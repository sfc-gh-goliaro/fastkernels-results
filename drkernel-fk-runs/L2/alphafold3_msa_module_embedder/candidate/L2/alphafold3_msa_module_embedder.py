import math
import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# -----------------------------
# Fused Triton kernel:
# Computes, for each effective batch b_eff:
#   C[b_eff, m, n] = sum_k A_m[b_eff, m, k_m] * Wm^T[k_m, n]  +  sum_k A_s[bb, m, k_s] * Ws^T[k_s, n]
# A_m: [B_eff, M, K_m]  (msa_feat viewed)
# Wm^T: [K_m, N]   (linear_m.weight.t())
# A_s: [B, M, K_s]  (s_input)
# Ws^T: [K_s, N]   (linear_s_input.weight.t())
# Result C: [B_eff, M, N]
# -----------------------------

if TRITON_AVAILABLE:
    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_KM': 32, 'BLOCK_KS': 32}, num_stages=2, num_warps=4),
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_KM': 32, 'BLOCK_KS': 32}, num_stages=3, num_warps=8),
            triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_KM': 32, 'BLOCK_KS': 32}, num_stages=3, num_warps=8),
            triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_KM': 32, 'BLOCK_KS': 32}, num_stages=2, num_warps=4),
        ],
        key=['M', 'N', 'K_m', 'K_s'],
    )
    @triton.jit
    def fused_mm_add_kernel(
        A_m_ptr, WmT_ptr, A_s_ptr, WsT_ptr, C_ptr,
        B_eff, M, N, K_m, K_s,
        stride_Amb, stride_Amm, stride_Amk,   # A_m strides: [b, m, k]
        stride_Wmn, stride_Wnn,                # Wm^T strides: [k, n]
        stride_Asb, stride_Asm, stride_Ask,   # A_s strides: [b, m, k]
        stride_Wsn, stride_Wsn2,              # Ws^T strides: [k, n]
        stride_cb, stride_cm, stride_cn,      # C strides: [b, m, n]
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
        BLOCK_KM: tl.constexpr, BLOCK_KS: tl.constexpr,
    ):
        # Program ids
        pid_b = tl.program_id(0)
        pid_m = tl.program_id(1)
        pid_n = tl.program_id(2)

        # Offsets
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_km = tl.arange(0, BLOCK_KM)
        offs_ks = tl.arange(0, BLOCK_KS)

        # Accumulators
        acc_m = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        acc_s = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # -------- First GEMM: A_m @ Wm^T --------
        for k in range(0, K_m, BLOCK_KM):
            a_m_ptrs = A_m_ptr + pid_b * stride_Amb + (offs_m[:, None] * stride_Amm) + ((k + offs_km)[None, :] * stride_Amk)
            w_m_ptrs = WmT_ptr + ((k + offs_km)[:, None] * stride_Wmn) + (offs_n[None, :] * stride_Wnn)

            k_mask_m = (k + offs_km) < K_m
            m_mask = offs_m < M
            n_mask = offs_n < N

            a_m = tl.load(a_m_ptrs, mask=m_mask[:, None] & k_mask_m[None, :], other=0.0)
            w_m = tl.load(w_m_ptrs, mask=k_mask_m[:, None] & n_mask[None, :], other=0.0)

            acc_m += tl.dot(a_m, w_m)  # -> float32

        # -------- Second GEMM: A_s @ Ws^T --------
        for k in range(0, K_s, BLOCK_KS):
            a_s_ptrs = A_s_ptr + pid_b * stride_Asb + (offs_m[:, None] * stride_Asm) + ((k + offs_ks)[None, :] * stride_Ask)
            w_s_ptrs = WsT_ptr + ((k + offs_ks)[:, None] * stride_Wsn) + (offs_n[None, :] * stride_Wsn2)

            k_mask_s = (k + offs_ks) < K_s
            a_s = tl.load(a_s_ptrs, mask=m_mask[:, None] & k_mask_s[None, :], other=0.0)
            w_s = tl.load(w_s_ptrs, mask=k_mask_s[:, None] & n_mask[None, :], other=0.0)

            acc_s += tl.dot(a_s, w_s)  # -> float32

        # Sum and store
        acc = acc_m + acc_s
        c_ptrs = C_ptr + pid_b * stride_cb + (offs_m[:, None] * stride_cm) + (offs_n[None, :] * stride_cn)
        m_mask = offs_m < M
        n_mask = offs_n < N
        mask = m_mask[:, None] & n_mask[None, :]
        tl.store(c_ptrs, acc, mask=mask)


def _torch_to_tl_dtype(torch_dtype):
    if torch_dtype == torch.float32:
        return tl.float32
    if torch_dtype == torch.float16:
        return tl.float16
    if torch_dtype == torch.bfloat16:
        return tl.bfloat16
    raise ValueError(f"Unsupported dtype for Triton matmul: {torch_dtype}")


def triton_fused_mm_add(a_m: torch.Tensor, wm_t: torch.Tensor, a_s: torch.Tensor, ws_t: torch.Tensor) -> torch.Tensor:
    """
    Fused compute:
      C = a_m @ wm_t + a_s @ ws_t
    Shapes:
      a_m:  [B_eff, M, K_m]
      wm_t:[K_m, N]
      a_s:  [B, M, K_s]
      ws_t:[K_s, N]
    Return:
      C: [B_eff, M, N]
    """
    if not TRITON_AVAILABLE:
        return a_m.matmul(wm_t) + a_s.matmul(ws_t)
    if (not a_m.is_cuda) or (not wm_t.is_cuda) or (not a_s.is_cuda) or (not ws_t.is_cuda):
        return a_m.matmul(wm_t) + a_s.matmul(ws_t)

    # Validate shapes
    assert a_m.dim() == 3 and a_s.dim() == 3, "Expected 3D tensors [B,M,K]"
    assert wm_t.dim() == 2 and ws_t.dim() == 2, "Expected 2D tensors [K,N]"
    Bm, M, K_m = a_m.shape
    Bs, Ms, K_s = a_s.shape
    Kw_m, N = wm_t.shape
    Kw_s, N2 = ws_t.shape
    assert M == Ms, f"M mismatch: {M} vs {Ms}"
    assert K_m == Kw_m, f"K_m mismatch: {K_m} vs {Kw_m}"
    assert K_s == Kw_s, f"K_s mismatch: {K_s} vs {Kw_s}"
    assert N == N2, f"N mismatch: {N} vs {N2}"

    # We allow a_s batch size different from a_m; in our use, a_s[B,T,K], but the kernel only uses A_s[b, m, :]
    # So we will relabel A_s batches to match grid dim 0 by iterating over Bm and using A_s[b % Bs, ...].
    # Easier: require Bs == Bm (typical). If not, fallback to torch.
    if Bs != Bm:
        return a_m.matmul(wm_t) + a_s.matmul(ws_t)

    # Dtype
    dtype = a_m.dtype
    assert a_s.dtype == dtype and wm_t.dtype == dtype and ws_t.dtype == dtype, "All dtypes must match"
    supported = {torch.float16, torch.bfloat16, torch.float32}
    assert dtype in supported, f"Unsupported dtype {dtype}; supported: {supported}"

    # Contiguous
    if not a_m.is_contiguous():
        a_m = a_m.contiguous()
    if not wm_t.is_contiguous():
        wm_t = wm_t.contiguous()
    if not a_s.is_contiguous():
        a_s = a_s.contiguous()
    if not ws_t.is_contiguous():
        ws_t = ws_t.contiguous()

    # Allocate output
    C = torch.empty((Bm, M, N), device=a_m.device, dtype=dtype)

    # Strides
    stride_Amb, stride_Amm, stride_Amk = a_m.stride(0), a_m.stride(1), a_m.stride(2)
    stride_Wmn, stride_Wnn = wm_t.stride(0), wm_t.stride(1)
    stride_Asb, stride_Asm, stride_Ask = a_s.stride(0), a_s.stride(1), a_s.stride(2)
    stride_Wsn, stride_Wsn2 = ws_t.stride(0), ws_t.stride(1)
    stride_cb, stride_cm, stride_cn = C.stride(0), C.stride(1), C.stride(2)

    # Grid over effective batch
    grid = (Bm, triton.cdiv(M, 64), triton.cdiv(N, 64))

    fused_mm_add_kernel[grid](
        a_m, wm_t, a_s, ws_t, C,
        Bm, M, N, K_m, K_s,
        stride_Amb, stride_Amm, stride_Amk,
        stride_Wmn, stride_Wnn,
        stride_Asb, stride_Asm, stride_Ask,
        stride_Wsn, stride_Wsn2,
        stride_cb, stride_cm, stride_cn,
    )

    return C


class ModelNew(nn.Module):
    """
    Triton-optimized version of the original Model.

    Keeps the same __init__ and forward signature.

    forward(batch, s_input) -> (m, msa_mask)
      where m: [*, N_seq, N_token, c_m]
            msa_mask: [*, N_seq, N_token]
    """

    def __init__(
        self,
        c_m_feats: int = 34,
        c_m: int = 64,
        c_s_input: int = 449,
    ):
        super().__init__()
        # Keep parameters as in original: no bias
        self.linear_m = nn.Linear(c_m_feats, c_m, bias=False)
        self.linear_s_input = nn.Linear(c_s_input, c_m, bias=False)

    def forward(
        self,
        batch: dict,
        s_input: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            batch: needs msa [*, N_msa, N_token, 32],
                   has_deletion [*, N_msa, N_token],
                   deletion_value [*, N_msa, N_token],
                   msa_mask [*, N_msa, N_token]
            s_input: [*, N_token, c_s_input]

        Returns:
            m: [*, N_msa, N_token, c_m]
            msa_mask: [*, N_msa, N_token]
        """
        # Build msa_feat
        msa = batch["msa"]
        has_del = batch["has_deletion"]
        del_val = batch["deletion_value"]
        assert msa.shape[-1] == 32, f"Expected msa last dim 32, got {msa.shape[-1]}"
        assert has_del.shape == del_val.shape == msa.shape[:-1], "Shape mismatch for has_deletion/deletion_value"

        msa_feat = torch.cat(
            [
                msa,
                has_del.unsqueeze(-1),
                del_val.unsqueeze(-1),
            ],
            dim=-1,
        )  # [* , S, T, 34]

        msa_mask = batch["msa_mask"]  # [* , S, T]

        # Extract shapes
        B = msa_feat.shape[0]
        S = msa_feat.shape[1]
        T = msa_feat.shape[2]
        Cin_m = msa_feat.shape[3]
        Cout = self.linear_m.out_features
        Cin_s = self.linear_s_input.in_features
        assert s_input.shape[-1] == Cin_s, f"Expected s_input last dim {Cin_s}, got {s_input.shape[-1]}"

        # Weights
        Wm = self.linear_m.weight  # [64, 34]
        Ws = self.linear_s_input.weight  # [64, 449]

        use_triton = TRITON_AVAILABLE and msa_feat.is_cuda and s_input.is_cuda and Wm.is_cuda and Ws.is_cuda

        if use_triton:
            # Prepare A_m: [B*S, T, 34] by viewing
            A_m = msa_feat.view(B * S, T, Cin_m).contiguous()
            Wm_t = Wm.t().contiguous()  # [34, 64]

            # Prepare A_s: [B, T, 449]
            A_s = s_input
            Ws_t = Ws.t().contiguous()  # [449, 64]

            # Fused: [B*S, T, 64]
            C_view = triton_fused_mm_add(A_m, Wm_t, A_s, Ws_t)  # [B*S, T, 64]
            m = C_view.view(B, S, T, Cout)
        else:
            # Torch fallback
            m = msa_feat.matmul(Wm.t())
            out_s = s_input.matmul(Ws.t())
            m = m + out_s.unsqueeze(1)

        return m, msa_mask

MSAModuleEmbedder = ModelNew
