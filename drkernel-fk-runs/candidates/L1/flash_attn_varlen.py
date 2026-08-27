import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False

# Import the original function to fallback
from fastkernels.infra.fa_utils import flash_attn_varlen_func


@triton.jit
def _flash_attn_equal_len_2pass(
    Q, K, V, Out, Lse,
    L,  # sequence length (runtime int)
    H,  # num heads (runtime int)
    BD, # dim Q/K (runtime int)
    BV, # dim V    (runtime int)
    stride_q_l, stride_q_h, stride_q_d,
    stride_k_l, stride_k_h, stride_k_d,
    stride_v_l, stride_v_h, stride_v_d,
    stride_o_l, stride_o_h, stride_o_d,
    stride_l_l, stride_l_h,
    softmax_scale,  # float
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
    STORE_LSE: tl.constexpr,  # whether to store lse
):
    pid_m = tl.program_id(0)  # tile id over length
    pid_h = tl.program_id(1)  # head id

    l_start = pid_m * BLOCK_M
    rows = l_start + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_D)

    # Helper pointers (element-based addressing)
    def q_ptr(l, d): return Q + l * stride_q_l + pid_h * stride_q_h + d * stride_q_d
    def k_ptr(l, d): return K + l * stride_k_l + pid_h * stride_k_h + d * stride_k_d
    def v_ptr(l, d): return V + l * stride_v_l + pid_h * stride_v_h + d * stride_v_d
    def o_ptr(l, d): return Out + l * stride_o_l + pid_h * stride_o_h + d * stride_o_d
    def lse_ptr(l):   return Lse  + l * stride_l_l + pid_h * stride_l_h

    row_mask = rows < L

    # Load Q tile [BM, BD] (accumulate in float32)
    q = tl.zeros((BLOCK_M, BD), dtype=tl.float32)
    for d0 in range(0, BD, BLOCK_D):
        qd = cols + d0
        d_mask = qd < BD
        ptr_q = q_ptr(rows[:, None], qd[None, :])
        q_curr = tl.load(ptr_q, mask=row_mask[:, None] & d_mask[None, :], other=0.0)
        q += q_curr.to(tl.float32)

    # Pass 1: compute lse per row
    m_i = tl.full((BLOCK_M,), -1.0e30, dtype=tl.float32)
    lse_i = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for k0 in range(0, L, BLOCK_K):
        k_rows = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_rows < L

        # K block [BK, BD]
        k = tl.zeros((BLOCK_K, BD), dtype=tl.float32)
        for d0 in range(0, BD, BLOCK_D):
            kd = cols + d0
            d_mask = kd < BD
            ptr_k = k_ptr(k_rows[:, None], kd[None, :])
            k_curr = tl.load(ptr_k, mask=k_mask[:, None] & d_mask[None, :], other=0.0)
            k += k_curr.to(tl.float32)

        # scores = Q @ K^T -> [BM, BK]
        scores = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
        for d0 in range(0, BD, BLOCK_D):
            qd = cols + d0
            kd = cols + d0
            d_mask_q = qd < BD
            d_mask_k = kd < BD
            # q_sub: [BM, BLOCK_D]
            ptr_q_sub = q_ptr(rows[:, None], qd[None, :])
            q_sub = tl.load(ptr_q_sub, mask=row_mask[:, None] & d_mask_q[None, :], other=0.0).to(tl.float32)
            # k_sub_T: [BLOCK_D, BK]
            ptr_k_sub_T = k_ptr(k_rows[None, :], kd[:, None])
            k_sub_T = tl.load(ptr_k_sub_T, mask=k_mask[None, :] & d_mask_k[:, None], other=0.0).to(tl.float32)
            scores += tl.dot(q_sub, k_sub_T)  # [BM,BK]

        scores = scores * softmax_scale
        # zero out invalid k
        scores = tl.where(k_mask[None, :], scores, -1.0e30)

        row_max = tl.max(scores, axis=1)
        new_m = tl.maximum(m_i, row_max)
        p = tl.exp(scores - new_m[:, None])
        sum_p = tl.sum(p, axis=1)
        lse = new_m + tl.log(sum_p)
        m_i = new_m
        lse_i = lse

    # Pass 2: compute output O = sum p @ V
    o = tl.zeros((BLOCK_M, BV), dtype=tl.float32)
    for k0 in range(0, L, BLOCK_K):
        k_rows = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_rows < L

        # K block [BK, BD]
        k = tl.zeros((BLOCK_K, BD), dtype=tl.float32)
        for d0 in range(0, BD, BLOCK_D):
            kd = cols + d0
            d_mask = kd < BD
            ptr_k = k_ptr(k_rows[:, None], kd[None, :])
            k_curr = tl.load(ptr_k, mask=k_mask[:, None] & d_mask[None, :], other=0.0)
            k += k_curr.to(tl.float32)

        # scores = Q @ K^T -> [BM, BK]
        scores = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
        for d0 in range(0, BD, BLOCK_D):
            qd = cols + d0
            kd = cols + d0
            d_mask_q = qd < BD
            d_mask_k = kd < BD
            ptr_q_sub = q_ptr(rows[:, None], qd[None, :])
            q_sub = tl.load(ptr_q_sub, mask=row_mask[:, None] & d_mask_q[None, :], other=0.0).to(tl.float32)
            ptr_k_sub_T = k_ptr(k_rows[None, :], kd[:, None])
            k_sub_T = tl.load(ptr_k_sub_T, mask=k_mask[None, :] & d_mask_k[:, None], other=0.0).to(tl.float32)
            scores += tl.dot(q_sub, k_sub_T)

        scores = scores * softmax_scale
        scores = tl.where(k_mask[None, :], scores, -1.0e30)

        # p = exp(scores - lse)
        p = tl.exp(scores - lse_i[:, None])  # [BM, BK]

        # V block [BK, BV]
        v = tl.zeros((BLOCK_K, BV), dtype=tl.float32)
        for d0v in range(0, BV, BLOCK_D):
            vd = cols + d0v
            v_mask = vd < BV
            ptr_v = v_ptr(k_rows[:, None], vd[None, :])
            v_curr = tl.load(ptr_v, mask=k_mask[:, None] & v_mask[None, :], other=0.0)
            v += v_curr.to(tl.float32)

        o += tl.dot(p, v)  # [BM, BV]

    # Store output element-wise in BV blocks
    for d0v in range(0, BV, BLOCK_D):
        vd = cols + d0v
        v_mask = vd < BV
        # loop over BLOCK_D to avoid slicing
        for jj in range(BLOCK_D):
            j = d0v + jj
            if j < BV:
                ptr_o_j = o_ptr(rows, j)
                tl.store(ptr_o_j, o[:, jj], mask=row_mask)

    # Optionally store lse
    if STORE_LSE:
        ptr_lse = lse_ptr(rows)
        tl.store(ptr_lse, lse_i, mask=row_mask)


class ModelNew(nn.Module):
    """
    Triton-optimized variant of the original Model.

    Uses a custom FlashAttention kernel for the common case:
      - Q and K have equal numbers of sequences and equal per-sequence lengths
      - Dq == Dk
      - CUDA device
    Falls back to vLLM's flash_attn_varlen_func otherwise.
    """
    def __init__(self):
        super().__init__()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        softmax_scale: float,
        causal: bool = True,
        return_softmax_lse: bool = False,
    ):
        # Fallback if Triton not available or not CUDA
        if (not TRITON_AVAILABLE) or (not q.is_cuda):
            return flash_attn_varlen_func(
                q, k, v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                softmax_scale=softmax_scale,
                causal=causal,
                return_softmax_lse=return_softmax_lse,
            )

        # Validate shapes
        assert q.ndim == 3 and k.ndim == 3 and v.ndim == 3, "Q/K/V must be [L, H, D]"
        Lq, Hq, BDq = q.shape
        Lk, Hk, BDk = k.shape
        Lv, Hv, BV  = v.shape
        assert Hq == Hk == Hv, f"Head mismatch: {Hq} vs {Hk} vs {Hv}"
        assert BDq == BDk, f"Q/K dim mismatch: {BDq} vs {BDk}"
        H = Hq

        # Check equal-length fast path
        Tq = cu_seqlens_q.numel()
        Tk = cu_seqlens_k.numel()
        assert Tq == Tk, f"Sequence counts mismatch: {Tq} vs {Tk}"

        # If T==1, L = cu[1]-cu[0]; else require all equal
        if Tq == 1:
            L = int(cu_seqlens_q[1].item() - cu_seqlens_q[0].item())
            all_equal = True
        else:
            lens_q = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).tolist()
            lens_k = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).tolist()
            unique_q = set(lens_q)
            unique_k = set(lens_k)
            all_equal = (len(unique_q) == 1) and (len(unique_k) == 1) and (unique_q == unique_k)
            L = int(unique_q.pop()) if all_equal else -1

        use_equal_len = (Lq == Lk) and BDq == BDk and all_equal and (L > 0)

        if not use_equal_len:
            # Fallback to varlen
            return flash_attn_varlen_func(
                q, k, v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                softmax_scale=softmax_scale,
                causal=causal,
                return_softmax_lse=return_softmax_lse,
            )

        # Make contiguous and view as [L, H, D]
        q_ = q.contiguous().view(L, H, BDq)
        k_ = k.contiguous().view(L, H, BDk)
        v_ = v.contiguous().view(L, H, BV)

        # Allocate output (float32)
        out = torch.empty((L, H, BV), device=q.device, dtype=torch.float32)

        # Optional lse buffer
        if return_softmax_lse:
            lse = torch.empty((L, H), device=q.device, dtype=torch.float32)
        else:
            # Dummy; kernel won't store due to STORE_LSE=False
            lse = torch.empty(1, device=q.device, dtype=torch.float32)

        # Strides in elements
        s_q_l, s_q_h, s_q_d = q_.stride(0), q_.stride(1), q_.stride(2)
        s_k_l, s_k_h, s_k_d = k_.stride(0), k_.stride(1), k_.stride(2)
        s_v_l, s_v_h, s_v_d = v_.stride(0), v_.stride(1), v_.stride(2)
        s_o_l, s_o_h, s_o_d = out.stride(0), out.stride(1), out.stride(2)
        s_l_l, s_l_h = lse.stride(0), lse.stride(1) if return_softmax_lse else (0, 0)

        # Choose block sizes (heuristics)
        BD = BDq
        BLOCK_D = 64 if BD >= 64 else 32
        BLOCK_K = 64 if L >= 64 else 32
        BLOCK_M = 64 if L >= 64 else 32

        grid = (triton.cdiv(L, BLOCK_M), H)

        _flash_attn_equal_len_2pass[grid](
            q_, k_, v_, out, lse,
            L, H, BD, BV,
            s_q_l, s_q_h, s_q_d,
            s_k_l, s_k_h, s_k_d,
            s_v_l, s_v_h, s_v_d,
            s_o_l, s_o_h, s_o_d,
            s_l_l, s_l_h,
            softmax_scale,
            BLOCK_M=BLOCK_M, BLOCK_D=BLOCK_D, BLOCK_K=BLOCK_K,
            STORE_LSE=return_softmax_lse,
            num_warps=4, num_stages=2,
        )

        if return_softmax_lse:
            return out.view_as(v), lse.view(L, H)
        else:
            return out.view_as(v)

FlashAttnVarlen = ModelNew
