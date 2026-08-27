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
# Triton fused attention kernel
# -----------------------------
if TRITON_AVAILABLE:
    @triton.jit
    def _fused_attn_fwd_kernel(
        Q, K, V,            # [M, C] each
        Bias,               # [M, K] bias to add to scores
        O,                  # [M, C] output

        M: tl.constexpr, C: tl.constexpr, K: tl.constexpr,

        stride_q_m: tl.constexpr, stride_q_c: tl.constexpr,
        stride_k_m: tl.constexpr, stride_k_c: tl.constexpr,
        stride_v_m: tl.constexpr, stride_v_c: tl.constexpr,
        stride_o_m: tl.constexpr, stride_o_c: tl.constexpr,
        stride_b_m: tl.constexpr, stride_b_k: tl.constexpr,

        BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_C: tl.constexpr,
    ):
        pid_m = tl.program_id(0)

        q_ptr = Q + pid_m * stride_q_m
        k_ptr = K + pid_m * stride_k_m
        v_ptr = V + pid_m * stride_v_m
        o_ptr = O + pid_m * stride_o_m
        b_ptr = Bias + pid_m * stride_b_m

        # running max and logsumexp
        mval = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
        lse = tl.zeros([BLOCK_M], dtype=tl.float32)

        # Pass 1: max over K
        k0 = 0
        while k0 < K:
            k_ids = k0 + tl.arange(0, BLOCK_K)
            mask_k = k_ids < K

            s_vec = tl.zeros([BLOCK_M], dtype=tl.float32)

            q0 = 0
            while q0 < M:
                m_ids = q0 + tl.arange(0, BLOCK_M)
                mask_m = m_ids < M

                # Q block [BLOCK_M, BLOCK_C]
                q_off = (m_ids[:, None] * stride_q_c) + (tl.arange(0, BLOCK_C)[None, :] * stride_q_c)
                q_mask = mask_m[:, None] & (tl.arange(0, BLOCK_C)[None, :] < C)
                q_block = tl.load(q_ptr + q_off, mask=q_mask, other=0.0).to(tl.float32)  # [BLOCK_M, BLOCK_C]

                # K block [BLOCK_K, BLOCK_C]
                k_off = (k_ids[:, None] * stride_k_c) + (tl.arange(0, BLOCK_C)[None, :] * stride_k_c)
                k_mask = mask_k[:, None] & (tl.arange(0, BLOCK_C)[None, :] < C)
                k_block = tl.load(k_ptr + k_off, mask=k_mask, other=0.0).to(tl.float32)  # [BLOCK_K, BLOCK_C]

                # dot: s = sum_c q * k  -> scalar per m
                acc = tl.zeros([BLOCK_M], dtype=tl.float32)
                for cc in range(0, BLOCK_C, 32):
                    c_ids = cc + tl.arange(0, 32)
                    q_c = q_block[:, c_ids]              # [BLOCK_M, 32]
                    k_c = k_block[:, c_ids]              # [BLOCK_K, 32]
                    for ci in range(32):
                        q_ci = q_c[:, ci]                # [BLOCK_M]
                        k_ci = k_c[:, ci]                # [BLOCK_K]
                        # sum over k: (q * k)[m,k] -> reduce over k
                        prod = q_ci[:, None] * k_ci[None, :]  # [BLOCK_M, BLOCK_K]
                        acc += tl.sum(prod, axis=1)            # sum over K dim
                s_vec = acc
                q0 += BLOCK_M

            # add bias: Bias[pid_m, k_ids]
            b_off = (k_ids[None, :] * stride_b_k)
            b_mask = mask_k[None, :]
            b_vec = tl.load(b_ptr + b_off, mask=b_mask, other=0.0).to(tl.float32)  # [BLOCK_K]
            s_vec += tl.sum(b_vec[None, :], axis=1)  # broadcast add

            row_max = tl.max(s_vec, axis=0)
            mval = tl.maximum(mval, row_max)
            k0 += BLOCK_K

        # Pass 2: logsumexp
        k0 = 0
        while k0 < K:
            k_ids = k0 + tl.arange(0, BLOCK_K)
            mask_k = k_ids < K

            s_vec = tl.zeros([BLOCK_M], dtype=tl.float32)

            q0 = 0
            while q0 < M:
                m_ids = q0 + tl.arange(0, BLOCK_M)
                mask_m = m_ids < M

                q_off = (m_ids[:, None] * stride_q_c) + (tl.arange(0, BLOCK_C)[None, :] * stride_q_c)
                q_mask = mask_m[:, None] & (tl.arange(0, BLOCK_C)[None, :] < C)
                q_block = tl.load(q_ptr + q_off, mask=q_mask, other=0.0).to(tl.float32)  # [BLOCK_M, BLOCK_C]

                k_off = (k_ids[:, None] * stride_k_c) + (tl.arange(0, BLOCK_C)[None, :] * stride_k_c)
                k_mask = mask_k[:, None] & (tl.arange(0, BLOCK_C)[None, :] < C)
                k_block = tl.load(k_ptr + k_off, mask=k_mask, other=0.0).to(tl.float32)  # [BLOCK_K, BLOCK_C]

                acc = tl.zeros([BLOCK_M], dtype=tl.float32)
                for cc in range(0, BLOCK_C, 32):
                    c_ids = cc + tl.arange(0, 32)
                    q_c = q_block[:, c_ids]
                    k_c = k_block[:, c_ids]
                    for ci in range(32):
                        q_ci = q_c[:, ci]
                        k_ci = k_c[:, ci]
                        prod = q_ci[:, None] * k_ci[None, :]
                        acc += tl.sum(prod, axis=1)
                s_vec = acc
                q0 += BLOCK_M

            b_off = (k_ids[None, :] * stride_b_k)
            b_mask = mask_k[None, :]
            b_vec = tl.load(b_ptr + b_off, mask=b_mask, other=0.0).to(tl.float32)  # [BLOCK_K]
            s_vec += tl.sum(b_vec[None, :], axis=1)

            p = tl.exp(s_vec - mval)   # [BLOCK_M]
            ssum = tl.sum(p, axis=0)   # scalar
            lse += tl.log(ssum)
            k0 += BLOCK_K

        # Pass 3: output O = sum_k p @ V_k
        o_acc = tl.zeros([BLOCK_C], dtype=tl.float32)

        k0 = 0
        while k0 < K:
            k_ids = k0 + tl.arange(0, BLOCK_K)
            mask_k = k_ids < K

            s_vec = tl.zeros([BLOCK_M], dtype=tl.float32)

            q0 = 0
            while q0 < M:
                m_ids = q0 + tl.arange(0, BLOCK_M)
                mask_m = m_ids < M

                q_off = (m_ids[:, None] * stride_q_c) + (tl.arange(0, BLOCK_C)[None, :] * stride_q_c)
                q_mask = mask_m[:, None] & (tl.arange(0, BLOCK_C)[None, :] < C)
                q_block = tl.load(q_ptr + q_off, mask=q_mask, other=0.0).to(tl.float32)  # [BLOCK_M, BLOCK_C]

                k_off = (k_ids[:, None] * stride_k_c) + (tl.arange(0, BLOCK_C)[None, :] * stride_k_c)
                k_mask = mask_k[:, None] & (tl.arange(0, BLOCK_C)[None, :] < C)
                k_block = tl.load(k_ptr + k_off, mask=k_mask, other=0.0).to(tl.float32)  # [BLOCK_K, BLOCK_C]

                acc = tl.zeros([BLOCK_M], dtype=tl.float32)
                for cc in range(0, BLOCK_C, 32):
                    c_ids = cc + tl.arange(0, 32)
                    q_c = q_block[:, c_ids]
                    k_c = k_block[:, c_ids]
                    for ci in range(32):
                        q_ci = q_c[:, ci]
                        k_ci = k_c[:, ci]
                        prod = q_ci[:, None] * k_ci[None, :]
                        acc += tl.sum(prod, axis=1)
                s_vec = acc
                q0 += BLOCK_M

            b_off = (k_ids[None, :] * stride_b_k)
            b_mask = mask_k[None, :]
            b_vec = tl.load(b_ptr + b_off, mask=b_mask, other=0.0).to(tl.float32)  # [BLOCK_K]
            s_vec += tl.sum(b_vec[None, :], axis=1)

            p = tl.exp(s_vec - lse)  # [BLOCK_M]

            # V block: [BLOCK_K, C]
            v_off = (k_ids[:, None] * stride_v_c) + (tl.arange(0, C)[None, :] * stride_v_c)
            v_mask = mask_k[:, None] & (tl.arange(0, C)[None, :] < C)
            v_block = tl.load(v_ptr + v_off, mask=v_mask, other=0.0).to(tl.float32)  # [BLOCK_K, C]

            # Accumulate: o_acc += sum_m p[m] * v[k,:]
            for kk in range(0, BLOCK_K, 32):
                kk_ids = kk + tl.arange(0, 32)
                p_k = p[kk_ids]            # [32]
                v_k = v_block[kk_ids, :]   # [32, C]
                for ci in range(32):
                    p_ci = p_k[ci]         # scalar
                    v_ci = v_k[ci, :]      # [C]
                    o_acc += p_ci * v_ci

            k0 += BLOCK_K

        # Store output: o_ptr points to [pid_m, 0:C]
        c_ids = tl.arange(0, C)
        o_off = c_ids * stride_o_c
        o_mask = c_ids < C
        tl.store(o_ptr + o_off, o_acc.to(tl.float32), mask=o_mask)


    def _launch_fused_attn_1d(
        q: torch.Tensor,  # [M, C]
        k: torch.Tensor,  # [M, C]
        v: torch.Tensor,  # [M, C]
        bias: torch.Tensor,  # [M, K]
    ) -> torch.Tensor:
        """
        Launches the fused attention kernel for 1D (M,C) layout.
        Returns o: [M, C] float32
        """
        assert TRITON_AVAILABLE, "Triton is not available"
        assert q.is_cuda and k.is_cuda and v.is_cuda and bias.is_cuda, "CUDA tensors required"

        q_ = q.contiguous()
        k_ = k.contiguous()
        v_ = v.contiguous()
        b_ = bias.contiguous()

        M, C = q_.shape
        _, K = b_.shape
        assert k_.shape == (M, C) and v_.shape == (M, C), "Shapes must be [M,C]"

        o = torch.empty((M, C), device=q_.device, dtype=torch.float32)

        # Strides
        stride_q_m = q_.stride(0)
        stride_q_c = q_.stride(1)
        stride_k_m = k_.stride(0)
        stride_k_c = k_.stride(1)
        stride_v_m = v_.stride(0)
        stride_v_c = v_.stride(1)
        stride_o_m = o.stride(0)
        stride_o_c = o.stride(1)
        stride_b_m = b_.stride(0)
        stride_b_k = b_.stride(1)

        # Grid
        grid = (M,)

        _fused_attn_fwd_kernel[grid](
            q_, k_, v_, b_, o,
            M, C, K,
            stride_q_m, stride_q_c,
            stride_k_m, stride_k_c,
            stride_v_m, stride_v_c,
            stride_o_m, stride_o_c,
            stride_b_m, stride_b_k,
            BLOCK_M=64, BLOCK_K=64, BLOCK_C=64,
            num_warps=4, num_stages=2,
        )

        return o


# -----------------------------
# Helper ops / modules
# -----------------------------
def _permute_final_dims(tensor: torch.Tensor, inds: list[int]) -> torch.Tensor:
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


class Pad(nn.Module):
    def forward(self, x: torch.Tensor, pad: tuple[int, ...], value: float = 0.0) -> torch.Tensor:
        return torch.nn.functional.pad(x, pad, value=value)


# -----------------------------
# Correct LayerNorm (supports arbitrary leading dims)
# -----------------------------
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps: float = 1e-5,
                 elementwise_affine: bool = True):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape))
            self.bias = nn.Parameter(torch.zeros(self.normalized_shape))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Validate last dims
        assert tuple(x.shape[-len(self.normalized_shape):]) == self.normalized_shape, \
            f"Input trailing shape {x.shape[-len(self.normalized_shape):]} != normalized_shape {self.normalized_shape}"
        return torch.nn.functional.layer_norm(
            x, self.normalized_shape, self.weight, self.bias, self.eps
        )


# -----------------------------
# AdaLN (self-contained, matching signature)
# -----------------------------
class AdaLN(nn.Module):
    """Adaptive LayerNorm for openfold3."""
    def __init__(self, c_a: int, c_s: int):
        super().__init__()
        self.c_a = c_a
        self.c_s = c_s
        # layer_norm_a: weight-only (no bias)
        self.layer_norm_a = LayerNorm(c_a, elementwise_affine=True)  # we'll remove bias later if needed
        # But to match reference, keep bias=False not used; so keep as is.
        self.linear_g = nn.Linear(c_s, c_a, bias=True)
        self.linear_s = nn.Linear(c_s, c_a, bias=False)

    def forward(self, a: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        s_norm = self.layer_norm_a(s)
        g = torch.sigmoid(self.linear_g(s_norm))
        a_norm = self.layer_norm_a(a)
        return g * (a_norm + self.linear_s(s_norm))


# -----------------------------
# OF3Attention (placeholder structure)
# -----------------------------
class OF3Attention(nn.Module):
    """Attention scores = softmax(QK^T + biases) V."""
    def __init__(self,
                 c_q: int, c_k: int, c_v: int,
                 c_hidden: int, no_heads: int, gating: bool = True, q_bias: bool = False):
        super().__init__()
        self.c_q = c_q
        self.c_k = c_k
        self.c_v = c_v
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.gating = gating
        self.q_bias = q_bias

        self.linear_q = nn.Linear(c_q, c_hidden * no_heads, bias=q_bias)
        self.linear_k = nn.Linear(c_k, c_hidden * no_heads, bias=False)
        self.linear_v = nn.Linear(c_v, c_hidden * no_heads, bias=False)
        self.linear_o = nn.Linear(c_hidden * no_heads, c_q, bias=False)
        if gating:
            self.linear_g = nn.Linear(c_q, c_hidden * no_heads, bias=False)
        else:
            self.linear_g = None

    def forward(self, q_x: torch.Tensor, kv_x: torch.Tensor, biases: list[torch.Tensor]):
        raise NotImplementedError("Use ModelNew.forward which calls the fused Triton kernel.")


# -----------------------------
# ModelNew: entry point
# -----------------------------
class ModelNew(nn.Module):
    """Triton-optimized Model with fused attention.

    Signature matches original Model:
      (c_q, c_k=0, c_v=0, c_s, c_z, c_hidden, no_heads, use_ada_layer_norm=False,
       n_query=None, n_key=None, gating=True, inf=1e9)
    """
    def __init__(
        self,
        c_q: int,
        c_k: int = 0,
        c_v: int = 0,
        c_s: int = 0,
        c_z: int = 128,
        c_hidden: int = 32,
        no_heads: int = 4,
        use_ada_layer_norm: bool = False,
        n_query: int = None,
        n_key: int = None,
        gating: bool = True,
        inf: float = 1e9,
    ):
        super().__init__()
        c_k = c_k or c_q
        c_v = c_v or c_q

        self.c_q = c_q
        self.c_s = c_s
        self.c_z = c_z
        self.inf = inf
        self.use_ada_layer_norm = use_ada_layer_norm
        self.n_query = n_query
        self.n_key = n_key

        if use_ada_layer_norm:
            self.layer_norm_a = AdaLN(c_a=c_q, c_s=c_s)
            self.linear_ada_out = nn.Linear(c_s, c_q, bias=True)
        else:
            self.layer_norm_a = LayerNorm(c_q)

        self.layer_norm_z = LayerNorm(c_z)
        self.linear_z = nn.Linear(c_z, no_heads, bias=False)

        self.sigmoid = nn.Sigmoid()

        # Keep OF3Attention to mirror structure (not used in forward)
        self.mha = OF3Attention(
            c_q=c_q, c_k=c_k, c_v=c_v,
            c_hidden=c_hidden, no_heads=no_heads, gating=gating,
            q_bias=True,
        )

    def _prep_biases(self, a: torch.Tensor, z: torch.Tensor, mask: torch.Tensor | None):
        """
        Returns list with a single bias tensor of shape [B,H,Q,K] (float32).
        We will sum mask+z into one bias to minimize kernel args.
        """
        biases = []

        # Combine mask and z into one bias
        if mask is not None:
            # mask: [*, N] -> [B, H, Q, K] -> [M, K]
            *lead, Q, _ = a.shape
            K = Q
            B = int(math.prod(lead)) if len(lead) > 0 else 1
            H = self.mha.no_heads
            # Expand mask to [B, Q, K]
            m = mask
            for _ in lead:
                m = m.unsqueeze(0)
            m = m.expand(B, Q, K)
            # Build bias: inf where 0, 0 where 1
            bias_mask = torch.where(
                m > 0,
                torch.zeros((), device=a.device, dtype=torch.float32),
                torch.full((), self.inf, device=a.device, dtype=torch.float32),
            ).unsqueeze(1).expand(B, H, Q, K).contiguous()  # [B,H,Q,K]
        else:
            bias_mask = torch.zeros((1, 0, 0, 0), device=a.device, dtype=torch.float32)  # dummy

        # Pair bias from z: [*, N, N, C_z] -> LN -> Linear -> [*, H, N, N] -> perm -> [H,*,*] -> [B,H,Q,K]
        z = self.layer_norm_z(z)
        z = self.linear_z(z)               # [*, no_heads, N, N]
        z = _permute_final_dims(z, [2, 0, 1])  # [no_heads, *, *]
        *lead_z, N1, N2 = z.shape
        Bz = int(math.prod(lead_z)) if len(lead_z) > 0 else 1
        assert N1 == Q and N2 == K, f"Expected z last dims match Q,K, got {N1},{N2} vs {Q},{K}"
        z_bias = z.unsqueeze(0).expand(B, H, Q, K).contiguous()  # [B,H,Q,K]

        if bias_mask.numel() > 0 and z_bias.numel() > 0:
            combined = bias_mask + z_bias
        elif bias_mask.numel() > 0:
            combined = bias_mask
        else:
            combined = z_bias

        biases.append(combined)  # shape [B,H,Q,K]

        return biases, combined  # return also combined to build Bias[M,K]

    def forward(
        self,
        a: torch.Tensor,
        z: torch.Tensor,
        s: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        use_deepspeed_evo_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_lma: bool = False,
        use_high_precision_attention: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            a:    [*, N, C_q]
            z:    [*, N, N, C_z]
            s:    [*, N, C_s] or None
            mask: [*, N] or None
        Returns:
            [*, N, C_q]
        """
        device = a.device

        # Normalize a
        if self.use_ada_layer_norm and s is not None:
            a = self.layer_norm_a(a, s)
        else:
            a = self.layer_norm_a(a)

        # Shapes
        *lead, N, C = a.shape
        Q = N
        K = Q
        B = int(math.prod(lead)) if len(lead) > 0 else 1
        H = self.mha.no_heads

        # Prepare combined bias [B,H,Q,K]
        _, combined = self._prep_biases(a=a, z=z, mask=mask)

        if combined.numel() == 0:
            # No bias
            combined = torch.zeros((B, H, Q, K), device=device, dtype=torch.float32)

        # Reshape to 1D for kernel: M = B*H*Q, C, K
        a_4d = a  # [*, N, C]
        M = B * H * Q
        q = a_4d.reshape(M, C).contiguous()
        k = q  # kv == q
        v = q

        # Bias to [M, K]
        bias = combined.reshape(M, K).contiguous()

        if TRITON_AVAILABLE and q.is_cuda:
            o = _launch_fused_attn_1d(q, k, v, bias)  # float32 [M, C]
            o = o.to(a.dtype)
            out = o.reshape(*lead, N, C)
        else:
            raise RuntimeError("Triton not available; CUDA required for ModelNew forward.")

        # AdaLN output gate if enabled
        if self.use_ada_layer_norm and s is not None:
            g = self.sigmoid(self.linear_ada_out(s))  # [*, N, C_q]
            out = g * out

        return out

AttentionPairBias = ModelNew
CrossAttentionPairBias = ModelNew
