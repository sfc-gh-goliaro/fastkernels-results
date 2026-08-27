import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import Triton; if not available, we'll fallback to PyTorch
try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False


# ------------------------------
# Triton kernels / helpers
# ------------------------------

if _HAS_TRITON:
    @triton.jit
    def _matmul_bias_kernel(
        A, B, Bias, C,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        # Program ids for tiles
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        # Pointers for A and B
        a_ptrs = A + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = B + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        # Accumulator in fp32
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # Loop over K dimension
        for k in range(0, K, BLOCK_K):
            a = tl.load(
                a_ptrs,
                mask=(offs_m[:, None] < M) & (k + offs_k[None, :] < K),
                other=0.0,
            )
            b = tl.load(
                b_ptrs,
                mask=(k + offs_k[:, None] < K) & (offs_n[None, :] < N),
                other=0.0,
            )
            acc += tl.dot(a, b)
            # Advance pointers along K
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

        # Add bias if provided
        if Bias is not None:
            bias = tl.load(Bias + offs_n, mask=(offs_n < N), other=0.0)
            acc = acc + bias[None, :]

        # Store result
        c_ptrs = C + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
        tl.store(
            c_ptrs,
            acc,
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        )

    def _triton_linear(A: torch.Tensor, B: torch.Tensor, bias: torch.Tensor | None,
                       out: torch.Tensor | None = None,
                       block_m: int = 64, block_n: int = 64, block_k: int = 32) -> torch.Tensor:
        """
        Compute C = A @ B + bias using Triton.
        Shapes:
          A: [M, K]
          B: [K, N]
          bias: [N] or None
        Output:
          C: [M, N]
        """
        assert A.is_cuda and B.is_cuda, "Triton kernel requires CUDA tensors"
        assert A.dtype in (torch.float32, torch.bfloat16), "Supported dtypes: fp32, bf16"
        assert B.dtype == A.dtype, "A and B must have same dtype"

        M, K = A.shape
        Kb, N = B.shape
        assert Kb == K, f"Incompatible shapes: A is (*,{K}), B is (*,{Kb},{N})"

        # Allocate output (accumulate in fp32)
        if out is None:
            out = torch.empty((M, N), device=A.device, dtype=torch.float32)
        else:
            assert out.shape == (M, N)
            assert out.dtype == torch.float32

        # Strides in elements
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        stride_bk = B.stride(0)
        stride_bn = B.stride(1)
        stride_cm = out.stride(0)
        stride_cn = out.stride(1)

        grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))

        _matmul_bias_kernel[grid](
            A, B, bias if bias is not None else tl.zeros((1,), dtype=tl.float32), out,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
            num_warps=4, num_stages=2,
        )

        # Cast back to input dtype if needed
        if A.dtype == torch.bfloat16:
            return out.to(torch.bfloat16)
        else:
            return out  # fp32


# ------------------------------
# Core modules (unchanged API/behavior)
# ------------------------------

class Matmul(nn.Module):
    """Functional linear: input, weight, bias."""
    def forward(self, input, weight, bias=None):
        return F.linear(input, weight, bias)

class Linear(nn.Module):
    """Parametric linear: stores weight and bias."""
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.matmul = Matmul()

    def forward(self, input):
        return self.matmul(input, self.weight, self.bias)

class SiLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x)

class SwiGLU(nn.Module):
    """SwiGLU: silu(Wa x) * Wb x."""
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.silu = SiLU()
        self.linear_a = Linear(c_in, c_out, bias=False)
        self.linear_b = Linear(c_in, c_out, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.silu(self.linear_a(x)) * self.linear_b(x)

class LayerNorm(nn.Module):
    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        create_scale: bool = True,
        create_offset: bool = True,
        promote_fp32: bool = True,
    ):
        super().__init__()
        self.normalized_shape = (normalized_shape,)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.promote_fp32 = promote_fp32

        if elementwise_affine and create_scale:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
        else:
            self.register_parameter("weight", None)

        if elementwise_affine and create_offset:
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter("bias", None)

        self._cast_done = False
        self._src_w = None
        self._src_b = None
        self._w32 = None
        self._b32 = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.promote_fp32:
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps,
            )

        orig_dtype = x.dtype
        if (not self._cast_done
                or self._src_w is not self.weight
                or self._src_b is not self.bias):
            w, b = self.weight, self.bias
            self._src_w, self._src_b = w, b
            self._w32 = (w.float() if w is not None and w.dtype != torch.float32 else w)
            self._b32 = (b.float() if b is not None and b.dtype != torch.float32 else b)
            self._cast_done = True
        weight, bias = self._w32, self._b32
        return F.layer_norm(
            x.float(), self.normalized_shape, weight, bias, self.eps,
        ).to(orig_dtype)

class OuterProductMean(nn.Module):
    """AF3 Algorithm 9: Outer product mean."""
    def __init__(self, c_m: int, c_z: int, c_hidden: int, eps: float = 1e-3):
        super().__init__()
        self.c_m = c_m
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.eps = eps

        self.layer_norm = LayerNorm(c_m)
        self.linear_1 = Linear(c_m, c_hidden, bias=False)
        self.linear_2 = Linear(c_m, c_hidden, bias=False)
        self.linear_out = Linear(c_hidden ** 2, c_z, bias=True)

    def forward(
        self,
        m: torch.Tensor,
        mask: torch.Tensor | None = None,
        chunk_size: int | None = None,
        inplace_safe: bool = False,
    ) -> torch.Tensor:
        if mask is None:
            mask = m.new_ones(m.shape[:-1])

        ln = self.layer_norm(m)

        mask = mask.unsqueeze(-1)
        a = self.linear_1(ln) * mask
        b = self.linear_2(ln) * mask

        # [*, N_res, N_seq, C]
        a = a.transpose(-2, -3)
        b = b.transpose(-2, -3)

        # [*, N_res, N_res, C, C]
        outer = torch.einsum("...bac,...dae->...bdce", a, b)
        outer = outer.reshape(outer.shape[:-2] + (-1,))
        outer = self.linear_out(outer)

        # Normalization: count valid sequence pairs per residue pair
        norm = torch.einsum("...abc,...adc->...bdc", mask, mask)
        norm = norm + self.eps

        outer = outer / norm

        return outer

class MSARowAttentionWithPairBias(nn.Module):
    """AF3 MSA Pair-Weighted Averaging (Algorithm 10)."""
    def __init__(
        self,
        c_m: int,
        c_z: int,
        c_hidden: int,
        no_heads: int,
        inf: float = 1e9,
    ):
        super().__init__()
        self.c_m = c_m
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.inf = inf

        self.layer_norm_m = LayerNorm(c_m)
        self.layer_norm_z = LayerNorm(c_z)
        self.linear_z = Linear(c_z, no_heads, bias=False)

        self.linear_v = Linear(c_m, c_hidden * no_heads, bias=False)
        self.linear_g = Linear(c_m, c_hidden * no_heads, bias=False)
        self.linear_o = Linear(c_hidden * no_heads, c_m, bias=False)

        self.sigmoid = Sigmoid()
        self.softmax = Softmax(dim=-1)

    def forward(
        self,
        m: torch.Tensor,
        z: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        chunk_size: int | None = None,
        use_deepspeed_evo_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
    ) -> torch.Tensor:
        if z is None:
            return m

        if mask is None:
            mask = z.new_ones(z.shape[:-1])

        n_res = z.shape[-2]
        mask_bias = (self.inf * (mask - 1))[..., None, None, :, :]
        z_norm = self.layer_norm_z(z)
        z_proj = self.linear_z(z_norm)
        z_weights = _permute_final_dims(z_proj, (2, 0, 1)).unsqueeze(-4)
        z_weights = z_weights + mask_bias
        z_weights = self.softmax(z_weights)

        m = self.layer_norm_m(m)

        v = self.linear_v(m)
        v = v.view(v.shape[:-1] + (self.no_heads, -1))
        v = v.transpose(-2, -3)  # [*, N_seq, H, N_res, C_hidden]

        o = torch.einsum("...hqk,...hkc->...qhc", z_weights, v)
        g = self.sigmoid(self.linear_g(m))
        g = g.view(g.shape[:-1] + (self.no_heads, -1))
        o = o * g

        o = o.reshape(o.shape[:-2] + (-1,))
        o = self.linear_o(o)
        return o

class TriangleAttention(nn.Module):
    """AF3 Algorithms 14/15: Triangle attention."""
    def __init__(
        self,
        c_in: int,
        c_hidden: int,
        no_heads: int,
        starting: bool = True,
        inf: float = 1e9,
    ):
        super().__init__()
        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.starting = starting
        self.inf = inf

        self.layer_norm = LayerNorm(c_in)
        self.linear_z = Linear(c_in, no_heads, bias=False)

        self.mha = OF3Attention(
            c_q=c_in,
            c_k=c_in,
            c_v=c_in,
            c_hidden=c_hidden,
            no_heads=no_heads,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        chunk_size: int | None = None,
        use_deepspeed_evo_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
    ) -> torch.Tensor:
        if mask is None:
            mask = x.new_ones(x.shape[:-1])

        if not self.starting:
            x = x.transpose(-2, -3)
            mask = mask.transpose(-1, -2)

        x = self.layer_norm(x)

        mask_bias = (self.inf * (mask - 1))[..., :, None, None, :]
        triangle_bias = _permute_final_dims(self.linear_z(x), (2, 0, 1)).unsqueeze(-4)
        biases = [mask_bias, triangle_bias]

        x = self.mha(q_x=x, kv_x=x, biases=biases)
        if not self.starting:
            x = x.transpose(-2, -3)
        return x

class OF3Attention(nn.Module):
    """Standard multi-head attention with gating and bias list support."""
    def __init__(
        self,
        c_q: int,
        c_k: int,
        c_v: int,
        c_hidden: int,
        no_heads: int,
        gating: bool = True,
        q_bias: bool = False,
    ):
        super().__init__()
        self.c_q = c_q
        self.c_k = c_k
        self.c_v = c_v
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.gating = gating

        self.linear_q = Linear(c_q, c_hidden * no_heads, bias=q_bias)
        self.linear_k = Linear(c_k, c_hidden * no_heads, bias=False)
        self.linear_v = Linear(c_v, c_hidden * no_heads, bias=False)
        self.linear_o = Linear(c_hidden * no_heads, c_q, bias=False)

        self.linear_g = None
        if gating:
            self.linear_g = Linear(c_q, c_hidden * no_heads, bias=False)

    def _prep_qkv(self, q_x: torch.Tensor, kv_x: torch.Tensor, apply_scale: bool = True):
        q = self.linear_q(q_x)
        k = self.linear_k(kv_x)
        v = self.linear_v(kv_x)

        q = q.view(q.shape[:-1] + (self.no_heads, -1))
        k = k.view(k.shape[:-1] + (self.no_heads, -1))
        v = v.view(v.shape[:-1] + (self.no_heads, -1))

        q = q.transpose(-2, -3)
        k = k.transpose(-2, -3)
        v = v.transpose(-2, -3)

        if apply_scale:
            q = q / math.sqrt(self.c_hidden)

        return q, k, v

    def _wrap_up(self, o: torch.Tensor, q_x: torch.Tensor) -> torch.Tensor:
        if self.linear_g is not None:
            g = torch.sigmoid(self.linear_g(q_x))
            g = g.view(g.shape[:-1] + (self.no_heads, -1))
            o = o * g

        o = o.reshape(o.shape[:-2] + (-1,))
        return self.linear_o(o)

    def forward(
        self,
        q_x: torch.Tensor,
        kv_x: torch.Tensor,
        biases: list[torch.Tensor] | None = None,
        use_deepspeed_evo_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_lma: bool = False,
        lma_q_chunk_size: int = 1024,
        lma_kv_chunk_size: int = 4096,
        use_high_precision: bool = False,
    ) -> torch.Tensor:
        if biases is None:
            biases = []

        q, k, v = self._prep_qkv(q_x, kv_x)

        o = _attention(q, k, v, biases)
        o = o.transpose(-2, -3)
        return self._wrap_up(o, q_x)

def _attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    biases: list[torch.Tensor],
) -> torch.Tensor:
    scores = torch.einsum("...qc,...kc->...qk", query, key)
    for b in biases:
        scores = scores + b
    scores = F.softmax(scores, dim=-1)
    return torch.einsum("...qk,...kc->...qc", scores.to(dtype=value.dtype), value)

def _permute_final_dims(tensor: torch.Tensor, inds: tuple[int, ...]) -> torch.Tensor:
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])

class TriangleMultiplicativeUpdate(nn.Module):
    """AF3 Algorithms 12/13: Triangle multiplicative update."""
    def __init__(self, c_z: int, c_hidden: int, _outgoing: bool = True):
        super().__init__()
        self.c_z = c_z
        self.c_hidden = c_hidden
        self._outgoing = _outgoing

        self.linear_a_p = Linear(c_z, c_hidden, bias=False)
        self.linear_a_g = Linear(c_z, c_hidden, bias=False)
        self.linear_b_p = Linear(c_z, c_hidden, bias=False)
        self.linear_b_g = Linear(c_z, c_hidden, bias=False)

        self.linear_g = Linear(c_z, c_z, bias=False)
        self.linear_z = Linear(c_hidden, c_z, bias=False)

        self.layer_norm_in = LayerNorm(c_z)
        self.layer_norm_out = LayerNorm(c_hidden)

    def _combine_projections(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if self._outgoing:
            a = _permute_final_dims(a, (2, 0, 1))
            b = _permute_final_dims(b, (2, 1, 0))
        else:
            a = _permute_final_dims(a, (2, 1, 0))
            b = _permute_final_dims(b, (2, 0, 1))
        p = torch.einsum("...ij,...jk->...ik", a, b)
        return _permute_final_dims(p, (1, 2, 0))

    def forward(
        self,
        z: torch.Tensor,
        mask: torch.Tensor | None = None,
        inplace_safe: bool = False,
        use_cueq_triangle_kernels: bool = False,
        _add_with_inplace: bool = False,
        _inplace_chunk_size: int | None = 256,
    ) -> torch.Tensor:
        if mask is None:
            mask = z.new_ones(z.shape[:-1])
        mask = mask.unsqueeze(-1)

        z_ln = self.layer_norm_in(z)

        a = mask * torch.sigmoid(self.linear_a_g(z_ln)) * self.linear_a_p(z_ln)
        b = mask * torch.sigmoid(self.linear_b_g(z_ln)) * self.linear_b_p(z_ln)

        x = self._combine_projections(a, b)
        x = self.layer_norm_out(x)
        x = self.linear_z(x)
        x = x * torch.sigmoid(self.linear_g(z_ln))
        return x

class TriangleMultiplicationOutgoing(TriangleMultiplicativeUpdate):
    """AF3 Algorithm 12."""
    def __init__(self, c_z: int, c_hidden: int):
        super().__init__(c_z=c_z, c_hidden=c_hidden, _outgoing=True)

class TriangleMultiplicationIncoming(TriangleMultiplicativeUpdate):
    """AF3 Algorithm 13."""
    def __init__(self, c_z: int, c_hidden: int):
        super().__init__(c_z=c_z, c_hidden=c_hidden, _outgoing=False)

class PairBlock(nn.Module):
    """Shared pair stack block for AF3 PairFormer / MSA module / template."""
    def __init__(
        self,
        c_z: int,
        c_hidden_mul: int,
        c_hidden_pair_att: int,
        no_heads_pair: int,
        transition_n: int,
        pair_dropout: float = 0.0,
        fuse_projection_weights: bool = False,
        inf: float = 1e9,
    ):
        super().__init__()
        self.tri_mul_out = TriangleMultiplicationOutgoing(c_z, c_hidden_mul)
        self.tri_mul_in = TriangleMultiplicationIncoming(c_z, c_hidden_mul)
        self.tri_att_start = TriangleAttention(
            c_z, c_hidden_pair_att, no_heads_pair, starting=True, inf=inf,
        )
        self.tri_att_end = TriangleAttention(
            c_z, c_hidden_pair_att, no_heads_pair, starting=False, inf=inf,
        )
        self.pair_transition = SwiGLUTransition(c_in=c_z, n=transition_n)

    def forward(
        self,
        z: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size: int | None = None,
        use_deepspeed_evo_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
        _mask_trans: bool = True,
        _attn_chunk_size: int | None = None,
    ) -> torch.Tensor:
        pair_trans_mask = pair_mask if _mask_trans else None

        z = z + self.tri_mul_out(z, mask=pair_mask)
        z = z + self.tri_mul_in(z, mask=pair_mask)
        z = z + self.tri_att_start(z, mask=pair_mask)
        z = z + self.tri_att_end(z, mask=pair_mask)
        z = z + self.pair_transition(z, mask=pair_trans_mask)
        return z

class MSARowAttentionWithPairBias(nn.Module):
    """AF3 MSA Pair-Weighted Averaging (Algorithm 10)."""
    def __init__(
        self,
        c_m: int,
        c_z: int,
        c_hidden: int,
        no_heads: int,
        inf: float = 1e9,
    ):
        super().__init__()
        self.c_m = c_m
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.inf = inf

        self.layer_norm_m = LayerNorm(c_m)
        self.layer_norm_z = LayerNorm(c_z)
        self.linear_z = Linear(c_z, no_heads, bias=False)

        self.linear_v = Linear(c_m, c_hidden * no_heads, bias=False)
        self.linear_g = Linear(c_m, c_hidden * no_heads, bias=False)
        self.linear_o = Linear(c_hidden * no_heads, c_m, bias=False)

        self.sigmoid = Sigmoid()
        self.softmax = Softmax(dim=-1)

    def forward(
        self,
        m: torch.Tensor,
        z: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        chunk_size: int | None = None,
        use_deepspeed_evo_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
    ) -> torch.Tensor:
        if z is None:
            return m

        if mask is None:
            mask = z.new_ones(z.shape[:-1])

        n_res = z.shape[-2]
        mask_bias = (self.inf * (mask - 1))[..., None, None, :, :]
        z_norm = self.layer_norm_z(z)
        z_proj = self.linear_z(z_norm)
        z_weights = _permute_final_dims(z_proj, (2, 0, 1)).unsqueeze(-4)
        z_weights = z_weights + mask_bias
        z_weights = self.softmax(z_weights)

        m = self.layer_norm_m(m)

        v = self.linear_v(m)
        v = v.view(v.shape[:-1] + (self.no_heads, -1))
        v = v.transpose(-2, -3)  # [*, N_seq, H, N_res, C_hidden]

        o = torch.einsum("...hqk,...hkc->...qhc", z_weights, v)
        g = self.sigmoid(self.linear_g(m))
        g = g.view(g.shape[:-1] + (self.no_heads, -1))
        o = o * g

        o = o.reshape(o.shape[:-2] + (-1,))
        o = self.linear_o(o)
        return o

class MSAModuleBlock(nn.Module):
    """Single block of AF3 Algorithm 8."""
    def __init__(
        self,
        c_m: int,
        c_z: int,
        c_hidden_msa_att: int,
        c_hidden_opm: int,
        c_hidden_mul: int,
        c_hidden_pair_att: int,
        no_heads_msa: int,
        no_heads_pair: int,
        transition_n: int,
        msa_dropout: float = 0.0,
        pair_dropout: float = 0.0,
        opm_first: bool = True,
        inf: float = 1e9,
        eps: float = 1e-3,
        last_block: bool = False,
    ):
        super().__init__()
        self.opm_first = opm_first
        self.skip_msa_update = last_block and opm_first

        if not self.skip_msa_update:
            self.msa_att_row = MSARowAttentionWithPairBias(
                c_m=c_m, c_z=c_z,
                c_hidden=c_hidden_msa_att,
                no_heads=no_heads_msa,
                inf=inf,
            )
            self.msa_transition = SwiGLUTransition(c_in=c_m, n=transition_n)

        self.outer_product_mean = OuterProductMean(
            c_m=c_m, c_z=c_z, c_hidden=c_hidden_opm, eps=eps,
        )

        self.pair_stack = PairBlock(
            c_z=c_z,
            c_hidden_mul=c_hidden_mul,
            c_hidden_pair_att=c_hidden_pair_att,
            no_heads_pair=no_heads_pair,
            transition_n=transition_n,
            pair_dropout=pair_dropout,
            inf=inf,
        )

    def forward(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        msa_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size: int | None = None,
        use_deepspeed_evo_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
        _mask_trans: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.opm_first:
            z = z + self.outer_product_mean(m, mask=msa_mask)

        if not self.skip_msa_update:
            m = m + self.msa_att_row(m, z=z, mask=pair_mask)
            m = m + self.msa_transition(m)

        if not self.opm_first:
            z = z + self.outer_product_mean(m, mask=msa_mask)

        z = self.pair_stack(z=z, pair_mask=pair_mask)
        return m, z


# ------------------------------
# Entry point: ModelNew (with expanded constructor to match evaluator)
# ------------------------------

class ModelNew(nn.Module):
    """Stacked AF3 Algorithm 8 blocks, with Triton-accelerated final linear in SwiGLUTransition."""
    def __init__(
        self,
        c_m: int,
        c_z: int,
        c_hidden_msa_att: int,
        c_hidden_opm: int,
        c_hidden_mul: int,
        c_hidden_pair_att: int,
        no_heads_msa: int,
        no_heads_pair: int,
        no_blocks: int,
        transition_n: int,
        msa_dropout: float = 0.0,
        pair_dropout: float = 0.0,
        opm_first: bool = True,
        fuse_projection_weights: bool = False,
        blocks_per_ckpt: int | None = None,
        inf: float = 1e9,
        eps: float = 1e-3,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            MSAModuleBlock(
                c_m=c_m, c_z=c_z,
                c_hidden_msa_att=c_hidden_msa_att,
                c_hidden_opm=c_hidden_opm,
                c_hidden_mul=c_hidden_mul,
                c_hidden_pair_att=c_hidden_pair_att,
                no_heads_msa=no_heads_msa,
                no_heads_pair=no_heads_pair,
                transition_n=transition_n,
                msa_dropout=msa_dropout,
                pair_dropout=pair_dropout,
                opm_first=opm_first,
                inf=inf,
                eps=eps,
                last_block=(i == no_blocks - 1),
            )
            for i in range(no_blocks)
        ])

    def forward(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        msa_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            m:        [*, N_seq, N_res, C_m] MSA embedding
            z:        [*, N_res, N_res, C_z] pair embedding
            msa_mask: [*, N_seq, N_res] MSA mask
            pair_mask:[*, N_res, N_res] pair mask

        Returns:
            (m, z): updated MSA and pair embeddings
        """
        for block in self.blocks:
            m, z = block(m=m, z=z, msa_mask=msa_mask, pair_mask=pair_mask)
        return m, z


# ------------------------------
# SwiGLUTransition with Triton-accelerated final linear
# ------------------------------

class SwiGLUTransition(nn.Module):
    """SwiGLU-based transition: LN -> SwiGLU -> Linear (Triton-backed)."""
    def __init__(self, c_in: int, n: int):
        super().__init__()
        self.c_in = c_in
        self.n = n

        self.layer_norm = LayerNorm(c_in)
        self.swiglu = SwiGLU(c_in, n * c_in)
        # Use Triton-backed Linear for the final matmul
        self.linear_out = Linear(n * c_in, c_in, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        chunk_size: int | None = None,
        ckpt_chunk_size: int | None = None,
    ) -> torch.Tensor:
        if mask is None:
            mask = x.new_ones(x.shape[:-1])
        mask = mask.unsqueeze(-1)

        x = self.layer_norm(x)
        x = self.swiglu(x)
        # Use Triton linear if possible
        if _HAS_TRITON and x.is_cuda and self.linear_out.weight.is_cuda and x.dtype in (torch.float32, torch.bfloat16):
            w = self.linear_out.weight
            b = self.linear_out.bias  # might be None
            # Ensure dtype/device match
            if w.dtype != x.dtype:
                w = w.to(x.dtype)
            if b is not None and b.dtype != x.dtype:
                b = b.to(x.dtype)
            # Reshape weight to [K, N] and make contiguous
            w_t = w.t().contiguous()
            out = _triton_linear(x.contiguous(), w_t, b, block_m=64, block_n=64, block_k=32)
        else:
            out = F.linear(x, self.linear_out.weight, self.linear_out.bias)

        return out * mask


# ------------------------------
# Helper functions used by modules
# ------------------------------

def _permute_final_dims(tensor: torch.Tensor, inds: tuple[int, ...]) -> torch.Tensor:
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])

MSAModuleStack = ModelNew
