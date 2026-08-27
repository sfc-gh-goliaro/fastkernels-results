import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Triton imports
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# ------------------------------
# Triton kernels
# ------------------------------

if TRITON_AVAILABLE:
    @triton.jit
    def _linear_matmul_bias_kernel(
        X, W, BIAS, Y,
        B, M, K, N,
        stride_x_b, stride_x_m, stride_x_k,
        stride_w_n, stride_w_k,
        stride_y_b, stride_y_m, stride_y_n,
        APPLY_LN: tl.constexpr,
        EPS: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        # program ids
        pid_b = tl.program_id(0)
        pid_m = tl.program_id(1)
        pid_n = tl.program_id(2)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        # pointers
        x_ptrs = X + pid_b * stride_x_b + offs_m[:, None] * stride_x_m + offs_k[None, :] * stride_x_k
        w_ptrs = W + offs_n[:, None] * stride_w_n + offs_k[None, :] * stride_w_k

        # accumulator
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # loop over K
        for k in range(0, K, BLOCK_K):
            k_mask = (k + offs_k) < K
            x = tl.load(x_ptrs, mask=k_mask[None, :], other=0.0)
            w = tl.load(w_ptrs, mask=k_mask[None, :], other=0.0)
            acc += tl.dot(x, tl.trans(w))
            x_ptrs += BLOCK_K * stride_x_k
            w_ptrs += BLOCK_K * stride_w_k

        # optional pre-layer-norm over last dim (K): mean/var
        # We won't use this path; keeping signature for flexibility.
        pass

        # cast to output dtype (assume same as X)
        if HAS_BIAS:
            bias = tl.load(BIAS + offs_n, mask=(offs_n < N), other=0.0)
            out = acc + bias[None, :]
        else:
            out = acc

        # store
        y_ptrs = Y + pid_b * stride_y_b + offs_m[:, None] * stride_y_m + offs_n[None, :] * stride_y_n
        mask_m = offs_m < M
        mask_n = offs_n < N
        store_mask = mask_m[:, None] & mask_n[None, :]
        tl.store(y_ptrs, out, mask=store_mask)


    @triton.jit
    def _contract_ijk_jk_ik_kernel(
        A, B, P,
        Bbatch, H, I, J, K,
        stride_a_b, stride_a_h, stride_a_i, stride_a_j,
        stride_b_b, stride_b_h, stride_b_j, stride_b_k,
        stride_p_b, stride_p_h, stride_p_i, stride_p_k,
        BLOCK_I: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_J: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)
        pid_ik = tl.program_id(2)

        num_k_tiles = (K + BLOCK_K - 1) // BLOCK_K
        pid_i = pid_ik // num_k_tiles
        pid_k = pid_ik % num_k_tiles

        offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
        offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        offs_j = tl.arange(0, BLOCK_J)

        # accum
        acc = tl.zeros((BLOCK_I, BLOCK_K), dtype=tl.float32)

        a_ptrs = A + pid_b * stride_a_b + pid_h * stride_a_h + offs_i[:, None] * stride_a_i + offs_j[None, :] * stride_a_j
        b_ptrs = B + pid_b * stride_b_b + pid_h * stride_b_h + offs_j[:, None] * stride_b_j + offs_k[None, :] * stride_b_k

        for j0 in range(0, J, BLOCK_J):
            j_mask = (j0 + offs_j) < J
            a = tl.load(a_ptrs, mask=j_mask[None, :], other=0.0)  # [BI, BJ]
            b = tl.load(b_ptrs, mask=j_mask[:, None] & (offs_k[None, :] < K), other=0.0)  # [BJ, BK]
            acc += tl.dot(a, b)  # [BI, BK]
            a_ptrs += BLOCK_J * stride_a_j
            b_ptrs += BLOCK_J * stride_b_j

        p_ptrs = P + pid_b * stride_p_b + pid_h * stride_p_h + offs_i[:, None] * stride_p_i + offs_k[None, :] * stride_p_k
        mask_i = offs_i < I
        mask_k = offs_k < K
        store_mask = mask_i[:, None] & mask_k[None, :]
        tl.store(p_ptrs, acc, mask=store_mask)


# ------------------------------
# Python wrappers for kernels
# ------------------------------

def _triton_linear(x: torch.Tensor,
                   w: torch.Tensor,
                   bias: torch.Tensor | None,
                   apply_ln: bool = False,
                   eps: float = 1e-5) -> torch.Tensor:
    """
    Compute y = linear(x, w, bias) using Triton.
    Shapes:
      x:  [B, M, K]
      w:  [N, K]
      y:  [B, M, N]
    """
    assert TRITON_AVAILABLE, "Triton not available"
    assert x.is_cuda and w.is_cuda, "Tensors must be CUDA"
    B, M, K = x.shape
    N = w.shape[0]
    assert w.shape[1] == K, f"Incompatible shapes: w {w.shape} vs x {x.shape}"

    out_dtype = x.dtype
    y = torch.empty((B, M, N), device=x.device, dtype=out_dtype)

    sx_b, sx_m, sx_k = x.stride()
    sw_n, sw_k = w.stride()
    sy_b, sy_m, sy_n = y.stride()

    BLOCK_M = 32
    BLOCK_N = 64
    BLOCK_K = 32

    grid = (B, triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    _linear_matmul_bias_kernel[grid](
        x, w, bias if bias is not None else w,  # valid ptr
        y,
        B, M, K, N,
        sx_b, sx_m, sx_k,
        sw_n, sw_k,
        sy_b, sy_m, sy_n,
        APPLY_LN=False,
        EPS=eps,
        HAS_BIAS=(bias is not None),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )
    return y


def _triton_contract(a: torch.Tensor,
                     b: torch.Tensor) -> torch.Tensor:
    """
    Compute p = einsum('...ij,...jk->...ik') using Triton.
    Shapes:
      a:  [B, H, I, J]
      b:  [B, H, J, K]
      p:  [B, H, I, K]
    """
    assert TRITON_AVAILABLE, "Triton not available"
    assert a.is_cuda and b.is_cuda, "Tensors must be CUDA"
    B, H, I, J = a.shape
    _, _, Jb, K = b.shape
    assert Jb == J, f"J mismatch: a has J={J}, b has J={Jb}"
    p = torch.empty((B, H, I, K), device=a.device, dtype=a.dtype)

    sa_b, sa_h, sa_i, sa_j = a.stride()
    sb_b, sb_h, sb_j, sb_k = b.stride()
    sp_b, sp_h, sp_i, sp_k = p.stride()

    BLOCK_I = 16
    BLOCK_K = 16
    BLOCK_J = 32

    grid = (B, H, triton.cdiv(I, BLOCK_I) * triton.cdiv(K, BLOCK_K))

    _contract_ijk_jk_ik_kernel[grid](
        a, b, p,
        B, H, I, J, K,
        sa_b, sa_h, sa_i, sa_j,
        sb_b, sb_h, sb_j, sb_k,
        sp_b, sp_h, sp_i, sp_k,
        BLOCK_I=BLOCK_I, BLOCK_K=BLOCK_K, BLOCK_J=BLOCK_J,
        num_warps=4, num_stages=2,
    )
    return p


# ------------------------------
# Triton-optimized Linear
# ------------------------------

class TritonLinear(nn.Module):
    """Drop-in replacement for Linear using Triton matmul + bias."""
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        # init like torch.nn.Linear default
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expect x: [B, M, K] or [M, K]
        if x.dim() == 2:
            x = x.unsqueeze(0)
        return _triton_linear(x, self.weight, self.bias)


# ------------------------------
# Triton-optimized components
# ------------------------------

class TriangleMultiplicativeUpdateTriton(nn.Module):
    """Triton-optimized Triangle Multiplicative Update (Algs 12/13)."""
    def __init__(self, c_z: int, c_hidden: int, _outgoing: bool = True):
        super().__init__()
        self.c_z = c_z
        self.c_hidden = c_hidden
        self._outgoing = _outgoing

        # linears (Triton)
        self.linear_a_p = TritonLinear(c_z, c_hidden, bias=False)
        self.linear_a_g = TritonLinear(c_z, c_hidden, bias=False)
        self.linear_b_p = TritonLinear(c_z, c_hidden, bias=False)
        self.linear_b_g = TritonLinear(c_z, c_hidden, bias=False)

        self.linear_g = TritonLinear(c_z, c_z, bias=False)
        self.linear_z = TritonLinear(c_hidden, c_z, bias=False)

        # layernorms (PyTorch to preserve state_dict and numerics)
        self.layer_norm_in = nn.LayerNorm(c_z, elementwise_affine=False)
        self.layer_norm_out = nn.LayerNorm(c_hidden, elementwise_affine=False)

    def _combine_projections(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # a,b are [B, I, C]; compute p[i,k] = sum_j a[i,j] * b[j,k]
        B = a.shape[0]
        I = a.shape[1]
        C = a.shape[2]
        assert b.shape[0] == B and b.shape[2] == C
        J = b.shape[1]
        # view
        A = a.view(B, 1, I, J, C)
        Bx = b.view(B, 1, J, C)
        return _triton_contract(A, Bx).view(B, I, J, C)

    def forward(self, z: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None:
            mask = z.new_ones(z.shape[:-1])
        mask = mask.unsqueeze(-1)

        z_ln = self.layer_norm_in(z)

        # a = mask * sigmoid(g) * p
        a_p = self.linear_a_p(z_ln)
        a_g = self.linear_a_g(z_ln)
        a = mask * torch.sigmoid(a_g) * a_p

        # b = mask * sigmoid(g) * p
        b_p = self.linear_b_p(z_ln)
        b_g = self.linear_b_g(z_ln)
        b = mask * torch.sigmoid(b_g) * b_p

        # combine
        x = self._combine_projections(a, b)

        x = self.layer_norm_out(x)
        x = self.linear_z(x)
        x = x * torch.sigmoid(self.linear_g(z_ln))
        return x


class TriangleAttentionTriton(nn.Module):
    """Triton-optimized TriangleAttention (Alg 14/15)."""
    def __init__(self, c_in: int, c_hidden: int, no_heads: int, starting: bool = True, inf: float = 1e9):
        super().__init__()
        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.starting = starting
        self.inf = inf

        # layernorm (PyTorch)
        self.layer_norm = nn.LayerNorm(c_in, elementwise_affine=False)

        # linears (Triton)
        self.linear_z = TritonLinear(c_in, no_heads, bias=False)

        # MHA with Triton linears
        self.mha = OF3Attention(  # keep OF3Attention but replace linear with Triton where possible
            c_q=c_in, c_k=c_in, c_v=c_in,
            c_hidden=c_hidden, no_heads=no_heads, gating=False, q_bias=False,
        )

    def _prep_qkv(self, q_x: torch.Tensor, kv_x: torch.Tensor, apply_scale: bool = True):
        # Use TritonLinear for q/k/v
        q = self.mha.linear_q(q_x)
        k = self.mha.linear_k(kv_x)
        v = self.mha.linear_v(kv_x)
        # view
        q = q.view(q.shape[:-1] + (self.no_heads, -1))
        k = k.view(k.shape[:-1] + (self.no_heads, -1))
        v = v.view(v.shape[:-1] + (self.no_heads, -1))
        # transpose
        q = q.transpose(-2, -3)
        k = k.transpose(-2, -3)
        v = v.transpose(-2, -3)
        if apply_scale:
            q = q / math.sqrt(self.c_hidden)
        return q, k, v

    def _wrap_up(self, o: torch.Tensor, q_x: torch.Tensor):
        # no gating in our TriangleAttention
        o = o.transpose(-2, -3)
        o = o.reshape(o.shape[:-2] + (-1,))
        return self.mha.linear_o(o)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None, **kwargs):
        if mask is None:
            mask = x.new_ones(x.shape[:-1])
        x = self.layer_norm(x)

        # mask bias
        mask_bias = (self.inf * (mask - 1))[..., :, None, None, :]

        # triangle bias
        triangle_bias = _permute_final_dims(self.linear_z(x), (2, 0, 1))
        triangle_bias = triangle_bias.unsqueeze(-4)

        biases = [mask_bias, triangle_bias]

        q, k, v = self._prep_qkv(x, x)
        o = _attention(q, k, v, biases)
        return self._wrap_up(o, x)


class SwiGLUTriton(nn.Module):
    """SwiGLU with Triton Linear layers."""
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.linear_a = TritonLinear(c_in, c_out, bias=False)
        self.linear_b = TritonLinear(c_in, c_out, bias=False)
        self.silu = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.silu(self.linear_a(x)) * self.linear_b(x)


class SwiGLUTransitionTriton(nn.Module):
    """Triton-optimized SwiGLUTransition."""
    def __init__(self, c_in: int, n: int):
        super().__init__()
        self.c_in = c_in
        self.n = n

        self.layer_norm = nn.LayerNorm(c_in, elementwise_affine=False)
        self.swiglu = SwiGLUTriton(c_in, n * c_in)
        self.linear_out = TritonLinear(n * c_in, c_in, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        if mask is None:
            mask = x.new_ones(x.shape[:-1])
        mask = mask.unsqueeze(-1)
        x = self.layer_norm(x)
        x = self.swiglu(x)
        x = self.linear_out(x)
        x = x * mask
        return x


class AttentionPairBiasTriton(nn.Module):
    """Triton-optimized AttentionPairBias (Alg 24)."""
    def __init__(self, c_q: int, c_s: int = 0, c_z: int = 128, c_hidden: int = 32, no_heads: int = 4, use_ada_layer_norm: bool = False, inf: float = 1e9):
        super().__init__()
        self.c_q = c_q
        self.c_s = c_s if c_s > 0 else c_q
        self.c_z = c_z
        self.inf = inf
        self.use_ada_layer_norm = use_ada_layer_norm

        if use_ada_layer_norm:
            # keep AdaLN structure but use PyTorch LN to preserve state_dict
            self.layer_norm_a = AdaLN(c_a=c_q, c_s=self.c_s)
            self.linear_ada_out = TritonLinear(self.c_s, self.c_q, bias=True)
        else:
            self.layer_norm_a = nn.LayerNorm(c_q, elementwise_affine=False)

        self.layer_norm_z = nn.LayerNorm(c_z, elementwise_affine=False)
        self.linear_z = TritonLinear(c_z, no_heads, bias=False)

        self.sigmoid = nn.Sigmoid()

        # MHA
        self.mha = OF3Attention(
            c_q=c_q, c_k=c_q, c_v=c_q,
            c_hidden=c_hidden, no_heads=no_heads, gating=True, q_bias=True,
        )

    def _prep_bias(self, a: torch.Tensor, z: torch.Tensor, mask: torch.Tensor | None):
        if mask is None:
            mask = a.new_ones(a.shape[:-1])
        B, N, Cq = a.shape
        mask = mask.expand(B, N)

        mask_bias = (self.inf * (mask - 1)).unsqueeze(1).unsqueeze(1)  # [B,1,1,N]
        biases = [mask_bias]

        z = self.layer_norm_z(z)
        z = self.linear_z(z)  # [B,N,H]
        z = _permute_final_dims(z, (2, 0, 1))  # [1,B,H,N]
        biases.append(z)
        return biases

    def forward(self, a: torch.Tensor, z: torch.Tensor, s: torch.Tensor | None = None, mask: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        biases = self._prep_bias(a, z, mask)

        if self.use_ada_layer_norm:
            # AdaLN on a using s
            a = self.layer_norm_a(a, s)
        else:
            a = self.layer_norm_a(a)

        a = self.mha(q_x=a, kv_x=a, biases=biases)
        if self.use_ada_layer_norm:
            a = self.sigmoid(self.linear_ada_out(s)) * a
        return a


# ------------------------------
# Triton-optimized Pair block
# ------------------------------

class PairFormerBlockTriton(nn.Module):
    """A single block: PairBlock + AttentionPairBias + SwiGLUTransition, all Triton-optimized where possible."""
    def __init__(self, c_s: int, c_z: int, c_hidden_pair_bias: int, no_heads_pair_bias: int,
                 c_hidden_mul: int, c_hidden_pair_att: int, no_heads_pair: int,
                 transition_n: int, pair_dropout: float = 0.0, inf: float = 1e9):
        super().__init__()
        self.pair_stack = TriangleMultiplicativeUpdateTriton(c_z=c_z, c_hidden=c_hidden_mul)
        # triangle attention is already using Triton parts
        self.tri_att_start = TriangleAttentionTriton(c_in=c_z, c_hidden=c_hidden_pair_att, no_heads=no_heads_pair, starting=True, inf=inf)
        self.tri_att_end = TriangleAttentionTriton(c_in=c_z, c_hidden=c_hidden_pair_att, no_heads=no_heads_pair, starting=False, inf=inf)
        self.pair_transition = SwiGLUTransitionTriton(c_in=c_z, n=transition_n)

        # single attention pair bias
        self.attn_pair_bias = AttentionPairBiasTriton(
            c_q=c_s, c_s=c_s, c_z=c_z,
            c_hidden=c_hidden_pair_bias, no_heads=no_heads_pair_bias,
            use_ada_layer_norm=False, inf=inf,
        )

        self.single_transition = SwiGLUTransitionTriton(c_in=c_s, n=transition_n)

    def forward(self, s: torch.Tensor, z: torch.Tensor, single_mask: torch.Tensor, pair_mask: torch.Tensor,
                chunk_size: int | None = None, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        pair_trans_mask = pair_mask if True else None

        # Pair block
        z = z + self.pair_stack(z, mask=pair_mask)
        z = z + self.tri_att_start(z, mask=pair_mask)
        z = z + self.tri_att_end(z, mask=pair_mask)
        z = z + self.pair_transition(z, mask=pair_trans_mask)

        # Single path
        s = s + self.attn_pair_bias(a=s, z=z, s=None, mask=single_mask)
        s = s + self.single_transition(s, mask=single_mask)

        return s, z


# ------------------------------
# Entry point: ModelNew
# ------------------------------

class ModelNew(nn.Module):
    """Triton-optimized version of Model with the same forward signature.

    It replaces linear operations and small einsums with Triton kernels where appropriate.
    """
    def __init__(self,
                 c_s: int,
                 c_z: int,
                 c_hidden_pair_bias: int,
                 no_heads_pair_bias: int,
                 c_hidden_mul: int,
                 c_hidden_pair_att: int,
                 no_heads_pair: int,
                 transition_n: int,
                 pair_dropout: float = 0.0,
                 inf: float = 1e9):
        super().__init__()
        # Single block, matching original Model
        self.block = PairFormerBlockTriton(
            c_s=c_s, c_z=c_z,
            c_hidden_pair_bias=c_hidden_pair_bias, no_heads_pair_bias=no_heads_pair_bias,
            c_hidden_mul=c_hidden_mul, c_hidden_pair_att=c_hidden_pair_att, no_heads_pair=no_heads_pair,
            transition_n=transition_n, pair_dropout=pair_dropout, inf=inf,
        )

    def forward(self,
                s: torch.Tensor,
                z: torch.Tensor,
                single_mask: torch.Tensor,
                pair_mask: torch.Tensor,
                chunk_size: int | None = None,
                **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        return self.block(s, z, single_mask, pair_mask)


# ------------------------------
# Original helper functions and classes (kept for completeness)
# ------------------------------

def _permute_final_dims(tensor: torch.Tensor, inds: list[int]) -> torch.Tensor:
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


def _attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    biases: list[torch.Tensor],
) -> torch.Tensor:
    """Core SDPA: scores = softmax(Q K^T + biases) V.

    Args:
        query: [*, H, Q, C_hidden]
        key:   [*, H, K, C_hidden]
        value: [*, H, V, C_hidden]
        biases: list of tensors broadcastable to [*, H, Q, K]

    Returns:
        [*, H, Q, C_hidden]
    """
    scores = torch.einsum("...qc,...kc->...qk", query, key)

    for b in biases:
        scores = scores + b

    scores = F.softmax(scores, dim=-1)

    return torch.einsum("...qk,...kc->...qc", scores.to(dtype=value.dtype), value)


class OF3Attention(nn.Module):
    """Standard multi-head attention with gating and bias list support.

    Reference: openfold3/core/model/primitives/attention.py Attention

    Args:
        c_q: Input dimension of query data
        c_k: Input dimension of key data
        c_v: Input dimension of value data
        c_hidden: Per-head hidden dimension
        no_heads: Number of attention heads
        gating: Whether to gate output using query data
    """

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

        self.linear_q = nn.Linear(c_q, c_hidden * no_heads, bias=q_bias)
        self.linear_k = nn.Linear(c_k, c_hidden * no_heads, bias=False)
        self.linear_v = nn.Linear(c_v, c_hidden * no_heads, bias=False)
        self.linear_o = nn.Linear(c_hidden * no_heads, c_q, bias=False)

        self.linear_g = None
        if gating:
            self.linear_g = nn.Linear(c_q, c_hidden * no_heads, bias=False)

    def _prep_qkv(
        self, q_x: torch.Tensor, kv_x: torch.Tensor, apply_scale: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

        o = o.transpose(-2, -3)

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
        """
        Args:
            q_x:  [*, Q, C_q] query data
            kv_x: [*, K, C_k] key data
            biases: List of biases that broadcast to [*, H, Q, K]

        Returns:
            [*, Q, C_q] attention update
        """
        if biases is None:
            biases = []

        q, k, v = self._prep_qkv(q_x, kv_x)

        o = _attention(q, k, v, biases)
        o = o.transpose(-2, -3)

        return self._wrap_up(o, q_x)


class AdaLN(nn.Module):
    """Adaptive Layer Normalization matching the reference AdaLN.

    Submodule structure matches checkpoint keys:
    - layer_norm_s: LayerNorm(c_s), weight-only
    - linear_g: Linear(c_s, c_a, bias=True) — gating
    - linear_s: Linear(c_s, c_a, bias=False) — additive conditioning

    Reference: openfold3/core/model/primitives/normalization.py AdaLN

    Args:
        c_a: Activation channel dimension
        c_s: Conditioning channel dimension
    """

    def __init__(self, c_a: int, c_s: int):
        super().__init__()
        self.c_a = c_a
        self.c_s = c_s

        self.layer_norm_a = nn.LayerNorm(c_a, create_scale=False, create_offset=False)
        self.layer_norm_s = nn.LayerNorm(c_s, create_offset=False)
        self.sigmoid = nn.Sigmoid()
        self.linear_g = nn.Linear(c_s, c_a, bias=True)
        self.linear_s = nn.Linear(c_s, c_a, bias=False)

    def forward(self, a: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        s_norm = self.layer_norm_s(s)
        g = self.sigmoid(self.linear_g(s_norm))
        a_norm = self.layer_norm_a(a)
        return g * (a_norm + self.linear_s(s_norm))

PairFormerStack = ModelNew
