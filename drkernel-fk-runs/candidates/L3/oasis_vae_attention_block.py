from math import pi
from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# Triton fused QKV matmul+bias
# ---------------------------

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


def _ceil_div(a, b):
    return (a + b - 1) // b


@triton.jit
def _qkv_matmul_bias_kernel(
    x_ptr,          # *f16 [B, L, D]
    w_ptr,          # *f16 [3D, D]
    b_q_ptr,        # *f16 [D]
    b_k_ptr,        # *f16 [D]
    b_v_ptr,        # *f16 [D]
    out_q_ptr,      # *f16 [B, L, D]
    out_k_ptr,      # *f16 [B, L, D]
    out_v_ptr,      # *f16 [B, L, D]
    B: tl.constexpr,
    L: tl.constexpr,
    D: tl.constexpr,
    stride_x_b: tl.constexpr,
    stride_x_l: tl.constexpr,
    stride_x_d: tl.constexpr,
    stride_w_m: tl.constexpr,
    stride_w_n: tl.constexpr,
    stride_out_b: tl.constexpr,
    stride_out_l: tl.constexpr,
    stride_out_d: tl.constexpr,
    BLOCK_N: tl.constexpr,  # tile over D
    BLOCK_K: tl.constexpr,  # tile over D (reduction)
):
    # Program ids
    pid_b = tl.program_id(0)
    pid_l = tl.program_id(1)
    pid_d_blk = tl.program_id(2)

    # Offsets
    d_offsets = pid_d_blk * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    d_mask = d_offsets < D

    # Accumulator
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    # Loop over reduction dimension K = D in BLOCK_K tiles
    for k in range(0, D, BLOCK_K):
        k_offsets = k + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        k_mask = k_offsets < D

        # Load x[b, l, k_offsets] -> [BLOCK_K]
        x_vals = tl.load(
            x_ptr + pid_b * stride_x_b + pid_l * stride_x_l + k_offsets * stride_x_d,
            mask=k_mask,
            other=0.0,
        ).to(tl.float32)  # [BLOCK_K], fp32

        # Load weight blocks: q (rows 0:D), k (rows D:2D), v (rows 2D:3D) -> [BLOCK_N, BLOCK_K]
        # Construct pointers for w[:, k_offsets] slices at d_offsets
        wq_ptrs = w_ptr + 0 * D * D + d_offsets[:, None] * stride_w_m + k_offsets[None, :] * stride_w_n
        wk_ptrs = w_ptr + 1 * D * D + d_offsets[:, None] * stride_w_m + k_offsets[None, :] * stride_w_n
        wv_ptrs = w_ptr + 2 * D * D + d_offsets[:, None] * stride_w_m + k_offsets[None, :] * stride_w_n

        mask = (d_offsets[:, None] < D) & (k_offsets[None, :] < D)
        wq = tl.load(wq_ptrs, mask=mask, other=0.0).to(tl.float32)  # [BLOCK_N, BLOCK_K]
        wk = tl.load(wk_ptrs, mask=mask, other=0.0).to(tl.float32)  # [BLOCK_N, BLOCK_K]
        wv = tl.load(wv_ptrs, mask=mask, other=0.0).to(tl.float32)  # [BLOCK_N, BLOCK_K]

        # Accumulate: acc += sum_k x[k] * w[:, k] for each output d
        # Do this by matrix multiply: [BLOCK_N, BLOCK_K] @ [BLOCK_K, 1] -> [BLOCK_N, 1], then squeeze
        # Equivalent to: for kk in range(BLOCK_K): acc += wq[:, kk] * x_vals[kk]
        # Vectorized: sum over axis=1 after elementwise multiply
        prod_q = wq * x_vals[None, :]          # [BLOCK_N, BLOCK_K]
        prod_k = wk * x_vals[None, :]          # [BLOCK_N, BLOCK_K]
        prod_v = wv * x_vals[None, :]          # [BLOCK_N, BLOCK_K]
        acc += tl.sum(prod_q, axis=1)
        acc += tl.sum(prod_k, axis=1)
        acc += tl.sum(prod_v, axis=1)

    # Add biases if provided
    bq = tl.load(b_q_ptr + d_offsets, mask=d_mask, other=0.0).to(tl.float32)
    bk = tl.load(b_k_ptr + d_offsets, mask=d_mask, other=0.0).to(tl.float32)
    bv = tl.load(b_v_ptr + d_offsets, mask=d_mask, other=0.0).to(tl.float32)
    acc_q = acc + bq
    acc_k = acc + bk
    acc_v = acc + bv

    # Store results as float16
    tl.store(
        out_q_ptr + pid_b * stride_out_b + pid_l * stride_out_l + d_offsets * stride_out_d,
        acc_q.to(tl.float16),
        mask=d_mask,
    )
    tl.store(
        out_k_ptr + pid_b * stride_out_b + pid_l * stride_out_l + d_offsets * stride_out_d,
        acc_k.to(tl.float16),
        mask=d_mask,
    )
    tl.store(
        out_v_ptr + pid_b * stride_out_b + pid_l * stride_out_l + d_offsets * stride_out_d,
        acc_v.to(tl.float16),
        mask=d_mask,
    )


def _triton_qkv_matmul_bias(x: torch.Tensor,
                            wqkv: torch.Tensor,
                            b_q: torch.Tensor | None,
                            b_k: torch.Tensor | None,
                            b_v: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    x: [B, L, D], float16
    wqkv: [3D, D], float16
    biases: [D] or None
    Returns (q, k, v) each [B, L, D], dtype float16
    """
    assert x.is_cuda, "Triton QKV requires CUDA tensor"
    assert wqkv.is_cuda, "Weights must be CUDA"
    B, L, D = x.shape
    assert wqkv.shape[0] == 3 * D and wqkv.shape[1] == D, f"Expected weight shape {(3*D, D)}, got {tuple(wqkv.shape)}"
    # Ensure contiguous
    x_c = x.contiguous()
    w_c = wqkv.contiguous()
    # Output
    out_dtype = x.dtype
    q = torch.empty((B, L, D), device=x.device, dtype=out_dtype)
    k = torch.empty((B, L, D), device=x.device, dtype=out_dtype)
    v = torch.empty((B, L, D), device=x.device, dtype=out_dtype)

    # Strides (elements)
    stride_x_b, stride_x_l, stride_x_d = x_c.stride()
    stride_w_m, stride_w_n = w_c.stride()  # (3D, D) row-major
    stride_out_b, stride_out_l, stride_out_d = q.stride()

    # Tile sizes
    BLOCK_N = 128
    BLOCK_K = 64

    grid = (B, L, _ceil_div(D, BLOCK_N))

    _qkv_matmul_bias_kernel[grid](
        x_c, w_c,
        (b_q if b_q is not None else torch.zeros(D, device=x.device, dtype=out_dtype)),
        (b_k if b_k is not None else torch.zeros(D, device=x.device, dtype=out_dtype)),
        (b_v if b_v is not None else torch.zeros(D, device=x.device, dtype=out_dtype)),
        q, k, v,
        B, L, D,
        stride_x_b, stride_x_l, stride_x_d,
        stride_w_m, stride_w_n,
        stride_out_b, stride_out_l, stride_out_d,
        BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )
    return q, k, v


# ---------------------------
# Original modules reused/adjusted
# ---------------------------

class Matmul(nn.Module):
    def forward(self, input, weight, bias=None):
        return F.linear(input, weight, bias)

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.matmul = Matmul()

    def forward(self, input):
        return self.matmul(input, self.weight, self.bias)

def _resolve_flash_attn_func():
    for mod in ("fa3_fwd_interface", "flash_attn_interface"):
        try:
            return __import__(mod, fromlist=["flash_attn_func"]).flash_attn_func
        except (ImportError, ModuleNotFoundError):
            pass
    from flash_attn import flash_attn_func
    return flash_attn_func

_CUDNN_MAX_HEAD_DIM = 128

class DenseAttention(nn.Module):
    def __init__(self, backend: Literal["auto", "sdpa", "flash_attn", "cudnn", "flex"] = "auto"):
        super().__init__()
        self.fa_func = None
        self.use_cudnn_kernel = False
        self.use_flex_kernel = False
        self._flex_fn = None

        if backend == "sdpa":
            return

        if backend == "cudnn":
            self.use_cudnn_kernel = True
            return

        if backend == "flex":
            from torch.nn.attention.flex_attention import flex_attention
            self.use_flex_kernel = True
            self._flex_fn = torch.compile(flex_attention, dynamic=False)
            return

        if backend == "flash_attn":
            self.fa_func = _resolve_flash_attn_func()
            return

        cc = (torch.cuda.get_device_capability()[0] * 10 + torch.cuda.get_device_capability()[1])
        if 80 <= cc < 100:
            self.fa_func = _resolve_flash_attn_func()
        elif cc >= 100:
            self.use_cudnn_kernel = True

    def forward(self, query, key, value, softmax_scale=None, causal=False, attn_mask: torch.Tensor | None = None):
        if self.fa_func is not None and attn_mask is None and query.dtype != torch.float32:
            out = self.fa_func(query, key, value, softmax_scale=softmax_scale, causal=causal)
            if isinstance(out, tuple):
                out = out[0]
            return out

        q = query.permute(0, 2, 1, 3)
        k = key.permute(0, 2, 1, 3)
        v = value.permute(0, 2, 1, 3)

        if self.use_flex_kernel:
            q = q.contiguous(); k = k.contiguous(); v = v.contiguous()
            raise NotImplementedError("FlexAttention path is not implemented here.")

        elif self.use_cudnn_kernel:
            from torch.nn.attention import sdpa_kernel, SDPBackend
            if attn_mask is not None and causal:
                raise ValueError("Pass either attn_mask or causal=True, not both.")
            q_shape = q.shape[-1]
            if q_shape > _CUDNN_MAX_HEAD_DIM:
                if attn_mask is not None and attn_mask.dtype not in (torch.bool, q.dtype):
                    attn_mask = attn_mask.to(dtype=q.dtype)
                with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                    out = F.scaled_dot_product_attention(
                        q, k, v,
                        attn_mask=attn_mask,
                        dropout_p=0.0,
                        is_causal=causal,
                        scale=softmax_scale,
                    )
            else:
                try:
                    with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
                        out = F.scaled_dot_product_attention(
                            q, k, v,
                            attn_mask=attn_mask,
                            dropout_p=0.0,
                            is_causal=causal,
                            scale=softmax_scale,
                        )
                except RuntimeError:
                    with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                        out = F.scaled_dot_product_attention(
                            q, k, v,
                            attn_mask=attn_mask,
                            dropout_p=0.0,
                            is_causal=causal,
                            scale=softmax_scale,
                        )
            return out.permute(0, 2, 1, 3)

        else:
            if attn_mask is not None and attn_mask.dtype != torch.bool:
                attn_mask = attn_mask.to(dtype=q.dtype)
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=0.0,
                is_causal=False if attn_mask is not None else causal,
                scale=softmax_scale,
            )
            return out.permute(0, 2, 1, 3)

# Rotary helpers
def oasis_rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = x.reshape(*x.shape[:-1], -1, 2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)

def oasis_apply_rotary_emb(freqs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    dtype = t.dtype
    rot_dim = freqs.shape[-1]
    t_left = t[..., :0]
    t_middle = t[..., :rot_dim]
    t_right = t[..., rot_dim:]
    t_transformed = (t_middle * freqs.cos()) + (oasis_rotate_half(t_middle) * freqs.sin())
    return torch.cat((t_left, t_transformed, t_right), dim=-1).to(dtype)

class OasisRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, freqs_for: str = "lang", theta: float = 10000.0, max_freq: float = 10.0):
        super().__init__()
        self.dim = dim
        self.freqs_for = freqs_for
        if freqs_for == "lang":
            freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        elif freqs_for == "pixel":
            freqs = torch.linspace(1.0, max_freq / 2, dim // 2) * pi
        else:
            raise ValueError(f"unsupported rotary mode: {freqs_for}")
        self.freqs = nn.Parameter(freqs, requires_grad=False)
        self.register_buffer("dummy", torch.tensor(0), persistent=False)

    @property
    def device(self) -> torch.device:
        return self.dummy.device

    def _forward_freqs(self, positions: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        freqs = torch.einsum("..., f -> ... f", positions.to(freqs.dtype), freqs)
        return freqs.repeat_interleave(2, dim=-1)

    def forward(self, t: torch.Tensor, freqs: torch.Tensor, seq_len: int | None = None, offset: int = 0) -> torch.Tensor:
        del seq_len, offset
        return self._forward_freqs(t, freqs)

    def rotate_queries_or_keys(self, t: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        seq_len = t.shape[-2]
        positions = torch.arange(seq_len, device=t.device, dtype=t.dtype)
        seq_freqs = self.forward(positions, freqs, seq_len=seq_len)
        return oasis_apply_rotary_emb(seq_freqs, t)

    def get_axial_freqs(self, *dims: int) -> torch.Tensor:
        colon = slice(None)
        all_freqs = []
        for index, dim in enumerate(dims):
            use_pixel = self.freqs_for == "pixel" and index >= len(dims) - 2
            if use_pixel:
                pos = torch.linspace(-1, 1, steps=dim, device=self.device)
            else:
                pos = torch.arange(dim, device=self.device)
            seq_freqs = self.forward(pos, self.freqs, seq_len=dim)
            axis = [None] * len(dims)
            axis[index] = colon
            all_freqs.append(seq_freqs[(Ellipsis, *axis, colon)])
        all_freqs = torch.broadcast_tensors(*all_freqs)
        return torch.cat(all_freqs, dim=-1)

# OasisVAEAttention
class OasisVAEAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, frame_height: int, frame_width: int, qkv_bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.frame_height = frame_height
        self.frame_width = frame_width
        # Single parametric Linear for QKV
        self.qkv = Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = Linear(dim, dim, bias=True)
        self.rotary = OasisRotaryEmbedding(
            dim=(dim // num_heads) // 4,
            freqs_for="pixel",
            max_freq=frame_height * frame_width,
        )
        self.register_buffer(
            "rotary_freqs",
            self.rotary.get_axial_freqs(frame_height, frame_width),
            persistent=False,
        )
        self.attn = DenseAttention(backend="sdpa")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        # Use Triton fused QKV when available; else fallback to F.linear
        if x.is_cuda and TRITON_AVAILABLE:
            q, k, v = _triton_qkv_matmul_bias(x, self.qkv.weight, self.qkv.bias, self.qkv.bias, self.qkv.bias)
        else:
            qkv = F.linear(x, self.qkv.weight, self.qkv.bias)
            q, k, v = qkv.chunk(3, dim=-1)

        # Reshape
        q = q.reshape(bsz, self.frame_height, self.frame_width, self.num_heads, -1).permute(0, 3, 1, 2, 4)
        k = k.reshape(bsz, self.frame_height, self.frame_width, self.num_heads, -1).permute(0, 3, 1, 2, 4)
        v = v.reshape(bsz, self.frame_height, self.frame_width, self.num_heads, -1).permute(0, 3, 1, 2, 4)

        # Apply pixel rotary
        q = oasis_apply_rotary_emb(self.rotary_freqs, q)
        k = oasis_apply_rotary_emb(self.rotary_freqs, k)

        # Flatten to (B, H*W, H, D)
        seq_len = self.frame_height * self.frame_width
        q = q.reshape(bsz, self.num_heads, seq_len, -1).transpose(1, 2)
        k = k.reshape(bsz, self.num_heads, seq_len, -1).transpose(1, 2)
        v = v.reshape(bsz, self.num_heads, seq_len, -1).transpose(1, 2)

        # Ensure contiguous for SDPA
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        # SDPA attention
        out = self.attn(q, k, v)

        # Project
        out = out.reshape(bsz, seq_len, -1)
        return self.proj(out)

# LayerNorm and MLP
class GELU(nn.Module):
    def __init__(self, approximate: str = "none"):
        super().__init__()
        self.approximate = approximate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x, approximate=self.approximate)

class OasisMLP(nn.Module):
    def __init__(self, in_features: int, hidden_features: int | None = None, out_features: int | None = None, approximate_tanh: bool = False):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = Linear(in_features, hidden_features, bias=True)
        self.act = GELU(approximate="tanh" if approximate_tanh else "none")
        self.fc2 = Linear(hidden_features, out_features, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-5, elementwise_affine: bool = True, create_scale: bool = True, create_offset: bool = True, promote_fp32: bool = True):
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
        self._src_w, self._src_b = None, None
        self._w32, self._b32 = None, None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.promote_fp32:
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

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
        return F.layer_norm(x.float(), self.normalized_shape, weight, bias, self.eps).to(orig_dtype)

# ---------------------------
# Entry point: ModelNew
# ---------------------------

class ModelNew(nn.Module):
    def __init__(self, dim: int, num_heads: int, frame_height: int, frame_width: int, mlp_ratio: float = 4.0, qkv_bias: bool = False):
        super().__init__()
        self.norm1 = LayerNorm(dim, eps=1e-6)
        self.attn = OasisVAEAttention(dim, num_heads, frame_height, frame_width, qkv_bias=qkv_bias)
        self.norm2 = LayerNorm(dim, eps=1e-6)
        self.mlp = OasisMLP(dim, hidden_features=int(dim * mlp_ratio), approximate_tanh=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

OasisVAEAttentionBlock = ModelNew
