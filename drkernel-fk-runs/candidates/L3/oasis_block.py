import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import triton
import triton.language as tl


# -------------------------
# Triton elementwise kernels (flat, no strides)
# -------------------------

@triton.jit
def _modulate_kernel(x_ptr, scale_ptr, shift_ptr, y_ptr, SIZE: tl.constexpr, D: tl.constexpr):
    # y[i] = x[i] * (1 + scale[d]) + shift[d], where d = i % D
    pid = tl.program_id(0)
    if pid >= SIZE:
        return
    i = pid
    d = i % D
    x = tl.load(x_ptr + i)
    s = tl.load(scale_ptr + d)
    h = tl.load(shift_ptr + d)
    y = x * (1.0 + s) + h
    tl.store(y_ptr + i, y)


@triton.jit
def _gate_kernel(x_ptr, g_ptr, y_ptr, SIZE: tl.constexpr, D: tl.constexpr):
    # y[i] = x[i] * g[d], where d = i % D
    pid = tl.program_id(0)
    if pid >= SIZE:
        return
    i = pid
    d = i % D
    x = tl.load(x_ptr + i)
    g = tl.load(g_ptr + d)
    y = x * g
    tl.store(y_ptr + i, y)


def _modulate_triton(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    # x: [B, T, H, W, D]; scale, shift: [B, 1, D]
    assert x.is_contiguous(), "x must be contiguous"
    assert scale.is_contiguous() and shift.is_contiguous(), "scale/shift must be contiguous"
    assert scale.shape[0] == x.shape[0] and shift.shape[0] == x.shape[0], "batch mismatch"
    B, T, H, W, D = x.shape
    x_flat = x.view(-1)
    y_flat = torch.empty_like(x_flat)
    SIZE = x_flat.numel()
    grid = (SIZE,)
    _modulate_kernel[grid](x_flat, scale.view(-1), shift.view(-1), y_flat, SIZE, D, num_warps=4)
    return y_flat.view_as(x)


def _gate_triton(x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    # x: [B, T, H, W, D]; g: [B, 1, D]
    assert x.is_contiguous(), "x must be contiguous"
    assert g.is_contiguous(), "g must be contiguous"
    assert g.shape[0] == x.shape[0], "batch mismatch"
    B, T, H, W, D = x.shape
    x_flat = x.view(-1)
    y_flat = torch.empty_like(x_flat)
    SIZE = x_flat.numel()
    grid = (SIZE,)
    _gate_kernel[grid](x_flat, g.view(-1), y_flat, SIZE, D, num_warps=4)
    return y_flat.view_as(x)


# -------------------------
# Reuse original definitions (kept as-is)
# -------------------------

# The following classes are copied verbatim from the provided snippet
# to ensure identical math and layout behavior.

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
    def __init__(self, backend: str = "auto"):
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

        cc = (torch.cuda.get_device_capability()[0] * 10
              + torch.cuda.get_device_capability()[1])
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
            out = self._flex_fn(q, k, v, block_mask=attn_mask, scale=softmax_scale)
        elif self.use_cudnn_kernel:
            from torch.nn.attention import sdpa_kernel, SDPBackend
            if attn_mask is not None and causal:
                raise ValueError("Pass either attn_mask or causal=True, not both.")
            if q.shape[-1] > _CUDNN_MAX_HEAD_DIM:
                with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                    out = F.scaled_dot_product_attention(q, k, v,
                                                         attn_mask=attn_mask,
                                                         dropout_p=0.0,
                                                         is_causal=causal,
                                                         scale=softmax_scale)
            else:
                try:
                    with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
                        out = F.scaled_dot_product_attention(q, k, v,
                                                             attn_mask=attn_mask,
                                                             dropout_p=0.0,
                                                             is_causal=causal,
                                                             scale=softmax_scale)
                except RuntimeError:
                    with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                        out = F.scaled_dot_product_attention(q, k, v,
                                                             attn_mask=attn_mask,
                                                             dropout_p=0.0,
                                                             is_causal=causal,
                                                             scale=softmax_scale)
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


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    fixed_dims = [1] * len(shift.shape[1:])
    shift = shift.repeat(x.shape[0] // shift.shape[0], *fixed_dims)
    scale = scale.repeat(x.shape[0] // shift.shape[0], *fixed_dims)
    while shift.dim() < x.dim():
        shift = shift.unsqueeze(-2)
        scale = scale.unsqueeze(-2)
    return x * (1 + scale) + shift


def _gate(x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    fixed_dims = [1] * len(g.shape[1:])
    g = g.repeat(x.shape[0] // g.shape[0], *fixed_dims)
    while g.dim() < x.dim():
        g = g.unsqueeze(-2)
    return g * x


class OasisRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, freqs_for: str = "lang", theta: float = 10000.0, max_freq: float = 10.0):
        super().__init__()
        self.dim = dim
        self.freqs_for = freqs_for
        if freqs_for == "lang":
            freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        elif freqs_for == "pixel":
            freqs = torch.linspace(1.0, max_freq / 2, dim // 2) * math.pi
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

    def forward(self, t: torch.Tensor, freqs: torch.Tensor, seq_len: int | None = None, offset: int = 0):
        del seq_len, offset
        return self._forward_freqs(t, freqs)

    def rotate_queries_or_keys(self, t: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        seq_len = t.shape[-2]
        positions = torch.arange(seq_len, device=t.device, dtype=t.dtype)
        seq_freqs = self.forward(positions, freqs, seq_len=seq_len)
        return oasis_rotate_half(t[..., :seq_len, :]) * seq_freqs + t[..., :seq_len, :]

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


class OasisTemporalAxialAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, rotary_emb: OasisRotaryEmbedding, is_causal: bool = True):
        super().__init__()
        self.heads = heads
        self.to_qkv = Linear(dim, dim_head * heads * 3, bias=False)
        self.to_out = Linear(dim_head * heads, dim, bias=True)
        self.rotary_emb = rotary_emb
        self.is_causal = is_causal
        self.attn = DenseAttention(backend="sdpa")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, time, height, width, _ = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q = q.reshape(bsz, time, height, width, self.heads, -1).permute(0, 2, 3, 4, 1, 5)
        k = k.reshape(bsz, time, height, width, self.heads, -1).permute(0, 2, 3, 4, 1, 5)
        v = v.reshape(bsz, time, height, width, self.heads, -1).permute(0, 2, 3, 4, 1, 5)

        q = q.reshape(bsz * height * width, self.heads, time, -1)
        k = k.reshape(bsz * height * width, self.heads, time, -1)
        v = v.reshape(bsz * height * width, self.heads, time, -1)

        q = self.rotary_emb.rotate_queries_or_keys(q, self.rotary_emb.freqs)
        k = self.rotary_emb.rotate_queries_or_keys(k, self.rotary_emb.freqs)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = self.attn(q, k, v, causal=self.is_causal)
        out = out.reshape(bsz, height, width, time, self.heads, -1)
        out = out.permute(0, 3, 1, 2, 4, 5).reshape(bsz, time, height, width, -1)
        return self.to_out(out.to(q.dtype))


class OasisSpatialAxialAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, rotary_emb: OasisRotaryEmbedding):
        super().__init__()
        self.heads = heads
        self.to_qkv = Linear(dim, dim_head * heads * 3, bias=False)
        self.to_out = Linear(dim_head * heads, dim, bias=True)
        self.rotary_emb = rotary_emb
        self.attn = DenseAttention(backend="sdpa")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, time, height, width, _ = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q = q.reshape(bsz * time, height, width, self.heads, -1).permute(0, 3, 1, 2, 4)
        k = k.reshape(bsz * time, height, width, self.heads, -1).permute(0, 3, 1, 2, 4)
        v = v.reshape(bsz * time, height, width, self.heads, -1).permute(0, 3, 1, 2, 4)

        freqs = self.rotary_emb.get_axial_freqs(height, width)
        q = oasis_apply_rotary_emb(freqs, q)
        k = oasis_apply_rotary_emb(freqs, k)

        q = q.reshape(bsz * time, self.heads, height * width, -1).transpose(1, 2)
        k = k.reshape(bsz * time, self.heads, height * width, -1).transpose(1, 2)
        v = v.reshape(bsz * time, self.heads, height * width, -1).transpose(1, 2)
        out = self.attn(q, k, v, causal=False)
        out = out.reshape(bsz, time, height, width, self.heads, -1).reshape(bsz, time, height, width, -1)
        return self.to_out(out.to(q.dtype))


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


class SiLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x)


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
            self._w32 = w.float() if w is not None and w.dtype != torch.float32 else w
            self._b32 = b.float() if b is not None and b.dtype != torch.float32 else b
            self._cast_done = True
        weight, bias = self._w32, self._b32
        return F.layer_norm(x.float(), self.normalized_shape, weight, bias, self.eps).to(orig_dtype)


# -------------------------
# Triton-optimized ModelNew
# -------------------------

class ModelNew(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        is_causal: bool = True,
        spatial_rotary_emb: OasisRotaryEmbedding = None,
        temporal_rotary_emb: OasisRotaryEmbedding = None,
    ):
        super().__init__()
        if spatial_rotary_emb is None:
            spatial_rotary_emb = OasisRotaryEmbedding(hidden_size, freqs_for="lang", theta=10000.0)
        if temporal_rotary_emb is None:
            temporal_rotary_emb = OasisRotaryEmbedding(hidden_size, freqs_for="lang", theta=10000.0)

        # Keep original modules
        self.s_norm1 = LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.s_attn = OasisSpatialAxialAttention(
            hidden_size,
            heads=num_heads,
            dim_head=hidden_size // num_heads,
            rotary_emb=spatial_rotary_emb,
        )
        self.s_norm2 = LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.s_mlp = OasisMLP(
            hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            approximate_tanh=True,
        )
        self.s_adaLN_modulation = nn.Sequential(
            SiLU(),
            Linear(hidden_size, 6 * hidden_size, bias=True),
        )

        self.t_norm1 = LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.t_attn = OasisTemporalAxialAttention(
            hidden_size,
            heads=num_heads,
            dim_head=hidden_size // num_heads,
            rotary_emb=temporal_rotary_emb,
            is_causal=is_causal,
        )
        self.t_norm2 = LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.t_mlp = OasisMLP(
            hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            approximate_tanh=True,
        )
        self.t_adaLN_modulation = nn.Sequential(
            SiLU(),
            Linear(hidden_size, 6 * hidden_size, bias=True),
        )

        # Triton elementwise
        self._triton_modulate = _modulate_triton
        self._triton_gate = _gate_triton

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # Ensure device
        if c.device != x.device:
            c = c.to(x.device)

        # Spatial block
        s_params = self.s_adaLN_modulation(c)  # [B, T, 6D]
        s_shift_msa, s_scale_msa, s_gate_msa, s_shift_mlp, s_scale_mlp, s_gate_mlp = s_params.chunk(6, dim=-1)

        # norm1 -> modulate -> attn -> gate -> add
        xn = self.s_norm1(x)  # F.layer_norm
        xn = xn.contiguous()
        mod = self._triton_modulate(xn, s_scale_msa.contiguous(), s_shift_msa.contiguous())
        attn_out = self.s_attn(mod)
        gated = self._triton_gate(attn_out, s_gate_msa.contiguous())
        x = x + gated

        # mlp
        xn2 = self.s_norm2(x)
        xn2 = xn2.contiguous()
        mod2 = self._triton_modulate(xn2, s_scale_mlp.contiguous(), s_shift_mlp.contiguous())
        mlp_out = self.s_mlp(mod2)
        gated2 = self._triton_gate(mlp_out, s_gate_mlp.contiguous())
        x = x + gated2

        # Temporal block
        t_params = self.t_adaLN_modulation(c)
        t_shift_msa, t_scale_msa, t_gate_msa, t_shift_mlp, t_scale_mlp, t_gate_mlp = t_params.chunk(6, dim=-1)

        xn3 = self.t_norm1(x)
        xn3 = xn3.contiguous()
        mod3 = self._triton_modulate(xn3, t_scale_msa.contiguous(), t_shift_msa.contiguous())
        attn_t = self.t_attn(mod3)
        gated3 = self._triton_gate(attn_t, t_gate_msa.contiguous())
        x = x + gated3

        xn4 = self.t_norm2(x)
        xn4 = xn4.contiguous()
        mod4 = self._triton_modulate(xn4, t_scale_mlp.contiguous(), t_shift_mlp.contiguous())
        mlp_t = self.t_mlp(mod4)
        gated4 = self._triton_gate(mlp_t, t_gate_mlp.contiguous())
        x = x + gated4

        return x

SpatioTemporalDiTBlock = ModelNew
