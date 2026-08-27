import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import triton
import triton.language as tl


@triton.jit
def _rotate_apply_axial_kernel(
    x_ptr,         # *dtype, shape [M, D], row-major
    freq_ptr,      # *dtype, shape [K, D//2], row-major
    out_ptr,       # *dtype, shape [M, D], row-major
    M: tl.constexpr,
    D: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # 2D program id: over rows (M) and features (D)
    pid_m = tl.program_id(0)
    pid_d = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    mask_m = offs_m < M
    mask_d = offs_d < D

    mm = offs_m[:, None]   # [BM, 1]
    dd = offs_d[None, :]   # [1, BD]

    base = mm * D + dd     # [BM, BD]
    x_ptrs = x_ptr + base
    out_ptrs = out_ptr + base

    J = D // 2
    for j in range(0, J):
        c0 = 2 * j
        c1 = c0 + 1

        # column masks (defensive, though c0,c1 are valid)
        mask_c0 = mask_d & (c0 < D)
        mask_c1 = mask_d & (c1 < D)

        # load x columns
        x_c0 = tl.load(x_ptrs + c0, mask=mask_m & mask_c0, other=0.0)
        x_c1 = tl.load(x_ptrs + c1, mask=mask_m & mask_c1, other=0.0)

        # load freq row j across BD features
        freq_j_ptrs = freq_ptr + j * (D // 2) + offs_d
        cos_j = tl.cos(freq_j_ptrs)
        sin_j = tl.sin(freq_j_ptrs)

        out_c0 = x_c0 * cos_j - x_c1 * sin_j
        out_c1 = x_c1 * cos_j + x_c0 * sin_j

        tl.store(out_ptrs + c0, out_c0, mask=mask_m & mask_c0)
        tl.store(out_ptrs + c1, out_c1, mask=mask_m & mask_c1)


def _rotate_apply_axial(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """
    Apply oasis-style rotate-half on x using provided freqs.
    x: [M, D] (contiguous), freqs: [K, D//2] (contiguous), both CUDA.
    Returns out: [M, D] same dtype/device.
    """
    assert x.is_cuda and freqs.is_cuda, "Triton kernel requires CUDA tensors"
    assert x.dim() == 2, f"Expected 2D tensor, got shape {tuple(x.shape)}"
    M, D = x.shape
    assert D % 2 == 0, f"Last dim must be even, got D={D}"
    assert freqs.dim() == 2, f"Expected 2D freqs, got shape {tuple(freqs.shape)}"
    K, K2 = freqs.shape
    assert K2 == D // 2, f"freqs second dim must be D//2={D//2}, got {K2}"

    if not x.is_contiguous():
        x = x.contiguous()
    if not freqs.is_contiguous():
        freqs = freqs.contiguous()

    out = torch.empty_like(x)

    BLOCK_D = 128
    BLOCK_M = 128
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(D, BLOCK_D))

    _rotate_apply_axial_kernel[grid](
        x, freqs, out,
        M, D, K,
        BLOCK_M=BLOCK_M,
        BLOCK_D=BLOCK_D,
        num_warps=4,
        num_stages=2,
    )
    return out


# Original helpers (kept for parity / testing)
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
    def __init__(
        self,
        dim: int,
        *,
        freqs_for: str = "lang",
        theta: float = 10000.0,
        max_freq: float = 10.0,
    ):
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

    def forward(
        self,
        t: torch.Tensor,
        freqs: torch.Tensor,
        seq_len: int | None = None,
        offset: int = 0,
    ) -> torch.Tensor:
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
            out = self._flex_fn(q, k, v, block_mask=attn_mask, scale=softmax_scale)
        elif self.use_cudnn_kernel:
            if attn_mask is not None and causal:
                raise ValueError("pass either attn_mask or causal=True, not both")
            if attn_mask is not None and not attn_mask.is_contiguous():
                attn_mask = attn_mask.contiguous()
            if q.shape[-1] > _CUDNN_MAX_HEAD_DIM:
                if attn_mask is not None and attn_mask.dtype not in (torch.bool, q.dtype):
                    attn_mask = attn_mask.to(dtype=q.dtype)
                with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION, torch.nn.attention.SDPBackend.MATH]):
                    out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=causal, scale=softmax_scale)
            else:
                try:
                    with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.CUDNN_ATTENTION]):
                        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=causal, scale=softmax_scale)
                except RuntimeError:
                    with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION, torch.nn.attention.SDPBackend.MATH]):
                        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=causal, scale=softmax_scale)
        else:
            if attn_mask is not None and attn_mask.dtype != torch.bool:
                attn_mask = attn_mask.to(dtype=q.dtype)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False if attn_mask is not None else causal, scale=softmax_scale)
        return out.permute(0, 2, 1, 3)


# Entry point expected by the harness: ModelNew
class ModelNew(nn.Module):
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

        # Reshape to [B*T, H, W, heads, dim_head]
        q2 = q.reshape(bsz * time, height, width, self.heads, -1).permute(0, 3, 1, 2, 4).reshape(-1, q.shape[-1])
        k2 = k.reshape(bsz * time, height, width, self.heads, -1).permute(0, 3, 1, 2, 4).reshape(-1, k.shape[-1])
        v2 = v.reshape(bsz * time, height, width, self.heads, -1).permute(0, 3, 1, 2, 4).reshape(-1, v.shape[-1])

        # Get axial freqs (CUDA tensor, matching dtype)
        freqs = self.rotary_emb.get_axial_freqs(height, width).to(device=x.device, dtype=x.dtype)

        # Apply fused Triton rotary
        q_r = _rotate_apply_axial(q2, freqs)
        k_r = _rotate_apply_axial(k2, freqs)

        # Rebuild [B*T, heads, L, dim_head]
        L = height * width
        q3 = q_r.reshape(bsz * time, self.heads, L, -1)
        k3 = k_r.reshape(bsz * time, self.heads, L, -1)
        v3 = v2.reshape(bsz * time, self.heads, L, -1)

        # SDPA expects [B, H, L, D]
        q4 = q3.reshape(bsz, time, self.heads, L, -1).reshape(bsz * time, self.heads, L, -1)
        k4 = k3.reshape(bsz, time, self.heads, L, -1).reshape(bsz * time, self.heads, L, -1)
        v4 = v3.reshape(bsz, time, self.heads, L, -1).reshape(bsz * time, self.heads, L, -1)

        out = self.attn(q4, k4, v4, causal=False)
        out = out.reshape(bsz, time, self.heads, L, -1).reshape(bsz, time, height, width, self.heads, -1)
        return self.to_out(out.to(q.dtype))

OasisSpatialAxialAttention = ModelNew
