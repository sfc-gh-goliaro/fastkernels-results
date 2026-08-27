from math import pi
import torch
import torch.nn as nn

import triton
import triton.language as tl


@triton.jit
def _rotate_half_pixel_kernel(
    x_ptr,          # *f16 / *bf16
    freq_ptr,       # *f32, length DH
    out_ptr,        # *f16 / *bf16
    D: tl.constexpr,
    DH: tl.constexpr,   # D // 2
    BLOCK_D: tl.constexpr,
):
    # 2D launch: pid0 over (B*Nhead*H*W), pid1 over blocks of D
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    d_start = pid1 * BLOCK_D
    offs_d = d_start + tl.arange(0, BLOCK_D)
    mask = offs_d < D

    # base linear index for this (b,n,h,w) row: each row has D contiguous elements
    base = pid0 * D
    row_start = x_ptr + base

    # parity: even = 2*i, odd = 2*i+1
    is_even = (offs_d % 2) == 0
    i = offs_d // 2  # in [0, DH)

    # indices
    even = i * 2
    odd  = even + 1

    mask_even = mask & is_even
    mask_odd  = mask & (~is_even)

    # load only even/odd needed, upcast to f32
    x_even = tl.load(row_start + even, mask=mask_even, other=0.0).to(tl.float32)
    x_odd  = tl.load(row_start + odd,  mask=mask_odd,  other=0.0).to(tl.float32)

    # load frequencies and compute cos/sin
    freq = tl.load(freq_ptr + i, mask=mask, other=0.0).to(tl.float32)
    c = tl.cos(freq)
    s = tl.sin(freq)

    # compute rotated results
    out_even = x_even * c - x_odd * s   # at positions 2*i
    out_odd  = x_odd  * c + x_even * s  # at positions 2*i+1

    # store results back to out
    tl.store(out_ptr + base + even, out_even, mask=mask_even)
    tl.store(out_ptr + base + odd,  out_odd,  mask=mask_odd)


def _triton_rotate_half_pixel(x: torch.Tensor, max_freq: float) -> torch.Tensor:
    """
    Fused Triton rotate-half for 'pixel' mode.
    x: [B, Nhead, H, W, D], CUDA, fp16/bf16
    Returns: same shape/dtype.
    """
    assert x.is_cuda, "Triton kernel requires CUDA tensor"
    assert x.dtype in (torch.float16, torch.bfloat16), "Supported dtypes: fp16, bf16"
    x = x.contiguous()
    assert x.dim() == 5, f"Expected 5D tensor [B,N,H,W,D], got shape {tuple(x.shape)}"
    B, Nhead, H, W, D = x.shape
    DH = D // 2
    assert DH * 2 == D, f"D must be even, got {D}"

    device = x.device

    # Replicate frequency computation from original code:
    # freqs = linspace(1, max_freq/2, DH) * pi
    ends = torch.tensor([1.0, float(max_freq) * 0.5], dtype=torch.float32, device=device)
    freq_idx = torch.linspace(ends[0], ends[1], DH, dtype=torch.float32, device=device)
    freqs = freq_idx * torch.tensor(pi, dtype=torch.float32, device=device)

    out = torch.empty_like(x)

    BLOCK_D = 128
    grid = (B * Nhead * H * W, triton.cdiv(D, BLOCK_D))

    _rotate_half_pixel_kernel[grid](
        x, freqs, out,
        D=D, DH=DH,
        BLOCK_D=BLOCK_D,
        num_warps=4,
        num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        frame_height: int,
        frame_width: int,
        *,
        qkv_bias: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.frame_height = frame_height
        self.frame_width = frame_width
        # Linear layers
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, F, D_in]
        qkv = self.qkv(x)                 # [B, F, 3D]
        q, k, v = qkv.chunk(3, dim=-1)    # each [B, F, D]

        # Reshape to [B, num_heads, H, W, d]
        d = q.shape[-1] // self.num_heads
        B = q.shape[0]
        F = q.shape[1]
        H, W = self.frame_height, self.frame_width
        assert F == H * W, f"Expected F={H*W}, got {F}"

        q = q.reshape(B, self.num_heads, H, W, d)
        k = k.reshape(B, self.num_heads, H, W, d)
        v = v.reshape(B, self.num_heads, H, W, d)

        # Apply Triton fused pixel-rotary (compute freqs internally, no external deps)
        assert q.is_cuda and k.is_cuda, "Triton kernel requires CUDA tensors"
        q = _triton_rotate_half_pixel(q, max_freq=float(H * W))
        k = _triton_rotate_half_pixel(k, max_freq=float(H * W))

        # Permute to [B, num_heads, S, d]
        q = q.reshape(B, self.num_heads, H * W, d).transpose(1, 2)
        k = k.reshape(B, self.num_heads, H * W, d).transpose(1, 2)
        v = v.reshape(B, self.num_heads, H * W, d).transpose(1, 2)

        # SDPA attention (no mask)
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
        )

        # Final projection
        out = out.reshape(B, H * W, -1)
        return self.proj(out)

OasisVAEAttention = ModelNew
