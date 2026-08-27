import torch
import triton
import triton.language as tl


@triton.jit
def _rotary_interleaved_vec64_kernel(
    OUT, X, COS, SIN,
    # strides in elements
    stride_out_b, stride_out_s, stride_out_h, stride_out_d,
    stride_x_b, stride_x_s, stride_x_h, stride_x_d,
    stride_cos_s, stride_cos_d,
    stride_sin_s, stride_sin_d,
    seqlen: tl.constexpr,
    nheads: tl.constexpr,
    batch: tl.constexpr,
    BLOCK_M: tl.constexpr,  # rows per program
):
    # Program ids
    pid_m = tl.program_id(axis=0)  # tile idx along seqlen
    pid_h = tl.program_id(axis=1)  # head idx
    pid_b = tl.program_id(axis=2)  # batch idx

    # Rows this program will process
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = rm < seqlen

    # Half features: 0..63
    rk = tl.arange(0, 64)

    # Base pointers for (b, h)
    base_out = OUT + pid_b * stride_out_b + pid_h * stride_out_h
    base_x = X + pid_b * stride_x_b + pid_h * stride_x_h

    # --- Load cos, sin: shape [BLOCK_M, 64] ---
    pos_cos = rm[:, None] * stride_cos_s + rk[None, :] * stride_cos_d
    pos_sin = rm[:, None] * stride_sin_s + rk[None, :] * stride_sin_d
    cos = tl.load(COS + pos_cos, mask=mask_m[:, None], other=0.0).to(tl.float32)  # [BM,64]
    sin = tl.load(SIN + pos_sin, mask=mask_m[:, None], other=0.0).to(tl.float32)  # [BM,64]

    # --- Load x0 (even), x1 (odd): shape [BM,64] ---
    # even k -> 2*j ; odd k -> 2*j + 1
    ptr_x0 = base_x + (rm[:, None] * stride_x_s) + (rk[None, :] * 2 * stride_x_d)       # 2*j
    ptr_x1 = base_x + (rm[:, None] * stride_x_s) + ((rk[None, :] * 2 + 1) * stride_x_d)  # 2*j+1
    x0 = tl.load(ptr_x0, mask=mask_m[:, None], other=0.0).to(tl.float32)  # [BM,64]
    x1 = tl.load(ptr_x1, mask=mask_m[:, None], other=0.0).to(tl.float32)  # [BM,64]

    # --- Compute outputs ---
    # even: x0*cos - x1*sin ; odd: x0*sin + x1*cos
    out_even = x0 * cos - x1 * sin   # [BM,64]
    out_odd  = x0 * sin + x1 * cos   # [BM,64]

    # --- Store back to OUT at positions k=2j and k=2j+1 ---
    ptr_out_even = base_out + (rm[:, None] * stride_out_s) + (rk[None, :] * 2 * stride_out_d)
    ptr_out_odd  = base_out + (rm[:, None] * stride_out_s) + ((rk[None, :] * 2 + 1) * stride_out_d)
    tl.store(ptr_out_even, out_even, mask=mask_m[:, None])
    tl.store(ptr_out_odd,  out_odd,  mask=mask_m[:, None])


def _apply_rotary_triton_interleaved_vec64(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    Specialized launcher for interleaved rotary (GPT-J style), vectorized over 64 features.
    Assumes:
      - x: (B, S, H, 128), contiguous
      - cos/sin: (S, 64), contiguous
    Returns out with same shape/dtype as x.
    """
    assert x.is_cuda and cos.is_cuda and sin.is_cuda, "Triton kernel requires CUDA tensors"
    assert x.dtype == cos.dtype == sin.dtype, "x, cos, sin must have same dtype"
    assert x.dim() == 4, f"Expected x of shape (B,S,H,D); got {tuple(x.shape)}"
    B, S, H, D = x.shape
    assert D == 128, f"Expected D=128; got {D}"
    assert cos.shape == (S, 64) and sin.shape == (S, 64), \
        f"Expected cos/sin shape (S,64); got {cos.shape}, {sin.shape}"

    # Make sure contiguous
    x = x.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()

    out = torch.empty_like(x)

    # Strides in elements
    stride_out_b, stride_out_s, stride_out_h, stride_out_d = out.stride(0), out.stride(1), out.stride(2), out.stride(3)
    stride_x_b,  stride_x_s,  stride_x_h,  stride_x_d  = x.stride(0),  x.stride(1),  x.stride(2),  x.stride(3)
    stride_cos_s, stride_cos_d = cos.stride(0), cos.stride(1)
    stride_sin_s, stride_sin_d = sin.stride(0), sin.stride(1)

    # Tiling
    BLOCK_M = 256  # process 256 rows per program; good for S=1536/4608
    grid = (triton.cdiv(S, BLOCK_M), H, B)

    _rotary_interleaved_vec64_kernel[grid](
        out, x, cos, sin,
        stride_out_b, stride_out_s, stride_out_h, stride_out_d,
        stride_x_b,  stride_x_s,  stride_x_h,  stride_x_d,
        stride_cos_s, stride_cos_d,
        stride_sin_s, stride_sin_d,
        seqlen=S, nheads=H, batch=B,
        BLOCK_M=BLOCK_M,
        num_warps=8,
        num_stages=2,
    )
    return out


class Model(nn.Module):
    """
    Triton-optimized rotary position embedding entry point.
    Matches the original Model API:
      - __init__(is_neox_style: bool = False) where False -> interleaved (GPT-J), True -> half-split (GPT-NeoX).
      - forward(x, cos, sin) -> tensor
    Optimized fast path for interleaved, D=128, half=64 (evaluation case).
    """
    def __init__(self, is_neox_style: bool = False):
        super().__init__()
        # is_neox_style=False -> GPT-J interleaved; True -> half-split
        self.interleaved = not is_neox_style

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """
        x: (B, S, H, D)
        cos, sin: (S, D//2) or (1, S, D//2) — we take first slice if shape[0]==1
        """
        # Normalize cos/sin: if leading dim is 1, take slice 0
        if cos.dim() == 3 and cos.size(0) == 1:
            cos = cos[0]
            sin = sin[0]

        assert x.is_cuda and cos.is_cuda and sin.is_cuda, "Triton kernel requires CUDA tensors"
        assert x.dim() == 4, f"Expected x of shape (B,S,H,D); got {tuple(x.shape)}"
        B, S, H, D = x.shape
        assert cos.shape == (S, D // 2) and sin.shape == (S, D // 2), \
            f"Expected cos/sin shape (S,{D//2}); got {cos.shape}, {sin.shape}"
        assert x.dtype == cos.dtype == sin.dtype, "x, cos, sin must have same dtype"

        # Fast path: interleaved layout, D=128
        if self.interleaved and (D == 128):
            return _apply_rotary_triton_interleaved_vec64(x, cos, sin)
        else:
            # Fallback: use the original helper (supports both layouts / more general)
            from rotary_triton import _apply_rotary  # assuming the original file is importable
            return _apply_rotary(x, cos, sin, interleaved=self.interleaved)

DiffusionRoPE = ModelNew
