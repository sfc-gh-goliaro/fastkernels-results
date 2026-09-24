import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False


@triton.jit
def _pad2d_const_kernel_v3(
    x_ptr, y_ptr,
    # sizes (constexpr -> shapes are known at launch)
    H: tl.constexpr,
    W: tl.constexpr,
    Ho: tl.constexpr,
    Wo: tl.constexpr,
    top: tl.constexpr,
    left: tl.constexpr,
    # strides (elements)
    STRIDE_XB: tl.constexpr, STRIDE_XH: tl.constexpr, STRIDE_XW: tl.constexpr,
    STRIDE_YB: tl.constexpr, STRIDE_YH: tl.constexpr, STRIDE_YW: tl.constexpr,
    # tiling
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
    # value as python float -> becomes scalar; Triton will cast on store
    VALUE,
):
    # Program ids
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)

    # Tile origins
    ho0 = pid_h * BLOCK_H
    wo0 = pid_w * BLOCK_W

    # Indices
    r = ho0 + tl.arange(0, BLOCK_H)  # [BH]
    c = wo0 + tl.arange(0, BLOCK_W)  # [BW]
    HO = r[:, None]  # [BH, 1]
    WO = c[None, :]  # [1, BW]

    # In-bounds mask
    mask_o = (HO < Ho) & (WO < Wo)

    # Base pointers for this batch
    x_base = x_ptr + pid_b * STRIDE_XB
    y_base = y_ptr + pid_b * STRIDE_YB

    # -----------------------
    # Phase 1: fill constants for the whole tile (within bounds)
    # -----------------------
    y_idx = y_base + HO * STRIDE_YH + WO * STRIDE_YW
    # Store scalar VALUE; Triton will broadcast and cast to y dtype
    tl.store(y_idx, VALUE, mask=mask_o)

    # -----------------------
    # Phase 2: overwrite inner region with input values
    # -----------------------
    # Inner mask: top <= HO < top+H and left <= WO < left+W
    inner = (HO >= top) & (HO < top + H) & (WO >= left) & (WO < left + W)

    # Input coordinates
    HI = HO - top
    WI = WO - left

    x_idx = x_base + HI * STRIDE_XH + WI * STRIDE_XW
    load_mask = mask_o & inner
    x_val = tl.load(x_idx, mask=load_mask, other=0)
    tl.store(y_idx, x_val, mask=load_mask)


class ModelNew(nn.Module):
    """
    Triton-optimized version of F.pad for constant padding on the last two dims.
    Falls back to torch F.pad when Triton/CUDA is not available or dtype is not supported.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, pad: tuple[int, ...], value: float = 0.0) -> torch.Tensor:
        # Only handle 4-element pad as constant on last 2 dims
        if not isinstance(pad, (tuple, list)) or len(pad) != 4:
            return F.pad(x, pad, value=value)

        left, right, top, bottom = map(int, pad)

        # Fallbacks
        if (not _HAS_TRITON) or (not x.is_cuda):
            return F.pad(x, pad, value=value)
        if x.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            return F.pad(x, pad, value=value)

        # Make contiguous
        if not x.is_contiguous():
            x = x.contiguous()

        shape = x.shape
        if x.dim() < 2:
            return F.pad(x, pad, value=value)

        # Flatten to [B, H, W]
        B = 1
        for d in shape[:-2]:
            B *= d
        H = shape[-2]
        W = shape[-1]

        Ho = H + top + bottom
        Wo = W + left + right

        # Allocate output
        out_shape = shape[:-2] + (Ho, Wo)
        y = torch.empty(out_shape, device=x.device, dtype=x.dtype)

        # Views as [B, H, W] and [B, Ho, Wo]
        x_view = x.view(B, H, W)
        y_view = y.view(B, Ho, Wo)

        # Strides in elements
        STRIDE_XB, STRIDE_XH, STRIDE_XW = x_view.stride()
        STRIDE_YB, STRIDE_YH, STRIDE_YW = y_view.stride()

        # Tiling: wide along W for coalescing; moderate H to利用 latency hiding
        BLOCK_W = 128  # matches common W=128; good coalescing
        BLOCK_H = 32   # balances register use and per-program work

        grid = (
            B,
            triton.cdiv(Ho, BLOCK_H),
            triton.cdiv(Wo, BLOCK_W),
        )

        # Launch params
        num_warps = 4
        num_stages = 3

        _pad2d_const_kernel_v3[grid](
            x_view, y_view,
            H, W, Ho, Wo,
            top, left,
            STRIDE_XB, STRIDE_XH, STRIDE_XW,
            STRIDE_YB, STRIDE_YH, STRIDE_YW,
            BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
            VALUE=float(value),
            num_warps=num_warps, num_stages=num_stages,
        )

        return y

Pad = ModelNew
