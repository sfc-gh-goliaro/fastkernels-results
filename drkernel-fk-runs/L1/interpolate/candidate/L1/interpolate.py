import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_H': 32, 'BLOCK_W': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_H': 64, 'BLOCK_W': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_H': 32, 'BLOCK_W': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_H': 64, 'BLOCK_W': 64}, num_warps=8, num_stages=2),
    ],
    key=['outH', 'outW'],
)
@triton.jit
def _nearest_2d_resize_nchw_kernel(
    in_ptr, out_ptr,
    N: tl.constexpr, C: tl.constexpr,
    H: tl.constexpr, W: tl.constexpr,
    outH: tl.constexpr, outW: tl.constexpr,
    # precomputed scales as float (H/outH, W/outW); used to multiply OH/OW
    scale_h: tl.constexpr, scale_w: tl.constexpr,
    stride_in_n, stride_in_c, stride_in_h, stride_in_w,
    stride_out_n, stride_out_c, stride_out_h, stride_out_w,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
):
    # Program ids
    pid_nc = tl.program_id(0)
    pid_oh = tl.program_id(1)
    pid_ow = tl.program_id(2)

    # Decode n and c
    n = pid_nc // C
    c = pid_nc % C

    # Tile origins
    oh0 = pid_oh * BLOCK_H
    ow0 = pid_ow * BLOCK_W

    # Output indices in tile
    oh = oh0 + tl.arange(0, BLOCK_H)
    ow = ow0 + tl.arange(0, BLOCK_W)

    # Bounds masks
    mask_h = oh < outH
    mask_w = ow < outW

    # 2D grids
    OH = oh[:, None]      # [BH, 1], int32
    OW = ow[None, :]      # [1, BW], int32

    # Floor-based index mapping matching PyTorch:
    # in_h = floor(oh * H / outH) = floor((oh / outH) * H)
    # in_w = floor(ow * W / outW) = floor((ow / outW) * W)
    # Use precomputed scales to avoid per-element divs where possible.
    OHf = OH.to(tl.float32)
    OWf = OW.to(tl.float32)
    scale_h_f = tl.full((), scale_h, dtype=tl.float32)
    scale_w_f = tl.full((), scale_w, dtype=tl.float32)

    in_h_f = OHf * scale_h_f
    in_w_f = OWf * scale_w_f

    # Convert to int32 and floor (for non-negative values, floor == int conversion)
    in_h = in_h_f.to(tl.int32)
    in_w = in_w_f.to(tl.int32)

    # Clamp to [0, H-1] and [0, W-1]
    Hm1 = H - 1
    Wm1 = W - 1
    zero = 0

    in_h = tl.maximum(tl.minimum(in_h, Hm1), zero)
    in_w = tl.maximum(tl.minimum(in_w, Wm1), zero)

    # Base pointers for (n, c)
    base_in = n * stride_in_n + c * stride_in_c
    base_out = n * stride_out_n + c * stride_out_c

    # Compute input and output element offsets (in elements)
    in_offsets = base_in + in_h * stride_in_h + in_w * stride_in_w   # [BH,BW]
    out_offsets = base_out + OH * stride_out_h + OW * stride_out_w    # [BH,BW]

    # Combine masks
    mask = (mask_h[:, None] & mask_w[None, :])

    # Load and store
    vals = tl.load(in_ptr + in_offsets, mask=mask, other=0)
    tl.store(out_ptr + out_offsets, vals, mask=mask)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
        size: int | tuple[int, ...] | None = None,
        scale_factor: float | tuple[float, ...] | None = None,
        mode: str = "nearest",
        align_corners: bool | None = None,
    ) -> torch.Tensor:
        # Fallbacks and validations
        if not TRITON_AVAILABLE:
            return F.interpolate(x, size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners)

        if mode != "nearest":
            return F.interpolate(x, size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners)

        if not x.is_cuda:
            return F.interpolate(x, size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners)

        if x.dim() != 4:
            return F.interpolate(x, size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners)

        # Ensure contiguous for best performance
        if not x.is_contiguous():
            x = x.contiguous()

        N, C, H, W = x.shape

        # Determine output size
        outH, outW = None, None
        use_size = size is not None
        use_scale = scale_factor is not None

        if use_size and use_scale:
            # Ambiguous; fallback
            return F.interpolate(x, size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners)

        if use_size:
            if isinstance(size, int):
                outH = outW = int(size)
            else:
                sz = tuple(size)
                if len(sz) != 2:
                    return F.interpolate(x, size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners)
                outH, outW = int(sz[0]), int(sz[1])
        else:
            if scale_factor is None:
                return F.interpolate(x, size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners)
            if isinstance(scale_factor, (float, int)):
                sf = float(scale_factor)
                outH = int(math.floor(H * sf))
                outW = int(math.floor(W * sf))
            else:
                sf = tuple(scale_factor)
                if len(sf) != 2:
                    return F.interpolate(x, size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners)
                sh, sw = float(sf[0]), float(sf[1])
                outH = int(math.floor(H * sh))
                outW = int(math.floor(W * sw))
            if outH <= 0 or outW <= 0:
                raise ValueError(f"Non-positive output size computed: {(outH, outW)}")

        # Allocate output
        out = torch.empty((N, C, outH, outW), device=x.device, dtype=x.dtype)

        # Get strides (in elements)
        s_in_n, s_in_c, s_in_h, s_in_w = x.stride()
        s_out_n, s_out_c, s_out_h, s_out_w = out.stride()

        # Precompute scales as Python floats: H/outH, W/outW
        scale_h = float(H) / float(outH)
        scale_w = float(W) / float(outW)

        # Launch configuration: grid over (N*C, tiles_H, tiles_W)
        # BLOCK sizes are selected by autotune; we just provide grid.
        def cdiv(a, b): return (a + b - 1) // b
        # We need a dummy BLOCK to build grid; Triton will override via config.
        # Use the smallest BLOCK to be safe for grid shape (autotune will re-launch with its BLOCKs).
        MIN_BLOCK = 32
        grid = (
            N * C,
            cdiv(outH, MIN_BLOCK),
            cdiv(outW, MIN_BLOCK),
        )

        _nearest_2d_resize_nchw_kernel[grid](
            x, out,
            N, C, H, W, outH, outW,
            scale_h, scale_w,
            s_in_n, s_in_c, s_in_h, s_in_w,
            s_out_n, s_out_c, s_out_h, s_out_w,
        )

        return out

Interpolate = ModelNew
