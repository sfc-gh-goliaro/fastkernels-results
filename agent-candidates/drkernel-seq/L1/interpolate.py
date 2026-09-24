import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 32},  num_warps=2, num_stages=2),
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 64},  num_warps=2, num_stages=2),
        triton.Config({"BLOCK_H": 16, "BLOCK_W": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_H": 32, "BLOCK_W": 32},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_H": 32, "BLOCK_W": 64},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_H": 32, "BLOCK_W": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_H": 64, "BLOCK_W": 64},  num_warps=8, num_stages=3),
        triton.Config({"BLOCK_H": 64, "BLOCK_W": 128}, num_warps=8, num_stages=3),
    ],
    key=["H_out", "W_out"],
)
@triton.jit
def nearest_2d_resize_kernel(
    x_ptr,                        # *const T
    y_ptr,                        # *T
    # sizes (constexpr so divisions can be optimized)
    N: tl.constexpr,              # int
    C: tl.constexpr,              # int
    H_in: tl.constexpr,           # int
    W_in: tl.constexpr,           # int
    H_out: tl.constexpr,          # int
    W_out: tl.constexpr,          # int
    # strides (in elements)
    stride_n: tl.constexpr,       # int
    stride_c: tl.constexpr,       # int
    stride_h: tl.constexpr,       # int
    stride_w: tl.constexpr,       # int
    out_stride_n: tl.constexpr,   # int
    out_stride_c: tl.constexpr,   # int
    out_stride_h: tl.constexpr,   # int
    out_stride_w: tl.constexpr,   # int
    # meta-params
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # Program ids
    pid_nc = tl.program_id(0)
    pid_h  = tl.program_id(1)
    pid_w  = tl.program_id(2)

    # Recover n and c from flattened nc
    n = pid_nc // C
    c = pid_nc % C

    # Tile start
    h_start = pid_h * BLOCK_H
    w_start = pid_w * BLOCK_W

    # Vectors for this tile (1D)
    rh = tl.arange(0, BLOCK_H)     # [BH]
    rw = tl.arange(0, BLOCK_W)     # [BW]

    # 2D coordinates from 1D
    HH = h_start + rh[:, None]     # [BH, 1]
    WW = w_start + rw[None, :]     # [1, BW]

    # Mask within bounds
    mask = (HH < H_out) & (WW < W_out)

    # Base offsets for this (n, c)
    base_in  = n * stride_n + c * stride_c
    base_out = n * out_stride_n + c * out_stride_c

    # Row/col strides
    in_row_stride  = stride_h
    in_col_stride  = stride_w
    out_row_stride = out_stride_h
    out_col_stride = out_stride_w

    # Compute source indices with reduced work:
    # in_h depends only on HH (row); in_w depends only on WW (col).
    HH_i32 = HH.to(tl.int32)
    WW_i32 = WW.to(tl.int32)
    H_in_i32 = tl.full((), H_in, tl.int32)
    W_in_i32 = tl.full((), W_in, tl.int32)
    H_out_i32 = tl.full((), H_out, tl.int32)
    W_out_i32 = tl.full((), W_out, tl.int32)

    # Row source indices [BH,1]
    num_h = HH_i32 * H_in_i32
    den_h = H_out_i32
    in_h = (num_h // den_h).to(tl.int32)  # [BH,1]

    # Col source indices [1,BW]
    num_w = WW_i32 * W_in_i32
    den_w = W_out_i32
    in_w = (num_w // den_w).to(tl.int32)  # [1,BW]

    # Expand to 2D offsets
    # Input: base + in_h*row + in_w*col
    inp_row = in_h * in_row_stride       # [BH,1]
    inp_col = in_w * in_col_stride       # [1,BW]
    inp_offsets = base_in + inp_row + inp_col  # [BH,BW]

    # Output: base + HH*row + WW*col
    out_row = HH * out_row_stride       # [BH,1]
    out_col = WW * out_col_stride       # [1,BW]
    out_offsets = base_out + out_row + out_col  # [BH,BW]

    # Load and store
    vals = tl.load(x_ptr + inp_offsets, mask=mask, other=0)
    tl.store(y_ptr + out_offsets, vals, mask=mask)


def _triton_nearest_resize(x: torch.Tensor,
                           out_size: tuple[int, int] | None = None,
                           scale_factor: tuple[float, float] | float | None = None) -> torch.Tensor:
    """
    Resize a 4D NCHW tensor using nearest-neighbor interpolation with Triton.
    Exactly one of out_size or scale_factor must be provided.

    Args:
      x: torch.Tensor, shape (N, C, H, W), CUDA.
      out_size: tuple(H_out, W_out)
      scale_factor: float or tuple(h_scale, w_scale)

    Returns:
      y: torch.Tensor with shape (N, C, H_out, W_out)
    """
    assert x.dim() == 4, f"Expected 4D NCHW, got shape {tuple(x.shape)}"
    assert x.is_cuda, "Triton kernel requires CUDA tensor"
    N, C, H_in, W_in = x.shape

    if out_size is None and scale_factor is None:
        raise ValueError("One of out_size or scale_factor must be provided")
    if out_size is not None and scale_factor is not None:
        raise ValueError("Provide only one of out_size or scale_factor")

    if out_size is not None:
        H_out, W_out = int(out_size[0]), int(out_size[1])
        scale_factor = None
    else:
        if isinstance(scale_factor, (float, int)):
            h_scale = float(scale_factor)
            w_scale = float(scale_factor)
        else:
            h_scale, w_scale = float(scale_factor[0]), float(scale_factor[1])
        H_out = max(int(math.floor(H_in * h_scale)), 1)
        W_out = max(int(math.floor(W_in * w_scale)), 1)

    # Allocate output
    y = torch.empty((N, C, H_out, W_out), device=x.device, dtype=x.dtype)

    # Get strides in elements
    sN, sC, sH, sW = x.stride()
    out_sN, out_sC, out_sH, out_sW = y.stride()

    # Launch grid; BLOCKs are chosen by autotune
    def grid(meta):
        BH = meta["BLOCK_H"]
        BW = meta["BLOCK_W"]
        return (N * C,
                triton.cdiv(H_out, BH),
                triton.cdiv(W_out, BW))

    nearest_2d_resize_kernel[grid](
        x, y,
        N, C, H_in, W_in, H_out, W_out,
        sN, sC, sH, sW,
        out_sN, out_sC, out_sH, out_sW,
    )

    return y


class ModelNew(nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        size: int | tuple[int, ...] | None = None,
        scale_factor: float | tuple[float, ...] | None = None,
        mode: str = "nearest",
        align_corners: bool | None = None,
    ) -> torch.Tensor:
        # Fallbacks and validations
        if mode != "nearest":
            return F.interpolate(
                x,
                size=size,
                scale_factor=scale_factor,
                mode=mode,
                align_corners=align_corners,
            )

        # Normalize size argument
        if isinstance(size, int):
            raise NotImplementedError("ModelNew Triton path currently supports 2D resizing only")
        if isinstance(size, (tuple, list)):
            if len(size) != 2:
                raise NotImplementedError("ModelNew Triton path currently supports 2D resizing only")
            out_size = (int(size[0]), int(size[1]))
            scale_factor = None
        else:
            out_size = None

        # CPU or non-fp16: fall back
        if not x.is_cuda or x.dtype != torch.float16:
            return F.interpolate(
                x,
                size=out_size,
                scale_factor=scale_factor,
                mode=mode,
                align_corners=align_corners,
            )

        # Use Triton
        y = _triton_nearest_resize(x, out_size=out_size, scale_factor=scale_factor)
        return y

Interpolate = ModelNew
