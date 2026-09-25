"""YOLOv10 tensor concatenation op."""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _concat_channels_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    x_batch_stride: tl.constexpr,
    y_batch_stride: tl.constexpr,
    out_batch_stride: tl.constexpr,
    x_blocks: tl.constexpr,
    BLOCK: tl.constexpr,
):
    block = tl.program_id(0)
    batch = tl.program_id(1)
    lane_offsets = tl.arange(0, BLOCK)

    from_y = block >= x_blocks
    source = tl.where(from_y, y_ptr, x_ptr)
    source_stride = tl.where(from_y, y_batch_stride, x_batch_stride)
    source_block = block - tl.where(from_y, x_blocks, 0)
    source_offsets = source_block * BLOCK + lane_offsets
    values = tl.load(source + batch * source_stride + source_offsets)
    out_offsets = source_offsets + tl.where(from_y, x_batch_stride, 0)
    tl.store(out_ptr + batch * out_batch_stride + out_offsets, values)


class YOLOConcat(nn.Module):
    def __init__(self, dimension: int = 1):
        super().__init__()
        self.d = dimension

    def forward(self, xs: list[torch.Tensor]) -> torch.Tensor:
        if (
            self.d != 1
            or len(xs) != 2
            or xs[0].ndim != 4
            or not xs[0].is_cuda
            or not xs[0].is_contiguous()
            or not xs[1].is_contiguous()
            or xs[0].dtype != xs[1].dtype
            or xs[0].device != xs[1].device
            or xs[0].shape[0] != xs[1].shape[0]
            or xs[0].shape[2:] != xs[1].shape[2:]
            or xs[0].shape[0] == 0
        ):
            return torch.cat(xs, self.d)

        x, y = xs
        n, c, h, w = x.shape
        x_batch_stride = x.numel() // n
        y_batch_stride = y.numel() // n
        out_batch_stride = x_batch_stride + y_batch_stride
        block = 2048 if out_batch_stride >= 1_000_000 else 1024
        if x_batch_stride % block != 0 or y_batch_stride % block != 0:
            return torch.cat(xs, self.d)

        out = torch.empty(
            (n, c + y.shape[1], h, w),
            dtype=x.dtype,
            device=x.device,
        )
        x_blocks = x_batch_stride // block
        y_blocks = y_batch_stride // block
        grid = (x_blocks + y_blocks, n)
        _concat_channels_kernel[grid](
            x,
            y,
            out,
            x_batch_stride,
            y_batch_stride,
            out_batch_stride,
            x_blocks,
            BLOCK=block,
            num_warps=8 if block == 2048 else 4,
        )
        return out
