"""Softmax / LogSoftmax activations (via torch.nn.functional)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _softmax_kernel(
    x,
    y,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    values = tl.load(x + row * n_cols + offsets, mask=mask, other=-float("inf"))
    values = values - tl.max(values, axis=0)
    numerator = tl.exp(values)
    result = numerator / tl.sum(numerator, axis=0)
    tl.store(y + row * n_cols + offsets, result, mask=mask)


@triton.jit
def _softmax_rows_kernel(
    x,
    y,
    n_rows: tl.constexpr,
    n_cols: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    cols = tl.arange(0, BLOCK_COLS)
    offsets = rows[:, None] * n_cols + cols[None, :]
    mask = (rows[:, None] < n_rows) & (cols[None, :] < n_cols)
    values = tl.load(x + offsets, mask=mask, other=-float("inf"))
    values = values - tl.max(values, axis=1)[:, None]
    numerator = tl.exp(values)
    result = numerator / tl.sum(numerator, axis=1)[:, None]
    tl.store(y + offsets, result, mask=mask)


@triton.jit
def _softmax_dim1_16_kernel(
    x,
    y,
    n_inner: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    outer = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    channels = tl.arange(0, 16)
    offsets = outer * (16 * n_inner) + channels[:, None] * n_inner + cols[None, :]
    mask = cols[None, :] < n_inner
    values = tl.load(x + offsets, mask=mask, other=-float("inf"))
    values = values - tl.max(values, axis=0)
    numerator = tl.exp(values)
    result = numerator / tl.sum(numerator, axis=0)
    tl.store(y + offsets, result, mask=mask)


@triton.jit
def _softmax_interleaved_16_kernel(x, y):
    block = tl.program_id(0)
    channels = tl.arange(0, 16)
    lanes = tl.arange(0, 8)
    offsets = block * 128 + channels[:, None] * 8 + lanes[None, :]
    values = tl.load(x + offsets)
    values = values - tl.max(values, axis=0)
    numerator = tl.exp(values)
    result = numerator / tl.sum(numerator, axis=0)
    tl.store(y + offsets, result)


def _softmax(x: torch.Tensor) -> torch.Tensor:
    n_cols = x.shape[-1]
    n_rows = x.numel() // n_cols
    output = torch.empty_like(x)

    if n_cols == 77:
        _softmax_rows_kernel[(triton.cdiv(n_rows, 2),)](
            x,
            output,
            n_rows=n_rows,
            n_cols=n_cols,
            BLOCK_ROWS=2,
            BLOCK_COLS=128,
            num_warps=4,
        )
        return output
    if n_cols == 512:
        _softmax_rows_kernel[(triton.cdiv(n_rows, 4),)](
            x,
            output,
            n_rows=n_rows,
            n_cols=n_cols,
            BLOCK_ROWS=4,
            BLOCK_COLS=512,
            num_warps=2,
        )
        return output

    if n_cols <= 16:
        block_size, num_warps = 32, 1
    elif n_cols <= 128:
        block_size, num_warps = 128, 1
    elif n_cols <= 512:
        block_size, num_warps = 512, 2
    else:
        block_size, num_warps = triton.next_power_of_2(n_cols), 8

    _softmax_kernel[(n_rows,)](
        x,
        output,
        n_cols=n_cols,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return output


class Softmax(nn.Module):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            self.dim == -1
            and x.ndim == 5
            and x.shape[-1] == 16
            and x.stride()[-3:] == (1, 128, 8)
        ):
            output = torch.empty_like(x)
            _softmax_interleaved_16_kernel[(x.numel() // 128,)](
                x, output, num_warps=1
            )
            return output
        if (
            self.dim == 1
            and x.ndim == 4
            and x.shape[1] == 16
            and x.stride(1) == x.shape[-1]
            and x.stride(2) == 16 * x.shape[-1]
            and x.stride(3) == 1
        ):
            output = torch.empty_like(x)
            n_inner = x.shape[-1]
            n_outer = x.numel() // (16 * n_inner)
            _softmax_dim1_16_kernel[
                (n_outer, triton.cdiv(n_inner, 64))
            ](
                x,
                output,
                n_inner=n_inner,
                BLOCK_N=64,
                num_warps=1,
            )
            return output
        if (self.dim == -1 or self.dim == x.ndim - 1) and x.is_contiguous():
            return _softmax(x)
        return F.softmax(x, dim=self.dim)


class LogSoftmax(nn.Module):
    """Numerically-stable log-softmax. Used by the TTT-E2E inner-loop CE loss."""

    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(x, dim=self.dim)
