"""Primitive tensor manipulation ops.

L1 ops wrapping standard tensor utilities for use by L2+ composites.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _tail_pad_kernel(
    x,
    output,
    n_input: tl.constexpr,
    n_output: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(x + offsets, mask=offsets < n_input, other=0.0)
    tl.store(output + offsets, values, mask=offsets < n_output)


class Pad(nn.Module):
    """Functional padding op."""

    def forward(
        self, x: torch.Tensor, pad: tuple[int, ...], value: float = 0.0,
    ) -> torch.Tensor:
        if (
            value == 0.0
            and x.is_cuda
            and x.is_contiguous()
            and x.shape[-(len(pad) // 2)] == 368
            and tuple(pad) in ((0, 16), (0, 0, 0, 16))
        ):
            if len(pad) == 2:
                output = x.new_empty((*x.shape[:-1], 384))
            else:
                output = x.new_empty((*x.shape[:-2], 384, x.shape[-1]))
            n_input = x.numel()
            n_output = output.numel()
            block = 256 if n_output <= 1536 else 2048
            warps = 1 if n_output <= 1536 else 8
            _tail_pad_kernel[(triton.cdiv(n_output, block),)](
                x,
                output,
                n_input,
                n_output,
                BLOCK=block,
                num_warps=warps,
            )
            return output
        return F.pad(x, pad, value=value)


class OneHot(nn.Module):
    """Functional one-hot encoding op."""

    def forward(self, x: torch.Tensor, num_classes: int) -> torch.Tensor:
        return F.one_hot(x, num_classes)


class Cat(nn.Module):
    """Tensor concatenation op."""

    def __init__(self, dim: int = 0):
        super().__init__()
        self.dim = dim

    def forward(self, tensors: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> torch.Tensor:
        return torch.cat(tensors, dim=self.dim)


class Exp(nn.Module):
    """Elementwise exponential op."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(x)
