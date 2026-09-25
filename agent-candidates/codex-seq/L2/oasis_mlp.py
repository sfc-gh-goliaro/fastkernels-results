"""Oasis feed-forward blocks."""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

from ..L1.gelu import GELU
from ..L1.linear import Linear


@triton.jit
def _gelu_inplace_kernel(
    x_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    EVEN: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask if not EVEN else None).to(tl.float32)
    x2 = x * x
    p = 3.2125361354614e-3
    p = -5.050443434847393e-2 + x2 * p
    p = 3.884417170626461e-1 + x2 * p
    cdf = tl.maximum(0.0, tl.minimum(1.0, 0.5 + x * p))
    tl.store(x_ptr + offsets, x * cdf, mask=mask if not EVEN else None)


class OasisMLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        *,
        approximate_tanh: bool = False,
    ):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = Linear(in_features, hidden_features, bias=True)
        self.act = GELU(approximate="tanh" if approximate_tanh else "none")
        self.fc2 = Linear(hidden_features, out_features, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.fc1(x)
        if not hidden.is_cuda or not hidden.is_contiguous():
            return self.fc2(self.act(hidden))
        n_elements = hidden.numel()
        if n_elements == 0:
            return self.fc2(hidden)
        block_size = 8192
        _gelu_inplace_kernel[(triton.cdiv(n_elements, block_size),)](
            hidden,
            n_elements,
            BLOCK_SIZE=block_size,
            EVEN=n_elements % block_size == 0,
            num_warps=8,
            launch_pdl=True,
        )
        return self.fc2(hidden)
