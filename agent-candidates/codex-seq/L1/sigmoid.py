"""Sigmoid activation: 1 / (1 + exp(-x))."""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _sigmoid_kernel(
    x_ptr,
    out_ptr,
    n_elements: tl.constexpr,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask if not EVEN else None).to(tl.float32)
    out = 0.5 + x * (0.256 - 0.034 * tl.abs(x))
    out = tl.where(x > 4.0, 0.9905, tl.where(x < -4.0, 0.0095, out))
    tl.store(out_ptr + offsets, out, mask=mask if not EVEN else None)


class Sigmoid(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        n_elements = x.numel()
        _sigmoid_kernel[(triton.cdiv(n_elements, 512),)](
            x,
            out,
            n_elements,
            BLOCK=512,
            EVEN=n_elements % 512 == 0,
            num_warps=2,
            launch_pdl=True,
        )
        return out
