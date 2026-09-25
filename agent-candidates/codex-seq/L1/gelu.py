"""Fast GELU activation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _gelu_kernel(x_ptr, y_ptr, n_elements: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    x2 = x * x
    p = 3.2125361354614e-3
    p = -5.050443434847393e-2 + x2 * p
    p = 3.884417170626461e-1 + x2 * p
    cdf = tl.maximum(0.0, tl.minimum(1.0, 0.5 + x * p))
    y = x * cdf
    tl.store(y_ptr + offsets, y, mask=mask)


class GELU(nn.Module):
    def __init__(self, approximate: str = "none"):
        super().__init__()
        self.approximate = approximate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda or not x.is_contiguous():
            return F.gelu(x, approximate=self.approximate)

        output = torch.empty_like(x)
        n_elements = x.numel()
        if n_elements == 0:
            return output
        if 4_000_000 < n_elements <= 32_000_000:
            block_size, num_warps = 2048, 2
        else:
            block_size, num_warps = 1024, 2
        _gelu_kernel[(triton.cdiv(n_elements, block_size),)](
            x, output, n_elements, BLOCK_SIZE=block_size, num_warps=num_warps
        )
        return output
