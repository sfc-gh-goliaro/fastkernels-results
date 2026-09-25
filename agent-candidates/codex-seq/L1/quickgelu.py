"""QuickGELU activation: x * sigmoid(1.702 * x).

Approximation of GELU used in Qwen2-VL vision encoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _quickgelu_kernel(x_ptr, out_ptr, n_elements: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    out = x * tl.sigmoid(1.702 * x)
    tl.store(out_ptr + offsets, out, mask=mask)



class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        n_elements = x.numel()
        _quickgelu_kernel[(triton.cdiv(n_elements, 512),)](
            x, out, n_elements, BLOCK_SIZE=512, num_warps=4
        )
        return out
