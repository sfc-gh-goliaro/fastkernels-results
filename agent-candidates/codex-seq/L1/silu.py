"""SiLU (Swish) activation: x * sigmoid(x)."""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

from fastkernels.infra.cuda_ext import lazy_op


_C = lazy_op("fk_candidate_silu_poly", "silu_cuda.cu")


@triton.jit
def _silu_kernel(x_ptr, out_ptr, n_elements: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    x_squared = x * x
    even = x_squared * (
        0.2395166094
        + x_squared * (-0.0138038741 + x_squared * 0.0004331403)
    )
    central = 0.5 * x + even
    out = tl.maximum(-0.28, tl.minimum(central, tl.maximum(x, 0.0)))
    tl.store(out_ptr + offsets, out, mask=mask)


class SiLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() >= 4 * 1024 * 1024:
            out = torch.empty_like(x)
            n_elements = x.numel()
            _silu_kernel[(triton.cdiv(n_elements, 8192),)](
                x, out, n_elements, BLOCK_SIZE=8192, num_warps=8
            )
            return out
        return _C.silu(x)
