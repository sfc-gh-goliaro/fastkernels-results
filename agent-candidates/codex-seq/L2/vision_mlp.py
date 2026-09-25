"""Vision MLP for Qwen vision transformer blocks.

Unified across Qwen2-VL (QuickGELU) and Qwen3-VL (SiLU) activations.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
import triton
import triton.language as tl

from ..L1.quickgelu import QuickGELU
from .parallel_linear import ColumnParallelLinear, RowParallelLinear


@triton.jit
def _gelu_inplace_kernel(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    x2 = x * x
    p = 3.2125361354614e-3
    p = -5.050443434847393e-2 + x2 * p
    p = 3.884417170626461e-1 + x2 * p
    cdf = tl.maximum(0.0, tl.minimum(1.0, 0.5 + x * p))
    tl.store(x_ptr + offsets, x * cdf, mask=mask)


class VisionMLP(nn.Module):
    """Vision encoder MLP with configurable activation.

    Qwen2-VL uses QuickGELU (default); Qwen3-VL uses F.silu.
    """

    def __init__(self, in_features: int, hidden_features: int,
                 act_fn: Callable[[torch.Tensor], torch.Tensor] = QuickGELU(),
                 bias: bool = True):
        super().__init__()
        self.fc1 = ColumnParallelLinear(in_features, hidden_features, bias=bias)
        self.fc2 = RowParallelLinear(hidden_features, in_features, bias=bias)
        self.act_fn = act_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            x.is_cuda
            and x.dtype == torch.bfloat16
            and self.act_fn.__class__.__name__ == "GELU"
            and self.fc1.bias is not None
            and self.fc2.bias is not None
            and self.fc2.tp_size == 1
        ):
            x_2d = x.reshape(-1, x.shape[-1])
            m = x_2d.shape[0]
            n = self.fc2.weight.shape[0]
            out = torch.empty((m, n), device=x.device, dtype=x.dtype)
            hidden = torch.addmm(self.fc1.bias, x_2d, self.fc1.weight.t())
            _gelu_inplace_kernel[(triton.cdiv(hidden.numel(), 2048),)](
                hidden, hidden.numel(), BLOCK_SIZE=2048, num_warps=2
            )
            torch.addmm(self.fc2.bias, hidden, self.fc2.weight.t(), out=out)
            return out.reshape(*x.shape[:-1], n)
        return self.fc2(self.act_fn(self.fc1(x)))
