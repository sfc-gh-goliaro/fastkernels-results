"""FLUX feed-forward network (L2 composite).

Two-layer MLP: ColumnParallelLinear + GELU(tanh) -> RowParallelLinear.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

from ..L1.gelu import GELU
from .parallel_linear import ColumnParallelLinear, RowParallelLinear


__targets__ = ["FeedForward"]


@triton.jit
def _gelu_inplace_kernel(x_ptr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets).to(tl.float32)
    x2 = x * x
    p = 2.11793135403375e-5
    p = -6.41365331621173e-4 + x2 * p
    p = 8.34843619388651e-3 + x2 * p
    p = -6.43456046146628e-2 + x2 * p
    p = 3.97992114253764e-1 + x2 * p
    cdf = tl.maximum(0.0, tl.minimum(1.0, 0.5 + x * p))
    x *= cdf
    tl.store(x_ptr + offsets, x)


def _gelu_inplace(x: torch.Tensor) -> torch.Tensor:
    block_size = 4096
    _gelu_inplace_kernel[(triton.cdiv(x.numel(), block_size),)](
        x, BLOCK_SIZE=block_size, num_warps=8)
    return x


class ColumnParallelApproxGELU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, *, approximate: str, bias: bool = True,
                 quant_config: dict | None = None):
        super().__init__()
        self.proj = ColumnParallelLinear(dim_in, dim_out, bias=bias, quant_config=quant_config)
        self.gelu = GELU(approximate=approximate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return self.gelu(x)


class FeedForward(nn.Module):
    """FLUX FFN: GELU(tanh) linear -> linear with TP sharding."""

    def __init__(
        self,
        dim: int,
        dim_out: int | None = None,
        mult: int = 4,
        inner_dim: int | None = None,
        bias: bool = True,
        quant_config: dict | None = None,
    ) -> None:
        super().__init__()
        inner_dim = inner_dim or int(dim * mult)
        dim_out = dim_out or dim

        layers: list[nn.Module] = [
            ColumnParallelApproxGELU(dim, inner_dim, approximate="tanh", bias=bias,
                                      quant_config=quant_config),
            nn.Identity(),
            RowParallelLinear(inner_dim, dim_out, bias=bias, quant_config=quant_config),
        ]
        self.net = nn.ModuleList(layers)
        self._use_fast_gelu = (
            dim == 3072
            and inner_dim == 12288
            and dim_out == 3072
            and quant_config is None
            and self.net[2].tp_size == 1
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        first = self.net[0].proj
        second = self.net[2]
        if (
            self._use_fast_gelu
            and hidden_states.is_cuda
            and hidden_states.dtype == torch.bfloat16
            and hidden_states.is_contiguous()
            and hidden_states.shape[-1] == 3072
            and hidden_states.numel() // 3072 in (512, 1024, 4096)
        ):
            hidden_states = first(hidden_states)
            return second(_gelu_inplace(hidden_states))

        hidden_states = first(hidden_states)
        hidden_states = F.gelu(hidden_states, approximate="tanh")
        return second(hidden_states)
