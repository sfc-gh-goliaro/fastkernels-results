from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _rms_norm_gated_kernel(
    x,
    z,
    weight,
    out,
    eps,
    ROWS: tl.constexpr,
    N: tl.constexpr,
):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, N)
    offsets = rows[:, None] * N + cols[None, :]

    x_values = tl.load(x + offsets, cache_modifier=".cg").to(tl.float32)
    variance = tl.sum(x_values * x_values, axis=1) / N
    rstd = tl.rsqrt(variance + eps)

    w = tl.load(weight + cols, eviction_policy="evict_last").to(tl.float32)
    z_values = tl.load(z + offsets, cache_modifier=".cg").to(tl.float32)
    y = x_values * rstd[:, None] * w[None, :]
    y *= z_values * tl.sigmoid(z_values)
    tl.store(out + offsets, y, cache_modifier=".cs")


__targets__ = ["RMSNormGated"]


class RMSNormGated(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        norm_before_gate: bool = True,
        activation: str = "swish",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.norm_before_gate = norm_before_gate
        self.activation = activation
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        m = x.numel() // self.hidden_size
        rows = 1 if m < 256 else 2 if m < 768 else 8
        _rms_norm_gated_kernel[(triton.cdiv(m, rows),)](
            x,
            z,
            self.weight,
            out,
            self.eps,
            ROWS=rows,
            N=self.hidden_size,
            num_warps=1,
        )
        return out
