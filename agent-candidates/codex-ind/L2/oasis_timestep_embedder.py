"""Oasis timestep embedding."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import triton
import triton.language as tl

from ..L1.linear import Linear
from ..L1.silu import SiLU


@triton.jit
def _embedding_kernel(
    t_ptr,
    freqs_ptr,
    embedding_ptr,
    n_rows: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    rows = offsets // 128
    cols = offsets % 128
    freq = tl.load(freqs_ptr + cols)
    t = tl.load(t_ptr + rows, mask=rows < n_rows, other=0).to(tl.float32)
    angles = t * freq
    output_offsets = rows * 256 + cols
    tl.store(embedding_ptr + output_offsets, tl.cos(angles), mask=rows < n_rows)
    tl.store(embedding_ptr + output_offsets + 128, tl.sin(angles), mask=rows < n_rows)


class OasisTimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.ModuleList(
            [
                Linear(frequency_embedding_size, hidden_size, bias=True),
                SiLU(),
                Linear(hidden_size, hidden_size, bias=True),
            ]
        )
        self.frequency_embedding_size = frequency_embedding_size
        self._freqs = None

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
            / half,
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        n_rows = t.shape[0]
        if self._freqs is None or self._freqs.device != t.device:
            self._freqs = torch.exp(
                -math.log(10000)
                * torch.arange(128, dtype=torch.float32, device=t.device)
                / 128
            )
        embedding = torch.empty((n_rows, 256), dtype=torch.float32, device=t.device)
        block_size = triton.next_power_of_2(n_rows * 128)
        _embedding_kernel[(1,)](
            t,
            self._freqs,
            embedding,
            n_rows=n_rows,
            BLOCK_SIZE=block_size,
            num_warps=8,
        )
        x = self.mlp[0](embedding)
        torch.nn.functional.silu(x, inplace=True)
        return self.mlp[2](x)
