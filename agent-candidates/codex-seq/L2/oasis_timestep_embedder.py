"""Oasis timestep embedding."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from fastkernels.infra.cuda_ext import lazy_op

from ..L1.linear import Linear
from ..L1.silu import SiLU


_C = lazy_op("fk_oasis_timestep_embedder", "oasis_timestep_embedder.cu")


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
        self.register_buffer("_cached_freqs", torch.empty(0), persistent=False)

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
        if (
            t.is_cuda
            and t.dtype == torch.int64
            and t.ndim == 1
            and self.frequency_embedding_size == 256
            and self.mlp[0].weight.shape == (1024, 256)
            and self.mlp[2].weight.shape == (1024, 1024)
            and self.mlp[0].weight.dtype == torch.float32
        ):
            if self._cached_freqs.device != t.device or self._cached_freqs.numel() != 128:
                self._cached_freqs = torch.exp(
                    -math.log(10000)
                    * torch.arange(128, dtype=torch.float32, device=t.device)
                    / 128
                )
            x = _C.embedding(t, self._cached_freqs)
            x = self.mlp[0](x)
            x = torch.nn.functional.silu(x, inplace=True)
            return self.mlp[2](x)

        x = self.timestep_embedding(t, self.frequency_embedding_size)
        x = self.mlp[0](x)
        x = torch.nn.functional.silu(x)
        x = self.mlp[2](x)
        return x
