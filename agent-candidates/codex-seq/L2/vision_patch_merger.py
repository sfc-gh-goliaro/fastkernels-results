"""Patch merger for Qwen vision encoders.

Merges spatial_merge_size^2 patches into one via norm + 2-layer MLP.

Unified across Qwen2-VL and Qwen3-VL:
  - use_postshuffle_norm: Qwen3 DeepStack mergers norm after spatial reshape.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

from ..L1.gelu import GELU
from ..L1.layer_norm import LayerNorm
from .parallel_linear import ColumnParallelLinear, RowParallelLinear


@triton.jit
def _layer_norm_inplace_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0).to(
        tl.float32
    )
    mean = tl.sum(x, axis=0) / n_cols
    centered = tl.where(mask, x - mean, 0.0)
    variance = tl.sum(centered * centered, axis=0) / n_cols
    y = centered * tl.rsqrt(variance + eps)
    y *= tl.load(weight_ptr + cols, mask=mask).to(tl.float32)
    y += tl.load(bias_ptr + cols, mask=mask).to(tl.float32)
    tl.store(x_ptr + row * n_cols + cols, y, mask=mask)


@triton.jit
def _gelu_inplace_kernel(x_ptr, n_elements: tl.constexpr,
                         BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    x2 = x * x
    p = 3.82411768152011e-8
    p = -2.3895584944491883e-6 + x2 * p
    p = 6.359391292506229e-5 + x2 * p
    p = -9.63761177160297e-4 + x2 * p
    p = 9.481815833772472e-3 + x2 * p
    p = -6.606990690746545e-2 + x2 * p
    p = 3.988825261356899e-1 + x2 * p
    cdf = tl.maximum(0.0, tl.minimum(1.0, 0.5 + x * p))
    cdf = tl.where(x > 4.0, 1.0, tl.where(x < -4.0, 0.0, cdf))
    y = x * cdf
    tl.store(x_ptr + offsets, y, mask=mask)


class VisionPatchMerger(nn.Module):
    """Patch merger: norm -> flatten -> MLP(fc1, GELU, fc2).

    Qwen3 DeepStack mergers set use_postshuffle_norm=True to norm after reshape.
    """

    def __init__(self, d_model: int, context_dim: int,
                 spatial_merge_size: int = 2,
                 use_postshuffle_norm: bool = False,
                 eps: float = 1e-6):
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size ** 2)
        self.use_postshuffle_norm = use_postshuffle_norm
        norm_dim = self.hidden_size if use_postshuffle_norm else context_dim
        # See VisionBlock: vLLM's vision path uses plain nn.LayerNorm on
        # bf16, and our fp32 promotion costs two full-tensor copies here.
        self.norm = LayerNorm(norm_dim, eps=eps, promote_fp32=False)
        self.fc1 = ColumnParallelLinear(self.hidden_size, self.hidden_size, bias=True)
        self.act = GELU()
        self.fc2 = RowParallelLinear(self.hidden_size, d_model, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_dim = self.norm.normalized_shape[0]
        if x.is_cuda and x.is_contiguous():
            n_rows = x.numel() // norm_dim
            block_size = triton.next_power_of_2(norm_dim)
            _layer_norm_inplace_kernel[(n_rows,)](
                x,
                self.norm.weight,
                self.norm.bias,
                n_cols=norm_dim,
                eps=self.norm.eps,
                BLOCK_SIZE=block_size,
                num_warps=1,
            )
            x = x.view(-1, self.hidden_size)
        elif self.use_postshuffle_norm:
            x = self.norm(x.view(-1, self.hidden_size))
        else:
            x = self.norm(x).view(-1, self.hidden_size)
        x = self.fc1(x)
        if x.is_cuda and x.is_contiguous():
            n_elements = x.numel()
            _gelu_inplace_kernel[(triton.cdiv(n_elements, 1024),)](
                x, n_elements, BLOCK_SIZE=1024, num_warps=1,
            )
        else:
            x = self.act(x)
        x = self.fc2(x)
        return x
