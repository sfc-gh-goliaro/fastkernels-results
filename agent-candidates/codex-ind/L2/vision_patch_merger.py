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
def _layer_norm_1152_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    eps: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < WIDTH
    x = tl.load(x_ptr + row * WIDTH + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / WIDTH
    centered = tl.where(mask, x - mean, 0.0)
    variance = tl.sum(centered * centered, axis=0) / WIDTH
    normalized = centered * tl.rsqrt(variance + eps)
    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + cols, mask=mask, other=0.0)
    tl.store(out_ptr + row * WIDTH + cols, normalized * weight + bias, mask=mask)


@triton.jit
def _bias_gelu_inplace_kernel(
    x_ptr,
    bias_ptr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    offsets = row * WIDTH + cols
    mask = cols < WIDTH
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    bias = tl.load(bias_ptr + cols, mask=mask).to(tl.float32)
    x += bias
    x3 = x * x * x
    out = x / (
        1.0 + tl.exp(-1.5957691216057308 * (x + 0.044715 * x3))
    )
    tl.store(x_ptr + offsets, out, mask=mask)


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
        if (not self.use_postshuffle_norm and x.is_cuda
                and x.dtype == torch.bfloat16 and x.shape[-1] == 1152):
            # Captured Qwen vision inputs have enough rows for one-warp CTAs.
            rows = x.numel() // 1152
            normalized = torch.empty_like(x)
            _layer_norm_1152_kernel[(rows,)](
                x, self.norm.weight, self.norm.bias, normalized,
                eps=self.norm.eps, WIDTH=1152, BLOCK=2048,
                num_warps=1,
            )
            x = normalized.view(-1, self.hidden_size)
        elif self.use_postshuffle_norm:
            x = self.norm(x.view(-1, self.hidden_size))
        else:
            x = self.norm(x).view(-1, self.hidden_size)
        if x.is_cuda and x.dtype == torch.bfloat16:
            # Fold fc1's bias into GELU and update its temporary in place.
            x = torch.mm(x, self.fc1.weight.t())
            _bias_gelu_inplace_kernel[
                (x.shape[0], triton.cdiv(self.hidden_size, 1024))
            ](
                x, self.fc1.bias, WIDTH=self.hidden_size, BLOCK=1024,
                num_warps=4,
            )
        else:
            x = self.act(self.fc1(x))
        return self.fc2(x)
