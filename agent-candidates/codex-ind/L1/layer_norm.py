"""Fused LayerNorm specialized for CUDA inference."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _layer_norm_kernel(
    x_ptr,
    y_ptr,
    weight_ptr,
    bias_ptr,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=0) / n_cols
    centered = tl.where(mask, x - mean, 0.0)
    variance = tl.sum(centered * centered, axis=0) / n_cols
    y = centered * tl.rsqrt(variance + eps)

    if HAS_WEIGHT:
        y *= tl.load(weight_ptr + cols, mask=mask).to(tl.float32)
    if HAS_BIAS:
        y += tl.load(bias_ptr + cols, mask=mask).to(tl.float32)
    tl.store(y_ptr + row * n_cols + cols, y, mask=mask)


class LayerNorm(nn.Module):
    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        create_scale: bool = True,
        create_offset: bool = True,
        promote_fp32: bool = True,
    ):
        super().__init__()
        self.normalized_shape = (normalized_shape,)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.promote_fp32 = promote_fp32

        if elementwise_affine and create_scale:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
        else:
            self.register_parameter("weight", None)

        if elementwise_affine and create_offset:
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            if not self.promote_fp32:
                return F.layer_norm(
                    x, self.normalized_shape, self.weight, self.bias, self.eps,
                )
            dtype = x.dtype
            weight = self.weight.float() if self.weight is not None else None
            bias = self.bias.float() if self.bias is not None else None
            return F.layer_norm(
                x.float(), self.normalized_shape, weight, bias, self.eps,
            ).to(dtype)
        if not x.is_contiguous():
            x = x.contiguous()

        n_cols = self.normalized_shape[0]
        n_rows = x.numel() // n_cols
        out = torch.empty_like(x)
        weight_ptr = self.weight if self.weight is not None else x
        bias_ptr = self.bias if self.bias is not None else x
        block_size = triton.next_power_of_2(n_cols)
        num_warps = 1 if block_size <= 512 else (
            2 if block_size >= 8192 else 4
        )
        _layer_norm_kernel[(n_rows,)](
            x,
            out,
            weight_ptr,
            bias_ptr,
            n_cols=n_cols,
            eps=self.eps,
            HAS_WEIGHT=self.weight is not None,
            HAS_BIAS=self.bias is not None,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
        )
        return out
