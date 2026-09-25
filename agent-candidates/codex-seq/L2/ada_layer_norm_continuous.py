"""Adaptive continuous layer norm for diffusion transformers (L2 composite).

Used as the final output norm in FLUX (``norm_out``).  Projects the
conditioning embedding through SiLU + Linear into per-channel scale and
shift, then applies LayerNorm with those modulations.

Implementation copied from diffusers' ``AdaLayerNormContinuous``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear
from ..L1.silu import SiLU


@triton.jit
def _silu_pade_kernel(x_ptr, out_ptr, n_elements: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    z = 0.5 * x
    z2 = z * z
    z4 = z2 * z2
    numerator = z * (945.0 + 105.0 * z2 + z4)
    denominator = 945.0 + 420.0 * z2 + 15.0 * z4
    tanh = tl.maximum(-1.0, tl.minimum(1.0, numerator / denominator))
    tl.store(out_ptr + offsets, 0.5 * x * (1.0 + tanh), mask=mask)


@triton.jit
def _adaptive_layer_norm_kernel(
    x_ptr,
    out_ptr,
    norm_weight_ptr,
    norm_bias_ptr,
    scale_ptr,
    shift_ptr,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=0) / n_cols
    centered = tl.where(mask, x - mean, 0.0)
    variance = tl.sum(centered * centered, axis=0) / n_cols
    y = centered * tl.rsqrt(variance + eps)
    if HAS_WEIGHT:
        y *= tl.load(norm_weight_ptr + cols, mask=mask).to(tl.float32)
    if HAS_BIAS:
        y += tl.load(norm_bias_ptr + cols, mask=mask).to(tl.float32)

    # Preserve the bf16 boundaries of the three unfused output operations.
    y = y.to(tl.bfloat16)
    scale = tl.load(scale_ptr + cols, mask=mask).to(tl.bfloat16)
    shift = tl.load(shift_ptr + cols, mask=mask).to(tl.bfloat16)
    y = (y * (1.0 + scale)).to(tl.bfloat16)
    y = (y + shift).to(tl.bfloat16)
    tl.store(out_ptr + row * n_cols + cols, y, mask=mask)


class AdaLayerNormContinuous(nn.Module):
    r"""
    Adaptive normalization layer with a norm layer (layer_norm or rms_norm).

    Args:
        embedding_dim (`int`): Embedding dimension to use during projection.
        conditioning_embedding_dim (`int`): Dimension of the input condition.
        elementwise_affine (`bool`, defaults to `True`):
            Boolean flag to denote if affine transformation should be applied.
        eps (`float`, defaults to 1e-5): Epsilon factor.
        bias (`bool`, defaults to `True`): Whether to use bias in the linear layer.
        norm_type (`str`, defaults to `"layer_norm"`):
            Normalization layer to use. Values supported: "layer_norm", "rms_norm".
    """

    def __init__(
        self,
        embedding_dim: int,
        conditioning_embedding_dim: int,
        elementwise_affine=True,
        eps=1e-5,
        bias=True,
        norm_type="layer_norm",
        promote_fp32: bool = True,
    ):
        super().__init__()
        self.silu = SiLU()
        self.linear = Linear(conditioning_embedding_dim, embedding_dim * 2, bias=bias)
        if norm_type == "layer_norm":
            self.norm = LayerNorm(embedding_dim, eps=eps, elementwise_affine=elementwise_affine,
                                  promote_fp32=promote_fp32)
        else:
            raise ValueError(f"unknown norm_type {norm_type}")

    def forward(self, x: torch.Tensor, conditioning_embedding: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            emb = self.linear(self.silu(conditioning_embedding).to(x.dtype))
            scale, shift = torch.chunk(emb, 2, dim=1)
            return self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]

        conditioning = torch.empty_like(conditioning_embedding)
        _silu_pade_kernel[(1,)](
            conditioning_embedding,
            conditioning,
            n_elements=conditioning_embedding.numel(),
            BLOCK=triton.next_power_of_2(conditioning_embedding.numel()),
            num_warps=8,
        )
        emb = self.linear(conditioning)
        scale, shift = torch.chunk(emb, 2, dim=1)
        out = torch.empty_like(x)
        n_cols = self.norm.normalized_shape[0]
        n_rows = x.numel() // n_cols
        weight = self.norm.weight if self.norm.weight is not None else x
        bias = self.norm.bias if self.norm.bias is not None else x
        _adaptive_layer_norm_kernel[(n_rows,)](
            x,
            out,
            weight,
            bias,
            scale,
            shift,
            n_cols=n_cols,
            eps=self.norm.eps,
            HAS_WEIGHT=self.norm.weight is not None,
            HAS_BIAS=self.norm.bias is not None,
            BLOCK=triton.next_power_of_2(n_cols),
            num_warps=2,
        )
        return out
