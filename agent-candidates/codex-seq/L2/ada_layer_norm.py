"""Adaptive Layer Norm modules for diffusion transformers (L2 composite).

AdaLayerNormZero: 6-output adaLN-Zero for dual-stream FLUX blocks.
AdaLayerNormZeroSingle: 3-output adaLN-Zero for single-stream FLUX blocks.

Implementation copied from diffusers' ``AdaLayerNormZero`` /
``AdaLayerNormZeroSingle`` to keep the code identical.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear
from ..L1.silu import SiLU


@triton.jit
def _ada_layer_norm_kernel(
    x_ptr,
    emb_ptr,
    out_ptr,
    n_cols: tl.constexpr,
    scale_offset: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=0) / n_cols
    centered = tl.where(mask, x - mean, 0.0)
    variance = tl.sum(centered * centered, axis=0) / n_cols
    normalized = (centered * tl.rsqrt(variance + eps)).to(tl.bfloat16)

    shift = tl.load(emb_ptr + cols, mask=mask)
    scale = tl.load(emb_ptr + scale_offset + cols, mask=mask)
    scaled = (normalized * (scale + 1.0)).to(tl.bfloat16)
    tl.store(out_ptr + row * n_cols + cols, scaled + shift, mask=mask)


def _modulated_layer_norm(
    x: torch.Tensor,
    emb: torch.Tensor,
    norm: LayerNorm,
) -> torch.Tensor:
    if not x.is_cuda:
        shift, scale = emb[:, :x.shape[-1]], emb[:, x.shape[-1]:2 * x.shape[-1]]
        return norm(x) * (1 + scale[:, None]) + shift[:, None]

    x = x.contiguous()
    out = torch.empty_like(x)
    n_cols = x.shape[-1]
    _ada_layer_norm_kernel[(x.numel() // n_cols,)](
        x,
        emb,
        out,
        n_cols=n_cols,
        scale_offset=n_cols,
        eps=norm.eps,
        BLOCK_SIZE=triton.next_power_of_2(n_cols),
        num_warps=4,
    )
    return out


@triton.jit
def _silu_kernel(x_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < 3072
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    tl.store(out_ptr + offsets, x * tl.sigmoid(x), mask=mask)


def _silu(x: torch.Tensor) -> torch.Tensor:
    if not x.is_cuda or x.numel() != 3072:
        return F.silu(x)
    out = torch.empty_like(x)
    _silu_kernel[(1,)](x, out, BLOCK_SIZE=4096, num_warps=4)
    return out


class AdaLayerNormZero(nn.Module):
    r"""
    Norm layer adaptive layer norm zero (adaLN-Zero).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
    """

    def __init__(self, embedding_dim: int, num_embeddings: int | None = None,
                 norm_type="layer_norm", bias=True, promote_fp32: bool = True):
        super().__init__()
        self.emb = None

        self.silu = SiLU()
        self.linear = Linear(embedding_dim, 6 * embedding_dim, bias=bias)
        if norm_type == "layer_norm":
            # promote_fp32=False (bf16 F.layer_norm already accumulates stats in
            # fp32) avoids a full fp32 up/down-cast; callers on bf16 pass False.
            self.norm = LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6,
                                  promote_fp32=promote_fp32)
        else:
            raise ValueError(
                f"Unsupported `norm_type` ({norm_type}) provided. Supported ones are: 'layer_norm'."
            )

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor | None = None,
        class_labels: torch.LongTensor | None = None,
        hidden_dtype: torch.dtype | None = None,
        emb: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.emb is not None:
            emb = self.emb(timestep, class_labels, hidden_dtype=hidden_dtype)
        emb = self.linear(_silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=1)
        x = _modulated_layer_norm(x, emb, self.norm)
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaLayerNormZeroSingle(nn.Module):
    r"""
    Norm layer adaptive layer norm zero (adaLN-Zero) for single-stream blocks.

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
    """

    def __init__(self, embedding_dim: int, norm_type="layer_norm", bias=True,
                 promote_fp32: bool = True):
        super().__init__()

        self.silu = SiLU()
        self.linear = Linear(embedding_dim, 3 * embedding_dim, bias=bias)
        if norm_type == "layer_norm":
            self.norm = LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6,
                                  promote_fp32=promote_fp32)
        else:
            raise ValueError(
                f"Unsupported `norm_type` ({norm_type}) provided. Supported ones are: 'layer_norm'."
            )

    def forward(
        self,
        x: torch.Tensor,
        emb: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.linear(_silu(emb))
        shift_msa, scale_msa, gate_msa = emb.chunk(3, dim=1)
        x = _modulated_layer_norm(x, emb, self.norm)
        return x, gate_msa
