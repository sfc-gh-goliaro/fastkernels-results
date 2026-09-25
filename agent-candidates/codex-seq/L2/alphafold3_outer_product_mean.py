"""Outer product mean for AlphaFold3 (L2).

Implements AF3 Algorithm 9. Computes an outer product of MSA
representations and averages over the MSA dimension to produce
a pair representation update.

Reference: openfold3/core/model/layers/outer_product_mean.py OuterProductMean
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear


@triton.jit
def _project_kernel(
    m_ptr,
    mask_ptr,
    ln_weight_ptr,
    ln_bias_ptr,
    w1_ptr,
    w2_ptr,
    projected_ptr,
    N_RES: tl.constexpr,
    C_M: tl.constexpr,
    C_HIDDEN: tl.constexpr,
    LN_EPS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    cols = tl.arange(0, C_M)
    x = tl.load(m_ptr + rows[:, None] * C_M + cols[None, :]).to(tl.float32)

    mean = tl.sum(x, axis=1) / C_M
    centered = x - mean[:, None]
    variance = tl.sum(centered * centered, axis=1) / C_M
    x = centered * tl.rsqrt(variance[:, None] + LN_EPS)
    scale = tl.load(ln_weight_ptr + cols).to(tl.float32)
    bias = tl.load(ln_bias_ptr + cols).to(tl.float32)
    x = (x * scale[None, :] + bias[None, :]).to(tl.bfloat16)

    out_cols = tl.arange(0, 2 * C_HIDDEN)
    weight_offsets = cols[:, None] + (
        out_cols[None, :] % C_HIDDEN
    ) * C_M
    weights = tl.where(
        out_cols[None, :] < C_HIDDEN,
        tl.load(w1_ptr + weight_offsets),
        tl.load(w2_ptr + weight_offsets),
    )
    projected = tl.dot(x, weights)
    row_mask = tl.load(mask_ptr + rows).to(tl.float32)
    projected *= row_mask[:, None]

    seq = rows // N_RES
    res = rows % N_RES
    dst = (
        (res[:, None] * 8 + seq[:, None]) * (2 * C_HIDDEN)
        + out_cols[None, :]
    )
    tl.store(projected_ptr + dst, projected)


@triton.jit
def _outer_out_kernel(
    projected_ptr,
    mask_ptr,
    out_weight_ptr,
    out_bias_ptr,
    out_ptr,
    EPS: tl.constexpr,
    C_HIDDEN: tl.constexpr,
    C_Z: tl.constexpr,
    BLOCK_Z: tl.constexpr,
):
    res_i = tl.program_id(0)
    z_block = tl.program_id(1)
    res_j = tl.arange(0, 16)
    seq = tl.arange(0, 8)
    z = z_block * BLOCK_Z + tl.arange(0, BLOCK_Z)
    hidden_e = tl.arange(0, C_HIDDEN)

    acc = tl.zeros((16, BLOCK_Z), tl.float32)
    for hidden_c in range(C_HIDDEN):
        a_offsets = (
            (res_i * 8 + seq) * (2 * C_HIDDEN) + hidden_c
        )
        av = tl.load(projected_ptr + a_offsets)
        b_offsets = (
            (res_j[:, None, None] * 8 + seq[None, :, None])
            * (2 * C_HIDDEN)
            + C_HIDDEN + hidden_e[None, None, :]
        )
        bv = tl.load(projected_ptr + b_offsets)
        outer = tl.sum(av[None, :, None] * bv, axis=1).to(tl.bfloat16)

        k = hidden_c * C_HIDDEN + hidden_e
        weights = tl.load(
            out_weight_ptr + z[None, :] * (C_HIDDEN * C_HIDDEN)
            + k[:, None]
        )
        acc += tl.dot(outer, weights)

    mask_i = tl.load(mask_ptr + seq * 16 + res_i)
    mask_j = tl.load(mask_ptr + seq[None, :] * 16 + res_j[:, None])
    norm = tl.sum(mask_i[None, :] * mask_j, axis=1) + EPS
    norm = norm.to(tl.bfloat16)
    bias = tl.load(out_bias_ptr + z)
    result = (acc + bias[None, :]) / norm[:, None]
    out_offsets = (res_i * 16 + res_j[:, None]) * C_Z + z[None, :]
    tl.store(out_ptr + out_offsets, result)


class OuterProductMean(nn.Module):
    """AF3 Algorithm 9: Outer product mean.

    Args:
        c_m: MSA embedding channel dimension
        c_z: Pair embedding channel dimension
        c_hidden: Hidden channel dimension
        eps: Epsilon for numerical stability
    """

    def __init__(self, c_m: int, c_z: int, c_hidden: int, eps: float = 1e-3):
        super().__init__()
        self.c_m = c_m
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.eps = eps

        self.layer_norm = LayerNorm(c_m)
        self.linear_1 = Linear(c_m, c_hidden, bias=False)
        self.linear_2 = Linear(c_m, c_hidden, bias=False)
        self.linear_out = Linear(c_hidden ** 2, c_z, bias=True)

    def forward(
        self,
        m: torch.Tensor,
        mask: torch.Tensor | None = None,
        chunk_size: int | None = None,
        inplace_safe: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            m:    [*, N_seq, N_res, C_m] MSA embedding
            mask: [*, N_seq, N_res] MSA mask

        Returns:
            [*, N_res, N_res, C_z] pair embedding update
        """
        if mask is None:
            mask = m.new_ones(m.shape[:-1])

        if (
            m.is_cuda
            and m.dtype == torch.bfloat16
            and m.shape == (1, 8, 16, 64)
            and mask.shape == (1, 8, 16)
            and self.c_m == 64
            and self.c_hidden == 32
            and self.c_z == 128
        ):
            projected = torch.empty((16, 8, 64), device=m.device, dtype=m.dtype)
            _project_kernel[(8,)](
                m,
                mask,
                self.layer_norm.weight,
                self.layer_norm.bias,
                self.linear_1.weight,
                self.linear_2.weight,
                projected,
                N_RES=16,
                C_M=64,
                C_HIDDEN=32,
                LN_EPS=self.layer_norm.eps,
                BLOCK_ROWS=16,
                num_warps=8,
            )
            out = torch.empty((1, 16, 16, 128), device=m.device, dtype=m.dtype)
            _outer_out_kernel[(16, 4)](
                projected,
                mask,
                self.linear_out.weight,
                self.linear_out.bias,
                out,
                EPS=self.eps,
                C_HIDDEN=32,
                C_Z=128,
                BLOCK_Z=32,
                num_warps=2,
                num_stages=5,
            )
            return out

        ln = self.layer_norm(m)

        mask = mask.unsqueeze(-1)
        a = self.linear_1(ln) * mask
        b = self.linear_2(ln) * mask

        del ln

        # [*, N_res, N_seq, C]
        a = a.transpose(-2, -3)
        b = b.transpose(-2, -3)

        # [*, N_res, N_res, C, C]
        outer = torch.einsum("...bac,...dae->...bdce", a, b)

        # [*, N_res, N_res, C * C]
        outer = outer.reshape(outer.shape[:-2] + (-1,))

        # [*, N_res, N_res, C_z]
        outer = self.linear_out(outer)

        # Normalization: count valid sequence pairs per residue pair
        norm = torch.einsum("...abc,...adc->...bdc", mask, mask)
        norm = norm + self.eps

        outer = outer / norm

        return outer
