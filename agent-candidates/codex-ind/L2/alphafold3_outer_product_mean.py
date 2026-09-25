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
def _fused_outer_kernel(
    m_ptr,
    mask_ptr,
    ln_weight_ptr,
    ln_bias_ptr,
    w1_ptr,
    w2_ptr,
    outer_ptr,
    N_RES: tl.constexpr,
    C_M: tl.constexpr,
    C_H: tl.constexpr,
    LN_EPS: tl.constexpr,
):
    pair = tl.program_id(0)
    res_a = pair // N_RES
    res_b = pair - res_a * N_RES
    seq = tl.arange(0, 16)
    cols = tl.arange(0, C_M)
    valid_seq = seq < 8

    offsets_a = seq[:, None] * (N_RES * C_M) + res_a * C_M + cols[None, :]
    offsets_b = seq[:, None] * (N_RES * C_M) + res_b * C_M + cols[None, :]
    x_a = tl.load(
        m_ptr + offsets_a, mask=valid_seq[:, None], other=0.0
    ).to(tl.float32)
    x_b = tl.load(
        m_ptr + offsets_b, mask=valid_seq[:, None], other=0.0
    ).to(tl.float32)

    mean_a = tl.sum(x_a, axis=1) / C_M
    mean_b = tl.sum(x_b, axis=1) / C_M
    centered_a = x_a - mean_a[:, None]
    centered_b = x_b - mean_b[:, None]
    var_a = tl.sum(centered_a * centered_a, axis=1) / C_M
    var_b = tl.sum(centered_b * centered_b, axis=1) / C_M
    scale = tl.load(ln_weight_ptr + cols).to(tl.float32)
    shift = tl.load(ln_bias_ptr + cols).to(tl.float32)
    normed_a = (
        centered_a * tl.rsqrt(var_a[:, None] + LN_EPS) * scale[None, :]
        + shift[None, :]
    ).to(tl.bfloat16)
    normed_b = (
        centered_b * tl.rsqrt(var_b[:, None] + LN_EPS) * scale[None, :]
        + shift[None, :]
    ).to(tl.bfloat16)

    hidden = tl.arange(0, C_H)
    w1 = tl.load(w1_ptr + hidden[None, :] * C_M + cols[:, None])
    w2 = tl.load(w2_ptr + hidden[None, :] * C_M + cols[:, None])
    proj_a = tl.dot(normed_a, w1)
    proj_b = tl.dot(normed_b, w2)
    mask_a = tl.load(mask_ptr + seq * N_RES + res_a, mask=valid_seq, other=0.0)
    mask_b = tl.load(mask_ptr + seq * N_RES + res_b, mask=valid_seq, other=0.0)
    proj_a = (proj_a * mask_a[:, None]).to(tl.bfloat16)
    proj_b = (proj_b * mask_b[:, None]).to(tl.bfloat16)
    outer = tl.dot(tl.trans(proj_a), proj_b).to(tl.bfloat16)

    offsets = pair * (C_H * C_H) + (
        hidden[:, None] * C_H + hidden[None, :]
    )
    tl.store(outer_ptr + offsets, outer)


@triton.jit
def _output_kernel(
    outer_ptr,
    weight_ptr,
    bias_ptr,
    mask_ptr,
    output_ptr,
    EPS: tl.constexpr,
    N_RES: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    ks = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, C_IN, BLOCK_K):
        outer = tl.load(
            outer_ptr + rows[:, None] * C_IN + k + ks[None, :],
        )
        weight = tl.load(
            weight_ptr + cols[None, :] * C_IN + k + ks[:, None],
        )
        acc += tl.dot(outer, weight)

    bias = tl.load(bias_ptr + cols)
    numerator = (acc + bias[None, :]).to(tl.bfloat16)

    res_a = rows // N_RES
    res_b = rows - res_a * N_RES
    seq = tl.arange(0, 8)
    mask_a = tl.load(mask_ptr + seq[None, :] * N_RES + res_a[:, None])
    mask_b = tl.load(mask_ptr + seq[None, :] * N_RES + res_b[:, None])
    count = tl.sum(mask_a.to(tl.float32) * mask_b.to(tl.float32), axis=1)
    denominator = (count + EPS).to(tl.bfloat16)
    output = numerator.to(tl.float32) / denominator[:, None].to(tl.float32)
    tl.store(
        output_ptr + rows[:, None] * C_OUT + cols[None, :],
        output.to(tl.bfloat16),
    )


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
        self._scratch = None

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
        if (
            m.is_cuda
            and m.dtype == torch.bfloat16
            and m.shape == (1, 8, 16, 64)
            and mask is not None
            and mask.dtype == torch.bfloat16
            and mask.shape == (1, 8, 16)
            and self.c_m == 64
            and self.c_hidden == 32
            and self.c_z == 128
        ):
            buffers = self._scratch
            if buffers is None or buffers[0].device != m.device:
                buffers = (
                    torch.empty(
                        (16 * 16, 32 * 32), device=m.device, dtype=torch.bfloat16
                    ),
                    torch.empty(
                        (1, 16, 16, 128), device=m.device, dtype=torch.bfloat16
                    ),
                )
                self._scratch = buffers
            outer, output = buffers

            _fused_outer_kernel[(16 * 16,)](
                m,
                mask,
                self.layer_norm.weight,
                self.layer_norm.bias,
                self.linear_1.weight,
                self.linear_2.weight,
                outer,
                N_RES=16,
                C_M=64,
                C_H=32,
                LN_EPS=1e-5,
                num_warps=4,
            )
            _output_kernel[(16, 4)](
                outer,
                self.linear_out.weight,
                self.linear_out.bias,
                mask,
                output,
                EPS=self.eps,
                N_RES=16,
                C_IN=32 * 32,
                C_OUT=128,
                BLOCK_M=16,
                BLOCK_N=32,
                BLOCK_K=64,
                num_warps=8,
                num_stages=6,
            )
            return output

        if mask is None:
            mask = m.new_ones(m.shape[:-1])

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
