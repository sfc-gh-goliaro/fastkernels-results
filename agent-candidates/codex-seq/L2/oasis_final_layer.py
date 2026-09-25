"""Oasis final DiT projection layer."""

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
def _modulation_matvec_kernel(
    c_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    rows,
    hidden: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(1)
    offs_n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_N,), tl.float32)

    for k in range(0, hidden, BLOCK_K):
        c = tl.load(c_ptr + pid_m * hidden + k + offs_k).to(tl.float32)
        c = (c * tl.sigmoid(c)).to(tl.float16)
        weight = tl.load(
            weight_ptr + offs_n[:, None] * hidden + k + offs_k[None, :]
        ).to(tl.float32)
        acc += tl.sum(weight * c[None, :], axis=1)

    bias = tl.load(bias_ptr + offs_n)
    tl.store(out_ptr + pid_m * (2 * hidden) + offs_n, acc + bias)


@triton.jit
def _project_kernel(
    x_ptr,
    modulation_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    rows,
    stride_condition,
    stride_hidden,
    hidden: tl.constexpr,
    out_features: tl.constexpr,
    rows_per_condition: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    row_mask = offs_m < rows
    condition = offs_m // rows_per_condition
    spatial = offs_m % rows_per_condition
    total = tl.zeros((BLOCK_M,), tl.float32)
    square_total = tl.zeros((BLOCK_M,), tl.float32)

    for k in range(0, hidden, BLOCK_K):
        x = tl.load(
            x_ptr
            + condition[None, :] * stride_condition
            + spatial[None, :]
            + (k + offs_k[:, None]) * stride_hidden,
            mask=row_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        total += tl.sum(x, axis=0)
        square_total += tl.sum(x * x, axis=0)
    mean = total / hidden
    variance = square_total / hidden - mean * mean
    rstd = tl.rsqrt(variance + 1e-6)

    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    for k in range(0, hidden, BLOCK_K):
        x = tl.load(
            x_ptr
            + condition[None, :] * stride_condition
            + spatial[None, :]
            + (k + offs_k[:, None]) * stride_hidden,
            mask=row_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        x = tl.trans(x)
        shift = tl.load(
            modulation_ptr
            + condition[:, None] * (2 * hidden)
            + k
            + offs_k[None, :],
            mask=row_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        scale = tl.load(
            modulation_ptr
            + condition[:, None] * (2 * hidden)
            + hidden
            + k
            + offs_k[None, :],
            mask=row_mask[:, None],
            other=0.0,
        ).to(tl.float32)

        normalized = ((x - mean[:, None]) * rstd[:, None]).to(tl.float16)
        factor = (1.0 + scale).to(tl.float16)
        modulated = (normalized * factor).to(tl.float16)
        modulated = (modulated + shift).to(tl.float16)
        weight = tl.load(
            weight_ptr + offs_n[None, :] * hidden + k + offs_k[:, None],
            mask=offs_n[None, :] < out_features,
            other=0.0,
        )
        acc = tl.dot(modulated, weight, acc)

    bias = tl.load(bias_ptr + offs_n, mask=offs_n < out_features, other=0.0)
    tl.store(
        out_ptr + offs_m[:, None] * out_features + offs_n[None, :],
        acc + bias[None, :],
        mask=row_mask[:, None] & (offs_n[None, :] < out_features),
    )


class OasisFinalLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.linear = Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.ModuleList(
            [
                SiLU(),
                Linear(hidden_size, 2 * hidden_size, bias=True),
            ]
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        if (
            x.is_cuda
            and x.dtype == torch.float16
            and x.shape[-1] == 1024
            and x.stride(-2) == 1
        ):
            hidden = 1024
            condition_rows = c.numel() // hidden
            rows = x.numel() // hidden
            out_features = self.linear.weight.shape[0]
            modulation = torch.empty(
                (condition_rows, 2 * hidden), device=x.device, dtype=x.dtype
            )
            _modulation_matvec_kernel[(triton.cdiv(2 * hidden, 8), condition_rows)](
                c,
                self.adaLN_modulation[1].weight,
                self.adaLN_modulation[1].bias,
                modulation,
                condition_rows,
                hidden=hidden,
                BLOCK_N=8,
                BLOCK_K=1024,
                num_warps=4,
                num_stages=1,
            )

            out = torch.empty(
                (*x.shape[:-1], out_features), device=x.device, dtype=x.dtype
            )
            _project_kernel[(triton.cdiv(rows, 16),)](
                x,
                modulation,
                self.linear.weight,
                self.linear.bias,
                out,
                rows,
                x.stride(1),
                x.stride(-1),
                hidden=hidden,
                out_features=out_features,
                rows_per_condition=rows // condition_rows,
                BLOCK_M=16,
                BLOCK_N=64,
                BLOCK_K=256,
                num_warps=4,
                num_stages=4,
            )
            return out

        modulation = c
        for layer in self.adaLN_modulation:
            modulation = layer(modulation)
        shift, scale = modulation.chunk(2, dim=-1)
        while shift.dim() < x.dim():
            shift = shift.unsqueeze(-2)
            scale = scale.unsqueeze(-2)
        x = self.norm_final(x) * (1 + scale) + shift
        return F.linear(x, self.linear.weight, self.linear.bias)
