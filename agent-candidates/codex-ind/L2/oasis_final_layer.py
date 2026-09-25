"""Oasis final DiT projection layer."""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear
from ..L1.silu import SiLU


@triton.jit
def _modulation_gemv_kernel(
    c_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    K: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, K)
    c = tl.load(c_ptr + row * K + offs_k).to(tl.float32)
    c = (c * tl.sigmoid(c)).to(tl.float16)
    weight = tl.load(
        weight_ptr + offs_n[:, None] * K + offs_k[None, :],
        mask=offs_n[:, None] < N,
        other=0.0,
    )
    acc = tl.sum((c[None, :] * weight).to(tl.float32), axis=1)
    acc += tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    tl.store(out_ptr + row * N + offs_n, acc, mask=offs_n < N)


@triton.jit
def _final_gemv_kernel(
    x_ptr,
    modulation_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    K: tl.constexpr,
    N: tl.constexpr,
    SPATIAL: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, K)
    frame = row // SPATIAL
    spatial = row - frame * SPATIAL
    x = tl.load(
        x_ptr + frame * K * SPATIAL + offs_k * SPATIAL + spatial
    ).to(tl.float32)
    mean = tl.sum(x, axis=0) * (1.0 / K)
    centered = x - mean
    variance = tl.sum(centered * centered, axis=0) * (1.0 / K)
    normalized = (centered * tl.rsqrt(variance + EPS)).to(tl.float16)
    shift = tl.load(modulation_ptr + frame * (2 * K) + offs_k)
    scale = tl.load(modulation_ptr + frame * (2 * K) + K + offs_k)
    projected_input = (
        normalized * (tl.full((K,), 1.0, tl.float16) + scale) + shift
    ).to(tl.float16)
    weight = tl.load(
        weight_ptr + offs_n[:, None] * K + offs_k[None, :],
        mask=offs_n[:, None] < N,
        other=0.0,
    )
    acc = tl.sum((projected_input[None, :] * weight).to(tl.float32), axis=1)
    acc += tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    tl.store(out_ptr + row * N + offs_n, acc, mask=offs_n < N)


@triton.jit
def _normalize_kernel(
    x_ptr,
    modulation_ptr,
    out_ptr,
    rows: tl.constexpr,
    K: tl.constexpr,
    SPATIAL: tl.constexpr,
    EPS: tl.constexpr,
):
    offs_m = tl.program_id(0)
    offs_k = tl.arange(0, K)
    frame = offs_m // SPATIAL
    spatial = offs_m - frame * SPATIAL

    # x is a view of contiguous [frame, channel, spatial] storage.
    x = tl.load(
        x_ptr + frame * K * SPATIAL + offs_k * SPATIAL + spatial
    ).to(tl.float32)
    mean = tl.sum(x, axis=0) * (1.0 / K)
    centered = x - mean
    variance = tl.sum(centered * centered, axis=0) * (1.0 / K)
    normalized = (centered * tl.rsqrt(variance + EPS)).to(tl.float16)

    shift = tl.load(
        modulation_ptr + frame * (2 * K) + offs_k
    )
    scale = tl.load(
        modulation_ptr + frame * (2 * K) + K + offs_k
    )
    one = tl.full((K,), 1.0, tl.float16)
    projected_input = (normalized * (one + scale) + shift).to(tl.float16)
    tl.store(out_ptr + offs_m * K + offs_k, projected_input)


@triton.jit
def _matmul_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    rows: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        input = tl.load(
            input_ptr + offs_m[:, None] * K + offs_k[None, :],
            mask=offs_m[:, None] < rows,
            other=0.0,
        )
        weight = tl.load(
            weight_ptr + offs_k[:, None] + offs_n[None, :] * K,
            mask=offs_n[None, :] < N,
            other=0.0,
        )
        acc = tl.dot(input, weight, acc)

    acc += tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)[None, :]
    tl.store(
        out_ptr + offs_m[:, None] * N + offs_n[None, :],
        acc,
        mask=(offs_m[:, None] < rows) & (offs_n[None, :] < N),
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
        frames = c.numel() // 1024
        modulation = torch.empty(
            (*c.shape[:-1], 2048), device=c.device, dtype=c.dtype
        )
        modulation_block_n = 32
        _modulation_gemv_kernel[(frames, triton.cdiv(2048, modulation_block_n))](
            c,
            self.adaLN_modulation[1].weight,
            self.adaLN_modulation[1].bias,
            modulation,
            1024,
            2048,
            BLOCK_N=modulation_block_n,
            num_warps=4,
        )

        rows = x.numel() // 1024
        output = torch.empty(
            (*x.shape[:-1], self.linear.weight.shape[0]),
            device=x.device,
            dtype=x.dtype,
        )
        if frames <= 4:
            final_block_n = 64 if frames == 2 else 32
            _final_gemv_kernel[
                (rows, triton.cdiv(64, final_block_n))
            ](
                x,
                modulation,
                self.linear.weight,
                self.linear.bias,
                output,
                1024,
                64,
                144,
                1e-6,
                BLOCK_N=final_block_n,
                num_warps=4,
            )
            return output

        normalized = torch.empty((rows, 1024), device=x.device, dtype=x.dtype)
        norm_warps = 16 if frames == 5 else 8
        _normalize_kernel[(rows,)](
            x,
            modulation,
            normalized,
            rows,
            1024,
            144,
            1e-6,
            num_warps=norm_warps,
        )

        block_m, block_n, block_k, stages = 64, 32, 32, 3
        _matmul_kernel[
            (triton.cdiv(rows, block_m), triton.cdiv(64, block_n))
        ](
            normalized,
            self.linear.weight,
            self.linear.bias,
            output,
            rows,
            1024,
            64,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=4,
            num_stages=stages,
        )
        return output
