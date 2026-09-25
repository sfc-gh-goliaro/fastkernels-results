"""Oasis spatial axial attention."""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

from ..L1.dense_attention import DenseAttention
from ..L1.linear import Linear
from ..L1.oasis_rotary import OasisRotaryEmbedding, oasis_apply_rotary_emb


@triton.jit
def _qkv_rotary_kernel(
    x,
    weight,
    cos,
    sin,
    qkv,
    M: tl.constexpr,
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
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(
            x + offs_m[:, None] * K + k + offs_k[None, :],
            mask=offs_m[:, None] < M,
            other=0.0,
        )
        b = tl.load(weight + offs_n[None, :] * K + k + offs_k[:, None])
        acc = tl.dot(a, b, acc)

    # F.linear rounds to fp16 before the eager rotary arithmetic.
    values = acc.to(tl.float16)
    if pid_n * BLOCK_N < 2048:
        pair_values = tl.reshape(values, (BLOCK_M, BLOCK_N // 2, 2))
        even, odd = tl.split(pair_values)
        pair_n = pid_n * (BLOCK_N // 2) + tl.arange(0, BLOCK_N // 2)
        pair_d = pair_n % 32
        spatial = offs_m % 144
        c = tl.load(
            cos + spatial[:, None] * 32 + pair_d[None, :],
            mask=offs_m[:, None] < M,
            other=0.0,
        )
        s = tl.load(
            sin + spatial[:, None] * 32 + pair_d[None, :],
            mask=offs_m[:, None] < M,
            other=0.0,
        )
        rotated = tl.join(even * c - odd * s, odd * c + even * s)
        values = tl.reshape(rotated, (BLOCK_M, BLOCK_N))

    tl.store(
        qkv + offs_m[:, None] * N + offs_n[None, :],
        values,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.jit
def _attention_kernel(
    qkv,
    out,
    T: tl.constexpr,
    sm_scale: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D: tl.constexpr,
):
    query_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch = batch_head // 16
    head = batch_head % 16

    offs_m = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    base = batch * 144 * 3072 + head * D

    q = tl.load(
        qkv + base + offs_m[:, None] * 3072 + offs_d[None, :],
        mask=offs_m[:, None] < 144,
        other=0.0,
    )
    q = (q * (sm_scale * 1.4426950408889634)).to(tl.float16)

    row_max = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    row_sum = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, D), tl.float32)

    for start_n in range(0, 144, BLOCK_N):
        valid_n = start_n + offs_n < 144
        k = tl.load(
            qkv
            + base
            + 1024
            + (start_n + offs_n)[:, None] * 3072
            + offs_d[None, :],
            mask=valid_n[:, None],
            other=0.0,
        )
        scores = tl.dot(q, tl.trans(k))
        scores = tl.where(valid_n[None, :], scores, -float("inf"))
        next_max = tl.maximum(row_max, tl.max(scores, axis=1))
        alpha = tl.exp2(row_max - next_max)
        probs = tl.exp2(scores - next_max[:, None])
        next_sum = row_sum * alpha + tl.sum(probs, axis=1)
        acc *= alpha[:, None]

        v = tl.load(
            qkv
            + base
            + 2048
            + (start_n + offs_n)[:, None] * 3072
            + offs_d[None, :],
            mask=valid_n[:, None],
            other=0.0,
        )
        acc = tl.dot(probs.to(tl.float16), v, acc)
        row_max = next_max
        row_sum = next_sum

    acc /= row_sum[:, None]
    tl.store(
        out
        + (batch * 144 + offs_m[:, None]) * 1024
        + head * D
        + offs_d[None, :],
        acc,
        mask=offs_m[:, None] < 144,
    )


@triton.jit
def _output_kernel(
    x,
    weight,
    bias,
    output,
    M: tl.constexpr,
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
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(
            x + offs_m[:, None] * K + k + offs_k[None, :],
            mask=offs_m[:, None] < M,
            other=0.0,
        )
        b = tl.load(weight + offs_n[None, :] * K + k + offs_k[:, None])
        acc = tl.dot(a, b, acc)

    acc += tl.load(bias + offs_n)[None, :]
    tl.store(
        output + offs_m[:, None] * N + offs_n[None, :],
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


class OasisSpatialAxialAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        rotary_emb: OasisRotaryEmbedding,
    ):
        super().__init__()
        self.heads = heads
        self.to_qkv = Linear(dim, dim_head * heads * 3, bias=False)
        self.to_out = Linear(dim_head * heads, dim, bias=True)
        self.rotary_emb = rotary_emb
        self.attn = DenseAttention(backend="sdpa")
        self._rotary_cache = None

    def _rotary_tables(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cache = self._rotary_cache
        if cache is None or cache[0].device != x.device or cache[0].dtype != x.dtype:
            freqs = self.rotary_emb.get_axial_freqs(9, 16)
            # One value per adjacent rotary pair; frequencies are duplicated.
            cache = (freqs.cos()[..., ::2].contiguous(), freqs.sin()[..., ::2].contiguous())
            self._rotary_cache = cache
        return cache

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, time, height, width, dim = x.shape
        if (
            bsz != 1
            or height != 9
            or width != 16
            or dim != 1024
            or self.heads != 16
            or x.dtype != torch.float16
            or not x.is_cuda
        ):
            q, k, v = self.to_qkv(x).chunk(3, dim=-1)
            q = q.reshape(bsz * time, height, width, self.heads, -1).permute(0, 3, 1, 2, 4)
            k = k.reshape(bsz * time, height, width, self.heads, -1).permute(0, 3, 1, 2, 4)
            v = v.reshape(bsz * time, height, width, self.heads, -1).permute(0, 3, 1, 2, 4)
            freqs = self.rotary_emb.get_axial_freqs(height, width)
            q = oasis_apply_rotary_emb(freqs, q)
            k = oasis_apply_rotary_emb(freqs, k)
            q = q.reshape(bsz * time, self.heads, height * width, -1).transpose(1, 2)
            k = k.reshape(bsz * time, self.heads, height * width, -1).transpose(1, 2)
            v = v.reshape(bsz * time, self.heads, height * width, -1).transpose(1, 2)
            out = self.attn(q, k, v, causal=False)
            out = out.reshape(bsz, time, height, width, -1)
            return self.to_out(out.to(q.dtype))

        m = time * 144
        cos, sin = self._rotary_tables(x)
        qkv = torch.empty((m, 3072), device=x.device, dtype=x.dtype)
        if time == 2:
            block_m, block_n, block_k, warps, stages = 64, 128, 128, 8, 3
        elif time <= 4:
            block_m, block_n, block_k, warps, stages = 64, 64, 64, 4, 4
        elif time == 5:
            block_m, block_n, block_k, warps, stages = 64, 128, 64, 8, 4
        else:
            block_m, block_n, block_k, warps, stages = 128, 256, 64, 8, 4
        _qkv_rotary_kernel[
            (triton.cdiv(m, block_m), triton.cdiv(3072, block_n))
        ](
            x,
            self.to_qkv.weight,
            cos,
            sin,
            qkv,
            M=m,
            K=1024,
            N=3072,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=warps,
            num_stages=stages,
        )

        attended = torch.empty((m, 1024), device=x.device, dtype=x.dtype)
        _attention_kernel[(3, time * 16)](
            qkv,
            attended,
            T=time,
            sm_scale=0.125,
            BLOCK_M=64,
            BLOCK_N=64,
            D=64,
            num_warps=4,
            num_stages=3,
        )

        output = torch.empty_like(attended)
        _output_kernel[(triton.cdiv(m, 64), 8)](
            attended,
            self.to_out.weight,
            self.to_out.bias,
            output,
            M=m,
            K=1024,
            N=1024,
            BLOCK_M=64,
            BLOCK_N=128,
            BLOCK_K=128,
            num_warps=8,
            num_stages=3,
        )
        return output.reshape(bsz, time, height, width, dim)
