"""Optimized YOLOv10 spatial attention block."""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

from .yolov10_conv import YOLOConv


@triton.jit
def _qkv_kernel(
    x,
    weight,
    bias,
    output,
    n_ctx: tl.constexpr,
    channels: tl.constexpr,
    out_channels: tl.constexpr,
    BLOCK_O: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    offs_o = tl.program_id(0) * BLOCK_O + tl.arange(0, BLOCK_O)
    offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    batch_idx = tl.program_id(2)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_O, BLOCK_N), tl.float32)

    for start_k in range(0, channels, BLOCK_K):
        cols = start_k + offs_k
        w_ptrs = weight + offs_o[:, None] * channels + cols[None, :]
        x_ptrs = (
            x
            + batch_idx * channels * n_ctx
            + cols[:, None] * n_ctx
            + offs_n[None, :]
        )
        w = tl.load(
            w_ptrs,
            mask=(offs_o[:, None] < out_channels) & (cols[None, :] < channels),
            other=0.0,
        )
        values = tl.load(
            x_ptrs,
            mask=(cols[:, None] < channels) & (offs_n[None, :] < n_ctx),
            other=0.0,
        )
        acc += tl.dot(w, values)

    acc += tl.load(bias + offs_o, mask=offs_o < out_channels, other=0.0)[:, None]
    out_ptrs = (
        output
        + batch_idx * out_channels * n_ctx
        + offs_o[:, None] * n_ctx
        + offs_n[None, :]
    )
    tl.store(
        out_ptrs,
        acc,
        mask=(offs_o[:, None] < out_channels) & (offs_n[None, :] < n_ctx),
    )


@triton.jit
def _attention_pe_proj_kernel(
    qkv,
    pe_weight,
    pe_bias,
    proj_weight,
    proj_bias,
    output,
    nheads: tl.constexpr,
    n_ctx: tl.constexpr,
    key_dim: tl.constexpr,
    head_dim: tl.constexpr,
    channels: tl.constexpr,
    scale: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0) * BLOCK_M
    batch_idx = tl.program_id(1)
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, key_dim)
    offs_v = tl.arange(0, head_dim)
    offs_o = tl.arange(0, channels)

    head_stride = (2 * key_dim + head_dim) * n_ctx
    batch_base = batch_idx * nheads * head_stride
    projected = tl.zeros((BLOCK_M, channels), tl.float32)

    for head in range(0, nheads):
        base = batch_base + head * head_stride
        q_ptrs = qkv + base + offs_m[:, None] + offs_k[None, :] * n_ctx
        q = tl.load(q_ptrs, mask=offs_m[:, None] < n_ctx, other=0.0)

        cols = offs_n
        k_ptrs = (
            qkv
            + base
            + key_dim * n_ctx
            + offs_k[:, None] * n_ctx
            + cols[None, :]
        )
        k = tl.load(k_ptrs, mask=cols[None, :] < n_ctx, other=0.0)
        scores = tl.dot(q, k) * scale
        scores = tl.where(cols[None, :] < n_ctx, scores, float("-inf"))
        scores -= tl.max(scores, axis=1)[:, None]
        p = tl.exp2(scores * 1.4426950408889634)
        p /= tl.sum(p, axis=1)[:, None]

        v_ptrs = (
            qkv
            + base
            + 2 * key_dim * n_ctx
            + offs_v[:, None] * n_ctx
            + cols[None, :]
        )
        v = tl.load(v_ptrs, mask=cols[None, :] < n_ctx, other=0.0)
        acc = tl.dot(p.to(v.dtype), tl.trans(v))

        channel = head * head_dim + offs_v
        row = offs_m // 20
        col = offs_m - row * 20
        pe = tl.zeros((BLOCK_M, head_dim), tl.float32)
        pe += tl.load(pe_bias + channel)[None, :]
        for ky in range(3):
            for kx in range(3):
                rr = row + ky - 1
                cc = col + kx - 1
                pos = rr * 20 + cc
                valid = (
                    (offs_m < n_ctx)
                    & (rr >= 0)
                    & (rr < 20)
                    & (cc >= 0)
                    & (cc < 20)
                )
                v_pe_ptrs = (
                    qkv
                    + base
                    + 2 * key_dim * n_ctx
                    + offs_v[None, :] * n_ctx
                    + pos[:, None]
                )
                v_pe = tl.load(v_pe_ptrs, mask=valid[:, None], other=0.0)
                w_pe = tl.load(pe_weight + channel * 9 + ky * 3 + kx)
                pe += v_pe.to(tl.float32) * w_pe[None, :]

        w_ptrs = (
            proj_weight
            + offs_v[:, None]
            + head * head_dim
            + offs_o[None, :] * channels
        )
        w_proj = tl.load(w_ptrs)
        projected += tl.dot((acc + pe).to(w_proj.dtype), w_proj)

    projected += tl.load(proj_bias + offs_o)[None, :]
    out_ptrs = (
        output
        + batch_idx * channels * n_ctx
        + offs_o[:, None] * n_ctx
        + offs_m[None, :]
    )
    tl.store(out_ptrs, tl.trans(projected), mask=offs_m[None, :] < n_ctx)


class YOLOAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, attn_ratio: float = 0.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim ** -0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = YOLOConv(dim, h, 1, act=False)
        self.proj = YOLOConv(dim, dim, 1, act=False)
        self.pe = YOLOConv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        n = h * w

        if not self.qkv._is_fused:
            self.qkv.fuse()
            self.proj.fuse()
            self.pe.fuse()

        out_channels = self.num_heads * (2 * self.key_dim + self.head_dim)
        qkv = torch.empty(
            (b, out_channels, h, w), device=x.device, dtype=x.dtype
        )
        qkv_grid = (triton.cdiv(out_channels, 128), triton.cdiv(n, 128), b)
        _qkv_kernel[qkv_grid](
            x,
            self.qkv.conv.weight,
            self.qkv.conv.bias,
            qkv,
            n_ctx=n,
            channels=c,
            out_channels=out_channels,
            BLOCK_O=128,
            BLOCK_N=128,
            BLOCK_K=64,
            num_warps=4,
            num_stages=3,
        )
        output = torch.empty_like(x)
        grid = (triton.cdiv(n, 16), b)
        _attention_pe_proj_kernel[grid](
            qkv,
            self.pe.conv.weight,
            self.pe.conv.bias,
            self.proj.conv.weight,
            self.proj.conv.bias,
            output,
            nheads=self.num_heads,
            n_ctx=n,
            key_dim=self.key_dim,
            head_dim=self.head_dim,
            channels=c,
            scale=self.scale,
            BLOCK_M=16,
            BLOCK_N=512,
            num_warps=4,
            num_stages=3,
        )
        return output
