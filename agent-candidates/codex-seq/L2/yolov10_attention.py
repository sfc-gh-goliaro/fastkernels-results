"""YOLOv10 spatial attention block."""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

from ..L1.softmax import Softmax
from .yolov10_conv import YOLOConv


@triton.jit
def _conv_bn_1x1_kernel(
    x,
    conv_weight,
    bn_weight,
    bn_bias,
    running_mean,
    running_var,
    output,
    n_tokens: tl.constexpr,
    in_channels: tl.constexpr,
    out_channels: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    blocks_n = tl.cdiv(n_tokens, BLOCK_N)
    block = tl.program_id(0)
    batch = tl.program_id(1)
    block_m = block // blocks_n
    block_n = block - block_m * blocks_n
    out_ch = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
    tokens = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
    in_ch = tl.arange(0, BLOCK_K)

    weights = tl.load(
        conv_weight + out_ch[:, None] * in_channels + in_ch[None, :],
        mask=(out_ch[:, None] < out_channels) & (in_ch[None, :] < in_channels),
        other=0.0,
    )
    values = tl.load(
        x + batch * in_channels * n_tokens + in_ch[:, None] * n_tokens + tokens[None, :],
        mask=(in_ch[:, None] < in_channels) & (tokens[None, :] < n_tokens),
        other=0.0,
    )
    conv = tl.dot(weights, values).to(tl.float16).to(tl.float32)

    mask_ch = out_ch < out_channels
    gamma = tl.load(bn_weight + out_ch, mask=mask_ch).to(tl.float32)
    beta = tl.load(bn_bias + out_ch, mask=mask_ch).to(tl.float32)
    mean = tl.load(running_mean + out_ch, mask=mask_ch).to(tl.float32)
    var = tl.load(running_var + out_ch, mask=mask_ch).to(tl.float32)
    result = (conv - mean[:, None]) * (
        gamma * tl.rsqrt(var + eps)
    )[:, None] + beta[:, None]
    tl.store(
        output
        + batch * out_channels * n_tokens
        + out_ch[:, None] * n_tokens
        + tokens[None, :],
        result,
        mask=mask_ch[:, None] & (tokens[None, :] < n_tokens),
    )


def _conv_bn_1x1(x: torch.Tensor, layer: YOLOConv) -> torch.Tensor:
    batch, in_channels, height, width = x.shape
    out_channels = layer.conv.weight.shape[0]
    n_tokens = height * width
    output = torch.empty(
        (batch, out_channels, height, width), device=x.device, dtype=x.dtype
    )
    _conv_bn_1x1_kernel[
        (triton.cdiv(out_channels, 64) * triton.cdiv(n_tokens, 64), batch)
    ](
        x,
        layer.conv.weight,
        layer.bn.weight,
        layer.bn.bias,
        layer.bn.running_mean,
        layer.bn.running_var,
        output,
        n_tokens=n_tokens,
        in_channels=in_channels,
        out_channels=out_channels,
        eps=layer.bn.eps,
        BLOCK_M=64,
        BLOCK_N=64,
        BLOCK_K=triton.next_power_of_2(in_channels),
        num_warps=4,
    )
    return output


@triton.jit
def _attention_kernel(
    qkv,
    output,
    pe_weight,
    pe_bn_weight,
    pe_bn_bias,
    pe_mean,
    pe_var,
    n_tokens: tl.constexpr,
    width: tl.constexpr,
    num_heads: tl.constexpr,
    scale: tl.constexpr,
    key_dim: tl.constexpr,
    head_dim: tl.constexpr,
    pe_eps: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    head = tl.program_id(1)
    value_block = tl.program_id(2)
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    qk_dim = tl.arange(0, BLOCK_K)
    value_dim = value_block * BLOCK_V + tl.arange(0, BLOCK_V)
    head_base = head * (2 * key_dim + head_dim) * n_tokens

    q = tl.load(
        qkv + head_base + rows[:, None] + qk_dim[None, :] * n_tokens,
        mask=(rows[:, None] < n_tokens) & (qk_dim[None, :] < key_dim),
        other=0.0,
    )
    row_max = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    row_sum = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_V), tl.float32)

    for start in range(0, n_tokens, BLOCK_N):
        cols = start + tl.arange(0, BLOCK_N)
        k = tl.load(
            qkv
            + head_base
            + key_dim * n_tokens
            + qk_dim[:, None] * n_tokens
            + cols[None, :],
            mask=(qk_dim[:, None] < key_dim) & (cols[None, :] < n_tokens),
            other=0.0,
        )
        scores = tl.dot(q, k)
        scores = (scores.to(tl.float16) * scale).to(tl.float16)
        scores = tl.where(cols[None, :] < n_tokens, scores, -float("inf"))

        tile_max = tl.max(scores, axis=1)
        new_max = tl.maximum(row_max, tile_max)
        alpha = tl.exp(row_max - new_max)
        probs = tl.exp(scores - new_max[:, None])
        row_sum = row_sum * alpha + tl.sum(probs, axis=1)

        v = tl.load(
            qkv
            + head_base
            + 2 * key_dim * n_tokens
            + cols[:, None]
            + value_dim[None, :] * n_tokens,
            mask=(cols[:, None] < n_tokens) & (value_dim[None, :] < head_dim),
            other=0.0,
        )
        acc = acc * alpha[:, None] + tl.dot(probs.to(tl.float16), v)
        row_max = new_max

    result = acc / row_sum[:, None]
    channel = (head % num_heads) * head_dim + value_dim
    row = rows // width
    col = rows - row * width
    positional = tl.zeros((BLOCK_M, BLOCK_V), tl.float32)
    for ky in range(0, 3):
        for kx in range(0, 3):
            in_row = row + ky - 1
            in_col = col + kx - 1
            in_bounds = (
                (rows[:, None] < n_tokens)
                & (value_dim[None, :] < head_dim)
                & (in_row[:, None] >= 0)
                & (in_row[:, None] < n_tokens // width)
                & (in_col[:, None] >= 0)
                & (in_col[:, None] < width)
            )
            value = tl.load(
                qkv
                + head_base
                + 2 * key_dim * n_tokens
                + value_dim[None, :] * n_tokens
                + in_row[:, None] * width
                + in_col[:, None],
                mask=in_bounds,
                other=0.0,
            )
            weight = tl.load(
                pe_weight + channel[None, :] * 9 + ky * 3 + kx,
                mask=value_dim[None, :] < head_dim,
                other=0.0,
            )
            positional += value.to(tl.float32) * weight.to(tl.float32)

    positional = positional.to(tl.float16).to(tl.float32)
    channel_mask = channel < num_heads * head_dim
    gamma = tl.load(pe_bn_weight + channel, mask=channel_mask).to(tl.float32)
    beta = tl.load(pe_bn_bias + channel, mask=channel_mask).to(tl.float32)
    mean = tl.load(pe_mean + channel, mask=channel_mask).to(tl.float32)
    var = tl.load(pe_var + channel, mask=channel_mask).to(tl.float32)
    positional = (
        (positional - mean[None, :])
        * (gamma * tl.rsqrt(var + pe_eps))[None, :]
        + beta[None, :]
    )
    tl.store(
        output + head * head_dim * n_tokens + value_dim[:, None] * n_tokens + rows[None, :],
        (result + positional).trans(1, 0),
        mask=(value_dim[:, None] < head_dim) & (rows[None, :] < n_tokens),
    )


def _attention(
    qkv: torch.Tensor,
    num_heads: int,
    key_dim: int,
    head_dim: int,
    scale: float,
    pe: YOLOConv,
) -> torch.Tensor:
    batch, _, height, width = qkv.shape
    n_tokens = height * width
    output = torch.empty(
        (batch, num_heads * head_dim, height, width),
        device=qkv.device,
        dtype=qkv.dtype,
    )
    block_m = 32 if batch == 1 else 64
    _attention_kernel[
        (triton.cdiv(n_tokens, block_m), batch * num_heads, triton.cdiv(head_dim, 32))
    ](
        qkv,
        output,
        pe.conv.weight,
        pe.bn.weight,
        pe.bn.bias,
        pe.bn.running_mean,
        pe.bn.running_var,
        n_tokens=n_tokens,
        width=width,
        num_heads=num_heads,
        scale=scale,
        key_dim=key_dim,
        head_dim=head_dim,
        pe_eps=pe.bn.eps,
        BLOCK_K=triton.next_power_of_2(key_dim),
        BLOCK_V=32,
        BLOCK_M=block_m,
        BLOCK_N=32,
        num_warps=8,
    )
    return output


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
        self._softmax = Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = _conv_bn_1x1(x, self.qkv)
        attended = _attention(
            qkv,
            self.num_heads,
            self.key_dim,
            self.head_dim,
            self.scale,
            self.pe,
        )
        return _conv_bn_1x1(attended, self.proj)
