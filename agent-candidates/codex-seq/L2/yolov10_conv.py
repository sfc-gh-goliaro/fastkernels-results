"""YOLOv10 Conv-BN-Act building block."""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

from ..L1.batch_norm2d import BatchNorm2d
from ..L1.conv2d import Conv2d
from ..L1.silu import SiLU


@triton.jit
def _conv_bn_silu(
    x,
    weight,
    bn_weight,
    bn_bias,
    running_mean,
    running_var,
    out,
    M: tl.constexpr,
    N: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    SH: tl.constexpr,
    SW: tl.constexpr,
    PH: tl.constexpr,
    PW: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    image = offs_m // (OH * OW)
    pos = offs_m % (OH * OW)
    oh = pos // OW
    ow = pos % OW
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    K: tl.constexpr = C * KH * KW
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        channel = offs_k // (KH * KW)
        kernel_pos = offs_k % (KH * KW)
        kh = kernel_pos // KW
        kw = kernel_pos % KW
        ih = oh[:, None] * SH - PH + kh[None, :]
        iw = ow[:, None] * SW - PW + kw[None, :]
        x_offsets = (
            image[:, None] * (C * H * W)
            + channel[None, :] * (H * W)
            + ih * W
            + iw
        )
        x_mask = (
            (offs_m[:, None] < M)
            & (offs_k[None, :] < K)
            & (ih >= 0)
            & (ih < H)
            & (iw >= 0)
            & (iw < W)
        )
        a = tl.load(x + x_offsets, mask=x_mask, other=0.0)
        b = tl.load(
            weight + offs_n[None, :] * K + offs_k[:, None],
            mask=(offs_n[None, :] < N) & (offs_k[:, None] < K),
            other=0.0,
        )
        acc = tl.dot(a, b, acc, input_precision="ieee")

    channel_mask = offs_n < N
    scale = tl.load(bn_weight + offs_n, mask=channel_mask, other=0.0).to(tl.float32)
    mean = tl.load(running_mean + offs_n, mask=channel_mask, other=0.0).to(tl.float32)
    variance = tl.load(running_var + offs_n, mask=channel_mask, other=1.0).to(tl.float32)
    bias = tl.load(bn_bias + offs_n, mask=channel_mask, other=0.0).to(tl.float32)
    value = (acc - mean[None, :]) * tl.rsqrt(variance[None, :] + EPS)
    value = value * scale[None, :] + bias[None, :]
    squared = value * value
    even = squared * (0.2395166094 + squared * (-0.0138038741 + squared * 0.0004331403))
    value = tl.maximum(-0.28, tl.minimum(0.5 * value + even, tl.maximum(value, 0.0)))
    out_offsets = (
        image[:, None] * (N * OH * OW)
        + offs_n[None, :] * (OH * OW)
        + pos[:, None]
    )
    tl.store(
        out + out_offsets,
        value,
        mask=(offs_m[:, None] < M) & channel_mask[None, :],
    )


@triton.jit
def _pointwise_bn_silu(
    x,
    weight,
    bn_weight,
    bn_bias,
    running_mean,
    running_var,
    out,
    M: tl.constexpr,
    N: tl.constexpr,
    C: tl.constexpr,
    S: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    image = offs_m // S
    pos = offs_m % S
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, C, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        a = tl.load(
            x + image[:, None] * (C * S) + offs_k[None, :] * S + pos[:, None],
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < C),
            other=0.0,
        )
        b = tl.load(
            weight + offs_n[None, :] * C + offs_k[:, None],
            mask=(offs_n[None, :] < N) & (offs_k[:, None] < C),
            other=0.0,
        )
        acc = tl.dot(a, b, acc)

    channel_mask = offs_n < N
    scale = tl.load(bn_weight + offs_n, mask=channel_mask, other=0.0).to(tl.float32)
    mean = tl.load(running_mean + offs_n, mask=channel_mask, other=0.0).to(tl.float32)
    variance = tl.load(running_var + offs_n, mask=channel_mask, other=1.0).to(tl.float32)
    bias = tl.load(bn_bias + offs_n, mask=channel_mask, other=0.0).to(tl.float32)
    value = (acc - mean[None, :]) * tl.rsqrt(variance[None, :] + EPS)
    value = value * scale[None, :] + bias[None, :]
    squared = value * value
    even = squared * (0.2395166094 + squared * (-0.0138038741 + squared * 0.0004331403))
    value = tl.maximum(-0.28, tl.minimum(0.5 * value + even, tl.maximum(value, 0.0)))
    tl.store(
        out + image[:, None] * (N * S) + offs_n[None, :] * S + pos[:, None],
        value,
        mask=(offs_m[:, None] < M) & channel_mask[None, :],
    )


def autopad(k: int | tuple[int, int], p=None, d: int = 1):
    if isinstance(k, tuple):
        if d > 1:
            k = tuple(d * (x - 1) + 1 for x in k)
        if p is None:
            return tuple(x // 2 for x in k)
        return p
    if d > 1:
        k = d * (k - 1) + 1
    return k // 2 if p is None else p


def _fuse_conv_bn(conv: Conv2d, bn: BatchNorm2d) -> tuple[torch.Tensor, torch.Tensor]:
    w_conv = conv.weight.clone().view(conv.weight.shape[0], -1)
    w_bn = torch.diag(
        bn.weight.to(dtype=conv.weight.dtype).div(
            torch.sqrt(bn.eps + bn.running_var.to(dtype=conv.weight.dtype))
        )
    )
    fused_weight = torch.mm(w_bn, w_conv).view_as(conv.weight)

    conv_bias = conv.bias
    if conv_bias is None:
        conv_bias = torch.zeros(conv.weight.shape[0], device=conv.weight.device, dtype=conv.weight.dtype)
    b_bn = (
        bn.bias.to(dtype=conv.weight.dtype)
        - bn.weight.to(dtype=conv.weight.dtype)
        .mul(bn.running_mean.to(dtype=conv.weight.dtype))
        .div(torch.sqrt(bn.running_var.to(dtype=conv.weight.dtype) + bn.eps))
    )
    fused_bias = torch.mm(w_bn, conv_bias.reshape(-1, 1)).reshape(-1) + b_bn
    return fused_weight, fused_bias


class YOLOConv(nn.Module):
    default_act = SiLU()

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 1,
        s: int = 1,
        p=None,
        g: int = 1,
        d: int = 1,
        act=True,
    ):
        super().__init__()
        self.conv = Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = BatchNorm2d(c2, eps=1e-3, momentum=0.03)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self._fused_silu = act is True
        self._is_fused = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._is_fused:
            return self.act(self.conv(x))
        if (
            self._fused_silu
            and not self.training
            and x.is_cuda
            and x.dtype == torch.float16
            and self.conv.groups == 1
            and self.conv.dilation == (1, 1)
            and x.is_contiguous()
        ):
            batch, channels, height, width = x.shape
            kh, kw = self.conv.kernel_size
            sh, sw = self.conv.stride
            ph, pw = self.conv.padding
            oh = (height + 2 * ph - kh) // sh + 1
            ow = (width + 2 * pw - kw) // sw + 1
            n = self.conv.out_channels
            m = batch * oh * ow
            key = (channels, n, kh, sh, height, width)

            if key == (16, 32, 3, 2, 320, 320):
                block_m, block_n, block_k, warps, stages = 64, 32, 32, 4, 2
            elif key == (64, 64, 3, 1, 20, 20):
                block_m, block_n, block_k, warps, stages = 16, 32, 128, 4, 2
            elif kh == kw == 1 and key in {
                (96, 64, 1, 1, 80, 80),
                (256, 128, 1, 1, 20, 20),
            }:
                if channels == 96:
                    block_m, block_n, block_k, warps, stages = 32, 64, 64, 4, 3
                elif batch == 1:
                    block_m, block_n, block_k, warps, stages = 16, 32, 128, 4, 3
                else:
                    block_m, block_n, block_k, warps, stages = 16, 64, 64, 4, 3
                out = torch.empty((batch, n, oh, ow), device=x.device, dtype=x.dtype)
                _pointwise_bn_silu[
                    (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
                ](
                    x,
                    self.conv.weight,
                    self.bn.weight,
                    self.bn.bias,
                    self.bn.running_mean,
                    self.bn.running_var,
                    out,
                    M=m,
                    N=n,
                    C=channels,
                    S=height * width,
                    EPS=self.bn.eps,
                    BLOCK_M=block_m,
                    BLOCK_N=block_n,
                    BLOCK_K=block_k,
                    num_warps=warps,
                    num_stages=stages,
                )
                return out
            else:
                return self.act(self.bn(self.conv(x)))

            out = torch.empty((batch, n, oh, ow), device=x.device, dtype=x.dtype)
            _conv_bn_silu[
                (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
            ](
                x,
                self.conv.weight,
                self.bn.weight,
                self.bn.bias,
                self.bn.running_mean,
                self.bn.running_var,
                out,
                M=m,
                N=n,
                C=channels,
                H=height,
                W=width,
                OH=oh,
                OW=ow,
                KH=kh,
                KW=kw,
                SH=sh,
                SW=sw,
                PH=ph,
                PW=pw,
                EPS=self.bn.eps,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                BLOCK_K=block_k,
                num_warps=warps,
                num_stages=stages,
            )
            return out
        return self.act(self.bn(self.conv(x)))

    @torch.no_grad()
    def fuse(self):
        if self._is_fused:
            return self
        fused_weight, fused_bias = _fuse_conv_bn(self.conv, self.bn)
        self.conv.weight.data.copy_(fused_weight)
        self.conv.bias = nn.Parameter(fused_bias)
        delattr(self, "bn")
        self._is_fused = True
        return self


def fuse_module(module: nn.Module) -> nn.Module:
    from .yolov10_repvggdw import YOLORepVGGDW

    for child in module.children():
        fuse_module(child)
    if isinstance(module, YOLOConv):
        module.fuse()
    elif isinstance(module, YOLORepVGGDW):
        module.fuse()
    return module
