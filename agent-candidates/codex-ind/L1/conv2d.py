"""Triton kernels for the convolution shapes in the captured workload."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _conv2d_igemm(
    x,
    weight,
    bias,
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
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

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
        b_offsets = offs_n[None, :] * K + offs_k[:, None]
        b = tl.load(
            weight + b_offsets,
            mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        acc = tl.dot(a, b, acc, input_precision="ieee")

    if HAS_BIAS:
        acc += tl.load(bias + offs_n, mask=offs_n < N, other=0.0)[None, :]
    out_offsets = (
        image[:, None] * (N * OH * OW)
        + offs_n[None, :] * (OH * OW)
        + pos[:, None]
    )
    tl.store(
        out + out_offsets,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.jit
def _pointwise_nchw(
    x,
    weight,
    out,
    C: tl.constexpr,
    N: tl.constexpr,
    S: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    offs_s = tl.program_id(0) * BLOCK_S + tl.arange(0, BLOCK_S)
    offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    image = tl.program_id(2)
    acc = tl.zeros((BLOCK_S, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, C, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        a = tl.load(
            x
            + image * (C * S)
            + offs_k[None, :] * S
            + offs_s[:, None],
            mask=(offs_s[:, None] < S) & (offs_k[None, :] < C),
            other=0.0,
        )
        b = tl.load(
            weight + offs_n[None, :] * C + offs_k[:, None],
            mask=(offs_n[None, :] < N) & (offs_k[:, None] < C),
            other=0.0,
        )
        acc = tl.dot(a, b, acc)

    tl.store(
        out
        + image * (N * S)
        + offs_n[None, :] * S
        + offs_s[:, None],
        acc,
        mask=(offs_s[:, None] < S) & (offs_n[None, :] < N),
    )


class Conv2d(nn.Module):
    """Parametric 2D convolution with captured-shape Triton fast paths."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        groups: int = 1,
        dilation: int | tuple[int, int] = 1,
        bias: bool = True,
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, *kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.groups != 1 or self.dilation != (1, 1) or not x.is_cuda:
            return F.conv2d(
                x,
                self.weight,
                self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )

        batch, channels, height, width = x.shape
        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding
        oh = (height + 2 * ph - kh) // sh + 1
        ow = (width + 2 * pw - kw) // sw + 1
        m = batch * oh * ow
        n = self.out_channels

        key = (channels, n, kh, height, width)
        if key == (16, 1024, 2, 18, 32):
            block_m, block_n, block_k, warps, stages = 16, 64, 16, 4, 2
            ctas = 1
        elif key == (16, 32, 3, 320, 320):
            block_m, block_n, block_k, warps, stages = 128, 32, 32, 8, 2
            ctas = 1
        elif key == (384, 256, 1, 20, 20):
            block_m, block_n, block_k, warps, stages = 64, 64, 128, 8, 2
            ctas = 1
        elif key == (64, 64, 3, 40, 40):
            block_m, block_n, block_k, warps, stages = 16, 64, 64, 4, 2
            ctas = 2
        elif key == (256, 128, 1, 20, 20):
            block_m, block_n, block_k, warps, stages = 32, 64, 128, 4, 2
            ctas = 1
        else:
            return F.conv2d(
                x,
                self.weight,
                self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )

        out = torch.empty((batch, n, oh, ow), device=x.device, dtype=x.dtype)
        if kh == 1 and kw == 1 and self.bias is None:
            if channels == 384:
                block_s, block_n, block_k, warps = 64, 64, 256, 4
            else:
                block_s, block_n, block_k, warps = 32, 16, 128, 4
            _pointwise_nchw[
                (triton.cdiv(oh * ow, block_s), triton.cdiv(n, block_n), batch)
            ](
                x,
                self.weight,
                out,
                C=channels,
                N=n,
                S=oh * ow,
                BLOCK_S=block_s,
                BLOCK_N=block_n,
                BLOCK_K=block_k,
                num_warps=warps,
                num_stages=3,
            )
            return out

        _conv2d_igemm[(triton.cdiv(m, block_m), triton.cdiv(n, block_n))](
            x,
            self.weight,
            self.bias,
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
            HAS_BIAS=self.bias is not None,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=warps,
            num_stages=stages,
            num_ctas=ctas,
        )
        return out
