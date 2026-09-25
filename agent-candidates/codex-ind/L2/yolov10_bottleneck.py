"""YOLOv10 bottleneck block."""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

from .yolov10_conv import YOLOConv


@triton.jit
def _conv_bn_silu_kernel(
    x,
    weight,
    bn_weight,
    bn_bias,
    running_mean,
    running_var,
    residual,
    out,
    total_m,
    height: tl.constexpr,
    width: tl.constexpr,
    channels: tl.constexpr,
    eps: tl.constexpr,
    add_residual: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    hw: tl.constexpr = height * width
    k_total: tl.constexpr = channels * 9

    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_n = pid_n * block_n + tl.arange(0, block_n)
    batch = offs_m // hw
    pos = offs_m % hw
    oy = pos // width
    ox = pos % width
    acc = tl.zeros((block_m, block_n), dtype=tl.float32)

    for k0 in range(0, k_total, block_k):
        offs_k = k0 + tl.arange(0, block_k)
        ic = offs_k // 9
        tap = offs_k % 9
        iy = oy[:, None] + tap[None, :] // 3 - 1
        ix = ox[:, None] + tap[None, :] % 3 - 1
        x_offs = (
            batch[:, None] * channels * hw
            + ic[None, :] * hw
            + iy * width
            + ix
        )
        x_mask = (
            (offs_m[:, None] < total_m)
            & (offs_k[None, :] < k_total)
            & (iy >= 0)
            & (iy < height)
            & (ix >= 0)
            & (ix < width)
        )
        a = tl.load(x + x_offs, mask=x_mask, other=0.0)

        w_offs = offs_n[:, None] * k_total + offs_k[None, :]
        w = tl.load(
            weight + w_offs,
            mask=(offs_n[:, None] < channels) & (offs_k[None, :] < k_total),
            other=0.0,
        )
        acc += tl.dot(a, tl.trans(w))

    gamma = tl.load(bn_weight + offs_n, mask=offs_n < channels, other=0.0)
    beta = tl.load(bn_bias + offs_n, mask=offs_n < channels, other=0.0)
    mean = tl.load(running_mean + offs_n, mask=offs_n < channels, other=0.0)
    var = tl.load(running_var + offs_n, mask=offs_n < channels, other=1.0)
    acc = (acc - mean[None, :]) * tl.rsqrt(var[None, :] + eps)
    acc = acc * gamma[None, :] + beta[None, :]
    acc = acc * tl.sigmoid(acc)

    out_offs = (
        batch[:, None] * channels * hw
        + offs_n[None, :] * hw
        + pos[:, None]
    )
    out_mask = (offs_m[:, None] < total_m) & (offs_n[None, :] < channels)
    if add_residual:
        acc += tl.load(residual + out_offs, mask=out_mask, other=0.0)
    tl.store(out + out_offs, acc, mask=out_mask)


class YOLOBottleneck(nn.Module):
    def __init__(self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k=(3, 3), e: float = 0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = YOLOConv(c1, c_, k[0], 1)
        self.cv2 = YOLOConv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        if (
            x.is_cuda
            and x.dtype == torch.float16
            and c == self.cv1.conv.weight.shape[1]
            and c == self.cv1.conv.weight.shape[0]
            and c == self.cv2.conv.weight.shape[0]
            and self.cv1.conv.weight.shape[2:] == (3, 3)
            and self.cv2.conv.weight.shape[2:] == (3, 3)
            and self.cv1.conv.groups == 1
            and self.cv2.conv.groups == 1
        ):
            y = torch.empty_like(x)
            out = torch.empty_like(x)
            block_m = 64 if c <= 32 else 16
            block_n = 32 if c <= 32 else 64
            block_k = 32 if c <= 32 else 64
            num_warps = 4
            grid = (triton.cdiv(n * h * w, block_m), triton.cdiv(c, block_n))
            _conv_bn_silu_kernel[grid](
                x,
                self.cv1.conv.weight,
                self.cv1.bn.weight,
                self.cv1.bn.bias,
                self.cv1.bn.running_mean,
                self.cv1.bn.running_var,
                x,
                y,
                n * h * w,
                h,
                w,
                c,
                self.cv1.bn.eps,
                False,
                block_m,
                block_n,
                block_k,
                num_warps=num_warps,
            )
            _conv_bn_silu_kernel[grid](
                y,
                self.cv2.conv.weight,
                self.cv2.bn.weight,
                self.cv2.bn.bias,
                self.cv2.bn.running_mean,
                self.cv2.bn.running_var,
                x,
                out,
                n * h * w,
                h,
                w,
                c,
                self.cv2.bn.eps,
                self.add,
                block_m,
                block_n,
                block_k,
                num_warps=num_warps,
            )
            return out
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y
