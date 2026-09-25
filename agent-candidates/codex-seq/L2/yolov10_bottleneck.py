"""YOLOv10 bottleneck block."""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

from .yolov10_conv import YOLOConv


@triton.jit
def _conv_bn_silu(
    x,
    weight,
    scale,
    shift,
    residual,
    out,
    x_batch_stride,
    residual_batch_stride,
    M: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    ADD_RESIDUAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    image = offs_m // (H * W)
    pos = offs_m % (H * W)
    oh = pos // W
    ow = pos % W
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    K: tl.constexpr = C * 9
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        channel = offs_k // 9
        kernel_pos = offs_k % 9
        ih = oh[:, None] + kernel_pos[None, :] // 3 - 1
        iw = ow[:, None] + kernel_pos[None, :] % 3 - 1
        x_offsets = (
            image[:, None] * x_batch_stride
            + channel[None, :] * H * W
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
            mask=(offs_n[None, :] < C) & (offs_k[:, None] < K),
            other=0.0,
        )
        acc = tl.dot(a, b, acc)

    channel_mask = offs_n < C
    channel_scale = tl.load(scale + offs_n, mask=channel_mask, other=0.0)
    channel_shift = tl.load(shift + offs_n, mask=channel_mask, other=0.0)
    z = acc * channel_scale[None, :] + channel_shift[None, :]
    y = z * tl.sigmoid(z)

    out_offsets = (
        image[:, None] * C * H * W
        + offs_n[None, :] * H * W
        + pos[:, None]
    )
    out_mask = (offs_m[:, None] < M) & channel_mask[None, :]
    if ADD_RESIDUAL:
        y = y.to(tl.float16)
        residual_offsets = (
            image[:, None] * residual_batch_stride
            + offs_n[None, :] * H * W
            + pos[:, None]
        )
        y += tl.load(residual + residual_offsets, mask=out_mask, other=0.0)
    tl.store(out + out_offsets, y, mask=out_mask)


class YOLOBottleneck(nn.Module):
    def __init__(self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k=(3, 3), e: float = 0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = YOLOConv(c1, c_, k[0], 1)
        self.cv2 = YOLOConv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2
        self._fast_path = c1 == c2 == c_ and g == 1 and k == (3, 3)
        self._bn_cache = None

    def _bn_affine(self):
        tensors = (
            self.cv1.bn.weight,
            self.cv1.bn.bias,
            self.cv1.bn.running_mean,
            self.cv1.bn.running_var,
            self.cv2.bn.weight,
            self.cv2.bn.bias,
            self.cv2.bn.running_mean,
            self.cv2.bn.running_var,
        )
        versions = tuple(t._version for t in tensors)
        if self._bn_cache is None or self._bn_cache[0] != versions:
            affine = []
            for cv in (self.cv1, self.cv2):
                scale = cv.bn.weight.float() * torch.rsqrt(
                    cv.bn.running_var.float() + cv.bn.eps
                )
                shift = cv.bn.bias.float() - cv.bn.running_mean.float() * scale
                affine.extend((scale, shift))
            self._bn_cache = (versions, *affine)
        return self._bn_cache[1:]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            self._fast_path
            and x.is_cuda
            and x.dtype == torch.float16
            and x.stride(1) == x.shape[2] * x.shape[3]
            and x.stride(2) == x.shape[3]
            and x.stride(3) == 1
            and not self.training
            and hasattr(self.cv1, "bn")
            and hasattr(self.cv2, "bn")
        ):
            batch, channels, height, width = x.shape
            if channels in (16, 32, 64, 128) and height * channels == 2560:
                if channels == 16:
                    block_m, block_n, block_k, warps = 128, 16, 16, 4
                    ctas = 1
                elif channels == 32:
                    block_m, block_n, block_k, warps = 64, 32, 32, 4
                    ctas = 1
                elif channels == 64:
                    block_m, block_n, block_k, warps = 16, 64, 64, 4
                    ctas = 1
                else:
                    block_m, block_n, block_k, warps = 16, 128, 128, 4
                    ctas = 2

                m = batch * height * width
                buffers = torch.empty((2, *x.shape), device=x.device, dtype=x.dtype)
                hidden, out = buffers.unbind(0)
                scale1, shift1, scale2, shift2 = self._bn_affine()
                grid = (triton.cdiv(m, block_m), triton.cdiv(channels, block_n))
                _conv_bn_silu[grid](
                    x,
                    self.cv1.conv.weight,
                    scale1,
                    shift1,
                    x,
                    hidden,
                    x.stride(0),
                    x.stride(0),
                    M=m,
                    C=channels,
                    H=height,
                    W=width,
                    ADD_RESIDUAL=False,
                    BLOCK_M=block_m,
                    BLOCK_N=block_n,
                    BLOCK_K=block_k,
                    num_warps=warps,
                    num_stages=2,
                    num_ctas=ctas,
                )
                _conv_bn_silu[grid](
                    hidden,
                    self.cv2.conv.weight,
                    scale2,
                    shift2,
                    x,
                    out,
                    hidden.stride(0),
                    x.stride(0),
                    M=m,
                    C=channels,
                    H=height,
                    W=width,
                    ADD_RESIDUAL=self.add,
                    BLOCK_M=block_m,
                    BLOCK_N=block_n,
                    BLOCK_K=block_k,
                    num_warps=warps,
                    num_stages=2,
                    num_ctas=ctas,
                )
                return out
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y
