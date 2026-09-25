"""YOLOv10 SCDown (spatial channel downsampling) block."""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

from .yolov10_conv import YOLOConv


@triton.jit
def _scdown_fused(
    x,
    pw_weight,
    dw_weight,
    bn1_scale,
    bn1_shift,
    bn2_scale,
    bn2_shift,
    out,
    CIN: tl.constexpr,
    COUT: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    K: tl.constexpr,
    STRIDE: tl.constexpr,
    PAD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pos = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    channel = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    batch = tl.program_id(2)
    oh = pos // OW
    ow = pos % OW
    pos_mask = pos < OH * OW
    channel_mask = channel < COUT

    scale1 = tl.load(bn1_scale + channel, mask=channel_mask, other=0.0)
    shift1 = tl.load(bn1_shift + channel, mask=channel_mask, other=0.0)

    result = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for ky in range(K):
        ih = oh * STRIDE + ky - PAD
        for kx in range(K):
            iw = ow * STRIDE + kx - PAD
            spatial_mask = pos_mask & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
            pointwise = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k0 in range(0, CIN, BLOCK_K):
                cin = k0 + tl.arange(0, BLOCK_K)
                x_offsets = (
                    (batch * CIN + cin[None, :]) * H * W
                    + ih[:, None] * W
                    + iw[:, None]
                )
                a = tl.load(
                    x + x_offsets,
                    mask=spatial_mask[:, None] & (cin[None, :] < CIN),
                    other=0.0,
                )
                b = tl.load(
                    pw_weight + channel[None, :] * CIN + cin[:, None],
                    mask=(cin[:, None] < CIN) & channel_mask[None, :],
                    other=0.0,
                )
                pointwise = tl.dot(a, b, pointwise)
            pointwise = pointwise * scale1[None, :] + shift1[None, :]
            pointwise = pointwise * tl.sigmoid(pointwise)
            depthwise = tl.load(
                dw_weight + channel * K * K + ky * K + kx,
                mask=channel_mask,
                other=0.0,
            )
            result += tl.where(
                spatial_mask[:, None],
                pointwise * depthwise[None, :],
                0.0,
            )

    scale2 = tl.load(bn2_scale + channel, mask=channel_mask, other=0.0)
    shift2 = tl.load(bn2_shift + channel, mask=channel_mask, other=0.0)
    result = result * scale2[None, :] + shift2[None, :]
    out_offsets = (
        batch * COUT * OH * OW + channel[None, :] * OH * OW + pos[:, None]
    )
    tl.store(out + out_offsets, result, mask=pos_mask[:, None] & channel_mask[None, :])


@triton.jit
def _silu_depthwise_bn(
    x,
    dw_weight,
    bn1_scale,
    bn1_shift,
    bn2_scale,
    bn2_shift,
    out,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    K: tl.constexpr,
    STRIDE: tl.constexpr,
    PAD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    bc = tl.program_id(0)
    batch = bc // C
    channel = bc % C
    pos = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    oh = pos // OW
    ow = pos % OW
    out_mask = pos < OH * OW

    scale1 = tl.load(bn1_scale + channel)
    shift1 = tl.load(bn1_shift + channel)

    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for ky in range(K):
        ih = oh * STRIDE + ky - PAD
        for kx in range(K):
            iw = ow * STRIDE + kx - PAD
            valid = out_mask & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
            offsets = (batch * C + channel) * H * W + ih * W + iw
            value = tl.load(x + offsets, mask=valid, other=0.0).to(tl.float32)
            value = value * scale1 + shift1
            value = value * tl.sigmoid(value)
            weight = tl.load(dw_weight + channel * K * K + ky * K + kx).to(
                tl.float32
            )
            acc += tl.where(valid, value * weight, 0.0)

    scale2 = tl.load(bn2_scale + channel)
    shift2 = tl.load(bn2_shift + channel)
    result = acc * scale2 + shift2
    out_offsets = (batch * C + channel) * OH * OW + pos
    tl.store(out + out_offsets, result, mask=out_mask)


class YOLOSCDown(nn.Module):
    def __init__(self, c1: int, c2: int, k: int, s: int):
        super().__init__()
        self.cv1 = YOLOConv(c1, c2, 1, 1)
        self.cv2 = YOLOConv(c2, c2, k=k, s=s, g=c2, act=False)

    def _bn_coefficients(self):
        cached = getattr(self, "_cached_bn_coefficients", None)
        if cached is not None:
            return cached

        coefficients = []
        for bn in (self.cv1.bn, self.cv2.bn):
            scale = bn.weight.float() * torch.rsqrt(bn.running_var.float() + bn.eps)
            shift = bn.bias.float() - bn.running_mean.float() * scale
            coefficients.extend((scale, shift))
        self._cached_bn_coefficients = tuple(coefficients)
        return self._cached_bn_coefficients

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            x.is_cuda
            and not self.training
            and self.cv1.bn.track_running_stats
            and self.cv2.bn.track_running_stats
        ):
            batch, in_channels, height, width = x.shape
            channels = self.cv1.conv.weight.shape[0]
            kernel = self.cv2.conv.weight.shape[-1]
            stride = self.cv2.conv.stride[0]
            pad = self.cv2.conv.padding[0]
            out_h = (height + 2 * pad - kernel) // stride + 1
            out_w = (width + 2 * pad - kernel) // stride + 1
            out = torch.empty(
                (batch, channels, out_h, out_w),
                device=x.device,
                dtype=x.dtype,
            )
            scale1, shift1, scale2, shift2 = self._bn_coefficients()
            if batch == 1:
                block_m, block_n, block_k = 16, 32, 64
                _scdown_fused[
                    (
                        triton.cdiv(out_h * out_w, block_m),
                        triton.cdiv(channels, block_n),
                        batch,
                    )
                ](
                    x,
                    self.cv1.conv.weight,
                    self.cv2.conv.weight,
                    scale1,
                    shift1,
                    scale2,
                    shift2,
                    out,
                    CIN=in_channels,
                    COUT=channels,
                    H=height,
                    W=width,
                    OH=out_h,
                    OW=out_w,
                    K=kernel,
                    STRIDE=stride,
                    PAD=pad,
                    BLOCK_M=block_m,
                    BLOCK_N=block_n,
                    BLOCK_K=block_k,
                    num_warps=4,
                    num_stages=2,
                )
            else:
                hidden = self.cv1.conv(x)
                block = 256
                _silu_depthwise_bn[
                    (batch * channels, triton.cdiv(out_h * out_w, block))
                ](
                    hidden,
                    self.cv2.conv.weight,
                    scale1,
                    shift1,
                    scale2,
                    shift2,
                    out,
                    C=channels,
                    H=height,
                    W=width,
                    OH=out_h,
                    OW=out_w,
                    K=kernel,
                    STRIDE=stride,
                    PAD=pad,
                    BLOCK=block,
                    num_warps=4,
                    num_stages=2,
                )
            return out
        return self.cv2(self.cv1(x))
