"""YOLOv10 SCDown (spatial channel downsampling) block."""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

from .yolov10_conv import YOLOConv


@triton.jit
def _pointwise_bn_silu(
    x,
    weight,
    bn_shift,
    out,
    spatial: tl.constexpr,
    total_p: tl.constexpr,
    c1: tl.constexpr,
    c2: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_p = tl.program_id(0)
    pid_c = tl.program_id(1)
    p = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
    c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    k = tl.arange(0, c1)
    p_mask = p < total_p

    n = p // spatial
    hw = p - n * spatial
    x_ptrs = x + n[None, :] * (c1 * spatial) + k[:, None] * spatial + hw[None, :]
    w_ptrs = weight + c[:, None] * c1 + k[None, :]
    values = tl.dot(
        tl.load(w_ptrs, mask=c[:, None] < c2, other=0.0),
        tl.load(x_ptrs, mask=p_mask[None, :], other=0.0),
    )

    shift = tl.load(bn_shift + c, mask=c < c2, other=0.0)
    values += shift[:, None]
    values = values * tl.sigmoid(values)

    out_ptrs = out + n[None, :] * (c2 * spatial) + c[:, None] * spatial + hw[None, :]
    tl.store(out_ptrs, values, mask=(c[:, None] < c2) & p_mask[None, :])


@triton.jit
def _depthwise_bn(
    x,
    weight,
    bn_shift,
    out,
    in_h: tl.constexpr,
    in_w: tl.constexpr,
    out_h: tl.constexpr,
    out_w: tl.constexpr,
    channels: tl.constexpr,
    BLOCK: tl.constexpr,
):
    nc = tl.program_id(0)
    p = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    out_spatial = out_h * out_w
    mask = p < out_spatial
    ow = p % out_w
    oh = p // out_w
    c = nc % channels

    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for ky in tl.static_range(3):
        iy = oh * 2 + ky - 1
        for kx in tl.static_range(3):
            ix = ow * 2 + kx - 1
            inside = mask & (iy >= 0) & (iy < in_h) & (ix >= 0) & (ix < in_w)
            x_offsets = nc * (in_h * in_w) + iy * in_w + ix
            v = tl.load(x + x_offsets, mask=inside, other=0.0)
            w = tl.load(weight + c * 9 + ky * 3 + kx)
            acc += v * w

    shift = tl.load(bn_shift + c)
    tl.store(out + nc * out_spatial + p, acc + shift, mask=mask)


class YOLOSCDown(nn.Module):
    def __init__(self, c1: int, c2: int, k: int, s: int):
        super().__init__()
        self.cv1 = YOLOConv(c1, c2, 1, 1)
        self.cv2 = YOLOConv(c2, c2, k=k, s=s, g=c2, act=False)
        self._fused_params = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._fused_params is None:
            scale1 = self.cv1.bn.weight.float() * torch.rsqrt(
                self.cv1.bn.running_var.float() + self.cv1.bn.eps
            )
            shift1 = self.cv1.bn.bias.float() - self.cv1.bn.running_mean.float() * scale1
            scale2 = self.cv2.bn.weight.float() * torch.rsqrt(
                self.cv2.bn.running_var.float() + self.cv2.bn.eps
            )
            shift2 = self.cv2.bn.bias.float() - self.cv2.bn.running_mean.float() * scale2
            pw_weight = (
                self.cv1.conv.weight.float() * scale1[:, None, None, None]
            ).to(self.cv1.conv.weight.dtype)
            dw_weight = (
                self.cv2.conv.weight.float() * scale2[:, None, None, None]
            ).to(self.cv2.conv.weight.dtype)
            self._fused_params = (pw_weight, shift1, dw_weight, shift2)
        pw_weight, shift1, dw_weight, shift2 = self._fused_params

        n, _, h, w = x.shape
        c1 = self.cv1.conv.weight.shape[1]
        c2 = self.cv1.conv.weight.shape[0]
        spatial = h * w
        out_h = (h + 1) // 2
        out_w = (w + 1) // 2
        out = torch.empty((n, c2, out_h, out_w), device=x.device, dtype=x.dtype)
        hidden = torch.empty((n, c2, h, w), device=x.device, dtype=x.dtype)

        block_p = 64
        block_c = 32 if n == 1 and spatial == 1600 else 128
        pointwise_warps = 8 if block_c == 128 else 4
        _pointwise_bn_silu[(triton.cdiv(n * spatial, block_p), triton.cdiv(c2, block_c))](
            x,
            pw_weight,
            shift1,
            hidden,
            spatial,
            n * spatial,
            c1,
            c2,
            BLOCK_P=block_p,
            BLOCK_C=block_c,
            num_warps=pointwise_warps,
        )

        out_spatial = out_h * out_w
        block = 256
        _depthwise_bn[(n * c2, triton.cdiv(out_spatial, block))](
            hidden,
            dw_weight,
            shift2,
            out,
            h,
            w,
            out_h,
            out_w,
            c2,
            BLOCK=block,
            num_warps=4,
        )
        return out
