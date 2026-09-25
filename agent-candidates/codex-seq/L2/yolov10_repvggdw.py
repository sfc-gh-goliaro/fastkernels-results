"""YOLOv10 RepVGG depthwise block."""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

from ..L1.silu import SiLU
from ..L1.tensor_ops import Pad
from .yolov10_conv import YOLOConv


@triton.jit
def _repvggdw_kernel(
    x,
    weight,
    bias,
    out,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)
    tile = pid % 2
    channel_plane = pid // 2
    channel = channel_plane % 256
    batch = channel_plane // 256
    pos = tile * BLOCK_S + tl.arange(0, BLOCK_S)
    oh = pos // 20
    ow = pos % 20
    base = (batch * 256 + channel) * 400

    acc7 = tl.zeros((BLOCK_S,), dtype=tl.float32)
    for kh in tl.static_range(0, 7):
        ih = oh + kh - 3
        for kw in tl.static_range(0, 7):
            iw = ow + kw - 3
            mask = (pos < 400) & (ih >= 0) & (ih < 20) & (iw >= 0) & (iw < 20)
            value = tl.load(x + base + ih * 20 + iw, mask=mask, other=0.0)
            kernel_weight = tl.load(weight + channel * 49 + kh * 7 + kw)
            acc7 += value.to(tl.float32) * kernel_weight

    value = acc7 + tl.load(bias + channel)
    value = value * tl.sigmoid(value)
    tl.store(out + base + pos, value, mask=pos < 400)


class YOLORepVGGDW(nn.Module):
    def __init__(self, ed: int):
        super().__init__()
        self.conv = YOLOConv(ed, ed, 7, 1, 3, g=ed, act=False)
        self.conv1 = YOLOConv(ed, ed, 3, 1, 1, g=ed, act=False)
        self.act = SiLU()
        self._pad = Pad()
        self._is_fused = False
        self.register_buffer("_kernel_weight", None, persistent=False)
        self.register_buffer("_kernel_bias", None, persistent=False)

    @torch.no_grad()
    def _prepare_kernel_weights(self):
        scale7 = self.conv.bn.weight.float() * torch.rsqrt(
            self.conv.bn.running_var.float() + self.conv.bn.eps
        )
        scale3 = self.conv1.bn.weight.float() * torch.rsqrt(
            self.conv1.bn.running_var.float() + self.conv1.bn.eps
        )
        weight = self.conv.conv.weight.float() * scale7[:, None, None, None]
        weight[:, :, 2:5, 2:5].add_(
            self.conv1.conv.weight.float() * scale3[:, None, None, None]
        )
        self._kernel_weight = weight.contiguous()
        self._kernel_bias = (
            self.conv.bn.bias.float()
            - self.conv.bn.running_mean.float() * scale7
            + self.conv1.bn.bias.float()
            - self.conv1.bn.running_mean.float() * scale3
        ).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            not self._is_fused
            and x.is_cuda
            and x.is_contiguous()
            and x.dtype == torch.float16
            and x.shape[1:] == (256, 20, 20)
            and x.shape[0] in (1, 4)
        ):
            if self._kernel_weight is None:
                self._prepare_kernel_weights()
            out = torch.empty_like(x)
            _repvggdw_kernel[(x.shape[0] * 512,)](
                x,
                self._kernel_weight,
                self._kernel_bias,
                out,
                BLOCK_S=256,
                num_warps=4,
                num_stages=1,
            )
            return out
        if self._is_fused:
            return self.act(self.conv(x))
        return self.act(self.conv(x) + self.conv1(x))

    @torch.no_grad()
    def fuse(self):
        if self._is_fused:
            return self
        self.conv.fuse()
        self.conv1.fuse()
        final_conv_w = self.conv.conv.weight.data + self._pad(self.conv1.conv.weight.data, [2, 2, 2, 2])
        final_conv_b = self.conv.conv.bias.data + self.conv1.conv.bias.data
        self.conv.conv.weight.data.copy_(final_conv_w)
        self.conv.conv.bias.data.copy_(final_conv_b)
        delattr(self, "conv1")
        self._is_fused = True
        return self
