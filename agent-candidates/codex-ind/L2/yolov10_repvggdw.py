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
def _repvggdw_20x20_kernel(
    x,
    weight,
    bias,
    out,
    CHANNELS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    nc = tl.program_id(0)
    channel = nc % CHANNELS
    offsets = tl.arange(0, BLOCK)
    active = offsets < 400
    oy = offsets // 20
    ox = offsets % 20
    image = x + nc * 400

    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for ky in tl.static_range(7):
        iy = oy + ky - 3
        valid_y = (iy >= 0) & (iy < 20)
        for kx in tl.static_range(7):
            ix = ox + kx - 3
            values = tl.load(
                image + iy * 20 + ix,
                mask=active & valid_y & (ix >= 0) & (ix < 20),
                other=0.0,
            )
            kernel_value = tl.load(weight + channel * 49 + ky * 7 + kx)
            acc += values * kernel_value

    value = acc + tl.load(bias + channel)
    value = value * tl.sigmoid(value)
    tl.store(out + nc * 400 + offsets, value, mask=active)


class YOLORepVGGDW(nn.Module):
    def __init__(self, ed: int):
        super().__init__()
        self.conv = YOLOConv(ed, ed, 7, 1, 3, g=ed, act=False)
        self.conv1 = YOLOConv(ed, ed, 3, 1, 1, g=ed, act=False)
        self.act = SiLU()
        self._pad = Pad()
        self._is_fused = False
        self.register_buffer("_kernel_weight", torch.empty(0), persistent=False)
        self.register_buffer("_kernel_bias", torch.empty(0), persistent=False)

    @torch.no_grad()
    def _refresh_kernel(self):
        scale7 = self.conv.bn.weight.float() * torch.rsqrt(
            self.conv.bn.running_var.float() + self.conv.bn.eps
        )
        scale3 = self.conv1.bn.weight.float() * torch.rsqrt(
            self.conv1.bn.running_var.float() + self.conv1.bn.eps
        )
        weight = self.conv.conv.weight.float() * scale7[:, None, None, None]
        weight[:, :, 2:5, 2:5] += (
            self.conv1.conv.weight.float() * scale3[:, None, None, None]
        )
        self._kernel_weight = weight.contiguous()
        self._kernel_bias = (
            self.conv.bn.bias.float()
            - self.conv.bn.running_mean.float() * scale7
            + self.conv1.bn.bias.float()
            - self.conv1.bn.running_mean.float() * scale3
        ).contiguous()

    def load_state_dict(self, *args, **kwargs):
        result = super().load_state_dict(*args, **kwargs)
        self._refresh_kernel()
        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._is_fused:
            return self.act(self.conv(x))
        if (
            not self.training
            and x.is_cuda
            and x.dtype == torch.float16
            and x.ndim == 4
            and x.shape[1:] == (256, 20, 20)
            and x.is_contiguous()
        ):
            if (
                self._kernel_weight.numel() != 256 * 49
                or self._kernel_weight.device != x.device
            ):
                self._refresh_kernel()
            out = torch.empty_like(x)
            _repvggdw_20x20_kernel[(x.shape[0] * 256,)](
                x,
                self._kernel_weight,
                self._kernel_bias,
                out,
                CHANNELS=256,
                BLOCK=512,
                num_warps=4,
                num_stages=2,
            )
            return out
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
