"""YOLOv10 C2f and C2fCIB blocks."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .yolov10_bottleneck import YOLOBottleneck
from .yolov10_cib import YOLOCIB
from .yolov10_conv import YOLOConv, fuse_module


def _conv_silu(module: YOLOConv, x: torch.Tensor) -> torch.Tensor:
    conv = module.conv
    y = F.conv2d(x, conv.weight, conv.bias, conv.stride, conv.padding, conv.dilation, conv.groups)
    return F.silu(y, inplace=True)


class YOLOC2f(nn.Module):
    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = YOLOConv(c1, 2 * self.c, 1, 1)
        self.cv2 = YOLOConv((2 + n) * self.c, c2, 1, 1)
        self.m = nn.ModuleList(
            YOLOBottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n)
        )
        self._fused_for_eval = False

    @staticmethod
    def _run_block(block: nn.Module, x: torch.Tensor) -> torch.Tensor:
        y = _conv_silu(block.cv1, x)
        y = _conv_silu(block.cv2, y)
        return y.add_(x) if block.add else y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._fused_for_eval:
            fuse_module(self)
            self._fused_for_eval = True
        y = list(_conv_silu(self.cv1, x).chunk(2, 1))
        for block in self.m:
            y.append(self._run_block(block, y[-1]))
        return _conv_silu(self.cv2, torch.cat(y, 1))


class YOLOC2fCIB(YOLOC2f):
    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, lk: bool = False, g: int = 1, e: float = 0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(YOLOCIB(self.c, self.c, shortcut, e=1.0, lk=lk) for _ in range(n))

    @staticmethod
    def _run_block(block: nn.Module, x: torch.Tensor) -> torch.Tensor:
        y = x
        for layer in block.cv1:
            if isinstance(layer, YOLOConv):
                y = _conv_silu(layer, y)
            else:
                y = _conv_silu(layer.conv, y)
        return y.add_(x) if block.add else y
