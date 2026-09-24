"""YOLOv10 Distribution Focal Loss layer -- one fused CUDA kernel.

The baseline chain is ``view -> transpose -> softmax(dim=1) -> 1x1 conv``,
which for every (batch, group g in 0..3, anchor a) computes

    out[b, g, a] = sum_c  c * softmax_c( x[b, g*16 + c, a] )

``yolov10_dfl_kernel.cu`` fuses all of it into a single memory-bound pass; the
kernel reads the captured (strided) layout directly, and the baseline chain
stays as the fallback for anything it does not cover (c1 != 16, non-fp16).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ....infra.cuda_ext import load_op
from ..L1.conv2d import Conv2d
from ..L1.softmax import Softmax

_C = load_op("fk_yolov10_dfl", "yolov10_dfl_kernel.cu")


class YOLODFL(nn.Module):
    def __init__(self, c1: int = 16):
        super().__init__()
        self.conv = Conv2d(c1, 1, 1, bias=False)
        self.conv.requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1
        self._softmax = Softmax(dim=1)
        self._fused = _C.dfl_forward if c1 == 16 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._fused is not None and x.dtype is torch.float16:
            return self._fused(x)
        b, _, a = x.shape
        return self.conv(self._softmax(x.view(b, 4, self.c1, a).transpose(2, 1))).view(b, 4, a)
