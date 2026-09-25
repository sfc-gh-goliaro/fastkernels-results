"""YOLOv10 Distribution Focal Loss layer."""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

from ..L1.conv2d import Conv2d
from ..L1.softmax import Softmax


@triton.jit
def _dfl_kernel(x, out, n_anchors: tl.constexpr, BLOCK: tl.constexpr):
    anchors = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    batch_group = tl.program_id(1)
    group = batch_group % 4
    batch = batch_group // 4
    bins = tl.arange(0, 16)
    offsets = ((batch * 64 + group * 16 + bins[:, None]) * n_anchors
               + anchors[None, :])
    values = tl.load(x + offsets, mask=anchors[None, :] < n_anchors,
                     other=0.0).to(tl.float32)
    probs = tl.exp(values)
    numerator = tl.sum(probs * bins[:, None], axis=0)
    denominator = tl.sum(probs, axis=0)
    out_offsets = (batch * 4 + group) * n_anchors + anchors
    tl.store(out + out_offsets, numerator / denominator,
             mask=anchors < n_anchors)


class YOLODFL(nn.Module):
    def __init__(self, c1: int = 16):
        super().__init__()
        self.conv = Conv2d(c1, 1, 1, bias=False)
        self.conv.requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1
        self._softmax = Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, a = x.shape
        out = torch.empty((b, 4, a), device=x.device, dtype=x.dtype)
        block = 32 if b == 1 else 64
        _dfl_kernel[(triton.cdiv(a, block), 4 * b)](
            x, out, a, BLOCK=block, num_warps=1,
        )
        return out
