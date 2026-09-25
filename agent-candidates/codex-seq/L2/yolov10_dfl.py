"""YOLOv10 Distribution Focal Loss layer."""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

from ..L1.conv2d import Conv2d
from ..L1.softmax import Softmax


@triton.jit
def _dfl_kernel(
    x,
    out,
    n_anchors: tl.constexpr,
    BLOCK_A: tl.constexpr,
):
    anchors = tl.program_id(0) * BLOCK_A + tl.arange(0, BLOCK_A)
    row = tl.program_id(1)
    bins = tl.arange(0, 16)
    offsets = row * (16 * n_anchors) + bins[:, None] * n_anchors + anchors[None, :]
    mask = anchors[None, :] < n_anchors

    values = tl.load(x + offsets, mask=mask, other=-float("inf")).to(tl.float32)
    values -= tl.max(values, axis=0)
    numerator = tl.exp(values)
    denominator = tl.sum(numerator, axis=0)
    result = tl.sum(numerator * bins[:, None], axis=0) / denominator
    tl.store(out + row * n_anchors + anchors, result, mask=anchors < n_anchors)


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
        if (
            x.is_cuda
            and x.dtype == torch.float16
            and self.c1 == 16
            and x.shape[1] == 64
        ):
            out = torch.empty((b, 4, a), device=x.device, dtype=x.dtype)
            block_a = 64
            _dfl_kernel[(triton.cdiv(a, block_a), b * 4)](
                x,
                out,
                n_anchors=a,
                BLOCK_A=block_a,
                num_warps=1,
                num_stages=4,
            )
            return out
        probabilities = self._softmax(
            x.view(b, 4, self.c1, a).transpose(2, 1)
        )
        return self.conv(probabilities).view(b, 4, a)
