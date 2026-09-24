"""YOLOv10 tensor concatenation op."""

from __future__ import annotations

import os

import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))


def _build():
    from torch.utils.cpp_extension import load

    return load(
        name="ako_yolov10_concat",
        sources=[os.path.join(_HERE, "cat_kernel.cu")],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )


try:
    _C = _build()
except Exception:  # pragma: no cover - no nvcc / no CUDA: stay on torch.cat
    _C = None


class YOLOConcat(nn.Module):
    def __init__(self, dimension: int = 1):
        super().__init__()
        self.d = dimension

    def forward(self, xs: list[torch.Tensor]) -> torch.Tensor:
        if _C is not None:
            out = _C.cat_fast(xs, self.d)
            if out is not None:
                return out
        return torch.cat(xs, self.d)
