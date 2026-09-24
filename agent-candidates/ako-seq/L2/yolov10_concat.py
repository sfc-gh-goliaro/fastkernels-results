"""YOLOv10 tensor concatenation op.

``torch.cat`` of two contiguous fp16 NCHW tensors on dim 1 is a batched pair of
contiguous runs, so the copy itself needs no per-element index math. But the
captured outputs are only 0.3-9.8 MB -- 0.09-2.8 us of B200 HBM traffic against
a ~1.8 us gap between consecutive stream operations, so what decides the score
is the *launch*, not the copy. This is one kernel for both inputs, issued with
Programmatic Dependent Launch so its CTAs are dispatched during the preceding
operation's tail and that gap disappears. Without PDL the same kernel is a wash
with ``torch.cat``; with it, the concat is free outright on the batch-1 shapes.
See ``yolo_cat2.cu`` for the profile this is built on.

The Python forward is two cheap checks and one call; everything else (shape
math, output allocation, alignment gating, and the ``at::cat`` fallback for
inputs the fast path does not cover) happens in C++, where it is ~free.
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    from fastkernels.infra.cuda_ext import load_op

    _C = load_op("yolo_cat2_pdl", "yolo_cat2.cu")
except Exception:  # no toolchain / build failure -> eager
    _C = None


class YOLOConcat(nn.Module):
    def __init__(self, dimension: int = 1):
        super().__init__()
        self.d = dimension
        self._fast = _C is not None and dimension == 1

    def forward(self, xs: list[torch.Tensor]) -> torch.Tensor:
        if self._fast and len(xs) == 2:
            return _C.cat2(xs[0], xs[1])
        return torch.cat(xs, self.d)
