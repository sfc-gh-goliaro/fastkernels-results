"""YOLOv10 tensor concatenation op."""

from __future__ import annotations

import torch
import torch.nn as nn

from fastkernels.infra.cuda_ext import lazy_op


_C = lazy_op("yolov10_concat_cuda", "yolov10_concat.cu")


class YOLOConcat(nn.Module):
    def __init__(self, dimension: int = 1):
        super().__init__()
        self.d = dimension

    def forward(self, xs: list[torch.Tensor]) -> torch.Tensor:
        if (
            self.d == 1
            and len(xs) == 2
            and xs[0].is_cuda
            and xs[0].dtype == torch.float16
            and xs[1].dtype == torch.float16
            and xs[0].ndim == 4
            and xs[1].ndim == 4
            and xs[0].is_contiguous()
            and xs[1].is_contiguous()
            and xs[0].shape[0] == xs[1].shape[0]
            and xs[0].shape[2:] == xs[1].shape[2:]
            and xs[0].numel() + xs[1].numel() >= 4 * 1024 * 1024
        ):
            return _C.concat_half(xs[0], xs[1])
        return torch.cat(xs, self.d)
