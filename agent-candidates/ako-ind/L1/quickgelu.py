"""QuickGELU activation: x * sigmoid(1.702 * x).

Approximation of GELU used in Qwen2-VL vision encoder.

One fused elementwise CUDA kernel replaces eager's three (1.702*x, sigmoid,
x*s), so the tensor makes one HBM round trip instead of three and pays one
launch instead of three. The launch is issued with Programmatic Dependent
Launch, which overlaps grid dispatch with the tail of the preceding kernel and
takes the remaining launch latency to ~0. See ``quickgelu.cu``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    from fastkernels.infra.cuda_ext import load_op

    _C = load_op("quickgelu_fused_pdl", "quickgelu.cu")
except Exception:  # no CUDA toolchain / build failure -> eager formula
    _C = None


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (_C is not None and x.dtype is torch.float32 and x.is_cuda
                and not x.requires_grad and x.is_contiguous()):
            return _C.quickgelu(x)
        return x * torch.sigmoid(1.702 * x)
