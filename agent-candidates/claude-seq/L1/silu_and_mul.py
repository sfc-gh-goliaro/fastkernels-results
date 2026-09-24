"""SiLU-and-Mul activation: fused CUDA kernel with a pure-PyTorch compile path."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastkernels.infra.cuda_ext import lazy_op

_C = lazy_op("silu_and_mul_fk", "silu_and_mul_fk.cu")


class SiluAndMul(nn.Module):
    def __init__(self):
        super().__init__()
        # Resolve the extension once so the hot path is a single pybind call.
        self._fwd = _C.silu_and_mul_fwd

    @staticmethod
    def forward_native(x: torch.Tensor) -> torch.Tensor:
        """Pure PyTorch implementation — visible to Inductor for fusion."""
        d = x.shape[-1] // 2
        return F.silu(x[..., :d]) * x[..., d:]

    @staticmethod
    def forward_cuda(x: torch.Tensor) -> torch.Tensor:
        return _C.silu_and_mul_fwd(x)

    def forward(self, x):
        if torch.compiler.is_compiling():
            return self.forward_native(x)
        return self._fwd(x)
