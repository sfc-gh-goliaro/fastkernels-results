"""Shape-tuned SiLU-and-Mul activation for BF16 CUDA inputs."""

from __future__ import annotations

import torch
import torch.nn as nn
from fastkernels.infra.cuda_ext import lazy_op

_C = lazy_op("silu_and_mul_candidate", "silu_and_mul.cu")


class SiluAndMul(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return _C.silu_and_mul(x)
