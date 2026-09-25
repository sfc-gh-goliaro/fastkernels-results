"""Fused MoE sum kernel: reduces top-k expert outputs into final output.

Uses a custom CUDA kernel for high-performance reduction.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from fastkernels.infra.cuda_ext import lazy_op

_C = lazy_op("moe_sum_candidate", "moe_sum_opt.cu")


class MoeSum(nn.Module):
    """Fused top-k reduction for MoE outputs using sgl_kernel."""

    def __init__(self):
        super().__init__()
        self._output = None
        self._rows = -1
        self._op = None

    def forward(
        self,
        input: torch.Tensor,
        topk: int,
    ) -> torch.Tensor:
        """Sum over the topk dimension.

        Args:
            input: [M * topk, D] tensor
            topk: number of experts per token

        Returns:
            output: [M, D] tensor
        """
        rows = input.size(0)
        if rows != self._rows:
            self._output = torch.empty(
                rows // topk, input.size(1), device=input.device, dtype=input.dtype
            )
            self._rows = rows
        if self._op is None:
            self._op = _C.moe_sum_bf16x4

        self._op(input, self._output)
        return self._output
