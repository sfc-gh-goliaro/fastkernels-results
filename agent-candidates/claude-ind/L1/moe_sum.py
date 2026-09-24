"""Fused MoE sum kernel: reduces top-k expert outputs into final output.

Memory-bound reduction -- reads ``topk * M * D`` elements and writes ``M * D``.
The kernel in ``moe_sum_fast.cu`` moves 16-byte vectors per thread with ``topk``
loads in flight and accumulates in fp32 (the same accumulation width as the
baseline's ``moe_sum_facc_kernel``, so results are bit-identical for topk=8).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from fastkernels.infra.cuda_ext import lazy_op

# A name distinct from the baseline's "moe_sum" extension: the JIT build
# directory is keyed by name, and both are loaded in the same process.
_C = lazy_op("moe_sum_vec16", "moe_sum_fast.cu")


class MoeSum(nn.Module):
    """Fused top-k reduction for MoE outputs."""

    def __init__(self):
        super().__init__()
        self._output = None
        self._view = None
        self._key = None

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
        D = input.size(1)
        M = input.size(0) // topk
        # The smallest captured shape (8x4096) is per-call-overhead bound, so the
        # buffer view is cached whole instead of re-sliced on every call.
        if self._key != (M, D):
            buf = self._output
            if buf is None or buf.size(0) < M or buf.size(1) < D:
                buf = self._output = torch.empty(
                    M, D, device=input.device, dtype=input.dtype)
            self._view = buf[:M, :D]
            self._key = (M, D)

        output = self._view
        _C.moe_sum(input, output, topk)
        return output
