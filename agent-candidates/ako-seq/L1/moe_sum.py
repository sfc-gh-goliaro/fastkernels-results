"""Fused MoE sum kernel: reduces top-k expert outputs into final output.

Uses a custom vectorized streaming CUDA kernel (``moe_sum_fast.cu``).

The forward path is deliberately thin. At the captured ``[8, 4096]`` shape the
whole call is host-bound -- the GPU work is 200 KB -- so the two
``aten::slice`` calls behind ``self._output[:M, :D]`` (~1.5 us) and the
``input.view(M, topk, D)`` (~0.8 us) cost more than the kernel does. Both are
removed: output views are memoised per ``(M, D)`` in a dict, and the raw 2-D
``[M*topk, D]`` input goes straight to C++, which derives ``M`` from ``topk``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from fastkernels.infra.cuda_ext import lazy_op

_C = lazy_op("moe_sum_fast", "moe_sum_fast.cu")


class MoeSum(nn.Module):
    """Fused top-k reduction for MoE outputs."""

    def __init__(self):
        super().__init__()
        # Flat backing store, grown monotonically; ``_views`` memoises the
        # contiguous [M, D] view carved out of it for each shape seen.
        self._base = None
        self._rows = 0
        self._cols = 0
        self._views = {}

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
        M = input.shape[0] // topk
        D = input.shape[1]
        out = self._views.get((M, D))
        if out is None:
            out = self._make_view(M, D, input)
        _C.moe_sum(input, out, topk)
        return out

    def _make_view(self, M: int, D: int, input: torch.Tensor) -> torch.Tensor:
        base = self._base
        if (
            base is None
            or self._rows < M
            or self._cols < D
            or base.dtype != input.dtype
            or base.device != input.device
        ):
            self._rows = max(M, self._rows)
            self._cols = max(D, self._cols)
            self._base = torch.empty(
                self._rows * self._cols, device=input.device, dtype=input.dtype
            )
            self._views.clear()
        # Slicing the flat store keeps the view contiguous for any (M, D),
        # unlike a 2-D row/col slice of an over-wide buffer.
        out = self._base[: M * D].view(M, D)
        self._views[(M, D)] = out
        return out
