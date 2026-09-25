"""Triton log-sigmoid activation used by GLA's gk gate."""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _log_sigmoid_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    ax = tl.abs(x)
    # Cubic approximation of log1p(exp(-abs(x))) on [0, 4].
    correction = (
        ((-0.012287877 * ax + 0.130475713) * ax - 0.494606823) * ax
        + 0.691007886
    )
    correction = tl.where(ax < 4.0, correction, 0.0)
    y = tl.minimum(x, 0.0) - correction
    tl.store(out_ptr + offsets, y, mask=mask)


class LogSigmoid(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        n = x.numel()
        _log_sigmoid_kernel[(triton.cdiv(n, 1024),)](
            x, out, n, BLOCK=1024, num_warps=4
        )
        return out
