"""L2 normalization along a single dimension: x / ||x||_2.

Fused single-pass Triton kernel. Each program owns one row of the flattened
[M, N] view and does exactly one masked load and one masked store: the sum of
squares is accumulated in fp32 registers, so the row never round-trips to
memory. Semantics match ``F.normalize(x, p=2, dim=...)`` exactly -- the divisor
is ``clamp_min(||x||_2, eps)``, not ``sqrt(sumsq + eps)``, so an all-zero row
normalizes to zeros rather than to NaN.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _l2_norm_row_kernel(
    x_ptr,
    out_ptr,
    n_cols,
    row_stride,
    eps,
    BLOCK_N: tl.constexpr,
):
    """One program per row: load, reduce, scale, store."""
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < n_cols
    offs = row * row_stride + cols

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    xf = x.to(tl.float32)
    sumsq = tl.sum(xf * xf, axis=0)
    norm = tl.sqrt(sumsq)
    # clamp_min(norm, eps), written as a select rather than tl.maximum so that a
    # NaN row propagates: CUDA's max.f32 is NaN-quieting and would hand back eps,
    # turning a NaN row into finite garbage. `norm < eps` is False for NaN, so
    # the select keeps NaN -- which is what torch.clamp_min does.
    denom = tl.where(norm < eps, eps, norm)
    # Zero rows: 0 * (1/eps) = 0, matching F.normalize rather than NaN.
    inv = 1.0 / denom
    tl.store(out_ptr + offs, xf * inv, mask=mask)


# Largest reduction width we are willing to hold in registers as a single tile.
_MAX_BLOCK_N = 8192


class L2Norm(nn.Module):
    def __init__(self, dim: int = -1, eps: float = 1e-12):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dim = self.dim
        ndim = x.ndim
        # Only the fused last-dim reduction over contiguous fp32 is specialised;
        # everything else defers to eager so semantics stay identical.
        if (
            ndim >= 1
            and (dim == -1 or dim == ndim - 1)
            and x.dtype == torch.float32
            and x.is_contiguous()
            and x.is_cuda
        ):
            n = x.shape[-1]
            if 0 < n <= _MAX_BLOCK_N:
                m = x.numel() // n
                if m > 0:
                    out = torch.empty_like(x)
                    _l2_norm_row_kernel[(m,)](
                        x,
                        out,
                        n,
                        n,
                        self.eps,
                        BLOCK_N=triton.next_power_of_2(n),
                        num_warps=4,
                    )
                    return out
        return F.normalize(x, p=2.0, dim=dim, eps=self.eps)
