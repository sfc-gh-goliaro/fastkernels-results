import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False


@triton.jit
def _l2norm_kernel(
    x_ptr,                 # *const T
    y_ptr,                 # *T
    M: tl.constexpr,       # number of rows
    K: tl.constexpr,       # number of cols (normalize dim size)
    stride_xm,             # int: stride for dim 0 (row) in elements
    stride_xk,             # int: stride for dim 1 (col) in elements
    stride_ym,             # int: stride for dim 0 (row) in elements
    stride_yk,             # int: stride for dim 1 (col) in elements
    eps,                   # float
    BLOCK_K: tl.constexpr  # block size along K
):
    m = tl.program_id(0)
    if m >= M:
        return

    # Single-pass fast path if K <= BLOCK_K: load once, compute, store.
    if K <= BLOCK_K:
        idx = tl.arange(0, BLOCK_K)
        mask = idx < K
        x = tl.load(x_ptr + m * stride_xm + idx * stride_xk, mask=mask, other=0.0)
        xf = x.to(tl.float32)
        # zero out invalid lanes before square (mask ensures OOB are zero)
        xf = tl.where(mask, xf, 0.0)
        acc = tl.sum(xf * xf, axis=0)
        norm = tl.sqrt(acc)
        denom = tl.maximum(norm, eps)
        y = (xf / denom).to(x.dtype)
        tl.store(y_ptr + m * stride_ym + idx * stride_yk, y, mask=mask)
        return

    # Fallback: two-pass for K > BLOCK_K
    acc = tl.zeros((), dtype=tl.float32)
    off = 0
    while off < K:
        idx = off + tl.arange(0, BLOCK_K)
        mask = idx < K
        x = tl.load(x_ptr + m * stride_xm + idx * stride_xk, mask=mask, other=0.0)
        xf = x.to(tl.float32)
        xf = tl.where(mask, xf, 0.0)
        acc += tl.sum(xf * xf, axis=0)
        off += BLOCK_K

    norm = tl.sqrt(acc)
    denom = tl.maximum(norm, eps)

    off = 0
    while off < K:
        idx = off + tl.arange(0, BLOCK_K)
        mask = idx < K
        x = tl.load(x_ptr + m * stride_xm + idx * stride_xk, mask=mask, other=0.0)
        xf = x.to(tl.float32)
        y = (xf / denom).to(x.dtype)
        tl.store(y_ptr + m * stride_ym + idx * stride_yk, y, mask=mask)
        off += BLOCK_K


class ModelNew(nn.Module):
    def __init__(self, dim: int = -1, eps: float = 1e-12):
        super().__init__()
        self.dim = dim
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fallback to PyTorch if Triton not available or tensor not on CUDA
        if (not _HAS_TRITON) or (not x.is_cuda):
            return F.normalize(x, p=2.0, dim=self._normalize_dim(x.dim(), self.dim), eps=self.eps)

        # Normalize dim to non-negative
        dim = self._normalize_dim(x.dim(), self.dim)

        # Move normalization dimension to the last, and make contiguous
        x_m = x.moveaxis(dim, -1).contiguous()
        *prefix, K = x_m.shape
        M = 1
        for s in prefix:
            M *= s
        # View as [M, K]
        x_2d = x_m.view(M, K)

        # Allocate output
        y_2d = torch.empty_like(x_2d)

        # Compute strides in elements
        stride_xm = x_2d.stride(0)
        stride_xk = x_2d.stride(1)
        stride_ym = y_2d.stride(0)
        stride_yk = y_2d.stride(1)

        # Choose BLOCK_K = K for single-pass fast path (K is small in provided shapes).
        block_k = K

        # num_warps heuristic based on block size
        if block_k <= 256:
            num_warps = 2
        elif block_k <= 1024:
            num_warps = 4
        else:
            num_warps = 8

        # Launch grid: one program per row
        grid = (M,)

        _l2norm_kernel[grid](
            x_2d, y_2d,
            M, K,
            stride_xm, stride_xk,
            stride_ym, stride_yk,
            self.eps,
            BLOCK_K=block_k,
            num_warps=num_warps,
            num_stages=1,  # simple kernel; 1 stage is sufficient
        )

        # Restore original shape and dimension order
        y_m = y_2d.view(*prefix, K)
        y = y_m.moveaxis(-1, dim)
        return y

    @staticmethod
    def _normalize_dim(ndim: int, dim: int) -> int:
        return dim if dim >= 0 else dim + ndim

L2Norm = ModelNew
