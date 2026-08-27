import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except Exception:
    _TRITON_AVAILABLE = False


# Autotuned, bandwidth-oriented elementwise ReLU kernel.
@triton.autotune(
    configs=[
        # Smaller blocks can help for small/medium N or certain GPUs
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4, num_stages=2),
        # Mid-to-large blocks
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8, num_stages=2),
        # Very large block for very large N (some GPUs like this)
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=8, num_stages=1),
    ],
    key=['N'],
)
@triton.jit
def _relu_kernel(x_ptr, y_ptr, N: tl.int32, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask, other=0)
    y = tl.maximum(x, 0)  # zero is cast to x's dtype
    tl.store(y_ptr + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        # No user params; autotune will pick good launch params per shape/GPU.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fallbacks for unsupported environments or dtypes
        if (not _TRITON_AVAILABLE) or (not x.is_cuda):
            return F.relu(x)
        if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            return F.relu(x)

        # Use input as-is if contiguous; else materialize a contiguous copy
        x_src = x if x.is_contiguous() else x.contiguous()
        y = torch.empty_like(x_src)

        N = x_src.numel()

        # Launch grid: one program per BLOCK_SIZE chunk
        def grid(meta):
            return (triton.cdiv(N, meta['BLOCK_SIZE']),)

        _relu_kernel[grid](x_src, y, N)
        return y

ReLU = ModelNew
