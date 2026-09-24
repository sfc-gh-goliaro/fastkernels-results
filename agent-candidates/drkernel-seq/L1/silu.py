import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except Exception:
    _TRITON_AVAILABLE = False


# Autotuned elementwise SiLU kernel.
# - Flattened 1D processing
# - Always compute in fp32 for numerical stability, then cast back to input dtype
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256},  num_warps=2, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=2, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=3),
    ],
    key=['N'],
)
@triton.jit
def _silu_kernel(x_ptr, y_ptr, N: tl.int32,
                 BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    # Load input; upcast to fp32 for computation
    x = tl.load(x_ptr + offs, mask=mask, other=0)
    x32 = x.to(tl.float32)

    # sigmoid(x) = 1 / (1 + exp(-x)); compute in fp32
    s = 1.0 / (1.0 + tl.exp(-x32))
    y32 = x32 * s

    # Cast back to original dtype and store
    y = y32.to(x.dtype)
    tl.store(y_ptr + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fallback to PyTorch if Triton not available or tensor not on CUDA
        if (not _TRITON_AVAILABLE) or (not x.is_cuda):
            return F.silu(x)

        # Ensure contiguous for coalesced access
        if not x.is_contiguous():
            x = x.contiguous()

        # Allocate output
        y = torch.empty_like(x)

        # Flatten to 1D
        N = x.numel()

        # Grid: 1D over blocks; autotune will supply BLOCK_SIZE
        def grid(meta):
            return (triton.cdiv(N, meta['BLOCK_SIZE']),)

        _silu_kernel[grid](
            x, y, N,
        )

        return y

SiLU = ModelNew
