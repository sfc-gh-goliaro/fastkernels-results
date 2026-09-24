import torch
import torch.nn as nn

import triton
import triton.language as tl


# Autotuned fused kernel: y = x * sigmoid(a * x)
# Compute in float32 for numerical stability; cast back to input dtype.
@triton.autotune(
    configs=[
        # Smaller blocks for small N; help occupancy on some GPUs
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4, num_stages=2),
        # Mid-to-large blocks that tend to work well for N ~ 1e5–1e6
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=2),
        # Slightly deeper pipeline; may help on一些 GPUs
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=3),
    ],
    key=['N'],  # tune based on problem size
)
@triton.jit
def quickgelu_kernel(X_ptr, Y_ptr, N, a, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    # Load; upcast to float32 for compute
    x = tl.load(X_ptr + offs, mask=mask, other=0)
    x32 = x.to(tl.float32)

    # z = a * x
    z = x32 * a

    # sigmoid(z) = 1 / (1 + exp(-z))
    one = 1.0
    s = one / (one + tl.exp(-z))

    # y = x * sigmoid(z)
    y32 = x32 * s

    # Cast back to original dtype and store
    y = y32.to(x.dtype)
    tl.store(Y_ptr + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, a: float = 1.702):
        super().__init__()
        self.a = float(a)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fallback to PyTorch if tensor is on CPU
        if not x.is_cuda:
            return x * torch.sigmoid(self.a * x)

        # Ensure contiguous for coalesced access
        x_contig = x.contiguous()
        y = torch.empty_like(x_contig)

        N = x_contig.numel()

        # 1D grid: one program per BLOCK_SIZE chunk
        def grid(meta):
            return (triton.cdiv(N, meta['BLOCK_SIZE']),)

        # Launch autotuned kernel
        quickgelu_kernel[grid](
            x_contig, y, N,
            self.a,
        )

        return y

QuickGELU = ModelNew
