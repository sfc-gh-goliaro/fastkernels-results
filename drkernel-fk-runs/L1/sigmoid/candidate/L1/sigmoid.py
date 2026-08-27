import torch
import torch.nn as nn

import triton
import triton.language as tl


# Autotuned elementwise sigmoid kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512},  num_warps=2, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=2),
        # A variant with more stages in case of deeper pipelines
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=3),
    ],
    key=['N'],  # tune based on problem size
)
@triton.jit
def sigmoid_kernel(X_ptr, Y_ptr, N, BLOCK_SIZE: tl.constexpr):
    """
    Elementwise sigmoid: y = 1 / (1 + exp(-x))
    Processes a flattened 1D array for coalesced access.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    # Load; upcast to fp32 for math
    x = tl.load(X_ptr + offs, mask=mask, other=0.0)
    x32 = x.to(tl.float32)

    # Compute sigmoid in fp32
    y32 = 1.0 / (1.0 + tl.exp(-x32))

    # Store; Triton will cast to Y_ptr's element type
    tl.store(Y_ptr + offs, y32, mask=mask)


class _SigmoidTritonFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        # Use Triton on CUDA; fallback to torch otherwise
        if not x.is_cuda:
            return torch.sigmoid(x)

        if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            return torch.sigmoid(x)

        x_contig = x.contiguous()
        n = x_contig.numel()
        y = torch.empty_like(x_contig)

        # Grid: 1D over blocks
        def grid(meta):
            return (triton.cdiv(n, meta['BLOCK_SIZE']),)

        sigmoid_kernel[grid](
            x_contig, y, n,
            # Meta-parameters are provided by autotune configs
        )

        # Save input for backward
        ctx.save_for_backward(x_contig)
        return y.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors

        # CPU fallback
        if not x.is_cuda:
            s = torch.sigmoid(x)
            return grad_output * s * (1.0 - s)

        # CUDA: compute gradient in fp32 for stability, then cast
        go = grad_output.contiguous()
        x32 = x.to(torch.float32)
        go32 = go.to(torch.float32)
        s32 = 1.0 / (1.0 + torch.exp(-x32))
        grad32 = go32 * s32 * (1.0 - s32)
        grad = grad32.to(x.dtype)
        return grad


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _SigmoidTritonFn.apply(x)


# Keep Model for compatibility; use the same implementation.
class Model(ModelNew):
    pass

Sigmoid = ModelNew
