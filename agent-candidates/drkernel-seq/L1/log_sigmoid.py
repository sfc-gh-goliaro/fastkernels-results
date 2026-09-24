import torch
import torch.nn as nn

import triton
import triton.language as tl


# Autotuned kernel over a set of sensible configs for elementwise ops.
@triton.autotune(
    configs=[
        # Small/Narrow problems
        triton.Config({'BLOCK_SIZE': 256},  num_warps=2, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=2, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=2),
        # Medium/Large problems
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=3),
        # Very large problems: more stages can help hide latency
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8, num_stages=4),
    ],
    key=['N'],  # problem size drives the best config
)
@triton.jit
def _logsigmoid_kernel_fp32math(x_ptr, y_ptr, N, BLOCK_SIZE: tl.constexpr):
    """
    Compute y = logsigmoid(x) elementwise in a numerically stable way.
    - Load as input dtype, upcast to float32 for math.
    - Use y = -(log(1 + exp(-x))) in float32.
    - Cast back to input dtype and store.
    1D grid over N elements.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    # Load and upcast to fp32 for stable math
    x = tl.load(x_ptr + offs, mask=mask, other=0)
    x32 = x.to(tl.float32)

    # Stable computation: exp(-x) in (0,1], no overflow; then log(1 + e)
    e = tl.exp(-x32)
    y32 = -tl.log(1.0 + e)

    # Cast back and store
    y = y32.to(x.dtype)
    tl.store(y_ptr + offs, y, mask=mask)


def _launch_logsigmoid(x: torch.Tensor) -> torch.Tensor:
    """
    Launch the Triton kernel to compute logsigmoid(x).
    - Upcasts compute to fp32 inside kernel.
    - Supports float16, bfloat16, float32 inputs; output matches input dtype.
    Requires CUDA tensor.
    """
    assert x.is_cuda, f"Expected CUDA tensor, got {x.device}"
    # Ensure contiguous for simple 1D addressing
    x_contig = x.contiguous()
    y = torch.empty_like(x_contig)

    N = x_contig.numel()

    # Grid: one program per BLOCK_SIZE chunk; autotune will set BLOCK_SIZE
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)

    _logsigmoid_kernel_fp32math[grid](
        x_contig, y, N,
    )
    return y.view_as(x)


class ModelNew(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CPU fallback
        if not x.is_cuda:
            return torch.nn.functional.logsigmoid(x)
        # CUDA path: Triton kernel
        return _launch_logsigmoid(x)

LogSigmoid = ModelNew
