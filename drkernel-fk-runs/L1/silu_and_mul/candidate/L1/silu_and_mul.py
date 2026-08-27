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


@triton.autotune(
    configs=[
        # Focus on BLOCK_M=1 for column-wise contiguous access; vary BLOCK_N and warps
        triton.Config({'BLOCK_N': 128,  'BLOCK_M': 1}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_N': 256,  'BLOCK_M': 1}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 512,  'BLOCK_M': 1}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 1024, 'BLOCK_M': 1}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 2048, 'BLOCK_M': 1}, num_warps=8, num_stages=2),

        # A few BM>1 to cover small-R regimes
        triton.Config({'BLOCK_N': 256,  'BLOCK_M': 4}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 512,  'BLOCK_M': 4}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 1024, 'BLOCK_M': 4}, num_warps=8, num_stages=2),

        triton.Config({'BLOCK_N': 512,  'BLOCK_M': 8}, num_warps=8, num_stages=2),
    ],
    key=['R', 'd'],  # tune based on problem size
)
@triton.jit
def _silu_and_mul_2d_kernel(
    out_ptr, x_ptr,
    R: tl.constexpr,       # number of rows (flattened leading dims)
    D: tl.constexpr,       # last dimension size
    d: tl.constexpr,       # output dimension size (= D//2)
    BLOCK_N: tl.constexpr, # columns per program
    BLOCK_M: tl.constexpr, # rows per program
):
    # 2D program ids
    pid_m = tl.program_id(0)  # over rows
    pid_n = tl.program_id(1)  # over cols

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Masks for bounds
    mask_rows = rows < R
    mask_cols = cols < d
    mask = mask_rows[:, None] & mask_cols[None, :]

    # x is viewed as [R, D]; out as [R, d]
    ptr_x1 = x_ptr + rows[:, None] * D + cols[None, :]
    ptr_x2 = x_ptr + rows[:, None] * D + (cols[None, :] + d)
    ptr_out = out_ptr + rows[:, None] * d + cols[None, :]

    # Load
    x1 = tl.load(ptr_x1, mask=mask, other=0.0)
    x2 = tl.load(ptr_x2, mask=mask, other=0.0)

    # Upcast to float32 for math
    x1_f = x1.to(tl.float32)
    x2_f = x2.to(tl.float32)

    # Sigmoid and SiLU: silu(x) = x / (1 + exp(-x))
    # implemented as x * sigmoid(x) with sigmoid = 1 / (1 + exp(-x))
    sig = 1.0 / (1.0 + tl.exp(-x1_f))
    silu = x1_f * sig

    out_f = silu * x2_f
    out = out_f.to(x1.dtype)

    # Store
    tl.store(ptr_out, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward_native(x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        return F.silu(x[..., :d]) * x[..., d:]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fallback conditions
        if (not _HAS_TRITON) or (not x.is_cuda) or (x.dtype not in (torch.float32, torch.bfloat16)):
            return self.forward_native(x)

        # Ensure contiguous
        if not x.is_contiguous():
            x = x.contiguous()

        # Shapes
        *prefix, D = x.shape
        d = D // 2
        R = int(x.numel() // D)  # flatten all leading dims

        # Allocate output
        out = torch.empty((*prefix, d), dtype=x.dtype, device=x.device)

        # 2D grid over (rows, cols); BLOCK sizes chosen by autotune
        def grid(meta):
            BN = meta['BLOCK_N']
            BM = meta['BLOCK_M']
            return (triton.cdiv(R, BM), triton.cdiv(d, BN))

        _silu_and_mul_2d_kernel[grid](
            out, x,
            R=R, D=D, d=d,
        )

        return out

SiluAndMul = ModelNew
