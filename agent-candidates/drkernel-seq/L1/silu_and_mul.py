import torch
import torch.nn as nn
import triton
import triton.language as tl


# 1D, column-only tiling: each program processes a contiguous block of 'd' for one row.
@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 128},  num_warps=2, num_stages=2),
        triton.Config({'BLOCK': 128},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK': 256},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK': 256},  num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 512},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK': 512},  num_warps=8, num_stages=3),
        triton.Config({'BLOCK': 1024}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK': 1024}, num_warps=8, num_stages=4),
    ],
    key=['d'],  # tune primarily on column size
)
@triton.jit
def silu_and_mul_1d_kernel(
    out_ptr,       # *T_out, shape [B, d]
    x_ptr,         # *T_in,  shape [B, N] (we use columns 0:d and d:2d)
    B: tl.constexpr,
    d: tl.constexpr,
    stride_x_row: tl.constexpr,   # elements
    stride_out_row: tl.constexpr, # elements
    OUT_DTYPE: tl.constexpr,      # tl.float16 / tl.bfloat16 / tl.float32
    BLOCK: tl.constexpr,
):
    # Program ids
    pid_b = tl.program_id(0)   # row id
    pid_blk = tl.program_id(1) # block id along columns

    # Column offsets this program will handle
    offs = pid_blk * BLOCK + tl.arange(0, BLOCK)
    mask = offs < d

    # Base for this row
    x_row = x_ptr + pid_b * stride_x_row
    out_row = out_ptr + pid_b * stride_out_row

    # Pointers: a at x[b, offs], b at x[b, d + offs]
    a_ptrs = x_row + offs
    b_ptrs = x_row + d + offs

    # Load with cache hint (streaming); upcast to f32
    a = tl.load(a_ptrs, mask=mask, other=0.0, cache_modifier=".cg").to(tl.float32)
    b = tl.load(b_ptrs, mask=mask, other=0.0, cache_modifier=".cg").to(tl.float32)

    # silu(a) = a * sigmoid(a); sigmoid = 1 / (1 + exp(-a))
    s = 1.0 / (1.0 + tl.exp(-a))
    out32 = (a * s) * b

    # Cast and store
    out_vals = out32.to(OUT_DTYPE)
    out_ptrs = out_row + offs
    tl.store(out_ptrs, out_vals, mask=mask)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute silu(x[..., :d]) * x[..., d:] where d = x.shape[-1] // 2.

        - CUDA + 2D: uses Triton kernel (1D over columns, autotuned).
        - CPU or non-2D: falls back to pure PyTorch.
        """
        # CPU or non-CUDA fallback
        if not x.is_cuda:
            d = x.shape[-1] // 2
            return torch.nn.functional.silu(x[..., :d]) * x[..., d:]

        # Enforce 2D for this implementation
        if x.dim() != 2:
            raise ValueError(f"ModelNew expects a 2D tensor [B, N]; got shape {tuple(x.shape)}")

        B, N = x.shape
        if (N % 2) != 0:
            raise ValueError(f"Last dimension N must be even; got N={N}")

        d = N // 2

        # Make contiguous to simplify strides and enable vectorized access
        x = x.contiguous()

        # Allocate output
        out = torch.empty((B, d), dtype=x.dtype, device=x.device)

        # Strides in elements
        stride_x_row = x.stride(0)  # should be 'N' for contiguous
        stride_out_row = out.stride(0)  # should be 'd' for contiguous

        # Map dtype to Triton dtype
        if x.dtype == torch.float16:
            OUT_DTYPE = tl.float16
        elif x.dtype == torch.bfloat16:
            OUT_DTYPE = tl.bfloat16
        elif x.dtype == torch.float32:
            OUT_DTYPE = tl.float32
        else:
            raise TypeError(f"Unsupported dtype {x.dtype}; supported: float16, bfloat16, float32")

        # Launch grid: (rows, column blocks)
        def grid(meta):
            BLOCK = meta['BLOCK']
            return (B, triton.cdiv(d, BLOCK))

        silu_and_mul_1d_kernel[grid](
            out, x,
            B, d,
            stride_x_row, stride_out_row,
            OUT_DTYPE=OUT_DTYPE,
        )

        return out

SiluAndMul = ModelNew
