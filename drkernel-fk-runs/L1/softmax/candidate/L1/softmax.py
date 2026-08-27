import math
import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


def _next_power_of_2(x: int) -> int:
    return 1 if x <= 1 else 1 << (x - 1).bit_length()


# One-pass softmax: assumes BLOCK_N >= N. Loads the whole row once.
@triton.jit
def _softmax_onepass_kernel(
    X, Y,
    M, N,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    OUT_IS_FP16: tl.constexpr,
    OUT_IS_BF16: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N

    # Load and upcast to fp32
    x = tl.load(X + row * stride_xm + offs * stride_xn, mask=mask, other=-float('inf'))
    x = x.to(tl.float32)

    # Max
    m = tl.max(x, axis=0)
    # Shift, exp, sum
    z = x - m
    numer = tl.exp(z)
    denom = tl.sum(numer, axis=0)
    y = numer / denom

    # Cast to output dtype
    if OUT_IS_FP16:
        y = y.to(tl.float16)
    elif OUT_IS_BF16:
        y = y.to(tl.bfloat16)

    # Store
    tl.store(Y + row * stride_ym + offs * stride_yn, y, mask=mask)


# Three-pass softmax for arbitrary N: max -> sum(exp) -> write
@triton.jit
def _softmax_three_pass_kernel(
    X, Y,
    M, N,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    OUT_IS_FP16: tl.constexpr,
    OUT_IS_BF16: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)

    # Pass 1: max
    m = -float('inf')
    col = 0
    while col < N:
        idx = col + offs
        mask = idx < N
        x = tl.load(X + row * stride_xm + idx * stride_xn, mask=mask, other=-float('inf')).to(tl.float32)
        local_max = tl.max(x, axis=0)
        m = tl.maximum(m, local_max)
        col += BLOCK_N

    # Pass 2: sum of exp(x - m)
    denom = 0.0
    col = 0
    while col < N:
        idx = col + offs
        mask = idx < N
        x = tl.load(X + row * stride_xm + idx * stride_xn, mask=mask, other=-float('inf')).to(tl.float32)
        denom += tl.sum(tl.exp(x - m), axis=0)
        col += BLOCK_N

    inv_denom = 1.0 / denom

    # Pass 3: write
    col = 0
    while col < N:
        idx = col + offs
        mask = idx < N
        x = tl.load(X + row * stride_xm + idx * stride_xn, mask=mask, other=-float('inf')).to(tl.float32)
        y = tl.exp(x - m) * inv_denom

        if OUT_IS_FP16:
            y = y.to(tl.float16)
        elif OUT_IS_BF16:
            y = y.to(tl.bfloat16)

        tl.store(Y + row * stride_ym + offs * stride_yn, y, mask=mask)
        col += BLOCK_N


class ModelNew(nn.Module):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fallbacks
        if (not TRITON_AVAILABLE) or (not x.is_cuda):
            return torch.softmax(x, dim=self.dim)

        # Normalize and move axis to last
        dim = self.dim
        if dim < 0:
            dim = dim + x.ndim
        assert 0 <= dim < x.ndim, f"Invalid dim={self.dim} for shape {tuple(x.shape)}"

        x_m = x.moveaxis(dim, -1).contiguous()
        *outer, N = x_m.shape
        M = 1
        for d in outer:
            M *= d
        x2 = x_m.view(M, N)

        # Output (same dtype as input)
        y2 = torch.empty_like(x2, dtype=x2.dtype)

        # Strides in elements
        stride_xm = x2.stride(0)
        stride_xn = x2.stride(1)
        stride_ym = y2.stride(0)
        stride_yn = y2.stride(1)

        # Output dtype flags for kernel
        out_is_fp16 = (x2.dtype == torch.float16)
        out_is_bf16 = (x2.dtype == torch.bfloat16)

        grid = (M,)

        # Choose BLOCK_N >= N, clamped
        BLOCK_N = _next_power_of_2(N)
        MAX_BLOCK = 4096
        if BLOCK_N > MAX_BLOCK:
            BLOCK_N = MAX_BLOCK

        # Heuristic for num_warps
        if BLOCK_N <= 128:
            num_warps = 4
        elif BLOCK_N <= 512:
            num_warps = 4
        elif BLOCK_N <= 1024:
            num_warps = 8
        else:
            num_warps = 8

        if N <= BLOCK_N:
            # One-pass: guarantees coverage since BLOCK_N >= N
            _softmax_onepass_kernel[grid](
                x2, y2,
                M, N,
                stride_xm, stride_xn,
                stride_ym, stride_yn,
                OUT_IS_FP16=out_is_fp16,
                OUT_IS_BF16=out_is_bf16,
                BLOCK_N=BLOCK_N,
                num_warps=num_warps,
                num_stages=2,
            )
        else:
            # Should not happen given BLOCK_N selection, but keep as safeguard.
            # Use a conservative BLOCK_N tile and loop (three-pass).
            _softmax_three_pass_kernel[grid](
                x2, y2,
                M, N,
                stride_xm, stride_xn,
                stride_ym, stride_yn,
                OUT_IS_FP16=out_is_fp16,
                OUT_IS_BF16=out_is_bf16,
                BLOCK_N=1024,
                num_warps=8,
                num_stages=2,
            )

        # Restore shape and axis
        y_m = y2.view(*outer, N)
        y = y_m.moveaxis(-1, dim)
        return y

Softmax = ModelNew
