import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


def _next_power_of_2(n: int) -> int:
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


def _choose_block_n(D: int, max_block: int = 1024) -> int:
    # Use D itself when it's <= max_block
    return D if D <= max_block else min(max_block, _next_power_of_2(max_block))


def _choose_block_m(M: int) -> int:
    # Choose a small power-of-two rows per program to balance register pressure and work
    if M >= 1 << 18:       # >= 262144
        return 32
    elif M >= 1 << 14:     # >= 16384
        return 16
    elif M >= 1 << 12:     # >= 4096
        return 8
    else:
        return 4


@triton.jit
def _rmsnorm_gate_after_rows_kernel(
    X,       # *ptr, shape [M, N]
    Z,       # *ptr, shape [M, N]
    W,       # *ptr, shape [N]
    B,       # *ptr, shape [N] or dummy
    Out,     # *ptr, shape [M, N]
    stride_x_row,
    stride_out_row,
    M: tl.constexpr,      # rows
    N: tl.constexpr,      # cols
    eps: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    # Program processes BLOCK_M rows starting at row_start
    pid = tl.program_id(0)
    row_start = pid * BLOCK_M

    rows = row_start + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)

    row_mask = rows < M
    col_mask = cols < N
    mask2d = row_mask[:, None] & col_mask[None, :]

    # Load X and Z tiles; compute in float32
    x = tl.load(X + rows[:, None] * stride_x_row + cols[None, :],
                mask=mask2d, other=0.0).to(tl.float32)
    z = tl.load(Z + rows[:, None] * stride_x_row + cols[None, :],
                mask=mask2d, other=0.0).to(tl.float32)

    # var per row: mean(x^2) over N
    x2 = x * x
    var = tl.sum(x2, axis=1) / N
    rstd = 1.0 / tl.sqrt(var + eps)  # shape [BLOCK_M]

    # Normalize
    x_hat = x * rstd[:, None]

    # Weight [N]
    w = tl.load(W + cols, mask=col_mask, other=0.0).to(tl.float32)
    # Broadcast weight over rows
    y = x_hat * w[None, :]

    # Bias if any
    if HAS_BIAS:
        b = tl.load(B + cols, mask=col_mask, other=0.0).to(tl.float32)
        y = y + b[None, :]

    # Gate after norm
    if ACTIVATION == "swish" or ACTIVATION == "silu":
        s = 1.0 / (1.0 + tl.exp(-z))
        y = y * (z * s)
    elif ACTIVATION == "sigmoid":
        s = 1.0 / (1.0 + tl.exp(-z))
        y = y * s

    # Store
    tl.store(Out + rows[:, None] * stride_out_row + cols[None, :],
             y, mask=mask2d)


# Fallback two-pass kernel if N > BLOCK_N (not expected for given shapes, but kept for completeness)
@triton.jit
def _rmsnorm_gate_after_rows_twopass_kernel(
    X, Z, W, B, Out,
    stride_x_row, stride_out_row,
    M: tl.constexpr, N: tl.constexpr, eps: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr,
    HAS_BIAS: tl.constexpr, ACTIVATION: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * BLOCK_M
    rows = row_start + tl.arange(0, BLOCK_M)
    row_mask = rows < M

    n_iters = (N + BLOCK_N - 1) // BLOCK_N

    # Pass 1: compute var per row
    var = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for k in range(0, n_iters):
        cols = k * BLOCK_N + tl.arange(0, BLOCK_N)
        col_mask = cols < N
        # loop over rows in tile
        for m in range(0, BLOCK_M):
            r = rows[m]
            if row_mask[m]:
                x = tl.load(X + r * stride_x_row + cols, mask=col_mask, other=0.0).to(tl.float32)
                var[m] += tl.sum(x * x, axis=0)
    var = var / N

    rstd = 1.0 / tl.sqrt(var + eps)

    # Pass 2: normalize, scale, bias, gate, store
    for k in range(0, n_iters):
        cols = k * BLOCK_N + tl.arange(0, BLOCK_N)
        col_mask = cols < N
        for m in range(0, BLOCK_M):
            r = rows[m]
            if row_mask[m]:
                x = tl.load(X + r * stride_x_row + cols, mask=col_mask, other=0.0).to(tl.float32)
                x_hat = x * rstd[m]
                w = tl.load(W + cols, mask=col_mask, other=0.0).to(tl.float32)
                y = x_hat * w
                if HAS_BIAS:
                    b = tl.load(B + cols, mask=col_mask, other=0.0).to(tl.float32)
                    y = y + b
                z = tl.load(Z + r * stride_x_row + cols, mask=col_mask, other=0.0).to(tl.float32)
                if ACTIVATION == "swish" or ACTIVATION == "silu":
                    s = 1.0 / (1.0 + tl.exp(-z))
                    y = y * (z * s)
                elif ACTIVATION == "sigmoid":
                    s = 1.0 / (1.0 + tl.exp(-z))
                    y = y * s
                tl.store(Out + r * stride_out_row + cols, y, mask=col_mask)


def _launch_rmsnorm_gate_after(x: torch.Tensor,
                               z: torch.Tensor,
                               weight: torch.Tensor,
                               bias: torch.Tensor | None,
                               eps: float,
                               out: torch.Tensor,
                               activation: str):
    assert x.is_cuda and z.is_cuda and out.is_cuda, "Triton kernel requires CUDA tensors"
    assert x.dtype == z.dtype == out.dtype, "Expect x, z, out same dtype"
    assert x.shape == z.shape, "x and z must have same shape"
    assert x.dim() == 2, "Expect 2D [M, N] tensors"
    M, N = x.shape

    # Ensure contiguous last dim
    if x.stride(-1) != 1:
        x = x.contiguous()
    if z.stride(-1) != 1:
        z = z.contiguous()
    out_c = out
    if out.stride(-1) != 1:
        out_c = out.contiguous()

    # weight, bias
    w = weight
    b = bias
    if w.stride(-1) != 1:
        w = w.contiguous()
    if b is not None and b.stride(-1) != 1:
        b = b.contiguous()

    HAS_BIAS = b is not None
    ACT = activation
    if ACT not in ("swish", "silu", "sigmoid"):
        raise ValueError(f"Unsupported activation: {ACT}; supported: swish, silu, sigmoid")

    # Choose blocks
    BLOCK_N = _choose_block_n(N, max_block=1024)
    BLOCK_M = _choose_block_m(M)

    grid = (triton.cdiv(M, BLOCK_M),)

    num_warps = 4 if BLOCK_N >= 128 else 2
    # Slightly more stages can help with register file
    num_stages = 3

    if N <= BLOCK_N:
        _rmsnorm_gate_after_rows_kernel[grid](
            x, z, w, (b if HAS_BIAS else w),
            out_c,
            x.stride(0), out_c.stride(0),
            M, N,
            eps,
            BLOCK_N=BLOCK_N, BLOCK_M=BLOCK_M,
            HAS_BIAS=HAS_BIAS, ACTIVATION=ACT,
            num_warps=num_warps, num_stages=num_stages,
        )
    else:
        # Fallback two-pass if needed
        _rmsnorm_gate_after_rows_twopass_kernel[grid](
            x, z, w, (b if HAS_BIAS else w),
            out_c,
            x.stride(0), out_c.stride(0),
            M, N,
            eps,
            BLOCK_N=_next_power_of_2(min(256, N)), BLOCK_M=BLOCK_M,
            HAS_BIAS=HAS_BIAS, ACTIVATION=ACT,
            num_warps=num_warps, num_stages=num_stages,
        )

    # If out was non-contiguous, copy back
    if out_c.data_ptr() != out.data_ptr():
        out.copy_(out_c)


class ModelNew(nn.Module):
    """
    Triton-optimized version of Model: Fused RMSNorm + elementwise gate (after norm).

    Signature preserved:
      __init__(hidden_size: int, eps: float = 1e-6,
               norm_before_gate: bool = True, activation: str = "swish")
    Forward:
      forward(x: [M, N], z: [M, N]) -> out: [M, N]
    Notes:
      - Implements norm_before_gate=True (matches usage).
      - activation in {"swish","silu","sigmoid"}.
      - Inference-only; no backward.
    """
    def __init__(self, hidden_size: int, eps: float = 1e-6,
                 norm_before_gate: bool = True, activation: str = "swish"):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.norm_before_gate = norm_before_gate  # we only implement after-norm gating
        if activation not in ("swish", "silu", "sigmoid"):
            raise ValueError(f"Unsupported activation: {activation}; supported: swish, silu, sigmoid")
        self.activation = activation
        # elementwise affine
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if not self.norm_before_gate:
            # Fallback: PyTorch implementation of before-norm gate (not fused).
            g = torch.sigmoid(z) if self.activation == "sigmoid" else (z * torch.sigmoid(z))
            xg = x * g
            var = (xg.float() ** 2).mean(dim=-1, keepdim=True)
            rstd = torch.rsqrt(var + self.eps)
            y = xg.float() * rstd
            y = y * self.weight.float()
            if self.bias is not None:
                y = y + self.bias.float()
            return y.to(x.dtype)

        out = torch.empty_like(x)
        _launch_rmsnorm_gate_after(
            x, z, self.weight, self.bias,
            self.eps, out, self.activation
        )
        return out

RMSNormGated = ModelNew
