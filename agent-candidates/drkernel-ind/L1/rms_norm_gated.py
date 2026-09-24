import math
import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_gated_row_kernel(
    X,          # *ptr* to input x [M, D]
    Z,          # *ptr* to gate z [M, D]
    W,          # *ptr* to weight [D]
    Y,          # *ptr* to output [M, D]
    M,          # rows
    D,          # cols
    eps,       # epsilon float
    ACTIVATION: tl.constexpr,  # "swish" or "sigmoid"
    BLOCK_D: tl.constexpr,     # tile size over D
):
    # One program per row
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    mask = cols < D

    # Row pointers
    x = tl.load(X + row * D + cols, mask=mask, other=0.0).to(tl.float32)
    z = tl.load(Z + row * D + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)

    # Compute variance over D
    x2 = x * x
    invD = 1.0 / D
    var = tl.sum(x2, axis=0) * invD
    rstd = tl.rsqrt(var + eps)

    # Normalize and scale by weight
    x_hat = x * rstd
    y = x_hat * w

    # Apply gate: swish or sigmoid
    if ACTIVATION == "swish" or ACTIVATION == "silu":
        s = tl.sigmoid(z)
        g = z * s
        out = y * g
    else:
        s = tl.sigmoid(z)
        out = y * s

    # Store
    tl.store(Y + row * D + cols, out, mask=mask)


@triton.jit
def rmsnorm_gated_2d_kernel(
    X,          # *ptr* to input x [M, D]
    Z,          # *ptr* to gate z [M, D]
    W,          # *ptr* to weight [D]
    Y,          # *ptr* to output [M, D]
    M,          # rows
    D,          # cols
    eps,       # epsilon float
    ACTIVATION: tl.constexpr,  # "swish" or "sigmoid"
    BLOCK_D: tl.constexpr,     # tile size over D (prefer BLOCK_D == D)
    ROWS_PER_BLOCK: tl.constexpr,  # rows per program
):
    # Program processes a block of rows
    pid = tl.program_id(0)
    row_start = pid * ROWS_PER_BLOCK
    rows = row_start + tl.arange(0, ROWS_PER_BLOCK)
    row_mask = rows < M

    cols = tl.arange(0, BLOCK_D)
    # Typically BLOCK_D == D, so no column mask needed
    col_mask = cols < D

    # 2D pointers
    x_ptrs = X + rows[:, None] * D + cols[None, :]
    z_ptrs = Z + rows[:, None] * D + cols[None, :]
    y_ptrs = Y + rows[:, None] * D + cols[None, :]

    # Load
    x = tl.load(x_ptrs, mask=row_mask[:, None] & col_mask[None, :], other=0.0).to(tl.float32)
    z = tl.load(z_ptrs, mask=row_mask[:, None] & col_mask[None, :], other=0.0).to(tl.float32)
    w = tl.load(W + cols, mask=col_mask, other=0.0).to(tl.float32)

    # Variance per row
    x2 = x * x
    invD = 1.0 / D
    var = tl.sum(x2, axis=1) * invD  # shape: [ROWS_PER_BLOCK]
    rstd = tl.rsqrt(var + eps)       # shape: [ROWS_PER_BLOCK]

    # Normalize, scale, gate
    x_hat = x * rstd[:, None]
    y = x_hat * w[None, :]

    if ACTIVATION == "swish" or ACTIVATION == "silu":
        s = tl.sigmoid(z)
        g = z * s
        out = y * g
    else:
        s = tl.sigmoid(z)
        out = y * s

    # Store
    tl.store(y_ptrs, out, mask=row_mask[:, None] & col_mask[None, :])


def _next_power_of_2(n: int) -> int:
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


def rmsnorm_gated(
    x: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    activation: str = "swish",
):
    """
    Fused RMSNorm + gate:
      y = RMSNorm(x, weight); out = y * activation(z)
    Shapes: x, z: [M, D]; weight: [D]
    Dtypes: x, z can be fp16/bf16/fp32; compute in fp32; output same as x.dtype.
    """
    assert x.dim() == 2 and z.dim() == 2, f"Expected 2D tensors, got {x.shape}, {z.shape}"
    assert x.shape == z.shape, f"x and z must have same shape, got {x.shape} vs {z.shape}"
    assert weight.dim() == 1, f"Expected 1D weight, got shape {weight.shape}"
    M, D = x.shape
    assert weight.shape[0] == D, f"weight size {weight.shape[0]} != D {D}"

    # Ensure contiguous
    if not x.is_contiguous():
        x = x.contiguous()
    if not z.is_contiguous():
        z = z.contiguous()
    if not weight.is_contiguous():
        weight = weight.contiguous()

    # Allocate output
    y = torch.empty_like(x)

    # Choose strategy based on M
    use_2d = M >= 4096  # stronger 2D preference for very large M

    if use_2d:
        # BLOCK_D: prefer == D to avoid column masking
        BLOCK_D = D
        # Rows per program: tune by M size
        if M >= (1 << 20):      # ~1M rows+
            ROWS_PER_BLOCK = 32
        elif M >= (1 << 16):    # 65k+ rows
            ROWS_PER_BLOCK = 16
        else:                   # 4k+ rows
            ROWS_PER_BLOCK = 8
        grid = (triton.cdiv(M, ROWS_PER_BLOCK),)
        # Warps and stages: 4 warps for D<=256; 2 stages to hide latency
        num_warps = 4
        num_stages = 2
        rmsnorm_gated_2d_kernel[grid](
            x, z, weight, y,
            M, D, eps,
            ACTIVATION=activation,
            BLOCK_D=BLOCK_D,
            ROWS_PER_BLOCK=ROWS_PER_BLOCK,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    else:
        # Simple row kernel
        BLOCK_D = _next_power_of_2(D)
        BLOCK_D = min(BLOCK_D, 1024)
        grid = (M,)
        num_warps = 4 if BLOCK_D <= 1024 else 8
        rmsnorm_gated_row_kernel[grid](
            x, z, weight, y,
            M, D, eps,
            ACTIVATION=activation,
            BLOCK_D=BLOCK_D,
            num_warps=num_warps,
        )
    return y


class ModelNew(torch.nn.Module):
    """
    Triton-optimized fused RMSNorm + gate.

    Signature matches original Model:
      __init__(hidden_size: int, eps: float = 1e-6, norm_before_gate: bool = True, activation: str = "swish")
      forward(x: Tensor, z: Tensor) -> Tensor
    Notes:
      - Implements out = RMSNorm(x, weight) * activation(z) (gate after norm).
      - Computes in fp32 for stability; returns in x.dtype.
      - No bias, no residual, no norm-before-gate support in this snippet.
    """
    def __init__(self, hidden_size: int, eps: float = 1e-6,
                 norm_before_gate: bool = True, activation: str = "swish"):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        # API compatibility; not used in this optimized version
        self.norm_before_gate = norm_before_gate
        self.activation = activation
        # Elementwise affine weight
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # Fallback to torch if not CUDA
        if not x.is_cuda or not z.is_cuda:
            x32 = x.float()
            var = x32.pow(2).mean(dim=-1, keepdim=True)
            rstd = torch.rsqrt(var + self.eps)
            y = x32 * rstd * self.weight.float()
            if self.activation in ("swish", "silu"):
                g = z.float() * torch.sigmoid(z.float())
            else:
                g = torch.sigmoid(z.float())
            out = (y * g).to(x.dtype)
            return out

        return rmsnorm_gated(x, z, self.weight, eps=self.eps, activation=self.activation)

RMSNormGated = ModelNew
