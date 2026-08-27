import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def layernorm_to_z_kernel(
    X,       # *ptr* to input
    Z,       # *ptr* to output (will hold (x - mean) * inv_std)
    H: tl.constexpr,            # last dimension size
    stride_row: tl.constexpr,   # row stride in elements
    BLOCK_SIZE: tl.constexpr,
    EPS: tl.constexpr,
):
    # One program per row
    row_id = tl.program_id(0)
    row_x = X + row_id * stride_row
    row_z = Z + row_id * stride_row

    # Pass 1: compute mean and variance in fp32
    sum_val = tl.zeros((), dtype=tl.float32)
    sum_sq = tl.zeros((), dtype=tl.float32)

    off = 0
    while off < H:
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < H
        x = tl.load(row_x + idx, mask=mask, other=0.0)
        xf = x.to(tl.float32)
        block_sum = tl.sum(xf, axis=0)
        block_sum_sq = tl.sum(xf * xf, axis=0)
        sum_val += block_sum
        sum_sq += block_sum_sq
        off += BLOCK_SIZE

    invH = 1.0 / H
    mean = sum_val * invH
    var = sum_sq * invH - mean * mean
    r_inv = 1.0 / tl.sqrt(var + EPS)  # precompute reciprocal sqrt

    # Pass 2: write z = (x - mean) * r_inv in input dtype
    off = 0
    while off < H:
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < H
        x = tl.load(row_x + idx, mask=mask, other=0.0)
        xf = x.to(tl.float32)
        z = (xf - mean) * r_inv
        z_cast = z.to(x.dtype)
        tl.store(row_z + idx, z_cast, mask=mask)
        off += BLOCK_SIZE


def _choose_block_and_warps(H: int):
    # Heuristic: for H ~ 1024, BLOCK=256, warps=8 is a solid choice.
    if H >= 2048:
        return 256, 8
    elif H >= 1024:
        return 256, 8
    elif H >= 512:
        return 128, 4
    else:
        return 64, 4


def layernorm_to_z(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Compute z = ((x - mean) / sqrt(var + eps)) in a Triton kernel over last dim.
    Returns z with same shape/dtype as x.
    x must be CUDA.
    """
    assert x.is_cuda, "layernorm_to_z requires a CUDA tensor"
    if not x.is_contiguous():
        x = x.contiguous()

    # Flatten all leading dims to rows
    *leading, H = x.shape
    B = 1
    for d in leading:
        B *= d
    x2 = x.view(B, H)
    z2 = torch.empty_like(x2)

    BLOCK, num_warps = _choose_block_and_warps(H)
    grid = (B,)
    layernorm_to_z_kernel[grid](
        x2, z2,
        H,
        x2.stride(0),
        BLOCK_SIZE=BLOCK,
        EPS=eps,
        num_warps=num_warps,
        num_stages=2,
    )
    return z2.view_as(x)


class SiLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.silu(x)


class Matmul(nn.Module):
    """Pure functional linear: takes input, weight, and optional bias as forward args."""
    def forward(self, input, weight, bias=None):
        return torch.nn.functional.linear(input, weight, bias)


class Linear(nn.Module):
    """Parametric linear: stores weight and bias internally."""
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.matmul = Matmul()

    def forward(self, input):
        return self.matmul(input, self.weight, self.bias)


class LayerNorm(nn.Module):
    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        create_scale: bool = True,
        create_offset: bool = True,
        promote_fp32: bool = True,
    ):
        super().__init__()
        # Placeholder to mirror API; not used in ModelNew fast path.
        self.normalized_shape = (normalized_shape,)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.promote_fp32 = promote_fp32

        if elementwise_affine and create_scale:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
        else:
            self.register_parameter("weight", None)

        if elementwise_affine and create_offset:
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # No custom here in ModelNew; kept for API parity.
        pass


class ModelNew(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.hidden_size = hidden_size
        # Keep.LayerNorm to mirror API, but we'll use a custom fast path
        self.norm_final = LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.linear = Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.ModuleList(
            [
                SiLU(),
                Linear(hidden_size, 2 * hidden_size, bias=True),
            ]
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # 1) Modulation: siLU + linear -> [B, S, 2H] -> chunk -> [B, S, H] each
        modulation = c
        for layer in self.adaLN_modulation:
            modulation = layer(modulation)
        shift, scale = modulation.chunk(2, dim=-1)
        # Broadcast to x
        while shift.dim() < x.dim():
            shift = shift.unsqueeze(-2)
            scale = scale.unsqueeze(-2)

        # 2) Fast path: compute z = norm(x) using Triton, then form y = z + z*scale + shift
        z = layernorm_to_z(x, eps=1e-6)  # z has same shape/dtype as x
        # Elementwise in input dtype
        y = z
        y = y + y * scale  # term: z*scale
        y = y + shift      # add shift

        # 3) Linear
        return self.linear(y)

OasisFinalLayer = ModelNew
