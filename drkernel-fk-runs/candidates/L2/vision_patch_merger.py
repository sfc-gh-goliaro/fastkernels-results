import math
import torch
import torch.nn as nn

import triton
import triton.language as tl


def _next_power_of_two(x: int) -> int:
    return 1 if x <= 1 else 2 ** ((x - 1).bit_length())


@triton.jit
def _layer_norm_forward_kernel(
    x_ptr,                  # *dtype pointer to input, shape [B, C]
    y_ptr,                  # *dtype pointer to output, shape [B, C]
    w_ptr,                  # *dtype pointer to weight (gamma), shape [C] or dummy
    b_ptr,                  # *dtype pointer to bias   (beta),  shape [C] or dummy
    stride_x_row: tl.constexpr,
    stride_x_col: tl.constexpr,
    stride_y_row: tl.constexpr,
    stride_y_col: tl.constexpr,
    C: tl.constexpr,        # feature dimension (normalized_shape)
    eps: tl.constexpr,      # epsilon
    has_weight: tl.constexpr,
    has_bias: tl.constexpr,
    OUT_DTYPE: tl.constexpr,  # tl.float16 / tl.bfloat16 / tl.float32
    BLOCK: tl.constexpr       # power-of-two tile >= C
):
    # One program per row
    row = tl.program_id(0)

    cols = tl.arange(0, BLOCK)

    # Base pointers for this row
    x_row_base = x_ptr + row * stride_x_row
    y_row_base = y_ptr + row * stride_y_row

    # First pass: compute sum and sum of squares in float32
    sum_x = 0.0
    sum_x2 = 0.0
    # Loop over tiles
    for t in range(0, C, BLOCK):
        col = t + cols
        mask = col < C
        x = tl.load(x_row_base + col * stride_x_col, mask=mask, other=0.0).to(tl.float32)
        sum_x += tl.sum(x, axis=0)
        sum_x2 += tl.sum(x * x, axis=0)

    mean = sum_x / C
    # var = E[x^2] - (E[x])^2 is fine for C=1152; still safe in fp32
    var = sum_x2 / C - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize, apply affine, store
    for t in range(0, C, BLOCK):
        col = t + cols
        mask = col < C
        x = tl.load(x_row_base + col * stride_x_col, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * rstd  # float32

        if has_weight:
            w = tl.load(w_ptr + col, mask=mask, other=1.0).to(tl.float32)
        else:
            w = 1.0
        if has_bias:
            b = tl.load(b_ptr + col, mask=mask, other=0.0).to(tl.float32)
        else:
            b = 0.0

        y = y * w + b  # float32
        y = y.to(OUT_DTYPE)
        tl.store(y_row_base + col * stride_y_col, y, mask=mask)


def _layer_norm_triton(x: torch.Tensor,
                       weight: torch.Tensor | None,
                       bias: torch.Tensor | None,
                       eps: float) -> torch.Tensor:
    """
    Triton LayerNorm over the last dimension.
    Shapes:
      x:      [..., C] (will be made contiguous)
      weight: [C] or None
      bias:   [C] or None
    Returns:
      y: same shape and dtype as x
    """
    assert x.is_cuda, "Triton LayerNorm requires CUDA tensor"
    # Make contiguous
    x_c = x.contiguous()
    # Last dimension is features
    *prefix, C = x_c.shape
    # Flatten to [B, C]
    B = int(math.prod(prefix)) if len(prefix) > 0 else 1
    x_2d = x_c.view(B, C)

    # Allocate output
    y_2d = torch.empty_like(x_2d)

    # Strides in elements
    stride_x_row = x_2d.stride(0)
    stride_x_col = x_2d.stride(1)
    stride_y_row = y_2d.stride(0)
    stride_y_col = y_2d.stride(1)

    # Dtypes
    if x.dtype == torch.bfloat16:
        out_dtype = tl.bfloat16
    elif x.dtype == torch.float16:
        out_dtype = tl.float16
    else:
        out_dtype = tl.float32

    has_weight = weight is not None
    has_bias = bias is not None
    # Ensure weight/bias are on device and 1D
    if has_weight:
        w = weight.contiguous()
    else:
        w = torch.empty(1, device=x.device, dtype=x.dtype)
    if has_bias:
        b = bias.contiguous()
    else:
        b = torch.empty(1, device=x.device, dtype=x.dtype)

    # Choose BLOCK as next power-of-two >= C, cap to a reasonable size
    BLOCK = _next_power_of_two(C)
    # Triton benefits from power-of-two blocks; 1024 or 2048 are good choices.
    # For C up to a few thousand, 2048 is safe.
    BLOCK = min(BLOCK, 2048)

    # Launch
    grid = (B,)
    _layer_norm_forward_kernel[grid](
        x_2d, y_2d, w, b,
        stride_x_row, stride_x_col,
        stride_y_row, stride_y_col,
        C,
        eps,
        has_weight, has_bias,
        OUT_DTYPE=out_dtype,
        BLOCK=BLOCK,
        num_warps=4,
        num_stages=2,
    )

    # Reshape back
    y = y_2d.view(*prefix, C)
    return y


class LayerNormTriton(nn.Module):
    """
    Drop-in replacement for LayerNorm that uses a Triton kernel on CUDA,
    falls back to torch.nn.functional.layer_norm otherwise.
    Matches the original LayerNorm signature used in Model.
    """
    def __init__(self, normalized_shape: int, eps: float = 1e-5,
                 elementwise_affine: bool = True,
                 create_scale: bool = True,
                 create_offset: bool = True,
                 promote_fp32: bool = True):
        super().__init__()
        if isinstance(normalized_shape, (list, tuple)):
            assert len(normalized_shape) == 1, "Only 1D normalized_shape supported here"
            normalized_shape = normalized_shape[0]
        self.normalized_shape = int(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        # Keep parameters if affine
        if elementwise_affine and create_scale:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape))
        else:
            self.register_parameter("weight", None)
        if elementwise_affine and create_offset:
            self.bias = nn.Parameter(torch.zeros(self.normalized_shape))
        else:
            self.register_parameter("bias", None)
        self.promote_fp32 = promote_fp32  # API parity; not used by kernel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CPU fallback
        if not x.is_cuda:
            return F.layer_norm(
                x, (self.normalized_shape,), self.weight, self.bias, self.eps
            )
        # CUDA Triton path
        return _layer_norm_triton(x, self.weight, self.bias, self.eps)


# Provide ModelNew that uses the Triton LayerNorm and standard PyTorch linears.

class ModelNew(nn.Module):
    """Patch merger: norm -> flatten -> MLP(fc1, GELU, fc2).

    Qwen3 DeepStack mergers set use_postshuffle_norm=True to norm after reshape.
    This version swaps in a Triton LayerNorm on CUDA while keeping GEMMs in PyTorch.
    """

    def __init__(self, d_model: int, context_dim: int,
                 spatial_merge_size: int = 2,
                 use_postshuffle_norm: bool = False,
                 eps: float = 1e-6):
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size ** 2)
        self.use_postshuffle_norm = use_postshuffle_norm
        norm_dim = self.hidden_size if use_postshuffle_norm else context_dim
        # Replace with Triton LayerNorm
        self.norm = LayerNormTriton(norm_dim, eps=eps, promote_fp32=False)
        # Standard linears (no tensor-parallel dependencies)
        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.act = nn.GELU()  # approximate='none' by default
        self.fc2 = nn.Linear(self.hidden_size, d_model, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [..., 1, 1152] in tests; we norm over last dim
        if self.use_postshuffle_norm:
            x = self.norm(x.view(-1, self.hidden_size))
        else:
            x = self.norm(x).view(-1, self.hidden_size)
        x = self.fc2(self.act(self.fc1(x)))
        return x

VisionPatchMerger = ModelNew
