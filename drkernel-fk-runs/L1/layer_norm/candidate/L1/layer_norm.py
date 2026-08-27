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


# Autotuned kernel: LayerNorm over last dimension (1D), with optional affine.
# Assumptions:
# - x is contiguous in the last dimension (enforced in forward).
# - Compute in fp32; store in original dtype.
@triton.autotune(
    configs=[
        # Small rows
        triton.Config({'BLOCK_SIZE': 32,  'num_warps': 1}, num_stages=2),
        triton.Config({'BLOCK_SIZE': 64,  'num_warps': 1}, num_stages=2),
        triton.Config({'BLOCK_SIZE': 64,  'num_warps': 2}, num_stages=2),
        # Medium rows
        triton.Config({'BLOCK_SIZE': 128, 'num_warps': 2}, num_stages=2),
        triton.Config({'BLOCK_SIZE': 128, 'num_warps': 4}, num_stages=3),
        triton.Config({'BLOCK_SIZE': 256, 'num_warps': 4}, num_stages=3),
        triton.Config({'BLOCK_SIZE': 256, 'num_warps': 8}, num_stages=3),
        # Large rows
        triton.Config({'BLOCK_SIZE': 512,  'num_warps': 8}, num_stages=3),
        triton.Config({'BLOCK_SIZE': 1024, 'num_warps': 8}, num_stages=4),
        triton.Config({'BLOCK_SIZE': 2048, 'num_warps': 8}, num_stages=4),
    ],
    key=['D'],  # tune by row length
)
@triton.jit
def _layer_norm_1d_kernel(
    X,               # *ptr* to input
    Y,               # *ptr* to output (will be written in input dtype)
    W,               # *ptr* to weight (can be dummy if no weight)
    B,               # *ptr* to bias   (can be dummy if no bias)
    stride_x,        # row stride in elements
    stride_y,        # row stride in elements
    D,               # row length
    eps,             # float
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    OUT_DTYPE: tl.constexpr,  # 0=float32, 1=float16, 2=bfloat16
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    x_row = X + row * stride_x
    y_row = Y + row * stride_y

    # Pass 1: compute sum and sum of squares in fp32
    sum_val = 0.0
    sum_sq = 0.0
    offset = 0
    while offset < D:
        idx = offset + tl.arange(0, BLOCK_SIZE)
        mask = idx < D
        x = tl.load(x_row + idx, mask=mask, other=0).to(tl.float32)
        sum_val += tl.sum(x, axis=0)
        sum_sq  += tl.sum(x * x, axis=0)
        offset += BLOCK_SIZE

    inv_D = 1.0 / D
    mean = sum_val * inv_D
    var  = sum_sq  * inv_D - mean * mean
    inv_std = tl.math.rsqrt(var + eps)  # faster than 1.0 / sqrt

    # Pass 2: normalize and apply affine, store to Y in desired dtype
    offset = 0
    while offset < D:
        idx = offset + tl.arange(0, BLOCK_SIZE)
        mask = idx < D
        x  = tl.load(x_row + idx, mask=mask, other=0).to(tl.float32)
        y  = (x - mean) * inv_std
        if HAS_WEIGHT:
            w = tl.load(W + idx, mask=mask, other=1).to(tl.float32)
            y = y * w
        if HAS_BIAS:
            b = tl.load(B + idx, mask=mask, other=0).to(tl.float32)
            y = y + b
        # Cast to output dtype before store
        if OUT_DTYPE == 0:
            out = y
        elif OUT_DTYPE == 1:
            out = y.to(tl.float16)
        else:
            out = y.to(tl.bfloat16)
        tl.store(y_row + idx, out, mask=mask)
        offset += BLOCK_SIZE


class ModelNew(nn.Module):
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

        # Cache for fp32 casts of params
        self._cast_done = False
        self._src_w = None
        self._src_b = None
        self._w32 = None
        self._b32 = None

    def _maybe_cast_params_to_fp32(self):
        # Lazy cast weight/bias to fp32 once
        w, b = self.weight, self.bias
        src_w, src_b = w, b
        w32 = (w.float() if (w is not None and w.dtype != torch.float32) else w)
        b32 = (b.float() if (b is not None and b.dtype != torch.float32) else b)
        self._cast_done = True
        self._src_w = src_w
        self._src_b = src_b
        self._w32 = w32
        self._b32 = b32

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fallbacks
        if (not _HAS_TRITON) or (not x.is_cuda):
            # Use PyTorch
            if self.promote_fp32:
                weight = self._w32 if self._w32 is not None else self.weight
                bias   = self._b32 if self._b32 is not None else self.bias
                return F.layer_norm(
                    x.float(), self.normalized_shape, weight, bias, self.eps
                ).to(x.dtype)
            else:
                return F.layer_norm(
                    x, self.normalized_shape, self.weight, self.bias, self.eps
                )

        # Training + autograd: fall back to PyTorch to get gradients unless you implement backward
        if self.training and x.requires_grad:
            if self.promote_fp32:
                weight = self._w32 if self._w32 is not None else self.weight
                bias   = self._b32 if self._b32 is not None else self.bias
                return F.layer_norm(
                    x.float(), self.normalized_shape, weight, bias, self.eps
                ).to(x.dtype)
            else:
                return F.layer_norm(
                    x, self.normalized_shape, self.weight, self.bias, self.eps
                )

        # Triton path
        x_in = x
        # Ensure last-dim contiguous
        x = x.contiguous()

        # Cast params to fp32 once if needed
        if self.promote_fp32 and self.elementwise_affine:
            if (not self._cast_done) or (self._src_w is not self.weight) or (self._src_b is not self.bias):
                self._maybe_cast_params_to_fp32()
            weight = self._w32
            bias   = self._b32
        else:
            weight = self.weight
            bias   = self.bias

        # Shapes
        shape = x.shape
        k = len(self.normalized_shape)
        if k != 1:
            # Fallback for non-1D normalized shapes
            if self.promote_fp32:
                return F.layer_norm(
                    x.float(), self.normalized_shape,
                    (self._w32 if self._w32 is not None else self.weight),
                    (self._b32 if self._b32 is not None else self.bias),
                    self.eps
                ).to(x.dtype)
            else:
                return F.layer_norm(
                    x, self.normalized_shape, self.weight, self.bias, self.eps
                )
        D = shape[-1]
        # Number of rows = product of preceding dims
        num_rows = 1
        for d in shape[:-1]:
            num_rows *= d

        # Allocate output in input dtype; kernel will write that dtype
        y = torch.empty_like(x)

        # Strides in elements (contiguous last dim => stride == D)
        stride_x = D
        stride_y = D

        has_weight = (weight is not None)
        has_bias   = (bias   is not None)

        # Map dtype to kernel OUT_DTYPE code
        if x.dtype == torch.float32:
            out_dtype_code = 0
        elif x.dtype == torch.float16:
            out_dtype_code = 1
        elif x.dtype == torch.bfloat16:
            out_dtype_code = 2
        else:
            # Unknown dtype: fallback to PyTorch
            if self.promote_fp32:
                weight = self._w32 if self._w32 is not None else self.weight
                bias   = self._b32 if self._b32 is not None else self.bias
                return F.layer_norm(
                    x.float(), self.normalized_shape, weight, bias, self.eps
                ).to(x.dtype)
            else:
                return F.layer_norm(
                    x, self.normalized_shape, self.weight, self.bias, self.eps
                )

        # Launch kernel: one program per row
        grid = (num_rows,)

        _layer_norm_1d_kernel[grid](
            x, y,
            weight if has_weight else x,   # dummy ptr if not used
            bias   if has_bias   else x,   # dummy ptr if not used
            stride_x,
            stride_y,
            D,
            self.eps,
            HAS_WEIGHT=has_weight,
            HAS_BIAS=has_bias,
            OUT_DTYPE=out_dtype_code,
        )

        return y

LayerNorm = ModelNew
