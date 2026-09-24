import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import Triton; if unavailable, we'll fallback to PyTorch ops.
try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False


# -----------------------------
# Fast-path Triton kernel: single-block LN over last dim of [B, L, D]
# One program per row; BLOCK_D >= D so we load once, compute, normalize, store.
# -----------------------------
if _HAS_TRITON:
    @triton.jit
    def _layer_norm_3d_singleblock_kernel(
        X,  # *ptr* to input
        Y,  # *ptr* to output
        W,  # *ptr* to weight (dummy if no affine)
        BIAS,  # *ptr* to bias (dummy if no affine)
        stride_b: tl.constexpr,  # strides in elements
        stride_l: tl.constexpr,
        stride_d: tl.constexpr,
        L: tl.constexpr,         # sequence length
        D: tl.constexpr,         # feature dimension
        EPS,                     # epsilon (float)
        HAS_AFFINE: tl.constexpr,  # bool: 1 if elementwise_affine, 0 otherwise
        BLOCK_D: tl.constexpr,   # block size over D, must be >= D
    ):
        # Grid: (B * L,)
        row = tl.program_id(0)
        b = row // L
        l = row % L

        base = b * stride_b + l * stride_l

        # Columns
        cols = tl.arange(0, BLOCK_D)
        mask = cols < D

        # Load entire row block once
        x = tl.load(X + base + cols * stride_d, mask=mask, other=0.0)
        x32 = x.to(tl.float32)

        # Sums
        sum_x = tl.sum(x32, axis=0)
        sum_x2 = tl.sum(x32 * x32, axis=0)

        inv_D = 1.0 / D
        mean = sum_x * inv_D
        var = sum_x2 * inv_D - mean * mean
        rstd = 1.0 / tl.sqrt(var + EPS)

        # Normalize
        norm = (x32 - mean) * rstd  # [BLOCK_D]

        if HAS_AFFINE:
            w = tl.load(W + cols, mask=mask, other=1.0).to(tl.float32)
            beta = tl.load(BIAS + cols, mask=mask, other=0.0).to(tl.float32)
            out32 = norm * w + beta
        else:
            out32 = norm

        out = out32.to(x.dtype)
        tl.store(Y + base + cols * stride_d, out, mask=mask)


# -----------------------------
# Fallback 2D kernel: processes ROWS rows per program
# Useful if D is very large and we can't set BLOCK_D >= D.
# -----------------------------
if _HAS_TRITON:
    @triton.jit
    def _layer_norm_3d_rows_kernel(
        X,  # *ptr* to input
        Y,  # *ptr* to output
        W,  # *ptr* to weight (dummy if no affine)
        BIAS,  # *ptr* to bias (dummy if no affine)
        stride_b: tl.constexpr,  # strides in elements
        stride_l: tl.constexpr,
        stride_d: tl.constexpr,
        L: tl.constexpr,         # sequence length
        D: tl.constexpr,         # feature dimension
        EPS,                     # epsilon (float)
        HAS_AFFINE: tl.constexpr,  # bool: 1 if elementwise_affine, 0 otherwise
        BLOCK_D: tl.constexpr,   # block size over D
        ROWS: tl.constexpr,      # number of rows per program
    ):
        # 2D grid: (ceil_div(L, ROWS), B)
        pid_l = tl.program_id(0)
        pid_b = tl.program_id(1)

        l_start = pid_l * ROWS
        rows = l_start + tl.arange(0, ROWS)  # [ROWS]
        valid_rows = rows < L

        base_b = pid_b * stride_b

        # Accumulators per row
        sum_x = tl.zeros((ROWS,), dtype=tl.float32)
        sum_x2 = tl.zeros((ROWS,), dtype=tl.float32)

        # Loop over feature blocks (first pass)
        for start in range(0, D, BLOCK_D):
            cols = start + tl.arange(0, BLOCK_D)  # [BLOCK_D]
            valid_cols = cols < D

            ptr = X + base_b + (rows[:, None] * stride_l) + (cols[None, :] * stride_d)
            mask = valid_rows[:, None] & valid_cols[None, :]
            x = tl.load(ptr, mask=mask, other=0.0)
            x32 = x.to(tl.float32)

            local_sum = tl.sum(x32, axis=1)         # [ROWS]
            local_sum2 = tl.sum(x32 * x32, axis=1)  # [ROWS]

            sum_x += local_sum
            sum_x2 += local_sum2

        inv_D = 1.0 / D
        mean = sum_x * inv_D  # [ROWS]
        var = sum_x2 * inv_D - mean * mean
        rstd = 1.0 / tl.sqrt(var + EPS)  # [ROWS]

        # Second pass: normalize and store
        for start in range(0, D, BLOCK_D):
            cols = start + tl.arange(0, BLOCK_D)
            valid_cols = cols < D

            ptr = X + base_b + (rows[:, None] * stride_l) + (cols[None, :] * stride_d)
            mask = valid_rows[:, None] & valid_cols[None, :]
            x = tl.load(ptr, mask=mask, other=0.0)
            x32 = x.to(tl.float32)

            norm = (x32 - mean[:, None]) * rstd[:, None]  # [ROWS, BLOCK_D]

            if HAS_AFFINE:
                w_ptr = W + cols
                beta_ptr = BIAS + cols
                valid_w = valid_cols
                w = tl.load(w_ptr, mask=valid_w, other=1.0).to(tl.float32)[None, :]  # [1, BLOCK_D]
                beta = tl.load(beta_ptr, mask=valid_w, other=0.0).to(tl.float32)[None, :]  # [1, BLOCK_D]
                out32 = norm * w + beta  # [ROWS, BLOCK_D]
            else:
                out32 = norm

            out = out32.to(x.dtype)
            y_ptr = Y + base_b + (rows[:, None] * stride_l) + (cols[None, :] * stride_d)
            tl.store(y_ptr, out, mask=mask)


# Python helper to launch the kernels
def _next_power_of_2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << ((n - 1).bit_length())


def _triton_layer_norm(x: torch.Tensor,
                       weight: torch.Tensor | None,
                       bias: torch.Tensor | None,
                       eps: float = 1e-5) -> torch.Tensor:
    """
    x: [B, L, D]
    weight, bias: [D] or None
    Returns y with same shape/dtype as x.
    """
    if not _HAS_TRITON or not x.is_cuda:
        return F.layer_norm(x, (x.shape[-1],), weight, bias, eps)

    # Ensure last-dim contiguous for simple addressing
    if not x.is_contiguous():
        x = x.contiguous()

    assert x.dim() == 3, f"Expected 3D tensor [B, L, D], got shape {tuple(x.shape)}"
    B, L, D = x.shape

    y = torch.empty_like(x)

    stride_b, stride_l, stride_d = x.stride(0), x.stride(1), x.stride(2)

    has_affine = int(weight is not None and bias is not None)
    w = weight if has_affine else x  # dummy
    b = bias if has_affine else x    # dummy

    # Fast path: single-block kernel if D is moderate (<= 4096)
    if D <= 4096:
        block_d = _next_power_of_2(D)
        block_d = min(block_d, 4096)
        num_warps = 8 if block_d >= 2048 else 4
        grid = (B * L,)
        _layer_norm_3d_singleblock_kernel[grid](
            x, y, w, b,
            stride_b, stride_l, stride_d,
            L, D,
            eps,
            has_affine,
            block_d,
            num_warps=num_warps,
            num_stages=2,
        )
        return y

    # Fallback: 2D rows kernel
    block_d = 1024 if D >= 1024 else 512
    rows = 4 if L >= 4 else max(1, L)
    num_warps = 4 if block_d <= 1024 else 8
    grid = (triton.cdiv(L, rows), B)
    _layer_norm_3d_rows_kernel[grid](
        x, y, w, b,
        stride_b, stride_l, stride_d,
        L, D,
        eps,
        has_affine,
        block_d,
        rows,
        num_warps=num_warps,
        num_stages=2,
    )
    return y


# -----------------------------
# Original modules (unchanged except using our Triton LN)
# -----------------------------
class SiLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x)


class Matmul(nn.Module):
    def forward(self, input, weight, bias=None):
        return F.linear(input, weight, bias)


class Linear(nn.Module):
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
        promote_fp32: bool = True,
    ):
        super().__init__()
        self.normalized_shape = (normalized_shape,)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.promote_fp32 = promote_fp32

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
        else:
            self.register_parameter("weight", None)

        if elementwise_affine:
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight
        b = self.bias
        if self.promote_fp32 and x.dtype in (torch.float16, torch.bfloat16):
            x32 = x.float()
            y32 = _triton_layer_norm(x32, w.float() if w is not None else None,
                                     b.float() if b is not None else None,
                                     eps=self.eps)
            return y32.to(x.dtype)
        else:
            return _triton_layer_norm(x, w, b, eps=self.eps)


# -----------------------------
# Adaptive LayerNorm (Model) with Triton LN
# -----------------------------
class ModelNew(nn.Module):
    r"""
    Adaptive normalization layer with a norm layer (layer_norm).

    Args:
        embedding_dim (`int`): Embedding dimension to use during projection.
        conditioning_embedding_dim (`int`): Dimension of the input condition.
        elementwise_affine (`bool`, defaults to `True`):
            Boolean flag to denote if affine transformation should be applied.
        eps (`float`, defaults to 1e-5): Epsilon factor.
        bias (`bool`, defaults to `True`): Whether to use bias in the linear layer.
        norm_type (`str`, defaults to `"layer_norm"`):
            Normalization layer to use. Only "layer_norm" is supported here.
        promote_fp32 (`bool`, defaults to `True`):
            If True and input is fp16/bf16, do reduction in fp32 for stability.
    """

    def __init__(
        self,
        embedding_dim: int,
        conditioning_embedding_dim: int,
        elementwise_affine=True,
        eps=1e-5,
        bias=True,
        norm_type="layer_norm",
        promote_fp32: bool = True,
    ):
        super().__init__()
        self.silu = SiLU()
        self.linear = Linear(conditioning_embedding_dim, embedding_dim * 2, bias=bias)
        if norm_type == "layer_norm":
            self.norm = LayerNorm(embedding_dim, eps=eps, elementwise_affine=elementwise_affine,
                                  promote_fp32=promote_fp32)
        else:
            raise ValueError(f"unknown norm_type {norm_type}")

    def forward(self, x: torch.Tensor, conditioning_embedding: torch.Tensor) -> torch.Tensor:
        ce = conditioning_embedding.to(x.dtype)
        emb = self.linear(self.silu(ce))  # [B, D*2]
        scale, shift = torch.chunk(emb, 2, dim=1)  # each [B, D]
        scale = scale.unsqueeze(1)  # [B, 1, D]
        shift = shift.unsqueeze(1)  # [B, 1, D]

        y = self.norm(x)  # [B, L, D]
        y = y * (1 + scale) + shift
        return y

AdaLayerNormContinuous = ModelNew
