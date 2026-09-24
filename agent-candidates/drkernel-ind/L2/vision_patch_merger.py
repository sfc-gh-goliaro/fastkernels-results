import math
import torch
import torch.nn as nn

# Try to import Triton
try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False


# -----------------------------
# Triton LayerNorm kernel (bf16)
# -----------------------------
@triton.jit
def _ln_bf16_kernel(
    X,             # *ptr* to bf16 [M, K]
    W, B,          # *ptr* to fp32 [K] (weight), [K] (bias) or dummy
    Y,             # *ptr* to bf16 [M, K]
    M, K,          # int
    stride_xm, stride_xk,  # strides for X
    stride_ym, stride_yk,  # strides for Y
    eps,           # float
    has_weight: tl.constexpr,  # bool
    has_bias: tl.constexpr,    # bool
    BLOCK_K: tl.constexpr,     # block size (power-of-two >= K, cap 2048)
):
    row = tl.program_id(0)
    # First pass: accumulate sum and sum of squares in fp32
    sum_val = tl.zeros((), dtype=tl.float32)
    sum_sq = tl.zeros((), dtype=tl.float32)

    for start in range(0, K, BLOCK_K):
        offs = start + tl.arange(0, BLOCK_K)
        mask = offs < K
        x = tl.load(X + row * stride_xm + offs * stride_xk, mask=mask, other=0.0)
        x32 = x.to(tl.float32)
        sum_val += tl.sum(x32, axis=0)
        sum_sq  += tl.sum(x32 * x32, axis=0)

    Kf = tl.full((), K, dtype=tl.float32)
    mean = sum_val / Kf
    ex2  = sum_sq  / Kf
    var  = ex2 - mean * mean
    rstd = tl.rsqrt(var + eps)

    # Second pass: normalize + affine + store
    for start in range(0, K, BLOCK_K):
        offs = start + tl.arange(0, BLOCK_K)
        mask = offs < K
        x = tl.load(X + row * stride_xm + offs * stride_xk, mask=mask, other=0.0).to(tl.float32)
        z = (x - mean) * rstd  # fp32

        if has_weight:
            w = tl.load(W + offs, mask=mask, other=1.0)
        else:
            w = 1.0
        if has_bias:
            b = tl.load(B + offs, mask=mask, other=0.0)
        else:
            b = 0.0

        y32 = z * w + b
        ybf = y32.to(tl.bfloat16)
        tl.store(Y + row * stride_ym + offs * stride_yk, ybf, mask=mask)


def _next_power_of_two(x: int) -> int:
    return 1 if x <= 1 else 1 << (x - 1).bit_length()


def _ln_bf16(x: torch.Tensor,
             weight: torch.Tensor | None,
             bias: torch.Tensor | None,
             eps: float) -> torch.Tensor:
    """
    Fast LayerNorm for bf16 tensors using Triton.
    Shapes:
      x:      [M, K], bf16, CUDA
      weight: [K] or None, fp32
      bias:   [K] or None, fp32
    Returns:
      y: bf16 [M, K]
    Falls back to torch if unsupported.
    """
    assert x.is_cuda, "Triton LayerNorm requires CUDA tensor"
    assert x.dtype == torch.bfloat16, f"Expected bf16, got {x.dtype}"
    assert x.dim() == 2, f"Expected 2D tensor, got shape {tuple(x.shape)}"

    M, K = x.shape
    if not x.is_contiguous():
        x = x.contiguous()

    # Choose BLOCK_K: power-of-two >= K, cap to 2048
    BLOCK_K = _next_power_of_two(K)
    if BLOCK_K > 2048:
        return torch.nn.functional.layer_norm(x, (K,), weight=weight, bias=bias, eps=eps)

    y = torch.empty_like(x)

    stride_xm, stride_xk = x.stride(0), x.stride(1)
    stride_ym, stride_yk = y.stride(0), y.stride(1)

    has_weight = weight is not None
    has_bias = bias is not None

    # Ensure device and dtype
    dev = x.device
    W = weight if has_weight else torch.ones(K, dtype=torch.float32, device=dev)
    B = bias if has_bias else torch.zeros(K, dtype=torch.float32, device=dev)

    grid = (M,)

    _ln_bf16_kernel[grid](
        x, W, B, y,
        M, K,
        stride_xm, stride_xk,
        stride_ym, stride_yk,
        eps,
        has_weight=has_weight,
        has_bias=has_bias,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=2,
    )
    return y


# -----------------------------
# Inline LayerNorm (fallback)
# -----------------------------
class _LayerNormNew(nn.Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-5,
                 elementwise_affine: bool = True, promote_fp32: bool = True):
        super().__init__()
        # normalized_shape is int (last-dim size)
        self.normalized_shape = int(normalized_shape)
        self.eps = float(eps)
        self.elementwise_affine = bool(elementwise_affine)
        self.promote_fp32 = bool(promote_fp32)

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape))
            self.bias = nn.Parameter(torch.zeros(self.normalized_shape))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight if self.elementwise_affine else None
        bias = self.bias if self.elementwise_affine else None
        return torch.nn.functional.layer_norm(x, (self.normalized_shape,), weight=weight, bias=bias, eps=self.eps)


# -----------------------------
# Inline GELU
# -----------------------------
class _GELU(nn.Module):
    def __init__(self, approximate: str = "none"):
        super().__init__()
        self.approximate = approximate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.gelu(x, approximate=self.approximate)


# -----------------------------
# Inline ColumnParallelLinear
# -----------------------------
class _ColumnParallelLinear(nn.Module):
    """
    Simplified version: splits input dim; forward uses F.linear.
    No custom all-reduce; not needed for inference.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        # in_features is total; each partition gets in_features // tp
        # But tp is not provided here; assume tp=1 for this benchmark.
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.weight = nn.Parameter(torch.empty(self.out_features, self.in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features))
        else:
            self.register_parameter("bias", None)
        # Optional: a real initializer would reset_parameters(); here we trust input.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [M, K]; weight: [N, K] -> output: [M, N]
        return torch.nn.functional.linear(x, self.weight, self.bias)


# -----------------------------
# Inline RowParallelLinear
# -----------------------------
class _RowParallelLinear(nn.Module):
    """
    Simplified version: splits output dim; forward uses F.linear.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.weight = nn.Parameter(torch.empty(self.out_features, self.in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [M, K]; weight: [N, K] -> output: [M, N]
        return torch.nn.functional.linear(x, self.weight, self.bias)


# -----------------------------
# ModelNew: same API, faster LN, self-contained
# -----------------------------
class ModelNew(nn.Module):
    """
    Patch merger: norm -> flatten -> MLP(fc1, GELU, fc2).

    - use_postshuffle_norm: if True, norm after view into hidden_size;
                            else norm before view.
    - eps: LayerNorm epsilon.
    """
    def __init__(self,
                 d_model: int,
                 context_dim: int,
                 spatial_merge_size: int = 2,
                 use_postshuffle_norm: bool = False,
                 eps: float = 1e-6):
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size ** 2)
        self.use_postshuffle_norm = use_postshuffle_norm
        norm_dim = self.hidden_size if use_postshuffle_norm else context_dim

        # Inline LayerNorm
        self.norm = _LayerNormNew(norm_dim, eps=eps, elementwise_affine=True, promote_fp32=False)

        # MLP (simplified, inference-friendly)
        self.fc1 = _ColumnParallelLinear(self.hidden_size, self.hidden_size, bias=True)
        self.act = _GELU()
        self.fc2 = _RowParallelLinear(self.hidden_size, d_model, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Norm
        if self.use_postshuffle_norm:
            xt = x.view(-1, self.hidden_size)
            if _HAS_TRITON and xt.is_cuda and xt.dtype == torch.bfloat16 and xt.dim() == 2:
                y = _ln_bf16(xt, self.norm.weight, self.norm.bias, self.norm.eps)
            else:
                y = torch.nn.functional.layer_norm(
                    xt, (self.hidden_size,), weight=self.norm.weight, bias=self.norm.bias, eps=self.norm.eps
                )
        else:
            if _HAS_TRITON and x.is_cuda and x.dtype == torch.bfloat16 and x.dim() == 2:
                y = _ln_bf16(x, self.norm.weight, self.norm.bias, self.norm.eps)
            else:
                y = torch.nn.functional.layer_norm(
                    x, (self.norm.normalized_shape,), weight=self.norm.weight, bias=self.norm.bias, eps=self.norm.eps
                )
            y = y.view(-1, self.hidden_size)

        # MLP
        y = self.fc2(self.act(self.fc1(y)))
        return y

VisionPatchMerger = ModelNew
