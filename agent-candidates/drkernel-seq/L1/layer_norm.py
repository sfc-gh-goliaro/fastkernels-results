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


@triton.jit
def lnorm_fwd_kernel(
    x_ptr,                 # *T, shape [M, D], contiguous
    y_ptr,                 # *T, shape [M, D]
    w_ptr,                 # *f32, shape [D] or dummy
    b_ptr,                 # *f32, shape [D] or dummy
    M: tl.constexpr,       # number of rows (programs)
    D: tl.constexpr,       # number of features per row
    stride_xm: tl.constexpr,
    stride_xd: tl.constexpr,
    stride_ym: tl.constexpr,
    stride_yd: tl.constexpr,
    eps: tl.constexpr,     # float
    has_weight: tl.constexpr,
    has_bias: tl.constexpr,
    OUT_DTYPE: tl.constexpr,  # 0: fp32, 1: fp16, 2: bf16
    BLOCK_D: tl.constexpr,
):
    # program id: which row m
    m = tl.program_id(0)
    if m >= M:
        return

    # Accumulators in fp32
    sum_x = tl.zeros((), dtype=tl.float32)
    sum_x2 = tl.zeros((), dtype=tl.float32)

    # First pass: accumulate sum and sum of squares
    for off in range(0, D, BLOCK_D):
        idx = off + tl.arange(0, BLOCK_D)
        mask = idx < D
        x = tl.load(x_ptr + m * stride_xm + idx * stride_xd, mask=mask, other=0.0)
        xf = x.to(tl.float32)
        sum_x += tl.sum(xf, axis=0)
        sum_x2 += tl.sum(xf * xf, axis=0)

    Df = tl.full((), D, dtype=tl.float32)
    mean = sum_x / Df
    var = sum_x2 / Df - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize and store
    for off in range(0, D, BLOCK_D):
        idx = off + tl.arange(0, BLOCK_D)
        mask = idx < D
        x = tl.load(x_ptr + m * stride_xm + idx * stride_xd, mask=mask, other=0.0)
        xf = x.to(tl.float32)
        z = (xf - mean) * rstd  # fp32

        if has_weight:
            w = tl.load(w_ptr + idx, mask=mask, other=1.0).to(tl.float32)
            z = z * w
        if has_bias:
            b = tl.load(b_ptr + idx, mask=mask, other=0.0).to(tl.float32)
            z = z + b

        # cast to output dtype
        if OUT_DTYPE == 0:
            out = z
        elif OUT_DTYPE == 1:
            out = z.to(tl.float16)
        elif OUT_DTYPE == 2:
            out = z.to(tl.bfloat16)
        else:
            out = z  # default fp32

        tl.store(y_ptr + m * stride_ym + idx * stride_yd, out, mask=mask)


def _torch_dtype_to_out_code(dtype: torch.dtype) -> int:
    if dtype == torch.float32:
        return 0
    if dtype == torch.float16:
        return 1
    if dtype == torch.bfloat16:
        return 2
    return 0  # fallback


def _pick_block_d(D: int) -> int:
    # power-of-two block, cap at 1024 or 2048
    if D <= 64:
        return 64
    elif D <= 128:
        return 128
    elif D <= 256:
        return 256
    elif D <= 512:
        return 512
    elif D <= 1024:
        return 1024
    else:
        return 2048


def _pick_num_warps(block_d: int) -> int:
    if block_d <= 128:
        return 4
    elif block_d <= 256:
        return 4
    elif block_d <= 512:
        return 8
    else:
        return 8


class ModelNew(nn.Module):
    def __init__(
        self,
        normalized_shape: int | tuple,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        create_scale: bool = True,
        create_offset: bool = True,
        promote_fp32: bool = True,
    ):
        super().__init__()
        if isinstance(normalized_shape, int):
            norm_shape = (normalized_shape,)
        else:
            norm_shape = tuple(normalized_shape)
        self.normalized_shape = norm_shape
        self.eps = float(eps)
        self.elementwise_affine = bool(elementwise_affine)
        self.promote_fp32 = bool(promote_fp32)

        if elementwise_affine and create_scale:
            self.weight = nn.Parameter(torch.ones(norm_shape))
        else:
            self.register_parameter("weight", None)

        if elementwise_affine and create_offset:
            self.bias = nn.Parameter(torch.zeros(norm_shape))
        else:
            self.register_parameter("bias", None)

        # Cache for fp32 params, filled on first use
        self._cast_done = False
        self._src_w = self.weight
        self._src_b = self.bias
        self._w32 = None
        self._b32 = None

    def _ensure_fp32_params(self):
        w, b = self.weight, self.bias
        if (not self._cast_done) or (self._src_w is not w) or (self._src_b is not b):
            self._src_w, self._src_b = w, b
            self._w32 = w.float() if w is not None and w.dtype != torch.float32 else w
            self._b32 = b.float() if b is not None and b.dtype != torch.float32 else b
            self._cast_done = True
        return self._w32, self._b32

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fallback to PyTorch if CPU or Triton not available
        if (not x.is_cuda) or (not _HAS_TRITON):
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )

        # Ensure contiguous
        x_in = x if x.is_contiguous() else x.contiguous()

        # Compute D = product(normalized_shape), handle empty tuple
        if len(self.normalized_shape) == 0:
            D = 1
        else:
            D = math.prod(int(d) for d in self.normalized_shape)

        # Total elements and M
        N = x_in.numel()
        if D == 0:
            # Degenerate; just use PyTorch
            return F.layer_norm(
                x_in, self.normalized_shape, self.weight, self.bias, self.eps
            )
        M = N // D

        # View as [M, D]; this assumes trailing dims match normalized_shape
        x2 = x_in.view(M, D)
        y2 = torch.empty_like(x2)

        # Prepare weight/bias as fp32
        has_w = self.weight is not None
        has_b = self.bias is not None
        w32, b32 = None, None
        if has_w or has_b:
            w32, b32 = self._ensure_fp32_params()
            if w32 is not None and w32.device != x2.device:
                w32 = w32.to(x2.device)
            if b32 is not None and b32.device != x2.device:
                b32 = b32.to(x2.device)
        if has_w and w32 is None:
            has_w = False
        if has_b and b32 is None:
            has_b = False

        # Flatten to 1D for simple strides (contiguous)
        x3 = x2.reshape(-1)
        y3 = y2.reshape(-1)

        # Strides in elements for contiguous
        stride_xm = D
        stride_xd = 1
        stride_ym = D
        stride_yd = 1

        # Block and launch config
        BLOCK_D = _pick_block_d(D)
        num_warps = _pick_num_warps(BLOCK_D)

        out_code = _torch_dtype_to_out_code(x2.dtype)

        # Launch kernel: grid = (M,)
        lnorm_fwd_kernel[(M,)](
            x3, y3,
            (w32.reshape(-1) if has_w else y3),  # dummy ptr if unused
            (b32.reshape(-1) if has_b else y3),  # dummy ptr if unused
            M, D,
            stride_xm, stride_xd,
            stride_ym, stride_yd,
            self.eps,
            has_weight=has_w,
            has_bias=has_b,
            OUT_DTYPE=out_code,
            BLOCK_D=BLOCK_D,
            num_warps=num_warps,
        )

        # Reshape back
        y = y2.view_as(x_in)
        return y

LayerNorm = ModelNew
