import math
import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False


# ---------------------------
# Triton kernels
# ---------------------------

@triton.jit
def _rmsnorm_fwd_kernel_single(
    x_ptr,               # *T, [M, N]
    w_ptr,               # *T, [N] or dummy
    y_ptr,               # *T, [M, N]
    M: tl.constexpr,
    N: tl.constexpr,
    stride_xm: tl.constexpr,
    stride_xn: tl.constexpr,
    stride_ym: tl.constexpr,
    stride_yn: tl.constexpr,
    eps: tl.constexpr,
    has_weight: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)

    # Pass 1: accumulate sum of squares of x
    sumsq = tl.zeros((), dtype=tl.float32)
    off = 0
    while off < N:
        idx = off + tl.arange(0, BLOCK)
        mask = idx < N
        x = tl.load(x_ptr + pid * stride_xm + idx * stride_xn, mask=mask, other=0).to(tl.float32)
        sumsq += tl.sum(x * x, axis=0)
        off += BLOCK

    mean = sumsq / N
    inv_rms = 1.0 / tl.sqrt(mean + eps)  # f32

    # Pass 2: y = x * inv_rms, * w if any
    off = 0
    while off < N:
        idx = off + tl.arange(0, BLOCK)
        mask = idx < N

        x = tl.load(x_ptr + pid * stride_xm + idx * stride_xn, mask=mask, other=0).to(tl.float32)
        y = x * inv_rms

        if has_weight:
            w = tl.load(w_ptr + idx, mask=mask, other=1).to(tl.float32)
            y = y * w

        tl.store(y_ptr + pid * stride_ym + idx * stride_yn, y, mask=mask)
        off += BLOCK


@triton.jit
def _rmsnorm_fwd_kernel_twoout(
    x_ptr,               # *T, [M, N]
    w_ptr,               # *T, [N] or dummy
    r_ptr,               # *T, [M, N]
    y_ptr,               # *T, [M, N]
    r_out_ptr,           # *T, [M, N]  (x + r)
    M: tl.constexpr,
    N: tl.constexpr,
    stride_xm: tl.constexpr,
    stride_xn: tl.constexpr,
    stride_rm: tl.constexpr,
    stride_rn: tl.constexpr,
    stride_ym: tl.constexpr,
    stride_yn: tl.constexpr,
    stride_rom: tl.constexpr,
    stride_ron: tl.constexpr,
    eps: tl.constexpr,
    has_weight: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)

    # Pass 1: accumulate sum of squares of v = x + r
    sumsq = tl.zeros((), dtype=tl.float32)
    off = 0
    while off < N:
        idx = off + tl.arange(0, BLOCK)
        mask = idx < N

        x = tl.load(x_ptr + pid * stride_xm + idx * stride_xn, mask=mask, other=0).to(tl.float32)
        r = tl.load(r_ptr + pid * stride_rm + idx * stride_rn, mask=mask, other=0).to(tl.float32)
        v = x + r
        sumsq += tl.sum(v * v, axis=0)
        off += BLOCK

    mean = sumsq / N
    inv_rms = 1.0 / tl.sqrt(mean + eps)  # f32

    # Pass 2: produce y and r_out = x + r, apply weight if any
    off = 0
    while off < N:
        idx = off + tl.arange(0, BLOCK)
        mask = idx < N

        x = tl.load(x_ptr + pid * stride_xm + idx * stride_xn, mask=mask, other=0).to(tl.float32)
        r = tl.load(r_ptr + pid * stride_rm + idx * stride_rn, mask=mask, other=0).to(tl.float32)

        v = x + r  # x + residual
        y = v * inv_rms  # f32

        if has_weight:
            w = tl.load(w_ptr + idx, mask=mask, other=1).to(tl.float32)
            y = y * w

        # store y in x.dtype
        tl.store(y_ptr + pid * stride_ym + idx * stride_yn, y, mask=mask)
        # store r_out = v in same dtype as x
        tl.store(r_out_ptr + pid * stride_rom + idx * stride_ron, v, mask=mask)

        off += BLOCK


# ---------------------------
# Python helpers
# ---------------------------

def _next_power_of_two(x: int, cap: int = 2048) -> int:
    if x <= 1:
        return 1
    p = 1 << (x - 1).bit_length()
    return min(p, cap)


def _choose_num_warps(block: int) -> int:
    if block <= 128:
        return 2
    if block <= 512:
        return 4
    return 8


# ---------------------------
# Triton wrappers
# ---------------------------

def _rmsnorm_triton_single(x: torch.Tensor,
                           weight: torch.Tensor | None,
                           eps: float) -> torch.Tensor:
    """
    Compute y = rmsnorm(x) * weight along last dimension using Triton.
    Returns y (same shape as x).
    """
    assert x.is_cuda, "Triton kernel requires CUDA tensor"
    assert x.dtype in (torch.float16, torch.bfloat16, torch.float32), f"Unsupported dtype {x.dtype}"

    orig_shape = x.shape
    N = orig_shape[-1]
    M = int(x.numel() // N)

    x2 = x.reshape(M, N).contiguous()
    y2 = torch.empty_like(x2)

    has_weight = weight is not None
    if has_weight:
        w = weight.contiguous()
    else:
        w = x2  # dummy

    stride_xm, stride_xn = x2.stride(0), x2.stride(1)
    stride_ym, stride_yn = y2.stride(0), y2.stride(1)

    BLOCK = _next_power_of_two(N, cap=2048)
    num_warps = _choose_num_warps(BLOCK)

    grid = (M,)

    _rmsnorm_fwd_kernel_single[grid](
        x2, w, y2,
        M, N,
        stride_xm, stride_xn,
        stride_ym, stride_yn,
        eps,
        has_weight,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=2,
    )

    return y2.reshape(orig_shape)


def _rmsnorm_triton_twoout(x: torch.Tensor,
                           weight: torch.Tensor | None,
                           eps: float,
                           residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute:
      y      = rmsnorm(x + residual) * weight
      r_out  = x + residual
    along last dimension using Triton.

    Returns (y, x + residual) with same shape as x.
    """
    assert x.is_cuda, "Triton kernel requires CUDA tensor"
    assert residual.is_cuda, "residual must be CUDA for Triton path"
    assert x.shape == residual.shape, f"Shapes must match: got {x.shape} vs {residual.shape}"
    assert x.dtype in (torch.float16, torch.bfloat16, torch.float32), f"Unsupported dtype {x.dtype}"

    orig_shape = x.shape
    N = orig_shape[-1]
    M = int(x.numel() // N)

    x2 = x.reshape(M, N).contiguous()
    r2 = residual.reshape(M, N).contiguous()
    y2 = torch.empty_like(x2)
    r_out2 = torch.empty_like(r2)

    has_weight = weight is not None
    if has_weight:
        w = weight.contiguous()
    else:
        w = x2  # dummy

    stride_xm, stride_xn = x2.stride(0), x2.stride(1)
    stride_ym, stride_yn = y2.stride(0), y2.stride(1)
    stride_rm, stride_rn = r2.stride(0), r2.stride(1)
    stride_rom, stride_ron = r_out2.stride(0), r_out2.stride(1)

    BLOCK = _next_power_of_two(N, cap=2048)
    num_warps = _choose_num_warps(BLOCK)

    grid = (M,)

    _rmsnorm_fwd_kernel_twoout[grid](
        x2, w, r2, y2, r_out2,
        M, N,
        stride_xm, stride_xn,
        stride_rm, stride_rn,
        stride_ym, stride_yn,
        stride_rom, stride_ron,
        eps,
        has_weight,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=2,
    )

    return y2.reshape(orig_shape), r_out2.reshape(orig_shape)


# ---------------------------
# Module: ModelNew
# ---------------------------

class ModelNew(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = float(eps)
        self.elementwise_affine = bool(elementwise_affine)
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(hidden_size))
        else:
            self.register_buffer("_unit_weight", torch.ones(hidden_size), persistent=False)

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None):
        # Handle residual is None: return single output
        if residual is None:
            if (not x.is_cuda) or (not _HAS_TRITON):
                y = torch.nn.functional.rms_norm(x, (x.shape[-1],), eps=self.eps)
                if self.elementwise_affine:
                    y = y * self.weight
                return y

            if self.elementwise_affine:
                w = self.weight
                if w.device != x.device:
                    w = w.to(device=x.device)
                if w.dtype != x.dtype:
                    w = w.to(dtype=x.dtype)
                weight = w
            else:
                weight = self._unit_weight.to(device=x.device, dtype=x.dtype)

            return _rmsnorm_triton_single(x, weight=weight, eps=self.eps)

        # residual is not None: return (y, x + residual)
        if (not x.is_cuda) or (not _HAS_TRITON):
            xr = x + residual
            y = torch.nn.functional.rms_norm(xr, (x.shape[-1],), eps=self.eps)
            if self.elementwise_affine:
                y = y * self.weight
            return y, xr

        if self.elementwise_affine:
            w = self.weight
            if w.device != x.device:
                w = w.to(device=x.device)
            if w.dtype != x.dtype:
                w = w.to(dtype=x.dtype)
            weight = w
        else:
            weight = self._unit_weight.to(device=x.device, dtype=x.dtype)

        return _rmsnorm_triton_twoout(x, weight=weight, eps=self.eps, residual=residual)

RMSNorm = ModelNew
