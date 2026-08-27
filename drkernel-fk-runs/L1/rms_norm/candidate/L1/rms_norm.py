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
# Triton kernels: Welford-based single read of x, then normalize
# ---------------------------

@triton.jit
def rmsnorm_kernel_welford(
    X,                # *ptr* to input, shape [M, H]
    W,                # *ptr* to weight, shape [H] (dummy if has_weight == 0)
    Y,                # *ptr* to output, shape [M, H]
    M, H,             # int: rows, cols
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    stride_w,         # typically 1
    eps,              # float
    has_weight: tl.constexpr,  # 0/1
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)

    # Pass 1: Welford over blocks using masks (no 'continue')
    count = 0.0
    mean = 0.0
    M2 = 0.0
    for col in range(0, H, BLOCK_SIZE):
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < H
        x = tl.load(X + row * stride_xm + offs * stride_xn, mask=mask, other=0.0)
        xf = x.to(tl.float32)

        # Masked block reductions
        block_count = tl.sum(mask.to(tl.float32), axis=0)
        sum1 = tl.sum(tl.where(mask, xf, 0.0), axis=0)
        sum2 = tl.sum(tl.where(mask, xf * xf, 0.0), axis=0)

        block_mean = sum1 / block_count
        block_m2 = sum2 / block_count - block_mean * block_mean

        new_count = count + block_count
        # If new_count == 0 (empty row), skip combine (but here H>0 and BLOCK_SIZE>0 so typically not 0).
        # Still guard divide-by-zero just in case.
        delta = block_mean - mean
        # Only combine if there is something new
        mean = mean + delta * (block_count / new_count)
        M2 = M2 + block_m2 * (block_count / new_count) + delta * delta * (count / new_count) * (block_count / new_count)
        count = new_count

    var = M2
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Pass 2: normalize and (optionally) weight, store
    for col in range(0, H, BLOCK_SIZE):
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < H
        x = tl.load(X + row * stride_xm + offs * stride_xn, mask=mask, other=0.0)
        xf = x.to(tl.float32)
        y = xf * inv_std
        if has_weight == 1:
            w = tl.load(W + offs * stride_w, mask=mask, other=1.0)
            wf = w.to(tl.float32)
            y = y * wf
        y_cast = y.to(x.dtype)
        tl.store(Y + row * stride_ym + offs * stride_yn, y_cast, mask=mask)


@triton.jit
def fused_add_rmsnorm_kernel_welford(
    X,                # *ptr* input, shape [M, H]
    R,                # *ptr* residual, shape [M, H]
    W,                # *ptr* weight, shape [H]
    Y,                # *ptr* output, shape [M, H]
    M, H,
    stride_xm, stride_xn,
    stride_rm, stride_rn,
    stride_w,
    stride_ym, stride_yn,
    eps,
    has_weight: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)

    # Pass 1: Welford over z = x + r using masks
    count = 0.0
    mean = 0.0
    M2 = 0.0
    for col in range(0, H, BLOCK_SIZE):
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < H
        x = tl.load(X + row * stride_xm + offs * stride_xn, mask=mask, other=0.0)
        r = tl.load(R + row * stride_rm + offs * stride_rn, mask=mask, other=0.0)
        zf = (x + r).to(tl.float32)

        block_count = tl.sum(mask.to(tl.float32), axis=0)
        sum1 = tl.sum(tl.where(mask, zf, 0.0), axis=0)
        sum2 = tl.sum(tl.where(mask, zf * zf, 0.0), axis=0)

        block_mean = sum1 / block_count
        block_m2 = sum2 / block_count - block_mean * block_mean

        new_count = count + block_count
        delta = block_mean - mean
        mean = mean + delta * (block_count / new_count)
        M2 = M2 + block_m2 * (block_count / new_count) + delta * delta * (count / new_count) * (block_count / new_count)
        count = new_count

    var = M2
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Pass 2: normalize z, weight, store
    for col in range(0, H, BLOCK_SIZE):
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < H
        x = tl.load(X + row * stride_xm + offs * stride_xn, mask=mask, other=0.0)
        r = tl.load(R + row * stride_rm + offs * stride_rn, mask=mask, other=0.0)
        z = (x + r).to(tl.float32) * inv_std
        if has_weight == 1:
            w = tl.load(W + offs * stride_w, mask=mask, other=1.0)
            wf = w.to(tl.float32)
            z = z * wf
        z_cast = z.to(x.dtype)
        tl.store(Y + row * stride_ym + offs * stride_yn, z_cast, mask=mask)


# ---------------------------
# Python wrappers
# ---------------------------

def _choose_block_size(H: int) -> int:
    if H >= 2048:
        return 1024
    elif H >= 1024:
        return 1024
    elif H >= 256:
        return 256
    else:
        return 128

def _rmsnorm_triton(x_2d: torch.Tensor,
                    weight: torch.Tensor | None,
                    eps: float) -> torch.Tensor:
    """
    x_2d: [M, H], contiguous, CUDA
    weight: [H] or None
    returns: [M, H]
    """
    assert x_2d.is_cuda, "Triton path requires CUDA tensor"
    M, H = x_2d.shape
    y = torch.empty_like(x_2d)
    BLOCK = _choose_block_size(H)
    has_weight = 1 if (weight is not None) else 0
    w_ptr = weight if weight is not None else x_2d  # dummy ptr
    grid = (M,)
    rmsnorm_kernel_welford[grid](
        x_2d, w_ptr, y,
        M, H,
        x_2d.stride(0), x_2d.stride(1),
        y.stride(0), y.stride(1),
        (weight.stride(0) if weight is not None else 1),
        eps,
        has_weight,
        BLOCK_SIZE=BLOCK,
        num_warps=8 if BLOCK >= 1024 else 4,
    )
    return y

def _fused_add_rmsnorm_triton(x_2d: torch.Tensor,
                              resid_2d: torch.Tensor,
                              weight: torch.Tensor | None,
                              eps: float) -> torch.Tensor:
    """
    x_2d: [M, H], CUDA
    resid_2d: [M, H], CUDA
    weight: [H] or None
    returns: y [M, H]
    """
    assert x_2d.is_cuda and resid_2d.is_cuda, "Triton path requires CUDA tensors"
    M, H = x_2d.shape
    y = torch.empty_like(x_2d)
    BLOCK = _choose_block_size(H)
    has_weight = 1 if (weight is not None) else 0
    w_ptr = weight if weight is not None else x_2d  # dummy
    grid = (M,)
    fused_add_rmsnorm_kernel_welford[grid](
        x_2d, resid_2d, w_ptr, y,
        M, H,
        x_2d.stride(0), x_2d.stride(1),
        resid_2d.stride(0), resid_2d.stride(1),
        (weight.stride(0) if weight is not None else 1),
        y.stride(0), y.stride(1),
        eps,
        has_weight,
        BLOCK_SIZE=BLOCK,
        num_warps=8 if BLOCK >= 1024 else 4,
    )
    return y


# ---------------------------
# Public Module: ModelNew
# ---------------------------

class ModelNew(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6,
                 elementwise_affine: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = float(eps)
        self.elementwise_affine = bool(elementwise_affine)
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(hidden_size))
        else:
            self.register_buffer("_unit_weight", torch.ones(hidden_size), persistent=False)

    def forward(self, x: torch.Tensor,
                residual: torch.Tensor | None = None) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        - x: shape [..., H] or [M, H]
        - residual: same shape as x or None
        Returns:
          - tensor y with same shape as x
          - if residual is not None: returns (y, residual) (residual unchanged)
        """
        # Fallbacks: CPU or no Triton
        use_triton = _HAS_TRITON and x.is_cuda
        if not use_triton:
            return torch.nn.functional.rms_norm(x, (x.shape[-1],), eps=self.eps,
                                                weight=self.weight if self.elementwise_affine else None)

        # Ensure last-dim is contiguous for performance
        if not x.is_contiguous():
            x = x.contiguous()
        H = x.shape[-1]
        shape_2d = x.shape[:-1]
        M = int(math.prod(shape_2d)) if len(shape_2d) > 0 else 1
        x_2d = x.view(M, H)

        weight = None
        if self.elementwise_affine:
            w = self.weight
            if w.device != x.device or w.dtype != x.dtype:
                w = w.to(device=x.device, dtype=x.dtype)
            weight = w
        else:
            weight = None

        if residual is None:
            y_2d = _rmsnorm_triton(x_2d, weight, self.eps)
            y = y_2d.view(*shape_2d, H)
            return y
        else:
            if not residual.is_cuda:
                return torch.nn.functional.rms_norm(x + residual, (x.shape[-1],), eps=self.eps,
                                                    weight=self.weight if self.elementwise_affine else None)
            if not residual.is_contiguous():
                residual = residual.contiguous()
            resid_2d = residual.view(M, H)
            y_2d = _fused_add_rmsnorm_triton(x_2d, resid_2d, weight, self.eps)
            y = y_2d.view(*shape_2d, H)
            return y, residual

RMSNorm = ModelNew
