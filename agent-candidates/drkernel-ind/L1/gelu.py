import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Autotune configurations for elementwise GELU
_tune_configs = [
    triton.Config({'BLOCK_SIZE': 128},  num_warps=4, num_stages=2),
    triton.Config({'BLOCK_SIZE': 256},  num_warps=4, num_stages=2),
    triton.Config({'BLOCK_SIZE': 512},  num_warps=4, num_stages=2),
    triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=2),
    triton.Config({'BLOCK_SIZE': 2048}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=2),
]


@triton.autotune(configs=_tune_configs, key=['n_elements'])
@triton.jit
def _gelu_exact_kernel(x_ptr, y_ptr, n_elements,
                       BLOCK_SIZE: tl.constexpr):
    # Program id and offsets
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load and upcast to f32 for math
    x = tl.load(x_ptr + offs, mask=mask, other=0)
    x32 = x.to(tl.float32)

    # Compute z = x / sqrt(2)
    inv_sqrt2 = 0.7071067811865476  # 1/sqrt(2)
    z = x32 * inv_sqrt2

    # Erf approximation (Abramowitz-Stegun 7.1.26)
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911

    az = tl.abs(z)
    t = 1.0 / (1.0 + p * az)
    # Polynomial P(t)
    pt = a5 * t + a4
    pt = pt * t + a3
    pt = pt * t + a2
    pt = pt * t + a1
    pt = pt * t

    e = tl.exp(-az * az)
    erf_az = 1.0 - pt * e
    sign = tl.where(z >= 0, 1.0, -1.0)
    erf_z = sign * erf_az

    # GELU exact: y = 0.5 * x * (1 + erf(z))
    y32 = 0.5 * x32 * (1.0 + erf_z)

    # Cast back and store
    y = y32.to(x.dtype)
    tl.store(y_ptr + offs, y, mask=mask)


@triton.autotune(configs=_tune_configs, key=['n_elements'])
@triton.jit
def _gelu_tanh_kernel(x_ptr, y_ptr, n_elements,
                      BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    x = tl.load(x_ptr + offs, mask=mask, other=0)
    x32 = x.to(tl.float32)

    # constants
    c = 0.7978845608028654  # sqrt(2/pi)
    a = 0.044715

    x2 = x32 * x32
    x3 = x2 * x32
    inner = c * (x32 + a * x3)
    # tanh(inner) = (exp(2*inner) - 1) / (exp(2*inner) + 1)
    e2 = tl.exp(2.0 * inner)
    t = (e2 - 1.0) / (e2 + 1.0)

    y32 = 0.5 * x32 * (1.0 + t)
    y = y32.to(x.dtype)
    tl.store(y_ptr + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, approximate: str = "none"):
        super().__init__()
        if approximate not in ("none", "tanh"):
            raise ValueError(f"approximate must be 'none' or 'tanh', got {approximate!r}")
        self.approximate = approximate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fallback if Triton not available or tensor not on CUDA
        if (not TRITON_AVAILABLE) or (not x.is_cuda):
            return F.gelu(x, approximate=self.approximate)

        # Ensure contiguous
        x_contig = x.contiguous()
        y = torch.empty_like(x_contig)

        n_elements = x_contig.numel()
        if n_elements == 0:
            return y.view_as(x)

        # Grid is 1D over elements
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

        if self.approximate == "tanh":
            _gelu_tanh_kernel[grid](
                x_contig, y, n_elements,
            )
        else:
            # approximate == "none": exact GELU via erf approximation in kernel
            _gelu_exact_kernel[grid](
                x_contig, y, n_elements,
            )

        return y.view_as(x)

GELU = ModelNew
