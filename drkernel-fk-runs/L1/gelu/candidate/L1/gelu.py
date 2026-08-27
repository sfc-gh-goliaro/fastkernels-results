import torch
import torch.nn as nn
import torch.nn.functional as F

import triton
import triton.language as tl


# Autotune configs: try a few block sizes, warps, and vector widths.
_autotune_configs = [
    triton.Config({'BLOCK': 1024, 'VEC': 1}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK': 2048, 'VEC': 1}, num_warps=8, num_stages=2),
    triton.Config({'BLOCK': 1024, 'VEC': 2}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK': 2048, 'VEC': 2}, num_warps=8, num_stages=2),
    triton.Config({'BLOCK': 1024, 'VEC': 4}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK': 2048, 'VEC': 4}, num_warps=8, num_stages=2),
]


# Fast GELU approximation using tanh computed via exp:
# tanh(u) = 1 - 2 / (exp(2u) + 1)
# gelu(x) ≈ 0.5 * x * (1 + tanh(c * (x + a*x^3)))
# c = sqrt(2/pi), a = 0.044715
@triton.autotune(configs=_autotune_configs, key=['n_elements'])
@triton.jit
def gelu_tanh_kernel(x_ptr, y_ptr, n_elements,
                     BLOCK: tl.constexpr, VEC: tl.constexpr):
    pid = tl.program_id(0)
    # 2D tile of indices: [BLOCK, VEC]
    base = pid * BLOCK * VEC
    row = tl.arange(0, BLOCK)[:, None]
    col = tl.arange(0, VEC)[None, :]
    offs = base + row * VEC + col
    mask = offs < n_elements

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    x32 = x.to(tl.float32)

    # constants
    c = 0.7978845608028654  # sqrt(2/pi)
    a = 0.044715

    u = c * (x32 + a * x32 * x32 * x32)
    # tanh(u) = 1 - 2 / (exp(2u) + 1)
    e2u = tl.exp(2.0 * u)
    t = 1.0 - 2.0 / (e2u + 1.0)

    y32 = 0.5 * x32 * (1.0 + t)
    y = y32.to(x.dtype)
    tl.store(y_ptr + offs, y, mask=mask)


# Exact GELU via high-accuracy polynomial approximation of erf
# Abramowitz and Stegun formula 7.1.26
@triton.autotune(configs=_autotune_configs, key=['n_elements'])
@triton.jit
def gelu_erf_kernel(x_ptr, y_ptr, n_elements,
                    BLOCK: tl.constexpr, VEC: tl.constexpr):
    pid = tl.program_id(0)
    base = pid * BLOCK * VEC
    row = tl.arange(0, BLOCK)[:, None]
    col = tl.arange(0, VEC)[None, :]
    offs = base + row * VEC + col
    mask = offs < n_elements

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    x32 = x.to(tl.float32)

    inv_sqrt2 = 0.7071067811865476  # 1/sqrt(2)
    z = x32 * inv_sqrt2

    # erf approximation (A&S 7.1.26)
    p = 0.3275911
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429

    az = tl.abs(z)
    t = 1.0 / (1.0 + p * az)

    # Horner evaluation
    poly = a5 * t + a4
    poly = poly * t + a3
    poly = poly * t + a2
    poly = poly * t + a1
    poly = poly * t

    e = tl.exp(-(az * az))
    erf_az = 1.0 - poly * e

    sign = tl.where(z >= 0, 1.0, -1.0)
    erf_z = sign * erf_az

    y32 = 0.5 * x32 * (1.0 + erf_z)
    y = y32.to(x.dtype)
    tl.store(y_ptr + offs, y, mask=mask)


def _gelu_triton(x: torch.Tensor, approximate: str) -> torch.Tensor:
    """
    Triton implementation of GELU. Supports:
      - approximate="tanh"  (fast approximation, via exp)
      - approximate="none"  (exact, via polynomial erf approximation)
    Falls back to torch if tensor is not CUDA.
    """
    if not x.is_cuda:
        return F.gelu(x, approximate=approximate)

    x_contig = x.contiguous()
    y = torch.empty_like(x_contig)

    n = x_contig.numel()

    if approximate == "tanh":
        gelu_tanh_kernel[(triton.cdiv(n, 1024),)](  # grid is ignored by autotune; only shape key matters
            x_contig, y, n,
        )
    elif approximate == "none":
        gelu_erf_kernel[(triton.cdiv(n, 1024),)](
            x_contig, y, n,
        )
    else:
        raise ValueError(f"Unsupported approximate mode: {approximate!r}")

    return y.view_as(x)


class ModelNew(nn.Module):
    def __init__(self, approximate: str = "none"):
        super().__init__()
        self.approximate = approximate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _gelu_triton(x, self.approximate)

GELU = ModelNew
