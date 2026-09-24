import math
import torch
import torch.nn as nn

# Honor frozen imports
from ..L1.gelu import GELU
from ..L1.linear import Linear

# Triton setup
try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False


if _HAS_TRITON:
    @triton.jit
    def _gelu_tanh_approx_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        """
        GELU via tanh approximation:
          gelu(x) ~ 0.5 * x * (1 + tanh(c * (x + 0.044715 * x^3)))
        where c = sqrt(2/pi).
        Implement tanh using exp: tanh(z) = (e^{2z} - 1) / (e^{2z} + 1)
        """
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements

        x = tl.load(x_ptr + offs, mask=mask, other=0)
        x_f32 = x.to(tl.float32)

        c = 0.7978845608028654  # sqrt(2/pi)
        x3 = x_f32 * x_f32 * x_f32
        z = c * (x_f32 + 0.044715 * x3)

        e2z = tl.exp(2.0 * z)
        t = (e2z - 1.0) / (e2z + 1.0)

        y_f32 = 0.5 * x_f32 * (1.0 + t)
        y = y_f32.to(x.dtype)
        tl.store(y_ptr + offs, y, mask=mask)

    @triton.jit
    def _gelu_erf_approx_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        """
        GELU using erf approximation:
          gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2))).
        Abramowitz & Stegun 7.1.26 approximation for erf.
        """
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements

        x = tl.load(x_ptr + offs, mask=mask, other=0)
        x_f32 = x.to(tl.float32)

        inv_sqrt2 = 0.7071067811865476  # 1/sqrt(2)
        u = x_f32 * inv_sqrt2
        sign = tl.where(u >= 0.0, 1.0, -1.0)
        absu = tl.abs(u)

        # A&S coefficients
        p = 0.3275911
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429

        t = 1.0 / (1.0 + p * absu)
        # Horner polynomial
        poly = a5
        poly = poly * t + a4
        poly = poly * t + a3
        poly = poly * t + a2
        poly = poly * t + a1
        poly = poly * t

        exp_term = tl.exp(-(absu * absu))
        approx_erf = sign * (1.0 - poly * exp_term)

        y_f32 = 0.5 * x_f32 * (1.0 + approx_erf)
        y = y_f32.to(x.dtype)
        tl.store(y_ptr + offs, y, mask=mask)


def _choose_launch_config(n: int):
    """
    Simple heuristic for BLOCK_SIZE and num_warps.
    Larger problems -> larger blocks and more warps.
    """
    if n >= (1 << 22):      # ~4M elements and up
        return 4096, 8
    elif n >= (1 << 20):    # ~1M+
        return 4096, 4
    elif n >= (1 << 18):    # ~262k+
        return 2048, 4
    else:
        return 1024, 4


def _triton_gelu(x: torch.Tensor, approximate_tanh: bool) -> torch.Tensor:
    """
    Run GELU via Triton:
      - approximate_tanh=True: tanh-approx kernel.
      - else: erf-approx kernel (closely matches PyTorch's default "none").
    Falls back to torch if Triton is not available or tensor is not CUDA.
    """
    if (not _HAS_TRITON) or (not x.is_cuda):
        approx = "tanh" if approximate_tanh else "none"
        return torch.nn.functional.gelu(x, approximate=approx)

    x_contig = x.contiguous()
    y = torch.empty_like(x_contig)

    n = x_contig.numel()
    BLOCK_SIZE, num_warps = _choose_launch_config(n)
    grid = (triton.cdiv(n, BLOCK_SIZE),)

    if approximate_tanh:
        _gelu_tanh_approx_kernel[grid](
            x_contig, y,
            n_elements=n,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
            num_stages=2,
        )
    else:
        _gelu_erf_approx_kernel[grid](
            x_contig, y,
            n_elements=n,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
            num_stages=2,
        )
    return y


class ModelNew(nn.Module):
    """
    Triton-optimized version of the original Model:
      - Keeps cuBLAS Linear layers (fc1, fc2).
      - Replaces GELU with custom Triton elementwise kernels with tuned launch.
    Entry point is ModelNew.
    """
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        *,
        approximate_tanh: bool = False,
    ):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features

        # Use torch.nn.Linear to leverage cuBLAS
        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=True)

        self._use_triton_gelu = _HAS_TRITON
        self._approximate_tanh = approximate_tanh

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First linear
        h = self.fc1(x)
        # GELU
        if self._use_triton_gelu and h.is_cuda:
            h = _triton_gelu(h, self._approximate_tanh)
        else:
            approx = "tanh" if self._approximate_tanh else "none"
            h = torch.nn.functional.gelu(h, approximate=approx)
        # Second linear
        y = self.fc2(h)
        return y

OasisMLP = ModelNew
