"""Oasis final DiT projection layer -- fused CUDA implementation.

The eager reference is a chain of ~10 tiny elementwise / GEMM launches (one of
which promotes x to fp32 just for the layernorm) on a problem holding only a
couple of microseconds of real GPU work, so it is dominated by fixed costs.
Everything is folded into two hand-written kernels -- the modulation GEMV, then
a fused layernorm + modulate + projection GEMM on tensor cores -- behind a
single dispatch.  See the .cu for why it is two kernels and not one.

The kernels handle the *frame-major* activation layout these captures use
(x strided as a dense [hidden][token] matrix per frame); any other layout,
dtype or shape falls through to the eager path below.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear
from ..L1.silu import SiLU
from ....infra.cuda_ext import lazy_op

_C = lazy_op("oasis_final_layer_fused", "oasis_final_layer_fused.cu")

_MAX_MOD_ROWS = 8  # max frames the kernel's scratch is sized for
_NCHUNK = 2        # k-chunks the modulation GEMV splits into; matches NCH6


class OasisFinalLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.linear = Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.ModuleList(
            [
                SiLU(),
                Linear(hidden_size, 2 * hidden_size, bias=True),
            ]
        )
        self._fast = None

    # -- reference path (also the fallback for shapes the kernel declines) ---
    def _eager(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        modulation = c
        for layer in self.adaLN_modulation:
            modulation = layer(modulation)
        shift, scale = modulation.chunk(2, dim=-1)
        while shift.dim() < x.dim():
            shift = shift.unsqueeze(-2)
            scale = scale.unsqueeze(-2)
        x = self.norm_final(x) * (1 + scale) + shift
        return self.linear(x)

    def _setup(self, x: torch.Tensor):
        """Bind the kernel's argument tuple once; ``False`` means stay in eager."""
        lin = self.linear
        ada = self.adaLN_modulation[1]
        ok = (
            x.is_cuda
            and x.dtype == torch.float16
            and lin.bias is not None
            and ada.bias is not None
            and lin.weight.dtype == torch.float16
            and ada.weight.dtype == torch.float16
            and self.norm_final.weight is None
            and self.norm_final.bias is None
            and float(self.norm_final.eps) == 1e-6
            and tuple(lin.weight.shape) == (64, 1024)
            and tuple(ada.weight.shape) == (2048, 1024)
        )
        if not ok:
            self._fast = False
            return
        # fp32 scratch for the modulation kernel's per-k-chunk partial sums,
        # reused across calls so the hot path performs exactly one allocation
        # (the output).  Layout: [chunk][frame][2 * hidden].
        buf = torch.empty(_NCHUNK * _MAX_MOD_ROWS * 2 * 1024,
                          device=x.device, dtype=torch.float32)
        self._fast = (buf, ada.weight, ada.bias, lin.weight, lin.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        fast = self._fast
        if fast is None:
            self._setup(x)
            fast = self._fast
        if fast is not False:
            out = _C.oasis_final(x, c, fast[0], fast[1], fast[2], fast[3], fast[4])
            if out is not None:
                return out
        return self._eager(x, c)
