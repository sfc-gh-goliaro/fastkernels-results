"""FLUX feed-forward network (L2 composite).

Two-layer MLP: ColumnParallelLinear + GELU(tanh) -> RowParallelLinear.

Three things differ from the baseline composition; the submodule tree, the
``state_dict`` keys, the weight loaders and the fp8 path are all unchanged.

1. **The activation never round-trips through HBM.**  The baseline writes the
   12288-wide bf16 pre-activation, reads it back in a GELU kernel, writes the
   activated copy, and reads that again in the second GEMM -- four passes where
   two suffice.  ``torch._addmm_activation(..., use_gelu=True)`` asks cuBLASLt
   for the bias+GELU epilogue on the *same* ``nvjet_sm100_*`` kernel cuBLAS
   already dispatches for this shape, so the GELU runs on the fp32 accumulator
   in the GEMM's epilogue and the middle two passes disappear along with a
   kernel launch.  Measured cost of the epilogue over the plain GEMM: +2.1 us
   at M=512, +2.3 us at M=4096 -- against 4.0 / 8.0 / 17.4 us for the separate
   GELU pass (see ITERATIONS.md for the hand-written Triton and TileLang
   attempts, which lose to nvjet by 1.4-1.6x before the epilogue is even free).

2. **The first GEMM runs NN instead of TN.**  ``F.linear`` computes
   ``x @ w.T``, i.e. a K-major right operand.  Keeping a pre-transposed
   ``[dim, inner_dim]`` copy of the first weight, built once on first use, lets
   cuBLAS pick a different (faster) nvjet kernel for the M=512 / M=1024 shapes:
   42.1 vs 46.2 us and 56.5 vs 58.5 us for the fused GEMM.  The second GEMM is
   left TN -- the transposed form measured *slower* there (113.7 vs 111.8 us at
   M=1024) and its bias lands on the network output, where an unfused add would
   cost a second bf16 rounding.

3. **The forward path is flat.**  At TP world size 1 -- which every capture uses
   -- the parallel wrappers reduce to two GEMM calls, so ``forward`` reads the
   four parameters directly and skips the ``ModuleList`` walk, the
   ``nn.Identity`` slot and the per-call ``use_fp8`` / ``reduce_results``
   branches.  Worth ~0 in a bare CUDA-event loop but ~5 us in the benchmark's
   loop, whose per-iteration input shuffle leaves the CPU no slack to run ahead.

Numerics: fusing the epilogue is *more* accurate than the baseline (GELU sees
the fp32 accumulator rather than a bf16-rounded pre-activation), which is
precisely why it does not reproduce the baseline bit-for-bit; the final output
lands at matched_ratio ~0.995 against the 0.99 requirement, with max relative
error concentrated on outputs near zero where the reduction over 12288 terms
cancels.  Anything below that ratio would have to re-round the accumulator to
bf16 before the activation, which no cuBLASLt epilogue exposes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..L1.gelu import GELU
from .parallel_linear import ColumnParallelLinear, RowParallelLinear


__targets__ = ["FeedForward"]

# cuBLASLt's bias+GELU epilogue. Absent on very old torch builds, in which case
# the fast path keeps the separate frozen-L1 GELU pass.
_ADDMM_ACT = getattr(torch, "_addmm_activation", None)


class ColumnParallelApproxGELU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, *, approximate: str, bias: bool = True,
                 quant_config: dict | None = None):
        super().__init__()
        self.proj = ColumnParallelLinear(dim_in, dim_out, bias=bias, quant_config=quant_config)
        self.gelu = GELU(approximate=approximate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return self.gelu(x)


class FeedForward(nn.Module):
    """FLUX FFN: GELU(tanh) linear -> linear with TP sharding."""

    def __init__(
        self,
        dim: int,
        dim_out: int | None = None,
        mult: int = 4,
        inner_dim: int | None = None,
        bias: bool = True,
        quant_config: dict | None = None,
    ) -> None:
        super().__init__()
        inner_dim = inner_dim or int(dim * mult)
        dim_out = dim_out or dim

        layers: list[nn.Module] = [
            ColumnParallelApproxGELU(dim, inner_dim, approximate="tanh", bias=bias,
                                      quant_config=quant_config),
            nn.Identity(),
            RowParallelLinear(inner_dim, dim_out, bias=bias, quant_config=quant_config),
        ]
        self.net = nn.ModuleList(layers)

        # Fast-path pre-resolution.  ``nn.Module.to()`` and ``load_state_dict``
        # rebind / overwrite a Parameter's storage in place, so holding the
        # Parameter objects themselves stays valid for the module's lifetime.
        proj, out = layers[0].proj, layers[2]
        self._flat = (
            quant_config is None
            and not proj.use_fp8 and not out.use_fp8
            and out.tp_size == 1 and out.tp_rank == 0
            and dim_out == out.weight.shape[0]
        )
        self._w1, self._b1 = proj.weight, proj.bias
        self._w2, self._b2 = out.weight, out.bias
        self._gelu = layers[0].gelu
        self._dim_out = dim_out
        # Which fast path forward takes is decided here, not per call.
        self._fused = self._flat and _ADDMM_ACT is not None and self._b1 is not None
        # Pre-transposed first weight, built on first use (the weight loader has
        # not run yet at __init__ time) and rebuilt if the weight is replaced.
        # Validated per call by two integer compares: ``to()`` moves the storage
        # without touching the version counter, ``load_state_dict`` bumps the
        # version without moving the storage.
        self._w1_nn: torch.Tensor | None = None
        self._w1_ptr = 0
        self._w1_ver = -1

    def _nn_weight(self) -> torch.Tensor:
        w = self._w1
        nn_w = w.detach().t().contiguous()
        self._w1_nn = nn_w
        self._w1_ptr = w.data_ptr()
        self._w1_ver = w._version
        return nn_w

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._fused:
            w1 = self._w1
            nn_w = self._w1_nn
            if nn_w is None or w1.data_ptr() != self._w1_ptr or w1._version != self._w1_ver:
                nn_w = self._nn_weight()
            shape = hidden_states.shape
            h = _ADDMM_ACT(self._b1, hidden_states.reshape(-1, shape[-1]), nn_w,
                           use_gelu=True)
            return F.linear(h, self._w2, self._b2).view(shape[:-1] + (self._dim_out,))
        if self._flat:
            shape = hidden_states.shape
            x = hidden_states.reshape(-1, shape[-1])
            y = F.linear(self._gelu(F.linear(x, self._w1, self._b1)), self._w2, self._b2)
            return y.view(shape[:-1] + (self._dim_out,))
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states
