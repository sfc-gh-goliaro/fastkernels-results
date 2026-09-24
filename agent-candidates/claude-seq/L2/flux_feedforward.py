"""FLUX feed-forward network (L2 composite) -- fastkernels candidate.

``h = GELU_tanh(x W1^T + b1)``, ``y = h W2^T + b2`` with dim=3072,
inner_dim=12288, bf16, at M = 512 / 1024 / 4096 rows.

Where the time goes (B200, measured per kernel inside the benchmarked
sequence with Nsight Compute, M=512 / 1024 / 4096):

    GEMM1 + bias   32.8 /  56.2 / 198.0 us      (nvjet_sm100 ..._2cta_..._bias)
    GELU           4.6  /   7.8 /  28.3 us
    GEMM2 + bias   33.2 /  57.3 / 213.0 us
    ----------------------------------------
    forward       80.9  / 123.9 / 422.9 us      (harness, incl. launch gaps)

So >90% of the operator is two tensor-core GEMMs, and the reference ones are
*good*: ``sm__pipe_tensor_cycles_active`` per *active* SM cycle is 82-85%, i.e.
the 2-CTA ``tcgen05`` mainloop is essentially saturated while it runs.  The only
structural slack is wave quantisation (``smsp__cycles_active`` is 73% at M=512,
77% at M=1024, 91% at M=4096), which a stream-K schedule could recover -- but
only from a mainloop that is itself at ~85% per-cycle efficiency.

Every replacement GEMM this workspace could produce was measured against that
bar and lost, so the two GEMMs keep the reference path:

* Triton (``tl.dot``, which does lower to ``tcgen05`` here): swept tiles
  64..256 x 64..256 x 32..128, 4/8 warps, 2..6 stages, persistent / plain,
  epilogue subtiling, ``warp_specialize``, ``num_ctas``, GROUP_M.  Best config
  (128x256x64, 4 warps, 4 stages, persistent, subtiled) reaches 0.61-0.68x of
  cuBLAS end to end: 31% / 46% / 62% tensor-pipe utilisation against the
  reference's 59% / 69% / 89% at equal SM occupancy, i.e. the gap is mainloop
  throughput, not scheduling.  ``num_ctas=2`` (the 2-SM MMA the reference uses)
  fails to compile in this Triton with TMA-store epilogues.
* Fusing the activation into a GEMM1 epilogue only pays if that GEMM lands
  within the activation's own cost of the reference (within 10% at M=512, 14%
  at M=1024, 13% at M=4096) -- the 0.65x kernel above is 30-40% short, so the
  fused variants measured 0.82-0.94x of this candidate.
* Splitting rows across 2-4 streams so one chunk's activation overlaps the next
  chunk's GEMM: 0.59-1.09x (the concurrent GEMMs lose more to smaller tiles and
  L2 contention than the overlap wins).
* ``torch._addmm_activation(..., use_gelu=True)`` (cuBLASLt's own GELU+bias
  epilogue, which does remove the pass): 1.024x / 1.104x / 0.991x of the
  baseline where this candidate measured 1.053x / 1.067x / 1.003x in the same
  process -- a wash, and its activation differs from
  ``F.gelu(approximate="tanh")`` by a bf16 ulp, which drops the scorer's matched
  ratio from 1.00000 to 0.99638 (threshold 0.99).
* Weight layout (``w.t()`` view vs. a K-major copy), ``torch.addmm`` vs.
  ``F.linear``, the cuBLASLt backend and ``CUBLAS_WORKSPACE_CONFIG`` (in case a
  bigger workspace unlocked a split-K algorithm at M=512) are all within noise.

What this candidate does change is the activation pass, which *is* ours:

* ``flux_feedforward.cu`` runs GELU(tanh) **in place** over the FFN's private
  [M, 12288] temporary: 32 B per thread (``LDG.E.128``/``STG.E.128``), fp32 math
  with a single hardware ``tanh.approx.f32``, grid-stride over a capped grid --
  bit-identical to the frozen L1 GELU, but with half the L2 footprint (the lines
  GEMM1 just wrote are the lines GEMM2 reads back) and one 12-100 MB allocation
  less per call.  Over the [M, 12288] buffer standalone it runs at the copy
  roofline -- 8.5 / 12.5 / 39.2 us against ``F.gelu``'s 13.6 / 22.9 / 82.3 us
  (accurate ``erff``, 4-byte accesses) -- and 4.6 / 7.8 / 28.3 us inside the
  sequence, where the buffer is still partly L2-resident.
* ``forward`` flattens to 2-D once and issues the two GEMMs as ``torch.addmm``
  directly, so a call is 3 kernels and ~4 dispatches instead of the baseline's
  module-per-layer walk.

The TP / FP8 paths (``tp_size > 1``, ``quant_config``) are untouched: they keep
the baseline's ``ColumnParallelLinear`` / ``RowParallelLinear`` forwards, which
own the all-reduce and the block-scaled FP8 GEMM.  Parameters keep the baseline
names (``net.0.proj.{weight,bias}``, ``net.2.{weight,bias}``) so the scorer's
``load_state_dict`` still shares weights.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..L1.gelu import GELU
from .parallel_linear import ColumnParallelLinear, RowParallelLinear


__targets__ = ["FeedForward"]


# ---------------------------------------------------------------------------
# In-place GELU(tanh) extension (falls back to the frozen L1 GELU if the build
# is unavailable -- no GPU / no nvcc).
# ---------------------------------------------------------------------------
try:
    from fastkernels.infra.cuda_ext import load_op

    _C = load_op("fk_l2_flux_ffn", "flux_feedforward.cu")
except Exception:  # pragma: no cover - no CUDA toolchain
    _C = None


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

        # Fast path only for the plain (non-quantised, single-rank) FFN, which is
        # what the captured FLUX configuration is.
        down = layers[2]
        self._fast = (
            _C is not None
            and not layers[0].proj.use_fp8
            and not down.use_fp8
            and down.tp_size == 1
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._fast:
            up, down = self.net[0], self.net[2]
            w_up, b_up = up.proj.weight, up.proj.bias
            if hidden_states.dtype == w_up.dtype and hidden_states.is_contiguous():
                x = hidden_states.reshape(-1, hidden_states.shape[-1])
                h = (torch.addmm(b_up, x, w_up.t()) if b_up is not None
                     else torch.mm(x, w_up.t()))
                if not _C.gelu_tanh_(h):          # dtype/layout we do not cover
                    h = up.gelu(h)
                w_dn, b_dn = down.weight, down.bias
                out = (torch.addmm(b_dn, h, w_dn.t()) if b_dn is not None
                       else torch.mm(h, w_dn.t()))
                return out.view(*hidden_states.shape[:-1], out.shape[-1])

        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states
