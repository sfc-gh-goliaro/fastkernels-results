"""Oasis feed-forward blocks.

``OasisMLP`` is ``fc2(gelu(fc1(x)))`` with ``in=out=1024``, ``hidden=4096`` and
fp16 activations.  The captured shapes collapse to a plain row count
``M = prod(x.shape[:-1])``: 288 / 432 / 576 / 720 / 864 for the 5-D denoiser
calls (``approximate_tanh=True``) and 576 / 3456 for the 3-D calls
(``approximate_tanh=False``).

What this file does
-------------------
It keeps the baseline's composition, which routes both matmuls through the
frozen ``L1.linear`` (cuBLAS for fp16) and the activation through the frozen
``L1.gelu`` (a hand-written CUDA elementwise kernel, ~2-3x the bandwidth of
``F.gelu``).  That is the whole speedup: 1.01-1.19x, largest on the 3456-row
shape where the activation is 14 M elements.

Why nothing is fused here (measured on this B200, SM clock 1155 MHz)
-------------------------------------------------------------------
Per-kernel GPU time for one forward, from Nsight/torch profiler:

    M=288   fc1 9.0us (268 TFLOP/s)   gelu 3.5us   fc2 13.0us (186 TFLOP/s)
    M=3456  fc1 32.0us (906 TFLOP/s)  gelu 11.1us  fc2 35.1us

So the obvious L2 win -- fold ``bias + gelu`` into the fc1 epilogue and drop the
activation pass -- is worth 3.5us of 25.5us (small M) or 11us of 78us (large M),
but only if the replacement GEMM matches cuBLAS.  It does not:

* **Triton** (``tl.dot`` lowers to ``tcgen05`` on sm_100): 1.5-2x slower than
  cuBLAS on these shapes, epilogue fusion included -- 58.5us vs 33us for
  M=3456xK=1024xN=4096.  ``torch.compile(mode="max-autotune")`` restricted to
  Triton templates lands at 0.67x of eager for the whole module.
* **Hand-written tcgen05 kernel** (in ``dev/mlp.cu``: TMA loads, ``cp.async``
  multistage pipeline, UMMA into TMEM, fused bias+GELU epilogue reading TMEM).
  Correct (max err 2e-3, i.e. fp16 rounding) but 0.65-0.94x of cuBLAS+gelu on
  gemm1 and 0.41-0.74x on gemm2.  The wall is the *1-CTA* UMMA issue rate: a
  back-to-back ``tcgen05.mma.cta_group::1.kind::f16`` storm with no loads and no
  barriers sustains only ~400 TFLOP/s (480 cycles per 128x256x16 instruction,
  tensor pipe 32% busy), and two warps issuing independent streams reach just
  534 TFLOP/s.  cuBLAS' ``nvjet_*_2cta_*`` kernels get 906 TFLOP/s by pairing
  SMs with ``cta_group::2``, which halves the per-SM operand traffic; without
  that path a fused kernel cannot pay for the GEMM it replaces.
* **fp8** would double the MMA rate and halve the weight traffic, but the
  round-trip error through two K=1024/4096 contractions is ~3.5% relative,
  against a bench bound of ``atol=1e-2 + 1e-2*|y|`` on outputs whose std is
  ~0.5 -- it misses the 99%-of-elements rule.

The activation pass itself is already at the memory/latency floor: under the
bench's L2-flush timing the frozen kernel, ``F.gelu`` minus its slow ``erff``,
and eight Triton variants all land within noise of each other (9.2us for the
1.2 M-element case, of which ~7.2us is the harness' own per-measurement floor),
so there is nothing left to win in that kernel either.

Other things measured and rejected: transposing the problem so cuBLAS sees
``W @ x^T`` (1.4-1.8x slower), split-K via ``bmm`` + reduction (the reduction
costs what the split saves), chunking M so the hidden tensor stays in L2 (the
extra launches cost more than the traffic saved), and ``_addmm_activation``
(cuBLASLt's own fused GELU epilogue -- faster than the baseline at large M, but
it is a vendor kernel, not compute written here).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.gelu import GELU
from ..L1.linear import Linear


class OasisMLP(nn.Module):
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
        self.fc1 = Linear(in_features, hidden_features, bias=True)
        self.act = GELU(approximate="tanh" if approximate_tanh else "none")
        self.fc2 = Linear(hidden_features, out_features, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))
