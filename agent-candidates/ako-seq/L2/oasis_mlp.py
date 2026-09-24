"""Oasis feed-forward blocks: fc1 -> GELU -> fc2, fp16, 1024 -> 4096 -> 1024.

Both GEMMs stay on vendor Blackwell kernels (cuBLAS/cuBLASLt ``nvjet_sm100_*``).
Round 2 measured that there is nothing left to take from them here -- see
ITERATIONS.md for the full tables; the short version is:

* Driving cuBLASLt directly and timing *every* heuristic candidate (64 per
  shape, 64 MB workspace) shows its default pick is already the best algo in its
  space for both GEMMs at all six M.  The picks are one-wave configs with 2-CTA
  clusters, i.e. the cluster multicast that was assumed to be out of reach is
  already what runs.  Every larger tile and every ``SPLITK_NUM`` candidate is at
  least one dispatch slot slower.
* Folding bias+GELU into fc1's epilogue (``torch._addmm_activation`` /
  ``CUBLASLT_EPILOGUE_GELU_BIAS``) does fuse, and the epilogue's GELU is within
  one fp16 ULP of both ``approximate`` modes, but cuBLASLt only serves that
  epilogue from ``cutlass3x_sm100_*`` kernels, which cost a full slot more than
  ``nvjet``; end to end it loses ~3 µs at every shape.
* CUDA-graph capture loses too, in both forms: the whole chain (which needs an
  input copy, since the harness hands a new pointer every call) and the tail
  alone (GELU+fc2 graphed, fc1 eager into the graph's static hidden buffer).
  Empty-kernel probes say a graphed launch should be ~1 µs against 2.05 µs
  eager, so the loss is the cuBLAS kernels themselves running slower when
  replayed from a graph.

So what this kernel does is what round 1 found -- both GEMMs on cuBLAS plus a
retuned in-place activation pass -- with one addition, ``_kmajor`` below.

**K-major weights.**  ``F.linear``/``addmm`` hand cuBLAS ``w.t()``, which for a
row-major ``[N, K]`` weight is non-contiguous, and it answers with a
``..._TNT`` nvjet kernel.  When ``w.t()`` is contiguous instead it picks
``..._NNT``, and that kernel is both slightly faster (0.1-0.4 µs standalone at
M <= 864) and, more importantly, *stable*: over 6 independent processes the
``w.t()`` path is bimodal at one shape per process, landing a whole 2.05 µs slot
slow (M=432 in one campaign, M=720 in the next), while the contiguous path pins
the fast branch at every shape.  Every campaign that measured it won: 8/8
processes for this form (+1.3%), 30/30 across all five campaigns for the idea,
and 4/4 alternating full benches (1.171-1.180 against the parent's 1.145-1.149,
all of the margin coming from M=288 crossing a slot).

Rather than *cache a transposed copy* -- which costs a second copy of every
weight and, worse, has to detect weight mutation, which cannot be done reliably
(``p.data.copy_(...)`` deliberately bypasses the version counter, so a
``(data_ptr, _version)`` guard silently serves a stale copy) -- ``_kmajor``
below rewrites the parameter's own storage into K-major order the first time it
is seen, in place, keeping the same ``Parameter`` object, shape, dtype, device
and values.  After that ``w.t()`` *is* contiguous, every later write lands in
that storage, and there is nothing to invalidate.  Cost is one transpose kernel
per weight on the first call and no extra resident memory; the guard is one
``stride(0)`` read per call.  ``state_dict()`` still returns the same ``[N, K]``
values, only with K-major strides.

**The activation pass.**  ``L1.gelu`` picks its launch shape from the tensor size
alone: 16 elements per thread, tuned on a 139 MB standalone tensor.  The hidden
activation here is 1.2-14 MB *and* is consumed immediately by fc2, so the pass is
latency-bound rather than bandwidth-bound and what wins is thread count, not
per-thread memory-level parallelism.  8 elements per thread drops a whole
dispatch slot at M in {432, 576, 720}; round 2 re-swept the launch shape at that
elements-per-thread and confirmed the plateau is flat (BLOCK 2048/8 warps, 4096/16
and 8192/32 tie to 0.03 µs at every shape), so BLOCK 4096 / 16 warps stands.

The pass runs in place: fc1's output is a fresh private buffer, so mutating it is
safe, and it keeps the read and write set on the same L2 lines.  A
graph-tracking caller would not see a raw pointer store, so ``requires_grad``
sends everything back to the out-of-place L1 modules.

Numerics come from ``L1.gelu``'s own fitted formulations, imported rather than
copied: the ``tanh.approx.f32`` half-angle form for ``approximate="tanh"`` (the
five 5-D captures) and the fitted quintic for exact mode (the two 3-D captures),
whose 3e-5 worst-case error is well inside one fp16 ULP.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl
from triton.language.extra.cuda import gdc_wait

from ..L1.gelu import GELU, _gelu_fast, _gelu_fast_tanh
from ..L1.linear import Linear

# 8 elements per thread over 512 threads. See the module docstring for why this
# beats L1's standalone-tuned 16-elements-per-thread shape on a hidden
# activation that fc2 consumes immediately.
_BLOCK = 4096
_WARPS = 16
# Below this the pass is short enough that L1's own small-tensor shape (BLOCK
# 512) already wins; no captured hidden tensor is anywhere near it.
_MIN_N = 524288
_FAST_DTYPES = (torch.float16, torch.bfloat16)


@triton.jit
def _gelu_inplace(X, TANH: tl.constexpr, BLOCK: tl.constexpr):
    """In-place GELU over a dense fp16/bf16 buffer whose length divides BLOCK."""
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    # fc1 may still be draining; PDL staged this grid, so wait before reading.
    gdc_wait()
    x = tl.load(X + off).to(tl.float32)
    y = _gelu_fast_tanh(x) if TANH else _gelu_fast(x)
    tl.store(X + off, y.to(X.dtype.element_ty))


def _kmajor(w: torch.Tensor) -> None:
    """Rewrite ``w`` (shape ``[N, K]``) in place so that ``w.t()`` is contiguous.

    Same ``Parameter``, same shape/dtype/device/values -- only the strides
    change, from ``(K, 1)`` to ``(1, N)``.  cuBLAS then serves the GEMM from a
    ``..._NNT`` nvjet kernel instead of ``..._TNT``; see the module docstring.
    Because this is the parameter's own storage and not a copy, a later weight
    update cannot be missed.
    """
    n, k = w.shape
    dst = torch.empty(k, n, dtype=w.dtype, device=w.device).t()
    with torch.no_grad():
        dst.copy_(w)
    w.data = dst


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
        self._tanh = approximate_tanh

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1, w2 = self.fc1.weight, self.fc2.weight
        b1, b2 = self.fc1.bias, self.fc2.bias
        if not (x.is_cuda and x.dtype is w1.dtype and x.ndim >= 2
                and x.is_contiguous() and x.shape[-1] == w1.shape[1]
                and b1 is not None and b2 is not None
                and w1.ndim == 2 and w2.ndim == 2
                and not torch.is_grad_enabled()):
            # Anything unusual (grad mode, a strided view, no bias, a dtype
            # mismatch) goes back to the L1 composition, which handles it all.
            # Note the gate is ``is_grad_enabled``, not ``requires_grad``: the
            # weights of an nn.Module are leaves that require grad by default,
            # so gating on that would send every inference call down here.
            return self.fc2(self.act(self.fc1(x)))
        # K-major once, then never again (a cast or a fresh weight resets it).
        if w1.stride(0) != 1:
            _kmajor(w1)
        if w2.stride(0) != 1:
            _kmajor(w2)
        h = torch.addmm(b1, x.view(-1, w1.shape[1]), w1.t())
        n = h.numel()
        if (h.dtype in _FAST_DTYPES and n >= _MIN_N and n % _BLOCK == 0
                and not h.requires_grad):
            _gelu_inplace[(n // _BLOCK,)](h, self._tanh, _BLOCK,
                                          num_warps=_WARPS, launch_pdl=True)
        else:
            h = self.act(h)
        return torch.addmm(b2, h, w2.t()).view(x.shape[:-1] + (w2.shape[0],))
