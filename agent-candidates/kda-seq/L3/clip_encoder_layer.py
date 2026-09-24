"""CLIP text-encoder layer (pre-norm attention + MLP, two residuals) for B200 / sm_100.

The captured problem is tiny -- ``[1, 77, 768]`` fp32, 12 heads of 64, 3072-wide MLP, about
0.4 GFLOP -- and the baseline spends **19 kernel launches** on it: two ``layer_norm``, three
projection GEMMs, two batched GEMMs for QK^T and PV, the softmax, the ``out_proj`` and
``fc1``/``fc2`` GEMMs, and eight elementwise kernels (``* scale``, ``+ mask``, the
``transpose(1, 2).contiguous()`` copy, two residual adds, and the three that make up
QuickGELU). Measured in the bench's own loop (``profile/00-baseline/probe_baseline.py``) the
module costs 224.6-232.7 us; an ``identity`` module in the same loop costs 12.3 us, and the
19 kernels account for only 116.9 us of GPU time. Wrapping those same 19 kernels in a CUDA
graph costs 144.1 us with bitwise-identical output
(``profile/00-baseline/probe_headroom.py``), so roughly 80 us of the baseline is exposed host
launch gap. **Launch count, not arithmetic, is the lever.**

This file runs the operator in **ten launches**. Traced GPU time below, at an SM clock of
1155 MHz -- these are only comparable to each other and to the 1155 MHz row further down,
for the reason the next section explains:

    1. ``F.layer_norm``                                     6.50 us
    2. one fused cuBLAS GEMM on ``cat([W_q, W_k, W_v])``    9.06 us   (was 3 GEMMs, 30 us)
    3. one Triton kernel: QK^T, scale, mask, softmax, PV,
       and a store straight into ``[S, embed]`` layout     11.04 us   (was 5 kernels, 28 us)
    4. ``out_proj`` through cuBLAS + the residual add       9.12 + 2.94 us
    5. ``F.layer_norm``                                     6.05 us
    6. ``fc1`` through cuBLAS                               8.70 us
    7. one Triton QuickGELU                                 3.36 us   (was 3 kernels, 9.5 us)
    8. ``fc2`` through cuBLAS + the residual add           16.16 + 2.66 us

against the baseline's 19 kernels. Those ten launches are then captured in a CUDA graph and
replayed as one, so the host issues four launches per call: two copies into the static input
buffers, the replay, and one copy of the static result into freshly allocated storage the caller
owns.

**The speedup is clock-dependent, and that is the single most important thing to know about the
numbers below.** All terms measured inside one process, on two leases:

    SM clock   baseline                        candidate                        ratio
    1155 MHz   197.76 us (116.7 GPU, 19 launches)  110.46 us (74.7 GPU, 10)     1.79x
    1965 MHz   229.68 us ( 70.7 GPU, 19 launches)   77.52 us (45.3 GPU, 10)     2.96x

GPU time scales with clock; host launch cost does not. The baseline carries 19 launches to this
implementation's 10, so at a high clock the baseline's fixed host cost is a larger share of its
total and the ratio rises. That is why the same unchanged code measures anywhere from **1.79x to
2.96x** depending on which lease it lands on, and why a reviewer independently measured 68.8 us
and 3.57x. Absolute microseconds are only comparable *within* one process; across processes only
ratios mean anything, and even those move with the clock.

Consequently the honest summary is a range, not a point: **1.79x-2.96x measured, PASSED in every
run**. The plan's 2.4x target is met on high-clock leases and missed on low-clock ones. Within a
single process the zero-launch-cost ceiling -- harness floor plus GPU time, i.e. what the layer
would cost if launches were free -- is 86.99 us against that process's 197.76 us baseline, so
**2.27x** is the most this pipeline can reach at that clock; getting there reliably needs GPU time
under about 70 us against the current 74.7. Removing launches cannot do it. The two candidates for
removing *kernel* work are recorded under the residual switches below and in ``docs/plan.md``, and
neither is in this revision.

Graph capture earns its place by a paired measurement rather than by two independent medians:
alternating capture on and off inside one process, seven repetitions, capture is ahead by a median
**+8.10 us with a 1.64 us standard deviation and wins 7 of 7 reps**, and its output is bitwise
identical to the eager path's.

Numerically the gate is *pinned*, not merely bounded, and that constraint shapes every line
below. The bench compares against the baseline with ``atol=1e-5 + rtol=1e-3*|ref|`` and needs
99% of elements inside it. The output is ``x + attn_branch + mlp_branch`` with ``|x| ~ 1``, so
0.76% of elements have ``|ref| < 0.01`` where the bound collapses to about 2e-5 -- and torch's
effective fp32 matmul policy here is **tf32**, so the reference is not IEEE fp32 either.
Measured over three seeds (``profile/00-baseline/probe_stage_numerics.py``), *both* directions
fail: bf16 operands score 0.381, fp16 0.909, exact fp32 everywhere 0.912-0.915, and computing
both LayerNorms in fp64 and rounding back scores 0.978-0.996. Being **more** accurate fails
just as hard as being less accurate.

Why it is that brittle: a tf32 operand ulp is ``2**-11 ~ 4.9e-4`` relative, about 4000x
coarser than an fp32 ulp. Perturbing a GEMM's A operand by ``d`` flips the tf32 rounding of a
fraction ``d / 4.9e-4`` of its elements, and each flip moves one product by a full tf32 ulp.
So deviation out of a GEMM grows like ``sqrt(d)``, not ``d``, and every downstream GEMM
re-amplifies it. Two GEMMs later a quarter-ulp input difference is a ~1e-4 output difference,
which is exactly the ``|ref| < 0.1`` failure band.

The per-stage error budget that follows is measured, not assumed -- relative noise injected at
one intermediate at a time, scored end to end under the bench rule
(``profile/01-error-budget/probe_budget.py``). Each stage below carries the budget it must
meet, and the measurement it came from:

    hn1 = LN1(x)        bitwise. 3e-8 relative noise already scores 0.9785, and one fp32 ulp
                        is 1.2e-7 -- the budget is below an ulp, so ``F.layer_norm`` is called
                        rather than reproduced. Reproducing ATen's Welford tree in Triton is
                        left to later work; it is worth ~11 us and two launches.
    qkv                 <= 3e-8 (1e-7 scores 0.9906). Bitwise, because the fused projection
                        stays on cuBLAS.
    prob (softmax)      <= 1e-7 (1e-6 scores 0.9891).
    attn (PV out)       <= 1e-6 (1e-5 scores 0.9832).
    h (post-residual)   <= 3e-7 (1e-7 scores 0.9989).
    hn2 = LN2(h)        <= 1e-7 (1e-6 scores 0.9894).
    fc1, g              <= 1e-6 (1e-5 scores 0.9888).

Single-stage budgets are necessary but not sufficient -- real per-stage errors are correlated,
not independent -- so the authority is always an end-to-end ``validate.py`` run, never a sum
of these numbers. ``profile/02-stage-numerics/`` holds the per-stage comparison this file is
re-checked against after every change.

What the harness actually reports for this revision, on three seeds:

* ``hn1``, ``qkv``, ``logits`` and ``prob`` are **exactly bitwise** -- ``torch.equal`` against
  the reference on every seed and, for ``prob``, on every tile configuration in the sweep. The
  softmax reduction reproduction is therefore not approximately right, it is exact.
* ``attn`` -- the PV dot -- is the **only** stage that introduces any deviation: 6e-8 to 9e-8 max
  absolute, 7.9e-7 to 1.1e-6 relative to the stage's own RMS, and identical across every tile
  configuration in the sweep.

  **What is measured about that deviation** (``profile/10-pv-bitwise/``), stated without the
  overreach an earlier revision of this comment committed -- it claimed the deviation was impossible
  to remove from *any* single-launch fused attention, which the experiment does not establish: The reference contracts PV over exactly ``k = 77``;
  a fused kernel holds a power-of-two probability tile and must contract over 128 with the tail
  zeroed. Zero-padding the contraction axis **alone changes torch's own matmul result** (1.4e-9 to
  1.9e-9) -- so the reference and a padded contraction are different computations before Triton is
  involved at all. And the Triton dot reproduces the padded ``torch.matmul`` **bitwise for every
  k-chunking ``tl.dot`` can express** (it requires ``K >= 16``, so tiles of 16, 32 and 64 were
  tested and all agree exactly). The kernel is therefore already exactly right against
  identically-padded operands; what remains is the padding, which is the price of the fusion. The
  alternative -- leaving PV on ``torch.matmul`` -- buys bitwise identity back for about five
  launches and the transpose copy this kernel eliminates, i.e. it undoes the optimisation.

  What this does **not** establish is that no kernel can contract exactly 77. Adding exact zeros to
  an fp32 accumulator is exact, so a schedule issuing only the reference's k-steps should agree, and
  the plan permits CUDA C++ extensions where the ``mma.sync.m16n8k8`` k-step sequence is chosen
  freely -- Triton's ``tl.dot`` cannot express it (``K >= 16``, power-of-two tiles), but that is a
  Triton limitation, not a mathematical one. An exact-K=77 PV via a local CUDA extension is
  therefore **open work, not a closed impossibility**, and this stage is judged against AC-2.1's
  documented relative budget, which it currently misses on one of three seeds.
* Every stage *after* ``attn`` comes out **bitwise** when fed the reference's own ``attn``
  (``sec_attribution`` in the probe). So the downstream arithmetic -- both residual adds, LN2,
  ``fc1``, QuickGELU, ``fc2`` -- contributes no error of its own, and the whole end-to-end
  deviation is the propagated image of that single stage. This is why the downstream stages are
  not judged against the injected-noise budgets: those budgets answer "how much *fresh* noise can
  this stage absorb", which is the wrong question for an error inherited from upstream.
* End to end, and this is the number that decides acceptance: ``matched_ratio`` **0.998410,
  0.999188 and 0.999104** across three scored ``validate.py`` runs, against the harness's 0.99
  gate -- passing with about a 6x margin on the failure fraction. On the probe's own three seeds
  the same build scores 0.999324 to 0.999882. The 0.999 figure this work adopted as a tripwire is
  missed on one of the three runs, and that is stated rather than smoothed: it is the floor set by
  the PV padding above, not an open defect.

The frozen ``candidate/L2/clip_attention.py`` was measured at this level rather than assumed
unusable, as the plan required. Its attention-stage deviation here is 5.8e-6 to 1.3e-5 max
absolute -- roughly **100x looser than this kernel's 6e-8 to 9e-8** -- which is the
decision-relevant comparison and is why a purpose-built kernel ships. Its end-to-end matched ratio
at this level is **0.998630 / 0.998309 / 0.999527** -- settled, once the probe's weight load was made
strict and its construction seed pinned; the earlier conflicting 1.000000 reading came from a
``strict=False`` load leaving unpinned weights. A residual ~3e-4 wobble across repeats remains and is
recorded as an open queued item (suspected buffer-alignment-dependent cuBLAS kernel selection), but
it does not move the third decimal place or the conclusion.

Three measurements make the bitwise strategy tractable
(``profile/01-error-budget/probe_gemm_layout.py``, ``probe_triton_gemm.py``):

* ``tl.dot(..., input_precision="tf32")`` **truncates** its fp32 operands to tf32 where cuBLAS
  round-to-nearest-evens them. Rounding both operands to nearest even *before* the dot makes
  Triton bitwise identical to cuBLAS on all four GEMM shapes -- 312/312 sweep configurations,
  max abs error exactly 0. Without the pre-rounding, none match. Identity is insensitive to
  tiling because the tf32 MMA rounds once per ``k = 8`` group in increasing ``k``, so any
  ``BK`` that is a multiple of 8 gives the same rounding sequence. That removes tile choice
  from the numerics question entirely and leaves it a pure speed/register question.
* Pre-rounding a *weight* to tf32 once, offline, does not change cuBLAS's own result -- so the
  cached pre-transposed, pre-rounded ``[K, N]`` copies the Triton GEMM reads are numerically
  free. Folding ``bias`` into a Triton epilogue as ``acc + bias`` matches cuBLAS's fused bias
  epilogue bitwise.
* ``libdevice.exp`` is bitwise identical to ``torch.exp`` (0 mismatches over 2**18 values).
  ``tl.exp`` is **not** (65% mismatch, max 1.3e-6 relative), nor is ``tl.exp2(x * log2e)``,
  and ``tl.sigmoid`` is not either (33%, 5.3e-7). Against a ``prob <= 1e-7`` budget, ``tl.exp``
  in a fused softmax is on its own enough to fail, so every transcendental here goes through
  ``libdevice``.

The softmax denominator is the one place where reproducing ATen means reproducing a *reduction
order*, so it is written out explicitly rather than left to ``tl.sum``. For a 77-long axis,
``at::native::softmax_warp_forward<float,float,float,7,false,false>`` (in the bundled
``ATen/native/cuda/PersistentSoftmax.cuh``) instantiates with ``WARP_SIZE = 32`` and
``WARP_ITERATIONS = 4``, and its load is
``element_index = local_idx + it * WARP_SIZE``. Lane ``l`` therefore owns the **strided**
elements ``l, l+32, l+64, l+96`` -- not four consecutive ones. Out-of-range slots load as
``-inf``, and the accumulation loop over them is unguarded, so they enter the sum as
``exp(-inf - max) = 0.0`` exactly and the padding is a true no-op. The per-lane partial is a
left-associated chain over ``it = 0, 1, 2, 3`` seeded from ``0.0f``, the cross-lane combine is
an XOR butterfly at offsets 16, 8, 4, 2, 1, and normalisation is a true per-element division
(not a reciprocal multiply). ``_softmax_denominator`` below reproduces all three.

Anything the fast path is not written for runs ``_reference()``, which is the baseline's
formula in explicit functional torch ops reading the parameters directly. It deliberately does
**not** call submodule forwards, and the parameter tree is built from local holders rather than
by importing the level-1 and level-2 winners: inside the candidate package
``..L1.layer_norm`` and ``..L2.clip_attention`` resolve to those *frozen candidate kernels*,
which are documented as not reproducing ATen, so routing a fallback through them would compare
one candidate against another instead of against the reference. (``..L2.clip_mlp`` has no
candidate file and would alias to baseline -- the asymmetry is exactly why neither resolution
is relied on here.) The holders register exactly ``weight`` and ``bias``, so ``state_dict``
keys still match the baseline's; that parity is tested rather than assumed, because the
harness wraps its ``load_state_dict`` in a bare ``except Exception: pass`` and a key mismatch
would fail *silently* as a numerics failure.
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTextConfig

_HEAD_DIM = 64
_KEY_BLOCK = 128        # the padded key axis the softmax reduction is shaped for, and so max S

# The fast path admits exactly the captured configuration. Comparing one ``torch.Size``
# against one constant is cheaper than unpacking a shape and bounds-checking it, and on a
# ~90 us budget the per-call host cost is worth that.
_CAPTURED_INPUT = torch.Size((1, 77, 768))
_CAPTURED_MASK = torch.Size((1, 1, 77, 77))
_CAPTURED_SEQ = 77
_CAPTURED_EMBED = 768
_CAPTURED_HEADS = 12
_CAPTURED_INTERMEDIATE = 3072

# Which implementation each residual stage ships, decided per stage by measurement rather than
# by symmetry. Timed in the harness's own loop, interleaved with the baseline and repeated five
# times so clock drift cancels (``profile/05-stage-variants/results-timing.txt``); the spread
# within a variant was under 0.03 us, so these differences are real:
#
#   stage4=cuBLAS stage8=cuBLAS   110.40 us   1.818x   10 launches   <- ships
#   stage4=Triton stage8=cuBLAS   118.82 us   1.689x    9 launches
#   stage4=cuBLAS stage8=Triton   163.81 us   1.225x    9 launches
#   stage4=Triton stage8=Triton   174.14 us   1.152x    8 launches
#
# **Saving a launch loses at both stages.** The launch trace explains it: the fused GEMM costs
# 21.92 us on the out_proj shape against cuBLAS's 9.54 plus a 2.91 us add, and it is far worse
# on fc2's K = 3072. At M = 77 there are only 36 CTAs on 148 SMs and the K loop is fully
# exposed, so this kernel is paying 12 dependent global loads where cuBLAS pipelines them. That
# contradicts the estimate this work started from, which had Triton winning stage 4 at 10.06 us
# -- so the switch is set from the measurement, and the estimate is recorded as not reproduced.
#
# Both variants stay in the file behind these switches, because the kernel is already
# bitwise-equal to cuBLAS-plus-a-separate-add on both shapes -- so turning it on later is a pure
# speed decision with no numerics work attached.
#
# What that speed work is, specifically: this kernel uses plain pointer arithmetic, a plain
# ``tl.dot``, and three pipeline stages, which is the bottom of the known progression for
# Blackwell matmul -- no pipelining 46% of peak, a 3-stage TMA/MMA overlap 62%, and warp
# specialization 98% (KernelWiki ``wiki/techniques/pipeline-stages.md``). Triton 3.6 is also
# explicit that its SM100 ``tcgen05``/TMEM lowering is only reliably reached through
# descriptor/TMA plus ``tl.range(..., warp_specialize=True)``, and that a plain ``tl.dot`` is
# *not* proven to lower that way (``sources/docs/triton-3.6-blackwell.md``). So the lever is a
# rewrite around ``tl.make_tensor_descriptor`` and warp specialization, not another tile sweep --
# retuning was tried and did not close the gap. Split-K stays out regardless: it inserts extra
# fp32 roundings and breaks the identity the gate depends on. See
# ``profile/08-kernelwiki/consultation.md``.
#
# Note that neither stage can be folded into one cuBLAS call: torch exposes either a matrix
# ``beta * C`` term or a fused bias epilogue, not both, so ``torch.addmm(residual, g, W.t())``
# would drop the bias and ``F.linear`` cannot take the residual. cuBLAS here is always two
# kernels.
_STAGE4_TRITON = False
_STAGE8_TRITON = False

# CUDA-graph capture of the fast path. One captured graph writing one static output, which every
# call clones before returning -- see the note above ``_replay`` for why a clone rather than a ring
# of static buffers.
#
# On by measurement, not by default: interleaved against the eager path five times
# (``profile/05-stage-variants/results-timing2.txt``), replay is 98.53 us against eager's
# 110.48 us -- 11.95 us apart where the spread within either variant is under 0.15 us -- and its
# output is bitwise identical to the eager path's. 2.092x against 1.866x.
_USE_CUDA_GRAPH = True

# tf32 is the reference's arithmetic, and the *effective cuBLAS policy* is what decides it --
# not ``torch.get_float32_matmul_precision()``. Those two disagree: setting
# ``torch.backends.cuda.matmul.fp32_precision = "ieee"`` leaves the generic getter at "high"
# while cuBLAS switches to IEEE fp32, and a kernel still mimicking tf32 would then be scored
# against an exact reference and land near 0.91. The generic setter propagates the other way
# ("highest" -> "ieee", "high"/"medium" -> "tf32"), so reading the backend value alone covers
# both APIs. Resolved once, here, because the public property costs 302 ns per read against
# 128 ns for the private getter it forwards to.
try:
    _fp32_policy = torch._C._get_fp32_precision_getter
    _fp32_policy("cuda", "matmul")
except Exception:  # private getter gone or renamed: fall back to the documented property
    def _fp32_policy(_backend: str, _op: str) -> str:
        return torch.backends.cuda.matmul.fp32_precision


# Swept over BM in {4, 8, 16, 32} x num_warps in {4, 8}, timed end to end through the harness's
# own loop. Deliberately not a compute-peak search: at 36 MFLOP this kernel is latency bound, and
# Nsight Compute says so plainly (``profile/06-ncu-encoder-v1/analysis/``) -- for this
# configuration, 60 CTAs, **0.203 waves per multiprocessor**, DRAM read at 0.7% of peak, 12.6%
# achieved occupancy, and top stall reasons long_scoreboard 1.43 and wait 1.29. Most of the
# machine is idle while dependent loads resolve.
#
# That profile argues for *smaller* tiles, to buy CTAs, and BM=4 does deliver on its own terms:
# 240 CTAs, 0.541 waves, L2 hit up from 32% to 56%, SM throughput up from 10% to 21%. **It does
# not translate into latency.** A paired A/B alternating the two configurations inside one
# process, nine repetitions (``profile/07-tile-sweep/results-ab.txt``), gives a median paired
# difference of +0.13 us against a 2.24 us standard deviation, with BM=4 ahead in 1 of 9 reps --
# the whole BM in {4, 8, 16} x warps in {4, 8} grid lands between 100 and 106 us. Two earlier
# sweeps in separate processes each declared a different winner, which is what a non-difference
# looks like. Recorded rather than resolved: the occupancy headroom is real and unspent, and it
# is the *dependent-load chain*, not the CTA count, that is binding.
#
# So the tile is chosen on the profile rather than on the clock: 119 registers against BM=4's
# 164 (more headroom if the kernel ever grows), **zero spilled bytes** either way, and far
# steadier run-to-run timing (0.36 us standard deviation against 2.14). The [BM, 128] fp32
# logits tile stacked on two [128, 64] operand tiles is the shape that would normally spill, and
# it does not here. There is no K loop to pipeline (S <= 128 is one key block), so ``num_stages``
# is not a free parameter.
#
# Numerics are independent of this choice, which is the point of the pre-rounding: every
# configuration in the sweep produced the *identical* 4.470e-08 max absolute deviation on the
# attention output, bitwise-identical softmax probabilities, and the identical end-to-end
# matched ratio.
_ATTN_BM = 16
_ATTN_WARPS = 8

# The residual-epilogue GEMM's tile, swept over the same kind of grid. BK is a multiple of 8,
# which is what keeps the tf32 rounding sequence identical to cuBLAS's.
_GEMM_BM = 32
_GEMM_BN = 64
_GEMM_BK = 64
_GEMM_WARPS = 4
_GEMM_STAGES = 3

_QUICKGELU_BLOCK = 1024

# Host-side diagnostics: plain ints, no device sync, nothing the harness's integrity guards
# watch. Without them a failed build degrades to the fallback and validation "passes" at
# baseline speed, which reads as a win rather than as an untested run.
_FASTPATH_HITS: dict[str, int] = {}
_FALLBACK_HITS: dict[str, int] = {}
_CACHE_BUILDS: dict[str, int] = {}

_ATTENTION = None
_QUICKGELU = None
_GEMM_RESIDUAL = None
_KERNEL_STATUS = "disabled:not-built"


def _count(table: dict[str, int], key: str) -> None:
    table[key] = table.get(key, 0) + 1


def kernel_status() -> str:
    """The build outcome, for probes and for the bench log. Not used on any hot path."""
    return _KERNEL_STATUS


def graph_generation(module) -> int:
    """How many times *module* has captured a graph. For tests that must prove a re-capture."""
    return getattr(module, "_graph_generation", 0)


def diagnostics() -> dict[str, dict[str, int]]:
    """Fast-path / fallback / cache-build counts. Host-side ints; reading them syncs nothing."""
    return {"fastpath": dict(_FASTPATH_HITS), "fallback": dict(_FALLBACK_HITS),
            "cache_builds": dict(_CACHE_BUILDS)}


# The counters exist so a degraded run cannot look like a win, but the bench runs each operator
# in a subprocess whose Python objects the caller never sees -- so opt in to having them printed
# at exit. Off unless asked for, one stderr line, after all timing is over.
if os.environ.get("FK_L3_ENCODER_DIAGNOSTICS"):
    import atexit

    @atexit.register
    def _dump_diagnostics() -> None:
        print(f"[candidate L3/clip_encoder_layer] status={_KERNEL_STATUS} "
              f"stage4={'triton' if _STAGE4_TRITON else 'cublas'} "
              f"stage8={'triton' if _STAGE8_TRITON else 'cublas'} "
              f"graph={'on' if _USE_CUDA_GRAPH else 'off'} {diagnostics()}",
              file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Rounding fp32 operands to tf32, to nearest even, on the host.
#
# The same arithmetic the kernels apply, so a pre-rounded cached weight and an in-kernel
# rounding of the same weight are the identical bit pattern -- which is what lets the GEMM skip
# rounding its B operand per launch. Asserted rather than assumed: the probe in
# ``profile/02-stage-numerics/`` checks host-against-kernel agreement and idempotence over a wide
# range plus the NaN, signed-zero, subnormal and overflow tails.
# ---------------------------------------------------------------------------
def _round_tf32_host(t: torch.Tensor) -> torch.Tensor:
    """Round every element of an fp32 tensor to tf32's 10-bit mantissa, to nearest even."""
    bits = t.contiguous().view(torch.int32)
    # (half - 1) plus the lowest kept bit, then clear the dropped 13 bits. Two's complement
    # addition increments the magnitude for negative floats too, because the sign bit is
    # untouched by a carry that stops inside the mantissa -- so this is symmetric.
    rounded = ((bits + 0x0FFF + ((bits >> 13) & 1)) & -8192).view(torch.float32)
    # A NaN whose payload lives only in the dropped 13 bits (0x7f800001, say) would round to
    # infinity, and 0x7fffffff carries all the way into the sign bit and lands on -0.0.
    # Infinities and finite overflow are already correct -- overflowing to infinity is what
    # round-to-nearest does -- so only NaN needs the select.
    return torch.where(t == t, rounded, t)


# ---------------------------------------------------------------------------
# Kernels.
# ---------------------------------------------------------------------------
def _build_kernels():
    """Compile the three Triton kernels and return their launchers, or raise."""
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    @triton.jit
    def _to_tf32_rne(x):
        """Round fp32 to tf32's 10-bit mantissa, to nearest even.

        ``tl.dot`` truncates its operands; cuBLAS rounds to nearest even. Doing it here first
        makes the two agree. Idempotent: after one application the dropped 13 bits are zero, so
        the increment cannot carry into the mantissa a second time and this composes safely
        with whatever the tensor core does to an already-representable operand.
        """
        b = x.to(tl.int32, bitcast=True)
        rounded = ((b + 0x0FFF + ((b >> 13) & 1)) & -8192).to(tl.float32, bitcast=True)
        return tl.where(x == x, rounded, x)

    @triton.jit
    def _softmax_denominator(e, BM: tl.constexpr, BN: tl.constexpr):
        """Sum a [BM, 128] tile of exponentials in ATen's warp-softmax order.

        Two orders have to be reproduced, and neither is what a plain ``tl.sum`` would do.

        First the per-lane fold. Lane ``l`` owns elements ``l, l+32, l+64, l+96`` -- strided,
        because ATen indexes ``local_idx + it * WARP_SIZE`` -- and folds them left to right
        from a ``0.0f`` seed. Adding that seed is exact here (every exponential is >= 0), so
        the chain is ``((e0 + e1) + e2) + e3``, which is *not* the balanced
        ``(e0 + e1) + (e2 + e3)`` a 4-wide reduction would produce. The four slices are pulled
        out with ``tl.split``, which needs the axis it splits to be last and of size 2, hence
        the permute.

        Then the cross-lane combine, an XOR butterfly at offsets 16, 8, 4, 2, 1. At offset 16
        lane ``l`` computes ``s[l] + s[l ^ 16]``; for ``l < 16`` that is ``s[l] + s[l + 16]``
        and for ``l >= 16`` the same sum with the operands swapped, which is bitwise identical
        because IEEE-754 addition is commutative. Lanes ``l`` and ``l ^ 16`` therefore hold the
        same value afterwards, and the invariant carries to every later offset -- so reshaping
        to ``[2, half]`` and reducing that size-2 axis reproduces the butterfly exactly. A
        size-2 reduction has only one pairing and both operand orders agree, so ``tl.sum`` is
        safe *here* specifically, where it is not safe over 4 or 128.

        ``BN`` is threaded through as a ``constexpr`` rather than read from the module: Triton
        refuses to close over a plain global int, and the width has to be a compile-time
        constant for the reshapes. The unrolled offsets below assume ``BN // 4 == 32``, i.e.
        one warp's worth of lanes, which is what a 77-long axis padded to 128 gives.
        """
        strided = tl.permute(tl.reshape(e, (BM, 4, BN // 4)), (0, 2, 1))
        pairs = tl.reshape(strided, (BM, BN // 4, 2, 2))
        even, odd = tl.split(pairs)          # it in {0, 2} and it in {1, 3}
        e0, e2 = tl.split(even)
        e1, e3 = tl.split(odd)
        partial = ((e0 + e1) + e2) + e3      # [BM, 32], left-associated as ATen folds it
        # The five butterfly levels, unrolled. Written out rather than looped because Triton's
        # codegen only iterates ``range`` / ``tl.static_range``, and because each line is one
        # shuffle offset -- 16, 8, 4, 2, 1 -- which is easier to check against ATen this way.
        partial = tl.sum(tl.reshape(partial, (BM, 2, 16)), axis=1)
        partial = tl.sum(tl.reshape(partial, (BM, 2, 8)), axis=1)
        partial = tl.sum(tl.reshape(partial, (BM, 2, 4)), axis=1)
        partial = tl.sum(tl.reshape(partial, (BM, 2, 2)), axis=1)
        partial = tl.sum(tl.reshape(partial, (BM, 2, 1)), axis=1)
        return partial                       # [BM, 1]

    @triton.jit
    def _fused_attention(QKV, MASK, OUT, seq, embed, qkv_row, mask_row, out_row, scale,
                         HAS_MASK: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                         HD: tl.constexpr):
        """One program per (row tile, head): QK^T, scale, mask, softmax, PV, transposed store.

        Deliberately *not* flash-style. ``S <= 128`` is a single key block, so there is nothing
        to tile over and no online rescaling to do, and a plain one-pass softmax keeps the
        arithmetic on the sequence the reference actually performs.
        """
        head = tl.program_id(1)
        rows = tl.program_id(0) * BM + tl.arange(0, BM)
        cols = tl.arange(0, BN)
        feat = head * HD + tl.arange(0, HD)
        row_ok = rows < seq
        col_ok = cols < seq

        # Q rows past S load as 0.0, not -inf: a fully padded row tile still has its softmax
        # evaluated before the store drops it, and it has to stay finite to avoid a NaN.
        q = tl.load(QKV + rows[:, None] * qkv_row + feat[None, :],
                    mask=row_ok[:, None], other=0.0)
        k = tl.load(QKV + cols[:, None] * qkv_row + (embed + feat)[None, :],
                    mask=col_ok[:, None], other=0.0)
        v = tl.load(QKV + cols[:, None] * qkv_row + (2 * embed + feat)[None, :],
                    mask=col_ok[:, None], other=0.0)

        logits = tl.dot(_to_tf32_rne(q), _to_tf32_rne(tl.trans(k)), input_precision="tf32")
        # Scale after the dot, then the mask -- the baseline computes ``bmm(q, k^T) * scale``
        # and only then adds the mask. The harness materialises any float argument named
        # *mask* as all ones, and softmax is invariant to a uniform shift, so the add is
        # algebraically a no-op; it is performed anyway because the fp32 addition perturbs low
        # bits exactly as the reference's does, and those bits are what the gate measures.
        logits = logits * scale
        if HAS_MASK:
            logits += tl.load(MASK + rows[:, None] * mask_row + cols[None, :],
                              mask=row_ok[:, None] & col_ok[None, :], other=0.0)
        # Only now the -inf tail, so padded columns cannot perturb the scale or the mask add.
        logits = tl.where(col_ok[None, :], logits, float("-inf"))
        # tl.max replaces ATen's sequential fold plus Max butterfly. Max is associative and
        # commutative on totally-ordered floats, so any order agrees -- but only while every
        # row holds at least one finite value and no NaN or +inf. Signed zero is the one
        # asymmetry (ATen's ``a < b ? b : a`` returns -0.0 for ``max(-0.0, +0.0)`` and +0.0 for
        # the reverse), and it is unobservable here: ``x - (-0.0)`` and ``x - (+0.0)`` agree for
        # every x that is not itself a zero, and ``exp`` of either zero is exactly 1.0.
        #
        # The precondition cannot be established by a shape check, and testing it per call would
        # need a device sync, so it is not in the admission predicate. It holds for the scored
        # workload by construction: the harness materialises any float argument named *mask* as
        # all ones, and the padded tail is the only -inf this kernel introduces, so all 77 real
        # columns stay finite. A caller passing a mask with -inf still gets a correct result as
        # long as each row keeps one finite entry; a fully-masked row, a NaN, or a +inf makes
        # both this kernel and ATen produce NaN -- which the bench rejects outright rather than
        # scoring, so the failure is loud and cannot be a silently wrong finite answer.
        #
        # exp goes through libdevice: tl.exp mismatches torch.exp on 65% of inputs.
        m = tl.max(logits, 1)
        e = libdevice.exp(logits - m[:, None])
        # Padded lanes are exp(-inf - m) == 0.0 exactly, so they contribute nothing here.
        #
        # tl.div_rn, not ``/``. Plain division lowers to ``div.full.f32``, which is the ~2-ulp
        # approximate divide, while ATen's store does a correctly rounded ``elements / sum``.
        # Against a 1e-7 budget on this stage that difference alone would fail; div_rn lowers
        # through precise_divf to ``div.rn.f32``. (``tl.fdiv`` is not the same thing.)
        prob = tl.div_rn(e, _softmax_denominator(e, BM, BN))

        acc = tl.dot(_to_tf32_rne(prob), _to_tf32_rne(v), input_precision="tf32")
        # Straight into [S, embed] layout, which is what transpose(1, 2).contiguous() produces
        # -- so the baseline's copy kernel has nowhere to go rather than moving.
        tl.store(OUT + rows[:, None] * out_row + feat[None, :], acc, mask=row_ok[:, None])

    @triton.jit
    def _quickgelu(X, OUT, n, BLOCK: tl.constexpr):
        """``x * sigmoid(1.702 * x)`` in one launch, in place of the reference's three.

        The multiply order is the reference's: ``1.702 * x``, then ATen's sigmoid as
        ``1 / (1 + expf(-x))``, then ``x * s``. Reassociating or folding the reciprocal would
        change the rounding sequence, and ``tl.sigmoid`` mismatches ``torch.sigmoid`` on 33% of
        inputs, so the form is written out and ``libdevice.exp`` is used.
        """
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        keep = offs < n
        x = tl.load(X + offs, mask=keep)
        s = tl.div_rn(1.0, 1.0 + libdevice.exp(-(1.702 * x)))
        tl.store(OUT + offs, x * s, mask=keep)

    @triton.jit
    def _gemm_residual(A, B, BIAS, RES, C, M, N, K, a_row, b_row, res_row, c_row,
                       BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                       B_PREROUNDED: tl.constexpr, HAS_RES: tl.constexpr):
        """``C = (A @ B + bias) + residual`` with tf32 operands rounded as cuBLAS rounds them.

        ``B`` is the pre-transposed ``[K, N]`` weight copy, so both operands stream
        contiguously. The epilogue order is the reference's: cuBLAS fuses ``bias`` into the
        GEMM and the residual add is a separate kernel afterwards, so it is ``acc + bias``
        first and ``+ residual`` second -- the reverse is a different fp32 result.

        The K loop runs in increasing ``k`` with ``BK`` a multiple of 8, which is what makes
        the accumulation order match: the tf32 MMA rounds once per ``k = 8`` group, so every
        such ``BK`` produces the same rounding sequence and tiling drops out of the numerics.
        """
        rows = tl.program_id(0) * BM + tl.arange(0, BM)
        cols = tl.program_id(1) * BN + tl.arange(0, BN)
        row_ok = rows < M
        col_ok = cols < N
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in tl.range(0, K, BK):
            ks = k0 + tl.arange(0, BK)
            k_ok = ks < K
            a = tl.load(A + rows[:, None] * a_row + ks[None, :],
                        mask=row_ok[:, None] & k_ok[None, :], other=0.0)
            b = tl.load(B + ks[:, None] * b_row + cols[None, :],
                        mask=k_ok[:, None] & col_ok[None, :], other=0.0)
            if not B_PREROUNDED:
                b = _to_tf32_rne(b)
            acc += tl.dot(_to_tf32_rne(a), b, input_precision="tf32")
        acc += tl.load(BIAS + cols, mask=col_ok, other=0.0)[None, :]
        if HAS_RES:
            acc += tl.load(RES + rows[:, None] * res_row + cols[None, :],
                           mask=row_ok[:, None] & col_ok[None, :], other=0.0)
        tl.store(C + rows[:, None] * c_row + cols[None, :], acc,
                 mask=row_ok[:, None] & col_ok[None, :])

    # ``enable_fp_fusion=False`` on every launch. Triton defaults it to True and *does*
    # contract an ``fmul`` feeding an ``fadd`` into ``fma.rn.f32`` even with no fast-math flags
    # on the IR -- verified in the compiled PTX. Here that would fuse ``logits * scale`` into
    # the mask add, which the reference performs as two separate elementwise kernels with an
    # fp32 store between them. For this operator ``scale`` happens to be 0.125, so the multiply
    # is exact and the contraction is numerically harmless -- but that is a property of one
    # constant, not a guarantee, and the flag also passes ``--fmad=false`` to ptxas, which
    # costs nothing measurable in three latency-bound kernels. It does not affect ``tl.dot``:
    # the MMA is not an fmad.
    def attention(qkv, mask, out, seq, embed, num_heads, scale, bm=_ATTN_BM,
                  warps=_ATTN_WARPS):
        _fused_attention[(triton.cdiv(seq, bm), num_heads)](
            qkv, mask if mask is not None else qkv, out,
            seq, embed, qkv.stride(0), mask.stride(2) if mask is not None else 0,
            out.stride(0), scale,
            HAS_MASK=mask is not None, BM=bm, BN=_KEY_BLOCK, HD=_HEAD_DIM,
            num_warps=warps, enable_fp_fusion=False)
        return out

    def quickgelu(x, out):
        n = x.numel()
        _quickgelu[(triton.cdiv(n, _QUICKGELU_BLOCK),)](
            x, out, n, BLOCK=_QUICKGELU_BLOCK, enable_fp_fusion=False)
        return out

    def gemm_residual(a, b_kn, bias, residual, out, *, prerounded=True,
                      bm=_GEMM_BM, bn=_GEMM_BN, bk=_GEMM_BK, warps=_GEMM_WARPS,
                      stages=_GEMM_STAGES):
        m, k = a.shape
        n = b_kn.shape[1]
        _gemm_residual[(triton.cdiv(m, bm), triton.cdiv(n, bn))](
            a, b_kn, bias, residual if residual is not None else a, out,
            m, n, k, a.stride(0), b_kn.stride(0),
            residual.stride(0) if residual is not None else 0, out.stride(0),
            BM=bm, BN=bn, BK=bk, B_PREROUNDED=prerounded,
            HAS_RES=residual is not None, num_warps=warps, num_stages=stages,
            enable_fp_fusion=False)
        return out

    attention.kernel = _fused_attention
    quickgelu.kernel = _quickgelu
    gemm_residual.kernel = _gemm_residual
    return attention, quickgelu, gemm_residual


def _warm(attention, quickgelu, gemm_residual) -> None:
    """Launch every reachable configuration once, then synchronize.

    Triton compiles on first launch. Doing it at import rather than on the harness's first
    correctness forward keeps compilation out of the timed window entirely, and happens while
    the bench worker is still producing output -- clear of the watchdog that stalls on a quiet
    log. The harness's ten warmup iterations would also cover it; this only makes a build
    failure surface earlier and louder.
    """
    embed = _CAPTURED_EMBED
    # Both the captured length and the kernel's widest supported one. Triton specializes on
    # argument divisibility, so a launch warmed only at M = 128 (a multiple of 16) does not cover
    # M = 77 (not one) -- that variant would compile on its first real call. Harmless here,
    # because the harness's warmup iterations precede its timed ones, but warming the shape the
    # operator actually runs keeps compilation out of the timed window by construction rather
    # than by relying on that ordering.
    for seq in (_CAPTURED_SEQ, _KEY_BLOCK):
        qkv = torch.zeros((seq, 3 * embed), device="cuda", dtype=torch.float32)
        out = torch.empty((seq, embed), device="cuda", dtype=torch.float32)
        mask = torch.zeros((1, 1, seq, seq), device="cuda", dtype=torch.float32)
        for arg in (mask, None):
            attention(qkv, arg, out, seq, embed, embed // _HEAD_DIM, _HEAD_DIM ** -0.5)
        wide = torch.zeros((seq, _CAPTURED_INTERMEDIATE), device="cuda", dtype=torch.float32)
        quickgelu(wide, torch.empty_like(wide))
        # Both residual GEMM shapes, since either switch can be on.
        for k, weight_shape in ((embed, (embed, embed)),
                                (_CAPTURED_INTERMEDIATE, (_CAPTURED_INTERMEDIATE, embed))):
            a = torch.zeros((seq, k), device="cuda", dtype=torch.float32)
            weight = torch.zeros(weight_shape, device="cuda", dtype=torch.float32)
            bias = torch.zeros((embed,), device="cuda", dtype=torch.float32)
            gemm_residual(a, weight, bias, out, torch.empty_like(out))
    torch.cuda.synchronize()


def _init_kernels() -> None:
    """Resolve the kernels once, at import. Any failure degrades to the reference path."""
    global _ATTENTION, _QUICKGELU, _GEMM_RESIDUAL, _KERNEL_STATUS
    if not torch.cuda.is_available():
        _KERNEL_STATUS = "disabled:no-cuda-device"
        return
    try:
        attention, quickgelu, gemm_residual = _build_kernels()
        _warm(attention, quickgelu, gemm_residual)
    except Exception as exc:  # compiler, driver, or architecture rejected a kernel
        # Flattened to one physical line: a Triton or ptxas failure is routinely multiline
        # (source excerpts, carets, a nested traceback) and interpolating it raw would break
        # the one-line promise exactly when the message matters most.
        detail = " ".join(str(exc).split())
        _KERNEL_STATUS = f"failed:{type(exc).__name__}: {detail}"[:400]
        # One line, at import, on stderr: the bench worker routes this to the per-operator log,
        # so a swallowed build failure stays visible instead of hiding behind a silent 1.00x.
        print(f"[candidate L3/clip_encoder_layer] kernels unavailable, delegating to torch: "
              f"{_KERNEL_STATUS}", file=sys.stderr, flush=True)
        return
    _ATTENTION, _QUICKGELU, _GEMM_RESIDUAL = attention, quickgelu, gemm_residual
    _KERNEL_STATUS = "built"


_init_kernels()


# ---------------------------------------------------------------------------
# Parameter holders.
#
# Local rather than imported so the fallback cannot reach a frozen candidate kernel, and so no
# level-1 CUDA extension has to build for this file to import. They register exactly ``weight``
# and ``bias``, which is what makes ``state_dict`` keys identical to the baseline's tree of
# ``Linear`` and ``LayerNorm`` submodules.
# ---------------------------------------------------------------------------
class _ParamPair(nn.Module):
    """``weight`` + ``bias``, plus the flags it raises when either object is replaced.

    ``m.q_proj.weight = nn.Parameter(...)`` changes neither the old object's version counter
    nor its storage pointer, so nothing a caller could poll would notice it and a cached copy
    built from the old object would go on being used. Catching it in ``__setattr__`` costs
    nothing per forward.

    Each flag is a one-element list shared with an owner rather than a back-reference, so it
    survives ``deepcopy`` pointing at the copy's own owner. There is a *list* of them because a
    holder can be owned by more than one module at once: ``a.fc1 = b.fc1`` leaves both modules
    holding it and both of their caches derived from it. Owners are therefore added, never
    replaced, and de-duplicated by identity.
    """

    def __init__(self, weight_shape, bias_shape, dirty: list, *, ones: bool = False):
        super().__init__()
        self._owners = [dirty]
        # Mirrors how the baseline's own holders initialise, so that the harness's
        # uninitialised-parameter sanitiser sees the same thing on both modules: the linears
        # allocate with ``empty`` and rely on a weight load, the norms start at ones/zeros.
        if ones:
            self.weight = nn.Parameter(torch.ones(weight_shape))
            self.bias = nn.Parameter(torch.zeros(bias_shape))
        else:
            self.weight = nn.Parameter(torch.empty(weight_shape))
            self.bias = nn.Parameter(torch.empty(bias_shape))

    def register_owner(self, dirty: list) -> None:
        for existing in self._owners:
            if existing is dirty:
                return
        self._owners.append(dirty)

    def owned_by(self, dirty: list) -> bool:
        for existing in self._owners:
            if existing is dirty:
                return True
        return False

    def _mark_owners(self) -> None:
        for dirty in self.__dict__.get("_owners", ()):
            dirty[0] = True

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name in ("weight", "bias"):
            self._mark_owners()

    def _apply(self, *args, **kwargs):
        # In a ``finally`` because a partially-applied module is exactly the case that must not
        # keep using the cache.
        try:
            return super()._apply(*args, **kwargs)
        finally:
            self._mark_owners()

    def _load_from_state_dict(self, *args, **kwargs):
        # Also in a ``finally``: a pre-hook that mutates a weight and then raises would
        # otherwise leave the cache both stale and trusted. Nothing here can raise on its own,
        # which matters because the harness wraps the whole load in a bare
        # ``except Exception: pass`` -- a hook that raised would be swallowed and the stale
        # cache would survive as a silent numerics failure.
        try:
            super()._load_from_state_dict(*args, **kwargs)
        finally:
            self._mark_owners()


class _HolderGroup(nn.Module):
    """A named group of ``_ParamPair`` holders, matching one baseline submodule's key prefix."""

    _MEMBERS: tuple[str, ...] = ()

    def __init__(self, dirty: list):
        super().__init__()
        self._dirty = dirty

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name in self._MEMBERS:
            dirty = self.__dict__.get("_dirty")
            if dirty is not None:
                dirty[0] = True
                if isinstance(value, _ParamPair):
                    # Added, never replaced: the donating module still owns it too, and its own
                    # cache is still derived from it.
                    value.register_owner(dirty)

    def _apply(self, *args, **kwargs):
        try:
            return super()._apply(*args, **kwargs)
        finally:
            self.__dict__.get("_dirty", [True])[0] = True


class _AttentionParams(_HolderGroup):
    """``q_proj`` / ``k_proj`` / ``v_proj`` / ``out_proj``, the baseline attention key prefix."""

    _MEMBERS = ("q_proj", "k_proj", "v_proj", "out_proj")

    def __init__(self, config: CLIPTextConfig, dirty: list):
        super().__init__(dirty)
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        for name in self._MEMBERS:
            setattr(self, name, _ParamPair((self.embed_dim, self.embed_dim),
                                           (self.embed_dim,), dirty))


class _MlpParams(_HolderGroup):
    """``fc1`` / ``fc2``, the baseline MLP key prefix."""

    _MEMBERS = ("fc1", "fc2")

    def __init__(self, config: CLIPTextConfig, dirty: list):
        super().__init__(dirty)
        hidden, inter = config.hidden_size, config.intermediate_size
        self.fc1 = _ParamPair((inter, hidden), (inter,), dirty)
        self.fc2 = _ParamPair((hidden, inter), (hidden,), dirty)


# ---------------------------------------------------------------------------
class CLIPEncoderLayer(nn.Module):
    """A CLIP text-encoder layer: pre-norm attention and MLP, each with a residual.

    Nothing derived from parameter *values* is computed in ``__init__``. The harness constructs
    the module, moves it, casts it, rewrites uninitialised parameters with
    ``normal_(0, 0.02)``, and only *then* loads the baseline's state dict -- anything
    precomputed here would be built from ``torch.empty`` garbage.
    """

    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.eps = config.layer_norm_eps
        self.normalized_shape = (self.embed_dim,)

        # Raised by the holders when a parameter object is swapped out, and by this module's
        # ``__setattr__`` when a whole group is. Created before the submodules, which capture it.
        self._dirty = [True]

        self.self_attn = _AttentionParams(config, self._dirty)
        self.layer_norm1 = _ParamPair((self.embed_dim,), (self.embed_dim,), self._dirty,
                                      ones=True)
        self.mlp = _MlpParams(config, self._dirty)
        self.layer_norm2 = _ParamPair((self.embed_dim,), (self.embed_dim,), self._dirty,
                                      ones=True)

        # Config-derived, not weight-derived, and fixed after construction. The head geometry
        # has to be checked separately from the input shape: a config with 16 heads of 48 also
        # presents a [1, 77, 768] input, and hidden_size=769 with 12 heads also gives
        # head_dim 64, so neither is caught by the shape comparison alone.
        attn = self.self_attn
        self._fast_ok = (self.embed_dim == _CAPTURED_EMBED
                         and attn.num_heads == _CAPTURED_HEADS
                         and attn.head_dim == _HEAD_DIM
                         and attn.num_heads * attn.head_dim == self.embed_dim
                         and config.intermediate_size == _CAPTURED_INTERMEDIATE)
        self._stage4_triton = _STAGE4_TRITON
        self._stage8_triton = _STAGE8_TRITON

        # Built on first use, never here. Plain attributes, so they stay out of ``state_dict``:
        # registering the fused weight as a parameter or buffer would add a key the baseline
        # does not have, and the harness's weight sharing is keyed on exact parity.
        self._cached = None
        self._cached_key = None
        self._graphs = None
        # Capture failures are recorded per key, never globally -- see ``_replay``.
        self._graphs_failed: set = set()
        # Bumped on every successful capture, so a test can prove a re-capture happened rather
        # than inferring it. Never read on the accepting path.
        self._graph_generation = 0

    # -- cache coherence ------------------------------------------------------
    #
    # Carried by *invalidation*, not by polling. Per call the accepting path reads one cached
    # reference and one list index. The alternative -- six ``(data_ptr, _version)`` reads --
    # was measured at 0.33-0.88 us in the frozen level-2 file, which is 3-9% of a ~90 us
    # budget, so it is not affordable here and is not needed: in this harness ``_apply`` and
    # the single ``load_state_dict`` both complete before the first forward, so a lazy build
    # alone is already correct and these hooks are defence in depth.
    #
    # What invalidates: ``_apply`` and ``_load_from_state_dict`` on this module and on every
    # holder; ``_ParamPair.__setattr__`` for a replaced ``weight``/``bias`` object; and
    # ``__setattr__`` on this module and on each group for a replaced submodule.
    #
    # What is left uncovered, and why it is acceptable here:
    #   * Ordinary in-place parameter writes (``p.normal_()``, ``p.copy_()``) outside a
    #     state-dict load. The scored workload never does this after the first forward -- the
    #     harness's own sanitiser runs before any forward, and its weight sharing goes through
    #     ``load_state_dict``, which invalidates.
    #   * Writes through the ``.data`` alias (``p.data.mul_(2)``) and ``p.data = X``. These
    #     bump no version counter *and* preserve ``data_ptr``, so no per-call check at any
    #     price detects them.
    #   * Publication across CUDA streams: the rebuild is enqueued on the calling stream and
    #     published to the host immediately, so another stream consuming the cache establishes
    #     no dependency on the ``cat``. Single-stream use, which is what the bench does, is
    #     ordered correctly.
    #   * Forward-mode AD, for which this build exposes no cheap predicate. Reverse mode is
    #     guarded by ``torch.is_grad_enabled()``.
    _GROUPS = ("self_attn", "layer_norm1", "mlp", "layer_norm2")

    def _invalidate(self) -> None:
        self._cached = None
        self._cached_key = None
        self._dirty[0] = True
        # Belt and braces: graph validity is already carried by weight-bundle identity in
        # ``_replay``, so dropping it here is redundant -- but it makes the invariant local and
        # costs one store.
        self._graphs = None

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name in self._GROUPS:
            dirty = self.__dict__.get("_dirty")
            if dirty is not None:
                dirty[0] = True
                self._cached = None
                if isinstance(value, (_ParamPair, _HolderGroup)):
                    register = getattr(value, "register_owner", None)
                    if register is not None:
                        register(dirty)

    def _apply(self, *args, **kwargs):
        try:
            return super()._apply(*args, **kwargs)
        finally:
            self._invalidate()

    def _load_from_state_dict(self, *args, **kwargs):
        try:
            super()._load_from_state_dict(*args, **kwargs)
        finally:
            self._invalidate()

    # -- the fused weights ----------------------------------------------------
    def _cache_key(self):
        """What the published bundle's contents depend on, beyond the parameters themselves.

        ``_rebuild`` decides whether to build the pre-transposed ``out_kn``/``fc2_kn`` copies from
        the stage switches, so a bundle is only valid for the switch settings it was built under.
        An earlier revision checked only the dirty flag here, so flipping a switch kept serving the
        old bundle and ``_pipeline`` silently kept the old implementation -- while ``_replay``,
        which *did* key on the switches, dutifully re-captured the unchanged pipeline. A test that
        compared graph against eager could not see it, because both read the same stale bundle.
        """
        return (self._stage4_triton, self._stage8_triton)

    def weights(self):
        """The cached weight bundle, rebuilt if stale. The out-of-band accessor, for probes.

        ``forward`` does not call this -- it uses ``_admit``, which folds the same two lines
        into the admission check so the timed path pays one Python call instead of two. The
        duplication is deliberate.
        """
        cached = self._cached
        if cached is not None and not self._dirty[0] and self._cached_key == self._cache_key():
            return cached
        return self._rebuild()

    def _rebuild(self):
        """Build the fused weight bundle, or return None if the parameters are not what the
        kernels assume -- in which case the caller delegates rather than launching.

        The shape checks are not defensive noise: a ``q_proj`` replaced by a 767-output holder
        makes the concatenation 2303 columns wide while the kernel indexes 2304, which is an
        out-of-bounds read. They are cheap here because this runs once, not per call.
        """
        if not self._fast_ok:
            return None
        dirty = self._dirty
        attn, mlp = self.self_attn, self.mlp
        if not isinstance(attn, _AttentionParams) or not isinstance(mlp, _MlpParams):
            return None
        # Only cache from holders that participate in the invalidation protocol and that this
        # module is a registered owner of. A plain ``nn.Linear`` assigned over a holder, or a
        # holder this module never registered with, cannot be relied on to report a later
        # parameter replacement -- so nothing is cached and every call delegates.
        holders = []
        for group, names in ((attn, _AttentionParams._MEMBERS), (mlp, _MlpParams._MEMBERS)):
            for name in names:
                holder = getattr(group, name, None)
                if not isinstance(holder, _ParamPair) or not holder.owned_by(dirty):
                    return None
                holders.append(holder)
        for norm in (self.layer_norm1, self.layer_norm2):
            if not isinstance(norm, _ParamPair) or not norm.owned_by(dirty):
                return None
            holders.append(norm)
        q, k, v, out_proj, fc1, fc2, ln1, ln2 = holders

        embed, inter = self.embed_dim, _CAPTURED_INTERMEDIATE
        expected = ((q, (embed, embed)), (k, (embed, embed)), (v, (embed, embed)),
                    (out_proj, (embed, embed)), (fc1, (inter, embed)), (fc2, (embed, inter)),
                    (ln1, (embed,)), (ln2, (embed,)))
        device = q.weight.device
        for holder, shape in expected:
            w, b = holder.weight, holder.bias
            bias_shape = (shape[0],)
            if (w is None or b is None or w.shape != shape or b.shape != bias_shape
                    or w.dtype is not torch.float32 or b.dtype is not torch.float32
                    or w.device != device or b.device != device):
                return None

        with torch.no_grad():
            qkv_weight = torch.cat([q.weight, k.weight, v.weight], 0)
            qkv_bias = torch.cat([q.bias, k.bias, v.bias], 0)
            # Pre-transposed, pre-rounded [K, N] copies for whichever residual stages run on
            # Triton. Rounding a weight to tf32 offline does not change cuBLAS's own result, so
            # these are numerically free, and they let the GEMM skip rounding its B operand on
            # every launch.
            out_kn = (_round_tf32_host(out_proj.weight.t().contiguous())
                      if self._stage4_triton else None)
            fc2_kn = (_round_tf32_host(fc2.weight.t().contiguous())
                      if self._stage8_triton else None)

        self._dirty[0] = False
        self._cached_key = self._cache_key()
        self._cached = (qkv_weight, qkv_bias, out_proj.weight, out_proj.bias,
                        fc1.weight, fc1.bias, fc2.weight, fc2.bias,
                        ln1.weight, ln1.bias, ln2.weight, ln2.bias, out_kn, fc2_kn)
        _count(_CACHE_BUILDS, "clip_encoder_layer")
        return self._cached

    # -- the one accepting-path helper ---------------------------------------
    def _admit(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None):
        """Admission and the warm-cache lookup in one call: the weight bundle, or None.

        One function rather than two because a Python call is ~70 ns and the timed path should
        pay for one. Ordered cheapest-first. The exact-``torch.Tensor`` check is on *both*
        arguments: they are independent, and a plain input paired with a
        ``Tensor._make_subclass`` mask would otherwise take the fast path, while a wrapper
        subclass may have no usable storage pointer at all.
        """
        if _ATTENTION is None:
            return None
        if hidden_states.dtype is not torch.float32 or type(hidden_states) is not torch.Tensor:
            return None
        if hidden_states.shape != _CAPTURED_INPUT:
            return None
        device = hidden_states.get_device()          # -1 when not on CUDA
        if device < 0 or not hidden_states.is_contiguous():
            return None
        # An inference-only path: it builds no autograd graph, and under autocast the reference
        # would run in a different dtype entirely.
        if torch.is_grad_enabled() or torch.is_autocast_enabled():
            return None
        # Mimicking cuBLAS's tf32 is only correct while cuBLAS is on tf32. An allow-list of
        # exactly "tf32" rather than a deny-list of "ieee", so a future backend precision this
        # was never measured against delegates instead of being assumed compatible. Wrapped
        # because mixing the legacy ``allow_tf32`` API with the new one makes the accessor
        # itself raise, and an unreadable policy is not one to bet on.
        try:
            if _fp32_policy("cuda", "matmul") != "tf32":
                return None
        except Exception:
            return None
        if attention_mask is not None:
            if (type(attention_mask) is not torch.Tensor
                    or attention_mask.dtype is not torch.float32
                    or attention_mask.shape != _CAPTURED_MASK
                    or not attention_mask.is_contiguous()
                    or attention_mask.get_device() != device):
                return None
        cached = self._cached
        if cached is not None and not self._dirty[0] and self._cached_key == self._cache_key():
            return cached
        return self._rebuild()

    # -- forward --------------------------------------------------------------
    def _pipeline(self, weights, residual: torch.Tensor,
                  attention_mask: torch.Tensor | None, out: torch.Tensor) -> torch.Tensor:
        """The fused sequence, on ``[S, embed]`` operands, writing its result into ``out``.

        ``out`` is passed in rather than returned so a captured graph can write into a buffer whose
        address is stable across replays, which is what capture requires.
        """
        (qkv_weight, qkv_bias, out_weight, out_bias, fc1_weight, fc1_bias,
         fc2_weight, fc2_bias, ln1_weight, ln1_bias, ln2_weight, ln2_bias,
         out_kn, fc2_kn) = weights
        seq, embed = _CAPTURED_SEQ, _CAPTURED_EMBED

        normed = F.layer_norm(residual, self.normalized_shape, ln1_weight, ln1_bias, self.eps)
        # One 2304-wide cuBLAS GEMM in place of three 768-wide ones: bitwise identical to the
        # baseline's, so Q, K and V reach the attention kernel bit-for-bit.
        qkv = F.linear(normed, qkv_weight, qkv_bias)
        attn = torch.empty((seq, embed), device=qkv.device, dtype=qkv.dtype)
        _ATTENTION(qkv, attention_mask, attn, seq, embed, _CAPTURED_HEADS,
                   self.self_attn.scale)

        if out_kn is not None:
            hidden = torch.empty((seq, embed), device=qkv.device, dtype=qkv.dtype)
            _GEMM_RESIDUAL(attn, out_kn, out_bias, residual, hidden)
        else:
            hidden = F.linear(attn, out_weight, out_bias)
            hidden = hidden.add_(residual)

        normed = F.layer_norm(hidden, self.normalized_shape, ln2_weight, ln2_bias, self.eps)
        projected = F.linear(normed, fc1_weight, fc1_bias)
        gated = _QUICKGELU(projected, torch.empty_like(projected))

        if fc2_kn is not None:
            _GEMM_RESIDUAL(gated, fc2_kn, fc2_bias, hidden, out)
        else:
            # ``out=`` rather than ``add_``: the add has to land in the caller's buffer, and this
            # is the same two kernels either way.
            torch.add(F.linear(gated, fc2_weight, fc2_bias), hidden, out=out)
        return out

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weights = self._admit(hidden_states, attention_mask)
        if weights is None:
            _count(_FALLBACK_HITS, "clip_encoder_layer")
            return self._reference(hidden_states, attention_mask)
        seq, embed = _CAPTURED_SEQ, _CAPTURED_EMBED
        residual = hidden_states.view(seq, embed)

        if _USE_CUDA_GRAPH:
            replayed = self._replay(weights, residual, attention_mask)
            if replayed is not None:
                _count(_FASTPATH_HITS, "clip_encoder_layer")
                return replayed.view(1, seq, embed)

        out = torch.empty((seq, embed), device=residual.device, dtype=residual.dtype)
        self._pipeline(weights, residual, attention_mask, out)
        _count(_FASTPATH_HITS, "clip_encoder_layer")
        return out.view(1, seq, embed)

    # -- graph replay ---------------------------------------------------------
    #
    # The whole fused sequence, captured once and replayed as a single host launch. Legitimate
    # under this harness's timing contract only if the dynamic input really is copied inside the
    # timed call: the shifting memory pool hands out a fresh ``data_ptr`` every iteration, so
    # embedding the caller's pointers in the graph would replay stale data. Those copies are
    # therefore issued eagerly, inside the window, and counted.
    #
    # **The graph is keyed on the identity of the weight bundle it was captured against.** A
    # captured graph hard-codes the pointers of every tensor it touched, so it is only valid for
    # the exact bundle ``_rebuild`` produced at capture time. An earlier revision keyed it on mask
    # presence alone and invalidated it only from the root's ``_invalidate``; that was wrong,
    # because the holder-level hooks (``_ParamPair.__setattr__`` and friends) only raise the shared
    # dirty flag -- they do not go through ``_invalidate``. So replacing ``q_proj.weight`` rebuilt
    # the cache but went on replaying the old graph against the old pointers, silently. Identity
    # against the live bundle closes that off by construction: ``_rebuild`` returns a *new tuple
    # object* on every rebuild, so any path that can stale the cache also fails this check, and the
    # graph can never be more stale than the cache it was built from. The stage switches are in the
    # key too, since flipping one changes which kernels the pipeline issues.
    #
    # In-place parameter mutation needs no invalidation **for the weights the pipeline reads
    # directly** -- ``out_proj``, ``fc1``, ``fc2`` and both norms -- because the graph replays
    # against the same storage and picks up the new values. It is **not** covered for Q, K and V:
    # those are read from the fused ``cat``, which is a *derived copy* made at rebuild time, so an
    # in-place write to ``q_proj.weight`` is invisible to both the cache and the graph. That is the
    # same residual window the cache documents above, and it is a property of the fusion rather
    # than of capture; an earlier version of this comment generalised from the one case that does
    # work. ``p.data = X`` replaces storage without going through any hook this module can see and
    # is likewise uncovered.
    #
    # Output lifetime: the captured graph writes into one static buffer, and every call copies that
    # buffer into a freshly allocated tensor before returning. That costs one launch and a 236 KB
    # copy, and it is what makes the module a genuine drop-in -- the caller owns its result and can
    # hold it indefinitely. A ring of static buffers was tried first and rejected: with N slots the
    # (N+1)-th call silently overwrites the first caller's tensor, which is not an ``nn.Module``
    # output contract, and a test that stops at N calls cannot see it.
    def _replay(self, weights, residual: torch.Tensor,
                attention_mask: torch.Tensor | None):
        """Copy in, replay, and return caller-owned storage. None if no graph is available."""
        has_mask = attention_mask is not None
        key = (id(weights), has_mask, self._stage4_triton, self._stage8_triton)
        state = self._graphs
        if state is None or state[0] is not weights or state[1] != has_mask \
                or state[2] != (self._stage4_triton, self._stage8_triton):
            # A capture failure is recorded against the exact key that failed, not globally. An
            # earlier revision latched a single boolean and tested it *before* looking at the key,
            # so one failure permanently disabled capture for every unrelated bundle, mask and
            # switch combination -- and nothing on the holder-level dirtying path ever cleared it.
            if key in self._graphs_failed:
                return None
            state = self._capture(weights, has_mask)
            if state is None:
                return None
        _, _, _, static_x, static_mask, graph, static_out = state
        static_x.copy_(residual, non_blocking=True)
        if static_mask is not None:
            static_mask.copy_(attention_mask, non_blocking=True)
        graph.replay()
        # A fresh allocation, not a view of the static buffer: the caller owns what it gets back.
        return static_out.clone()

    def _capture(self, weights, has_mask: bool):
        """Capture the pipeline once, off the timed path. None if capture is unavailable."""
        seq, embed = _CAPTURED_SEQ, _CAPTURED_EMBED
        device = weights[0].device
        try:
            static_x = torch.empty((seq, embed), device=device, dtype=torch.float32)
            static_mask = (torch.ones((1, 1, seq, seq), device=device, dtype=torch.float32)
                           if has_mask else None)
            static_out = torch.empty((seq, embed), device=device, dtype=torch.float32)
            # Run once eagerly first: every kernel the capture will record has to be compiled and
            # every cuBLAS handle warmed, or capture would try to synchronize.
            self._pipeline(weights, static_x, static_mask, static_out)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self._pipeline(weights, static_x, static_mask, static_out)
            torch.cuda.synchronize()
        except Exception as exc:  # capture unsupported, or a kernel refused to be captured
            detail = " ".join(str(exc).split())
            print(f"[candidate L3/clip_encoder_layer] graph capture unavailable for this "
                  f"configuration, staying eager: {type(exc).__name__}: {detail}"[:400],
                  file=sys.stderr, flush=True)
            self._graphs = None
            self._graphs_failed.add((id(weights), has_mask,
                                     self._stage4_triton, self._stage8_triton))
            return None
        self._graph_generation += 1
        _count(_CACHE_BUILDS, "graph_capture")
        self._graphs = (weights, has_mask, (self._stage4_triton, self._stage8_triton),
                        static_x, static_mask, graph, static_out)
        return self._graphs

    # -- fallback -------------------------------------------------------------
    def _norm(self, x: torch.Tensor, holder: _ParamPair) -> torch.Tensor:
        """The baseline norm's own body, including its fp32 promotion.

        The baseline holder promotes to fp32 for the reduction and casts the result back, which
        is a no-op for an fp32 input but not for a lower-precision one -- so it is reproduced
        rather than simplified away, and the fallback stays faithful for any dtype the baseline
        accepts.
        """
        weight, bias = holder.weight, holder.bias
        if x.dtype is torch.float32:
            return F.layer_norm(x, self.normalized_shape, weight, bias, self.eps)
        weight = weight.float() if weight is not None and weight.dtype != torch.float32 else weight
        bias = bias.float() if bias is not None and bias.dtype != torch.float32 else bias
        return F.layer_norm(x.float(), self.normalized_shape, weight, bias,
                            self.eps).to(x.dtype)

    def _reference(self, hidden_states: torch.Tensor,
                   attention_mask: torch.Tensor | None) -> torch.Tensor:
        """The baseline's operation sequence, op for op, on the parameters directly.

        ``F.layer_norm`` / ``F.linear`` / ``torch.matmul`` / ``F.softmax`` and
        ``x * torch.sigmoid(1.702 * x)`` are what the baseline's holders reduce to, so this is
        bit-identical to the reference for every input, at the reference's cost. It reads the
        parameters rather than calling submodule forwards on purpose -- see the module
        docstring on which relative imports resolve to frozen candidate kernels.
        """
        attn = self.self_attn
        residual = hidden_states
        normed = self._norm(hidden_states, self.layer_norm1)

        batch_size, seq_length, _ = normed.shape
        queries = F.linear(normed, attn.q_proj.weight, attn.q_proj.bias)
        keys = F.linear(normed, attn.k_proj.weight, attn.k_proj.bias)
        values = F.linear(normed, attn.v_proj.weight, attn.v_proj.bias)
        shape = (batch_size, seq_length, attn.num_heads, attn.head_dim)
        queries = queries.view(shape).transpose(1, 2)
        keys = keys.view(shape).transpose(1, 2)
        values = values.view(shape).transpose(1, 2)

        weights = torch.matmul(queries, keys.transpose(-1, -2)) * attn.scale
        if attention_mask is not None:
            weights = weights + attention_mask
        weights = F.softmax(weights.float(), dim=-1).to(queries.dtype)

        attn_out = torch.matmul(weights, values).transpose(1, 2).contiguous()
        attn_out = attn_out.reshape(batch_size, seq_length, attn.embed_dim)
        hidden = residual + F.linear(attn_out, attn.out_proj.weight, attn.out_proj.bias)

        residual = hidden
        normed = self._norm(hidden, self.layer_norm2)
        gated = F.linear(normed, self.mlp.fc1.weight, self.mlp.fc1.bias)
        gated = gated * torch.sigmoid(1.702 * gated)
        return residual + F.linear(gated, self.mlp.fc2.weight, self.mlp.fc2.bias)
