"""Oasis feed-forward blocks for B200 / sm_100.

Same operator as ``baseline.py``: ``fc2(act(fc1(x)))``, two fp16 GEMMs with bias
separated by a GELU. Over the five cases ``fastkernels bench --target oasis_mlp``
selects, that is ``M x 1024 . 4096x1024^T`` then ``M x 4096 . 1024x4096^T`` at
M in {288, 432, 576, 720} in ``approximate="tanh"`` mode and M = 3456 in exact mode.

Two measured facts set the shape of this file.

**The reference is plain torch, and the frozen L1 winners are ours alone.**
``bench._load_candidate_class`` rewrites only ``.tasks.baseline.`` ->
``.tasks.candidate.`` in *this* operator's dotted path, and
``list._CandidateFinder.find_spec`` returns early for any name that does not start
with ``fastkernels.tasks.candidate.``. So the baseline instance's ``..L1.gelu`` and
``..L1.linear`` resolve to ``tasks/baseline/L1/*`` -- verified to be exactly
``F.gelu`` and ``F.linear`` -- while the imports below resolve through the finder to
the frozen winners. Composition alone therefore buys something: the frozen L1 GELU
does the exact-mode pass in 14.43 us against ``F.gelu``'s 27.45 us, which read
1.000 / 1.000 / 1.001 / 0.999 / 1.099 over the five cases, geomean 1.0189x
(``profile/00-baseline/probe4.clean.log``). The four ties are because that module
deliberately delegates ``approximate="tanh"`` to ``F.gelu``.

**The window has a large common-mode floor, so only multi-microsecond kernel wins
show.** Measured in ``profile/00-baseline/probe3.py``:

    window ~= 8.7 us fixed + sum(kernel GPU time) + ~1.5 us per extra launch

which predicts 37.5 us against 35.87 measured at M=288 and 117.2 against 113.60 at
M=3456. Host launch cost is hidden rather than absent: the ``l2.zero_()`` that
``bench._time_module`` enqueues before each timed iteration lets the CPU run ahead, and
the events then measure GPU time only -- the same one-kernel ``x*2`` read
11.31 / 11.28 / 11.30 us through eager ATen, a Triton launch and a C++ extension despite
host enqueue costs of 8.5 / 12.5 / 7.2 us. So optimise GPU time and launch count, not
host work.

How much run-ahead there is was re-measured against the shipped candidate rather than
estimated (``profile/06-host-cost/host-vs-gpu.log``), because the plan warned that
falling GPU time could re-expose host cost. It cannot, by a wide margin: the flush is a
265 MB buffer (``2 x L2_cache_size``, and L2 is 132.6 MB) and costs **~114 us** of GPU
time per iteration, not the ~30 us the plan estimated, leaving 106-175 us of headroom
rather than ~20. This module's dispatch chain is invisible in it -- 38.6-46.5 us of host
time per iteration against the baseline's 38.7-40.6 us.

The fixed 8.7 us is common-mode, so it compresses every speedup -- halving a 25.8 us GPU
cost reads as 1.66x, not 2x.

Where the baseline's time goes (``profile/00-baseline/probe4.py``, one process):

    ==========================  M=288 (tanh)   M=3456 (exact)
    fc1  nvjet_sm100_hsh_*            8.09 us         42.28 us
    F.gelu  vectorized_elementwise    4.29 us         27.45 us
    fc2  nvjet_sm100_hsh_*           13.39 us         35.79 us
    GPU total                        25.77 us        105.52 us
    window                           35.87 us        113.60 us

Deleting the activation outright measures 1.240 / 1.187 / 1.229 / 1.270 / 1.335,
geomean 1.251x. That is the ceiling on fusion while the GEMMs stay nvjet's, and it
is the largest single item on the table.

Structure
---------
``fc1`` / ``act`` / ``fc2`` are the baseline's own submodules under the baseline's
own names, because ``bench._bench_one_case`` shares weights with
``candidate.load_state_dict(baseline.state_dict(), strict=False)`` inside a bare
``try/except: pass``. A renamed parameter does not raise there -- it silently leaves
this module on its own ``torch.empty`` storage (made deterministic but *different* by
``_sanitize_float_params``) and the case fails as ``INCORRECT_NUMERICAL`` with no
diagnostic. Keeping the structure also makes the fallback free: anything the
admission table does not name runs ``self.fc2(self.act(self.fc1(x)))``, which is the
baseline's own expression over the frozen L1 modules.

Nothing is derived from a weight in ``__init__``. ``_prepare_module`` moves and casts
the parameters and ``load_state_dict`` overwrites their storage, both *after*
construction, so anything precomputed there would be stale by the first forward.
Both captured weights are already K-contiguous, which is the layout the tensor cores
want, so nothing needs relaying out; if that ever changes, the cache key must include
``weight.data_ptr()`` **and** ``weight._version``, since the storage can be rewritten
under a stable ``id()``.

Dispatch
--------
``_ADMITTED`` maps the full problem tuple to one measured route, and an empty table is a
valid shipping state rather than an unfinished one.

**Nothing here runs with locked clocks, and no entry may rest on comparing absolute
times across processes.** An earlier version of this file claimed ``validate.py`` locks
clocks via ``bench._CLOCK_PRESETS``; that was wrong. ``bench`` calls ``_lock_clocks()``
only when ``--lock-clocks`` is passed, and in this container the lock cannot be taken at
all -- ``sudo -n nvidia-smi -lgc 1500`` returns *"The current user does not have
permission to change clocks"*, so the bench prints one line and continues unlocked
(``profile/07-clock-lock/``). ``validate.py`` now passes the flag anyway, purely so every
log records the fact instead of leaving it to be assumed.

The consequence is that absolute windows are process-local: the same M=3456 case read
58 us in one bench process and 93 us in another, a 1.6x spread with no code change. Two
kinds of comparison survive that, and admission uses both:

* the harness's own per-case ``speedup``, which times baseline and candidate inside the
  *same* process for the same case and is therefore paired by construction; and
* ``tools/same_process_ab.py``, which toggles this table in place inside one process and
  reports the median paired difference over alternating repeats. Its resolution is measured
  rather than asserted, using the four keys that have **no** admitted entry: both arms there
  run identical code, so whatever they show is the method's own floor. Over four runs that
  is 16 null observations with medians spanning **-0.40 to +0.10 us** and individual paired
  repeats as wide as -1.78 us. So the method resolves effects at roughly the half-microsecond
  level, not better.

``_plan`` consults the table last, after predicates that each guard something a route
actually relies on, and returns ``None`` for everything unmeasured. Those predicates run
inside the timed window on every call, so they stay host-side and cheap: integer and
attribute checks, two set lookups and a dict lookup, with no CUDA call, no ``.item()``, no
tensor allocation and no sync. ``_FASTPATH_HITS`` counts entries per key with saturating
host-side ints, which is what distinguishes "the fast path ran and tied" from "the fast
path never ran".

What the four tanh readings mean
--------------------------------
In ``approximate="tanh"`` mode nothing here is admitted, so the candidate runs
``F.linear -> F.gelu -> F.linear`` -- the frozen L1 ``Linear`` delegates fp16 and the
frozen L1 ``GELU`` delegates that mode -- which is *the same three ATen calls the
baseline makes*, confirmed kernel-for-kernel in ``profile/00-baseline/probe4.py``
(13.59 / 8.04 / 4.30 us against 13.39 / 8.09 / 4.29). Those four cases therefore
cannot regress; they can only be *read* as regressing.

They are, routinely. Over 20 ``validate.py`` runs (``profile/03-bench/``) M=576 drew
0.956-1.001 and M=720 drew 0.959-1.001 on that identical work, and the spread sits
entirely in which side of the ~2.04 us quantum a ~42-48 us window lands on -- the
baseline itself read 45.94-48.06 us for one shape across runs. The bench takes exactly
one median-of-50 draw per module per case, so a single draw on these shapes is not
evidence in either direction, which is the same conclusion ``candidate/L1/gelu.py``
reached about its own tanh path. Everything admitted below is therefore gated on a
margin over that quantum, reproduced across runs, and grouped by machine regime.

Deferred designs
----------------
Two kernels are designed and deliberately not implemented here, because each
compounds an unproven thing onto another. Both are gated on an own fc1 first reaching
parity with the kernel it would replace -- which is the epilogue-fused cuBLASLt kernel
the route below already ships, not plain nvjet -- and it does not: 0.765x, measured in
``profile/09-fc1-ladder/`` (see ``_FUSED_FC1_STATUS``). The full derivations, with the
numbers a later attempt should start from, are in ``docs/deferred-designs.md``.

*Split-K fc2.* fc2 costs more than fc1 for identical FLOPs (13.39 vs 8.09 us at
M=288) because cuBLAS picks a 64x32 tile for N=1024 against 64x144 for N=4096. At
BM=64/BN=32 each CTA streams (64+32)*4096*2 B = 786 KB, so 160 CTAs move ~126 MB
through L2 for a 2.4 GFLOP problem: arithmetic intensity, not parallelism, is the
defect. BM=128/BN=128 would move ~50 MB but field only 24 CTAs, which is why cuBLAS
does not choose it; BM=128, BN=128 with an 8-way split over K=4096 gives 192 CTAs at
~60 MB and would put fc2 near its ~1.7 us roofline instead of 13.4 us.

*Single-kernel fused MLP.* grid = (m_tile, hidden_chunk); each CTA computes
``h[BM,BH] = x[BM,:] . W1[:,chunk]``, applies GELU, then
``partial[BM,1024] = g . W2[chunk,:]``, and the hidden tensor never reaches DRAM. The
hidden-chunk axis *is* fc2's split-K axis, so this subsumes the design above. Two hard
constraints bound it. TMEM on SM100 is 128 lanes x 512 columns x 4 B = 256 KB per SM,
while an fc2 accumulator of BM x 1024 x 4 B needs 512 KB at BM=128 -- it does not fit;
BM <= 48 fits but wrecks MMA efficiency. And partials of S*M*1024*4 B against a hidden
tensor of M*4096*2 B break even at S=2: at M=288, S=16 is 18.9 MB and stays inside the
126 MB L2, but at M=3456 it is 226 MB and does not -- where split-K is not needed
anyway, at 216 CTAs. So the fused form is a small-M design and the two-kernel form a
large-M design, which is an argument for the per-shape table above rather than for one
universal kernel.

Numerics
--------
What the bench measures, stated as measured: ``max_abs_error`` 1.95e-03 on the
exact-mode case and 0.00e+00 on the four tanh cases, with ``matched_ratio`` 1.000000 --
that is, *every* element inside ``atol + rtol*|y|`` -- on all three correctness rounds of
every run. The bench reports the *worst* matched ratio and the *maximum* error over
``rounds = 3`` independent seeds, so this holds per seed and not merely on average.

This is an approximation admitted by the tolerance, not an algebraic identity: the
exact-mode GELU in the frozen L1 kernel is a fitted degree-4 polynomial and the epilogue's
is cuBLASLt's own. Only ``approximate="tanh"`` would earn the equivalence claim, and that
mode is unadmitted here. Note also that 1.95e-03 happens to equal one fp16 ULP at
``|y|max`` 2.705, but that pairing is not evidence: the largest error and the largest
output need not occur at the same element, so no per-element ULP claim is made.

The fp16 boundary at the fc1 output was the open question here: the reference computes
GELU on the fp16-rounded fc1 output while a fused epilogue holds an fp32 accumulator, so
the two could disagree. For **the shipped cuBLASLt route** they do not, at these
magnitudes: the epilogue's output sits within one fp16 ULP (6.10e-05 at ``|h|`` <= 0.13)
of *both* candidate references, and those two are within one ULP of each other, so which
side of the boundary cuBLASLt uses is not observable and this file has no knob for it.

That is a fact about cuBLASLt's epilogue, not a general one. For **an own fused kernel**,
where the choice is ours, the emulation was measured and is the right default: rounding the
fp32 accumulator to fp16 and back before the activation costs +0.040 us on 14.2 M elements
(+0.09%, 25x less than the plan estimated) and is never worse over the three correctness
seeds -- see ``profile/10-f16-boundary/``. Any future fused route here should carry it.

Environment switches, sampled once at import and never written:

``FK_OASIS_MLP_DELEGATE=1``
    Ignore the admission table and delegate every call, for A/B-ing a shipped route against
    the fallback across two processes. Prefer ``tools/same_process_ab.py``, which toggles
    the table in place inside *one* process and is what the admission below actually rests
    on; this switch remains useful for running the whole bench in a delegating
    configuration.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
from torch.autograd import forward_ad as _forward_ad

from ..L1.gelu import GELU
from ..L1.linear import Linear

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

# cuBLASLt's GELU epilogue, reached through the private ATen op. Guarded with getattr
# so a torch build without it degrades to delegation instead of raising at import.
_ADDMM_ACTIVATION = getattr(torch, "_addmm_activation", None)

_EPILOGUE = "cublaslt-gelu-epilogue"

_DELEGATE_ALL = os.environ.get("FK_OASIS_MLP_DELEGATE", "") not in ("", "0")


class Route:
    """One admitted fast path, plus the measurement that admitted it.

    ``speedup`` is the harness's own per-case figure for this key -- ``baseline_ms /
    candidate_ms``, both timed in one bench process -- so it is measured against the
    *plain-torch baseline*, not against whatever route preceded this one. The incremental
    value of admitting the entry is a different and smaller number, and lives in the
    comment on the entry itself; keep the two apart, because routes overlap and a gain
    counted against the wrong reference is counted twice.
    """

    __slots__ = ("name", "speedup", "note")

    def __init__(self, name: str, speedup: float = 0.0, note: str = ""):
        self.name = name
        self.speedup = speedup
        self.note = note

    def __repr__(self):
        return f"Route({self.name!r}, speedup={self.speedup:.3f})"


# ---------------------------------------------------------------------------
# No own fused fc1 + bias + GELU kernel ships. This is a measured decision, taken with the
# kernel installed and judged by `python validate.py`, not by a probe.
#
# `profile/09-fc1-ladder/` walks the additive ladder the plan names, in the order
# `docs/task12-fc1-prospective-ladder.md` froze before any of it was measured:
#
#   v0  128x256x64 tiled + bias, no GELU              60.5 us
#   v1  + fp16 boundary + exact GELU epilogue         70.7 us   (the activation costs 10.2)
#   T1  + num_warps / num_stages calibration          70.6 us   (nothing to win: see below)
#   v2  + GROUP_M=8 swizzle                           70.7 us
#   v3  + 2-CTA cluster (`num_ctas=2`)                93.1 us   (a 22.4 us REGRESSION)
#   v4  + cluster-aware persistence                   76.8 us
#
# against a bar of 51.3 us for the `cutlass3x_..._2sm_bias_f16_gelu_aux_f16` kernel that
# the epilogue route above already runs for this key. Best rung 0.71-0.77x of the bar,
# depending on process; 0.765x in the one ncu report that holds every rung.
#
# Three results worth keeping, because each cost real time to establish:
#
# * **The W/S calibration has nothing in it.** num_warps=8 is optimal by a wide margin --
#   4 costs 28 us and 16 costs 108 us -- and num_stages 2/3/4 span 0.09 us, which is noise.
#   The barrier-stall symptom (8.1 instructions against the vendor kernels' 0.22-0.61) is
#   not addressable by stage count or warp count on this surface, which is where the
#   pre-registered analysis expected the cheapest win to be.
# * **`num_ctas=2` forms a cluster and nothing more.** The PTX is *identical* at
#   `num_ctas=1` and `num_ctas=2` apart from `clusterid.x` and a few cluster ops: the same
#   two `cp.async.bulk.tensor.2d.shared::cluster...` loads, no `.multicast::cluster`
#   anywhere, and `tcgen05.mma.cta_group::1` either way. The counters say what the pair
#   does instead of cooperating -- CTAs 432 -> 864, waves/SM 2.92 -> 5.84, L1 global load
#   sectors 6,912 -> 13,824, all exactly 2x for the same 432 output tiles. Both CTAs run
#   the same program on the same tile and duplicate the work, which is the whole 22 us.
#   So *2-CTA clusters with TMA multicast* is not expressible on Triton 3.6's `tl.*`
#   surface here; it needs Gluon, CuTe DSL or CUDA C++.
# * **tcgen05 and TMEM are genuinely reached, and that was never the problem.**
#   `tcgen05.alloc`, `tcgen05.mma.cta_group::1.kind::f16`,
#   `tcgen05.ld/st.sync.aligned.32x32b.x64.b32`, and no `wgmma` or `mma.sync`. The claim is
#   "no GEMM FMA fallback" rather than "no FMA instructions" -- the 10.3% FMA-pipe activity
#   is the GELU epilogue. The residual gap is MMA feed rate: 28-40% tensor-pipe activity
#   against nvjet's 75.3% and the bar's 66.1%.
#
# Installed behind a temporary admission key and put through the bench, the kernel was
# correct (5/5 PASSED, max_abs 1.95e-03, matched 1.000000) and slower: M=3456 read 0.9694
# and 0.9626 against the epilogue route's 1.14-1.19, dropping the geomean to 0.9984 /
# 1.0008. A same-process whole-module A/B agrees -- fused 102.4 us, epilogue 82.0 us,
# delegation 88.2 us -- and puts the loss in the kernel rather than the dispatch wrapper,
# since building the TMA descriptors per call costs nothing measurable (70.69 us against
# 70.66 hoisted).
#
# The scaffolding was removed afterwards rather than left dormant: an unadmitted route
# still costs import-time compilation and a branch in `forward`, and buys nothing.
_FUSED_FC1 = None
_FUSED_FC1_STATUS = "absent:measured-0.71x-of-the-shipped-epilogue-and-0.96x-under-the-bench"


# Keyed ``(M, in_features, hidden_features, out_features, approximate, dtype)``.
#
# The key carries the whole geometry rather than M alone because both L3 consumers
# size the hidden layer from a config ratio -- ``oasis_block`` passes
# ``hidden_features=int(hidden_size * mlp_ratio)`` and ``oasis_vae_attention_block``
# ``int(dim * mlp_ratio)``, both leaving ``out_features`` at ``None``. The captures
# happen to be 1024/4096/1024 throughout, but the ratio is not fixed by the operator,
# so a table keyed on M would route an unmeasured geometry into a measured entry.
#
# ``approximate`` is in the key because the two modes select different work, and
# ``dtype`` because only fp16 has been measured here.
_ADMITTED: dict[tuple, Route] = {}

if _ADDMM_ACTIVATION is not None:
    # The one key the epilogue measured a win on. It is admitted for the exact-mode
    # M=3456 case and nowhere else.
    #
    # The four tanh keys were each tried under `validate.py`, three runs, rather than
    # taken on the earlier probe's word -- `profile/03-bench/epilogue-on-tanh-keys.log`.
    # Candidate windows came out unusually stable (43.01 us three times at M=576, 47.10
    # three times at M=720), which makes the readings unusually easy to trust:
    #
    #   M=288   36.83-36.93 us   0.967 / 0.971 / 0.967   a reproducible ~1.15 us LOSS
    #   M=432   38.88-38.91 us   1.025 / 1.025 / 1.004   a reproducible ~0.95 us win
    #   M=576   43.01 us         1.001 / 1.003 / 0.990   inside the quantum, both signs
    #   M=720   47.10 us         1.019 / 1.019 / 1.000   inside the quantum
    #
    # So the earlier probe's 0.921 / 0.974 / 0.978 / 1.019 overstated the harm: only
    # M=288 is a consistent regression. But nothing here clears the gate either. The
    # largest effect in any direction is under half of the ~2.04 us quantum, and the
    # rule this file is held to is a margin *over* the quantum, reproduced -- not a
    # reproducible sub-quantum difference, which is what a stable reading of a
    # launch-floored shape produces either way. None is admitted.
    #
    # The unscored exact-mode [1,576,1024] case (M=576, "none") is not admitted either:
    # nothing has measured it, and it must not inherit an entry keyed to M=3456.
    #
    # Why it is per-shape rather than universal: requesting the epilogue does fuse --
    # two kernels instead of three, confirmed in profile/00-baseline/probe4.py -- but
    # it also flips kernel selection away from nvjet, from
    # nvjet_sm100_hsh_64x144_64x12_2x2_2cta (8.09 us) to
    # cutlass3x_sm100_tensorop_s128x256x16gemm_f16_f16_f32 (14.95 us). That is
    # +6.86 us to save 4.29 us at M=288 and +6.11 us to save 27.45 us at M=3456, so
    # the trade is only worth taking where the activation is large. The mechanism
    # that fuses into nvjet cannot use nvjet.
    # What admitted it. Two independent lines of evidence, both paired within a process,
    # because absolute windows are not comparable across processes here.
    #
    # (a) The harness's own per-case speedup, which times baseline and candidate in one
    #     process, over 9 runs with this entry against 11 with it removed. Every run with
    #     the entry beat every run without it, with no overlap:
    #         with entry     1.1398 - 1.1969   (n=9)
    #         without        1.1004 - 1.1324   (n=11)
    #     `profile/03-bench/epilogue-vs-delegation.log`.
    #
    # (b) `tools/same_process_ab.py`, which toggles this table in place and takes the
    #     median paired difference over alternating repeats. Four runs, 6 repeats each,
    #     so 24 paired deltas for this key, none of them negative:
    #         window ~58 us   delta +2.02 us  [+2.00..+2.08]   3.3% of the window
    #         window ~88 us   delta +4.20 us  [+4.18..+4.90]   4.6%
    #         window ~90 us   delta +3.39 us  [+3.15..+5.09]   3.6%
    #         window ~88 us   delta +4.48 us  [+4.18..+4.93]   4.8%
    #
    #     The method's own floor, from the 16 null observations on the four keys with no
    #     entry, is a median span of -0.40 to +0.10 us. So these four sit 5-10x outside
    #     the floor, and three of the four also clear the ~2.04 us quantum outright; the
    #     fourth reads 2.02 us in the fastest-clocked process, where every absolute effect
    #     shrinks. Separately, one whole-module single-pass timing (`tools/route_ab.py`,
    #     not the alternating method) read +4.19 us.
    #
    # Two corrections to earlier versions of this comment, both of them overstatements of
    # mine: "~4.9 us, 2.4x the quantum" came from cross-process absolute windows, which is
    # not an admissible comparison here; and "a measured noise floor of +-0.05 us, 40x
    # below the quantum" generalised the tightest of the four runs to all of them. No
    # other case moved in either line of evidence.
    _ADMITTED[(3456, 1024, 4096, 1024, "none", torch.float16)] = Route(
        _EPILOGUE, speedup=1.194,
        note="harness per-case speedup vs the plain-torch baseline, median of 9 runs; "
             "the incremental gain over delegating is +3.4 to +4.5 us, see above")

# The coarsest components of the admitted keys, so a call that cannot possibly be in
# the table reaches the fallback on two set lookups instead of the whole predicate
# chain. That matters for the four tanh-mode cases: they never have an admitted route,
# they run in 25-48 us windows where host cost is only hidden by ~20 us of headroom,
# and they are the cases whose readings are already launch-floored.
_ADMITTED_MODES = frozenset(k[4] for k in _ADMITTED)
_ADMITTED_DTYPES = frozenset(k[5] for k in _ADMITTED)

# Fast-path entries per key, so "the fast path ran and tied" is distinguishable from "the
# fast path never ran". Host-side only: no CUDA call, no sync, no thread, nothing the
# harness's integrity guards watch. It is not strictly allocation-free -- past CPython's
# small-int cache each increment builds a new int object -- but that is a host-side
# allocation on a path with ~20-175 us of CPU run-ahead per iteration
# (`profile/06-host-cost/`), and the counter is capped below so it cannot grow without
# bound in a long-lived process.
_FASTPATH_HITS: dict[tuple, int] = {}

# The counter saturates. Its only job is to answer "did this key ever take the fast path",
# and a bounded value keeps every increment inside CPython's small-int cache once the cap
# is reached, so a long-lived L3 process cannot accumulate int objects on this path.
_HIT_CAP = 256

# 128-bit vectorised access needs 16-byte-aligned bases. The harness's shifting pool
# hands out 256-byte-aligned slots, but that is a property of the pool rather than a
# guarantee about every caller.
_ALIGN_BYTES = 16


def _carries_forward_grad(x: torch.Tensor) -> bool:
    """Does *x* hold a forward-mode tangent?

    A dual tensor has exact type ``torch.Tensor`` and ``requires_grad=False``, so no
    other predicate here would stop it, and a fused route writes into fresh storage
    with no tangent attached -- the derivative would vanish silently. Checking the
    active dual level first makes this one integer comparison when nobody is doing
    forward AD, which is always, under the bench. Same guard, same reason, as
    ``candidate/L1/gelu.py``.
    """
    if getattr(_forward_ad, "_current_level", -1) < 0:
        return False
    return _forward_ad.unpack_dual(x).tangent is not None


def _plan(x: torch.Tensor, w1: torch.Tensor, b1, w2: torch.Tensor, b2,
          approximate) -> tuple[Route | None, tuple]:
    """Return ``(route, key)`` for this call, with *route* ``None`` to delegate.

    Pure and cheap by construction: integer and attribute checks only. Every
    predicate guards something a route actually relies on, and the shape must
    additionally be in ``_ADMITTED``.
    """
    if _DELEGATE_ALL or not _ADMITTED:
        return None, ()
    # The two coarsest components of the key first, so a mode or dtype with no admitted
    # route at all costs two set lookups rather than every predicate below. The mode is
    # whatever `self.act.approximate` holds right now; an unrecognised value is not this
    # file's business to interpret, so it delegates and lets ATen raise its own message
    # from the frozen L1 module. The `str` check also keeps an unhashable value out of
    # the set lookup.
    if type(approximate) is not str or approximate not in _ADMITTED_MODES:
        return None, ()
    dt = x.dtype
    if dt not in _ADMITTED_DTYPES:
        return None, ()
    # A tensor subclass, a fake tensor or a functorch wrapper would be dropped on the
    # floor by a route that reads raw storage, and `_check_lazy_outputs` tests
    # `type(t) is torch.Tensor` strictly on the way out.
    if type(x) is not torch.Tensor:
        return None, ()
    # Inference-only routes: they build no graph, so grad mode must delegate. The
    # bench runs both correctness and timing inside `torch.no_grad()`, so this is
    # False there.
    if torch.is_grad_enabled():
        return None, ()
    # Every operand in the input's dtype -- which the check above has already narrowed
    # to a dtype some admitted key uses, fp16 being the only one measured here.
    if w1.dtype is not dt or w2.dtype is not dt:
        return None, ()
    if b1 is None or b2 is None or b1.dtype is not dt or b2.dtype is not dt:
        return None, ()
    # One device: every operand has to live where the launch goes.
    if not x.is_cuda:
        return None, ()
    dev = x.device
    if w1.device != dev or w2.device != dev or b1.device != dev or b2.device != dev:
        return None, ()
    if w1.dim() != 2 or w2.dim() != 2 or x.dim() < 1:
        return None, ()
    K = w1.shape[1]          # in_features
    H = w1.shape[0]          # hidden_features
    if w2.shape[1] != H:
        return None, ()
    N = w2.shape[0]          # out_features
    if x.shape[-1] != K:
        return None, ()
    # Degenerate problems stay with torch, which handles a zero-length dimension.
    if K <= 0 or H <= 0 or N <= 0:
        return None, ()
    # A hidden `.contiguous()` would cost a launch worth more than any route here wins
    # back, so a non-contiguous input delegates instead. The reshape on the fast path
    # is a view only under this predicate.
    if not x.is_contiguous():
        return None, ()
    if w1.stride(-1) != 1 or w1.stride(0) != K:
        return None, ()
    if w2.stride(-1) != 1 or w2.stride(0) != H:
        return None, ()
    if b1.dim() != 1 or b1.shape[0] != H or not b1.is_contiguous():
        return None, ()
    if b2.dim() != 1 or b2.shape[0] != N or not b2.is_contiguous():
        return None, ()
    M = x.numel() // K
    if M <= 0:
        return None, ()
    if (x.data_ptr() % _ALIGN_BYTES or w1.data_ptr() % _ALIGN_BYTES or
            w2.data_ptr() % _ALIGN_BYTES or b1.data_ptr() % _ALIGN_BYTES or
            b2.data_ptr() % _ALIGN_BYTES):
        return None, ()
    if _carries_forward_grad(x):
        return None, ()
    key = (M, K, H, N, approximate, dt)
    return _ADMITTED.get(key), key


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
        # The baseline's own two lines, including the falsy-zero semantics: 0 resolves
        # to `in_features`, which `hidden_features or in_features` gives and
        # `if hidden_features is None` would not.
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = Linear(in_features, hidden_features, bias=True)
        self.act = GELU(approximate="tanh" if approximate_tanh else "none")
        self.fc2 = Linear(hidden_features, out_features, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1, b1 = self.fc1.weight, self.fc1.bias
        w2, b2 = self.fc2.weight, self.fc2.bias
        # Read the mode from the attribute on every call, as the frozen L1 `GELU` and
        # the reference module both do, so reassigning it after construction keeps
        # working.
        approximate = self.act.approximate
        route, key = _plan(x, w1, b1, w2, b2, approximate)
        if route is None:
            return self.fc2(self.act(self.fc1(x)))
        n = _FASTPATH_HITS.get(key, 0)
        if n < _HIT_CAP:
            _FASTPATH_HITS[key] = n + 1
        if route.name == _EPILOGUE:
            # cuBLASLt fuses bias and GELU into the fc1 epilogue: two kernels instead
            # of three. `x.reshape` and `w1.t()` are both views under the predicates
            # above, so nothing is copied on the way in.
            h = _ADDMM_ACTIVATION(b1, x.reshape(-1, x.shape[-1]), w1.t(),
                                  use_gelu=True)
        else:  # pragma: no cover - no other route is admitted
            return self.fc2(self.act(self.fc1(x)))
        # fc2 stays nvjet's: it is the larger of the two GEMMs and what it needs is
        # split-K, which is recorded as a deferred design rather than attempted here.
        return self.fc2(h).view(*x.shape[:-1], w2.shape[0])
