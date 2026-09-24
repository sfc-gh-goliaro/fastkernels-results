"""Diffusion conditioning for AlphaFold3 -- the whole forward in one Triton launch.

Produces conditioned single and pair representations from trunk outputs and the
diffusion time step. Numerically equivalent to ``baseline.py`` (in fact slightly
*more* accurate -- see the numerics note); structurally a different program.

Why: at the captured shape (N_token=16 -- si_input[1,16,449], si_trunk[1,16,384],
zij_trunk[1,16,16,128], scalar t) every intermediate is under 140 KB, so nothing
here is FLOP bound. The baseline forward is ~140 ATen calls / ~90 kernel launches
(relpos_complex alone is ~20, then two cats, two projections, an 8-launch Fourier
chain over 256 elements, and 4x SwiGLUTransition at LN + 2 GEMM + SiLU + mul +
GEMM + mask + residual each): 1.29 ms of *host* time against 0.28 ms of device
time. The harness's own shifting-input pool re-copies ~24 tensors inside the
timed window, so ~146 us of the measurement is a floor no candidate can go under,
and what remains is dominated by **host** time: measured on this benchmark, a
microsecond of host time costs ~1 us of the score and a microsecond of device
time only ~0.1 us (the GPU finishes our work while the host is still issuing the
harness's copies). Everything below follows from that.

Structure -- one launch, four phases, three grid-wide barriers:

0. ``_pair_body`` (16 programs) + ``_proj_body`` (6). The pair branch is
   row-local from raw input to final output, so one program owns a block of
   (i, j) rows through LayerNorm -> linear_z -> both residual SwiGLU
   transitions. The single projection is independent of it, so it shares the
   grid: neither branch fills the GPU, and this way the whole 11.7 us pair
   branch hides behind the 13 us projection. ``_prefetch`` CTAs (64 of them,
   read-only) stream the later phases' weights into L2 meanwhile -- the
   benchmark flushes L2 before every call, so those 3.5 MB would otherwise be
   fetched from DRAM on the critical path.
1. ``_tr_body``: first single transition, hidden dim split over 24 programs,
   each adding its fp32 partial into a scratch accumulator.
2. ``_tr_body`` again: completes the row from phase 1's partial (residual +
   mask), then the second transition's hidden slice.
3. ``_fin_body``: reduce, mask, residual, store bf16.

The single branch is split across programs because it has only 16 rows: a fused
one-program version must stream all 4.4 MB of its weights through one SM's load
path, which measured 127 us at 6% occupancy and 0.4% of DRAM throughput -- pure
latency starvation (NCU). Splitting its reduction dims cut that to ~30 us, at the
price of three producer/consumer boundaries that need a device-wide sync.

Those three syncs are what ``_gbar`` replaces. Four launches cost ~4.5 us of host
time *each*; three grid-wide barriers cost ~0.3 us of device time in total, which
is then ~90% hidden. The barrier is only legal because the grid is *proven*
co-resident at plan time (``_resident_ctas``) -- 86 CTAs of 148 SMs at the
captured shape -- and the four-launch path (``_first`` / ``_s_tr`` / ``_s_fin``,
still here, unchanged) is used whenever that proof fails, e.g. N_token=64, which
needs 280 CTAs. ``_gbar`` derives its release target on the device rather than
taking one from the host, so there is no host-side counter that can drift out of
step and hang the next launch; see its docstring for the memory-ordering
argument.

Two things make the arithmetic cheaper than the reference's, independent of the
launch structure:

* **The relpos one-hot never has to exist.** ``relpos_complex`` builds a
  [*, N, N, 139] tensor out of ~20 launches of integer bucketing, and
  ``_binned_one_hot`` compares against ``arange``, so each of its four blocks is a
  *thermometer* code: a prefix of ones of length ``ceil(final_offset)``. Its
  contribution to the LayerNorm moments is therefore just that prefix length
  (sum == sumsq, the entries being 0/1), and its contribution to ``linear_z`` is a
  prefix sum of weight columns -- precomputed once in ``_build_plan``, so the
  kernel gathers three table rows and two vectors per pair. The 139-wide tensor,
  the four ``arange``s, the comparisons and the cat all disappear.
* **Both concats and the Fourier chain are virtual.** The single projection runs
  one reduction loop over [si_trunk | si_input | fourier]: no [*, 833] copy, and
  the Fourier tail chunks feed the same ``tl.dot`` accumulator as the LayerNorm'd
  input rows, so ``linear_n``, its LayerNorm and the 8-launch cos chain cost one
  more k-chunk.

Numerics: the Fourier chain is evaluated *with the reference's per-op bf16
rounding* (``_rb``). That is not pedantry -- ``cos(2*pi*x)`` with x held in bf16
carries ~4e-2 of quantization noise, and after ``layer_norm_n`` + ``linear_n``
that is ~0.02 absolute on si, more than the bf16 tolerance allows; an fp32 chain
fails the benchmark. Elsewhere the reference's rounding comes for free, since
``tl.dot`` wants bf16 operands anyway. Checked against an exact op-for-op
emulation of the reference: this kernel matches it to 1e-4 rms where the baseline
itself deviates by 7e-3 (its small-shape cuBLAS GEMMs), so the residual
disagreement with the baseline is the *baseline's* bf16 noise. The fp32 partials
are combined with atomics, so the summation order is not fixed and a borderline
element can move by one bf16 ulp between calls -- the four-launch path does the
same, and the two paths agree to within one ulp over 144 shape/seed pairs.

Host cost is cut the way ``candidate/L1/layer_norm.py`` does it, and then some:
weights are packed once into one bf16 and one fp32 blob (a launch passes two
pointers instead of thirty), the fp32 scratch belongs to the plan rather than
being allocated per call, every launch config is resolved at plan time, the
compiled kernel is invoked through its own C launcher with a prebuilt argument
list, and the per-call validation of the five relpos features, the mask and the
five positional tensors is two tuple comparisons against signatures assembled at
plan time. What remains per call is ~13.5 us: one launch (4.5), two output
allocations (2.3), that validation (2.4), and Python.

The one constraint all this adds: a single module instance must not have two
forwards *in flight concurrently on different streams*, since the scratch buffer
and the prebuilt argument list are per-instance state. Sequential calls, however
deeply queued, are ordered by the stream and safe.
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._C import _cuda_getCurrentRawStream as _raw_stream
from torch._C import _cuda_getDevice as _cur_device

from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear
from .alphafold3_input_embedder import relpos_complex
from .alphafold3_swiglu_transition import SwiGLUTransition


__targets__ = ["DiffusionConditioning"]


# ---------------------------------------------------------------------------
# Tunables (measured; see ITERATIONS.md)
# ---------------------------------------------------------------------------
_PAIR_BM = 16        # pair rows per program
_PAIR_BH = 128       # hidden tile in the pair transitions
_S_BM = 16           # tokens per program (single branch)
_S_BK = 128          # reduction tile of the single projection
_S_BN = 64           # output-channel tile of the single projection
_S_HS = 32           # hidden units per program in a single transition
_S_BH = 32           # hidden tile inside one program (must divide _S_HS)
_S_PROJ_WARPS = 8
_S_PROJ_STAGES = 3
_S_TR_WARPS = 4
_S_TR_STAGES = 2
_MEGA_WARPS = 8      # one launch means one warp count for all four phases
_MEGA_STAGES = 3
# Extra CTAs that do nothing but pull the later phases' weights into L2 while
# phase 0 runs (the benchmark flushes L2 before every call, so those 3.5 MB are
# otherwise fetched from DRAM on the critical path). 0 disables.
_PRE_CTAS = 64
_PRE_TILE = 4096     # bf16 elements per prefetch load (8 KB)
_ALIGN = 128         # blob block alignment, in elements

# One launch for the whole forward, with hand-rolled grid-wide barriers between
# the four phases, whenever the grid is provably co-resident (``_resident_ctas``).
# ``AKO_PERSIST=0`` forces the four-launch path (A/B testing).
_PERSIST = os.environ.get("AKO_PERSIST", "1") != "0"

# Placeholder pointers for the relpos features when ``asym_id`` is absent: the
# kernel is then compiled with HAS_RELPOS=False and never dereferences them.
_NULL = (0, 0, 0, 0, 0)


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------
@triton.jit
def _rb(x):
    """Round an fp32 value through bf16, as an ATen op on bf16 tensors does.

    Applied only where the reference materializes a bf16 tensor, so the
    accumulated rounding matches op for op.
    """
    return x.to(tl.bfloat16).to(tl.float32)


@triton.jit
def _gbar(BAR, SLOT: tl.constexpr, NCTA: tl.constexpr):
    """Barrier across every CTA in the grid.

    Legal only because the launch is *proven* co-resident: ``_resident_ctas``
    computes the occupancy-limited number of simultaneously resident CTAs for
    this kernel's shared memory / registers / warps and the plan refuses the
    single-launch path unless the whole grid fits (a barrier with a CTA that has
    not been scheduled deadlocks, which is far worse than any speed loss).

    One int64 counter per barrier, incremented once per CTA, never reset. The
    target is derived *on the device* from the arrival's own ticket:
    ``(old // NCTA) * NCTA + NCTA``. Every CTA of one launch draws a ticket in
    ``[g*NCTA, g*NCTA + NCTA)`` -- a CTA cannot arrive twice, and the barrier
    itself stops any CTA from reaching generation g+1 before all of them passed
    g -- so they all compute the same target, and consecutive launches (which the
    stream serializes) simply continue the numbering. int64 puts overflow ~10^17
    calls away, so nothing ever has to be rewound.

    Deriving the target on the device instead of passing it in costs ~1.7 us per
    barrier (the arrival's return value has to come back before the spin can
    start) and is worth it: a host-side generation counter has to stay in
    lockstep with the device's, and *any* divergence -- an exception between the
    increment and the launch, a re-plan that changes NCTA, a replayed argument
    list -- leaves the next launch waiting for arrivals that never come. That is
    a permanent hang, a far worse failure than a slow kernel, and it is easy to
    cause (it happened three times while building the measurement rigs for this
    round). This form cannot diverge: the counter is the only state, and it lives
    on the device.

    Ordering: the arrival is a **release** atomic at device scope, which is what
    publishes this phase's writes; the spin is a **volatile** load, which PTX
    defines as ``relaxed.sys`` -- so it cannot be hoisted out of the loop, and
    every cross-phase read (also volatile) is ordered after it and cannot be
    served by a stale L1 line. The two ``bar.sync``es keep the CTA's own warps
    from arriving early or from racing past the release.
    """
    tl.debug_barrier()
    old = tl.atomic_add(BAR + SLOT, 1, sem="release", scope="gpu")
    targ = (old // NCTA) * NCTA + NCTA
    while tl.load(BAR + SLOT, volatile=True) < targ:
        pass
    tl.debug_barrier()


@triton.jit
def _fourier(idx, tval, WF, O_FW: tl.constexpr, O_FB: tl.constexpr,
             SIGMA: tl.constexpr):
    """``cos(2*pi*(0.25*log(t/sigma)*w[idx] + b[idx]))``, rounded like the reference.

    One ``_rb`` per reference op: divide, log, scale, ``*w``, ``+b``, ``2*pi*``,
    cos. The scale and the (power-of-two) divide are exact, but rounding them
    costs nothing and keeps the correspondence checkable.
    """
    n = _rb(0.25 * _rb(tl.log(_rb(tval / SIGMA))))
    x = _rb(_rb(n * tl.load(WF + O_FW + idx)) + tl.load(WF + O_FB + idx))
    return _rb(tl.cos(_rb(6.283185307179586 * x)))


@triton.jit
def _transition(x, mask, WB, WF,
                O_LNW: tl.constexpr, O_LNB: tl.constexpr,
                O_WAB: tl.constexpr, O_WO: tl.constexpr,
                C: tl.constexpr, CP: tl.constexpr, H: tl.constexpr,
                BH: tl.constexpr, BM: tl.constexpr, EPS: tl.constexpr,
                HAS_LNB: tl.constexpr, HAS_MASK: tl.constexpr):
    """One residual SwiGLUTransition block, start to finish in registers.

    ``x`` is [BM, C] fp32 holding bf16-valued rows; returns ``x + mask *
    linear_out(SiLU(linear_a(LN(x))) * linear_b(LN(x)))``.

    ``linear_a``/``linear_b`` are one packed [C, 2H] matrix, so the two
    projections are two dots against one tile, and the out-GEMM accumulates over
    hidden tiles in fp32 -- which is what one cuBLAS GEMM over the whole hidden
    dim does anyway.

    Every Triton tile dim must be a power of two, so a channel count like 384 is
    carried as a ``CP``-wide tile whose tail lanes are held at *exactly zero*.
    The LayerNorm vectors are zero-padded and the weight tiles are *predicated*
    at ``c < C`` with ``other=0.0`` -- identical arithmetic to zero-padded
    weights, but a predicated lane issues no load, so the 33% of weight bytes
    that the padding used to stream (at c_s=384: 512 rows read for 384 real
    ones) never leave memory. Bytes per CTA is what this kernel is limited by.
    Only the variance needs an explicit mask on top, because the tail lanes of
    ``x - mean`` are not zero.
    """
    c = tl.arange(0, CP)
    m = tl.sum(x, 1) * (1.0 / C)
    d = x - m[:, None]
    if C == CP:
        v = tl.sum(d * d, 1) * (1.0 / C)
    else:
        v = tl.sum(tl.where(c[None, :] < C, d * d, 0.0), 1) * (1.0 / C)
    r = 1.0 / tl.sqrt(v + EPS)
    xn = d * r[:, None] * tl.load(WF + O_LNW + c)[None, :]
    if HAS_LNB:
        xn = xn + tl.load(WF + O_LNB + c)[None, :]
    xb = xn.to(tl.bfloat16)
    hc = tl.arange(0, BH)
    wabp = WB + O_WAB + c[:, None] * (2 * H) + hc[None, :]
    wop = WB + O_WO + hc[:, None] * C + c[None, :]
    km = (c < C)[:, None]
    nm = (c < C)[None, :]
    acc = tl.zeros((BM, CP), dtype=tl.float32)
    # A *dynamic* loop: tl.static_range would unroll H/BH copies of two dots and
    # a SiLU, whose overlapping live ranges push this kernel from 0 spills to
    # >1500 (and 8x the device time at the captured shape).
    for _ in range(H // BH):
        if C == CP:
            wa = tl.load(wabp)
            wbq = tl.load(wabp + H)
            wo = tl.load(wop)
        else:
            wa = tl.load(wabp, mask=km, other=0.0)
            wbq = tl.load(wabp + H, mask=km, other=0.0)
            wo = tl.load(wop, mask=nm, other=0.0)
        a = _rb(tl.dot(xb, wa))
        b = _rb(tl.dot(xb, wbq))
        hh = _rb(_rb(a * tl.sigmoid(a)) * b).to(tl.bfloat16)
        acc = tl.dot(hh, wo, acc)
        wabp += BH
        wop += BH * C
    o = _rb(acc)
    if HAS_MASK:
        o = _rb(o * mask[:, None])
    return _rb(x + o)


# ---------------------------------------------------------------------------
# Pair branch: relpos + LayerNorm + linear_z + two transitions, one launch
# ---------------------------------------------------------------------------
@triton.jit
def _pair_body(pid, Z, OUT, RES, TOKI, ASYM, ENT, SYM, TM, WB, WF,
               R, N: tl.constexpr, CZ: tl.constexpr, NCAT: tl.constexpr,
              NREL: tl.constexpr, KPOS: tl.constexpr, KCH: tl.constexpr,
              EPS: tl.constexpr, EPST: tl.constexpr, BM: tl.constexpr,
              H: tl.constexpr, BH: tl.constexpr,
              O_LNW: tl.constexpr, O_LNB: tl.constexpr, O_T1: tl.constexpr,
              O_T2: tl.constexpr, O_T3: tl.constexpr, O_TE: tl.constexpr,
              O_GV: tl.constexpr, O_BV: tl.constexpr, O_WZ: tl.constexpr,
              O_A0: tl.constexpr, O_B0: tl.constexpr, O_W0: tl.constexpr,
              O_O0: tl.constexpr, O_A1: tl.constexpr, O_B1: tl.constexpr,
              O_W1: tl.constexpr, O_O1: tl.constexpr,
              HAS_RELPOS: tl.constexpr, HAS_MASK: tl.constexpr,
              HAS_LNB: tl.constexpr, HAS_TLNB: tl.constexpr,
              EVEN_R: tl.constexpr):
    """One program per block of ``BM`` (i, j) pairs; rows never interact."""
    rows = pid * BM + tl.arange(0, BM)
    c = tl.arange(0, CZ)
    if EVEN_R:
        z = tl.load(Z + rows[:, None] * CZ + c[None, :]).to(tl.float32)
    else:
        rm = rows < R
        z = tl.load(Z + rows[:, None] * CZ + c[None, :],
                    mask=rm[:, None], other=0.0).to(tl.float32)
        rows = tl.where(rm, rows, 0)

    # --- relpos: the four thermometer lengths, never the one-hot -------------
    nn = N * N
    bb = rows // nn
    rem = rows - bb * nn
    bi = bb * N + rem // N
    bj = bb * N + rem % N
    if HAS_RELPOS:
        same_chain = tl.load(ASYM + bi) == tl.load(ASYM + bj)
        same_ent = tl.load(ENT + bi) == tl.load(ENT + bj)
        ri = tl.load(RES + bi).to(tl.float32)
        rj = tl.load(RES + bj).to(tl.float32)
        # clamp(offset + k, 0, 2k) in bf16; the "different chain" bucket is 2k+1.
        f1 = tl.where(same_chain,
                      tl.minimum(tl.maximum(_rb(_rb(ri - rj) + KPOS), 0.0),
                                 2.0 * KPOS),
                      2.0 * KPOS + 1.0)
        ti = tl.load(TOKI + bi).to(tl.float32)
        tj = tl.load(TOKI + bj).to(tl.float32)
        f2 = tl.where(same_chain & (ri == rj),
                      tl.minimum(tl.maximum(_rb(_rb(ti - tj) + KPOS), 0.0),
                                 2.0 * KPOS),
                      2.0 * KPOS + 1.0)
        si = tl.load(SYM + bi).to(tl.float32)
        sj = tl.load(SYM + bj).to(tl.float32)
        f3 = tl.where(same_ent,
                      tl.minimum(tl.maximum(_rb(_rb(si - sj) + KCH), 0.0),
                                 2.0 * KCH),
                      2.0 * KCH + 1.0)
        # #{k in [0, nbins) : k < f} == ceil(f): the thermometer's prefix length.
        c1 = tl.ceil(f1).to(tl.int32)
        c2 = tl.ceil(f2).to(tl.int32)
        c3 = tl.ceil(f3).to(tl.int32)
        se = tl.where(same_ent, 1.0, 0.0)
        cnt = c1.to(tl.float32) + c2.to(tl.float32) + c3.to(tl.float32) + se
    else:
        cnt = tl.zeros((BM,), dtype=tl.float32)

    # --- LayerNorm over the (128 + 139)-wide concat --------------------------
    # relpos entries are 0/1, so they contribute ``cnt`` to both the sum and the
    # sum of squares; everything else is the z block's own moment.
    mean = (tl.sum(z, 1) + cnt) * (1.0 / NCAT)
    d = z - mean[:, None]
    var = (tl.sum(d * d, 1) + cnt * (1.0 - 2.0 * mean)
           + NREL * mean * mean) * (1.0 / NCAT)
    rstd = 1.0 / tl.sqrt(var + EPS)

    # --- linear_z ------------------------------------------------------------
    zn = d * rstd[:, None] * tl.load(WF + O_LNW + c)[None, :]
    if HAS_LNB:
        zn = zn + tl.load(WF + O_LNB + c)[None, :]
    acc = tl.dot(zn.to(tl.bfloat16),
                 tl.load(WB + O_WZ + c[:, None] * CZ + c[None, :]))
    # relpos block: (f - mean) * rstd * ln_w folded into prefix-sum tables.
    acc -= (mean * rstd)[:, None] * tl.load(WF + O_GV + c)[None, :]
    if HAS_RELPOS:
        acc += rstd[:, None] * (
            tl.load(WF + O_T1 + c1[:, None] * CZ + c[None, :])
            + tl.load(WF + O_T2 + c2[:, None] * CZ + c[None, :])
            + tl.load(WF + O_T3 + c3[:, None] * CZ + c[None, :])
            + se[:, None] * tl.load(WF + O_TE + c)[None, :])
    if HAS_LNB:
        acc += tl.load(WF + O_BV + c)[None, :]
    x = _rb(acc)

    # --- two residual transitions -------------------------------------------
    if HAS_MASK:
        mask = _rb(tl.load(TM + bi).to(tl.float32)
                   * tl.load(TM + bj).to(tl.float32))
    else:
        mask = cnt   # unused
    x = _transition(x, mask, WB, WF, O_A0, O_B0, O_W0, O_O0,
                    CZ, CZ, H, BH, BM, EPST, HAS_TLNB, HAS_MASK)
    x = _transition(x, mask, WB, WF, O_A1, O_B1, O_W1, O_O1,
                    CZ, CZ, H, BH, BM, EPST, HAS_TLNB, HAS_MASK)
    if EVEN_R:
        tl.store(OUT + rows[:, None] * CZ + c[None, :], x.to(tl.bfloat16))
    else:
        tl.store(OUT + rows[:, None] * CZ + c[None, :], x.to(tl.bfloat16),
                 mask=rm[:, None])


# ---------------------------------------------------------------------------
# Single branch: four stages, split across CTAs
#
# The single branch has only N_token=16 rows, so a fused one-CTA version has to
# stream all 4.4 MB of its weights through a single SM's load path: measured 127
# us at 6% occupancy, 0.4% of DRAM throughput and 46% of L1/TEX throughput --
# latency-starved, not FLOP or bandwidth bound (NCU). Splitting the *reduction*
# dims across CTAs is what fixes it, and that needs cross-CTA reductions, hence
# four launches:
#
#   _s_proj  grid=(c_s/BN, rows)  LayerNorm + linear_s + linear_n, split by
#                                 output channel -- no reduction needed
#   _s_tr    grid=(H/HS, rows)    one transition's hidden slice -> fp32 partial
#                                 (atomic-add; the region was zeroed by the
#                                 previous stage, which the stream orders)
#   _s_tr    again for the second transition, reducing the first's partial first
#   _s_fin   grid=(rows,)         reduce + mask + residual -> bf16 si
#
# Each CTA now reads ~100-200 KB of weights instead of 4.4 MB. Four launches
# cost ~10 us of host time, against ~115 us of device time saved.
# ---------------------------------------------------------------------------
@triton.jit
def _proj_body(pid, pidr, SIT, SII, TT, SF, WB, WF,
               R, CS: tl.constexpr, CIN: tl.constexpr, NCAT: tl.constexpr,
            CF: tl.constexpr, KPAD: tl.constexpr, KTOT: tl.constexpr,
            BK: tl.constexpr, BN: tl.constexpr, BM: tl.constexpr,
            EPS: tl.constexpr, EPSF: tl.constexpr, SIGMA: tl.constexpr,
            O_LNW: tl.constexpr, O_LNB: tl.constexpr, O_NW: tl.constexpr,
            O_NB: tl.constexpr, O_FW: tl.constexpr, O_FB: tl.constexpr,
            O_WS: tl.constexpr, S_SI0: tl.constexpr, S_Z: tl.constexpr,
            HAS_LNB: tl.constexpr, HAS_NLNB: tl.constexpr,
            EVEN_N: tl.constexpr, EVEN_R: tl.constexpr):
    """LayerNorm(concat) @ linear_s.T + LayerNorm(fourier) @ linear_n.T.

    One reduction loop over the *virtual* concat [si_trunk | si_input | fourier]:
    the [*, 833] cat is never built, and the Fourier tail chunks feed the same
    ``tl.dot`` accumulator as the LayerNorm'd input rows -- so linear_n, its
    LayerNorm and the whole 8-launch Fourier chain cost one more k-chunk.
    Programs split the output channels, which needs no cross-CTA reduction.
    """
    n = pid * BN + tl.arange(0, BN)
    rows = pidr * BM + tl.arange(0, BM)
    rm = rows < R
    ka = tl.arange(0, BK)
    tval = tl.load(TT).to(tl.float32)

    # --- Fourier embedding + its LayerNorm moments --------------------------
    fk = tl.arange(0, CF)
    e = _fourier(fk, tval, WF, O_FW, O_FB, SIGMA)
    fmean = tl.sum(e, 0) * (1.0 / CF)
    fd = e - fmean
    frstd = 1.0 / tl.sqrt(tl.sum(fd * fd, 0) * (1.0 / CF) + EPSF)

    # --- LayerNorm moments of the concat ------------------------------------
    # Shifted one-pass: subtract the row's own first element, so the
    # sq/N - off^2 cancellation only ever removes the *shifted* mean. Two
    # dynamic loops (si_trunk chunks, then si_input chunks) rather than one
    # unrolled loop with compile-time branches -- the unrolled form costs ~1500
    # register spills here.
    shift = tl.load(SIT + rows * CS, mask=rm, other=0.0).to(tl.float32)
    ssum = tl.zeros((BM,), dtype=tl.float32)
    ssq = tl.zeros((BM,), dtype=tl.float32)
    ap = SIT + rows[:, None] * CS + ka[None, :]
    for _ in range(CS // BK):
        a = tl.load(ap, mask=rm[:, None], other=0.0).to(tl.float32) - shift[:, None]
        ssum += tl.sum(a, 1)
        ssq += tl.sum(a * a, 1)
        ap += BK
    ap = SII + rows[:, None] * CIN + ka[None, :]
    kc = ka
    for _ in range((KPAD - CS) // BK):
        km = rm[:, None] & (kc < CIN)[None, :]
        a = tl.where(km, tl.load(ap, mask=km, other=0.0).to(tl.float32)
                     - shift[:, None], 0.0)
        ssum += tl.sum(a, 1)
        ssq += tl.sum(a * a, 1)
        ap += BK
        kc += BK
    off = ssum * (1.0 / NCAT)
    var = ssq * (1.0 / NCAT) - off * off
    mean = shift + off
    rstd = 1.0 / tl.sqrt(tl.maximum(var, 0.0) + EPS)

    # --- the projection ------------------------------------------------------
    nmask = (n < CS)[None, :]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    wp = WB + O_WS + ka[:, None] * CS + n[None, :]
    lnp = WF + O_LNW + ka
    lnbp = WF + O_LNB + ka
    ap = SIT + rows[:, None] * CS + ka[None, :]
    for _ in range(CS // BK):
        a = tl.load(ap, mask=rm[:, None], other=0.0).to(tl.float32)
        xn = (a - mean[:, None]) * rstd[:, None] * tl.load(lnp)[None, :]
        if HAS_LNB:
            xn = xn + tl.load(lnbp)[None, :]
        w = tl.load(wp) if EVEN_N else tl.load(wp, mask=nmask, other=0.0)
        acc = tl.dot(xn.to(tl.bfloat16), w, acc)
        ap += BK
        wp += BK * CS
        lnp += BK
        lnbp += BK
    ap = SII + rows[:, None] * CIN + ka[None, :]
    kc = ka
    for _ in range((KPAD - CS) // BK):
        a = tl.load(ap, mask=rm[:, None] & (kc < CIN)[None, :],
                    other=0.0).to(tl.float32)
        xn = (a - mean[:, None]) * rstd[:, None] * tl.load(lnp)[None, :]
        if HAS_LNB:
            xn = xn + tl.load(lnbp)[None, :]
        w = tl.load(wp) if EVEN_N else tl.load(wp, mask=nmask, other=0.0)
        acc = tl.dot(xn.to(tl.bfloat16), w, acc)
        ap += BK
        kc += BK
        wp += BK * CS
        lnp += BK
        lnbp += BK
    fi = ka
    nwp = WF + O_NW + ka
    nbp = WF + O_NB + ka
    for _ in range(CF // BK):
        u = (_fourier(fi, tval, WF, O_FW, O_FB, SIGMA) - fmean) * frstd
        u = u * tl.load(nwp)
        if HAS_NLNB:
            u = u + tl.load(nbp)
        xb = tl.broadcast_to(_rb(u)[None, :], (BM, BK)).to(tl.bfloat16)
        w = tl.load(wp) if EVEN_N else tl.load(wp, mask=nmask, other=0.0)
        acc = tl.dot(xb, w, acc)
        fi += BK
        nwp += BK
        nbp += BK
        wp += BK * CS

    sm = nmask if EVEN_R else (nmask & rm[:, None])
    o = SF + rows[:, None] * CS + n[None, :]
    tl.store(o + S_SI0, _rb(acc), mask=sm)
    # Zero the next stage's partial accumulator. Kernels on one stream are
    # ordered, so this is a free, sync-less way to prepare the atomics.
    tl.store(o + S_Z, 0.0, mask=sm)


@triton.jit
def _first(Z, ZOUT, RES, TOKI, ASYM, ENT, SYM, TM, SIT, SII, TT, SF, WB, WF,
           RP, RS, NP: tl.constexpr, NCOL: tl.constexpr,
           N: tl.constexpr, CZ: tl.constexpr, NCAT: tl.constexpr,
           NREL: tl.constexpr, KPOS: tl.constexpr, KCH: tl.constexpr,
           EPSZ: tl.constexpr, EPSTZ: tl.constexpr, PBM: tl.constexpr,
           HZ: tl.constexpr, PBH: tl.constexpr,
           O_LNW: tl.constexpr, O_LNB: tl.constexpr, O_T1: tl.constexpr,
           O_T2: tl.constexpr, O_T3: tl.constexpr, O_TE: tl.constexpr,
           O_GV: tl.constexpr, O_BV: tl.constexpr, O_WZ: tl.constexpr,
           O_A0: tl.constexpr, O_B0: tl.constexpr, O_W0: tl.constexpr,
           O_O0: tl.constexpr, O_A1: tl.constexpr, O_B1: tl.constexpr,
           O_W1: tl.constexpr, O_O1: tl.constexpr,
           HAS_RELPOS: tl.constexpr, HAS_MASK: tl.constexpr,
           HAS_LNB: tl.constexpr, HAS_TLNB: tl.constexpr,
           EVEN_RP: tl.constexpr,
           CS: tl.constexpr, CIN: tl.constexpr, NCATS: tl.constexpr,
           CF: tl.constexpr, KPAD: tl.constexpr, KTOT: tl.constexpr,
           BK: tl.constexpr, BN: tl.constexpr, SBM: tl.constexpr,
           EPSS: tl.constexpr, EPSF: tl.constexpr, SIGMA: tl.constexpr,
           O_SLNW: tl.constexpr, O_SLNB: tl.constexpr, O_NW: tl.constexpr,
           O_NB: tl.constexpr, O_FW: tl.constexpr, O_FB: tl.constexpr,
           O_WS: tl.constexpr, S_SI0: tl.constexpr, S_Z: tl.constexpr,
           HAS_SLNB: tl.constexpr, HAS_NLNB: tl.constexpr,
           EVEN_N: tl.constexpr, EVEN_RS: tl.constexpr):
    """Both branches' first stage in one launch.

    The pair branch and the single projection share no data, so putting them in
    one grid lets them run *concurrently* on different SMs -- two kernels on one
    stream cannot overlap, and neither fills the GPU (16 + 3 CTAs of 148 SMs).
    Worth the whole shorter branch: 11.7 us of pair time disappears behind the
    projection, plus one launch of host time.
    """
    pid = tl.program_id(0)
    if pid < NP:
        _pair_body(pid, Z, ZOUT, RES, TOKI, ASYM, ENT, SYM, TM, WB, WF,
                   RP, N, CZ, NCAT, NREL, KPOS, KCH, EPSZ, EPSTZ, PBM, HZ, PBH,
                   O_LNW, O_LNB, O_T1, O_T2, O_T3, O_TE, O_GV, O_BV, O_WZ,
                   O_A0, O_B0, O_W0, O_O0, O_A1, O_B1, O_W1, O_O1,
                   HAS_RELPOS, HAS_MASK, HAS_LNB, HAS_TLNB, EVEN_RP)
    else:
        q = pid - NP
        _proj_body(q % NCOL, q // NCOL, SIT, SII, TT, SF, WB, WF,
                   RS, CS, CIN, NCATS, CF, KPAD, KTOT, BK, BN, SBM,
                   EPSS, EPSF, SIGMA, O_SLNW, O_SLNB, O_NW, O_NB, O_FW, O_FB,
                   O_WS, S_SI0, S_Z, HAS_SLNB, HAS_NLNB, EVEN_N, EVEN_RS)


@triton.jit
def _tr_body(ph, pr, SF, TM, WB, WF, R,
             C: tl.constexpr, CP: tl.constexpr, H: tl.constexpr,
             HS: tl.constexpr,
             BH: tl.constexpr, BM: tl.constexpr, EPS: tl.constexpr,
             O_LNW: tl.constexpr, O_LNB: tl.constexpr, O_WAB: tl.constexpr,
             O_WO: tl.constexpr, S_X: tl.constexpr, S_R: tl.constexpr,
             S_XO: tl.constexpr, S_P: tl.constexpr, S_Z: tl.constexpr,
             HAS_LNB: tl.constexpr, HAS_MASK: tl.constexpr,
             REDUCE: tl.constexpr, WRITE_X: tl.constexpr,
             ZERO: tl.constexpr, VOL: tl.constexpr):
    """One residual SwiGLUTransition, hidden dim split over programs.

    Program (h, r) owns hidden units [h*HS, (h+1)*HS) of row block r: it norms
    the row, computes that slice of ``SiLU(linear_a(x)) * linear_b(x)``, and
    atomically adds its share of ``linear_out``'s output into an fp32 partial.
    The residual, the mask and the next LayerNorm are applied by whoever reads
    that partial (the next ``_s_tr`` or ``_s_fin``).

    With ``REDUCE`` the row is first completed from the *previous* transition's
    partial, so two transitions cost two launches rather than four.

    Every Triton tile dim must be a power of two, so a channel count like 384 is
    carried as a ``CP``-wide tile whose tail lanes are held at *exactly zero*:
    the packed weights are zero-padded (rows of ``linear_a``/``linear_b``,
    columns of ``linear_out``, and both LayerNorm vectors), which keeps the
    residual and the moments padded with zeros too. Only the variance needs an
    explicit mask, since the tail lanes of ``x - mean`` are not zero.
    """
    h0 = ph * HS
    rows = pr * BM + tl.arange(0, BM)
    rm = rows < R
    c = tl.arange(0, CP)
    cm = (c < C)[None, :] & rm[:, None]
    xp = SF + rows[:, None] * C + c[None, :]
    # ``VOL``: on the single-launch path these two reads cross a barrier rather
    # than a launch boundary, so they must bypass L1 and stay ordered after the
    # release (see ``_gbar``).
    x = tl.load(xp + S_X, mask=cm, other=0.0, volatile=VOL)
    if REDUCE:
        o = _rb(tl.load(xp + S_R, mask=cm, other=0.0, volatile=VOL))
        if HAS_MASK:
            o = _rb(o * _rb(tl.load(TM + rows, mask=rm, other=0.0).to(tl.float32))[:, None])
        x = _rb(x + o)
        if WRITE_X:
            tl.store(xp + S_XO, x, mask=cm)

    m = tl.sum(x, 1) * (1.0 / C)
    d = x - m[:, None]
    if C == CP:
        v = tl.sum(d * d, 1) * (1.0 / C)
    else:
        v = tl.sum(tl.where(c[None, :] < C, d * d, 0.0), 1) * (1.0 / C)
    xn = d * (1.0 / tl.sqrt(v + EPS))[:, None] * tl.load(WF + O_LNW + c)[None, :]
    if HAS_LNB:
        xn = xn + tl.load(WF + O_LNB + c)[None, :]
    xb = xn.to(tl.bfloat16)

    hc = h0 + tl.arange(0, BH)
    wabp = WB + O_WAB + c[:, None] * (2 * H) + hc[None, :]
    wop = WB + O_WO + hc[:, None] * C + c[None, :]
    km = (c < C)[:, None]
    nm = (c < C)[None, :]
    acc = tl.zeros((BM, CP), dtype=tl.float32)
    for _ in range(HS // BH):
        if C == CP:
            wa = tl.load(wabp)
            wbq = tl.load(wabp + H)
            wo = tl.load(wop)
        else:
            # Predicated at c < C: the tail lanes of the CP-wide tile issue no
            # load at all, so the padding costs FLOPs (which are free here) but
            # not bytes (which are not).
            wa = tl.load(wabp, mask=km, other=0.0)
            wbq = tl.load(wabp + H, mask=km, other=0.0)
            wo = tl.load(wop, mask=nm, other=0.0)
        a = _rb(tl.dot(xb, wa))
        b = _rb(tl.dot(xb, wbq))
        hh = _rb(_rb(a * tl.sigmoid(a)) * b).to(tl.bfloat16)
        acc = tl.dot(hh, wo, acc)
        wabp += BH
        wop += BH * C
    tl.atomic_add(xp + S_P, acc, mask=cm, sem="relaxed")
    if ZERO:
        # Prepare the *following* transition's accumulator. Ordered by the
        # launch boundary (four-launch path) or by the next barrier's release
        # (single-launch path); either way it lands before anyone adds to it.
        if h0 == 0:
            tl.store(xp + S_Z, 0.0, mask=cm)


@triton.jit
def _s_tr(SF, TM, WB, WF, R,
          C: tl.constexpr, CP: tl.constexpr, H: tl.constexpr, HS: tl.constexpr,
          BH: tl.constexpr, BM: tl.constexpr, EPS: tl.constexpr,
          O_LNW: tl.constexpr, O_LNB: tl.constexpr, O_WAB: tl.constexpr,
          O_WO: tl.constexpr, S_X: tl.constexpr, S_R: tl.constexpr,
          S_XO: tl.constexpr, S_P: tl.constexpr, S_Z: tl.constexpr,
          HAS_LNB: tl.constexpr, HAS_MASK: tl.constexpr,
          REDUCE: tl.constexpr, WRITE_X: tl.constexpr, ZERO: tl.constexpr):
    """One transition stage as its own launch (the four-launch fallback path)."""
    _tr_body(tl.program_id(0), tl.program_id(1), SF, TM, WB, WF, R,
             C, CP, H, HS, BH, BM, EPS, O_LNW, O_LNB, O_WAB, O_WO,
             S_X, S_R, S_XO, S_P, S_Z, HAS_LNB, HAS_MASK,
             REDUCE, WRITE_X, ZERO, False)


@triton.jit
def _fin_body(pc, pr, SF, TM, OUT, R, C: tl.constexpr, BN: tl.constexpr,
              BM: tl.constexpr, S_X: tl.constexpr, S_P: tl.constexpr,
              HAS_MASK: tl.constexpr, VOL: tl.constexpr):
    """Reduce the last transition's partial, mask, add the residual, store bf16.

    Split over output channels as well as rows: the work is one pass over 60 KB,
    so with a single program the kernel is all latency (4.6 us, of which ~2.5 us
    is the launch itself).
    """
    c = pc * BN + tl.arange(0, BN)
    rows = pr * BM + tl.arange(0, BM)
    rm = rows < R
    cm = (c < C)[None, :] & rm[:, None]
    xp = SF + rows[:, None] * C + c[None, :]
    x = tl.load(xp + S_X, mask=cm, other=0.0, volatile=VOL)
    o = _rb(tl.load(xp + S_P, mask=cm, other=0.0, volatile=VOL))
    if HAS_MASK:
        o = _rb(o * _rb(tl.load(TM + rows, mask=rm, other=0.0).to(tl.float32))[:, None])
    tl.store(OUT + rows[:, None] * C + c[None, :], _rb(x + o).to(tl.bfloat16),
             mask=cm)


@triton.jit
def _s_fin(SF, TM, OUT, R, C: tl.constexpr, BN: tl.constexpr,
           BM: tl.constexpr, S_X: tl.constexpr, S_P: tl.constexpr,
           HAS_MASK: tl.constexpr):
    """The final reduce as its own launch (the four-launch fallback path)."""
    _fin_body(tl.program_id(0), tl.program_id(1), SF, TM, OUT, R, C, BN, BM,
              S_X, S_P, HAS_MASK, False)


@triton.jit
def _prefetch(q, WB, SF, PRE0: tl.constexpr, PREN: tl.constexpr,
              PER: tl.constexpr, TILE: tl.constexpr, S_D: tl.constexpr):
    """Touch 1/NPRE of the later phases' weights, to leave them in L2.

    The benchmark flushes a 2x-L2 buffer before each timed call, so all ~4.7 MB
    of weights this forward reads come from DRAM, and the measured device time is
    DRAM *latency* bound (~120 GB/s aggregate, ~1.5% of this GPU's bandwidth)
    rather than bandwidth bound: 24 working CTAs cannot keep enough loads in
    flight. Phase 0 occupies 22 CTAs of 148 SMs for ~14 us, so the cheapest way
    to raise the in-flight count is to spend idle SMs fetching what phases 1-3
    will need.

    One whole ``TILE`` per load (8 KB, i.e. two 16-byte loads per thread) and a
    dynamic loop, so ``num_stages`` keeps several tiles in flight per CTA -- a
    prefetch that trickles is worthless, it has to finish inside phase 0.

    Read-only, bounds-checked against the end of the blob, and the reduction is
    stored to a scratch slot nobody reads -- it exists only so that the loads are
    not dead code.
    """
    base = PRE0 + q * PER
    off = tl.arange(0, TILE)
    acc = tl.zeros((TILE,), dtype=tl.float32)
    for i in range(PER // TILE):
        p = base + i * TILE + off
        acc += tl.load(WB + p, mask=p < PREN, other=0.0).to(tl.float32)
    tl.store(SF + S_D + q, tl.sum(acc))


# ---------------------------------------------------------------------------
# The whole forward in one launch: four phases separated by grid-wide barriers
# ---------------------------------------------------------------------------
@triton.jit
def _mega(Z, ZOUT, SOUT, RES, TOKI, ASYM, ENT, SYM, TM, SIT, SII, TT,
          SF, WB, WF, BAR, RP, RS,
          NCTA: tl.constexpr,
          NP: tl.constexpr, NCOL: tl.constexpr, N0: tl.constexpr,
          N1: tl.constexpr, N3: tl.constexpr, NH: tl.constexpr,
          NFC: tl.constexpr, NPRE: tl.constexpr, PRE0: tl.constexpr,
          PREN: tl.constexpr, PRE_PER: tl.constexpr, PRE_TILE: tl.constexpr,
          S_D: tl.constexpr,
          N: tl.constexpr, CZ: tl.constexpr, NCAT: tl.constexpr,
          NREL: tl.constexpr, KPOS: tl.constexpr, KCH: tl.constexpr,
          EPSZ: tl.constexpr, EPSTZ: tl.constexpr, PBM: tl.constexpr,
          HZ: tl.constexpr, PBH: tl.constexpr,
          O_LNW: tl.constexpr, O_LNB: tl.constexpr, O_T1: tl.constexpr,
          O_T2: tl.constexpr, O_T3: tl.constexpr, O_TE: tl.constexpr,
          O_GV: tl.constexpr, O_BV: tl.constexpr, O_WZ: tl.constexpr,
          O_A0: tl.constexpr, O_B0: tl.constexpr, O_W0: tl.constexpr,
          O_O0: tl.constexpr, O_A1: tl.constexpr, O_B1: tl.constexpr,
          O_W1: tl.constexpr, O_O1: tl.constexpr,
          HAS_RELPOS: tl.constexpr, HAS_MASK: tl.constexpr,
          HAS_LNB: tl.constexpr, HAS_TLNB: tl.constexpr,
          EVEN_RP: tl.constexpr,
          CS: tl.constexpr, CIN: tl.constexpr, NCATS: tl.constexpr,
          CF: tl.constexpr, KPAD: tl.constexpr, KTOT: tl.constexpr,
          BK: tl.constexpr, BN: tl.constexpr, SBM: tl.constexpr,
          EPSS: tl.constexpr, EPSF: tl.constexpr, SIGMA: tl.constexpr,
          O_SLNW: tl.constexpr, O_SLNB: tl.constexpr, O_NW: tl.constexpr,
          O_NB: tl.constexpr, O_FW: tl.constexpr, O_FB: tl.constexpr,
          O_WS: tl.constexpr, S_SI0: tl.constexpr, S_P1: tl.constexpr,
          HAS_SLNB: tl.constexpr, HAS_NLNB: tl.constexpr,
          EVEN_N: tl.constexpr, EVEN_RS: tl.constexpr,
          CSP: tl.constexpr, TH: tl.constexpr, THS: tl.constexpr,
          TBH: tl.constexpr, TEPS: tl.constexpr,
          U_A0: tl.constexpr, U_B0: tl.constexpr, U_W0: tl.constexpr,
          U_O0: tl.constexpr, U_A1: tl.constexpr, U_B1: tl.constexpr,
          U_W1: tl.constexpr, U_O1: tl.constexpr,
          S_SI1: tl.constexpr, S_P2: tl.constexpr, FBN: tl.constexpr):
    """The four dependent stages of the forward, in one launch.

    Phase boundaries are producer/consumer boundaries, which is the only reason
    the four-launch path exists; a grid-wide barrier replaces each of the three
    launch boundaries, saving three ``cuLaunchKernelEx`` calls (~3.6 us of host
    time each) and three kernel-to-kernel gaps, and leaving the fp32 partials in
    L2 across phases instead of being re-fetched by a fresh grid.

    Every CTA executes every barrier unconditionally -- the phase guards are
    *inside* the barrier-free stretches, never around a barrier -- so a CTA that
    has no work in a phase still arrives. Grid size is the max over phases, and
    ``_resident_ctas`` proves the whole grid is co-resident before this path is
    used at all.

    The pointer arguments are ordered so that everything that changes per call
    (both outputs, the five relpos features, the mask and the three inputs) is
    one contiguous run, which lets the host patch them into a prebuilt argument
    list with one slice assignment.
    """
    pid = tl.program_id(0)
    if pid < NP:
        _pair_body(pid, Z, ZOUT, RES, TOKI, ASYM, ENT, SYM, TM, WB, WF,
                   RP, N, CZ, NCAT, NREL, KPOS, KCH, EPSZ, EPSTZ, PBM, HZ, PBH,
                   O_LNW, O_LNB, O_T1, O_T2, O_T3, O_TE, O_GV, O_BV, O_WZ,
                   O_A0, O_B0, O_W0, O_O0, O_A1, O_B1, O_W1, O_O1,
                   HAS_RELPOS, HAS_MASK, HAS_LNB, HAS_TLNB, EVEN_RP)
    elif pid < N0:
        q = pid - NP
        _proj_body(q % NCOL, q // NCOL, SIT, SII, TT, SF, WB, WF,
                   RS, CS, CIN, NCATS, CF, KPAD, KTOT, BK, BN, SBM,
                   EPSS, EPSF, SIGMA, O_SLNW, O_SLNB, O_NW, O_NB, O_FW, O_FB,
                   O_WS, S_SI0, S_P1, HAS_SLNB, HAS_NLNB, EVEN_N, EVEN_RS)
    elif NPRE > 0 and pid < N0 + NPRE:
        _prefetch(pid - N0, WB, SF, PRE0, PREN, PRE_PER, PRE_TILE, S_D)
    _gbar(BAR, 0, NCTA)
    if pid < N1:
        _tr_body(pid % NH, pid // NH, SF, TM, WB, WF, RS,
                 CS, CSP, TH, THS, TBH, SBM, TEPS, U_A0, U_B0, U_W0, U_O0,
                 S_SI0, 0, 0, S_P1, S_P2, HAS_TLNB, HAS_MASK,
                 False, False, True, True)
    _gbar(BAR, 1, NCTA)
    if pid < N1:
        _tr_body(pid % NH, pid // NH, SF, TM, WB, WF, RS,
                 CS, CSP, TH, THS, TBH, SBM, TEPS, U_A1, U_B1, U_W1, U_O1,
                 S_SI0, S_P1, S_SI1, S_P2, 0, HAS_TLNB, HAS_MASK,
                 True, True, False, True)
    _gbar(BAR, 2, NCTA)
    if pid < N3:
        _fin_body(pid % NFC, pid // NFC, SF, TM, SOUT, RS, CS, FBN, SBM,
                  S_SI1, S_P2, HAS_MASK, True)


# ---------------------------------------------------------------------------
# Host side: weight packing, launch plan, direct launcher
# ---------------------------------------------------------------------------
def _resident_ctas(kern, num_warps: int, dev) -> int:
    """How many CTAs of *kern* can be resident on this GPU **at once**.

    The grid-wide barrier in ``_gbar`` is only legal if every CTA of the grid is
    scheduled before any of them waits, so the single-launch path is gated on
    this number rather than on the belief that a small grid "obviously" fits: a
    barrier with one unscheduled CTA hangs the process, which is a far worse
    outcome than the four-launch path's few microseconds.

    Occupancy limits, all per SM: shared memory, registers (allocated per warp,
    rounded up to a multiple of 8 per thread), resident threads, and the
    hardware's 32-CTA cap. Returns 0 if anything cannot be determined, which
    sends the plan to the four-launch path.
    """
    try:
        # n_regs is only known once the binary is loaded, which a compile-only
        # warmup does not do.
        kern._init_handles()
        props = torch.cuda.get_device_properties(dev)
        sms = int(props.multi_processor_count)
        smem_sm = int(props.shared_memory_per_multiprocessor)
        regs_sm = int(props.regs_per_multiprocessor)
        thr_sm = int(props.max_threads_per_multi_processor)
        smem = int(kern.metadata.shared) + 2048   # + slack for static/driver use
        regs = int(kern.n_regs)
    except Exception:
        return 0
    threads = num_warps * 32
    if min(sms, smem_sm, regs_sm, thr_sm, threads) <= 0 or regs <= 0:
        return 0
    per_sm = min(smem_sm // max(smem, 1),
                 regs_sm // (((regs + 7) // 8 * 8) * threads),
                 thr_sm // threads,
                 32)
    return per_sm * sms if per_sm >= 1 else 0


class _Blob:
    """Concatenate weight blocks into one tensor, returning element offsets.

    One pointer per dtype instead of one per weight: a launch of either kernel
    passes ``(WB, WF)`` and reads everything else off compile-time offsets.
    Blocks are aligned so Triton's divisibility analysis still sees each block
    start as 16-byte aligned.
    """

    def __init__(self, dtype: torch.dtype, device):
        self._parts: list[torch.Tensor] = []
        self._n = 0
        self._dtype = dtype
        self._device = device

    def add(self, t: torch.Tensor) -> int:
        pad = (-self._n) % _ALIGN
        if pad:
            self._parts.append(torch.zeros(pad, dtype=self._dtype,
                                           device=self._device))
            self._n += pad
        off = self._n
        flat = t.reshape(-1).to(device=self._device, dtype=self._dtype)
        self._parts.append(flat)
        self._n += flat.numel()
        return off

    def build(self) -> torch.Tensor:
        if not self._parts:
            return torch.zeros(_ALIGN, dtype=self._dtype, device=self._device)
        return torch.cat(self._parts)


def _lnw(ln: nn.Module, n: int, dev, pad: int = 0):
    """(weight, bias) of a LayerNorm as fp32 vectors of length ``n + pad``.

    ``create_scale=False`` becomes an explicit ones vector, so the kernel has a
    single code path; a missing bias stays ``None`` (a constexpr flag drops the
    add). Padding is zero on both, so padded reduction lanes contribute nothing.
    """
    w = ln.weight
    if w is None:
        w32 = torch.ones(n, dtype=torch.float32, device=dev)
    else:
        w32 = w.detach().float().reshape(-1)
    b = ln.bias
    b32 = None if b is None else b.detach().float().reshape(-1)
    if pad:
        w32 = torch.cat([w32, torch.zeros(pad, dtype=torch.float32, device=dev)])
        if b32 is not None:
            b32 = torch.cat([b32,
                             torch.zeros(pad, dtype=torch.float32, device=dev)])
    return w32, b32


class FourierEmbedding(nn.Module):
    """Fourier time embedding for diffusion conditioning.

    Buffers ``w`` / ``b`` match the reference's seeded initialization (the
    benchmark loads them through ``state_dict``); the forward is kept for the
    fallback path -- the fast path evaluates the same chain inside
    ``_single_fwd``.

    Args:
        c: Embedding dimension (256 in the reference)
        seed: Random seed for weight initialization
    """

    def __init__(self, c: int = 256, seed: int = 42):
        super().__init__()
        self.c = c
        generator = torch.Generator()
        generator.manual_seed(seed)
        self.register_buffer(
            "w", torch.randn(c, generator=generator),
        )
        self.register_buffer(
            "b", torch.randn(c, generator=generator),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        x = t * self.w + self.b
        return torch.cos(2 * math.pi * x)


class DiffusionConditioning(nn.Module):
    """Conditioning for diffusion module.

    Matches the reference:
    - Pair: concat([zij_trunk, relpos], dim=-1) -> LayerNorm -> Linear -> 2x SwiGLU transition
    - Single: concat([si_trunk, si_input], dim=-1) -> LayerNorm -> Linear + fourier -> 2x SwiGLU transition

    Reference: openfold3/core/model/layers/diffusion_conditioning.py

    Args:
        c_s: Single representation channel dimension
        c_z: Pair representation channel dimension
        c_s_input: Input single representation dimension (449)
        sigma_data: Noise level scaling for Fourier embedding
        relpos_k: Maximum relative position for pair bias
        max_relative_chain: Maximum relative chain index
        c_fourier_emb: Fourier embedding dimension (256)
        seed_fourier_emb: Fourier embedding random seed
    """

    def __init__(
        self,
        c_s: int = 384,
        c_z: int = 128,
        c_s_input: int = 449,
        sigma_data: float = 16.0,
        relpos_k: int = 32,
        max_relative_chain: int = 2,
        c_fourier_emb: int = 256,
        seed_fourier_emb: int = 42,
    ):
        super().__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.c_s_input = c_s_input
        self.c_fourier_emb = c_fourier_emb
        self.sigma_data = sigma_data
        self.relpos_k = relpos_k
        self.max_relative_chain = max_relative_chain

        num_rel_pos_bins = 2 * relpos_k + 2
        num_rel_token_bins = 2 * relpos_k + 2
        num_rel_chain_bins = 2 * max_relative_chain + 2
        num_same_entity_features = 1
        num_relpos_dims = (
            num_rel_pos_bins + num_rel_token_bins
            + num_rel_chain_bins + num_same_entity_features
        )
        self._bins = (num_rel_pos_bins, num_rel_token_bins,
                      num_rel_chain_bins, num_relpos_dims)

        self.layer_norm_z = LayerNorm(num_relpos_dims + c_z, create_offset=False)
        self.linear_z = Linear(num_relpos_dims + c_z, c_z, bias=False)

        self.transition_z = nn.ModuleList([
            SwiGLUTransition(c_in=c_z, n=2)
            for _ in range(2)
        ])

        self.layer_norm_s = LayerNorm(c_s + c_s_input, create_offset=False)
        self.linear_s = Linear(c_s + c_s_input, c_s, bias=False)

        self.fourier_emb = FourierEmbedding(c=c_fourier_emb, seed=seed_fourier_emb)
        self.layer_norm_n = LayerNorm(c_fourier_emb, create_offset=False)
        self.linear_n = Linear(c_fourier_emb, c_s, bias=False)

        self.transition_s = nn.ModuleList([
            SwiGLUTransition(c_in=c_s, n=2)
            for _ in range(2)
        ])

        # Packed-weight plan, built on the first forward that can use it (the
        # benchmark loads reference weights after construction, so nothing may
        # be derived here), and dropped whenever weights move or are reloaded.
        self._plan = None
        self._plan_failed = False
        self._fast = None
        # Per-instance override of the module-level default, so the four-launch
        # fallback can be exercised without re-importing.
        self._persist = _PERSIST

    # -- cache invalidation --------------------------------------------------
    # The packed blobs are copies of the parameters, so any weight reload or
    # device move has to drop them. ``load_state_dict`` calls
    # ``_load_from_state_dict`` on every module in the tree (including this
    # one), and ``.to()`` / ``.cuda()`` / ``.float()`` all route through
    # ``_apply``, so these two overrides cover every supported mutation.
    def _drop_plan(self) -> None:
        self._plan = None
        self._plan_failed = False
        self._fast = None

    def _load_from_state_dict(self, *args, **kwargs):
        self._drop_plan()
        return super()._load_from_state_dict(*args, **kwargs)

    def _apply(self, *args, **kwargs):
        self._drop_plan()
        return super()._apply(*args, **kwargs)

    # -- plan ----------------------------------------------------------------
    def _pack_transitions(self, mods, c: int, cp: int, wb: _Blob, wf: _Blob):
        """Offsets for a pair of SwiGLUTransition blocks, or None if unsupported.

        ``linear_a`` and ``linear_b`` are stacked into one [c, 2h] operand and
        pre-transposed for the ``tl.dot`` layout, so each block's two input
        projections read one tile; ``linear_out`` is pre-transposed to [h, c].
        Neither is padded up to the kernel's power-of-two channel tile -- the
        kernel predicates the tail lanes instead, which reads 33% fewer weight
        bytes at c_s=384.
        """
        out = []
        h = None
        eps = None
        has_b = None
        for m in mods:
            wa = m.swiglu.linear_a.weight
            wbb = m.swiglu.linear_b.weight
            wo = m.linear_out.weight
            if (m.swiglu.linear_a.bias is not None
                    or m.swiglu.linear_b.bias is not None
                    or m.linear_out.bias is not None):
                return None
            hh = wa.shape[0]
            if (wa.shape != (hh, c) or wbb.shape != (hh, c)
                    or wo.shape != (c, hh)):
                return None
            lw, lb = _lnw(m.layer_norm, c, wa.device, pad=cp - c)
            if h is None:
                h, eps, has_b = hh, m.layer_norm.eps, lb is not None
            elif hh != h or m.layer_norm.eps != eps or (lb is not None) != has_b:
                return None
            o_lnw = wf.add(lw)
            o_lnb = wf.add(lb) if lb is not None else o_lnw
            o_wab = wb.add(torch.cat([wa, wbb], 0).t().contiguous())
            o_wo = wb.add(wo.t().contiguous())
            out += [o_lnw, o_lnb, o_wab, o_wo]
        return out, h, eps, has_b

    def _sources(self):
        """Every tensor object ``_build_plan`` copies into the packed blobs.

        The blobs are copies, so the plan is only valid while these are the live
        weights. Compared by *identity* on every re-plan, which catches a weight
        replaced behind the module's back (``m.linear_z.weight = ...``, a direct
        ``_parameters[...] = ...``): ``load_state_dict`` and ``_apply`` already
        drop the plan through their hooks, and the per-call fast path keeps two
        of these as sentinels. An *in-place* mutation of a weight's storage
        (``p.copy_()``) changes no object and is not detectable here -- the
        frozen ``candidate/L1/layer_norm.py`` caches its fp32 affine on the same
        terms, so the baseline is equally stale in that case.
        """
        out = [self.linear_z.weight, self.linear_s.weight, self.linear_n.weight,
               self.fourier_emb.w, self.fourier_emb.b]
        for ln in (self.layer_norm_z, self.layer_norm_s, self.layer_norm_n):
            out += [ln.weight, ln.bias]
        for mods in (self.transition_z, self.transition_s):
            for m in mods:
                out += [m.layer_norm.weight, m.layer_norm.bias,
                        m.swiglu.linear_a.weight, m.swiglu.linear_b.weight,
                        m.linear_out.weight]
        return tuple(out)

    def _build_plan(self):
        """Pack every weight and resolve both launch configs, or return None.

        Everything the fast path assumes about the module tree is checked here
        once; anything unexpected (a non-bf16 weight, a hidden dim the tiling
        cannot cover, a reshaped submodule) leaves ``_plan`` None and the forward
        on the reference path.
        """
        wz = self.linear_z.weight
        ws = self.linear_s.weight
        wn = self.linear_n.weight
        dev = wz.device
        cz, cs, cin, cf = self.c_z, self.c_s, self.c_s_input, self.c_fourier_emb
        nb1, nb2, nb3, nrel = self._bins
        ncat_z, ncat_s = nrel + cz, cs + cin
        if (dev.type != "cuda" or wz.dtype is not torch.bfloat16
                or ws.dtype is not torch.bfloat16
                or wn.dtype is not torch.bfloat16
                or wz.shape != (cz, ncat_z) or ws.shape != (cs, ncat_s)
                or wn.shape != (cs, cf)
                or self.linear_z.bias is not None
                or self.linear_s.bias is not None
                or self.linear_n.bias is not None
                or self.fourier_emb.w.shape != (cf,)
                or self.fourier_emb.b.shape != (cf,)
                or cz % 16 or cs % _S_BK or cf % _S_BK or cs % 16):
            return None

        wb = _Blob(torch.bfloat16, dev)
        wf = _Blob(torch.float32, dev)

        # --- pair projection ------------------------------------------------
        lnzw, lnzb = _lnw(self.layer_norm_z, ncat_z, dev)
        gw = wz[:, cz:].detach().float() * lnzw[cz:][None, :]     # [cz, nrel]

        def prefix(block):
            """[cz, nb] -> [nb + 1, cz]: row c is the sum of the first c columns."""
            cs_ = torch.cumsum(block, dim=1)
            z = torch.zeros(block.shape[0], 1, dtype=torch.float32, device=dev)
            return torch.cat([z, cs_], 1).t().contiguous()

        o_lnw = wf.add(lnzw[:cz])
        o_lnb = wf.add(lnzb[:cz]) if lnzb is not None else o_lnw
        o_t1 = wf.add(prefix(gw[:, :nb1]))
        o_t2 = wf.add(prefix(gw[:, nb1:nb1 + nb2]))
        o_t3 = wf.add(prefix(gw[:, nb1 + nb2 + 1:]))
        o_te = wf.add(gw[:, nb1 + nb2].contiguous())
        o_gv = wf.add(gw.sum(1))
        if lnzb is not None:
            o_bv = wf.add((wz[:, cz:].detach().float()
                           * lnzb[cz:][None, :]).sum(1))
        else:
            o_bv = o_gv
        o_wz = wb.add(wz[:, :cz].t().contiguous())

        packed_z = self._pack_transitions(self.transition_z, cz, cz, wb, wf)
        if packed_z is None:
            return None
        off_z, hz, eps_tz, tlnb = packed_z
        if hz % _PAIR_BH or hz % 16:
            return None

        # --- single projection ----------------------------------------------
        csp = triton.next_power_of_2(cs)
        kpad = cs + (-cin % _S_BK) + cin
        ktot = kpad + cf
        lnsw, lnsb = _lnw(self.layer_norm_s, ncat_s, dev, pad=kpad - ncat_s)
        lnnw, lnnb = _lnw(self.layer_norm_n, cf, dev)
        o_slnw = wf.add(lnsw)
        o_slnb = wf.add(lnsb) if lnsb is not None else o_slnw
        o_nw = wf.add(lnnw)
        o_nb = wf.add(lnnb) if lnnb is not None else o_nw
        o_fw = wf.add(self.fourier_emb.w.detach().float())
        o_fb = wf.add(self.fourier_emb.b.detach().float())
        wsblk = torch.zeros(ktot, cs, dtype=torch.bfloat16, device=dev)
        wsblk[:ncat_s] = ws.detach().t()
        wsblk[kpad:] = wn.detach().t()
        o_ws = wb.add(wsblk)

        packed_s = self._pack_transitions(self.transition_s, cs, csp, wb, wf)
        if packed_s is None:
            return None
        off_s, hs, eps_ts, tlnb_s = packed_s
        if (hs % _S_HS or _S_HS % _S_BH or hs % 16 or tlnb_s != tlnb
                or eps_ts != eps_tz):
            return None

        blob_b = wb.build()
        blob_f = wf.build()
        pair_const = (cz, ncat_z, nrel, float(self.relpos_k),
                      float(self.max_relative_chain),
                      self.layer_norm_z.eps, eps_tz, _PAIR_BM, hz, _PAIR_BH,
                      o_lnw, o_lnb, o_t1, o_t2, o_t3, o_te, o_gv, o_bv, o_wz,
                      *off_z)
        proj_const = (cs, cin, ncat_s, cf, kpad, ktot, _S_BK, _S_BN, _S_BM,
                      self.layer_norm_s.eps, self.layer_norm_n.eps,
                      float(self.sigma_data),
                      o_slnw, o_slnb, o_nw, o_nb, o_fw, o_fb, o_ws)
        tr_const = tuple(
            (cs, csp, hs, _S_HS, _S_BH, _S_BM, eps_ts) + tuple(off_s[4 * i:4 * i + 4])
            for i in range(2))
        # Barrier counters for the single-launch path: one slot per barrier,
        # monotonically increasing, never reset (see ``_gbar``). Allocated once
        # per plan, so no per-call allocation and no per-call zeroing.
        bar = torch.zeros(8, dtype=torch.int64, device=dev)
        return {
            "wb": blob_b, "wf": blob_f,
            "wbp": blob_b.data_ptr(), "wfp": blob_f.data_ptr(),
            "pair": pair_const, "proj": proj_const, "tr": tr_const,
            "trm": (csp, hs, _S_HS, _S_BH, eps_ts) + tuple(off_s),
            # The bf16-blob span holding everything phases 1-3 read: the two
            # single transitions' packed [c, 2h] and [h, c] weights, which
            # ``_pack_transitions`` appends last and in order. Prefetch CTAs warm
            # exactly this range (see ``_prefetch``).
            "pre": (off_s[2], off_s[7] + hs * cs),
            "bar": bar, "barp": bar.data_ptr(),
            "csp": csp, "hs": hs,
            "tlnb": tlnb, "dev": dev,
            # The blobs are copies, so a weight replaced *without* going through
            # load_state_dict / _apply (``p.data = ...``, or the harness's
            # buffer cast, both of which swap the tensor object in place) has to
            # be caught per call. Two dict lookups and two identity compares.
            "src": (self.linear_z._parameters["weight"],
                    self.fourier_emb._buffers["w"]),
            "srcs": self._sources(),
        }

    # -- reference path ------------------------------------------------------
    def _reference(self, batch, t, si_input, si_trunk, zij_trunk,
                   use_conditioning, chunk_size=None):
        """The baseline forward, verbatim -- anything the fast path declines."""
        if use_conditioning:
            if "asym_id" in batch:
                relpos_zij = relpos_complex(
                    batch=batch,
                    max_relative_idx=self.relpos_k,
                    max_relative_chain=self.max_relative_chain,
                ).to(dtype=zij_trunk.dtype)
            else:
                relpos_dim = self.linear_z.weight.shape[-1] - self.c_z
                relpos_zij = zij_trunk.new_zeros(
                    zij_trunk.shape[:-1] + (relpos_dim,),
                )
            zij = torch.cat([zij_trunk, relpos_zij], dim=-1)
            zij = self.linear_z(self.layer_norm_z(zij))
            si = torch.cat([si_trunk, si_input], dim=-1)
            si = self.linear_s(self.layer_norm_s(si))
        else:
            zij = zij_trunk.new_zeros(zij_trunk.shape)
            si = si_trunk.new_zeros(si_trunk.shape[:-1] + (self.c_s,))

        n = 0.25 * torch.log(t / self.sigma_data)
        n_emb = self.fourier_emb(n.unsqueeze(-1) if n.dim() == 0 else n)
        si = si + self.linear_n(self.layer_norm_n(n_emb)).unsqueeze(-2)

        token_mask = batch.get("token_mask")
        if token_mask is not None:
            pair_mask = token_mask[..., :, None] * token_mask[..., None, :]
        else:
            pair_mask = None

        for layer in self.transition_z:
            zij = zij + layer(zij, mask=pair_mask)
        for layer in self.transition_s:
            si = si + layer(si, mask=token_mask)
        return si, zij

    # -- fast path setup -----------------------------------------------------
    def _setup(self, batch, t, si_input, si_trunk, zij_trunk):
        """Compile both kernels for this input signature, then memoize their
        launchers.

        Triton's ``kernel[grid](...)`` re-binds and re-specializes every argument
        on each call -- tens of microseconds of Python here, against ~150 us for
        the whole operator. Everything it derives is invariant for this module
        (every argument is a constexpr or a pointer) except pointer alignment,
        hence the ``& 15`` guard in ``forward`` and the refusal to memoize off a
        misaligned first call. So we hold ``CompiledKernel.run.launch`` -- the
        generated C entry point -- plus the invariant prefix
        (function handle, cooperative/PDL flags, scratch slots, packed metadata,
        hook slots) and the constexpr suffix, and a launch becomes two
        star-unpacks.
        """
        plan = self._plan
        if plan is not None:
            # A re-plan means something about this call did not match the
            # compiled program -- possibly a weight that was swapped underneath
            # us, in which case the packed blobs are stale and must be rebuilt
            # rather than reused.
            srcs = self._sources()
            if len(srcs) != len(plan["srcs"]) or any(
                    a is not b for a, b in zip(srcs, plan["srcs"])):
                plan = self._plan = None
        if plan is None:
            if self._plan_failed:
                return None
            plan = self._plan = self._build_plan()
            if plan is None:
                self._plan_failed = True
                return None
        dev = plan["dev"]
        if si_trunk.device != dev or si_trunk.dtype is not torch.bfloat16:
            return None
        n_tok = si_trunk.shape[-2]
        if (si_input.shape[-1] != self.c_s_input or si_trunk.shape[-1] != self.c_s
                or zij_trunk.shape[-1] != self.c_z
                or zij_trunk.shape[-2] != n_tok or zij_trunk.shape[-3] != n_tok
                or si_input.shape[:-1] != si_trunk.shape[:-1]
                or zij_trunk.shape[:-3] != si_trunk.shape[:-2]
                or t.dim() != 0 or t.dtype is not torch.bfloat16
                or n_tok < 1 or not si_trunk.is_contiguous()
                or not si_input.is_contiguous() or not zij_trunk.is_contiguous()
                or si_input.device != dev or zij_trunk.device != dev
                or t.device != dev):
            return None
        idx = self._batch_idx(batch, si_trunk.shape[:-1])
        if idx is False:
            return None
        has_relpos = idx is not None
        tm = batch.get("token_mask")
        has_mask = tm is not None
        if has_mask and (type(tm) is not torch.Tensor
                         or tm.dtype is not torch.bfloat16
                         or tm.shape != si_trunk.shape[:-1]
                         or not tm.is_contiguous() or tm.device != dev):
            return None

        rows_p = zij_trunk.numel() // self.c_z
        rows_s = si_trunk.numel() // self.c_s
        cs, csp = self.c_s, plan["csp"]
        # fp32 scratch: si after the projection, si after the first transition,
        # and one partial accumulator per transition. 96 KB at the captured
        # shape, with the offsets baked in as constexprs.
        #
        # Owned by the *plan*, not allocated per call: `torch.empty` of this size
        # is 1.5 us of the ~20 us the whole forward spends on the host, and every
        # byte of it is written before it is read (the projection writes si0 and
        # zeroes the first accumulator, each transition zeroes the next one), so
        # nothing carries between calls. The one constraint this adds is that a
        # single module instance cannot have two forwards *in flight
        # concurrently* on different streams -- the same constraint a cuBLAS
        # workspace carries. Sequential calls, including a queue of hundreds of
        # unsynchronized ones on one stream, are ordered by the stream and fine.
        sz = rows_s * cs
        s_si0, s_si1, s_p1, s_p2 = 0, sz, 2 * sz, 3 * sz
        s_dummy = 4 * sz          # one fp32 per prefetch CTA, written, never read
        need = 4 * sz + max(_PRE_CTAS, 1)
        sf = plan.get("scr")
        if sf is None or sf.numel() < need:
            sf = plan["scr"] = torch.empty(need, dtype=torch.float32,
                                           device=dev)
        zij = torch.empty_like(zij_trunk)
        si = torch.empty_like(si_trunk)
        dummy = zij_trunk
        tmt = tm if has_mask else dummy
        lnb_s = self.layer_norm_s.bias is not None
        tlnb = plan["tlnb"]

        ncol = triton.cdiv(cs, _S_BN)
        npair = triton.cdiv(rows_p, _PAIR_BM)
        nr = triton.cdiv(rows_s, _S_BM)
        nh = plan["hs"] // _S_HS
        # Programs per phase, and the grid of the merged kernel: the max over
        # phases, with the idle tail still arriving at every barrier.
        n0, n1, n3 = npair + ncol * nr, nh * nr, ncol * nr
        # Prefetch CTAs sit past phase 0's own programs; they are part of the
        # grid, so they arrive at every barrier like everyone else. Sized so the
        # weight span divides evenly into whole tiles.
        pre0, pren = plan["pre"]
        # Never let the prefetch push the grid past one CTA per SM: that would
        # cost the *whole* single-launch win (~10 us of host) at shapes where the
        # working grid alone still fits, in exchange for ~2.5 us of hidden device
        # time. At the captured shape phase 0 uses 22 of 148 SMs, so this is not
        # binding there.
        try:
            sms = torch.cuda.get_device_properties(dev).multi_processor_count
        except Exception:
            sms = 0
        npre = max(0, min(_PRE_CTAS, sms - n0))
        pre_per = 0
        if npre:
            pre_per = -(-(pren - pre0) // npre)             # ceil
            pre_per = -(-pre_per // _PRE_TILE) * _PRE_TILE  # whole tiles
        ncta = max(n0 + npre, n1, n3)
        fargs0 = (zij_trunk, zij,
                  idx[0] if has_relpos else dummy,
                  idx[1] if has_relpos else dummy,
                  idx[2] if has_relpos else dummy,
                  idx[3] if has_relpos else dummy,
                  idx[4] if has_relpos else dummy,
                  tmt, si_trunk, si_input, t, sf, plan["wb"], plan["wf"])
        fconst0 = ((rows_p, rows_s, npair, ncol, n_tok) + plan["pair"]
                   + (has_relpos, has_mask,
                      self.layer_norm_z.bias is not None, tlnb,
                      rows_p % _PAIR_BM == 0)
                   + plan["proj"]
                   + (s_si0, s_p1, lnb_s,
                      self.layer_norm_n.bias is not None,
                      cs % _S_BN == 0, rows_s % _S_BM == 0))
        targs = (sf, tmt, plan["wb"], plan["wf"])
        tconst = (
            (rows_s,) + plan["tr"][0] + (s_si0, 0, 0, s_p1, s_p2,
                                         tlnb, has_mask, False, False, True),
            (rows_s,) + plan["tr"][1] + (s_si0, s_p1, s_si1, s_p2, 0,
                                         tlnb, has_mask, True, True, False),
        )
        fargs = (sf, tmt, si)
        fconst = (rows_s, cs, _S_BN, _S_BM, s_si1, s_p2, has_mask)

        # --- one launch, if the grid is provably co-resident -----------------
        margs = (zij_trunk, zij, si,
                 idx[0] if has_relpos else dummy,
                 idx[1] if has_relpos else dummy,
                 idx[2] if has_relpos else dummy,
                 idx[3] if has_relpos else dummy,
                 idx[4] if has_relpos else dummy,
                 tmt, si_trunk, si_input, t,
                 sf, plan["wb"], plan["wf"], plan["bar"])
        mconst = ((rows_p, rows_s, ncta, npair, ncol, n0, n1, n3, nh, ncol,
                   npre, pre0, pren, pre_per, _PRE_TILE, s_dummy, n_tok)
                  + plan["pair"]
                  + (has_relpos, has_mask,
                     self.layer_norm_z.bias is not None, tlnb,
                     rows_p % _PAIR_BM == 0)
                  + plan["proj"]
                  + (s_si0, s_p1, lnb_s,
                     self.layer_norm_n.bias is not None,
                     cs % _S_BN == 0, rows_s % _S_BM == 0)
                  + plan["trm"] + (s_si1, s_p2, _S_BN))
        mkern = None
        if self._persist:
            # Compile *without* launching, then check occupancy: the barrier is
            # only legal if the whole grid is resident at once.
            try:
                mkern = _mega.warmup(*margs, *mconst, grid=(ncta,),
                                     num_warps=_MEGA_WARPS,
                                     num_stages=_MEGA_STAGES)
            except Exception:
                mkern = None
            if mkern is not None and hasattr(mkern, "result"):
                mkern = mkern.result()
            if mkern is not None and _resident_ctas(mkern, _MEGA_WARPS,
                                                    dev) < ncta:
                mkern = None

        if mkern is not None:
            # A previous plan of this module may have left the counters at an
            # arrival count that is not a multiple of *this* plan's NCTA, which
            # would misalign the generations, so restart from zero (stream
            # ordered against the launch below).
            plan["bar"].zero_()
            grids = ((ncta, 1),)
            kerns = (_mega[(ncta,)](*margs, *mconst,
                                    num_warps=_MEGA_WARPS,
                                    num_stages=_MEGA_STAGES),)
            consts = (mconst,)
        else:
            grids = ((npair + ncol * nr, 1),
                     (nh, nr),
                     (nh, nr),
                     (ncol, nr))
            kerns = (
                # The pair path shares this launch, so it shares its warp count
                # and pipeline depth; measured the same for the pair shape at 4
                # and 8 warps, so the projection's choice wins.
                _first[grids[0]](*fargs0, *fconst0, num_warps=_S_PROJ_WARPS,
                                 num_stages=_S_PROJ_STAGES),
                _s_tr[grids[1]](*targs, *tconst[0], num_warps=_S_TR_WARPS,
                                num_stages=_S_TR_STAGES),
                _s_tr[grids[2]](*targs, *tconst[1], num_warps=_S_TR_WARPS,
                                num_stages=_S_TR_STAGES),
                _s_fin[grids[3]](*fargs, *fconst, num_warps=4, num_stages=2),
            )
            consts = (fconst0, tconst[0], tconst[1], fconst)

        plans = []
        for kern in kerns:
            run = None if kern is None else kern.run
            raw = getattr(run, "launch", None)
            if (raw is None
                    or getattr(run, "global_scratch_size", None) != 0
                    or getattr(run, "profile_scratch_size", None) != 0):
                plans = None
                break
            plans.append((raw, (kern.function, run.launch_cooperative_grid,
                                run.launch_pdl, None, None,
                                kern.packed_metadata, None, None, None)))
        shp_m_t = si_trunk.shape[:-1]
        aligned = not any(x.data_ptr() & 15 for x in
                          (zij_trunk, zij, si_trunk, si_input, t, si, sf,
                           plan["wb"], plan["wf"]))
        if plans is not None and aligned and si_trunk.get_device() == _cur_device():
            self._fast = (
                tuple(zip((p[0] for p in plans), (p[1] for p in plans),
                          consts, grids)),
                si_trunk.get_device(), plan["wbp"], plan["wfp"], sf.data_ptr(),
                tuple(si_input.shape), tuple(si_trunk.shape),
                tuple(zij_trunk.shape), tuple(si_trunk.shape[:-1]),
                has_relpos, has_mask, plan["src"][0], plan["src"][1],
                # The single-launch path's whole argument list, prebuilt: only
                # the stream and the 12 per-call pointers (one contiguous run,
                # see ``_mega``) are patched in, which is ~0.7 us cheaper per
                # call than re-unpacking ~100 arguments at the call site.
                ([grids[0][0], 1, 1, 0, *plans[0][1],
                  *(x.data_ptr() for x in margs), *mconst]
                 if mkern is not None else None),
                # The per-call guard's expected value, assembled once here so the
                # hot path is one comparison (see ``forward``).
                (tuple(si_input.shape), tuple(si_trunk.shape),
                 tuple(zij_trunk.shape), torch.bfloat16, torch.bfloat16,
                 torch.bfloat16, torch.bfloat16, True, True, True, 0,
                 si_trunk.get_device(), si_trunk.get_device(), False),
                ((torch.bfloat16,) * 6 + (tuple(shp_m_t),) * 6 + (True,) * 6
                 if (has_relpos and has_mask) else None))
        return si, zij

    @staticmethod
    def _batch_idx(batch, shp):
        """The five integer features relpos needs, or None (no relpos) / False.

        ``False`` means "present but not in the form the kernel reads" -- a
        non-bf16 index, a strided view, a shape that does not match the tokens --
        and sends the call to the reference path, which will reproduce whatever
        the baseline does with it (including raising).
        """
        if "asym_id" not in batch:
            return None
        out = []
        for k in ("residue_index", "token_index", "asym_id", "entity_id",
                  "sym_id"):
            v = batch.get(k)
            if (type(v) is not torch.Tensor or v.dtype is not torch.bfloat16
                    or v.shape != shp or not v.is_contiguous()):
                return False
            out.append(v)
        return out

    # -- forward -------------------------------------------------------------
    def forward(
        self,
        batch: dict,
        t: torch.Tensor,
        si_input: torch.Tensor,
        si_trunk: torch.Tensor,
        zij_trunk: torch.Tensor,
        use_conditioning: bool,
        chunk_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            batch:     Feature dictionary (needs asym_id, entity_id etc. for relpos)
            t:         [*] noise level
            si_input:  [*, N_token, c_s_input] input embedding
            si_trunk:  [*, N_token, c_s] trunk single rep
            zij_trunk: [*, N_token, N_token, c_z] trunk pair rep
            use_conditioning: Whether to condition with trunk reps

        Returns:
            si:  [*, N_token, c_s] conditioned single rep
            zij: [*, N_token, N_token, c_z] conditioned pair rep
        """
        f = self._fast
        if f is not None and use_conditioning:
            (ks, dev, wbp, wfp, fp, shp_i, shp_t, shp_z, shp_m,
             has_relpos, has_mask, g_wz, g_fw, pers, sig, rsig) = f
            bf = torch.bfloat16
            # Every property the compiled program assumes about the five
            # positional tensors, checked as *one* tuple compare rather than a
            # chain of 14 short-circuit tests: same checks, 1.55 -> 0.89 us, and
            # the host is what this operator is bound by. The two weight
            # sentinels stay out of the tuple -- comparing Parameters with ``==``
            # would launch an elementwise kernel.
            if ((si_input.shape, si_trunk.shape, zij_trunk.shape,
                 si_input.dtype, si_trunk.dtype, zij_trunk.dtype, t.dtype,
                 si_input.is_contiguous(), si_trunk.is_contiguous(),
                 zij_trunk.is_contiguous(), t.dim(), si_trunk.get_device(),
                 _cur_device(), torch.is_grad_enabled()) == sig
                    and self.linear_z._parameters["weight"] is g_wz
                    and self.fourier_emb._buffers["w"] is g_fw):
                # ``ip`` doubles as the "this call still matches the compiled
                # program" flag: None means a batch feature the kernels were
                # specialized on changed (relpos features or the mask appeared /
                # vanished / changed layout), which needs a re-plan, not a
                # launch.
                ip = _NULL
                mp = 0
                if rsig is not None:
                    # Both relpos and the mask are present (the captured case):
                    # one tuple compare covers all six batch tensors. A missing
                    # key or a non-tensor raises, which is the re-plan signal.
                    try:
                        v0 = batch["residue_index"]
                        v1 = batch["token_index"]
                        v2 = batch["asym_id"]
                        v3 = batch["entity_id"]
                        v4 = batch["sym_id"]
                        tm = batch["token_mask"]
                        if ((v0.dtype, v1.dtype, v2.dtype, v3.dtype, v4.dtype,
                             tm.dtype, v0.shape, v1.shape, v2.shape, v3.shape,
                             v4.shape, tm.shape, v0.is_contiguous(),
                             v1.is_contiguous(), v2.is_contiguous(),
                             v3.is_contiguous(), v4.is_contiguous(),
                             tm.is_contiguous()) == rsig):
                            ip = (v0.data_ptr(), v1.data_ptr(), v2.data_ptr(),
                                  v3.data_ptr(), v4.data_ptr())
                            mp = tm.data_ptr()
                        else:
                            ip = None
                    except (KeyError, AttributeError):
                        ip = None
                else:
                    # Anything else (no relpos, no mask, or one of them absent):
                    # not the captured shape, so keep the general form.
                    if has_relpos:
                        ip = self._batch_idx(batch, shp_m)
                        ip = ((ip[0].data_ptr(), ip[1].data_ptr(),
                               ip[2].data_ptr(), ip[3].data_ptr(),
                               ip[4].data_ptr())
                              if type(ip) is list else None)
                    elif "asym_id" in batch:
                        ip = None   # relpos reappeared: wrong program
                    if has_mask:
                        tm = batch.get("token_mask")
                        if (type(tm) is not torch.Tensor or tm.dtype is not bf
                                or tm.shape != shp_m or not tm.is_contiguous()):
                            ip = None
                        else:
                            mp = tm.data_ptr()
                    elif batch.get("token_mask") is not None:
                        ip = None   # a mask appeared: likewise
                zp, sp, ap, tp = (zij_trunk.data_ptr(), si_trunk.data_ptr(),
                                  si_input.data_ptr(), t.data_ptr())
                if ip is not None and not ((zp | sp | ap | tp) & 15):
                    # Two ``empty_like`` calls are 2.3 us of the ~13 us this
                    # forward spends on the host, and they are the cheapest form
                    # available (``dev/allocvar.py``): one allocation plus two
                    # ``as_strided`` views is 3.4 us, and slice / narrow / split
                    # views are 6.5-6.9 us. Measure this on an *idle* node -- the
                    # caching allocator's ordering inverts under load.
                    zij = torch.empty_like(zij_trunk)
                    si = torch.empty_like(si_trunk)
                    st = _raw_stream(dev)
                    zop, sop = zij.data_ptr(), si.data_ptr()
                    if pers is not None:
                        # One launch: the four phases are separated by grid-wide
                        # barriers instead of launch boundaries. Nothing about
                        # the barriers is host state (see ``_gbar``), so the call
                        # is just "patch the pointers that changed, launch".
                        pers[3] = st
                        pers[13:25] = (zp, zop, sop, *ip, mp, sp, ap, tp)
                        ks[0][0](*pers)
                        return si, zij
                    (l0, p0, c0, g0), (l1, p1, c1, g1), (l2, p2, c2, g2), \
                        (l3, p3, c3, g3) = ks
                    l0(g0[0], 1, 1, st, *p0, zp, zop, *ip, mp, sp, ap, tp, fp,
                       wbp, wfp, *c0)
                    l1(g1[0], g1[1], 1, st, *p1, fp, mp, wbp, wfp, *c1)
                    l2(g2[0], g2[1], 1, st, *p2, fp, mp, wbp, wfp, *c2)
                    l3(g3[0], g3[1], 1, st, *p3, fp, mp, sop, *c3)
                    return si, zij
        # Either no plan yet, or this call's signature is not the one the plan was
        # compiled for (a different N_token, a mask that appeared or vanished, a
        # reloaded weight): re-plan rather than falling back forever.
        if (use_conditioning and not self._plan_failed
                and not torch.is_grad_enabled()):
            out = self._setup(batch, t, si_input, si_trunk, zij_trunk)
            if out is not None:
                return out
        return self._reference(batch, t, si_input, si_trunk, zij_trunk,
                               use_conditioning, chunk_size)
