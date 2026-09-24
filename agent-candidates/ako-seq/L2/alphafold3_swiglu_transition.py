"""SwiGLU transition composites for AlphaFold3 (L2).

SwiGLUTransition: LayerNorm -> SwiGLU -> Linear (AF3 Algorithm 11)
ConditionedTransitionBlock: AdaLN -> SwiGLU -> gated output (AF3 Algorithm 25)

Reference: openfold3/core/model/layers/transition.py SwiGLUTransition
           openfold3/core/model/layers/transition.py ConditionedTransitionBlock

WHAT THIS FILE IS FOR
=====================
Both composites are towers of ``nn.Module`` calls, and at the captured shapes
each call costs far more in dispatch and launch than in arithmetic.  Measured on
this box (B200, 148 SMs) with the benchmark's own timing loop, on the seed:

===================  =========  ============  =========
case                 harness    host enqueue  true GPU
===================  =========  ============  =========
CTB   c_a=768        217 us      219 us        43.6 us
CTB   c_a=128 (4D)   221 us      225 us        45.2 us
CTB   c_a=128 (3D)   211 us      221 us        45.1 us
SGT   c_in=128       59 us        80 us        18.5 us
SGT   c_in=384       59 us        80 us        18.5 us
SGT   c_in=64        60 us        90 us        16.4 us
===================  =========  ============  =========

("true GPU" = the same call captured in a CUDA graph and replayed, so the host
is out of the way.)  The reported number tracks *host enqueue*, not device
time.  ``bench.py::_time_module`` does zero a 2x-L2 buffer (~252 MB, ~50 us of
GPU) before recording the start event, but that only buys the CPU a ~50 us head
start per iteration and CTB needs ~220 us of Python to enqueue its ~16 kernels,
so the GPU sits starved for most of the measured window.  Host cost on this
benchmark is *not* hidden.  And the device-side 43 us is itself ~16 x 2.7 us of
launch-to-launch gap rather than work -- the arithmetic is nanoseconds.

So both terms are per-kernel overhead, and the lever that moves both is
**kernel count**.  This file collapses the towers to three ``@triton.jit``
kernels -- two launches for SwiGLUTransition and three for
ConditionedTransitionBlock, down from ~7 and ~16 -- with every normalization,
activation, gate, bias and mask folded into a GEMM prologue or epilogue, and the
sibling GEMMs that share an input issued as two ``tl.dot`` calls against one
register-resident A operand so no c_out-sized intermediate ever reaches HBM.
Three further things each mattered as much as the fusion:

* **The launch path.** ``kernel[grid](...)`` re-binds and re-specializes every
  argument: ~12 us of Python per launch, more than three fused kernels' worth of
  budget.  Every kernel argument except the pointers is ``tl.constexpr`` and the
  plan is keyed on the exact shape, so after the first call the launch goes
  through the compiled kernel's own C entry point at ~4 us.
* **Everything else in ``forward``.** ``torch.broadcast_shapes`` alone is 6.6 us
  a call and the layout validation wanted two of them; the parameter attribute
  chain another 3.5 us.  All of it is resolved once per shape into a flat "hot"
  tuple.  This took CTB c_a=768 from 45.7 us of host to 15.5 us.
* **PDL.** ``gdc_launch_dependents`` / ``gdc_wait`` across the two or three
  dependent kernels, which is the one lever that attacks the *GPU-side* gap: 10%
  on CTB c_a=768, 15% on CTB c_a=128, 40% on SGT c_in=64.  See ``_pdl_flags``
  for why every kernel waits, including the first.

The remaining floor is the harness itself: ``_ShiftingPool`` copies every input
tensor inside the timed region, which costs 15.3 us for CTB's three tensors,
11.2 us for SGT's two and 7.2 us for SGT's one (measured with a module that
launches nothing).  It is additive on candidate and reference alike and cannot
be optimized away.  Final per-case candidate microseconds against that floor:

===================  =====  =========  =====  ====  ====  ===========
case                 seed   this file  floor  host  GPU   bound
===================  =====  =========  =====  ====  ====  ===========
CTB   c_a=768        217     29.7      15.3   18.4  20.4  GPU
CTB   c_a=128 (4D)   221     23.6      15.3   17.7  16.4  host ~= GPU
CTB   c_a=128 (3D)   211     23.6      15.3   17.8  16.4  host ~= GPU
SGT   c_in=128       59      15.3      11.2   13.2  12.3  host ~= GPU
SGT   c_in=384       59      19.5      11.2   13.0  16.4  GPU
SGT   c_in=64        60      11.2       7.2   12.5  10.2  host
===================  =====  =========  =====  ====  ====  ===========

``host`` is wall-clock enqueue with no sync; ``GPU`` is a CUDA-graph replay with
the L2 flushed first.  The harness reports roughly ``max(host, GPU + pool)``, and
after round 1 the two terms are within a couple of microseconds of each other on
every case.  **That is the fact that governs what is still worth doing here**, and
it is easy to get wrong: round 1's notes say launch count is the dominant
controllable term, which was true of round 1's *starting point* and is no longer
true of its result.  Removing a further launch now only pays if it does not cost
CTAs -- and the obvious way to remove one (make the SwiGLU hidden dimension the
grid dimension of a single fused kernel, so ``hidden`` never reaches HBM) pays for
its launch *in* CTAs, because it turns a parallel dimension into a reduction
dimension.  That was built, swept and measured in round 2: it took SGT c_in=128's
host from 14.5 to 8.2 us exactly as intended and still lost 4 us, because GPU time
went 10.2 -> 14.3 as the grid collapsed from 384 CTAs to 16.  It is slower on all
five shapes; the implementation and the full accounting are in
``dev/merged_experiment_kernel.py`` and ``ITERATIONS.md``.  Do not re-derive it.

WEIGHT / CTA ACCOUNTING (per case, for the next round)
=====================================================
bf16 weight bytes streamed once per call, and the CTA count each kernel gets at
the tile shapes in ``_TILE``:

Note the ``n`` on the SwiGLUTransition rows.  Every captured SwiGLUTransition is
built with **n=4**, not n=2 -- the capture lists two ``init_variant_ids`` per
forward variant and the harness takes the first -- so the hidden widths are
512 / 1536 / 256.  Round 1's tile entries were keyed on the n=2 widths and
therefore never matched the benchmark at all; see ``_TILE``.

  CTB c_a=768 c_s=384 n=2 (M=16, 36 KB of activations):
      adaln 1.18 MB / 24 CTA, swiglu 4.72 MB / 96 CTA,
      out 2.36+0.59 MB / 48 CTA.  Total 8.85 MB.
  CTB c_a=128 c_s=128 n=2 (M=368):
      adaln 66 KB / 184 CTA, swiglu 131 KB / 368 CTA,
      out 66+33 KB / 184 CTA.  Total 0.29 MB.
  SGT c_in=384 n=4 (M=16):   swiglu 2.36 MB / 48 CTA, out 1.18 MB / 12 CTA.
      Total 3.54 MB.
  SGT c_in=128 n=4 (M=256):  swiglu 262 KB / 256 CTA, out 131 KB / 128 CTA.
      Total 0.39 MB.
  SGT c_in=64 n=4 (M=128):   swiglu 66 KB / 64 CTA, out 33 KB / 32 CTA.
      Total 0.10 MB.

So >99% of the traffic is weights -- but at 6.5-8 TB/s even the largest of those
streams is ~1.3 us, against a ~15 us pool floor and ~4 us per launch.  **Every
case here is launch- or latency-bound, not bandwidth-bound.**  Do not trade
launches for weight reuse.  The corollary the next round can use: redundant
weight reads are *cheap* relative to a launch, so merging the two dependent
GEMMs into one kernel by replicating the first GEMM's work across CTAs is
affordable in *bytes* on the small cases (CTB c_a=128 would re-read ~6 MB, ~2 us,
to save a ~4 us launch).  Round 2 built exactly that and it still lost, because
bytes were never the binding constraint -- CTAs were.  See the table above.

TILING
======
``BM = 16`` everywhere: rows are the scarce dimension (M is 16 on the two
highest-count cases), so a taller row tile idles MMA lanes *and* removes CTAs.
The dominant effect in the sweep was **pipeline depth, not tile shape** -- at
BN=64/num_stages=2 the CTB c_a=768 case runs 54 us and at num_stages>=4 it runs
40 us, with BN worth only 1-2 us on top -- which is what a latency bound rather
than a bandwidth bound looks like: what buys time is bytes in flight per CTA and
CTAs, not fewer re-reads.  Configs are constexpr in ``_TILE`` rather than
``@triton.autotune`` so the choice cannot drift between runs.

NUMERICS
========
Every reduction and every accumulator is fp32.  The kernels additionally round
each intermediate that the reference materializes in bf16 (the two SwiGLU
projections, the AdaLN pieces, the output gate) back through bf16 before using
it, so the fused result *tracks the reference's rounding* rather than merely
being more accurate than it: at the harness's own weight scale that makes
SwiGLUTransition bit-exact against the reference (max_abs 0.00e+00) and CTB
9.8e-04, against a tolerance of atol 1e-2 / rtol 1e-2 on 99% of elements --
which is only ~2 bf16 roundings wide once a GEMM, a gate and a second GEMM are
chained.  Weights are read as bf16; nothing is quantized below that.

Both composites are ill-conditioned at weight scales well above the captured
ones (bf16 activations reaching 1e9, and ~96% cancellation in the final GEMM);
there the reference's own bf16 result is just as far from an fp64 evaluation as
this one is, so it is a property of the problem, not something to fix here.

FALLBACK
========
Anything the fast path cannot prove goes to ``_reference``, which is the
baseline composition over the same submodules: non-bf16, non-contiguous, a mask
whose broadcast cannot be shown to be row-aligned, a set chunk_size or
ckpt_chunk_size, grad enabled, a relocated parameter, or an AdaLN whose affine
structure differs from the one ``AdaLN.__init__`` builds.
"""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._C import _cuda_getCurrentRawStream as _raw_stream
from torch._C import _cuda_getDevice as _cur_device

try:  # Triton 3.6+; a no-op on the device unless launch_pdl=True is passed
    from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait
    _HAVE_PDL = True
except ImportError:  # pragma: no cover - older Triton
    _HAVE_PDL = False

# Programmatic Dependent Launch between the (strictly dependent) kernels of one
# composite: the consumer's CTAs are allowed to start while the producer's tail
# drains, which targets the GPU-side launch-to-launch gap. Flipped by
# ``_pdl_meta`` below; see ITERATIONS.md for the measurement.
_PDL = True

from ..L1.sigmoid import Sigmoid
from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear
from .alphafold3_swiglu import AdaLN, SwiGLU

_LN_EPS = 1e-5


# ###########################################################################
# Kernels
# ###########################################################################
@triton.jit
def _ln_swiglu(X, LNW, LNB, WA, WB, HOUT,
               M: tl.constexpr, C: tl.constexpr, HID: tl.constexpr,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
               DO_LN: tl.constexpr, HAS_W: tl.constexpr, HAS_B: tl.constexpr,
               EVEN_M: tl.constexpr, EVEN_N: tl.constexpr, EPS: tl.constexpr,
               PDL_IN: tl.constexpr = False, PDL_OUT: tl.constexpr = False):
    """``HOUT[m, n] = silu(xn[m] . WA[n]) * (xn[m] . WB[n])``, with
    ``xn = layer_norm(X[m])`` folded into the prologue when ``DO_LN``.

    This is the whole of ``LayerNorm -> SwiGLU`` (5 kernels in the reference:
    the fp32 cast sandwich or a fused LN, two GEMMs, a SiLU and a multiply) in
    one launch, and for ConditionedTransitionBlock the same kernel with
    ``DO_LN=False`` is the whole of ``SwiGLU`` (4 kernels).

    ``linear_a`` and ``linear_b`` consume the *same* normalized row, so they
    are two ``tl.dot`` calls against one A operand inside one K loop rather
    than a concatenated ``[C, 2*HID]`` weight buffer: identical traffic and
    identical launch count, but no lazily-built copy of the weights to keep
    coherent with ``load_state_dict``, and the two dots share the A tile in
    registers.  The ``HID``-sized products never leave the SM -- only
    ``silu(a) * b`` is stored.

    The row is streamed twice, not held in registers.  ``C`` reaches 768 here
    and a resident ``[BM, 768]`` fp32 tile would force the weight tiles out of
    the MMA pipeline; X is at most 24 KB per call, so re-reading it costs
    nothing next to the 1.2-4.7 MB weight stream it lets Triton pipeline. Pass
    one takes both moments in fp32 by the shifted one-pass formula (subtract
    the row's own first element, then accumulate ``sum(d)`` and ``sum(d*d)``
    together so the two reduction trees pipeline); shifting by a real data
    point keeps ``sq/C - mean^2`` on the scale of the row's spread, so the
    subtraction cancels only the shifted mean.  Pass two re-reads the row,
    normalizes, applies the affine and feeds the MMA directly.
    """
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    mm = rm < M
    mn = rn < HID
    rk = tl.arange(0, BK)
    xb = X + rm[:, None].to(tl.int64) * C
    if PDL_IN:
        gdc_wait()

    # ---- prologue: row moments, fp32, shifted one-pass ----------------
    if DO_LN:
        if EVEN_M:
            shift = tl.load(X + rm.to(tl.int64) * C).to(tl.float32)
        else:
            shift = tl.load(X + rm.to(tl.int64) * C, mask=mm,
                            other=0.0).to(tl.float32)
        msum = tl.zeros([BM], dtype=tl.float32)
        msq = tl.zeros([BM], dtype=tl.float32)
        for kk in range(0, C, BK):
            if EVEN_M:
                d = tl.load(xb + (kk + rk)[None, :]).to(tl.float32)
            else:
                d = tl.load(xb + (kk + rk)[None, :], mask=mm[:, None],
                            other=0.0).to(tl.float32)
            d = d - shift[:, None]
            msum += tl.sum(d, axis=1)
            msq += tl.sum(d * d, axis=1)
        mean = msum * (1.0 / C)
        rstd = 1.0 / tl.sqrt(
            tl.maximum(msq * (1.0 / C) - mean * mean, 0.0) + EPS)

    # ---- the two projections, one K loop, one shared A operand --------
    acc_a = tl.zeros([BM, BN], dtype=tl.float32)
    acc_b = tl.zeros([BM, BN], dtype=tl.float32)
    wa = WA + rn[None, :].to(tl.int64) * C + rk[:, None]
    wb = WB + rn[None, :].to(tl.int64) * C + rk[:, None]
    for kk in range(0, C, BK):
        kc = kk + rk
        if EVEN_M:
            x = tl.load(xb + kc[None, :]).to(tl.float32)
        else:
            x = tl.load(xb + kc[None, :], mask=mm[:, None],
                        other=0.0).to(tl.float32)
        if DO_LN:
            x = (x - shift[:, None] - mean[:, None]) * rstd[:, None]
            if HAS_W:
                x = x * tl.load(LNW + kc).to(tl.float32)[None, :]
            if HAS_B:
                x = x + tl.load(LNB + kc).to(tl.float32)[None, :]
        a = x.to(tl.bfloat16)
        if EVEN_N:
            ta = tl.load(wa + kk)
            tb = tl.load(wb + kk)
        else:
            ta = tl.load(wa + kk, mask=mn[None, :], other=0.0)
            tb = tl.load(wb + kk, mask=mn[None, :], other=0.0)
        acc_a = tl.dot(a, ta, acc_a)
        acc_b = tl.dot(a, tb, acc_b)

    # ---- epilogue: SiLU gate.  Round through bf16 first: the reference
    # materializes both projections in bf16 before F.silu and the multiply,
    # and this GEMM's output feeds a second GEMM, so tracking its rounding
    # is worth more than keeping the extra fp32 bits. --------------------
    ga = acc_a.to(tl.bfloat16).to(tl.float32)
    gb = acc_b.to(tl.bfloat16).to(tl.float32)
    h = (ga * tl.sigmoid(ga)).to(tl.bfloat16).to(tl.float32) * gb
    hp = HOUT + rm[:, None].to(tl.int64) * HID + rn[None, :]
    if EVEN_M and EVEN_N:
        tl.store(hp, h.to(tl.bfloat16))
    else:
        tl.store(hp, h.to(tl.bfloat16), mask=mm[:, None] & mn[None, :])
    if PDL_OUT:
        gdc_launch_dependents()


@triton.jit
def _adaln(A, S, LNSW, WG, BG, WS, ACOND,
           M: tl.constexpr, CA: tl.constexpr, CS: tl.constexpr,
           BM: tl.constexpr, BN: tl.constexpr,
           BKS: tl.constexpr, BKA: tl.constexpr,
           EVEN_M: tl.constexpr, EVEN_N: tl.constexpr, EPS: tl.constexpr,
           PDL_IN: tl.constexpr = False, PDL_OUT: tl.constexpr = False):
    """AdaLN in one launch: ``sigmoid(Wg . sn + bg) * (an + Ws . sn)`` with
    ``sn = layer_norm_s(S[m])`` and ``an = layer_norm_a(A[m])``.

    The reference spends 7 kernels here (two LayerNorms -- each a three-kernel
    fp32 cast sandwich in plain PyTorch -- two GEMMs, a sigmoid, a multiply and
    an add).  ``linear_g`` and ``linear_s`` both consume ``sn``, so as in
    ``_ln_swiglu`` they are two dots over one A operand in one K loop.

    ``layer_norm_a`` reduces over the full ``CA`` row while this program owns
    only ``BN`` of its columns, so A is read twice: once streamed for the
    moments, once for the ``[BM, BN]`` slice the epilogue needs.  A is 24 KB at
    the largest captured shape against 1.18 MB of weights, so the redundant
    read is free; the alternative (a separate normalization kernel) is a whole
    launch.  ``layer_norm_a`` has neither scale nor offset and ``layer_norm_s``
    has scale only, matching how ``AdaLN.__init__`` builds them.
    """
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    mm = rm < M
    mn = rn < CA
    row = rm.to(tl.int64)
    sb = S + row[:, None] * CS
    ab = A + row[:, None] * CA
    if PDL_IN:
        gdc_wait()

    # ---- layer_norm_s moments (fp32, shifted one-pass) ----------------
    ks = tl.arange(0, BKS)
    if EVEN_M:
        s_sh = tl.load(S + row * CS).to(tl.float32)
    else:
        s_sh = tl.load(S + row * CS, mask=mm, other=0.0).to(tl.float32)
    ssum = tl.zeros([BM], dtype=tl.float32)
    ssq = tl.zeros([BM], dtype=tl.float32)
    for kk in range(0, CS, BKS):
        if EVEN_M:
            d = tl.load(sb + (kk + ks)[None, :]).to(tl.float32)
        else:
            d = tl.load(sb + (kk + ks)[None, :], mask=mm[:, None],
                        other=0.0).to(tl.float32)
        d = d - s_sh[:, None]
        ssum += tl.sum(d, axis=1)
        ssq += tl.sum(d * d, axis=1)
    s_mean = ssum * (1.0 / CS)
    s_rstd = 1.0 / tl.sqrt(
        tl.maximum(ssq * (1.0 / CS) - s_mean * s_mean, 0.0) + EPS)

    # ---- layer_norm_a moments (fp32, shifted one-pass, no affine) -----
    ka = tl.arange(0, BKA)
    if EVEN_M:
        a_sh = tl.load(A + row * CA).to(tl.float32)
    else:
        a_sh = tl.load(A + row * CA, mask=mm, other=0.0).to(tl.float32)
    asum = tl.zeros([BM], dtype=tl.float32)
    asq = tl.zeros([BM], dtype=tl.float32)
    for kk in range(0, CA, BKA):
        if EVEN_M:
            d = tl.load(ab + (kk + ka)[None, :]).to(tl.float32)
        else:
            d = tl.load(ab + (kk + ka)[None, :], mask=mm[:, None],
                        other=0.0).to(tl.float32)
        d = d - a_sh[:, None]
        asum += tl.sum(d, axis=1)
        asq += tl.sum(d * d, axis=1)
    a_mean = asum * (1.0 / CA)
    a_rstd = 1.0 / tl.sqrt(
        tl.maximum(asq * (1.0 / CA) - a_mean * a_mean, 0.0) + EPS)

    # ---- linear_g(sn) and linear_s(sn), one K loop over CS ------------
    acc_g = tl.zeros([BM, BN], dtype=tl.float32)
    acc_s = tl.zeros([BM, BN], dtype=tl.float32)
    wg = WG + rn[None, :].to(tl.int64) * CS + ks[:, None]
    ws = WS + rn[None, :].to(tl.int64) * CS + ks[:, None]
    for kk in range(0, CS, BKS):
        kc = kk + ks
        if EVEN_M:
            s = tl.load(sb + kc[None, :]).to(tl.float32)
        else:
            s = tl.load(sb + kc[None, :], mask=mm[:, None],
                        other=0.0).to(tl.float32)
        s = (s - s_sh[:, None] - s_mean[:, None]) * s_rstd[:, None]
        s = s * tl.load(LNSW + kc).to(tl.float32)[None, :]
        sn = s.to(tl.bfloat16)
        if EVEN_N:
            acc_g = tl.dot(sn, tl.load(wg + kk), acc_g)
            acc_s = tl.dot(sn, tl.load(ws + kk), acc_s)
        else:
            acc_g = tl.dot(sn, tl.load(wg + kk, mask=mn[None, :], other=0.0),
                           acc_g)
            acc_s = tl.dot(sn, tl.load(ws + kk, mask=mn[None, :], other=0.0),
                           acc_s)
    acc_g += tl.load(BG + rn, mask=mn, other=0.0).to(tl.float32)[None, :]

    # ---- epilogue: the A slice this program owns, normalized ----------
    ap = ab + rn[None, :]
    if EVEN_M and EVEN_N:
        av = tl.load(ap).to(tl.float32)
    else:
        av = tl.load(ap, mask=mm[:, None] & mn[None, :], other=0.0).to(tl.float32)
    an = ((av - a_mean[:, None] - a_sh[:, None]) * a_rstd[:, None]
          ).to(tl.bfloat16).to(tl.float32)
    g = tl.sigmoid(acc_g.to(tl.bfloat16).to(tl.float32))
    out = g.to(tl.bfloat16).to(tl.float32) * (
        an + acc_s.to(tl.bfloat16).to(tl.float32))
    op = ACOND + row[:, None] * CA + rn[None, :]
    if EVEN_M and EVEN_N:
        tl.store(op, out.to(tl.bfloat16))
    else:
        tl.store(op, out.to(tl.bfloat16), mask=mm[:, None] & mn[None, :])
    if PDL_OUT:
        gdc_launch_dependents()


@triton.jit
def _out(H, WO, Y, MASK, S, WGO, BGO,
         M: tl.constexpr, HID: tl.constexpr, CO: tl.constexpr,
         CS: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
         BK: tl.constexpr, BKS: tl.constexpr,
         HAS_MASK: tl.constexpr, HAS_GATE: tl.constexpr,
         EVEN_M: tl.constexpr, EVEN_N: tl.constexpr,
         PDL_IN: tl.constexpr = False, PDL_OUT: tl.constexpr = False):
    """``Y[m, n] = linear_out(H)[m, n] * sigmoid(WGO . S[m] + bgo)[n] * MASK[m]``
    -- ``linear_out`` with the output gate and the mask folded into its
    epilogue.

    The gate is computed here rather than in ``_adaln`` (where its K dimension,
    ``CS``, is already resident) precisely because it is an *epilogue*: doing
    it here needs a second dot over a tiny K, doing it there needs an extra
    ``[M, CO]`` buffer plus its store, its reload, and a ``torch.empty`` on the
    host -- and the host is what this file is short of.  The gate reads *raw*
    ``S``, not ``layer_norm_s(S)``; only AdaLN's own gate uses the normalized
    conditioning.

    ``MASK = None`` is a constexpr specialization, not a ones tensor: the
    reference's ``x.new_ones(x.shape[:-1])`` is a whole extra kernel plus an
    allocation to multiply by 1.0.
    """
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    mm = rm < M
    mn = rn < CO
    row = rm.to(tl.int64)
    rk = tl.arange(0, BK)
    hb = H + row[:, None] * HID
    wo = WO + rn[None, :].to(tl.int64) * HID + rk[:, None]
    if PDL_IN:
        gdc_wait()
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for kk in range(0, HID, BK):
        if EVEN_M:
            h = tl.load(hb + (kk + rk)[None, :])
        else:
            h = tl.load(hb + (kk + rk)[None, :], mask=mm[:, None], other=0.0)
        if EVEN_N:
            w = tl.load(wo + kk)
        else:
            w = tl.load(wo + kk, mask=mn[None, :], other=0.0)
        acc = tl.dot(h, w, acc)
    res = acc.to(tl.bfloat16).to(tl.float32)

    if HAS_GATE:
        ks = tl.arange(0, BKS)
        sb = S + row[:, None] * CS
        wg = WGO + rn[None, :].to(tl.int64) * CS + ks[:, None]
        accg = tl.zeros([BM, BN], dtype=tl.float32)
        for kk in range(0, CS, BKS):
            if EVEN_M:
                s = tl.load(sb + (kk + ks)[None, :])
            else:
                s = tl.load(sb + (kk + ks)[None, :], mask=mm[:, None],
                            other=0.0)
            if EVEN_N:
                accg = tl.dot(s, tl.load(wg + kk), accg)
            else:
                accg = tl.dot(s, tl.load(wg + kk, mask=mn[None, :], other=0.0),
                              accg)
        accg += tl.load(BGO + rn, mask=mn, other=0.0).to(tl.float32)[None, :]
        g = tl.sigmoid(accg.to(tl.bfloat16).to(tl.float32))
        res = g.to(tl.bfloat16).to(tl.float32) * res

    if HAS_MASK:
        res = res * tl.load(MASK + rm, mask=mm, other=0.0).to(tl.float32)[:, None]

    yp = Y + row[:, None] * CO + rn[None, :]
    if EVEN_M and EVEN_N:
        tl.store(yp, res.to(tl.bfloat16))
    else:
        tl.store(yp, res.to(tl.bfloat16), mask=mm[:, None] & mn[None, :])
    if PDL_OUT:
        gdc_launch_dependents()


# Tile configs -- constexpr per shape, never autotuned
# ###########################################################################
# (kernel, M, K, N) -> (BN, num_warps, num_stages).  The kernel tag is part of
# the key because the same GEMM shape shows up in two different kernels -- case
# CTB c_a=768 runs `_adaln` at (16, 384, 768) and case SGT c_in=384 runs
# `_ln_swiglu` at the same (16, 384, 768) -- and they do not want the same
# config.
#
# BM is 16 on every path: M is 16 on the two highest-count captured cases, so a
# taller row tile idles MMA lanes *and* removes CTAs from a grid already well
# under the 148 SMs.  BN is set to keep the CTA count up rather than to minimize
# weight re-reads -- see the module docstring's byte accounting.
#
# The dominant effect in the sweep was **pipeline depth, not tile shape**: at
# BN=64/stages=2 the CTB c_a=768 case runs 54 us and at stages>=4 it runs 40 us,
# with BN mattering only 1-2 us on top.  These shapes are latency-bound, not
# bandwidth-bound -- 8.85 MB would be 1.3 us at HBM speed -- so what buys time
# is bytes in flight per CTA (num_stages) and CTAs (small BN), which is why the
# M=16 cases want stages 4-6 and the M=368 cases, which already have 23 row
# blocks of CTAs, are flat past stages=3.
#
# A table rather than `@triton.autotune` so the chosen config cannot drift
# between runs, with a deterministic rule for shapes not in it (a novel shape
# then still takes the fast path, just untuned, instead of falling back to the
# 7-or-16-kernel composite).
_TILE: dict[tuple[str, int, int, int], tuple[int, int, int, int]] = {
    # --- ConditionedTransitionBlock c_a=768 c_s=384 n=2, M=16 (1920 calls) ---
    # `ada` at num_warps=8 is worth 4.1 us on this case (33.8 -> 29.7), the
    # same lever that was worth 4.1 us on SGT c_in=384: at M=16 the two
    # conditioning projections are the whole kernel and they want every warp
    # available. Round 1 swept this stage and chose 4; the ~2 us harness
    # quantization plus the +-35% reference noise is enough to lose a 4 us
    # difference if the sweep is read off the score instead of off candidate
    # microseconds.
    ("ada", 16, 384, 768): (32, 8, 3, 128),
    ("swi", 16, 768, 1536): (16, 2, 4, 256),
    ("out", 16, 1536, 768): (16, 4, 4, 256),
    # --- ConditionedTransitionBlock c_a=128 c_s=128 n=2, M=368 (480 + 24) ---
    ("ada", 368, 128, 128): (16, 2, 4, 128),
    ("swi", 368, 128, 256): (16, 2, 4, 128),
    ("out", 368, 256, 128): (16, 2, 4, 128),
    # --- SwiGLUTransition c_in=128 n=4, M=256 (992 calls) ---
    # NOTE the `n`. Every captured SwiGLUTransition instance is built with
    # **n=4**, not n=2: the capture's forward variant for x[1,16,16,128] lists
    # `init_variant_ids [2, 4]` and `_collect_cases` takes the *first*, which is
    # `{c_in: 128, n: 4}`. So the hidden width is 512 / 1536 / 256, not
    # 256 / 768 / 128, and round 1's n=2 entries below never matched the
    # benchmark at all -- all three SGT cases ran on `_tile`'s untuned fallback.
    # Re-sweeping at the real widths is worth 4.1 us on c_in=384 and 2.0 us on
    # c_in=128, entirely from `num_warps=8` on the projection stage (the fallback
    # guessed 4): at M=16, HID=1536 the projections want every warp they can get.
    ("lns", 256, 128, 512): (32, 8, 2, 128),
    ("out", 256, 512, 128): (16, 2, 3, 256),
    # --- SwiGLUTransition c_in=384 n=4, M=16 (928 calls) ---
    ("lns", 16, 384, 1536): (32, 8, 3, 128),
    ("out", 16, 1536, 384): (32, 2, 4, 256),
    # --- SwiGLUTransition c_in=64 n=4, M=128, mask=None (48 calls) ---
    ("lns", 128, 64, 256): (32, 8, 3, 64),
    ("out", 128, 256, 64): (16, 8, 4, 256),
    # --- n=2 variants: not reached by this benchmark, kept for L3 callers ---
    ("lns", 256, 128, 256): (16, 2, 3, 128),
    ("out", 256, 256, 128): (16, 2, 3, 128),
    ("lns", 16, 384, 768): (32, 4, 4, 64),
    ("out", 16, 768, 384): (16, 2, 4, 256),
    ("lns", 128, 64, 128): (16, 4, 4, 64),
    ("out", 128, 128, 64): (16, 4, 4, 64),
}


def _tile(tag: str, m: int, k: int, n: int) -> tuple[int, int, int, int]:
    """(BN, num_warps, num_stages, BK).  A table entry may omit BK, in which
    case the largest exact power-of-two K tile is used."""
    hit = _TILE.get((tag, m, k, n))
    if hit is None:
        # BN must be a power of two (`tl.arange`) and at least 16 (`tl.dot`).
        # An N below 16, or not a multiple of it, is covered by the EVEN_N mask.
        hit = (32 if n % 32 == 0 else 16, 4, 4)
    if len(hit) == 4:
        return hit
    return hit + (_bk(k),)


def _pdl_meta(warps: int, stages: int) -> dict:
    """Launch metadata. ``launch_pdl`` is resolved at compile time and lands on
    the ``CudaLauncher``, so the memoized direct-launch path inherits it from
    the ``pre`` tuple with no per-call cost."""
    if _PDL and _HAVE_PDL:
        return dict(num_warps=warps, num_stages=stages, launch_pdl=True)
    return dict(num_warps=warps, num_stages=stages)


def _pdl_flags(last: bool) -> tuple[bool, bool]:
    """(PDL_IN, PDL_OUT) for a kernel at that position in the chain.

    ``PDL_IN`` is set on *every* kernel, including the first. ``launch_pdl``
    says "you may begin before your predecessor has finished", and the only
    thing that makes that safe is the ``gdc_wait()`` the kernel itself executes
    before touching memory it did not write. That applies to the first kernel
    too: its predecessor is whatever the caller ran last, and the frozen L1
    Sigmoid kernel *does* trigger programmatic completion early, so in an L3
    stack it could be a producer of this op's input. Measured on the SGT
    c_in=64 case, dropping ``launch_pdl`` from the first kernel to avoid that
    hazard costs 4 us; waiting instead keeps the overlap and costs nothing.

    ``PDL_OUT`` is *not* set on the last kernel. Triggering early there would
    only help a successor that opted into PDL, and would silently create the
    same hazard for any successor that opted in without waiting. A producer
    that never triggers is safe for every consumer.
    """
    if not (_PDL and _HAVE_PDL):
        return (False, False)
    return (True, not last)


def _bk(k: int, cap: int = 128) -> int:
    """Largest power-of-two K tile that divides *k* exactly, or 0 if there is
    none at least 16 wide.

    The GEMM loops are deliberately unmasked along K -- every captured width
    (64/128/256/384/768/1536) admits 64 or 128, and a K mask would cost a
    predicate on the hottest loop for nothing.  So a width that admits no such
    tile (8, 12, 24, 100, 120, ...) must not reach the fast path at all: 0
    makes `_prepare` return None and the call goes to the reference composite.
    A tile below 16 is no use either, since `tl.dot` needs K >= 16.
    """
    bk = min(cap, 1 << (k.bit_length() - 1)) if k > 0 else 0
    while bk >= 16 and k % bk:
        bk //= 2
    return bk if bk >= 16 else 0


# ###########################################################################
# ###########################################################################
# Host side
# ###########################################################################
# The benchmark is host-bound (module docstring), so `forward` is written to a
# host budget, not just a launch budget. Measured costs of the pieces at CTB
# c_a=768 on this box:
#
#   torch.broadcast_shapes(...)                   6.6 us   <- twice, in the
#                                                              layout validation
#   attribute chain for the 9 parameters          3.5 us
#   one memoized raw Triton launch                4-5 us
#   Triton's own kernel[grid](...) binder        ~12 us
#   torch.empty_like                              1.2 us
#   is_contiguous / dtype / data_ptr / device    ~0.06 us each
#
# Two consequences drive the structure below. First, *everything* that only
# depends on the shape signature -- the broadcast legality of `mask` and of `s`
# against `a`, the tile config, the grid, every constexpr, and the parameter
# tensors themselves -- is resolved once into a flat "hot" tuple and never
# recomputed. Second, the launch goes through the compiled kernel's own C entry
# point rather than `kernel[grid](...)`.
#
# What is deliberately NOT cached: the output tensor (a fresh `empty_like` every
# call, so nothing a caller can see is ever aliased between calls) and the
# parameter *addresses* (re-read from the cached tensors each call, so an
# ordinary in-place weight update -- including `load_state_dict`'s default
# `param.copy_()`, and `.to()`'s `param.data = ...` -- is picked up with no
# invalidation at all).


class _Launch:
    """One Triton kernel at one fully-constexpr signature, compiled once and
    then launched through the compiled kernel's own C entry point.

    `kernel[grid](...)` re-binds and re-specializes every argument on every
    call. Every argument here except the pointers is `tl.constexpr` and the
    plan is keyed on the exact shape, so there is nothing left for the binder
    to derive, and after the first (compiling) call the launch is one call into
    the generated launcher: ~4 us against ~12 us.

    Triton still specializes on 16-byte pointer alignment, so `warm` refuses to
    memoize off a misaligned first call and the caller re-checks alignment.
    Everything reached for inside `CudaLauncher` is fetched defensively: if a
    future Triton reshapes it, or the kernel wants scratch (these never do), we
    never memoize and keep using the supported path -- slower, still right.
    """

    __slots__ = ("kern", "grid", "cargs", "meta", "gx", "gy", "run", "pre")

    def __init__(self, kern, grid, cargs, meta):
        self.kern = kern
        self.grid = grid
        self.gx, self.gy = grid
        self.cargs = cargs
        self.meta = meta
        self.run = None
        self.pre = ()

    def warm(self, ptrs):
        compiled = self.kern[self.grid](*ptrs, *self.cargs, **self.meta)
        launcher = None if compiled is None else compiled.run
        raw = getattr(launcher, "launch", None)
        if (raw is not None
                and getattr(launcher, "global_scratch_size", None) == 0
                and getattr(launcher, "profile_scratch_size", None) == 0
                and all(t is None or not (t.data_ptr() & 15) for t in ptrs)):
            self.run = raw
            self.pre = (compiled.function,
                        launcher.launch_cooperative_grid, launcher.launch_pdl,
                        None, None,               # global / profile scratch
                        compiled.packed_metadata,
                        None, None, None)         # launch metadata, 2 hooks

    def slow(self, ptrs):
        self.kern[self.grid](*ptrs, *self.cargs, **self.meta)


def _rows(t: torch.Tensor, width: int) -> int:
    """Rows of a `[..., width]` tensor, or -1 if it is not a contiguous stack
    of them. No reshape: the kernels index `row * width + col` off the base
    pointer, so `[1, 1, M, C]` and the 3D/4D/5D leading-singleton layouts all
    reach the same flat path with no copy and no view."""
    if t.ndim < 1 or t.shape[-1] != width or not t.is_contiguous():
        return -1
    return t.numel() // width


def _row_aligned(mask: torch.Tensor, ref: torch.Tensor, rows: int) -> bool:
    """Does `mask.unsqueeze(-1)` broadcast against *ref* to exactly
    `ref.shape`, with its elements in *ref*'s row order?

    `numel == rows` plus an exact broadcast is sufficient: a mask dim of 1
    where *ref* is k>1 would make `numel` too small by a factor of k, and a
    mask dim of k where *ref* is 1 would grow the broadcast past `ref.shape`.
    The masks that survive both tests are exactly those whose flat order is
    *ref*'s row order. Called once per shape, never in the hot path --
    `torch.broadcast_shapes` alone is 6.6 us.
    """
    if not mask.is_contiguous() or mask.numel() != rows:
        return False
    try:
        return torch.broadcast_shapes(mask.shape + (1,), ref.shape) == ref.shape
    except RuntimeError:
        return False


class _Cached(nn.Module):
    """Shared cache bookkeeping for the two composites.

    `_hot` holds one fully-resolved shape signature; `_plans` keeps the
    compiled `_Launch` objects so a second shape does not recompile. Both are
    dropped whenever a parameter *object* could have been replaced under us:
    `_apply` covers `.to()` / `.cuda()` / `.float()`, and the load-state-dict
    post hook covers `load_state_dict(..., assign=True)`. The default
    `load_state_dict` path copies in place and needs no invalidation, since the
    hot tuple holds the parameter tensors and re-reads their addresses.
    """

    def __init__(self):
        super().__init__()
        self._hot = None
        self._plans: dict = {}
        self._scratch: dict = {}
        self.register_load_state_dict_post_hook(
            lambda mod, incompatible_keys: mod._invalidate())

    _warned = False

    def _invalidate(self):
        self._hot = None
        self._plans.clear()
        self._scratch.clear()

    def _fell_over(self, exc: BaseException) -> None:
        """A compile or launch problem must never be worse than being slow, so
        we drop the plan and let the reference composition answer.

        But warn, once per class: a fast path that silently stops engaging looks
        exactly like a correct operator and costs ~6x here, which is a much
        worse failure to debug than an exception would have been.
        """
        self._invalidate()
        cls = type(self)
        if not cls.__dict__.get("_warned"):
            cls._warned = True
            warnings.warn(
                f"{cls.__name__}: fused path unavailable ({exc!r}); "
                f"falling back to the reference composition",
                RuntimeWarning, stacklevel=3)

    def _apply(self, *args, **kwargs):
        self._invalidate()
        return super()._apply(*args, **kwargs)

    def _buf(self, slot: int, numel: int, device) -> torch.Tensor:
        """A reused intermediate. `torch.empty` is ~1.5 us of host, ~10% of a
        two- or three-launch budget, so the buffers that never leave the
        operator are allocated once per shape. Only the returned output is
        freshly allocated."""
        key = (slot, numel, device)
        buf = self._scratch.get(key)
        if buf is None:
            buf = self._scratch[key] = torch.empty(
                numel, dtype=torch.bfloat16, device=device)
        return buf


# ###########################################################################
# Modules
# ###########################################################################
class SwiGLUTransition(_Cached):
    """AF3 Algorithm 11: SwiGLU-based transition.

    Two launches: `_ln_swiglu` (LayerNorm prologue, both projections, SiLU gate
    epilogue) then `_out` (linear_out with the mask folded into its epilogue).

    Args:
        c_in: Input channel dimension
        n: Factor multiplied to c_in for hidden dimension
    """

    def __init__(self, c_in: int, n: int):
        super().__init__()
        self.c_in = c_in
        self.n = n

        self.layer_norm = LayerNorm(c_in)
        self.swiglu = SwiGLU(c_in, n * c_in)
        self.linear_out = Linear(n * c_in, c_in, bias=False)

        self._hid = n * c_in

    # -- the reference composition; anything the fast path cannot prove --
    def _reference(self, x, mask, chunk_size, ckpt_chunk_size):
        if mask is None:
            mask = x.new_ones(x.shape[:-1])
        mask = mask.unsqueeze(-1)
        x = self.layer_norm(x)
        x = self.swiglu(x)
        x = self.linear_out(x)
        return x * mask

    def _prepare(self, x, mask):
        """Resolve one shape signature: validate the layout, compile both
        kernels, and fold everything shape-dependent into the hot tuple."""
        c, hid = self.c_in, self._hid
        ln = self.layer_norm
        lnw, lnb = ln.weight, ln.bias
        wa, wb = self.swiglu.linear_a.weight, self.swiglu.linear_b.weight
        wo = self.linear_out.weight
        rows = _rows(x, c)
        dev = x.get_device()
        if (rows <= 0 or x.dtype is not torch.bfloat16 or not x.is_cuda
                or not _bk(c) or not _bk(hid)
                or wa.dtype is not torch.bfloat16
                or wb.dtype is not torch.bfloat16
                or wo.dtype is not torch.bfloat16
                or wa.shape != (hid, c) or wb.shape != (hid, c)
                or wo.shape != (c, hid)
                or not (wa.is_contiguous() and wb.is_contiguous()
                        and wo.is_contiguous())
                or wa.get_device() != dev or wb.get_device() != dev
                or wo.get_device() != dev
                or (lnw is not None and (lnw.dtype is not torch.bfloat16
                                         or not lnw.is_contiguous()
                                         or lnw.shape != (c,)
                                         or lnw.get_device() != dev))
                or (lnb is not None and (lnb.dtype is not torch.bfloat16
                                         or not lnb.is_contiguous()
                                         or lnb.shape != (c,)
                                         or lnb.get_device() != dev))
                or (mask is not None
                    and (mask.dtype is not torch.bfloat16 or not mask.is_cuda
                         or mask.get_device() != dev
                         or not _row_aligned(mask, x, rows)))):
            return None

        key = (rows, mask is not None)
        plan = self._plans.get(key)
        if plan is None:
            bn1, w1, st1, bk1 = _tile("lns", rows, c, hid)
            k1 = _Launch(
                _ln_swiglu, (-(-rows // 16), -(-hid // bn1)),
                (rows, c, hid, 16, bn1, bk1, True,
                 lnw is not None, lnb is not None,
                 rows % 16 == 0, hid % bn1 == 0, ln.eps)
                + _pdl_flags(False),
                _pdl_meta(w1, st1))
            bn2, w2, st2, bk2 = _tile("out", rows, hid, c)
            k2 = _Launch(
                _out, (-(-rows // 16), -(-c // bn2)),
                (rows, hid, c, 0, 16, bn2, bk2, 16,
                 mask is not None, False, rows % 16 == 0, c % bn2 == 0)
                + _pdl_flags(True),
                _pdl_meta(w2, st2))
            plan = self._plans[key] = (k1, k2)
        k1, k2 = plan

        h = self._buf(0, rows * hid, x.device)
        y = torch.empty_like(x)
        p1 = (x, lnw, lnb, wa, wb, h)
        p2 = (h, wo, y, mask, None, None, None)
        if k1.run is None or k2.run is None:
            k1.warm(p1)
            k2.warm(p2)
        else:
            k1.slow(p1)
            k2.slow(p2)
        if k1.run is not None and k2.run is not None:
            self._hot = (
                x.shape, None if mask is None else mask.shape, dev,
                k1.run, k1.gx, k1.gy, k1.pre, k1.cargs,
                k2.run, k2.gx, k2.gy, k2.pre, k2.cargs,
                # The *owning* `_parameters` dicts, not the parameter tensors.
                # A cached tensor cannot detect its own replacement -- reading
                # `d["weight"]` each call does, and a dict lookup is ~0.03 us
                # against the ~1 us that `nn.Module.__getattr__` costs.
                self.layer_norm._parameters, self.swiglu.linear_a._parameters,
                self.swiglu.linear_b._parameters,
                self.linear_out._parameters,
                (0 if lnw is None else lnw.data_ptr(),
                 0 if lnb is None else lnb.data_ptr(),
                 wa.data_ptr(), wb.data_ptr(), wo.data_ptr()),
                h.data_ptr(), h,
            )
        return y

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        chunk_size: int | None = None,
        ckpt_chunk_size: int | None = None,
    ) -> torch.Tensor:
        hot = self._hot
        if hot is not None and chunk_size is None and ckpt_chunk_size is None:
            (shape, mshape, dev,
             run1, gx1, gy1, pre1, post1,
             run2, gx2, gy2, pre2, post2,
             dln, dwa, dwb, dwo, cp, hp, _h) = hot
            if (x.shape == shape and x.dtype is torch.bfloat16
                    and x.is_contiguous()
                    and (mask is None if mshape is None
                         else (mask is not None and mask.shape == mshape
                               and mask.dtype is torch.bfloat16
                               and mask.is_contiguous()
                               and mask.get_device() == dev))
                    and not torch.is_grad_enabled()
                    and x.get_device() == dev and _cur_device() == dev):
                lnw = dln["weight"]
                lnb = dln["bias"]
                lnwp = 0 if lnw is None else lnw.data_ptr()
                lnbp = 0 if lnb is None else lnb.data_ptr()
                wap = dwa["weight"].data_ptr()
                wbp = dwb["weight"].data_ptr()
                wop = dwo["weight"].data_ptr()
                y = torch.empty_like(x)
                xp = x.data_ptr()
                yp = y.data_ptr()
                mp = 0 if mask is None else mask.data_ptr()
                # One test for two things: the 16-byte alignment the compiled
                # kernels were specialized on (only the per-call pointers can
                # break it -- the weights and the scratch buffer were checked
                # when the plan was built), and whether any parameter moved.
                # Reading the live parameter out of its owning `_parameters`
                # dict catches both a replaced Parameter object and a relocated
                # storage (`p.data = fn(p.data)`, which is what `.to()` does);
                # either sends us back through `_prepare` to re-validate dtype,
                # shape and contiguity. An ordinary in-place weight update
                # keeps the address and passes straight through with the new
                # values.
                if not (((xp | yp | mp) & 15)
                        | (lnwp ^ cp[0]) | (lnbp ^ cp[1]) | (wap ^ cp[2])
                        | (wbp ^ cp[3]) | (wop ^ cp[4])):
                    st = _raw_stream(dev)
                    run1(gx1, gy1, 1, st, *pre1,
                         xp, lnwp or None, lnbp or None, wap, wbp, hp, *post1)
                    run2(gx2, gy2, 1, st, *pre2,
                         hp, wop, yp, mp or None, None, None, None, *post2)
                    return y
        if (chunk_size is None and ckpt_chunk_size is None
                and x.is_cuda and not torch.is_grad_enabled()):
            try:
                y = self._prepare(x, mask)
            except Exception as exc:  # noqa: BLE001
                self._fell_over(exc)
                y = None
            if y is not None:
                return y
        return self._reference(x, mask, chunk_size, ckpt_chunk_size)


class ConditionedTransitionBlock(_Cached):
    """AF3 Algorithm 25: SwiGLU transition with AdaLN-Zero conditioning.

    Three launches: `_adaln` (both LayerNorms, both conditioning projections,
    the sigmoid gate, the add and the multiply), `_ln_swiglu` with `DO_LN=False`
    (both SwiGLU projections and the SiLU gate), then `_out` (linear_out with
    the output gate's GEMM, its sigmoid and the mask in the epilogue).

    Submodule names match the reference checkpoint:
    - layer_norm: AdaLN
    - swiglu: SwiGLU
    - linear_g: output gate Linear(c_s, c_a, bias=True)
    - linear_out: Linear(n*c_a, c_a, bias=False)

    Reference: openfold3/core/model/layers/transition.py ConditionedTransitionBlock

    Args:
        c_a: Activation channel dimension
        c_s: Conditioning channel dimension
        n: Factor for hidden dimension
    """

    def __init__(self, c_a: int, c_s: int, n: int):
        super().__init__()
        self.layer_norm = AdaLN(c_a=c_a, c_s=c_s)
        self.swiglu = SwiGLU(c_a, n * c_a)
        self.sigmoid = Sigmoid()
        self.linear_g = Linear(c_s, c_a, bias=True)
        self.linear_out = Linear(n * c_a, c_a, bias=False)

        self._c_a = c_a
        self._c_s = c_s
        self._hid = n * c_a

    # -- the reference composition; anything the fast path cannot prove --
    def _reference(self, a, s, mask, chunk_size):
        if mask is None:
            mask = a.new_ones(a.shape[:-1])
        mask = mask.unsqueeze(-1)
        a = self.layer_norm(a, s)
        b = self.swiglu(a)
        a = self.sigmoid(self.linear_g(s)) * self.linear_out(b)
        return a * mask

    def _prepare(self, a, s, mask):
        ca, cs, hid = self._c_a, self._c_s, self._hid
        ada = self.layer_norm
        lnsw = ada.layer_norm_s.weight
        wg, bg, ws = ada.linear_g.weight, ada.linear_g.bias, ada.linear_s.weight
        wa, wb = self.swiglu.linear_a.weight, self.swiglu.linear_b.weight
        wo = self.linear_out.weight
        wgo, bgo = self.linear_g.weight, self.linear_g.bias
        rows = _rows(a, ca)
        dev = a.get_device()
        pars = (lnsw, wg, bg, ws, wa, wb, wo, wgo, bgo)
        if (rows <= 0 or _rows(s, cs) != rows
                or a.dtype is not torch.bfloat16
                or s.dtype is not torch.bfloat16 or not a.is_cuda
                or not s.is_cuda or s.get_device() != dev
                or not _bk(cs) or not _bk(ca) or not _bk(hid)
                # layer_norm_a is built with neither scale nor offset and
                # layer_norm_s with scale only; the kernel bakes that in.
                or ada.layer_norm_a.weight is not None
                or ada.layer_norm_a.bias is not None
                or ada.layer_norm_s.bias is not None
                # One EPS constexpr serves both AdaLN LayerNorms.
                or ada.layer_norm_s.eps != ada.layer_norm_a.eps
                or any(t is None or t.dtype is not torch.bfloat16
                       or not t.is_contiguous() or t.get_device() != dev
                       for t in pars)
                or wg.shape != (ca, cs) or ws.shape != (ca, cs)
                or wgo.shape != (ca, cs) or bg.shape != (ca,)
                or bgo.shape != (ca,) or lnsw.shape != (cs,)
                or wa.shape != (hid, ca) or wb.shape != (hid, ca)
                or wo.shape != (ca, hid)
                # `sigmoid(linear_g(s)) * linear_out(b)` must broadcast to
                # a.shape, i.e. s's rows line up with a's.
                or torch.broadcast_shapes(s.shape[:-1] + (ca,),
                                          a.shape) != a.shape
                or (mask is not None
                    and (mask.dtype is not torch.bfloat16 or not mask.is_cuda
                         or mask.get_device() != dev
                         or not _row_aligned(mask, a, rows)))):
            return None

        key = (rows, mask is not None)
        plan = self._plans.get(key)
        if plan is None:
            bn1, w1, st1, bk1 = _tile("ada", rows, cs, ca)
            k1 = _Launch(
                _adaln, (-(-rows // 16), -(-ca // bn1)),
                (rows, ca, cs, 16, bn1, bk1, _bk(ca),
                 rows % 16 == 0, ca % bn1 == 0, ada.layer_norm_a.eps)
                + _pdl_flags(False),
                _pdl_meta(w1, st1))
            bn2, w2, st2, bk2 = _tile("swi", rows, ca, hid)
            k2 = _Launch(
                _ln_swiglu, (-(-rows // 16), -(-hid // bn2)),
                (rows, ca, hid, 16, bn2, bk2, False, False, False,
                 rows % 16 == 0, hid % bn2 == 0, 0.0)
                + _pdl_flags(False),
                _pdl_meta(w2, st2))
            bn3, w3, st3, bk3 = _tile("out", rows, hid, ca)
            k3 = _Launch(
                _out, (-(-rows // 16), -(-ca // bn3)),
                (rows, hid, ca, cs, 16, bn3, bk3, _bk(cs),
                 mask is not None, True, rows % 16 == 0, ca % bn3 == 0)
                + _pdl_flags(True),
                _pdl_meta(w3, st3))
            plan = self._plans[key] = (k1, k2, k3)
        k1, k2, k3 = plan

        cond = self._buf(0, rows * ca, a.device)
        h = self._buf(1, rows * hid, a.device)
        y = torch.empty_like(a)
        p1 = (a, s, lnsw, wg, bg, ws, cond)
        p2 = (cond, None, None, wa, wb, h)
        p3 = (h, wo, y, mask, s, wgo, bgo)
        if k1.run is None or k2.run is None or k3.run is None:
            k1.warm(p1)
            k2.warm(p2)
            k3.warm(p3)
        else:
            k1.slow(p1)
            k2.slow(p2)
            k3.slow(p3)
        if k1.run is not None and k2.run is not None and k3.run is not None:
            self._hot = (
                a.shape, s.shape, None if mask is None else mask.shape, dev,
                k1.run, k1.gx, k1.gy, k1.pre, k1.cargs,
                k2.run, k2.gx, k2.gy, k2.pre, k2.cargs,
                k3.run, k3.gx, k3.gy, k3.pre, k3.cargs,
                # The *owning* `_parameters` dicts, not the parameter tensors:
                # a cached tensor cannot detect its own replacement. See the
                # matching comment in SwiGLUTransition.
                ada.layer_norm_s._parameters, ada.linear_g._parameters,
                ada.linear_s._parameters, self.swiglu.linear_a._parameters,
                self.swiglu.linear_b._parameters, self.linear_out._parameters,
                self.linear_g._parameters,
                tuple(t.data_ptr() for t in pars),
                cond.data_ptr(), h.data_ptr(), cond, h,
            )
        return y

    def forward(
        self,
        a: torch.Tensor,
        s: torch.Tensor,
        mask: torch.Tensor | None = None,
        chunk_size: int | None = None,
    ) -> torch.Tensor:
        hot = self._hot
        if hot is not None and chunk_size is None:
            (ashape, sshape, mshape, dev,
             run1, gx1, gy1, pre1, post1,
             run2, gx2, gy2, pre2, post2,
             run3, gx3, gy3, pre3, post3,
             dlns, dag, das, dwa, dwb, dwo, dlg,
             cpars, cp, hp, _c, _h) = hot
            if (a.shape == ashape and s.shape == sshape
                    and a.dtype is torch.bfloat16
                    and s.dtype is torch.bfloat16
                    and a.is_contiguous() and s.is_contiguous()
                    and s.get_device() == dev
                    and (mask is None if mshape is None
                         else (mask is not None and mask.shape == mshape
                               and mask.dtype is torch.bfloat16
                               and mask.is_contiguous()
                               and mask.get_device() == dev))
                    and not torch.is_grad_enabled()
                    and a.get_device() == dev and _cur_device() == dev):
                lnswp = dlns["weight"].data_ptr()
                wgp = dag["weight"].data_ptr()
                bgp = dag["bias"].data_ptr()
                wsp = das["weight"].data_ptr()
                wap = dwa["weight"].data_ptr()
                wbp = dwb["weight"].data_ptr()
                wop = dwo["weight"].data_ptr()
                wgop = dlg["weight"].data_ptr()
                bgop = dlg["bias"].data_ptr()
                y = torch.empty_like(a)
                ap = a.data_ptr()
                sp = s.data_ptr()
                yp = y.data_ptr()
                mp = 0 if mask is None else mask.data_ptr()
                # As in SwiGLUTransition: one test covering both the 16-byte
                # alignment the kernels were specialized on (only the per-call
                # pointers can break it) and whether any parameter was replaced
                # or relocated.
                if not (((ap | sp | yp | mp) & 15)
                        | (lnswp ^ cpars[0]) | (wgp ^ cpars[1])
                        | (bgp ^ cpars[2]) | (wsp ^ cpars[3])
                        | (wap ^ cpars[4]) | (wbp ^ cpars[5])
                        | (wop ^ cpars[6]) | (wgop ^ cpars[7])
                        | (bgop ^ cpars[8])):
                    st = _raw_stream(dev)
                    run1(gx1, gy1, 1, st, *pre1,
                         ap, sp, lnswp, wgp, bgp, wsp, cp, *post1)
                    run2(gx2, gy2, 1, st, *pre2,
                         cp, None, None, wap, wbp, hp, *post2)
                    run3(gx3, gy3, 1, st, *pre3,
                         hp, wop, yp, mp or None, sp, wgop, bgop, *post3)
                    return y
        if chunk_size is None and a.is_cuda and not torch.is_grad_enabled():
            try:
                y = self._prepare(a, s, mask)
            except Exception as exc:  # noqa: BLE001
                self._fell_over(exc)
                y = None
            if y is not None:
                return y
        return self._reference(a, s, mask, chunk_size)
