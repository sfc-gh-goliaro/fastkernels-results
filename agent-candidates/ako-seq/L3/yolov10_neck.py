"""YOLOv10 neck as ONE fused plan rather than a chain of optimized blocks.

The measurement that sets the whole strategy: the composed frozen-L2 neck --
already 6.35x over the baseline -- is **100% host-bound**.  Its scored window is
235 us at batch 1 and 241 us at batch 4, and the host time to *issue* the
forward without ever synchronizing is 231 us and 241 us.  The window equals the
Python time, and it does not move when the FLOPs go up 4x, so the GPU is idle
waiting on the interpreter for essentially the whole window.  Twelve sub-block
dispatches (2 upsamples, 4 concats, 4 C2f blocks, 2 downsamples) each re-derive
a ``(data_ptr, _version)`` cache guard over its own weights, look up its own
shape-keyed plan and issue its own graph replay or megakernel; that bookkeeping,
not any convolution, is what the operator costs.

So the neck is treated as one plan with one guard, and the twelve blocks stop
existing as runtime entities:

* **One plan, one guard, one workspace.**  Every BatchNorm in the neck (22
  convolutions) is folded once, lazily, on the first forward; all 22 weight
  tensors are packed into one buffer and all 22 bias vectors into another; every
  internal activation is a view into a single flat workspace cached on
  ``(batch, dtype, device)``.  Steady state does no allocation, no ``torch.cat``,
  and one ``(data_ptr, _version)`` check for the whole neck.
* **The data-movement ops are erased into their consumer's addressing.**  Both
  2x nearest upsamples are an index halving, so ``cat1``/``cat2`` become a
  two-source K axis on the following 1x1 with ``pix = (oh>>1)*Ws + (ow>>1)`` on
  the upsampled source -- there is no upsampled tensor.  ``cat4`` is a two-source
  K axis with no gather.  ``cat3`` disappears entirely: ``p4`` is *written* into
  channels [64, 192) of the 192-channel buffer ``c2f_n4.cv1`` reads, and
  ``down_p3`` writes channels [0, 64) of the same buffer, extending L2's
  shared-concat-buffer trick across block boundaries.  Four concats and two
  upsamples: six device ops and six dispatches, gone, replaced by constant
  offsets inside kernels that had to run anyway.
* **The stage table is executed behind one capture.**  Issuing the 22 stages
  eagerly costs 445 us of *host* time -- Triton's Python dispatch is ~20 us per
  launch once a kernel carries 30-odd specialization arguments -- which is worse
  than the composed blocks it replaces, so the plan is only a win behind a
  whole-neck CUDA graph.  The eager list (``FK_NECK_PATH=eager``) is the capture
  vehicle and the attribution path; the *fallback* is ``_reference``.
* **Consecutive graph nodes overlap via PDL.**  The 22 stages are a strict serial
  chain on one stream, so each stage's CTAs are dispatched during its
  predecessor's tail and only the first load of predecessor-written data waits
  (``gdc_wait`` sits after the index arithmetic).  Worth 5-9% measured.
* **Every stage's tile is measured on its own shape** (``_TILE`` / ``_DW_TILE`` /
  ``_S2_TILE`` / ``_UPF_TILE``, swept by ``tools/sweep.py``, which checks a
  candidate's output against the shipped config's before believing its time).  Once the host cost is gone the
  window is device time, and these stages are bound by A/W tile traffic, so the
  lever is tile *area*: the rule ported from L2 caps ``BLOCK_CO`` at 32, and
  raising it to the full output channel axis is 1.7x on the ``cat1`` 1x1 alone.
  The 22-stage device sum is 198 -> 138 us at batch 4 on tiles alone.
* **The stride-2 convolution reads int32 PAIRS.**  ``down_p3`` is the one stage
  whose input span is strided, and even after tiling it was 3.9x slower than the
  stride-1 conv with identical COUT, K, output pixels and batch.  With pad 1 the
  three x-taps of an input row live in exactly two adjacent 32-bit words, so
  viewing the fp16 input as int32 turns the gather into two *contiguous* 4-byte
  loads that feed all three taps -- 197.7 -> 183.4 us at batch 4.  See
  ``_conv_s2_kernel``.

Round 2 measured where the rest of the window is, and it was not where round 1
thought.  The 22 stages captured as ONE serial chain cost only 3.5-4.5 us more
than the sum of the stages timed individually, so the inter-node gap is ~0.2 us
and not the ~1.9 us round 1 inferred; what is expensive is an **eager** launch,
at ~4 us each in the scored loop.  Three things follow, and they are this round:

* **Six eager copy nodes became two.**  The graph's static input buffers are three
  views of one slab and its static outputs likewise, so the fill is a single
  ``torch.cat(out=)`` and the drain a single ``clone`` whose three views are
  returned.  Identical bytes, four fewer eager launches: **-16.9 us at batch 1,
  -13.0 us at batch 4**.  See ``_slab_views`` and ``_COPY``.
* **A 1x1 conv commutes with 2x nearest upsampling**, so the two upsample-fed 1x1s
  evaluate ``up(W @ x)`` instead of ``W @ up(x)`` -- their upsampled source's dot
  runs once per source pixel instead of four times.  It is 2/3 of the K axis in
  both, so their MAC count halves exactly, with no extra node: a program owns an
  ``RS x CS`` tile of *source* pixels and doubles its accumulator in registers.
  Stage 0 9.88 -> 7.27 us and stage 4 15.14 -> 6.12 us at batch 4.  See
  ``_gemm_up_kernel``.
* **The graph's first node warms L2 with the weights.**  The harness zeroes a
  253 MiB L2-flush buffer immediately before each timed region, so every stage's
  weight tile is a cold HBM read while its activation tile -- just written by the
  predecessor -- is still resident.  That is 19 us at batch 1 and 17 us at batch 4
  inside the window (``graph.replay()`` measured 98.2 us behind the flush against
  79.0 us without it), and it is per stage, because a serial chain at 100-400 CTAs
  cannot hide a 600 ns miss.  One 1.3 MB streaming read of the folded weight+bias
  allocation up front costs ~0.2 us and is worth **-8.2 us at batch 1, -6.2 at
  batch 4**.  See ``_touch_kernel``.

Three things were tried, measured to be *worse*, and survive only as knobs
defaulted off so the next round does not re-derive them.

* ``FK_NECK_NOMASK`` reads halo taps unmasked out of slack-padded slabs, on the
  theory that a mask varying along the contiguous pixel axis blocks widening:
  batch 4 159.7 -> 171.0 us.
* ``FK_NECK_S1`` makes the 3x3's tap loop outer so each tap address is the pixel
  base plus a literal (``_conv_s1_kernel``): batch 1 113.6 -> 134.0 us.  It also
  pins the dot's k extent at one tap's channels, and that dominates.
* ``FK_NECK_FUSE`` fuses each block's chain-out 3x3 into its cv2 1x1
  (``_fuse_kernel``), 22 stages -> 19: batch 1 105.4 -> 109.6 us, batch 4 neutral.
  Removing a boundary is worth ~0.2 us of gap (measured: the 22 stages captured as
  one chain cost only 3.5-4.5 us more than the sum of them timed alone) plus ~1.2
  us of one kernel's fixed cost, and it *costs* the grid's output-channel
  dimension, because phase A needs every producer channel resident.  At batch 1
  that dimension is where half the CTAs came from.

The first two say the same thing and the third confirms it from the other side:
**these convolutions are limited by the dot at low occupancy** -- not by how the A
tile is addressed, masked or widened, and not by stage boundaries.

Where this leaves the window: **105 us at batch 1 and 153 us at batch 4**, of which
~72 / ~122 us is the 22-stage chain itself, ~24 / ~30 us is the two remaining eager
copies plus the harness' own in-window input copies, ~11 us is residual L2-flush
penalty and ~2 us is the graph launch.

``p4`` is consumed twice (by the ``cat2`` upsample and by ``cat3``) so it stays
materialized, as channels [64, 192) of that shared buffer; ``p3``, ``n4`` and
``n5`` are returned as contiguous NCHW tensors -- views of one freshly cloned
slab, which is what makes the drain a single node.

Contracts.  Every submodule keeps its baseline name and type, so the harness'
``load_state_dict(baseline.state_dict(), strict=False)`` shares weights exactly
(``tools/harness.py`` asserts the candidate's key set covers the baseline's -- a
rename here would silently drop weights and "pass" on garbage).  The folded weights live outside
the module tree.  The plan is guarded by ``(data_ptr, _version)`` over every
tensor it was derived from plus a ``load_state_dict`` hook, so a later weight
load, an in-place BN edit or a call to a submodule's ``fuse()`` rebuilds it.
Training mode, grad-enabled autograd, any dtype but fp16, CPU tensors,
non-contiguous or unexpected geometry, and a missing Triton each fall back to
``_reference``, which is the baseline forward over the frozen L2 blocks.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn

from ..L1.interpolate import Interpolate
from ..L2.yolov10_c2f import YOLOC2f, YOLOC2fCIB
from ..L2.yolov10_concat import YOLOConcat
from ..L2.yolov10_conv import YOLOConv
from ..L2.yolov10_scdown import YOLOSCDown

try:
    from ..L2.yolov10_c2f import _conv_step, _repvgg_step
except Exception:  # noqa: BLE001 - fold locally if the L2 helpers move
    _conv_step = _repvgg_step = None

try:
    import triton
    import triton.language as tl
except Exception:  # noqa: BLE001 - no Triton: the reference path is the whole op
    triton = None

# Programmatic Dependent Launch.  The 22 stages are a strict serial chain on one
# stream, so every stage's CTAs can be *dispatched* during its predecessor's tail
# and only the first load of predecessor-written data has to wait -- which is
# exactly what PDL buys, and it is the one lever that attacks the ~1.1 us
# per-node gap a replayed graph pays without changing the stage table at all.
_HAS_PDL = False
if triton is not None:
    try:
        from triton.language.extra.cuda import gdc_launch_dependents as _gdc_launch
        from triton.language.extra.cuda import gdc_wait as _gdc_wait
        _HAS_PDL = True
    except Exception:  # noqa: BLE001 - older Triton: the PDL branches stay dead
        @triton.jit
        def _gdc_wait():
            tl.debug_barrier()

        @triton.jit
        def _gdc_launch():
            tl.debug_barrier()

# Knobs, for attributing each lever in its own bench rather than for tuning.
# "eager" (attribution / capture vehicle) | "graph" | "auto"
_PATH = os.environ.get("FK_NECK_PATH", "auto")
_USE_PDL = os.environ.get("FK_NECK_PDL", "1") != "0"
# "auto" (per-stage, from addressing), or "0"/"1" to force, for A/B.
_ROWT_MODE = os.environ.get("FK_NECK_ROWT", "auto")
# How the graph's static buffers are filled and drained.  This is worth a knob
# because the copies are SIX of the graph's 28 nodes and the node gap is ~2 us:
#   "plain"   three ``copy_`` in, three ``clone`` out  (r1's shipped form)
#   "slab"    the static in/out buffers are views of ONE contiguous slab each,
#             so copy-in is a single ``torch.cat(out=)`` node and copy-out a
#             single ``clone`` -- 6 nodes become 2, same bytes moved.
#   "foreach" one multi-tensor-apply kernel each way (measured 3-4 us WORSE than
#             "plain" in r1; kept only so it is not re-measured by accident).
_COPY = os.environ.get("FK_NECK_COPY", "slab")
# Compute an upsampled source's contribution at the SOURCE resolution, in the
# same kernel; see _gemm_up_kernel.  Halves the MAC count of the two
# upsample-fed 1x1s (stages 0 and 4) and adds no graph node.
_USE_UPF = os.environ.get("FK_NECK_UPF", "1") != "0"
# Fuse each C2f block's chain-out 3x3 into its cv2 1x1; see _fuse_kernel.
_USE_FUSE = os.environ.get("FK_NECK_FUSE", "0") != "0"
# Warm L2 with the folded weights as the graph's first node; see _touch_kernel.
_USE_PF = os.environ.get("FK_NECK_PF", "1") != "0"
# Tap-major body for the shape-preserving stride-1 convs; see _conv_s1_kernel.
# MEASURED LOSS, and a large one: whole-neck window batch 1 113.6 -> 134.0 us,
# batch 4 159.9 -> 177.1 us.  Making the tap the outer loop does remove the
# [BLOCK_K, BLOCK_P] address tensor and the 4-term mask, but it also fixes the
# dot's k extent at ONE tap's channels (64, or 32 at 80x80) instead of BLOCK_K,
# and nine k=64 dots are worth much less than five k=128 ones.  Default OFF.
_USE_S1 = os.environ.get("FK_NECK_S1", "0") != "0"
# Read halo taps UNMASKED out of slack-padded slabs; see _SLACK / NOMASK.
# MEASURED LOSS, kept only as a knob so it is not re-derived: whole-neck window
# batch 4 159.7 -> 171.0 us, batch 1 113.7 -> 115.7 us.  The reasoning was that a
# halo mask varies along the contiguous pixel axis so a predicated load cannot
# widen; the measurement says that is not what those stages pay for, and the
# `tl.where` over a [BLOCK_K, BLOCK_P] register tile plus the extra halo bytes
# costs more than the predication saved.  Default OFF.
_USE_NOMASK = os.environ.get("FK_NECK_NOMASK", "0") != "0"
# int32-pair path for the 3x3 stride-2 conv (``down_p3``); see _conv_s2_kernel.
# Measured against the generic body with its swept tile, whole-neck window:
# batch 4 197.7 -> 183.4 us, batch 1 131.3 -> 132.1 us (inside run-to-run noise).
# The stage is 24% of batch-4 device time and this is where its addressing cost
# goes, so it ships on; batch 1 has too few CTAs for the extra load volume to pay.
_USE_S2 = os.environ.get("FK_NECK_S2", "1") != "0"
# Re-raise instead of falling back, so a broken stage is a crash in development
# rather than a silent 240 us regression.  ``_graph_build``'s blanket ``except``
# is correct for shipping and awful for debugging: with the (unfinished) int32
# stride-2 body enabled, capture failed, the neck quietly ran ``_reference``, and
# the run looked "numerically exact" because the reference path is exact.
_DEBUG = os.environ.get("FK_NECK_DEBUG", "0") != "0"
_MAX_PLANS = 8
# Elements of unused padding at BOTH ends of every slab a haloed load may read
# past (the workspace and the three returned outputs).  A k x k convolution's
# halo tap at output pixel 0 addresses ``-(k//2)*IMW - k//2`` relative to its
# plane, and the last plane's bottom-right tap addresses that far past the end;
# with the padding present the address is always a legal read, so the load does
# not need a mask, and an *unmasked* contiguous fp16 load widens to 16 bytes
# where a predicated one degenerates to one 2-byte access per element.  The halo
# lanes are zeroed after the load instead.  4 KiB at each end covers every
# geometry in the neck with margin -- the worst reach is the 20x20 stride-2
# depthwise, whose 512-pixel tile runs 480 elements past its 40x40 source -- and
# ``_reach`` derives the reach per stage rather than trusting this comment.
_SLACK = 2048
_MISSING = object()
_SMS = None


def _num_sms():
    global _SMS
    if _SMS is None:
        _SMS = torch.cuda.get_device_properties(
            torch.cuda.current_device()).multi_processor_count
    return _SMS


# ---------------------------------------------------------------------------
# Kernels.
#
# Three stage bodies cover all 22 convolutions in the neck:
#
#   _gemm_kernel   1x1, stride 1, ONE OR TWO sources, each optionally read
#                  through a 2x nearest upsample gather.  This is where cat1,
#                  cat2, cat4 and both upsamples live.
#   _conv_kernel   dense k x k, odd k, padding k//2, any stride (the bottleneck
#                  3x3s and ``down_p3``'s 3x3 stride 2).
#   _dw_kernel     depthwise k x k, odd k, padding k//2, any stride (the CIB's
#                  3x3/7x7 and ``down_n4``'s 3x3 stride 2).
#
# All three end in the same epilogue -- ``silu(acc + bias) + residual`` on the
# fp32 accumulator, one narrowing store -- and all three address source,
# residual and destination as a *constant element offset* into a tensor plus
# (batch, channel) strides, which is what lets a stage read one channel slice of
# a shared buffer and write another with no copy and no ``contiguous()``.  The
# offsets are ``tl.constexpr`` and every one is a multiple of 8 halves, so
# Triton's alignment analysis still widens the loads.
#
# Structure (accumulator laid out [BLOCK_CO, BLOCK_P] so the store walks the
# output's contiguous pixel axis; the A tile gathered straight out of NCHW
# inside a statically unrolled k loop; the halo masked rather than padded; the
# weight pre-transposed to [COUT, KH*KW, C] so a k tile is contiguous along the
# channel axis it blocks over) is the one ``candidate/L1/conv2d.py`` established
# and ``candidate/L2/yolov10_c2f.py`` swept for exactly these shapes.  What is
# new here is the multi-source gather, the stride, and that the offsets span the
# whole neck instead of one block.
# ---------------------------------------------------------------------------
if triton is not None:

    @triton.jit
    def _touch_kernel(SRC, OUT, N: tl.constexpr, BLOCK: tl.constexpr,
                      PDL: tl.constexpr):
        """Pull the whole folded weight/bias buffer into L2 as the graph's node 0.

        The harness zeroes a 253 MiB L2-flush buffer immediately before each timed
        region, so every stage's *weight* tile is a cold HBM read while its
        activation tile was just written by the predecessor and is still resident.
        That costs 19 us at batch 1 and 17 us at batch 4 -- measured as
        ``graph.replay()`` with the flush (98.2 us) against without it (79.0 us) --
        and it is per stage, because a serial chain at 100-400 CTAs cannot hide a
        600 ns miss behind other work.

        One 1.3 MB streaming read up front turns 22 rounds of exposed HBM latency
        into 22 rounds of L2 hits, for ~0.2 us of bandwidth and one graph node
        (~0.2 us).  The sum is stored so the loads cannot be dead-coded; ``OUT`` is
        scratch nothing reads.
        """
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        v = tl.load(SRC + offs, mask=offs < N, other=0.0)
        tl.store(OUT + pid, tl.sum(v.to(tl.float32)))
        if PDL:
            _gdc_launch()


    @triton.jit
    def _gemm_kernel(X0, X1, WGT, BIA, RES, Y,
                     C0: tl.constexpr, C1: tl.constexpr, K: tl.constexpr,
                     O0: tl.constexpr, O1: tl.constexpr,
                     SN0: tl.constexpr, SN1: tl.constexpr,
                     SC0: tl.constexpr, SC1: tl.constexpr,
                     UP0: tl.constexpr, UP1: tl.constexpr,
                     WS0: tl.constexpr, WS1: tl.constexpr,
                     COUT: tl.constexpr, P: tl.constexpr, OW: tl.constexpr,
                     WOFF: tl.constexpr, BOFF: tl.constexpr,
                     YOFF: tl.constexpr, YSN: tl.constexpr, YSC: tl.constexpr,
                     ROFF: tl.constexpr, RSN: tl.constexpr, RSC: tl.constexpr,
                     ACT: tl.constexpr, HAS_RES: tl.constexpr,
                     BLOCK_CO: tl.constexpr, BLOCK_P: tl.constexpr,
                     BK0: tl.constexpr, NK0: tl.constexpr, EK0: tl.constexpr,
                     BK1: tl.constexpr, NK1: tl.constexpr, EK1: tl.constexpr,
                     EVEN_CO: tl.constexpr, EVEN_P: tl.constexpr,
                     ROWT: tl.constexpr, TPR: tl.constexpr,
                     PDL: tl.constexpr):
        """1x1 conv over a K axis that may span two differently-addressed sources.

        ``UPs`` turns source *s* into its own 2x nearest upsample: output pixel
        (oh, ow) reads source pixel (oh >> 1, ow >> 1), so the upsampled tensor
        is never written.  The reads are 4-way redundant, but they are L1/L2 hits
        against a store that had to happen anyway.
        """
        pid_p = tl.program_id(0)
        pid_co = tl.program_id(1)
        n = tl.program_id(2)
        oc = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
        mc = oc < COUT
        if ROWT:
            # One output ROW per program (plus a column tile).  The upsample
            # gather ``(oh>>1)*Ws + (ow>>1)`` is only affine in the flat pixel
            # index while a tile stays inside one row; a tile that straddles a
            # row boundary changes the source *row* mid-tile, which scatters the
            # A-tile load.  OW is 40 or 80 here and no power-of-two tile divides
            # 40, so the flat tiling straddles constantly -- confining the tile
            # to a row costs masked lanes and buys back the coalescing.
            ohs = pid_p // TPR
            owt = pid_p - ohs * TPR
            ow_v = owt * BLOCK_P + tl.arange(0, BLOCK_P)
            mp = ow_v < OW
            op = ohs * OW + ow_v
        else:
            op = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
            mp = op < P
        acc = tl.zeros((BLOCK_CO, BLOCK_P), dtype=tl.float32)

        if UP0:
            if ROWT:
                pix0 = (ohs // 2) * WS0 + (ow_v // 2)
            else:
                oh0 = op // OW
                pix0 = (oh0 // 2) * WS0 + ((op - oh0 * OW) // 2)
        else:
            pix0 = op
        x0 = X0 + O0 + n * SN0 + pix0[None, :]
        if PDL:
            _gdc_wait()
        for kb in tl.static_range(NK0):
            ok = kb * BK0 + tl.arange(0, BK0)
            if EK0 and EVEN_P:
                a = tl.load(x0 + ok[:, None] * SC0)
            else:
                am = mp[None, :] if EK0 else (
                    (ok < C0)[:, None] if EVEN_P else (ok < C0)[:, None] & mp[None, :])
                a = tl.load(x0 + ok[:, None] * SC0, mask=am, other=0.0)
            wp = WGT + WOFF + oc[:, None] * K + ok[None, :]
            if EVEN_CO and EK0:
                # An unmasked weight tile is the difference between one widened
                # vector load and one predicated load per element: the mask
                # varies along the contiguous (k) axis so Triton cannot widen a
                # masked one, and weight traffic here is the same order as
                # activation traffic.
                w = tl.load(wp)
            else:
                w = tl.load(wp, mask=(mc[:, None] & (ok < C0)[None, :]), other=0.0)
            acc = tl.dot(w, a, acc=acc)

        if NK1 > 0:
            if UP1:
                if ROWT:
                    pix1 = (ohs // 2) * WS1 + (ow_v // 2)
                else:
                    oh1 = op // OW
                    pix1 = (oh1 // 2) * WS1 + ((op - oh1 * OW) // 2)
            else:
                pix1 = op
            x1 = X1 + O1 + n * SN1 + pix1[None, :]
            for kb in tl.static_range(NK1):
                ok = kb * BK1 + tl.arange(0, BK1)
                if EK1 and EVEN_P:
                    a = tl.load(x1 + ok[:, None] * SC1)
                else:
                    am = mp[None, :] if EK1 else (
                        (ok < C1)[:, None] if EVEN_P else (ok < C1)[:, None] & mp[None, :])
                    a = tl.load(x1 + ok[:, None] * SC1, mask=am, other=0.0)
                wp = WGT + WOFF + oc[:, None] * K + (C0 + ok)[None, :]
                if EVEN_CO and EK1:
                    w = tl.load(wp)
                else:
                    w = tl.load(wp, mask=(mc[:, None] & (ok < C1)[None, :]),
                                other=0.0)
                acc = tl.dot(w, a, acc=acc)

        if BOFF >= 0:
            if EVEN_CO:
                acc += tl.load(BIA + BOFF + oc)[:, None].to(tl.float32)
            else:
                acc += tl.load(BIA + BOFF + oc, mask=mc, other=0.0)[:, None].to(tl.float32)
        if ACT:
            acc *= tl.sigmoid(acc)
        om = mc[:, None] & mp[None, :]
        if HAS_RES:
            r = RES + ROFF + n * RSN + oc[:, None] * RSC + op[None, :]
            if EVEN_CO and EVEN_P:
                acc += tl.load(r).to(tl.float32)
            else:
                acc += tl.load(r, mask=om, other=0.0).to(tl.float32)
        o = Y + YOFF + n * YSN + oc[:, None] * YSC + op[None, :]
        if EVEN_CO and EVEN_P:
            tl.store(o, acc.to(Y.dtype.element_ty))
        else:
            tl.store(o, acc.to(Y.dtype.element_ty), mask=om)
        if PDL:
            _gdc_launch()


    @triton.jit
    def _gemm_up_kernel(X0, X1, WGT, BIA, RES, Y,
                        C0: tl.constexpr, C1: tl.constexpr, K: tl.constexpr,
                        O0: tl.constexpr, O1: tl.constexpr,
                        SN0: tl.constexpr, SN1: tl.constexpr,
                        SC0: tl.constexpr, SC1: tl.constexpr,
                        SH0: tl.constexpr, SW0: tl.constexpr,
                        COUT: tl.constexpr, OW: tl.constexpr,
                        WOFF: tl.constexpr, BOFF: tl.constexpr,
                        YOFF: tl.constexpr, YSN: tl.constexpr, YSC: tl.constexpr,
                        ROFF: tl.constexpr, RSN: tl.constexpr, RSC: tl.constexpr,
                        ACT: tl.constexpr, HAS_RES: tl.constexpr,
                        BLOCK_CO: tl.constexpr, RS: tl.constexpr, CS: tl.constexpr,
                        BK0: tl.constexpr, NK0: tl.constexpr, EK0: tl.constexpr,
                        BK1: tl.constexpr, NK1: tl.constexpr, EK1: tl.constexpr,
                        EVEN_CO: tl.constexpr, EVEN_S: tl.constexpr,
                        TPR: tl.constexpr, PDL: tl.constexpr):
        """A 1x1 conv whose first source is a 2x nearest upsample, with that
        source's contribution computed at the SOURCE resolution.

        A 1x1 convolution commutes with nearest-neighbour upsampling:

            (W @ up(x))[:, 2i+dy, 2j+dx] == up(W @ x)[:, 2i+dy, 2j+dx]

        because every one of the four output pixels of source pixel (i, j) reads
        the *same* source vector.  ``_gemm_kernel`` evaluates the left-hand side,
        so it does that source's dot four times per source pixel; this kernel
        evaluates the right-hand side, once.  Two of the neck's stages are of this
        form and the upsampled source is 2/3 of their K axis in both, so their
        MAC count drops to 1/6 + 1/3 = **half**.  It is exact, not approximate:
        the four output pixels get the same fp32 accumulator, bit for bit, and
        the second source is then accumulated into it in the same order
        ``_gemm_kernel`` uses.

        The tiling is what makes it one kernel instead of two.  A program owns an
        ``RS x CS`` tile of *source* pixels and all ``BLOCK_CO`` output channels
        of it, so the partial never leaves registers and no node is added:

          phase 1   acc0[BLOCK_CO, RS*CS]      = W[:, :C0] @ x0[tile]
          expand    accE[BLOCK_CO, RS*2*CS]    = each source lane doubled, via
                                                 ``join`` + ``reshape`` (lane
                                                 ``2l+j`` takes lane ``l``), which
                                                 is the dx duplication
          phase 2   for dy in (0, 1): acc = accE + W[:, C0:] @ x1[out tile]

        Lane ``m`` of the expanded tile is source lane ``l = m // 2`` and
        ``dx = m & 1``, so for each of the ``RS`` source rows the ``2*CS`` output
        lanes are one contiguous run of output columns -- the store and the
        second source's load stay coalesced in runs of ``2*CS`` halves.
        """
        pid_p = tl.program_id(0)
        pid_co = tl.program_id(1)
        n = tl.program_id(2)
        oc = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
        mc = oc < COUT
        rt = pid_p // TPR
        ct = pid_p - rt * TPR
        lane = tl.arange(0, RS * CS)
        sr = rt * RS + lane // CS
        sc = ct * CS + (lane % CS)
        ms = (sr < SH0) & (sc < SW0)
        spix = sr * SW0 + sc
        x0 = X0 + O0 + n * SN0 + spix[None, :]
        acc0 = tl.zeros((BLOCK_CO, RS * CS), dtype=tl.float32)
        if PDL:
            _gdc_wait()
        for kb in tl.static_range(NK0):
            ok = kb * BK0 + tl.arange(0, BK0)
            if EK0 and EVEN_S:
                a = tl.load(x0 + ok[:, None] * SC0)
            else:
                am = ms[None, :] if EK0 else (
                    (ok < C0)[:, None] if EVEN_S else (ok < C0)[:, None] & ms[None, :])
                a = tl.load(x0 + ok[:, None] * SC0, mask=am, other=0.0)
            wp = WGT + WOFF + oc[:, None] * K + ok[None, :]
            if EVEN_CO and EK0:
                w = tl.load(wp)
            else:
                w = tl.load(wp, mask=(mc[:, None] & (ok < C0)[None, :]), other=0.0)
            acc0 = tl.dot(w, a, acc=acc0)

        # dx duplication: [BLOCK_CO, RS*CS] -> [BLOCK_CO, RS*CS*2], lane 2l+j <- l.
        accE = tl.reshape(tl.join(acc0, acc0), (BLOCK_CO, RS * CS * 2))
        msE = tl.reshape(tl.join(ms, ms), (RS * CS * 2,))
        lane2 = tl.arange(0, RS * CS * 2)
        l2 = lane2 // 2
        r2 = l2 // CS
        ow2 = 2 * (ct * CS + (l2 - r2 * CS)) + (lane2 - 2 * l2)
        for dy in tl.static_range(2):
            op = (2 * (rt * RS + r2) + dy) * OW + ow2
            acc = accE
            if NK1 > 0:
                x1 = X1 + O1 + n * SN1 + op[None, :]
                for kb in tl.static_range(NK1):
                    ok = kb * BK1 + tl.arange(0, BK1)
                    if EK1 and EVEN_S:
                        a = tl.load(x1 + ok[:, None] * SC1)
                    else:
                        am = msE[None, :] if EK1 else (
                            (ok < C1)[:, None] if EVEN_S
                            else (ok < C1)[:, None] & msE[None, :])
                        a = tl.load(x1 + ok[:, None] * SC1, mask=am, other=0.0)
                    wp = WGT + WOFF + oc[:, None] * K + (C0 + ok)[None, :]
                    if EVEN_CO and EK1:
                        w = tl.load(wp)
                    else:
                        w = tl.load(wp, mask=(mc[:, None] & (ok < C1)[None, :]),
                                    other=0.0)
                    acc = tl.dot(w, a, acc=acc)
            if BOFF >= 0:
                if EVEN_CO:
                    acc += tl.load(BIA + BOFF + oc)[:, None].to(tl.float32)
                else:
                    acc += tl.load(BIA + BOFF + oc, mask=mc,
                                   other=0.0)[:, None].to(tl.float32)
            if ACT:
                acc *= tl.sigmoid(acc)
            om = mc[:, None] & msE[None, :]
            if HAS_RES:
                r = RES + ROFF + n * RSN + oc[:, None] * RSC + op[None, :]
                if EVEN_CO and EVEN_S:
                    acc += tl.load(r).to(tl.float32)
                else:
                    acc += tl.load(r, mask=om, other=0.0).to(tl.float32)
            o = Y + YOFF + n * YSN + oc[:, None] * YSC + op[None, :]
            if EVEN_CO and EVEN_S:
                tl.store(o, acc.to(Y.dtype.element_ty))
            else:
                tl.store(o, acc.to(Y.dtype.element_ty), mask=om)
        if PDL:
            _gdc_launch()


    @triton.jit
    def _conv_kernel(X, WGT, BIA, RES, Y,
                     C: tl.constexpr, K: tl.constexpr,
                     IMH: tl.constexpr, IMW: tl.constexpr,
                     XOFF: tl.constexpr, XSN: tl.constexpr, XSC: tl.constexpr,
                     COUT: tl.constexpr, P: tl.constexpr, OW: tl.constexpr,
                     KH: tl.constexpr, KW: tl.constexpr,
                     SH: tl.constexpr, SW: tl.constexpr,
                     PH: tl.constexpr, PW: tl.constexpr,
                     WOFF: tl.constexpr, BOFF: tl.constexpr,
                     YOFF: tl.constexpr, YSN: tl.constexpr, YSC: tl.constexpr,
                     ROFF: tl.constexpr, RSN: tl.constexpr, RSC: tl.constexpr,
                     ACT: tl.constexpr, HAS_RES: tl.constexpr,
                     BLOCK_CO: tl.constexpr, BLOCK_P: tl.constexpr,
                     BLOCK_K: tl.constexpr, NUM_K: tl.constexpr,
                     EVEN_K: tl.constexpr, EVEN_CO: tl.constexpr,
                     EVEN_P: tl.constexpr, ROWT: tl.constexpr,
                     TPR: tl.constexpr, NOMASK: tl.constexpr,
                     PDL: tl.constexpr):
        """Dense k x k implicit GEMM, M = OH*OW, N = COUT, K = C*KH*KW."""
        pid_p = tl.program_id(0)
        pid_co = tl.program_id(1)
        n = tl.program_id(2)
        oc = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
        mc = oc < COUT
        if ROWT:
            # One output row per program.  For a stride-1 shape-preserving conv
            # the tap address is ``op + (i-PH)*IMW + (j-PW)`` -- affine in the
            # flat pixel index, so flat tiling is already contiguous.  At stride
            # 2 it is ``2*op + 2*oh*OW + const``, which is *not*, and a tile that
            # crosses a row boundary turns the A-tile load into a scatter.  This
            # is what makes ``down_p3`` 5x slower than the identical-FLOP
            # stride-1 3x3 (48.0 us against 9.8 us at batch 4) under flat tiling.
            ohs = pid_p // TPR
            owt = pid_p - ohs * TPR
            ow_v = owt * BLOCK_P + tl.arange(0, BLOCK_P)
            mp = ow_v < OW
            op = ohs * OW + ow_v
            ih0 = ohs * SH - PH + tl.zeros((BLOCK_P,), dtype=tl.int32)
            iw0 = ow_v * SW - PW
        else:
            op = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
            mp = op < P
            oh = op // OW
            ih0 = oh * SH - PH
            iw0 = (op - oh * OW) * SW - PW
        xn = X + XOFF + n * XSN
        acc = tl.zeros((BLOCK_CO, BLOCK_P), dtype=tl.float32)
        if PDL:
            _gdc_wait()
        for kb in tl.static_range(NUM_K):
            ok = kb * BLOCK_K + tl.arange(0, BLOCK_K)
            tap = ok // C
            cc = ok - tap * C
            if NOMASK and not EVEN_K:
                # Lanes past K are zeroed below, but their *address* must still
                # land inside the padded slab, so clamp the tap.  The clamp is a
                # function of the k axis alone, so the pixel axis -- the one the
                # load widens along -- is untouched.
                tap = tl.minimum(tap, KH * KW - 1)
            ih = ih0[None, :] + (tap // KW)[:, None]
            iw = iw0[None, :] + (tap % KW)[:, None]
            am = (ih >= 0) & (ih < IMH) & (iw >= 0) & (iw < IMW)
            if not EVEN_K:
                am = am & (ok < K)[:, None]
            if not EVEN_P:
                am = am & mp[None, :]
            ap = xn + cc[:, None] * XSC + ih * IMW + iw
            if NOMASK:
                # Every address above is a legal read (see _SLACK), so the load
                # is unmasked -- 16-byte vector loads along the contiguous pixel
                # axis instead of one predicated 2-byte access per element -- and
                # the halo is zeroed afterwards in registers.
                a = tl.where(am, tl.load(ap), 0.0)
            else:
                a = tl.load(ap, mask=am, other=0.0)
            wp = WGT + WOFF + oc[:, None] * K + ok[None, :]
            if EVEN_CO and EVEN_K:
                w = tl.load(wp)
            else:
                w = tl.load(wp, mask=(mc[:, None] & (ok < K)[None, :]), other=0.0)
            acc = tl.dot(w, a, acc=acc)
        if BOFF >= 0:
            if EVEN_CO:
                acc += tl.load(BIA + BOFF + oc)[:, None].to(tl.float32)
            else:
                acc += tl.load(BIA + BOFF + oc, mask=mc, other=0.0)[:, None].to(tl.float32)
        if ACT:
            acc *= tl.sigmoid(acc)
        om = mc[:, None] & mp[None, :]
        if HAS_RES:
            r = RES + ROFF + n * RSN + oc[:, None] * RSC + op[None, :]
            if EVEN_CO and EVEN_P:
                acc += tl.load(r).to(tl.float32)
            else:
                acc += tl.load(r, mask=om, other=0.0).to(tl.float32)
        o = Y + YOFF + n * YSN + oc[:, None] * YSC + op[None, :]
        if EVEN_CO and EVEN_P:
            tl.store(o, acc.to(Y.dtype.element_ty))
        else:
            tl.store(o, acc.to(Y.dtype.element_ty), mask=om)
        if PDL:
            _gdc_launch()


    @triton.jit
    def _conv_s1_kernel(X, WGT, BIA, RES, Y,
                        C: tl.constexpr, K: tl.constexpr,
                        IMH: tl.constexpr, IMW: tl.constexpr,
                        XOFF: tl.constexpr, XSN: tl.constexpr, XSC: tl.constexpr,
                        COUT: tl.constexpr, P: tl.constexpr, OW: tl.constexpr,
                        KH: tl.constexpr, KW: tl.constexpr,
                        PH: tl.constexpr, PW: tl.constexpr,
                        WOFF: tl.constexpr, BOFF: tl.constexpr,
                        YOFF: tl.constexpr, YSN: tl.constexpr, YSC: tl.constexpr,
                        ROFF: tl.constexpr, RSN: tl.constexpr, RSC: tl.constexpr,
                        ACT: tl.constexpr, HAS_RES: tl.constexpr,
                        BLOCK_CO: tl.constexpr, BLOCK_P: tl.constexpr,
                        BKC: tl.constexpr, NKC: tl.constexpr, EKC: tl.constexpr,
                        EVEN_CO: tl.constexpr, EVEN_P: tl.constexpr,
                        ROWT: tl.constexpr, TPR: tl.constexpr,
                        PDL: tl.constexpr):
        """Shape-preserving stride-1 k x k conv, TAP-MAJOR instead of k-major.

        Same idea that made ``_conv_s2_kernel`` pay, applied to the six 3x3
        stride-1 stages -- which are the largest single block of device time left
        (26.6 us at batch 4 for the four at 40x40, 14.4 for the two at 80x80).

        ``_conv_kernel`` walks one flat K axis of ``C*KH*KW`` and recovers the tap
        from it (``tap = ok // C``), so every k block materialises a full
        ``[BLOCK_K, BLOCK_P]`` tensor of *computed* addresses and a matching
        4-term mask -- 8192 int32 lanes of index arithmetic per block, rebuilt
        five times.  But when the convolution preserves its resolution at stride 1
        the tap address is exactly

            op + (i - PH) * IMW + (j - PW)

        i.e. the pixel base plus a COMPILE-TIME SCALAR.  So this kernel blocks the
        k axis by *channels* and makes the tap the outer static loop: the
        ``[BKC, BLOCK_P]`` address tensor is built once per channel block and each
        of the ``KH*KW`` taps adds a literal to it, while the mask collapses to two
        ``[BLOCK_P]`` comparisons per tap (one scalar in the row-confined case).
        The dot's k extent is ``BKC`` (one tap's channels) instead of ``BLOCK_K``.
        """
        pid_p = tl.program_id(0)
        pid_co = tl.program_id(1)
        n = tl.program_id(2)
        oc = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
        mc = oc < COUT
        if ROWT:
            ohs = pid_p // TPR
            owt = pid_p - ohs * TPR
            ow_v = owt * BLOCK_P + tl.arange(0, BLOCK_P)
            mp = ow_v < OW
            op = ohs * OW + ow_v
            oh_v = ohs + tl.zeros((BLOCK_P,), dtype=tl.int32)
        else:
            op = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
            mp = op < P
            oh_v = op // OW
            ow_v = op - oh_v * OW
        acc = tl.zeros((BLOCK_CO, BLOCK_P), dtype=tl.float32)
        if PDL:
            _gdc_wait()
        for kb in tl.static_range(NKC):
            ok = kb * BKC + tl.arange(0, BKC)
            # Built once per channel block; every tap below is this plus a literal.
            apb = X + XOFF + n * XSN + ok[:, None] * XSC + op[None, :]
            for t in tl.static_range(KH * KW):
                i = t // KW
                j = t - i * KW
                ihv = oh_v + (i - PH)
                iwv = ow_v + (j - PW)
                m = (ihv >= 0) & (ihv < IMH) & (iwv >= 0) & (iwv < IMW)
                if not EVEN_P:
                    m = m & mp
                am = m[None, :] if EKC else (m[None, :] & (ok < C)[:, None])
                a = tl.load(apb + ((i - PH) * IMW + (j - PW)), mask=am, other=0.0)
                wp = WGT + WOFF + oc[:, None] * K + (t * C + ok)[None, :]
                if EVEN_CO and EKC:
                    w = tl.load(wp)
                else:
                    w = tl.load(wp, mask=(mc[:, None] & (ok < C)[None, :]),
                                other=0.0)
                acc = tl.dot(w, a, acc=acc)
        if BOFF >= 0:
            if EVEN_CO:
                acc += tl.load(BIA + BOFF + oc)[:, None].to(tl.float32)
            else:
                acc += tl.load(BIA + BOFF + oc, mask=mc, other=0.0)[:, None].to(tl.float32)
        if ACT:
            acc *= tl.sigmoid(acc)
        om = mc[:, None] & mp[None, :]
        if HAS_RES:
            r = RES + ROFF + n * RSN + oc[:, None] * RSC + op[None, :]
            if EVEN_CO and EVEN_P:
                acc += tl.load(r).to(tl.float32)
            else:
                acc += tl.load(r, mask=om, other=0.0).to(tl.float32)
        o = Y + YOFF + n * YSN + oc[:, None] * YSC + op[None, :]
        if EVEN_CO and EVEN_P:
            tl.store(o, acc.to(Y.dtype.element_ty))
        else:
            tl.store(o, acc.to(Y.dtype.element_ty), mask=om)
        if PDL:
            _gdc_launch()


    @triton.jit
    def _conv_s2_kernel(X32, WGT, BIA, RES, Y,
                        C: tl.constexpr, IMH: tl.constexpr, IMW2: tl.constexpr,
                        XOFF2: tl.constexpr, XSN2: tl.constexpr,
                        XSC2: tl.constexpr,
                        COUT: tl.constexpr, K: tl.constexpr,
                        OW: tl.constexpr,
                        WOFF: tl.constexpr, BOFF: tl.constexpr,
                        YOFF: tl.constexpr, YSN: tl.constexpr, YSC: tl.constexpr,
                        ROFF: tl.constexpr, RSN: tl.constexpr, RSC: tl.constexpr,
                        ACT: tl.constexpr, HAS_RES: tl.constexpr,
                        BLOCK_CO: tl.constexpr, BLOCK_P: tl.constexpr,
                        TPR: tl.constexpr,
                        EVEN_CO: tl.constexpr, EVEN_P: tl.constexpr,
                        NOMASK: tl.constexpr, PDL: tl.constexpr):
        """3x3 stride-2 pad-1 dense conv whose input span is read as int32 PAIRS.

        ``TPR`` sits with the tile constexprs rather than next to ``OW`` on
        purpose: everything up to ``HAS_RES`` is passed positionally by ``_launch``
        and everything after it comes from ``st["consts"]`` as keywords, so a
        constant that lives in ``consts`` must not also occupy a positional slot.

        This exists because the generic kernel is 3.9x slower on this one stage
        than on the stride-1 conv with identical COUT, K, output pixel count and
        batch (25.48 us against 6.50 us at batch 4, after both were tile-swept).
        The whole difference is the gather: consecutive output columns read input
        columns two apart, so every A-tile element is its own 2-byte load and
        nothing widens.

        The fix uses the fact that stride 2 with pad 1 asks for input column
        ``2*ow + j - 1``, so across the three x-taps the *pairs* of adjacent
        halves are reused:

            word[ow-1] = (in[2ow-2], in[2ow-1])   -> its high half is tap j=0
            word[ow]   = (in[2ow],   in[2ow+1])   -> low half is j=1, high is j=2

        Viewing the input as int32 makes both of those a 4-byte, fully contiguous,
        vectorizable load, and TWO of them feed all THREE taps of an input row.
        The k axis is one tap-block of exactly C channels, so ``j`` is a literal
        in the unrolled nest, the weight columns of a block are contiguous, and
        the only non-scalar mask left is the single ``ow >= 1`` left-edge term.
        """
        pid_p = tl.program_id(0)
        pid_co = tl.program_id(1)
        n = tl.program_id(2)
        oc = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
        mc = oc < COUT
        ohs = pid_p // TPR
        owt = pid_p - ohs * TPR
        ow = owt * BLOCK_P + tl.arange(0, BLOCK_P)
        mp = ow < OW
        cc = tl.arange(0, C)
        xn = X32 + XOFF2 + n * XSN2 + cc[:, None] * XSC2
        acc = tl.zeros((BLOCK_CO, BLOCK_P), dtype=tl.float32)
        if PDL:
            _gdc_wait()
        for i in tl.static_range(3):
            ih = ohs * 2 - 1 + i
            okh = (ih >= 0) & (ih < IMH)
            row = xn + ih * IMW2
            # Both masks are 2D and both keep the ``ow < OW`` term even when the
            # tile divides OW: without it the lanes past OW would read into the
            # next row, and on the last row of the last channel past the end of
            # the buffer.  The only term that varies along the contiguous axis is
            # the left edge, which is what keeps the load widenable.
            m1 = okh & mp[None, :]
            m0 = m1 & (ow >= 1)[None, :]
            if NOMASK:
                # Same slack argument as _conv_kernel, in int32 words: the left
                # edge and the ``ih == -1`` row both read into the slab's
                # padding.  This matters more here than anywhere else, because
                # ``ow >= 1`` is a mask term that varies along the *contiguous*
                # axis, which is exactly what stopped the pair load from widening.
                w0 = tl.where(m0, tl.load(row + (ow - 1)[None, :]), 0)
                w1 = tl.where(m1, tl.load(row + ow[None, :]), 0)
            else:
                w0 = tl.load(row + (ow - 1)[None, :], mask=m0, other=0)
                w1 = tl.load(row + ow[None, :], mask=m1, other=0)
            u0 = w0.to(tl.uint32, bitcast=True)
            u1 = w1.to(tl.uint32, bitcast=True)
            a0 = (u0 >> 16).to(tl.uint16).to(Y.dtype.element_ty, bitcast=True)
            a1 = (u1 & 0xFFFF).to(tl.uint16).to(Y.dtype.element_ty, bitcast=True)
            a2 = (u1 >> 16).to(tl.uint16).to(Y.dtype.element_ty, bitcast=True)
            for j in tl.static_range(3):
                wp = WGT + WOFF + oc[:, None] * K + ((i * 3 + j) * C + cc)[None, :]
                if EVEN_CO:
                    wv = tl.load(wp)
                else:
                    wv = tl.load(wp, mask=mc[:, None], other=0.0)
                if j == 0:
                    acc = tl.dot(wv, a0, acc=acc)
                elif j == 1:
                    acc = tl.dot(wv, a1, acc=acc)
                else:
                    acc = tl.dot(wv, a2, acc=acc)
        op = ohs * OW + ow
        if BOFF >= 0:
            if EVEN_CO:
                acc += tl.load(BIA + BOFF + oc)[:, None].to(tl.float32)
            else:
                acc += tl.load(BIA + BOFF + oc, mask=mc, other=0.0)[:, None].to(tl.float32)
        if ACT:
            acc *= tl.sigmoid(acc)
        om = mc[:, None] & mp[None, :]
        if HAS_RES:
            r = RES + ROFF + n * RSN + oc[:, None] * RSC + op[None, :]
            if EVEN_CO and EVEN_P:
                acc += tl.load(r).to(tl.float32)
            else:
                acc += tl.load(r, mask=om, other=0.0).to(tl.float32)
        o = Y + YOFF + n * YSN + oc[:, None] * YSC + op[None, :]
        if EVEN_CO and EVEN_P:
            tl.store(o, acc.to(Y.dtype.element_ty))
        else:
            tl.store(o, acc.to(Y.dtype.element_ty), mask=om)
        if PDL:
            _gdc_launch()


    @triton.jit
    def _fuse_kernel(XA, XB, WGT, BIA, Y,
                     CIN: tl.constexpr, KA: tl.constexpr,
                     IMH: tl.constexpr, IMW: tl.constexpr,
                     AOFF: tl.constexpr, ASN: tl.constexpr, ASC: tl.constexpr,
                     CMID: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
                     WOFFA: tl.constexpr, BOFFA: tl.constexpr,
                     CB: tl.constexpr, KB: tl.constexpr,
                     BOFF_: tl.constexpr, BSN: tl.constexpr, BSC: tl.constexpr,
                     COUT: tl.constexpr, P: tl.constexpr, OW: tl.constexpr,
                     WOFFB: tl.constexpr, BOFFB: tl.constexpr,
                     YOFF: tl.constexpr, YSN: tl.constexpr, YSC: tl.constexpr,
                     ACTA: tl.constexpr, ACTB: tl.constexpr,
                     BLOCK_P: tl.constexpr,
                     BKA: tl.constexpr, NKA: tl.constexpr, EKA: tl.constexpr,
                     BKB: tl.constexpr, NKB: tl.constexpr, EKB: tl.constexpr,
                     EVEN_P: tl.constexpr, PDL: tl.constexpr):
        """A C2f block's chain-out 3x3 and its cv2 1x1, as ONE kernel.

        This is the round's headline lever and the one the direction is built on:
        a 1x1 consumer needs no halo -- at output pixel p it reads only producer
        outputs at pixel p, over all channels -- so a program that owns a pixel
        tile and ALL ``CMID`` producer channels can do both convolutions with no
        barrier and no intermediate in HBM.

        The price is parallelism.  Phase A needs every producer channel in one
        program, so the consumer's output channels cannot be split either (a
        second co-tile would have to recompute phase A), and the grid loses its
        ``ceil(COUT/BLOCK_CO)`` dimension -- which at batch 1 is where half the
        CTAs came from.  That is the trade this kernel exists to measure.

            phase A   accA[CMID, BLOCK_P]  = silu(WA @ x[tile with halo] + bA)
            phase B   accB[COUT, BLOCK_P]  = WB[:, :2c] @ buf[:2c, tile]
                                           + WB[:, 2c:] @ accA
                                           (+ bB, silu)

        ``accA`` is already ``[k, pixels]``, the exact layout a dot's A operand
        wants, and it is narrowed to fp16 before phase B -- which is the same value
        the unfused pair would have round-tripped through HBM, so the result is
        bit-identical to the two-kernel form.
        """
        pid_p = tl.program_id(0)
        n = tl.program_id(1)
        op = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
        mp = op < P
        oh = op // OW
        ih0 = oh - KH // 2
        iw0 = (op - oh * OW) - KW // 2
        ocm = tl.arange(0, CMID)
        accA = tl.zeros((CMID, BLOCK_P), dtype=tl.float32)
        if PDL:
            _gdc_wait()
        xa = XA + AOFF + n * ASN
        for kb in tl.static_range(NKA):
            ok = kb * BKA + tl.arange(0, BKA)
            tap = ok // CIN
            cc = ok - tap * CIN
            ih = ih0[None, :] + (tap // KW)[:, None]
            iw = iw0[None, :] + (tap % KW)[:, None]
            am = (ih >= 0) & (ih < IMH) & (iw >= 0) & (iw < IMW)
            if not EKA:
                am = am & (ok < KA)[:, None]
            if not EVEN_P:
                am = am & mp[None, :]
            a = tl.load(xa + cc[:, None] * ASC + ih * IMW + iw, mask=am, other=0.0)
            wp = WGT + WOFFA + ocm[:, None] * KA + ok[None, :]
            if EKA:
                w = tl.load(wp)
            else:
                w = tl.load(wp, mask=(ok < KA)[None, :], other=0.0)
            accA = tl.dot(w, a, acc=accA)
        if BOFFA >= 0:
            accA += tl.load(BIA + BOFFA + ocm)[:, None].to(tl.float32)
        if ACTA:
            accA *= tl.sigmoid(accA)
        mid = accA.to(Y.dtype.element_ty)

        oc = tl.arange(0, COUT)
        accB = tl.zeros((COUT, BLOCK_P), dtype=tl.float32)
        xb = XB + BOFF_ + n * BSN + op[None, :]
        for kb in tl.static_range(NKB):
            ok = kb * BKB + tl.arange(0, BKB)
            if EKB and EVEN_P:
                b = tl.load(xb + ok[:, None] * BSC)
            else:
                bm = mp[None, :] if EKB else (
                    (ok < CB)[:, None] if EVEN_P
                    else (ok < CB)[:, None] & mp[None, :])
                b = tl.load(xb + ok[:, None] * BSC, mask=bm, other=0.0)
            wp = WGT + WOFFB + oc[:, None] * KB + ok[None, :]
            if EKB:
                w = tl.load(wp)
            else:
                w = tl.load(wp, mask=(ok < CB)[None, :], other=0.0)
            accB = tl.dot(w, b, acc=accB)
        w = tl.load(WGT + WOFFB + oc[:, None] * KB + (CB + ocm)[None, :])
        accB = tl.dot(w, mid, acc=accB)
        if BOFFB >= 0:
            accB += tl.load(BIA + BOFFB + oc)[:, None].to(tl.float32)
        if ACTB:
            accB *= tl.sigmoid(accB)
        o = Y + YOFF + n * YSN + oc[:, None] * YSC + op[None, :]
        if EVEN_P:
            tl.store(o, accB.to(Y.dtype.element_ty))
        else:
            tl.store(o, accB.to(Y.dtype.element_ty), mask=mp[None, :])
        if PDL:
            _gdc_launch()


    @triton.jit
    def _dw_kernel(X, WGT, BIA, RES, Y,
                   C: tl.constexpr, IMH: tl.constexpr, IMW: tl.constexpr,
                   XOFF: tl.constexpr, XSN: tl.constexpr, XSC: tl.constexpr,
                   P: tl.constexpr, OW: tl.constexpr,
                   KH: tl.constexpr, KW: tl.constexpr,
                   SH: tl.constexpr, SW: tl.constexpr,
                   PH: tl.constexpr, PW: tl.constexpr,
                   WOFF: tl.constexpr, BOFF: tl.constexpr,
                   YOFF: tl.constexpr, YSN: tl.constexpr, YSC: tl.constexpr,
                   ROFF: tl.constexpr, RSN: tl.constexpr, RSC: tl.constexpr,
                   ACT: tl.constexpr, HAS_RES: tl.constexpr,
                   BLOCK: tl.constexpr, EVEN_P: tl.constexpr,
                   NOMASK: tl.constexpr, PDL: tl.constexpr):
        """Depthwise k x k: one program owns a pixel tile of one (n, c) plane, so
        the k*k taps are a fully unrolled static nest over one resident plane and
        every tap load is contiguous along ow."""
        nc = tl.program_id(0)
        n = nc // C
        c = nc - n * C
        offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mp = offs < P
        oh = offs // OW
        ow = offs - oh * OW
        ih0 = oh * SH - PH
        iw0 = (ow * SW - PW)
        xb = X + XOFF + n * XSN + c * XSC
        wb = WGT + WOFF + c * (KH * KW)
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        if PDL:
            _gdc_wait()
        for i in tl.static_range(KH):
            ih = ih0 + i
            ok_h = (ih >= 0) & (ih < IMH)
            if not EVEN_P:
                ok_h = ok_h & mp
            row = xb + ih * IMW
            for j in tl.static_range(KW):
                iw = iw0 + j
                m = ok_h & (iw >= 0) & (iw < IMW)
                if NOMASK:
                    a = tl.where(m, tl.load(row + iw), 0.0)
                else:
                    a = tl.load(row + iw, mask=m, other=0.0)
                acc += a.to(tl.float32) * tl.load(wb + i * KW + j).to(tl.float32)
        if BOFF >= 0:
            acc += tl.load(BIA + BOFF + c).to(tl.float32)
        if ACT:
            acc *= tl.sigmoid(acc)
        if HAS_RES:
            r = RES + ROFF + n * RSN + c * RSC + offs
            if EVEN_P:
                acc += tl.load(r).to(tl.float32)
            else:
                acc += tl.load(r, mask=mp, other=0.0).to(tl.float32)
        o = Y + YOFF + n * YSN + c * YSC + offs
        if EVEN_P:
            tl.store(o, acc.to(Y.dtype.element_ty))
        else:
            tl.store(o, acc.to(Y.dtype.element_ty), mask=mp)
        if PDL:
            _gdc_launch()



# ---------------------------------------------------------------------------
# Tile shapes.  Ported from ``candidate/L2/yolov10_c2f.py``'s ``_dense_cfg`` /
# ``_dw_cfg``, which were swept over exactly these 18 dense convolutions and 4
# depthwise ones on this GPU; ``_fill_machine``'s occupancy cliff (narrow the
# pixel tile until the launch covers at least half the SMs, and only then) is
# the part that must not be dropped.
# ---------------------------------------------------------------------------
def _dense_cfg(p, cout, c, k, n, padded):
    """Tile rule for a dense stage NOT in the measured ``_TILE`` table.

    Ported verbatim from ``candidate/L2/yolov10_c2f.py``'s ``_dense_cfg_base`` plus
    its ``_fill_machine`` occupancy cliff (narrow the pixel tile until the launch
    covers at least half the SMs, and only then -- below about half a wave halving
    the pixel tile is worth 1.1-1.3x, at or above it the wider tile's reuse wins).
    It was swept on L2's shapes, not these, and this round's sweep shows it is
    wrong here in one specific way: it caps ``BLOCK_CO`` at 32, and the whole
    output-channel axis is what these stages want (1.7x on the cat1 1x1 at batch
    4).  It is kept as-is anyway, because the two *scored* shapes are covered by
    ``_TILE`` and a rule re-fit to two batches would be a rule fit to noise --
    what reaches this function is an unscored geometry, where an L2-swept rule is
    better evidence than an extrapolation from here.
    """
    pow2 = triton.next_power_of_2
    fat_m = p >= 4096
    if padded:
        if fat_m:
            cfg = (min(pow2(cout), 32), 64, min(pow2(k), 128), 4)
        else:
            cfg = (min(pow2(cout), 32), 32, min(pow2(k), 256), 4)
    elif fat_m:
        if cout >= 64:
            cfg = (min(pow2(cout), 64), 64, min(pow2(k), 256), 8)
        else:
            cfg = (min(pow2(cout), 32), 64, min(pow2(k), 128), 4)
    elif -(-p // 32) * -(-cout // 32) * n >= 2 * _num_sms():
        cfg = (min(pow2(cout), 32), 64, min(pow2(k), 128), 4)
    else:
        cfg = (min(pow2(cout), 32), 64, min(pow2(k), 256), 8)
    block_co, block_p, block_k, warps = cfg
    half = max(1, _num_sms() // 2)
    co_tiles = -(-cout // block_co) * n
    while block_p > 16 and -(-p // block_p) * co_tiles < half:
        block_p //= 2
        warps = 4
    return block_co, block_p, block_k, warps


_PDL_OK = None


def _pdl_ok():
    """One-time probe that this backend takes ``launch_pdl`` as a launch option.

    It is a field of the CUDA backend's options dataclass (consumed out of the
    call-site kwargs), not a parameter of ``JITFunction.run``, so that is what
    has to be checked -- and on a backend without it, passing the kwarg would
    raise rather than be ignored.
    """
    global _PDL_OK
    if _PDL_OK is None:
        try:
            import dataclasses
            from triton.backends.nvidia.compiler import CUDAOptions
            _PDL_OK = any(f.name == "launch_pdl"
                          for f in dataclasses.fields(CUDAOptions))
        except Exception:  # noqa: BLE001 - not the CUDA backend
            _PDL_OK = False
    return bool(_PDL_OK)


# ``num_stages`` was never varied in r1's sweep.  Every k loop here is a
# ``tl.static_range``, so it is fully unrolled and the software pipeliner has no
# loop to stage -- the expectation is that this does nothing, which is exactly why
# it is one line and worth one A/B rather than an argument.
_NS = int(os.environ.get("FK_NECK_NS", "1"))


def _cfg(warps, pdl):
    cfg = {"num_warps": warps, "num_stages": _NS}
    if pdl:
        cfg["launch_pdl"] = True
    return cfg


def _dw_cfg(p):
    block = min(512, triton.next_power_of_2(p))
    return block, max(1, min(8, block // 64))


# (C, OH, OW, KH, SH, batch) -> (BLOCK, num_warps), measured; see _TILE.  The
# depthwise landscape is flat to within 0.2-0.6 us (the kernel is bound by issuing
# k*k masked gathers over one resident plane, not by the tile), so these are small
# corrections, not a cliff.
_DW_TILE: dict = {
    (128, 20, 20, 3, 1, 4): (512, 8),
    (128, 20, 20, 3, 2, 1): (256, 8),
    (128, 20, 20, 3, 2, 4): (512, 8),
    (256, 20, 20, 7, 1, 1): (256, 4),
    (256, 20, 20, 7, 1, 4): (512, 4),
}




# ---------------------------------------------------------------------------
# The plan: fold every BatchNorm once, pack the weights, lay out one workspace
# and build the 22-stage table that all three forward paths execute.
#
# Workspace layout.  One flat fp16 slab holds every internal activation, so a
# stage addresses its source, residual and destination as a constant element
# offset plus (batch, channel) strides -- which is what makes cat3 free and lets
# a block's cv2 read a contiguous concat buffer nothing ever copied into:
#
#   W40a  [N, 3*c_p4, 40, 40]   c2f_p4's concat buffer
#   X3    [N, 64+128, 40, 40]   c2f_n4's input: [0,64) down_p3, [64,192) p4
#   W80a  [N, 3*c_p3, 80, 80]   c2f_p3's concat buffer
#   W40b  [N, 3*c_n4, 40, 40]   c2f_n4's concat buffer
#   D2A   [N, 128, 40, 40]      down_n4's 1x1 output
#   D2B   [N, 128, 20, 20]      down_n4's strided depthwise output
#   W20a  [N, 3*c_n5, 20, 20]   c2fcib_n5's concat buffer
#   plus two ping-pong scratch slabs per resolution for block-internal chains
#
# ``p4`` is written straight into X3[:, 64:192] by ``c2f_p4.cv2`` and read from
# there twice (the cat2 upsample gather and cat3), which is the whole of cat3.
# ---------------------------------------------------------------------------
_G_GEMM, _G_CONV, _G_DW, _G_FUSE = 0, 1, 2, 3

# Buffer kinds.  "ws" is the workspace slab; the rest are per-call tensors
# (the three inputs and the three returned outputs).
_WS = "ws"


class _Buf:
    """[N, C, H, W] living at a fixed element offset inside a named tensor."""

    __slots__ = ("kind", "off", "c", "h", "w")

    def __init__(self, kind, off, c, h, w):
        self.kind = kind
        self.off = off
        self.c = int(c)
        self.h = int(h)
        self.w = int(w)

    @property
    def sn(self):
        return self.c * self.h * self.w

    @property
    def sc(self):
        return self.h * self.w

    def at(self, c0):
        return self.off + int(c0) * self.h * self.w


def _align8(v):
    return (v + 7) & ~7


def _cin_of(step):
    """Input channels a folded conv consumes (depthwise: its own channel count)."""
    if step.groups > 1:
        return int(step.w.shape[0])
    return int(step.w.shape[1])


def _kind(step):
    """(body, kh, kw, sh, sw) for a folded conv, or None if unsupported."""
    cout, cin_per_g = int(step.w.shape[0]), int(step.w.shape[1])
    kh, kw = int(step.w.shape[2]), int(step.w.shape[3])
    sh, sw = int(step.stride[0]), int(step.stride[1])
    ph, pw = int(step.padding[0]), int(step.padding[1])
    if kh % 2 == 0 or kw % 2 == 0 or (ph, pw) != (kh // 2, kw // 2):
        return None
    if step.groups == cout and cin_per_g == 1:
        return _G_DW, kh, kw, sh, sw
    if step.groups != 1:
        return None
    if (kh, kw) == (1, 1) and (sh, sw) == (1, 1):
        return _G_GEMM, 1, 1, 1, 1
    return _G_CONV, kh, kw, sh, sw


def _dense_weight(w):
    """[COUT, C, KH, KW] -> [COUT, KH*KW*C] with the channel axis innermost, so a
    k tile is contiguous along the axis the implicit GEMM blocks over."""
    return w.permute(0, 2, 3, 1).reshape(w.shape[0], -1).contiguous()


def _c2f_parts(block):
    """``(cv1, chain, add, cv2)`` of folded convs for a C2f/C2fCIB with n == 1."""
    if len(block.m) != 1:
        return None
    m = block.m[0]
    seq = getattr(m, "cv1", None)
    if isinstance(seq, nn.Sequential):
        inner = list(seq)                      # YOLOCIB
    elif seq is not None and hasattr(m, "cv2"):
        inner = [seq, m.cv2]                   # YOLOBottleneck
    else:
        return None
    return block.cv1, inner, bool(m.add), block.cv2


class _Plan:
    __slots__ = ("ok", "sig", "src", "wgt", "bia", "wb", "scr", "stages",
                 "ws_elems", "dtype", "device", "n", "outs", "graph", "work")

    def __init__(self, dtype=None, device=None):
        self.ok = False
        self.sig = ()
        self.src = ()
        self.dtype = dtype
        self.device = device
        self.graph = {}
        self.wb = None
        self.scr = None


def _build_plan(neck, shapes, dtype, device):
    """Fold, pack and lay out the whole neck for one (batch, dtype, device)."""
    plan = _Plan(dtype, device)
    src = []
    if _conv_step is None or triton is None:
        return plan
    try:
        parts = [_c2f_parts(b) for b in (neck.c2f_p4, neck.c2f_p3, neck.c2f_n4,
                                         neck.c2fcib_n5)]
        if any(p is None for p in parts):
            return plan
        folded = []
        for cv1, inner, add, cv2 in parts:
            chain = []
            for s in inner:
                if isinstance(s, YOLOConv):
                    chain.append(_conv_step(s, src))
                elif hasattr(s, "conv") and hasattr(s, "conv1"):
                    chain.append(_repvgg_step(s, src))       # YOLORepVGGDW
                else:
                    return plan
            folded.append((_conv_step(cv1, src), chain, add,
                           _conv_step(cv2, src)))
        dp3 = _conv_step(neck.down_p3, src)
        dn4a = _conv_step(neck.down_n4.cv1, src)
        dn4b = _conv_step(neck.down_n4.cv2, src)
    except Exception:  # noqa: BLE001 - an edited / unexpected module tree
        return plan
    finally:
        plan.src = tuple(src)
        plan.sig = tuple((t.data_ptr(), t._version) for t in src)

    allsteps = [dp3, dn4a, dn4b]
    for cv1, chain, _add, cv2 in folded:
        allsteps += [cv1, cv2, *chain]
    for s in allsteps:
        if s.w.dtype != dtype or not s.w.is_cuda or _kind(s) is None:
            return plan
        if s.b is not None and s.b.dtype != dtype:
            return plan

    (bn, c3, h3, w3), (_b4, c4, h4, w4), (_b5, c5, h5, w5) = shapes
    if not (h3 == 2 * h4 == 4 * h5 and w3 == 2 * w4 == 4 * w5):
        return plan
    if h5 % 2 or w5 % 2 or w5 < 8:
        return plan
    plan.n = bn

    (p4cv1, p4ch, p4add, p4cv2) = folded[0]
    (p3cv1, p3ch, p3add, p3cv2) = folded[1]
    (n4cv1, n4ch, n4add, n4cv2) = folded[2]
    (n5cv1, n5ch, n5add, n5cv2) = folded[3]

    def cout(s):
        return int(s.w.shape[0])

    c_p4, c_p3 = cout(p4cv1) // 2, cout(p3cv1) // 2
    c_n4, c_n5 = cout(n4cv1) // 2, cout(n5cv1) // 2
    # Every block must be the standard C2f split: cv1 -> 2c, chain ends at c,
    # cv2 consumes (2 + 1) * c.  Otherwise the shared concat buffer is wrong.
    for (cv1, chain, _a, cv2), c in ((folded[0], c_p4), (folded[1], c_p3),
                                     (folded[2], c_n4), (folded[3], c_n5)):
        if cout(cv1) != 2 * c or cout(chain[-1]) != c or _cin_of(cv2) != 3 * c:
            return plan
        if _cin_of(chain[0]) != c:
            return plan
    if _cin_of(p4cv1) != c5 + c4 or _cin_of(p3cv1) != cout(p4cv2) + c3:
        return plan
    if _cin_of(n5cv1) != cout(dn4b) + c5:
        return plan
    if _cin_of(dp3) != cout(p3cv2) or _cin_of(n4cv1) != cout(dp3) + cout(p4cv2):
        return plan
    if _cin_of(dn4a) != cout(n4cv2) or _cin_of(dn4b) != cout(dn4a):
        return plan

    # -- workspace regions --------------------------------------------------
    # The slab opens and closes with _SLACK unused elements so a haloed load can
    # read past a plane -- in either direction, and past the last region -- and
    # still be a legal access, which is what lets those loads drop their mask.
    off = [_SLACK]

    def region(c, h, w):
        b = _Buf(_WS, off[0], c, h, w)
        off[0] = _align8(off[0] + bn * c * h * w)
        return b

    w40a = region(3 * c_p4, h4, w4)
    x3 = region(cout(dp3) + cout(p4cv2), h4, w4)
    w80a = region(3 * c_p3, h3, w3)
    w40b = region(3 * c_n4, h4, w4)
    d2a = region(cout(dn4a), h4, w4)
    d2b = region(cout(dn4b), h5, w5)
    w20a = region(3 * c_n5, h5, w5)
    scratch = {}
    for h, w, chains in ((h4, w4, (p4ch, n4ch)), (h3, w3, (p3ch,)),
                         (h5, w5, (n5ch,))):
        wide = max(cout(s) for ch in chains for s in ch[:-1])
        scratch[h] = (region(wide, h, w), region(wide, h, w))
    plan.ws_elems = off[0] + _SLACK

    in3 = _Buf("p3_backbone", 0, c3, h3, w3)
    in4 = _Buf("p4_backbone", 0, c4, h4, w4)
    in5 = _Buf("p5_backbone", 0, c5, h5, w5)
    out3 = _Buf("p3", 0, cout(p3cv2), h3, w3)
    out4 = _Buf("n4", 0, cout(n4cv2), h4, w4)
    out5 = _Buf("n5", 0, cout(n5cv2), h5, w5)

    # -- packed weight / bias buffers --------------------------------------
    wparts, bparts, cur = [], [], [0, 0]

    def take(parts, which, t):
        """Append at a 16-byte-aligned offset, so the constexpr offset the kernel
        adds keeps Triton's widened loads aligned."""
        o = _align8(cur[which])
        if o > cur[which]:
            parts.append(torch.zeros(o - cur[which], dtype=t.dtype,
                                     device=t.device))
        parts.append(t.reshape(-1))
        cur[which] = o + t.numel()
        return o

    stages = []

    bad = []

    def emit(step, srcs, dst, res=None, desc=""):
        """One stage.  ``srcs`` = [(buf, c0, up)], ``dst``/``res`` = (buf, c0)."""
        g, kh, kw, sh, sw = _kind(step)
        co = cout(step)
        dbuf, dc0 = dst
        # Geometry the kernels assume, checked rather than trusted: the
        # destination slab really is this conv's output size, an upsampled source
        # really is half the output resolution in both axes, and the source slice
        # really holds the channels the K axis will read.
        sbuf = srcs[0][0]
        if (dbuf.h != (sbuf.h + 2 * (kh // 2) - kh) // sh + 1
                or dbuf.w != (sbuf.w + 2 * (kw // 2) - kw) // sw + 1):
            if not (srcs[0][2] and dbuf.h == sbuf.h * 2 and dbuf.w == sbuf.w * 2):
                bad.append(desc)
        for b, c0, up in srcs:
            if up and not (dbuf.h == 2 * b.h and dbuf.w == 2 * b.w):
                bad.append(desc)
        if dc0 + co > dbuf.c or (res is not None and res[1] + co > res[0].c):
            bad.append(desc)
        st = {
            "g": g, "kh": kh, "kw": kw, "sh": sh, "sw": sw, "cout": co,
            "act": bool(step.act), "desc": desc,
            "oh": dbuf.h, "ow": dbuf.w, "p": dbuf.h * dbuf.w,
            "y": (dbuf.kind, dbuf.at(dc0), dbuf.sn, dbuf.sc),
            "boff": -1 if step.b is None else take(bparts, 1, step.b),
            "res": None if res is None else
                   (res[0].kind, res[0].at(res[1]), res[0].sn, res[0].sc),
        }
        if g == _G_DW:
            st["w"] = take(wparts, 0, step.w.reshape(co, kh * kw).contiguous())
        elif g == _G_GEMM:
            st["w"] = take(wparts, 0, step.w.reshape(co, -1).contiguous())
        else:
            st["w"] = take(wparts, 0, _dense_weight(step.w))
        cins = _split_cin(step, srcs)
        for (b, c0, _u), cin in zip(srcs, cins):
            if c0 + cin > b.c:
                bad.append(desc)
        if sum(cins) != (co if g == _G_DW else int(step.w.shape[1])):
            bad.append(desc)
        st["src"] = [(b.kind, b.at(c0), b.sn, b.sc, b.h, b.w, int(up), cin)
                     for (b, c0, up), cin in zip(srcs, cins)]
        stages.append(st)

    def _split_cin(step, srcs):
        """Channels each source contributes to the conv's K axis."""
        if len(srcs) == 1:
            return [_cin_of(step)]
        return [b.c - c0 if c0 else b.c for (b, c0, _u) in srcs]

    def do_block(cv1, chain, add, cv2, srcs, buf, c, hh, out):
        sa, sb = scratch[hh]
        emit(cv1, srcs, (buf, 0), desc="cv1")
        cur_src = (buf, c, 0)
        for t, s in enumerate(chain[:-1]):
            tgt = sa if t % 2 == 0 else sb
            emit(s, [cur_src], (tgt, 0), desc=f"chain{t}")
            cur_src = (tgt, 0, 0)
        emit(chain[-1], [cur_src], (buf, 2 * c),
             res=(buf, c) if add else None, desc="chain-out")
        emit(cv2, [(buf, 0, 0)], out, desc="cv2")

    #  cat1 = cat(up2x(p5_backbone), p4_backbone) -> c2f_p4 -> p4 == X3[:,64:]
    do_block(p4cv1, p4ch, p4add, p4cv2, [(in5, 0, 1), (in4, 0, 0)],
             w40a, c_p4, h4, (x3, cout(dp3)))
    #  cat2 = cat(up2x(p4), p3_backbone) -> c2f_p3 -> p3 (returned)
    do_block(p3cv1, p3ch, p3add, p3cv2,
             [(x3, cout(dp3), 1), (in3, 0, 0)], w80a, c_p3, h3, (out3, 0))
    #  down_p3(p3) -> X3[:, :64];  cat3 is now nothing at all
    emit(dp3, [(out3, 0, 0)], (x3, 0), desc="down_p3")
    do_block(n4cv1, n4ch, n4add, n4cv2, [(x3, 0, 0)], w40b, c_n4, h4, (out4, 0))
    #  down_n4 = SCDown: 1x1 then strided depthwise
    emit(dn4a, [(out4, 0, 0)], (d2a, 0), desc="down_n4.cv1")
    emit(dn4b, [(d2a, 0, 0)], (d2b, 0), desc="down_n4.cv2")
    #  cat4 = cat(down_n4(n4), p5_backbone) -> c2fcib_n5 -> n5 (returned)
    do_block(n5cv1, n5ch, n5add, n5cv2, [(d2b, 0, 0), (in5, 0, 0)],
             w20a, c_n5, h5, (out5, 0))

    if len(stages) != 22 or bad:
        return plan
    if _USE_FUSE:
        stages = _fuse_pairs(stages)
    # Weights and biases share ONE allocation (they stay separate *views*, so
    # every stage's constexpr offsets are unchanged) purely so the L2 prefetch is
    # a single contiguous read covering both.  ``_align8`` keeps the bias view's
    # base 16-byte aligned.
    wcat = torch.cat([t.to(dtype) for t in wparts])
    bcat = (torch.cat([t.to(dtype) for t in bparts]) if bparts
            else torch.zeros(8, dtype=dtype, device=device))
    nw = _align8(wcat.numel())
    both = torch.zeros(nw + bcat.numel(), dtype=dtype, device=device)
    both[:wcat.numel()] = wcat
    both[nw:] = bcat
    plan.wb = both
    plan.wgt = both[:nw]
    plan.bia = both[nw:]
    plan.outs = (out3, out4, out5)
    plan.scr = torch.empty(_align8(-(-plan.wb.numel() // 4096)) + 8,
                           dtype=torch.float32, device=device)
    if not _finish_stages(stages, bn):
        return plan
    plan.stages = tuple(stages)
    plan.work = sum(s["work"] for s in stages)
    plan.ok = True
    return plan


def _fusable(a, b):
    """Is ``a`` a C2f chain-out 3x3 whose only consumer is the cv2 1x1 ``b``?

    Everything ``_fuse_kernel`` assumes, checked: ``a`` is a stride-1
    shape-preserving dense conv with no residual whose whole output is ``b``'s
    last channel block, ``b`` is a single-source 1x1 with no residual and no
    upsample, and both channel counts are powers of two (they index a program's
    full ``tl.arange``).  ``a``'s destination has to be exactly the top third of
    ``b``'s source, because that is the concat layout the split relies on.
    """
    if a["g"] != _G_CONV or b["g"] != _G_GEMM or a["res"] or b["res"]:
        return False
    if (a["sh"], a["sw"]) != (1, 1) or len(b["src"]) != 1:
        return False
    src = a["src"][0]
    if src[4] != a["oh"] or src[5] != a["ow"]:
        return False
    bs = b["src"][0]
    if bs[6] or bs[4] != b["oh"] or bs[5] != b["ow"]:
        return False
    cmid, cb = a["cout"], bs[7] - a["cout"]
    if cb <= 0 or bs[7] != 3 * cmid or cb != 2 * cmid:
        return False
    for v in (cmid, b["cout"], cb):
        if v < 16 or v & (v - 1):
            return False
    # ``a`` writes b's source at channel offset ``cb`` (same buffer, same plane).
    return (a["y"][0] == bs[0] and a["y"][1] == bs[1] + cb * bs[3]
            and a["y"][2] == bs[2] and a["y"][3] == bs[3])


def _fuse_pairs(stages):
    """Collapse every fusable (chain-out, cv2) pair into one _G_FUSE stage."""
    out, i = [], 0
    while i < len(stages):
        a = stages[i]
        b = stages[i + 1] if i + 1 < len(stages) else None
        if b is not None and _fusable(a, b):
            out.append({
                "g": _G_FUSE, "desc": a["desc"] + "+" + b["desc"],
                "a": a, "b": b,
                "cout": b["cout"], "oh": b["oh"], "ow": b["ow"], "p": b["p"],
                "kh": a["kh"], "kw": a["kw"], "sh": 1, "sw": 1,
                "act": b["act"], "res": None, "y": b["y"], "src": a["src"],
            })
            i += 2
        else:
            out.append(a)
            i += 1
    return out


def _rowt_needed(st):
    """True when this stage's tap address is not affine in the flat pixel index.

    A stride-1 shape-preserving conv reads ``op + (i-PH)*IMW + (j-PW)`` -- affine,
    so a flat pixel tile is contiguous however it straddles rows.  A stride-2 conv
    reads ``2*op + 2*oh*OW + const`` and an upsample gather reads
    ``(oh>>1)*Ws + (ow>>1)``; neither is, so a tile that crosses an output row
    boundary scatters the A-tile load.  Those stages get row-confined tiles.
    """
    if _ROWT_MODE in ("0", "1"):
        return _ROWT_MODE == "1"
    if st["g"] == _G_GEMM:
        return any(s[6] for s in st["src"])
    if st["g"] == _G_CONV:
        return (st["sh"], st["sw"]) != (1, 1) or st["src"][0][5] != st["ow"]
    return False


def _apply_tile(st, n, bco, bp, bk, warps, rowt, pdl):
    """Fill ``consts`` / ``grid`` / ``cfg`` / ``work`` for one dense stage.

    Shared by plan construction and ``tools/stages.py``'s sweep, so a swept
    config and a shipped config are computed by the same code.
    """
    pow2 = triton.next_power_of_2
    p, co, ow, oh = st["p"], st["cout"], st["ow"], st["oh"]
    tpr = -(-ow // bp) if rowt else 0
    even_p = (ow % bp == 0) if rowt else (p % bp == 0)
    grid_p = tpr * oh if rowt else -(-p // bp)
    common = dict(BLOCK_CO=bco, BLOCK_P=bp, EVEN_CO=(co % bco == 0),
                  EVEN_P=even_p, ROWT=rowt, TPR=tpr, PDL=pdl)
    if st["g"] != _G_GEMM:
        # A row-confined tile overruns in columns (``ow`` up to TPR*BLOCK_P-1);
        # a flat tile overruns in whole rows and its ``ow`` is always in range.
        oh_max, ow_max = ((st["oh"] - 1, tpr * bp - 1) if rowt
                          else ((-(-p // bp) * bp - 1) // st["ow"], st["ow"] - 1))
        common["NOMASK"] = _nomask_ok(
            st, *_reach(st, oh_max, ow_max, st["sh"], st["sw"],
                        st["kh"], st["kw"]))
    if st["g"] == _G_GEMM:
        c0 = st["src"][0][7]
        c1 = st["src"][1][7] if len(st["src"]) > 1 else 0
        bk0 = min(bk, pow2(c0))
        bk1 = min(bk, pow2(c1)) if c1 else 16
        if min(bco, bp, bk0, bk1) < 16:
            return False
        common.update(BK0=bk0, NK0=-(-c0 // bk0), EK0=(c0 % bk0 == 0),
                      BK1=bk1, NK1=(-(-c1 // bk1) if c1 else 0),
                      EK1=(c1 % bk1 == 0 if c1 else True))
        k = st["k"]
    else:
        k = st["k"]
        bk = min(bk, pow2(k))
        if min(bco, bp, bk) < 16:
            return False
        common.update(BLOCK_K=bk, NUM_K=-(-k // bk), EVEN_K=(k % bk == 0))
    st["consts"] = common
    st["grid"] = (grid_p, -(-co // bco), n)
    st["cfg"] = _cfg(warps, pdl)
    st["work"] = grid_p * st["grid"][1] * n * bco * bp * k
    return True


def _apply_fuse(st, n, pdl):
    """Tile a fused stage.  The grid has NO output-channel dimension -- phase A
    needs every producer channel resident -- so the only knob is the pixel tile,
    and the whole trade this kernel measures is that lost dimension against a
    round trip through HBM and one kernel's fixed cost."""
    a, b = st["a"], st["b"]
    cmid = a["cout"]
    ka = a["src"][0][7] * a["kh"] * a["kw"]
    cb = b["src"][0][7] - cmid
    p = st["p"]
    bp = _FUSE_TILE.get((st["cout"], cmid, st["oh"], st["ow"], n))
    if bp is None:
        # Pick the widest pixel tile that still fills the machine, since the grid
        # is one-dimensional now: ceil(P/BLOCK_P)*n CTAs, nothing else.
        bp = 128
        while bp > 16 and -(-p // bp) * n < _num_sms():
            bp //= 2
    bka = min(256, triton.next_power_of_2(ka))
    bkb = min(256, triton.next_power_of_2(cb))
    if min(cmid, st["cout"], bp, bka, bkb) < 16:
        return False
    st["consts"] = dict(
        BLOCK_P=bp, BKA=bka, NKA=-(-ka // bka), EKA=(ka % bka == 0),
        BKB=bkb, NKB=-(-cb // bkb), EKB=(cb % bkb == 0),
        EVEN_P=(p % bp == 0), PDL=pdl)
    st["grid"] = (-(-p // bp), n)
    st["cfg"] = _cfg(8, pdl)
    st["work"] = st["grid"][0] * n * bp * (cmid * ka + st["cout"] * 3 * cmid)
    st["ka"], st["kb"], st["cb"], st["cmid"] = ka, 3 * cmid, cb, cmid
    return True


# (COUT, CMID, OH, OW, batch) -> BLOCK_P for _fuse_kernel.
_FUSE_TILE: dict = {}


def _pdl_on():
    return _HAS_PDL and _USE_PDL and _pdl_ok()


def _finish_stages(stages, n):
    """Attach tile shape, grid, launch config and the constexpr arguments.

    Everything a launch needs is resolved here, once per plan, so the steady
    state is ``kernel[grid](*tensors, *consts)`` with no per-call arithmetic.
    """
    pdl = _pdl_on()
    for st in stages:
        p, co = st["p"], st["cout"]
        if st["g"] == _G_FUSE:
            if not _apply_fuse(st, n, pdl):
                return False
            continue
        if st["g"] == _G_DW:
            block, warps = _DW_TILE.get(
                (st["cout"], st["oh"], st["ow"], st["kh"], st["sh"], n),
                _dw_cfg(p))
            st["grid"] = (n * co, -(-p // block))
            st["cfg"] = _cfg(warps, pdl)
            st["consts"] = dict(
                BLOCK=block, EVEN_P=(p % block == 0), PDL=pdl,
                NOMASK=_nomask_ok(st, *_reach(
                    st, (-(-p // block) * block - 1) // st["ow"], st["ow"] - 1,
                    st["sh"], st["sw"], st["kh"], st["kw"])))
            st["work"] = n * co * -(-p // block) * block * st["kh"] * st["kw"]
            continue
        if st["g"] == _G_GEMM:
            st["k"] = sum(s[7] for s in st["src"])
            c = st["k"]
            padded = False
        else:
            c = st["src"][0][7]
            st["k"] = c * st["kh"] * st["kw"]
            padded = True
        if _upf_ok(st):
            ent = _UPF_TILE.get((co, st["src"][0][7], st["src"][1][7],
                                 st["src"][0][4], st["src"][0][5], n))
            if ent is None:
                ent = _upf_cfg(st["src"][0][4], st["src"][0][5], co,
                               st["src"][0][7], st["src"][1][7], n)
            if ent is not None and _apply_upf(st, n, *ent, pdl):
                continue
        if _s1_ok(st):
            ent = _S1_TILE.get((co, c, st["oh"], st["ow"], st["kh"], n))
            if ent is None:
                bco, bp, _bk, warps, rowt = _tile_for(st, p, co, c, st["k"], n,
                                                      padded)
                ent = (bco, bp, min(c, 128), warps, rowt)
            if _apply_s1(st, n, *ent, pdl):
                continue
        if _s2_ok(st):
            bco, bp, warps = _S2_TILE.get(
                (co, c, st["oh"], st["ow"], n),
                (min(triton.next_power_of_2(co), 64), 32, 8))
            st["s2"] = True
            tpr = -(-st["ow"] // bp)
            # ``_conv_s2_kernel`` addresses int32 WORDS: the ``ih == -1`` row
            # reaches back IMW/2 + 1 words (== IMW + 2 halves) and a trailing
            # column tile runs up to ``bp`` words (2*bp halves) past the plane,
            # so the pixel index does not have to divide the row for the unmasked
            # form to be legal -- unlike the generic conv, which is why the bound
            # is passed as ``extra`` and ``even_p`` is forced True here.
            st["consts"] = dict(BLOCK_CO=bco, BLOCK_P=bp, TPR=tpr,
                                EVEN_CO=(co % bco == 0),
                                EVEN_P=(st["ow"] % bp == 0), PDL=pdl,
                                NOMASK=_nomask_ok(
                                    st, st["src"][0][5] + 2, 2 * (bp + 1)))
            st["grid"] = (tpr * st["oh"], -(-co // bco), n)
            st["cfg"] = _cfg(warps, pdl)
            st["work"] = st["grid"][0] * st["grid"][1] * n * bco * bp * st["k"]
            continue
        bco, bp, bk, warps, rowt = _tile_for(st, p, co, c, st["k"], n, padded)
        if not _apply_tile(st, n, bco, bp, bk, warps, rowt, pdl):
            return False
    return True


# Buffer kinds allocated with _SLACK padding at both ends.  The three *inputs*
# are the harness' own tensors and carry no padding -- which is fine, because
# every stage that reads an input reads it with a 1x1, and a 1x1 has no halo.
_SLACK_KINDS = frozenset((_WS, "p3", "n4", "n5"))


def _nomask_ok(st, lo, hi):
    """May this stage read its halo taps unmasked?

    *lo* / *hi* are how far, in elements, the A-tile load may reach before the
    source's first element and past its last.  Reading into a neighbouring region
    of the same slab is harmless (the value is zeroed in registers afterwards);
    reading outside the slab is not, so the reach has to fit in the padding.
    Computed from the chosen tile, never assumed.
    """
    if not _USE_NOMASK:
        return False
    kind, _off, _sn, _sc, _h, _w, up, _c = st["src"][0]
    if up or kind not in _SLACK_KINDS:
        return False
    return max(lo, hi) <= _SLACK


def _reach(st, oh_max, ow_max, sh, sw, kh, kw):
    """(lo, hi) for a ``kh x kw`` tap nest whose lanes carry output coordinates up
    to (*oh_max*, *ow_max*).

    Those bounds are past the real plane whenever the pixel tile does not divide
    it -- a flat tile overruns in whole rows, a row-confined tile in columns --
    and that overrun is the part that is easy to get wrong, so both are passed in
    from the tile rather than re-derived here.
    """
    _k, _off, _sn, sc, _imh, imw, _up, c = st["src"][0]
    ph, pw = kh // 2, kw // 2
    ih_max = oh_max * sh + (kh - 1 - ph)
    iw_max = ow_max * sw + (kw - 1 - pw)
    hi = (c - 1) * sc + ih_max * imw + iw_max - (c * sc - 1)
    return ph * imw + pw, max(0, hi)


def _s1_ok(st):
    """Can this stage use the tap-major stride-1 body?

    Needs the property the whole trick rests on -- the tap address is the pixel
    base plus a compile-time scalar -- which holds exactly when the convolution is
    stride 1 and preserves its resolution, so ``oh*OW + ow == op`` and
    ``ih*IMW + iw == op + (i-PH)*IMW + (j-PW)``.  Also needs one tap's channel
    count to be a legal ``tl.dot`` k extent on its own.
    """
    if not (_USE_S1 and st["g"] == _G_CONV):
        return False
    if (st["sh"], st["sw"]) != (1, 1):
        return False
    _k, _off, _sn, _sc, imh, imw, up, c = st["src"][0]
    if up or imh != st["oh"] or imw != st["ow"]:
        return False
    if c < 16 or c > 256 or (c & (c - 1)):
        return False
    return st["k"] == c * st["kh"] * st["kw"]


def _apply_s1(st, n, bco, bp, bkc, warps, rowt, pdl):
    """Fill ``consts`` / ``grid`` / ``cfg`` / ``work`` for a tap-major stage."""
    c = st["src"][0][7]
    bkc = min(bkc, c)
    if min(bco, bp, bkc) < 16:
        return False
    p, co, ow, oh = st["p"], st["cout"], st["ow"], st["oh"]
    tpr = -(-ow // bp) if rowt else 0
    even_p = (ow % bp == 0) if rowt else (p % bp == 0)
    grid_p = tpr * oh if rowt else -(-p // bp)
    st["s1"] = True
    st["consts"] = dict(BLOCK_CO=bco, BLOCK_P=bp, BKC=bkc, NKC=-(-c // bkc),
                        EKC=(c % bkc == 0), EVEN_CO=(co % bco == 0),
                        EVEN_P=even_p, ROWT=rowt, TPR=tpr, PDL=pdl)
    st["grid"] = (grid_p, -(-co // bco), n)
    st["cfg"] = _cfg(warps, pdl)
    st["work"] = grid_p * st["grid"][1] * n * bco * bp * st["k"]
    return True


# (COUT, C, OH, OW, KH, batch) -> (BLOCK_CO, BLOCK_P, BKC, warps, ROWT).
_S1_TILE: dict = {}


def _s2_ok(st):
    """Can this stage use the int32-pair stride-2 body?

    Everything the trick needs, checked rather than assumed: a 3x3 stride-2 pad-1
    dense conv, a power-of-two channel count that is a legal ``tl.dot`` K axis in
    one block, and a source whose element offset, batch stride, channel stride and
    row width are all even, so the fp16 buffer can be viewed as int32 pairs at all.
    """
    if not (_USE_S2 and st["g"] == _G_CONV):
        return False
    if (st["kh"], st["kw"], st["sh"], st["sw"]) != (3, 3, 2, 2):
        return False
    _k, off, sn, sc, imh, imw, up, c = st["src"][0]
    if up or not (16 <= c <= 256) or (c & (c - 1)):
        return False
    if imw % 2 or off % 2 or sn % 2 or sc % 2:
        return False
    return imh == 2 * st["oh"] and imw == 2 * st["ow"]


# (COUT, C, OH, OW, batch) -> (BLOCK_CO, BLOCK_P, num_warps) for _conv_s2_kernel.
# ``tools/sweep.py``: BLOCK_P 32 -> 16 is worth 0.84 us at B=1 and 0.25 at B=4 on
# ``down_p3``, the neck's single most expensive stage.  16 does not divide OW=40
# either, but a narrower column tile means fewer wasted lanes past the row.
_S2_TILE: dict = {
    (64, 64, 40, 40, 1): (64, 16, 8),
    (64, 64, 40, 40, 4): (64, 16, 8),
}


def _upf_ok(st):
    """Can this stage move its upsampled source's dot to the source resolution?

    Needs exactly the shape ``_gemm_up_kernel`` assumes: a stride-1 1x1 with two
    sources, the first read through the 2x nearest gather and the second at the
    output resolution, and an output that is exactly twice the first source in
    both axes (so the ``dy``/``dx`` loop covers it with no leftover row).
    """
    if not (_USE_UPF and st["g"] == _G_GEMM):
        return False
    src = st["src"]
    if len(src) != 2 or not src[0][6] or src[1][6]:
        return False
    _k0, _o0, _sn0, _sc0, h0, w0, _u0, _c0 = src[0]
    if 2 * h0 != st["oh"] or 2 * w0 != st["ow"]:
        return False
    return src[1][4] == st["oh"] and src[1][5] == st["ow"]


def _upf_cfg(sh0, sw0, co, c0, c1, n):
    """(BLOCK_CO, RS, CS, BK0, BK1, warps) for a source tile of ``RS x CS``.

    ``RS * CS`` is the phase-1 dot's N, so it must be at least 16; the source
    resolutions here are 20x20 and 40x40, and 20 is not a power of two, which is
    why the tile is two-dimensional (4x4 divides 20x20 exactly) rather than a
    row segment.  Whatever divides both axes exactly is preferred -- a masked
    lane in phase 1 costs a masked *pair* in phase 2.
    """
    best = None
    for rs in (1, 2, 4, 8):
        for cs in (2, 4, 8, 16, 32):
            if rs * cs < 16 or rs * cs > 64 or 2 * cs < 8:
                continue
            if sh0 % rs or sw0 % cs:
                continue
            grid_p = (sh0 // rs) * (sw0 // cs)
            # Prefer the tile whose launch covers the machine, then the widest
            # contiguous output run (2*CS halves).
            score = (min(grid_p * n * max(1, co // 32), 4 * _num_sms()), cs)
            if best is None or score > best[0]:
                best = (score, rs, cs)
    if best is None:
        return None
    _sc, rs, cs = best
    pow2 = triton.next_power_of_2
    bco = min(pow2(co), 64)
    while bco > 32 and (sh0 // rs) * (sw0 // cs) * (-(-co // bco)) * n < _num_sms():
        bco //= 2
    return bco, rs, cs, min(128, pow2(c0)), min(128, pow2(c1)), 4


def _apply_upf(st, n, bco, rs, cs, bk0, bk1, warps, pdl):
    """Fill ``consts`` / ``grid`` / ``cfg`` / ``work`` for an upfold stage.

    Shared by plan construction and ``tools/sweep.py``, so a swept config and a
    shipped config are computed by the same code.
    """
    src = st["src"]
    c0, c1 = src[0][7], src[1][7]
    sh0, sw0 = src[0][4], src[0][5]
    if rs * cs < 16 or sh0 % rs or sw0 % cs:
        return False
    bk0 = min(bk0, triton.next_power_of_2(c0))
    bk1 = min(bk1, triton.next_power_of_2(c1))
    if min(bco, bk0, bk1) < 16:
        return False
    tpr = sw0 // cs
    st["upf"] = True
    st["consts"] = dict(
        BLOCK_CO=bco, RS=rs, CS=cs,
        BK0=bk0, NK0=-(-c0 // bk0), EK0=(c0 % bk0 == 0),
        BK1=bk1, NK1=-(-c1 // bk1), EK1=(c1 % bk1 == 0),
        EVEN_CO=(st["cout"] % bco == 0), EVEN_S=True, TPR=tpr, PDL=pdl)
    st["grid"] = (tpr * (sh0 // rs), -(-st["cout"] // bco), n)
    # MACs actually issued: phase 1 over the source tile, phase 2 over both
    # output rows of it.
    lanes = st["grid"][0] * rs * cs
    st["work"] = st["grid"][1] * n * bco * (lanes * c0 + 2 * 2 * lanes * c1)
    st["cfg"] = _cfg(warps, pdl)
    return True


# (COUT, C0, C1, SH0, SW0, batch) -> (BLOCK_CO, RS, CS, BK0, BK1, warps),
# from ``tools/sweep.py``.  The source resolutions are 20x20 and 40x40 and 20 is
# not a power of two, which is why the source tile is 2-D (4x4 divides 20x20
# exactly): a masked lane in phase 1 costs a masked *pair* in phase 2.
_UPF_TILE: dict = {
    (128, 256, 128, 20, 20, 1): (32, 4, 4, 256, 128, 4),
    (128, 256, 128, 20, 20, 4): (64, 4, 4, 128, 128, 4),
    (64, 128, 64, 40, 40, 1): (32, 4, 8, 128, 64, 4),
    (64, 128, 64, 40, 40, 4): (64, 2, 8, 64, 64, 4),
}


def _tile_key(st, n):
    """Shape identity of a dense stage: what the sweep in ``tools/stages.py``
    keys its measurements on."""
    return (st["g"], st["cout"], st["k"], st["oh"], st["ow"], st["kh"], st["sh"],
            len(st["src"]), st["src"][0][6], n)


def _tile_for(st, p, co, c, k, n, padded):
    """(BLOCK_CO, BLOCK_P, BLOCK_K, warps, ROWT) for one dense stage.

    ``_TILE`` is the measured table over the 15 distinct stage shapes the two
    scored cases issue (``tools/stages.py --sweep``); anything not in it -- a
    different batch, a re-configured neck -- falls back to the tile rule ported
    from L2's sweep plus the addressing test in ``_rowt_needed``.
    """
    ent = _TILE.get(_tile_key(st, n))
    if ent is not None:
        return ent
    bco, bp, bk, warps = _dense_cfg(p, co, c, k, n, padded)
    rowt = _rowt_needed(st)
    if rowt:
        # Where ROWT won in the sweep it won with a 16-pixel tile: that is the
        # widest one that divides OW=80 exactly, and row confinement is only worth
        # its masked lanes when the waste is small.
        bp, warps = 16, 4
    return bco, bp, bk, warps, rowt


# Measured tiles, from ``tools/stages.py --sweep`` over the 15 distinct stage
# shapes the two scored cases issue, timed through a graph replay so Triton's
# ~20 us of Python dispatch is outside the event window.  A shape appears here
# only when a swept config actually beat the rule below; the occurrence-weighted
# 22-stage device sum goes 95.2 -> ~77 us at B=1 and 198.4 -> ~138 us at B=4.
#
# Two things the sweep established that the rule cannot express:
#  * The dominant lever is BLOCK_CO.  Stage 0 at B=4 goes 16.6 -> 9.8 us on
#    BLOCK_CO 32 -> 128 alone; these stages are bound by A/W tile traffic and
#    tile area is what sets their arithmetic intensity.  L2's rule caps it at 32.
#  * ROWT pays exactly where the pixel tile divides OW.  It wins the cat2 1x1 at
#    OW=80 (p16, exact) and ``down_p3``'s stride-2 3x3 at B=1 (27.0 -> 9.5 us),
#    and loses that same stride-2 conv at B=4 (25.72 row against 25.48 flat),
#    where 40/16 wastes 37% of the lanes.
#
# (body, COUT, K, OH, OW, KH, SH, n_sources, up0, batch)
#   -> (BLOCK_CO, BLOCK_P, BLOCK_K, num_warps, ROWT)
_TILE: dict = {
    (0, 64, 96, 80, 80, 1, 1, 1, 0, 4): (64, 64, 256, 4, False),
    (0, 64, 192, 80, 80, 1, 1, 2, 1, 1): (64, 16, 128, 4, True),
    (0, 64, 192, 80, 80, 1, 1, 2, 1, 4): (64, 16, 128, 4, True),
    (0, 128, 192, 40, 40, 1, 1, 1, 0, 4): (128, 64, 256, 8, False),
    (0, 128, 384, 40, 40, 1, 1, 2, 1, 1): (128, 16, 128, 8, True),
    (0, 128, 384, 40, 40, 1, 1, 2, 1, 4): (128, 32, 128, 4, False),
    (0, 256, 384, 20, 20, 1, 1, 1, 0, 1): (32, 64, 128, 4, False),
    (0, 256, 384, 20, 20, 1, 1, 2, 0, 1): (32, 64, 128, 4, False),
    (0, 256, 384, 20, 20, 1, 1, 2, 0, 4): (64, 64, 256, 8, False),
    (1, 32, 288, 80, 80, 3, 1, 1, 0, 1): (32, 64, 128, 4, False),
    (1, 64, 576, 40, 40, 3, 1, 1, 0, 4): (64, 64, 128, 8, False),
    (1, 64, 576, 40, 40, 3, 2, 1, 0, 1): (64, 16, 256, 8, True),
    (1, 64, 576, 40, 40, 3, 2, 1, 0, 4): (64, 64, 128, 8, False),
    # r2 additions (tools/sweep.py, coordinate descent over the same axes):
    (0, 64, 96, 80, 80, 1, 1, 1, 0, 1): (32, 128, 128, 4, False),
    (0, 256, 128, 20, 20, 1, 1, 1, 0, 1): (32, 32, 256, 8, False),
    (0, 256, 128, 20, 20, 1, 1, 1, 0, 4): (32, 128, 256, 8, False),
    (0, 256, 384, 20, 20, 1, 1, 1, 0, 4): (32, 128, 128, 4, False),
    (1, 32, 288, 80, 80, 3, 1, 1, 0, 4): (32, 64, 64, 4, False),
}


# ---------------------------------------------------------------------------
# Launching.  ``_launch`` issues one stage; ``_issue`` issues the whole neck.
# Both take a ``dict`` of the six per-call tensors plus the workspace, so the
# same code drives the eager path and the graph capture.
# ---------------------------------------------------------------------------
def _launch(st, T, wgt, bia):
    if st["g"] == _G_FUSE:
        a, b = st["a"], st["b"]
        ak, aoff, asn, asc, aimh, aimw, _u, cin = a["src"][0]
        bk, boff, bsn, bsc, _bh, _bw, _u2, _cb = b["src"][0]
        y_k, y_off, y_sn, y_sc = st["y"]
        _fuse_kernel[st["grid"]](
            T[ak], T[bk], wgt, bia, T[y_k],
            cin, st["ka"], aimh, aimw, aoff, asn, asc,
            st["cmid"], a["kh"], a["kw"], a["w"], a["boff"],
            st["cb"], st["kb"], boff, bsn, bsc,
            st["cout"], st["p"], st["ow"], b["w"], b["boff"],
            y_off, y_sn, y_sc, a["act"], b["act"],
            **st["consts"], **st["cfg"])
        return
    y_k, y_off, y_sn, y_sc = st["y"]
    res = st["res"]
    if res is None:
        r_k, r_off, r_sn, r_sc, has_res = y_k, 0, 0, 0, False
    else:
        r_k, r_off, r_sn, r_sc = res
        has_res = True
    Y, R = T[y_k], T[r_k]
    src = st["src"]
    g = st["g"]
    if g == _G_GEMM:
        if st.get("upf"):
            k0, o0, sn0, sc0, h0, w0, _up0, c0 = src[0]
            k1, o1, sn1, sc1, _h1, _w1, _up1, c1 = src[1]
            _gemm_up_kernel[st["grid"]](
                T[k0], T[k1], wgt, bia, R, Y,
                c0, c1, st["k"], o0, o1, sn0, sn1, sc0, sc1, h0, w0,
                st["cout"], st["ow"], st["w"], st["boff"],
                y_off, y_sn, y_sc, r_off, r_sn, r_sc, st["act"], has_res,
                **st["consts"], **st["cfg"])
            return
        k0, o0, sn0, sc0, _h0, w0, up0, c0 = src[0]
        if len(src) > 1:
            k1, o1, sn1, sc1, _h1, w1, up1, c1 = src[1]
        else:
            k1, o1, sn1, sc1, w1, up1, c1 = k0, 0, 0, 0, 1, 0, 0
        _gemm_kernel[st["grid"]](
            T[k0], T[k1], wgt, bia, R, Y,
            c0, c1, st["k"], o0, o1, sn0, sn1, sc0, sc1, up0, up1, w0, w1,
            st["cout"], st["p"], st["ow"], st["w"], st["boff"],
            y_off, y_sn, y_sc, r_off, r_sn, r_sc, st["act"], has_res,
            **st["consts"], **st["cfg"])
    elif g == _G_CONV:
        k0, o0, sn0, sc0, h0, w0, _up0, c0 = src[0]
        if st.get("s1"):
            _conv_s1_kernel[st["grid"]](
                T[k0], wgt, bia, R, Y,
                c0, st["k"], h0, w0, o0, sn0, sc0,
                st["cout"], st["p"], st["ow"], st["kh"], st["kw"],
                st["kh"] // 2, st["kw"] // 2, st["w"], st["boff"],
                y_off, y_sn, y_sc, r_off, r_sn, r_sc, st["act"], has_res,
                **st["consts"], **st["cfg"])
            return
        if st.get("s2"):
            _conv_s2_kernel[st["grid"]](
                T[k0 + _I32], wgt, bia, R, Y,
                c0, h0, w0 // 2, o0 // 2, sn0 // 2, sc0 // 2,
                st["cout"], st["k"], st["ow"], st["w"], st["boff"],
                y_off, y_sn, y_sc, r_off, r_sn, r_sc, st["act"], has_res,
                **st["consts"], **st["cfg"])
            return
        _conv_kernel[st["grid"]](
            T[k0], wgt, bia, R, Y,
            c0, st["k"], h0, w0, o0, sn0, sc0,
            st["cout"], st["p"], st["ow"], st["kh"], st["kw"], st["sh"], st["sw"],
            st["kh"] // 2, st["kw"] // 2, st["w"], st["boff"],
            y_off, y_sn, y_sc, r_off, r_sn, r_sc, st["act"], has_res,
            **st["consts"], **st["cfg"])
    else:
        k0, o0, sn0, sc0, h0, w0, _up0, c0 = src[0]
        _dw_kernel[st["grid"]](
            T[k0], wgt, bia, R, Y,
            c0, h0, w0, o0, sn0, sc0,
            st["p"], st["ow"], st["kh"], st["kw"], st["sh"], st["sw"],
            st["kh"] // 2, st["kw"] // 2, st["w"], st["boff"],
            y_off, y_sn, y_sc, r_off, r_sn, r_sc, st["act"], has_res,
            **st["consts"], **st["cfg"])


_I32 = "\0i32"


def _issue(plan, T):
    """Issue all 22 stages.  A stage on the int32-pair stride-2 body needs an
    int32 *view* of its source; those are built once per ``_issue`` (so at capture
    time, not per replay) and keyed off the buffer name."""
    for st in plan.stages:
        if st.get("s2"):
            key = st["src"][0][0] + _I32
            if key not in T:
                T[key] = T[st["src"][0][0]].view(torch.int32)
    wgt, bia = plan.wgt, plan.bia
    if _USE_PF and plan.scr is not None:
        n, blk, pdl = plan.wb.numel(), 4096, _pdl_on()
        _touch_kernel[(-(-n // blk),)](plan.wb, plan.scr, n, blk, pdl,
                                       **_cfg(8, pdl))
    for st in plan.stages:
        _launch(st, T, wgt, bia)


def _slab_views(numels, shapes, dtype, device, slack=0):
    """One contiguous allocation with *slack* elements of padding at both ends,
    viewed as the given tensors.

    Two jobs.  (1) Node count: three separate static buffers force three
    ``copy_`` nodes in and three ``clone`` nodes out, while three *views of one
    slab* let the whole fill be one ``torch.cat(out=)`` and the whole drain one
    ``clone`` -- and an eager node costs ~4 us in the scored loop against ~0.2 us
    for a node inside the capture, so six of them are ~24 us of the window
    against the 0.35 us of HBM time the 1.4 MB actually needs.  (2) The padding,
    which is what makes ``NOMASK`` legal for a stage whose source lives here.
    """
    raw = torch.empty(sum(numels) + 2 * slack, dtype=dtype, device=device)
    flat = raw[slack:raw.numel() - slack] if slack else raw
    views, off = [], 0
    for n, shp in zip(numels, shapes):
        views.append(flat[off:off + n].view(shp))
        off += n
    return flat, views


def _out_tensors(plan, device, flat=False):
    """The three returned tensors, as views of one slack-padded slab.

    ``p3`` is read by ``down_p3``'s 3x3 stride-2 convolution, whose halo taps
    address before and after it, so this allocation carries the same padding the
    workspace does -- and it must, on *every* path (the eager path and
    ``tools/stages.py`` call this too, not just the capture)."""
    n = plan.n
    oflat, views = _slab_views([n * b.c * b.h * b.w for b in plan.outs],
                               [(n, b.c, b.h, b.w) for b in plan.outs],
                               plan.dtype, device, _SLACK)
    return (oflat, views) if flat else views


# ---------------------------------------------------------------------------
# Whole-neck CUDA graph.  All 22 nodes are captured once per shape, so the
# graph's fixed cost and its per-node scheduling are paid once for the whole
# operator rather than once per block -- which is the point of the round.
#
# This is not an optional nicety: issuing the 22 stages eagerly costs 445 us of
# *host* time (Triton's Python dispatch is ~20 us per launch once a kernel takes
# 30-odd specialization arguments), which is worse than the composed frozen-L2
# neck's 235 us.  The capture is what turns the plan into a win.
#
# The harness hands a fresh ``data_ptr`` for every input on every iteration (its
# shifting memory pool), so the graph owns static input buffers and copies into
# them, and the three returned tensors are copied out of static buffers.  Those
# copies are the ONLY eager launches left, and an eager launch costs ~4 us in the
# scored loop against ~0.2 us for a node inside the capture (``tools/where.py``),
# so their *count* is what matters and not their 1.4 MB: the static buffers are
# views of one slab each, so the fill is one ``torch.cat(out=)`` and the drain one
# ``clone`` -- 6 nodes -> 2, worth 17 us at batch 1 and 13 us at batch 4.  A fused
# Triton copy kernel is not an option (outside the capture, ~20 us of Python
# dispatch), and ``torch._foreach_copy_`` measured 3-4 us worse than three plain
# copies, so it is kept only as a knob.  See _COPY.
# ---------------------------------------------------------------------------
def _graph_build(mod, plan, x3, x4, x5, device):
    """Capture the whole neck for this shape; ``None`` if it is not capturable."""
    try:
        ws = mod._workspace(plan, device)
        ins = (x3, x4, x5)
        iflat, sin = _slab_views([t.numel() for t in ins],
                                 [tuple(t.shape) for t in ins], plan.dtype, device)
        for s, t in zip(sin, ins):
            s.copy_(t)
        oflat, sout = _out_tensors(plan, device, flat=True)
        T = {_WS: ws, "p3_backbone": sin[0], "p4_backbone": sin[1],
             "p5_backbone": sin[2], "p3": sout[0], "n4": sout[1], "n5": sout[2]}
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):      # JIT every stage before capture
                _issue(plan, T)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _issue(plan, T)
        torch.cuda.synchronize()
    except Exception:  # noqa: BLE001 - uncapturable: stay on the reference path
        if _DEBUG:
            raise
        return None
    numels = [b.c * b.h * b.w * plan.n for b in plan.outs]
    ends = [sum(numels[:i + 1]) for i in range(3)]
    views_of = [(ends[i] - numels[i], ends[i], (plan.n, b.c, b.h, b.w))
                for i, b in enumerate(plan.outs)]
    return (graph, iflat, sin, oflat, sout, views_of)


def _graph_run(e, x3, x4, x5):
    graph, iflat, sin, oflat, sout, views_of = e
    if _COPY == "slab":
        # One node.  ``cat`` over three contiguous 1-D sources into a
        # pre-sized ``out`` is ATen's batched copy kernel -- one launch, no
        # allocation -- where three ``copy_`` are three launches of the same
        # total bytes.
        torch.cat((x3.view(-1), x4.view(-1), x5.view(-1)), out=iflat)
    elif _COPY == "foreach":
        torch._foreach_copy_(sin, (x3, x4, x5))
    else:
        sin[0].copy_(x3)
        sin[1].copy_(x4)
        sin[2].copy_(x5)
    graph.replay()
    if _COPY == "slab":
        # One node, and the three returned tensors are contiguous NCHW views of
        # it (a view of a contiguous slab is itself contiguous).
        flat = oflat.clone()
        return [flat[a:b].view(s) for a, b, s in views_of]
    if _COPY == "plain":
        return [sout[0].clone(), sout[1].clone(), sout[2].clone()]
    flat = torch.empty(oflat.numel(), dtype=oflat.dtype, device=oflat.device)
    outs = [flat[a:b].view(s) for a, b, s in views_of]
    if _COPY == "foreach":
        torch._foreach_copy_(outs, sout)
    else:
        outs[0].copy_(sout[0])
        outs[1].copy_(sout[1])
        outs[2].copy_(sout[2])
    return outs


# ---------------------------------------------------------------------------
# The module.  Every submodule keeps its baseline name and class, so the
# state_dict keys are the baseline's exactly and ``_reference`` is the baseline
# forward over the frozen L2 blocks.
# ---------------------------------------------------------------------------
class YOLOv10Neck(nn.Module):
    def __init__(self):
        super().__init__()
        self._upsample = Interpolate()
        self.cat1 = YOLOConcat(1)
        self.c2f_p4 = YOLOC2f(384, 128, n=1, shortcut=False)
        self.cat2 = YOLOConcat(1)
        self.c2f_p3 = YOLOC2f(192, 64, n=1, shortcut=False)
        self.down_p3 = YOLOConv(64, 64, 3, 2)
        self.cat3 = YOLOConcat(1)
        self.c2f_n4 = YOLOC2f(192, 128, n=1, shortcut=False)
        self.down_n4 = YOLOSCDown(128, 128, 3, 2)
        self.cat4 = YOLOConcat(1)
        self.c2fcib_n5 = YOLOC2fCIB(384, 256, n=1, shortcut=True, lk=True)
        self._plans: dict = {}          # (shapes, dtype, device) -> _Plan
        self._ws: dict = {}             # (elems, dtype, device) -> workspace
        self.register_load_state_dict_post_hook(_invalidate_hook)

    # -- plan cache --------------------------------------------------------
    def _get_plan(self, key, shapes, dtype, device):
        plan = self._plans.get(key)
        if plan is not None:
            if plan.sig == tuple((t.data_ptr(), t._version) for t in plan.src):
                return plan
            self._plans.clear()
            self._ws.clear()
        if len(self._plans) >= _MAX_PLANS:
            return None
        plan = _build_plan(self, shapes, dtype, device)
        self._plans[key] = plan
        return plan

    def _workspace(self, plan, device):
        key = (plan.ws_elems, plan.dtype, device)
        ws = self._ws.get(key)
        if ws is None:
            ws = torch.empty(plan.ws_elems, dtype=plan.dtype, device=device)
            self._ws[key] = ws
        return ws

    # -- forward -----------------------------------------------------------
    def _reference(self, feats):
        p3_backbone = feats["p3_backbone"]
        p4_backbone = feats["p4_backbone"]
        p5_backbone = feats["p5_backbone"]

        x = self._upsample(p5_backbone, scale_factor=2.0, mode="nearest")
        x = self.cat1([x, p4_backbone])
        p4 = self.c2f_p4(x)

        x = self._upsample(p4, scale_factor=2.0, mode="nearest")
        x = self.cat2([x, p3_backbone])
        p3 = self.c2f_p3(x)

        x = self.down_p3(p3)
        x = self.cat3([x, p4])
        n4 = self.c2f_n4(x)

        x = self.down_n4(n4)
        x = self.cat4([x, p5_backbone])
        n5 = self.c2fcib_n5(x)
        return [p3, n4, n5]

    def forward(self, feats: dict[str, torch.Tensor]):
        if triton is None or self.training or torch.is_grad_enabled():
            return self._reference(feats)
        try:
            x3 = feats["p3_backbone"]
            x4 = feats["p4_backbone"]
            x5 = feats["p5_backbone"]
        except (KeyError, TypeError):
            return self._reference(feats)
        if not (x3.is_cuda and x4.is_cuda and x5.is_cuda):
            return self._reference(feats)
        if not (x3.dtype is torch.float16 and x4.dtype is x3.dtype
                and x5.dtype is x3.dtype):
            return self._reference(feats)
        if not (x3.is_contiguous() and x4.is_contiguous() and x5.is_contiguous()):
            return self._reference(feats)
        if x3.dim() != 4 or x4.dim() != 4 or x5.dim() != 4:
            return self._reference(feats)
        shapes = (tuple(x3.shape), tuple(x4.shape), tuple(x5.shape))
        if not (shapes[0][0] == shapes[1][0] == shapes[2][0]):
            return self._reference(feats)
        device = x3.device
        key = (shapes, x3.dtype, device)
        plan = self._get_plan(key, shapes, x3.dtype, device)
        if plan is None or not plan.ok:
            return self._reference(feats)

        if _PATH in ("graph", "auto"):
            entry = plan.graph.get("g", _MISSING)
            if entry is _MISSING:
                entry = _graph_build(self, plan, x3, x4, x5, device)
                plan.graph["g"] = entry
            if entry is not None:
                return _graph_run(entry, x3, x4, x5)
        if _PATH == "eager":
            ws = self._workspace(plan, device)
            outs = _out_tensors(plan, device)
            T = {_WS: ws, "p3_backbone": x3, "p4_backbone": x4,
                 "p5_backbone": x5, "p3": outs[0], "n4": outs[1], "n5": outs[2]}
            _issue(plan, T)
            return outs
        # Nothing captured: the composed frozen-L2 blocks are 235 us against the
        # 445 us the eager stage list costs in host dispatch, so the reference
        # path is the better fallback, not the worse one.
        return self._reference(feats)


def _invalidate_hook(module, incompatible_keys):  # noqa: ARG001
    module._plans.clear()
    module._ws.clear()
