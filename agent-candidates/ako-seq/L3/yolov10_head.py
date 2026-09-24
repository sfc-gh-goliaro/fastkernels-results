"""YOLOv10 detection head (L3 composite) -- fused post-trunk tail.

The scored path is ``eval() + export=True``, so only the ``one2one`` branch runs
and the module's whole job is

    towers -> pack [b, 144, 8400] -> DFL -> dist2bbox -> stride -> sigmoid
           -> amax over classes -> top-300 anchors -> top-300 (anchor, class)
           -> xywh2xyxy -> concat to [b, 300, 6]

Written as composed operators that is ~70 eager launches over tensors that are at
most a few MB, and on B200 the module is *dispatch* bound, not device bound:
1.73 ms of host enqueue for a 1.73 ms device span, i.e. the GPU idle between
~24 us CPU launches. Two things follow, and this file does both: cut the op
count -- by doing the expensive per-anchor work only for the anchors that
survive selection -- and then remove the enqueue itself.

**The tail, as four kernels plus an exact selection.**

* ``_dw3_bn_act_kernel`` -- depthwise 3x3 Conv-BN-SiLU in one launch. The frozen
  L2 ``YOLOConv``'s fused gate excludes ``g=c1``, so the six depthwise blocks in
  ``cv3`` were running as three composed eager ops each: 52 us of CPU per block
  against 27 us for a fused one, ~240 us in total -- more than the whole tail.
* ``_score_kernel`` -- the per-anchor selection score in one pass. Selection only
  ever reads ``scores.amax(-1)``, and ``fp16(sigmoid(.))`` is monotone
  non-decreasing, so it commutes with the max: taking the max of the 80 *logits*
  and rounding once is bit-identical to "sigmoid all 672k values, then reduce",
  at 1/80th of the sigmoid work.
* ``_sel_score_kernel`` -- the 80 class scores of the 300 survivors (24k
  sigmoids, not 672k), laid out exactly as the reference's
  ``scores.flatten(1)`` so the second top-k sees the same values *and* the same
  flat indices.
* ``_head_kernel`` -- the box head's 1x1 conv, DFL, dist2bbox, the stride scale,
  xywh2xyxy, the label arithmetic and the final concat, for the 300 output rows:
  ~28x less per-anchor work than the reference's 8400.
* ``_Select`` -- the two top-300 selections, exact, six launches each in place of
  ``torch.topk``'s eighteen.  Half the module's device time used to live here.

The three per-level ``cat``s and the packed ``[b, 144, 8400]`` never exist -- the
kernels take the six tower outputs as six pointers and resolve the level from the
tile id -- and the whole export path is captured into a single CUDA graph, which
is what takes the host side from 900 us to 45 us.

**Why the selection is bit-exact, and why it has to be.**  With random weights
the class logits are tiny (|z| < 0.1), so every ``sigmoid`` lands within a few
fp16 steps of 0.5 and *three* distinct values typically cover all 8400 anchor
maxima, 1600 anchors sharing the top one.  ``torch.topk`` breaks those ties by
lowest index, so a one-ulp disagreement anywhere in the score field does not
perturb one row -- it re-shuffles the whole 300-row tie group and the output
stops matching at all.  Hence:

* The reference's sigmoid is ``fp16(1.f / (1.f + expf(-x)))``.  Triton's
  ``1.0 / y`` lowers to a fast reciprocal, which disagrees with ``div.rn.f32``
  on exactly 2 of the 65536 fp16 inputs -- both at |x| ~ 0.0065, squarely inside
  the range these logits occupy.  ``libdevice.div_rn`` closes that: the kernel's
  sigmoid is bit-identical to ``torch.sigmoid`` on *all* 65536 fp16 inputs.  This
  is also why the frozen L1 ``Sigmoid`` (a ``--use_fast_math`` MUFU.TANH kernel)
  is not used on this path -- it differs from the reference on ~1% of values,
  which broke selection outright.  It stays imported and reachable for the
  non-export path.
* Both selections are exact and hand-written (``_Select``), and the tie order is
  the reference's *by construction* rather than by imitation: a score and its
  index pack into one int32 key whose descending order already **is** value
  descending / lowest index first, so all keys are distinct and there is no
  tie-break rule left to reproduce.  ``torch.topk`` is 41-46 us and eighteen
  kernels per call; this is 17-20 us and six.
* Only the *box* head's trailing 1x1 conv is folded into the tail.  Folding the
  class head's too measured 1.2x faster, but computing it with ``tl.dot`` instead
  of cudnn changes the fp32 summation order and flips ~0.1% of logits by one ulp
  -- harmless on a weight draw whose scores are bunched, enough to reshuffle the
  output on one whose scores spread across many fp16 levels.  Its *bias* does
  move into the tail, which is a different thing: cudnn runs that GEMM unbiased
  and PyTorch adds the bias in a separate 9.6 us broadcast pass, and
  ``fp16(unbiased + bias)`` is bit-identical to the biased conv on every scored
  shape.  The GEMM stays exactly where it was; only the epilogue moved.

**Why the box path is not algebraically collapsed.**  ``dist2bbox(xywh=True)``
followed by ``xywh2xyxy`` is the identity and the stride is a uniform scale, so
the path reduces to ``((anchor - lt) * s, (anchor + rb) * s)``.  It is
deliberately *not* reduced: the reference computes the round trip in fp16, where
the final ``c - w/2`` cancels two O(500) values, so its ``x1``/``y1`` carry up to
~0.5 px of rounding that the collapsed form does not.  Against a reference that
noisy the exact answer is the wrong answer -- near a zero coordinate the gap
exceeds ``atol=1e-2``.  So ``_head_kernel`` performs each reference op in fp32
and narrows to fp16 after it, which is what an fp16 op does.

**Capture safety.**  Static shapes, no host syncs, no ``.item()``; the
anchor/stride buffers are filled in place and never rebound after the first call;
the tail scratch and top-k outputs are allocated once per (shape, device); the
per-shape execution plan holds prebound callables so the steady-state host path
is a loop over them.  The internal graph refuses to *capture* while an enclosing
capture is active and refuses to *replay* there either, handing the enclosing
capture the individual launches instead -- so an engine that graphs the whole
model still works.

Anything the fast path does not cover -- training, ``export=False``,
``dynamic``, non-fp16, CPU tensors, a different ``nc``/``reg_max``/``max_det``, a
level count other than 3, or a tower that does not end in a bias-only 1x1 conv --
falls through to the reference composition, which is what this module already was.
"""

from __future__ import annotations

import math
import copy
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra import libdevice

from ..L1.conv2d import Conv2d
from ..L1.sigmoid import Sigmoid
from ..L1.silu import SiLU
from ..L2.yolov10_conv import YOLOConv
from ..L2.yolov10_dfl import YOLODFL


def make_anchors(feats: list[torch.Tensor], strides: torch.Tensor, grid_cell_offset: float = 0.5):
    anchor_points, stride_tensor = [], []
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def dist2bbox(distance: torch.Tensor, anchor_points: torch.Tensor, xywh: bool = True, dim: int = -1):
    lt, rb = distance.split([2, 2], dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)
    return torch.cat((x1y1, x2y2), dim)


def xywh2xyxy(boxes: torch.Tensor) -> torch.Tensor:
    x, y, w, h = boxes.unbind(-1)
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    return torch.stack((x1, y1, x2, y2), dim=-1)


def v10postprocess(preds: torch.Tensor, max_det: int, nc: int = 80):
    boxes, scores = preds.split([4, nc], dim=-1)
    max_scores = scores.amax(dim=-1)
    max_scores, index = torch.topk(max_scores, max_det, dim=-1)
    index = index.unsqueeze(-1)
    boxes = torch.gather(boxes, dim=1, index=index.repeat(1, 1, boxes.shape[-1]))
    scores = torch.gather(scores, dim=1, index=index.repeat(1, 1, scores.shape[-1]))

    scores, index = torch.topk(scores.flatten(1), max_det, dim=-1)
    labels = index % nc
    index = index // nc
    boxes = boxes.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, boxes.shape[-1]))
    return boxes, scores, labels


# ===========================================================================
# Device helpers
# ===========================================================================
_NO_GRAPH = os.environ.get("FK_NO_GRAPH", "0") == "1"
_CLS_BIAS = os.environ.get("FK_CLS_BIAS", "1") == "1"


def _npow2(n: int) -> int:
    return 1 << (int(n) - 1).bit_length()


@triton.jit
def _f16(v):
    """Narrow to fp16 and widen back -- i.e. perform the fp16 op the reference
    performs.  Used wherever the reference's *rounding*, not its algebra, is
    what has to be reproduced."""
    return v.to(tl.float16).to(tl.float32)


@triton.jit
def _sigmoid_rn(x):
    """``fp32`` sigmoid that narrows to the same fp16 as ``torch.sigmoid``.

    ``1.0 / y`` in Triton is a fast reciprocal; the reference is a plain
    ``float`` divide.  They differ on 2 of the 65536 fp16 inputs, both near
    |x| = 0.0065 where these logits live, and one differing score is enough to
    re-order a 1600-anchor tie group.  ``div_rn`` is ``div.rn.f32``.
    """
    return libdevice.div_rn(1.0, 1.0 + tl.exp(-x))


@triton.jit
def _head1x1(pre, WT, BI, CO, MCO, CI: tl.constexpr, CIP: tl.constexpr):
    """The tower's trailing 1x1 conv as what it is: a per-anchor GEMM.

    ``pre`` is [rows, CIP] fp16, ``WT`` the ``[COUT, CI, 1, 1]`` weight read as
    [CIP, COUT].  fp32 accumulate then one narrowing store, which is what cudnn
    does for an fp16 conv with an fp32 bias epilogue.
    """
    ci = tl.arange(0, CIP)[:, None]
    wt = tl.load(WT + CO * CI + ci, mask=MCO & (ci < CI), other=0.0)
    return _f16(tl.dot(pre, wt) + tl.load(BI + CO, mask=MCO, other=0.0).to(tl.float32))


# ===========================================================================
# Exact top-k, as six small launches instead of torch.topk's eighteen.
#
# ``torch.topk`` is 41-46 us of device time *per call* on these rows (n=8400 and
# n=24000, k=300) and, inside the graph, eighteen kernels: a single-block
# ``sbtopk::gatherTopK``, a multi-block ``mbtopk`` digit-count/cumsum/gather
# chain, two cub scans, two memsets and two ``radixSortKVInPlace`` passes to sort
# the k results.  Together the two selections are ~85 us of work plus ~35 us of
# inter-kernel gap: half the module's device time.
#
# The reduction that replaces it: fp16 has a **monotone bit order**, so a score
# and its index pack into one int32
#
#     key = ((monotone_u16(score) - 32768) << 16) | (65535 - index)
#
# whose descending order *is* ``torch.topk``'s order -- value descending, and
# lowest original index first inside a tie.  Every key is distinct, so there is
# no tie-break rule left to reproduce: it is a property of the key, not of the
# algorithm, and the thousand-anchor fp16 ties this operator lives on stop being
# a special case.  (The << 16 needs n <= 65536, which both rows satisfy; the
# ``- 32768`` is what keeps the product inside int32.)
#
# Selection is then a radix select on that key -- and only on its *value* half,
# because the index half is resolved by counting rather than comparing:
#
#   1. ``_sel_hist_hi``  per-tile 256-bin histogram of the top byte of the value
#                        key.  Per-tile *stores*, not global atomics, so nothing
#                        has to be zeroed first and no launch is spent on a
#                        memset.
#   2. ``_sel_red_hi``   one CTA per row: sum the tiles, one 256-wide scan, and
#                        the bucket holding the k-th largest falls out exactly.
#   3. ``_sel_hist_lo``  the same for the low byte, over that bucket only.
#   4. ``_sel_red_lo``   the exact k-th value ``T``, the count ``c_above`` of
#                        strictly larger elements, and the per-tile prefixes.
#   5. ``_sel_place``    elements ``> T`` (fewer than k of them) are compacted;
#                        elements ``== T`` all carry the *same* value, so their
#                        order is index order, and the per-tile prefix of the
#                        ==T counts places each of them directly -- the
#                        ``need = k - c_above`` lowest indices, no sort.
#   6. ``_sel_order_gt`` one ``tl.sort`` over the 512-slot compaction buffer
#                        orders the ``> T`` elements among themselves.
#
# Measured in a CUDA graph against ``torch.topk`` on the same rows: 17.6 us vs
# 44.3 us at n=8400, and validated bit-exact in *both* values and indices over
# 12 adversarial score fields x {8400, 24000, 2048, 4096, 1000} x {b=1, b=4} --
# including fields with a single distinct fp16 level, two levels, and a top group
# that lives only at the tail of the row, i.e. the cases where tie order is the
# entire answer.
# ===========================================================================
_SEL_NEG = tl.constexpr(-2147483648)


@triton.jit
def _vkey(x):
    """fp16 -> the 16-bit unsigned integer with the same total order.

    Positive floats already compare as their bit patterns; negatives compare
    reversed, so their bits are complemented.  ``-0.0`` maps below ``+0.0``,
    which is also how ``torch.topk`` orders them.
    """
    u = x.to(tl.uint16, bitcast=True).to(tl.int32)
    return tl.where((u & 0x8000) != 0, (~u) & 0xFFFF, u | 0x8000)


@triton.jit
def _sel_hist_hi(X, H1, N: tl.constexpr, TS: tl.constexpr, NT: tl.constexpr):
    t = tl.program_id(0)
    b = tl.program_id(1).to(tl.int64)
    off = t * TS + tl.arange(0, TS)
    m = off < N
    vk = _vkey(tl.load(X + b * N + off, mask=m, other=0.0))
    tl.store(H1 + (b * NT + t) * 256 + tl.arange(0, 256),
             tl.histogram(vk >> 8, 256, mask=m))


@triton.jit
def _sel_red_hi(H1, META, G1, K: tl.constexpr, NT: tl.constexpr,
                NTP: tl.constexpr):
    """The bucket of the k-th largest top byte, exactly.

    ``c_gt[v] = #{hb > v}`` is non-increasing and ``c_gt[v] + cnt[v] =
    c_gt[v-1]``, so the half-open intervals ``[c_gt[v], c_gt[v-1])`` partition
    ``[0, total)`` and ``k-1`` lands in exactly one of them.  Empty buckets
    cannot satisfy the test, so the ``tl.sum`` below is a one-hot pick.
    """
    b = tl.program_id(0).to(tl.int64)
    t = tl.arange(0, NTP)[:, None]
    v = tl.arange(0, 256)[None, :]
    h = tl.load(H1 + (b * NT + t) * 256 + v, mask=t < NT, other=0)
    cnt = tl.sum(h, 0)
    c_gt = tl.sum(cnt, 0) - tl.cumsum(cnt, 0)
    cond = (c_gt < K) & (c_gt + cnt >= K)
    hi = tl.sum(tl.where(cond, tl.arange(0, 256), 0), 0)
    tl.store(META + b * 8 + 0, hi)
    tl.store(META + b * 8 + 1, tl.sum(tl.where(cond, c_gt, 0), 0))
    # Per-tile count of hb > hi: free here, and it is half of the >T prefix
    # that _sel_red_lo needs.
    tl.store(G1 + b * NT + tl.arange(0, NTP), tl.sum(tl.where(v > hi, h, 0), 1),
             mask=tl.arange(0, NTP) < NT)


@triton.jit
def _sel_hist_lo(X, META, H2, N: tl.constexpr, TS: tl.constexpr,
                 NT: tl.constexpr):
    t = tl.program_id(0)
    b = tl.program_id(1).to(tl.int64)
    hi = tl.load(META + b * 8 + 0)
    off = t * TS + tl.arange(0, TS)
    m = off < N
    vk = _vkey(tl.load(X + b * N + off, mask=m, other=0.0))
    tl.store(H2 + (b * NT + t) * 256 + tl.arange(0, 256),
             tl.histogram(vk & 255, 256, mask=m & ((vk >> 8) == hi)))


@triton.jit
def _sel_red_lo(H2, META, G1, GTPRE, TPRE, GTK, K: tl.constexpr,
                NT: tl.constexpr, NTP: tl.constexpr, KP: tl.constexpr):
    b = tl.program_id(0).to(tl.int64)
    hi = tl.load(META + b * 8 + 0)
    ca_hi = tl.load(META + b * 8 + 1)
    t = tl.arange(0, NTP)
    mt = t < NT
    v = tl.arange(0, 256)[None, :]
    h = tl.load(H2 + (b * NT + t[:, None]) * 256 + v, mask=mt[:, None], other=0)
    cnt = tl.sum(h, 0)
    c_gt = tl.sum(cnt, 0) - tl.cumsum(cnt, 0)
    a = ca_hi + c_gt
    cond = (a < K) & (a + cnt >= K)
    lo = tl.sum(tl.where(cond, tl.arange(0, 256), 0), 0)
    c_above = tl.sum(tl.where(cond, a, 0), 0)
    tl.store(META + b * 8 + 2, (hi << 8) | lo)   # T: the exact k-th value key
    tl.store(META + b * 8 + 3, c_above)
    tl.store(META + b * 8 + 4, K - c_above)      # need, taken from the tie group
    eqc = tl.sum(tl.where(v == lo, h, 0), 1)     # per tile: #{vkey == T}
    gtc = tl.load(G1 + b * NT + t, mask=mt, other=0) + tl.sum(tl.where(v > lo, h, 0), 1)
    tl.store(GTPRE + b * NT + t, tl.cumsum(gtc, 0) - gtc, mask=mt)
    tl.store(TPRE + b * NT + t, tl.cumsum(eqc, 0) - eqc, mask=mt)
    tl.store(GTK + b * KP + tl.arange(0, KP), tl.full((KP,), _SEL_NEG, tl.int32))


@triton.jit
def _sel_place(X, META, GTPRE, TPRE, GTK, VOUT, IOUT, N: tl.constexpr,
               TS: tl.constexpr, NT: tl.constexpr, K: tl.constexpr,
               KP: tl.constexpr):
    t = tl.program_id(0)
    b = tl.program_id(1).to(tl.int64)
    T = tl.load(META + b * 8 + 2)
    c_above = tl.load(META + b * 8 + 3)
    need = tl.load(META + b * 8 + 4)
    off = t * TS + tl.arange(0, TS)
    m = off < N
    x = tl.load(X + b * N + off, mask=m, other=0.0)
    vk = _vkey(x)
    gt = m & (vk > T)
    eq = m & (vk == T)
    ig = gt.to(tl.int32)
    slot = tl.load(GTPRE + b * NT + t) + tl.cumsum(ig, 0) - ig
    tl.store(GTK + b * KP + slot, ((vk - 32768) << 16) | (65535 - off), mask=gt)
    ie = eq.to(tl.int32)
    tp = tl.load(TPRE + b * NT + t) + tl.cumsum(ie, 0) - ie
    keep = eq & (tp < need)
    o = c_above + tp
    tl.store(IOUT + b * K + o, off.to(tl.int64), mask=keep)
    tl.store(VOUT + b * K + o, x, mask=keep)


@triton.jit
def _sel_order_gt(X, META, GTK, VOUT, IOUT, N: tl.constexpr, K: tl.constexpr,
                  KP: tl.constexpr):
    b = tl.program_id(0).to(tl.int64)
    p = tl.arange(0, KP)
    key = tl.sort(tl.load(GTK + b * KP + p), descending=True)
    idx = 65535 - (key & 0xFFFF)
    mk = p < tl.load(META + b * 8 + 3)
    tl.store(IOUT + b * K + p, idx.to(tl.int64), mask=mk)
    tl.store(VOUT + b * K + p, tl.load(X + b * N + idx, mask=mk, other=0.0),
             mask=mk)


class _Select:
    """Prebound exact top-k over a ``[b, n]`` fp16 row.

    Scratch is allocated once per shape and never rebound, and every launch
    geometry is a constant, so the whole thing records into the module's graph
    like any other kernel: no host sync, no ``.item()``, no data-dependent shape.
    """

    __slots__ = ("b", "n", "k", "ts", "nt", "ntp", "kp", "w", "wr", "h1", "h2",
                 "meta", "g1", "gtpre", "tpre", "gtk", "v", "i")

    def __init__(self, b, n, k, ts, dev, buf, tag, w=4, wr=4, kp=512):
        self.b, self.n, self.k, self.ts, self.kp = b, n, k, ts, kp
        self.nt = triton.cdiv(n, ts)
        self.ntp = _npow2(self.nt)
        self.w, self.wr = w, wr
        i32, i64, f16 = torch.int32, torch.int64, torch.float16
        self.h1 = buf(tag + "h1", (b, self.nt, 256), i32, dev)
        self.h2 = buf(tag + "h2", (b, self.nt, 256), i32, dev)
        self.meta = buf(tag + "meta", (b, 8), i32, dev)
        self.g1 = buf(tag + "g1", (b, self.nt), i32, dev)
        self.gtpre = buf(tag + "gtpre", (b, self.nt), i32, dev)
        self.tpre = buf(tag + "tpre", (b, self.nt), i32, dev)
        self.gtk = buf(tag + "gtk", (b, kp), i32, dev)
        self.v = buf(tag + "v", (b, k), f16, dev)
        self.i = buf(tag + "i", (b, k), i64, dev)

    def __call__(self, x):
        b, n, k, ts, nt, ntp, kp = (self.b, self.n, self.k, self.ts, self.nt,
                                    self.ntp, self.kp)
        _sel_hist_hi[(nt, b)](x, self.h1, N=n, TS=ts, NT=nt, num_warps=self.w)
        _sel_red_hi[(b,)](self.h1, self.meta, self.g1, K=k, NT=nt, NTP=ntp,
                          num_warps=self.wr)
        _sel_hist_lo[(nt, b)](x, self.meta, self.h2, N=n, TS=ts, NT=nt,
                              num_warps=self.w)
        _sel_red_lo[(b,)](self.h2, self.meta, self.g1, self.gtpre, self.tpre,
                          self.gtk, K=k, NT=nt, NTP=ntp, KP=kp, num_warps=self.wr)
        _sel_place[(nt, b)](x, self.meta, self.gtpre, self.tpre, self.gtk,
                            self.v, self.i, N=n, TS=ts, NT=nt, K=k, KP=kp,
                            num_warps=self.w)
        _sel_order_gt[(b,)](x, self.meta, self.gtk, self.v, self.i, N=n, K=k,
                            KP=kp, num_warps=self.wr)
        return self.v, self.i


@triton.jit
def _add_bias(pre, BI, idx, m, BIAS: tl.constexpr):
    """The class head's bias epilogue, exactly where cudnn left it.

    ``F.conv2d(x, w, b)`` on these shapes is an *unbiased* nvjet GEMM followed by
    a separate broadcast ``add`` -- 9.6 us and 3 launches at b=1 to add 80 numbers
    to 1.3 MB -- and ``fp16(unbiased + bias)`` is bit-identical to the biased conv
    on every scored shape (``dev/bias_probe.py``, 6/6, max diff 0).  So the GEMM
    stays on cudnn (which is the constraint: it is the fp32 summation order that
    selection is sensitive to, not the epilogue) and only the add moves in here,
    where these logits are read anyway.
    """
    if BIAS == 1:
        return pre.to(tl.float32) + tl.load(BI + idx, mask=m, other=0.0).to(tl.float32)
    return pre.to(tl.float32)


@triton.jit
def _add_bias3(pre, B0, B1, B2, L0, L1, idx, m, BIAS: tl.constexpr):
    """``_add_bias`` when the rows of ``pre`` come from different levels."""
    if BIAS == 1:
        b0 = tl.load(B0 + idx, mask=m, other=0.0).to(tl.float32)
        b1 = tl.load(B1 + idx, mask=m, other=0.0).to(tl.float32)
        b2 = tl.load(B2 + idx, mask=m, other=0.0).to(tl.float32)
        # Unlike the score kernel, every one of these values is stored, so the
        # rounding cannot be deferred to a reduction.
        return _f16(pre.to(tl.float32) + tl.where(L0, b0, tl.where(L1, b1, b2)))
    return pre.to(tl.float32)


# ---------------------------------------------------------------------------
# Kernel 1: per-anchor selection score, with the class head's 1x1 conv folded
# in.
#
# ``pid(0)`` is the anchor tile so concurrently dispatched CTAs walk adjacent
# addresses, and the level is resolved from the tile id -- which is what lets
# the three per-level tensors (and the three per-level head weights) be read in
# place instead of being cat'd into a packed [b, 144, 8400].  Because the level
# is a scalar the branch is real, so each arm is one GEMM, not three.
# ---------------------------------------------------------------------------
@triton.jit
def _score_kernel(C0, C1, C2, W0, W1, W2, A0, A1, A2, S,
                  NC: tl.constexpr, NCP: tl.constexpr,
                  CI: tl.constexpr, CIP: tl.constexpr,
                  P0: tl.constexpr, P1: tl.constexpr, P2: tl.constexpr,
                  T0: tl.constexpr, T1: tl.constexpr,
                  BLOCK: tl.constexpr, FUSE: tl.constexpr,
                  BIAS: tl.constexpr):
    t = tl.program_id(0)
    boff = tl.program_id(1).to(tl.int64)
    o = tl.arange(0, BLOCK)
    ci = tl.arange(0, CIP)[None, :]
    co = tl.arange(0, NCP)[None, :]
    mco = co < NC
    if t < T0:
        a = t * BLOCK + o
        m = a < P0
        pre = tl.load(C0 + boff * (CI * P0) + ci * P0 + a[:, None],
                      mask=m[:, None] & (ci < CI), other=0.0)
        logit = (_head1x1(pre, W0, A0, co, mco, CI, CIP) if FUSE
                 else _add_bias(pre, A0, ci, ci < CI, BIAS))
        g = a
    elif t < T1:
        a = (t - T0) * BLOCK + o
        m = a < P1
        pre = tl.load(C1 + boff * (CI * P1) + ci * P1 + a[:, None],
                      mask=m[:, None] & (ci < CI), other=0.0)
        logit = (_head1x1(pre, W1, A1, co, mco, CI, CIP) if FUSE
                 else _add_bias(pre, A1, ci, ci < CI, BIAS))
        g = P0 + a
    else:
        a = (t - T1) * BLOCK + o
        m = a < P2
        pre = tl.load(C2 + boff * (CI * P2) + ci * P2 + a[:, None],
                      mask=m[:, None] & (ci < CI), other=0.0)
        logit = (_head1x1(pre, W2, A2, co, mco, CI, CIP) if FUSE
                 else _add_bias(pre, A2, ci, ci < CI, BIAS))
        g = P0 + P1 + a
    # Selection only reads scores.amax(-1), and fp16(sigmoid(.)) is monotone
    # non-decreasing, so the max may be taken on the *logits* and rounded once:
    # bit-identical to the reference's 672k sigmoids, at 1/80th of the work.
    # fp16 rounding is monotone non-decreasing, so max_c fp16(z_c + b_c) is
    # fp16(max_c (z_c + b_c)): the bias epilogue's rounding can be done *once* on
    # the reduced value instead of on all NC of them, which is bit-identical and
    # is what keeps absorbing the epilogue cheaper than the launch it deletes.
    acc = _f16(tl.max(tl.where(mco, logit, float("-inf")), 1))
    tl.store(S + boff * (P0 + P1 + P2) + g,
             _sigmoid_rn(acc).to(tl.float16), mask=m)


@triton.jit
def _gather3(C0, C1, C2, boff, A, M0, M1, M2, CH,
             P0: tl.constexpr, P1: tl.constexpr, P2: tl.constexpr,
             NCH: tl.constexpr):
    """``X[b, ch, a]`` for global anchor ids that may straddle levels.

    A tile of *selected* anchors is unordered, so the level cannot be hoisted to
    a scalar branch as in ``_score_kernel``; all three loads are issued
    predicated on the level mask instead.  Two of the three are dead per
    element, which costs issue slots on 24k (class) / 19k (box) elements -- below
    the measurement grain, and it keeps the packed tensor unmaterialized.
    """
    v0 = tl.load(C0 + boff * (NCH * P0) + CH * P0 + A, mask=M0, other=0.0)
    v1 = tl.load(C1 + boff * (NCH * P1) + CH * P1 + (A - P0), mask=M1, other=0.0)
    v2 = tl.load(C2 + boff * (NCH * P2) + CH * P2 + (A - P0 - P1), mask=M2, other=0.0)
    return tl.where(M0, v0, tl.where(M1, v1, v2))


@triton.jit
def _lvl3(pre, W0, W1, W2, A0, A1, A2, L0, L1, CO, MCO,
          CI: tl.constexpr, CIP: tl.constexpr):
    """``_head1x1`` when the rows of ``pre`` come from different levels.

    Each level has its own head conv, and a tile of selected anchors mixes
    levels, so all three GEMMs are evaluated and selected per row.  Three
    [rows, 128] x [128, cols] dots on 300 rows is far below the launch it saves.
    """
    y0 = _head1x1(pre, W0, A0, CO, MCO, CI, CIP)
    y1 = _head1x1(pre, W1, A1, CO, MCO, CI, CIP)
    y2 = _head1x1(pre, W2, A2, CO, MCO, CI, CIP)
    return tl.where(L0, y0, tl.where(L1, y1, y2))


# ---------------------------------------------------------------------------
# Kernel 2: the NC class scores of the K selected anchors, laid out exactly as
# the reference's ``scores.flatten(1)`` (row = selection rank, col = class) so
# the second top-k sees identical values *and* identical flat indices -- which
# is what makes its lowest-index tie-break reproduce the reference's.
# ---------------------------------------------------------------------------
@triton.jit
def _sel_score_kernel(C0, C1, C2, W0, W1, W2, A0, A1, A2, IDX, G,
                      NC: tl.constexpr, NCP: tl.constexpr,
                      CI: tl.constexpr, CIP: tl.constexpr,
                      P0: tl.constexpr, P1: tl.constexpr, P2: tl.constexpr,
                      K: tl.constexpr, BLOCK_R: tl.constexpr,
                      FUSE: tl.constexpr, BIAS: tl.constexpr):
    r = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    boff = tl.program_id(1).to(tl.int64)
    mr = r < K
    a = tl.load(IDX + boff * K + r, mask=mr, other=0).to(tl.int32)
    ci = tl.arange(0, CIP)[None, :]
    a2 = a[:, None]
    l0 = (a2 < P0) & mr[:, None]
    l1 = (a2 >= P0) & (a2 < P0 + P1) & mr[:, None]
    l2 = (a2 >= P0 + P1) & mr[:, None]
    keep = ci < CI
    pre = _gather3(C0, C1, C2, boff, a2, l0 & keep, l1 & keep, l2 & keep,
                   ci, P0, P1, P2, CI)
    co = tl.arange(0, NCP)[None, :]
    mco = co < NC
    logit = (_lvl3(pre, W0, W1, W2, A0, A1, A2, l0, l1, co, mco, CI, CIP)
             if FUSE else _add_bias3(pre, A0, A1, A2, l0, l1, ci, keep, BIAS))
    tl.store(G + boff * (K * NC) + r[:, None] * NC + co,
             _sigmoid_rn(logit).to(tl.float16), mask=mr[:, None] & mco)


@triton.jit
def _dfl(pre, W0, W1, W2, A0, A1, A2, DW, L0, L1, g: tl.constexpr,
         CI: tl.constexpr, CIP: tl.constexpr, NB: tl.constexpr,
         RM: tl.constexpr):
    """One DFL group, with the box head's 1x1 conv folded in.

    The reference runs the 1x1 over all 8400 anchors, then a ``Softmax`` over
    the 16 bins that materializes an fp16 probability tensor, then a 1x1 conv
    with weight ``arange(16)`` that re-reads it -- three passes over ~4 MB for a
    result only 300 anchors need.  Here the 1x1 lands on the survivors and the
    softmax + expectation collapse into one reduction with no intermediate.  The
    probabilities are still narrowed to fp16 before the weighted sum, because
    the reference's ``Softmax`` writes fp16 and that rounding is visible in the
    box coordinates it produces.
    """
    t = tl.arange(0, RM)[None, :]
    ch = g * RM + t
    z = _lvl3(pre, W0, W1, W2, A0, A1, A2, L0, L1, ch, ch < NB, CI, CIP)
    e = tl.exp(z - tl.max(z, 1)[:, None])
    p = _f16(libdevice.div_rn(e, tl.sum(e, 1)[:, None]))
    # The expectation weight is *read*, not assumed to be ``arange(reg_max)``:
    # that is what ``YOLODFL``'s 1x1 conv holds, and reading it costs 16
    # L1-resident loads while removing the one place this kernel would silently
    # disagree with a reference whose DFL weight is anything else.
    return _f16(tl.sum(p * tl.load(DW + t).to(tl.float32), 1))


# ---------------------------------------------------------------------------
# Kernel 3: everything downstream of the second top-k, for the K output rows.
#
# ``_f16`` after every arithmetic step is not defensive: it *is* the reference,
# which runs dist2bbox / *strides / xywh2xyxy in fp16.  ``dist2bbox(xywh=True)``
# followed by ``xywh2xyxy`` is algebraically the identity and the stride is a
# uniform scale, so the whole path collapses to ``((anchor-lt)*s, (anchor+rb)*s)``
# -- and it is deliberately *not* collapsed, because the reference's final
# ``c - w/2`` cancels two O(500) fp16 values and so carries ~0.5 px of rounding
# that the collapsed form does not.  Against a reference that noisy the exact
# answer is the wrong answer: near a zero coordinate the gap exceeds
# ``atol=1e-2``.
# ---------------------------------------------------------------------------
@triton.jit
def _head_kernel(B0, B1, B2, W0, W1, W2, A0, A1, A2, DW, IDX, J2, V2,
                 ANCH, STR, OUT,
                 NA: tl.constexpr, NB: tl.constexpr, RM: tl.constexpr,
                 NC: tl.constexpr, CI: tl.constexpr, CIP: tl.constexpr,
                 P0: tl.constexpr, P1: tl.constexpr, P2: tl.constexpr,
                 K: tl.constexpr, BLOCK_R: tl.constexpr):
    r = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    boff = tl.program_id(1).to(tl.int64)
    mr = r < K
    j = tl.load(J2 + boff * K + r, mask=mr, other=0).to(tl.int32)
    sc = tl.load(V2 + boff * K + r, mask=mr, other=0)
    pos = j // NC
    lab = j % NC
    a = tl.load(IDX + boff * K + pos, mask=mr, other=0).to(tl.int32)

    ci = tl.arange(0, CIP)[None, :]
    a2 = a[:, None]
    l0 = (a2 < P0) & mr[:, None]
    l1 = (a2 >= P0) & (a2 < P0 + P1) & mr[:, None]
    l2 = (a2 >= P0 + P1) & mr[:, None]
    keep = ci < CI
    pre = _gather3(B0, B1, B2, boff, a2, l0 & keep, l1 & keep, l2 & keep,
                   ci, P0, P1, P2, CI)
    d0 = _dfl(pre, W0, W1, W2, A0, A1, A2, DW, l0, l1, 0, CI, CIP, NB, RM)
    d1 = _dfl(pre, W0, W1, W2, A0, A1, A2, DW, l0, l1, 1, CI, CIP, NB, RM)
    d2 = _dfl(pre, W0, W1, W2, A0, A1, A2, DW, l0, l1, 2, CI, CIP, NB, RM)
    d3 = _dfl(pre, W0, W1, W2, A0, A1, A2, DW, l0, l1, 3, CI, CIP, NB, RM)

    ax = tl.load(ANCH + a, mask=mr, other=0.0).to(tl.float32)
    ay = tl.load(ANCH + NA + a, mask=mr, other=0.0).to(tl.float32)
    s = tl.load(STR + a, mask=mr, other=0.0).to(tl.float32)

    x1 = _f16(ax - d0)
    y1 = _f16(ay - d1)
    x2 = _f16(ax + d2)
    y2 = _f16(ay + d3)
    cx = _f16(_f16(x1 + x2) * 0.5)
    cy = _f16(_f16(y1 + y2) * 0.5)
    bw = _f16(x2 - x1)
    bh = _f16(y2 - y1)
    cx = _f16(cx * s)
    cy = _f16(cy * s)
    bw = _f16(_f16(bw * s) * 0.5)
    bh = _f16(_f16(bh * s) * 0.5)

    o = OUT + boff * (K * 6) + r * 6
    tl.store(o + 0, _f16(cx - bw).to(tl.float16), mask=mr)
    tl.store(o + 1, _f16(cy - bh).to(tl.float16), mask=mr)
    tl.store(o + 2, _f16(cx + bw).to(tl.float16), mask=mr)
    tl.store(o + 3, _f16(cy + bh).to(tl.float16), mask=mr)
    tl.store(o + 4, sc, mask=mr)
    tl.store(o + 5, lab.to(tl.float16), mask=mr)


# ---------------------------------------------------------------------------
# Depthwise 3x3 Conv-BN-SiLU, one launch.
#
# The direction's premise -- "the conv towers already run on tuned L2 YOLOConv
# kernels" -- holds for 6 of the 12 tower blocks.  The other 6 are the
# ``g=c1`` depthwise blocks in ``cv3``, which the L2 winner's gate explicitly
# excludes, so each of them runs as three composed eager ops (F.conv2d ->
# F.batch_norm -> F.silu).  Measured 52 us of CPU per block against 27 us for a
# fused one, i.e. ~240 us of the 1.27 ms candidate -- the single largest
# remaining term after the tail, and bigger than the whole tail.
#
# Fusing it is a launch play, not a FLOP play: 9 taps on a plane of at most
# 6400 pixels is nothing.  So the kernel is shaped for simplicity and for
# *matching the reference bit-for-bit as closely as a different accumulation
# order allows*: the fp32 accumulator is narrowed to fp16 after the conv, the
# BN is evaluated in the reference's ``(x - mean) * invstd * w + b`` form (not
# the algebraically equal ``x * scale + shift``) and narrowed again, and only
# then is SiLU applied -- three roundings, exactly like the composed path.
# Collapsing them to one rounding would be more accurate and is what the L2
# block does, but this composite's top-k selection keys on fp16 sigmoid values
# whose ties span thousands of anchors, so *matching* the reference is worth
# more here than being closer to the real number.
# ---------------------------------------------------------------------------
@triton.jit
def _dw3_bn_act_kernel(X, WT, BW, BB, BM, BV, Y, eps,
                       C: tl.constexpr, H: tl.constexpr, IW: tl.constexpr,
                       P: tl.constexpr, ACT: tl.constexpr, BLOCK: tl.constexpr):
    nc = tl.program_id(1)
    c = nc % C
    p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mp = p < P
    oh = p // IW
    ow = p % IW
    xp = X + nc.to(tl.int64) * P
    acc = tl.zeros((BLOCK,), tl.float32)
    for i in tl.static_range(3):
        ih = oh + i - 1
        mh = mp & (ih >= 0) & (ih < H)
        for j in tl.static_range(3):
            iw = ow + j - 1
            v = tl.load(xp + ih * IW + iw,
                        mask=mh & (iw >= 0) & (iw < IW), other=0.0).to(tl.float32)
            acc += v * tl.load(WT + c * 9 + i * 3 + j).to(tl.float32)
    y = acc.to(tl.float16).to(tl.float32)
    mean = tl.load(BM + c).to(tl.float32)
    var = tl.load(BV + c).to(tl.float32)
    invstd = libdevice.div_rn(1.0, libdevice.sqrt(var + eps))
    y = (y - mean) * invstd * tl.load(BW + c).to(tl.float32) + tl.load(BB + c).to(tl.float32)
    y = y.to(tl.float16).to(tl.float32)
    if ACT == 1:
        y = y * libdevice.div_rn(1.0, 1.0 + tl.exp(-y))
    tl.store(Y + nc.to(tl.int64) * P + p, y.to(tl.float16), mask=mp)


def _dw_act_code(act) -> int | None:
    if isinstance(act, (SiLU, nn.SiLU)):
        return 1
    if isinstance(act, nn.Identity):
        return 0
    return None


def _dw_plan(m, x: torch.Tensor):
    """A prebound launcher for a depthwise 3x3 Conv-BN-Act block, or None.

    The gate below is ~10 us of Python -- as much as the launch it guards, on a
    module where *every* op costs ~24 us of CPU and the GPU is idle waiting.  So
    it runs once per (module, input shape) and what survives is a closure that
    does one ``empty_like`` and one launch.  ``conv``/``bn`` are captured as
    modules, not as tensors, so an in-place weight update *or* a wholesale
    ``Parameter`` replacement is still picked up on the next call.
    """
    if getattr(m, "_is_fused", True) or not (x.is_cuda and x.dtype == torch.float16):
        return None
    if not x.is_contiguous() or x.dim() != 4:
        return None
    conv, bn = m.conv, getattr(m, "bn", None)
    if bn is None or bn.training or not bn.affine or not bn.track_running_stats:
        return None
    if bn.running_mean is None or bn.running_var is None:
        return None
    w = conv.weight
    c = x.shape[1]
    if (conv.groups != c or conv.bias is not None or w.shape[0] != c or w.shape[1] != 1
            or tuple(w.shape[2:]) != (3, 3) or tuple(conv.stride) != (1, 1)
            or tuple(conv.padding) != (1, 1) or tuple(conv.dilation) != (1, 1)):
        return None
    act = _dw_act_code(m.act)
    if act is None or not w.is_contiguous():
        return None
    n, _, h, iw = x.shape
    p = h * iw
    eps = float(bn.eps)
    grid = (triton.cdiv(p, 256), n * c)
    kern = _dw3_bn_act_kernel

    def run(t, conv=conv, bn=bn, grid=grid, c=c, h=h, iw=iw, p=p, act=act,
            eps=eps, kern=kern):
        y = torch.empty_like(t)
        kern[grid](t, conv.weight, bn.weight, bn.bias, bn.running_mean,
                   bn.running_var, y, eps,
                   C=c, H=h, IW=iw, P=p, ACT=act, BLOCK=256, num_warps=4)
        return y

    return run


def _leaves(seq):
    out = []
    for m in seq:
        if isinstance(m, nn.Sequential):
            out.extend(_leaves(m))
        else:
            out.append(m)
    return out


def _head_conv(m, x: torch.Tensor):
    """``m`` if it is a plain 1x1 conv with bias, i.e. a pure per-anchor GEMM."""
    if not isinstance(m, (Conv2d, nn.Conv2d)):
        return None
    w, bias = m.weight, m.bias
    if bias is None or w.dtype != torch.float16 or not w.is_contiguous():
        return None
    if (getattr(m, "groups", 1) != 1 or tuple(w.shape[2:]) != (1, 1)
            or tuple(m.stride) != (1, 1) or tuple(m.padding) != (0, 0)
            or tuple(m.dilation) != (1, 1) or w.shape[1] != x.shape[1]):
        return None
    return m


def _plan_leaves(lv, x: torch.Tensor):
    """Flatten a tower's leaves into a list of callables.

    Walking the nested ``Sequential`` with ``isinstance`` checks per layer cost
    ~166 us across the six towers -- 7 us per layer of pure dispatch on a 1 ms
    module.  The walk happens once; steady state is ``for f in steps: x = f(x)``
    over prebound callables, which is the cheapest form a data-dependent chain
    can take in Python.
    """
    steps = []
    for m in lv:
        run = _dw_plan(m, x) if isinstance(m, YOLOConv) else None
        if run is None:
            run = m
        steps.append(run)
        x = run(x)
    return steps, x


def _plan_tower(seq, x: torch.Tensor):
    """``(steps, pre_activation, head_1x1_conv)`` or ``None``.

    The box tower ends in a bias-only 1x1 conv -- a pure per-anchor GEMM.  Left in
    the tower it costs a launch *and* materializes ``[b, 4*reg_max, 8400]`` that
    the tail then reads for 300 of 8400 anchors.  So it is handed to the tail
    kernels, which fold it into their own loads.
    """
    lv = _leaves(seq)
    if len(lv) < 2:
        return None
    steps, x = _plan_leaves(lv[:-1], x)
    hc = _head_conv(lv[-1], x)
    if hc is None:
        return None
    return steps, x, hc


def _plan_cls(seq, x: torch.Tensor):
    """``(steps, raw_logits, head_1x1_conv)`` for the *class* tower, or ``None``.

    Unlike the box head, the class head's GEMM is deliberately left on cudnn --
    computing it with ``tl.dot`` changes the fp32 summation order and flips ~0.1%
    of logits by one ulp, which is enough to reshuffle selection on a weight draw
    whose scores are spread (r1 iters 07/08).  What moves is only its *bias*: the
    conv runs unbiased (byte for byte the same GEMM cudnn already ran) and the
    two tail kernels add the bias where they read the logits, which is exact and
    saves the 9.6 us / 3-launch broadcast ``add`` PyTorch spends on it.
    """
    lv = _leaves(seq)
    if len(lv) < 2:
        return None
    steps, x = _plan_leaves(lv[:-1], x)
    hc = _head_conv(lv[-1], x)
    if hc is None:
        return None
    if not _CLS_BIAS:
        steps.append(hc)
        return steps, hc(x), hc
    w = hc.weight

    def run(t, w=w):
        return F.conv2d(t, w, None)

    steps.append(run)
    return steps, run(x), hc


class _TorchSelect:
    """``torch.topk`` behind the same interface, for A/B and as the fallback."""

    __slots__ = ("k", "v", "i")

    def __init__(self, b, n, k, ts, dev, buf, tag, **kw):
        self.k = k
        self.v = buf(tag + "v", (b, k), torch.float16, dev)
        self.i = buf(tag + "i", (b, k), torch.int64, dev)

    def __call__(self, x):
        torch.topk(x, self.k, dim=-1, out=(self.v, self.i))
        return self.v, self.i


_SEL_TS = int(os.environ.get("FK_SEL_TS", "1024"))
_SEL_WARPS = int(os.environ.get("FK_SEL_WARPS", "4"))
_TORCH_TOPK = os.environ.get("FK_TORCH_TOPK", "0") == "1"


def _mk_select(b, n, k, dev, buf, tag):
    """The exact selection for a ``[b, n]`` fp16 row.

    The packed key needs ``n <= 65536`` (16 bits of index) and ``k`` no larger
    than the compaction buffer; outside that, ``torch.topk`` -- which is what
    this replaces and is exact by definition -- still stands in.
    """
    kp = _npow2(max(int(k), 2))
    if _TORCH_TOPK or n > 65536 or k > kp or k > n:
        return _TorchSelect(b, n, k, _SEL_TS, dev, buf, tag)
    return _Select(b, n, k, _SEL_TS, dev, buf, tag, w=_SEL_WARPS, kp=kp)


class YOLOv10DetectHead(nn.Module):
    dynamic = False
    export = True
    shape = None
    max_det = 300

    def __init__(self, nc: int = 80, ch: tuple[int, int, int] = (256, 512, 1024)):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = 16
        self.no = nc + self.reg_max * 4
        self.stride = torch.tensor([8.0, 16.0, 32.0])
        c2 = max((16, ch[0] // 4, self.reg_max * 4))
        c3 = max(ch[0], min(self.nc, 100))
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                YOLOConv(x, c2, 3),
                YOLOConv(c2, c2, 3),
                Conv2d(c2, 4 * self.reg_max, 1),
            )
            for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(YOLOConv(x, x, 3, g=x), YOLOConv(x, c3, 1)),
                nn.Sequential(YOLOConv(c3, c3, 3, g=c3), YOLOConv(c3, c3, 1)),
                Conv2d(c3, self.nc, 1),
            )
            for x in ch
        )
        self.dfl = YOLODFL(self.reg_max)
        self._sigmoid = Sigmoid()
        self.one2one_cv2 = copy.deepcopy(self.cv2)
        self.one2one_cv3 = copy.deepcopy(self.cv3)
        self.register_buffer("anchors", torch.empty(0))
        self.register_buffer("strides", torch.empty(0))
        self._scratch: dict = {}
        self._plan: tuple | None = None

    def forward_feat(self, x: list[torch.Tensor], cv2, cv3):
        y = []
        for i in range(self.nl):
            y.append(torch.cat((cv2[i](x[i]), cv3[i](x[i])), 1))
        return y

    # -- anchors ------------------------------------------------------------
    def _sync_anchors(self, feats: list[torch.Tensor]) -> None:
        """Anchors depend only on feature-map HW.

        Keyed on ``(h, w)`` of the finest level, not on the full BCHW: keying on
        the batch too rebuilt them on every batch-size change and freed the
        buffers a CUDA graph had captured.  On a rebuild the existing storage is
        filled in place whenever the element count matches, so the buffers are
        never rebound after the first call.
        """
        _, _, h, w = feats[0].shape
        spatial = (h, w)
        if self.dynamic or self.shape != spatial:
            anchors, strides = (t.transpose(0, 1).contiguous()
                                for t in make_anchors(feats, self.stride, 0.5))
            if self.anchors.numel() == anchors.numel() and self.strides.numel() == strides.numel():
                self.anchors.copy_(anchors)
                self.strides.copy_(strides)
            else:
                self.register_buffer("anchors", anchors)
                self.register_buffer("strides", strides)
            self.shape = spatial

    def inference(self, x: list[torch.Tensor]):
        b, _, h, w = x[0].shape
        x_cat = torch.cat([xi.view(b, self.no, -1) for xi in x], 2)
        self._sync_anchors(x)
        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        return torch.cat((dbox, self._sigmoid(cls)), 1)

    # -- fused export path --------------------------------------------------
    def _make_plan(self, xs):
        """Everything shape-derived, computed once.

        Returns the two tower programs, the box head's trailing 1x1 conv per
        level, and a flat tuple of launch geometry + scratch tensors -- or
        ``None`` to record that this shape is not on the fast path, so the gate is
        not re-run per call.  Building the plan runs the towers once, which also
        warms the Triton compiles before any capture.

        **Only the box head's 1x1 is folded into the tail; the class head's is
        not** (``FUSE=0``), even though folding it measured 1.2x faster.  The
        class logits decide the two top-k selections, and with thousand-anchor
        fp16 ties a single differing logit re-orders the whole tie group rather
        than one row.  Computing that 1x1 with ``tl.dot`` instead of cudnn
        changes the fp32 summation order, which flips ~0.1% of logits by one
        ulp -- harmless on a weight draw whose score field is bunched, but on a
        draw whose scores spread across many fp16 levels it reshuffles the
        output and the case fails (measured: `iter 07`, b=4 matched 0.9218).
        The box path has no such sensitivity -- it only feeds coordinates that
        are compared with rtol=1e-2 -- so its 1x1 stays folded, which also drops
        it from 8400 anchors to 300.
        """
        if self.dynamic or self.nl != 3 or self.reg_max != 16 or self.max_det < 1:
            return None
        dw = getattr(getattr(self.dfl, "conv", None), "weight", None)
        if (dw is None or dw.numel() != self.reg_max or dw.dtype != torch.float16
                or not dw.is_contiguous()):
            return None
        for t in xs:
            if not (t.is_cuda and t.dtype == torch.float16 and t.is_contiguous()
                    and t.dim() == 4):
                return None
        nc, rm, k = self.nc, self.reg_max, self.max_det
        steps2, steps3, hb, hc, outs = [], [], [], [], []
        for i in range(3):
            r2 = _plan_tower(self.one2one_cv2[i], xs[i])
            if r2 is None:
                return None
            s2, pb, cb = r2
            # cv3's trailing 1x1 is deliberately NOT folded -- see _make_plan's
            # note above the FUSE flag.  Only its bias moves into the tail.
            r3 = _plan_cls(self.one2one_cv3[i], xs[i])
            if r3 is None:
                return None
            s3, pc, cc = r3
            if not (pb.is_contiguous() and pc.is_contiguous()
                    and pb.dtype == torch.float16 and pc.dtype == torch.float16
                    and pb.shape[0] == pc.shape[0] and pb.shape[2:] == pc.shape[2:]
                    and cb.weight.shape[0] == 4 * rm and pc.shape[1] == nc):
                return None
            steps2.append(s2)
            steps3.append(s3)
            hb.append(cb)
            hc.append(cc)
            outs.append(pc)
        # one CI per kernel, so all three levels must agree on the head's fan-in
        cib = hb[0].weight.shape[1]
        if any(m.weight.shape[1] != cib for m in hb):
            return None
        self._sync_anchors(outs)
        b = xs[0].shape[0]
        p0, p1, p2 = (o.shape[2] * o.shape[3] for o in outs)
        na = p0 + p1 + p2
        if k > na or k > na * nc or self.anchors.numel() != 2 * na:
            return None
        dev = xs[0].device
        blk = 128
        t0 = triton.cdiv(p0, blk)
        t1 = t0 + triton.cdiv(p1, blk)
        nt = t1 + triton.cdiv(p2, blk)
        pp = (
            (nt, b),
            (nc, _npow2(nc), nc, _npow2(nc), p0, p1, p2, t0, t1, blk, 0,
             int(_CLS_BIAS)),
            (triton.cdiv(k, 32), b),
            (nc, _npow2(nc), nc, _npow2(nc), p0, p1, p2, k, 32, 0,
             int(_CLS_BIAS)),
            (triton.cdiv(k, 32), b),
            (na, 4 * rm, rm, nc, cib, _npow2(cib), p0, p1, p2, k, 32),
            self._buf("score", (b, na), torch.float16, dev),
            self._buf("sel", (b, k * nc), torch.float16, dev),
            _mk_select(b, na, k, dev, self._buf, "s1_"),
            _mk_select(b, k * nc, k, dev, self._buf, "s2_"),
            k, b, dev, hb, dw, hc,
        )
        return steps2, steps3, pp

    def _run(self, xs, plan):
        steps2, steps3, pp = plan
        (gs, cs, gg, cg, gh, ch, score, sel, sel1, sel2, k, b, dev, hb, dw,
         hc) = pp

        b0 = xs[0]
        for f in steps2[0]:
            b0 = f(b0)
        c0 = xs[0]
        for f in steps3[0]:
            c0 = f(c0)
        b1 = xs[1]
        for f in steps2[1]:
            b1 = f(b1)
        c1 = xs[1]
        for f in steps3[1]:
            c1 = f(c1)
        b2 = xs[2]
        for f in steps2[2]:
            b2 = f(b2)
        c2 = xs[2]
        for f in steps3[2]:
            c2 = f(c2)

        # class-head weights are unused (FUSE=0); a0/a1/a2 carry its bias, which
        # the two tail kernels apply where they read the logits.
        w0 = w1 = w2 = self.strides
        a0, a1, a2 = hc[0].bias, hc[1].bias, hc[2].bias
        _score_kernel[gs](c0, c1, c2, w0, w1, w2, a0, a1, a2, score, *cs,
                          num_warps=4)
        i1 = sel1(score)[1]
        _sel_score_kernel[gg](c0, c1, c2, w0, w1, w2, a0, a1, a2, i1, sel, *cg,
                              num_warps=4)
        v2, j2 = sel2(sel)
        out = torch.empty((b, k, 6), dtype=torch.float16, device=dev)
        _head_kernel[gh](b0, b1, b2, hb[0].weight, hb[1].weight, hb[2].weight,
                         hb[0].bias, hb[1].bias, hb[2].bias, dw, i1, j2, v2,
                         self.anchors, self.strides, out, *ch, num_warps=4)
        return out

    def _try_capture(self, xs, plan):
        """Capture the export path into one graph, or return ``None``.

        With the tail fused and the tower glue hoisted, what is left is the
        enqueue itself: ~24 us of CPU per op on this host against single-digit-us
        kernels, and the GPU idle in between (measured: host enqueue time and
        device span agree to 0.1%).  No amount of further kernel work removes a
        launch's *CPU* cost; a graph does.

        It refuses to capture, and stays on the eager path, when the caller is
        already capturing -- so an engine that graphs the whole model still
        works, which is the case this module has to be safe for and where a
        nested capture would be an error rather than a slowdown.  Everything a
        replay touches is address-stable by construction: the anchor/stride
        buffers are filled in place and never rebound, the tail scratch is
        allocated once per shape, and weights are read through their modules, so
        an in-place weight update is picked up without recapture.
        """
        if _NO_GRAPH:
            return None
        try:
            if torch.cuda.is_current_stream_capturing():
                return None
            inp = [torch.empty_like(t) for t in xs]
            for d, sc in zip(inp, xs):
                d.copy_(sc)
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    self._run(inp, plan)
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                out = self._run(inp, plan)
            return (g, inp, out)
        except Exception:
            return None

    def _forward_export(self, x):
        xs = [xi.detach() for xi in x] if torch.is_grad_enabled() else x
        key = (xs[0].shape, xs[1].shape, xs[2].shape, xs[0].dtype, xs[0].device)
        plan = self._plan
        if plan is None or plan[0] != key:
            p = self._make_plan(xs)
            plan = self._plan = (key, p, self._try_capture(xs, p) if p else None)
        if plan[1] is None:
            return None
        cap = plan[2]
        if cap is not None and not torch.cuda.is_current_stream_capturing():
            # A replay cannot be recorded into an enclosing capture ("Cannot
            # prepare for replay during capturing stage"), so when an engine is
            # graphing the whole model we hand it the individual launches
            # instead -- which is the outcome it wants anyway.
            g, inp, out = cap
            inp[0].copy_(xs[0])
            inp[1].copy_(xs[1])
            inp[2].copy_(xs[2])
            g.replay()
            # The replay writes a buffer the graph owns; hand back a copy so the
            # returned tensor keeps ordinary value semantics across calls.
            return out.clone()
        return self._run(xs, plan[1])

    def _buf(self, key, shape, dtype, device):
        t = self._scratch.get(key)
        if t is None or t.shape != shape or t.dtype != dtype or t.device != device:
            t = torch.empty(shape, dtype=dtype, device=device)
            self._scratch[key] = t
        return t

    def forward(self, x: list[torch.Tensor]):
        if not self.training and self.export:
            out = self._forward_export(x)
            if out is not None:
                return out
            one2one = self.inference(self.forward_feat(
                [xi.detach() for xi in x], self.one2one_cv2, self.one2one_cv3))
            bb, sc, lb = v10postprocess(one2one.permute(0, 2, 1), self.max_det, self.nc)
            return torch.cat([xywh2xyxy(bb), sc.unsqueeze(-1),
                              lb.unsqueeze(-1).to(bb.dtype)], dim=-1)

        one2one = self.forward_feat([xi.detach() for xi in x], self.one2one_cv2, self.one2one_cv3)
        if not self.training:
            one2one = self.inference(one2one)
        one2many = self.forward_feat(x, self.cv2, self.cv3)
        if self.training:
            return {"one2many": one2many, "one2one": one2one}
        one2many = self.inference(one2many)
        return {"one2many": one2many, "one2one": one2one}

    def bias_init(self):
        for a, b, s in zip(self.cv2, self.cv3, self.stride):
            a[-1].bias.data[:] = 1.0
            b[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / s) ** 2)
        for a, b, s in zip(self.one2one_cv2, self.one2one_cv3, self.stride):
            a[-1].bias.data[:] = 1.0
            b[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / s) ** 2)
