"""Multi-head attention with bias list support for AlphaFold3 (L2).

Composes QKV projections + SDPA + gated output.

Reference: openfold3/core/model/primitives/attention.py Attention

Every captured shape is tiny (Q=16-32, K=16-128, C_hidden=24-48, H=4-16, bf16)
and the operator runs ~4.9k times, so wall time is set by per-call cost rather
than by FLOPs: the reference forward spends ~220 us of Python and dispatch to
launch ~15 us of GPU work, and an empty forward on the same inputs already
measures ~13 us.  Collapsing launches -- and the Python that issues them -- is
the whole optimization here; tuning tiles for throughput is close to irrelevant.

The reference pays ~15 launches per call: four projection GEMMs, a bmm, an add
per bias, a softmax, a cast, a second bmm, a sigmoid, a gating multiply and the
output projection, plus several materializing transposes and reshapes.  This
leaves one to three, chosen per shape.

**The attention program.**  One Triton program per (head, batch) does QK^T, the
bias adds, the softmax, PV, the sigmoid gate and the head-flatten.  K <= 128
everywhere, so a whole score row fits in registers: no flash-style online
softmax, no KV loop, and no ``[*, H, Q, K]`` score tensor ever reaches HBM.  The
``1/sqrt(c_hidden)`` scale, the head-split view and the -2/-3 transposes are
folded into kernel indexing (strides are arguments), so no permute is
materialized on either side.  This is the fusion that carried the win:
1.00x -> 5.44x.

**The projections.**  ``linear_q``/``linear_g`` share one cached ``[q|g]`` weight
and ``linear_k``/``linear_v`` one cached ``[k|v]`` weight, so four GEMMs become
two.  Either of those two can additionally be folded *into* the attention program
(``FUSE_Q`` / ``FUSE_KV``), removing its launch outright -- but only when the
serial GEMM chain that adds to each program is small enough, which is decided per
side in `_build_plan`.  On its own this fusion is worth far less than the
attention core (1.25x at the torch level), and it is worth ~1.4x on top of it,
once the launches it removes are what is left.

**The output projection.**  ``linear_o`` can go into the same launch too
(``FUSE_O``), which leaves the whole module in one.  It reduces over the entire
``H*C`` axis while a program only produced its own head's ``C`` columns of it, so
the ``H`` programs of a batch rendezvous through a device-scope flag array and
then re-partition the ``[Q x H*C] @ [H*C x c_q]`` GEMM by output column, program
``h`` taking columns ``[h*BN, (h+1)*BN)``.  The rendezvous is not free -- ~1.1 us
of device time against the 2.4-3.4 us ``nvjet_sm100`` launch it removes -- so
``linear_o`` is not simply a third candidate for the fusion budget; see
``_fuse_flags``.  What makes it pay at all is loading its weight tile *before*
publishing, so the cold-HBM latency of the one operand that depends on no other
program overlaps the rendezvous instead of following it.

A *separate* Triton launch for this GEMM is a different and worse proposition,
and a measured dead end: on B200 bf16 dispatches Blackwell-native ``nvjet_sm100``
kernels that a hand-written replacement lost to on four of five shapes even after
a 72-point tile sweep.  Fusing wins only because it adds no launch.

Biases arrive as an arbitrary-length list that merely broadcasts to
``[*, H, Q, K]``, so the host normalizes each one to four strides (zero on any
axis it broadcasts over) and the kernel is specialized on the bias count.  The
whole plan -- strides, tile shape, grid, fusion mode, scratch buffer -- is
memoized per (shape, stride, dtype) signature, so a steady-state forward is a
dict hit plus the launches.

The launch is then the dominant remaining cost, so each plan holds a pre-bound
handle to Triton's C launcher: ~5 us against ~16 us for ``kernel[grid](...)``,
by skipping argument binding, cache-key hashing and launch-metadata construction.
Pointer parameters are declared ``do_not_specialize_on_alignment`` so a pre-bound
kernel stays valid whatever the input alignment.

Numerics follow the reference: fp32 GEMM accumulation, bf16 q/k/v/g, fp32
softmax, bf16 probabilities before PV.  Worst error over the captured shapes is
9.8e-04, against a bf16 tolerance of atol = rtol = 1e-2.

One caveat about *why* the fusion decisions look the way they do.  The benchmark's
timing loop flushes twice the L2 before each timed region, which gives the CPU a
68 us head start on the GPU, and the two then finish within ~15% of each other on
these shapes.  The loop therefore sits at the crossover rather than deep in either
regime: each shape has a few microseconds of CPU slack in which removing host work
buys literally nothing, and past it device time is what shows.  That is why the
thresholds here are calibrated per shape on measured time rather than derived from
launch counts, and why a fusion that plainly removes host work can still be a
regression.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

from ..L1.linear import Linear
from ..L1.softmax import Softmax


# ---------------------------------------------------------------------------
# The module in one launch:
#   q/k/v/g projections -> scores -> biases -> softmax -> PV -> gate -> flatten
#   -> rendezvous -> output projection
# Each of the three GEMM groups is independently in or out; see `_fuse_flags`.
# ---------------------------------------------------------------------------
@triton.jit(do_not_specialize_on_alignment=["OUT", "QX", "KVX", "WQG", "WKV",
                                            "QB", "B0", "B1", "B2", "WO",
                                            "RES", "FLAG"])
def _attn(OUT, QX, KVX, WQG, WKV, QB, B0, B1, B2, WO, RES, FLAG,
          sob, soq, sxb, sxq, syb, syq, swq, swk, GOFF, VOFF, CQ, CK,
          s0b, s0h, s0q, s0k, s1b, s1h, s1q, s1k, s2b, s2h, s2q, s2k,
          swo, srb, srq, HC,
          QN, KN, C, SCALE,
          BQ: tl.constexpr, BK: tl.constexpr, BC: tl.constexpr,
          BJ: tl.constexpr, NBIAS: tl.constexpr, HAS_G: tl.constexpr,
          HAS_QB: tl.constexpr, EVEN_K: tl.constexpr,
          FUSE_Q: tl.constexpr, FUSE_KV: tl.constexpr,
          FUSE_O: tl.constexpr, BN: tl.constexpr, BJO: tl.constexpr,
          BH: tl.constexpr, BHC: tl.constexpr):
    """One program per (head, batch): out[b, q, h*C + c] for every q, c.

    ``FUSE_Q``, ``FUSE_KV`` and ``FUSE_O`` independently fold a GEMM into the
    program; with all three set the module is a single launch.
    When set, ``QX``/``KVX`` is ``q_x``/``kv_x`` and the program multiplies it by
    the ``C`` weight rows of head ``h`` out of the cached ``[q|g]`` / ``[k|v]``
    weight (``[2*H*C, C_in]``, row stride ``swq``/``swk``, second half at element
    offset ``GOFF``/``VOFF``); when clear, ``QX``/``KVX`` is the corresponding
    pre-projected activation from cuBLAS and the program just loads its slice.
    With both set the whole module except ``linear_o`` is one launch.

    Fusing duplicates no FLOPs -- each head's weight rows are read by exactly one
    program -- but it does prepend a serial GEMM chain to every program, and with
    only B*H programs (16-64 here) there is nothing else on the SM to hide that
    chain behind.  The two sides are therefore budgeted separately by the host:
    at these shapes ``q_x`` is short (Q <= 32) while ``kv_x`` can be four times
    as long (K <= 128), so it is common for one side to be worth fusing and the
    other not.

    Either way ``q``/``k``/``v``/``g`` are rounded to the weight dtype at exactly
    the points the reference materializes them -- fp32 GEMM accumulation, bf16
    q/k/v/g, fp32 scores, fp32 softmax, bf16 probabilities, fp32 PV -- so the
    arithmetic matches.  Each bias carries its own four strides, zero on any
    axis it broadcasts over.
    """
    h = tl.program_id(0)
    b = tl.program_id(1)

    rq = tl.arange(0, BQ)
    rk = tl.arange(0, BK)
    rc = tl.arange(0, BC)
    mq = rq < QN
    mk = rk < KN
    mc = rc < C
    qm = mq[:, None] & mc[None, :]
    km = mk[:, None] & mc[None, :]

    if FUSE_Q or FUSE_KV:
        rj = tl.arange(0, BJ)
        wrow = (h * C + rc)[:, None]

    # --- q, g ---------------------------------------------------------------
    # q and g share the q_x tile (as k and v share kv_x), so a fused side loads
    # its input once and feeds it to two weight tiles: one pipelined loop per
    # side instead of one per projection, and half the activation traffic.  The
    # 1/sqrt(c_hidden) scale is folded into q's bf16 rounding when fused (the
    # reference divides an already-bf16 q) and applied to the scores otherwise.
    if FUSE_Q:
        xqp = QX + b * sxb + rq[:, None] * sxq
        wqp = WQG + wrow * swq
        accq = tl.zeros([BQ, BC], dtype=tl.float32)
        accg = tl.zeros([BQ, BC], dtype=tl.float32)
        for j0 in range(0, CQ, BJ):
            mj = (j0 + rj) < CQ
            x = tl.load(xqp + j0 + rj[None, :], mask=mq[:, None] & mj[None, :],
                        other=0.0)
            wm = mc[:, None] & mj[None, :]
            w = tl.load(wqp + j0 + rj[None, :], mask=wm, other=0.0)
            accq = tl.dot(x, tl.trans(w), accq)
            if HAS_G:
                w = tl.load(wqp + GOFF + j0 + rj[None, :], mask=wm, other=0.0)
                accg = tl.dot(x, tl.trans(w), accg)
        if HAS_QB:
            accq += tl.load(QB + h * C + rc, mask=mc,
                            other=0.0).to(tl.float32)[None, :]
        qt = (accq * SCALE).to(WQG.dtype.element_ty)
        gt = accg.to(WQG.dtype.element_ty)
    else:
        qp = QX + b * sxb + h * C + rq[:, None] * sxq + rc[None, :]
        qt = tl.load(qp, mask=qm, other=0.0)
        if HAS_G:
            gt = tl.load(qp + GOFF, mask=qm, other=0.0)

    # --- k, v ---------------------------------------------------------------
    if FUSE_KV:
        xkp = KVX + b * syb + rk[:, None] * syq
        wkp = WKV + wrow * swk
        acck = tl.zeros([BK, BC], dtype=tl.float32)
        accv = tl.zeros([BK, BC], dtype=tl.float32)
        for j0 in range(0, CK, BJ):
            mj = (j0 + rj) < CK
            x = tl.load(xkp + j0 + rj[None, :], mask=mk[:, None] & mj[None, :],
                        other=0.0)
            wm = mc[:, None] & mj[None, :]
            w = tl.load(wkp + j0 + rj[None, :], mask=wm, other=0.0)
            acck = tl.dot(x, tl.trans(w), acck)
            w = tl.load(wkp + VOFF + j0 + rj[None, :], mask=wm, other=0.0)
            accv = tl.dot(x, tl.trans(w), accv)
        kt = acck.to(WKV.dtype.element_ty)
        vt = accv.to(WKV.dtype.element_ty)
    else:
        kp = KVX + b * syb + h * C + rk[:, None] * syq + rc[None, :]
        kt = tl.load(kp, mask=km, other=0.0)
        vt = tl.load(kp + VOFF, mask=km, other=0.0)

    # --- scores ------------------------------------------------------------
    # Every tile this program reads from HBM has been issued by now -- q, g, k, v
    # and the biases -- and none of them depends on another, so their (cold, L2
    # having just been flushed by the benchmark) latencies overlap each other and
    # the two dots instead of stacking up.  Only the score dot has to wait, and
    # only on q and k.
    bm = mq[:, None] & mk[None, :]
    if NBIAS > 0:
        bb0 = tl.load(B0 + b * s0b + h * s0h + rq[:, None] * s0q
                      + rk[None, :] * s0k, mask=bm, other=0.0)
    if NBIAS > 1:
        bb1 = tl.load(B1 + b * s1b + h * s1h + rq[:, None] * s1q
                      + rk[None, :] * s1k, mask=bm, other=0.0)
    if NBIAS > 2:
        bb2 = tl.load(B2 + b * s2b + h * s2h + rq[:, None] * s2q
                      + rk[None, :] * s2k, mask=bm, other=0.0)

    s = tl.dot(qt, tl.trans(kt), out_dtype=tl.float32)
    if not FUSE_Q:
        s = s * SCALE
    if NBIAS > 0:
        s += bb0.to(tl.float32)
    if NBIAS > 1:
        s += bb1.to(tl.float32)
    if NBIAS > 2:
        s += bb2.to(tl.float32)

    # --- softmax over K (fp32), then cast to the value dtype for PV --------
    if not EVEN_K:
        s = tl.where(mk[None, :], s, float("-inf"))
    p = tl.exp(s - tl.max(s, 1)[:, None])
    p = (p / tl.sum(p, 1)[:, None]).to(OUT.dtype.element_ty)

    # --- o = p v -----------------------------------------------------------
    o = tl.dot(p, vt, out_dtype=tl.float32)

    # --- sigmoid gate + head-flattened store -------------------------------
    if HAS_G:
        o = o * tl.sigmoid(gt.to(tl.float32))
    tl.store(OUT + b * sob + h * C + rq[:, None] * soq + rc[None, :],
             o.to(OUT.dtype.element_ty), mask=qm)

    # --- linear_o, behind a grid barrier -----------------------------------
    # `linear_o` reduces over the whole H*C axis, but a program only produced
    # its own head's C columns of it.  The H programs of this batch therefore
    # rendezvous -- each publishes its slice of `OUT` with a release atomic on
    # its own flag word, then spins on an *acquire* atomic until every flag in
    # the group has reached the same launch count.  The group is one batch, not
    # the whole grid: nothing here needs another batch's rows.
    #
    # The spin has to be the acquire atomic itself.  A `tl.load(..., volatile=
    # True)` is cheaper but PTX only orders a volatile against other volatiles,
    # so the `OUT` loads below could be hoisted above the wait -- and a trailing
    # `tl.atomic_add(..., 0, sem="acquire")` to fence them does not survive: it
    # adds zero, so LLVM deletes it (checked in the emitted PTX by `dev/ptx.py`,
    # which is why that tool exists).
    #
    # Flag words are only ever incremented, and the wait compares the *signed
    # difference* to the arriving program's own count, so the rendezvous stays
    # correct when the counters wrap.  Nothing is reset, so no host-side memset
    # is needed per call -- which is the point, one would cost more than the
    # `F.linear` this replaces.
    #
    # The launch is cooperative, so the driver refuses a grid it cannot make
    # co-resident instead of letting the spin deadlock; `_build_plan` also
    # keeps the grid under the SM count.
    if FUSE_O:
        if BJO == 0:
            rnp = h * BN + tl.arange(0, BN)
            rjp = tl.arange(0, BHC)
            wpre = tl.load(WO + rnp[:, None] * swo + rjp[None, :],
                           mask=(rnp < CQ)[:, None] & (rjp < HC)[None, :],
                           other=0.0)
        nh = tl.num_programs(0)
        fl = FLAG + b * nh
        my = tl.atomic_add(fl + h, 1, sem="release", scope="gpu") + 1
        rh = tl.arange(0, BH)
        while tl.min(tl.atomic_add(fl + rh, 0, mask=rh < nh, sem="acquire",
                                   scope="gpu") - my) < 0:
            pass

        # Program h owns output columns [h*BN, (h+1)*BN); `_build_plan` only
        # takes this path when BN * H covers CQ, so every column has an owner.
        n0 = h * BN
        if n0 < CQ:
            rn = n0 + tl.arange(0, BN)
            mn = rn < CQ
            mp = OUT + b * sob + rq[:, None] * soq
            wop = WO + rn[:, None] * swo
            if BJO == 0:
                # The whole [BN, H*C] weight tile fits, so it is loaded *before*
                # the rendezvous: it depends on no other program, and getting its
                # cold-HBM latency out from behind the barrier is most of what
                # this phase costs.  `mid` is then the only post-barrier load and
                # it is L2-hot, having just been written.
                rjo = tl.arange(0, BHC)
                mj = rjo < HC
                mt = tl.load(mp + rjo[None, :],
                             mask=mq[:, None] & mj[None, :], other=0.0)
                acco = tl.dot(mt, tl.trans(wpre), out_dtype=tl.float32)
            else:
                rjo = tl.arange(0, BJO)
                acco = tl.zeros([BQ, BN], dtype=tl.float32)
                for j0 in range(0, HC, BJO):
                    mj = (j0 + rjo) < HC
                    mt = tl.load(mp + j0 + rjo[None, :],
                                 mask=mq[:, None] & mj[None, :], other=0.0)
                    wt = tl.load(wop + j0 + rjo[None, :],
                                 mask=mn[:, None] & mj[None, :], other=0.0)
                    acco = tl.dot(mt, tl.trans(wt), acco)
            tl.store(RES + b * srb + rq[:, None] * srq + rn[None, :],
                     acco.to(RES.dtype.element_ty),
                     mask=mq[:, None] & mn[None, :])


# ---------------------------------------------------------------------------
# Pre-bound launch: skip JIT argument binding / cache hashing / launch metadata
# ---------------------------------------------------------------------------
_raw_stream = torch._C._cuda_getCurrentRawStream

# Constexpr parameters of `_attn`, in declaration order (the C launcher takes
# every parameter positionally, constexprs included).
_ATTN_CST = ("BQ", "BK", "BC", "BJ", "NBIAS", "HAS_G", "HAS_QB", "EVEN_K",
             "FUSE_Q", "FUSE_KV", "FUSE_O", "BN", "BJO", "BH", "BHC")


class _Plan:
    """A pre-bound launch of one kernel specialization."""

    __slots__ = ("kern", "gx", "gy", "dev", "ints", "kw", "tail", "raw", "fn",
                 "coop", "pdl", "pm")

    def __init__(self, kern, cst, gx, gy, dev, ints, kw):
        self.kern = kern
        self.gx = gx
        self.gy = gy
        self.dev = dev
        self.ints = ints
        self.kw = kw
        self.tail = (*ints, *(kw[n] for n in cst))
        self.raw = None

    def run(self, ptrs):
        raw = self.raw
        if raw is None:
            self._bind(self.kern[(self.gx, self.gy)](*ptrs, *self.ints, **self.kw))
            return
        raw(self.gx, self.gy, 1, _raw_stream(self.dev), self.fn, self.coop,
            self.pdl, None, None, self.pm, None, None, None, *ptrs, *self.tail)

    def _bind(self, compiled):
        """Cache Triton's C launcher and its immutable per-kernel arguments."""
        try:
            launcher = compiled.run
            if launcher.global_scratch_size or launcher.profile_scratch_size:
                return
            self.fn = compiled.function
            self.coop = launcher.launch_cooperative_grid
            self.pdl = launcher.launch_pdl
            self.pm = compiled.packed_metadata
            self.raw = launcher.launch
        except AttributeError:  # unfamiliar Triton build: keep the JIT path
            self.raw = None


class _Recipe:
    """The fused launch plus the scratch buffer for one forward signature.

    ``mid`` holds the gated, head-flattened attention output that ``linear_o``
    consumes.  It is a private temporary -- never returned and never read across
    calls -- and every kernel touching it runs on the caller's stream, so one
    buffer per signature is reused instead of paying ~2.4 us of ``torch.empty``
    per forward.  (A caller driving the same module from two streams
    concurrently would need one recipe per stream; that is outside the
    single-stream contract this module is used under.)
    """

    __slots__ = ("fuse_q", "fuse_kv", "fuse_o", "attn", "mid_shape", "mid",
                 "out_shape", "flag")

    def __init__(self, fuse_q, fuse_kv, fuse_o, attn, mid_shape, out_shape,
                 flag):
        self.fuse_q = fuse_q
        self.fuse_kv = fuse_kv
        self.fuse_o = fuse_o
        self.attn = attn
        self.mid_shape = mid_shape
        self.out_shape = out_shape
        self.flag = flag
        self.mid = None


_MAX_BQ = 64
_MAX_BK = 256
_MAX_BC = 128
_MAX_BIAS = 3
_MAX_BJ = 128
_MAX_BN = 128
_PRE_WO_ELEMS = 1 << 14
# GEMM MACs per program above which cuBLAS does that GEMM instead; see
# `_fuse_flags`.
_FUSE_MACS = 3 << 17
# Rows in the `linear_o` GEMM above which cuBLAS fills the GPU with it and
# wins; see `_fuse_flags`.
_FUSE_O_ROWS = 64


def _sm_count(dev):
    """Multiprocessor count of *dev*, memoized (the query costs ~30 us)."""
    n = _SMS.get(dev)
    if n is None:
        n = _SMS[dev] = torch.cuda.get_device_properties(dev).multi_processor_count
    return n


_SMS = {}


def _fuse_flags(macs_q, macs_kv, macs_o, orows, o_ok):
    """Which GEMMs to fold into the attention program.

    Folding one in removes a cuBLAS launch but prepends a *serial* GEMM chain to
    every program, and with only B*H programs -- 16 to 64 here, against 148 SMs
    -- there is nothing else on the SM to hide that chain behind, so past some
    size it costs more GPU than it saves.  One budget, ``_FUSE_MACS`` MACs per
    program, on the total chain a program runs, spent on the cheapest side first.

    ``linear_o`` is not a third candidate for that budget, for two reasons that
    the per-shape sweep separates cleanly.

    It also pays the rendezvous (~1.1 us of device time against the 2.4-3.4 us
    ``nvjet`` launch it removes), so it only earns its place when it *displaces* a
    projection whose chain is longer than its own; when the budget already covers
    both projections, taking them removes two launches for less added work than
    ``linear_o`` removes one.

    And it is only worth taking when the GEMM it replaces is one cuBLAS cannot
    fill the GPU with.  ``linear_o`` is ``[B*Q, H*C] @ [H*C, c_q]``, so ``B*Q``
    sets how many row tiles ``nvjet`` gets: at ``B*Q = 16`` (one batch, Q = 16) it
    launches ~12 CTAs and is almost pure overhead, which is where fusing wins
    outright (31.7 -> 28.7 us on ``q_x[1,16,384]``); at ``B*Q = 256`` and 384 it
    gets a full GPU and wins, and fusing costs 0.5-2 us.  ``_FUSE_O_ROWS`` sits at
    the geometric midpoint of that measured bracket -- it is calibrated on five
    shapes, not derived, which is why it is a named constant.
    """
    order = (((macs_q, 0), (macs_kv, 1)) if macs_q <= macs_kv
             else ((macs_kv, 1), (macs_q, 0)))
    total = 0
    proj = [False, False]
    for macs, which in order:
        if total + macs <= _FUSE_MACS:
            total += macs
            proj[which] = True
    fuse_o = False
    if (o_ok and macs_o <= _FUSE_MACS and orows <= _FUSE_O_ROWS
            and not (proj[0] and proj[1])):
        if not (proj[0] or proj[1]):
            fuse_o = True
        elif macs_o < (macs_q if proj[0] else macs_kv):
            fuse_o = True
            proj = [False, False]
    return [proj[0], proj[1], fuse_o]


def _align(t, ndim):
    """Right-align a tensor to *ndim* dims; broadcast axes get stride 0."""
    sizes = [1] * (ndim - t.dim()) + list(t.shape)
    strides = [0] * (ndim - t.dim()) + list(t.stride())
    return sizes, [0 if s == 1 else st for s, st in zip(sizes, strides)]


def _flat_batch_stride(bsizes, strides):
    """Single stride ``s`` with ``offset(flat_b) == flat_b * s``, or None.

    The score tensor's batch axes are collapsed into one grid index; a tensor
    can follow that only if its batch strides look like a contiguous layout
    scaled by ``s`` (or it is fully broadcast, in which case ``s == 0``).
    """
    s = None
    mult = 1
    for d in range(len(bsizes) - 1, -1, -1):
        if bsizes[d] > 1:
            st = strides[d]
            if st % mult:
                return None
            cand = st // mult
            if s is not None and s != cand:
                return None
            s = cand
            mult *= bsizes[d]
    return 0 if s is None else s


class OF3Attention(nn.Module):
    """Standard multi-head attention with gating and bias list support.

    Reference: openfold3/core/model/primitives/attention.py Attention

    Args:
        c_q: Input dimension of query data
        c_k: Input dimension of key data
        c_v: Input dimension of value data
        c_hidden: Per-head hidden dimension
        no_heads: Number of attention heads
        gating: Whether to gate output using query data
    """

    def __init__(
        self,
        c_q: int,
        c_k: int,
        c_v: int,
        c_hidden: int,
        no_heads: int,
        gating: bool = True,
        q_bias: bool = False,
    ):
        super().__init__()
        self.c_q = c_q
        self.c_k = c_k
        self.c_v = c_v
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.gating = gating

        self.linear_q = Linear(c_q, c_hidden * no_heads, bias=q_bias)
        self.linear_k = Linear(c_k, c_hidden * no_heads, bias=False)
        self.linear_v = Linear(c_v, c_hidden * no_heads, bias=False)
        self.linear_o = Linear(c_hidden * no_heads, c_q, bias=False)

        self.linear_g = None
        if gating:
            self.linear_g = Linear(c_q, c_hidden * no_heads, bias=False)

        self._proj_cache = None
        self._plans = {}
        self._scale = 1.0 / math.sqrt(c_hidden)

    # -- cached [q|g] / [k|v] weights ---------------------------------------
    def _projections(self):
        """The cached ``[q|g]`` / ``[k|v]`` weights, rebuilt if a source changed.

        Parameters are read out of the live ``_modules`` / ``_parameters`` dicts
        rather than as ``self.linear_q.weight``: both are plain ``__dict__``
        entries, so this is the same lookup ``nn.Module.__getattr__`` performs
        (and stays correct if a submodule or a ``Parameter`` is replaced) without
        the ~0.3 us of interpreter overhead per access -- 1.6 us per forward at
        five weights, against a ~21 us budget.

        The guard pairs each source's storage address with its version counter,
        which between them catch every way a weight can change: ``p.data = ...``
        rebinding (dtype/device casts) moves the storage, and in-place writes
        (``load_state_dict``, ``normal_``) bump the version.  ``linear_o`` is in
        the guard although it is not concatenated into anything: the plans bake in
        its row stride and are bound to its dtype.
        """
        mods = self._modules
        pq = mods["linear_q"]._parameters
        wq = pq["weight"]
        bq = pq.get("bias")
        wk = mods["linear_k"]._parameters["weight"]
        wv = mods["linear_v"]._parameters["weight"]
        wo = mods["linear_o"]._parameters["weight"]
        lg = mods.get("linear_g")
        wg = None if lg is None else lg._parameters["weight"]
        guard = (wq.data_ptr(), wq._version, wk.data_ptr(), wk._version,
                 wv.data_ptr(), wv._version, wo.data_ptr(), wo._version,
                 None if bq is None else (bq.data_ptr(), bq._version),
                 None if wg is None else (wg.data_ptr(), wg._version))
        cache = self._proj_cache
        if cache is not None and cache[0] == guard:
            return cache
        w_qg = wq if wg is None else torch.cat((wq, wg), 0)
        b_qg = bq
        if bq is not None and wg is not None:
            b_qg = torch.cat((bq, torch.zeros_like(bq)), 0)
        # Every plan is bound to these weights -- their dtype selects the
        # compiled specialization the plan pre-binds, and `linear_o`'s row stride
        # is baked into its argument list -- so a change here retires the plans
        # too.  That is also why the plan key does not carry the weight dtype:
        # any cast rebinds `p.data`, which moves the storage this guard watches.
        cache = (guard, w_qg, b_qg, torch.cat((wk, wv), 0))
        self._proj_cache = cache
        self._plans.clear()
        return cache

    # -- launch plan --------------------------------------------------------
    def _build_plan(self, q_x, kv_x, biases):
        """Normalize this signature into launch arguments, or None to defer."""
        h = self.no_heads
        c = self.c_hidden
        hc = h * c
        bsizes = list(q_x.shape[:-2])
        nbatch = len(bsizes)
        qn, cq = q_x.shape[-2], q_x.shape[-1]
        kn, ck = kv_x.shape[-2], kv_x.shape[-1]
        nbias = len(biases)
        # `tl.dot` requires a contraction dim of at least 16, and BC / BK are the
        # contractions of the two dots, so tiles are floored there rather than
        # only rounded up.
        bq = max(16, triton.next_power_of_2(qn))
        bk = max(16, triton.next_power_of_2(kn))
        bc = max(16, triton.next_power_of_2(c))
        dt = self.linear_q.weight.dtype
        # fp16/bf16 only.  For fp32 the reference silently switches between
        # exact-fp32 and TF32 cuBLAS kernels depending on shape while `tl.dot`
        # defaults to TF32, and the fp32 tolerance (atol 1e-5 / rtol 1e-3 on 99%
        # of elements) is tight enough that guessing wrong fails -- the same
        # reason candidate/L1/linear.py leaves fp32 to the reference.
        if dt not in (torch.float16, torch.bfloat16):
            return None
        if (bq > _MAX_BQ or bk > _MAX_BK or bc > _MAX_BC or nbias > _MAX_BIAS
                or qn == 0 or kn == 0 or cq == 0 or ck == 0
                or kv_x.shape[:-2] != q_x.shape[:-2]
                or cq != self.c_q or ck != self.c_k or self.c_v != self.c_k
                or q_x.dtype is not dt or kv_x.dtype is not dt
                or q_x.stride(-1) != 1 or kv_x.stride(-1) != 1
                or q_x.device.type != "cuda"):
            return None
        bstr = []
        for t in biases:
            strides = _align(t, nbatch + 3)[1]
            sb = _flat_batch_stride(bsizes, strides[:nbatch])
            if sb is None or t.dtype is not dt:
                return None
            bstr += [sb, strides[nbatch], strides[nbatch + 1], strides[nbatch + 2]]
        bstr += [0] * (4 * (_MAX_BIAS - nbias))
        batch = 1
        for size in bsizes:
            batch *= size
        has_g = self.linear_g is not None

        # Can `linear_o` be folded in?  It reduces over the whole H*C axis while
        # a program only produced its own head's C columns of it, so the fused
        # form rendezvouses the H programs of a batch and then re-partitions the
        # [Q x H*C] @ [H*C x c_q] GEMM by output column: program h takes columns
        # [h*BN, (h+1)*BN).  That needs BN*H to cover c_q, the whole group to be
        # co-resident (so the spin cannot deadlock -- the launch is cooperative
        # and the grid is kept under the SM count as well), and the [BQ, BN]
        # accumulator to fit alongside the attention body.
        w_o = self._modules["linear_o"]._parameters["weight"]
        bn = max(16, triton.next_power_of_2(-(-cq // h)))
        o_ok = (bn <= _MAX_BN and bn * h >= cq and bq * bn <= 4096
                and h * batch <= _sm_count(q_x.device)
                and w_o.dtype is dt and w_o.stride(-1) == 1
                and tuple(w_o.shape) == (cq, hc))
        macs_q = c * (2 if has_g else 1) * qn * cq
        macs_kv = c * 2 * kn * ck
        macs_o = qn * bn * hc
        fuse_q, fuse_kv, fuse_o = _fuse_flags(macs_q, macs_kv, macs_o,
                                              batch * qn, o_ok)
        if fuse_q:
            sxb = _flat_batch_stride(bsizes, _align(q_x, nbatch + 2)[1][:nbatch])
            if sxb is None:
                return None
            # g is a row offset into the [2*H*C, C_q] weight.
            sxq, goff = q_x.stride(-2), hc * cq
        else:
            # QX is the pre-projected [q|g] activation instead, so g is a column
            # offset -- and the activation is only H*C wide when there is no gate.
            nq = (2 if has_g else 1) * hc
            sxb, sxq, goff = qn * nq, nq, hc
        if fuse_kv:
            syb = _flat_batch_stride(bsizes, _align(kv_x, nbatch + 2)[1][:nbatch])
            if syb is None:
                return None
            syq, voff = kv_x.stride(-2), hc * ck
        else:
            syb, syq, voff = kn * 2 * hc, 2 * hc, hc
        bj = 64 if (fuse_kv and bk >= 64) else _MAX_BJ
        bj = max(16, min(bj, triton.next_power_of_2(max(cq, ck))))
        bhc = max(16, triton.next_power_of_2(hc))
        bjo = 0 if bn * bhc <= _PRE_WO_ELEMS else max(16, min(_MAX_BJ, bhc))
        warps = 8 if bq * bk >= 2048 or bk * bc >= 4096 else 4
        ints = (qn * hc, hc, sxb, sxq, syb, syq, cq, ck, goff, voff, cq, ck,
                *bstr, w_o.stride(0), qn * cq, cq, hc, qn, kn, c, self._scale)
        kw = dict(BQ=bq, BK=bk, BC=bc, BJ=bj, NBIAS=nbias, HAS_G=has_g,
                  HAS_QB=self.linear_q.bias is not None, EVEN_K=(bk == kn),
                  FUSE_Q=fuse_q, FUSE_KV=fuse_kv, FUSE_O=fuse_o, BN=bn,
                  BJO=bjo, BH=triton.next_power_of_2(h), BHC=bhc,
                  num_warps=warps,
                  num_stages=2 if (fuse_q or fuse_kv or fuse_o) else 1)
        if fuse_o:
            kw["launch_cooperative_grid"] = True
        # One int32 word per program, only ever incremented; see the barrier in
        # `_attn`.  Allocated with the plan so no per-call zeroing is needed.
        flag = (torch.zeros(batch * h, dtype=torch.int32, device=q_x.device)
                if fuse_o else None)
        return _Recipe(fuse_q, fuse_kv, fuse_o,
                       _Plan(_attn, _ATTN_CST, h, batch, q_x.device.index,
                             ints, kw),
                       tuple(q_x.shape[:-1]) + (hc,), tuple(q_x.shape), flag)

    def forward(
        self,
        q_x: torch.Tensor,
        kv_x: torch.Tensor,
        biases: list[torch.Tensor] | None = None,
        use_deepspeed_evo_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_lma: bool = False,
        lma_q_chunk_size: int = 1024,
        lma_kv_chunk_size: int = 4096,
        use_high_precision: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            q_x:  [*, Q, C_q] query data
            kv_x: [*, K, C_k] key data
            biases: List of biases that broadcast to [*, H, Q, K]

        Returns:
            [*, Q, C_q] attention update
        """
        if biases is None:
            biases = []
        nbias = len(biases)
        # `_projections` also retires every plan when any weight changes, so the
        # plan key does not have to carry the weight dtype a compiled kernel is
        # bound to: casting the module rebinds `p.data`, which the guard sees.
        _, w_qg, b_qg, w_kv = self._projections()
        key = (q_x.shape, q_x.stride(), kv_x.shape, kv_x.stride(), q_x.dtype,
               q_x.device,
               *[(b.shape, b.stride(), b.dtype) for b in biases])
        plans = self._plans
        if key in plans:
            rec = plans[key]
        else:
            rec = plans[key] = self._build_plan(q_x, kv_x, biases)
        if rec is None:
            return self._forward_ref(q_x, kv_x, biases)

        qx = q_x if rec.fuse_q else F.linear(q_x, w_qg, b_qg)
        kvx = kv_x if rec.fuse_kv else F.linear(kv_x, w_kv)
        mid = rec.mid
        if mid is None:
            mid = rec.mid = torch.empty(rec.mid_shape, dtype=q_x.dtype,
                                        device=q_x.device)
        b0 = biases[0] if nbias else mid
        w_o = self._modules["linear_o"]._parameters["weight"]
        if rec.fuse_o:
            out = torch.empty(rec.out_shape, dtype=q_x.dtype, device=q_x.device)
            rec.attn.run((mid, qx, kvx, w_qg, w_kv,
                          b_qg if b_qg is not None else mid, b0,
                          biases[1] if nbias > 1 else b0,
                          biases[2] if nbias > 2 else b0, w_o, out, rec.flag))
            return out
        rec.attn.run((mid, qx, kvx, w_qg, w_kv, b_qg if b_qg is not None else mid,
                      b0, biases[1] if nbias > 1 else b0,
                      biases[2] if nbias > 2 else b0, w_o, mid, mid))
        return F.linear(mid, w_o)

    # -- reference path for signatures the fused kernel does not cover ------
    def _forward_ref(self, q_x, kv_x, biases):
        """Literal reference implementation, op for op.

        Used for any signature `_build_plan` declines (fp32, K past the
        register-resident cap, a bias layout that will not collapse to one batch
        stride, ...).  Deliberately *not* a faster rewrite: a deferral has to be
        exactly right, and at fp32 tolerances even fusing the projection weights
        into one wider GEMM can move cuBLAS's kernel choice and the result with
        it.  Speed here does not matter -- no captured shape takes this path.
        """
        h = self.no_heads
        q = self.linear_q(q_x)
        k = self.linear_k(kv_x)
        v = self.linear_v(kv_x)
        q = q.view(q.shape[:-1] + (h, -1)).transpose(-2, -3)
        k = k.view(k.shape[:-1] + (h, -1)).transpose(-2, -3)
        v = v.view(v.shape[:-1] + (h, -1)).transpose(-2, -3)
        q = q / math.sqrt(self.c_hidden)

        scores = torch.einsum("...qc,...kc->...qk", q, k)
        for b in biases:
            scores = scores + b
        scores = F.softmax(scores, dim=-1)
        o = torch.einsum("...qk,...kc->...qc", scores.to(dtype=v.dtype), v)
        o = o.transpose(-2, -3)

        if self.linear_g is not None:
            g = torch.sigmoid(self.linear_g(q_x))
            o = o * g.view(g.shape[:-1] + (h, -1))
        return self.linear_o(o.reshape(o.shape[:-2] + (-1,)))
