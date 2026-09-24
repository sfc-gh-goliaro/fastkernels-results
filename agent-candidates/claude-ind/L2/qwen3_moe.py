"""Qwen3 Mixture-of-Experts block with a hand-written fused FP8 MoE pipeline.

The baseline composes the block out of generic building blocks (activation
quantization -> materialized permutation -> grouped GEMM -> fused act+quant ->
grouped GEMM -> gather + weighted reduce).  Most of its cost is *not* the two
GEMMs: the permutation is built out of a dozen small PyTorch ops (argsort,
searchsorted, scatter_add, boolean index_put) which both launch-bound the block
at small batch and move far more bytes than the GEMMs at large batch.

This version replaces the whole expert pipeline with a small set of Triton
kernels:

  1. ``_quant_act``   per-token-group (128) FP8 quantization of the activation,
                      done once on the *unpermuted* [M, K] tensor.
  2. ``_count_and_fill`` / ``_scan_meta`` / ``_scatter_rows``
                      a counting sort of the (token, slot) pairs by expert id,
                      emitting directly the BLOCK_M-padded row table the GEMMs
                      consume.  Three O(pairs) kernels, no host sync and no
                      torch op in between; ``_prep_small`` collapses all of it
                      (plus the quantization) into one block for decode-sized
                      batches, where launch cost dominates.
  3. ``_moe_gemm1``   block-scaled FP8 grouped GEMM that *gathers* its A rows
                      through the sorted table, so the permuted activation is
                      never materialized.
  4. ``_act_quant``   fused SiLU-mul + per-token-group FP8 requantization.
  5. ``_moe_gemm2``   block-scaled FP8 grouped GEMM writing straight into
                      token-major slot order.
  6. ``_reduce_topk`` weighted sum over the top-k slots of each token.

The whole block is then replayed from a captured CUDA graph, because at decode
batch size the eight launches cost several times more host time than the device
work they describe.

FP8 W8A8 block-scaled expert weights are kept in the checkpoint layout
([E, N, K] weights, [E, N/128, K/128] fp32 scales); the scales are folded into
the fp32 accumulator once per 128-deep K block.  The reference's quantization
(UE8M0 scales, bf16-rounded SiLU, and *where* it rounds the weighted top-k
partials) is reproduced exactly, so the two implementations differ only in
accumulation order.
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
import triton
import triton.language as tl

from ....infra.tp import _tp_rank, _tp_size
from ..L1.allreduce import AllReduce
from ..L1.moe_grouped_gemm import _valid_deep_gemm
from ..L1.topk_softmax import TopKSoftmax
from ..L2.fused_experts import FusedExperts
from ..L2.parallel_linear import ReplicatedLinear

_FP8_BLOCK = 128

# constexpr mirrors of the FP8 quantization constants, for use inside kernels
_TL_FP8_MAX = tl.constexpr(448.0)
_TL_EPS = tl.constexpr(1e-10)


def _dg_reduction_order(x, w13, w2) -> bool:
    """Whether the reference keeps the routed weight *out* of the second GEMM's
    epilogue.  The baseline dispatches on exactly this predicate, and the two
    branches round the top-k partials to bf16 at different points, which is
    observable in the output, so the rounding point has to follow it."""
    return (_valid_deep_gemm(x, w13, w2)
            and not torch.cuda.is_current_stream_capturing())


@triton.jit
def _tile_id(NTN: tl.constexpr, GM: tl.constexpr):
    """Flat program id -> (row tile, column tile).  ``GM`` groups GM row tiles
    with all NTN column tiles so that the weight tile of one expert is shared by
    several concurrently-resident blocks (L2 reuse); GM == 1 walks columns
    fastest, which is what the bandwidth-bound small-batch shapes want."""
    pid = tl.program_id(0)
    if GM == 1:
        return pid // NTN, pid % NTN
    ng = GM * NTN
    return (pid // ng) * GM + ((pid % ng) % GM), (pid % ng) // GM


# ---------------------------------------------------------------------------
# Routing metadata: counting sort of the (token, slot) pairs by expert.
# ---------------------------------------------------------------------------
@triton.jit
def _count_and_fill(ids_ptr, cnt_ptr, srt_ptr, R, NCB, PAD, E,
                    BLOCK: tl.constexpr, BINS: tl.constexpr):
    """Per-expert histogram of the routed ids, plus the -1 pre-fill of the padded
    row table.  Both are pure O(R + PAD) passes fused into one launch: the fill
    only has to land before ``_scatter_rows`` overwrites the live slots, and
    stream order guarantees that."""
    pid = tl.program_id(0)
    if pid < NCB:
        idx = pid * BLOCK + tl.arange(0, BLOCK)
        # Tail lanes land in the spare upper half of the bin range, which is
        # never read back, so no correction pass is needed.
        v = tl.load(ids_ptr + idx, mask=idx < R, other=BINS - 1)
        h = tl.histogram(v.to(tl.int32), BINS)
        b = tl.arange(0, BINS)
        tl.atomic_add(cnt_ptr + b, h, mask=(b < E) & (h != 0))
    else:
        idx = (pid - NCB) * BLOCK + tl.arange(0, BLOCK)
        tl.store(srt_ptr + idx, -1, mask=idx < PAD)


@triton.jit
def _scan_meta(cnt_ptr, tile_off_ptr, pos_ptr, meta_ptr, E, BM: tl.constexpr,
               EP2: tl.constexpr):
    """Exclusive scan of per-expert tile counts.  Expert e owns tiles
    [tile_off[e], tile_off[e] + ceil(cnt[e]/BM)) and therefore rows
    [BM*tile_off[e], ...), i.e. tile t always starts at row t*BM."""
    e = tl.arange(0, EP2)
    c = tl.load(cnt_ptr + e, mask=e < E, other=0)
    nt = (c + (BM - 1)) // BM
    off = tl.cumsum(nt, 0) - nt
    tl.store(tile_off_ptr + e, off, mask=e < E)
    tl.store(pos_ptr + e, off * BM, mask=e < E)
    tl.store(cnt_ptr + e, 0, mask=e < E)   # ready for the next call
    tl.store(meta_ptr, tl.sum(nt, 0))


@triton.jit
def _scatter_rows(ids_ptr, pos_ptr, srt_ptr, R, BLOCK: tl.constexpr):
    """Place each (token, slot) pair into its expert's padded row range.  The
    slot within an expert comes from an atomic bump, so the order inside a block
    is arbitrary -- every row's output depends only on its own token and expert,
    so the grouping is observationally irrelevant."""
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = idx < R
    v = tl.load(ids_ptr + idx, mask=m, other=0)
    slot = tl.atomic_add(pos_ptr + v, 1, mask=m)
    tl.store(srt_ptr + slot, idx.to(tl.int32), mask=m)


# ---------------------------------------------------------------------------
# Activation quantization (matches per_token_group_quant_fp8, UE8M0 scales).
# ---------------------------------------------------------------------------
@triton.jit
def _quant_act(x_ptr, q_ptr, s_ptr, cnt_ptr, M, K, NG, E, BM: tl.constexpr,
               G: tl.constexpr, EP2: tl.constexpr):
    pid = tl.program_id(0)
    g = tl.program_id(1)
    if pid == 0 and g == 0:
        ei = tl.arange(0, EP2)
        tl.store(cnt_ptr + ei, 0, mask=ei < E)
    rows = pid * BM + tl.arange(0, BM)
    rmask = rows < M
    cols = g * G + tl.arange(0, G)
    off = rows[:, None].to(tl.int64) * K + cols[None, :]
    x = tl.load(x_ptr + off, mask=rmask[:, None], other=0.0).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x), 1), _TL_EPS)
    # Divide (not reciprocal-multiply) to match the reference CUDA kernel for
    # this stage; see the note in ``_act_quant``, which mirrors a Triton one.
    ys = amax / _TL_FP8_MAX
    ys = tl.math.exp2(tl.math.ceil(tl.math.log2(ys)))
    q = tl.clamp(x / ys[:, None], -_TL_FP8_MAX, _TL_FP8_MAX)
    tl.store(q_ptr + off, q.to(q_ptr.dtype.element_ty), mask=rmask[:, None])
    tl.store(s_ptr + rows * NG + g, ys, mask=rmask)


# ---------------------------------------------------------------------------
# Single-block prep for decode-sized batches.
#
# At M*top_k <= a few dozen pairs the four prep launches cost far more CPU than
# the work they do, and the whole job fits in one block: quantize, histogram,
# scan and place, in that order, with no grid-wide synchronization needed.
# ---------------------------------------------------------------------------
@triton.jit
def _prep_small(x_ptr, xq_ptr, xs_ptr, ids_ptr, srt_ptr, tile_off_ptr, meta_ptr,
                NGT, R, E, BM: tl.constexpr, EP2: tl.constexpr,
                RMAX: tl.constexpr, CH: tl.constexpr, G: tl.constexpr):
    # 1) per-token-group FP8 quantization.  K is a whole number of groups, so
    #    group ``gi`` is exactly x[gi*G : (gi+1)*G] and its scale is xs[gi].
    cols = tl.arange(0, G)
    for c0 in range(0, NGT, CH):
        gi = c0 + tl.arange(0, CH)
        gm = gi < NGT
        off = gi[:, None].to(tl.int64) * G + cols[None, :]
        v = tl.load(x_ptr + off, mask=gm[:, None], other=0.0).to(tl.float32)
        amax = tl.maximum(tl.max(tl.abs(v), 1), _TL_EPS)
        ys = tl.math.exp2(tl.math.ceil(tl.math.log2(amax / _TL_FP8_MAX)))
        q = tl.clamp(v / ys[:, None], -_TL_FP8_MAX, _TL_FP8_MAX)
        tl.store(xq_ptr + off, q.to(xq_ptr.dtype.element_ty), mask=gm[:, None])
        tl.store(xs_ptr + gi, ys, mask=gm)

    # 2) histogram + 3) exclusive scan of the per-expert tile counts.
    idx = tl.arange(0, RMAX)
    rm = idx < R
    v = tl.load(ids_ptr + idx, mask=rm, other=EP2)
    ei = tl.arange(0, EP2)
    cnt = tl.sum(tl.where(v[:, None] == ei[None, :], 1, 0).to(tl.int32), 0)
    nt = (cnt + (BM - 1)) // BM
    off = tl.cumsum(nt, 0) - nt
    tl.store(tile_off_ptr + ei, off, mask=ei < E)
    tl.store(meta_ptr, tl.sum(nt, 0))

    # 4) mark the padding slots, then place the pairs.  The two slot sets are
    #    disjoint, so no ordering between the stores is required.
    j = tl.arange(0, BM)
    pslot = (off * BM + cnt)[:, None] + j[None, :]
    tl.store(srt_ptr + pslot, -1, mask=(cnt[:, None] + j[None, :]) < nt[:, None] * BM)

    rank = tl.sum(((v[:, None] == v[None, :]) & (idx[None, :] < idx[:, None])
                   ).to(tl.int32), 1)
    base = tl.sum(tl.where(ei[None, :] == v[:, None], off[None, :] * BM, 0), 1)
    tl.store(srt_ptr + base + rank, idx.to(tl.int32), mask=rm)


# ---------------------------------------------------------------------------
# Grouped GEMM 1: gather A rows through the sorted table, so the permuted
# activation is never materialized.
# ---------------------------------------------------------------------------
@triton.jit
def _moe_gemm1(xq_ptr, xs_ptr, w_ptr, ws_ptr, out_ptr,
               sorted_ptr, tile_off_ptr, meta_ptr,
               N, K, NG, NGW, E,
               TOPK_LOG2: tl.constexpr, EP2: tl.constexpr,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
               NTN: tl.constexpr, GM: tl.constexpr, AMASK: tl.constexpr,
               SB: tl.constexpr):
    t, pid_n = _tile_id(NTN, GM)
    if t >= tl.load(meta_ptr):
        return

    ei = tl.arange(0, EP2)
    toff = tl.load(tile_off_ptr + ei, mask=ei < E, other=0x3FFFFFFF)
    e = tl.sum(tl.where(toff <= t, 1, 0), 0) - 1

    slots = t * BM + tl.arange(0, BM)
    pairs = tl.load(sorted_ptr + slots)
    valid = pairs >= 0
    # Padding rows are pointed at token 0 so the gathered address is in range
    # even though the load below is predicated off (and the store at the end
    # drops them).
    tok = tl.where(valid, pairs >> TOPK_LOG2, 0)

    offs_n = pid_n * BN + tl.arange(0, BN)
    wbase = w_ptr + e.to(tl.int64) * (N * K)
    wsb = ws_ptr + e.to(tl.int64) * (NGW * NG)
    abase = tok.to(tl.int64) * K
    sbase = tok.to(tl.int64) * NG
    offs_k = tl.arange(0, BK)

    acc = tl.zeros([BM, BN], tl.float32)
    for k0 in range(0, K, BK):
        if AMASK:
            a = tl.load(xq_ptr + abase[:, None] + (k0 + offs_k)[None, :],
                        mask=valid[:, None], other=0.0)
            asc = tl.load(xs_ptr + sbase + (k0 // 128), mask=valid, other=0.0)
        else:
            a = tl.load(xq_ptr + abase[:, None] + (k0 + offs_k)[None, :])
            asc = tl.load(xs_ptr + sbase + (k0 // 128))
        b = tl.load(wbase + offs_n[:, None].to(tl.int64) * K + (k0 + offs_k)[None, :])
        if SB:  # one weight-scale block per column tile -> scalar
            bsc = tl.load(wsb + (pid_n * BN // 128) * NG + (k0 // 128))
            acc += tl.dot(a, tl.trans(b)) * (asc * bsc)[:, None]
        else:
            bsc = tl.load(wsb + (offs_n // 128) * NG + (k0 // 128))
            acc += tl.dot(a, tl.trans(b)) * (asc[:, None] * bsc[None, :])

    oo = slots[:, None].to(tl.int64) * N + offs_n[None, :]
    tl.store(out_ptr + oo, acc.to(out_ptr.dtype.element_ty), mask=valid[:, None])


# ---------------------------------------------------------------------------
# Fused SiLU-mul + FP8 requantization (matches silu_mul_quant_fp8).
# ---------------------------------------------------------------------------
@triton.jit
def _act_quant(y_ptr, q_ptr, s_ptr, meta_ptr, N2, NQ, NGQ, BMG,
               BM: tl.constexpr, NG: tl.constexpr, G: tl.constexpr):
    r0 = tl.program_id(0) * BM
    if r0 >= tl.load(meta_ptr) * BMG:
        return
    gb = tl.program_id(1) * NG
    rows = r0 + tl.arange(0, BM)
    cols = gb * G + tl.arange(0, NG * G)
    ybase = rows[:, None].to(tl.int64) * N2 + cols[None, :]
    a = tl.load(y_ptr + ybase)
    u = tl.load(y_ptr + ybase + NQ)
    af = a.to(tl.float32)
    # SiLU is rounded to the activation dtype before the gate multiply, and the
    # product is rounded again -- both reference kernels do exactly this, and the
    # FP8 bucket a value lands in is sensitive to it.
    silu = (af / (1.0 + tl.exp(-af))).to(y_ptr.dtype.element_ty)
    y = tl.reshape((silu * u).to(tl.float32), (BM, NG, G))
    amax = tl.maximum(tl.max(tl.abs(y), 2), _TL_EPS)
    # Reciprocal-multiply, not divide: ``silu_mul_quant_fp8`` (the reference for
    # this stage) scales by 1/fp8_max, and the two differ by an ULP that can move
    # ceil(log2(...)) by one whole binade for scales that are exact powers of two.
    ys = amax * (1.0 / _TL_FP8_MAX)
    ys = tl.math.exp2(tl.math.ceil(tl.math.log2(ys)))
    q = tl.clamp(y / ys[:, :, None], -_TL_FP8_MAX, _TL_FP8_MAX)
    qoff = rows[:, None].to(tl.int64) * NQ + cols[None, :]
    tl.store(q_ptr + qoff, tl.reshape(q, (BM, NG * G)).to(q_ptr.dtype.element_ty))
    tl.store(s_ptr + rows[:, None] * NGQ + (gb + tl.arange(0, NG))[None, :], ys)


# ---------------------------------------------------------------------------
# Grouped GEMM 2: contiguous A rows, scatter C rows to token-major slots.
# ---------------------------------------------------------------------------
@triton.jit
def _moe_gemm2(aq_ptr, as_ptr, w_ptr, ws_ptr, out_ptr, tw_ptr,
               sorted_ptr, tile_off_ptr, meta_ptr,
               N, K, NG, NGW, E,
               EP2: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
               BK: tl.constexpr, MUL_W: tl.constexpr, NTN: tl.constexpr,
               GM: tl.constexpr, AMASK: tl.constexpr, SB: tl.constexpr):
    t, pid_n = _tile_id(NTN, GM)
    if t >= tl.load(meta_ptr):
        return

    ei = tl.arange(0, EP2)
    toff = tl.load(tile_off_ptr + ei, mask=ei < E, other=0x3FFFFFFF)
    e = tl.sum(tl.where(toff <= t, 1, 0), 0) - 1

    slots = t * BM + tl.arange(0, BM)
    pairs = tl.load(sorted_ptr + slots)
    valid = pairs >= 0

    offs_n = pid_n * BN + tl.arange(0, BN)
    wbase = w_ptr + e.to(tl.int64) * (N * K)
    wsb = ws_ptr + e.to(tl.int64) * (NGW * NG)
    abase = slots[:, None].to(tl.int64) * K
    sbase = slots.to(tl.int64) * NG
    offs_k = tl.arange(0, BK)

    acc = tl.zeros([BM, BN], tl.float32)
    for k0 in range(0, K, BK):
        if AMASK:
            a = tl.load(aq_ptr + abase + (k0 + offs_k)[None, :],
                        mask=valid[:, None], other=0.0)
            asc = tl.load(as_ptr + sbase + (k0 // 128), mask=valid, other=0.0)
        else:
            a = tl.load(aq_ptr + abase + (k0 + offs_k)[None, :])
            asc = tl.load(as_ptr + sbase + (k0 // 128))
        b = tl.load(wbase + offs_n[:, None].to(tl.int64) * K + (k0 + offs_k)[None, :])
        if SB:
            bsc = tl.load(wsb + (pid_n * BN // 128) * NG + (k0 // 128))
            acc += tl.dot(a, tl.trans(b)) * (asc * bsc)[:, None]
        else:
            bsc = tl.load(wsb + (offs_n // 128) * NG + (k0 // 128))
            acc += tl.dot(a, tl.trans(b)) * (asc[:, None] * bsc[None, :])

    pos = tl.where(valid, pairs, 0)
    if MUL_W:
        # The reference applies the routed weight to the fp32 accumulator and
        # only then rounds to bf16 (Triton grouped-GEMM epilogue + moe_sum);
        # rounding order is visible in the output wherever the top-k terms
        # cancel, so it has to be reproduced exactly.
        acc *= tl.load(tw_ptr + pos, mask=valid, other=0.0)[:, None]
    oo = pos.to(tl.int64)[:, None] * N + offs_n[None, :]
    tl.store(out_ptr + oo, acc.to(out_ptr.dtype.element_ty), mask=valid[:, None])


# ---------------------------------------------------------------------------
# Weighted reduce over the top-k slots of each token.
# ---------------------------------------------------------------------------
@triton.jit
def _reduce_topk(part_ptr, tw_ptr, out_ptr, H, TOPK: tl.constexpr,
                 BN: tl.constexpr, HB: tl.constexpr, USE_W: tl.constexpr):
    pid = tl.program_id(0)
    m = pid // HB
    cols = (pid % HB) * BN + tl.arange(0, BN)
    acc = tl.zeros([BN], tl.float32)
    for j in tl.static_range(TOPK):
        v = tl.load(part_ptr + (m * TOPK + j).to(tl.int64) * H + cols).to(tl.float32)
        if USE_W:
            v *= tl.load(tw_ptr + m * TOPK + j)
        acc += v
    tl.store(out_ptr + m.to(tl.int64) * H + cols, acc.to(out_ptr.dtype.element_ty))


# ---------------------------------------------------------------------------
# Scratch buffers, shared by every layer (they run sequentially).
# ---------------------------------------------------------------------------
class _Scratch:
    __slots__ = ("bufs", "gen")

    def __init__(self):
        self.bufs = {}
        self.gen = 0

    def get(self, key, numel, dtype, device):
        b = self.bufs.get(key)
        if b is None or b.numel() < numel or b.dtype != dtype:
            b = torch.empty(numel, dtype=dtype, device=device)
            self.bufs[key] = b
            # Any reallocation moves a pointer that a captured graph baked in.
            self.gen += 1
        return b


_SCRATCH = _Scratch()


# ---------------------------------------------------------------------------
# CUDA-graph capture of the whole block.
#
# The pipeline is eight-or-so kernel launches of a few microseconds each; at
# decode batch size the host cannot even submit them as fast as the GPU retires
# them (measured ~170us of launch time against ~40us of device time), so the
# block is latency-bound on the CPU. Replaying a captured graph collapses that to
# one launch. Every pointer the graph hard-codes -- our scratch, the router's
# top-k buffers -- is folded into a signature that invalidates the cache when it
# moves, and any capture failure falls back to eager submission for good.
# ---------------------------------------------------------------------------
_MAX_GRAPHS = 12
_GRAPHS: dict = {}
_GRAPH_SIG = None
_GRAPHS_OK = not os.environ.get("FK_MOE_NOGRAPH")


def _graphed(body, x, key, router):
    """Run ``body(x)`` through a cached CUDA graph; None if unavailable."""
    global _GRAPH_SIG, _GRAPHS_OK
    if not _GRAPHS_OK or torch.cuda.is_current_stream_capturing():
        return None
    ent = _GRAPHS.get(key) if _state_sig(router) == _GRAPH_SIG else None
    if ent is None:
        out = body(x)
        # Sizing the scratch (or the router's own buffers) may have moved
        # pointers, so re-read the signature after the warm-up run.
        sig = _state_sig(router)
        if sig != _GRAPH_SIG:
            _GRAPHS.clear()
            _GRAPH_SIG = sig
        if len(_GRAPHS) >= _MAX_GRAPHS:
            return out
        try:
            static_in = x.clone()
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(2):
                    body(static_in)
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                static_out = body(static_in)
        except Exception:      # capture unsupported here -- stay eager
            _GRAPHS_OK = False
            return out
        _GRAPHS[key] = ent = (graph, static_in, static_out)
    graph, static_in, static_out = ent
    static_in.copy_(x)
    graph.replay()
    return static_out.clone()


def _state_sig(router):
    """Identity of every buffer a captured graph hard-codes: our own scratch plus
    the router's pre-allocated top-k outputs."""
    sc = _SCRATCH
    tw = getattr(router, "_topk_weights", None)
    return (sc.gen, tuple(sorted(b.data_ptr() for b in sc.bufs.values())),
            None if tw is None else tw.data_ptr())


def _round_tiles(n: int, gm: int) -> int:
    return n if gm == 1 else -(-n // gm) * gm


# Tile/launch configuration for both grouped GEMMs, tuned on B200 across the
# captured batch sizes (1 ... 16384 tokens):
#
#   BM=64    Blackwell's MMA works in 64-row steps, so a narrower row tile buys
#            less padding waste but no less work -- measured 3x worse at 314
#            tokens even though it padded half as much.
#   BN=128   equal to the weight-scale block, which makes the B scale a scalar
#            per K step instead of a vector (measured ~10% on both GEMMs).
#   BK=128   the activation scale group; smaller splits re-apply scales for no
#            gain (~1.6x worse at 64).
#   warps=4, stages=4  best of {4,8} x {3..6}; 5+ stages runs out of shared
#            memory for the B tile and spills into a ~1.5x regression.
_CFG = (64,                  # BM
        128, 128, 4, 4, 1,   # GEMM1: BN, BK, warps, stages, group-M
        128, 128, 4, 4, 1,   # GEMM2: BN, BK, warps, stages, group-M
        1, 0, 1)             # mask A in GEMM1 / in GEMM2, scalar B scale

# Pair count below which the four prep launches are collapsed into one block.
_SMALL_R = 64

_TUNE = bool(os.environ.get("FK_MOE_TUNE"))


def _pick_cfg():
    if _TUNE:  # offline tuning hook; never taken in a normal run
        return tuple(int(v) for v in os.environ["FK_MOE_CFG"].split(","))
    return _CFG


def _fused_moe_fp8(x, w13, w13_scale, w2, w2_scale, topk_weights, topk_ids,
                   num_experts, top_k, late_weight):
    """FP8 W8A8 block-scaled fused MoE.  Returns [M, K] bf16."""
    M, K = x.shape
    E = num_experts
    N2 = w13.size(1)
    N = N2 // 2
    R = M * top_k
    dev = x.device

    (bm, bn1, bk1, w1, s1, gm1, bn2, bk2, w2c, s2, gm2,
     am, am2, sb) = _pick_cfg()
    max_tiles = min(-(-R // bm) + E, R)
    max_pad = max_tiles * bm
    ng_a = K // 128
    ng_i = N // 128

    sc = _SCRATCH
    xq = sc.get("xq", M * K, torch.float8_e4m3fn, dev)[:M * K].view(M, K)
    xs = sc.get("xs", M * ng_a, torch.float32, dev)[:M * ng_a].view(M, ng_a)
    aux = sc.get("aux", 3 * E + 8, torch.int32, dev)
    cnt, pos, toff, meta = aux[:E], aux[E:2 * E], aux[2 * E:3 * E], aux[3 * E:]
    srt = sc.get("srt", max_pad, torch.int32, dev)[:max_pad]
    i1 = sc.get("i1", max_pad * N2, torch.bfloat16, dev)[:max_pad * N2]
    i2q = sc.get("i2q", max_pad * N, torch.float8_e4m3fn, dev)[:max_pad * N]
    i2s = sc.get("i2s", max_pad * ng_i, torch.float32, dev)[:max_pad * ng_i]
    part = sc.get("part", R * K, torch.bfloat16, dev)[:R * K]

    # --- routing metadata + activation quantization -------------------------
    ep2 = triton.next_power_of_2(E)
    if R <= _SMALL_R:
        _prep_small[(1,)](x, xq, xs, topk_ids, srt, toff, meta,
                          M * ng_a, R, E, BM=bm, EP2=ep2,
                          RMAX=max(16, triton.next_power_of_2(R)), CH=32,
                          G=128, num_warps=8)
    else:
        qbm = 32 if M >= 32 else 8
        _quant_act[(-(-M // qbm), ng_a)](x, xq, xs, cnt, M, K, ng_a, E, BM=qbm,
                                         G=128, EP2=ep2, num_warps=4)
        cb = 1024
        ncb = -(-R // cb)
        _count_and_fill[(ncb + -(-max_pad // cb),)](
            topk_ids, cnt, srt, R, ncb, max_pad, E, BLOCK=cb, BINS=2 * ep2,
            num_warps=4)
        _scan_meta[(1,)](cnt, toff, pos, meta, E, BM=bm, EP2=ep2, num_warps=4)
        _scatter_rows[(ncb,)](topk_ids, pos, srt, R, BLOCK=cb, num_warps=4)

    # --- GEMM 1 -------------------------------------------------------------
    ntn1 = N2 // bn1
    _moe_gemm1[(_round_tiles(max_tiles, gm1) * ntn1,)](
        xq, xs, w13, w13_scale, i1, srt, toff, meta,
        N2, K, ng_a, N2 // 128, E,
        TOPK_LOG2=top_k.bit_length() - 1, EP2=ep2,
        BM=bm, BN=bn1, BK=bk1, NTN=ntn1, GM=gm1, AMASK=bool(am),
        SB=bool(sb), num_warps=w1, num_stages=s1,
    )

    # --- fused activation + requantization ----------------------------------
    agq = 4 if ng_i % 4 == 0 else 1
    abm = min(16, bm)
    _act_quant[(-(-max_pad // abm), ng_i // agq)](
        i1, i2q, i2s, meta, N2, N, ng_i, bm,
        BM=abm, NG=agq, G=128, num_warps=4)

    # --- GEMM 2 -------------------------------------------------------------
    ntn2 = K // bn2
    _moe_gemm2[(_round_tiles(max_tiles, gm2) * ntn2,)](
        i2q, i2s, w2, w2_scale, part, topk_weights, srt, toff, meta,
        K, N, ng_i, K // 128, E,
        EP2=ep2, BM=bm, BN=bn2, BK=bk2,
        MUL_W=not late_weight, NTN=ntn2, GM=gm2, AMASK=bool(am2),
        SB=bool(sb), num_warps=w2c, num_stages=s2,
    )

    # --- weighted reduce ----------------------------------------------------
    out = torch.empty(M, K, dtype=x.dtype, device=dev)
    bnr = 1024
    _reduce_topk[(M * (K // bnr),)](
        part, topk_weights, out, K, TOPK=top_k, BN=bnr, HB=K // bnr,
        USE_W=late_weight, num_warps=8)
    return out


class Qwen3MoE(nn.Module):
    """Qwen3 Mixture-of-Experts with fused Triton grouped GEMM.

    Weights (FP8 mode):
      gate:     [num_experts, hidden_size] (bfloat16, replicated)
      w13:      [E, 2*moe_intermediate_per_tp, hidden_size] (float8_e4m3fn)
      w13_scale:[E, scale_rows_13, scale_cols_13] (float32)
      w2:       [E, hidden_size, moe_intermediate_per_tp] (float8_e4m3fn)
      w2_scale: [E, scale_rows_2, scale_cols_2] (float32)

    Weights (BF16 mode):
      gate:  [num_experts, hidden_size]
      w13:   [E, 2*moe_intermediate_per_tp, hidden_size]
      w2:    [E, hidden_size, moe_intermediate_per_tp]
    """

    def __init__(self, config, quant_config: dict | None = None):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.hidden_size = config.hidden_size
        tp = _tp_size()
        self.tp_size = tp
        self.intermediate_per_tp = config.moe_intermediate_size // tp
        self.renormalize = getattr(config, "norm_topk_prob", True)
        self.use_fp8 = quant_config is not None

        self.gate = ReplicatedLinear(
            config.hidden_size, config.num_experts, bias=False,
        )

        w13_rows = 2 * self.intermediate_per_tp
        w2_cols = self.intermediate_per_tp

        if self.use_fp8:
            block_size = quant_config.get("weight_block_size", [128, 128])
            self.block_shape = block_size
            block_n, block_k = block_size[0], block_size[1]

            self.w13 = nn.Parameter(torch.empty(
                config.num_experts, w13_rows, config.hidden_size,
                dtype=torch.float8_e4m3fn,
            ), requires_grad=False)
            self.w13_scale = nn.Parameter(torch.ones(
                config.num_experts,
                math.ceil(w13_rows / block_n),
                math.ceil(config.hidden_size / block_k),
                dtype=torch.float32,
            ), requires_grad=False)

            self.w2 = nn.Parameter(torch.empty(
                config.num_experts, config.hidden_size, w2_cols,
                dtype=torch.float8_e4m3fn,
            ), requires_grad=False)
            self.w2_scale = nn.Parameter(torch.ones(
                config.num_experts,
                math.ceil(config.hidden_size / block_n),
                math.ceil(w2_cols / block_k),
                dtype=torch.float32,
            ), requires_grad=False)

            self.w13.weight_loader = self._w13_weight_loader_fp8
            self.w13_scale.weight_loader = self._w13_scale_loader
            self.w2.weight_loader = self._w2_weight_loader_fp8
            self.w2_scale.weight_loader = self._w2_scale_loader
        else:
            self.block_shape = None
            self.w13 = nn.Parameter(torch.empty(
                config.num_experts, w13_rows, config.hidden_size,
            ))
            self.w13.weight_loader = self._w13_weight_loader

            self.w2 = nn.Parameter(torch.empty(
                config.num_experts, config.hidden_size, w2_cols,
            ))
            self.w2.weight_loader = self._w2_weight_loader

            self.w13_scale = None
            self.w2_scale = None

        self.topk_softmax = TopKSoftmax()
        self.fused_experts = FusedExperts()
        self.allreduce = AllReduce()

        # Custom-op dispatch for torch.compile (set by engine after model init)
        self._use_custom_op = False
        self._layer_name = ""

    # --- BF16 weight loaders ---

    def _w13_weight_loader(self, param, loaded_weight, expert_id: int, is_w1: bool):
        tp, rank = _tp_size(), _tp_rank()
        N = self.intermediate_per_tp
        shard = loaded_weight.narrow(0, rank * N, N)
        offset = 0 if is_w1 else N
        param.data[expert_id, offset:offset + N, :].copy_(shard)

    def _w2_weight_loader(self, param, loaded_weight, expert_id: int):
        tp, rank = _tp_size(), _tp_rank()
        N = self.intermediate_per_tp
        param.data[expert_id].copy_(loaded_weight.narrow(1, rank * N, N))

    # --- FP8 weight loaders ---

    def _w13_weight_loader_fp8(self, param, loaded_weight, expert_id: int, is_w1: bool):
        tp, rank = _tp_size(), _tp_rank()
        N = self.intermediate_per_tp
        shard = loaded_weight.narrow(0, rank * N, N)
        offset = 0 if is_w1 else N
        param.data[expert_id, offset:offset + N, :].copy_(shard)

    def _w13_scale_loader(self, param, loaded_weight, expert_id: int, is_w1: bool):
        tp, rank = _tp_size(), _tp_rank()
        block_n = self.block_shape[0]
        N = self.intermediate_per_tp
        scale_rows_per_shard = math.ceil(N / block_n)
        full_scale_rows = loaded_weight.shape[0]
        rows_per_tp = full_scale_rows // tp
        src = loaded_weight.narrow(0, rank * rows_per_tp, rows_per_tp)
        offset = 0 if is_w1 else scale_rows_per_shard
        param.data[expert_id, offset:offset + rows_per_tp, :].copy_(src)

    def _w2_weight_loader_fp8(self, param, loaded_weight, expert_id: int):
        tp, rank = _tp_size(), _tp_rank()
        N = self.intermediate_per_tp
        param.data[expert_id].copy_(loaded_weight.narrow(1, rank * N, N))

    def _w2_scale_loader(self, param, loaded_weight, expert_id: int):
        tp, rank = _tp_size(), _tp_rank()
        block_k = self.block_shape[1]
        N = self.intermediate_per_tp
        scale_cols_per_shard = math.ceil(N / block_k)
        full_scale_cols = loaded_weight.shape[1]
        cols_per_tp = full_scale_cols // tp
        src = loaded_weight.narrow(1, rank * cols_per_tp, cols_per_tp)
        param.data[expert_id].copy_(src)

    # --- fast-path eligibility ---

    def _fast_path_ok(self, x: torch.Tensor) -> bool:
        if not self.use_fp8:
            return False
        if self.block_shape is None or list(self.block_shape) != [128, 128]:
            return False
        if x.dtype not in (torch.bfloat16, torch.float16):
            return False
        k = self.hidden_size
        n = self.intermediate_per_tp
        if k % 128 or n % 128:
            return False
        tk = self.top_k
        if tk <= 0 or (tk & (tk - 1)) != 0:
            return False
        if x.size(0) < 1:
            return False
        if not (x.is_contiguous() and self.w13.is_contiguous()
                and self.w2.is_contiguous()
                and self.w13_scale.is_contiguous()
                and self.w2_scale.is_contiguous()):
            return False
        return True

    def forward_impl(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Core MoE logic, callable from both eager and custom-op paths."""
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_size)

        if self._fast_path_ok(hidden_states):
            # Whether the reference keeps the routed weight out of GEMM 2's
            # epilogue is resolved *before* any capture: the predicate consults
            # the capture state, which would answer differently inside one.
            late = _dg_reduction_order(hidden_states, self.w13, self.w2)

            def body(h):
                rl = self.gate(h)
                tw, ti = self.topk_softmax(
                    rl, self.top_k, renormalize=self.renormalize)
                return _fused_moe_fp8(h, self.w13, self.w13_scale, self.w2,
                                      self.w2_scale, tw, ti, self.num_experts,
                                      self.top_k, late)

            key = (hidden_states.shape, hidden_states.dtype, late)
            out = _graphed(body, hidden_states, key, self.topk_softmax)
            if out is None:
                out = body(hidden_states)
        else:
            router_logits = self.gate(hidden_states)
            topk_weights, topk_ids = self.topk_softmax(
                router_logits, self.top_k, renormalize=self.renormalize,
            )
            out = self.fused_experts(
                hidden_states, self.w13, self.w2,
                topk_weights, topk_ids, self.num_experts,
                w13_scale=self.w13_scale,
                w2_scale=self.w2_scale,
                w13_scale_dg=getattr(self, 'w13_scale_dg', None),
                w2_scale_dg=getattr(self, 'w2_scale_dg', None),
                use_fp8_w8a8=self.use_fp8,
                block_shape=self.block_shape,
            )

        if self.tp_size > 1 and not self._use_custom_op:
            out = self.allreduce(out)

        return out.view(orig_shape)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._use_custom_op:
            out = torch.ops.fastkernels.moe_forward(hidden_states, self._layer_name)
            if self.tp_size > 1:
                out = self.allreduce(out)
            return out
        return self.forward_impl(hidden_states)
