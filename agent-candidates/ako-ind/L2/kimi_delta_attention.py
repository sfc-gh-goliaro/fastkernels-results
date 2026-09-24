"""FastKernels L2/kimi_delta_attention candidate.

Harness note
------------
The bench's module prep (``bench._prep_kimi_recurrent`` ->
``_locate_recurrent_attn``) finds the KDA layer with
``isinstance(sub, baseline.KimiDeltaAttention)``. A candidate class that is not a
subclass of the baseline class is never prepped, so every case reports
``skip: no KDA/GDN submodule found`` -- verified with a byte-identical copy of
the baseline as the candidate (5/5 SKIPPED). Subclassing keeps the class name,
the ``__init__`` signature and the ``forward`` contract while letting the prep
run; the hot paths are all overridden below.

What the score is made of
-------------------------
The bench selects five shapes, ``hidden_states[T, 2304]`` for
T = 1, 26, 64, 443, 16384, and scores the geometric mean of the per-shape
speedups. Measured baseline wall times: 1.14 / 1.11 / 1.10 / 1.10 / 4.55 ms,
against 132 / ~150 / 167 / ~190 / 4977 us of *device* time -- i.e. the four
small shapes are ~85% host overhead, and the host is the limiter even at
T=16384 because the forward contains a device-to-host sync.

Overheads removed here
----------------------
* the ``pf_state_indices[~pf_has_initial]`` boolean-mask gather: a
  ``nonzero`` + ``cudaStreamSynchronize`` + ``cudaMemcpyAsync`` inside the timed
  region on every call. ``KimiLinearMetadata`` already publishes the host-side
  answer (``any_have_initial_state`` / ``all_have_initial_state``, documented as
  existing precisely so a layer need not derive it on device), so the branch is
  taken on the host and the "no initial state" case passes ``initial_state=None``
  instead of gathering a zeroed 2 MB slice.
* advanced indexing on the recurrent state (``state[idx]`` /
  ``state[idx] = ...``) replaced by ``index_select`` / ``index_copy_``: the
  advanced-indexing path costs an ``aten::index`` + ``_index_put_impl_`` pair
  that profiled at 214 us of host time per call.
* five projection GEMMs -> two launches: ``q/k/v/b``, ``f_a`` and ``g_a`` all
  read ``hidden_states``, so they concatenate into one [12576, 2304] weight;
  ``f_b`` and ``g_b`` then share one batched GEMM (same FLOPs, one launch).
* ``einops.rearrange`` -> ``view``/``unflatten`` (no kernel, but ~10 us of host
  each), ``torch.zeros`` for ``core_attn_out`` -> the chunk kernel's own output
  buffer (it was overwritten in full immediately afterwards), and the
  ``core_attn_out[:, :n] = pf_out`` copy that followed it.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ....infra.context import get_context
from ..L1.gated_delta_rule import (
    FLA_CHUNK_SIZE,
    prepare_chunk_indices,
    prepare_chunk_offsets,
)
from ...baseline.L2.kimi_delta_attention import (
    KimiDeltaAttention as _BaselineKDA,
)


# ---------------------------------------------------------------------------
# Fused prefill front-end.
#
# The baseline spends eight launches between the projection GEMMs and the chunk
# pipeline: three ``causal_conv1d_fn`` (q/k/v), two ``l2norm_fwd`` (q/k), the
# gate cumsum, and the bf16->fp32 cast + sigmoid for beta. Every one of them is
# a per-(token, head) elementwise/reduction pass over data the next one reads
# straight back, so they fuse into a single kernel with one program per
# (chunk, head).
#
# Kept bit-compatible with the pieces it replaces:
#   * conv is width-4 causal depthwise + silu, and the l2 norm reads the
#     *bf16-rounded* conv output (``causal_conv1d_fn`` materializes bf16 before
#     ``l2norm_fwd`` sees it), so the rounding happens in the same place;
#   * the gate is ``-exp(A_log) * softplus(g + dt_bias)`` with the same
#     threshold=20 branch, cumulatively summed within the chunk and scaled by
#     RCP_LN2 for the exp2-based chunk kernels. ``tl.cumsum`` replaces FLA's
#     ``tl.dot(lower_triangular_ones, gate)``: same value, and it drops a
#     [64,64]x[64,BD] non-tf32 fp32 matmul that costs 507 us at T=16384.
#   * conv state semantics: state[slot][j] holds token ``eos - 3 + j`` of the
#     sequence, falling back to the incoming state (shift-left) when the
#     sequence is shorter than the state, and slot 0 is the null block that
#     ``causal_conv1d`` skips.
# ---------------------------------------------------------------------------
_RCP_LN2 = tl.constexpr(1.4426950216)  # keep in sync with L1/kda.py

# Cap on the backward gate exponent in the WY kernel (see the ``eb`` comment
# there). Measured max within-16-token gate drop on the benched shapes is 33.0
# log2 units (tools/gc_range.py), so this never binds in practice; k is
# l2-normalised (|k_d| <= 1), so exp2(100) * 128 accumulation terms stays at
# 1.6e32, four orders of magnitude inside fp32 range, which makes an inf
# structurally impossible instead of merely unobserved.
_EB_CAP = tl.constexpr(100.0)

# num_warps per chunk size. 8 is a sharp optimum at BT=64 (measured: 4 -> 4.91 ms
# and 16 -> 9.41 ms against 3.47 at T=16384), but a [16, BT] tile has nothing for
# 8 warps to do, so the smaller specializations get their own entry.
_NUM_WARPS = {16: 8, 32: 8, 64: 8}

# Largest chunk size for which the single-launch megakernel beats the
# three-kernel split. Measured wall (tools/prof.py), mega vs split:
#   BT=16 (T=1)  51.2 vs 78.3   BT=32 (T=26) 60.4 vs 74.7   BT=64 (T=64) 105.4 vs 97.3
# At BT=64 the merged program's peak liveness (a [128,128] fp32 state
# accumulator on top of five [64,64] fp32 tiles) pushes ptxas to 255 regs/thread
# and it still spills, and the lost instruction-level parallelism costs more than
# the two launches the fusion saves.
_MEGA_MAX_BT = 32

# Chunk size for the multi-chunk path. FLA's own 64 is not obviously right here:
# the WY triangular inverse costs T * BT^2 * 2*log2(BT) and is latency-bound, so
# a smaller BT is cheaper *and* gives more programs, at the price of more
# sequential steps in the inter-chunk state scan. Swept below.
_MC_BT = FLA_CHUNK_SIZE

# Split the front-end kernel's four independent jobs into four programs while the
# unsplit grid would leave SMs idle. B200 has 148 SMs.
_PSPLIT_MAX_PROGS = 148

# V-block of the inter-chunk state scan: grid is (head_dim // _BV, N * H), so
# this is the only parallelism that kernel has beyond the head count -- 128
# programs at _BV=32 against 148 SMs. Swept over BV x warps x stages; 32/4/3 wins
# at both T=443 and T=16384 and BV=64 with 2 warps is catastrophic (8.2 ms).
_BV = 32


def _nb_bc(bt):
    """(number of gate-reference sub-blocks, their size) for a chunk of ``bt``.

    A / A_qk factorise as exp2(gc_i - gc_n) * exp2(gc_n - gc_j) about a reference
    row n, and the two halves individually have to stay inside fp32: the forward
    half underflows and the backward half overflows once |gc_n - gc| passes ~127
    log2 units. FLA (and r1) use 16-row sub-blocks, which costs
    ``bt/BC`` *full* [bt,D]x[D,bt] dots to keep only bt/BC rows of each -- a
    BC-fold MMA waste. The measured worst-case gate drop is 33.0 log2 units per
    16 tokens (r1, tools/gc_range.py), so BC=32 bounds the exponent at ~66:
    exp2(-66) = 1.4e-20 and exp2(66) * 128 accumulation terms = 9.4e21, both
    comfortably inside fp32, and it halves the number of those dots. BC=64 does
    not work -- exp2(132) is +inf.
    """
    bc = min(bt, 32)
    return max(1, bt // bc), bc


@triton.jit
def _conv_tile(
    Xc, Wc, Sc, slot_off, bos, eos, t0, has_init, DTP,
    stride_x, stride_s_tok,
    D: tl.constexpr, BT: tl.constexpr, W: tl.constexpr, SL: tl.constexpr,
    NORM: tl.constexpr, HAS_INIT: tl.constexpr, EPS: tl.constexpr,
):
    """One [BT, D] tile of causal depthwise conv + silu (+ optional l2 norm).

    Returns the tile in ``DTP``'s element type instead of storing it, so a
    fused kernel can consume it out of registers. Rows past ``eos`` come back
    exactly zero (masked loads -> silu(0) = 0 -> 0 * rsqrt(EPS) = 0), which is
    what the separate-kernel version's ``mask=mt`` store + ``other=0.0`` load
    also produced.
    """
    od = tl.arange(0, D)
    ot = t0 + tl.arange(0, BT)
    mt = ot < eos

    acc = tl.zeros([BT, D], dtype=tl.float32)
    for j in tl.static_range(W):
        pos = ot - (W - 1 - j)
        inb = pos >= bos
        xv = tl.load(Xc + pos[:, None] * stride_x + od[None, :],
                     mask=(inb & mt)[:, None], other=0.0).to(tl.float32)
        if HAS_INIT:
            srow = pos - bos + SL
            sm = (((inb == 0) & mt) & (srow >= 0))[:, None] & has_init
            sv = tl.load(Sc + slot_off + srow[:, None] * stride_s_tok + od[None, :],
                         mask=sm, other=0.0).to(tl.float32)
            xv = tl.where(inb[:, None], xv, sv)
        wj = tl.load(Wc + od * W + j).to(tl.float32)
        acc += xv * wj[None, :]

    # silu. ``tl.fdiv(..., ieee_rounding=False)`` lowers to div.approx.f32
    # instead of div.rn: the result is rounded to bf16 immediately below, so the
    # extra mantissa bits of an IEEE divide are thrown away anyway, and this
    # kernel issues 3 * 64 * 128 of these per program.
    acc = tl.fdiv(acc, 1.0 + tl.exp(-acc), ieee_rounding=False)
    ob = acc.to(DTP.dtype.element_ty)
    if NORM:
        xf = ob.to(tl.float32)
        rs = tl.rsqrt(tl.sum(xf * xf, axis=1) + EPS)
        ob = (xf * rs[:, None]).to(DTP.dtype.element_ty)
    return ob


@triton.jit
def _conv_state(
    Xc, Sc, slot_off, bos, eos, has_init,
    stride_x, stride_s_tok,
    D: tl.constexpr, SL: tl.constexpr, SLP: tl.constexpr,
    HAS_INIT: tl.constexpr,
):
    """Conv-state writeback: state[slot][j] = token ``eos - SL + j``, falling
    back to the (shifted) incoming state for a sequence shorter than SL."""
    od = tl.arange(0, D)
    sr = tl.arange(0, SLP)
    ms = sr < SL
    spos = eos - SL + sr
    sinb = spos >= bos
    sx = tl.load(Xc + spos[:, None] * stride_x + od[None, :],
                 mask=(sinb & ms)[:, None], other=0.0)
    if HAS_INIT:
        srow2 = spos - bos + SL
        sm2 = (((sinb == 0) & ms) & (srow2 >= 0))[:, None] & has_init
        sv2 = tl.load(Sc + slot_off + srow2[:, None] * stride_s_tok + od[None, :],
                      mask=sm2, other=0.0)
        tl.debug_barrier()
        sx = tl.where(sinb[:, None], sx, sv2)
    tl.store(Sc + slot_off + sr[:, None] * stride_s_tok + od[None, :], sx,
             mask=ms[:, None])


@triton.jit
def _conv_head(
    Xc, Wc, Oc, Sc, slot_off, bos, eos, t0, has_init, do_state,
    stride_x, stride_o, stride_s_tok,
    D: tl.constexpr, BT: tl.constexpr, W: tl.constexpr,
    SL: tl.constexpr, SLP: tl.constexpr,
    NORM: tl.constexpr, HAS_INIT: tl.constexpr, EPS: tl.constexpr,
):
    od = tl.arange(0, D)
    ot = t0 + tl.arange(0, BT)
    mt = ot < eos
    ob = _conv_tile(Xc, Wc, Sc, slot_off, bos, eos, t0, has_init, Oc,
                    stride_x, stride_s_tok, D, BT, W, SL, NORM, HAS_INIT, EPS)
    tl.store(Oc + ot[:, None] * stride_o + od[None, :], ob, mask=mt[:, None])
    if do_state:
        _conv_state(Xc, Sc, slot_off, bos, eos, has_init, stride_x,
                    stride_s_tok, D, SL, SLP, HAS_INIT)


@triton.jit
def _kda_pre_kernel(
    X, WFB, QO, KO, VO, GC, BETA,
    CW, ALOG, GBIAS,
    SQ, SK, SV, CIDX, HINIT,
    cu_seqlens, chunk_indices,
    stride_x, stride_wfb, stride_o, stride_gc, stride_s_tok, stride_s_slot,
    P: tl.constexpr, H: tl.constexpr, D: tl.constexpr, BT: tl.constexpr,
    W: tl.constexpr, SL: tl.constexpr, SLP: tl.constexpr,
    HAS_INIT: tl.constexpr, THRESH: tl.constexpr, EPS: tl.constexpr,
    PSPLIT: tl.constexpr,
):
    # The four jobs in here (gate+beta, and the q/k/v conv+silu[+l2norm]) are
    # independent. With PSPLIT they become four programs instead of four phases
    # of one program, which quadruples the grid. That is worthless when the grid
    # already covers the machine, but the small shapes launch only
    # nt * H = 32..224 programs on 148 SMs, and there each program is a pure
    # latency chain -- so the split converts idle SMs into speed.
    i_t = tl.program_id(0)
    ih = tl.program_id(1)
    if PSPLIT:
        iz = tl.program_id(2)
    else:
        iz = -1
    i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
    i_c = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
    bos = tl.load(cu_seqlens + i_n).to(tl.int32)
    eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    t0 = bos + i_c * BT

    od = tl.arange(0, D)
    ot = t0 + tl.arange(0, BT)
    mt = ot < eos
    hd = ih * D + od

    do_gate = (iz < 0) or (iz == 0)
    do_q = (iz < 0) or (iz == 1)
    do_k = (iz < 0) or (iz == 2)
    do_v = (iz < 0) or (iz == 3)

    # --- gate: -exp(A_log) * softplus(raw_g + dt_bias), chunk-local cumsum ---
    # ``raw_g = f_b(f_a(hidden))`` is projected here rather than in its own
    # GEMM: this program needs exactly the [D, D] slice of f_b's weight for its
    # own head, so it costs one extra tl.dot and both the launch and the
    # [T, proj] intermediate disappear. Rounded to the activation dtype first,
    # matching the bf16 output the separate GEMM produced.
    if do_gate:
        bfa = tl.load(X + ot[:, None] * stride_x + (3 * P + H + od)[None, :],
                      mask=mt[:, None], other=0.0)
        bg = tl.dot(bfa, tl.load(WFB + od[:, None] * stride_wfb + hd[None, :])).to(
            X.dtype.element_ty).to(tl.float32)
        bg += tl.load(GBIAS + hd).to(tl.float32)[None, :]
        ba = -tl.exp(tl.load(ALOG + ih).to(tl.float32))
        sp = tl.where(bg > THRESH, bg, tl.log(1.0 + tl.exp(bg)))
        tl.store(GC + ot[:, None] * stride_gc + hd[None, :],
                 tl.cumsum(ba * sp, axis=0) * _RCP_LN2, mask=mt[:, None])

        # --- beta = sigmoid(b_proj) in fp32 ---
        bb = tl.load(X + ot * stride_x + (3 * P + ih),
                     mask=mt, other=0.0).to(tl.float32)
        tl.store(BETA + ot * H + ih, tl.sigmoid(bb), mask=mt)

    # --- conv + silu for q/k/v (q/k also l2-normalised) ---
    slot = tl.load(CIDX + i_n).to(tl.int64)
    if slot == 0:  # NULL_BLOCK_ID: causal_conv1d skips these sequences
        return  # noqa: RET502 - gate/beta above are already stored
    if HAS_INIT:
        hi = tl.load(HINIT + i_n)
    else:
        hi = False
    so = slot * stride_s_slot
    ds = (t0 + BT) >= eos
    if do_q:
        _conv_head(X + ih * D, CW + (ih * D) * W, QO + ih * D, SQ + ih * D, so,
                   bos, eos, t0, hi, ds, stride_x, stride_o, stride_s_tok,
                   D, BT, W, SL, SLP, True, HAS_INIT, EPS)
    if do_k:
        _conv_head(X + P + ih * D, CW + (P + ih * D) * W, KO + ih * D,
                   SK + ih * D, so,
                   bos, eos, t0, hi, ds, stride_x, stride_o, stride_s_tok,
                   D, BT, W, SL, SLP, True, HAS_INIT, EPS)
    if do_v:
        _conv_head(X + 2 * P + ih * D, CW + (2 * P + ih * D) * W, VO + ih * D,
                   SV + ih * D, so,
                   bos, eos, t0, hi, ds, stride_x, stride_o, stride_s_tok,
                   D, BT, W, SL, SLP, False, HAS_INIT, EPS)



# ---------------------------------------------------------------------------
# Fused A / A_qk / (I+A)^-1  (replaces 7 launches)
#
# FLA spends seven launches to get from (q, k, v, gc, beta) to the WY transform:
# ``torch.zeros`` x2 for A and A_qk, ``chunk_kda_scaled_dot_kkt`` x2 (an
# inter-sub-block and an intra-sub-block kernel), ``zeros_like`` + the 16x16 ->
# 64x64 triangular-inverse merge. Measured host cost of that group alone is
# 68 us (kkt) + 29 us (solve_tril). All of it is per-(chunk, head) work on a
# [64, 64] tile that fits in one program's registers, so it becomes one kernel.
#
# The two structural simplifications versus FLA:
#
# 1. FLA splits the 64x64 tile into 4x4 sub-blocks of 16 and uses a different
#    gate reference row per sub-block, because the naive factorisation
#    exp2(gc_i - gc_j) = exp2(gc_i) * exp2(-gc_j) overflows: gc is a cumsum of
#    negative gates, so exp2(-gc_j) grows without bound down the chunk. But the
#    *block-start* reference works for the diagonal block too (gc is monotone
#    decreasing, so for any j >= 16*I, gc[16*I] - gc[j] <= 0), which collapses
#    FLA's inter+intra split into one loop of four rank-128 matmuls. Entries
#    whose exponent would be positive are the ones the causal mask discards
#    anyway; ``tl.minimum(expo, 0)`` clamps them so no inf is ever produced,
#    and every *kept* entry has a non-positive exponent so the clamp is a no-op
#    on it.
# 2. The triangular inverse is the same algorithm as FLA (four 16x16 forward
#    substitutions, then two block-doubling merges), but written on the whole
#    [64, 64] tile: the four substitutions run simultaneously because a
#    block-diagonal Z makes ``sum(V[:, None] * Z, 0)`` compute all four
#    per-block row updates at once, and each merge is
#    X_2s = X_s - X_s @ A_offdiag @ X_s, which expands to exactly FLA's
#    Ai_21/31/32/41/42/43 expressions (verified by hand).
# ---------------------------------------------------------------------------


@triton.jit
def _kda_wy_kernel(
    Q, K, V, GC, BETA, AQK, U, W, KG,
    cu_seqlens, chunk_indices,
    stride_qk, stride_gc, stride_beta, stride_a,
    scale,
    H: tl.constexpr, D: tl.constexpr, BT: tl.constexpr, NB: tl.constexpr,
    BC: tl.constexpr, NLV: tl.constexpr, PREC: tl.constexpr,
    NEED_W: tl.constexpr,
):
    i_t = tl.program_id(0)
    ih = tl.program_id(1)
    i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
    i_c = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
    bos = tl.load(cu_seqlens + i_n).to(tl.int32)
    eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    t0 = bos + i_c * BT

    o = tl.arange(0, BT)
    od = tl.arange(0, D)
    hd = ih * D + od
    ot = t0 + o
    mt = ot < eos

    bq = tl.load(Q + ot[:, None] * stride_qk + hd[None, :],
                 mask=mt[:, None], other=0.0)
    bk = tl.load(K + ot[:, None] * stride_qk + hd[None, :],
                 mask=mt[:, None], other=0.0)
    bg = tl.load(GC + ot[:, None] * stride_gc + hd[None, :],
                 mask=mt[:, None], other=0.0)
    bb = tl.load(BETA + ot * stride_beta + ih, mask=mt, other=0.0)

    lower = o[:, None] > o[None, :]
    diag = o[:, None] == o[None, :]
    blk = o // BC

    bA = tl.zeros([BT, BT], dtype=tl.float32)
    bAq = tl.zeros([BT, BT], dtype=tl.float32)
    for ib in tl.static_range(NB):
        gn = tl.load(GC + (t0 + ib * BC) * stride_gc + hd,
                     mask=(od >= 0) & ((t0 + ib * BC) < eos), other=0.0)
        # gc is a cumsum of negative gates, so it decreases down the chunk.
        # ``ef`` is exp2(gc_i - gc_n) with i >= n on every row this iteration
        # keeps -> exponent <= 0, and the clamp only touches discarded rows
        # (it just stops exp2 from producing inf there).
        # ``eb`` is exp2(gc_n - gc_j) and must NOT be clamped to 0: for j inside
        # block ib (the diagonal sub-block, which FLA handles with a separate
        # row loop) the exponent is *positive*, and clamping it to 0 silently
        # drops the intra-block decay -- that was a real bug, see ITERATIONS.md.
        # On every (i, j) the mask keeps it is bounded by the gate decay across
        # one 16-token block; ``_EB_CAP`` only bites on columns belonging to
        # blocks after ib, whose outputs ``lower``/``diag`` discards, and stops
        # those from turning into inf (and then, via 0 * inf, into a NaN that
        # the triangular inverse would spread into live rows).
        ef = tl.exp2(tl.minimum(bg - gn[None, :], 0.0))
        eb = tl.exp2(tl.minimum(gn[None, :] - bg, _EB_CAP))
        kt = tl.trans(bk * eb)
        # ``mt`` in the row predicate matters: for a chunk shorter than 64 the
        # padded rows have bk == 0 and an unbounded ``eb``, so kt picks up
        # 0 * inf = NaN there. Those entries are never *stored* (the stores mask
        # on mt), but the triangular inverse below multiplies the whole [64, 64]
        # tile, and 0 * NaN = NaN would carry the padding into live rows.
        rows = ((blk == ib) & mt)[:, None]
        bA = tl.where(
            rows & lower,
            tl.dot(bk * ef, kt, input_precision=PREC) * bb[:, None],
            bA)
        bAq = tl.where(
            rows & (lower | diag),
            tl.dot(bq * ef * scale, kt, input_precision=PREC),
            bAq)

    tl.store(AQK + ot[:, None] * stride_a + ih * BT + o[None, :],
             bAq.to(AQK.dtype.element_ty), mask=mt[:, None])

    # ---- (I + bA)^-1, bA strictly lower ----------------------------------
    # Block doubling all the way down: X_2s = X_s - X_s @ A_off(s) @ X_s, where
    # A_off(s) is bA restricted to (same 2s-group) & (different s-block). At
    # s = 1 X_s is the identity, so the first level is just I - A_off(1) and
    # costs no matmul; the remaining log2(BT) - 1 levels are two [BT, BT]
    # tensor-core matmuls each.
    #
    # FLA instead runs a 14-step sequential forward substitution per 16x16
    # diagonal block and only uses matmuls for the two merges. Transcribing
    # that onto the full [64, 64] tile needs 28 reduce-along-axis-0 passes,
    # which on a 64x64 fp32 tile spread over 8 warps are cross-warp shuffles --
    # measured 2.30 ms at T=16384 for this kernel. Doubling is more FLOPs and
    # far less of them are serialised.
    off = tl.where((o[:, None] // 2 == o[None, :] // 2) & lower, bA, 0.0)
    Z = tl.where(diag, 1.0, 0.0) - off
    for lg in tl.static_range(1, NLV):
        sb = 1 << lg
        off = tl.where((o[:, None] // (2 * sb) == o[None, :] // (2 * sb))
                       & (o[:, None] // sb > o[None, :] // sb), bA, 0.0)
        Z = Z - tl.dot(tl.dot(Z, off, input_precision=PREC), Z,
                       input_precision=PREC)

    # ---- WY transform, in the same program that produced the inverse -----
    # ``recompute_w_u_fwd_kda`` is a separate launch only because FLA writes the
    # inverse to global memory first. Everything it needs (the inverse, k, v,
    # beta, gc) is already live here, so the inverse never has to be stored.
    # The bf16 round-trip is kept: ``solve_tril`` materialises bf16 and
    # ``recompute_w_u`` reads it back, so the dots see the same operand.
    Zb = Z.to(K.dtype.element_ty)
    bv = tl.load(V + ot[:, None] * stride_qk + hd[None, :],
                 mask=mt[:, None], other=0.0)
    tl.store(U + ot[:, None] * stride_qk + hd[None, :],
             tl.dot(Zb, (bv * bb[:, None]).to(V.dtype.element_ty)).to(
                 U.dtype.element_ty),
             mask=mt[:, None])
    if NEED_W:
        eg = tl.exp2(bg)
        tl.store(W + ot[:, None] * stride_qk + hd[None, :],
                 tl.dot(Zb, (bk * bb[:, None] * eg).to(K.dtype.element_ty)).to(
                     W.dtype.element_ty),
                 mask=mt[:, None])
        last = tl.minimum(t0 + BT, eos) - 1
        gl = tl.load(GC + last * stride_gc + hd)
        tl.store(KG + ot[:, None] * stride_qk + hd[None, :],
                 (bk * tl.exp2(gl[None, :] - bg)).to(KG.dtype.element_ty),
                 mask=mt[:, None])


# ---------------------------------------------------------------------------
# Inter-chunk state scan (replaces the last FLA wrapper on the hot path).
#
# ``chunk_gated_delta_rule_fwd_h`` is the only FLA entry point left in the
# multi-chunk path and it is an expensive way to reach one kernel: a
# ``prepare_chunk_offsets`` cache lookup, two ``new_empty`` and one
# ``empty_like``, and a kwargs-style launch of an ``@triton.autotune``d kernel
# (r1 measured autotuned launches at 13.3 us of host against 10.0 for a plain
# one). Its results then need ``recurrent_state.index_copy_`` to get the final
# state into the caller's slot, which is another 8.5 us of host and an
# ``index_elementwise_kernel``.
#
# This is the same recurrence, specialised to what this layer actually calls it
# with (K = V = head_dim <= 128, gk only, exp2, varlen, save v_new), launched
# positionally with cached output buffers, and -- when there is a single prefill
# sequence, which is every benched case -- writing the final state directly into
# ``recurrent_state[slot]`` so the scatter disappears entirely.
# ---------------------------------------------------------------------------


@triton.jit
def _kda_h_kernel(
    KG, U, WW, VNEW, GC, HS, RSTATE, CIDX, HINIT,
    cu_seqlens, chunk_offsets,
    stride_t, stride_gc, stride_rs,
    H: tl.constexpr, D: tl.constexpr, BT: tl.constexpr, BV: tl.constexpr,
    HAS_INIT: tl.constexpr,
):
    i_v = tl.program_id(0)
    i_nh = tl.program_id(1)
    i_n = i_nh // H
    i_h = i_nh % H
    bos = tl.load(cu_seqlens + i_n).to(tl.int32)
    eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    nt = tl.cdiv(eos - bos, BT)

    ov = i_v * BV + tl.arange(0, BV)
    od = tl.arange(0, D)
    ot = tl.arange(0, BT)

    sbase = (tl.load(CIDX + i_n).to(tl.int64) * stride_rs + i_h * D * D
             + ov[:, None] * D + od[None, :])
    bh = tl.zeros([BV, D], dtype=tl.float32)
    if HAS_INIT:
        # read the incoming state out of the caller's own slot (masked by
        # has_initial_state) instead of an index_select'd copy of it
        bh += tl.where(tl.load(HINIT + i_n),
                       tl.load(RSTATE + sbase), 0.0)

    kgp = KG + (bos * H + i_h) * D
    up = U + (bos * H + i_h) * D
    wp = WW + (bos * H + i_h) * D
    vnp = VNEW + (bos * H + i_h) * D
    gcp = GC + bos * stride_gc + i_h * D
    hp = HS + ((boh * H) + i_h) * D * D

    for i_t in range(nt):
        t0 = i_t * BT
        mt = (t0 + ot) < (eos - bos)
        # h for this chunk is the state *before* it
        tl.store(hp + i_t * H * D * D + ov[:, None] * D + od[None, :],
                 bh.to(HS.dtype.element_ty))

        bw = tl.load(wp + (t0 + ot)[:, None] * stride_t + od[None, :],
                     mask=mt[:, None], other=0.0)
        bu = tl.load(up + (t0 + ot)[:, None] * stride_t + ov[None, :],
                     mask=mt[:, None], other=0.0)
        bv = bu - tl.dot(bw, tl.trans(bh).to(bw.dtype))
        tl.store(vnp + (t0 + ot)[:, None] * stride_t + ov[None, :],
                 bv.to(VNEW.dtype.element_ty), mask=mt[:, None])

        last = tl.minimum(t0 + BT, eos - bos) - 1
        gl = tl.load(gcp + last * stride_gc + od)
        bh *= tl.exp2(gl)[None, :]

        bk = tl.load(kgp + (t0 + ot)[:, None] * stride_t + od[None, :],
                     mask=mt[:, None], other=0.0)
        bh += tl.trans(tl.dot(tl.trans(bk), bv.to(bk.dtype)))

    tl.store(RSTATE + sbase, bh)


# ---------------------------------------------------------------------------
# Fused output + gated RMS norm (replaces 2 launches)
#
#   o = A_qk(lower) @ v_new  +  scale * (q * exp2(gc)) @ h_chunk^T
#   y = rms_norm(o) * o_norm.weight * sigmoid(g_proj)
#
# ``chunk_gla_fwd_o_gk`` writes ``o`` to global memory only for
# ``FusedRMSNormGated`` to read it straight back and reduce over the same 128
# lanes the program already holds. ``o`` is rounded to bf16 before the norm
# reads it, matching the bf16 ``core_attn_out`` buffer the two-launch version
# passes between them.
# ---------------------------------------------------------------------------


@triton.jit
def _kda_out_kernel(
    Q, GC, VNEW, HS, AQK, X, WGB, WNORM, Y,
    cu_seqlens, chunk_indices,
    stride_qk, stride_gc, stride_v, stride_a, stride_x, stride_wgb, stride_y,
    scale, eps, P: tl.constexpr,
    H: tl.constexpr, D: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr,
):
    i_t = tl.program_id(0)
    ih = tl.program_id(1)
    i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
    i_c = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
    bos = tl.load(cu_seqlens + i_n).to(tl.int32)
    eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    t0 = bos + i_c * BT

    o = tl.arange(0, BT)
    od = tl.arange(0, D)
    ok = tl.arange(0, BK)
    hd = ih * D + od
    ot = t0 + o
    mt = ot < eos

    acc = tl.zeros([BT, D], dtype=tl.float32)
    hbase = HS + (i_t * H + ih) * D * D
    for ik in tl.static_range(D // BK):
        cq = ih * D + ik * BK + ok
        bq = tl.load(Q + ot[:, None] * stride_qk + cq[None, :],
                     mask=mt[:, None], other=0.0).to(tl.float32)
        bg = tl.load(GC + ot[:, None] * stride_gc + cq[None, :],
                     mask=mt[:, None], other=0.0)
        qg = (bq * scale * tl.exp2(bg)).to(Q.dtype.element_ty)
        bh = tl.load(hbase + od[:, None] * D + (ik * BK + ok)[None, :])
        acc += tl.dot(qg, tl.trans(bh))

    ba = tl.load(AQK + ot[:, None] * stride_a + ih * BT + o[None, :],
                 mask=mt[:, None], other=0.0)
    ba = tl.where(o[:, None] >= o[None, :], ba, 0.0)
    bv = tl.load(VNEW + ot[:, None] * stride_v + hd[None, :],
                 mask=mt[:, None], other=0.0)
    acc += tl.dot(ba, bv)

    of = acc.to(Y.dtype.element_ty).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(of * of, axis=1) / D + eps)
    bw = tl.load(WNORM + od).to(tl.float32)
    # output gate g_b(g_a(hidden)), projected here for the same reason raw_g is
    # projected inside the front-end kernel: one [D, D] weight slice per head.
    bga = tl.load(X + ot[:, None] * stride_x + (3 * P + H + D + od)[None, :],
                  mask=mt[:, None], other=0.0)
    bgt = tl.dot(bga, tl.load(WGB + od[:, None] * stride_wgb + hd[None, :])).to(
        Y.dtype.element_ty).to(tl.float32)
    y = of * rstd[:, None] * bw[None, :] * tl.sigmoid(bgt)
    tl.store(Y + ot[:, None] * stride_y + hd[None, :],
             y.to(Y.dtype.element_ty), mask=mt[:, None])



# ---------------------------------------------------------------------------
# Single-chunk megakernel: the entire post-projection layer in ONE launch.
#
# When every prefill sequence fits in one chunk and starts from a zero recurrent
# state -- prefill of a fresh short sequence, four of the five benched shapes --
# the front-end, the WY transform and the output all run with
# exactly one program per (sequence, head) and each one only ever reads what the
# previous one wrote *for its own program*. So there is nothing to synchronize
# across programs and the three can be one kernel with q/k/v/gc/beta/A_qk/u
# living in registers instead of taking a round trip through HBM.
#
# Cost removed at T <= 64 (measured, tools/prof.py --mode prof):
#   * 2 of the 5 launches -- 4.2 us of host each (STEP 0's marginal launch cost);
#   * 2 kernel drains, i.e. two full grid-wide tail/head serialisations;
#   * 7 [BT, 128] tiles + one [BT, BT] tile of store-then-reload traffic.
#
# What is deliberately *not* fused in: the two ``torch.mm`` calls. They are
# weight-read bound (58 MB for the fused input projection, 19 MB for o_proj) and
# folding the low-rank f_a/g_a GEMM in would re-read its weight once per head,
# i.e. 32x.
#
# Numerics are held identical to the three-kernel version by reproducing what
# the intermediate stores/loads did to padded rows: ``gc`` and ``beta`` are
# forced to 0 past ``eos`` (a masked store followed by an ``other=0.0`` load did
# that implicitly; in registers the cumsum and the sigmoid would otherwise carry
# a nonzero value there), and every fp32 -> bf16 rounding the separate kernels
# performed on an intermediate is performed here too.
# ---------------------------------------------------------------------------


@triton.jit
def _kda_mega_kernel(
    X, WFB, WGB, WNORM, CW, ALOG, GBIAS,
    SQ, SK, SV, RSTATE, CIDX, HINIT, Y, GC,
    QI, KI, VI, BETAI,
    cu_seqlens, chunk_indices,
    stride_x, stride_wfb, stride_wgb, stride_y, stride_gc, stride_s_tok,
    stride_s_slot, stride_rs, stride_qk, stride_beta, scale, eps,
    P: tl.constexpr, H: tl.constexpr, D: tl.constexpr, BT: tl.constexpr,
    W: tl.constexpr, SL: tl.constexpr, SLP: tl.constexpr,
    NB: tl.constexpr, BC: tl.constexpr, NLV: tl.constexpr,
    PREC: tl.constexpr, THRESH: tl.constexpr, EPS: tl.constexpr,
    HAS_INIT: tl.constexpr, FUSE_PRE: tl.constexpr,
):
    # FUSE_PRE=False drops the front-end half and reads q/k/v/gc/beta from the
    # buffers ``_kda_pre_kernel`` wrote, i.e. this becomes a merged WY+output
    # kernel. That is the right split at BT=64: the fully fused form spills
    # (three [64,128] fp32 conv accumulators on top of the [64,64] tiles) and
    # measured 97 us at T=64 against 79 for the five-launch pipeline, while the
    # front-end on its own benefits from PSPLIT, which a single fused program
    # cannot use.
    i_t = tl.program_id(0)
    ih = tl.program_id(1)
    i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
    bos = tl.load(cu_seqlens + i_n).to(tl.int32)
    eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    t0 = bos

    o = tl.arange(0, BT)
    od = tl.arange(0, D)
    hd = ih * D + od
    ot = t0 + o
    mt = ot < eos

    # ---- gate: -exp(A_log) * softplus(f_b(f_a(x)) + dt_bias), cumsum ---------
    if FUSE_PRE:
        bfa = tl.load(X + ot[:, None] * stride_x + (3 * P + H + od)[None, :],
                      mask=mt[:, None], other=0.0)
        bg = tl.dot(bfa,
                    tl.load(WFB + od[:, None] * stride_wfb + hd[None, :])).to(
            X.dtype.element_ty).to(tl.float32)
        bg += tl.load(GBIAS + hd).to(tl.float32)[None, :]
        ba = -tl.exp(tl.load(ALOG + ih).to(tl.float32))
        sp = tl.where(bg > THRESH, bg, tl.log(1.0 + tl.exp(bg)))
    # zero past eos: the three-kernel version stored gc with ``mask=mt`` and
    # reloaded it with ``other=0.0``, so padded rows were 0 there. In registers
    # the cumsum keeps accumulating (a masked-out row still has dt_bias), and a
    # nonzero padded gc would feed ``exp2(gn - gc)`` in the WY block below.
        bg = tl.where(mt[:, None],
                      tl.cumsum(ba * sp, axis=0) * _RCP_LN2, 0.0)
    # The WY block below needs gc's row at each 16-token sub-block start, and the
    # state writeback needs its last valid row. Extracting a row from a register
    # tile costs an axis-0 block reduction (shared-memory staged, barriered) and
    # there are NB + 1 of them, which measured *more* than the two launches the
    # fusion saves. Round-tripping the tile through its own global buffer and
    # loading the rows back as scalars is what the three-kernel version did, and
    # is far cheaper: one 32 KB store plus a `bar.sync` (which orders a block's
    # global writes for its own subsequent reads).
    if FUSE_PRE:
        tl.store(GC + ot[:, None] * stride_gc + hd[None, :], bg,
                 mask=mt[:, None])
        tl.debug_barrier()
    # ...and once it is there, the [BT, D] fp32 tile does not need to stay in
    # registers either. At BT=64 it is 32 KB of the ~255 KB a program can hold,
    # and it is live across the whole triangular inverse otherwise. Reloading it
    # at each of its two use sites costs two 32 KB reads.
    bgcs = GC + ot[:, None] * stride_gc + hd[None, :]

    # ---- beta = sigmoid(b_proj); 0 past eos for the same reason -------------
    slot = tl.load(CIDX + i_n).to(tl.int64)
    if HAS_INIT:
        hi = tl.load(HINIT + i_n)
    else:
        hi = False
    so = slot * stride_s_slot
    if FUSE_PRE:
        bb = tl.load(X + ot * stride_x + (3 * P + ih),
                     mask=mt, other=0.0).to(tl.float32)
        bb = tl.where(mt, tl.sigmoid(bb), 0.0)
        bq = _conv_tile(X + ih * D, CW + (ih * D) * W, SQ + ih * D, so,
                        bos, eos, t0, hi, Y, stride_x, stride_s_tok,
                        D, BT, W, SL, True, HAS_INIT, EPS)
        bk = _conv_tile(X + P + ih * D, CW + (P + ih * D) * W, SK + ih * D, so,
                        bos, eos, t0, hi, Y, stride_x, stride_s_tok,
                        D, BT, W, SL, True, HAS_INIT, EPS)
    else:
        bb = tl.load(BETAI + ot * stride_beta + ih, mask=mt, other=0.0)
        bq = tl.load(QI + ot[:, None] * stride_qk + hd[None, :],
                     mask=mt[:, None], other=0.0)
        bk = tl.load(KI + ot[:, None] * stride_qk + hd[None, :],
                     mask=mt[:, None], other=0.0)

    # ---- A, A_qk ------------------------------------------------------------
    lower = o[:, None] > o[None, :]
    diag = o[:, None] == o[None, :]
    blk = o // BC

    bg = tl.load(bgcs, mask=mt[:, None], other=0.0)
    bA = tl.zeros([BT, BT], dtype=tl.float32)
    bAq = tl.zeros([BT, BT], dtype=tl.float32)
    for ib in tl.static_range(NB):
        gn = tl.load(GC + (t0 + ib * BC) * stride_gc + hd,
                     mask=(t0 + ib * BC) < eos, other=0.0)
        ef = tl.exp2(tl.minimum(bg - gn[None, :], 0.0))
        eb = tl.exp2(tl.minimum(gn[None, :] - bg, _EB_CAP))
        kt = tl.trans(bk * eb)
        rows = ((blk == ib) & mt)[:, None]
        bA = tl.where(
            rows & lower,
            tl.dot(bk * ef, kt, input_precision=PREC) * bb[:, None],
            bA)
        bAq = tl.where(
            rows & (lower | diag),
            tl.dot(bq * ef * scale, kt, input_precision=PREC),
            bAq)

    # ---- (I + bA)^-1 by block doubling -------------------------------------
    off = tl.where((o[:, None] // 2 == o[None, :] // 2) & lower, bA, 0.0)
    Z = tl.where(diag, 1.0, 0.0) - off
    for lg in tl.static_range(1, NLV):
        sb = 1 << lg
        off = tl.where((o[:, None] // (2 * sb) == o[None, :] // (2 * sb))
                       & (o[:, None] // sb > o[None, :] // sb), bA, 0.0)
        Z = Z - tl.dot(tl.dot(Z, off, input_precision=PREC), Z,
                       input_precision=PREC)
    Zb = Z.to(Y.dtype.element_ty)

    # ---- u = Z @ (beta * v) -------------------------------------------------
    if FUSE_PRE:
        bv = _conv_tile(X + 2 * P + ih * D, CW + (2 * P + ih * D) * W,
                        SV + ih * D, so, bos, eos, t0, hi, Y, stride_x,
                        stride_s_tok, D, BT, W, SL, False, HAS_INIT, EPS)
    else:
        bv = tl.load(VI + ot[:, None] * stride_qk + hd[None, :],
                     mask=mt[:, None], other=0.0)
    bu = tl.dot(Zb, (bv * bb[:, None]).to(Y.dtype.element_ty)).to(
        Y.dtype.element_ty)

    # ---- o = tril(A_qk) @ u, then the gated RMS norm ------------------------
    # ``bAq`` went through global memory as bf16 between the two kernels, so it
    # is rounded here as well before the dot.
    ba = tl.where(o[:, None] >= o[None, :], bAq.to(Y.dtype.element_ty), 0.0)
    of = tl.dot(ba.to(Y.dtype.element_ty), bu).to(Y.dtype.element_ty).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(of * of, axis=1) / D + eps)
    bw = tl.load(WNORM + od).to(tl.float32)
    bga = tl.load(X + ot[:, None] * stride_x + (3 * P + H + D + od)[None, :],
                  mask=mt[:, None], other=0.0)
    bgt = tl.dot(bga, tl.load(WGB + od[:, None] * stride_wgb + hd[None, :])).to(
        Y.dtype.element_ty).to(tl.float32)
    tl.store(Y + ot[:, None] * stride_y + hd[None, :],
             (of * rstd[:, None] * bw[None, :] * tl.sigmoid(bgt)).to(
                 Y.dtype.element_ty),
             mask=mt[:, None])

    # ---- conv states + final recurrent state -------------------------------
    if FUSE_PRE:
        _conv_state(X + ih * D, SQ + ih * D, so, bos, eos, hi, stride_x,
                    stride_s_tok, D, SL, SLP, HAS_INIT)
        _conv_state(X + P + ih * D, SK + ih * D, so, bos, eos, hi, stride_x,
                    stride_s_tok, D, SL, SLP, HAS_INIT)
        _conv_state(X + 2 * P + ih * D, SV + ih * D, so, bos, eos, hi,
                    stride_x, stride_s_tok, D, SL, SLP, HAS_INIT)

    # state[v, k] = sum_t u[t, v] * kg[t, k],  kg = k * exp2(gc_last - gc).
    gl = tl.load(GC + (eos - 1) * stride_gc + hd)
    kg = (bk * tl.exp2(gl[None, :] - tl.load(bgcs, mask=mt[:, None], other=0.0))
          ).to(Y.dtype.element_ty)
    # The separate output kernel split this over k in BK-wide blocks because it
    # reloaded k and gc from global per block; here the whole [BT, D] kg tile is
    # already live, so one [D, BT] x [BT, D] dot writes the state in full.
    base = RSTATE + slot * stride_rs + ih * D * D
    tl.store(base + od[:, None] * D + od[None, :], tl.dot(tl.trans(bu), kg))


class KimiDeltaAttention(_BaselineKDA):
    def __init__(self, config, layer_idx: int, quant_config: dict | None = None):
        super().__init__(config, layer_idx, quant_config)
        # Built lazily on the first forward: the harness loads weights (and runs
        # ``_prepare_module``'s fp32 -> bf16 cast) after ``__init__``, so
        # anything derived from a weight has to wait until it is final.
        self._fused_w = None
        self._ws = {}
        self._x_ws = {}
        self._y_ws = {}
        self._gc_ws = {}
        self._h_ws = {}
        self._co_ws = {}
        self._ci_ws = {}

    # ------------------------------------------------------------------ setup
    def _build_fused(self) -> None:
        p = self.head_dim * self.local_num_heads
        self._p = p
        # [3*proj + H + 2*head_dim, hidden]: q/k/v/b/f_a/g_a in one GEMM.
        self._w_in = torch.cat(
            [
                self.qkvb_proj.weight.data,
                self.f_a_proj.weight.data,
                self.g_a_proj.weight.data,
            ],
            0,
        ).contiguous()
        # [2*head_dim, 2*proj] block diagonal so f_b and g_b are one plain GEMM
        # over the [T, 2*head_dim] f_a|g_a activations. A batched GEMM would do
        # half the FLOPs, but needs unflatten + permute + select on the host
        # every call and cannot take a contiguous operand; at T<=64 the layer is
        # launch-bound, so the host ops cost more than the extra FLOPs.
        wab = torch.zeros(2 * self.head_dim, 2 * p, dtype=self._w_in.dtype,
                          device=self._w_in.device)
        wab[:self.head_dim, :p] = self.f_b_proj.weight.data.t()
        wab[self.head_dim:, p:] = self.g_b_proj.weight.data.t()
        self._w_ab = wab.contiguous()
        # [head_dim, proj] each: the per-head [D, D] slice a kernel program needs
        # is a contiguous column block of these.
        self._w_fb_t = self.f_b_proj.weight.data.t().contiguous()
        self._w_gb_t = self.g_b_proj.weight.data.t().contiguous()
        self._cw_q = self.q_conv1d.weight.data.view(p, self.conv_size)
        self._cw_k = self.k_conv1d.weight.data.view(p, self.conv_size)
        self._cw_v = self.v_conv1d.weight.data.view(p, self.conv_size)
        # One [3*proj, width] weight so the fused front-end addresses q/k/v conv
        # channels with a single base pointer.
        self._cw_all = torch.cat(
            [self._cw_q, self._cw_k, self._cw_v], 0
        ).contiguous()
        self._alog = self.A_log.data.view(-1)
        self._dtb = self.dt_bias.data.view(-1)
        self._scale = self.head_dim ** -0.5
        # ``.t()`` views held once instead of rebuilt per call.
        self._w_in_t = self._w_in.t()
        self._w_out_t = self.o_proj.weight.data.t()
        self._fused_w = True

    def _hbuf(self, nt, dtype, device):
        b = self._h_ws.get((nt, dtype, device))
        if b is None:
            b = torch.empty((nt, self.local_num_heads, self.head_dim,
                             self.head_dim), dtype=dtype, device=device)
            self._h_ws[(nt, dtype, device)] = b
        return b

    def _coff(self, cu, bt):
        key = (id(cu), bt)
        ent = self._co_ws.get(key)
        if ent is None or ent[0] is not cu:
            co = prepare_chunk_offsets(cu, bt)
            if len(self._co_ws) > 32:
                self._co_ws.clear()
            self._co_ws[key] = (cu, co)
            return co
        return ent[1]

    def _gcbuf(self, t, device):
        b = self._gc_ws.get((t, device))
        if b is None:
            b = torch.empty((t, self._p), dtype=torch.float32, device=device)
            self._gc_ws[(t, device)] = b
        return b

    def _ybuf(self, t, dtype, device):
        b = self._y_ws.get((t, dtype, device))
        if b is None:
            b = torch.empty((t, self._p), dtype=dtype, device=device)
            self._y_ws[(t, dtype, device)] = b
        return b

    def _xbuf(self, t, dtype, device):
        b = self._x_ws.get((t, dtype, device))
        if b is None:
            b = torch.empty((t, self._w_in.shape[0]), dtype=dtype, device=device)
            self._x_ws[(t, dtype, device)] = b
        return b

    # -------------------------------------------------------------- metadata
    @staticmethod
    def _cu(meta):
        cu = meta.query_start_loc_int32
        if cu is None:
            cu = meta.query_start_loc.to(torch.int32)
        return cu

    @staticmethod
    def _idx_long(meta):
        idx = meta.state_indices_long
        if idx is None:
            idx = meta.state_indices.long()
        return idx


    # ---------------------------------------------------------------- fused
    def _scratch(self, n, bt, dtype, device):
        """Per-shape scratch views, built once and reused.

        Two `torch.empty` calls instead of ~14 (the aten allocation calls alone
        profiled at ~52 us of host time per forward against 87 us of device
        time), and -- the larger saving at T=64 -- the ~16 slice/view calls that
        carve the buffer up are done once per shape rather than once per
        forward, at ~1.5 us of host each.

        The buffers are pure intermediates: every one is fully written before it
        is read, and the tensor this layer returns is a fresh GEMM output, so
        the reuse is not observable to a caller.
        """
        key = (n, bt, dtype, device)
        plan = self._ws.get(key)
        if plan is None:
            p, h = self._p, self.local_num_heads
            np_, na = n * p, n * h * bt
            bf = torch.empty(7 * np_ + na, dtype=dtype, device=device)
            f32 = torch.empty(np_, dtype=torch.float32, device=device)
            q, k, v, u, w, kg = (bf[i * np_:(i + 1) * np_].view(n, p)
                                 for i in range(6))
            plan = (q, k, v, u, w, kg,
                    bf[6 * np_:6 * np_ + na].view(n, h, bt),
                    bf[6 * np_ + na:].view(n, p),
                    f32.view(n, p),
                    torch.empty((n, h), dtype=torch.float32, device=device))
            self._ws[key] = plan
        return plan

    def _pre(self, x, n, kda_state, meta, cu, chunk_indices, bufs, bt, nw):
        """Run the fused conv+l2norm+gate+beta front-end. Returns q,k,v,gc,beta."""
        p, h, d = self._p, self.local_num_heads, self.head_dim
        dev = x.device
        q, k, v, gc, beta = bufs
        sq = kda_state.q_conv_states[self.layer_idx]
        sk = kda_state.k_conv_states[self.layer_idx]
        sv = kda_state.v_conv_states[self.layer_idx]
        w = self.conv_size
        psplit = chunk_indices.shape[0] * h < _PSPLIT_MAX_PROGS
        grid = ((chunk_indices.shape[0], h, 4) if psplit
                else (chunk_indices.shape[0], h))
        _kda_pre_kernel[grid](
            x, self._w_fb_t, q, k, v, gc, beta,
            self._cw_all, self._alog, self._dtb,
            sq, sk, sv, meta.state_indices, meta.has_initial_state,
            cu, chunk_indices,
            x.stride(0), p, p, p, sq.stride(1), sq.stride(0),
            P=p, H=h, D=d, BT=bt,
            W=w, SL=w - 1, SLP=triton.next_power_of_2(w - 1),
            HAS_INIT=bool(meta.any_have_initial_state),
            THRESH=20.0, EPS=1e-6, PSPLIT=psplit,
            num_warps=nw,
        )
        return q, k, v, gc, beta

    def _prefill_fused(self, x, n, kda_state, meta):
        p, h, d = self._p, self.local_num_heads, self.head_dim
        # --- chunk size ------------------------------------------------------
        # A sequence that fits in one chunk is computed identically for any
        # BT >= its length: the extra rows load as zeros, contribute exactly 0
        # to every dot, and are masked out of every store. So for a fresh
        # single-chunk prefill BT shrinks to the smallest tile the tensor cores
        # accept (16), which is a pure waste removal, not an approximation --
        # and it matters a lot, because the [BT, BT] triangular inverse is
        # 2*(log2(BT) - 1) serial BT^3 dots and at BT=64 that chain alone is
        # ~20 us of latency for a *one-token* sequence.
        bt = _MC_BT
        mq = meta.max_query_len
        if not meta.any_have_initial_state and mq is not None and mq <= bt:
            bt = max(16, triton.next_power_of_2(int(mq)))
        nw = _NUM_WARPS[bt]
        nb, bc = _nb_bc(bt)
        cu = self._cu(meta)
        # own identity cache: FLA's ``tensor_cache`` linear-scans up to 8 entries
        # comparing arg tuples, ~3 us of the ~100 us host budget at T=64.
        ckey = (id(cu), bt)
        ent = self._ci_ws.get(ckey)
        if ent is None or ent[0] is not cu:
            ci = prepare_chunk_indices(cu, bt)
            if len(self._ci_ws) > 32:
                self._ci_ws.clear()
            # the cu tensor is kept in the entry so its id cannot be recycled
            # under us, and re-checked with ``is`` in case it ever were
            self._ci_ws[ckey] = (cu, ci)
        else:
            ci = ent[1]
        nt = ci.shape[0]
        recurrent_state = kda_state.recurrent_states[self.layer_idx]
        li = self.layer_idx
        # every prefill sequence is exactly one chunk and starts from zero: the
        # whole layer tail is one kernel, so none of q/k/v/gc/beta/A_qk/u needs a
        # buffer at all.
        if not meta.any_have_initial_state and nt == meta.num_prefills:
            # single-chunk: the whole tail is one kernel. Up to BT=32 it also
            # absorbs the front-end (3 launches for the layer); at BT=64 the
            # front-end stays separate so it can use PSPLIT (4 launches).
            fuse_pre = bt <= _MEGA_MAX_BT
            if fuse_pre:
                y = self._ybuf(n, x.dtype, x.device)
                q = k = v = beta = y
                gc = self._gcbuf(n, x.device)
            else:
                q, k, v, u, w, kg, aqk, y, gc, beta = self._scratch(
                    n, bt, x.dtype, x.device)
                self._pre(x, n, kda_state, meta, cu, ci,
                          (q, k, v, gc, beta), bt, nw)
            _kda_mega_kernel[(nt, h)](
                x, self._w_fb_t, self._w_gb_t, self.o_norm.weight,
                self._cw_all, self._alog, self._dtb,
                kda_state.q_conv_states[li], kda_state.k_conv_states[li],
                kda_state.v_conv_states[li], recurrent_state,
                meta.state_indices, meta.has_initial_state, y,
                gc, q, k, v, beta,
                cu, ci,
                x.stride(0), p, p, p, p,
                kda_state.q_conv_states[li].stride(1),
                kda_state.q_conv_states[li].stride(0),
                recurrent_state.stride(0), p, h, self._scale, self.o_norm.eps,
                P=p, H=h, D=d, BT=bt, W=self.conv_size, SL=self.conv_size - 1,
                SLP=triton.next_power_of_2(self.conv_size - 1),
                NB=nb, BC=bc,
                NLV=bt.bit_length() - 1, PREC="tf32", THRESH=20.0, EPS=1e-6,
                HAS_INIT=False, FUSE_PRE=fuse_pre, num_warps=nw,
            )
            return y

        q, k, v, u, w, kg, aqk, y, gc, beta = self._scratch(n, bt, x.dtype,
                                                            x.device)
        self._pre(x, n, kda_state, meta, cu, ci, (q, k, v, gc, beta), bt, nw)

        has_init = bool(meta.any_have_initial_state)
        _kda_wy_kernel[(nt, h)](
            q, k, v, gc, beta, aqk, u, w, kg,
            cu, ci, p, p, h, h * bt, self._scale,
            H=h, D=d, BT=bt, NB=nb, BC=bc,
            NLV=bt.bit_length() - 1, PREC="tf32", NEED_W=True,
            num_warps=nw,
        )
        # own inter-chunk scan. ``v`` (the conv output) is dead once the WY
        # kernel has consumed it, so it is reused as the ``v_new`` buffer instead
        # of allocating one -- that was FLA's ``torch.empty_like(u)``.
        hst = self._hbuf(nt, x.dtype, x.device)
        _kda_h_kernel[(d // _BV, (cu.shape[0] - 1) * h)](
            kg, u, w, v, gc, hst, recurrent_state, meta.state_indices,
            meta.has_initial_state, cu, self._coff(cu, bt),
            p, p, recurrent_state.stride(0),
            H=h, D=d, BT=bt, BV=_BV, HAS_INIT=has_init,
            num_warps=8, num_stages=3,
        )
        _kda_out_kernel[(nt, h)](
            q, gc, v, hst, aqk, x, self._w_gb_t, self.o_norm.weight, y,
            cu, ci, p, p, p, h * bt, x.stride(0), p, p,
            self._scale, self.o_norm.eps, P=p,
            H=h, D=d, BT=bt, BK=64, num_warps=nw,
        )
        return y

    # ------------------------------------------------------------------ core


    def forward(self, hidden_states: torch.Tensor, state_manager=None) -> torch.Tensor:
        del state_manager
        ctx = get_context()
        kda_state = getattr(ctx, "kda_state", None)
        meta = getattr(ctx, "kda_metadata", None)
        # Everything except an all-prefill batch defers to the baseline: the
        # decode branch (causal_conv1d_update / fused_kda_gate /
        # fused_recurrent_kda) is not what this layer spends its time in, and a
        # hand-rolled copy of it is pure risk. Verified against the baseline in
        # tools/decode_check.py.
        if (self.qkvb_proj is None or kda_state is None or meta is None
                or meta.num_prefills == 0 or meta.num_decode_tokens != 0):
            return super().forward(hidden_states)
        if self._fused_w is None:
            self._build_fused()
        self._ensure_triton_allocator(hidden_states.device)
        x = torch.mm(hidden_states, self._w_in_t,
                     out=self._xbuf(hidden_states.shape[0], hidden_states.dtype,
                                    hidden_states.device))
        y = self._prefill_fused(x, meta.num_actual_tokens, kda_state, meta)
        return torch.mm(y, self._w_out_t)

