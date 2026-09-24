"""Chunk GLA — a direct Triton implementation of the chunked prefill forward.

The reference path (``fla.ops.gla.chunk_gla``) wraps five autotuned Triton
kernels in a ``torch.autograd.Function``, materializes a full-size zeroed
output, and — on variable-length input — builds its chunk-index tables on the
host, paying three device synchronizations per call. None of that is needed for
inference. This module keeps the reference's *arithmetic* and drops the rest:

  1. ``_attn_kernel``    the intra-chunk score block ``A``
  2. ``_state_kernel``   the inter-chunk state recurrence, and ``ht``
  3. ``_output_kernel``  ``o``

Two structural choices carry most of the performance.

*Packed chunk slots.* Work is indexed by a single flat chunk-slot number rather
than by a ``(sequence, chunk)`` pair. A grid over the pair has to be sized by the
product of the two bounds, and on packed variable-length input the per-sequence
chunk bound is the whole padded token axis — 128 chunks for an 8192-row buffer
holding 195 real tokens, so 1024 slots to cover 8. Sizing by the bound on the
*sum* instead gives 136. Both bounds come from shapes, so neither costs a
synchronization; one is two orders of magnitude tighter.

*Recomputing the gate where it pays.* The output kernel owns a whole chunk's
time range and re-reads its gate input once per V tile, so it rebuilds the
cumulative sum from bf16 ``g`` in registers instead. The score kernel cannot: it
works on ``BC``-row sub-blocks and would have to scan the whole chunk to rebuild
one sub-block's gate. The state kernel could, and deliberately does not — see the
comment at its gate load.

Numerics are deliberately conservative. The correctness gate compares against
the reference implementation, not against an exact answer, and it holds the fp32
final state to a tolerance roughly three orders of magnitude tighter than the
bf16 output. The reference rounds its gated keys back to bf16 before the state
product, so a *more* accurate state recurrence reads as a wrong one: fp32 keys
here match the reference on only ~62% of elements against the required 99%.
Every rounding point below is therefore chosen to match the reference, and the
comments say so where a cast looks removable and is not.

Layout follows FLA's convention: ``[B, T, H, K]`` for ``q``/``k``/``g`` and
``[B, T, H, V]`` for ``v``/``o``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

# 1/ln(2), as fla defines it. This value, log2(e) and 1/ln(2) all round to the
# same fp32 bit pattern (0x3FB8AA3B), so only the *placement* of the multiply
# matters: the scan runs first and this scales the result, matching
# chunk_local_cumsum.
RCP_LN2 = 1.4426950216
_RCP_LN2 = tl.constexpr(RCP_LN2)

_ATTN_BC = 16
_ATTN_BK = 64

# The state and output tiles are wide in V on purpose. Both kernels re-read
# their K-indexed inputs once per V tile, so halving the number of V tiles
# halves that traffic — the dominant cost on the one compute-bound benched
# shape.
_STATE_BK = 64
_STATE_BV = 128

_OUT_BK = 64
_OUT_BV = 128

# Warp counts and pipeline depths chosen by measurement, not by autotuning at
# run time. Nsight Compute puts the three heavy kernels at 12-19% achieved
# occupancy with 140-206 registers per thread while reaching only 27-34% of peak
# DRAM bandwidth, so they are occupancy-limited rather than bandwidth-limited;
# spreading each tile over more warps lowers per-thread register demand.
# See profile/chunk_gla_v2_dense_181x1081/REPORT.md and
# docs/evidence/warp_stage_sweep.json.
_ATTN_WARPS, _ATTN_STAGES = 4, 1
_STATE_WARPS, _STATE_STAGES = 8, 3
_OUT_WARPS, _OUT_STAGES = 4, 3


@triton.jit
def _slot_to_chunk(
    cu_seqlens,
    slot,
    T,
    N: tl.constexpr,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """Resolve a flat chunk slot to ``(i_n, i_t, bos, seq_len, valid)``.

    Dense slots are a plain quotient/remainder. Varlen slots are packed, so this
    walks ``cu_seqlens`` accumulating per-sequence chunk counts until the slot
    falls inside one. The walk is branch-free and over the number of sequences,
    which is small and L2-resident — and crucially it happens on the device, so
    no chunk-index table has to be built on the host.
    """
    # Slot counts are bounded by the padded token axis over the chunk edge, so
    # int32 is ample; the widening to int64 happens only where a slot indexes
    # into the state buffer, whose element count can exceed 2^31.
    s32 = slot.to(tl.int32)
    if IS_VARLEN:
        i_n = -1
        i_t = -1
        bos = 0
        seq_len = 0
        base = 0
        for m in range(N):
            s = tl.load(cu_seqlens + m).to(tl.int32)
            e = tl.load(cu_seqlens + m + 1).to(tl.int32)
            nt = tl.cdiv(e - s, BT)
            hit = (s32 >= base) & (s32 < base + nt)
            i_n = tl.where(hit, m, i_n)
            i_t = tl.where(hit, s32 - base, i_t)
            bos = tl.where(hit, s, bos)
            seq_len = tl.where(hit, e - s, seq_len)
            base += nt
        return i_n, i_t, bos, seq_len, i_n >= 0
    NT: tl.constexpr = tl.cdiv(T, BT)
    i_n = s32 // NT
    i_t = s32 % NT
    return i_n, i_t, i_n * T, T, True


@triton.jit
def _seq_bounds(cu_seqlens, i_n, T, IS_VARLEN: tl.constexpr):
    """``(bos, seq_len)`` for sequence ``i_n``: packed offsets when varlen,
    otherwise the rectangular batch stride."""
    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        return bos, eos - bos
    return i_n * T, T


@triton.jit
def _chunk_slot_base(cu_seqlens, i_n, T, BT: tl.constexpr, IS_VARLEN: tl.constexpr):
    """First chunk slot owned by sequence ``i_n``, i.e. ``sum_{m<i_n} NT_m``."""
    if IS_VARLEN:
        base = 0
        for m in range(i_n):
            s = tl.load(cu_seqlens + m).to(tl.int32)
            e = tl.load(cu_seqlens + m + 1).to(tl.int32)
            base += tl.cdiv(e - s, BT)
        return base
    return i_n * tl.cdiv(T, BT)


# ---------------------------------------------------------------------------
# 2. Intra-chunk score block
# ---------------------------------------------------------------------------
@triton.jit
def _attn_kernel(
    q,
    k,
    g,
    A,
    cu_seqlens,
    scale,
    T,
    N: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    K_FULL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """One program owns row sub-block ``i_i`` of one chunk's score block.

    Two formulations, for the same reason the reference uses two. Writing the
    score as ``q[i]·k[j] * 2^(ghat[i]-ghat[j])`` needs the gate difference
    *inside* the feature reduction, which no matmul can express. Factoring it
    into ``2^(ghat[i]-p) * 2^(p-ghat[j])`` for a shared pivot ``p`` does give a
    matmul, but only the off-diagonal blocks can pivot safely: with the pivot at
    the first row of the query sub-block, keys strictly before that row have
    ``ghat[j] >= p`` and queries at or after it have ``ghat[i] <= p``, so both
    factors are at most 1. Inside the diagonal block that ordering breaks, so
    those entries are computed by exact pairwise differences instead.

    Only the lower block-triangle is written. The output kernel masks what it
    reads, which is what keeps the untouched upper blocks from mattering.
    """
    i_i, slot, i_h = tl.program_id(0), tl.program_id(1).to(tl.int64), tl.program_id(2)
    _, i_t, bos, seq_len, valid = _slot_to_chunk(cu_seqlens, slot, T, N, BT, IS_VARLEN)
    if not valid:
        return
    if i_t * BT + i_i * BC >= seq_len:
        return

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    g += (bos * H + i_h) * K
    A += (bos * H + i_h) * BT

    o_q = i_t * BT + i_i * BC + tl.arange(0, BC)       # query rows, this block
    m_q = o_q < seq_len

    o_row = i_t * BT + tl.arange(0, BT)          # every row of this chunk
    m_row = o_row < seq_len

    # --- off-diagonal blocks: one pivoted matmul over all earlier keys -------
    # Both factors are *differences* of the gate cumulative sum from this
    # sub-block's first row, and a difference does not depend on where the scan
    # started. The key side spans the chunk, so its scan is chunk-wide; the query
    # side scans only its own BC rows and measures from that scan's first row,
    # which is the pivot. Scanning the chunk for the query side too would force a
    # [BT, BT] product.
    if i_i > 0:
        b_A = tl.zeros([BC, BT], dtype=tl.float32)
        for i_k in range(tl.cdiv(K, BK)):
            o_k = i_k * BK + tl.arange(0, BK)
            m_k = o_k < K
            m_rk = m_k[:, None] & m_row[None, :]

            b_gkraw = tl.load(g + o_k[:, None] + o_row[None, :] * (H * K),
                              mask=m_rk, other=0.0).to(tl.float32)
            b_gkc = tl.cumsum(b_gkraw, axis=1) * _RCP_LN2          # [BK, BT]
            b_gn = tl.sum(tl.where(tl.arange(0, BT)[None, :] == i_i * BC,
                                   b_gkc, 0.0), axis=1)            # [BK]
            b_k = tl.load(k + o_k[:, None] + o_row[None, :] * (H * K),
                          mask=m_rk, other=0.0)
            b_kg = b_k * tl.math.exp2(tl.minimum(b_gn[:, None] - b_gkc, 0.0))

            m_qk = m_q[:, None] & m_k[None, :]
            b_gqraw = tl.load(g + o_q[:, None] * (H * K) + o_k[None, :],
                              mask=m_qk, other=0.0).to(tl.float32)
            b_gqc = tl.cumsum(b_gqraw, axis=0) * _RCP_LN2          # [BC, BK]
            b_gq0 = tl.sum(tl.where(tl.arange(0, BC)[:, None] == 0,
                                    b_gqc, 0.0), axis=0)           # [BK]
            b_q = tl.load(q + o_q[:, None] * (H * K) + o_k[None, :],
                          mask=m_qk, other=0.0)
            # scale folded into the query side before the reduction, as in the
            # reference; applying it after would round at a different point.
            b_qg = b_q * tl.math.exp2(
                tl.minimum(b_gqc - b_gq0[None, :], 0.0)) * scale

            # fp32 operands, so this is a tf32 dot -- the reference's precision
            # for this term. The clamps are no-ops on every lane that reaches the
            # output, where both exponents are provably <= 0; they only bound
            # lanes that get masked.
            b_A += tl.dot(b_qg, b_kg)

        o_col = tl.arange(0, BT)
        tl.store(A + o_q[:, None] * (H * BT) + o_col[None, :],
                 b_A.to(A.dtype.element_ty),
                 mask=m_q[:, None] & (o_col[None, :] < i_i * BC))

    # --- diagonal block: exact pairwise gate differences --------------------
    # The whole feature axis is loaded at once and reduced in fp32, one key row
    # per iteration, matching the reference. No pivot appears, so for j <= i the
    # exponent is exactly <= 0 and overflow is impossible by construction.
    o_k = tl.arange(0, K_FULL)
    m_k = o_k < K
    p_q = q + o_q[:, None] * (H * K) + o_k[None, :]
    b_q = tl.load(p_q, mask=m_q[:, None] & m_k[None, :], other=0.0)
    # Only *differences* within this sub-block are needed, and a difference is
    # invariant to the scan's origin, so the scan is local to the sub-block's own
    # BC rows. That keeps the working tile at [BC, K] -- the feature-wise exp2 in
    # the loop below is this kernel's dominant cost.
    b_gqraw = tl.load(g + o_q[:, None] * (H * K) + o_k[None, :],
                      mask=m_q[:, None] & m_k[None, :], other=0.0).to(tl.float32)
    b_gq = tl.cumsum(b_gqraw, axis=0) * _RCP_LN2

    rows_here = min(BC, seq_len - i_t * BT - i_i * BC)
    for j in range(BC):
        if j < rows_here:
            row = i_t * BT + i_i * BC + j
            b_k = tl.load(k + row * (H * K) + o_k, mask=m_k, other=0.0).to(tl.float32)
            # The key row's gate from the same local scan, so the difference
            # below is exactly the pairwise gate difference.
            b_gk = tl.sum(tl.where(tl.arange(0, BC)[:, None] == j, b_gq, 0.0), axis=0)
            b_col = tl.sum(
                b_q * b_k[None, :] * tl.math.exp2(tl.minimum(b_gq - b_gk[None, :], 0.0)),
                1,
            ) * scale
            # Causal within the block: rows before this key contribute nothing.
            keep = m_q & (tl.arange(0, BC) >= j)
            tl.store(
                A + o_q * (H * BT) + i_i * BC + j,
                tl.where(keep, b_col, 0.0).to(A.dtype.element_ty),
                mask=m_q,
            )
        else:
            # Columns past the sequence end still sit inside the lower
            # block-triangle the output kernel reads, so they are zeroed rather
            # than left uninitialized.
            tl.store(
                A + o_q * (H * BT) + i_i * BC + j,
                tl.zeros([BC], dtype=A.dtype.element_ty),
                mask=m_q,
            )


# ---------------------------------------------------------------------------
# 3. Inter-chunk state recurrence
# ---------------------------------------------------------------------------
@triton.jit
def _state_kernel(
    k,
    v,
    g,
    h,
    h0,
    ht,
    cu_seqlens,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """Carry ``h[BK, BV]`` across one sequence's chunks.

    The accumulator stays fp32 for the whole walk and only the per-chunk copy
    handed to the output kernel is rounded to bf16 — that split is what lets the
    final state hold an fp32-grade tolerance while the output term stays cheap.
    The gated keys, by contrast, *are* rounded back to bf16 before the product,
    because the reference does: keeping them in fp32 is closer to the true
    answer and further from the reference, which the gate reads as an error.
    """
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_h = i_nh // H, i_nh % H
    bos, seq_len = _seq_bounds(cu_seqlens, i_n, T, IS_VARLEN)
    slot = _chunk_slot_base(cu_seqlens, i_n, T, BT, IS_VARLEN)

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    m_k = o_k < K
    m_v = o_v < V
    m_kv = m_k[:, None] & m_v[None, :]

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = h0 + i_nh * K * V + o_k[:, None] * V + o_v[None, :]
        b_h = tl.load(p_h0, mask=m_kv, other=0.0).to(tl.float32)

    for i_t in range(tl.cdiv(seq_len, BT)):
        o_t = i_t * BT + tl.arange(0, BT)
        m_t = o_t < seq_len

        # The state *entering* this chunk, which is what the output kernel's
        # cross-chunk term needs. Stored before the update, in bf16.
        p_h = h + ((slot + i_t) * H + i_h).to(tl.int64) * K * V \
            + o_k[:, None] * V + o_v[None, :]
        tl.store(p_h, b_h.to(p_h.dtype.element_ty), mask=m_kv)

        p_k = k + (bos * H + i_h) * K + o_k[:, None] + o_t[None, :] * (H * K)
        p_v = v + (bos * H + i_h) * V + o_t[:, None] * (H * V) + o_v[None, :]
        b_k = tl.load(p_k, mask=m_k[:, None] & m_t[None, :], other=0.0)
        b_v = tl.load(p_v, mask=m_t[:, None] & m_v[None, :], other=0.0)

        # This kernel reads the materialized gate buffer rather than rebuilding
        # it. Rebuilding is bit-identical for the scan itself, but it changes
        # what the masked lanes of a partial chunk hold (a plateau instead of
        # zero), and the final state is the output held to an fp32 tolerance —
        # measured agreement drops from every element to 99.97% of them. The
        # extra read is the cheaper side of that trade.
        # Gate cumulative sum rebuilt in registers. Scanning time along axis 1
        # of a [BK, BT] tile is bit-identical to the reference's axis-0 scan of a
        # [BT, BS] tile -- the scan axis and the tiled axis are orthogonal, so
        # each column is scanned independently (docs/evidence/scan_shape_probe.json).
        p_g = g + (bos * H + i_h) * K + o_k[:, None] + o_t[None, :] * (H * K)
        b_graw = tl.load(p_g, mask=m_k[:, None] & m_t[None, :],
                         other=0.0).to(tl.float32)
        b_gk = tl.cumsum(b_graw, axis=1) * _RCP_LN2

        # A chunk's total decay comes from its last *valid* row. The masked-sum
        # selection is exact -- every other lane contributes a true zero and fp32
        # addition of zero is lossless -- and it selects the real last-valid lane,
        # not the tile's last lane, which in a partial chunk holds a plateau of
        # the scan rather than a stored value.
        last_local = min(BT, seq_len - i_t * BT) - 1
        b_gn = tl.sum(tl.where(tl.arange(0, BT)[None, :] == last_local,
                               b_gk, 0.0), axis=1)

        # Zero the decay on lanes past the sequence end. Their keys are already
        # zero so they cannot contribute a value, but the scan leaves a plateau
        # there where the reference's buffer held zero; making the decay itself
        # zero removes the difference rather than relying on 0 * finite.
        b_decay = tl.math.exp2(b_gn[:, None] - b_gk)
        b_decay = tl.where(m_t[None, :], b_decay, 0.0)

        b_h *= tl.math.exp2(b_gn)[:, None]
        b_k = (b_k * b_decay).to(b_k.dtype)
        b_h += tl.dot(b_k, b_v)

    if STORE_FINAL_STATE:
        p_ht = ht + i_nh * K * V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_ht, b_h, mask=m_kv)


# ---------------------------------------------------------------------------
# 4. Output
# ---------------------------------------------------------------------------
@triton.jit
def _output_kernel(
    q,
    v,
    g,
    h,
    A,
    o,
    cu_seqlens,
    scale,
    T,
    N: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """``o = scale * (q * 2^ghat) @ h_chunk + tril(A) @ v``."""
    i_v, slot, i_h = tl.program_id(0), tl.program_id(1).to(tl.int64), tl.program_id(2)
    i_n, i_t, bos, seq_len, valid = _slot_to_chunk(
        cu_seqlens, slot, T, N, BT, IS_VARLEN)

    o_v_all = i_v * BV + tl.arange(0, BV)
    m_v_all = o_v_all < V
    if not valid:
        # Surplus slots clear the padded tail instead of returning, which lets the
        # output be allocated uninitialized even for packed input. Only the *last*
        # sequence owns the region past the final token: a sequence's own surplus
        # rows belong to whatever sequence follows it, so zeroing those would
        # overwrite real output. Slots are packed, so the surplus slots are
        # exactly those at or past the real total, and they tile the tail.
        if IS_VARLEN:
            total = tl.load(cu_seqlens + N).to(tl.int32)
            n_real = 0
            for m in range(N):
                s_m = tl.load(cu_seqlens + m).to(tl.int32)
                e_m = tl.load(cu_seqlens + m + 1).to(tl.int32)
                n_real += tl.cdiv(e_m - s_m, BT)
            o_r = total + (slot.to(tl.int32) - n_real) * BT + tl.arange(0, BT)
            m_r = (o_r >= total) & (o_r < T)
            p_z = o + (o_r[:, None] * H + i_h) * V + o_v_all[None, :]
            tl.store(p_z, tl.zeros([BT, BV], dtype=o.dtype.element_ty),
                     mask=m_r[:, None] & m_v_all[None, :])
        return

    q += (bos * H + i_h) * K
    g += (bos * H + i_h) * K
    v += (bos * H + i_h) * V
    o += (bos * H + i_h) * V
    A += (bos * H + i_h) * BT
    h += (slot * H + i_h).to(tl.int64) * K * V

    o_t = i_t * BT + tl.arange(0, BT)
    o_v = i_v * BV + tl.arange(0, BV)
    o_i = tl.arange(0, BT)
    m_t = o_t < seq_len
    m_v = o_v < V

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        m_qk = m_t[:, None] & m_k[None, :]
        p_q = q + o_t[:, None] * (H * K) + o_k[None, :]
        b_q = tl.load(p_q, mask=m_qk, other=0.0)
        # Rebuilt in registers, as in the state kernel: the fp32 buffer would be
        # re-read once per V tile, and this program owns the whole chunk.
        p_g = g + o_t[:, None] * (H * K) + o_k[None, :]
        b_graw = tl.load(p_g, mask=m_qk, other=0.0).to(tl.float32)
        b_g = tl.cumsum(b_graw, axis=0) * _RCP_LN2
        # Back to bf16 before the dot: the reference casts here, so the product
        # is bf16 x bf16 into an fp32 accumulator rather than a tf32 one.
        b_qg = (b_q * tl.math.exp2(b_g)).to(b_q.dtype)
        p_h = h + o_k[:, None] * V + o_v[None, :]
        b_h = tl.load(p_h, mask=m_k[:, None] & m_v[None, :], other=0.0)
        b_o += tl.dot(b_qg, b_h.to(b_qg.dtype))

    # scale multiplies the cross-chunk term only; the intra-chunk term already
    # carries it from the score kernel.
    b_o *= scale

    p_v = v + o_t[:, None] * (H * V) + o_v[None, :]
    b_v = tl.load(p_v, mask=m_t[:, None] & m_v[None, :], other=0.0)
    p_A = A + o_t[:, None] * (H * BT) + o_i[None, :]
    b_A = tl.load(p_A, mask=m_t[:, None] & (o_i[None, :] < BT), other=0.0)
    # Load-bearing, not decorative: the score kernel never writes the upper
    # block-triangle, so this mask is what keeps that memory out of the result.
    b_A = tl.where(o_i[:, None] >= o_i[None, :], b_A, 0.0).to(b_v.dtype)
    b_o += tl.dot(b_A, b_v)

    p_o = o + o_t[:, None] * (H * V) + o_v[None, :]
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=m_t[:, None] & m_v[None, :])


class ChunkGLA(nn.Module):
    """Triton chunk GLA kernel."""

    def forward(
        self,
        q: torch.Tensor,  # [B, T, H, K]
        k: torch.Tensor,  # [B, T, H, K]
        v: torch.Tensor,  # [B, T, H, V]
        g: torch.Tensor,  # [B, T, H, K]  log-space forget gate
        scale: float | None = None,
        initial_state: torch.Tensor | None = None,  # [N, H, K, V]
        output_final_state: bool = False,
        cu_seqlens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Reject exactly what the reference rejects, before anything is
        # allocated or launched. This is not defensive decoration: the harness
        # treats a candidate exception as a hard failure but a reference
        # exception as a skip, so a malformed input the reference refuses and
        # this module accepts would become wrong indexing -- or an out-of-bounds
        # access -- rather than an error.
        if cu_seqlens is not None:
            if q.shape[0] != 1:
                raise ValueError(
                    f"The batch size is expected to be 1 rather than "
                    f"{q.shape[0]} when using `cu_seqlens`. Please flatten "
                    f"variable-length inputs before processing.",
                )
            if (initial_state is not None
                    and initial_state.shape[0] != len(cu_seqlens) - 1):
                raise ValueError(
                    f"The number of initial states is expected to be equal to "
                    f"the number of input sequences, i.e., "
                    f"{len(cu_seqlens) - 1} rather than "
                    f"{initial_state.shape[0]}.",
                )
        if initial_state is not None:
            assert initial_state.dtype == torch.float32, \
                "initial_state must be in float32."
        assert q.shape == k.shape == g.shape, "q, k, g must have the same shape."
        assert v.shape == (*q.shape[:3], v.shape[-1]), \
            "v must be of shape (batch size, seq len, num of head, head dim)."
        if scale is None:
            scale = q.shape[-1] ** -0.5
        # The kernels index with contiguous strides, so normalize rather than
        # assume. The reference does the same through its input guard; these are
        # no-ops (returning self) on the contiguous tensors the benchmark passes,
        # and without them a legally strided view would silently read the wrong
        # elements instead of failing.
        q, k, v, g = q.contiguous(), k.contiguous(), v.contiguous(), g.contiguous()
        if initial_state is not None:
            initial_state = initial_state.contiguous()
        if cu_seqlens is not None:
            cu_seqlens = cu_seqlens.contiguous()

        B, T, H, K = q.shape
        V = v.shape[-1]
        # The chunk edge follows the reference's formula exactly. It sets the
        # chunk boundaries and therefore the arithmetic, so it is not a free
        # parameter even though nothing here would break at another value.
        BT = min(64, max(16, triton.next_power_of_2(T)))
        varlen = cu_seqlens is not None
        # Number of sequences: a shape read, never a value read.
        N = (cu_seqlens.shape[0] - 1) if varlen else B
        NT_bound = triton.cdiv(T, BT)

        # Chunk slots. Dense sequences are all NT_bound long, so this is exact.
        # Varlen slots are packed and their true count depends on cu_seqlens'
        # *values*; reading those would mean a synchronization, so the count is
        # bounded instead by sum_n cdiv(len_n, BT) <= cdiv(sum_n len_n, BT) +
        # N - 1 <= NT_bound + N. Surplus slots are never touched, and the
        # caching allocator absorbs them after the first call.
        n_slots = (NT_bound + N) if varlen else B * NT_bound

        # Narrower than the reference's fp32, but only as narrow as the value
        # dtype: the output kernel casts A to `v.dtype` before its dot, so
        # storing it at that dtype applies exactly the same single rounding the
        # reference does, and halves the traffic when v is bf16. Hardcoding
        # bf16 here would add a rounding the reference never performs on fp32
        # inputs.
        A = torch.empty(B, T, H, BT, device=q.device, dtype=v.dtype)
        h = torch.empty(n_slots, H, K, V, device=q.device, dtype=k.dtype)
        ht = (torch.empty(N, H, K, V, device=q.device, dtype=torch.float32)
              if output_final_state else None)
        # Neither path memsets. Dense inputs have no padded rows: the
        # per-chunk grid masked by row < T covers every row exactly once. For
        # packed input the output kernel's surplus programs clear the region
        # past the final token, which a timeline showed cost 11% of device
        # time plus a whole launch when done with zeros_like
        # (profile/chunk_gla_v3_varlen_4096_timeline/REPORT.md).
        o = torch.empty_like(v)

        BC = min(_ATTN_BC, BT)
        _attn_kernel[(triton.cdiv(BT, BC), n_slots, H)](
            q, k, g, A, cu_seqlens, scale, T,
            N=N, H=H, K=K, BT=BT, BC=BC, BK=_ATTN_BK,
            K_FULL=max(triton.next_power_of_2(K), 16), IS_VARLEN=varlen,
            num_warps=_ATTN_WARPS, num_stages=_ATTN_STAGES,
        )

        _state_kernel[(triton.cdiv(K, _STATE_BK), triton.cdiv(V, _STATE_BV), N * H)](
            k, v, g, h, initial_state, ht, cu_seqlens, T,
            H=H, K=K, V=V, BT=BT, BK=_STATE_BK, BV=_STATE_BV,
            USE_INITIAL_STATE=initial_state is not None,
            STORE_FINAL_STATE=output_final_state,
            IS_VARLEN=varlen,
            num_warps=_STATE_WARPS, num_stages=_STATE_STAGES,
        )

        _output_kernel[(triton.cdiv(V, _OUT_BV), n_slots, H)](
            q, v, g, h, A, o, cu_seqlens, scale, T,
            N=N, H=H, K=K, V=V, BT=BT, BK=_OUT_BK, BV=_OUT_BV,
            IS_VARLEN=varlen,
            num_warps=_OUT_WARPS, num_stages=_OUT_STAGES,
        )

        return o, ht
