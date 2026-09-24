"""Qwen3-Next Gated Delta Net (GDN) linear attention -- fused prefill path.

The baseline runs the block as thirteen launches: a GEMM for the merged input
projection, a Triton deinterleave, vLLM's causal-conv1d, vLLM's post-conv prep,
a gather of the recurrent state, ``exp(g)``, FlashInfer's chunked
gated-delta-rule, a scatter of the final state, the gated RMSNorm, the output
GEMM, and three dtype copies. At the captured prefill sizes -- 1..69 tokens is
18 600 of the 21 700 captured calls -- almost none of that is work: the block is
112 us of GPU time inside 344 us of wall clock, against a ~30 us floor set by
streaming the 67 MiB of projection weights once. So this path is built to cut
both the kernel count and the per-kernel dispatch cost.

Four or five launches replace the thirteen, for a single-sequence prefill:

``_gdn_fused_proj_prep_kernel`` (sequences of at most 32 tokens)
    One program per 128-column block of the merged projection. It computes its
    own slice of ``x @ W.T`` with tensor cores and then, still in registers, does
    whatever that slice feeds: causal conv1d + SiLU (+ L2 norm for q/k) for the
    8192 conv channels, a plain store for the 4096 ``z`` channels, and
    ``-exp(A_log) * softplus(a + dt_bias)`` / ``sigmoid(b)`` for the 64 ``ba``
    channels. A column block owns *every* token, so the conv's three-token
    history is already in the block -- reached by multiplying the tile with a 0/1
    shift matrix, which is exact and costs three small matmuls. The projection
    itself is never written to memory. Past a 32-token tile Triton lowers that
    GEMM to tcgen05 and the kernel slows by an order of magnitude, so wider
    sequences use a projection GEMM plus:

``_gdn_prep_kernel``
    The same epilogue reading a materialized projection. It still folds the
    deinterleave, the conv, the L2 norm and the gating into one pass over the
    projection, replacing four of the baseline's launches, and never materializes
    the contiguous ``z`` copy -- the norm reads ``z`` out of the projection's own
    columns.

``_gdn_wy_kernel`` (sequences over 48 tokens)
    The chunk-local ``(I + A)^-1`` for every (chunk, v-head) at once. Inside the
    scan this inverse puts ``2*log2(C)`` dependent matmuls on the chunk-to-chunk
    critical path; it depends on nothing carried between chunks, so hoisting it
    into its own launch leaves the scan three dependent matmuls per chunk.

``_gdn_chunk_scan_kernel``
    The recurrence, one program per (v-head, V block), chunked and carrying the
    state in registers, so FLA's per-chunk ``h`` tensor is never written. It
    reads the initial state straight out of the recurrent cache and writes the
    final state back in place, which removes the baseline's ``index_select`` /
    ``index_copy_`` / fp32 round-trip entirely.

``_gdn_norm_gate_kernel`` then the output GEMM
    ``RMSNorm(o) * silu(z)``, reading ``z`` from wherever it already lives.

The unit-lower-triangular inverse comes from
``(I + B)^-1 = prod_j (I + B^(2^j))`` with ``B = -A``, which terminates exactly
at ``j = log2(C)`` because ``A`` is strictly lower triangular: a handful of small
register matmuls instead of FLA's separate 16x16-solve and merge launches.

Long sequences (over 1280 tokens), decode, and varlen batches this path does not
model fall through to the baseline; at 16k tokens the chunk-serial scan's
dependency chain costs more than the baseline's whole pipeline.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ....infra.context import get_context as _get_context
from ...baseline.L2.qwen3_next_gdn_attention import (
    Qwen3NextGDNAttention as _BaselineGDN,
)

__targets__ = ["Qwen3NextGDNAttention"]


# ---------------------------------------------------------------------------
# Column-block layout of the merged projection
# ---------------------------------------------------------------------------
# ``in_proj_qkvz`` emits one group of ``GS = 2K + 2*VP*V`` columns per K head,
# laid out [q(K) k(K) v(VP*V) z(VP*V)]; ``in_proj_ba``'s rows, stacked
# underneath by ``process_weights_after_loading``, are grouped per K head too,
# as [b(VP) a(VP)]. The conv wants [q_all | k_all | v_all] packed and the gate
# wants z as [T, HV, V], so both kernels below address the projection in its
# native grouped layout and the permutation becomes register naming.
#
#   cid < H                  -> q of head cid
#   H     <= cid < 2H        -> k of head cid-H
#   2H    <= cid < 2H+HV     -> v of v-head cid-2H
#   2H+HV <= cid < 2H+2HV    -> z of v-head cid-2H-HV
#   cid == 2H+2HV            -> the ba block (gating)


@triton.jit
def _col_map(cid, H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr,
             V: tl.constexpr, VP: tl.constexpr, GS: tl.constexpr):
    """(projection column base, conv-channel base, kind) for column block cid.

    kind: 0 = q, 1 = k, 2 = v, 3 = z.
    """
    is_q = cid < H
    is_k = (cid >= H) & (cid < 2 * H)
    is_v = (cid >= 2 * H) & (cid < 2 * H + HV)
    hqk = tl.where(is_q, cid, cid - H)
    hv = tl.where(is_v, cid - 2 * H, cid - 2 * H - HV)
    grp = (hv // VP) * GS + (hv % VP) * V
    pcol = tl.where(
        is_q | is_k,
        hqk * GS + tl.where(is_k, K, 0),
        grp + tl.where(is_v, 2 * K, 2 * K + VP * V),
    )
    cch = tl.where(is_q | is_k, hqk * K + tl.where(is_k, H * K, 0),
                   2 * H * K + hv * V)
    kind = tl.where(is_q, 0, tl.where(is_k, 1, tl.where(is_v, 2, 3)))
    return pcol, cch, kind


@triton.jit
def _gate_from_ba(bb, aa, A_log_ptr, dt_ptr, ohv, mh):
    """(g, beta) from the raw ba columns -- vLLM's prep, in one expression."""
    al = tl.load(A_log_ptr + ohv, mask=mh, other=0.0).to(tl.float32)
    dtb = tl.load(dt_ptr + ohv, mask=mh, other=0.0).to(tl.float32)
    x = aa + dtb[None, :]
    sp = tl.where(x > 0, x + tl.log(1.0 + tl.exp(-x)), tl.log(1.0 + tl.exp(x)))
    sp = tl.where(x <= 20.0, sp, x)
    return -tl.exp(al)[None, :] * sp, tl.sigmoid(bb)


@triton.jit
def _l2_or_pass(y, kind, dt: tl.constexpr):
    """L2-normalise a q/k tile over its 128 channels; pass a v tile through."""
    if kind != 2:
        yf = y.to(tl.float32)
        ss = tl.sum(yf * yf, 1)
        return (yf * (1.0 / tl.sqrt(ss + 1e-6))[:, None]).to(dt)
    return y


# ---------------------------------------------------------------------------
# Fused projection + conv + gating (sequence fits one token block)
# ---------------------------------------------------------------------------
@triton.jit
def _gdn_fused_proj_prep_kernel(
    x_ptr, w_ptr, convw_ptr, A_log_ptr, dt_ptr,
    qkv_ptr, z_ptr, g_ptr, beta_ptr, cstate_ptr,
    cu_ptr, sidx_ptr,
    s_x, s_w, s_qkv, s_z, s_g,
    s_cs_seq, s_cs_dim, s_cs_tok,
    D: tl.constexpr, H: tl.constexpr, HV: tl.constexpr,
    K: tl.constexpr, V: tl.constexpr, VP: tl.constexpr,
    GS: tl.constexpr, QKVZ: tl.constexpr, HVP: tl.constexpr,
    WIDTH: tl.constexpr, BT: tl.constexpr, BD: tl.constexpr,
    BKK: tl.constexpr, USE_INIT: tl.constexpr, SAVE_STATE: tl.constexpr,
):
    cid = tl.program_id(0)
    DT = qkv_ptr.dtype.element_ty
    bos = tl.load(cu_ptr + 0).to(tl.int32)
    Tn = tl.load(cu_ptr + 1).to(tl.int32) - bos
    sid = tl.load(sidx_ptr + 0).to(tl.int64)

    ot = tl.arange(0, BT)
    mt = ot < Tn
    NBA: tl.constexpr = 2 * H + 2 * HV

    if cid == NBA:
        # ba block: two narrow GEMMs straight into the b / a column sets.
        ohv = tl.arange(0, HVP)
        mh = ohv < HV
        bcol = QKVZ + (ohv // VP) * (2 * VP) + (ohv % VP)
        accb = tl.zeros([BT, HVP], dtype=tl.float32)
        acca = tl.zeros([BT, HVP], dtype=tl.float32)
        for k0 in range(0, D, BKK):
            ok = k0 + tl.arange(0, BKK)
            mk = ok < D
            bx = tl.load(x_ptr + (bos + ot)[:, None] * s_x + ok[None, :],
                         mask=mt[:, None] & mk[None, :], other=0.0)
            # W is [out, in] row-major, so the [BD, BKK] orientation is the
            # coalesced one; tl.dot takes the transpose from there.
            wb = tl.load(w_ptr + bcol[:, None] * s_w + ok[None, :],
                         mask=mh[:, None] & mk[None, :], other=0.0)
            wa = tl.load(w_ptr + (bcol + VP)[:, None] * s_w + ok[None, :],
                         mask=mh[:, None] & mk[None, :], other=0.0)
            accb += tl.dot(bx, tl.trans(wb))
            acca += tl.dot(bx, tl.trans(wa))
        gv, bv = _gate_from_ba(accb.to(DT).to(tl.float32),
                               acca.to(DT).to(tl.float32),
                               A_log_ptr, dt_ptr, ohv, mh)
        po = (bos + ot)[:, None] * s_g + ohv[None, :]
        tl.store(g_ptr + po, gv, mask=mt[:, None] & mh[None, :])
        tl.store(beta_ptr + po, bv, mask=mt[:, None] & mh[None, :])
        return

    od = tl.arange(0, BD)
    pcol, cch, kind = _col_map(cid, H, HV, K, V, VP, GS)

    acc = tl.zeros([BT, BD], dtype=tl.float32)
    for k0 in range(0, D, BKK):
        ok = k0 + tl.arange(0, BKK)
        mk = ok < D
        bx = tl.load(x_ptr + (bos + ot)[:, None] * s_x + ok[None, :],
                     mask=mt[:, None] & mk[None, :], other=0.0)
        bw = tl.load(w_ptr + (pcol + od)[:, None] * s_w + ok[None, :],
                     mask=mk[None, :], other=0.0)
        acc += tl.dot(bx, tl.trans(bw))
    cur = acc.to(DT)

    if kind == 3:
        tl.store(z_ptr + (bos + ot)[:, None] * s_z + (cch - 2 * H * K + od)[None, :],
                 cur, mask=mt[:, None])
        return

    # Causal conv1d. Tap i needs the tile shifted down by WIDTH-1-i rows; a 0/1
    # shift matrix through tl.dot reproduces those rows exactly (one non-zero
    # product per output element) without touching memory.
    out = tl.zeros([BT, BD], dtype=tl.float32)
    for i in tl.static_range(WIDTH):
        if i == WIDTH - 1:
            x = cur
        else:
            r = ot - (WIDTH - 1 - i)
            shm = (r[:, None] == ot[None, :]).to(DT)
            x = tl.dot(shm, cur).to(DT)
            if USE_INIT:
                xs = tl.load(
                    cstate_ptr + sid * s_cs_seq + (cch + od)[None, :] * s_cs_dim
                    + (r + WIDTH - 1)[:, None] * s_cs_tok,
                    mask=(r < 0)[:, None], other=0.0,
                ).to(DT)
                x = tl.where((r >= 0)[:, None], x, xs)
        w = tl.load(convw_ptr + (cch + od) * WIDTH + i)
        out += (x * w[None, :]).to(tl.float32)
    out = out / (1.0 + tl.exp(-out))
    y = _l2_or_pass(out.to(DT), kind, DT)
    tl.store(qkv_ptr + (bos + ot)[:, None] * s_qkv + (cch + od)[None, :],
             y, mask=mt[:, None])

    if SAVE_STATE:
        # New history = the last WIDTH-1 pre-conv tokens, selected out of the
        # tile the same way; short sequences keep the tail of the old state.
        oi = tl.arange(0, 16)
        r = Tn - (WIDTH - 1) + oi
        sel = (r[:, None] == ot[None, :]).to(DT)
        new = tl.dot(sel, cur).to(DT)
        old = tl.load(
            cstate_ptr + sid * s_cs_seq + (cch + od)[None, :] * s_cs_dim
            + (Tn + oi)[:, None] * s_cs_tok,
            mask=(r < 0)[:, None] & (oi < WIDTH - 1)[:, None], other=0.0,
        ).to(DT)
        val = tl.where((r >= 0)[:, None], new, old)
        tl.store(
            cstate_ptr + sid * s_cs_seq + (cch + od)[None, :] * s_cs_dim
            + oi[:, None] * s_cs_tok, val, mask=(oi < WIDTH - 1)[:, None],
        )


# ---------------------------------------------------------------------------
# Conv + gating from a materialized projection (long sequences)
# ---------------------------------------------------------------------------
@triton.jit
def _gdn_prep_kernel(
    proj_ptr, convw_ptr, A_log_ptr, dt_ptr,
    qkv_ptr, g_ptr, beta_ptr, cstate_ptr,
    cu_ptr, sidx_ptr,
    s_proj, s_qkv, s_g,
    s_cs_seq, s_cs_dim, s_cs_tok,
    H: tl.constexpr, HV: tl.constexpr,
    K: tl.constexpr, V: tl.constexpr, VP: tl.constexpr,
    GS: tl.constexpr, QKVZ: tl.constexpr, HVP: tl.constexpr,
    WIDTH: tl.constexpr, BT: tl.constexpr, BD: tl.constexpr,
    USE_INIT: tl.constexpr, SAVE_STATE: tl.constexpr,
):
    pid_t = tl.program_id(0)
    cid = tl.program_id(1)
    DT = qkv_ptr.dtype.element_ty
    bos = tl.load(cu_ptr + 0).to(tl.int32)
    Tn = tl.load(cu_ptr + 1).to(tl.int32) - bos
    t0 = pid_t * BT
    if t0 >= Tn:
        return
    sid = tl.load(sidx_ptr + 0).to(tl.int64)

    ot = t0 + tl.arange(0, BT)
    mt = ot < Tn
    NCONV: tl.constexpr = 2 * H + HV

    if cid == NCONV:
        ohv = tl.arange(0, HVP)
        mh = ohv < HV
        bcol = QKVZ + (ohv // VP) * (2 * VP) + (ohv % VP)
        pb = proj_ptr + (bos + ot)[:, None] * s_proj
        m2 = mt[:, None] & mh[None, :]
        bb = tl.load(pb + bcol[None, :], mask=m2, other=0.0).to(tl.float32)
        aa = tl.load(pb + (bcol + VP)[None, :], mask=m2, other=0.0).to(tl.float32)
        gv, bv = _gate_from_ba(bb, aa, A_log_ptr, dt_ptr, ohv, mh)
        po = (bos + ot)[:, None] * s_g + ohv[None, :]
        tl.store(g_ptr + po, gv, mask=m2)
        tl.store(beta_ptr + po, bv, mask=m2)
        return

    od = tl.arange(0, BD)
    pcol, cch, kind = _col_map(cid, H, HV, K, V, VP, GS)

    out = tl.zeros([BT, BD], dtype=tl.float32)
    for i in tl.static_range(WIDTH):
        r = ot - (WIDTH - 1) + i
        x = tl.load(proj_ptr + (bos + r)[:, None] * s_proj + (pcol + od)[None, :],
                    mask=mt[:, None] & (r >= 0)[:, None], other=0.0)
        if USE_INIT:
            xs = tl.load(
                cstate_ptr + sid * s_cs_seq + (cch + od)[None, :] * s_cs_dim
                + (r + WIDTH - 1)[:, None] * s_cs_tok,
                mask=mt[:, None] & (r < 0)[:, None], other=0.0,
            )
            x = tl.where((r >= 0)[:, None], x, xs)
        w = tl.load(convw_ptr + (cch + od) * WIDTH + i)
        out += (x * w[None, :]).to(tl.float32)
    out = out / (1.0 + tl.exp(-out))
    y = _l2_or_pass(out.to(DT), kind, DT)
    tl.store(qkv_ptr + (bos + ot)[:, None] * s_qkv + (cch + od)[None, :],
             y, mask=mt[:, None])

    if SAVE_STATE:
        if Tn <= t0 + BT:
            oi = tl.arange(0, 4)
            mi = oi < WIDTH - 1
            rs = Tn - (WIDTH - 1) + oi
            newv = tl.load(
                proj_ptr + (bos + rs)[:, None] * s_proj + (pcol + od)[None, :],
                mask=mi[:, None] & (rs >= 0)[:, None], other=0.0,
            )
            oldv = tl.load(
                cstate_ptr + sid * s_cs_seq + (cch + od)[None, :] * s_cs_dim
                + (Tn + oi)[:, None] * s_cs_tok,
                mask=mi[:, None] & (rs < 0)[:, None], other=0.0,
            )
            val = tl.where((rs >= 0)[:, None], newv, oldv)
            tl.debug_barrier()
            tl.store(
                cstate_ptr + sid * s_cs_seq + (cch + od)[None, :] * s_cs_dim
                + oi[:, None] * s_cs_tok, val, mask=mi[:, None],
            )


# ---------------------------------------------------------------------------
# Chunked gated-delta-rule scan
# ---------------------------------------------------------------------------
@triton.jit
def _unit_tri_inv(A, LOG: tl.constexpr):
    """``(I + A)^-1`` for strictly lower triangular ``A`` [C, C].

    ``B = -A`` is nilpotent with ``B**C == 0``, so
    ``sum_{i<C} B**i == prod_{j<log2(C)} (I + B**(2**j))`` holds exactly:
    log2(C) squarings and as many products, all in registers, replacing FLA's
    separate 16x16-solve and merge launches.
    """
    C: tl.constexpr = A.shape[0]
    o = tl.arange(0, C)
    B = -A
    P = tl.where(o[:, None] == o[None, :], 1.0, 0.0) + B
    Bp = B
    for _ in tl.static_range(LOG - 1):
        Bp = tl.dot(Bp, Bp)
        P = P + tl.dot(P, Bp)
    return P


@triton.jit
def _gdn_wy_kernel(
    qkv_ptr, g_ptr, beta_ptr, ai_ptr, cu_ptr,
    s_qkv, s_g,
    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, VP: tl.constexpr,
    C: tl.constexpr, LOG: tl.constexpr,
):
    """``(I + A)^-1`` for every (chunk, v-head), in parallel.

    Building it inside the scan puts ``2*log2(C)`` dependent matmuls on the
    chunk-to-chunk critical path, which is what made the scan cost ~0.25 us per
    token. It depends on nothing carried between chunks, so hoisting it into its
    own launch -- one program per (chunk, v-head), thousands of them -- leaves
    the scan with three dependent matmuls per chunk instead of eleven.
    """
    i_t = tl.program_id(0)
    i_hv = tl.program_id(1)
    i_h = i_hv // VP
    DT = qkv_ptr.dtype.element_ty
    bos = tl.load(cu_ptr + 0).to(tl.int32)
    Tn = tl.load(cu_ptr + 1).to(tl.int32) - bos
    t0 = i_t * C
    if t0 >= Tn:
        return
    oc = tl.arange(0, C)
    ok = tl.arange(0, K)
    ot = t0 + oc
    mt = ot < Tn
    k_base = qkv_ptr + bos * s_qkv + H * K + i_h * K
    bk = tl.load(k_base + ot[:, None] * s_qkv + ok[None, :],
                 mask=mt[:, None], other=0.0)
    gr = tl.load(g_ptr + (bos + ot) * s_g + i_hv, mask=mt, other=0.0)
    bb = tl.load(beta_ptr + (bos + ot) * s_g + i_hv, mask=mt, other=0.0)
    gc = tl.cumsum(gr, axis=0)
    A = tl.dot(bk * bb[:, None].to(DT), tl.trans(bk))
    A = A * tl.exp(gc[:, None] - gc[None, :])
    A = tl.where((oc[:, None] > oc[None, :]) & mt[:, None] & mt[None, :], A, 0.0)
    Ai = _unit_tri_inv(A, LOG)
    tl.store(ai_ptr + (i_t * HV + i_hv) * (C * C) + oc[:, None] * C + oc[None, :],
             Ai.to(ai_ptr.dtype.element_ty))


@triton.jit
def _gdn_chunk_scan_kernel(
    qkv_ptr, g_ptr, beta_ptr, o_ptr, state_ptr, ai_ptr,
    cu_ptr, sidx_ptr,
    s_qkv, s_g, s_o, scale,
    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    VP: tl.constexpr, C: tl.constexpr, LOG: tl.constexpr, BV: tl.constexpr,
    USE_INIT: tl.constexpr, STATE_STRIDE: tl.constexpr, AI_IN: tl.constexpr,
):
    """One program per (v-head, V block); chunk-serial, state in registers.

    The state never leaves registers between chunks, so FLA's per-chunk ``h``
    tensor (537 MiB at 16 k tokens) is never written, and the initial/final state
    is read and written straight in the recurrent cache instead of through the
    baseline's gather / fp32 copy / scatter.

    ``v_new`` is formed as ``Ai @ (beta*v - (beta*k*e^g) @ S.T)`` rather than
    FLA's ``u - w @ S.T``: reassociating drops a [C, K]-wide accumulator -- the
    Blackwell tensor-memory budget is 512 columns and only fits so many -- and is
    fewer FLOPs besides.
    """
    pid = tl.program_id(0)
    i_hv = pid // (V // BV)
    i_v = pid % (V // BV)
    i_h = i_hv // VP
    DT = qkv_ptr.dtype.element_ty
    bos = tl.load(cu_ptr + 0).to(tl.int32)
    Tn = tl.load(cu_ptr + 1).to(tl.int32) - bos
    sid = tl.load(sidx_ptr + 0).to(tl.int64)

    ok = tl.arange(0, K)
    ov = i_v * BV + tl.arange(0, BV)
    oc = tl.arange(0, C)

    q_base = qkv_ptr + bos * s_qkv + i_h * K
    k_base = q_base + H * K
    v_base = qkv_ptr + bos * s_qkv + 2 * H * K + i_hv * V
    g_base = g_ptr + bos * s_g + i_hv
    b_base = beta_ptr + bos * s_g + i_hv
    o_base = o_ptr + bos * s_o + i_hv * V
    sp = (state_ptr + sid * STATE_STRIDE + i_hv * V * K
          + ov[:, None] * K + ok[None, :])

    S = tl.zeros([BV, K], dtype=tl.float32)
    if USE_INIT:
        S += tl.load(sp).to(tl.float32)

    for t0 in range(0, Tn, C):
        ot = t0 + oc
        mt = ot < Tn
        bq = tl.load(q_base + ot[:, None] * s_qkv + ok[None, :],
                     mask=mt[:, None], other=0.0)
        bk = tl.load(k_base + ot[:, None] * s_qkv + ok[None, :],
                     mask=mt[:, None], other=0.0)
        bv = tl.load(v_base + ot[:, None] * s_qkv + ov[None, :],
                     mask=mt[:, None], other=0.0)
        gr = tl.load(g_base + ot * s_g, mask=mt, other=0.0)
        bb = tl.load(b_base + ot * s_g, mask=mt, other=0.0)
        gc = tl.cumsum(gr, axis=0)
        last = tl.minimum(t0 + C, Tn) - 1
        g_last = tl.sum(tl.where(ot == last, gc, 0.0))
        egc = tl.exp(gc)
        mm = mt[:, None] & mt[None, :]
        decay = tl.exp(gc[:, None] - gc[None, :])
        St = tl.trans(S).to(DT)

        # WY representation of the chunk's delta-rule updates.
        if AI_IN:
            Ai = tl.load(ai_ptr + ((t0 // C) * HV + i_hv) * (C * C)
                         + oc[:, None] * C + oc[None, :])
        else:
            A = tl.dot(bk * bb[:, None].to(DT), tl.trans(bk)) * decay
            Ai = _unit_tri_inv(tl.where((oc[:, None] > oc[None, :]) & mm, A, 0.0),
                               LOG).to(DT)
        rhs = (bv * bb[:, None].to(DT)).to(tl.float32) - tl.dot(
            (bk * (bb * egc)[:, None].to(DT)).to(DT), St)
        vn = tl.dot(Ai, rhs.to(DT))

        # Output: carried state plus in-chunk attention.
        Aq = tl.dot(bq, tl.trans(bk)) * decay
        Aq = tl.where((oc[:, None] >= oc[None, :]) & mm, Aq, 0.0)
        o = tl.dot(bq, St) * egc[:, None] + tl.dot(Aq.to(DT), vn.to(DT))
        tl.store(o_base + ot[:, None] * s_o + ov[None, :], (o * scale).to(DT),
                 mask=mt[:, None])

        # Carry the state over the chunk boundary.
        vd = (vn * tl.where(mt, tl.exp(g_last - gc), 0.0)[:, None]).to(DT)
        S = S * tl.exp(g_last) + tl.trans(tl.dot(tl.trans(bk), vd))

    tl.store(sp, S.to(state_ptr.dtype.element_ty))


# ---------------------------------------------------------------------------
# Gated RMSNorm folded into the output projection
# ---------------------------------------------------------------------------
@triton.jit
def _gdn_norm_gate_kernel(
    o_ptr, z_ptr, y_ptr, nw_ptr, cu_ptr,
    s_o, s_z, s_y, eps,
    HV: tl.constexpr, V: tl.constexpr, VP: tl.constexpr,
    Z_GS: tl.constexpr, Z_OFF: tl.constexpr, BT: tl.constexpr,
):
    """``RMSNorm(o) * silu(z)`` for one (token block, v-head).

    Reads ``z`` straight out of whichever buffer holds it -- the standalone z
    the fused projection writes, or the materialized projection's z columns --
    so the baseline's contiguous ``[T, HV, V]`` z copy is never made.
    """
    pid_t = tl.program_id(0)
    i_hv = tl.program_id(1)
    bos = tl.load(cu_ptr + 0).to(tl.int32)
    Tn = tl.load(cu_ptr + 1).to(tl.int32) - bos
    ot = pid_t * BT + tl.arange(0, BT)
    mt = ot < Tn
    ov = tl.arange(0, V)
    oo = tl.load(o_ptr + (bos + ot)[:, None] * s_o + (i_hv * V + ov)[None, :],
                 mask=mt[:, None], other=0.0).to(tl.float32)
    rstd = 1.0 / tl.sqrt(tl.sum(oo * oo, 1) / V + eps)
    nw = tl.load(nw_ptr + ov).to(tl.float32)
    y = oo * rstd[:, None] * nw[None, :]
    zcol = (i_hv // VP) * Z_GS + Z_OFF + (i_hv % VP) * V
    zz = tl.load(z_ptr + (bos + ot)[:, None] * s_z + (zcol + ov)[None, :],
                 mask=mt[:, None], other=0.0).to(tl.float32)
    y = y * (zz * tl.sigmoid(zz))
    tl.store(y_ptr + (bos + ot)[:, None] * s_y + (i_hv * V + ov)[None, :],
             y.to(y_ptr.dtype.element_ty), mask=mt[:, None])


# ---------------------------------------------------------------------------
# Launch plan
# ---------------------------------------------------------------------------
# At the captured prefill sizes this block is dispatch-bound, not work-bound: the
# baseline's thirteen kernels are 117 us of GPU time inside 344 us of wall clock.
# Four launches instead of thirteen recovers most of that, but Triton's per-call
# argument binding (signature key, specialization, bound-args dict, a fresh
# closure and metadata object per launch) is then itself a double-digit fraction
# of what is left. So the first call through a given token count builds a
# ``_Prog``: the compiled kernels, their grids, and their full argument lists,
# with the projection/recurrence scratch buffers allocated once. Later calls
# overwrite the one slot that changes -- the activation pointer -- and go
# straight to each kernel's launcher.
#
# Reuse is only sound while the specialization those kernels were compiled under
# still holds, so ``_Prog.matches`` re-checks the activation's stride and
# 16-byte alignment and the identity of every engine-owned buffer, and anything
# unexpected rebuilds the plan (or falls back to the baseline).

try:  # Triton's launch hooks; the raw launcher path is only safe with none set.
    from triton import knobs as _tl_knobs
except ImportError:  # pragma: no cover - older Triton
    _tl_knobs = None


def _hooks_clear() -> bool:
    if _tl_knobs is None:
        return False
    try:
        return not (_tl_knobs.runtime.launch_enter_hook.calls
                    or _tl_knobs.runtime.launch_exit_hook.calls)
    except AttributeError:  # pragma: no cover - unexpected knobs layout
        return False


class _Step:
    """One compiled kernel plus the grid and argument list it launches with.

    ``launch`` is what ``JITFunction.run`` does after it has finished binding:
    call the generated launcher with the grid, the stream and the same argument
    list. ``launch_safe`` goes through ``CompiledKernel.__getitem__`` instead,
    for when the launcher protocol is not the one assumed here.
    """

    __slots__ = ("ck", "grid", "args", "run", "fn", "meta", "ok")

    def __init__(self, ck, grid, args):
        self.ck = ck
        self.grid = grid
        self.args = args
        self.run = ck.run          # binds, and realises the launcher
        self.fn = getattr(ck, "function", None)
        self.meta = getattr(ck, "packed_metadata", None)
        self.ok = self.fn is not None and self.meta is not None

    def launch(self, stream):
        self.run(self.grid[0], self.grid[1], self.grid[2], stream, self.fn,
                 self.meta, None, None, None, *self.args)

    def launch_safe(self):
        self.ck[self.grid](*self.args)


class _Prog:
    """Everything one token count needs, built once and replayed."""

    __slots__ = ("steps", "x_slots", "sx", "yn", "woT", "y_dim", "dtype",
                 "device", "n", "use_init", "owners", "mm_in", "raw")

    def __init__(self, n, use_init, sx, owners):
        self.n = n
        self.use_init = use_init
        self.sx = sx
        self.owners = owners
        self.steps = []
        self.x_slots = []
        self.mm_in = None
        self.raw = True

    def matches(self, x, use_init, owners):
        own = self.owners
        return (use_init == self.use_init and x.stride(0) == self.sx
                and x.stride(1) == 1 and x.data_ptr() % 16 == 0
                and owners[0] is own[0] and owners[1] is own[1]
                and owners[2] is own[2] and owners[3] is own[3])

    def seal(self):
        self.raw = all(step.ok for step in self.steps)

    def launch(self, x):
        for step, slot in zip(self.steps, self.x_slots):
            if slot >= 0:
                step.args[slot] = x
        if self.mm_in is not None:
            torch.mm(x, self.mm_in[0], out=self.mm_in[1])
        if self.raw and _hooks_clear():
            stream = torch.cuda.current_stream(self.device).cuda_stream
            for step in self.steps:
                step.launch(stream)
        else:
            for step in self.steps:
                step.launch_safe()


class Qwen3NextGDNAttention(_BaselineGDN):
    """Gated Delta Net linear attention for Qwen3-Next (fused prefill)."""

    # A fused projection program holds one column block for every token at once,
    # so its GEMM tile is [T, 128]. Past a 32-token tile Triton lowers that dot
    # to tcgen05 and the kernel slows by an order of magnitude, so wider
    # sequences use a projection GEMM plus the conv/gating pass.
    _FUSE_MAX_T = 32
    # Past this the chunk-serial scan's dependency chain costs more than the
    # baseline's whole pipeline; those sequences take the baseline path.
    _MAX_T = 1280
    # Past this the WY inverse is hoisted into its own parallel launch.
    _WY_MIN_T = 48
    _CHUNK = 16
    _CHUNK_WY = 64
    _BV = 32
    _PREP_BT = 32
    _NORM_BT = 16

    def _fast_ok(self, md, state_manager) -> bool:
        op = self.out_proj
        return (
            self._in_proj_w is not None
            and md.num_prefills > 0
            and md.num_decodes == 0
            and self.head_k_dim == self.head_v_dim
            and self.conv_kernel_size == 4
            and self.head_v_dim % self._BV == 0
            and not op.use_fp8
            and op.bias is None
            and not (op.reduce_results and op.tp_size > 1)
            and state_manager.recurrent[self.layer_idx] is not None
        )

    def _build_prog(self, N, x, cu, sidx, conv_state, recurrent, use_init):
        H = self.local_k_heads
        HV = self.local_v_heads
        K = self.head_k_dim
        V = self.head_v_dim
        VP = self.v_per_k
        GS = 2 * K + 2 * VP * V
        HVP = triton.next_power_of_2(HV)
        WID = self.conv_kernel_size
        dev, dt = x.device, x.dtype
        cs0, cs1, cs2 = conv_state.stride()
        w = self._in_proj_w

        fused = N <= self._FUSE_MAX_T
        bt = max(16, triton.next_power_of_2(N)) if fused else self._PREP_BT
        wy = N > self._WY_MIN_T
        C = self._CHUNK_WY if wy else self._CHUNK
        NT = (N + C - 1) // C

        qkv = torch.empty(N, 2 * H * K + HV * V, device=dev, dtype=dt)
        g = torch.empty(N, HV, device=dev, dtype=torch.float32)
        beta = torch.empty(N, HV, device=dev, dtype=torch.float32)
        o = torch.empty(N, HV * V, device=dev, dtype=dt)
        yn = torch.empty(N, HV * V, device=dev, dtype=dt)
        ai = (torch.empty(NT * HV * C * C, device=dev, dtype=dt) if wy
              else qkv)

        prog = _Prog(N, use_init, x.stride(0),
                     (cu, sidx, conv_state, recurrent))
        prog.device = dev
        prog.dtype = dt
        prog.yn = yn
        prog.woT = self.out_proj.weight.t()
        prog.y_dim = self.out_proj.weight.shape[0]

        if fused:
            z = torch.empty(N, HV * V, device=dev, dtype=dt)
            args = [x, w, self.conv1d.weight, self.A_log, self.dt_bias,
                    qkv, z, g, beta, conv_state, cu, sidx,
                    x.stride(0), w.stride(0), qkv.stride(0), z.stride(0),
                    g.stride(0), cs0, cs1, cs2,
                    self.hidden_size, H, HV, K, V, VP, GS, self._qkvz_dim,
                    HVP, WID, bt, K, 128, use_init, True]
            ck = _gdn_fused_proj_prep_kernel[(2 * H + 2 * HV + 1, 1, 1)](
                *args, num_warps=4, num_stages=3)
            prog.steps.append(_Step(ck, (2 * H + 2 * HV + 1, 1, 1), args))
            prog.x_slots.append(0)
            z_ptr, s_z, z_gs, z_off = z, z.stride(0), VP * V, 0
        else:
            proj = torch.empty(N, w.shape[0], device=dev, dtype=dt)
            prog.mm_in = (w.t(), proj)
            torch.mm(x, prog.mm_in[0], out=proj)
            args = [proj, self.conv1d.weight, self.A_log, self.dt_bias,
                    qkv, g, beta, conv_state, cu, sidx,
                    proj.stride(0), qkv.stride(0), g.stride(0), cs0, cs1, cs2,
                    H, HV, K, V, VP, GS, self._qkvz_dim, HVP, WID, bt, K,
                    use_init, True]
            grid = ((N + bt - 1) // bt, 2 * H + HV + 1, 1)
            ck = _gdn_prep_kernel[grid](*args, num_warps=4, num_stages=2)
            prog.steps.append(_Step(ck, grid, args))
            prog.x_slots.append(-1)
            z_ptr, s_z = proj, proj.stride(0)
            z_gs, z_off = GS, 2 * K + VP * V

        if wy:
            args = [qkv, g, beta, ai, cu, qkv.stride(0), g.stride(0),
                    H, HV, K, VP, C, C.bit_length() - 1]
            ck = _gdn_wy_kernel[(NT, HV, 1)](*args, num_warps=4, num_stages=2)
            prog.steps.append(_Step(ck, (NT, HV, 1), args))
            prog.x_slots.append(-1)

        args = [qkv, g, beta, o, recurrent, ai, cu, sidx,
                qkv.stride(0), g.stride(0), o.stride(0), K ** -0.5,
                H, HV, K, V, VP, C, C.bit_length() - 1, self._BV,
                use_init, HV * V * K, wy]
        grid = (HV * (V // self._BV), 1, 1)
        ck = _gdn_chunk_scan_kernel[grid](*args, num_warps=4, num_stages=1)
        prog.steps.append(_Step(ck, grid, args))
        prog.x_slots.append(-1)

        nbt = self._NORM_BT
        args = [o, z_ptr, yn, self.norm.weight, cu,
                o.stride(0), s_z, yn.stride(0), self.norm.eps,
                HV, V, VP, z_gs, z_off, nbt]
        grid = ((N + nbt - 1) // nbt, HV, 1)
        ck = _gdn_norm_gate_kernel[grid](*args, num_warps=4, num_stages=2)
        prog.steps.append(_Step(ck, grid, args))
        prog.x_slots.append(-1)
        prog.seal()
        return prog

    def forward_impl(self, hidden_states, state_manager=None):
        ctx = _get_context()
        md = ctx.kda_metadata
        if state_manager is None:
            state_manager = ctx.kda_state
        if md is None or state_manager is None:
            raise RuntimeError(
                "Qwen3NextGDNAttention requires engine-managed recurrent state "
                "and metadata",
            )
        x = hidden_states
        if x.dim() != 2:
            x = x.reshape(-1, self.hidden_size)
        N = x.shape[0]
        cu = md.query_start_loc_int32
        if cu is None:
            cu = md.non_spec_query_start_loc.to(torch.int32)
        if N > self._MAX_T or cu.numel() != 2 or not self._fast_ok(
                md, state_manager):
            return super().forward_impl(hidden_states, state_manager)

        use_init = bool(
            md.has_initial_state is None or md.all_have_initial_state
            or md.any_have_initial_state
        )
        li = self.layer_idx
        owners = (cu, md.non_spec_state_indices_tensor,
                  state_manager.gdn_conv[li], state_manager.recurrent[li])
        progs = self.__dict__.get("_gdn_progs")
        if progs is None:
            progs = self.__dict__["_gdn_progs"] = {}
        prog = progs.get(N)
        if prog is None or not prog.matches(x, use_init, owners):
            if x.stride(1) != 1 or x.data_ptr() % 16 != 0:
                return super().forward_impl(hidden_states, state_manager)
            if len(progs) > 16:
                progs.clear()
            prog = self._build_prog(N, x, *owners, use_init)
            progs[N] = prog
        else:
            prog.launch(x)

        y = torch.empty(N, prog.y_dim, device=prog.device, dtype=prog.dtype)
        return torch.mm(prog.yn, prog.woT, out=y)
