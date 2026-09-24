"""Input embedder for AlphaFold3 -- fused Triton implementation.

Produces initial single (s) and pair (z) representations from token and
atom features.

Reference: openfold3/core/model/feature_embedders/input_embedders.py
           InputEmbedderAllAtom

Why this file looks nothing like the baseline
---------------------------------------------
At the captured shape (1 batch, 16 tokens, 368 atoms) the baseline issues 627
CUDA kernels per forward for about 1 GFLOP of work.  Nothing is bandwidth- or
flop-bound: every kernel is launch and memory latency, and on this GPU a launch
costs ~7 us of host time either way, so the operator is bound by *how many
kernels* run and by *how many warps each one keeps resident*.  Two rules follow,
and they pull against each other:

* Fuse aggressively -- 627 launches become 15.
* But give every kernel a grid of a few hundred blocks.  At these sizes a
  program's throughput is set by its outstanding loads, so a kernel with 24
  blocks moves ~0.2 TB/s while the same bytes across 384 blocks move ~1.6 TB/s.
  That is why the work is partitioned over *output channels* as well as rows,
  and why widening a row tile (fewer, fatter programs, fewer redundant weight
  reads) measures slower every time it was tried.

The resulting schedule, in order:

  1. ``_k_cl``     -- the reference-atom feature embedding, output channels split
     across programs.  Also publishes ``n_real`` (the reference computes the atom
     count as a bf16 sum, which actually shifts the key windows) and clears the
     token-aggregation accumulator, so neither costs a launch.
  2. ``_k_cond``   -- every AdaLN conditioning tensor for all three transformer
     blocks, one program per (block, source, half).  These depend only on ``cl``,
     which is the atom transformer's ``s`` and is constant across blocks, so all
     24 conditioning GEMMs are hoisted out of the block loop and done once.
  3. ``_k_zb``     -- the blocked atom pair representation: offset,
     inverse-square-distance and valid-mask features, the ``cl_lm`` term, the
     3-layer pair MLP, ``layer_norm_z`` and every block's ``linear_z`` head bias.
     ``plm`` is never materialized; only the [query, key, head] attention bias
     survives, which is 16x smaller.
  4-12. per transformer block: ``_k_proj`` (AdaLN + one of Q/gate/K/V for one
     column slice, per atom), ``_k_attn_a`` (sequence-local attention, one
     program per query tile and head), ``_k_attn_b`` (output projection,
     residual, and the SwiGLU transition walked in hidden-dim chunks).
  13. ``_k_agg``   -- ``linear_q`` + relu + the atom->token mean, reduced inside
     each atom tile before the atomics so contention stays near 2-way.
  14. ``_k_sz``    -- ``s_input``, ``s``, ``z_i`` and ``z_j``; ``linear_s`` and
     both ``linear_z`` are stacked into one weight so a program owns a 64-column
     stripe of all three.
  15. ``_k_z``     -- ``z``, with ``relpos_complex`` evaluated as three
     cumulative-sum table lookups instead of building the 139-channel one-hot
     and multiplying it out.

Numerics follow the baseline's bf16 rounding points -- every ``Linear``, every
``LayerNorm``, every activation rounds to bf16 while reductions accumulate in
fp32 -- so the fused path is not merely inside tolerance, it rounds where torch
rounds.  The worst observed output difference is one bf16 ulp.

Anything the fast path does not recognise (no atom features, different
head/block geometry, non-bf16, CPU tensors, autograd enabled, no Triton) falls
through to the reference implementation below, which is kept verbatim.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear
from ..L1.tensor_ops import OneHot, Pad
from .alphafold3_atom_attention import AtomAttentionEncoder

try:
    import triton
    import triton.language as tl

    _HAVE_TRITON = True
except Exception:  # noqa: BLE001 - no triton: the reference path still runs
    _HAVE_TRITON = False


def _binned_one_hot(
    x: torch.Tensor, boundaries: torch.Tensor,
) -> torch.Tensor:
    """One-hot encoding with bin boundaries (matches reference binned_one_hot)."""
    return (x[..., None] > boundaries).to(dtype=x.dtype)


def relpos_complex(
    batch: dict,
    max_relative_idx: int,
    max_relative_chain: int,
) -> torch.Tensor:
    """Build relative position features matching the reference implementation.

    Produces 139 features when max_relative_idx=32, max_relative_chain=2:
      66 (rel_pos) + 66 (rel_token) + 1 (same_entity) + 6 (rel_chain)

    Reference: openfold3/core/utils/relpos.py relpos_complex
    """
    res_idx = batch["residue_index"]
    asym_id = batch["asym_id"]
    entity_id = batch["entity_id"]
    same_chain = asym_id[..., None] == asym_id[..., None, :]
    same_res = res_idx[..., None] == res_idx[..., None, :]
    same_entity = entity_id[..., None] == entity_id[..., None, :]

    def _relpos(
        pos: torch.Tensor, condition: torch.BoolTensor, rel_clip_idx: int,
    ) -> torch.Tensor:
        offset = pos[..., None] - pos[..., None, :]
        clipped_offset = torch.clamp(offset + rel_clip_idx, min=0, max=2 * rel_clip_idx)
        final_offset = torch.where(
            condition,
            clipped_offset,
            (2 * rel_clip_idx + 1) * torch.ones_like(clipped_offset),
        )
        boundaries = torch.arange(
            start=0, end=2 * rel_clip_idx + 2, device=final_offset.device,
        ).to(dtype=final_offset.dtype)
        return _binned_one_hot(final_offset, boundaries)

    rel_pos = _relpos(pos=res_idx, condition=same_chain, rel_clip_idx=max_relative_idx)
    rel_token = _relpos(
        pos=batch["token_index"],
        condition=same_chain & same_res,
        rel_clip_idx=max_relative_idx,
    )
    rel_chain = _relpos(
        pos=batch["sym_id"],
        condition=same_entity,
        rel_clip_idx=max_relative_chain,
    )

    same_entity_feat = same_entity[..., None].to(dtype=rel_pos.dtype)

    return torch.cat([rel_pos, rel_token, same_entity_feat, rel_chain], dim=-1)


# ###########################################################################
# Fused Triton path
# ###########################################################################
if _HAVE_TRITON:

    @triton.jit
    def _bfr(x):
        """Round an fp32 value through bf16, the way every torch op here does."""
        return x.to(tl.bfloat16).to(tl.float32)

    @triton.jit
    def _keys(nb, n_real, Q: tl.constexpr, K: tl.constexpr, K0=0, KN: tl.constexpr = 0):
        """Replay ``_get_block_key_indices`` for one query block.

        The reference does this arithmetic in bf16 (an int32 ``initial`` meets a
        bf16 ``n_real``, so torch promotes the whole expression), which actually
        shifts the window: with 368 atoms ``n_real - 1`` rounds 367 -> 368 and
        ``initial[-1]`` rounds 431 -> 432.  Reproducing the rounding is the only
        way to land on the same key set.
        """
        kk = K0 + tl.arange(0, KN if KN > 0 else K)
        center = Q // 2 + nb * Q
        initial = (center - K // 2 + kk).to(tl.float32)
        under = tl.maximum(0.0, (K // 2 - center).to(tl.float32))
        nreal_m1 = _bfr(n_real - 1.0)
        last = (center + K // 2 - 1).to(tl.float32)
        over = tl.maximum(0.0, _bfr(_bfr(last) - nreal_m1))
        shift = _bfr(tl.where(under > 0.0, under, -over))
        final = _bfr(_bfr(initial) + shift)
        valid = (final >= 0.0) & (final < n_real)
        safe = tl.minimum(tl.maximum(final, 0.0), tl.maximum(nreal_m1, 0.0))
        return safe.to(tl.int32), valid

    @triton.jit
    def _ln(x, C: tl.constexpr, EPS: tl.constexpr):
        """Row-wise LayerNorm with no affine, fp32 reduction, bf16 result.

        An all-zero row (atom padding) normalizes to zero, which is what
        ``F.layer_norm`` gives and what the reference relies on.
        """
        mu = tl.sum(x, 1) / C
        xc = x - mu[:, None]
        var = tl.sum(xc * xc, 1) / C
        return _bfr(xc * tl.rsqrt(var + EPS)[:, None])

    # -- 1a. cl: the reference-atom feature embedding -------------------------
    @triton.jit
    def _k_cl(
        RP, RC, RM, RE, RCH, AM,
        W_SMALL, W_ELEM, W_CHARS, CL, SCAL, ACCB,
        A, AP, TT,
        BM: tl.constexpr, C: tl.constexpr, NC: tl.constexpr, EP: tl.constexpr,
        CH: tl.constexpr, APAD: tl.constexpr, EDIM: tl.constexpr,
        CHDIM: tl.constexpr, CTOK1: tl.constexpr,
    ):
        pm = tl.program_id(0)
        pn = tl.program_id(1)
        pb = tl.program_id(2)
        rows = pm * BM + tl.arange(0, BM)
        rmask = rows < A
        cc = pn * NC + tl.arange(0, NC)

        # cl is the sum of five Linear projections of the reference atom features.
        # The 3+1+1 scalar features are packed into one 16-wide tile so this is
        # three tl.dot calls rather than five (K < 16 has no tensor-core path),
        # and the output channels are split across programs so each one touches a
        # slice of each weight.
        o16 = tl.arange(0, 16)
        pos = tl.load(
            RP + pb * A * 3 + rows[:, None] * 3 + o16[None, :],
            mask=rmask[:, None] & (o16[None, :] < 3), other=0.0,
        ).to(tl.float32)
        chg = tl.load(RC + pb * A + rows, mask=rmask, other=0.0).to(tl.float32)
        mag = tl.abs(chg)
        asinh = _bfr(
            tl.where(chg >= 0.0, 1.0, -1.0) * tl.log(mag + tl.sqrt(mag * mag + 1.0))
        )
        rmk = tl.load(RM + pb * A + rows, mask=rmask, other=0.0).to(tl.float32)
        small = (pos
                 + tl.where(o16[None, :] == 3, asinh[:, None], 0.0)
                 + tl.where(o16[None, :] == 4, rmk[:, None], 0.0))
        acc = tl.dot(small.to(tl.bfloat16),
                     tl.load(W_SMALL + o16[:, None] * C + cc[None, :]))
        oe = tl.arange(0, EP)
        acc = tl.dot(
            tl.load(RE + pb * A * EDIM + rows[:, None] * EDIM + oe[None, :],
                    mask=rmask[:, None] & (oe[None, :] < EDIM), other=0.0),
            tl.load(W_ELEM + oe[:, None] * C + cc[None, :]), acc)
        och = tl.arange(0, CH)
        acc = tl.dot(
            tl.load(RCH + pb * A * CHDIM + rows[:, None] * CHDIM + och[None, :],
                    mask=rmask[:, None] & (och[None, :] < CHDIM), other=0.0),
            tl.load(W_CHARS + och[:, None] * C + cc[None, :]), acc)
        tl.store(CL + pb * AP * C + rows[:, None] * C + cc[None, :],
                 tl.where(rmask[:, None], _bfr(acc), 0.0).to(tl.bfloat16))

        if pm == 0 and pn == 0:
            # n_real is a bf16 sum in the reference; torch accumulates it in fp32
            # and rounds once, so do exactly that and publish it for the later
            # kernels instead of re-reducing it per program.
            oa = tl.arange(0, APAD)
            amv = tl.load(AM + pb * A + oa, mask=oa < A, other=0.0).to(tl.float32)
            tl.store(SCAL + pb, _bfr(tl.sum(amv)))
            # Clear the token-aggregation accumulator here; the aggregation kernel
            # atomically adds into it, and a separate zero_() would be a launch.
            nz = TT * CTOK1
            for z0 in range(0, nz, 1024):
                oz = z0 + tl.arange(0, 1024)
                tl.store(ACCB + pb * nz + oz, tl.zeros([1024], tl.float32),
                         mask=oz < nz)

    # -- 1b. AdaLN conditioning: one program per (block, source, half) --------
    @triton.jit
    def _k_cond(
        CL, W_LM, LNW, WCOND, BCOND, PLM, COND,
        AP, SCOND,
        BM: tl.constexpr, C: tl.constexpr, P: tl.constexpr, NSRC: tl.constexpr,
        EPS: tl.constexpr,
    ):
        pm = tl.program_id(0)
        ps = tl.program_id(1)
        pb = tl.program_id(2)
        half = ps % 2
        src = (ps // 2) % NSRC
        blk = ps // (2 * NSRC)
        rows = pm * BM + tl.arange(0, BM)
        cc = tl.arange(0, C)
        cl = tl.load(CL + pb * AP * C + rows[:, None] * C + cc[None, :]).to(tl.float32)

        if ps == 0:
            # linear_l / linear_m are per-atom, so the pair kernel gets them from
            # here instead of recomputing them for every (query, key) pair.
            o2p = tl.arange(0, 2 * P)
            lm = _bfr(tl.dot(tl.maximum(cl, 0.0).to(tl.bfloat16),
                             tl.load(W_LM + cc[:, None] * (2 * P) + o2p[None, :])))
            tl.store(PLM + pb * AP * (2 * P) + rows[:, None] * (2 * P) + o2p[None, :],
                     lm.to(tl.bfloat16))

        # layer_norm_s(cl) is shared by the q / k / transition AdaLNs up to the
        # per-block scale, but that scale is applied before the bf16 round, so it
        # cannot be folded into the weight: normalize, then rescale.
        lw = tl.load(LNW + (blk * NSRC + src) * C + cc).to(tl.float32)
        x = tl.where(src == NSRC - 1, cl,
                     _bfr(_ln(cl, C, EPS) * lw[None, :])).to(tl.bfloat16)
        v = _bfr(tl.dot(x, tl.load(WCOND + ((blk * NSRC + src) * 2 + half) * C * C
                                   + cc[:, None] * C + cc[None, :]))
                 + tl.load(BCOND + ((blk * NSRC + src) * 2 + half) * C + cc)[None, :])
        # linear_g outputs are gated; linear_s outputs are not -- except source
        # NSRC-1, which is linear_ada_out and the transition gate, both gated.
        v = tl.where((half == 0) | (src == NSRC - 1), _bfr(tl.sigmoid(v)), v)
        tl.store(COND + (blk * 8 + src * 2 + half) * SCOND + pb * AP * C
                 + rows[:, None] * C + cc[None, :], v.to(tl.bfloat16))

    # -- 2. blocked pair representation -> per-head attention bias -----------
    @triton.jit
    def _k_zb(
        RP, UID, AM, PLM, SCAL,
        W_OFF, W_INV, W_VAL, W_MLP, LNZ, WZ, ZB,
        A, AP, NB,
        Q: tl.constexpr, K: tl.constexpr, KT: tl.constexpr, P: tl.constexpr,
        NH: tl.constexpr, NZ: tl.constexpr, EPS: tl.constexpr,
    ):
        pid = tl.program_id(0)
        pb = tl.program_id(1)
        nkt = K // KT
        nb = pid // (Q * nkt)
        ql = (pid // nkt) % Q
        kt = pid % nkt
        row = nb * Q + ql

        n_real = tl.load(SCAL + pb)
        safe, valid = _keys(nb, n_real, Q, K, kt * KT, KT)
        op = tl.arange(0, P)

        mq = tl.load(AM + pb * A + row, mask=row < A, other=0.0).to(tl.float32)
        amk = tl.load(AM + pb * A + safe, mask=safe < A, other=0.0).to(tl.float32)
        bm = _bfr(mq * _bfr(tl.where(valid, 1.0, 0.0) * amk))          # [K]

        rpb = RP + pb * A * 3
        dl0 = tl.load(rpb + row * 3 + 0, mask=row < A, other=0.0).to(tl.float32)
        dl1 = tl.load(rpb + row * 3 + 1, mask=row < A, other=0.0).to(tl.float32)
        dl2 = tl.load(rpb + row * 3 + 2, mask=row < A, other=0.0).to(tl.float32)
        km = (safe < A) & valid
        dm0 = tl.load(rpb + safe * 3 + 0, mask=km, other=0.0).to(tl.float32)
        dm1 = tl.load(rpb + safe * 3 + 1, mask=km, other=0.0).to(tl.float32)
        dm2 = tl.load(rpb + safe * 3 + 2, mask=km, other=0.0).to(tl.float32)
        vl = tl.load(UID + pb * A + row, mask=row < A, other=0.0).to(tl.float32)
        vm = tl.load(UID + pb * A + safe, mask=km, other=0.0).to(tl.float32)

        d0 = _bfr(_bfr(dl0 - dm0) * bm)
        d1 = _bfr(_bfr(dl1 - dm1) * bm)
        d2 = _bfr(_bfr(dl2 - dm2) * bm)
        vlm = _bfr(tl.where(vl == vm, 1.0, 0.0) * bm)
        # ``dlm ** 2`` is an elementwise bf16 op in the reference, so each square
        # rounds before the length-3 reduction does.
        s2 = _bfr(_bfr(d0 * d0) + _bfr(d1 * d1) + _bfr(d2 * d2))
        inv = _bfr(1.0 / _bfr(1.0 + s2))

        w0 = tl.load(W_OFF + 0 * P + op)
        w1 = tl.load(W_OFF + 1 * P + op)
        w2 = tl.load(W_OFF + 2 * P + op)
        plm = _bfr(_bfr(d0[:, None] * w0[None, :]
                        + d1[:, None] * w1[None, :]
                        + d2[:, None] * w2[None, :]) * vlm[:, None])
        plm = _bfr(plm + _bfr(_bfr(inv[:, None] * tl.load(W_INV + op)[None, :])
                              * vlm[:, None]))
        plm = _bfr(plm + _bfr(_bfr(vlm[:, None] * tl.load(W_VAL + op)[None, :])
                              * vlm[:, None]))

        # cl_lm: linear_l / linear_m are per-atom, so they were computed once in
        # _k_cond and only the [query + key] outer sum happens here.
        pl = tl.load(PLM + pb * AP * (2 * P) + row * (2 * P) + op,
                     mask=row < AP, other=0.0).to(tl.float32)
        pmk = tl.load(PLM + pb * AP * (2 * P) + safe[:, None] * (2 * P) + P + op[None, :],
                      mask=(safe < AP)[:, None], other=0.0).to(tl.float32)
        plm = _bfr(plm + _bfr(_bfr(pl[None, :] + pmk) * bm[:, None]))

        h = tl.maximum(plm, 0.0).to(tl.bfloat16)
        h = tl.maximum(_bfr(tl.dot(
            h, tl.load(W_MLP + 0 * P * P + op[:, None] * P + op[None, :]))), 0.0)
        h = tl.maximum(_bfr(tl.dot(
            h.to(tl.bfloat16),
            tl.load(W_MLP + 1 * P * P + op[:, None] * P + op[None, :]))), 0.0)
        h = _bfr(tl.dot(
            h.to(tl.bfloat16),
            tl.load(W_MLP + 2 * P * P + op[:, None] * P + op[None, :])))
        plm = _bfr(_bfr(plm + h) * bm[:, None])

        # layer_norm_z is applied once by the transformer, then each block's
        # linear_z head projection.  All three blocks' heads fit in one 16-wide
        # dot, so the whole bias table is produced with a single tl.dot.
        zn = _bfr(_ln(plm, P, EPS) * tl.load(LNZ + op)[None, :]).to(tl.bfloat16)
        zb = _bfr(tl.dot(zn, tl.load(WZ + op[:, None] * NZ + tl.arange(0, NZ)[None, :])))
        onz = tl.arange(0, NZ)
        tl.store(
            ZB + ((pb * NB + nb) * Q + ql) * NZ * K + kt * KT
            + onz[:, None] * K + tl.arange(0, KT)[None, :],
            tl.trans(zb).to(tl.bfloat16),
        )

    # -- 3a. per-atom AdaLN + one attention projection column slice ----------
    @triton.jit
    def _k_proj(
        AIN, COND, WATT, BQ, PROJ, AP, SCOND, SPROJ, SCALE,
        BLK: tl.constexpr, BM: tl.constexpr, C: tl.constexpr, NSC: tl.constexpr,
        EPS: tl.constexpr,
    ):
        """AdaLN(a, cl), then one of Q / gate / K / V for one column slice.

        Q, K, V and the output gate are all per-atom, so splitting them over the
        atom axis (instead of per query block, where each atom sits in ~4 key
        windows) removes a 4x redundancy; splitting the output channels on top of
        that is what makes the grid big enough to keep the machine busy, since
        every kernel here is limited by resident warps rather than by flops.
        """
        pm = tl.program_id(0)
        pj = tl.program_id(1)
        pb = tl.program_id(2)
        ns = C // NSC
        j = pj // ns
        rows = pm * BM + tl.arange(0, BM)
        cc = tl.arange(0, C)
        off = pb * AP * C + rows[:, None] * C + cc[None, :]

        an = _ln(tl.load(AIN + off).to(tl.float32), C, EPS)
        qside = j < 2
        g = tl.load(COND + (BLK * 8 + tl.where(qside, 0, 2)) * SCOND + off
                    ).to(tl.float32)
        sv = tl.load(COND + (BLK * 8 + tl.where(qside, 1, 3)) * SCOND + off
                     ).to(tl.float32)
        aa = _bfr(g * _bfr(an + sv)).to(tl.bfloat16)

        # PROJ slot order is q, gate, k, v; linear_* order in WATT is q, k, v, g.
        widx = tl.where(j == 0, 0, tl.where(j == 1, 3, tl.where(j == 2, 1, 2)))
        oc = (pj % ns) * NSC + tl.arange(0, NSC)
        v = _bfr(tl.dot(aa, tl.load(WATT + (BLK * 5 + widx) * C * C
                                    + cc[:, None] * C + oc[None, :]))
                 + tl.where(j == 0, tl.load(BQ + BLK * C + oc), 0.0)[None, :])
        v = tl.where(j == 0, _bfr(v / SCALE), v)
        v = tl.where(j == 1, _bfr(tl.sigmoid(v)), v)
        tl.store(PROJ + j * SPROJ + pb * AP * C + rows[:, None] * C + oc[None, :],
                 v.to(tl.bfloat16))

    # -- 3b. sequence-local attention, one program per (query block, head) ---
    @triton.jit
    def _k_attn_a(
        AM, SCAL, ZB, PROJ, OV, A, AP, NB, SPROJ,
        BLK: tl.constexpr, Q: tl.constexpr, QT: tl.constexpr, K: tl.constexpr,
        C: tl.constexpr, NH: tl.constexpr, CH: tl.constexpr, NZ: tl.constexpr,
        INF: tl.constexpr,
    ):
        # Query rows are independent given the block's key window, so the grid is
        # split QT rows at a time: at this size every kernel is limited by how
        # many warps are resident machine-wide, and block count is the only knob
        # that moves that.
        nb = tl.program_id(0) // (Q // QT)
        qt = tl.program_id(0) % (Q // QT)
        h = tl.program_id(1)
        pb = tl.program_id(2)
        rows = nb * Q + qt * QT + tl.arange(0, QT)
        oh = h * CH + tl.arange(0, CH)

        n_real = tl.load(SCAL + pb)
        safe, valid = _keys(nb, n_real, Q, K)
        mq = tl.load(AM + pb * A + rows, mask=rows < A, other=0.0).to(tl.float32)
        amk = tl.load(AM + pb * A + safe, mask=safe < A, other=0.0).to(tl.float32)
        bm = _bfr(mq[:, None] * _bfr(tl.where(valid, 1.0, 0.0) * amk)[None, :])
        # A fully masked row gets -INF on every key, so softmax falls back to the
        # uniform distribution the reference's does -- no NaN, and the row is
        # zeroed downstream anyway.  Keys that the window clamped are masked the
        # same way, which is why the gathered K/V need no separate zeroing.
        mbias = _bfr(INF * (bm - 1.0))

        pin = PROJ + pb * AP * C
        kmask = (safe < AP)[:, None]
        qh = tl.load(pin + rows[:, None] * C + oh[None, :])
        gg = tl.load(pin + SPROJ + rows[:, None] * C + oh[None, :]).to(tl.float32)
        kh = tl.load(pin + 2 * SPROJ + safe[:, None] * C + oh[None, :],
                     mask=kmask, other=0.0)
        vh = tl.load(pin + 3 * SPROJ + safe[:, None] * C + oh[None, :],
                     mask=kmask, other=0.0)
        zb = tl.load(ZB + ((pb * NB + nb) * Q + qt * QT) * NZ * K
                     + tl.arange(0, QT)[:, None] * NZ * K + (BLK * NH + h) * K
                     + tl.arange(0, K)[None, :]).to(tl.float32)
        sc = _bfr(_bfr(_bfr(tl.dot(qh, tl.trans(kh))) + mbias) + zb)
        e = tl.exp(sc - tl.max(sc, 1)[:, None])
        pr = _bfr(e / tl.sum(e, 1)[:, None]).to(tl.bfloat16)
        tl.store(OV + pb * AP * C + rows[:, None] * C + oh[None, :],
                 _bfr(_bfr(tl.dot(pr, vh)) * gg).to(tl.bfloat16))

    # -- 3c. output projection, residual, transition -- all per atom ---------
    @triton.jit
    def _k_attn_b(
        AIN, AOUT, AM, COND, OV, WATT, WAB, WOUT, A, AP, SCOND,
        BLK: tl.constexpr, BM: tl.constexpr, C: tl.constexpr, HID: tl.constexpr,
        HT: tl.constexpr, EPS: tl.constexpr,
    ):
        pm = tl.program_id(0)
        pb = tl.program_id(1)
        rows = pm * BM + tl.arange(0, BM)
        rmask = rows < A
        cc = tl.arange(0, C)
        off = rows[:, None] * C + cc[None, :]
        cb = COND + pb * AP * C + off
        mq = tl.load(AM + pb * A + rows, mask=rmask, other=0.0).to(tl.float32)

        # linear_o over the concatenated heads: one dot with fp32 accumulation,
        # exactly what the reference's single 128-wide GEMM does.
        o = _bfr(tl.dot(tl.load(OV + pb * AP * C + off),
                        tl.load(WATT + (BLK * 5 + 4) * C * C
                                + cc[:, None] * C + cc[None, :])))
        a0 = tl.load(AIN + pb * AP * C + off).to(tl.float32)
        a1 = _bfr(a0 + _bfr(tl.load(cb + (BLK * 8 + 6) * SCOND).to(tl.float32) * o))

        x = _bfr(tl.load(cb + (BLK * 8 + 4) * SCOND).to(tl.float32)
                 * _bfr(_ln(a1, C, EPS)
                        + tl.load(cb + (BLK * 8 + 5) * SCOND).to(tl.float32))
                 ).to(tl.bfloat16)
        # SwiGLU walked HT hidden units at a time: the gate is elementwise in the
        # hidden axis and linear_out reduces over it, so a chunked loop keeps the
        # resident weight tile at [C, HT] instead of [C, 2*HID] (128 KiB, which
        # alone capped the SM at one block) and lets the loads pipeline.
        wab = WAB + BLK * C * 2 * HID + cc[:, None] * (2 * HID)
        wo = WOUT + BLK * HID * C
        ohd = tl.arange(0, HT)
        acc = tl.zeros([BM, C], tl.float32)
        for hc in tl.range(0, HID // HT):
            oh = hc * HT + ohd
            ha = _bfr(tl.dot(x, tl.load(wab + oh[None, :])))
            hb = _bfr(tl.dot(x, tl.load(wab + HID + oh[None, :])))
            acc = tl.dot(_bfr(_bfr(ha * tl.sigmoid(ha)) * hb).to(tl.bfloat16),
                         tl.load(wo + oh[:, None] * C + cc[None, :]), acc)
        a2 = tl.where(rmask[:, None], _bfr(a1 + _bfr(
            _bfr(tl.load(cb + (BLK * 8 + 7) * SCOND).to(tl.float32) * _bfr(acc))
            * mq[:, None])), 0.0)
        tl.store(AOUT + pb * AP * C + off, a2.to(tl.bfloat16))

    # -- 3d. linear_q + relu + the atom -> token mean -------------------------
    @triton.jit
    def _k_agg(
        AIN, AM, A2T, W_AGG, ACCB, A, AP, TT,
        BM: tl.constexpr, C: tl.constexpr, CTOK: tl.constexpr,
        CTOKP: tl.constexpr, CTOK1: tl.constexpr, CAGG: tl.constexpr,
        TP: tl.constexpr,
    ):
        pm = tl.program_id(0)
        cg = tl.program_id(1)
        pb = tl.program_id(2)
        rows = pm * BM + tl.arange(0, BM)
        rmask = rows < A
        cc = tl.arange(0, C)
        oc = cg * CAGG + tl.arange(0, CAGG)
        mq = tl.load(AM + pb * A + rows, mask=rmask, other=0.0).to(tl.float32)
        ab = _bfr(tl.load(AIN + pb * AP * C + rows[:, None] * C + cc[None, :]
                          ).to(tl.float32) * mq[:, None]).to(tl.bfloat16)
        av = tl.maximum(_bfr(tl.dot(ab, tl.load(
            W_AGG + cc[:, None] * CTOKP + oc[None, :]))), 0.0)

        # Atoms are ordered by token, so a BM-atom tile touches one or two tokens.
        # Reducing inside the tile first (a one-hot matmul with fp32 accumulation)
        # turns ~23-way atomic contention per address into ~2-way.
        tok = tl.where(rmask, tl.load(A2T + pb * A + rows, mask=rmask, other=0), -1)
        ot = tl.arange(0, TP)
        hot = tl.where(tok[None, :] == ot[:, None], 1.0, 0.0)
        keep = (tl.sum(hot, 1) > 0.0) & (ot < TT)
        tl.atomic_add(ACCB + pb * TT * CTOK1 + ot[:, None] * CTOK1 + oc[None, :],
                      tl.dot(hot.to(tl.bfloat16),
                             _bfr(av * mq[:, None]).to(tl.bfloat16)),
                      mask=keep[:, None] & (oc[None, :] < CTOK))
        if cg == 0:
            tl.atomic_add(ACCB + pb * TT * CTOK1 + ot * CTOK1 + CTOK,
                          tl.sum(hot * mq[None, :], 1), mask=keep)

    @triton.jit
    def _si_cols(ACCB, TF, rows, rmask, cols, TD,
                 CTOK: tl.constexpr, CTOK1: tl.constexpr):
        """s_input columns ``cols``: the token-mean atom features, then the three
        token_features slices the reference concatenates onto them.

        restype lands at [CTOK, CTOK+32) from token_features[:32] and profile at
        [CTOK+32, CTOK+64) from token_features[32:64) -- both read source column
        ``col - CTOK``, so one masked load covers the pair.
        """
        acc = tl.load(ACCB + rows[:, None] * CTOK1 + cols[None, :],
                      mask=rmask[:, None] & (cols[None, :] < CTOK), other=0.0)
        cnt = tl.load(ACCB + rows * CTOK1 + CTOK, mask=rmask, other=1.0)
        ai = tl.where(cols[None, :] < CTOK,
                      _bfr(acc / tl.maximum(cnt, 1.0)[:, None]), 0.0)
        tfv = tl.load(TF + rows[:, None] * TD + (cols[None, :] - CTOK),
                      mask=(rmask[:, None] & (cols[None, :] >= CTOK)
                            & (cols[None, :] < CTOK + 64)), other=0.0).to(tl.float32)
        dmv = tl.load(TF + rows * TD + (TD - 1), mask=rmask, other=0.0).to(tl.float32)
        return ai + tfv + tl.where(cols[None, :] == CTOK + 64, dmv[:, None], 0.0)

    # -- 4a. s_input, s, z_i, z_j -- one program per 64 output columns -------
    @triton.jit
    def _k_sz(
        ACCB, TF, W_ALL, SI_OUT, S_OUT, ZI, ZJ, TT, TD,
        TP: tl.constexpr, SPAD: tl.constexpr, KB: tl.constexpr, NC: tl.constexpr,
        NCOL: tl.constexpr, CSQ: tl.constexpr, CS: tl.constexpr, CZ: tl.constexpr,
        CTOK: tl.constexpr, CTOK1: tl.constexpr, SIDIM: tl.constexpr,
    ):
        pn = tl.program_id(0)
        pb = tl.program_id(1)
        rows = tl.arange(0, TP)
        rmask = rows < TT
        base = pn * NC
        oc = base + tl.arange(0, NC)
        accb = ACCB + pb * TT * CTOK1
        tfb = TF + pb * TT * TD

        # linear_s, linear_z_i and linear_z_j are stacked into one [SIDIM, NCOL]
        # weight, so a program owns a 64-column stripe of the three of them and
        # touches 1/10th of the weight bytes a single program would.
        acc = tl.zeros([TP, NC], tl.float32)
        for kb in tl.static_range(SPAD // KB):
            ck = kb * KB + tl.arange(0, KB)
            sib = _si_cols(accb, tfb, rows, rmask, ck, TD, CTOK, CTOK1).to(tl.bfloat16)
            if pn == 0:
                tl.store(SI_OUT + pb * TT * SIDIM + rows[:, None] * SIDIM + ck[None, :],
                         sib, mask=rmask[:, None] & (ck[None, :] < SIDIM))
            acc = tl.dot(sib, tl.load(W_ALL + ck[:, None] * NCOL + oc[None, :]), acc)
        v = _bfr(acc).to(tl.bfloat16)
        if base < CSQ:
            tl.store(S_OUT + pb * TT * CS + rows[:, None] * CS + oc[None, :], v,
                     mask=rmask[:, None] & (oc[None, :] < CS))
        elif base < CSQ + CZ:
            tl.store(ZI + pb * TT * CZ + rows[:, None] * CZ + (oc[None, :] - CSQ), v,
                     mask=rmask[:, None])
        else:
            tl.store(ZJ + pb * TT * CZ + rows[:, None] * CZ + (oc[None, :] - CSQ - CZ),
                     v, mask=rmask[:, None])

    # -- 4b. z -- one program per row of the pair representation --------------
    @triton.jit
    def _k_z(
        ZI, ZJ, RI, TI, ASYM, ENT, SYM, TB, P1, P2, P3, W_SE, W_BOND, Z_OUT, TT,
        TP: tl.constexpr, CZ: tl.constexpr, K1: tl.constexpr, K2: tl.constexpr,
        NB1: tl.constexpr, NB3: tl.constexpr,
    ):
        ti = tl.program_id(0)
        pb = tl.program_id(1)
        rows = tl.arange(0, TP)
        rmask = rows < TT
        ocz = tl.arange(0, CZ)

        # relpos_complex: every feature block is a contiguous run of one-hot bins,
        # so the 139-wide one-hot times linear_relpos is a prefix sum of the weight
        # columns indexed by the bin count -- three table lookups instead of
        # building [T, T, 139] and multiplying it out.
        ri = tl.load(RI + pb * TT + rows, mask=rmask, other=0.0).to(tl.float32)
        tix = tl.load(TI + pb * TT + rows, mask=rmask, other=0.0).to(tl.float32)
        asym = tl.load(ASYM + pb * TT + rows, mask=rmask, other=0.0).to(tl.float32)
        ent = tl.load(ENT + pb * TT + rows, mask=rmask, other=0.0).to(tl.float32)
        sym = tl.load(SYM + pb * TT + rows, mask=rmask, other=0.0).to(tl.float32)
        ri_i = tl.load(RI + pb * TT + ti).to(tl.float32)
        ti_i = tl.load(TI + pb * TT + ti).to(tl.float32)
        as_i = tl.load(ASYM + pb * TT + ti).to(tl.float32)
        en_i = tl.load(ENT + pb * TT + ti).to(tl.float32)
        sy_i = tl.load(SYM + pb * TT + ti).to(tl.float32)
        same_chain = as_i == asym
        same_ent = en_i == ent
        f1 = tl.where(same_chain,
                      tl.minimum(tl.maximum(_bfr(_bfr(ri_i - ri) + K1), 0.0),
                                 float(2 * K1)), float(2 * K1 + 1))
        f2 = tl.where(same_chain & (ri_i == ri),
                      tl.minimum(tl.maximum(_bfr(_bfr(ti_i - tix) + K1), 0.0),
                                 float(2 * K1)), float(2 * K1 + 1))
        f3 = tl.where(same_ent,
                      tl.minimum(tl.maximum(_bfr(_bfr(sy_i - sym) + K2), 0.0),
                                 float(2 * K2)), float(2 * K2 + 1))
        c1 = tl.minimum(tl.maximum(tl.ceil(f1), 0.0), float(NB1)).to(tl.int32)
        c2 = tl.minimum(tl.maximum(tl.ceil(f2), 0.0), float(NB1)).to(tl.int32)
        c3 = tl.minimum(tl.maximum(tl.ceil(f3), 0.0), float(NB3)).to(tl.int32)
        rel = _bfr(tl.load(P1 + c1[:, None] * CZ + ocz[None, :])
                   + tl.load(P2 + c2[:, None] * CZ + ocz[None, :])
                   + tl.load(P3 + c3[:, None] * CZ + ocz[None, :])
                   + tl.where(same_ent, 1.0, 0.0)[:, None]
                   * tl.load(W_SE + ocz)[None, :])

        zi = tl.load(ZI + pb * TT * CZ + ti * CZ + ocz).to(tl.float32)
        zj = tl.load(ZJ + pb * TT * CZ + rows[:, None] * CZ + ocz[None, :],
                     mask=rmask[:, None], other=0.0).to(tl.float32)
        tb = tl.load(TB + pb * TT * TT + ti * TT + rows, mask=rmask, other=0.0
                     ).to(tl.float32)
        z = _bfr(_bfr(_bfr(zi[None, :] + zj) + rel)
                 + _bfr(tb[:, None] * tl.load(W_BOND + ocz)[None, :]))
        tl.store(Z_OUT + pb * TT * TT * CZ + ti * TT * CZ + rows[:, None] * CZ
                 + ocz[None, :], z.to(tl.bfloat16), mask=rmask[:, None])


def _np2(n: int) -> int:
    return 1 << max(4, (int(n) - 1).bit_length())


class _Plan:
    """Everything the fused path needs, resolved once per (module, shape).

    Weight packing (transposes, zero padding to power-of-two tiles, the
    ``linear_relpos`` prefix sums) and the scratch allocations are loop-invariant;
    keeping them here means a forward is fifteen launches and three ``empty`` calls,
    with no host work that scales with the tensors.
    """

    __slots__ = (
        "sig", "B", "A", "AP", "NB", "T", "TD", "SCOND", "SPROJ", "scale",
        "NBLK", "SIDIM", "CS", "CZ", "w", "scratch", "jobs", "runners",
    )


def _launch(pl, i, args):
    """Launch job ``i``, reusing the compiled kernel's own launcher.

    The first call goes through ``JITFunction.run``, which re-derives the
    specialization key from ~35 arguments -- about 18 us of host time here, and
    the host is half of what this operator costs.  After that the
    ``CompiledKernel`` launcher is called directly (~8 us), which skips the
    binder, the cache lookup and the grid canonicalization.  If that private
    entry point is missing, every call just takes the public path.
    """
    fn, grid, cst, nw = pl.jobs[i]
    r = pl.runners[i]
    if r is not None:
        if r is not False:
            r(*args, *cst)
        else:
            fn[grid](*args, *cst, num_warps=nw)
        return
    ck = fn[grid](*args, *cst, num_warps=nw)
    try:
        pl.runners[i] = ck[grid]
    except Exception:  # noqa: BLE001 - stay on the public path for this job
        pl.runners[i] = False


def _build_plan(mod, token_features, batch):  # noqa: PLR0911, PLR0912
    if not _HAVE_TRITON or not token_features.is_cuda:
        return None
    if token_features.dtype is not torch.bfloat16 or token_features.dim() != 3:
        return None
    ae = mod.atom_attn_enc
    if ae.noisy_position_embedder is not None:
        return None
    need = ("ref_pos", "ref_charge", "ref_mask", "ref_element",
            "ref_atom_name_chars", "ref_space_uid", "atom_mask", "token_mask",
            "atom_to_token_index", "residue_index", "token_index", "asym_id",
            "entity_id", "sym_id", "token_bonds")
    if any(k not in batch for k in need):
        return None
    # The reference reads restype / profile / deletion_mean out of the batch when
    # present; the fused path only implements the token_features slicing.
    if any(batch.get(k) is not None for k in ("restype", "profile", "deletion_mean")):
        return None

    rfe = ae.ref_atom_feature_embedder
    blocks = ae.atom_transformer.blocks
    NBLK = len(blocks)
    b0 = blocks[0]
    mha0 = b0.attention_pair_bias.mha
    if mha0.linear_g is None or not b0.attention_pair_bias.use_ada_layer_norm:
        return None
    if getattr(ae.atom_transformer, "layer_norm_z", None) is None:
        return None

    C = rfe.linear_ref_pos.weight.shape[0]
    P = rfe.linear_ref_offset.weight.shape[0]
    NH, CH = mha0.no_heads, mha0.c_hidden
    HID = b0.conditioned_transition.swiglu.linear_a.weight.shape[0]
    CTOK = ae.linear_q[0].weight.shape[0]
    CS, CZ = mod.c_s, mod.c_z
    EDIM = rfe.linear_ref_element.weight.shape[1]
    CHDIM = rfe.linear_ref_atom_chars.weight.shape[1]
    Q, K = ae.n_query, ae.n_key
    SIDIM = mod.c_s_input
    if (NH * CH != C or C != _np2(C) or P != _np2(P) or CH != _np2(CH)
            or Q != _np2(Q) or K != _np2(K) or CZ != _np2(CZ) or HID != _np2(HID)
            or C < 16 or P < 16 or CH < 16 or Q < 16 or K < 16
            or SIDIM != CTOK + 65):
        return None

    eps = b0.attention_pair_bias.layer_norm_a_q.layer_norm_a.eps
    epss = [ae.atom_transformer.layer_norm_z.eps]
    for blk in blocks:
        apb, ct = blk.attention_pair_bias, blk.conditioned_transition
        epss += [apb.layer_norm_a_q.layer_norm_a.eps, apb.layer_norm_a_q.layer_norm_s.eps,
                 apb.layer_norm_a_k.layer_norm_a.eps, apb.layer_norm_a_k.layer_norm_s.eps,
                 ct.layer_norm.layer_norm_a.eps, ct.layer_norm.layer_norm_s.eps]
    if any(e != eps for e in epss):
        return None

    dev = token_features.device
    Bn, T, TD = token_features.shape
    if TD < 64 or batch["token_mask"].shape[-1] != T:
        return None
    A = batch["atom_mask"].shape[-1]
    NB = -(-A // Q)
    AP = NB * Q
    # Several kernels hold a whole padded token axis (or the atom mask) in one
    # tile.  Those are register-resident, so bail out rather than compile a tile
    # that would spill -- shapes this large are not what this operator sees.
    if _np2(T) > 64 or _np2(A) > 8192:
        return None

    shapes = {
        "ref_pos": (Bn, A, 3), "ref_charge": (Bn, A), "ref_mask": (Bn, A),
        "ref_element": (Bn, A, EDIM), "ref_space_uid": (Bn, A),
        "atom_mask": (Bn, A), "token_mask": (Bn, T),
        "atom_to_token_index": (Bn, A), "residue_index": (Bn, T),
        "token_index": (Bn, T), "asym_id": (Bn, T), "entity_id": (Bn, T),
        "sym_id": (Bn, T), "token_bonds": (Bn, T, T),
    }
    for k, shp in shapes.items():
        t = batch[k]
        if tuple(t.shape) != shp or not t.is_contiguous():
            return None
        if k == "atom_to_token_index":
            if t.dtype not in (torch.int64, torch.int32):
                return None
        elif t.dtype is not torch.bfloat16:
            return None
    rc = batch["ref_atom_name_chars"]
    if (not rc.is_contiguous() or rc.dtype is not torch.bfloat16
            or rc.shape[0] != Bn or rc.shape[1] != A or rc.numel() != Bn * A * CHDIM):
        return None

    bf = torch.bfloat16
    z_ = lambda *s, d=bf: torch.zeros(*s, dtype=d, device=dev)  # noqa: E731
    EP, CHP, CTOKP = _np2(EDIM), _np2(CHDIM), _np2(CTOK)
    # 8-aligned row stride: CTOK+1 would start every row off a 32-byte
    # boundary and block vectorized loads when _k_sz reads it back.
    CTOK8 = CTOK + 8
    NZ = _np2(max(16, NBLK * NH))
    SPAD = _np2(SIDIM)
    NSRC = 4

    w_small = z_(16, C)
    w_small[0:3] = rfe.linear_ref_pos.weight.t()
    w_small[3:4] = rfe.linear_ref_charge.weight.t()
    w_small[4:5] = rfe.linear_ref_mask.weight.t()
    w_elem = z_(EP, C)
    w_elem[:EDIM] = rfe.linear_ref_element.weight.t()
    w_chars = z_(CHP, C)
    w_chars[:CHDIM] = rfe.linear_ref_atom_chars.weight.t()
    w_lm = torch.cat([ae.linear_l.weight.t(), ae.linear_m.weight.t()], 1).contiguous()

    lnw = z_(NBLK * NSRC, C)
    wcond = z_(NBLK * NSRC * 2, C, C)
    bcond = z_(NBLK * NSRC * 2, C, d=torch.float32)
    watt = z_(NBLK * 5, C, C)
    bq = z_(NBLK, C, d=torch.float32)
    wab = z_(NBLK, C, 2 * HID)
    wout = z_(NBLK, HID, C)
    wz = z_(P, NZ)
    for i, blk in enumerate(blocks):
        apb, ct = blk.attention_pair_bias, blk.conditioned_transition
        srcs = (
            (apb.layer_norm_a_q.layer_norm_s.weight,
             apb.layer_norm_a_q.linear_g, apb.layer_norm_a_q.linear_s),
            (apb.layer_norm_a_k.layer_norm_s.weight,
             apb.layer_norm_a_k.linear_g, apb.layer_norm_a_k.linear_s),
            (ct.layer_norm.layer_norm_s.weight,
             ct.layer_norm.linear_g, ct.layer_norm.linear_s),
            (None, apb.linear_ada_out, ct.linear_g),
        )
        for j, (lw, la, lb) in enumerate(srcs):
            if lw is not None:
                lnw[i * NSRC + j] = lw
            s0 = (i * NSRC + j) * 2
            wcond[s0] = la.weight.t()
            wcond[s0 + 1] = lb.weight.t()
            if la.bias is not None:
                bcond[s0] = la.bias.float()
            if lb.bias is not None:
                bcond[s0 + 1] = lb.bias.float()
        mha = apb.mha
        watt[i * 5 + 0] = mha.linear_q.weight.t()
        watt[i * 5 + 1] = mha.linear_k.weight.t()
        watt[i * 5 + 2] = mha.linear_v.weight.t()
        watt[i * 5 + 3] = mha.linear_g.weight.t()
        watt[i * 5 + 4] = mha.linear_o.weight.t()
        if mha.linear_q.bias is not None:
            bq[i] = mha.linear_q.bias.float()
        wab[i] = torch.cat([ct.swiglu.linear_a.weight.t(),
                            ct.swiglu.linear_b.weight.t()], 1)
        wout[i] = ct.linear_out.weight.t()
        wz[:, i * NH:(i + 1) * NH] = apb.linear_z.weight.t()

    w_mlp = torch.stack([ae.pair_mlp[1].weight.t(), ae.pair_mlp[3].weight.t(),
                         ae.pair_mlp[5].weight.t()]).contiguous()
    w_agg = z_(C, CTOKP)
    w_agg[:, :CTOK] = ae.linear_q[0].weight.t()
    NC = 64 if CZ % 64 == 0 else 16
    CSQ = -(-CS // NC) * NC
    NCOL = CSQ + 2 * CZ
    w_all = z_(SPAD, NCOL)
    w_all[:SIDIM, :CS] = mod.linear_s.weight.t()
    w_all[:SIDIM, CSQ:CSQ + CZ] = mod.linear_z_i.weight.t()
    w_all[:SIDIM, CSQ + CZ:] = mod.linear_z_j.weight.t()

    k1, k2 = mod.relpos_k, mod.max_relative_chain
    n1, n3 = 2 * k1 + 2, 2 * k2 + 2
    W = mod.linear_relpos.weight.float()
    if W.shape[1] != 2 * n1 + 1 + n3 or W.shape[0] != CZ:
        return None
    pad0 = torch.zeros(CZ, 1, dtype=torch.float32, device=dev)
    pre = lambda sl: torch.cat([pad0, W[:, sl].cumsum(-1)], -1).t().contiguous()  # noqa: E731
    p1 = pre(slice(0, n1))
    p2 = pre(slice(n1, 2 * n1))
    p3 = pre(slice(2 * n1 + 1, 2 * n1 + 1 + n3))
    w_se = W[:, 2 * n1].contiguous()
    w_bond = mod.linear_token_bonds.weight.reshape(-1).float().contiguous()

    pl = _Plan()
    pl.sig = (tuple(token_features.shape), A)
    pl.B, pl.A, pl.AP, pl.NB, pl.T, pl.TD = Bn, A, AP, NB, T, TD
    pl.NBLK, pl.SIDIM, pl.CS, pl.CZ = NBLK, SIDIM, CS, CZ
    pl.SCOND = Bn * AP * C
    pl.SPROJ = Bn * AP * C
    pl.scale = math.sqrt(CH)
    pl.w = (w_small, w_elem, w_chars, w_lm, lnw, wcond, bcond,
            rfe.linear_ref_offset.weight.t().float().contiguous(),
            rfe.linear_inv_sq_dists.weight.reshape(-1).float().contiguous(),
            rfe.linear_valid_mask.weight.reshape(-1).float().contiguous(),
            w_mlp, ae.atom_transformer.layer_norm_z.weight.float().contiguous(), wz,
            watt, bq, wab, wout, w_agg, w_all, p1, p2, p3, w_se, w_bond)
    pl.scratch = (
        z_(Bn, AP, C), z_(Bn, AP, 2 * P), z_(NBLK * 8, Bn, AP, C),
        z_(Bn, NB, Q, NZ, K), z_(Bn, AP, C), z_(Bn, AP, C),
        z_(4, Bn, AP, C), z_(Bn, AP, C),
        z_(Bn, T, CZ), z_(Bn, T, CZ),
        z_(Bn, T, CTOK8, d=torch.float32), z_(Bn, d=torch.float32),
    )

    BM = 16
    NSC = 32
    HT = 64
    inf = float(b0.attention_pair_bias.inf)
    NCL = 16
    cst_cl = (Q, C, NCL, EP, CHP, _np2(A), EDIM, CHDIM, CTOK8)
    cst_cond = (Q, C, P, NSRC, eps)
    KT = 32
    cst_zb = (Q, K, KT, P, NH, NZ, eps)
    QT = 16
    cst_a = (Q, QT, K, C, NH, CH, NZ, inf)
    cst_b = (BM, C, HID, HT, eps)
    cst_agg = (BM, C, CTOK, CTOKP, CTOK8, 128, _np2(T))
    cst_sz = (_np2(T), SPAD, 512, NC, NCOL, CSQ, CS, CZ, CTOK, CTOK8, SIDIM)
    cst_z = (_np2(T), CZ, k1, k2, n1, n3)
    cst_proj = (BM, C, NSC, eps)
    pl.jobs = [
        (_k_cl, (NB, C // NCL, Bn), cst_cl, 4),
        (_k_cond, (NB, NBLK * NSRC * 2, Bn), cst_cond, 4),
        (_k_zb, (NB * Q * (K // KT), Bn, 1), cst_zb, 2),
    ]
    for i in range(NBLK):
        pl.jobs.append((_k_proj, (AP // BM, 4 * (C // NSC), Bn), (i,) + cst_proj, 4))
        pl.jobs.append((_k_attn_a, (NB * (Q // QT), NH, Bn), (i,) + cst_a, 4))
        pl.jobs.append((_k_attn_b, (AP // BM, Bn, 1), (i,) + cst_b, 8))
    pl.jobs.append((_k_agg, (AP // BM, CTOKP // 128, Bn), cst_agg, 4))
    pl.jobs.append((_k_sz, (NCOL // NC, Bn, 1), cst_sz, 4))
    pl.jobs.append((_k_z, (T, Bn, 1), cst_z, 8))
    pl.runners = [None] * len(pl.jobs)
    return pl


def _run_plan(pl, token_features, batch):
    (w_small, w_elem, w_chars, w_lm, lnw, wcond, bcond, w_off, w_inv, w_val,
     w_mlp, lnz, wz, watt, bq, wab, wout, w_agg, w_all,
     p1, p2, p3, w_se, w_bond) = pl.w
    cl, plm, cond, zb, a_b, a_c, pj0, ov, zzi, zzj, accb, scal = pl.scratch
    am = batch["atom_mask"]
    rp = batch["ref_pos"]
    A, AP, NB, T, SCOND, SPROJ = pl.A, pl.AP, pl.NB, pl.T, pl.SCOND, pl.SPROJ
    sc = pl.scale

    _launch(pl, 0, (rp, batch["ref_charge"], batch["ref_mask"], batch["ref_element"],
                    batch["ref_atom_name_chars"], am, w_small, w_elem, w_chars,
                    cl, scal, accb, A, AP, T))
    _launch(pl, 1, (cl, w_lm, lnw, wcond, bcond, plm, cond, AP, SCOND))
    _launch(pl, 2, (rp, batch["ref_space_uid"], am, plm, scal, w_off, w_inv, w_val,
                    w_mlp, lnz, wz, zb, A, AP, NB))

    a2t = batch["atom_to_token_index"]
    ain, aout = cl, a_b
    for i in range(pl.NBLK):
        _launch(pl, 3 + 3 * i, (ain, cond, watt, bq, pj0, AP, SCOND, SPROJ, sc))
        _launch(pl, 4 + 3 * i, (am, scal, zb, pj0, ov, A, AP, NB, SPROJ))
        _launch(pl, 5 + 3 * i, (ain, aout, am, cond, ov, watt, wab, wout,
                                A, AP, SCOND))
        ain, aout = aout, (a_c if aout is a_b else a_b)

    n = 3 + 3 * pl.NBLK
    _launch(pl, n, (ain, am, a2t, w_agg, accb, A, AP, T))

    dev = cl.device
    s_input = torch.empty(pl.B, T, pl.SIDIM, dtype=torch.bfloat16, device=dev)
    s = torch.empty(pl.B, T, pl.CS, dtype=torch.bfloat16, device=dev)
    z = torch.empty(pl.B, T, T, pl.CZ, dtype=torch.bfloat16, device=dev)
    n += 1
    _launch(pl, n, (accb, token_features, w_all, s_input, s, zzi, zzj, T, pl.TD))
    _launch(pl, n + 1, (zzi, zzj, batch["residue_index"], batch["token_index"],
                        batch["asym_id"], batch["entity_id"], batch["sym_id"],
                        batch["token_bonds"], p1, p2, p3, w_se, w_bond, z, T))
    return s_input, s, z


class InputEmbedder(nn.Module):
    """Produces initial single and pair representations from token features.

    Matches InputEmbedderAllAtom: runs AtomAttentionEncoder to get a
    token-level representation, concatenates with restype/profile/deletion_mean
    to form s_input (449 dims), then projects to s and z.

    Args:
        c_s_input: Input single representation dimension (449 for all-atom)
        c_s: Single representation dimension
        c_z: Pair representation dimension
        relpos_k: Maximum relative residue position
        max_relative_chain: Maximum relative chain index
        c_atom: Atom single representation dim
        c_atom_pair: Atom pair representation dim
        c_token: Token dim for atom attention encoder output
    """

    def __init__(
        self,
        c_s_input: int,
        c_s: int,
        c_z: int,
        relpos_k: int = 32,
        max_relative_chain: int = 2,
        c_atom: int = 128,
        c_atom_pair: int = 16,
        c_token: int | None = None,
    ):
        super().__init__()
        self.c_s_input = c_s_input
        self.c_s = c_s
        self.c_z = c_z
        self.relpos_k = relpos_k
        self.max_relative_chain = max_relative_chain
        self._one_hot = OneHot()
        self._pad = Pad()

        if c_token is None:
            c_token = c_s

        self.atom_attn_enc = AtomAttentionEncoder(
            c_atom=c_atom,
            c_atom_pair=c_atom_pair,
            c_token=c_token,
            add_noisy_pos=False,
        )

        self.linear_s = Linear(c_s_input, c_s, bias=False)
        self.linear_z_i = Linear(c_s_input, c_z, bias=False)
        self.linear_z_j = Linear(c_s_input, c_z, bias=False)

        num_rel_pos_bins = 2 * relpos_k + 2
        num_rel_token_bins = 2 * relpos_k + 2
        num_rel_chain_bins = 2 * max_relative_chain + 2
        num_same_entity_features = 1
        n_relpos_features = (
            num_rel_pos_bins + num_rel_token_bins
            + num_rel_chain_bins + num_same_entity_features
        )
        self.linear_relpos = Linear(n_relpos_features, c_z, bias=False)

        self.linear_token_bonds = Linear(1, c_z, bias=False)

        # Fused-path state.  The plan is built on the first forward that matches
        # (weights are loaded by then) and invalidated whenever a parameter is
        # replaced or moved -- ``_apply`` covers .to()/.cuda(), and the
        # state-dict hook covers weight loading.
        self._fk_plan = None
        self._fk_off = not _HAVE_TRITON
        self._register_load_state_dict_pre_hook(self._fk_drop_hook)

    def _fk_drop_hook(self, *args, **kwargs):
        self._fk_plan = None

    def _apply(self, *args, **kwargs):
        self._fk_plan = None
        return super()._apply(*args, **kwargs)

    def _fk(self, token_features, batch):
        """Return the fused outputs, or None if this call is not supported."""
        pl = self._fk_plan
        if pl is not None:
            if (token_features.shape == pl.sig[0]
                    and batch.get("atom_mask") is not None
                    and batch["atom_mask"].shape[-1] == pl.sig[1]):
                return _run_plan(pl, token_features, batch)
            return None
        if self._fk_off:
            return None
        self._fk_off = True  # only ever attempt to build once
        pl = _build_plan(self, token_features, batch)
        if pl is None:
            return None
        self._fk_plan = pl
        self._fk_off = False
        return _run_plan(pl, token_features, batch)

    def forward(
        self,
        token_features: torch.Tensor,
        residue_index: torch.Tensor,
        batch: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            token_features: [*, N_token, c_s_input] per-token features.
                If batch contains ref_pos (atom features), only restype/profile/deletion_mean
                are expected here and atom_attn_enc produces the remaining features.
                Otherwise, treated as pre-built s_input.
            residue_index:  [*, N_token] residue indices
            batch: Feature dict for relpos and atom attention.

        Returns:
            s_input: [*, N_token, c_s_input] input single representation
            s: [*, N_token, C_s] single representation
            z: [*, N_token, N_token, C_z] pair representation
        """
        if (batch is not None and not torch.is_grad_enabled()
                and "ref_pos" in batch and "asym_id" in batch):
            out = self._fk(token_features, batch)
            if out is not None:
                return out

        if batch is not None and "ref_pos" in batch:
            a, _, _, _ = self.atom_attn_enc(batch=batch)
            s_input = torch.cat(
                [
                    a,
                    batch.get("restype", token_features[..., :32]),
                    batch.get("profile", token_features[..., 32:64]),
                    batch.get("deletion_mean", token_features[..., -1:]).unsqueeze(-1)
                    if batch.get("deletion_mean") is not None and batch["deletion_mean"].dim() == token_features.dim() - 1
                    else batch.get("deletion_mean", token_features[..., -1:]),
                ],
                dim=-1,
            )
        else:
            s_input = token_features

        s = self.linear_s(s_input)

        z_i = self.linear_z_i(s_input)[..., :, None, :]
        z_j = self.linear_z_j(s_input)[..., None, :, :]
        z = z_i + z_j

        if batch is not None and "asym_id" in batch:
            relpos_feats = relpos_complex(
                batch=batch,
                max_relative_idx=self.relpos_k,
                max_relative_chain=self.max_relative_chain,
            ).to(dtype=z.dtype)
        else:
            d = residue_index[..., :, None] - residue_index[..., None, :]
            d = d.clamp(-self.relpos_k, self.relpos_k) + self.relpos_k
            n_bins = 2 * self.relpos_k + 2
            relpos_feats = self._one_hot(d.long(), n_bins).to(
                dtype=z.dtype,
            )
            n_relpos_in = self.linear_relpos.weight.shape[-1]
            if relpos_feats.shape[-1] < n_relpos_in:
                pad_size = n_relpos_in - relpos_feats.shape[-1]
                relpos_feats = self._pad(relpos_feats, (0, pad_size))

        z = z + self.linear_relpos(relpos_feats)

        if batch is not None and "token_bonds" in batch:
            token_bonds_emb = self.linear_token_bonds(
                batch["token_bonds"].unsqueeze(-1).to(dtype=s.dtype)
            )
            z = z + token_bonds_emb

        return s_input, s, z
