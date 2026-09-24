"""Diffusion transformer for AlphaFold3 -- fused Triton implementation.

24-block transformer used inside the diffusion module. Each block:
AttentionPairBias + ConditionedTransitionBlock (AdaLN-Zero).

Reference: openfold3/core/model/layers/diffusion_transformer.py

The eager reference spends essentially all of its time in kernel-launch and
dispatch overhead: the captured shapes are tiny (16 tokens / 368 atoms), so the
~80 tensor ops per block are latency bound, not compute bound.  This candidate
keeps the module tree (and therefore the checkpoint keys) identical to the
reference and replaces the forward with

  * weight preprocessing done once: LayerNorm scales folded into the following
    linear, and q/k/v/gate (resp. the two SwiGLU halves) concatenated into a
    single matrix laid out [K, N] so a skinny GEMM reads it contiguously;
  * every activation derived only from ``s`` / ``z`` hoisted out of the block
    loop and evaluated for all blocks in one GEMM each (they do not depend on
    ``a``, which is the only thing the block loop mutates);
  * five fused Triton kernels per block (AdaLN+QKV/gate GEMM, attention,
    out-projection with the gated residual, AdaLN+SwiGLU GEMM, and the gated
    transition residual);
  * the whole stack captured into a CUDA graph, so a call costs one 32-byte
    address upload plus one graph launch instead of ~2000 dispatches.

Every intermediate is rounded to bf16 at exactly the points the reference rounds
(after each linear, after the AdaLN product, around the softmax, and at both
residual adds).  That is not cosmetic: keeping the residual stream in fp32
instead drifts far enough over 24 blocks to fail the 99%-within-tolerance check.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..L2.alphafold3_attention_pair_bias import AttentionPairBias, CrossAttentionPairBias
from ..L2.alphafold3_swiglu_transition import ConditionedTransitionBlock


__targets__ = ["DiffusionTransformer"]

try:
    import triton
    import triton.language as tl
    _HAVE_TRITON = True
except Exception:  # pragma: no cover - triton always present on the bench box
    _HAVE_TRITON = False


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------
if _HAVE_TRITON:

    # Programmatic dependent launch shaves a few percent off the launch latency
    # of this (very launch-bound) chain, but the early trigger let a dependent
    # grid observe stale data here -- even with a membar ahead of it -- so it is
    # left off; the hooks below are compiled out when it is.
    _PDL = False

    @triton.jit
    def _pdl_wait():
        tl.extra.cuda.gdc_wait()

    @triton.jit
    def _pdl_go():
        tl.extra.cuda.gdc_launch_dependents()

    @triton.jit
    def _k_copy_in(tab, da, ds, dz, dm,
                   NA: tl.constexpr, NS: tl.constexpr, NZ: tl.constexpr,
                   NM: tl.constexpr, G: tl.constexpr, BLK: tl.constexpr,
                   PDL: tl.constexpr):
        """Copy the four forward inputs into the graph's static buffers.

        The source addresses change from call to call (the harness hands out a
        fresh slot per iteration), so they are read from ``tab`` at run time
        instead of being baked into the launch -- that keeps the whole stack in
        one CUDA graph, with only an 8-word H2D copy per call.  Everything is
        moved as 64-bit words; each input is a whole number of them.
        """
        pid = tl.program_id(0)
        o = pid * BLK + tl.arange(0, BLK)
        if PDL:
            # must precede the table read: the addresses are written by the
            # host-side copy that this launch is allowed to overlap with.
            _pdl_wait()
        pa = tl.load(tab + 0).to(tl.pointer_type(tl.int64))
        ps = tl.load(tab + 1).to(tl.pointer_type(tl.int64))
        pz = tl.load(tab + 2).to(tl.pointer_type(tl.int64))
        pm = tl.load(tab + 3).to(tl.pointer_type(tl.int64))
        for i in range(0, NA, G * BLK):
            k = o + i
            tl.store(da + k, tl.load(pa + k, mask=k < NA, other=0), mask=k < NA)
        for i in range(0, NS, G * BLK):
            k = o + i
            tl.store(ds + k, tl.load(ps + k, mask=k < NS, other=0), mask=k < NS)
        for i in range(0, NM, G * BLK):
            k = o + i
            tl.store(dm + k, tl.load(pm + k, mask=k < NM, other=0), mask=k < NM)
        for i in range(0, NZ, G * BLK):
            k = o + i
            tl.store(dz + k, tl.load(pz + k, mask=k < NZ, other=0), mask=k < NZ)
        if PDL:
            _pdl_go()

    @triton.jit
    def _rowstats(x_ptr, rm, rok, SX: tl.constexpr, K: tl.constexpr,
                  BM: tl.constexpr, BK: tl.constexpr, MSK: tl.constexpr):
        """Per-row mean / reciprocal std over K.

        Single pass over the row: the inputs are bf16, so the fp32 sums are
        exact to ~1e-7 relative and E[x^2] - E[x]^2 is safe for activations with
        a small mean (which is what LayerNorm-fed residual streams have).
        """
        m1 = tl.zeros([BM], tl.float32)
        m2 = tl.zeros([BM], tl.float32)
        for k0 in range(0, K, BK):
            p = x_ptr + rm[:, None] * SX + (k0 + tl.arange(0, BK))[None, :]
            f = (tl.load(p, mask=rok[:, None], other=0.0) if MSK else tl.load(p)).to(tl.float32)
            m1 += tl.sum(f, 1)
            m2 += tl.sum(f * f, 1)
        mu = m1 / K
        return mu, 1.0 / tl.sqrt(tl.maximum(m2 / K - mu * mu, 0.0) + 1e-5)

    @triton.jit
    def _k_proj(x_ptr, w_ptr, b_ptr, o_ptr, o2_ptr, R,
                K: tl.constexpr, NC: tl.constexpr, NW: tl.constexpr,
                SX: tl.constexpr, SO: tl.constexpr, BM: tl.constexpr,
                BN: tl.constexpr, BK: tl.constexpr, DO_LN: tl.constexpr,
                HAS_B: tl.constexpr, CMB: tl.constexpr, MSK: tl.constexpr,
                TRANS: tl.constexpr, PDL: tl.constexpr):
        """Projection of an ``s``/``z`` derived tensor for every block at once.

        ``CMB`` selects the epilogue: 0 plain, 1 sigmoid, 2 the AdaLN gate pair
        (columns n and n+NC are the gate/shift halves -> sigmoid(g), sigmoid(g)*d).
        """
        pm = tl.program_id(0)
        pn = tl.program_id(1)
        rm = pm * BM + tl.arange(0, BM)
        rok = rm < R
        offn = pn * BN + tl.arange(0, BN)
        if PDL:
            _pdl_wait()
        if DO_LN:
            mu, rstd = _rowstats(x_ptr, rm, rok, SX, K, BM, BK, MSK)
        acc = tl.zeros([BM, BN], tl.float32)
        if CMB == 2:
            acc2 = tl.zeros([BM, BN], tl.float32)
        for k0 in range(0, K, BK):
            ok = k0 + tl.arange(0, BK)
            p = x_ptr + rm[:, None] * SX + ok[None, :]
            xv = tl.load(p, mask=rok[:, None], other=0.0) if MSK else tl.load(p)
            if DO_LN:
                xv = ((xv.to(tl.float32) - mu[:, None]) * rstd[:, None]).to(tl.bfloat16)
            wp = w_ptr + ok[:, None] * NW + offn[None, :]
            acc = tl.dot(xv, tl.load(wp), acc)
            if CMB == 2:
                acc2 = tl.dot(xv, tl.load(wp + NC), acc2)
        if HAS_B:
            acc += tl.load(b_ptr + offn)[None, :]
        if CMB == 1:
            acc = tl.sigmoid(acc)
        elif CMB == 2:
            acc = tl.sigmoid(acc).to(tl.bfloat16).to(tl.float32)
        if TRANS:
            tp = o_ptr + offn[:, None] * SO + rm[None, :]
            if MSK:
                tl.store(tp, tl.trans(acc).to(tl.bfloat16), mask=rok[None, :])
            else:
                tl.store(tp, tl.trans(acc).to(tl.bfloat16))
        else:
            op = o_ptr + rm[:, None] * SO + offn[None, :]
            if MSK:
                tl.store(op, acc.to(tl.bfloat16), mask=rok[:, None])
                if CMB == 2:
                    tl.store(o2_ptr + rm[:, None] * SO + offn[None, :],
                             acc2.to(tl.bfloat16), mask=rok[:, None])
            else:
                tl.store(op, acc.to(tl.bfloat16))
                if CMB == 2:
                    tl.store(o2_ptr + rm[:, None] * SO + offn[None, :],
                             acc2.to(tl.bfloat16))
        if PDL:
            _pdl_go()

    @triton.jit
    def _k_gemm_adaln(x_ptr, gm_ptr, ga_ptr, w_ptr, b_ptr, o_ptr, sti_ptr, stz_ptr,
                      acc_ptr, lk_ptr, R, GA, GB, NCOL_A,
                      K: tl.constexpr, NC: tl.constexpr, NW: tl.constexpr,
                      SX: tl.constexpr, SGD: tl.constexpr, SO: tl.constexpr,
                      BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                      HAS_B: tl.constexpr, DUAL: tl.constexpr, USE_ST: tl.constexpr,
                      MSK: tl.constexpr, SPK: tl.constexpr, NTOT: tl.constexpr,
                      PDL: tl.constexpr):
        """o = (G * LN(x) + Gd) @ w + b.

        ``DUAL`` fuses the SwiGLU: the two halves of the projection (columns n
        and n+NC) are combined as silu(lo) * hi before the store.  With
        ``SPK > 1`` the K range is split over programs which accumulate into
        ``acc_ptr``; the last one to arrive runs the epilogue (and clears the
        scratch for the next replay).
        """
        pm = tl.program_id(0)
        pn = tl.program_id(1)
        sid = tl.program_id(2)
        rm = pm * BM + tl.arange(0, BM)
        rok = rm < R
        n0 = pn * BN
        offn = n0 + tl.arange(0, BN)
        if PDL:
            _pdl_wait()
        if USE_ST:
            # row sums accumulated by whichever kernel last wrote x
            m1 = tl.load(sti_ptr + rm, mask=rok, other=0.0)
            m2 = tl.load(sti_ptr + R + rm, mask=rok, other=0.0)
            mu = m1 / K
            rstd = 1.0 / tl.sqrt(tl.maximum(m2 / K - mu * mu, 0.0) + 1e-5)
        else:
            mu, rstd = _rowstats(x_ptr, rm, rok, SX, K, BM, BK, MSK)
        if pn == 0 and sid == 0:
            tl.store(stz_ptr + rm, tl.zeros([BM], tl.float32), mask=rok)
            tl.store(stz_ptr + R + rm, tl.zeros([BM], tl.float32), mask=rok)
        goff = GA if n0 < NCOL_A else GB
        gbase = rm[:, None] * SGD + goff
        acc = tl.zeros([BM, BN], tl.float32)
        if DUAL:
            acc2 = tl.zeros([BM, BN], tl.float32)
        KS: tl.constexpr = K // SPK
        for k0 in range(sid * KS, (sid + 1) * KS, BK):
            ok = k0 + tl.arange(0, BK)
            if MSK:
                xv = tl.load(x_ptr + rm[:, None] * SX + ok[None, :], mask=rok[:, None], other=0.0)
                gv = tl.load(gm_ptr + gbase + ok[None, :], mask=rok[:, None], other=0.0)
                dv = tl.load(ga_ptr + gbase + ok[None, :], mask=rok[:, None], other=0.0)
            else:
                xv = tl.load(x_ptr + rm[:, None] * SX + ok[None, :])
                gv = tl.load(gm_ptr + gbase + ok[None, :])
                dv = tl.load(ga_ptr + gbase + ok[None, :])
            # matches g * (LN(x) + d) with a bf16 round after each reference op
            xh = (((xv.to(tl.float32) - mu[:, None]) * rstd[:, None]).to(tl.bfloat16)
                  .to(tl.float32) + dv.to(tl.float32)).to(tl.bfloat16)
            a1 = (gv.to(tl.float32) * xh.to(tl.float32)).to(tl.bfloat16)
            wp = w_ptr + ok[:, None] * NW + offn[None, :]
            acc = tl.dot(a1, tl.load(wp), acc)
            if DUAL:
                acc2 = tl.dot(a1, tl.load(wp + NC), acc2)
        last = True
        if SPK > 1:
            aoff = rm[:, None] * NTOT + offn[None, :]
            tl.atomic_add(acc_ptr + aoff, acc, mask=rok[:, None], sem="relaxed")
            if DUAL:
                tl.atomic_add(acc_ptr + R * NTOT + aoff, acc2, mask=rok[:, None], sem="relaxed")
            tok = tl.atomic_add(lk_ptr + pm * (NTOT // BN) + pn + tl.arange(0, 1), 1,
                                sem="acq_rel")
            last = tl.max(tok) == SPK - 1
            if last:
                acc = tl.load(acc_ptr + aoff, mask=rok[:, None], other=0.0, volatile=True)
                tl.store(acc_ptr + aoff, tl.zeros([BM, BN], tl.float32), mask=rok[:, None])
                if DUAL:
                    acc2 = tl.load(acc_ptr + R * NTOT + aoff, mask=rok[:, None],
                                   other=0.0, volatile=True)
                    tl.store(acc_ptr + R * NTOT + aoff, tl.zeros([BM, BN], tl.float32),
                             mask=rok[:, None])
                tl.store(lk_ptr + pm * (NTOT // BN) + pn + tl.arange(0, 1),
                         tl.zeros([1], tl.int32))
        if last:
            if HAS_B:
                acc += tl.load(b_ptr + offn)[None, :]
            if DUAL:
                acc = acc * tl.sigmoid(acc) * acc2
            op = o_ptr + rm[:, None] * SO + offn[None, :]
            if MSK:
                tl.store(op, acc.to(tl.bfloat16), mask=rok[:, None])
            else:
                tl.store(op, acc.to(tl.bfloat16))
        if PDL:
            _pdl_go()

    @triton.jit
    def _k_gemm_res(x_ptr, w_ptr, res_ptr, gd_ptr, m_ptr, sto_ptr, acc_ptr, lk_ptr,
                    R, GOFF, K: tl.constexpr, N: tl.constexpr, SX: tl.constexpr,
                    SGD: tl.constexpr, SR: tl.constexpr, BM: tl.constexpr,
                    BN: tl.constexpr, BK: tl.constexpr, APPLY_MASK: tl.constexpr,
                    MSK: tl.constexpr, SPK: tl.constexpr, PDL: tl.constexpr):
        """res += gate * (x @ w) [* mask]; ``gate`` is pre-sigmoided."""
        pm = tl.program_id(0)
        pn = tl.program_id(1)
        sid = tl.program_id(2)
        rm = pm * BM + tl.arange(0, BM)
        rok = rm < R
        offn = pn * BN + tl.arange(0, BN)
        if PDL:
            _pdl_wait()
        acc = tl.zeros([BM, BN], tl.float32)
        KS: tl.constexpr = K // SPK
        for k0 in range(sid * KS, (sid + 1) * KS, BK):
            ok = k0 + tl.arange(0, BK)
            p = x_ptr + rm[:, None] * SX + ok[None, :]
            xv = tl.load(p, mask=rok[:, None], other=0.0) if MSK else tl.load(p)
            acc = tl.dot(xv, tl.load(w_ptr + ok[:, None] * N + offn[None, :]), acc)
        last = True
        if SPK > 1:
            aoff = rm[:, None] * N + offn[None, :]
            tl.atomic_add(acc_ptr + aoff, acc, mask=rok[:, None], sem="relaxed")
            tok = tl.atomic_add(lk_ptr + pm * (N // BN) + pn + tl.arange(0, 1), 1,
                                sem="acq_rel")
            last = tl.max(tok) == SPK - 1
            if last:
                acc = tl.load(acc_ptr + aoff, mask=rok[:, None], other=0.0, volatile=True)
                tl.store(acc_ptr + aoff, tl.zeros([BM, BN], tl.float32), mask=rok[:, None])
                tl.store(lk_ptr + pm * (N // BN) + pn + tl.arange(0, 1),
                         tl.zeros([1], tl.int32))
        if last:
            gp = gd_ptr + rm[:, None] * SGD + GOFF + offn[None, :]
            rp = res_ptr + rm[:, None] * SR + offn[None, :]
            acc = acc.to(tl.bfloat16).to(tl.float32)
            if MSK:
                acc = acc * tl.load(gp, mask=rok[:, None], other=0.0).to(tl.float32)
            else:
                acc = acc * tl.load(gp).to(tl.float32)
            acc = acc.to(tl.bfloat16).to(tl.float32)
            if APPLY_MASK:
                acc = (acc * tl.load(m_ptr + rm, mask=rok, other=0.0)
                       .to(tl.float32)[:, None]).to(tl.bfloat16).to(tl.float32)
            if MSK:
                acc += tl.load(rp, mask=rok[:, None], other=0.0).to(tl.float32)
            else:
                acc += tl.load(rp).to(tl.float32)
            out = acc.to(tl.bfloat16)
            if MSK:
                tl.store(rp, out, mask=rok[:, None])
            else:
                tl.store(rp, out)
            f = out.to(tl.float32)
            tl.atomic_add(sto_ptr + rm, tl.sum(f, 1), mask=rok, sem="relaxed")
            tl.atomic_add(sto_ptr + R + rm, tl.sum(f * f, 1), mask=rok, sem="relaxed")
        if PDL:
            _pdl_go()

    @triton.jit
    def _k_attn_full(qkvg_ptr, zb_ptr, m_ptr, o_ptr, R, ZCOL, INF, QSC,
                     OK_: tl.constexpr, OV: tl.constexpr, OG: tl.constexpr,
                     D: tl.constexpr, DP: tl.constexpr,
                     SQ: tl.constexpr, SZB: tl.constexpr, SO: tl.constexpr,
                     BM: tl.constexpr, BKN: tl.constexpr, PDL: tl.constexpr):
        """Dense attention with pair bias, one program per (head, query block)."""
        h = tl.program_id(0)
        pm = tl.program_id(1)
        rq = pm * BM + tl.arange(0, BM)
        qok = rq < R
        kk = tl.arange(0, BKN)
        kok = kk < R
        od = tl.arange(0, DP)
        dok = od < D
        if PDL:
            _pdl_wait()
        q = tl.load(qkvg_ptr + rq[:, None] * SQ + h * D + od[None, :],
                    mask=qok[:, None] & dok[None, :], other=0.0)
        k = tl.load(qkvg_ptr + kk[:, None] * SQ + OK_ + h * D + od[None, :],
                    mask=kok[:, None] & dok[None, :], other=0.0)
        q = (q.to(tl.float32) * QSC).to(tl.bfloat16)
        sc = tl.dot(q, tl.trans(k)).to(tl.bfloat16).to(tl.float32)
        zb = tl.load(zb_ptr + (ZCOL + h) * SZB + rq[:, None] * R + kk[None, :],
                     mask=qok[:, None] & kok[None, :], other=0.0)
        mv = tl.load(m_ptr + kk, mask=kok, other=0.0).to(tl.float32)
        sc = (sc + INF * (mv - 1.0)[None, :]).to(tl.bfloat16).to(tl.float32)
        sc = (sc + zb.to(tl.float32)).to(tl.bfloat16).to(tl.float32)
        sc = tl.where(kok[None, :], sc, float("-inf"))
        p = tl.exp(sc - tl.max(sc, 1)[:, None])
        p = (p / tl.sum(p, 1)[:, None]).to(tl.bfloat16)
        v = tl.load(qkvg_ptr + kk[:, None] * SQ + OV + h * D + od[None, :],
                    mask=kok[:, None] & dok[None, :], other=0.0)
        o = tl.dot(p, v).to(tl.bfloat16).to(tl.float32)
        g = tl.load(qkvg_ptr + rq[:, None] * SQ + OG + h * D + od[None, :],
                    mask=qok[:, None] & dok[None, :], other=0.0).to(tl.float32)
        o = o * tl.sigmoid(g).to(tl.bfloat16).to(tl.float32)
        tl.store(o_ptr + rq[:, None] * SO + h * D + od[None, :], o.to(tl.bfloat16),
                 mask=qok[:, None] & dok[None, :])
        if PDL:
            _pdl_go()

    @triton.jit
    def _k_attn_local(qkvg_ptr, zb_ptr, m_ptr, o_ptr, R, ZCOL, INF, QSC,
                      OK_: tl.constexpr, OV: tl.constexpr, OG: tl.constexpr,
                      D: tl.constexpr, DP: tl.constexpr,
                      SQ: tl.constexpr, SZB: tl.constexpr, SO: tl.constexpr,
                      NQ: tl.constexpr, NK: tl.constexpr, RP: tl.constexpr,
                      PDL: tl.constexpr):
        """Sequence-local (blocked) attention with pair bias, per (block, head)."""
        j = tl.program_id(0)
        h = tl.program_id(1)
        od = tl.arange(0, DP)
        dok = od < D
        oq = tl.arange(0, NQ)
        ok = tl.arange(0, NK)
        if PDL:
            _pdl_wait()
        # number of real atoms (0/1 mask); decides where each key window sits
        rp = tl.arange(0, RP)
        nreal = tl.sum(tl.load(m_ptr + rp, mask=rp < R, other=0.0).to(tl.float32), 0).to(tl.int32)
        center = NQ // 2 + j * NQ
        init0 = center - NK // 2
        under = tl.maximum(-init0, 0)
        over = tl.maximum(center + NK // 2 - 1 - (nreal - 1), 0)
        fin = init0 + tl.where(under > 0, under, -over) + ok
        invalid = (fin < 0) | (fin >= nreal)
        safe = tl.minimum(tl.maximum(fin, 0), tl.maximum(nreal - 1, 0))
        rq = j * NQ + oq
        qok = rq < R
        mq = tl.load(m_ptr + rq, mask=qok, other=0.0).to(tl.float32)
        mk = tl.where(invalid, 0.0, tl.load(m_ptr + safe, mask=safe < R, other=0.0).to(tl.float32))
        q = tl.load(qkvg_ptr + rq[:, None] * SQ + h * D + od[None, :],
                    mask=qok[:, None] & dok[None, :], other=0.0)
        k = tl.load(qkvg_ptr + safe[:, None] * SQ + OK_ + h * D + od[None, :],
                    mask=dok[None, :], other=0.0)
        q = (q.to(tl.float32) * QSC).to(tl.bfloat16)
        sc = tl.dot(q, tl.trans(k)).to(tl.bfloat16).to(tl.float32)
        zb = tl.load(zb_ptr + (ZCOL + h) * SZB + (j * NQ + oq[:, None]) * NK + ok[None, :])
        sc = (sc + INF * (mq[:, None] * mk[None, :] - 1.0)).to(tl.bfloat16).to(tl.float32)
        sc = (sc + zb.to(tl.float32)).to(tl.bfloat16).to(tl.float32)
        p = tl.exp(sc - tl.max(sc, 1)[:, None])
        p = (p / tl.sum(p, 1)[:, None]).to(tl.bfloat16)
        v = tl.load(qkvg_ptr + safe[:, None] * SQ + OV + h * D + od[None, :],
                    mask=dok[None, :], other=0.0)
        o = tl.dot(p, v).to(tl.bfloat16).to(tl.float32)
        g = tl.load(qkvg_ptr + rq[:, None] * SQ + OG + h * D + od[None, :],
                    mask=qok[:, None] & dok[None, :], other=0.0).to(tl.float32)
        o = o * tl.sigmoid(g).to(tl.bfloat16).to(tl.float32)
        tl.store(o_ptr + rq[:, None] * SO + h * D + od[None, :], o.to(tl.bfloat16),
                 mask=qok[:, None] & dok[None, :])
        if PDL:
            _pdl_go()


def _p2(n: int) -> int:
    return 1 << max(0, (int(n) - 1).bit_length())


def _rup(n: int, m: int) -> int:
    return -(-int(n) // int(m)) * int(m)


# Launch geometry measured on the captured shapes (B200).  Keyed by
# (stage, rows, K, N); anything else falls back to the heuristic below.
_TUNED: dict = {
    # atom transformer (cross attention): 368 atoms, c_a=128, 3 blocks
    ("qkvg", 368, 128, 512): (16, 64, 128, 4, 2, 1),
    ("sw", 368, 128, 256): (16, 32, 16, 2, 4, 1),
    ("tout", 368, 256, 128): (32, 16, 256, 8, 4, 1),
    ("aout", 368, 128, 128): (32, 64, 64, 4, 4, 1),
    ("s1", 368, 128, 1152): (16, 64, 128, 8, 6, 1),
    ("z", 49152, 16, 16): (128, 16, 16, 2, 2, 1),
    ("attn", 368, 0, 0): (16, 0, 0, 4, 1, 1),
    # token transformer (self attention): 16 tokens, c_a=768, 24 blocks
    ("qkvg", 16, 768, 3072): (16, 128, 32, 2, 3, 6),
    ("sw", 16, 768, 1536): (16, 64, 16, 2, 3, 12),
    ("tout", 16, 1536, 768): (16, 32, 64, 2, 3, 8),
    ("s1", 16, 384, 36864): (16, 128, 32, 4, 4, 1),
}

_NSLOT = 64      # pinned staging ring for the per-call address table
_CFG_OVERRIDE: dict = {}


def _tune_cfg():
    """(BM, BN, BK, num_warps, num_stages, split_k) per stage tag.

    ``BM * BK`` is capped: the fused prologues keep a handful of [BM, BK] fp32
    tiles live, and spilling those costs far more than any tiling win.
    """
    def cfg(tag, rows, K, N):
        if tag in _CFG_OVERRIDE:
            return _CFG_OVERRIDE[tag]
        hit = _TUNED.get((tag, rows, K, N))
        if hit is not None:
            return hit if len(hit) == 6 else hit + (1,)
        if tag == "attn":
            return (16, 0, 0, 4, 2, 1)
        bk = min(128, _p2(K))
        while bk > 16 and K % bk:
            bk //= 2
        bm = min(_p2(rows), max(16, 2048 // bk))
        bn = 256
        nrb = -(-rows // bm)
        while bn > 16 and (bn > N or N % bn or nrb * (N // bn) < 192):
            bn //= 2
        return (bm, min(bn, N), bk, 4, 4, 1)
    return cfg


class _Plan:
    """Fused weights, scratch buffers and the captured graph for one shape."""

    __slots__ = ("graph", "out", "keep", "ready", "tab", "stage", "sview", "slot")

    def __init__(self):
        self.graph = None
        self.out = None
        self.keep = []
        self.ready = False
        self.tab = None
        self.stage = []
        self.sview = []
        self.slot = 0


class DiffusionTransformerBlock(nn.Module):
    """AF3 Algorithm 23: Diffusion transformer block.

    Args:
        c_a: Token activation channel dimension
        c_s: Single activation channel dimension
        c_z: Pair activation channel dimension
        c_hidden: Per-head hidden dimension
        no_heads: Number of attention heads
        n_transition: Transition layer scale
        use_ada_layer_norm: Whether to use AdaLN-Zero
        inf: Large masking constant
    """

    def __init__(
        self,
        c_a: int,
        c_s: int,
        c_z: int,
        c_hidden: int,
        no_heads: int,
        n_transition: int,
        use_ada_layer_norm: bool = True,
        n_query: int | None = None,
        n_key: int | None = None,
        inf: float = 1e9,
    ):
        super().__init__()
        self.use_cross_attention = n_query is not None

        if not self.use_cross_attention:
            self.attention_pair_bias = AttentionPairBias(
                c_q=c_a, c_k=c_a, c_v=c_a,
                c_s=c_s, c_z=c_z,
                c_hidden=c_hidden,
                no_heads=no_heads,
                use_ada_layer_norm=use_ada_layer_norm,
                gating=True,
                inf=inf,
            )
        else:
            self.attention_pair_bias = CrossAttentionPairBias(
                c_q=c_a, c_k=c_a, c_v=c_a,
                c_s=c_s, c_z=c_z,
                c_hidden=c_hidden,
                no_heads=no_heads,
                use_ada_layer_norm=use_ada_layer_norm,
                n_query=n_query,
                n_key=n_key,
                gating=True,
                inf=inf,
            )

        self.conditioned_transition = ConditionedTransitionBlock(
            c_a=c_a, c_s=c_s, n=n_transition,
        )

    def forward(
        self,
        a: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor | None = None,
        use_deepspeed_evo_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_lma: bool = False,
        use_high_precision_attention: bool = False,
        _mask_trans: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            a:    [*, N, C_token] token-level embedding
            s:    [*, N, C_s] single embedding
            z:    [*, N, N, C_z] pair embedding
            mask: [*, N] mask

        Returns:
            [*, N, C_token] updated token embedding
        """
        a = a + self.attention_pair_bias(a=a, z=z, s=s, mask=mask)

        trans_mask = mask if _mask_trans else None
        a = a + self.conditioned_transition(a=a, s=s, mask=trans_mask)

        return a


class DiffusionTransformer(nn.Module):
    """AF3 Algorithm 23: Diffusion transformer stack.

    Args:
        c_a: Token activation channel dimension
        c_s: Single activation channel dimension
        c_z: Pair activation channel dimension
        c_hidden: Per-head hidden dimension
        no_heads: Number of attention heads
        no_blocks: Number of transformer blocks
        n_transition: Transition layer scale
        use_ada_layer_norm: Whether to use AdaLN-Zero
        inf: Large masking constant
    """

    def __init__(
        self,
        c_a: int,
        c_s: int,
        c_z: int,
        c_hidden: int,
        no_heads: int,
        no_blocks: int,
        n_transition: int,
        use_ada_layer_norm: bool = True,
        n_query: int | None = None,
        n_key: int | None = None,
        inf: float = 1e9,
        blocks_per_ckpt: int | None = None,
        **kwargs,
    ):
        super().__init__()
        from ..L1.layer_norm import LayerNorm

        self.use_cross_attention = n_query is not None
        if self.use_cross_attention:
            self.layer_norm_z = LayerNorm(c_z, create_offset=False)

        self.blocks = nn.ModuleList([
            DiffusionTransformerBlock(
                c_a=c_a, c_s=c_s, c_z=c_z,
                c_hidden=c_hidden, no_heads=no_heads,
                n_transition=n_transition,
                use_ada_layer_norm=use_ada_layer_norm,
                n_query=n_query,
                n_key=n_key,
                inf=inf,
            )
            for _ in range(no_blocks)
        ])

        self.c_a = c_a
        self.c_s = c_s
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.no_blocks = no_blocks
        self.n_transition = n_transition
        self.use_ada_layer_norm = use_ada_layer_norm
        self.n_query = n_query
        self.n_key = n_key
        self.inf = inf
        self._plans: dict = {}
        self._fast_ok = _HAVE_TRITON and use_ada_layer_norm

    # -- reference path -----------------------------------------------------
    def _ref_forward(self, a, s, z, mask, _mask_trans=True):
        if self.use_cross_attention:
            z = self.layer_norm_z(z)
        for block in self.blocks:
            a = block(a=a, s=s, z=z, mask=mask, _mask_trans=_mask_trans)
        return a

    def forward(
        self,
        a: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor | None = None,
        use_deepspeed_evo_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_lma: bool = False,
        use_high_precision_attention: bool = False,
        _mask_trans: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            a:    [*, N, C_token] token-level embedding
            s:    [*, N, C_s] single embedding
            z:    [*, N, N, C_z] pair embedding
            mask: [*, N] mask

        Returns:
            [*, N, C_token] updated token embedding
        """
        plan = self._plans.get((a.shape, a.dtype, mask is None, _mask_trans))
        if (plan is not None and plan.ready and a.is_contiguous()
                and s.is_contiguous() and z.is_contiguous() and mask.is_contiguous()):
            i = plan.slot
            plan.slot = (i + 1) & (_NSLOT - 1)
            plan.sview[i][:] = (a.data_ptr(), s.data_ptr(), z.data_ptr(), mask.data_ptr())
            plan.tab.copy_(plan.stage[i], non_blocking=True)
            plan.graph.replay()
            return plan.out
        return self._slow_forward(a, s, z, mask, _mask_trans)

    # -- fast path setup ----------------------------------------------------
    def _slow_forward(self, a, s, z, mask, _mask_trans):
        key = (a.shape, a.dtype, mask is None, _mask_trans)
        if key in self._plans:                     # unsupported -> reference
            return self._ref_forward(a, s, z, mask, _mask_trans)
        if not (self._fast_ok and a.is_cuda and mask is not None and _mask_trans
                and a.dtype == torch.bfloat16 and s.dtype == torch.bfloat16
                and z.dtype == torch.bfloat16 and mask.dtype == torch.bfloat16
                and a.is_contiguous() and s.is_contiguous()
                and z.is_contiguous() and mask.is_contiguous()):
            self._plans[key] = _Plan()
            return self._ref_forward(a, s, z, mask, _mask_trans)
        try:
            plan = self._build(a, s, z, mask)
        except Exception:
            self._plans[key] = _Plan()
            return self._ref_forward(a, s, z, mask, _mask_trans)
        self._plans[key] = plan
        return self.forward(a, s, z, mask, _mask_trans=_mask_trans)

    def _build(self, a, s, z, mask):
        ca, cs, cz = self.c_a, self.c_s, self.c_z
        H, D, nb = self.no_heads, self.c_hidden, self.no_blocks
        nca = self.n_transition * ca
        dev = a.device
        R = a.shape[-2]
        if a.shape[-1] != ca or s.shape[-1] != cs or z.shape[-1] != cz:
            raise RuntimeError("unexpected channel dims")
        if a.numel() != R * ca or s.numel() != R * cs or mask.numel() != R:
            raise RuntimeError("batched inputs unsupported")
        if H * D != ca:
            raise RuntimeError("no_heads * c_hidden != c_a")

        cross = self.use_cross_attention
        if cross:
            nq, nk = self.n_query, self.n_key
            nblk = -(-R // nq)
            if z.numel() != nblk * nq * nk * cz:
                raise RuntimeError("unexpected z shape")
            Rz = nblk * nq * nk
            nad = 3                      # AdaLN instances per block: q, k, transition
        else:
            if z.numel() != R * R * cz:
                raise RuntimeError("unexpected z shape")
            Rz = R * R
            nq = nk = nblk = 0
            nad = 2                      # AdaLN instances per block: attn, transition

        plan = _Plan()
        keep = plan.keep
        f32 = torch.float32

        def hold(t):
            keep.append(t)
            return t

        def zeros(n):
            return torch.zeros(n, device=dev, dtype=f32)

        # ---- fused weights (LayerNorm scales folded in, [K, N] layout) ------
        wg, wd, bg, wr, br, wz = [], [], [], [], [], []
        Wqkvg, Bqkvg, Wao, Wsw, Wto = [], [], [], [], []
        zln = self.layer_norm_z.weight.float() if cross else None
        for blk in self.blocks:
            apb = blk.attention_pair_bias
            tr = blk.conditioned_transition
            adas = ([apb.layer_norm_a_q, apb.layer_norm_a_k, tr.layer_norm] if cross
                    else [apb.layer_norm_a, tr.layer_norm])
            for ada in adas:
                lnw = ada.layer_norm_s.weight.float()[None, :]
                wg.append((ada.linear_g.weight.float() * lnw).t())
                wd.append((ada.linear_s.weight.float() * lnw).t())
                bg.append(ada.linear_g.bias.float())
            for li in (apb.linear_ada_out, tr.linear_g):
                wr.append(li.weight.float().t())
                br.append(li.bias.float() if li.bias is not None else zeros(ca))
            zw = apb.linear_z.weight.float()
            zw = zw * (zln if cross else apb.layer_norm_z.weight.float())[None, :]
            wz.append(zw.t())
            mha = apb.mha
            order = ([mha.linear_q, mha.linear_g, mha.linear_k, mha.linear_v] if cross
                     else [mha.linear_q, mha.linear_k, mha.linear_v, mha.linear_g])
            Wqkvg.append(hold(torch.cat([li.weight.float() for li in order], 0)
                              .t().contiguous().bfloat16()))
            Bqkvg.append(hold(torch.cat([mha.linear_q.bias.float(), zeros(3 * ca)], 0)))
            Wao.append(hold(mha.linear_o.weight.float().t().contiguous().bfloat16()))
            Wsw.append(hold(torch.cat([tr.swiglu.linear_a.weight.float(),
                                       tr.swiglu.linear_b.weight.float()], 0)
                            .t().contiguous().bfloat16()))
            Wto.append(hold(tr.linear_out.weight.float().t().contiguous().bfloat16()))

        NG = nb * nad * ca
        Wl = hold(torch.cat(wg + wd, 1).contiguous().bfloat16())   # [cs, 2*NG]
        Bl = hold(torch.cat(bg, 0).contiguous())                   # [NG]
        NR = nb * 2 * ca
        Wr = hold(torch.cat(wr, 1).contiguous().bfloat16())        # [cs, NR]
        Br = hold(torch.cat(br, 0).contiguous())
        NZ = _rup(nb * H, 16)
        Wzt = torch.zeros(cz, NZ, device=dev, dtype=f32)
        for b in range(nb):
            Wzt[:, b * H:(b + 1) * H] = wz[b]
        Wzt = hold(Wzt.contiguous().bfloat16())

        # ---- scratch -------------------------------------------------------
        in_a = hold(torch.empty_like(a))
        in_s = hold(torch.empty_like(s))
        in_z = hold(torch.empty_like(z))
        in_m = hold(torch.empty_like(mask))
        va, vs, vz, vm = (in_a.view(R, ca), in_s.view(R, cs),
                          in_z.view(Rz, cz), in_m.view(R))
        emp = torch.empty
        gmul = hold(emp(R, NG, device=dev, dtype=torch.bfloat16))
        gadd = hold(emp(R, NG, device=dev, dtype=torch.bfloat16))
        gdr = hold(emp(R, NR, device=dev, dtype=torch.bfloat16))
        zb = hold(torch.zeros(NZ, Rz, device=dev, dtype=torch.bfloat16))
        qkvg = hold(emp(R, 4 * ca, device=dev, dtype=torch.bfloat16))
        ao = hold(emp(R, ca, device=dev, dtype=torch.bfloat16))
        hsw = hold(emp(R, nca, device=dev, dtype=torch.bfloat16))
        st = hold(torch.zeros(2, 2, R, device=dev, dtype=f32))    # (m1, m2) x2
        # split-K scratch: fp32 partial sums + one arrival counter per output tile
        spk_acc = hold(torch.zeros(4, 2 * R * max(4 * ca, nca), device=dev, dtype=f32))
        spk_lk = hold(torch.zeros(4, 8192, device=dev, dtype=torch.int32))
        plan.out = in_a
        plan.tab = hold(torch.zeros(4, dtype=torch.int64, device=dev))
        # a small ring of pinned staging words: the H2D copy is async, so a
        # single buffer could be rewritten while still in flight.
        for _ in range(_NSLOT):
            t = torch.zeros(4, dtype=torch.int64, pin_memory=True)
            plan.stage.append(t)
            plan.sview.append(t.numpy())
        nel = [in_a.numel() // 4, in_s.numel() // 4, in_z.numel() // 4, in_m.numel() // 4]
        if any(n * 4 != t.numel() for n, t in zip(nel, (in_a, in_s, in_z, in_m))):
            raise RuntimeError("inputs not a whole number of 64-bit words")
        vi64 = [t.view(torch.int64) if t.numel() % 4 == 0 else None
                for t in (in_a, in_s, in_z, in_m)]

        pdl = _PDL
        cfg = _tune_cfg()
        qsc = 1.0 / math.sqrt(D)

        def st_z():
            bm, bn, bk, nw, ns, _ = cfg("z", Rz, cz, NZ)
            _k_proj[(-(-Rz // bm), NZ // bn)](
                vz, Wzt, Wzt, zb, zb, Rz, K=cz, NC=NZ, NW=NZ, SX=cz, SO=Rz,
                BM=bm, BN=bn, BK=bk, DO_LN=True, HAS_B=False, CMB=0,
                MSK=(Rz % bm != 0), TRANS=True, PDL=pdl,
                num_warps=nw, num_stages=ns, launch_pdl=pdl)

        def st_s1():
            bm, bn, bk, nw, ns, _ = cfg("s1", R, cs, NG)
            _k_proj[(-(-R // bm), NG // bn)](
                vs, Wl, Bl, gmul, gadd, R, K=cs, NC=NG, NW=2 * NG, SX=cs, SO=NG,
                BM=bm, BN=bn, BK=bk, DO_LN=True, HAS_B=True, CMB=2,
                MSK=(R % bm != 0), TRANS=False, PDL=pdl,
                num_warps=nw, num_stages=ns, launch_pdl=pdl)

        def st_s2():
            bm, bn, bk, nw, ns, _ = cfg("s2", R, cs, NR)
            _k_proj[(-(-R // bm), NR // bn)](
                vs, Wr, Br, gdr, gdr, R, K=cs, NC=NR, NW=NR, SX=cs, SO=NR,
                BM=bm, BN=bn, BK=bk, DO_LN=False, HAS_B=True, CMB=1,
                MSK=(R % bm != 0), TRANS=False, PDL=pdl,
                num_warps=nw, num_stages=ns, launch_pdl=pdl)

        def st_qkvg(b, ga, gb2):
            bm, bn, bk, nw, ns, spk = cfg("qkvg", R, ca, 4 * ca)
            if b == 0:
                spk = 1                      # block 0 computes its own LN stats
            _k_gemm_adaln[(-(-R // bm), (4 * ca) // bn, spk)](
                va, gmul, gadd, Wqkvg[b], Bqkvg[b], qkvg, st[0], st[1],
                spk_acc[0], spk_lk[0], R, ga, gb2, 2 * ca,
                K=ca, NC=0, NW=4 * ca, SX=ca, SGD=NG, SO=4 * ca,
                BM=bm, BN=bn, BK=bk, HAS_B=True, DUAL=False, USE_ST=(b > 0),
                MSK=(R % bm != 0), SPK=spk, NTOT=4 * ca, PDL=pdl,
                num_warps=nw, num_stages=ns, launch_pdl=pdl)

        def st_attn(b):
            bm, bn, bk, nw, ns, _ = cfg("attn", R, 0, 0)
            if cross:
                _k_attn_local[(nblk, H)](
                    qkvg, zb, vm, ao, R, b * H, self.inf, qsc,
                    OK_=2 * ca, OV=3 * ca, OG=ca, D=D, DP=_p2(D),
                    SQ=4 * ca, SZB=Rz, SO=ca,
                    NQ=nq, NK=nk, RP=_p2(R), PDL=pdl,
                    num_warps=nw, num_stages=ns, launch_pdl=pdl)
            else:
                _k_attn_full[(H, -(-R // bm))](
                    qkvg, zb, vm, ao, R, b * H, self.inf, qsc,
                    OK_=ca, OV=2 * ca, OG=3 * ca, D=D, DP=_p2(D),
                    SQ=4 * ca, SZB=Rz, SO=ca,
                    BM=bm, BKN=_p2(R), PDL=pdl,
                    num_warps=nw, num_stages=ns, launch_pdl=pdl)

        def st_aout(b, gr):
            bm, bn, bk, nw, ns, spk = cfg("aout", R, ca, ca)
            _k_gemm_res[(-(-R // bm), ca // bn, spk)](
                ao, Wao[b], va, gdr, vm, st[1], spk_acc[1], spk_lk[1],
                R, gr, K=ca, N=ca, SX=ca, SGD=NR, SR=ca,
                BM=bm, BN=bn, BK=bk, APPLY_MASK=False, MSK=(R % bm != 0),
                SPK=spk, PDL=pdl, num_warps=nw, num_stages=ns, launch_pdl=pdl)

        def st_sw(b, gt):
            bm, bn, bk, nw, ns, spk = cfg("sw", R, ca, nca)
            _k_gemm_adaln[(-(-R // bm), nca // bn, spk)](
                va, gmul, gadd, Wsw[b], Wsw[b], hsw, st[1], st[0],
                spk_acc[2], spk_lk[2], R, gt, gt, nca,
                K=ca, NC=nca, NW=2 * nca, SX=ca, SGD=NG, SO=nca,
                BM=bm, BN=bn, BK=bk, HAS_B=False, DUAL=True, USE_ST=True,
                MSK=(R % bm != 0), SPK=spk, NTOT=nca, PDL=pdl,
                num_warps=nw, num_stages=ns, launch_pdl=pdl)

        def st_tout(b, gr):
            bm, bn, bk, nw, ns, spk = cfg("tout", R, nca, ca)
            _k_gemm_res[(-(-R // bm), ca // bn, spk)](
                hsw, Wto[b], va, gdr, vm, st[0], spk_acc[3], spk_lk[3],
                R, gr + ca, K=nca, N=ca, SX=nca,
                SGD=NR, SR=ca, BM=bm, BN=bn, BK=bk, APPLY_MASK=True,
                MSK=(R % bm != 0), SPK=spk, PDL=pdl,
                num_warps=nw, num_stages=ns, launch_pdl=pdl)

        def st_copy():
            blk = 512
            grid = 256
            _k_copy_in[(grid,)](
                plan.tab, vi64[0], vi64[1], vi64[2], vi64[3],
                NA=nel[0], NS=nel[1], NZ=nel[2], NM=nel[3], G=grid, BLK=blk,
                PDL=pdl, num_warps=4, num_stages=1, launch_pdl=pdl)

        def launch():
            st_copy()
            st_z()
            st_s1()
            st_s2()
            for b in range(nb):
                g0 = b * nad * ca
                gr = b * 2 * ca
                ga, gb2 = (g0, g0 + ca) if cross else (g0, g0)
                gt = g0 + (2 if cross else 1) * ca
                st_qkvg(b, ga, gb2)
                st_attn(b)
                st_aout(b, gr)
                st_sw(b, gt)
                st_tout(b, gr)

        # warm up (compiles every kernel) then capture
        plan.sview[0][:] = (a.data_ptr(), s.data_ptr(), z.data_ptr(), mask.data_ptr())
        plan.tab.copy_(plan.stage[0])
        launch()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            launch()
        plan.graph = g
        plan.ready = True
        return plan
