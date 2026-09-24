"""Gated linear attention (covers both GLA and RetNet).

The forward signature matches FLA's ``GatedLinearAttention.forward``
exactly so fastkernels kernels are drop-in for FLA users:

    forward(hidden_states, attention_mask=None,
            past_key_values=None, use_cache=False, **kwargs)
        -> (output, attentions, past_key_values)

Per the "Condense Variants" rule, this single class subsumes FLA's
``GatedLinearAttention`` (GLA, learned data-dependent gate) and
``MultiScaleRetention`` (RetNet, fixed-per-head decay + rotary).
The two architectures differ only in:

  * ``decay_mode``:
      - ``"learned_low_rank"`` (GLA): per-token, per-head, per-channel gk
        from a low-rank projection: ``gk = logsigmoid(W2(W1(x))) / norm``.
      - ``"fixed_per_head"`` (RetNet): data-independent gk[..., t, :] =
        log(gamma_h) for ``gamma_h = 1 - 2^(-5-h)``, broadcast across T.
  * ``use_rotary``: RetNet applies rotary to q/k; GLA does not.

Both feed into the SAME L1 recurrence kernel ``naive_recurrent_gla``
(RetNet is the constant-gk special case), and both finish with a per-head
RMSNorm + swish output gate. This consolidation keeps the L2 surface
small while preserving FLA's two distinct config knobs.

``nn.Sequential`` and ``nn.ModuleList`` are used here as pure-Python
*containers* over L1 ops (mirroring how every L4 model uses
``nn.ModuleList`` to hold L3 layers); the L2 "no torch.nn" rule applies
to *kernel* modules (Linear, LayerNorm, GroupNorm, activations) which we
unconditionally route through L1.

Performance notes (decode, T == 1, the dominant serving shape)
--------------------------------------------------------------
The reference decomposition issues ~13 kernels per token: four input
projections plus the two-``Linear`` ``gk_proj`` all re-reading the same
``hidden_states``, ``logsigmoid``, a scalar divide, the fused-recurrent
kernel (which itself split-Ks into an fp32 scratch, reduces it, and casts
back), an RMSNorm-weight cast, RMSNorm, SiLU and a multiply, then
``o_proj``.  At T == 1 that is entirely launch/Python bound.  This version
collapses the token path to three launches:

  1. one concatenated GEMM for ``q|k|v|g|gk_proj.0`` (the fused weight is a
     lazily-built buffer that the individual ``*_proj.weight`` params are
     *views* into, so eager writes -- ``load_state_dict``, in-place param
     updates -- stay visible with no per-call validation beyond a
     data-pointer identity check),
  2. ``_gla_decode_fwd_kernel``: the T == 1 recurrence *fused with* the
     gated-RMSNorm epilogue (RMSNorm(o) * silu(g)) in one Triton kernel,
     reading q/k/v/g straight out of the strided fused-GEMM output, and
  3. ``o_proj``.

Performance notes (chunked prefill, T >= 64)
-------------------------------------------
The reference routes prefill through the L1 ``ChunkGLA`` op (FLA's
``chunk_gla``), preceded by a ``logsigmoid`` + normalizer pass.  On the
captured ``[181, 1081, 2560]`` case that block is ~8.4 ms of the 15.9 ms
of device time.  It is replaced here by four own kernels -- gate+cumsum,
chunk-state scan, intra-chunk score block, output -- which cost ~4.8 ms.
The structure is deliberately FLA's (the chunk state does go through HBM;
keeping it on chip and using it as an MMA *operand* measures far slower on
Blackwell than the round trip); the wins are that the masked intra block is
one tf32 dot per K-block instead of two kernels including a per-column
elementwise loop, that ``BT`` is therefore free to be 128 so the state
tensor halves, and that the gate is one pass instead of three.  See
``ITERATIONS.md``.  Everything the four kernels do not cover -- packed
varlen, autograd, RetNet, launches too small to fill the GPU -- falls back
to the L1 op.

For T > 1 (and for the rotary / packed-varlen / CPU-fallback paths) the
original op composition is kept, with the RMSNorm+SiLU+multiply epilogue
routed through the fused L1 ``rms_norm_gated`` Triton kernel.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..L1.chunk_gla import ChunkGLA
from ..L1.chunk_retention import ChunkRetention
from ..L1.fused_recurrent_gla import FusedRecurrentGLA
from ..L1.fused_recurrent_retention import FusedRecurrentRetention
from ..L1.gla_recurrence import NaiveRecurrentGLA
from ..L1.linear import Linear
from ..L1.log_sigmoid import LogSigmoid
from ..L1.rms_norm import RMSNorm
from ..L1.rms_norm_gated import layer_norm_fwd
from ..L1.rotary_emb import RotaryEmbedding
from ..L1.silu import SiLU

try:  # Triton is present on every CUDA target; keep CPU import working.
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover - CPU-only environment
    _HAS_TRITON = False

# Threshold (matches FLA's own dispatch in fla.layers.rwkv7) — below this
# the chunk kernel's launch overhead exceeds its parallel speedup, so the
# fused-recurrent path is faster for short sequences (typical decode T=1).
_CHUNK_THRESHOLD = 64

# Largest head_v_dim the single-program decode kernel will take: the whole
# head must live in registers for the RMSNorm reduction to be fusable.
_MAX_FUSED_V = 1024


def _next_pow2(n: int) -> int:
    return 1 if n < 1 else 1 << (n - 1).bit_length()


# Tile shapes for the chunked-prefill kernels, measured on the scored
# [181, 1081, 5, 256, 512] case (see ITERATIONS.md).  ``_SMALL`` trades tile
# size for CTA count and is used when the default tiling would not fill the GPU.
_CHUNK_BT = 128
_CHUNK_CFG = {
    "gate": dict(BK=32, num_warps=4),
    "state": dict(BK=64, BV=256, num_warps=8, num_stages=2),
    "A": dict(BK=64, num_warps=4, num_stages=2),
    "o": dict(BK=32, BV=256, num_warps=8, num_stages=3),
}
_CHUNK_CFG_SMALL = {
    "gate": dict(BK=32, num_warps=4),
    "state": dict(BK=32, BV=64, num_warps=4, num_stages=2),
    "A": dict(BK=64, num_warps=4, num_stages=2),
    "o": dict(BK=32, BV=64, num_warps=4, num_stages=3),
}
# Minimum CTAs a launch must have before the chunk path is worth taking over
# the L1 op, whose autotuner picks much smaller tiles for tiny batches.
_MIN_CTAS = 128


if _HAS_TRITON:

    @triton.jit
    def _gla_decode_fwd_kernel(
        X,          # [N_ROW, F] fused q|k|v|g|gk_proj.0 GEMM output
        W2,         # gk_proj.1.weight [H*K, R]
        B2,         # gk_proj.1.bias   [H*K]
        LG,         # log_gamma [H] (fixed_per_head decay)
        NW,         # g_norm_swish_gate.weight [V]
        H0,         # initial recurrent state [N, H, K, V] fp32
        HT,         # final recurrent state   [N, H, K, V] fp32
        Y,          # [N_ROW, H*V] gated-normed output
        scale,      # 1/sqrt(K)
        gnorm_inv,  # 1 / gate_logit_normalizer
        eps,
        F: tl.constexpr,
        OFF_Q: tl.constexpr,
        OFF_K: tl.constexpr,
        OFF_V: tl.constexpr,
        OFF_G: tl.constexpr,
        OFF_L: tl.constexpr,
        H: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        R: tl.constexpr,
        BK: tl.constexpr,
        BV: tl.constexpr,
        BR: tl.constexpr,
        LOW_RANK: tl.constexpr,
        USE_H0: tl.constexpr,
        STORE_HT: tl.constexpr,
    ):
        """One program per (row, head) for a single-token step.

        With ``h`` the recurrent state, one GLA step is
        ``h' = h * exp(gk) + k v^T`` and ``o = q^T h'``, i.e.

            o[v] = sum_k q[k] exp(gk[k]) h[k, v]  +  (q . k) v[v]

        so the K reduction can be streamed in blocks -- no per-program
        [K, V] state residency, hence no split-K scratch and no separate
        reduction kernel.  ``exp(gk)`` only ever multiplies ``h``, so when
        there is no incoming state the gate is mathematically inert and the
        ``gk`` projection is skipped entirely (the ``USE_H0`` specialization);
        it is computed from the low-rank features in-kernel otherwise, which
        also removes the ``gk_proj.1`` GEMM from the token path.
        """
        i_bh = tl.program_id(0)
        i_b = i_bh // H
        i_h = i_bh % H

        o_v = tl.arange(0, BV)
        m_v = o_v < V
        x_row = X + i_b.to(tl.int64) * F
        b_v = tl.load(x_row + OFF_V + i_h * V + o_v, mask=m_v, other=0.0).to(tl.float32)
        b_g = tl.load(x_row + OFF_G + i_h * V + o_v, mask=m_v, other=0.0).to(tl.float32)

        if USE_H0 and LOW_RANK:
            o_r = tl.arange(0, BR)
            m_r = o_r < R
            b_low = tl.load(x_row + OFF_L + o_r, mask=m_r, other=0.0).to(tl.float32)
        if USE_H0 or STORE_HT:
            state_base = (i_b * H + i_h).to(tl.int64) * (K * V)

        acc = tl.zeros([BV], dtype=tl.float32)
        qk = tl.zeros([], dtype=tl.float32)
        for i_k in range(0, tl.cdiv(K, BK)):
            o_k = i_k * BK + tl.arange(0, BK)
            m_k = o_k < K
            b_q = tl.load(
                x_row + OFF_Q + i_h * K + o_k, mask=m_k, other=0.0
            ).to(tl.float32) * scale
            b_k = tl.load(x_row + OFF_K + i_h * K + o_k, mask=m_k, other=0.0).to(tl.float32)
            qk += tl.sum(b_q * b_k, axis=0)

            if USE_H0 or STORE_HT:
                p_h = state_base + o_k[:, None].to(tl.int64) * V + o_v[None, :]
                m_h = m_k[:, None] & m_v[None, :]
            if USE_H0:
                if LOW_RANK:
                    b_w2 = tl.load(
                        W2 + (i_h * K + o_k)[:, None] * R + o_r[None, :],
                        mask=m_k[:, None] & m_r[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    g_raw = tl.sum(b_w2 * b_low[None, :], axis=1) + tl.load(
                        B2 + i_h * K + o_k, mask=m_k, other=0.0
                    ).to(tl.float32)
                    # logsigmoid(x) = min(x, 0) - log1p(exp(-|x|)); then / norm.
                    b_e = tl.exp(
                        (tl.minimum(g_raw, 0.0)
                         - tl.log(1.0 + tl.exp(-tl.abs(g_raw)))) * gnorm_inv
                    )
                else:
                    b_e = tl.exp(tl.load(LG + i_h).to(tl.float32)) + tl.zeros([BK], tl.float32)
                b_h = tl.load(H0 + p_h, mask=m_h, other=0.0).to(tl.float32) * b_e[:, None]
                acc += tl.sum(b_h * b_q[:, None], axis=0)
            if STORE_HT:
                b_ht = b_k[:, None] * b_v[None, :]
                if USE_H0:
                    b_ht += b_h
                tl.store(HT + p_h, b_ht.to(HT.dtype.element_ty), mask=m_h)

        # Fused epilogue: per-head RMSNorm followed by the swish output gate.
        b_o = tl.where(m_v, acc + qk * b_v, 0.0)
        rstd = 1.0 / tl.sqrt(tl.sum(b_o * b_o, axis=0) / V + eps)
        b_w = tl.load(NW + o_v, mask=m_v, other=0.0).to(tl.float32)
        y = b_o * rstd * b_w * b_g * tl.sigmoid(b_g)
        tl.store(
            Y + i_b.to(tl.int64) * (H * V) + i_h * V + o_v,
            y.to(Y.dtype.element_ty),
            mask=m_v,
        )


    @triton.jit
    def _logsigmoid_scale_kernel(X, Y, n, inv, BLK: tl.constexpr):
        """``logsigmoid(x) / normalizer`` in one pass.

        The reference does this as ``F.logsigmoid(x)`` followed by a scalar
        divide -- two full passes over [B, T, key_dim].  Fused here, and in
        fp32 rather than the input dtype.
        """
        o = tl.program_id(0) * BLK + tl.arange(0, BLK)
        m = o < n
        x = tl.load(X + o, mask=m, other=0.0).to(tl.float32)
        y = (tl.minimum(x, 0.0) - tl.log(1.0 + tl.exp(-tl.abs(x)))) * inv
        tl.store(Y + o, y.to(Y.dtype.element_ty), mask=m)


    # ------------------------------------------------------------------
    # Chunked prefill: own four-kernel forward (see _chunk_own below)
    # ------------------------------------------------------------------
    # log2(e): the chunk path works in log2 space so every decay is an exp2.
    _RCP_LN2 = tl.constexpr(1.4426950408889634)

    @triton.jit
    def _gate_cumsum_kernel(
        GR, GC, ginv, T,
        H: tl.constexpr, K: tl.constexpr,
        BT: tl.constexpr, BK: tl.constexpr,
    ):
        """``cumsum_within_chunk(logsigmoid(gr) / normalizer) * log2(e)``.

        One pass over the raw gate logits.  The reference needs three
        (``logsigmoid``, the normalizer divide, then FLA's ``chunk_local_cumsum``)
        and materializes the intermediate; here only the fp32 cumsum is written.
        Everything downstream works in log2 space so the decay is ``exp2``.
        """
        i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1).to(tl.int64), tl.program_id(2)
        i_b, i_h = i_bh // H, i_bh % H
        o_t = i_t * BT + tl.arange(0, BT)
        o_k = i_k * BK + tl.arange(0, BK)
        m = (o_t < T)[:, None] & (o_k < K)[None, :]
        p = (i_b * T * H + i_h) * K + o_t[:, None] * (H * K) + o_k[None, :]

        b_g = tl.load(GR + p, mask=m, other=0.0)
        f = b_g.to(tl.float32)
        # The reference rounds the normalized gate back to the input dtype
        # before the cumsum, so round here too rather than carrying fp32.
        b_g = ((tl.minimum(f, 0.0) - tl.log(1.0 + tl.exp(-tl.abs(f)))) * ginv
               ).to(GR.dtype.element_ty)
        b_c = tl.cumsum(tl.where(m, b_g.to(tl.float32), 0.0), axis=0) * _RCP_LN2
        tl.store(GC + p, b_c, mask=m)


    @triton.jit
    def _chunk_state_kernel(
        Kt, Vt, GC, Ht, H0, HT, T,
        H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
        BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
        USE_H0: tl.constexpr, STORE_HT: tl.constexpr,
    ):
        """Sequential chunk-state scan; snapshots the state entering each chunk.

        ``h[c] = h[c-1] * 2^gl + sum_{s in chunk c-1} k[s] 2^(gl - gc[s]) v[s]^T``.
        The [BK, BV] accumulator stays in registers across the whole scan and is
        only ever a ``tl.dot`` *accumulator*, never an operand -- keeping it as an
        operand (i.e. fusing this with the output kernel) costs far more in
        layout conversions than the HBM round trip does.
        """
        i_k, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2).to(tl.int64)
        i_b, i_h = i_bh // H, i_bh % H
        NT = tl.cdiv(T, BT)

        o_k = i_k * BK + tl.arange(0, BK)
        o_v = i_v * BV + tl.arange(0, BV)
        o_i = tl.arange(0, BT)
        m_k = o_k < K
        m_v = o_v < V
        m_kv = m_k[:, None] & m_v[None, :]

        b_h = tl.zeros([BK, BV], dtype=tl.float32)
        if USE_H0:
            b_h = tl.load(H0 + i_bh * (K * V) + o_k[:, None] * V + o_v[None, :],
                          mask=m_kv, other=0.0).to(tl.float32)

        kb = (i_b * T * H + i_h) * K
        vb = (i_b * T * H + i_h) * V
        for i_t in range(0, NT):
            o_t = i_t * BT + o_i
            m_t = o_t < T
            tl.store(Ht + ((i_b * NT + i_t) * H + i_h).to(tl.int64) * (K * V)
                     + o_k[:, None] * V + o_v[None, :],
                     b_h.to(Ht.dtype.element_ty), mask=m_kv)

            p_kg = kb + o_k[:, None] + o_t[None, :] * (H * K)
            m_kt = m_k[:, None] & m_t[None, :]
            b_k = tl.load(Kt + p_kg, mask=m_kt, other=0.0)
            b_gc = tl.load(GC + p_kg, mask=m_kt, other=0.0)
            b_v = tl.load(Vt + vb + o_t[:, None] * (H * V) + o_v[None, :],
                          mask=m_t[:, None] & m_v[None, :], other=0.0)
            # The chunk's total decay is gc at its last *valid* token. Columns
            # past T load as 0, so BT-1 is only right for a full chunk -- using
            # it on a short trailing chunk would drop that chunk's decay from
            # the carried state (and hence from the returned final state).
            last = tl.minimum(BT, T - i_t * BT) - 1
            b_gl = tl.sum(tl.where(o_i[None, :] == last, b_gc, 0.0), axis=1)
            b_k = (b_k * tl.exp2(b_gl[:, None] - b_gc)).to(Kt.dtype.element_ty)
            b_h = b_h * tl.exp2(b_gl)[:, None] + tl.dot(b_k, b_v)

        if STORE_HT:
            tl.store(HT + i_bh * (K * V) + o_k[:, None] * V + o_v[None, :], b_h,
                     mask=m_kv)


    @triton.jit
    def _chunk_A_kernel(
        Q, Kt, GC, A, scale, T,
        H: tl.constexpr, K: tl.constexpr,
        BT: tl.constexpr, BK: tl.constexpr, NK: tl.constexpr,
    ):
        """Causally masked intra-chunk score block, one tf32 dot per K-block.

        ``A[t, s] = scale * sum_k q[t,k] k[s,k] 2^(gc[t,k] - gc[s,k])`` for
        ``s <= t``.  The reference spends two kernels here -- a sub-chunk-pair
        pass plus a 64-iteration elementwise loop for the diagonal block -- and
        keeps A in fp32; one masked dot per K-block covers the whole block, and
        A is written as bf16 because that is what the output kernel casts it to.

        The decay is referenced to the chunk midpoint so neither the q-side
        factor (``2^(gc-ref) <= 1``) nor the k-side one (``2^(ref-gc) >= 1``)
        spans more than half the chunk's dynamic range.  tf32 operands match the
        reference, whose intra dot also runs tf32 "to improve precision"; plain
        bf16 here measurably loses accuracy against the exact recurrence.
        """
        i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1)
        i_b, i_h = i_bh // H, i_bh % H
        o_t = i_t * BT + tl.arange(0, BT)
        o_i = tl.arange(0, BT)
        m_t = o_t < T
        base = (i_b * T * H + i_h) * K

        b_A = tl.zeros([BT, BT], dtype=tl.float32)
        for i_k in range(0, NK):
            o_k = i_k * BK + tl.arange(0, BK)
            m = m_t[:, None] & (o_k < K)[None, :]
            p = base + o_t[:, None] * (H * K) + o_k[None, :]
            b_gc = tl.load(GC + p, mask=m, other=0.0)
            last = tl.minimum(BT, T - i_t * BT) - 1
            b_ref = tl.sum(tl.where(o_i[:, None] == last, b_gc, 0.0), axis=0) * 0.5
            b_q = tl.load(Q + p, mask=m, other=0.0).to(tl.float32)
            b_k = tl.load(Kt + p, mask=m, other=0.0).to(tl.float32)
            b_A += tl.dot(b_q * tl.exp2(b_gc - b_ref[None, :]),
                          tl.trans(b_k * tl.exp2(b_ref[None, :] - b_gc)),
                          input_precision="tf32")

        b_A = tl.where(o_i[:, None] >= o_i[None, :], b_A * scale, 0.0)
        tl.store(A + (i_b * T * H + i_h) * BT + o_t[:, None] * (H * BT) + o_i[None, :],
                 b_A.to(A.dtype.element_ty), mask=m_t[:, None])


    @triton.jit
    def _chunk_o_kernel(
        Q, Vt, GC, Ht, A, O, scale, T,
        H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
        BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, NK: tl.constexpr,
    ):
        """``o = scale * (q * 2^gc) @ h  +  A @ v``."""
        i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1).to(tl.int64), tl.program_id(2)
        i_b, i_h = i_bh // H, i_bh % H
        NT = tl.cdiv(T, BT)
        dt = Q.dtype.element_ty

        o_t = i_t * BT + tl.arange(0, BT)
        o_v = i_v * BV + tl.arange(0, BV)
        o_i = tl.arange(0, BT)
        m_t = o_t < T
        m_v = o_v < V
        m_tv = m_t[:, None] & m_v[None, :]

        qb = (i_b * T * H + i_h) * K
        vb = (i_b * T * H + i_h) * V
        hb = ((i_b * NT + i_t) * H + i_h).to(tl.int64) * (K * V)

        b_o = tl.zeros([BT, BV], dtype=tl.float32)
        for i_k in range(0, NK):
            o_k = i_k * BK + tl.arange(0, BK)
            m_k = o_k < K
            p = qb + o_t[:, None] * (H * K) + o_k[None, :]
            m = m_t[:, None] & m_k[None, :]
            b_q = tl.load(Q + p, mask=m, other=0.0).to(tl.float32)
            b_gc = tl.load(GC + p, mask=m, other=0.0)
            b_h = tl.load(Ht + hb + o_k[:, None] * V + o_v[None, :],
                          mask=m_k[:, None] & m_v[None, :], other=0.0)
            b_o += tl.dot((b_q * tl.exp2(b_gc)).to(dt), b_h)
        b_o *= scale

        b_A = tl.load(A + (i_b * T * H + i_h) * BT
                      + o_t[:, None] * (H * BT) + o_i[None, :],
                      mask=m_t[:, None], other=0.0)
        b_v = tl.load(Vt + vb + o_t[:, None] * (H * V) + o_v[None, :],
                      mask=m_tv, other=0.0)
        b_o += tl.dot(b_A, b_v)
        tl.store(O + vb + o_t[:, None] * (H * V) + o_v[None, :],
                 b_o.to(O.dtype.element_ty), mask=m_tv)


class GatedLinearAttention(nn.Module):
    """Unified L2 attention for GLA and RetNet.

    Args:
        hidden_size: Model hidden size.
        num_heads: Number of attention heads.
        expand_k: Key expansion ratio (GLA: 0.5, RetNet: 1.0).
        expand_v: Value expansion ratio (GLA: 1.0, RetNet: 2.0).
        decay_mode: Which forget-gate mechanism to use.
        gate_low_rank_dim: Low-rank dim for the GLA gate (ignored for
            ``fixed_per_head``).
        gate_logit_normalizer: Normalizer applied after logsigmoid in the
            GLA gate (ignored for ``fixed_per_head``).
        use_rotary: Whether to apply rotary to q/k (RetNet uses this).
        rotary_base: Rotary base (theta).
        rotary_max_position: Max sequence length the rotary cache covers.
        norm_eps: RMSNorm epsilon for the per-head output norm.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        expand_k: float = 0.5,
        expand_v: float = 1.0,
        decay_mode: Literal["learned_low_rank", "fixed_per_head"] = "learned_low_rank",
        gate_low_rank_dim: int = 16,
        gate_logit_normalizer: int = 16,
        use_rotary: bool = False,
        rotary_base: float = 10000.0,
        rotary_max_position: int = 8192,
        norm_eps: float = 1e-6,
        use_fast_kernels: bool = True,
    ):
        super().__init__()
        assert decay_mode in ("learned_low_rank", "fixed_per_head"), (
            f"unknown decay_mode: {decay_mode!r}"
        )
        self.num_heads = num_heads
        self.decay_mode = decay_mode
        self.use_rotary = use_rotary
        self.gate_logit_normalizer = gate_logit_normalizer

        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads

        self.q_proj = Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = Linear(hidden_size, self.value_dim, bias=False)
        self.g_proj = Linear(hidden_size, self.value_dim, bias=False)
        self.o_proj = Linear(self.value_dim, hidden_size, bias=False)

        if decay_mode == "learned_low_rank":
            # FLA stores this as ``gk_proj = nn.Sequential(Linear, Linear)``
            # so the checkpoint paths are ``gk_proj.0.weight`` and
            # ``gk_proj.1.{weight,bias}``. nn.Sequential is used here purely
            # as a container; both children are L1 Linear ops.
            self.gk_proj = nn.Sequential(
                Linear(hidden_size, gate_low_rank_dim, bias=False),
                Linear(gate_low_rank_dim, self.key_dim, bias=True),
            )
            self.log_sigmoid = LogSigmoid()
        else:
            # RetNet: fixed per-head decay gamma_h = 1 - 2^(-5-h).
            # Stored as a non-persistent buffer so it auto-moves with the
            # module and is not written to checkpoints.
            h_idx = torch.arange(num_heads, dtype=torch.float32)
            gamma = 1.0 - torch.pow(torch.tensor(2.0, dtype=torch.float32), -5.0 - h_idx)
            log_gamma = torch.log(gamma)
            self.register_buffer("log_gamma", log_gamma, persistent=False)

        if use_rotary:
            self.rotary_emb = RotaryEmbedding(
                head_dim=self.head_k_dim,
                max_position_embeddings=rotary_max_position,
                rope_theta=rotary_base,
            )

        # Fast paths (Triton, FLA-vendored) + naive fallback (pure PyTorch).
        # The fast/slow choice is decided per-forward based on T and
        # ``use_fast_kernels``: chunk for prefill (T >= 64), fused-recurrent
        # for decode (T < 64). The naive path stays available for CPU
        # fallback / numerical reference.
        self.use_fast_kernels = use_fast_kernels
        self.naive_recurrence = NaiveRecurrentGLA()
        if use_fast_kernels:
            if decay_mode == "learned_low_rank":
                self.fused_recurrence = FusedRecurrentGLA()
                self.chunk = ChunkGLA()
            else:
                self.fused_recurrence = FusedRecurrentRetention()
                self.chunk = ChunkRetention()

        self.g_norm_swish_gate = RMSNorm(self.head_v_dim, eps=norm_eps)
        self.gate_act = SiLU()

        # --- fused-projection bookkeeping (shape-independent, precomputed) ---
        # Concatenated-GEMM row layout: q | k | v | g | gk_proj.0.
        proj = [self.q_proj, self.k_proj, self.v_proj, self.g_proj]
        if decay_mode == "learned_low_rank":
            proj.append(self.gk_proj[0])
        self._proj_mods = tuple(proj)
        splits = [m.weight.shape[0] for m in self._proj_mods]
        self._fused_out = sum(splits)
        self._off_q = 0
        self._off_k = splits[0]
        self._off_v = self._off_k + splits[1]
        self._off_g = self._off_v + splits[2]
        self._off_l = self._off_g + splits[3]
        self._fused_w = None
        self._fused_ptrs = None
        self._fuse_ok = True
        self._scale = self.head_k_dim ** -0.5
        # Decode kernel tiling (shape-independent).
        self._bv = _next_pow2(self.head_v_dim)
        self._low_rank_dim = gate_low_rank_dim if decay_mode == "learned_low_rank" else 1
        self._br = _next_pow2(self._low_rank_dim)
        self._bk_plain = min(_next_pow2(self.head_k_dim), 128)
        self._bk_state = min(_next_pow2(self.head_k_dim), 16)
        # Own chunked-prefill path: needs the raw gate logits (so the learned
        # low-rank gate only), a head layout the kernels can tile, and Triton.
        self._chunk_ok = (
            _HAS_TRITON
            and use_fast_kernels
            and decay_mode == "learned_low_rank"
            and self.head_k_dim * self.num_heads == self.key_dim
            and self.head_v_dim * self.num_heads == self.value_dim
        )
        self._decode_ok = (
            _HAS_TRITON
            and not use_rotary
            and use_fast_kernels
            and self._bv <= _MAX_FUSED_V
            and self.head_k_dim * self.num_heads == self.key_dim
            and self.head_v_dim * self.num_heads == self.value_dim
            and (decay_mode == "fixed_per_head" or self.gk_proj[1].bias is not None)
        )

    # ------------------------------------------------------------------
    # Fused input projection
    # ------------------------------------------------------------------
    def _fused_proj_weight(self) -> torch.Tensor | None:
        """The concatenated ``q|k|v|g|gk_proj.0`` weight, or None if the
        individual weights cannot be fused (mixed dtype/device).

        The concatenation is the *owning* storage and each ``*_proj.weight``
        is re-pointed at a slice of it, so every eager mutation route that
        writes through the parameter (``load_state_dict``, ``p.copy_()``,
        ``p.normal_()``, ``p.data.copy_()``) lands in the fused buffer with
        no copy and no per-call validation.  Routes that *replace* the
        storage (``p.data = ...``, ``module.to(dtype)``,
        ``nn.Parameter(...)``) are caught by the data-pointer check and
        trigger a rebuild.
        """
        mods = self._proj_mods
        ptrs = tuple(m.weight.data_ptr() for m in mods)
        if ptrs == self._fused_ptrs:
            return self._fused_w
        if not self._fuse_ok:
            return None
        try:
            with torch.no_grad():
                fused = torch.cat([m.weight.detach() for m in mods], dim=0).contiguous()
                off = 0
                for m in mods:
                    n = m.weight.shape[0]
                    m.weight.data = fused[off:off + n]
                    off += n
        except RuntimeError:  # mixed dtype / device: keep the unfused path
            self._fuse_ok = False
            self._fused_w = None
            self._fused_ptrs = None
            return None
        self._fused_w = fused
        self._fused_ptrs = tuple(m.weight.data_ptr() for m in mods)
        return fused

    # ------------------------------------------------------------------
    # Gate helpers
    # ------------------------------------------------------------------
    def _compute_gk(
        self, hidden_states: torch.Tensor, B: int, T: int,
    ) -> torch.Tensor:
        """Returns gk shaped [B, num_heads, T, head_k_dim] in log-space.

        Used by the naive recurrence path. The fast path uses
        :meth:`_compute_gk_bthk` to skip an unnecessary transpose.
        """
        if self.decay_mode == "learned_low_rank":
            gk = self.gk_proj(hidden_states)
            gk = self.log_sigmoid(gk) / self.gate_logit_normalizer
            return gk.view(B, T, self.num_heads, self.head_k_dim).transpose(1, 2)
        return self.log_gamma.to(hidden_states.dtype).view(
            1, self.num_heads, 1, 1
        ).expand(B, self.num_heads, T, self.head_k_dim)

    def _compute_gk_bthk(
        self, hidden_states: torch.Tensor, B: int, T: int,
    ) -> torch.Tensor:
        """Returns gk shaped [B, T, num_heads, head_k_dim] in log-space."""
        return self._scale_gk(self.gk_proj(hidden_states), B, T)

    def _scale_gk(self, gk: torch.Tensor, B: int, T: int) -> torch.Tensor:
        """``logsigmoid(gk) / normalizer``, reshaped to [B, T, H, head_k_dim]."""
        if _HAS_TRITON and gk.is_cuda:
            n = gk.numel()
            gk = gk.contiguous()
            _logsigmoid_scale_kernel[(triton.cdiv(n, 4096),)](
                gk, gk, n, 1.0 / self.gate_logit_normalizer, BLK=4096, num_warps=8)
        else:
            gk = self.log_sigmoid(gk) / self.gate_logit_normalizer
        return gk.view(B, T, self.num_heads, self.head_k_dim)

    def _gated_norm(self, o: torch.Tensor, g: torch.Tensor, rows: int) -> torch.Tensor:
        """``RMSNorm(o) * silu(g)`` per head, in one fused L1 Triton kernel.

        Replaces the RMSNorm + SiLU + multiply triple (plus the per-call
        norm-weight dtype cast the L1 RMSNorm module does).
        """
        w = self.g_norm_swish_gate.weight
        if w.dtype != o.dtype or w.device != o.device:
            w = w.to(device=o.device, dtype=o.dtype)
        y, _, _ = layer_norm_fwd(
            o.reshape(rows, self.head_v_dim),
            w,
            None,
            self.g_norm_swish_gate.eps,
            z=g.reshape(rows, self.head_v_dim),
            norm_before_gate=True,
            is_rms_norm=True,
            activation="swish",
        )
        return y

    # ------------------------------------------------------------------
    # Chunked prefill (T >= 64) fast path
    # ------------------------------------------------------------------
    def _chunk_cfg(self, B: int, T: int):
        """(BT, cfg) for this launch, or None when the L1 op should be used.

        The four kernels have exactly the L1 op's parallel structure (chunks are
        independent for A/o, and the state scan splits K and V), so there is no
        small-batch penalty *in kind* -- only in degree, because the L1 op's
        autotuner drops to much smaller tiles for tiny launches.  Rather than
        chase that, fall back whenever the grids here would not fill the GPU.
        """
        H, K, V = self.num_heads, self.head_k_dim, self.head_v_dim
        BT = _CHUNK_BT if T >= 4 * _CHUNK_BT else 64
        nbh = B * H
        for cfg in (_CHUNK_CFG, _CHUNK_CFG_SMALL):
            bks = min(cfg["state"]["BK"], _next_pow2(K))
            bvs = min(cfg["state"]["BV"], _next_pow2(V))
            bvo = min(cfg["o"]["BV"], _next_pow2(V))
            nt = -(-T // BT)
            ctas = min(
                nbh * -(-K // bks) * -(-V // bvs),   # state scan
                nbh * nt,                            # intra A
                nbh * nt * -(-V // bvo),             # output
            )
            if ctas >= _MIN_CTAS:
                return BT, cfg
        return None

    def _chunk_own(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        gr: torch.Tensor,
        initial_state: torch.Tensor | None,
        output_final_state: bool,
    ):
        """Own chunked GLA forward: gate+cumsum, state scan, intra A, output.

        Returns ``(o, final_state)`` in the L1 op's layout, or None when this
        path does not apply (caller then uses the L1 op).  ``gr`` is the *raw*
        ``gk_proj`` output -- the logsigmoid and the normalizer are folded into
        the cumsum kernel, so the gate never round-trips through HBM in its
        ungated form.
        """
        B, T, H, K = q.shape
        V = v.shape[-1]
        picked = self._chunk_cfg(B, T)
        if picked is None:
            return None
        BT, cfg = picked
        NT = -(-T // BT)
        scale = self._scale
        gr = gr.view(B, T, H, K)

        gc = q.new_empty(B, T, H, K, dtype=torch.float32)
        c = cfg["gate"]
        bk = min(c["BK"], _next_pow2(K))
        _gate_cumsum_kernel[(-(-K // bk), NT, B * H)](
            gr, gc, 1.0 / self.gate_logit_normalizer, T,
            H=H, K=K, BT=BT, BK=bk, num_warps=c["num_warps"])

        h = q.new_empty(B, NT, H, K, V)
        ht = q.new_empty(B, H, K, V, dtype=torch.float32) if output_final_state else None
        c = cfg["state"]
        bk = min(c["BK"], _next_pow2(K))
        bv = min(c["BV"], _next_pow2(V))
        _chunk_state_kernel[(-(-K // bk), -(-V // bv), B * H)](
            k, v, gc, h, initial_state, ht, T,
            H=H, K=K, V=V, BT=BT, BK=bk, BV=bv,
            USE_H0=initial_state is not None, STORE_HT=output_final_state,
            num_warps=c["num_warps"], num_stages=c["num_stages"])

        A = q.new_empty(B, T, H, BT)
        c = cfg["A"]
        bk = min(c["BK"], _next_pow2(K))
        _chunk_A_kernel[(NT, B * H)](
            q, k, gc, A, scale, T, H=H, K=K, BT=BT, BK=bk, NK=-(-K // bk),
            num_warps=c["num_warps"], num_stages=c["num_stages"])

        o = torch.empty_like(v)
        c = cfg["o"]
        bk = min(c["BK"], _next_pow2(K))
        bv = min(c["BV"], _next_pow2(V))
        _chunk_o_kernel[(-(-V // bv), NT, B * H)](
            q, v, gc, h, A, o, scale, T,
            H=H, K=K, V=V, BT=BT, BK=bk, BV=bv, NK=-(-K // bk),
            num_warps=c["num_warps"], num_stages=c["num_stages"])
        return o, ht

    def _chunk_own_ok(self, q: torch.Tensor, initial_state) -> bool:
        """Whether the own chunk path's assumptions hold for this call."""
        return (
            self._chunk_ok
            and q.is_cuda
            and not torch.is_grad_enabled()
            and q.dtype in (torch.bfloat16, torch.float16)
            and (initial_state is None
                 or (initial_state.dtype == torch.float32
                     and initial_state.is_contiguous()))
        )

    # ------------------------------------------------------------------
    # Decode (T == 1) fast path
    # ------------------------------------------------------------------
    def _decode_step(
        self,
        hidden_states: torch.Tensor,
        rows: int,
        initial_state: torch.Tensor | None,
        output_final_state: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Single-token step: one concatenated GEMM, one fused
        recurrence+gated-RMSNorm kernel, one ``o_proj``."""
        fused_w = self._fused_proj_weight()
        if fused_w is None:
            return None, None
        x = F.linear(hidden_states.reshape(rows, -1), fused_w)

        H, K, V = self.num_heads, self.head_k_dim, self.head_v_dim
        low_rank = self.decay_mode == "learned_low_rank"
        use_h0 = initial_state is not None
        if use_h0:
            initial_state = initial_state.contiguous()
        ht = (
            hidden_states.new_empty(rows, H, K, V, dtype=torch.float32)
            if output_final_state
            else None
        )
        y = hidden_states.new_empty(rows, H * V)
        _gla_decode_fwd_kernel[(rows * H,)](
            x,
            self.gk_proj[1].weight if low_rank else None,
            self.gk_proj[1].bias if low_rank else None,
            None if low_rank else self.log_gamma,
            self.g_norm_swish_gate.weight,
            initial_state,
            ht,
            y,
            self._scale,
            1.0 / self.gate_logit_normalizer,
            self.g_norm_swish_gate.eps,
            F=self._fused_out,
            OFF_Q=self._off_q,
            OFF_K=self._off_k,
            OFF_V=self._off_v,
            OFF_G=self._off_g,
            OFF_L=self._off_l,
            H=H,
            K=K,
            V=V,
            R=self._low_rank_dim,
            BK=self._bk_state if (use_h0 or output_final_state) else self._bk_plain,
            BV=self._bv,
            BR=self._br,
            LOW_RANK=low_rank,
            USE_H0=use_h0,
            STORE_HT=output_final_state,
            num_warps=4,
        )
        return y, ht

    def _finish(self, o, final_state, g, B, T, output_final_state, past_key_values):
        """Cache store + fused gated-RMSNorm epilogue + o_proj."""
        if output_final_state:
            if not hasattr(past_key_values, "states"):
                past_key_values.states = {}
            past_key_values.states[id(self)] = final_state
        o = self._gated_norm(o, g, B * T * self.num_heads)
        return self.o_proj(o.view(B, T, self.value_dim)), None, past_key_values

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        use_cache: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, None, object | None]:
        B, T, _ = hidden_states.shape
        cu_seqlens = kwargs.get("cu_seqlens")
        if cu_seqlens is not None and B != 1:
            raise ValueError("cu_seqlens prefill expects packed hidden_states with batch size 1")

        initial_state = None
        if past_key_values is not None and getattr(past_key_values, "states", None):
            initial_state = past_key_values.states.get(id(self))
        # The final state is only ever *read* back through ``past_key_values``;
        # with no cache to write it into, asking the recurrence for it is a pure
        # dead store (an [N, H, K, V] fp32 write, which at B=256 is the single
        # largest kernel in the token path).
        output_final_state = bool(use_cache) and past_key_values is not None

        # Dispatch length: with a packed batch every segment is <= T, so a
        # short packed batch needs no ``max()`` at all -- that spared
        # ``.item()`` is a full device sync on the decode hot path.
        if cu_seqlens is None or T < _CHUNK_THRESHOLD:
            dispatch_len = T
        else:
            lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            dispatch_len = int(lengths.max().item()) if lengths.numel() else 0

        # --- single-token fast path -------------------------------------
        if (
            T == 1
            and self._decode_ok
            and hidden_states.is_cuda
            and (cu_seqlens is None or cu_seqlens.numel() == 2)
        ):
            y, final_state = self._decode_step(
                hidden_states, B, initial_state, output_final_state)
            if y is not None:
                if output_final_state:
                    if not hasattr(past_key_values, "states"):
                        past_key_values.states = {}
                    past_key_values.states[id(self)] = final_state
                return self.o_proj(y).view(B, T, -1), None, past_key_values

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        g = self.g_proj(hidden_states)

        if self.use_rotary:
            # Build per-token absolute positions. For uncached single-shot
            # forward we use 0..T-1 per row. For cached prefill / decode the
            # engine passes ``past_key_values.seq_offsets`` (int or [B]
            # int64) giving the global position of token 0 in this call,
            # per row. Without that offset, RoPE would re-encode every
            # decode step at position 0 — totally breaking RetNet.
            #
            # NOTE: must materialize a contiguous int64 buffer with B*T real
            # elements. ``arange(T).expand(B, T).reshape(-1)`` returns a
            # stride-0 view (only T elements of storage); the CUDA RoPE
            # kernel does flat ``positions[token_idx]`` indexing which would
            # read out-of-bounds for token_idx >= T → illegal access.
            offsets = None
            if past_key_values is not None:
                offsets = getattr(past_key_values, "seq_offsets", None)
            if cu_seqlens is not None:
                # Packed varlen [1, total_T]: positions restart at each
                # sequence boundary. token t's position = its per-sequence local
                # index + that sequence's global start offset (seq_offsets, or
                # 0). This must be a flat [total_T] vector -- the dense branch
                # below builds [B*T], which is wrong for a packed batch and
                # feeds the RoPE kernel a positions length != query rows.
                lengths = cu_seqlens[1:] - cu_seqlens[:-1]
                seg_start = torch.repeat_interleave(cu_seqlens[:-1], lengths)
                positions = torch.arange(T, device=q.device, dtype=torch.int64) - seg_start
                if isinstance(offsets, int):
                    positions = positions + offsets
                elif offsets is not None:
                    positions = positions + torch.repeat_interleave(
                        offsets.to(device=q.device, dtype=torch.int64), lengths)
                positions = positions.contiguous()
            else:
                local = torch.arange(T, device=q.device, dtype=torch.int64)
                if offsets is None:
                    positions = local.repeat(B)
                elif isinstance(offsets, int):
                    positions = (local + offsets).repeat(B)
                else:
                    # [B] int64 tensor of per-row prefix lengths
                    positions = (offsets.to(device=q.device, dtype=torch.int64)
                                 .unsqueeze(1) + local.unsqueeze(0)).reshape(-1)
                    positions = positions.contiguous()
            q_flat = q.reshape(B * T, self.num_heads * self.head_k_dim).contiguous()
            k_flat = k.reshape(B * T, self.num_heads * self.head_k_dim).contiguous()
            q_flat, k_flat = self.rotary_emb(positions, q_flat, k_flat)
            q = q_flat.view(B, T, self.num_heads, self.head_k_dim)
            k = k_flat.view(B, T, self.num_heads, self.head_k_dim)
        else:
            q = q.view(B, T, self.num_heads, self.head_k_dim)
            k = k.view(B, T, self.num_heads, self.head_k_dim)

        v = v.view(B, T, self.num_heads, self.head_v_dim)

        # Dispatch:
        #   T >= 64 + fast kernels -> chunk (prefill / training)
        #   T  < 64 + fast kernels -> fused_recurrent (decode)
        #   no fast kernels         -> naive PyTorch (CPU / debug / reference)
        fast = self.use_fast_kernels and q.is_cuda
        if fast:
            if self.decay_mode == "learned_low_rank":
                gk_raw = self.gk_proj(hidden_states)
                own = None
                if (dispatch_len >= _CHUNK_THRESHOLD
                        and cu_seqlens is None
                        and self._chunk_own_ok(q, initial_state)):
                    own = self._chunk_own(
                        q, k, v, gk_raw, initial_state, output_final_state)
                if own is not None:
                    return self._finish(*own, g, B, T,
                                        output_final_state, past_key_values)
                # gk in [B, T, H, K] log-space, NOT transposed
                gk_btHK = self._scale_gk(gk_raw, B, T)
                if dispatch_len >= _CHUNK_THRESHOLD:
                    o, final_state = self.chunk(
                        q=q, k=k, v=v, g=gk_btHK,
                        initial_state=initial_state,
                        output_final_state=output_final_state,
                        cu_seqlens=cu_seqlens,
                    )
                else:
                    o, final_state = self.fused_recurrence(
                        q=q, k=k, v=v, gk=gk_btHK,
                        initial_state=initial_state,
                        output_final_state=output_final_state,
                        cu_seqlens=cu_seqlens,
                    )
            else:  # RetNet — kernel bakes in the per-head decay
                if dispatch_len >= _CHUNK_THRESHOLD:
                    o, final_state = self.chunk(
                        q=q, k=k, v=v,
                        initial_state=initial_state,
                        output_final_state=output_final_state,
                        cu_seqlens=cu_seqlens,
                    )
                else:
                    o, final_state = self.fused_recurrence(
                        q=q, k=k, v=v,
                        initial_state=initial_state,
                        output_final_state=output_final_state,
                        cu_seqlens=cu_seqlens,
                    )
            # Fast-path output is already [B, T, H, V] — no transpose needed.
        else:
            # Naive path expects [B, H, T, D]
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            gk = self._compute_gk(hidden_states, B, T)
            o, final_state = self.naive_recurrence(
                q, k, v, gk,
                initial_state=initial_state,
                output_final_state=output_final_state,
            )
            o = o.transpose(1, 2)  # [B, H, T, V] -> [B, T, H, V]

        if output_final_state:
            if not hasattr(past_key_values, "states"):
                past_key_values.states = {}
            past_key_values.states[id(self)] = final_state

        if fast:
            o = self._gated_norm(o, g, B * T * self.num_heads)
        else:
            o = self.g_norm_swish_gate(o.reshape(-1, self.head_v_dim))
            o = o.view(B, T, self.value_dim)
            o = o * self.gate_act(g)

        return self.o_proj(o.view(B, T, self.value_dim)), None, past_key_values
