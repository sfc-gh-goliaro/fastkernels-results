"""Kimi Delta Attention -- fused rewrite.

The baseline is a faithful port of vLLM's KDA layer: six cuBLAS GEMMs, three
``causal_conv1d`` launches, two ``l2norm`` launches, the fused-gate cumsum and
then FLA's five-kernel chunked delta-rule pipeline, plus a boolean-mask state
gather whose ``nonzero`` forces a device sync every call.  Profiled at the
captured prefill widths that is ~20 kernels costing ~930 us of *host* time
against ~265 us of device time (n=64): the layer is launch-bound, not
compute-bound, so the win is in collapsing launches, not in tuning tiles.

Three paths, by prefill width:

* ``T <= 64`` (one chunk -- the hot captured widths) -- ``_kda_prep_kernel``
  (conv + silu + l2norm + beta + gate) then ``_kda_chunk1_kernel``, which is
  the chunk math *and* the whole epilogue.  With an empty incoming state one
  chunk collapses: ``v_new == u``, the inter-chunk output term vanishes and the
  final state is just ``kg^T u``, so ``w`` and ``q*exp2(g)`` never exist.
* ``T <= 512`` -- prep, ``_kda_chunk_kernel`` (per-chunk matrices and the UT
  transform) and ``_kda_scan_kernel`` (the serial state recurrence, both output
  terms, the gated RMS norm and the state writeback).
* wider -- same prep and chunk kernels, then FLA's ``fwd_h``/``fwd_o``, whose
  state scan is parallel over the value dimension where the fused scan is one
  program per head.  Past a few hundred tokens that occupancy wins.

Two pieces of the math are done differently from FLA:

* ``(I+A)^-1`` uses the binary decomposition
  ``(I-A)(I+A^2)(I+A^4)(I+A^8)(I+A^16)(I+A^32)``, which is exact for a
  strictly-lower-triangular 64x64 ``A``.  Ten 64x64 matmuls replace FLA's 56
  sequential rank-1 forward-substitution steps, whose kernel is the slowest in
  the baseline profile at 24 us.
* ``exp2(g_i - g_j)`` is factored per 16-row block rather than per chunk, so
  both factors only ever span a 16-row stretch of decay on the growing side.
  A chunk-wide reference overflows (or worse, hits ``0 * inf`` inside the dot)
  once the gate decays by ~128 log2 units across the chunk.

Anything the fast path does not cover -- decode batches, multi-sequence
prefill, a live incoming recurrent state, quantized projections, an unexpected
head geometry -- falls through to the baseline implementation unchanged.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ...baseline.L2.kimi_delta_attention import (
    KimiDeltaAttention as _BaselineKimiDeltaAttention,
)
from ..L1.rms_norm_gated import FusedRMSNormGated
from ..L1.kda import chunk_gla_fwd_o_gk
from ..L1.gated_delta_rule import chunk_gated_delta_rule_fwd_h

_CHUNK = 64
_RCP_LN2 = 1.4426950216
_BC = 16
# Past this width FLA's value-parallel state kernel beats a single program per
# head (measured crossover is between 443 and 1024 tokens).
_FUSED_SCAN_MAX_T = 512
_L2EPS = 1e-6
# Triton can only close over globals that are already ``tl.constexpr``.
_BCC = tl.constexpr(_BC)
_L2EPSC = tl.constexpr(_L2EPS)
_RCPC = tl.constexpr(_RCP_LN2)
_NW_PREP, _NW_CHUNK, _NW_SCAN = 8, 8, 8
_NS_PREP, _NS_SCAN = 2, 2


# ---------------------------------------------------------------------------
# Shared device functions.
# ---------------------------------------------------------------------------
@triton.jit
def _conv_silu(P, NIN, col, o_t, m_t, o_d, Wc, WCS,
               W: tl.constexpr, BT: tl.constexpr, D: tl.constexpr,
               L2NORM: tl.constexpr, DT: tl.constexpr):
    """Depthwise causal conv of width *W* over the token axis, then silu (and
    the l2 norm for q/k).  ``out[t] = sum_s w[c,s] * x[t - (W-1) + s]``; rows
    before the sequence start read as zero, which is what an empty conv state
    means."""
    acc = tl.zeros([BT, D], dtype=tl.float32)
    for s in tl.static_range(W):
        rr = o_t - (W - 1 - s)
        acc += tl.load(
            P + rr[:, None] * NIN + (col + o_d)[None, :],
            mask=m_t[:, None] & (rr >= 0)[:, None], other=0.0,
        ).to(tl.float32) * tl.load(Wc + s * WCS + col + o_d).to(tl.float32)[None, :]
    y = acc * tl.sigmoid(acc)
    # the baseline rounds the conv output to bf16 before l2norm reads it back
    y = y.to(DT).to(tl.float32)
    if L2NORM:
        y = y * tl.rsqrt(tl.sum(y * y, axis=1) + _L2EPSC)[:, None]
    return y


@triton.jit
def _gate_cumsum(rg, GAB, hoff, i_h, HD: tl.constexpr):
    """dt_bias, softplus, -exp(A_log), chunk-local cumsum, and into log2 units
    so the downstream exp2 reproduces exp(g)."""
    rgf = rg.to(tl.float32) + tl.load(GAB + hoff)[None, :]
    sp = tl.where(rgf > 20.0, rgf, tl.log(1.0 + tl.exp(rgf)))
    return tl.cumsum(-tl.load(GAB + HD + i_h) * sp, axis=0) * _RCPC


@triton.jit
def _kda_intra(b_q, b_k, b_g, b_b, G, goff, t0, T, scale,
               GS: tl.constexpr, BT: tl.constexpr, BC: tl.constexpr,
               D: tl.constexpr, DT: tl.constexpr):
    """Per-chunk ``(I+A)^-1``, ``Aqk`` and the last row of the gate."""
    o_i = tl.arange(0, BT)
    blk = o_i // BC
    g_last = tl.load(G + (tl.minimum(t0 + BT, T) - 1) * GS + goff)

    # Factor exp2(g_i - g_j) per BC-row block: rows divide out the gate at the
    # start of their own block, columns multiply it back.  Both factors then
    # only grow across a BC-row stretch of decay, so neither overflows nor
    # underflows into a 0 * inf inside the dot.
    gr = tl.load(G + tl.minimum(t0 + blk * BC, T - 1)[:, None] * GS
                 + goff[None, :])
    b_L = tl.exp2(b_g - gr)
    kL = (b_k * b_L).to(DT)
    qL = (b_q * b_L).to(DT)

    Mkk = tl.zeros([BT, BT], dtype=tl.float32)
    Mqk = tl.zeros([BT, BT], dtype=tl.float32)
    for c in tl.static_range(BT // BC):
        eR = tl.load(G + tl.minimum(t0 + c * BC, T - 1) * GS + goff)[None, :] - b_g
        Rb = (b_k * tl.exp2(tl.where((o_i < (c + 1) * BC)[:, None], eR, -1.0e30))
              ).to(DT)
        rowm = (blk == c)[:, None]
        Mkk = tl.where(rowm, tl.dot(kL, tl.trans(Rb)), Mkk)
        Mqk = tl.where(rowm, tl.dot(qL, tl.trans(Rb)), Mqk)

    b_A = tl.where(o_i[:, None] > o_i[None, :], Mkk * b_b[:, None], 0.0).to(DT)
    b_Aqk = tl.where(o_i[:, None] >= o_i[None, :], Mqk * scale, 0.0).to(DT)

    # (I+A)^-1 = (I-A)(I+A^2)(I+A^4)(I+A^8)(I+A^16)(I+A^32), exact for a
    # strictly lower triangular (hence nilpotent) A of order <= 64
    Ai = (tl.where(o_i[:, None] == o_i[None, :], 1.0, 0.0) - b_A).to(DT)
    Pw = tl.dot(b_A, b_A).to(DT)
    Ai = (Ai + tl.dot(Ai, Pw)).to(DT)
    for _ in tl.static_range(4):
        Pw = tl.dot(Pw, Pw).to(DT)
        Ai = (Ai + tl.dot(Ai, Pw)).to(DT)
    return Ai, b_Aqk, g_last


@triton.jit
def _store_conv_state(P, NIN, CS0, CS1, CS2, sidx, i_h, o_d, T,
                      W: tl.constexpr, HD: tl.constexpr, D: tl.constexpr):
    """The pre-activation tail of the sequence, laid out as the update kernel
    expects: ``state[slot, s, c] = x[T - (W-1) + s, c]``."""
    for s in tl.static_range(W - 1):
        row = T - (W - 1) + s
        for j in tl.static_range(3):
            col = j * HD + i_h * D
            xt = tl.load(P + row * NIN + col + o_d,
                         mask=(row >= 0) & (o_d < D), other=0.0)
            dst = sidx * ((W - 1) * HD) + s * HD + col + o_d
            if j == 0:
                tl.store(CS0 + dst, xt)
            elif j == 1:
                tl.store(CS1 + dst, xt)
            else:
                tl.store(CS2 + dst, xt)


# ---------------------------------------------------------------------------
# Multi-chunk prefill: prep -> per-chunk -> scan.
# ---------------------------------------------------------------------------
@triton.jit
def _kda_prep_kernel(
    P, RG,                   # raw gate rows (wide path)
    WFG, Wc, GAB,
    QKV,                     # [3+, T, H, D] bf16 (slot 3 = output gate)
    BETA,                    # [T, H] fp32
    G,                       # [T, H, D] fp32 -- cumulative gate, log2 units
    CS0, CS1, CS2, SIDX,
    T,
    NIN: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    W: tl.constexpr,
    BT: tl.constexpr,
    FUSE_FG: tl.constexpr,
):
    HD = H * D
    i_t, i_h, job = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    o_d = tl.arange(0, D)
    hoff = i_h * D + o_d
    dt = QKV.dtype.element_ty

    if job < 3:
        col = job * HD + i_h * D
        y = _conv_silu(P, NIN, col, o_t, m_t, o_d, Wc, 3 * HD,
                       W, BT, D, job < 2, dt)
        tl.store(QKV + job * T * H * D + o_t[:, None] * (H * D) + hoff[None, :],
                 y.to(dt), mask=m_t[:, None])
        if i_t == (T - 1) // BT:
            if job == 0:
                _store_conv_state(P, NIN, CS0, CS1, CS2,
                                  tl.load(SIDX).to(tl.int64), i_h, o_d, T, W, HD, D)
    else:
        tl.store(BETA + o_t * H + i_h,
                 tl.sigmoid(tl.load(P + o_t * NIN + 3 * HD + i_h, mask=m_t,
                                    other=0.0).to(tl.float32)), mask=m_t)
        aoff = 3 * HD + H
        if FUSE_FG:
            wsel = hoff[:, None] * D + o_d[None, :]
            rg = tl.dot(tl.load(P + o_t[:, None] * NIN + (aoff + o_d)[None, :],
                                mask=m_t[:, None], other=0.0),
                        tl.trans(tl.load(WFG + wsel)))
            tl.store(QKV + 3 * T * H * D + o_t[:, None] * (H * D) + hoff[None, :],
                     tl.dot(tl.load(P + o_t[:, None] * NIN
                                    + (aoff + D + o_d)[None, :],
                                    mask=m_t[:, None], other=0.0),
                            tl.trans(tl.load(WFG + H * D * D + wsel))).to(dt),
                     mask=m_t[:, None])
        else:
            rg = tl.load(RG + o_t[:, None] * HD + hoff[None, :],
                         mask=m_t[:, None], other=0.0)
        tl.store(G + o_t[:, None] * (H * D) + hoff[None, :],
                 _gate_cumsum(rg, GAB, hoff, i_h, HD), mask=m_t[:, None])


@triton.jit
def _kda_chunk_kernel(
    QKV, G, B,
    AK,                  # [NT, H, BT+D, BT] fused, or [T, H, BT] for FLA
    WT, UT, QG, KGO,     # [T, H, D]
    GL,                  # [NT, H, D] fp32, exp2(g_last)
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    FLA_OUT: tl.constexpr,
):
    i_t, i_h = tl.program_id(0), tl.program_id(1)
    t0 = i_t * BT
    o_i = tl.arange(0, BT)
    o_t = t0 + o_i
    m_t = o_t < T
    o_d = tl.arange(0, D)
    off = o_t[:, None] * (H * D) + (i_h * D + o_d)[None, :]
    qkvs = T * H * D          # q | k | v share one slab
    dt = WT.dtype.element_ty

    b_g = tl.load(G + off, mask=m_t[:, None], other=0.0)
    b_k = tl.load(QKV + qkvs + off, mask=m_t[:, None], other=0.0)
    b_b = tl.load(B + o_t * H + i_h, mask=m_t, other=0.0)
    scale = D ** -0.5
    Ai, b_Aqk, g_last = _kda_intra(
        tl.load(QKV + off, mask=m_t[:, None], other=0.0), b_k, b_g, b_b,
        G, i_h * D + o_d, t0, T, scale, H * D, BT, _BCC, D, dt)

    if FLA_OUT:
        tl.store(AK + (o_t[:, None] * H + i_h) * BT + o_i[None, :], b_Aqk,
                 mask=m_t[:, None])
    else:
        tl.store(AK + (i_t * H + i_h) * (BT + D) * BT
                 + o_i[:, None] * BT + o_i[None, :], b_Aqk)

    eg = tl.exp2(b_g)
    tl.store(UT + off,
             tl.dot(Ai, (tl.load(QKV + 2 * qkvs + off, mask=m_t[:, None], other=0.0)
                         * b_b[:, None]).to(dt)).to(dt), mask=m_t[:, None])
    tl.store(WT + off, tl.dot(Ai, (b_k * (b_b[:, None] * eg)).to(dt)).to(dt),
             mask=m_t[:, None])
    b_kg = (b_k * tl.exp2(g_last[None, :] - b_g)).to(dt)
    if FLA_OUT:
        tl.store(KGO + off, b_kg, mask=m_t[:, None])
    else:
        # kg transposed here so the serial scan reads [D, BT] straight from memory
        tl.store(AK + (i_t * H + i_h) * (BT + D) * BT
                 + (BT + o_d[:, None]) * BT + o_i[None, :], tl.trans(b_kg))
        tl.store(QG + off,
                 (tl.load(QKV + off, mask=m_t[:, None], other=0.0)
                  * (scale * eg)).to(dt), mask=m_t[:, None])
        tl.store(GL + (i_t * H + i_h) * D + o_d, tl.exp2(g_last))


@triton.jit
def _kda_chunk1_kernel(
    QKV, G, B, NW, O, HST, SIDX,
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    RMSEPS: tl.constexpr,
):
    """One-chunk prefill core.  With an empty incoming state a single chunk
    collapses: ``v_new == u``, the inter-chunk output term vanishes and the
    final state is just ``kg^T u``, so ``w`` and ``q*exp2(g)`` are never
    needed and the whole epilogue (both output terms, the gated RMS norm, the
    state writeback) fits in this launch."""
    i_h = tl.program_id(0)
    o_i = tl.arange(0, BT)
    m_t = o_i < T
    o_d = tl.arange(0, D)
    off = o_i[:, None] * (H * D) + (i_h * D + o_d)[None, :]
    qkvs = T * H * D
    dt = QKV.dtype.element_ty

    b_g = tl.load(G + off, mask=m_t[:, None], other=0.0)
    b_k = tl.load(QKV + qkvs + off, mask=m_t[:, None], other=0.0)
    b_b = tl.load(B + o_i * H + i_h, mask=m_t, other=0.0)
    Ai, b_Aqk, g_last = _kda_intra(
        tl.load(QKV + off, mask=m_t[:, None], other=0.0), b_k, b_g, b_b,
        G, i_h * D + o_d, 0, T, D ** -0.5, H * D, BT, _BCC, D, dt)

    b_u = tl.dot(Ai, (tl.load(QKV + 2 * qkvs + off, mask=m_t[:, None], other=0.0)
                      * b_b[:, None]).to(dt)).to(dt)
    b_o = tl.dot(b_Aqk, b_u)
    rstd = tl.rsqrt(tl.sum(b_o * b_o, axis=1) / D + RMSEPS)
    b_g2 = tl.load(QKV + 3 * qkvs + off, mask=m_t[:, None], other=0.0).to(
        tl.float32)
    tl.store(O + off,
             (b_o * rstd[:, None] * tl.load(NW + o_d).to(tl.float32)[None, :]
              * tl.sigmoid(b_g2)).to(O.dtype.element_ty), mask=m_t[:, None])
    tl.store(HST + (tl.load(SIDX).to(tl.int64) * H + i_h) * D * D
             + o_d[:, None] * D + o_d[None, :],
             tl.dot(tl.trans(b_u), (b_k * tl.exp2(g_last[None, :] - b_g)).to(dt)))


@triton.jit
def _kda_scan_kernel(
    AK, WT, UT, QG, GL, G2,
    NW, O, HST, SIDX,
    T, NT,
    H: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    RMSEPS: tl.constexpr,
):
    i_h = tl.program_id(0)
    o_i = tl.arange(0, BT)
    o_d = tl.arange(0, D)
    b_nw = tl.load(NW + o_d).to(tl.float32)
    dt = WT.dtype.element_ty
    b_S = tl.zeros([D, D], dtype=tl.float32)      # [k, v] -- no transpose needed

    for i_t in range(NT):
        o_t = i_t * BT + o_i
        m_t = o_t < T
        off = o_t[:, None] * (H * D) + (i_h * D + o_d)[None, :]
        akbase = AK + (i_t * H + i_h) * (BT + D) * BT
        Sb = b_S.to(dt)

        v_new = tl.load(UT + off, mask=m_t[:, None], other=0.0).to(tl.float32) \
            - tl.dot(tl.load(WT + off, mask=m_t[:, None], other=0.0), Sb)
        v_nb = v_new.to(dt)

        b_o = tl.dot(tl.load(QG + off, mask=m_t[:, None], other=0.0), Sb)
        b_o += tl.dot(tl.load(akbase + o_i[:, None] * BT + o_i[None, :]), v_nb)

        # gated RMS norm -- this program owns the whole head dim
        rstd = tl.rsqrt(tl.sum(b_o * b_o, axis=1) / D + RMSEPS)
        b_g2 = tl.load(G2 + o_t[:, None] * (H * D) + (i_h * D + o_d)[None, :],
                       mask=m_t[:, None], other=0.0).to(tl.float32)
        tl.store(O + off,
                 (b_o * rstd[:, None] * b_nw[None, :] * tl.sigmoid(b_g2)).to(
                     O.dtype.element_ty), mask=m_t[:, None])

        b_S *= tl.load(GL + (i_t * H + i_h) * D + o_d)[:, None]
        b_S += tl.dot(tl.load(akbase + (BT + o_d[:, None]) * BT + o_i[None, :]),
                      v_nb)

    sidx = tl.load(SIDX).to(tl.int64)
    tl.store(HST + (sidx * H + i_h) * D * D + o_d[:, None] * D + o_d[None, :],
             tl.trans(b_S))


class KimiDeltaAttention(_BaselineKimiDeltaAttention):
    """Fused KDA -- baseline parameters, baseline forward signature."""

    def __init__(self, config, layer_idx: int, quant_config: dict | None = None):
        super().__init__(config, layer_idx, quant_config)
        # the frozen L1 winner, not the baseline gated norm super() picked up
        self.o_norm = FusedRMSNormGated(
            self.head_dim, eps=config.rms_norm_eps, activation="sigmoid",
        )
        self._rms_eps = float(config.rms_norm_eps)
        self._fused = None

    def process_weights_after_loading(self) -> None:
        self._build_fused()

    def _build_fused(self) -> None:
        D, H = self.head_dim, self.local_num_heads
        HD = H * D
        W = self.conv_size
        if self.qkvb_proj is None or D not in (32, 64, 128, 256) or not 2 <= W <= 8:
            self._fused = False
            return
        w = self.qkvb_proj.weight
        dt = w.dtype
        self._w_in_t = torch.cat(
            [w, self.f_a_proj.weight.to(dt), self.g_a_proj.weight.to(dt)], 0
        ).contiguous().t()
        self._nin = self._w_in_t.shape[1]
        # lag-major so each conv tap is a contiguous [D] load
        self._w_conv = torch.cat(
            [self.q_conv1d.weight.reshape(HD, W),
             self.k_conv1d.weight.reshape(HD, W),
             self.v_conv1d.weight.reshape(HD, W)], 0,
        ).t().contiguous().to(dt)
        self._wfg = torch.cat(
            [self.f_b_proj.weight, self.g_b_proj.weight], 0).contiguous()
        self._wfb_t = self.f_b_proj.weight.t()
        self._wgb_t = self.g_b_proj.weight.t()
        self._gab = torch.cat(
            [self.dt_bias.detach().reshape(-1).float(),
             self.A_log.detach().reshape(-1).float().exp()]).contiguous()
        # skip RowParallelLinear.forward for the plain tp=1 bf16 case: its
        # nn.Module dispatch costs about as much host time as the GEMM launch
        self._wo_t = (self.o_proj.weight.t()
                      if (not self.o_proj.use_fp8 and self.o_proj.bias is None
                          and self.o_proj.tp_size == 1) else None)
        self._fused = True

    def forward(self, hidden_states: torch.Tensor, state_manager=None) -> torch.Tensor:
        del state_manager
        if self._fused is None:
            self._build_fused()
        if self._fused:
            out = self._forward_fused(hidden_states)
            if out is not None:
                return out
        return super().forward(hidden_states)

    def _out_proj(self, out, ntok: int):
        if out.shape[0] != ntok:
            full = out.new_zeros((ntok, out.shape[1]))
            full[:out.shape[0]] = out
            out = full
        if self._wo_t is not None:
            return torch.mm(out, self._wo_t)
        return self.o_proj(out)

    def _forward_fused(self, hs: torch.Tensor):
        state_view, meta = self._get_state()
        if state_view is None or meta is None:
            return None
        # one contiguous prefill starting at token 0, empty recurrent state
        if (meta.num_prefills != 1 or meta.num_decodes != 0
                or meta.any_have_initial_state):
            return None
        T, ntok = int(meta.num_actual_tokens), hs.shape[0]
        if T <= 0 or T > ntok:
            return None
        self._ensure_triton_allocator(hs.device)

        D, H = self.head_dim, self.local_num_heads
        HD = H * D
        dev, dt = hs.device, self._w_conv.dtype
        NT = triton.cdiv(T, _CHUNK)
        scale = D ** -0.5
        rst = state_view.recurrent_state
        sidx = meta.non_spec_state_indices_tensor
        P = torch.mm(hs[:T] if T != ntok else hs, self._w_in_t)
        out = torch.empty((T, HD), device=dev, dtype=dt)

        wide = T > _FUSED_SCAN_MAX_T
        aoff = 3 * HD + H
        qkv = torch.empty((3 if wide else 4, T, H, D), device=dev, dtype=dt)
        if wide:
            # a dedicated GEMM each: fusing f_b/g_b into the prep kernel doubles
            # its weight traffic, which only pays when the GEMM is launch-bound
            rg = torch.mm(P.narrow(1, aoff, D), self._wfb_t)
            g2 = torch.mm(P.narrow(1, aoff + D, D), self._wgb_t)
        else:
            rg = g2 = qkv[3]
        beta = torch.empty((T, H), device=dev, dtype=torch.float32)
        g = torch.empty((T, H, D), device=dev, dtype=torch.float32)
        _kda_prep_kernel[(NT, H, 4)](
            P, rg, self._wfg, self._w_conv, self._gab, qkv, beta, g,
            state_view.q_conv_state, state_view.k_conv_state,
            state_view.v_conv_state, sidx, T,
            NIN=self._nin, H=H, D=D, W=self.conv_size, BT=_CHUNK,
            FUSE_FG=not wide,
            num_warps=_NW_PREP, num_stages=_NS_PREP,
        )

        if NT == 1:
            _kda_chunk1_kernel[(H,)](
                qkv, g, beta, self.o_norm.weight, out, rst, sidx, T,
                H=H, D=D, BT=_CHUNK, RMSEPS=self._rms_eps, num_warps=_NW_CHUNK,
            )
            return self._out_proj(out, ntok)

        if not wide:
            ak = torch.empty((NT, H, _CHUNK + D, _CHUNK), device=dev, dtype=dt)
            wuq = torch.empty((3, T, H, D), device=dev, dtype=dt)
            gl = torch.empty((NT, H, D), device=dev, dtype=torch.float32)
            _kda_chunk_kernel[(NT, H)](
                qkv, g, beta, ak, wuq[0], wuq[1], wuq[2], wuq[2], gl, T,
                H=H, D=D, BT=_CHUNK, FLA_OUT=False, num_warps=_NW_CHUNK,
            )
            _kda_scan_kernel[(H,)](
                ak, wuq[0], wuq[1], wuq[2], gl, g2, self.o_norm.weight, out,
                rst, sidx, T, NT,
                H=H, D=D, BT=_CHUNK, RMSEPS=self._rms_eps,
                num_warps=_NW_SCAN, num_stages=_NS_SCAN,
            )
            return self._out_proj(out, ntok)

        # one kernel for what FLA spends four on (scaled_dot_kkt x2, solve_tril,
        # recompute_w_u), then FLA's value-parallel scan and output kernels
        Aqk = torch.empty((1, T, H, _CHUNK), device=dev, dtype=dt)
        wuk = torch.empty((3, T, H, D), device=dev, dtype=dt)
        _kda_chunk_kernel[(NT, H)](
            qkv, g, beta, Aqk, wuk[0], wuk[1], wuk[1], wuk[2], wuk[1], T,
            H=H, D=D, BT=_CHUNK, FLA_OUT=True, num_warps=_NW_CHUNK,
        )
        h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
            k=wuk[2][None], w=wuk[0][None], u=wuk[1][None], gk=g[None],
            initial_state=None, output_final_state=True, cu_seqlens=None,
            use_exp2=True,
        )
        chunk_gla_fwd_o_gk(
            q=qkv[0][None], v=v_new, g=g[None], A=Aqk, h=h, o=qkv[2][None],
            scale=scale, cu_seqlens=None, chunk_size=_CHUNK,
        )
        rst.index_copy_(0, sidx[:1].long(), final_state.to(rst.dtype))
        return self._out_proj(
            self.o_norm(qkv[2], g2.view(T, H, D)).view(T, HD), ntok)
