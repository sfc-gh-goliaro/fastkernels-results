"""Kimi Delta Attention -- latency-optimized candidate.

The baseline is dominated by two things that have nothing to do with the math.

1. **Host-side launch overhead.** Its forward is ~43 kernel launches, most
   through Triton's Python launch path, so every captured shape from 1 to 443
   tokens measures ~1.0 ms end to end while the GPU only has ~140 us of work.
   Two of those launches -- ``state[~has_initial] = 0`` and the ``numel()`` that
   guards it -- are boolean-mask index ops, i.e. a ``nonzero`` plus a
   device->host copy, so the host also blocks mid-forward twice per call.
2. **Redundant passes and mask-shaped matmuls.** Three separate depthwise convs
   (one per projection, each reading a strided slice of the fused qkvb GEMM)
   plus two l2-norm passes move the 12288-channel activation across HBM five
   times; the gate cumsum runs a 64x64 fp32 ``tl.dot`` against a triangular mask
   instead of a scan; ``solve_tril`` forward-substitutes through ~56 serialized
   global loads; and the kkt kernel that fills A's diagonal blocks walks its 16
   columns one at a time, re-exponentiating a whole [16, K] tile per column.

What this file does about it:

* the mask-index state zeroing is replaced by the host-side summary the metadata
  already carries (``any_/all_have_initial_state``), and the chunk index/offset
  tables are pre-computed per token count, so the forward never touches the
  host;
* conv1d + silu + per-head l2-norm for q, k and v, the conv-state update and the
  beta sigmoid collapse into **one** kernel over a single read of the fused GEMM;
* the gate cumsum is rewritten around ``tl.cumsum``;
* A, Aqk and ``(I + A)^-1`` come out of one kernel that keeps all three in
  registers, built from a pair decomposition and a block-inverse identity that
  both run over the same ``log2(BT)`` power-of-two levels (see
  ``_kkt_solve_kernel`` / ``_single_chunk_kernel``);
* for a single chunk with no carried-in state the whole delta-rule core --
  A through output and final state -- is one kernel, because ``h`` is
  identically zero there;
* the qkvb projection and the two rank-128 gate down-projections are merged into
  one GEMM, and the ``zeros`` + slice-assign staging of ``core_attn_out`` is
  gone;
* and a CUDA graph is cached per token count, which removes what host overhead
  is left.

Anything that does not match the fast path's preconditions (quantized weights,
decode tokens, more than one prefill sequence, a live initial state, an
unexpected head dim) falls through to ``super().forward`` unchanged.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from fastkernels.tasks.baseline.L2.kimi_delta_attention import (
    KimiDeltaAttention as _BaselineKDA,
)
from ..L1.kda import (
    RCP_LN2,
    chunk_gla_fwd_o_gk,
    recompute_w_u_fwd_kda,
)
from ..L1.gated_delta_rule import (
    FLA_CHUNK_SIZE,
    chunk_gated_delta_rule_fwd_h,
)

_L2_EPS = 1e-6
_MAX_GRAPHS = 8
_GRAPH_MAX_TOKENS = 4096
_SOFTPLUS_THRESHOLD = 20.0
_TRIL_PRECISION = "tf32"
_KKT_WARPS = 8
_NO_GRAPH = object()
_FUSED_MAX_TOKENS = 64


# ---------------------------------------------------------------------------
# conv1d + silu + per-head l2-norm (+ conv-state update + beta sigmoid).
#
# ``X`` is the raw ``[T, 3*P + H]`` qkvb GEMM output; columns ``[0, 3P)`` hold
# q|k|v back to back, so one kernel covers all three convs. Program
# (i_t, i_p, i_h) owns ``BT`` tokens of head ``i_h`` of projection ``i_p``: it
# accumulates the ``WIDTH``-tap causal convolution in fp32 (rounding each
# product to bf16 so the arithmetic matches ``causal_conv1d``'s bf16 multiply),
# applies silu, and for q/k divides by the per-token, per-head L2 norm before
# storing into the packed ``[3, T, P]`` output.
#
# Two unrelated bits of bookkeeping ride along rather than pay for their own
# launch: the programs on the last token block write the trailing ``WIDTH-1``
# *pre-activation* tokens to the conv-state cache (what ``causal_conv1d_fn``
# leaves behind: right-aligned, zero-padded), and the ``i_p == 0`` programs turn
# the strided beta tail of the same GEMM into ``sigmoid(fp32(.))``.
# ---------------------------------------------------------------------------
@triton.jit
def _conv_silu_l2_kernel(
    X, OUT, W, SQ, SK, SV, IDX, BETA,
    T,
    stride_x,
    stride_o_p,
    i_t_last,
    D: tl.constexpr,
    P: tl.constexpr,
    C3: tl.constexpr,
    NH: tl.constexpr,
    WIDTH: tl.constexpr,
    BT: tl.constexpr,
    EPS: tl.constexpr,
):
    i_t = tl.program_id(0)
    i_p = tl.program_id(1)
    i_h = tl.program_id(2)

    o_d = tl.arange(0, D)
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    c = i_p * P + i_h * D + o_d

    acc = tl.zeros((BT, D), dtype=tl.float32)
    for j in tl.static_range(WIDTH):
        b_w = tl.load(W + j * C3 + c)
        tt = o_t - (WIDTH - 1) + j
        m = m_t & (tt >= 0)
        b_x = tl.load(X + tt[:, None] * stride_x + c[None, :], mask=m[:, None],
                      other=0.0)
        acc += (b_x * b_w[None, :]).to(tl.bfloat16).to(tl.float32)

    acc = acc / (1.0 + tl.exp(-acc))
    b_y = acc.to(tl.bfloat16)
    if i_p < 2:
        b_f = b_y.to(tl.float32)
        b_var = tl.sum(b_f * b_f, axis=1)
        b_y = (b_f * tl.rsqrt(b_var + EPS)[:, None]).to(tl.bfloat16)

    tl.store(OUT + i_p * stride_o_p + o_t[:, None] * P + (i_h * D + o_d)[None, :],
             b_y, mask=m_t[:, None])

    if i_p == 0:
        b_b = tl.load(X + o_t * stride_x + C3 + i_h, mask=m_t, other=0.0)
        tl.store(BETA + o_t * NH + i_h,
                 1.0 / (1.0 + tl.exp(-b_b.to(tl.float32))), mask=m_t)

    if i_t == i_t_last:
        slot = tl.load(IDX).to(tl.int64)
        if slot >= 0:
            for j in tl.static_range(WIDTH - 1):
                ts = T - (WIDTH - 1) + j
                b_s = tl.load(X + ts * stride_x + c,
                              mask=(o_d >= 0) & (ts >= 0), other=0.0)
                off = slot * ((WIDTH - 1) * P) + j * P + i_h * D + o_d
                if i_p == 0:
                    tl.store(SQ + off, b_s)
                elif i_p == 1:
                    tl.store(SK + off, b_s)
                else:
                    tl.store(SV + off, b_s)


# ---------------------------------------------------------------------------
# Chunk-local cumulative gate. Same math as ``kda_gate_cumsum_fwd_kernel``
# (softplus -> -exp(A_log) scaling -> chunk-local prefix sum, pre-scaled by
# 1/ln2 for the exp2-based consumers) with the triangular-mask ``tl.dot``
# replaced by ``tl.cumsum``.
# ---------------------------------------------------------------------------
@triton.jit
def _gate_cumsum_kernel(
    G, Y, A_LOG, BIAS,
    T,
    scale,
    threshold,
    H: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    i_d = tl.program_id(0)
    i_t = tl.program_id(1)
    i_h = tl.program_id(2)

    o_t = i_t * BT + tl.arange(0, BT)
    o_d = i_d * BD + tl.arange(0, BD)
    m_t = o_t < T
    m_d = o_d < D
    base = o_t[:, None] * (H * D) + i_h * D + o_d[None, :]

    b_g = tl.load(G + base, mask=m_t[:, None] & m_d[None, :], other=0.0).to(tl.float32)
    b_g += tl.load(BIAS + i_h * D + o_d, mask=m_d, other=0.0).to(tl.float32)[None, :]
    b_a = tl.exp(tl.minimum(tl.load(A_LOG + i_h).to(tl.float32), 80.0))
    b_sp = tl.where(b_g > threshold, b_g, tl.log(1.0 + tl.exp(b_g)))
    # Saturate instead of letting the gate reach -inf. Every consumer only ever
    # looks at 2^(g_i - g_j), and at -1e30 per step that is already exactly 0 --
    # but an infinite g turns the differences into inf - inf = NaN.
    b_gate = tl.where(m_t[:, None],
                      -tl.minimum(b_a * b_sp, 1e30), 0.0)
    tl.store(Y + base, tl.cumsum(b_gate, axis=0) * scale,
             mask=m_t[:, None] & m_d[None, :])


# ---------------------------------------------------------------------------
# Whole delta-rule core for a single chunk (T <= 64), one kernel.
#
# When the sequence fits in one chunk and there is no carried-in recurrent
# state, the chunked algorithm collapses: the inter-chunk state ``h`` is zero
# throughout, so ``v_new == u``, the ``w`` projection is never used, and the
# output is just ``Aqk @ u``. That removes the reason to materialize A, Aqk, its
# triangular inverse, w/u/kg and the per-chunk state tensor in HBM -- ten
# launches (two kkt passes, solve_tril, recompute_w_u, the state recurrence, the
# output matmul, the state scatter and three ``zeros`` fills) become one kernel
# that keeps all of it in registers:
#
#   A[i,j]  = beta_i * sum_d k_i,d k_j,d 2^(g_i,d - g_j,d)         (i > j)
#   Aqk[i,j]= scale  * sum_d q_i,d k_j,d 2^(g_i,d - g_j,d)         (i >= j)
#   Ai      = (I + A)^-1
#   u       = Ai @ (beta * v)
#   out     = Aqk @ u
#   state   = u^T @ (k * 2^(g_last - g))
#
# **Decomposition.** Every pair i > j is handled at exactly one level
# ``L = 2^t``, t being the highest bit where i and j differ: there
# ``i // L == j // L + 1`` with ``i // L`` odd, and the anchor
# ``a = (i // L) * L`` satisfies ``j < a <= i``. Because the chunk-local gate is
# a cumulative sum of non-positive increments, that ordering makes *both*
# factors of the split ``2^(g_i - g_a) * 2^(g_a - g_j)`` have a non-positive
# exponent, so neither can overflow -- they underflow to zero together, which is
# the right answer. (Splitting at a fixed anchor instead lets one factor reach
# +inf while its partner underflows to 0, and ``0 * inf`` poisons the tile with
# NaN; ``chunk_kda_scaled_dot_kkt`` avoids that the same way, by only ever
# anchoring *between* the two rows of a pair.) ``log2(BT)`` masked matmuls cover
# the whole triangle; the diagonal, where the factor is 1, is added directly.
#
# ``Ai`` uses the matching 2x2 block-inverse identity
# ``[[L11,0],[L21,L22]]^-1 = [[Ai11,0],[-Ai22 L21 Ai11, Ai22]]`` at every
# power-of-two block size (``Ai <- Ai - Ai @ mask_L(A) @ Ai``), which replaces
# ``solve_tril``'s 56 serially-dependent scalar loads with register-resident
# matmuls over the same level structure.
# ---------------------------------------------------------------------------
@triton.jit
def _single_chunk_kernel(
    Q, K_, V_, G, BETA, OUT, RS, IDX,
    T,
    scale,
    H: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    LOG_BT: tl.constexpr,
    DOTP: tl.constexpr,
):
    i_h = tl.program_id(0)
    i_v = tl.program_id(1)
    o_t = tl.arange(0, BT)
    o_d = tl.arange(0, D)
    o_v = i_v * BV + tl.arange(0, BV)
    m_t = o_t < T
    hd = i_h * D
    srow = H * D
    p_kq = o_t[:, None] * srow + hd + o_d[None, :]

    b_g = tl.load(G + p_kq, mask=m_t[:, None], other=0.0)
    b_q = tl.load(Q + p_kq, mask=m_t[:, None], other=0.0)
    b_k = tl.load(K_ + p_kq, mask=m_t[:, None], other=0.0)
    b_beta = tl.load(BETA + o_t * H + i_h, mask=m_t, other=0.0)

    b_A = tl.zeros((BT, BT), dtype=tl.float32)
    b_Aq = tl.zeros((BT, BT), dtype=tl.float32)
    for lvl in tl.static_range(LOG_BT):
        i_b = o_t // (1 << lvl)
        o_ra = i_b * (1 << lvl)
        o_ca = o_ra + (1 << lvl)
        b_ga = tl.load(G + o_ra[:, None] * srow + hd + o_d[None, :],
                       mask=(o_ra < T)[:, None], other=0.0)
        b_gc = tl.load(G + o_ca[:, None] * srow + hd + o_d[None, :],
                       mask=(o_ca < T)[:, None], other=0.0)
        b_ea = tl.exp2(b_g - b_ga)
        b_kb = tl.trans((b_k * tl.exp2(b_gc - b_g)).to(tl.bfloat16))
        m_off = (i_b[:, None] == i_b[None, :] + 1) & (i_b[:, None] % 2 == 1)
        b_A = tl.where(m_off, tl.dot((b_k * b_ea).to(tl.bfloat16), b_kb), b_A)
        b_Aq = tl.where(m_off, tl.dot((b_q * b_ea).to(tl.bfloat16), b_kb), b_Aq)

    m_diag = o_t[:, None] == o_t[None, :]
    b_Aq = tl.where(m_diag,
                    tl.sum(b_q.to(tl.float32) * b_k.to(tl.float32), 1)[:, None],
                    b_Aq)
    m_both = m_t[:, None] & m_t[None, :]
    b_A = tl.where((o_t[:, None] > o_t[None, :]) & m_both,
                   b_A * b_beta[:, None], 0.0)
    b_Aq = tl.where((o_t[:, None] >= o_t[None, :]) & m_both, b_Aq * scale, 0.0)

    b_Ai = tl.where(m_diag, 1.0, 0.0)
    for lvl in tl.static_range(LOG_BT):
        i_b = o_t // (1 << lvl)
        m_off = (i_b[:, None] == i_b[None, :] + 1) & (i_b[:, None] % 2 == 1)
        b_Ai -= tl.dot(tl.dot(b_Ai, tl.where(m_off, b_A, 0.0),
                              input_precision=DOTP),
                       b_Ai, input_precision=DOTP)

    p_v = o_t[:, None] * srow + hd + o_v[None, :]
    b_v = tl.load(V_ + p_v, mask=m_t[:, None], other=0.0)
    b_u = tl.dot(b_Ai.to(tl.bfloat16),
                 (b_v * b_beta[:, None]).to(tl.bfloat16),
                 input_precision="ieee").to(tl.bfloat16)
    tl.store(OUT + p_v,
             tl.dot(b_Aq.to(tl.bfloat16), b_u).to(tl.bfloat16),
             mask=m_t[:, None])

    slot = tl.load(IDX).to(tl.int64)
    if slot >= 0:
        b_gl = tl.load(G + (T - 1) * srow + hd + o_d)
        b_kg = (b_k * tl.exp2(b_gl[None, :] - b_g)).to(tl.bfloat16)
        tl.store(RS + slot * (H * D * D) + i_h * (D * D)
                 + o_v[:, None] * D + o_d[None, :],
                 tl.dot(tl.trans(b_u), b_kg))


# ---------------------------------------------------------------------------
# Per-chunk A / Aqk / (I+A)^-1 in one pass.
#
# The vendored pair does this in two kernels plus ``solve_tril``, and
# ``chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_intra`` -- the one that fills
# the 16x16 diagonal blocks -- walks its 16 columns one at a time, re-exponentiating
# a whole [16, K] tile per column. That is 16x more ``exp2`` than the data needs
# and it is the single most expensive kernel in the 16k-token step.
#
# The level decomposition used by ``_single_chunk_kernel`` (see its comment for
# why splitting the ``2^(g_i - g_j)`` factor at an anchor *between* i and j keeps
# both exponents non-positive, hence overflow-free) covers the diagonal blocks
# with plain matmuls instead, so the whole triangle costs ``log2(BT)`` masked
# matmuls. ``Ai`` then falls out of the same level structure in registers, so A
# itself never reaches HBM and the three ``zeros`` fills the vendored path needs
# are gone as well.
# ---------------------------------------------------------------------------
@triton.jit
def _kkt_solve_kernel(
    Q, K_, G, BETA, AQK, AI,
    T,
    scale,
    H: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    LOG_BT: tl.constexpr,
    DOTP: tl.constexpr,
):
    i_t = tl.program_id(0)
    i_h = tl.program_id(1)
    o = tl.arange(0, BT)
    o_d = tl.arange(0, D)
    o_r = i_t * BT + o
    m = o_r < T
    srow = H * D
    hd = i_h * D
    p = o_r[:, None] * srow + hd + o_d[None, :]

    b_g = tl.load(G + p, mask=m[:, None], other=0.0)
    b_q = tl.load(Q + p, mask=m[:, None], other=0.0)
    b_k = tl.load(K_ + p, mask=m[:, None], other=0.0)
    b_beta = tl.load(BETA + o_r * H + i_h, mask=m, other=0.0)

    b_A = tl.zeros((BT, BT), dtype=tl.float32)
    b_Aq = tl.zeros((BT, BT), dtype=tl.float32)
    for lvl in tl.static_range(LOG_BT):
        i_b = o // (1 << lvl)
        o_ra = i_b * (1 << lvl)
        o_ca = o_ra + (1 << lvl)
        b_ga = tl.load(G + (i_t * BT + o_ra)[:, None] * srow + hd + o_d[None, :],
                       mask=((i_t * BT + o_ra) < T)[:, None], other=0.0)
        # The column anchor of the last L-block of a chunk falls in the *next*
        # chunk, where the gate cumsum has restarted; no pair at this level can
        # use it (its partner rows all sit outside the tile), so read zero.
        b_gc = tl.load(G + (i_t * BT + o_ca)[:, None] * srow + hd + o_d[None, :],
                       mask=((o_ca < BT) & ((i_t * BT + o_ca) < T))[:, None],
                       other=0.0)
        b_ea = tl.exp2(b_g - b_ga)
        b_kb = tl.trans((b_k * tl.exp2(b_gc - b_g)).to(tl.bfloat16))
        m_off = (i_b[:, None] == i_b[None, :] + 1) & (i_b[:, None] % 2 == 1)
        b_A = tl.where(m_off, tl.dot((b_k * b_ea).to(tl.bfloat16), b_kb), b_A)
        b_Aq = tl.where(m_off, tl.dot((b_q * b_ea).to(tl.bfloat16), b_kb), b_Aq)

    m_diag = o[:, None] == o[None, :]
    b_Aq = tl.where(m_diag,
                    tl.sum(b_q.to(tl.float32) * b_k.to(tl.float32), 1)[:, None],
                    b_Aq)
    m_both = m[:, None] & m[None, :]
    b_A = tl.where((o[:, None] > o[None, :]) & m_both, b_A * b_beta[:, None], 0.0)
    p_a = o_r[:, None] * (H * BT) + i_h * BT + o[None, :]
    tl.store(AQK + p_a,
             tl.where((o[:, None] >= o[None, :]) & m_both, b_Aq * scale, 0.0),
             mask=m[:, None])

    b_Ai = tl.where(m_diag, 1.0, 0.0)
    for lvl in tl.static_range(LOG_BT):
        i_b = o // (1 << lvl)
        m_off = (i_b[:, None] == i_b[None, :] + 1) & (i_b[:, None] % 2 == 1)
        b_Ai -= tl.dot(tl.dot(b_Ai, tl.where(m_off, b_A, 0.0),
                              input_precision=DOTP),
                       b_Ai, input_precision=DOTP)
    tl.store(AI + p_a, b_Ai.to(AI.dtype.element_ty), mask=m[:, None])


# ---------------------------------------------------------------------------
# recurrent_state[slot] = final_state  (a plain strided copy; ``index_copy_``
# dispatches a generic gather/scatter kernel that costs several microseconds).
# ---------------------------------------------------------------------------
@triton.jit
def _scatter_state_kernel(SRC, DST, IDX, NUMEL, BLK: tl.constexpr):
    o = tl.program_id(0) * BLK + tl.arange(0, BLK)
    m = o < NUMEL
    slot = tl.load(IDX).to(tl.int64)
    if slot >= 0:
        tl.store(DST + slot * NUMEL + o, tl.load(SRC + o, mask=m), mask=m)


class _GraphEntry:
    __slots__ = ("graph", "static_in", "static_out", "idx", "sig", "src_idx",
                 "src_ver")

    def __init__(self, graph, static_in, static_out, idx, sig):
        self.graph = graph
        self.static_in = static_in
        self.static_out = static_out
        self.idx = idx
        self.sig = sig
        self.src_idx = None
        self.src_ver = -1


class KimiDeltaAttention(_BaselineKDA):
    """Same weights, same math, fewer launches and no host syncs."""

    # -- lazily built, param-keyed derived weights ---------------------------
    def _derived(self):
        """(conv weights as [WIDTH, 3P], qkvb|f_a|g_a merged into one GEMM)."""
        srcs = (self.q_conv1d.weight, self.k_conv1d.weight, self.v_conv1d.weight,
                self.qkvb_proj.weight, self.f_a_proj.weight, self.g_a_proj.weight)
        key = tuple((w.data_ptr(), w._version) for w in srcs)
        cached = getattr(self, "_derived_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1], cached[2]
        conv_w = torch.cat(
            [w.view(w.size(0), w.size(2)) for w in srcs[:3]], 0
        ).t().contiguous()
        merged_w = torch.cat([w for w in srcs[3:]], 0).contiguous()
        self._derived_cache = (key, conv_w, merged_w)
        return conv_w, merged_w

    def _plan(self, n: int, device):
        """Per-token-count constants: cu_seqlens plus the FLA chunk index /
        offset tables. Built on the host once per n and reused, so neither the
        steady-state forward nor a graph capture ever reads device memory."""
        plans = getattr(self, "_plan_cache", None)
        if plans is None:
            plans = self._plan_cache = {}
        ent = plans.get(n)
        if ent is not None and ent[0].device == device:
            return ent
        cu = torch.tensor([0, n], device=device, dtype=torch.int32)
        nt = triton.cdiv(n, FLA_CHUNK_SIZE)
        ci = torch.stack(
            [torch.zeros(nt, dtype=torch.int32), torch.arange(nt, dtype=torch.int32)],
            1,
        ).to(device)
        co = torch.tensor([0, nt], device=device, dtype=torch.int32)
        ent = (cu, ci, co)
        if len(plans) > 64:
            plans.clear()
        plans[n] = ent
        return ent

    # -- fast path ----------------------------------------------------------
    def _fast_state(self, hidden_states):
        """(state_view, meta) when the fused path applies, else (None, None)."""
        if self.qkvb_proj is None or hidden_states.dim() != 2:
            return None, None
        if hidden_states.stride(-1) != 1:
            return None, None
        if self.head_dim not in (32, 64, 128, 256) or self.conv_size < 2:
            return None, None
        state_view, meta = self._get_state()
        if state_view is None or meta is None:
            return None, None
        n = hidden_states.size(0)
        if (meta.num_prefills != 1 or meta.num_decodes != 0
                or meta.num_decode_tokens != 0
                or meta.num_prefill_tokens != n
                or meta.num_actual_tokens != n
                or meta.any_have_initial_state
                or meta.non_spec_state_indices_tensor is None
                or meta.non_spec_query_start_loc is None):
            return None, None
        if state_view.recurrent_state.dtype != torch.float32:
            return None, None
        return state_view, meta

    @staticmethod
    def _slot_index(meta):
        idx = meta.state_indices_long
        if idx is not None and idx.dtype == torch.int64:
            return idx[:1]
        return meta.non_spec_state_indices_tensor[:1].long()

    def _fused_core(self, hidden_states, sv, n, cu, ci, co, idx):
        p = self._ps_local
        nh = self._nh_local
        d = self.head_dim
        width = self.conv_size
        dev = hidden_states.device
        dt = hidden_states.dtype
        conv_w, merged_w = self._derived()

        fused = F.linear(hidden_states, merged_w)
        stride_x = fused.stride(0)
        lowr = fused[:, 3 * p + nh:]

        qkv = torch.empty((3, n, p), device=dev, dtype=dt)
        beta = torch.empty((1, n, nh), device=dev, dtype=torch.float32)
        cbt = 8 if n <= 8 else (16 if n <= 64 else 32)
        nblk = triton.cdiv(n, cbt)
        _conv_silu_l2_kernel[(nblk, 3, nh)](
            fused, qkv, conv_w,
            sv.q_conv_state, sv.k_conv_state, sv.v_conv_state, idx, beta,
            n, stride_x, n * p, nblk - 1,
            D=d, P=p, C3=3 * p, NH=nh, WIDTH=width, BT=cbt, EPS=_L2_EPS,
            num_warps=4 if cbt <= 16 else 8,
        )

        raw_g = F.linear(lowr[:, :d], self.f_b_proj.weight).view(1, n, nh, d)
        g2 = F.linear(lowr[:, d:], self.g_b_proj.weight).view(1, n, nh, d)

        q = qkv[0].view(1, n, nh, d)
        k = qkv[1].view(1, n, nh, d)
        v = qkv[2].view(1, n, nh, d)
        scale = d ** -0.5

        g = torch.empty((1, n, nh, d), device=dev, dtype=torch.float32)
        bd = 64 if d % 64 == 0 else d
        _gate_cumsum_kernel[(triton.cdiv(d, bd), triton.cdiv(n, FLA_CHUNK_SIZE), nh)](
            raw_g, g, self.A_log.reshape(-1), self.dt_bias.reshape(-1),
            n, RCP_LN2, _SOFTPLUS_THRESHOLD,
            H=nh, D=d, BT=FLA_CHUNK_SIZE, BD=bd, num_warps=4,
        )

        if n <= _FUSED_MAX_TOKENS:
            o = torch.empty((1, n, nh, d), device=dev, dtype=dt)
            # Only tile as many rows as the sequence needs: both the pair
            # decomposition and the triangular inverse cost O(log BT) [BT, BT]
            # matmuls, so a 1-token step has no business running the 64-row
            # version.
            bt = 16 if n <= 16 else FLA_CHUNK_SIZE
            bv = (32 if bt <= 16 else 64) if d % 64 == 0 else d
            _single_chunk_kernel[(nh, d // bv)](
                q, k, v, g, beta, o, sv.recurrent_state, idx,
                n, scale, H=nh, D=d, BT=bt, BV=bv,
                LOG_BT=bt.bit_length() - 1, DOTP=_TRIL_PRECISION,
                num_warps=4 if bt <= 16 else 8,
            )
            o = self.o_norm(o, g2)
            return self.o_proj(o.view(n, nh * d))

        bt = FLA_CHUNK_SIZE
        nt = triton.cdiv(n, bt)
        Aqk = torch.empty((1, n, nh, bt), device=dev, dtype=torch.float32)
        Ai = torch.empty((1, n, nh, bt), device=dev, dtype=dt)
        _kkt_solve_kernel[(nt, nh)](
            q, k, g, beta, Aqk, Ai, n, scale,
            H=nh, D=d, BT=bt, LOG_BT=bt.bit_length() - 1,
            DOTP=_TRIL_PRECISION, num_warps=_KKT_WARPS,
        )
        w, u, _, kg = recompute_w_u_fwd_kda(
            k=k, v=v, beta=beta, A=Ai, gk=g, cu_seqlens=cu, chunk_indices=ci,
        )
        del Ai
        h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
            k=kg, w=w, u=u, gk=g, initial_state=None, output_final_state=True,
            cu_seqlens=cu, chunk_indices=ci, chunk_offsets=co, use_exp2=True,
        )
        del w, u, kg
        o = chunk_gla_fwd_o_gk(
            q=q, v=v_new, g=g, A=Aqk, h=h, o=v, scale=scale, cu_seqlens=cu,
            chunk_indices=ci, chunk_size=FLA_CHUNK_SIZE,
        )
        del Aqk, v_new, h

        numel = final_state.numel()
        _scatter_state_kernel[(triton.cdiv(numel, 1024),)](
            final_state, sv.recurrent_state, idx, numel, BLK=1024, num_warps=4,
        )
        o = self.o_norm(o, g2)
        return self.o_proj(o.view(n, nh * d))

    # -- CUDA graph cache ---------------------------------------------------
    def _graph_sig(self, sv):
        conv_w, merged_w = self._derived()
        return (
            sv.q_conv_state.data_ptr(), sv.k_conv_state.data_ptr(),
            sv.v_conv_state.data_ptr(), sv.recurrent_state.data_ptr(),
            conv_w.data_ptr(), merged_w.data_ptr(),
            self.o_proj.weight.data_ptr(), self.o_proj.weight._version,
            self.A_log.data_ptr(), self.A_log._version,
            self.dt_bias.data_ptr(), self.dt_bias._version,
            self.f_b_proj.weight.data_ptr(), self.f_b_proj.weight._version,
            self.g_b_proj.weight.data_ptr(), self.g_b_proj.weight._version,
            self.o_norm.weight.data_ptr(), self.o_norm.weight._version,
        )

    def _capture(self, hidden_states, sv, meta, n, sig):
        dev = hidden_states.device
        cu, ci, co = self._plan(n, dev)
        idx = torch.empty(1, device=dev, dtype=torch.int64)
        idx.copy_(self._slot_index(meta))
        static_in = torch.empty_like(hidden_states)
        static_in.copy_(hidden_states)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self._fused_core(static_in, sv, n, cu, ci, co, idx)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_out = self._fused_core(static_in, sv, n, cu, ci, co, idx)
        return _GraphEntry(graph, static_in, static_out, idx, sig)

    @torch.no_grad()
    def forward(self, hidden_states: torch.Tensor, state_manager=None):
        del state_manager
        self._ensure_triton_allocator(hidden_states.device)
        sv, meta = self._fast_state(hidden_states)
        if sv is None:
            return _BaselineKDA.forward(self, hidden_states)

        n = hidden_states.size(0)
        if (n > _GRAPH_MAX_TOKENS or torch.cuda.is_current_stream_capturing()
                or torch.compiler.is_compiling()):
            cu, ci, co = self._plan(n, hidden_states.device)
            return self._fused_core(
                hidden_states, sv, n, cu, ci, co, self._slot_index(meta),
            )

        graphs = getattr(self, "_graph_cache", None)
        if graphs is None:
            graphs = self._graph_cache = {}
        sig = self._graph_sig(sv)
        ent = graphs.get(n)
        if ent is None or (ent is not _NO_GRAPH and ent.sig != sig):
            if len(graphs) >= _MAX_GRAPHS:
                graphs.clear()
            try:
                ent = self._capture(hidden_states, sv, meta, n, sig)
            except Exception:
                # Remember the failure rather than re-attempting a capture (and
                # its warmup) on every call.
                graphs[n] = _NO_GRAPH
                ent = _NO_GRAPH
            else:
                graphs[n] = ent
        if ent is _NO_GRAPH:
            cu, ci, co = self._plan(n, hidden_states.device)
            return self._fused_core(
                hidden_states, sv, n, cu, ci, co, self._slot_index(meta),
            )

        ent.static_in.copy_(hidden_states)
        # The captured graph reads the state slot out of ``ent.idx``. Refresh it
        # only when the metadata's own index tensor is not provably the one we
        # last copied (we hold a reference, so the identity check is sound).
        src = meta.non_spec_state_indices_tensor
        if src is not ent.src_idx or src._version != ent.src_ver:
            ent.idx.copy_(self._slot_index(meta))
            ent.src_idx = src
            ent.src_ver = src._version
        ent.graph.replay()
        return ent.static_out.clone()
