"""Oasis DiT blocks -- fused Triton implementation.

The captured workload is a *latency* problem, not a throughput one: 288-864
tokens of width 1024 per call, which the eager baseline spreads over ~200 tiny
CUDA launches (rotary tables rebuilt every call, six chunked modulation
tensors, ``repeat``/``unsqueeze`` broadcasts, four fp32-promoted LayerNorms, two
SDPA calls).  Nearly all of the 1.7 ms the baseline spends is launch latency.

The module tree is unchanged (so ``load_state_dict`` from the baseline transfers
verbatim), but ``forward`` becomes a CUDA graph over hand-written Triton
kernels, so a call costs two input copies plus one replay:

  * One GEMM kernel serves every projection, with the epilogue selected per call
    site: bias, tanh-GELU, or ``out = residual + gate * out``.  The residual is
    updated in place -- element (m, n) is read and written by the single program
    that owns it.  A SiLU-on-A variant lets the adaLN projection reuse it.
  * Both adaLN projections share one weight matrix and one SiLU.  The temporal
    half is only needed by the second half of the block, and like the input
    transpose it is bandwidth-bound, so both ride a side stream captured into the
    graph and overlap the compute-bound spatial GEMMs.
  * LayerNorm folds its fp32 reduction and the ``(1 + scale) * x + shift``
    modulation into one pass; the captured channel-major ``x`` stride is turned
    into the token-major residual by a single transposing copy.
  * Rotary runs as one in-place pass over the q/k thirds of the fused qkv
    output.  The q/k rows of that projection are permuted once at setup so each
    head's channels arrive as [even | odd] halves, which makes ``rotate_half``
    two contiguous half-width tiles rather than a ``d ^ 1`` permutation -- the
    interleaved form forces either a scalarized gather or a shared-memory
    reshape, and at these tile sizes either costs several times the attention
    it feeds.  Attention then reads q/k/v straight out of the packed buffer.

Launch configs for every captured frame count were measured on the target GPU
under the benchmark's own timing conditions (see ``_GEMM_CFG`` and friends).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear
from ..L1.oasis_rotary import OasisRotaryEmbedding
from ..L1.silu import SiLU
from ..L2.oasis_mlp import OasisMLP
from ..L2.oasis_spatial_axial_attention import OasisSpatialAxialAttention
from ..L2.oasis_temporal_axial_attention import OasisTemporalAxialAttention


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    fixed_dims = [1] * len(shift.shape[1:])
    shift = shift.repeat(x.shape[0] // shift.shape[0], *fixed_dims)
    scale = scale.repeat(x.shape[0] // shift.shape[0], *fixed_dims)
    while shift.dim() < x.dim():
        shift = shift.unsqueeze(-2)
        scale = scale.unsqueeze(-2)
    return x * (1 + scale) + shift


def _gate(x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    fixed_dims = [1] * len(g.shape[1:])
    g = g.repeat(x.shape[0] // g.shape[0], *fixed_dims)
    while g.dim() < x.dim():
        g = g.unsqueeze(-2)
    return g * x


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------
@triton.jit
def _tanh(x):
    # 1 - 2/(e^{2x}+1): saturates correctly at both ends, so no range guard.
    return 1.0 - 2.0 / (tl.exp(2.0 * x) + 1.0)


@triton.jit
def _k_trans(X, O, D, P, NPB, NDB, sx,
             BP: tl.constexpr, BD: tl.constexpr):
    """[frames, channels, pixels] -> [tokens, channels].

    The captured ``x`` stride is exactly channel-major within a frame, so the
    read side is 16 contiguous fp16 (one 32 B sector) per channel row and the
    write side is a full ``BD``-wide token row.
    """
    pid = tl.program_id(0)
    db = pid % NDB
    r = pid // NDB
    pb = r % NPB
    bt = r // NPB
    rp = pb * BP + tl.arange(0, BP)
    rd = db * BD + tl.arange(0, BD)
    mp = rp < P
    v = tl.load(X + bt * sx + rd[:, None] * P + rp[None, :], mask=mp[None, :], other=0.0)
    tl.store(O + (bt * P + rp)[:, None] * D + rd[None, :], tl.trans(v), mask=mp[:, None])


@triton.jit
def _k_lnmod(X, MOD, XN, N, D, P, smod, SOFF, EPS,
             BR: tl.constexpr, BD: tl.constexpr):
    """Row-wise fp32 LayerNorm (no affine) + adaLN modulation."""
    pid = tl.program_id(0)
    rn = pid * BR + tl.arange(0, BR)
    rd = tl.arange(0, BD)
    mn = rn < N
    md = mn[:, None] & (rd < D)[None, :]
    # issue the activation and modulation loads together: they are independent,
    # and at these sizes the kernel is bound by memory round trips, not bandwidth
    mb = MOD + (rn // P)[:, None] * smod + SOFF
    v = tl.load(X + rn[:, None] * D + rd[None, :], mask=md, other=0.0)
    sh = tl.load(mb + rd[None, :], mask=md, other=0.0)
    sl = tl.load(mb + D + rd[None, :], mask=md, other=0.0)
    v = v.to(tl.float32)
    mean = tl.sum(v, 1) / D
    rstd = 1.0 / tl.sqrt(tl.sum(v * v, 1) / D - mean * mean + EPS)
    y = ((v - mean[:, None]) * rstd[:, None] * (1.0 + sl.to(tl.float32))
         + sh.to(tl.float32))
    tl.store(XN + rn[:, None] * D + rd[None, :], y.to(XN.dtype.element_ty), mask=md)


@triton.jit
def _k_gemm(A, W, BI, RES, MOD, O, M, N, K, P, GOFF, sa, sw, so, smod,
            BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
            EPI: tl.constexpr, HAS_B: tl.constexpr, GM: tl.constexpr,
            TRANS: tl.constexpr, SILU: tl.constexpr):
    """O = A @ W.T (+ bias) with a fused epilogue.

    EPI 0 plain, 1 tanh-GELU, 2 ``O = RES + gate * O`` (O may alias RES).
    SILU applies the activation to the A tile, which lets the adaLN projection
    (SiLU then Linear, for both halves at once) reuse this kernel.
    """
    pid = tl.program_id(0)
    npm = tl.cdiv(M, BM)
    npn = N // BN
    nig = GM * npn
    gid = pid // nig
    fpm = gid * GM
    gsm = tl.minimum(npm - fpm, GM)
    pm = fpm + ((pid % nig) % gsm)
    pn = (pid % nig) // gsm
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    mm = rm < M
    ap = A + rm[:, None] * sa + rk[None, :]
    if TRANS:
        wp = W + rn[:, None] * sw + rk[None, :]
    else:
        wp = W + rk[:, None] * sw + rn[None, :]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for _ in range(K // BK):
        a = tl.load(ap, mask=mm[:, None], other=0.0)
        if SILU:
            af = a.to(tl.float32)
            a = (af / (1.0 + tl.exp(-af))).to(A.dtype.element_ty)
        if TRANS:
            b = tl.trans(tl.load(wp))
            wp += BK
        else:
            b = tl.load(wp)
            wp += BK * sw
        acc = tl.dot(a, b, acc)
        ap += BK
    if HAS_B:
        acc += tl.load(BI + rn)[None, :].to(tl.float32)
    if EPI == 1:
        acc = 0.5 * acc * (1.0 + _tanh(0.7978845608028654 *
                                       (acc + 0.044715 * acc * acc * acc)))
    if EPI == 2:
        g = tl.load(MOD + (rm // P)[:, None] * smod + GOFF + rn[None, :],
                    mask=mm[:, None], other=0.0).to(tl.float32)
        r = tl.load(RES + rm[:, None] * so + rn[None, :],
                    mask=mm[:, None], other=0.0).to(tl.float32)
        acc = r + g * acc
    tl.store(O + rm[:, None] * so + rn[None, :], acc.to(O.dtype.element_ty),
             mask=mm[:, None])


@triton.jit
def _k_gemm_sk(A, W, PART, M, N, K, sa, sw, sp,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
               SK: tl.constexpr, TRANS: tl.constexpr, SILU: tl.constexpr):
    """Split-K partial products for the narrow/deep GEMM (N small, K large).

    With M <= 864 a single-pass tile grid leaves most of the machine idle, so the
    K range is cut SK ways and the partials are summed by ``_k_red_epi``.
    """
    pid = tl.program_id(0)
    ks = tl.program_id(1)
    npn = N // BN
    pm = pid // npn
    pn = pid % npn
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    mm = rm < M
    kc = K // SK
    ap = A + rm[:, None] * sa + (ks * kc + rk)[None, :]
    if TRANS:
        wp = W + rn[:, None] * sw + (ks * kc + rk)[None, :]
    else:
        wp = W + (ks * kc + rk)[:, None] * sw + rn[None, :]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for _ in range(kc // BK):
        a = tl.load(ap, mask=mm[:, None], other=0.0)
        if SILU:
            af = a.to(tl.float32)
            a = (af / (1.0 + tl.exp(-af))).to(A.dtype.element_ty)
        if TRANS:
            b = tl.trans(tl.load(wp))
            wp += BK
        else:
            b = tl.load(wp)
            wp += BK * sw
        acc = tl.dot(a, b, acc)
        ap += BK
    tl.store(PART + ks * sp + rm[:, None] * N + rn[None, :],
             acc.to(PART.dtype.element_ty), mask=mm[:, None])


@triton.jit
def _k_red_epi(PART, BI, RES, MOD, O, M, N, P, GOFF, sp, so, smod,
               BM: tl.constexpr, BN: tl.constexpr, SK: tl.constexpr,
               EPI: tl.constexpr, HAS_B: tl.constexpr):
    """Sum split-K partials and run the same epilogue as ``_k_gemm``."""
    pid = tl.program_id(0)
    npn = N // BN
    rm = (pid // npn) * BM + tl.arange(0, BM)
    rn = (pid % npn) * BN + tl.arange(0, BN)
    mm = rm < M
    off = rm[:, None] * N + rn[None, :]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for i in range(SK):
        acc += tl.load(PART + i * sp + off, mask=mm[:, None], other=0.0).to(tl.float32)
    if HAS_B:
        acc += tl.load(BI + rn)[None, :].to(tl.float32)
    if EPI == 1:
        acc = 0.5 * acc * (1.0 + _tanh(0.7978845608028654 *
                                       (acc + 0.044715 * acc * acc * acc)))
    if EPI == 2:
        g = tl.load(MOD + (rm // P)[:, None] * smod + GOFF + rn[None, :],
                    mask=mm[:, None], other=0.0).to(tl.float32)
        r = tl.load(RES + rm[:, None] * so + rn[None, :],
                    mask=mm[:, None], other=0.0).to(tl.float32)
        acc = r + g * acc
    tl.store(O + rm[:, None] * so + rn[None, :], acc.to(O.dtype.element_ty),
             mask=mm[:, None])


@triton.jit
def _k_rope(QKV, CSH, SNH, N, sqn, P, TT, NSL,
            BR: tl.constexpr, DH: tl.constexpr, RMODE: tl.constexpr):
    """Rotate the q and k thirds of the packed qkv buffer in place.

    The q/k rows of the projection weight were permuted at setup so each head's
    channels arrive as [even | odd] halves, which turns ``rotate_half`` into two
    contiguous half-width tiles.  The interleaved form would need a ``d ^ 1``
    gather (Triton scalarizes it) or a shared-memory reshape, either of which
    costs several times this kernel.  A dot product is invariant to permuting q
    and k alike and v is left in the original order, so attention and the output
    projection are unaffected.
    """
    pid = tl.program_id(0)
    nb = pid // NSL
    sl = pid % NSL                     # (third, head) slot, 0 .. 2*NH-1
    hd = DH // 2
    rn = nb * BR + tl.arange(0, BR)
    rh = tl.arange(0, DH // 2)
    mn = (rn < N)[:, None]
    base = QKV + rn[:, None] * sqn + sl * DH
    if RMODE == 0:
        idx = rn % P
    else:
        idx = (rn // P) % TT
    off = idx[:, None] * hd + rh[None, :]
    cs = tl.load(CSH + off, mask=mn, other=0.0)
    sn = tl.load(SNH + off, mask=mn, other=0.0)
    a0 = tl.load(base + rh[None, :], mask=mn, other=0.0).to(tl.float32)
    a1 = tl.load(base + hd + rh[None, :], mask=mn, other=0.0).to(tl.float32)
    dt = QKV.dtype.element_ty
    tl.store(base + rh[None, :], (a0 * cs - a1 * sn).to(dt), mask=mn)
    tl.store(base + hd + rh[None, :], (a1 * cs + a0 * sn).to(dt), mask=mn)


@triton.jit
def _k_attn_sp(QKV, O, S, D, sqn, SCALE,
               BM: tl.constexpr, BN: tl.constexpr, DH: tl.constexpr):
    """Spatial axial attention over the H*W pixels of one frame (non-causal),
    reading q/k/v straight out of the packed (already rotated) qkv buffer."""
    pm = tl.program_id(0)
    h = tl.program_id(1)
    bt = tl.program_id(2)
    rd = tl.arange(0, DH)
    rm = pm * BM + tl.arange(0, BM)
    mq = rm < S
    base = QKV + bt * S * sqn + h * DH
    q = tl.load(base + rm[:, None] * sqn + rd[None, :], mask=mq[:, None], other=0.0)
    acc = tl.zeros((BM, DH), dtype=tl.float32)
    mi = tl.full((BM,), float("-inf"), tl.float32)
    li = tl.zeros((BM,), dtype=tl.float32)
    for st in range(0, S, BN):
        rn = st + tl.arange(0, BN)
        mk = (rn < S)[:, None]
        k = tl.load(base + D + rn[:, None] * sqn + rd[None, :], mask=mk, other=0.0)
        v = tl.load(base + 2 * D + rn[:, None] * sqn + rd[None, :], mask=mk, other=0.0)
        sc = tl.dot(q, tl.trans(k)) * SCALE
        sc = tl.where(tl.trans(mk), sc, float("-inf"))
        mn = tl.maximum(mi, tl.max(sc, 1))
        pr = tl.exp(sc - mn[:, None])
        al = tl.exp(mi - mn)
        li = li * al + tl.sum(pr, 1)
        acc = acc * al[:, None] + tl.dot(pr.to(QKV.dtype.element_ty), v)
        mi = mn
    acc = acc / li[:, None]
    tl.store(O + (bt * S + rm)[:, None] * D + h * DH + rd[None, :],
             acc.to(O.dtype.element_ty), mask=mq[:, None])


@triton.jit
def _k_attn_tp(QKV, O, T, P, D, sqn, SCALE, NH, NP,
               BP: tl.constexpr, TP: tl.constexpr, DH: tl.constexpr,
               CAUSAL: tl.constexpr):
    """Temporal axial attention: T <= 8 frames, one head and BP pixels per
    program.  The BP*TP tokens are stacked into one tile and the score matrix is
    masked block-diagonally (plus causally), which keeps the MMA shapes legal
    without one program per pixel."""
    pid = tl.program_id(0)
    h = pid % NH
    pb = pid // NH
    rr = tl.arange(0, BP * TP)
    lp = rr // TP
    lt = rr % TP
    gp = pb * BP + lp
    ok = (gp < NP) & (lt < T)
    b = gp // P
    p = gp % P
    rd = tl.arange(0, DH)
    qp = QKV + (b * T * P + lt * P + p)[:, None] * sqn + h * DH
    q = tl.load(qp + rd[None, :], mask=ok[:, None], other=0.0)
    k = tl.load(qp + D + rd[None, :], mask=ok[:, None], other=0.0)
    v = tl.load(qp + 2 * D + rd[None, :], mask=ok[:, None], other=0.0)
    sc = tl.dot(q, tl.trans(k)) * SCALE
    m2 = ok[:, None] & ok[None, :] & (lp[:, None] == lp[None, :])
    if CAUSAL:
        m2 = m2 & (lt[:, None] >= lt[None, :])
    sc = tl.where(m2, sc, float("-inf"))
    mx = tl.max(sc, 1)
    mx = tl.where(mx == float("-inf"), 0.0, mx)
    pr = tl.where(m2, tl.exp(sc - mx[:, None]), 0.0)
    ls = tl.sum(pr, 1)
    ls = tl.where(ls == 0.0, 1.0, ls)
    o = tl.dot((pr / ls[:, None]).to(QKV.dtype.element_ty), v)
    tl.store(O + (b * T * P + lt * P + p)[:, None] * D + h * DH + rd[None, :],
             o.to(O.dtype.element_ty), mask=ok[:, None])


# ---------------------------------------------------------------------------
# Launch configuration
# ---------------------------------------------------------------------------
# Per-GEMM launch configs, measured on the target GPU under the benchmark's
# timing conditions (L2 flushed, one launch) for every captured frame count.
# Keyed by (M, N, K), with a per-(N, K) fallback:
#   ("p", TRANS, BM, BN, BK, warps, stages, GROUP_M)  single-pass tile grid
#   ("s", TRANS, BM, BN, BK, SK, warps, stages, rBM, rBN, rwarps)  split-K
# TRANS selects the weight layout (1 = [N, K] as stored by Linear, 0 =
# pre-transposed to [K, N]); both are materialized when configs disagree.
_GEMM_DEF = {
    (3072, 1024): ("p", 1, 128, 256, 64, 8, 4, 8),
    (1024, 1024): ("p", 1, 64, 128, 64, 4, 6, 8),
    (4096, 1024): ("p", 1, 128, 256, 64, 8, 4, 8),
    (1024, 4096): ("p", 0, 128, 64, 128, 8, 4, 8),
    (6144, 1024): ("p", 0, 16, 64, 128, 2, 5, 8),
}
_GEMM_CFG = {
    (288, 3072, 1024): ("p", 0, 128, 128, 64, 8, 5, 8),
    (288, 1024, 1024): ("p", 0, 64, 64, 128, 4, 5, 8),
    (288, 4096, 1024): ("p", 0, 128, 128, 64, 8, 5, 8),
    (288, 1024, 4096): ("p", 0, 64, 64, 128, 4, 5, 8),
    (432, 3072, 1024): ("p", 1, 128, 128, 64, 8, 4, 8),
    (432, 1024, 1024): ("p", 0, 64, 64, 128, 4, 5, 8),
    (432, 4096, 1024): ("p", 0, 128, 128, 64, 8, 5, 8),
    (432, 1024, 4096): ("p", 0, 64, 64, 128, 4, 5, 8),
    (576, 3072, 1024): ("p", 1, 128, 256, 64, 8, 4, 8),
    (576, 1024, 1024): ("p", 0, 64, 64, 128, 4, 5, 8),
    (576, 4096, 1024): ("p", 0, 64, 128, 64, 4, 4, 8),
    (576, 1024, 4096): ("p", 0, 64, 64, 128, 4, 5, 8),
    (720, 3072, 1024): ("p", 0, 128, 256, 64, 8, 4, 8),
    (720, 1024, 1024): ("p", 1, 64, 128, 64, 4, 6, 8),
    (720, 4096, 1024): ("p", 1, 128, 256, 64, 8, 4, 8),
    (720, 1024, 4096): ("s", 0, 64, 128, 64, 2, 8, 4, 64, 128, 4),
    (864, 3072, 1024): ("p", 1, 128, 256, 64, 8, 4, 8),
    (864, 1024, 1024): ("p", 0, 128, 64, 128, 8, 4, 8),
    (864, 4096, 1024): ("p", 1, 128, 256, 64, 8, 4, 8),
    (864, 1024, 4096): ("s", 0, 64, 128, 64, 2, 8, 4, 64, 128, 4),
}


def _round_up(a, b):
    return -(-a // b) * b


# Only shapes the capture actually exercises are measured; anything else gets a
# conservative single-pass config that is valid for any N, K divisible by 64.
_GEMM_ANY = ("p", 1, 64, 64, 64, 4, 3, 8)


def _cfg_for(M, N, K):
    return _GEMM_CFG.get((M, N, K)) or _GEMM_DEF.get((N, K)) or _GEMM_ANY


def _trans_set(N, K):
    """Weight layouts any config for this (N, K) asks for."""
    out = {(_GEMM_DEF.get((N, K)) or _GEMM_ANY)[1]}
    for (m, n, k), c in _GEMM_CFG.items():
        if n == N and k == K:
            out.add(c[1])
    return out


_OVERLAP = True                          # run the temporal adaLN on a side stream
_TRANS_CFG = (32, 64, 2, 1)              # BP, BD, warps, stages
_LNMOD_CFG = (1, 2, 3)                   # BR, warps, stages
_ROPE_CFG = (32, 8, 2)                   # BR, warps, stages
_ATTN_SP_CFG = (16, 32, 1, 3)            # BM, BN, warps, stages
_ATTN_TP_CFG = (16, 1, 3)                # tile rows, warps, stages


class _Weights:
    __slots__ = ("sig", "wa", "ba", "qkv", "op_w", "op_b", "f1w", "f1b", "f2w",
                 "f2b", "cos_sp", "sin_sp", "wa_h")


class _Plan:
    __slots__ = ("sx", "sc", "out", "graph", "cos_t", "sin_t", "bufs",
                 "sx_view", "out_view")


class SpatioTemporalDiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        *,
        mlp_ratio: float = 4.0,
        is_causal: bool = True,
        spatial_rotary_emb: OasisRotaryEmbedding,
        temporal_rotary_emb: OasisRotaryEmbedding,
    ):
        super().__init__()
        self.s_norm1 = LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.s_attn = OasisSpatialAxialAttention(
            hidden_size,
            heads=num_heads,
            dim_head=hidden_size // num_heads,
            rotary_emb=spatial_rotary_emb,
        )
        self.s_norm2 = LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.s_mlp = OasisMLP(
            hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            approximate_tanh=True,
        )
        self.s_adaLN_modulation = nn.Sequential(
            SiLU(),
            Linear(hidden_size, 6 * hidden_size, bias=True),
        )

        self.t_norm1 = LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.t_attn = OasisTemporalAxialAttention(
            hidden_size,
            heads=num_heads,
            dim_head=hidden_size // num_heads,
            rotary_emb=temporal_rotary_emb,
            is_causal=is_causal,
        )
        self.t_norm2 = LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.t_mlp = OasisMLP(
            hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            approximate_tanh=True,
        )
        self.t_adaLN_modulation = nn.Sequential(
            SiLU(),
            Linear(hidden_size, 6 * hidden_size, bias=True),
        )

        self._hidden = hidden_size
        self._heads = num_heads
        self._fh = int(hidden_size * mlp_ratio)
        self._causal = bool(is_causal)
        self._side = None
        self._w: _Weights | None = None
        self._plans: dict = {}
        self._plan_dims: dict = {}

    # -- reference path -----------------------------------------------------
    def _forward_ref(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        s = self.s_adaLN_modulation(c).chunk(6, dim=-1)
        x = x + _gate(self.s_attn(_modulate(self.s_norm1(x), s[0], s[1])), s[2])
        x = x + _gate(self.s_mlp(_modulate(self.s_norm2(x), s[3], s[4])), s[5])
        t = self.t_adaLN_modulation(c).chunk(6, dim=-1)
        x = x + _gate(self.t_attn(_modulate(self.t_norm1(x), t[0], t[1])), t[2])
        x = x + _gate(self.t_mlp(_modulate(self.t_norm2(x), t[3], t[4])), t[5])
        return x

    # -- fused path ---------------------------------------------------------
    def _eligible(self, x: torch.Tensor, c: torch.Tensor) -> bool:
        D = self._hidden
        if x.dim() != 5 or c.dim() != 3:
            return False
        if x.device.type != "cuda" or x.dtype not in (torch.float16, torch.bfloat16):
            return False
        if c.dtype != x.dtype or c.device != x.device:
            return False
        if x.shape[4] != D or c.shape[2] != D or x.shape[0] != c.shape[0]:
            return False
        if x.shape[1] != c.shape[1] or x.shape[1] > 8:
            return False
        dh = D // self._heads
        if dh * self._heads != D or dh not in (32, 64, 128):
            return False
        # get_axial_freqs over the two pixel axes yields 2 * dim channels
        if 2 * self.s_attn.rotary_emb.dim != dh or self.s_attn.rotary_emb.dim % 2:
            return False
        if self.t_attn.rotary_emb.dim != dh:
            return False
        for ln in (self.s_norm1, self.s_norm2, self.t_norm1, self.t_norm2):
            if ln.weight is not None or ln.bias is not None or not ln.promote_fp32:
                return False
        if self.s_norm1.eps != self.s_norm2.eps or self.t_norm1.eps != self.s_norm1.eps:
            return False
        if self.t_norm2.eps != self.s_norm1.eps:
            return False
        # every GEMM tile config in use needs these divisibilities
        if self._fh % 256 or D % 256 or (x.shape[2] * x.shape[3]) % 16:
            return False
        return True

    def _sig(self):
        return (self.s_adaLN_modulation[1].weight.data_ptr(),
                self.t_adaLN_modulation[1].weight.data_ptr(),
                self.s_attn.to_qkv.weight.data_ptr(),
                self.t_mlp.fc2.weight.data_ptr(),
                self.s_attn.rotary_emb.freqs.data_ptr(),
                self.t_attn.rotary_emb.freqs.data_ptr())

    def _build_weights(self, H: int, W: int) -> _Weights:
        w = _Weights()
        D = self._hidden
        dh = D // self._heads
        nh = self._heads

        def pack(t, N, K):
            t = t.detach()
            return {tr: (t.contiguous() if tr else t.t().contiguous())
                    for tr in _trans_set(N, K)}

        sa = self.s_adaLN_modulation[1]
        ta = self.t_adaLN_modulation[1]
        # The two adaLN projections share silu(c), so their weights are stored
        # as one matrix; per-half views let the temporal projection run on a side
        # stream (it is not needed until the second half of the block).
        w.wa = pack(torch.cat([sa.weight.detach(), ta.weight.detach()], 0),
                    6 * D, D)
        w.wa_h = tuple({tr: (t[i * 6 * D:(i + 1) * 6 * D] if tr
                             else t[:, i * 6 * D:(i + 1) * 6 * D])
                        for tr, t in w.wa.items()} for i in (0, 1))
        ba = torch.cat([sa.bias.detach(), ta.bias.detach()], 0).contiguous()
        w.ba = (ba[:6 * D], ba[6 * D:])
        # q/k rows are reordered to [even | odd] per head so the rotary kernel
        # sees two contiguous half-width tiles instead of a d ^ 1 permutation.
        sub = torch.cat([torch.arange(0, dh, 2), torch.arange(1, dh, 2)])
        qk = ((torch.arange(2 * nh) * dh).view(-1, 1) + sub).reshape(-1)
        perm = torch.cat([qk, torch.arange(2 * D, 3 * D)]).to(sa.weight.device)
        w.qkv = tuple(pack(m.to_qkv.weight.detach()[perm], 3 * D, D)
                      for m in (self.s_attn, self.t_attn))
        w.op_w = tuple(pack(m.to_out.weight, D, D)
                       for m in (self.s_attn, self.t_attn))
        w.op_b = tuple(m.to_out.bias.detach().contiguous()
                       for m in (self.s_attn, self.t_attn))
        w.f1w = tuple(pack(m.fc1.weight, self._fh, D)
                      for m in (self.s_mlp, self.t_mlp))
        w.f1b = tuple(m.fc1.bias.detach().contiguous() for m in (self.s_mlp, self.t_mlp))
        w.f2w = tuple(pack(m.fc2.weight, D, self._fh)
                      for m in (self.s_mlp, self.t_mlp))
        w.f2b = tuple(m.fc2.bias.detach().contiguous() for m in (self.s_mlp, self.t_mlp))
        # Rotary tables come through the module's own helpers so they see exactly
        # the (fp16-cast) ``freqs`` parameter the baseline sees.  Only the
        # pair-level (even-index) entries are needed: repeat_interleave(2)
        # duplicates each angle across its pair.
        fr = self.s_attn.rotary_emb.get_axial_freqs(H, W).reshape(H * W, -1)
        w.cos_sp = fr.cos()[:, 0::2].float().contiguous()
        w.sin_sp = fr.sin()[:, 0::2].float().contiguous()
        w.sig = self._sig()
        return w

    def _temporal_tables(self, T: int, x: torch.Tensor):
        rot = self.t_attn.rotary_emb
        pos = torch.arange(T, device=x.device, dtype=x.dtype)
        fr = rot.forward(pos, rot.freqs, seq_len=T)
        return (fr.cos()[:, 0::2].float().contiguous(),
                fr.sin()[:, 0::2].float().contiguous())

    # -- kernel launches ----------------------------------------------------
    def _gemm(self, a, wd, bias, res, mod, out, M, N, K, P, goff, epi, part,
              silu=False):
        cfg = _cfg_for(M, N, K)
        wt = wd[cfg[1]]
        smod = mod.stride(0) if mod is not None else 0
        if cfg[0] == "p":
            _, tr, BM, BN, BK, nw, ns, gm = cfg
            _k_gemm[(triton.cdiv(M, BM) * (N // BN),)](
                a, wt, bias, res, mod, out, M, N, K, P, goff,
                a.stride(0), wt.stride(0), out.stride(0), smod,
                BM=BM, BN=BN, BK=BK, EPI=epi, HAS_B=bias is not None, GM=gm,
                TRANS=tr, SILU=silu, num_warps=nw, num_stages=ns)
            return
        _, tr, BM, BN, BK, SK, nw, ns, rbm, rbn, rnw = cfg
        sp = _round_up(M, BM) * N
        _k_gemm_sk[(triton.cdiv(M, BM) * (N // BN), SK)](
            a, wt, part, M, N, K, a.stride(0), wt.stride(0), sp,
            BM=BM, BN=BN, BK=BK, SK=SK, TRANS=tr, SILU=silu,
            num_warps=nw, num_stages=ns)
        _k_red_epi[(triton.cdiv(M, rbm) * (N // rbn),)](
            part, bias, res, mod, out, M, N, P, goff, sp, out.stride(0), smod,
            BM=rbm, BN=rbn, SK=SK, EPI=epi, HAS_B=bias is not None,
            num_warps=rnw, num_stages=1)

    def _launch(self, p: _Plan, w: _Weights, B: int, T: int, H: int, W: int, bufs):
        D = self._hidden
        NH = self._heads
        DH = D // NH
        FH = self._fh
        P = H * W
        Nt = B * T * P
        M = B * T
        eps = float(self.s_norm1.eps)
        scale = DH ** -0.5
        mod, xr, xn, qkv, ao, hid, part = bufs

        cur = torch.cuda.current_stream()
        side = self._side
        bp, bd, nw, ns = _TRANS_CFG
        cflat = p.sc.view(M, D)
        trans = lambda: _k_trans[(M * triton.cdiv(P, bp) * (D // bd),)](
            p.sx, xr, D, P, triton.cdiv(P, bp), D // bd, p.sx.stride(1),
            BP=bp, BD=bd, num_warps=nw, num_stages=ns)
        ada_t = lambda: self._gemm(cflat, w.wa_h[1], w.ba[1], None, None,
                                   mod[:, 6 * D:], M, 6 * D, D, P, 0, 0, part,
                                   silu=True)
        if _OVERLAP:
            # the transpose is independent of the spatial adaLN, and the temporal
            # adaLN is not needed until the second half of the block: both are
            # bandwidth-bound, so they ride a side stream under the compute-bound
            # spatial GEMMs instead of extending the critical path
            side.wait_stream(cur)
            with torch.cuda.stream(side):
                trans()
            self._gemm(cflat, w.wa_h[0], w.ba[0], None, None, mod,
                       M, 6 * D, D, P, 0, 0, part, silu=True)
            cur.wait_stream(side)
            side.wait_stream(cur)
            with torch.cuda.stream(side):
                ada_t()
        else:
            self._gemm(cflat, w.wa_h[0], w.ba[0], None, None, mod,
                       M, 6 * D, D, P, 0, 0, part, silu=True)
            ada_t()
            trans()

        br, nw, ns = _LNMOD_CFG
        bd = triton.next_power_of_2(D)
        rbr, rnw, rns = _ROPE_CFG
        abm, abn, anw, ans = _ATTN_SP_CFG
        tnr, tnw, tns = _ATTN_TP_CFG
        tp = triton.next_power_of_2(T)
        tbp = max(1, tnr // tp)
        for half in range(2):
            if half and _OVERLAP:
                cur.wait_stream(side)
            mo = half * 6 * D
            for stage in range(2):
                _k_lnmod[(triton.cdiv(Nt, br),)](
                    xr, mod, xn, Nt, D, P, mod.stride(0),
                    mo + (3 * D if stage else 0), eps,
                    BR=br, BD=bd, num_warps=nw, num_stages=ns)
                if stage == 0:
                    self._gemm(xn, w.qkv[half], None, None, None, qkv,
                               Nt, 3 * D, D, P, 0, 0, part)
                    cs, sn = (w.cos_sp, w.sin_sp) if half == 0 else (p.cos_t, p.sin_t)
                    _k_rope[(triton.cdiv(Nt, rbr) * 2 * NH,)](
                        qkv, cs, sn, Nt, 3 * D, P, T, 2 * NH,
                        BR=rbr, DH=DH, RMODE=half, num_warps=rnw, num_stages=rns)
                    if half == 0:
                        _k_attn_sp[(triton.cdiv(P, abm), NH, M)](
                            qkv, ao, P, D, 3 * D, scale,
                            BM=abm, BN=abn, DH=DH, num_warps=anw, num_stages=ans)
                    else:
                        _k_attn_tp[(triton.cdiv(B * P, tbp) * NH,)](
                            qkv, ao, T, P, D, 3 * D, scale,
                            NH, B * P, BP=tbp, TP=tp, DH=DH, CAUSAL=self._causal,
                            num_warps=tnw, num_stages=tns)
                    self._gemm(ao, w.op_w[half], w.op_b[half], xr, mod, xr,
                               Nt, D, D, P, mo + 2 * D, 2, part)
                else:
                    self._gemm(xn, w.f1w[half], w.f1b[half], None, None, hid,
                               Nt, FH, D, P, 0, 1, part)
                    self._gemm(hid, w.f2w[half], w.f2b[half], xr, mod, xr,
                               Nt, D, FH, P, mo + 5 * D, 2, part)

    def _part_buf(self, Nt, MT, D, FH, dev, dt):
        """Scratch for split-K partials, sized from the selected configs."""
        need = 0
        for (M, N, K) in ((Nt, 3 * D, D), (Nt, D, D), (Nt, FH, D), (Nt, D, FH),
                          (MT, 6 * D, D)):
            cfg = _cfg_for(M, N, K)
            if cfg[0] == "s":
                need = max(need, cfg[5] * _round_up(M, cfg[2]) * N)
        return torch.empty(need, device=dev, dtype=dt)

    def _build_plan(self, x: torch.Tensor, c: torch.Tensor) -> _Plan:
        B, T, H, W, D = x.shape
        FH = self._fh
        P = H * W
        Nt = B * T * P
        dev, dt = x.device, x.dtype
        if self._w is None or self._w.sig != self._sig():
            self._w = self._build_weights(H, W)
        w = self._w

        if self._side is None:
            self._side = torch.cuda.Stream(device=dev)
        p = _Plan()
        sxp = torch.empty((B, T, D, P), device=dev, dtype=dt)
        p.sx = sxp
        p.sc = torch.empty((B, T, D), device=dev, dtype=dt)
        p.out = torch.empty((Nt, D), device=dev, dtype=dt)
        p.cos_t, p.sin_t = self._temporal_tables(T, x)
        p.sx_view = sxp.view(B, T, D, H, W).permute(0, 1, 3, 4, 2)
        p.out_view = p.out.view(B, T, H, W, D)
        bufs = (
            torch.empty((B * T, 12 * D), device=dev, dtype=dt),   # modulation
            p.out,                                                # residual
            torch.empty((Nt, D), device=dev, dtype=dt),           # normed
            torch.empty((Nt, 3 * D), device=dev, dtype=dt),       # qkv
            torch.empty((Nt, D), device=dev, dtype=dt),           # attn out
            torch.empty((Nt, FH), device=dev, dtype=dt),          # mlp hidden
            self._part_buf(Nt, B * T, D, FH, dev, dt),            # split-K scratch
        )
        p.bufs = bufs

        for _ in range(2):        # compile everything outside the capture
            self._launch(p, w, B, T, H, W, bufs)
        torch.cuda.synchronize()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            self._launch(p, w, B, T, H, W, bufs)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        try:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self._launch(p, w, B, T, H, W, bufs)
            p.graph = g
        except Exception:
            # No graph (e.g. capture already active): replay eagerly instead.
            p.graph = None
            self._plan_dims[id(p)] = (B, T, H, W)
        self._plans[(B, T, H, W)] = p
        return p

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Copy the inputs into the captured buffers and replay.

        The result is the plan's own residual buffer, reused across calls (the
        usual graph-runner contract): each call overwrites it, and a caller that
        needs to keep a result past the next call of *this* block must copy it.
        Chaining blocks is unaffected -- every block stages its input into its
        own buffer -- and the residual chain inside the block is safe because the
        graph rewrites the buffer from the input transpose before reading it.
        """
        p = self._plans.get((x.shape[0], x.shape[1], x.shape[2], x.shape[3]))
        if p is None or torch.is_grad_enabled():
            # Autograd is never exercised by this operator's workload, and the
            # fused path does not build a graph, so defer to the reference.
            if torch.is_grad_enabled() or not self._eligible(x, c):
                return self._forward_ref(x, c)
            with torch.no_grad():
                p = self._build_plan(x, c)
        p.sx_view.copy_(x)
        p.sc.copy_(c)
        if p.graph is None:
            self._launch(p, self._w, *self._plan_dims[id(p)], p.bufs)
        else:
            p.graph.replay()
        return p.out_view
