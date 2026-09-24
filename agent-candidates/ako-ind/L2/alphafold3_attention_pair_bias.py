"""Attention with pair bias for AlphaFold3 -- launch-collapsed Triton kernels.

AttentionPairBias: Used in PairFormer and diffusion transformer. Uses a single
    layer_norm_a for both Q and K (AdaLN or LayerNorm).
CrossAttentionPairBias: Used in atom attention (sequence-local). Uses separate
    layer_norm_a_q and layer_norm_a_k, no layer_norm_z.

Reference: openfold3/core/model/layers/attention_pair_bias.py

Every captured workload is tiny (16 tokens, or 368 atoms in 12 blocks of
32x128) and the eager composition fires 23-117 kernels to move tens of KB, so
essentially all measured time is per-op CPU dispatch + launch overhead. The
whole forward is collapsed into 3 launches (AttentionPairBias) / 2 launches
(CrossAttentionPairBias): the norm/AdaLN chains, the concatenated q/k/v/g
projection, the pair + mask bias, softmax, linear_o and both output gates are
fused, and no [*, H, Q, K] bias tensor is ever handed back to eager code.

Launch *argument* count matters as much as launch count on this box (~8.2us
fixed + ~0.4us per argument for a `triton.jit` launch), so every weight lives in
one flat bf16 buffer and every intermediate in another, addressed by constexpr
offsets that cost nothing at launch time.

Submodule and parameter names/shapes are byte-identical to the baseline (the
bench shares weights via ``load_state_dict(..., strict=False)``); the packed
buffers are derived lazily on the first forward and invalidated by an identity
check on the source parameters.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import triton
import triton.language as tl

from ..L1.sigmoid import Sigmoid
from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear
from .alphafold3_swiglu import AdaLN
from .alphafold3_of3_attention import OF3Attention


def _permute_final_dims(tensor: torch.Tensor, inds: list[int]) -> torch.Tensor:
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


# Warp counts, kept here so a sweep is a one-line edit.
# Warp counts and tile widths, swept on B200 against candidate_ms.
_W_PREP = 8
_W_ATTN = 4
_W_OUT = 4
_W_CPREP = 16
_W_CATTN = 4


# ---------------------------------------------------------------------------
# Triton helpers
# ---------------------------------------------------------------------------
@triton.jit
def _rb(x):
    """Round an fp32 value through bf16, matching the eager op's output dtype.

    The reference chain rounds to bf16 after every op (F.linear, einsum,
    layer_norm, sigmoid, ...). Reproducing those rounding points keeps the fused
    kernels bit-close to the baseline rather than merely inside tolerance --
    AttentionPairBias comes out bit-exact.
    """
    return x.to(tl.bfloat16).to(tl.float32)


@triton.jit
def _sigmoid(x):
    return 1.0 / (1.0 + tl.exp(-x))


@triton.jit
def _layer_norm(x, C: tl.constexpr, EPS: tl.constexpr):
    """Row-wise fp32 normalization of a [M, C] tile, reduction kept in fp32 (the
    L1 LayerNorm promotes to fp32 on purpose). ``C`` must be the true row width:
    with a padded tile the padding lanes hold ``-mean`` and would inflate the
    variance, so callers with C != tile width use ``_layer_norm_p``."""
    xc = x - (tl.sum(x, 1) / C)[:, None]
    return xc * (1.0 / tl.sqrt(tl.sum(xc * xc, 1) / C + EPS))[:, None]


@triton.jit
def _row_stats(ptr, C: tl.constexpr, NP: tl.constexpr, N: tl.constexpr,
               BK: tl.constexpr, EPS: tl.constexpr):
    """(mean, rstd) per row of a [N, C] bf16 matrix, streamed in BK-wide tiles.

    Streaming beats loading one [NP, next_pow2(C)] tile: C is 384/768 here, so
    that tile would be a third padding, and the padding lanes have to be masked
    out of the variance -- masked loads cost vectorization on the *whole* tile.
    Two exact passes instead of E[x^2]-mean^2 keeps the reduction faithful to
    F.layer_norm at negligible cost (the rows sit in L1 after the first pass).
    """
    m = tl.arange(0, NP)
    mv = m < N
    s1 = tl.zeros((NP,), tl.float32)
    for k0 in range(0, C, BK):
        s1 += tl.sum(tl.load(ptr + m[:, None] * C + (k0 + tl.arange(0, BK))[None, :],
                             mask=mv[:, None], other=0.0).to(tl.float32), 1)
    mean = s1 / C
    s2 = tl.zeros((NP,), tl.float32)
    for k0 in range(0, C, BK):
        x = tl.load(ptr + m[:, None] * C + (k0 + tl.arange(0, BK))[None, :],
                    mask=mv[:, None], other=0.0).to(tl.float32) - mean[:, None]
        s2 += tl.sum(x * x, 1)
    return mean, 1.0 / tl.sqrt(s2 / C + EPS)


# ---------------------------------------------------------------------------
# AttentionPairBias kernels
#
# Both the packed weight buffer and the scratch buffer are addressed by offsets
# *derived* from the shape constexprs rather than passed in: every extra launch
# argument costs ~0.4us through Triton's binder (~0.13us even on the fast
# re-launch path), which is real money against a ~30us forward.
# ---------------------------------------------------------------------------
@triton.jit
def _apb_woff(CZ: tl.constexpr, H: tl.constexpr, CQ: tl.constexpr,
              HDP: tl.constexpr):
    """layer_norm_z weight/bias, linear_z^T, linear_q bias, the concatenated
    q/k/v/g weight, linear_o^T, and the start of the variant-specific tail."""
    o_bq: tl.constexpr = 2 * CZ + CZ * H
    o_w: tl.constexpr = o_bq + HDP
    o_wot: tl.constexpr = o_w + 4 * CQ * HDP
    return 0, CZ, 2 * CZ, o_bq, o_w, o_wot, o_wot + HDP * CQ


@triton.jit
def _apb_boff(NP: tl.constexpr, CQ: tl.constexpr, HDP: tl.constexpr,
              H: tl.constexpr):
    """Scratch: pair bias [H, NP, NP], a_norm, gated attention out, output gate."""
    b_an: tl.constexpr = H * NP * NP
    b_og: tl.constexpr = b_an + NP * CQ
    return 0, b_an, b_og, b_og + NP * HDP


@triton.jit
def _zbias_row(z_ptr, w_ptr, buf_ptr, q,
               N: tl.constexpr, NP: tl.constexpr, CZ: tl.constexpr,
               H: tl.constexpr, HP: tl.constexpr, O_LNZW: tl.constexpr,
               O_LNZB: tl.constexpr, O_WZT: tl.constexpr, B_ZB: tl.constexpr,
               HAS_LNZB: tl.constexpr, EPS: tl.constexpr):
    # bias is kept as a [H, NP, NP] bf16 plane inside the scratch buffer so the
    # attention kernel can read it without a mask; padding stays zero.
    """bias[h, q, :] = permute(linear_z(layer_norm_z(z)))[h, q, :]."""
    k = tl.arange(0, NP)
    kv = k < N
    c = tl.arange(0, CZ)
    z = tl.load(z_ptr + (q * N + k[:, None]) * CZ + c[None, :],
                mask=kv[:, None], other=0.0).to(tl.float32)
    y = _layer_norm(z, CZ, EPS) * tl.load(w_ptr + O_LNZW + c).to(tl.float32)[None, :]
    if HAS_LNZB:
        y = y + tl.load(w_ptr + O_LNZB + c).to(tl.float32)[None, :]
    h = tl.arange(0, HP)
    hv = h < H
    wzt = tl.load(w_ptr + O_WZT + c[:, None] * H + h[None, :],
                  mask=hv[None, :], other=0.0)
    b = _rb(tl.dot(_rb(y).to(tl.bfloat16), wzt))
    tl.store(buf_ptr + B_ZB + h[None, :] * (NP * NP) + q * NP + k[:, None],
             b.to(tl.bfloat16), mask=kv[:, None] & hv[None, :])


@triton.jit
def _apb_prep_ln(a_ptr, z_ptr, w_ptr, buf_ptr,
                 N: tl.constexpr, NP: tl.constexpr, CZ: tl.constexpr,
                 H: tl.constexpr, HP: tl.constexpr, CQ: tl.constexpr,
                 HDP: tl.constexpr, BK: tl.constexpr,
                 HAS_LNZB: tl.constexpr, EPS: tl.constexpr):
    """Plain-LayerNorm variant: pair bias (N programs) + layer_norm_a (1)."""
    O_LNZW, O_LNZB, O_WZT, _, _, _, O_TAIL = _apb_woff(CZ, H, CQ, HDP)
    O_LNAW = O_TAIL
    O_LNAB = O_TAIL + CQ
    B_ZB, B_AN, _, _ = _apb_boff(NP, CQ, HDP, H)
    pid = tl.program_id(0)
    if pid < N:
        _zbias_row(z_ptr, w_ptr, buf_ptr, pid, N, NP, CZ, H, HP,
                   O_LNZW, O_LNZB, O_WZT, B_ZB, HAS_LNZB, EPS)
    else:
        m = tl.arange(0, NP)
        mv = m < N
        mean, rstd = _row_stats(a_ptr, CQ, NP, N, BK, EPS)
        for k0 in range(0, CQ, BK):
            kk = k0 + tl.arange(0, BK)
            x = tl.load(a_ptr + m[:, None] * CQ + kk[None, :],
                        mask=mv[:, None], other=0.0).to(tl.float32)
            y = (x - mean[:, None]) * rstd[:, None]
            y = y * tl.load(w_ptr + O_LNAW + kk).to(tl.float32)[None, :]
            y = y + tl.load(w_ptr + O_LNAB + kk).to(tl.float32)[None, :]
            tl.store(buf_ptr + B_AN + m[:, None] * CQ + kk[None, :], y.to(tl.bfloat16))


@triton.jit
def _apb_prep_adaln(a_ptr, z_ptr, s_ptr, w_ptr, buf_ptr,
                    N: tl.constexpr, NP: tl.constexpr, CZ: tl.constexpr,
                    H: tl.constexpr, HP: tl.constexpr, CQ: tl.constexpr,
                    CS: tl.constexpr, HDP: tl.constexpr, BN: tl.constexpr,
                    BK: tl.constexpr, HAS_LNZB: tl.constexpr,
                    EPS: tl.constexpr):
    """AdaLN variant: pair bias (N programs) + AdaLN(a, s) and the trailing
    sigmoid(linear_ada_out(s)) gate (CQ/BN programs, one column tile each).

    layer_norm_s / layer_norm_a statistics are recomputed inside every column
    tile so the tiles stay independent -- a few thousand redundant loads buys
    one fewer launch, which is the expensive resource here.
    """
    O_LNZW, O_LNZB, O_WZT, _, _, _, O_TAIL = _apb_woff(CZ, H, CQ, HDP)
    O_LNSW = O_TAIL
    O_BG = O_TAIL + CS
    O_BAO = O_BG + CQ
    O_WGT = O_BAO + CQ
    O_WST = O_WGT + CS * CQ
    O_WAOT = O_WST + CS * CQ
    B_ZB, B_AN, _, B_GATE = _apb_boff(NP, CQ, HDP, H)
    pid = tl.program_id(0)
    if pid < N:
        _zbias_row(z_ptr, w_ptr, buf_ptr, pid, N, NP, CZ, H, HP,
                   O_LNZW, O_LNZB, O_WZT, B_ZB, HAS_LNZB, EPS)
    else:
        n = (pid - N) * BN + tl.arange(0, BN)
        m = tl.arange(0, NP)
        mv = m < N
        smean, srstd = _row_stats(s_ptr, CS, NP, N, BK, EPS)
        amean, arstd = _row_stats(a_ptr, CQ, NP, N, BK, EPS)

        accg = tl.zeros((NP, BN), tl.float32)
        accs = tl.zeros((NP, BN), tl.float32)
        acco = tl.zeros((NP, BN), tl.float32)
        for k0 in range(0, CS, BK):
            kk = k0 + tl.arange(0, BK)
            sc = tl.load(s_ptr + m[:, None] * CS + kk[None, :],
                         mask=mv[:, None], other=0.0)
            wl = tl.load(w_ptr + O_LNSW + kk).to(tl.float32)
            sn = _rb((sc.to(tl.float32) - smean[:, None]) * srstd[:, None]
                     * wl[None, :]).to(tl.bfloat16)
            wo = kk[:, None] * CQ + n[None, :]
            accg += tl.dot(sn, tl.load(w_ptr + O_WGT + wo))
            accs += tl.dot(sn, tl.load(w_ptr + O_WST + wo))
            acco += tl.dot(sc, tl.load(w_ptr + O_WAOT + wo))

        g = _rb(_sigmoid(_rb(accg + tl.load(w_ptr + O_BG + n).to(tl.float32)[None, :])))
        og = _rb(_sigmoid(_rb(acco + tl.load(w_ptr + O_BAO + n).to(tl.float32)[None, :])))
        aslice = tl.load(a_ptr + m[:, None] * CQ + n[None, :],
                         mask=mv[:, None], other=0.0).to(tl.float32)
        aln = _rb((aslice - amean[:, None]) * arstd[:, None])
        tl.store(buf_ptr + B_AN + m[:, None] * CQ + n[None, :],
                 _rb(g * _rb(aln + _rb(accs))).to(tl.bfloat16), mask=mv[:, None])
        tl.store(buf_ptr + B_GATE + m[:, None] * CQ + n[None, :],
                 og.to(tl.bfloat16), mask=mv[:, None])


@triton.jit
def _apb_attn(mask_ptr, w_ptr, buf_ptr,
              N: tl.constexpr, NP: tl.constexpr, CQ: tl.constexpr,
              HDP: tl.constexpr, CHP: tl.constexpr, CZ: tl.constexpr,
              H: tl.constexpr, BK: tl.constexpr, SCALE: tl.constexpr,
              INF: tl.constexpr, HAS_BQ: tl.constexpr,
              HAS_MASK: tl.constexpr):
    """One program per head: concatenated q/k/v/g projection, then QK^T + mask
    bias + pair bias + softmax + PV + output gate, entirely in registers.

    Every tile here is an exact power of two and every buffer is padded to match
    (``CHP`` >= c_hidden per head, scratch rows padded to ``NP``), so the inner
    GEMM issues unmasked vector loads. The padding columns of the projection
    weight are zero, so they produce an exactly-zero gated output and the
    matching zero rows of linear_o's weight ignore them.
    """
    _, _, _, O_BQ, O_W, _, _ = _apb_woff(CZ, H, CQ, HDP)
    B_ZB, B_AN, B_OG, _ = _apb_boff(NP, CQ, HDP, H)
    h = tl.program_id(0)
    m = tl.arange(0, NP)
    mv = m < N
    col = h * CHP + tl.arange(0, CHP)
    accq = tl.zeros((NP, CHP), tl.float32)
    acck = tl.zeros((NP, CHP), tl.float32)
    accv = tl.zeros((NP, CHP), tl.float32)
    accg = tl.zeros((NP, CHP), tl.float32)
    for k0 in range(0, CQ, BK):
        kk = k0 + tl.arange(0, BK)
        at = tl.load(buf_ptr + B_AN + m[:, None] * CQ + kk[None, :])
        wo = w_ptr + O_W + kk[:, None] * (4 * HDP) + col[None, :]
        accq += tl.dot(at, tl.load(wo))
        acck += tl.dot(at, tl.load(wo + HDP))
        accv += tl.dot(at, tl.load(wo + 2 * HDP))
        accg += tl.dot(at, tl.load(wo + 3 * HDP))
    if HAS_BQ:
        accq += tl.load(w_ptr + O_BQ + col).to(tl.float32)[None, :]
    q = _rb(_rb(accq) * SCALE)
    sc = _rb(tl.dot(q.to(tl.bfloat16), tl.trans(_rb(acck).to(tl.bfloat16))))
    if HAS_MASK:
        mk = tl.load(mask_ptr + m, mask=mv, other=1.0).to(tl.float32)
        sc = _rb(sc + _rb(INF * _rb(mk - 1.0))[None, :])
    sc = _rb(sc + tl.load(buf_ptr + B_ZB + h * (NP * NP) + m[:, None] * NP
                          + m[None, :]).to(tl.float32))
    sc = tl.where(mv[None, :], sc, -float("inf"))
    e = tl.exp(sc - tl.max(sc, 1)[:, None])
    p = _rb(e / tl.sum(e, 1)[:, None]).to(tl.bfloat16)
    o = _rb(tl.dot(p, _rb(accv).to(tl.bfloat16)))
    o = _rb(o * _rb(_sigmoid(_rb(accg))))
    tl.store(buf_ptr + B_OG + m[:, None] * HDP + col[None, :], o.to(tl.bfloat16))


@triton.jit
def _apb_out(out_ptr, w_ptr, buf_ptr,
             N: tl.constexpr, NP: tl.constexpr, HDP: tl.constexpr,
             CQ: tl.constexpr, CZ: tl.constexpr, H: tl.constexpr,
             BN: tl.constexpr, BK: tl.constexpr, HAS_GATE: tl.constexpr):
    """linear_o with the trailing AdaLN-Zero output gate in the epilogue."""
    _, _, _, _, _, O_WOT, _ = _apb_woff(CZ, H, CQ, HDP)
    _, _, B_OG, B_GATE = _apb_boff(NP, CQ, HDP, H)
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    m = tl.arange(0, NP)
    mv = m < N
    acc = tl.zeros((NP, BN), tl.float32)
    for k0 in range(0, HDP, BK):
        kk = k0 + tl.arange(0, BK)
        o = tl.load(buf_ptr + B_OG + m[:, None] * HDP + kk[None, :])
        acc += tl.dot(o, tl.load(w_ptr + O_WOT + kk[:, None] * CQ + n[None, :]))
    y = _rb(acc)
    if HAS_GATE:
        y = _rb(y * tl.load(buf_ptr + B_GATE + m[:, None] * CQ
                            + n[None, :]).to(tl.float32))
    tl.store(out_ptr + m[:, None] * CQ + n[None, :], y.to(tl.bfloat16),
             mask=mv[:, None])


# ---------------------------------------------------------------------------
# CrossAttentionPairBias kernels
# ---------------------------------------------------------------------------
@triton.jit
def _key_idx(mask_ptr, b, j, NA: tl.constexpr, NAPP: tl.constexpr,
             NQ: tl.constexpr, NK: tl.constexpr):
    """Sequence-local key gather indices, bit-matching the reference.

    ``_get_block_key_indices`` promotes its int32 index arithmetic to the mask's
    dtype (bf16 here): ``initial + total_shift`` is int32 + bf16 -> bf16. bf16
    holds integers exactly only to 256, so for 368 atoms the indices of the
    later blocks round to even and collapse onto each other (blocks 10/11 have
    65 distinct keys out of 128, and one index lands on n_real and is flagged
    invalid). That rounding decides *which atoms* a key block sees, so it is
    reproduced step-for-step -- computing the indices "correctly" in int32
    gathers different rows and fails correctness.
    """
    r = tl.arange(0, NAPP)
    nreal = _rb(tl.sum(tl.load(mask_ptr + r, mask=r < NA, other=0.0).to(tl.float32), 0))
    nm1 = _rb(nreal - 1.0)
    first = NQ // 2 + b * NQ - NK // 2
    underflow = tl.maximum(-first, 0).to(tl.float32)
    last = _rb((first + NK - 1).to(tl.float32))
    overflow = tl.maximum(_rb(last - nm1), 0.0)
    shift = tl.where(underflow > 0.0, _rb(underflow), -overflow)
    final = _rb(_rb((first + j).to(tl.float32)) + shift)
    invalid = (final < 0.0) | (final >= nreal)
    safe = tl.minimum(tl.maximum(final, 0.0), tl.maximum(nm1, 0.0))
    return safe.to(tl.int32), invalid


@triton.jit
def _adaln_tile(a_f, s_f, w_ptr, O_LNSW: tl.constexpr, O_WGT: tl.constexpr,
                O_BG: tl.constexpr, O_WST: tl.constexpr, C: tl.constexpr,
                EPS: tl.constexpr):
    """AdaLN over a [M, C] pair of (activation, conditioning) fp32 tiles."""
    c = tl.arange(0, C)
    sn = _rb(_layer_norm(s_f, C, EPS)
             * tl.load(w_ptr + O_LNSW + c).to(tl.float32)[None, :]).to(tl.bfloat16)
    wo = c[:, None] * C + c[None, :]
    g = _rb(_sigmoid(_rb(tl.dot(sn, tl.load(w_ptr + O_WGT + wo))
                         + tl.load(w_ptr + O_BG + c).to(tl.float32)[None, :])))
    add = _rb(tl.dot(sn, tl.load(w_ptr + O_WST + wo)))
    return _rb(g * _rb(_rb(_layer_norm(a_f, C, EPS)) + add))


@triton.jit
def _capb_wmat(C: tl.constexpr):
    """AdaLN_q/AdaLN_k linear_g / linear_s weights: slots 0..3 of the [C, C] stack."""
    return 0, C * C, 2 * C * C, 3 * C * C


@triton.jit
def _capb_wproj(C: tl.constexpr):
    """mha linear_q / linear_g / linear_k / linear_v and linear_ada_out: slots 4..8."""
    return 4 * C * C, 5 * C * C, 6 * C * C, 7 * C * C, 8 * C * C


@triton.jit
def _capb_wvec(C: tl.constexpr):
    """The [C] vectors (slot 9 of the stack is linear_o), then linear_z."""
    v: tl.constexpr = 10 * C * C
    return v, v + C, v + 2 * C, v + 3 * C, v + 4 * C, v + 5 * C, v + 6 * C


@triton.jit
def _capb_buf(NB: tl.constexpr, NQ: tl.constexpr, NK: tl.constexpr,
              HD: tl.constexpr, C: tl.constexpr, H: tl.constexpr):
    """Scratch layout: q, g, k, v, out-gate, pair bias, query mask, key mask."""
    q: tl.constexpr = NB * NQ * HD
    k: tl.constexpr = NB * NK * HD
    b_v: tl.constexpr = 2 * q + k
    b_gate: tl.constexpr = b_v + k
    b_zb: tl.constexpr = b_gate + NB * NQ * C
    b_mq: tl.constexpr = b_zb + NB * H * NQ * NK
    return 0, q, 2 * q, b_v, b_gate, b_zb, b_mq, b_mq + NB * NQ


@triton.jit
def _capb_prep(a_ptr, s_ptr, mask_ptr, z_ptr, w_ptr, buf_ptr,
               NA: tl.constexpr, NAPP: tl.constexpr, NB: tl.constexpr,
               NQ: tl.constexpr, NK: tl.constexpr, BKR: tl.constexpr,
               C: tl.constexpr, CH: tl.constexpr, H: tl.constexpr,
               HP: tl.constexpr, CZ: tl.constexpr, BR: tl.constexpr,
               SCALE: tl.constexpr, EPS: tl.constexpr):
    """Everything ahead of the attention itself, in one launch: key-block gather
    + AdaLN_k + k/v projection (NB*NKC programs), query AdaLN_q + q/g projection
    + output gate (NB programs), and the linear_z pair bias (NB programs).

    Section offsets are *derived* from the shape constexprs rather than passed
    in: 24 extra launch arguments cost ~3us of pure per-call overhead here,
    which is a tenth of the whole forward.
    """
    HD: tl.constexpr = CH * H
    NKC: tl.constexpr = NK // BKR
    NZC: tl.constexpr = (NQ * NK) // BR
    O_WGTQ, O_WSTQ, O_WGTK, O_WSTK = _capb_wmat(C)
    O_WQT, O_WG, O_WKT, O_WVT, O_WAOT = _capb_wproj(C)
    O_LNSWQ, O_BGQ, O_LNSWK, O_BGK, O_BQ, O_BAO, O_WZT = _capb_wvec(C)
    B_Q, B_G, B_K, B_V, B_GATE, B_ZB, B_MQ, B_KV = _capb_buf(NB, NQ, NK, HD, C, H)
    pid = tl.program_id(0)
    c = tl.arange(0, C)
    hd = tl.arange(0, HD)
    if pid < NB * NKC:
        b = pid // NKC
        j = (pid % NKC) * BKR + tl.arange(0, BKR)
        idx, invalid = _key_idx(mask_ptr, b, j, NA, NAPP, NQ, NK)
        amk = tl.load(mask_ptr + idx, mask=idx < NA, other=0.0).to(tl.float32)
        tl.store(buf_ptr + B_KV + b * NK + j,
                 _rb(tl.where(invalid, 0.0, 1.0) * amk).to(tl.bfloat16))
        ok = (~invalid) & (idx < NA)
        af = tl.load(a_ptr + idx[:, None] * C + c[None, :],
                     mask=ok[:, None], other=0.0).to(tl.float32)
        sf = tl.load(s_ptr + idx[:, None] * C + c[None, :],
                     mask=ok[:, None], other=0.0).to(tl.float32)
        akn = _adaln_tile(af, sf, w_ptr, O_LNSWK, O_WGTK, O_BGK, O_WSTK,
                          C, EPS).to(tl.bfloat16)
        wo = c[:, None] * HD + hd[None, :]
        off = (b * NK + j[:, None]) * HD + hd[None, :]
        tl.store(buf_ptr + B_K + off,
                 _rb(tl.dot(akn, tl.load(w_ptr + O_WKT + wo))).to(tl.bfloat16))
        tl.store(buf_ptr + B_V + off,
                 _rb(tl.dot(akn, tl.load(w_ptr + O_WVT + wo))).to(tl.bfloat16))
    elif pid < NB * NKC + NB:
        b = pid - NB * NKC
        i = tl.arange(0, NQ)
        r = b * NQ + i
        ok = r < NA
        tl.store(buf_ptr + B_MQ + r,
                 tl.load(mask_ptr + r, mask=ok, other=0.0))
        af = tl.load(a_ptr + r[:, None] * C + c[None, :],
                     mask=ok[:, None], other=0.0).to(tl.float32)
        sf = tl.load(s_ptr + r[:, None] * C + c[None, :],
                     mask=ok[:, None], other=0.0).to(tl.float32)
        aqn = _adaln_tile(af, sf, w_ptr, O_LNSWQ, O_WGTQ, O_BGQ, O_WSTQ,
                          C, EPS).to(tl.bfloat16)
        wo = c[:, None] * HD + hd[None, :]
        off = (b * NQ + i[:, None]) * HD + hd[None, :]
        q = _rb(tl.dot(aqn, tl.load(w_ptr + O_WQT + wo))
                + tl.load(w_ptr + O_BQ + hd).to(tl.float32)[None, :])
        tl.store(buf_ptr + B_Q + off, _rb(q * SCALE).to(tl.bfloat16))
        tl.store(buf_ptr + B_G + off,
                 _rb(tl.dot(aqn, tl.load(w_ptr + O_WG + wo))).to(tl.bfloat16))
        og = _rb(tl.dot(sf.to(tl.bfloat16),
                        tl.load(w_ptr + O_WAOT + c[:, None] * C + c[None, :]))
                 + tl.load(w_ptr + O_BAO + c).to(tl.float32)[None, :])
        tl.store(buf_ptr + B_GATE + (b * NQ + i[:, None]) * C + c[None, :],
                 _rb(_sigmoid(og)).to(tl.bfloat16))
    else:
        b = pid - NB * NKC - NB
        cz = tl.arange(0, CZ)
        hb = tl.arange(0, HP)
        wz = tl.load(w_ptr + O_WZT + cz[:, None] * H + hb[None, :],
                     mask=hb[None, :] < H, other=0.0)
        for ci in range(NZC):
            r = ci * BR + tl.arange(0, BR)
            zt = tl.load(z_ptr + (b * (NQ * NK) + r[:, None]) * CZ + cz[None, :])
            tl.store(buf_ptr + B_ZB + b * (H * NQ * NK) + hb[None, :] * (NQ * NK)
                     + r[:, None], _rb(tl.dot(zt, wz)).to(tl.bfloat16),
                     mask=hb[None, :] < H)


@triton.jit
def _capb_attn(out_ptr, w_ptr, buf_ptr,
               NA: tl.constexpr, NQ: tl.constexpr, NK: tl.constexpr,
               C: tl.constexpr, CH: tl.constexpr, H: tl.constexpr,
               NB: tl.constexpr, INF: tl.constexpr):
    """One program per key block: biased softmax for every head, output gate,
    linear_o and the trailing AdaLN-Zero gate, all without leaving registers."""
    HD: tl.constexpr = CH * H
    O_WOT: tl.constexpr = 9 * C * C
    B_Q, B_G, B_K, B_V, B_GATE, B_ZB, B_MQ, B_KV = _capb_buf(NB, NQ, NK, HD, C, H)
    b = tl.program_id(0)
    i = tl.arange(0, NQ)
    j = tl.arange(0, NK)
    c = tl.arange(0, C)
    mq = tl.load(buf_ptr + B_MQ + b * NQ + i).to(tl.float32)
    kvd = tl.load(buf_ptr + B_KV + b * NK + j).to(tl.float32)
    mb = _rb(INF * _rb(_rb(mq[:, None] * kvd[None, :]) - 1.0))
    acc = tl.zeros((NQ, C), tl.float32)
    for h in range(H):
        cc = h * CH + tl.arange(0, CH)
        qh = tl.load(buf_ptr + B_Q + (b * NQ + i[:, None]) * HD + cc[None, :])
        kh = tl.load(buf_ptr + B_K + (b * NK + j[:, None]) * HD + cc[None, :])
        sc = _rb(_rb(tl.dot(qh, tl.trans(kh))) + mb)
        sc = _rb(sc + tl.load(buf_ptr + B_ZB + (b * H + h) * (NQ * NK)
                              + i[:, None] * NK + j[None, :]).to(tl.float32))
        e = tl.exp(sc - tl.max(sc, 1)[:, None])
        p = _rb(e / tl.sum(e, 1)[:, None]).to(tl.bfloat16)
        vh = tl.load(buf_ptr + B_V + (b * NK + j[:, None]) * HD + cc[None, :])
        gh = tl.load(buf_ptr + B_G + (b * NQ + i[:, None]) * HD + cc[None, :])
        oh = _rb(_rb(tl.dot(p, vh)) * _rb(_sigmoid(gh.to(tl.float32))))
        acc += tl.dot(oh.to(tl.bfloat16),
                      tl.load(w_ptr + O_WOT + cc[:, None] * C + c[None, :]))
    y = _rb(_rb(acc) * tl.load(buf_ptr + B_GATE + (b * NQ + i[:, None]) * C
                               + c[None, :]).to(tl.float32))
    r = b * NQ + i
    tl.store(out_ptr + r[:, None] * C + c[None, :], y.to(tl.bfloat16),
             mask=(r < NA)[:, None])


# ---------------------------------------------------------------------------
# Host-side plan construction
# ---------------------------------------------------------------------------
def _p2(n: int, lo: int = 1) -> int:
    v = 1
    while v < n:
        v *= 2
    return max(v, lo)


def _tc(w: torch.Tensor) -> torch.Tensor:
    return w.t().contiguous()


def _pad_in(w: torch.Tensor, H: int, CH: int, CHP: int) -> torch.Tensor:
    """[H*CH, C] projection weight -> [C, H*CHP] with each head's slice zero-padded.

    Padding the per-head hidden dimension up to a power of two lets the fused
    projection load its weight tile without a column mask; the zero columns
    yield an exactly-zero gated output, which linear_o's matching zero rows
    then ignore.
    """
    C = w.shape[1]
    t = w.new_zeros((H, CHP, C))
    t[:, :CH, :] = w.reshape(H, CH, C)
    return t.reshape(H * CHP, C).t().contiguous()


def _pad_out(w: torch.Tensor, H: int, CH: int, CHP: int) -> torch.Tensor:
    """[C, H*CH] linear_o weight -> [H*CHP, C], zero rows at the head padding."""
    C = w.shape[0]
    t = w.new_zeros((C, H, CHP))
    t[:, :, :CH] = w.reshape(C, H, CH)
    return t.reshape(C, H * CHP).t().contiguous()


def _pad_vec(v: torch.Tensor | None, H: int, CH: int, CHP: int):
    if v is None:
        return None
    t = v.new_zeros((H, CHP))
    t[:, :CH] = v.reshape(H, CH)
    return t.reshape(-1)


def _cg(*t: torch.Tensor) -> torch.Tensor:
    return torch.cat(t, dim=1).contiguous()


def _ct(t):
    if t is None or t.is_contiguous():
        return t
    return t.contiguous()


class _Seq:
    """A fixed sequence of Triton launches with a fast re-entry path.

    Each step is ``[jit_fn, grid, num_warps, sel, static_args]``, where *sel*
    picks this step's per-call tensors out of ``(a, z, s, mask, out)``; they are
    declared first in every kernel signature so they occupy the leading slots of
    Triton's bound-argument list.

    ``fn[grid](...)`` costs ~11.5us for these kernels -- roughly 8us fixed plus
    0.4us for every *declared* parameter, constexprs included -- and all of it
    is Python-side specialization work that is invariant here: dtypes, shapes
    and pointer alignments are fixed once the plan exists. So the first call
    goes through the normal JIT path (which compiles and records the
    specialization) and later calls re-enter the compiled kernel's launcher
    directly, at ~5.8us. Pointer alignment is the one specialized property a
    later call could still violate, so it is re-checked per call and the JIT
    path is used whenever the check fails (or whenever launch hooks are
    installed, since the fast path does not run them).
    """

    __slots__ = ("steps", "fast")

    def __init__(self, steps):
        self.steps = steps
        self.fast = None

    def slow(self, t) -> None:
        for fn, grid, nw, sel, sa in self.steps:
            fn[grid](*[t[i] for i in sel], *sa, num_warps=nw)

    def _bind(self, t):
        try:
            from triton import knobs
            from triton.runtime.driver import driver
            for hook in (knobs.runtime.launch_enter_hook, knobs.runtime.launch_exit_hook):
                if getattr(hook, "calls", True):
                    return None
            dev = driver.active.get_current_device()
            gcs = driver.active.get_current_stream
            fast = []
            for fn, grid, nw, sel, sa in self.steps:
                args = [t[i] for i in sel] + list(sa)
                ck = fn[grid](*args, num_warps=nw)
                if ck is None:
                    return None
                if hasattr(ck, "result"):
                    ck = ck.result()
                binder = fn.device_caches[dev][-1]
                bound, spec, _ = binder(*args, num_warps=nw)
                vals = list(bound.values())
                k = len(sel)
                if len(vals) < len(args) or any(vals[i] is not args[i] for i in range(k)):
                    return None
                if any(sp[1] != "D" for sp in spec[:k]):
                    return None
                g = tuple(grid) + (1, 1)
                fast.append((ck.run, g[0], g[1], g[2], ck.function,
                             ck.packed_metadata, vals, sel, k))
            return (fast, gcs, dev)
        except Exception:
            return None

    def __call__(self, t) -> None:
        f = self.fast
        if f is None:
            self.slow(t)
            self.fast = self._bind(t) or False
            return
        if f is False:
            self.slow(t)
            return
        for x in t:
            if x is not None and (x.data_ptr() & 15):
                self.slow(t)
                return
        steps, gcs, dev = f
        stream = gcs(dev)
        for run, g0, g1, g2, cfn, pm, vals, sel, k in steps:
            for i in range(k):
                vals[i] = t[sel[i]]
            run(g0, g1, g2, stream, cfn, pm, None, None, None, *vals)


class AttentionPairBias(nn.Module):
    """AF3 Algorithm 24: Attention with pair bias.

    When use_ada_layer_norm is True, uses two separate AdaLN instances
    (layer_norm_a_q, layer_norm_a_k) for query and key normalization,
    plus a linear_ada_out for output gating.

    Reference: openfold3/core/model/layers/attention_pair_bias.py

    Args:
        c_q: Input dimension of query/key/value
        c_s: Single activation channel dimension (for AdaLN)
        c_z: Pair activation channel dimension
        c_hidden: Per-head hidden dimension
        no_heads: Number of attention heads
        use_ada_layer_norm: Whether to use AdaLN-Zero conditioning
        gating: Whether to gate output
        inf: Large constant for masking
    """

    def __init__(
        self,
        c_q: int,
        c_k: int = 0,
        c_v: int = 0,
        c_s: int = 0,
        c_z: int = 128,
        c_hidden: int = 32,
        no_heads: int = 4,
        use_ada_layer_norm: bool = False,
        gating: bool = True,
        inf: float = 1e9,
    ):
        super().__init__()
        c_k = c_k or c_q
        c_v = c_v or c_q

        self.c_q = c_q
        self.c_s = c_s
        self.c_z = c_z
        self.inf = inf
        self.use_ada_layer_norm = use_ada_layer_norm

        if use_ada_layer_norm:
            self.layer_norm_a = AdaLN(c_a=c_q, c_s=c_s)
            self.linear_ada_out = Linear(c_s, c_q, bias=True)
        else:
            self.layer_norm_a = LayerNorm(c_q)

        self.layer_norm_z = LayerNorm(
            c_z, create_offset=not use_ada_layer_norm,
        )
        self.linear_z = Linear(c_z, no_heads, bias=False)

        self.sigmoid = Sigmoid()

        self.mha = OF3Attention(
            c_q=c_q, c_k=c_k, c_v=c_v,
            c_hidden=c_hidden, no_heads=no_heads, gating=gating,
            q_bias=True,
        )
        self._plan = None

    # -- reference path (any shape / dtype the fused path rejects) -----------
    def _prep_bias(
        self, a: torch.Tensor, z: torch.Tensor, mask: torch.Tensor | None,
    ) -> list[torch.Tensor]:
        if mask is None:
            mask = a.new_ones(a.shape[:-1])

        batch_dims = a.shape[:-2]
        mask = mask.expand((*batch_dims, -1))

        mask_bias = (self.inf * (mask - 1))[..., None, None, :]
        biases = [mask_bias]

        z = self.layer_norm_z(z)
        z = self.linear_z(z)
        z = _permute_final_dims(z, [2, 0, 1])
        biases.append(z)

        return biases

    def _ref_forward(self, a, z, s, mask):
        biases = self._prep_bias(a=a, z=z, mask=mask)
        a = self.layer_norm_a(a, s) if self.use_ada_layer_norm else self.layer_norm_a(a)
        a = self.mha(q_x=a, kv_x=a, biases=biases)
        if self.use_ada_layer_norm:
            a = self.sigmoid(self.linear_ada_out(s)) * a
        return a

    # -- fused path ---------------------------------------------------------
    def _make_plan(self, a, z, s, mask):
        mha = self.mha
        dt = a.dtype
        if (mha.linear_g is None or a.device.type != "cuda"
                or dt is not torch.bfloat16):
            return None
        N, CQ, CZ, H = a.shape[-2], self.c_q, self.c_z, mha.no_heads
        CH, HD = mha.c_hidden, mha.c_hidden * mha.no_heads
        ada = self.use_ada_layer_norm
        if (a.shape[-1] != CQ or a.numel() != N * CQ or N < 1
                or tuple(z.shape[-3:]) != (N, N, CZ) or z.numel() != N * N * CZ
                or z.dtype is not dt or CZ != _p2(CZ)
                or mha.c_k != CQ or mha.c_v != CQ):
            return None
        if mask is not None and (mask.dtype is not dt or mask.numel() != N):
            return None
        if ada and (s is None or s.dtype is not dt or s.shape[-1] != self.c_s
                    or s.numel() != N * self.c_s):
            return None
        dev = a.device
        NP, HP, CHP = _p2(N), _p2(H, 16), _p2(CH, 16)
        HDP, EPS = H * CHP, self.layer_norm_z.eps
        BN, BK, BNO = 64, 128, 64
        CS = self.c_s if ada else CQ
        # Every tiled dimension must divide exactly: the kernels drop their
        # column masks in exchange for unmasked vector loads.
        if CQ % BK or CQ % BN or CQ % BNO or CS % BK or HDP % BK:
            return None
        lnzb = self.layer_norm_z.bias
        haslnzb = lnzb is not None
        bq = mha.linear_q.bias
        hasbq = bq is not None
        lnzw = self.layer_norm_z.weight
        # Packed weights, in the order _apb_woff() reconstructs: layer_norm_z
        # weight and bias (the bias slot is always present, zero-filled when the
        # variant has none), linear_z^T, linear_q's head-padded bias, the
        # concatenated head-padded q/k/v/g weight, linear_o^T, then the
        # variant-specific tail.
        head = [lnzw,
                lnzb if haslnzb else torch.zeros_like(lnzw),
                _tc(self.linear_z.weight),
                _pad_vec(bq, H, CH, CHP) if hasbq
                else torch.zeros(HDP, dtype=dt, device=dev),
                _cg(*[_pad_in(w, H, CH, CHP) for w in
                      (mha.linear_q.weight, mha.linear_k.weight,
                       mha.linear_v.weight, mha.linear_g.weight)]),
                _pad_out(mha.linear_o.weight, H, CH, CHP)]
        # Scratch, laid out by _apb_boff().
        BUF = torch.zeros(H * NP * NP + 2 * NP * CQ + NP * HDP,
                          dtype=dt, device=dev)
        scale, inf = 1.0 / math.sqrt(CH), self.inf
        gattn, gout = (H,), (CQ // BNO,)
        hasm = mask is not None
        if ada:
            ln = self.layer_norm_a
            if ln.linear_g.bias is None or self.linear_ada_out.bias is None:
                return None
            W = torch.cat([t.reshape(-1) for t in head + [
                ln.layer_norm_s.weight, ln.linear_g.bias,
                self.linear_ada_out.bias, _tc(ln.linear_g.weight),
                _tc(ln.linear_s.weight), _tc(self.linear_ada_out.weight)]])
            seq = _Seq([
                [_apb_prep_adaln, (N + CQ // BN,), _W_PREP, (0, 1, 2),
                 (W, BUF, N, NP, CZ, H, HP, CQ, CS, HDP, BN, BK, haslnzb, EPS)],
                [_apb_attn, gattn, _W_ATTN, (3,) if hasm else (0,),
                 (W, BUF, N, NP, CQ, HDP, CHP, CZ, H, BK, scale, inf, hasbq,
                  hasm)],
                [_apb_out, gout, _W_OUT, (4,),
                 (W, BUF, N, NP, HDP, CQ, CZ, H, BNO, BK, True)],
            ])
        else:
            lnaw, lnab = self.layer_norm_a.weight, self.layer_norm_a.bias
            if lnaw is None or lnab is None:
                return None
            W = torch.cat([t.reshape(-1) for t in head + [lnaw, lnab]])
            seq = _Seq([
                [_apb_prep_ln, (N + 1,), _W_PREP, (0, 1),
                 (W, BUF, N, NP, CZ, H, HP, CQ, HDP, BK, haslnzb, EPS)],
                [_apb_attn, gattn, _W_ATTN, (3,) if hasm else (0,),
                 (W, BUF, N, NP, CQ, HDP, CHP, CZ, H, BK, scale, inf, hasbq,
                  hasm)],
                [_apb_out, gout, _W_OUT, (4,),
                 (W, BUF, N, NP, HDP, CQ, CZ, H, BNO, BK, False)],
            ])
        return [mha.linear_q.weight, N, dt, seq, dev, tuple(a.shape)]

    def forward(
        self,
        a: torch.Tensor,
        z: torch.Tensor,
        s: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        use_deepspeed_evo_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_lma: bool = False,
        use_high_precision_attention: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            a:    [*, N, C_q] token/atom-level embedding
            z:    [*, N, N, C_z] pair embedding
            s:    [*, N, C_s] single embedding (for AdaLN)
            mask: [*, N] mask

        Returns:
            [*, N, C_q] attention update
        """
        p = self._plan
        if (p is None or p[0] is not self.mha.linear_q.weight
                or p[1] != a.shape[-2] or p[2] is not a.dtype):
            p = self._make_plan(a, z, s, mask)
            if p is None:
                return self._ref_forward(a, z, s, mask)
            self._plan = p
        out = torch.empty(p[5], dtype=p[2], device=p[4])
        p[3]((_ct(a), _ct(z), _ct(s), _ct(mask), out))
        return out


class CrossAttentionPairBias(nn.Module):
    """AF3 Algorithm 24: Cross-attention with pair bias for atom attention.

    Uses separate layer_norm_a_q and layer_norm_a_k for query/key, and
    does NOT apply layer_norm_z (pair bias goes through linear_z directly).
    Handles sequence-local blocked inputs.

    Reference: openfold3/core/model/layers/attention_pair_bias.py CrossAttentionPairBias

    Args:
        c_q: Input dimension of query/key/value
        c_s: Single activation channel dimension (for AdaLN)
        c_z: Pair activation channel dimension
        c_hidden: Per-head hidden dimension
        no_heads: Number of attention heads
        use_ada_layer_norm: Whether to use AdaLN-Zero conditioning
        n_query: Block size for queries
        n_key: Block size for keys
        gating: Whether to gate output
        inf: Large constant for masking
    """

    def __init__(
        self,
        c_q: int,
        c_k: int = 0,
        c_v: int = 0,
        c_s: int = 0,
        c_z: int = 16,
        c_hidden: int = 16,
        no_heads: int = 4,
        use_ada_layer_norm: bool = True,
        n_query: int | None = None,
        n_key: int | None = None,
        gating: bool = True,
        inf: float = 1e9,
    ):
        super().__init__()
        c_k = c_k or c_q
        c_v = c_v or c_q

        self.c_q = c_q
        self.c_s = c_s
        self.c_z = c_z
        self.inf = inf
        self.use_ada_layer_norm = use_ada_layer_norm
        self.n_query = n_query
        self.n_key = n_key

        if use_ada_layer_norm:
            self.layer_norm_a_q = AdaLN(c_a=c_q, c_s=c_s)
            self.layer_norm_a_k = AdaLN(c_a=c_q, c_s=c_s)
            self.linear_ada_out = Linear(c_s, c_q, bias=True)
        else:
            self.layer_norm_a_q = LayerNorm(c_q)
            self.layer_norm_a_k = LayerNorm(c_q)

        self.linear_z = Linear(c_z, no_heads, bias=False)

        self.sigmoid = Sigmoid()

        self.mha = OF3Attention(
            c_q=c_q, c_k=c_k, c_v=c_v,
            c_hidden=c_hidden, no_heads=no_heads, gating=gating,
            q_bias=True,
        )
        self._plan = None

    # -- reference path -----------------------------------------------------
    def _ref_forward(self, a, z, s, mask):
        from .alphafold3_atom_attention import (
            _convert_single_rep_to_blocks, _apply_block_indices,
        )

        batch_dims = a.shape[:-2]
        n_atom, n_dim = a.shape[-2:]

        if mask is None:
            mask = a.new_ones(a.shape[:-1])

        a_query, a_key, block_mask = _convert_single_rep_to_blocks(
            ql=a, n_query=self.n_query, n_key=self.n_key, atom_mask=mask,
        )

        mask_bias = (self.inf * (block_mask - 1))[..., None, :, :]
        biases = [mask_bias]

        z_bias = self.linear_z(z)
        z_bias = _permute_final_dims(z_bias, [2, 0, 1])
        biases.append(z_bias)

        if self.use_ada_layer_norm:
            s_q, s_k, _ = _apply_block_indices(
                ql=s, n_query=self.n_query, n_key=self.n_key, atom_mask=mask,
            )
            a_q = self.layer_norm_a_q(a_query, s_q)
            a_k = self.layer_norm_a_k(a_key, s_k)
        else:
            a_q = self.layer_norm_a_q(a_query)
            a_k = self.layer_norm_a_k(a_key)

        a_out = self.mha(q_x=a_q, kv_x=a_k, biases=biases)

        a_out = a_out.reshape((*batch_dims, -1, n_dim))[..., :n_atom, :]

        if self.use_ada_layer_norm:
            a_out = self.sigmoid(self.linear_ada_out(s)) * a_out

        return a_out

    # -- fused path ---------------------------------------------------------
    def _make_plan(self, a, z, s, mask):
        mha = self.mha
        NQ, NK = self.n_query, self.n_key
        if (mha.linear_g is None or not self.use_ada_layer_norm
                or NQ is None or NK is None or a.device.type != "cuda"
                or mask is None or s is None or mha.linear_q.bias is None):
            return None
        dt = a.dtype
        if dt is not torch.bfloat16 or z.dtype is not dt or s.dtype is not dt:
            return None
        C, CZ, H, CH = self.c_q, self.c_z, mha.no_heads, mha.c_hidden
        HD = CH * H
        NA = a.shape[-2]
        NB = -(-NA // NQ)
        if (self.c_s != C or mha.c_k != C or mha.c_v != C or a.shape[-1] != C
                or a.numel() != NA * C or s.numel() != NA * C
                or s.shape[-1] != C or mask.numel() != NA
                or mask.dtype is not dt
                or tuple(z.shape[-4:]) != (NB, NQ, NK, CZ)
                or z.numel() != NB * NQ * NK * CZ):
            return None
        if (C != _p2(C) or HD != _p2(HD) or NQ != _p2(NQ) or NK != _p2(NK)
                or CZ != _p2(CZ) or NQ < 16 or NK < 16 or CH < 16
                or (NQ * NK) % 512 != 0):
            return None
        lq, lk = self.layer_norm_a_q, self.layer_norm_a_k
        if (lq.linear_g.bias is None or lk.linear_g.bias is None
                or self.linear_ada_out.bias is None or HD != C):
            return None
        dev = a.device
        BKR, BR = 32, 256
        NKC = NK // BKR
        NAPP, HP, EPS = _p2(NA), _p2(H, 16), lq.layer_norm_a.eps
        # Packed weights: ten [C, C] matrices, then six [C] vectors, then
        # linear_z. The kernels rebuild these offsets from C alone -- see
        # _capb_wmat / _capb_wproj / _capb_wvec, and keep the order in sync.
        W = torch.cat([m.reshape(-1) for m in (
            _tc(lq.linear_g.weight), _tc(lq.linear_s.weight),
            _tc(lk.linear_g.weight), _tc(lk.linear_s.weight),
            _tc(mha.linear_q.weight), _tc(mha.linear_g.weight),
            _tc(mha.linear_k.weight), _tc(mha.linear_v.weight),
            _tc(self.linear_ada_out.weight), _tc(mha.linear_o.weight),
            lq.layer_norm_s.weight, lq.linear_g.bias,
            lk.layer_norm_s.weight, lk.linear_g.bias,
            mha.linear_q.bias, self.linear_ada_out.bias,
            _tc(self.linear_z.weight))])
        # Scratch, laid out by _capb_buf.
        BUF = torch.empty(2 * NB * NQ * HD + 2 * NB * NK * HD + NB * NQ * C
                          + NB * H * NQ * NK + NB * NQ + NB * NK,
                          dtype=dt, device=dev)
        scale, inf = 1.0 / math.sqrt(CH), self.inf
        seq = _Seq([
            [_capb_prep, (NB * NKC + 2 * NB,), _W_CPREP, (0, 2, 3, 1),
             (W, BUF, NA, NAPP, NB, NQ, NK, BKR, C, CH, H, HP, CZ, BR,
              scale, EPS)],
            [_capb_attn, (NB,), _W_CATTN, (4,),
             (W, BUF, NA, NQ, NK, C, CH, H, NB, inf)],
        ])
        return [mha.linear_q.weight, NA, dt, seq, dev, tuple(a.shape)]

    def forward(
        self,
        a: torch.Tensor,
        z: torch.Tensor,
        s: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            a:    [*, N_atom, C_q] atom-level embedding
            z:    [*, N_blocks, n_key, n_key, C_z] blocked pair embedding
            s:    [*, N_atom, C_s] single embedding (for AdaLN)
            mask: [*, N_atom] mask

        Returns:
            [*, N_atom, C_q] attention update
        """
        p = self._plan
        if (p is None or p[0] is not self.mha.linear_q.weight
                or p[1] != a.shape[-2] or p[2] is not a.dtype):
            p = self._make_plan(a, z, s, mask)
            if p is None:
                return self._ref_forward(a, z, s, mask)
            self._plan = p
        out = torch.empty(p[5], dtype=p[2], device=p[4])
        p[3]((_ct(a), _ct(z), _ct(s), _ct(mask), out))
        return out
