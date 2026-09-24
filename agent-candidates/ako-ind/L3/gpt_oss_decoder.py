"""GPT-OSS decoder layer: attention + MoE with RMSNorm residual connections.

Decode-path specialisation.  The captured workload is dominated by tiny-token
decode steps (T = 1 / 26 / 60 / 274) where the layer is latency- and
bandwidth-bound, not compute-bound, and where per-op eager dispatch is a large
fraction of the measured wall time.  For those shapes the MoE runs through a
hand-written Triton pipeline (routing -> gather -> two block-scaled MXFP4
GEMMs -> weighted combine) over a repacked, *unpadded* expert layout, replacing
the trtllm-gen fused-MoE call and its Python wrapper.

The trtllm-gen path is kept verbatim for prefill-sized batches, where the
grouped GEMM is compute-bound and its tiling wins.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn

from ..L1.rms_norm import RMSNorm
from ..L2.attention import LlamaAttention
from ..L2.gpt_oss_moe import GptOssMoE

os.environ.setdefault("TRITON_CACHE_DIR", os.path.expanduser("~/.triton_cache_fk"))

import triton
import triton.language as tl

# --------------------------------------------------------------------------
# Repacked MXFP4 expert layout.
#
# The checkpoint format is [E, N, K/2] uint8 (two e2m1 nibbles per byte, along
# K) plus [E, N, K/32] E8M0 block scales, stored into a 256-aligned buffer
# (hidden 2880 -> 3072) because that is what trtllm-gen wants.  Our GEMM wants
# a *tile-major* layout so every CTA load is one contiguous run, and it does
# not need 256 alignment: K is padded only to a multiple of BK (2880 -> 2944,
# +2.2%) instead of to 3072 (+6.7%).
#
#   W : [E, NB, KB, BN, BK/2] uint8   tile (nb, kb) is BN*BK/2 contiguous bytes
#   S : [E, NB, KB, BN, BK/32] uint8
#
# For GEMM1 the 2*I rows are re-ordered so each BN=128 tile holds 64 gate rows
# followed by the 64 matching up rows -- SwiGLU then needs no cross-lane data
# movement, and 2880 = 45 * 64 divides exactly (no row padding at all).
# --------------------------------------------------------------------------
BN = 128
BK = 128
BM = 16                      # token slots per expert tile
MOE_SMALL_M = 512            # above this, fall back to trtllm-gen
USE_GRAPH = os.environ.get("FK_GPTOSS_GRAPH", "1") == "1"


def _pad_to(x: int, m: int) -> int:
    return (x + m - 1) // m * m


@triton.jit
def _cvt_fp4x2(w):
    """Two e2m1 nibbles per uint8 -> two fp16 tiles (native SM100 cvt)."""
    return tl.inline_asm_elementwise(
        """
        {
          .reg .b8  lo8, hi8;
          .reg .b32 t;
          mov.b16 {lo8, hi8}, $2;
          cvt.rn.f16x2.e2m1x2 t, lo8;
          mov.b32 {$0, $1}, t;
        }
        """,
        "=h,=h,h", [w], dtype=(tl.float16, tl.float16), is_pure=True, pack=1)


@triton.jit
def _load_wf(W, S, woff, soff, BN_: tl.constexpr, BK_: tl.constexpr):
    """One [BN, BK] fp16 weight tile from the packed fp4 + E8M0 scale tiles."""
    r = tl.arange(0, BN_)[:, None]
    w = tl.load(W + woff + r * (BK_ // 2) + tl.arange(0, BK_ // 2)[None, :])
    s = tl.load(S + soff + r * (BK_ // 32) + tl.arange(0, BK_ // 32)[None, :])
    lo, hi = _cvt_fp4x2(w.to(tl.uint16))
    j = tl.join(lo, hi)
    wf = tl.reshape(j, (BN_, BK_))
    sc = (s.to(tl.uint16) << 10).to(tl.float16, bitcast=True)
    sc = tl.reshape(tl.broadcast_to(sc[:, :, None], (BN_, BK_ // 32, 32)), (BN_, BK_))
    return wf * sc


@triton.jit
def _tile_lookup(CNT, pid_p, E: tl.constexpr, BM_: tl.constexpr):
    """(expert, first slot, valid slots) for flat tile index ``pid_p``.

    Recomputed per CTA from the 128-entry expert histogram: cheaper than a
    launch-visible prefix sum, and it keeps the whole MoE free of any
    device->host sync."""
    idx = tl.arange(0, E)
    cnt = tl.load(CNT + idx)
    ntl = (cnt + (BM_ - 1)) // BM_
    cs = tl.cumsum(ntl, 0)
    before = cs <= pid_p
    e = tl.sum(before.to(tl.int32))
    base = tl.sum(tl.where(before, ntl, 0))
    local = pid_p - base
    ce = tl.sum(tl.where(idx == e, cnt, 0))
    return e, local * BM_, ce - local * BM_


# --------------------------------------------------------------------------
# 1) routing: top-k softmax over the router logits + per-expert bins
# --------------------------------------------------------------------------
@triton.jit
def _route(LOGITS, CNT, BIN, TOPE, TOPP, TOPW, M, stride_l,
           E: tl.constexpr, TOPK: tl.constexpr, BINCAP: tl.constexpr):
    t = tl.program_id(0)
    if t >= M:
        return
    idx = tl.arange(0, E)
    # The reference router is an F.linear with a bf16 output, and trtllm-gen
    # routes on those bf16 logits; round to bf16 so top-k tie-breaking matches.
    lg = tl.load(LOGITS + t * stride_l + idx).to(tl.bfloat16).to(tl.float32)
    mx = tl.max(lg, 0)
    acc = lg
    wsum = tl.zeros((), dtype=tl.float32)
    for j in tl.static_range(TOPK):
        m = tl.max(acc, 0)
        e = tl.min(tl.where(acc == m, idx, E), 0)
        wsum += tl.exp(m - mx)
        acc = tl.where(idx == e, float("-inf"), acc)
        tl.store(TOPE + t * TOPK + j, e.to(tl.int32))
        tl.store(TOPW + t * TOPK + j, tl.exp(m - mx))
    for j in tl.static_range(TOPK):
        e = tl.load(TOPE + t * TOPK + j)
        tl.store(TOPW + t * TOPK + j, tl.load(TOPW + t * TOPK + j) / wsum)
        p = tl.atomic_add(CNT + e, 1)
        tl.store(BIN + e * BINCAP + p, t)
        tl.store(TOPP + t * TOPK + j, p)


# --------------------------------------------------------------------------
# 2) gather: expert tiles of activations, transposed to [K, slot]
# --------------------------------------------------------------------------
@triton.jit
def _gather(X, XG, CNT, BIN, stride_x, KVAL,
            KP: tl.constexpr, E: tl.constexpr, BM_: tl.constexpr,
            BK_: tl.constexpr, BINCAP: tl.constexpr):
    pid_k = tl.program_id(0)
    pid_p = tl.program_id(1)
    e, slot0, nval = _tile_lookup(CNT, pid_p, E, BM_)
    if e < E:
        s = tl.arange(0, BM_)
        ok = s < nval
        tok = tl.load(BIN + e * BINCAP + slot0 + s, mask=ok, other=0)
        k = pid_k * BK_ + tl.arange(0, BK_)
        v = tl.load(X + tok[:, None] * stride_x + k[None, :],
                    mask=ok[:, None] & (k[None, :] < KVAL), other=0.0).to(tl.float16)
        tl.store(XG + pid_p * (KP * BM_) + k[:, None] * BM_ + s[None, :],
                 tl.trans(v), mask=(k[:, None] < KVAL))


# --------------------------------------------------------------------------
# 3) GEMM1 + OAI SwiGLU  ->  hact[tile, i, slot]
# --------------------------------------------------------------------------
@triton.jit
def _gemm1(XG, W, S, R, B, GRAW, CNT, KB: tl.constexpr, NB: tl.constexpr,
           KP: tl.constexpr, IP: tl.constexpr, E: tl.constexpr,
           LIMIT: tl.constexpr, GMIN: tl.constexpr,
           BM_: tl.constexpr, BN_: tl.constexpr, BK_: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_p = tl.program_id(1)
    e, _, nval = _tile_lookup(CNT, pid_p, E, BM_)
    if e >= E:
        return
    acc = tl.zeros((BN_, BM_), dtype=tl.float32)
    wbase = (e * NB + pid_n) * KB
    xbase = pid_p * (KP * BM_)
    for kb in range(KB):
        wf = _load_wf(W, S, (wbase + kb) * (BN_ * BK_ // 2),
                      (wbase + kb) * (BN_ * BK_ // 32), BN_, BK_)
        x = tl.load(XG + xbase + (kb * BK_ + tl.arange(0, BK_))[:, None] * BM_
                    + tl.arange(0, BM_)[None, :])
        acc = tl.dot(wf, x, acc=acc)
    # Epilogue must stay *elementwise* on the [BN, BM] accumulator: splitting it
    # with reshape/permute/split forces a layout conversion that Triton
    # propagates back into the k-loop and costs 3.4x on this kernel. The gate /
    # up de-interleave is therefore done by the store address instead, and the
    # activation itself by `_swiglu` over the two planes.
    HF: tl.constexpr = BN_ // 2
    r = tl.arange(0, BN_)
    off = (e * NB + pid_n) * BN_ + r
    acc = acc * tl.load(R + off)[:, None] + tl.load(B + off)[:, None]
    is_g = r < HF
    # Clamp in place: gate to [GMIN, LIMIT] (g*sigmoid(alpha*g) has already
    # underflown to 0 by GMIN), up to [-LIMIT, LIMIT]. Keeps fp16 storage safe.
    lo = tl.where(is_g, GMIN, -LIMIT)
    v = tl.minimum(tl.maximum(acc, lo[:, None]), LIMIT)
    i = pid_n * HF + (r - tl.where(is_g, 0, HF))
    plane = tl.where(is_g, 0, IP * BM_)
    tl.store(GRAW + pid_p * (2 * IP * BM_) + plane[:, None] + i[:, None] * BM_
             + tl.arange(0, BM_)[None, :], v.to(tl.float16))


@triton.jit
def _swiglu(GRAW, HACT, CNT, IP: tl.constexpr, I_: tl.constexpr, E: tl.constexpr,
            ALPHA: tl.constexpr, BM_: tl.constexpr, HF: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_p = tl.program_id(1)
    e, _, nval = _tile_lookup(CNT, pid_p, E, BM_)
    if e >= E:
        return
    o = pid_p * (2 * IP * BM_) + (pid_n * HF + tl.arange(0, HF))[:, None] * BM_ \
        + tl.arange(0, BM_)[None, :]
    g = tl.load(GRAW + o).to(tl.float32)
    u = tl.load(GRAW + o + IP * BM_).to(tl.float32)
    h = (u + 1.0) * g * tl.sigmoid(ALPHA * g)
    tl.store(HACT + pid_p * (IP * BM_) + (pid_n * HF + tl.arange(0, HF))[:, None] * BM_
             + tl.arange(0, BM_)[None, :], h.to(tl.float16))


# --------------------------------------------------------------------------
# 4) GEMM2  ->  partials[tile, slot, h]
# --------------------------------------------------------------------------
@triton.jit
def _gemm2(HACT, W, S, R, PART, CNT, KB: tl.constexpr, NB: tl.constexpr,
           IP: tl.constexpr, HP: tl.constexpr, E: tl.constexpr,
           BM_: tl.constexpr, BN_: tl.constexpr, BK_: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_p = tl.program_id(1)
    e, _, nval = _tile_lookup(CNT, pid_p, E, BM_)
    if e >= E:
        return
    acc = tl.zeros((BN_, BM_), dtype=tl.float32)
    wbase = (e * NB + pid_n) * KB
    hbase = pid_p * (IP * BM_)
    for kb in range(KB):
        wf = _load_wf(W, S, (wbase + kb) * (BN_ * BK_ // 2),
                      (wbase + kb) * (BN_ * BK_ // 32), BN_, BK_)
        x = tl.load(HACT + hbase + (kb * BK_ + tl.arange(0, BK_))[:, None] * BM_
                    + tl.arange(0, BM_)[None, :])
        acc = tl.dot(wf, x, acc=acc)
    n = pid_n * BN_ + tl.arange(0, BN_)
    acc = acc * tl.load(R + e * (NB * BN_) + n)[:, None]
    tl.store(PART + pid_p * (BM_ * HP) + tl.arange(0, BM_)[None, :] * HP + n[:, None],
             acc.to(tl.float32))


# --------------------------------------------------------------------------
# 5) combine: sum_j w_j * (partial_j + b2[e_j])
# --------------------------------------------------------------------------
@triton.jit
def _combine(PART, B2, TOPE, TOPP, TOPW, CNT, OUT, M, HP: tl.constexpr,
             stride_o, E: tl.constexpr, TOPK: tl.constexpr, BM_: tl.constexpr,
             H: tl.constexpr, BH: tl.constexpr):
    t = tl.program_id(0)
    pid_h = tl.program_id(1)
    if t >= M:
        return
    idx = tl.arange(0, E)
    cnt = tl.load(CNT + idx)
    ntl = (cnt + (BM_ - 1)) // BM_
    cs = tl.cumsum(ntl, 0) - ntl                     # exclusive prefix
    h = pid_h * BH + tl.arange(0, BH)
    hm = h < H
    out = tl.zeros((BH,), dtype=tl.float32)
    for j in tl.static_range(TOPK):
        e = tl.load(TOPE + t * TOPK + j)
        p = tl.load(TOPP + t * TOPK + j)
        w = tl.load(TOPW + t * TOPK + j)
        tile = tl.sum(tl.where(idx == e, cs, 0)) + p // BM_
        slot = p % BM_
        v = tl.load(PART + tile * (BM_ * HP) + slot * HP + h, mask=hm, other=0.0)
        b = tl.load(B2 + e * HP + h, mask=hm, other=0.0)
        out += w * (v + b)
    tl.store(OUT + t * stride_o + h, out.to(tl.bfloat16), mask=hm)




# --------------------------------------------------------------------------
# Layer-level kernels: fused residual-add RMSNorm, attention with sinks,
# and RMSNorm fused into the router GEMV.
# --------------------------------------------------------------------------
@triton.jit
def _norm_add(X, R, W, OUT, RO, sx, sr, so, EPS: tl.constexpr,
              N: tl.constexpr, BLK: tl.constexpr, HAS_R: tl.constexpr):
    t = tl.program_id(0)
    c = tl.arange(0, BLK)
    m = c < N
    x = tl.load(X + t * sx + c, mask=m, other=0.0).to(tl.float32)
    if HAS_R:
        x += tl.load(R + t * sr + c, mask=m, other=0.0).to(tl.float32)
    xb = x.to(tl.bfloat16)
    tl.store(RO + t * so + c, xb, mask=m)
    xf = xb.to(tl.float32)
    v = tl.sum(xf * xf, 0) / N
    w = tl.load(W + c, mask=m, other=0.0).to(tl.float32)
    tl.store(OUT + t * so + c, (xf * tl.rsqrt(v + EPS) * w).to(tl.bfloat16), mask=m)


@triton.jit
def _attn(QKV, SINKS, O, M, sq, so, NKVG: tl.constexpr, KOFF: tl.constexpr,
          VOFF: tl.constexpr, D: tl.constexpr, SCALE: tl.constexpr,
          WINDOW: tl.constexpr, BQ: tl.constexpr, BN_A: tl.constexpr):
    pid_q = tl.program_id(0)
    h = tl.program_id(1)
    kv = h // NKVG
    qs = pid_q * BQ + tl.arange(0, BQ)
    d = tl.arange(0, D)
    qm = qs < M
    q = tl.load(QKV + qs[:, None] * sq + h * D + d[None, :], mask=qm[:, None], other=0.0)
    m_i = tl.zeros((BQ,), dtype=tl.float32) + tl.load(SINKS + h).to(tl.float32)
    l_i = tl.full((BQ,), 1.0, tl.float32)
    acc = tl.zeros((BQ, D), dtype=tl.float32)
    hi = (pid_q + 1) * BQ
    lo = 0
    if WINDOW > 0:
        lo = tl.maximum((pid_q * BQ - WINDOW + 1) // BN_A, 0) * BN_A
    for n0 in range(lo, hi, BN_A):
        ks = n0 + tl.arange(0, BN_A)
        km = ks < M
        k = tl.load(QKV + ks[:, None] * sq + KOFF + kv * D + d[None, :],
                    mask=km[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k)) * SCALE
        ok = (qs[:, None] >= ks[None, :]) & km[None, :]
        if WINDOW > 0:
            ok = ok & (qs[:, None] - ks[None, :] < WINDOW)
        s = tl.where(ok, s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(QKV + ks[:, None] * sq + VOFF + kv * D + d[None, :],
                    mask=km[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
    tl.store(O + qs[:, None] * so + h * D + d[None, :],
             (acc / l_i[:, None]).to(tl.bfloat16), mask=qm[:, None])


# --------------------------------------------------------------------------
# Host-side repack + module
# --------------------------------------------------------------------------


def _repack(w, s, row_perm, KP, device):
    """[Erows, K/2] fp4 + [Erows, K/32] E8M0 scales -> tile-major (W, S, rowscale).

    ``row_perm`` maps destination row -> source row.  The E8M0 exponent
    2**(S-127) does not fit an fp16 multiplier, so it is split per row into
    ``2**(Smax-127)`` (a float32 factor applied once to the fp32 accumulator)
    times ``2**(S-Smax)``, stored as the fp16 exponent byte ``S-Smax+15``.
    Blocks more than 2**-15 below the row max store 0 (exactly the fp16
    rounding of a contribution that small).
    """
    K = s.shape[-1] * 32
    N = row_perm.numel()
    nb, kb = N // BN, KP // BK
    # nibble-expand along K so the row permutation and K padding are trivial
    lo = (w & 0x0F)
    hi = (w >> 4)
    nib = torch.stack((lo, hi), dim=-1).reshape(w.shape[0], -1)     # [rows, K]
    nib = nib[row_perm]
    sc = s[row_perm]
    if KP > K:
        nib = torch.cat((nib, torch.zeros(N, KP - K, dtype=nib.dtype, device=device)), 1)
        sc = torch.cat((sc, torch.zeros(N, (KP - K) // 32, dtype=sc.dtype, device=device)), 1)
    smax = sc.max(dim=1).values                                   # [N]
    rowscale = torch.exp2(smax.float() - 127.0)
    sc = (sc.int() - smax.int()[:, None] + 15).clamp_(min=0).to(torch.uint8)
    nib = nib.reshape(nb, BN, kb, BK).permute(0, 2, 1, 3).contiguous()
    packed = (nib[..., 0::2] | (nib[..., 1::2] << 4)).contiguous()
    sc = sc.reshape(nb, BN, kb, BK // 32).permute(0, 2, 1, 3).contiguous()
    return packed, sc, rowscale


class _FusedGptOssMoE(GptOssMoE):
    """GptOssMoE with a hand-written small-M expert path."""

    def __init__(self, config):
        super().__init__(config)
        self._fused_ready = False
        self._bufs = {}

    # -- weight prep ------------------------------------------------------
    def process_weights_after_loading(self):
        if self._processed:
            return
        if self.use_trtllm:
            self._build_fused_weights()
        super().process_weights_after_loading()

    def _build_fused_weights(self):
        E, I, H = self.num_experts, self.intermediate_per_tp, self.hidden_size
        dev = self.w13_weight.device
        KP1 = _pad_to(H, BK)                 # gemm1 K  (2880 -> 2944)
        IP = _pad_to(I, BK)                  # gemm2 K / hact width
        HP = _pad_to(H, BN)                  # gemm2 N  (2880 -> 2944)
        # gate/up interleave -> [64 gate | 64 up] per BN tile
        half = BN // 2
        blk = torch.arange(I, device=dev).reshape(-1, half)          # [nb, 64]
        perm1 = torch.cat((2 * blk, 2 * blk + 1), dim=1).reshape(-1)  # [nb*BN]
        perm2 = torch.cat((torch.arange(H, device=dev),
                           torch.zeros(HP - H, dtype=torch.long, device=dev)))
        w13 = self.w13_weight.data[:, :2 * I, :H // 2]
        s13 = self.w13_weight_scale.data[:, :2 * I, :H // 32]
        w2 = self.w2_weight.data[:, :H, :I // 2]
        s2 = self.w2_weight_scale.data[:, :H, :I // 32]
        W1, S1, R1, W2, S2, R2 = [], [], [], [], [], []
        for e in range(E):
            a, b, c = _repack(w13[e], s13[e], perm1, KP1, dev)
            W1.append(a); S1.append(b); R1.append(c)
            a, b, c = _repack(w2[e], s2[e], perm2, IP, dev)
            W2.append(a); S2.append(b); R2.append(c)
        self._fw1 = torch.stack(W1); self._fs1 = torch.stack(S1)
        self._fr1 = torch.stack(R1).contiguous()
        self._fw2 = torch.stack(W2); self._fs2 = torch.stack(S2)
        self._fr2 = torch.stack(R2).contiguous()
        del W1, S1, R1, W2, S2, R2
        b13 = self.w13_bias.data[:, :2 * I].float()
        self._fb1 = b13[:, perm1].reshape(E, -1).contiguous()
        b2 = torch.zeros(E, HP, dtype=torch.float32, device=dev)
        b2[:, :H] = self.w2_bias.data[:, :H].float()
        self._fb2 = b2.contiguous()
        self._KP1, self._IP, self._HP = KP1, IP, HP
        self._NB1, self._KB1 = (2 * I) // BN, KP1 // BK
        self._NB2, self._KB2 = HP // BN, IP // BK
        self._fused_ready = True
        torch.cuda.empty_cache()

    # -- buffers ----------------------------------------------------------
    def _get_bufs(self, M, device):
        ntiles = min(self.num_experts, self.top_k * M) + (self.top_k * M) // BM
        key = (M, ntiles)
        b = self._bufs.get(key)
        if b is None:
            if len(self._bufs) > 8:          # bound the per-M workspace cache
                self._bufs.clear()
            E = self.num_experts
            bincap = _pad_to(self.top_k * M, 8)
            b = dict(
                xg=torch.zeros(ntiles, self._KP1, BM, dtype=torch.float16, device=device),
                graw=torch.empty(ntiles, 2, self._IP, BM, dtype=torch.float16, device=device),
                hact=torch.zeros(ntiles, self._IP, BM, dtype=torch.float16, device=device),
                part=torch.empty(ntiles, BM, self._HP, dtype=torch.float32, device=device),
                cnt=torch.empty(E, dtype=torch.int32, device=device),
                bin=torch.empty(E * bincap, dtype=torch.int32, device=device),
                tope=torch.empty(M * self.top_k, dtype=torch.int32, device=device),
                topp=torch.empty(M * self.top_k, dtype=torch.int32, device=device),
                topw=torch.empty(M * self.top_k, dtype=torch.float32, device=device),
                ntiles=ntiles, bincap=bincap,
            )
            self._bufs[key] = b
        return b

    # -- forward ----------------------------------------------------------
    def forward_fused(self, hidden_states, router_logits):
        M, H = hidden_states.shape
        E, K = self.num_experts, self.top_k
        dev = hidden_states.device
        b = self._get_bufs(M, dev)
        b["cnt"].zero_()
        _route[(M,)](router_logits, b["cnt"], b["bin"], b["tope"], b["topp"],
                     b["topw"], M, router_logits.stride(0),
                     E=E, TOPK=K, BINCAP=b["bincap"], num_warps=4)
        nt = b["ntiles"]
        _gather[(self._KP1 // BK, nt)](hidden_states, b["xg"], b["cnt"], b["bin"],
                                       hidden_states.stride(0), H, KP=self._KP1, E=E, BM_=BM,
                                       BK_=BK, BINCAP=b["bincap"], num_warps=4)
        _gemm1[(self._NB1, nt)](b["xg"], self._fw1, self._fs1, self._fr1, self._fb1,
                                b["graw"], b["cnt"], self._KB1, self._NB1, self._KP1,
                                self._IP, E=E, LIMIT=7.0, GMIN=-40.0, BM_=BM, BN_=BN,
                                BK_=BK, num_warps=4, num_stages=4)
        _swiglu[(self._NB1, nt)](b["graw"], b["hact"], b["cnt"], self._IP,
                                 self.intermediate_per_tp, E=E, ALPHA=1.702,
                                 BM_=BM, HF=BN // 2, num_warps=4)
        _gemm2[(self._NB2, nt)](b["hact"], self._fw2, self._fs2, self._fr2, b["part"], b["cnt"],
                                self._KB2, self._NB2, self._IP, self._HP,
                                E=E, BM_=BM, BN_=BN, BK_=BK, num_warps=4, num_stages=4)
        out = torch.empty(M, H, dtype=torch.bfloat16, device=dev)
        BH = 512
        _combine[(M, (H + BH - 1) // BH)](b["part"], self._fb2, b["tope"], b["topp"],
                                          b["topw"], b["cnt"], out, M, self._HP,
                                          out.stride(0), E=E, TOPK=K, BM_=BM, H=H,
                                          BH=BH, num_warps=4)
        return out

    def forward(self, hidden_states):
        if (self._fused_ready and hidden_states.ndim == 2
                and hidden_states.shape[0] <= MOE_SMALL_M and self.tp_size == 1):
            logits = self.router(hidden_states)
            return self.forward_fused(hidden_states, logits)
        return super().forward(hidden_states)


class GptOssDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.self_attn = LlamaAttention(
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
            bias=True,
            o_proj_bias=True,
            use_sinks=True,
            sliding_window=config.sliding_window,
            layer_idx=layer_idx,
        )
        self.mlp = _FusedGptOssMoE(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self._BLK = triton.next_power_of_2(config.hidden_size)
        self._window = config.sliding_window if layer_idx % 2 == 0 else 0
        self._graphs = {}

    def _forward_baseline(self, positions, hidden_states, residual, rotary_emb):
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states, rotary_emb=rotary_emb)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual

    def forward(self, positions, hidden_states, residual, rotary_emb):
        M = hidden_states.shape[0]
        if (rotary_emb is not None or hidden_states.ndim != 2
                or M > MOE_SMALL_M or not self.mlp._fused_ready):
            return self._forward_baseline(positions, hidden_states, residual, rotary_emb)
        if not USE_GRAPH:
            return self._run_fused(hidden_states, residual)
        key = (M, residual is not None, hidden_states.dtype)
        if key not in self._graphs and len(self._graphs) > 8:
            self._graphs.clear()
        g = self._graphs[key] if key in self._graphs else self._capture(
            key, hidden_states, residual)
        if g is None:
            return self._run_fused(hidden_states, residual)
        g["hs"].copy_(hidden_states)
        if g["rs"] is not None:
            g["rs"].copy_(residual)
        g["graph"].replay()
        return g["out"], g["res"]

    def _capture(self, key, hs, rs):
        """One graph per (tokens, has-residual): the decode path is 10 kernels
        deep with < 30 us of GPU work at T=1, so eager launch latency dominates
        the measured time. Shapes are fixed per key, so the launch sequence is
        replayable; only the two input rows have to be staged in."""
        shs = hs.clone()
        srs = rs.clone() if rs is not None else None
        try:
            st = torch.cuda.Stream()
            st.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(st):
                for _ in range(3):
                    self._run_fused(shs, srs)
            torch.cuda.current_stream().wait_stream(st)
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr):
                out, res = self._run_fused(shs, srs)
        except Exception:
            self._graphs[key] = None
            return None
        ent = {"graph": gr, "hs": shs, "rs": srs, "out": out, "res": res}
        self._graphs[key] = ent
        return ent

    def _run_fused(self, hidden_states, residual):
        M = hidden_states.shape[0]
        attn, mlp = self.self_attn, self.mlp
        H = hidden_states.shape[1]
        dev = hidden_states.device
        h1 = torch.empty_like(hidden_states)
        r1 = torch.empty_like(hidden_states)
        _norm_add[(M,)](hidden_states, residual if residual is not None else hidden_states,
                        self.input_layernorm.weight, h1, r1,
                        hidden_states.stride(0),
                        residual.stride(0) if residual is not None else 0,
                        h1.stride(0), EPS=self.input_layernorm.eps, N=H,
                        BLK=self._BLK, HAS_R=residual is not None, num_warps=8)

        qkv = torch.nn.functional.linear(h1, attn.qkv_proj.weight, attn.qkv_proj.bias)
        nh, nkv, D = attn.num_heads, attn.num_kv_heads, attn.head_dim
        o = torch.empty(M, nh * D, dtype=qkv.dtype, device=dev)
        BQ = 16
        _attn[(triton.cdiv(M, BQ), nh)](
            qkv, attn.sinks, o, M, qkv.stride(0), o.stride(0),
            NKVG=nh // nkv, KOFF=nh * D, VOFF=nh * D + nkv * D, D=D,
            SCALE=attn.attn.scale, WINDOW=self._window, BQ=BQ,
            BN_A=64 if M > 16 else 16, num_warps=4)

        a = torch.nn.functional.linear(o, attn.o_proj.weight, attn.o_proj.bias)
        h2 = torch.empty_like(hidden_states)
        r2 = torch.empty_like(hidden_states)
        _norm_add[(M,)](a, r1, self.post_attention_layernorm.weight, h2, r2,
                        a.stride(0), r1.stride(0), h2.stride(0),
                        EPS=self.post_attention_layernorm.eps, N=H, BLK=self._BLK,
                        HAS_R=True, num_warps=8)
        logits = torch.nn.functional.linear(h2, mlp.router.weight, mlp.router.bias)
        return mlp.forward_fused(h2, logits), r2
