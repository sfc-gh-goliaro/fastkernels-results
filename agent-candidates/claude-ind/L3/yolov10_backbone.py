"""YOLOv10 native backbone -- fused NHWC Triton implementation.

The baseline is ~40 Conv(-BN)-SiLU blocks on 640x640 inputs: ~12 GFLOP of work
spread over >100 tiny cuDNN / batch-norm / SiLU launches, so it is entirely
launch- and latency-bound rather than math-bound.  This version

  * folds every BatchNorm into its convolution (the re-layout happens on the
    first forward, once the bench has shared the baseline's weights),
  * keeps every activation in NHWC fp16, so each conv is an implicit GEMM whose
    reduction axis is contiguous, and tiles spatially (scalar output row,
    vector output column) so every gather is affine and coalesced,
  * passes every shape and stride as a compile-time constant, and merges the
    three horizontal taps of each 3x3 conv into one K = 3*Cin GEMM,
  * fuses bias + SiLU + the bottleneck residual add into the conv epilogue,
  * keeps every C2f / SPPF branch in its own *contiguous* buffer and teaches
    the following 1x1 conv to read several of them, so ``torch.cat`` disappears
    without costing sector efficiency,
  * rewrites the three chained 5x5 SPPF max-pools as one separable 5/9/13 pass
    (stride-1 max-pooling composes: mp5 o mp5 == mp9),
  * runs PSA's attention as a single fused softmax+matmul kernel, and
  * replays the whole network from a CUDA graph, so the launch cost of ~40
    small kernels collapses.

Every compute kernel below is written from scratch in Triton; nothing calls
cuDNN or cuBLAS.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

from ..L2.yolov10_c2f import YOLOC2f
from ..L2.yolov10_conv import YOLOConv
from ..L2.yolov10_psa import YOLOPSA
from ..L2.yolov10_scdown import YOLOSCDown
from ..L2.yolov10_sppf import YOLOSPPF

_NEG = tl.constexpr(float("-inf"))
_NLOG2E = tl.constexpr(-1.4426950408889634)
_LOG2E = tl.constexpr(1.4426950408889634)


@triton.jit
def _silu(a):
    """x * sigmoid(x) via the hardware ex2 / approximate-reciprocal units."""
    return tl.fdiv(a, 1.0 + tl.exp2(a * _NLOG2E), ieee_rounding=False)


# ---------------------------------------------------------------------------
# 1x1 convolution == GEMM over the (N*H*W, Cin) matrix.  Up to four separately
# allocated inputs are concatenated along K on the fly (C2f / SPPF), and the
# output can be split across two buffers (C2f's cv1 feeding a bottleneck).
# ---------------------------------------------------------------------------
@triton.jit
def _k_gemm(X, W, B, R, Y, YB,
            M: tl.constexpr, F: tl.constexpr, KS: tl.constexpr,
            NSRC: tl.constexpr, SS: tl.constexpr, sx: tl.constexpr,
            sy: tl.constexpr, sr: tl.constexpr,
            BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
            ACT: tl.constexpr, RES: tl.constexpr, DUAL: tl.constexpr,
            NS: tl.constexpr):
    """Y[m, n] = sum_k X[m, k] * W[k, n] + B[n]  (+ SiLU) (+ R[m, n]).

    ``NSRC`` separately stored inputs (``SS`` elements apart, e.g. the branches
    of a C2f / SPPF concatenation) are joined along K on the fly, and the output
    may be split across two buffers (C2f's cv1 feeding a bottleneck).
    """
    pn = tl.program_id(1)
    om = tl.program_id(0) * BM + tl.arange(0, BM)
    on = pn * BN + tl.arange(0, BN)
    mm = om < M
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for kk in tl.range(0, NSRC * KS, BK, num_stages=NS):
        i = kk // KS
        ok = kk - i * KS + tl.arange(0, BK)
        if KS % BK == 0:
            a = tl.load(X + i * SS + om[:, None] * sx + ok[None, :],
                        mask=mm[:, None], other=0.0)
            b = tl.load(W + (i * KS + ok)[:, None] * F + on[None, :])
        else:
            mk = ok < KS
            a = tl.load(X + i * SS + om[:, None] * sx + ok[None, :],
                        mask=mm[:, None] & mk[None, :], other=0.0)
            b = tl.load(W + (i * KS + ok)[:, None] * F + on[None, :],
                        mask=mk[:, None], other=0.0)
        acc = tl.dot(a, b, acc)
    acc += tl.load(B + on)[None, :]
    if ACT:
        acc = _silu(acc)
    if RES:
        acc += tl.load(R + om[:, None] * sr + on[None, :],
                       mask=mm[:, None], other=0.0)
    v = acc.to(tl.float16)
    if DUAL:
        oc = tl.arange(0, BN)
        if pn == 0:
            tl.store(Y + om[:, None] * BN + oc[None, :], v, mask=mm[:, None])
        else:
            tl.store(YB + om[:, None] * BN + oc[None, :], v, mask=mm[:, None])
    else:
        tl.store(Y + om[:, None] * sy + on[None, :], v, mask=mm[:, None])


# ---------------------------------------------------------------------------
# 3x3 convolution, pad 1, stride ST.  One program owns BW consecutive output
# columns of one output row, and each of the three kernel rows is a single
# K = 3*Cin GEMM covering that row's three horizontal taps.
# ---------------------------------------------------------------------------
@triton.jit
def _k_conv3x3(X, W, B, R, Y,
               IH: tl.constexpr, IW: tl.constexpr,
               OH: tl.constexpr, OW: tl.constexpr,
               C: tl.constexpr, F: tl.constexpr, ST: tl.constexpr,
               sx: tl.constexpr, sy: tl.constexpr, sr: tl.constexpr,
               NB2: tl.constexpr, BW: tl.constexpr, BN: tl.constexpr,
               ACT: tl.constexpr, RES: tl.constexpr, TAPS: tl.constexpr,
               NS: tl.constexpr):
    oh = tl.program_id(1)
    p2 = tl.program_id(2)
    nb = p2 // NB2
    ow = tl.program_id(0) * BW + tl.arange(0, BW)
    on = (p2 % NB2) * BN + tl.arange(0, BN)
    mw = ow < OW
    ih0 = oh * ST - 1
    iw0 = ow * ST - 1
    nrow = nb * IH
    acc = tl.zeros((BW, BN), dtype=tl.float32)
    if TAPS == 3:
        # Four horizontal taps per GEMM (the fourth has zero weights) so the K
        # extent stays a power of two and the A tile stays contiguous.
        kk = tl.arange(0, 4 * C)
        ks = kk // C
        iwk = iw0[:, None] + ks[None, :]
        vk = (iwk >= 0) & (iwk < IW) & mw[:, None] & (ks < 3)[None, :]
        if sx == C:
            aoff = iw0[:, None] * sx + kk[None, :]
        else:
            aoff = iwk * sx + (kk % C)[None, :]
        wcol = kk[:, None] * F + on[None, :]
        for r in tl.range(0, 3, num_stages=NS):
            ih = ih0 + r
            a = tl.load(X + (nrow + ih) * IW * sx + aoff,
                        mask=vk & (ih >= 0) & (ih < IH), other=0.0)
            b = tl.load(W + r * 4 * C * F + wcol)
            acc = tl.dot(a, b, acc)
    else:
        ok = tl.arange(0, C)
        wcol = ok[:, None] * F + on[None, :]
        for t in tl.range(0, 9, num_stages=NS):
            r = t // 3
            s = t - r * 3
            ih = ih0 + r
            iw = iw0 + s
            v = mw & (ih >= 0) & (ih < IH) & (iw >= 0) & (iw < IW)
            a = tl.load(X + (nrow + ih) * IW * sx + iw[:, None] * sx + ok[None, :],
                        mask=v[:, None], other=0.0)
            b = tl.load(W + (r * 4 + s) * C * F + wcol)
            acc = tl.dot(a, b, acc)
    acc += tl.load(B + on)[None, :]
    if ACT:
        acc = _silu(acc)
    om = (nb * OH + oh) * OW + ow
    if RES:
        acc += tl.load(R + om[:, None] * sr + on[None, :],
                       mask=mw[:, None], other=0.0)
    tl.store(Y + om[:, None] * sy + on[None, :], acc.to(tl.float16),
             mask=mw[:, None])


# ---------------------------------------------------------------------------
# Stem: NCHW 3-channel input -> NHWC 4-channel (channel 3 zero), then a 3x3
# stride-2 conv whose K axis is 16 contiguous packed elements per kernel row.
# ---------------------------------------------------------------------------
@triton.jit
def _k_pack4(X, Y, HW: tl.constexpr, BM: tl.constexpr):
    n = tl.program_id(1)
    k = tl.program_id(0) * (4 * BM) + tl.arange(0, 4 * BM)
    pix = k // 4
    ch = k % 4
    mp = pix < HW
    a = tl.load(X + n * 3 * HW + tl.minimum(ch, 2) * HW + pix,
                mask=mp & (ch < 3), other=0.0)
    tl.store(Y + n * HW * 4 + k, a, mask=mp)


@triton.jit
def _k_stem(X, W, B, Y,
            IH: tl.constexpr, IW: tl.constexpr,
            OH: tl.constexpr, OW: tl.constexpr,
            F: tl.constexpr, sy: tl.constexpr,
            BW: tl.constexpr, ROWS: tl.constexpr, NS: tl.constexpr):
    oh = tl.program_id(1)
    nb = tl.program_id(2)
    ow = tl.program_id(0) * BW + tl.arange(0, BW)
    on = tl.arange(0, F)
    mw = ow < OW
    ih0 = oh * 2 - 1
    iw0 = (ow * 2 - 1) * 4
    nrow = nb * IH
    acc = tl.zeros((BW, F), dtype=tl.float32)
    if ROWS == 3:
        kk = tl.arange(0, 48)
        kr = kk // 16
        kj = kk % 16
        ih = ih0 + kr[None, :]
        col = iw0[:, None] + kj[None, :]
        off = (nrow + ih) * IW * 4 + col
        v = mw[:, None] & (ih >= 0) & (ih < IH) & (col >= 0) & (col < IW * 4)
        a = tl.load(X + off, mask=v, other=0.0)
        b = tl.load(W + kk[:, None] * F + on[None, :])
        acc = tl.dot(a, b, acc)
    else:
        kk = tl.arange(0, 16)
        col = iw0[:, None] + kk[None, :]
        vc = mw[:, None] & (col >= 0) & (col < IW * 4)
        for r in tl.range(0, 3, num_stages=NS):
            ih = ih0 + r
            a = tl.load(X + (nrow + ih) * IW * 4 + col,
                        mask=vc & (ih >= 0) & (ih < IH), other=0.0)
            b = tl.load(W + r * 16 * F + kk[:, None] * F + on[None, :])
            acc = tl.dot(a, b, acc)
    acc += tl.load(B + on)[None, :]
    acc = _silu(acc)
    om = (nb * OH + oh) * OW + ow
    tl.store(Y + om[:, None] * sy + on[None, :], acc.to(tl.float16),
             mask=mw[:, None])


# ---------------------------------------------------------------------------
# Depthwise 3x3, pad 1, stride ST, bias only.
# ---------------------------------------------------------------------------
@triton.jit
def _k_dw3x3(X, W, B, Y,
             IH: tl.constexpr, IW: tl.constexpr,
             OH: tl.constexpr, OW: tl.constexpr,
             C: tl.constexpr, ST: tl.constexpr,
             sx: tl.constexpr, sy: tl.constexpr,
             NB2: tl.constexpr, BW: tl.constexpr, BN: tl.constexpr,
             NS: tl.constexpr):
    oh = tl.program_id(1)
    p2 = tl.program_id(2)
    nb = p2 // NB2
    ow = tl.program_id(0) * BW + tl.arange(0, BW)
    on = (p2 % NB2) * BN + tl.arange(0, BN)
    mw = ow < OW
    ih0 = oh * ST - 1
    iw0 = ow * ST - 1
    nrow = nb * IH
    acc = tl.zeros((BW, BN), dtype=tl.float32)
    for t in tl.range(0, 9, num_stages=NS):
        r = t // 3
        s = t - r * 3
        ih = ih0 + r
        iw = iw0 + s
        v = mw & (ih >= 0) & (ih < IH) & (iw >= 0) & (iw < IW)
        a = tl.load(X + (nrow + ih) * IW * sx + iw[:, None] * sx + on[None, :],
                    mask=v[:, None], other=0.0)
        w = tl.load(W + t * C + on)
        acc += a.to(tl.float32) * w.to(tl.float32)[None, :]
    acc += tl.load(B + on)[None, :]
    om = (nb * OH + oh) * OW + ow
    tl.store(Y + om[:, None] * sy + on[None, :], acc.to(tl.float16),
             mask=mw[:, None])


# ---------------------------------------------------------------------------
# SPPF: separable 5/9/13 max-pool (mp5 o mp5 == mp9 at stride 1).
# ---------------------------------------------------------------------------
@triton.jit
def _k_mp_h(X, YB, SY: tl.constexpr, HH: tl.constexpr, WW: tl.constexpr,
            C: tl.constexpr, NB2: tl.constexpr,
            BW: tl.constexpr, BN: tl.constexpr, NS: tl.constexpr):
    oh = tl.program_id(1)
    p2 = tl.program_id(2)
    nb = p2 // NB2
    ow = tl.program_id(0) * BW + tl.arange(0, BW)
    on = (p2 % NB2) * BN + tl.arange(0, BN)
    mw = ow < WW
    base = (nb * HH + oh) * WW
    m5 = tl.full((BW, BN), _NEG, tl.float16)
    m9 = m5
    m13 = m5
    for d in tl.range(-6, 7, num_stages=NS):
        wx = ow + d
        v = mw & (wx >= 0) & (wx < WW)
        a = tl.load(X + (base + wx)[:, None] * C + on[None, :],
                    mask=v[:, None], other=_NEG)
        m13 = tl.maximum(m13, a)
        m9 = tl.maximum(m9, tl.where((d >= -4) & (d <= 4), a, _NEG))
        m5 = tl.maximum(m5, tl.where((d >= -2) & (d <= 2), a, _NEG))
    off = (base + ow)[:, None] * C + on[None, :]
    tl.store(YB + off, m5, mask=mw[:, None])
    tl.store(YB + SY + off, m9, mask=mw[:, None])
    tl.store(YB + 2 * SY + off, m13, mask=mw[:, None])


@triton.jit
def _k_mp_v(XB, YB, SX: tl.constexpr, SY: tl.constexpr,
            HH: tl.constexpr, WW: tl.constexpr,
            C: tl.constexpr, NB2: tl.constexpr,
            BW: tl.constexpr, BN: tl.constexpr, NS: tl.constexpr):
    """Vertical halves of the 5/9/13 max-pools, straight into the SPPF branches."""
    oh = tl.program_id(1)
    p2 = tl.program_id(2)
    nb = p2 // NB2
    ow = tl.program_id(0) * BW + tl.arange(0, BW)
    on = (p2 % NB2) * BN + tl.arange(0, BN)
    mw = ow < WW
    for q in tl.range(0, 3):
        lim = 2 * q + 2
        mq = tl.full((BW, BN), _NEG, tl.float16)
        for d in tl.range(-lim, lim + 1, num_stages=NS):
            hx = oh + d
            v = mw & (hx >= 0) & (hx < HH)
            off = ((nb * HH + hx) * WW + ow)[:, None] * C + on[None, :]
            mq = tl.maximum(mq, tl.load(XB + q * SX + off, mask=v[:, None],
                                        other=_NEG))
        tl.store(YB + q * SY + ((nb * HH + oh) * WW + ow)[:, None] * C
                 + on[None, :], mq, mask=mw[:, None])


# ---------------------------------------------------------------------------
# PSA attention: softmax(q^T k * scale) then v @ p^T, plus the depthwise pe(v).
# QKV channels are laid out [q_h0 k_h0 ... q_hN k_hN | v_h0 ... v_hN] so the v
# block is contiguous -- that is what ``pe`` reads.
# ---------------------------------------------------------------------------
@triton.jit
def _k_attn(QKV, PE, OUT, scale,
            sq: tl.constexpr, so: tl.constexpr, NPIX: tl.constexpr,
            NH: tl.constexpr, KD: tl.constexpr, HD: tl.constexpr,
            BI: tl.constexpr, BJ: tl.constexpr, NS: tl.constexpr):
    bh = tl.program_id(1)
    b = bh // NH
    h = bh % NH
    oi = tl.program_id(0) * BI + tl.arange(0, BI)
    mi = oi < NPIX
    okd = tl.arange(0, KD)
    ohd = tl.arange(0, HD)
    rb = b * NPIX * sq
    qb = QKV + rb + h * 2 * KD
    vb = QKV + rb + NH * 2 * KD + h * HD
    q = tl.load(qb + oi[:, None] * sq + okd[None, :], mask=mi[:, None], other=0.0)
    mx = tl.full((BI,), _NEG, tl.float32)
    den = tl.zeros((BI,), tl.float32)
    o = tl.zeros((BI, HD), tl.float32)
    for j0 in tl.range(0, NPIX, BJ, num_stages=NS):
        oj = j0 + tl.arange(0, BJ)
        mj = oj < NPIX
        kt = tl.load(qb + KD + oj[None, :] * sq + okd[:, None],
                     mask=mj[None, :], other=0.0)
        sc = tl.dot(q, kt) * scale
        sc = tl.where(mj[None, :], sc, _NEG)
        m2 = tl.maximum(mx, tl.max(sc, 1))
        alpha = tl.exp2((mx - m2) * _LOG2E)
        pj = tl.exp2((sc - m2[:, None]) * _LOG2E)
        den = den * alpha + tl.sum(pj, 1)
        v = tl.load(vb + oj[:, None] * sq + ohd[None, :], mask=mj[:, None], other=0.0)
        o = o * alpha[:, None] + tl.dot(pj.to(tl.float16), v)
        mx = m2
    o = tl.fdiv(o, den[:, None], ieee_rounding=False)
    ob = rb * 0 + b * NPIX * so + h * HD
    o += tl.load(PE + ob + oi[:, None] * so + ohd[None, :],
                 mask=mi[:, None], other=0.0)
    tl.store(OUT + ob + oi[:, None] * so + ohd[None, :], o.to(tl.float16),
             mask=mi[:, None])


# ---------------------------------------------------------------------------
# Weight folding / re-layout
# ---------------------------------------------------------------------------
def _folded(m: YOLOConv):
    """(weight fp32 (F, C, kh, kw), bias fp32 (F,)) with BatchNorm folded in."""
    conv = m.conv
    w = conv.weight.detach().float()
    f = w.shape[0]
    b = (conv.bias.detach().float() if conv.bias is not None
         else torch.zeros(f, device=w.device, dtype=torch.float32))
    bn = getattr(m, "bn", None)
    if bn is not None:
        scale = bn.weight.detach().float() / torch.sqrt(
            bn.running_var.detach().float() + bn.eps)
        b = bn.bias.detach().float() + (b - bn.running_mean.detach().float()) * scale
        w = w * scale.view(-1, 1, 1, 1)
    return w, b


def _w1(m):
    w, b = _folded(m)
    return w.view(w.shape[0], -1).t().contiguous().half(), b.contiguous()


def _w3(m):
    """(12*C, F) fp16: index (r*4 + s)*C + c -> w[f, c, r, s] (s padded to 4)."""
    w, b = _folded(m)
    f, c = w.shape[0], w.shape[1]
    t = torch.zeros(3, 4, c, f, device=w.device, dtype=torch.float32)
    t[:, :3, :, :] = w.permute(2, 3, 1, 0)
    return t.reshape(12 * c, f).contiguous().half(), b.contiguous()


def _wdw(m):
    w, b = _folded(m)
    return w.reshape(w.shape[0], 9).t().contiguous().half(), b.contiguous()


def _wstem(m):
    """(48, F) fp16: index r*16 + s*4 + c -> w[f, c, r, s] (s, c padded to 4)."""
    w, b = _folded(m)
    f = w.shape[0]
    t = torch.zeros(3, 4, 4, f, device=w.device, dtype=torch.float32)
    t[:, :3, :3, :] = w.permute(2, 3, 1, 0)
    return t.reshape(48, f).contiguous().half(), b.contiguous()


# ---------------------------------------------------------------------------
# Launch configuration.  ``_CFG`` holds tuned (block0, block1, warps, extra)
# tuples found offline with dev/tune.py; anything missing falls back to a
# heuristic default.
# ---------------------------------------------------------------------------
_CFG: dict = {
    ('stem', 4, 320, 16): (32, 0, 4, 1, 1),
    ('c3', 4, 160, 16, 32, 2, 1, 0, 16, 32): (32, 32, 4, 1, 1),
    ('gemm', 102400, 32, 1, 32, 1, 0, 1, 32, 16, 32, 0): (32, 16, 4, 0, 3),
    ('c3', 4, 160, 16, 16, 1, 1, 0, 16, 16): (32, 16, 4, 3, 3),
    ('c3', 4, 160, 16, 16, 1, 1, 1, 16, 16): (32, 16, 4, 1, 1),
    ('gemm', 102400, 16, 3, 32, 1, 0, 0, 16, 32, 16, 1638400): (256, 32, 4, 0, 3),
    ('c3', 4, 80, 32, 64, 2, 1, 0, 32, 64): (32, 64, 4, 1, 3),
    ('gemm', 25600, 64, 1, 64, 1, 0, 1, 64, 32, 64, 0): (32, 32, 4, 0, 1),
    ('c3', 4, 80, 32, 32, 1, 1, 0, 32, 32): (32, 32, 4, 1, 3),
    ('c3', 4, 80, 32, 32, 1, 1, 1, 32, 32): (32, 32, 4, 1, 3),
    ('gemm', 25600, 32, 4, 64, 1, 0, 0, 32, 64, 32, 819200): (32, 64, 4, 0, 3),
    ('gemm', 25600, 64, 1, 128, 1, 0, 0, 64, 128, 64, 0): (32, 128, 4, 0, 1),
    ('dw', 4, 40, 128, 2, 128, 128): (16, 64, 4, 0, 1),
    ('gemm', 6400, 128, 1, 128, 1, 0, 1, 128, 64, 128, 0): (32, 64, 4, 0, 1),
    ('c3', 4, 40, 64, 64, 1, 1, 0, 64, 64): (16, 64, 4, 1, 3),
    ('c3', 4, 40, 64, 64, 1, 1, 1, 64, 64): (16, 64, 4, 1, 3),
    ('gemm', 6400, 64, 4, 128, 1, 0, 0, 64, 128, 64, 409600): (32, 64, 8, 0, 3),
    ('gemm', 6400, 128, 1, 256, 1, 0, 0, 128, 256, 128, 0): (32, 128, 4, 0, 1),
    ('dw', 4, 20, 256, 2, 256, 256): (32, 64, 4, 0, 3),
    ('gemm', 1600, 256, 1, 256, 1, 0, 1, 256, 128, 256, 0): (32, 128, 8, 0, 3),
    ('c3', 4, 20, 128, 128, 1, 1, 0, 128, 128): (32, 128, 4, 1, 3),
    ('c3', 4, 20, 128, 128, 1, 1, 1, 128, 128): (32, 64, 4, 1, 3),
    ('gemm', 1600, 128, 3, 256, 1, 0, 0, 128, 256, 128, 204800): (32, 128, 8, 0, 3),
    ('gemm', 1600, 256, 1, 128, 1, 0, 0, 256, 128, 256, 0): (32, 32, 4, 0, 3),
    ('mph', 4, 20, 128): (16, 32, 4, 0, 1),
    ('mpv', 4, 20, 128): (16, 64, 8, 0, 1),
    ('gemm', 1600, 128, 4, 256, 1, 0, 0, 128, 256, 128, 204800): (32, 32, 4, 0, 3),
    ('gemm', 1600, 256, 1, 256, 1, 0, 0, 256, 256, 256, 0): (16, 64, 4, 0, 3),
    ('gemm', 1600, 128, 1, 256, 0, 0, 0, 256, 256, 256, 0): (32, 128, 4, 0, 3),
    ('dw', 4, 20, 128, 1, 256, 128): (32, 32, 8, 0, 1),
    ('attn', 4, 400, 2, 32, 64): (32, 512, 4, 0, 2),
    ('gemm', 1600, 128, 1, 128, 0, 1, 0, 128, 256, 256, 0): (32, 64, 8, 0, 1),
    ('gemm', 1600, 128, 1, 256, 1, 0, 0, 256, 256, 256, 0): (32, 128, 4, 0, 1),
    ('gemm', 1600, 256, 1, 128, 0, 1, 0, 256, 256, 256, 0): (32, 32, 4, 0, 3),
    ('stem', 1, 320, 16): (32, 0, 4, 1, 1),
    ('c3', 1, 160, 16, 32, 2, 1, 0, 16, 32): (32, 32, 4, 3, 3),
    ('gemm', 25600, 32, 1, 32, 1, 0, 1, 32, 16, 32, 0): (32, 16, 4, 0, 1),
    ('c3', 1, 160, 16, 16, 1, 1, 0, 16, 16): (32, 16, 4, 3, 1),
    ('c3', 1, 160, 16, 16, 1, 1, 1, 16, 16): (32, 16, 4, 3, 1),
    ('gemm', 25600, 16, 3, 32, 1, 0, 0, 16, 32, 16, 409600): (32, 32, 4, 0, 1),
    ('c3', 1, 80, 32, 64, 2, 1, 0, 32, 64): (32, 64, 4, 1, 3),
    ('gemm', 6400, 64, 1, 64, 1, 0, 1, 64, 32, 64, 0): (32, 32, 4, 0, 3),
    ('c3', 1, 80, 32, 32, 1, 1, 0, 32, 32): (32, 32, 4, 3, 3),
    ('c3', 1, 80, 32, 32, 1, 1, 1, 32, 32): (16, 32, 4, 3, 3),
    ('gemm', 6400, 32, 4, 64, 1, 0, 0, 32, 64, 32, 204800): (32, 64, 4, 0, 3),
    ('gemm', 6400, 64, 1, 128, 1, 0, 0, 64, 128, 64, 0): (32, 64, 4, 0, 1),
    ('dw', 1, 40, 128, 2, 128, 128): (16, 32, 8, 0, 1),
    ('gemm', 1600, 128, 1, 128, 1, 0, 1, 128, 64, 128, 0): (16, 64, 4, 0, 1),
    ('c3', 1, 40, 64, 64, 1, 1, 0, 64, 64): (16, 64, 4, 1, 3),
    ('c3', 1, 40, 64, 64, 1, 1, 1, 64, 64): (16, 64, 4, 1, 3),
    ('gemm', 1600, 64, 4, 128, 1, 0, 0, 64, 128, 64, 102400): (16, 64, 4, 0, 3),
    ('gemm', 1600, 128, 1, 256, 1, 0, 0, 128, 256, 128, 0): (32, 64, 4, 0, 1),
    ('dw', 1, 20, 256, 2, 256, 256): (16, 32, 8, 0, 1),
    ('gemm', 400, 256, 1, 256, 1, 0, 1, 256, 128, 256, 0): (16, 128, 8, 0, 3),
    ('c3', 1, 20, 128, 128, 1, 1, 0, 128, 128): (16, 64, 4, 1, 3),
    ('c3', 1, 20, 128, 128, 1, 1, 1, 128, 128): (32, 32, 4, 1, 3),
    ('gemm', 400, 128, 3, 256, 1, 0, 0, 128, 256, 128, 51200): (16, 64, 4, 0, 3),
    ('gemm', 400, 256, 1, 128, 1, 0, 0, 256, 128, 256, 0): (16, 32, 4, 0, 3),
    ('mph', 1, 20, 128): (16, 64, 8, 0, 1),
    ('mpv', 1, 20, 128): (16, 32, 4, 0, 1),
    ('gemm', 400, 128, 4, 256, 1, 0, 0, 128, 256, 128, 51200): (32, 32, 4, 0, 3),
    ('gemm', 400, 256, 1, 256, 1, 0, 0, 256, 256, 256, 0): (16, 64, 8, 0, 3),
    ('gemm', 400, 128, 1, 256, 0, 0, 0, 256, 256, 256, 0): (32, 64, 4, 0, 1),
    ('dw', 1, 20, 128, 1, 256, 128): (16, 64, 4, 0, 1),
    ('attn', 1, 400, 2, 32, 64): (16, 512, 4, 0, 1),
    ('gemm', 400, 128, 1, 128, 0, 1, 0, 128, 256, 256, 0): (16, 32, 8, 0, 3),
    ('gemm', 400, 128, 1, 256, 1, 0, 0, 256, 256, 256, 0): (32, 32, 8, 0, 1),
    ('gemm', 400, 256, 1, 128, 0, 1, 0, 256, 256, 256, 0): (16, 32, 4, 0, 3),
}

_TUNE = False


def _bk(K):
    if K <= 128:
        k = 16
        while k < K:
            k *= 2
        return k
    return 128


class _Op:
    def __init__(self):
        self.key = None

    def setup(self):
        self.apply(_CFG.get(self.key) or self.default())

    def configs(self):
        return [self.default()]


class _Gemm(_Op):
    kind = "gemm"

    def __init__(self, x, wb, y, act, res=None, yb=None, nsrc=1):
        self.x = x if x.dim() == 2 else x[0]
        self.SS = 0 if x.dim() == 2 else x.stride(0)
        self.w, self.b = wb
        self.F = self.w.shape[1]
        self.NSRC = nsrc if x.dim() == 2 else x.shape[0]
        self.KS = self.w.shape[0] // self.NSRC
        self.M = self.x.shape[0]
        self.sx = self.x.stride(0)
        self.y, self.yb = y, yb if yb is not None else y
        self.r = res if res is not None else self.x
        self.sy = y.stride(0)
        self.sr = self.r.stride(0)
        self.act = bool(act)
        self.res = res is not None
        self.dual = yb is not None
        self.BK = _bk(self.KS)
        self.key = (self.kind, self.M, self.KS, self.NSRC, self.F,
                    int(self.act), int(self.res), int(self.dual), self.sx,
                    self.sy, self.sr, self.SS)
        self.setup()

    def default(self):
        bn = self.F // 2 if self.dual else min(self.F, 64)
        bm = max(16, min(32, 2048 // bn))
        return (bm, bn, 4, 0, 3)

    def configs(self):
        if self.dual:
            bns = [self.F // 2]
        else:
            bns = [bn for bn in (32, 64, 128, 256) if bn <= self.F]
        out = []
        for bn in bns:
            if self.F % bn:
                continue
            for bm in (16, 32, 64, 128, 256):
                if not (512 <= bm * bn <= 8192):
                    continue
                for w in (4, 8):
                    for ns in (1, 3):
                        out.append((bm, bn, w, 0, ns))
        return out

    def apply(self, cfg):
        self.BM, self.BN, self.warps, _, self.NS = cfg
        self.grid = (triton.cdiv(self.M, self.BM), self.F // self.BN)

    def __call__(self):
        _k_gemm[self.grid](self.x, self.w, self.b, self.r, self.y, self.yb,
                           self.M, self.F, self.KS, self.NSRC, self.SS, self.sx,
                           self.sy, self.sr,
                           self.BM, self.BN, self.BK,
                           self.act, self.res, self.dual, self.NS,
                           num_warps=self.warps)


class _Conv3x3(_Op):
    kind = "c3"

    def __init__(self, x, wb, y, shape, st, act, res=None, nb=1):
        self.x, self.y = x, y
        self.r = res if res is not None else x
        self.w, self.b = wb
        self.IH, self.IW, self.OH, self.OW = shape
        self.nb = nb
        self.C = self.w.shape[0] // 12
        self.F = self.w.shape[1]
        self.ST = st
        self.sx, self.sy, self.sr = x.stride(0), y.stride(0), self.r.stride(0)
        self.act = bool(act)
        self.res = res is not None
        self.key = (self.kind, nb, self.OW, self.C, self.F, st, int(self.act),
                    int(self.res), self.sx, self.sy)
        self.setup()

    def default(self):
        bn = self.F if self.F <= 64 else self.F // 2
        bw = max(16, min(32, 2048 // bn))
        return (bw, bn, 4, 1, 3)

    def configs(self):
        out = []
        for bn in [b for b in (16, 32, 64, 128, 256) if b <= self.F]:
            if self.F % bn:
                continue
            for bw in (16, 32, 64, 128):
                if not (256 <= bw * bn <= 8192):
                    continue
                for w in (4, 8):
                    for taps in (1, 3):
                        for ns in (1, 3):
                            out.append((bw, bn, w, taps, ns))
        return out

    def apply(self, cfg):
        self.BW, self.BN, self.warps, self.TAPS, self.NS = cfg
        self.NB2 = self.F // self.BN
        self.grid = (triton.cdiv(self.OW, self.BW), self.OH, self.nb * self.NB2)

    def __call__(self):
        _k_conv3x3[self.grid](self.x, self.w, self.b, self.r, self.y,
                              self.IH, self.IW, self.OH, self.OW,
                              self.C, self.F, self.ST,
                              self.sx, self.sy, self.sr,
                              self.NB2, self.BW, self.BN,
                              self.act, self.res, self.TAPS, self.NS,
                              num_warps=self.warps)


class _Dw3x3(_Op):
    kind = "dw"

    def __init__(self, x, wb, y, shape, st, nb=1):
        self.x, self.y = x, y
        self.w, self.b = wb
        self.IH, self.IW, self.OH, self.OW = shape
        self.nb = nb
        self.C = self.w.shape[1]
        self.ST = st
        self.sx, self.sy = x.stride(0), y.stride(0)
        self.key = (self.kind, nb, self.OW, self.C, st, self.sx, self.sy)
        self.setup()

    def default(self):
        bn = min(self.C, 64)
        return (max(16, min(32, 2048 // bn)), bn, 4, 0, 1)

    def configs(self):
        out = []
        for bn in [b for b in (32, 64, 128, 256) if b <= self.C]:
            if self.C % bn:
                continue
            for bw in (16, 32, 64, 128):
                if not (256 <= bw * bn <= 8192):
                    continue
                for w in (4, 8):
                    for ns in (1, 3):
                        out.append((bw, bn, w, 0, ns))
        return out

    def apply(self, cfg):
        self.BW, self.BN, self.warps, _, self.NS = cfg
        self.NB2 = self.C // self.BN
        self.grid = (triton.cdiv(self.OW, self.BW), self.OH, self.nb * self.NB2)

    def __call__(self):
        _k_dw3x3[self.grid](self.x, self.w, self.b, self.y,
                            self.IH, self.IW, self.OH, self.OW,
                            self.C, self.ST, self.sx, self.sy,
                            self.NB2, self.BW, self.BN, self.NS,
                            num_warps=self.warps)


class _Pack4(_Op):
    kind = "pack"

    def __init__(self, x, y, hw, nb):
        self.x, self.y = x, y
        self.HW = hw
        self.nb = nb
        self.key = (self.kind, nb, hw)
        self.setup()

    def default(self):
        return (512, 0, 4, 0, 1)

    def configs(self):
        return [(bm, 0, w, 0, 1) for bm in (128, 256, 512, 1024) for w in (4, 8)]

    def apply(self, cfg):
        self.BM, _, self.warps, _, _ = cfg
        self.grid = (triton.cdiv(self.HW, self.BM), self.nb)

    def __call__(self):
        _k_pack4[self.grid](self.x, self.y, self.HW, self.BM, num_warps=self.warps)


class _Stem(_Op):
    kind = "stem"

    def __init__(self, x, wb, y, shape, nb):
        self.x, self.y = x, y
        self.w, self.b = wb
        self.IH, self.IW, self.OH, self.OW = shape
        self.nb = nb
        self.F = self.w.shape[1]
        self.sy = y.stride(0)
        self.key = (self.kind, nb, self.OW, self.F)
        self.setup()

    def default(self):
        return (32, 0, 4, 1, 1)

    def configs(self):
        return [(bw, 0, w, rows, ns) for bw in (16, 32, 64, 128, 256)
                for w in (4, 8) for rows in (1, 3) for ns in (1, 3)]

    def apply(self, cfg):
        self.BW, _, self.warps, self.ROWS, self.NS = cfg
        self.grid = (triton.cdiv(self.OW, self.BW), self.OH, self.nb)

    def __call__(self):
        _k_stem[self.grid](self.x, self.w, self.b, self.y,
                           self.IH, self.IW, self.OH, self.OW,
                           self.F, self.sy, self.BW, self.ROWS, self.NS,
                           num_warps=self.warps)


class _MpH(_Op):
    kind = "mph"

    def __init__(self, x, mph, hw, nb):
        self.x = x
        self.yb = mph[0]
        self.SY = mph.stride(0)
        self.HH = self.WW = hw
        self.nb = nb
        self.C = x.shape[1]
        self.key = (self.kind, nb, hw, self.C)
        self.setup()

    def default(self):
        return (32, min(self.C, 64), 4, 0, 1)

    def configs(self):
        return [(bw, bn, w, 0, ns) for bn in (32, 64, 128) if self.C % bn == 0
                for bw in (16, 32, 64) for w in (4, 8) for ns in (1, 3)
                if 256 <= bw * bn <= 8192]

    def apply(self, cfg):
        self.BW, self.BN, self.warps, _, self.NS = cfg
        self.NB2 = self.C // self.BN
        self.grid = (triton.cdiv(self.WW, self.BW), self.HH, self.nb * self.NB2)

    def __call__(self):
        _k_mp_h[self.grid](self.x, self.yb, self.SY,
                           self.HH, self.WW, self.C, self.NB2,
                           self.BW, self.BN, self.NS, num_warps=self.warps)


class _MpV(_Op):
    kind = "mpv"

    def __init__(self, mph, outs, hw, nb):
        self.xb = mph[0]
        self.SX = mph.stride(0)
        self.yb = outs[0]
        self.SY = outs.stride(0)
        self.HH = self.WW = hw
        self.nb = nb
        self.C = mph.shape[2]
        self.key = (self.kind, nb, hw, self.C)
        self.setup()

    def default(self):
        return (32, min(self.C, 64), 4, 0, 1)

    def configs(self):
        return [(bw, bn, w, 0, ns) for bn in (32, 64, 128) if self.C % bn == 0
                for bw in (16, 32, 64) for w in (4, 8) for ns in (1, 3)
                if 256 <= bw * bn <= 8192]

    def apply(self, cfg):
        self.BW, self.BN, self.warps, _, self.NS = cfg
        self.NB2 = self.C // self.BN
        self.grid = (triton.cdiv(self.WW, self.BW), self.HH, self.nb * self.NB2)

    def __call__(self):
        _k_mp_v[self.grid](self.xb, self.yb, self.SX, self.SY,
                           self.HH, self.WW, self.C, self.NB2,
                           self.BW, self.BN, self.NS, num_warps=self.warps)


class _Attn(_Op):
    kind = "attn"

    def __init__(self, qkv, pe, out, B, npix, nh, kd, hd, scale):
        self.qkv, self.pe, self.out = qkv, pe, out
        self.npix, self.nh, self.kd, self.hd = npix, nh, kd, hd
        self.scale = scale
        self.B = B
        self.sq, self.so = qkv.stride(0), out.stride(0)
        self.key = (self.kind, B, npix, nh, kd, hd)
        self.setup()

    def default(self):
        return (32, 128, 4, 0, 1)

    def configs(self):
        return [(bi, bj, w, 0, ns) for bi in (16, 32, 64, 128)
                for bj in (64, 128, 256, 512) for w in (4, 8) for ns in (1, 2)]

    def apply(self, cfg):
        self.BI, self.BJ, self.warps, _, self.NS = cfg
        self.grid = (triton.cdiv(self.npix, self.BI), self.B * self.nh)

    def __call__(self):
        _k_attn[self.grid](self.qkv, self.pe, self.out, self.scale,
                           self.sq, self.so, self.npix, self.nh, self.kd,
                           self.hd, self.BI, self.BJ, self.NS,
                           num_warps=self.warps)


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------
class YOLOv10Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem1 = YOLOConv(3, 16, 3, 2)
        self.stem2 = YOLOConv(16, 32, 3, 2)
        self.stage2 = YOLOC2f(32, 32, n=1, shortcut=True)
        self.down3 = YOLOConv(32, 64, 3, 2)
        self.stage3 = YOLOC2f(64, 64, n=2, shortcut=True)
        self.down4 = YOLOSCDown(64, 128, 3, 2)
        self.stage4 = YOLOC2f(128, 128, n=2, shortcut=True)
        self.down5 = YOLOSCDown(128, 256, 3, 2)
        self.stage5 = YOLOC2f(256, 256, n=1, shortcut=True)
        self.sppf = YOLOSPPF(256, 256, 5)
        self.psa = YOLOPSA(256, 256)
        self._w = None
        self._plans: dict = {}

    # -- weights ------------------------------------------------------------
    def _weights(self):
        if self._w is not None:
            return self._w
        w = {"stem1": _wstem(self.stem1), "stem2": _w3(self.stem2),
             "down3": _w3(self.down3)}
        for name in ("stage2", "stage3", "stage4", "stage5"):
            st = getattr(self, name)
            w[name + ".cv1"] = _w1(st.cv1)
            w[name + ".cv2"] = _w1(st.cv2)
            for i, bl in enumerate(st.m):
                w[f"{name}.m{i}.cv1"] = _w3(bl.cv1)
                w[f"{name}.m{i}.cv2"] = _w3(bl.cv2)
        for name in ("down4", "down5"):
            sc = getattr(self, name)
            w[name + ".cv1"] = _w1(sc.cv1)
            w[name + ".cv2"] = _wdw(sc.cv2)
        w["sppf.cv1"] = _w1(self.sppf.cv1)
        w["sppf.cv2"] = _w1(self.sppf.cv2)
        p = self.psa
        w["psa.cv1"] = _w1(p.cv1)
        w["psa.cv2"] = _w1(p.cv2)
        w["psa.proj"] = _w1(p.attn.proj)
        w["psa.pe"] = _wdw(p.attn.pe)
        w["psa.ffn1"] = _w1(p.ffn[0])
        w["psa.ffn2"] = _w1(p.ffn[1])
        a = p.attn
        kd, hd, nh = a.key_dim, a.head_dim, a.num_heads
        stride = 2 * kd + hd
        idx = [torch.arange(h * stride, h * stride + 2 * kd) for h in range(nh)]
        idx += [torch.arange(h * stride + 2 * kd, (h + 1) * stride) for h in range(nh)]
        perm = torch.cat(idx).to(a.qkv.conv.weight.device)
        wq, bq = _w1(a.qkv)
        w["psa.qkv"] = (wq.index_select(1, perm).contiguous(),
                        bq.index_select(0, perm).contiguous())
        self._w = w
        return w

    # -- plan ---------------------------------------------------------------
    def _build_plan(self, B: int, device):
        w = self._weights()
        f16 = torch.float16
        ops = []

        def buf(hw, c):
            return torch.zeros((B * hw * hw, c), dtype=f16, device=device)

        xin = torch.zeros((B, 3, 640, 640), dtype=f16, device=device)
        pk = torch.zeros((B * 640 * 640, 4), dtype=f16, device=device)
        pre = _Pack4(xin, pk, 640 * 640, B)

        s1 = buf(320, 16)
        ops.append(_Stem(pk, w["stem1"], s1, (640, 640, 320, 320), B))
        s2 = buf(160, 32)
        ops.append(_Conv3x3(s1, w["stem2"], s2, (320, 320, 160, 160), 2, True, nb=B))

        def c2f(name, x, hw, cout, n, out):
            c = cout // 2
            ysb = torch.zeros((2 + n, B * hw * hw, c), dtype=f16, device=device)
            ys = [ysb[i] for i in range(2 + n)]
            ops.append(_Gemm(x, w[name + ".cv1"], ys[0], True, yb=ys[1]))
            shape = (hw, hw, hw, hw)
            for i in range(n):
                tmp = buf(hw, c)
                ops.append(_Conv3x3(ys[1 + i], w[f"{name}.m{i}.cv1"], tmp, shape,
                                    1, True, nb=B))
                ops.append(_Conv3x3(tmp, w[f"{name}.m{i}.cv2"], ys[2 + i], shape,
                                    1, True, res=ys[1 + i], nb=B))
            ops.append(_Gemm(ysb, w[name + ".cv2"], out, True))

        st2 = buf(160, 32)
        c2f("stage2", s2, 160, 32, 1, st2)
        d3 = buf(80, 64)
        ops.append(_Conv3x3(st2, w["down3"], d3, (160, 160, 80, 80), 2, True, nb=B))
        p3 = buf(80, 64)
        c2f("stage3", d3, 80, 64, 2, p3)

        d4a = buf(80, 128)
        ops.append(_Gemm(p3, w["down4.cv1"], d4a, True))
        d4 = buf(40, 128)
        ops.append(_Dw3x3(d4a, w["down4.cv2"], d4, (80, 80, 40, 40), 2, nb=B))
        p4 = buf(40, 128)
        c2f("stage4", d4, 40, 128, 2, p4)

        d5a = buf(40, 256)
        ops.append(_Gemm(p4, w["down5.cv1"], d5a, True))
        d5 = buf(20, 256)
        ops.append(_Dw3x3(d5a, w["down5.cv2"], d5, (40, 40, 20, 20), 2, nb=B))
        st5 = buf(20, 256)
        c2f("stage5", d5, 20, 256, 1, st5)

        spb = torch.zeros((4, B * 400, 128), dtype=f16, device=device)
        ops.append(_Gemm(st5, w["sppf.cv1"], spb[0], True))
        mph = torch.zeros((3, B * 400, 128), dtype=f16, device=device)
        ops.append(_MpH(spb[0], mph, 20, B))
        ops.append(_MpV(mph, spb[1:], 20, B))
        sp = buf(20, 256)
        ops.append(_Gemm(spb, w["sppf.cv2"], sp, True))

        pc = buf(20, 256)
        ops.append(_Gemm(sp, w["psa.cv1"], pc, True))
        bsl = pc[:, 128:]
        qkv = buf(20, 256)
        ops.append(_Gemm(bsl, w["psa.qkv"], qkv, False))
        pe = buf(20, 128)
        ops.append(_Dw3x3(qkv[:, 128:], w["psa.pe"], pe, (20, 20, 20, 20), 1, nb=B))
        ao = buf(20, 128)
        at = self.psa.attn
        ops.append(_Attn(qkv, pe, ao, B, 400, at.num_heads, at.key_dim,
                         at.head_dim, at.scale))
        ops.append(_Gemm(ao, w["psa.proj"], bsl, False, res=bsl))
        fh = buf(20, 256)
        ops.append(_Gemm(bsl, w["psa.ffn1"], fh, True))
        ops.append(_Gemm(fh, w["psa.ffn2"], bsl, False, res=bsl))
        p5 = buf(20, 256)
        ops.append(_Gemm(pc, w["psa.cv2"], p5, True))

        return {
            "x": xin, "pre": pre, "ops": ops, "graph": None,
            "out": {
                "p3_backbone": p3.view(B, 80, 80, 64).permute(0, 3, 1, 2),
                "p4_backbone": p4.view(B, 40, 40, 128).permute(0, 3, 1, 2),
                "p5_backbone": p5.view(B, 20, 20, 256).permute(0, 3, 1, 2),
            },
        }

    @staticmethod
    def _run(ops):
        for op in ops:
            op()

    def forward(self, x: torch.Tensor):
        plan = self._plans.get(x.shape[0])
        if plan is None:
            plan = self._build_plan(x.shape[0], x.device)
            self._plans[x.shape[0]] = plan
            if not _TUNE:
                self._capture(plan)
        pre = plan["pre"]
        pre.x = x if x.is_contiguous() else x.contiguous()
        pre()
        g = plan["graph"]
        if g is None:
            self._run(plan["ops"])
        else:
            g.replay()
        return plan["out"]

    def _capture(self, plan):
        ops = plan["ops"]
        try:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    plan["pre"]()
                    self._run(ops)
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self._run(ops)
            plan["graph"] = g
        except Exception:
            plan["graph"] = None
