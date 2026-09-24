"""YOLOv10 Conv-BN-Act building block as a single fused Triton launch.

The block is ``act(bn(conv(x)))`` before ``fuse()`` and ``act(conv(x) + bias)``
after it -- three and two launches respectively when written as composed
operators.  Every captured shape is tiny (fp16, batch 1-4, 32-256 channels,
20x20 to 80x80; the largest scored output is a few hundred KB), so most of these
blocks are nowhere near a bandwidth or FLOP limit: what they cost is *launches*
and the HBM round-trip between them.

So both forward paths are reduced to the same shape,

    out[n, co, oh, ow] = act( scale[co] * conv(x)[n, co, oh, ow] + shift[co] )

and that whole expression is computed by one kernel.  The per-channel
``(scale, shift)`` pair is the *only* thing the epilogue ever sees:

* pre-``fuse()``  -- ``scale = bn.weight / sqrt(running_var + eps)`` and
  ``shift = bn.bias - running_mean * scale``, i.e. exactly the affine
  ``_fuse_conv_bn`` folds into the weight, but kept as a vector so the fold
  happens on the fp32 accumulator instead of on fp16 weights;
* post-``fuse()`` -- ``scale`` is 1 (dropped as a ``constexpr``) and ``shift``
  is the fused ``conv.bias``.

The pair is derived *inside* the epilogue, from the four BN tensors, on a
``BLOCK_CO``-long vector -- not precomputed on the host and cached.  A cache
keyed on ``(data_ptr, _version)`` cannot be made correct here: the caching
allocator hands a replaced ``running_mean`` back at the *same* address with
``_version == 0`` (verified in ``dev/t_cache.py``), and ``p.data.mul_()``
mutates a parameter without bumping its version at all.  Recomputing
``rsqrt`` on 32-256 elements per CTA is far below the measurement grain, so the
cache bought nothing and could go stale.  The fold is still done in fp32 --
dividing by ``sqrt(var + eps)`` in half precision is where it loses accuracy --
and scale, shift and the activation are applied to the fp32 GEMM accumulator
before the single narrowing store, so the fused output carries *one* rounding to
fp16 where the composed path carries three.

Three gathers cover the scored shapes; all are one launch, NCHW in and NCHW out,
fp32 accumulation, no im2col buffer and no padding pass:

* ``_conv1x1_act_kernel`` -- ``k=1``, stride 1, pad 0.  A plain GEMM: the input
  plane ``x[n]`` *is* the ``[C, P]`` A-matrix (``P = H*W``), so there is no
  gather at all.  Note this is not a re-litigation of "1x1 conv through Triton",
  which the frozen L1 Conv2d measured as a loss (0.82x / 1.01x) and leaves on
  cudnn: the claim here is about the *block*.  On these shapes cudnn's 1x1 is
  ~9 us of window and the bn+act passes behind it are another ~14, so a Triton
  GEMM that merely ties cudnn still wins by deleting those passes.
* ``_conv3x3_act_kernel`` -- 3x3, pad 1, any stride.  Flat implicit GEMM with
  the K axis ordered ``(tap, c)``, ``c`` innermost, matching the
  ``[COUT, KH*KW, C]`` weight transpose.  Used for stride 1 (and as the general
  form).
* ``_conv3x3s2_act_kernel`` -- 3x3, pad 1, stride **2**, the only scored case
  with real device work (``[4,16,320,320]`` 16->32).  See its docstring: the
  point is that pad 1 puts the span a CTA needs at the *odd* offset
  ``2*ow_t - 1``, which is 2-byte aligned, so Triton has to emit one
  ``LDG.E.U16`` per element; the span at the even base ``2*ow_t`` vectorises,
  and its two lane parities are two of the three x-taps.

Anything the gate does not match -- fp32/bf16, grouped or depthwise (the
``g=c1`` YOLOConv variants), dilated, other kernel sizes, training-mode BN, a
custom ``act`` module, or a problem large enough to be genuinely compute-bound --
falls back to the composed frozen L1 ``Conv2d`` / ``BatchNorm2d`` / ``SiLU``
modules, which is what the un-fused block already was.
"""

from __future__ import annotations

import functools

import torch
import torch.nn as nn
import triton
import triton.language as tl

from ..L1.batch_norm2d import BatchNorm2d
from ..L1.conv2d import Conv2d
from ..L1.silu import SiLU


def autopad(k: int | tuple[int, int], p=None, d: int = 1):
    if isinstance(k, tuple):
        if d > 1:
            k = tuple(d * (x - 1) + 1 for x in k)
        if p is None:
            return tuple(x // 2 for x in k)
        return p
    if d > 1:
        k = d * (k - 1) + 1
    return k // 2 if p is None else p


def _fuse_conv_bn(conv: Conv2d, bn: BatchNorm2d) -> tuple[torch.Tensor, torch.Tensor]:
    w_conv = conv.weight.clone().view(conv.weight.shape[0], -1)
    w_bn = torch.diag(
        bn.weight.to(dtype=conv.weight.dtype).div(
            torch.sqrt(bn.eps + bn.running_var.to(dtype=conv.weight.dtype))
        )
    )
    fused_weight = torch.mm(w_bn, w_conv).view_as(conv.weight)

    conv_bias = conv.bias
    if conv_bias is None:
        conv_bias = torch.zeros(conv.weight.shape[0], device=conv.weight.device, dtype=conv.weight.dtype)
    b_bn = (
        bn.bias.to(dtype=conv.weight.dtype)
        - bn.weight.to(dtype=conv.weight.dtype)
        .mul(bn.running_mean.to(dtype=conv.weight.dtype))
        .div(torch.sqrt(bn.running_var.to(dtype=conv.weight.dtype) + bn.eps))
    )
    fused_bias = torch.mm(w_bn, conv_bias.reshape(-1, 1)).reshape(-1) + b_bn
    return fused_weight, fused_bias


# ---------------------------------------------------------------------------
# The shared epilogue: the BN fold, then scale/shift/activation, all on the
# fp32 accumulator.
#
# ``acc`` is [BLOCK_CO, BLOCK_P] in every kernel, so the per-channel vectors
# broadcast along the contiguous (oh, ow) axis and the store walks the output
# row-wise.  ``Aff*`` is four pointers whose meaning is picked by ``FOLD``:
#
#   FOLD == 1  (pre-fuse):  bn.weight, bn.bias, running_mean, running_var.
#                           The harness casts *parameters* to fp16 but leaves
#                           *buffers* fp32, so these four are mixed dtype; each
#                           is widened to fp32 before the fold.
#   FOLD == 0  (post-fuse): conv.bias in Aff0, the rest aliased and unused --
#                           the affine is already inside the weight, so scale
#                           is 1 and drops out as a constexpr.
# ---------------------------------------------------------------------------
@triton.jit
def _epilogue(acc, Aff0, Aff1, Aff2, Aff3, eps, offs_co, m_co,
              FOLD: tl.constexpr, ACT: tl.constexpr, EVEN_CO: tl.constexpr):
    if EVEN_CO:
        b = tl.load(Aff0 + offs_co).to(tl.float32)
    else:
        b = tl.load(Aff0 + offs_co, mask=m_co, other=0.0).to(tl.float32)
    if FOLD == 1:
        if EVEN_CO:
            bb = tl.load(Aff1 + offs_co).to(tl.float32)
            mean = tl.load(Aff2 + offs_co).to(tl.float32)
            var = tl.load(Aff3 + offs_co).to(tl.float32)
        else:
            bb = tl.load(Aff1 + offs_co, mask=m_co, other=0.0).to(tl.float32)
            mean = tl.load(Aff2 + offs_co, mask=m_co, other=0.0).to(tl.float32)
            var = tl.load(Aff3 + offs_co, mask=m_co, other=1.0).to(tl.float32)
        # fp32 throughout: 1/sqrt(var + eps) in half precision is where this
        # fold loses the accuracy the separate passes keep.
        s = b * tl.rsqrt(var + eps)
        acc = acc * s[:, None] + (bb - mean * s)[:, None]
    else:
        acc = acc + b[:, None]
    if ACT == 1:  # SiLU / swish
        acc = acc * tl.sigmoid(acc)
    return acc


@triton.jit
def _wtile(Wt, offs_co, m_co, c, m_c, K: tl.constexpr, C: tl.constexpr,
           tap: tl.constexpr, EVEN_C: tl.constexpr, EVEN_CO: tl.constexpr):
    """One tap's ``[BLOCK_CO, BLOCK_C]`` weight tile out of ``[COUT, KH*KW, C]``.

    ``tap`` is a Python int (the callers unroll with ``range`` over a constexpr
    bound), so the whole offset is affine in the two aranges and the tile is
    BLOCK_C *contiguous* halves per output channel -- it loads as LDG.E.128.
    Reading the natural NCHW weight layout instead strides by KH*KW and turns
    every weight tile into a 9x amplified 2-byte gather.
    """
    off = offs_co[:, None] * K + (tap * C + c)[None, :]
    if EVEN_C and EVEN_CO:
        return tl.load(Wt + off)
    return tl.load(Wt + off, mask=m_co[:, None] & m_c[None, :], other=0.0)


# ---------------------------------------------------------------------------
# 1x1 (stride 1, pad 0): out[n, co, p] = act(scale[co] * sum_c W[co,c] X[n,c,p]
#                                            + shift[co]),  p in [0, H*W)
#
# Both operands are already in GEMM layout -- X[n] is [C, P] contiguous and W is
# [COUT, C] contiguous -- so there is no address arithmetic beyond two strided
# 2-D tile loads.  Every problem constant is a ``constexpr``: the shapes are
# fixed per module instance, so specializing buys a fully unrolled K loop and
# statically-resolved masks, which is most of the kernel at these sizes.
# ---------------------------------------------------------------------------
@triton.jit
def _conv1x1_act_kernel(
    X, W, Aff0, Aff1, Aff2, Aff3, Y, eps,
    P: tl.constexpr,          # H*W
    COUT: tl.constexpr,
    C: tl.constexpr,          # == K
    FOLD: tl.constexpr,
    ACT: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    EVEN_CO: tl.constexpr,
    EVEN_P: tl.constexpr,
):
    pid_p = tl.program_id(0)
    pid_co = tl.program_id(1)
    n = tl.program_id(2)

    offs_co = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    offs_p = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
    m_co = offs_co < COUT
    m_p = offs_p < P

    xn = X + n * (C * P)
    acc = tl.zeros((BLOCK_CO, BLOCK_P), dtype=tl.float32)
    for kb in tl.static_range(NUM_K):
        offs_k = kb * BLOCK_K + tl.arange(0, BLOCK_K)
        a_off = offs_k[:, None] * P + offs_p[None, :]
        w_off = offs_co[:, None] * C + offs_k[None, :]
        if EVEN_K:
            a_mask = None if EVEN_P else m_p[None, :]
            w_mask = None if EVEN_CO else m_co[:, None]
        else:
            m_k = offs_k < C
            a_mask = m_k[:, None] if EVEN_P else (m_k[:, None] & m_p[None, :])
            w_mask = m_k[None, :] if EVEN_CO else (m_co[:, None] & m_k[None, :])
        if a_mask is None:
            a = tl.load(xn + a_off)
        else:
            a = tl.load(xn + a_off, mask=a_mask, other=0.0)
        if w_mask is None:
            w = tl.load(W + w_off)
        else:
            w = tl.load(W + w_off, mask=w_mask, other=0.0)
        acc = tl.dot(w, a, acc=acc)

    acc = _epilogue(acc, Aff0, Aff1, Aff2, Aff3, eps, offs_co, m_co,
                    FOLD, ACT, EVEN_CO)
    y_off = n * (COUT * P) + offs_co[:, None] * P + offs_p[None, :]
    if EVEN_CO and EVEN_P:
        tl.store(Y + y_off, acc.to(Y.dtype.element_ty))
    else:
        tl.store(Y + y_off, acc.to(Y.dtype.element_ty), mask=m_co[:, None] & m_p[None, :])


# ---------------------------------------------------------------------------
# 3x3, pad 1, any stride -- implicit GEMM over M = OH*OW, N = Cout,
# K = C*KH*KW, gathering the A tile straight out of NCHW inside the k loop.
#
# The filter is a few KB here (73 KB at the scored channel counts), so the k loop
# is a fully unrolled static nest and the weight tile stays hot in L1.  The pad-1
# halo is masked loads inside that loop: no padding pass, no im2col buffer.
#
# This is the stride-1 form (and the general fallback).  Stride 2 has its own
# kernel below, because at stride 2 the lane axis is affine but not *contiguous*
# and the load degenerates.
# ---------------------------------------------------------------------------
@triton.jit
def _conv3x3_act_kernel(
    X, Wt, Aff0, Aff1, Aff2, Aff3, Y, eps,
    C: tl.constexpr,
    IMH: tl.constexpr,
    IMW: tl.constexpr,
    COUT: tl.constexpr,
    OW: tl.constexpr,
    P: tl.constexpr,          # OH*OW
    K: tl.constexpr,          # C*KH*KW
    KW: tl.constexpr,
    SH: tl.constexpr,
    SW: tl.constexpr,
    PH: tl.constexpr,
    PW: tl.constexpr,
    FOLD: tl.constexpr,
    ACT: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    EVEN_CO: tl.constexpr,
    EVEN_P: tl.constexpr,
):
    pid_p = tl.program_id(0)
    pid_co = tl.program_id(1)
    n = tl.program_id(2)

    offs_co = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    m_co = offs_co < COUT
    offs_p = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
    oh = offs_p // OW
    ow = offs_p - oh * OW
    m_p = offs_p < P
    ih0 = oh * SH - PH
    iw0 = ow * SW - PW

    xn = X + n * (C * IMH * IMW)
    acc = tl.zeros((BLOCK_CO, BLOCK_P), dtype=tl.float32)
    # K = (tap, c) with c innermost, matching the [COUT, KH*KW, C] weight
    # transpose, so a k tile is BLOCK_K *contiguous* halves per output channel.
    for kb in tl.static_range(NUM_K):
        offs_k = kb * BLOCK_K + tl.arange(0, BLOCK_K)
        tap = offs_k // C
        c = offs_k - tap * C
        ky = tap // KW
        kx = tap - ky * KW
        ih = ih0[None, :] + ky[:, None]
        iw = iw0[None, :] + kx[:, None]
        ok = (ih >= 0) & (ih < IMH) & (iw >= 0) & (iw < IMW)
        if not EVEN_K:
            ok = ok & (offs_k < K)[:, None]
        if not EVEN_P:
            ok = ok & m_p[None, :]
        a = tl.load(xn + c[:, None] * (IMH * IMW) + ih * IMW + iw, mask=ok, other=0.0)
        w_off = offs_co[:, None] * K + offs_k[None, :]
        if EVEN_CO and EVEN_K:
            w = tl.load(Wt + w_off)
        else:
            w_mask = m_co[:, None] if EVEN_K else m_co[:, None] & (offs_k < K)[None, :]
            w = tl.load(Wt + w_off, mask=w_mask, other=0.0)
        acc = tl.dot(w, a, acc=acc)

    acc = _epilogue(acc, Aff0, Aff1, Aff2, Aff3, eps, offs_co, m_co,
                    FOLD, ACT, EVEN_CO)
    y_off = n * (COUT * P) + offs_co[:, None] * P + offs_p[None, :]
    if EVEN_CO and EVEN_P:
        tl.store(Y + y_off, acc.to(Y.dtype.element_ty))
    else:
        tl.store(Y + y_off, acc.to(Y.dtype.element_ty), mask=m_co[:, None] & m_p[None, :])


# ---------------------------------------------------------------------------
# 3x3, pad 1, stride 2 -- the alignment kernel.
#
# This is the only scored case with real device work ([4,16,320,320] 16->32:
# 13.1 MB in, 6.55 MB out) and every flat tiling of it plateaued, so it is worth
# saying exactly where its time went.  ncu says DRAM is at 5.7% of peak and
# L1/TEX at 73%, and the SASS (dev/sass.py) says why: **96 LDG.E.U16 per thread
# and 0 vectorised input loads**.  One p tile owns output columns
# [ow_t, ow_t + BLOCK_P), so tap kx reads input columns 2*ow_t - 1 + kx + 2j --
# and the base 2*ow_t - 1 that pad 1 forces is *odd*.  A 2-byte-aligned base
# cannot be widened, so Triton correctly issues one 16-bit load per element,
# for all three x-taps.
#
# The span at the *even* base 2*ow_t is a multiple of 8 halves whenever IMW is,
# so it loads as LDG.E.128 -- and its two lane parities are two of the three
# taps outright:
#
#     even lanes  2*ow_t + 2j     = 2*ow      -> kx = 1
#     odd  lanes  2*ow_t + 2j + 1 = 2*ow + 1  -> kx = 2
#
# kx = 0 needs 2*ow - 1, which is the odd lanes shifted down one column, and no
# aligned load can produce it: a shift inside a stride-2 subset is not a
# power-of-two-aligned slice, and power-of-two-aligned slices (tl.reshape +
# tl.split, optionally tl.permute) are the only ones a register tile admits.
# What it *can* have is half the alignment: read it as the odd lanes of a span
# based at 2*ow_t - 2 (a multiple of 2) instead of the even lanes of one based at
# 2*ow_t - 1 (a multiple of 1), and the fetch widens to LDG.E.32.  Two
# alternatives to fetching it at all were measured and both lost:
#
#   * a narrow stride-2 fetch of just its BLOCK_P elements -- 26.7 us, worse
#     than the 2*BLOCK_P contiguous one despite fetching half as much;
#   * ``tl.dot`` against the 0/1 band matrix S[m, j] = (m + 1 == j), which is an
#     exact shift on fp16 and needs no second fetch at all -- 22.1 us, i.e. the
#     shift matmul costs more than the misaligned load it removes.
#
# Two loads per input row, one of them vectorised, against three unvectorised:
# 22.6 -> 18.2 us on case #2 (dev/c2.py, differential timer).  Reusing the one
# input row that consecutive output rows share (TY output rows per CTA, so
# 2*TY+1 rows instead of 3*TY) was also measured and is a small loss at every
# tile, so this kernel does one output row per CTA.
# ---------------------------------------------------------------------------
@triton.jit
def _conv3x3s2_act_kernel(
    X, Wt, Aff0, Aff1, Aff2, Aff3, Y, eps,
    C: tl.constexpr,
    IMH: tl.constexpr,
    IMW: tl.constexpr,
    COUT: tl.constexpr,
    OW: tl.constexpr,
    P: tl.constexpr,          # OH*OW
    TPR: tl.constexpr,        # p tiles per output row
    K: tl.constexpr,          # C*9
    FOLD: tl.constexpr,
    ACT: tl.constexpr,
    ALIGN8: tl.constexpr,     # IMW % 8 == 0, so the even-base span vectorises
    ALIGN2: tl.constexpr,     # IMW % 2 == 0, so the kx=0 span is 2-aligned
    SAFE_W: tl.constexpr,     # no span can run past the row's right edge
    BLOCK_CO: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_C: tl.constexpr,
    NUM_CB: tl.constexpr,
    EVEN_C: tl.constexpr,
    EVEN_CO: tl.constexpr,
    EVEN_P: tl.constexpr,
):
    pid_p = tl.program_id(0)
    pid_co = tl.program_id(1)
    n = tl.program_id(2)

    offs_co = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    m_co = offs_co < COUT
    oh = pid_p // TPR
    ow_t = (pid_p - oh * TPR) * BLOCK_P
    ow = ow_t + tl.arange(0, BLOCK_P)
    m_p = ow < OW
    iw_t = ow_t * 2 - 1                       # odd: what pad 1 forces
    xn = X + n * (C * IMH * IMW)

    span = tl.arange(0, 2 * BLOCK_P)
    if ALIGN8:
        span = tl.max_contiguous(tl.multiple_of(span, 2 * BLOCK_P), 2 * BLOCK_P)
    # A *separate* arange for the kx=0 fetch: its base is only 2-aligned, and
    # letting it share the hinted value above hands it a contiguity claim that
    # belongs to the aligned span.
    span2 = tl.arange(0, 2 * BLOCK_P)

    acc = tl.zeros((BLOCK_CO, BLOCK_P), dtype=tl.float32)
    for cb in range(NUM_CB):
        c = cb * BLOCK_C + tl.arange(0, BLOCK_C)
        m_c = c < C
        cstride = c * (IMH * IMW)
        for ky in range(3):
            ih = oh * 2 - 1 + ky
            row_ok = (ih >= 0) & (ih < IMH)
            off = cstride + ih * IMW
            # aligned span, base 2*ow_t: kx = 1 (even lanes), kx = 2 (odd lanes)
            hi = off + (iw_t + 1)
            if ALIGN8:
                hi = tl.multiple_of(hi, 8)
            mh = m_c[:, None] & row_ok
            if not SAFE_W:
                mh = mh & (iw_t + 1 + span < IMW)[None, :]
            v = tl.load(xn + hi[:, None] + span[None, :], mask=mh, other=0.0)
            a1, a2 = tl.split(tl.reshape(v, (BLOCK_C, BLOCK_P, 2)))
            # kx = 0 needs 2*ow - 1.  Read it as the *odd* lanes of a span
            # based at 2*ow_t - 2 rather than the even lanes of one based at
            # 2*ow_t - 1: same elements, but the base goes from a multiple of 1
            # to a multiple of 2, so the fetch widens from LDG.E.U16 to
            # LDG.E.32 -- two halves per instruction for the one tap that no
            # aligned span can supply.  Worth 1.9 us on case #2.
            lo = iw_t - 1 + span2
            ml = m_c[:, None] & row_ok & (lo >= 0)[None, :]
            if not SAFE_W:
                ml = ml & (lo < IMW)[None, :]
            loff = off + (iw_t - 1)
            if ALIGN2:
                # True only when IMW is even: off carries c*IMH*IMW + ih*IMW, and
                # iw_t - 1 = 2*ow_t - 2 is even on its own.  Claiming it for an
                # odd IMW would let Triton widen a load whose address is only
                # 2-byte aligned -- a misaligned-address fault, not a wrong
                # answer, and one no *scored* shape can reach (all four scored
                # stride-2-eligible widths are even).
                loff = tl.multiple_of(loff, 2)
            v0 = tl.load(xn + loff[:, None] + span2[None, :], mask=ml, other=0.0)
            _, a0 = tl.split(tl.reshape(v0, (BLOCK_C, BLOCK_P, 2)))
            for kx in range(3):
                w = _wtile(Wt, offs_co, m_co, c, m_c, K, C, ky * 3 + kx,
                           EVEN_C, EVEN_CO)
                if kx == 0:
                    acc = tl.dot(w, a0, acc=acc)
                elif kx == 1:
                    acc = tl.dot(w, a1, acc=acc)
                else:
                    acc = tl.dot(w, a2, acc=acc)

    acc = _epilogue(acc, Aff0, Aff1, Aff2, Aff3, eps, offs_co, m_co,
                    FOLD, ACT, EVEN_CO)
    y_off = n * (COUT * P) + offs_co[:, None] * P + (oh * OW + ow)[None, :]
    if EVEN_CO and EVEN_P:
        tl.store(Y + y_off, acc.to(Y.dtype.element_ty))
    else:
        tl.store(Y + y_off, acc.to(Y.dtype.element_ty), mask=m_co[:, None] & m_p[None, :])


# ---------------------------------------------------------------------------
# Host side: gate + plan (constexpr bundle + launch config), built once per
# (shape, dtype) and cached on the module.
#
# The gate is deliberately narrow -- exactly the (dtype, kernel, pad, stride,
# groups, activation) patterns benched in ITERATIONS.md.  fp16 only: every
# captured YOLOConv call is fp16, so bf16/fp32 stay on the composed fallback
# rather than on an unmeasured path.  ``_MAX_FUSED_FLOPS`` is a safety bound, not
# a measured boundary: it keeps a much larger, genuinely compute-bound conv --
# where a tuned cudnn GEMM should win, and where the launch-count argument for
# fusing stops applying -- off this path.
# ---------------------------------------------------------------------------
_MAX_FUSED_FLOPS = 4 << 30
_ACT_NONE, _ACT_SILU = 0, 1


def _tile(v: int, cap: int) -> int:
    """A power-of-two tile edge for a problem axis of length *v*, capped at *cap*.

    Clamped up to 16 because ``tl.dot`` rejects any dimension below 16, and the
    captured YOLOConv set includes convolutions narrower than that -- the network's
    first layer is ``c1=3`` (so a channel tile would be 4).  Over-wide tiles are
    correct, just masked.
    """
    return min(max(triton.next_power_of_2(v), 16), cap)


@functools.lru_cache(maxsize=None)
def _num_sms() -> int:
    return torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count


def _pick_1x1_cfg(p: int, cout: int, c: int, n: int) -> dict:
    """Tile shape for the 1x1 GEMM.

    Swept over 192 configs on the three scored 1x1 cases (see ITERATIONS.md).
    Two things came out of it:

    * ``BLOCK_CO = 16`` wins everywhere.  Widening the output-channel tile
      halves the CTA count without shortening the k loop (K is one dot here), so
      it just costs parallelism: every ``BLOCK_CO = 32`` variant lost a whole
      2.048 us window quantum on cases #4/#5.
    * ``BLOCK_P`` is entirely a *wave-count* choice, not a shape choice.  Cases
      #4 and #5 have identical (P, Cout, C) and differ only in batch, and their
      optima differ (P tile 16 vs 32) -- what they agree on is landing the grid
      at roughly 1.4-3 CTAs per SM.  Below ~1 wave the SMs are idle; past ~3 the
      block turnover costs more than the tile saves (case #5 at 800 CTAs
      measures 15.3 us against 13.3 us at 416).  So pick the *narrowest* p tile
      whose grid still fits in 3 waves.

    K is one dot: the whole reduction is <= 256 channels, and splitting it only
    adds a loop-carried dependency on the accumulator.
    """
    tiles_co = triton.cdiv(cout, 16)
    block_p = 128
    for cand in (16, 32, 64, 128):
        if triton.cdiv(p, cand) * tiles_co * n <= 3 * _num_sms():
            block_p = cand
            break
    return {"BLOCK_CO": 16, "BLOCK_P": block_p,
            "BLOCK_K": _tile(c, 256),
            "num_warps": 4, "num_stages": 1}


def _pick_3x3s2_cfg(cout: int, c: int, ow: int, oh: int, n: int) -> dict:
    """Tile shape for the stride-2 alignment kernel.

    Swept with the differential timer in ``dev/c2.py`` -- the harness window is
    quantised to 2.048 us, which is too coarse to tune against -- over 54
    (BLOCK_P x BLOCK_CO x warps x TY) configs on scored case #2 and 54
    (BLOCK_C x BLOCK_CO x warps) on the three largest captured-but-unscored
    stride-2 shapes: ``[4,32,160,160]`` 32->64 (14.3 us), ``[4,64,40,40]``
    64->128 (10.3 us) and the network's own first convolution
    ``[1,3,640,640]`` 3->16 (10.7 us).

    * ``BLOCK_CO * BLOCK_P <= 2048`` sets the p tile, and ``BLOCK_P`` is capped
      at 64.  That product is the fp32 accumulator, so it is the register
      budget, and it is what separates the four shapes: case #2 and the c1=3
      first convolution take ``BLOCK_CO`` 32 and 16 and want the wide p tile
      (case #2 is 16.6 us at 64 against 20.8 at 32 -- the kx=0 fetch moves two
      halves per instruction, so a longer span amortises better), while
      ``[4,32,160,160]`` takes ``BLOCK_CO=64`` and wants 32 (14.8 vs 16.5), and
      ``[4,64,40,40]`` has OW=20 so a 64-wide tile would be mostly masked.
      ``BLOCK_P=128`` is 7 us worse on case #2 at any warp count.  ``TY``
      (sharing the one input row consecutive output rows have in common) is a
      small loss at every tile, so this kernel does one output row per CTA.
    * ``BLOCK_C`` spans the whole channel axis.  Every shape's best config took
      the widest channel tile available (case #2 at 16 is 2.1 us better than a
      split reduction; ``[4,32,160,160]`` wants 32, ``[4,64,40,40]`` wants 64),
      because a narrower one re-fetches the same input span per c block.
    * ``BLOCK_CO`` is a *wave-count* choice, exactly as on the 1x1 path.  With
      the channel axis already whole, widening the output-channel tile only
      trades CTAs for work per CTA: ``[4,32,160,160]`` still has 960 CTAs at
      ``BLOCK_CO=64`` and wants that, while ``[4,64,40,40]`` would drop to 160
      and prefers 32 (12.5 vs 13.4).  So take the widest tile that still leaves
      ~2 waves of CTAs, then let the product bound above pick ``BLOCK_P``.
    * ``num_warps = 2``, except one warp when the channel axis is tiny (the
      c1=3 first convolution measures 10.66 at w1 against 11.12 at w2).  Note
      this differs from the previous kernel, where one warp won everywhere
      (31.73 vs 33.70 on case #2); that kernel's input loads were entirely
      unvectorised, and splitting a vectorised span across two warps now pays.
    """
    tiles_p = triton.cdiv(ow, 32) * oh * n
    block_co = 16
    for cand in (64, 32, 16):
        if cand > triton.next_power_of_2(cout):
            continue
        block_co = cand
        if tiles_p * triton.cdiv(cout, cand) >= 2 * _num_sms():
            break
    block_co = max(block_co, 16)
    block_p = 64 if (ow >= 64 and block_co * 64 <= 2048) else 32
    return {"BLOCK_CO": block_co, "BLOCK_P": block_p,
            "BLOCK_C": _tile(c, 64),
            "num_warps": 1 if c < 8 else 2, "num_stages": 1}


def _pick_3x3_cfg(p: int, cout: int, c: int, k: int, n: int) -> dict:
    """Tile shape for the stride-1 padded implicit GEMM.

    Swept over 216 flat configs on the scored 3x3 cases; see ITERATIONS.md.

    * few CTAs -- case #1 ``[1,64,20,20]`` 64->64 s1, 52 CTAs on 148 SMs.
      Nothing outside the CTA hides memory latency, so the win is a wide k
      block: the whole 73 KB filter and its A tile become one batch of
      independent loads feeding one dot (BLOCK_K 128 -> 256 moves it
      15.4 -> 13.3 us).  ``num_stages`` is flat (the k loop is a static_range,
      so it is already unrolled).
    * many CTAs -- no *scored* case lands here, so it is tuned on the three
      largest captured-but-unscored stride-1 shapes instead: ``[4,64,80,80]``
      64->64 (23.6 us), ``[4,128,40,40]`` 128->128 (27.7 us) and
      ``[4,32,80,80]`` 32->32 (17.4 us).  A wide output channel tile is what
      these want -- the opposite of the 1x1 path -- because here it also
      amortises the much larger filter across the k loop.
    """
    ctas = triton.cdiv(p, 32) * triton.cdiv(cout, 16) * n
    if ctas < 2 * _num_sms():
        return {"BLOCK_CO": _tile(cout, 16), "BLOCK_P": 32,
                "BLOCK_K": _tile(k, 256), "num_warps": 4, "num_stages": 1}
    return {"BLOCK_CO": _tile(cout, 64), "BLOCK_P": 64,
            "BLOCK_K": _tile(k, 128), "num_warps": 4, "num_stages": 1}


def _plan(x: torch.Tensor, conv: Conv2d, act: int, fold: int) -> dict | None:
    """Return a launch plan when (x, conv, act) match a benched fused pattern."""
    weight = conv.weight
    if x.dim() != 4 or weight.dim() != 4:
        return None
    if x.dtype is not torch.float16 or weight.dtype is not x.dtype:
        return None
    if conv.groups != 1 or conv.dilation != (1, 1):
        return None
    if not weight.is_contiguous():
        return None

    n, c, h, w = (int(v) for v in x.shape)
    cout = int(weight.shape[0])
    if int(weight.shape[1]) != c:
        return None
    kh, kw = int(weight.shape[2]), int(weight.shape[3])
    ph, pw = int(conv.padding[0]), int(conv.padding[1])
    sh, sw = int(conv.stride[0]), int(conv.stride[1])

    if (kh, kw) == (1, 1) and (ph, pw) == (0, 0) and (sh, sw) == (1, 1):
        kind = "1x1"
    elif (kh, kw) == (3, 3) and (ph, pw) == (1, 1) and sh == sw == 2:
        kind = "3x3s2"
    elif (kh, kw) == (3, 3) and (ph, pw) == (1, 1) and sh == sw == 1:
        kind = "3x3"
    else:
        return None

    oh = (h + 2 * ph - kh) // sh + 1
    ow = (w + 2 * pw - kw) // sw + 1
    if oh <= 0 or ow <= 0:
        return None
    p = oh * ow
    k = c * kh * kw
    if 2 * n * p * cout * k > _MAX_FUSED_FLOPS:
        return None

    if kind == "1x1":
        cfg = _pick_1x1_cfg(p, cout, c, n)
        consts = {"P": p, "COUT": cout, "C": c,
                  "NUM_K": triton.cdiv(c, cfg["BLOCK_K"]),
                  "EVEN_K": c % cfg["BLOCK_K"] == 0}
        grid_p = triton.cdiv(p, cfg["BLOCK_P"])
    elif kind == "3x3s2":
        cfg = _pick_3x3s2_cfg(cout, c, ow, oh, n)
        block_p = cfg["BLOCK_P"]
        consts = {"C": c, "IMH": h, "IMW": w, "COUT": cout, "OW": ow, "P": p,
                  "TPR": triton.cdiv(ow, block_p), "K": k,
                  # The even-base span is (c*IMH*IMW + ih*IMW + 2*ow_t) + [0, 2*BP),
                  # every term of which is a multiple of 8 halves once IMW is
                  # (BLOCK_P >= 16, so 2*BLOCK_P always is).  X itself is
                  # 16 B aligned via Triton's own pointer specialization.
                  "ALIGN8": w % 8 == 0,
                  "ALIGN2": w % 2 == 0,
                  # Right edge: the aligned span reaches 2*OW - 1 and the
                  # misaligned one 2*OW - 2, so both stay inside the row exactly
                  # when the p tiles divide OW and 2*OW <= IMW (which fails for
                  # odd IMW, where OW = (IMW+1)/2).
                  "SAFE_W": ow % block_p == 0 and 2 * ow <= w,
                  "BLOCK_C": cfg["BLOCK_C"],
                  "NUM_CB": triton.cdiv(c, cfg["BLOCK_C"]),
                  "EVEN_C": c % cfg["BLOCK_C"] == 0}
        grid_p = consts["TPR"] * oh
    else:
        cfg = _pick_3x3_cfg(p, cout, c, k, n)
        consts = {"C": c, "IMH": h, "IMW": w, "COUT": cout, "OW": ow, "P": p,
                  "K": k, "KW": kw, "SH": sh, "SW": sw, "PH": ph, "PW": pw,
                  "NUM_K": triton.cdiv(k, cfg["BLOCK_K"]),
                  "EVEN_K": k % cfg["BLOCK_K"] == 0}
        grid_p = triton.cdiv(p, cfg["BLOCK_P"])

    block_p = cfg["BLOCK_P"]
    consts.update({
        "FOLD": fold, "ACT": act,
        "BLOCK_CO": cfg["BLOCK_CO"], "BLOCK_P": block_p,
        "EVEN_CO": cout % cfg["BLOCK_CO"] == 0,
        "EVEN_P": (ow % block_p == 0) if kind == "3x3s2" else (p % block_p == 0),
    })
    launch = {kk: cfg[kk] for kk in ("num_warps", "num_stages")}
    if kind != "3x3s2":
        consts["BLOCK_K"] = cfg["BLOCK_K"]
    return {
        "kind": kind,
        "out_shape": (n, cout, oh, ow),
        "grid": (grid_p, triton.cdiv(cout, cfg["BLOCK_CO"]), n),
        "cfg": launch,
        "consts": consts,
    }


def _transposed_weight(weight: torch.Tensor) -> torch.Tensor:
    """3x3 paths: ``[COUT, C, KH, KW] -> [COUT, KH*KW, C]``, so a weight tile is
    contiguous along the channel axis it blocks over -- see ``_wtile``."""
    return weight.permute(0, 2, 3, 1).reshape(weight.shape[0], -1).contiguous()


_KERNELS = {"1x1": _conv1x1_act_kernel, "3x3": _conv3x3_act_kernel,
            "3x3s2": _conv3x3s2_act_kernel}


class YOLOConv(nn.Module):
    default_act = SiLU()

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 1,
        s: int = 1,
        p=None,
        g: int = 1,
        d: int = 1,
        act=True,
    ):
        super().__init__()
        self.conv = Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = BatchNorm2d(c2, eps=1e-3, momentum=0.03)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self._is_fused = False
        # Fused-path caches. Plain attributes, not buffers/parameters: they are
        # derived state, must not appear in ``state_dict`` and must not be cast
        # by a ``.half()`` on the module.
        self._plans: dict = {}
        self._wt: tuple | None = None

    # -- fused-path helpers -------------------------------------------------
    def _act_code(self) -> int | None:
        """``ACT`` constexpr for ``self.act``, or None if it is not fusable."""
        act = self.act
        if isinstance(act, (SiLU, nn.SiLU)):
            return _ACT_SILU
        if isinstance(act, nn.Identity):
            return _ACT_NONE
        return None

    def _affine(self):
        """The four tensors the epilogue folds, plus ``eps``, or None.

        Nothing is precomputed and nothing is cached -- the epilogue derives
        ``(scale, shift)`` itself, on a ``BLOCK_CO``-long vector.  Returns None
        when BN is not in its running-stats affine regime, in which case the
        caller must fall back to the composed modules.
        """
        if self._is_fused:
            bias = self.conv.bias
            if bias is None:
                return None
            return 0, bias, bias, bias, bias, 0.0
        bn = self.bn
        if (bn.training or not bn.track_running_stats or not bn.affine
                or bn.running_mean is None or bn.running_var is None):
            return None
        return (1, bn.weight, bn.bias, bn.running_mean, bn.running_var,
                float(bn.eps))

    def _plan_for(self, x: torch.Tensor, act: int, fold: int):
        key = (x.shape, x.dtype, act, fold)
        plan = self._plans.get(key, False)
        if plan is False:
            plan = _plan(x, self.conv, act, fold)
            self._plans[key] = plan
        return plan

    def _packed_weight(self, plan: dict) -> torch.Tensor:
        w = self.conv.weight
        if plan["kind"] == "1x1":
            return w  # [COUT, C, 1, 1] is already the [COUT, C] GEMM operand
        key = (w.data_ptr(), w._version)
        cached = self._wt
        if cached is None or cached[0] != key:
            cached = (key, _transposed_weight(w))
            self._wt = cached
        return cached[1]

    # -- forward ------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda and self.conv.weight.is_cuda:
            act = self._act_code()
            if act is not None:
                affine = self._affine()
                if affine is not None:
                    fold, a0, a1, a2, a3, eps = affine
                    plan = self._plan_for(x, act, fold)
                    if plan is not None:
                        return _run(x if x.is_contiguous() else x.contiguous(),
                                    self._packed_weight(plan),
                                    a0, a1, a2, a3, eps, plan)
        if self._is_fused:
            return self.act(self.conv(x))
        return self.act(self.bn(self.conv(x)))

    @torch.no_grad()
    def fuse(self):
        if self._is_fused:
            return self
        fused_weight, fused_bias = _fuse_conv_bn(self.conv, self.bn)
        self.conv.weight.data.copy_(fused_weight)
        self.conv.bias = nn.Parameter(fused_bias)
        delattr(self, "bn")
        self._is_fused = True
        # Drop the fused-path caches.  ``weight.data.copy_()`` above
        # deliberately bypasses autograd, so it leaves ``weight._version``
        # untouched (verified) -- a version-counter key cannot see this change
        # on its own, which is what makes clearing here load-bearing rather
        # than defensive.
        self._plans.clear()
        self._wt = None
        return self


def _run(x: torch.Tensor, weight: torch.Tensor, a0, a1, a2, a3, eps: float,
         plan: dict) -> torch.Tensor:
    y = torch.empty(plan["out_shape"], dtype=x.dtype, device=x.device)
    _KERNELS[plan["kind"]][plan["grid"]](
        x, weight, a0, a1, a2, a3, y, eps, **plan["consts"], **plan["cfg"])
    return y


def fuse_module(module: nn.Module) -> nn.Module:
    from .yolov10_repvggdw import YOLORepVGGDW

    for child in module.children():
        fuse_module(child)
    if isinstance(module, YOLOConv):
        module.fuse()
    elif isinstance(module, YOLORepVGGDW):
        module.fuse()
    return module
