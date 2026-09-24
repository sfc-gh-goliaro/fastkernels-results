"""YOLOv10 C2f and C2fCIB blocks, restructured for inference.

The captured workloads are small (batch 1 or 4, 20x20..160x160, 32..384
channels).  Two measurements set the whole strategy:

* **The scored window starts host-launch-bound.**  The baseline's time is a
  dead-straight line through ``~13 us x (ops issued per forward)``: 14 ops ->
  165-190 us (C2f n=1), 21 -> 302 us (n=2), 26 -> 345 us (C2fCIB).  Nothing
  here is FLOP-bound.
* **Once the host cost is gone, cudnn's per-conv overhead is.**  Profiling one
  ``F.conv2d(bias=...)`` 3x3 on [1,128,20,20]: ``nchwToNhwc`` 2.1 +
  ``nchwToNhwc`` 2.4 + cutlass implicit GEMM 4.7 + ``nhwcToNchw`` 2.1 + a
  separate 3.0 us bias elementwise = 15.2 us, of which 10.5 us is layout
  shuffling and an unfused bias.

Five changes, in the order they were measured (geometric-mean speedup over the
seven scored cases after each):

1. **BatchNorm folding** -- 2.04x.  The bench calls ``eval()`` and never calls
   ``YOLOConv.fuse()``, so the baseline pays a full activation-sized
   ``batch_norm`` read+write after *every* conv (4 in C2f n=1, 8 in the CIB).
   Each conv's BN is folded into its weight/bias once, lazily on the first
   forward (the weights are only final after the harness' fp16 cast plus
   ``load_state_dict``), in fp32, cast back to the conv dtype.  Folding
   ``YOLORepVGGDW``'s 3x3 into its 7x7 -- exactly what the module's own
   ``fuse()`` does -- collapses 6 ops to 2 on the ``lk=True`` CIB path.
2. **Whole-block CUDA graph** -- 2.68x.  Captured once per (shape, dtype): copy
   the input into the graph's static buffer, replay, clone the static output.
   Three host ops for any block size, which is what turns the block from
   host-bound into device-bound.
3. **Fused epilogue + shared concat buffer** -- 4.65x.  One Triton kernel does
   ``dst[slice] = silu(raw + bias) + residual`` with strided source and
   destination, so the bias pass (3.0 us), the SiLU pass (2.1 us), the residual
   add (1.6 us) and ``torch.cat`` (1.7-6.8 us) become one 2 us pass.  ``cv1``
   writes channels [0, 2c) of a single ``(2+n)*c``-channel buffer and each
   bottleneck/CIB writes its own slice, so ``cv2`` consumes a contiguous tensor
   with no copy at all.  Dense 3x3s went to ``candidate/L1/conv2d.py``'s fused
   NCHW implicit GEMM here (15.2 -> 5.4 us).
4. **Fused depthwise conv** -- 4.97x.  The CIB's three ``groups == channels``
   convs, conv and epilogue in one launch: the folded 7x7 on [4,256,20,20]
   goes 20.1 -> 7.6 us against cudnn + silu + add, the 3x3s 9.7 -> 2.3 us.
5. **Fused dense conv** -- 5.90x.  The same treatment for every remaining conv
   (1x1 and dense 3x3): one implicit-GEMM launch that ends in the epilogue, so
   nothing writes an activation-sized intermediate anywhere in the block.

That leaves one device op per conv plus the graph's copy-in and clone-out, and
at that point the *scheduling* of those 6-9 nodes is 35-45% of the window --
which is what the last two changes attack.

6. **Tile for occupancy in the few-CTA regime** (``_fill_machine``).  The dense
   3x3 on [1,128,20,20] launched 52 CTAs on 148 SMs; narrowing the pixel tile
   until the launch covers at least half the machine takes it 7.76 -> 5.90 us,
   and above that threshold the wider tile's reuse wins instead, so it is a
   cliff and not a preference.  Case 0's device time 25.8 -> 22.4 us.
7. **The megakernel** -- one launch for the *entire block*, replacing the
   graph-of-N-convs with a persistent 148-CTA (one per SM) grid that runs every
   stage in sequence with a grid-wide barrier between them.  A barrier costs
   1.54 us against 6.02 us of fixed graph cost plus 1.10 us per graph node plus
   two memcpys, so the arithmetic closes on the five smaller cases and the
   window drops 1.14-1.37x.  On the two batch-4 cases the fixed grid is the
   bottleneck instead and the graph path still wins, so the choice is made per
   shape from measured tile work (``_mega_wins``).  Both paths ship.

  ops per forward   baseline -> here      device kernels   window us
  C2f n=1              14 -> 1 or 6          1 or 6         27.6-56.4
  C2f n=2              21 -> 1                  1              41.0
  C2fCIB(lk) n=1       26 -> 1 or 9          1 or 9         37.9-46.1

Correctness and contracts.  Every parameter and buffer keeps its baseline name
(the folded tensors live outside the module tree), so the harness'
``load_state_dict(baseline.state_dict(), strict=False)`` shares weights
exactly -- ``tools/robust.py`` asserts the two key sets are identical, because
a rename here would silently drop weights and "pass" on garbage.  The
fold/graph cache is guarded by ``(data_ptr, _version)`` over every tensor it
was derived from, so a later ``load_state_dict`` (also hooked), an in-place
weight or BN-statistic edit, or a call to ``fuse()`` (which bumps
``conv.weight``'s version and drops ``bn``) invalidates it and the plan is
rebuilt.  ``tools/robust.py`` checks each of those against the baseline with
shared weights and non-identity BN statistics, plus training mode, grad-enabled
autograd, non-contiguous inputs, CPU, fp32 and bf16, ``lk=False``, and ``n=0``.
Anything unrecognized in the module tree, a missing Triton, or an uncapturable
graph each fall back one step, ending at the literal baseline forward.

One caveat the megakernel carries: a persistent grid-barrier kernel needs all
its CTAs co-resident, which one CTA per SM guarantees only as long as nothing
else is running on the device.  The bench pins one worker per GPU, and the
block's own work is serialized on one stream, so that holds here; it is why the
grid is exactly ``_num_sms()`` and never larger, and why the graph path stays
live rather than being deleted.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .yolov10_bottleneck import YOLOBottleneck
from .yolov10_cib import YOLOCIB
from .yolov10_conv import YOLOConv
from .yolov10_repvggdw import YOLORepVGGDW

try:  # the frozen L1 winner: fused NCHW padded-3x3 implicit GEMM, one launch
    from ..L1.conv2d import Conv2d as _L1Conv2d
except Exception:  # noqa: BLE001 - fall back to F.conv2d for the 3x3s
    _L1Conv2d = None

try:
    import triton
    import triton.language as tl
except Exception:  # noqa: BLE001 - epilogues fall back to torch ops
    triton = None

_ONE = (1, 1)

# Knobs, for attributing each lever in its own bench rather than for tuning.
_USE_GRAPH = os.environ.get("FK_C2F_GRAPH", "1") != "0"
_USE_L1 = os.environ.get("FK_C2F_L1CONV", "1") != "0"
_USE_TRITON_EPI = os.environ.get("FK_C2F_EPI", "1") != "0"
_USE_DW = os.environ.get("FK_C2F_DW", "1") != "0"
# Which dense convs take the fused epilogue kernel: "off", "1x1" or "all".
_DENSE_MODE = os.environ.get("FK_C2F_DENSE", "all")
# Megakernel (one launch for the whole block): "0" off, "1" always, "auto"
# per-shape from the measured table in ``_mega_wins``.
_MEGA_MODE = os.environ.get("FK_C2F_MEGA", "auto")
_MAX_GRAPHS = 16
_MISSING = object()
_SMS = None
# One 128-byte line per stage counter, so a CTA polling stage s never shares a
# line with an arrival at stage s+1.
_LOCK_STRIDE = 32
# Warps per CTA for the megakernel.  Barrier cost roughly doubles from 4 to 8
# warps (1.54 -> 2.73 us at 148 CTAs), so 4 is the default.
_MEGA_WARPS = int(os.environ.get("FK_C2F_MEGA_WARPS", "4"))
# Megakernel tile policy: "cap" (fits 4 warps) or "raw" (the swept tile).
_MEGA_TILE = os.environ.get("FK_C2F_MEGA_TILE", "raw")


# ---------------------------------------------------------------------------
# Fused conv epilogue:  dst[n, c, p] = silu(src[n, c, p] + bias[c]) + res[n, c, p]
#
# One launch for what otherwise costs a bias broadcast-add, a SiLU pass, a
# residual add and a slice of torch.cat.  Source, residual and destination each
# carry their own (batch, channel) strides so any of them can be a channel slice
# of the shared (2+n)*c concat buffer; the pixel axis of an NCHW channel slice
# is always contiguous, so the inner access stays a straight vector load.
# ``C`` is a constexpr: the channel count is fixed per call site and it turns
# the flat program id -> (n, c) split into a magic-number division.
# ---------------------------------------------------------------------------
if triton is not None:

    @triton.jit
    def _epi_kernel(SRC, BIAS, RES, DST,
                    P,
                    src_sn, src_sc, res_sn, res_sc, dst_sn, dst_sc,
                    C: tl.constexpr,
                    HAS_BIAS: tl.constexpr,
                    ACT: tl.constexpr,
                    HAS_RES: tl.constexpr,
                    BLOCK: tl.constexpr,
                    EVEN_P: tl.constexpr):
        nc = tl.program_id(0)
        n = nc // C
        c = nc - n * C
        offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        if EVEN_P:
            v = tl.load(SRC + n * src_sn + c * src_sc + offs).to(tl.float32)
        else:
            m = offs < P
            v = tl.load(SRC + n * src_sn + c * src_sc + offs, mask=m, other=0.0).to(tl.float32)
        if HAS_BIAS:
            v += tl.load(BIAS + c).to(tl.float32)
        if ACT:
            v *= tl.sigmoid(v)
        if HAS_RES:
            r = RES + n * res_sn + c * res_sc + offs
            if EVEN_P:
                v += tl.load(r).to(tl.float32)
            else:
                v += tl.load(r, mask=offs < P, other=0.0).to(tl.float32)
        o = DST + n * dst_sn + c * dst_sc + offs
        if EVEN_P:
            tl.store(o, v.to(DST.dtype.element_ty))
        else:
            tl.store(o, v.to(DST.dtype.element_ty), mask=offs < P)


    # -----------------------------------------------------------------------
    # Fused depthwise conv + epilogue (NCHW in and out, stride 1, odd kernel):
    #
    #   y[n,c,oh,ow] = silu(sum_{i,j} x[n,c,oh-PH+i,ow-PW+j]*w[c,i,j] + b[c])
    #                  + res[n,c,oh,ow]
    #
    # cudnn runs the CIB's depthwise convs at 2.6 us (3x3) and 16.0 us (the
    # RepVGGDW-folded 7x7 on [4,256,20,20]) and needs a second kernel for the
    # bias -- for 0.8 MB of unique data, i.e. two orders of magnitude off
    # bandwidth.  Each output channel here reads exactly one input channel, so
    # one program owns a whole (n, c) plane: the k*k taps are a fully unrolled
    # static nest over one resident plane, every tap load is contiguous along
    # ow, and the halo is masked instead of padded.  One launch, no layout
    # change, bias/SiLU/residual/destination-slice all in the epilogue.
    # -----------------------------------------------------------------------
    @triton.jit
    def _dw_kernel(X, W, B, RES, Y,
                   P,
                   x_sn, x_sc, res_sn, res_sc, y_sn, y_sc,
                   C: tl.constexpr, IMH: tl.constexpr, IMW: tl.constexpr,
                   OW: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
                   PH: tl.constexpr, PW: tl.constexpr,
                   HAS_BIAS: tl.constexpr, ACT: tl.constexpr, HAS_RES: tl.constexpr,
                   BLOCK: tl.constexpr, EVEN_P: tl.constexpr):
        nc = tl.program_id(0)
        n = nc // C
        c = nc - n * C
        offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        m_p = offs < P
        oh = offs // OW
        ow = offs - oh * OW
        xb = X + n * x_sn + c * x_sc
        wb = W + c * (KH * KW)
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        for i in tl.static_range(KH):
            ih = oh - PH + i
            ok_h = (ih >= 0) & (ih < IMH)
            if not EVEN_P:
                ok_h = ok_h & m_p
            row = xb + ih * IMW
            for j in tl.static_range(KW):
                iw = ow - PW + j
                ok = ok_h & (iw >= 0) & (iw < IMW)
                a = tl.load(row + iw, mask=ok, other=0.0)
                acc += a.to(tl.float32) * tl.load(wb + i * KW + j).to(tl.float32)
        if HAS_BIAS:
            acc += tl.load(B + c).to(tl.float32)
        if ACT:
            acc *= tl.sigmoid(acc)
        if HAS_RES:
            r = RES + n * res_sn + c * res_sc + offs
            if EVEN_P:
                acc += tl.load(r).to(tl.float32)
            else:
                acc += tl.load(r, mask=m_p, other=0.0).to(tl.float32)
        o = Y + n * y_sn + c * y_sc + offs
        if EVEN_P:
            tl.store(o, acc.to(Y.dtype.element_ty))
        else:
            tl.store(o, acc.to(Y.dtype.element_ty), mask=m_p)



    # -----------------------------------------------------------------------
    # Fused dense conv + epilogue (NCHW in and out, groups 1, stride 1,
    # shape-preserving padding), as one implicit GEMM with
    # ``M = OH*OW``, ``N = Cout``, ``K = Cin*KH*KW``:
    #
    #   y[n,co,p] = silu(sum_{c,i,j} x[n,c,ih,iw]*w[co,c,i,j] + b[co]) + res[n,co,p]
    #
    # The structure (acc laid out [BLOCK_CO, BLOCK_P] so the store walks the
    # output's contiguous pixel axis, the A tile gathered straight out of NCHW
    # inside a statically unrolled k loop, the pad-1 halo handled by masked
    # loads rather than a padding pass, the weight pre-transposed to
    # [COUT, KH*KW, C] so a k tile is contiguous along the channel axis it
    # blocks over) is the one ``candidate/L1/conv2d.py`` established and swept
    # for exactly this yolov10n backbone conv.  What is added here is the fused
    # epilogue and explicit (batch, channel) strides on x / res / y, which is
    # what lets a conv read one channel slice of the shared concat buffer and
    # write another without a contiguous() copy or a separate pass.
    #
    # ``PADDED`` is False for 1x1, where the gather is affine and every mask
    # folds away.
    # -----------------------------------------------------------------------
    @triton.jit
    def _dense_kernel(X, WT, B, RES, Y,
                      x_sn, x_sc, res_sn, res_sc, y_sn, y_sc,
                      C: tl.constexpr, IMH: tl.constexpr, IMW: tl.constexpr,
                      COUT: tl.constexpr, OW: tl.constexpr, P: tl.constexpr,
                      K: tl.constexpr, KW: tl.constexpr,
                      PH: tl.constexpr, PW: tl.constexpr,
                      HAS_BIAS: tl.constexpr, ACT: tl.constexpr, HAS_RES: tl.constexpr,
                      PADDED: tl.constexpr,
                      BLOCK_CO: tl.constexpr, BLOCK_P: tl.constexpr,
                      BLOCK_K: tl.constexpr, NUM_K: tl.constexpr,
                      EVEN_K: tl.constexpr, EVEN_CO: tl.constexpr,
                      EVEN_P: tl.constexpr):
        pid_p = tl.program_id(0)
        pid_co = tl.program_id(1)
        n = tl.program_id(2)

        offs_co = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
        m_co = offs_co < COUT
        offs_p = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
        m_p = offs_p < P
        xn = X + n * x_sn
        acc = tl.zeros((BLOCK_CO, BLOCK_P), dtype=tl.float32)

        if PADDED:
            oh = offs_p // OW
            ih0 = oh - PH
            iw0 = (offs_p - oh * OW) - PW

        for kb in tl.static_range(NUM_K):
            offs_k = kb * BLOCK_K + tl.arange(0, BLOCK_K)
            tap = offs_k // C
            c = offs_k - tap * C
            if PADDED:
                ih = ih0[None, :] + (tap // KW)[:, None]
                iw = iw0[None, :] + (tap % KW)[:, None]
                ok = (ih >= 0) & (ih < IMH) & (iw >= 0) & (iw < IMW)
                if not EVEN_K:
                    ok = ok & (offs_k < K)[:, None]
                if not EVEN_P:
                    ok = ok & m_p[None, :]
                a = tl.load(xn + c[:, None] * x_sc + ih * IMW + iw, mask=ok, other=0.0)
            else:
                a_off = c[:, None] * x_sc + offs_p[None, :]
                if EVEN_K and EVEN_P:
                    a = tl.load(xn + a_off)
                else:
                    ok = (offs_k < K)[:, None] if EVEN_P else (
                        m_p[None, :] if EVEN_K else (offs_k < K)[:, None] & m_p[None, :])
                    a = tl.load(xn + a_off, mask=ok, other=0.0)
            w_off = offs_co[:, None] * K + offs_k[None, :]
            if EVEN_CO and EVEN_K:
                w = tl.load(WT + w_off)
            else:
                w_mask = m_co[:, None] if EVEN_K else (
                    (offs_k < K)[None, :] if EVEN_CO else m_co[:, None] & (offs_k < K)[None, :])
                w = tl.load(WT + w_off, mask=w_mask, other=0.0)
            acc = tl.dot(w, a, acc=acc)

        if HAS_BIAS:
            if EVEN_CO:
                acc += tl.load(B + offs_co)[:, None].to(tl.float32)
            else:
                acc += tl.load(B + offs_co, mask=m_co, other=0.0)[:, None].to(tl.float32)
        if ACT:
            acc *= tl.sigmoid(acc)
        if HAS_RES:
            r = RES + n * res_sn + offs_co[:, None] * res_sc + offs_p[None, :]
            if EVEN_CO and EVEN_P:
                acc += tl.load(r).to(tl.float32)
            else:
                acc += tl.load(r, mask=m_co[:, None] & m_p[None, :], other=0.0).to(tl.float32)
        o = Y + n * y_sn + offs_co[:, None] * y_sc + offs_p[None, :]
        if EVEN_CO and EVEN_P:
            tl.store(o, acc.to(Y.dtype.element_ty))
        else:
            tl.store(o, acc.to(Y.dtype.element_ty), mask=m_co[:, None] & m_p[None, :])


def _dw_conv(x, weight, bias, act, res, dst, kh, kw, ph, pw, cfg=None):
    """One fused depthwise conv + epilogue launch."""
    n, c, imh, imw = x.shape
    oh, ow = imh, imw  # stride 1, padding k//2
    p = oh * ow
    if dst is None:
        dst = torch.empty((n, c, oh, ow), dtype=x.dtype, device=x.device)
    block, warps = _dw_cfg(p) if cfg is None else cfg
    _dw_kernel[(n * c, (p + block - 1) // block)](
        x, weight, bias, res, dst,
        p,
        x.stride(0), x.stride(1),
        0 if res is None else res.stride(0), 0 if res is None else res.stride(1),
        dst.stride(0), dst.stride(1),
        C=c, IMH=imh, IMW=imw, OW=ow, KH=kh, KW=kw, PH=ph, PW=pw,
        HAS_BIAS=bias is not None, ACT=act, HAS_RES=res is not None,
        BLOCK=block, EVEN_P=(p % block == 0), num_warps=warps,
    )
    return dst


def _num_sms():
    global _SMS
    if _SMS is None:
        _SMS = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    return _SMS


def _dense_cfg(p, cout, c, k, n, padded):
    """Tile shape + warps for the fused dense conv, swept over every dense conv
    the seven scored cases issue (18 shapes, ~40 configs each; see
    ITERATIONS.md for the table).

    ``BLOCK_P = 64`` won or tied everywhere except the few-CTA 3x3s; what
    actually moves the number is how wide the k block is and how many warps,
    and that splits on two axes:

    * **pixels per image** (``p >= 4096``: [4,*,80,80] and [1,*,160,160]).  A
      fat M dimension wants a narrow k block -- past the channel count a wider
      one only buys masked-off lanes -- while a small one wants the widest k
      block available so the whole filter plus its A tile is a single batch of
      independent loads feeding one MMA.  This is the same effect
      ``candidate/L1/conv2d.py`` swept for the 3x3.
    * **CTA count** ``cdiv(p,32)*cdiv(cout,32)*n`` against two per SM.  Batch 4
      at 20x20 has only 400 pixels but 416 CTAs, so it is occupancy-rich
      despite the small image and behaves like the fat-M regime; keying on
      ``n*p`` alone mispredicts it by 2x (9.9 us against 5.2 us).

    Measured (us, this kernel / cudnn+epilogue / L1-triton+epilogue):
      3x3 [1,128,20,20]    5.6 / 13.7 / 7.2     3x3 [4,32,80,80]    8.0 / 16.5 / 12.6
      3x3 [1,16,160,160]   4.7 / 18.1 / 8.4     3x3 [1,64,40,40]    5.2 / 13.2 / 6.9
      1x1 [1,256,20,20]    2.8 /  4.5 / 7.7     1x1 [4,192,80,80]   6.6 /  6.5 / 12.7
      1x1 [1,32,160,160]   3.1 /  4.9 / 8.6     1x1 [1,128,40,40]   2.4 /  4.5 / 7.5
      1x1 [4,384,20,20]    5.2 /  5.2 / 8.7     1x1 [1,128,20,20]   2.3 /  4.3 / 7.4
    i.e. a clear win everywhere except the one 1x1 where cudnn is already at
    its bandwidth floor, and there it is a tie -- and even there the fused form
    still saves the separate epilogue node.
    """
    return _fill_machine(_dense_cfg_base(p, cout, c, k, n, padded), p, cout, n)


def _dense_cfg_base(p, cout, c, k, n, padded):
    """The parent's swept tile, before the fill-the-machine correction."""
    pow2 = triton.next_power_of_2
    fat_m = p >= 4096
    if padded:
        if fat_m:
            cfg = (min(pow2(cout), 32), 64, min(pow2(k), 128), 4, 1)
        else:
            cfg = (min(pow2(cout), 32), 32, min(pow2(k), 256), 4, 1)
    elif fat_m:
        if cout >= 64:
            cfg = (min(pow2(cout), 64), 64, min(pow2(k), 256), 8, 1)
        else:
            cfg = (min(pow2(cout), 32), 64, min(pow2(k), 128), 4, 1)
    elif -(-p // 32) * -(-cout // 32) * n >= 2 * _num_sms():
        cfg = (min(pow2(cout), 32), 64, min(pow2(k), 128), 4, 1)
    else:
        cfg = (min(pow2(cout), 32), 64, min(pow2(k), 256), 8, 1)
    return cfg


def _fill_machine(cfg, p, cout, n):
    """Narrow the pixel tile until the launch covers at least half the machine.

    This is the answer to the sweep the parent abandoned on compile time.  With
    the k loop taken off ``tl.static_range`` (``tl.range(..., loop_unroll_factor)``)
    the sweep runs in seconds; ``tools/sweep3x3.py`` then re-times the top tiles
    in the shipped fully-unrolled form, which is what these numbers are:

      dense 3x3 [1,128,20,20]  BLOCK_P 32 ->  52 CTAs  7.76 us
                               BLOCK_P 16 -> 100 CTAs  5.90 us   <-- 1.31x
      dense 3x3 [1, 64,40,40]  BLOCK_P 32 -> 100 CTAs  5.02 us   <-- already full
                               BLOCK_P 16 -> 200 CTAs  6.32 us
      1x1 [1,256,20,20]->256   BLOCK_P 64 ->  56 CTAs  2.94 us
                               BLOCK_P 32 -> 104 CTAs  2.69 us
      1x1 [1,256,20,20]->128   BLOCK_P 64 ->  28 CTAs  2.90 us
                               BLOCK_P 16 -> 100 CTAs  2.58 us
      1x1 [4,256,20,20]->128   BLOCK_P 64 -> 112 CTAs  3.01 us   <-- already full
      dense 3x3 [4, 32,80,80]  BLOCK_P 64 -> 400 CTAs  7.68 us   <-- already full

    So the parent's ``BLOCK_P = 64 won or tied everywhere EXCEPT the few-CTA
    3x3s`` exception is an occupancy cliff and nothing more: below about half a
    wave, halving the pixel tile is worth 1.1-1.3x; at or above it, the wider
    tile's reuse wins and narrowing costs 1.0-1.26x.  Half of ``_num_sms()`` is
    the measured crossover (52 CTAs narrows, 100 does not).  Dropping to 4 warps
    with the narrow tile is part of the win -- 8 warps on a 16-pixel tile leaves
    most lanes with nothing to do.
    """
    block_co, block_p, block_k, warps, stages = cfg
    half = max(1, _num_sms() // 2)
    co_tiles = -(-cout // block_co) * n
    while block_p > 16 and -(-p // block_p) * co_tiles < half:
        block_p //= 2
        warps = 4
    return (block_co, block_p, block_k, warps, stages)


def _dense_weight(weight):
    """[COUT, Cin, KH, KW] -> [COUT, KH*KW, Cin], contiguous."""
    return weight.permute(0, 2, 3, 1).reshape(weight.shape[0], -1).contiguous()


def _dense_conv(x, wt, bias, act, res, dst, cout, kh, kw, ph, pw, cfg=None):
    """One fused dense conv + epilogue launch (stride 1, shape-preserving)."""
    n, c, imh, imw = x.shape
    p = imh * imw
    k = c * kh * kw
    if dst is None:
        dst = torch.empty((n, cout, imh, imw), dtype=x.dtype, device=x.device)
    padded = kh * kw > 1
    block_co, block_p, block_k, warps, stages = (
        _dense_cfg(p, cout, c, k, n, padded) if cfg is None else cfg)
    _dense_kernel[(-(-p // block_p), -(-cout // block_co), n)](
        x, wt, bias, res, dst,
        x.stride(0), x.stride(1),
        0 if res is None else res.stride(0), 0 if res is None else res.stride(1),
        dst.stride(0), dst.stride(1),
        C=c, IMH=imh, IMW=imw, COUT=cout, OW=imw, P=p, K=k, KW=kw, PH=ph, PW=pw,
        HAS_BIAS=bias is not None, ACT=act, HAS_RES=res is not None, PADDED=padded,
        BLOCK_CO=block_co, BLOCK_P=block_p, BLOCK_K=block_k,
        NUM_K=-(-k // block_k), EVEN_K=(k % block_k == 0),
        EVEN_CO=(cout % block_co == 0), EVEN_P=(p % block_p == 0),
        num_warps=warps, num_stages=stages,
    )
    return dst


# ---------------------------------------------------------------------------
# THE MEGAKERNEL: the whole block in one launch.
#
# Measured harness constants (tools/floor.py, tools/floor2.py; all in scored
# window units) are what make this the largest remaining lever:
#
#   floor of the window (event pair + Python + the pool's input copy)  7.17 us
#   per eager node                                                    2.05 us
#   per CUDA-graph node                          1.10 us + 6.02 us fixed
#   grid-wide barrier, 148 CTAs (1/SM), 4 warps                       1.54 us
#
# The parent's 6-9 node graph pays 6.0 + 1.1 x nodes plus two memcpys of pure
# scheduling; one launch pays 2.05 + 1.54 x (stages - 1) instead, and it also
# deletes the graph's static input (no copy-in) and static output (no clone-out).
#
# Grid.  Exactly ``_num_sms()`` CTAs, one per SM.  That is the only grid size
# where co-residency is structurally guaranteed -- and therefore the only one
# where a grid barrier cannot deadlock -- and it is also the cheapest, since
# barrier cost grows linearly in both grid size (296 CTAs: 3.73 us) and warps per
# CTA (8 warps: 2.73 us).  Each CTA grid-strides over its stage's tiles, so a
# stage with fewer tiles than the grid leaves CTAs idle; those CTAs still fall
# through to the barrier, which is what keeps it from hanging.
#
# Why the kernel is *generated* rather than written once with a constexpr stage
# table.  Every stage needs its own tile shape, kernel size, channel offsets and
# unroll counts as compile-time constants.  Triton 3.6 will not carry those from
# a ``tl.constexpr`` tuple into the places that require a constant: arithmetic on
# a constexpr created inside a jit body raises ``_semantic argument must be
# provided outside of JIT functions``, and a tuple field passed to a
# ``tl.constexpr``-annotated parameter of a jit helper arrives in a form
# ``tl.arange`` rejects (``arange's arguments must be of type tl.constexpr``).
# Emitting the source with the constants as *literals* is the form Triton handles
# unambiguously -- it is exactly what a hand-written kernel looks like -- and it
# also gives the compiler the best alignment and divisibility information.  The
# generated text is cached per plan, so this costs one compile per shape.
#
# Cross-stage visibility.  L1 is not coherent across SMs, so every load of a
# value another CTA produced in an earlier stage carries
# ``cache_modifier=".cg"`` (bypass L1, hit the coherent L2), and so does every
# store another CTA will read.  The intermediate buffer is 100-800 KB and stays
# L2-resident, so that costs latency, not bandwidth.
# ---------------------------------------------------------------------------
_MEGA_HEADER = '''import triton
import triton.language as tl


@triton.jit(do_not_specialize=["TARGET"])
def {name}(X, WGT, BIA, WS, OUT, LOCK, TARGET):
    pid = tl.program_id(0)
'''

# The arrival counter is never reset: ``TARGET`` is ``launch_index * GRID``, and
# exactly GRID CTAs arrive per launch with launches serialized on the stream, so
# one counter is correct for every launch and there is no sense reversal or
# last-arriver branch to get wrong.  ``tl.zeros((1,), ...)`` keeps the arrival to
# a single lane; ``tl.sum`` makes the spin's load block-uniform so the ``while``
# does not diverge.  The two ``debug_barrier``s are ``__syncthreads``: no thread
# announces arrival before its CTA finished the stage, none starts the next stage
# before release.
_MEGA_BARRIER = '''
    # ---- grid barrier {s} ----
    tl.debug_barrier()
    z{s} = tl.zeros((1,), dtype=tl.int32)
    tl.atomic_add(LOCK + {off} + z{s}, z{s} + 1, sem="release")
    a{s} = z{s}
    while tl.sum(a{s}) < TARGET:
        a{s} = tl.load(LOCK + {off} + z{s}, volatile=True)
    tl.debug_barrier()
'''


def _mega_src(stages, grid, imh, imw, p, strides, lock_stride, name):
    """Emit the Triton source for one block's megakernel."""
    out = [_MEGA_HEADER.format(name=name)]
    for s, st in enumerate(stages):
        out.append(f"\n    # ==== stage {s}: {st['desc']} ====\n")
        out.append(_mega_stage_src(s, st, grid, imh, imw, p, strides))
        if s + 1 < len(stages):
            out.append(_MEGA_BARRIER.format(s=s, off=s * lock_stride))
    return "".join(out)


def _mega_stage_src(s, st, grid, imh, imw, p, strides):
    """Emit one stage: a grid-stride loop over its tiles.

    ``dense`` is the implicit GEMM of ``_dense_kernel`` (M = P, N = COUT,
    K = CIN*KH*KW, accumulator laid out [BCO, BP] so the store walks the
    contiguous pixel axis, pad-1 halo masked rather than padded); ``dw`` is the
    flat-tile depthwise of ``_dw_kernel``, with the tap address written as the
    tile base plus a constant because ``(oh-PH+i)*IMW + (ow-PW+j)`` equals
    ``offs + (i-PH)*IMW + (j-PW)``.
    """
    src_sn, src_sc = strides[st["sb"]]
    dst_sn, dst_sc = strides[st["db"]]
    res_sn, res_sc = strides[1]
    SRC = f"({st['sbuf']} + {st['soff']})"
    DST = f"({st['dbuf']} + {st['doff']})"
    RES = f"(WS + {st['roff']})"
    scg = ', cache_modifier=".cg"' if st["sb"] == 1 else ""
    dcg = ', cache_modifier=".cg"' if st["db"] == 1 else ""
    L = []
    a = L.append
    if st["kind"] == _MEGA_KIND_DENSE:
        bco, bp, bk = st["bco"], st["bp"], st["bk"]
        cin, cout, kh, kw = st["cin"], st["cout"], st["kh"], st["kw"]
        k, nkb, np_, nco, npco, nt = (st["k"], st["nkb"], st["np"], st["nco"],
                                      st["npco"], st["nt"])
        padded = kh * kw > 1
        even_k, even_co, even_p = k % bk == 0, cout % bco == 0, p % bp == 0
        a(f"    for t{s} in range(pid, {nt}, {grid}):\n")
        i = "        "
        if npco == nt:
            a(f"{i}n{s} = 0\n{i}r{s} = t{s}\n")
        else:
            a(f"{i}n{s} = t{s} // {npco}\n{i}r{s} = t{s} - n{s} * {npco}\n")
        if nco == 1:
            a(f"{i}co{s} = 0\n{i}pp{s} = r{s}\n")
        else:
            a(f"{i}co{s} = r{s} // {np_}\n{i}pp{s} = r{s} - co{s} * {np_}\n")
        a(f"{i}oc{s} = co{s} * {bco} + tl.arange(0, {bco})\n")
        a(f"{i}op{s} = pp{s} * {bp} + tl.arange(0, {bp})\n")
        a(f"{i}mc{s} = oc{s} < {cout}\n")
        a(f"{i}mp{s} = op{s} < {p}\n")
        a(f"{i}xn{s} = {SRC} + n{s} * {src_sn}\n")
        a(f"{i}acc{s} = tl.zeros(({bco}, {bp}), dtype=tl.float32)\n")
        if padded:
            a(f"{i}oh{s} = op{s} // {imw}\n")
            a(f"{i}ih{s} = oh{s} - {kh // 2}\n")
            a(f"{i}iw{s} = (op{s} - oh{s} * {imw}) - {kw // 2}\n")
        loop = nkb > 1
        if loop:
            a(f"{i}for kb{s} in tl.static_range({nkb}):\n")
            j = i + "    "
            a(f"{j}ok{s} = kb{s} * {bk} + tl.arange(0, {bk})\n")
        else:
            j = i
            a(f"{j}ok{s} = tl.arange(0, {bk})\n")
        if cin == bk and not padded:
            a(f"{j}cc{s} = ok{s}\n")
        else:
            a(f"{j}tp{s} = ok{s} // {cin}\n{j}cc{s} = ok{s} - tp{s} * {cin}\n")
        if padded:
            a(f"{j}rh{s} = ih{s}[None, :] + (tp{s} // {kw})[:, None]\n")
            a(f"{j}rw{s} = iw{s}[None, :] + (tp{s} % {kw})[:, None]\n")
            m = (f"(rh{s} >= 0) & (rh{s} < {imh}) & (rw{s} >= 0) & (rw{s} < {imw})")
            if not even_k:
                m += f" & (ok{s} < {k})[:, None]"
            if not even_p:
                m += f" & mp{s}[None, :]"
            a(f"{j}am{s} = {m}\n")
            a(f"{j}ap{s} = xn{s} + cc{s}[:, None] * {src_sc} + rh{s} * {imw} + rw{s}\n")
            a(f"{j}av{s} = tl.load(ap{s}, mask=am{s}, other=0.0{scg})\n")
        else:
            a(f"{j}ap{s} = xn{s} + cc{s}[:, None] * {src_sc} + op{s}[None, :]\n")
            if even_k and even_p:
                a(f"{j}av{s} = tl.load(ap{s}{scg})\n")
            else:
                m = []
                if not even_k:
                    m.append(f"(ok{s} < {k})[:, None]")
                if not even_p:
                    m.append(f"mp{s}[None, :]")
                a(f"{j}am{s} = {' & '.join(m)}\n")
                a(f"{j}av{s} = tl.load(ap{s}, mask=am{s}, other=0.0{scg})\n")
        a(f"{j}wp{s} = WGT + {st['woff']} + oc{s}[:, None] * {k} + ok{s}[None, :]\n")
        if even_co and even_k:
            a(f"{j}wv{s} = tl.load(wp{s})\n")
        else:
            wm = []
            if not even_co:
                wm.append(f"mc{s}[:, None]")
            if not even_k:
                wm.append(f"(ok{s} < {k})[None, :]")
            a(f"{j}wv{s} = tl.load(wp{s}, mask={' & '.join(wm)}, other=0.0)\n")
        a(f"{j}acc{s} = tl.dot(wv{s}, av{s}, acc=acc{s})\n")
        if st["boff"] >= 0:
            bm = "" if even_co else f", mask=mc{s}, other=0.0"
            a(f"{i}acc{s} += tl.load(BIA + {st['boff']} + oc{s}{bm})"
              f"[:, None].to(tl.float32)\n")
        if st["act"]:
            a(f"{i}acc{s} *= tl.sigmoid(acc{s})\n")
        a(f"{i}om{s} = mc{s}[:, None] & mp{s}[None, :]\n")
        if st["has_res"]:
            a(f"{i}rp{s} = {RES} + n{s} * {res_sn} + oc{s}[:, None] * {res_sc}"
              f" + op{s}[None, :]\n")
            a(f'{i}acc{s} += tl.load(rp{s}, mask=om{s}, other=0.0,'
              f' cache_modifier=".cg").to(tl.float32)\n')
        a(f"{i}yp{s} = {DST} + n{s} * {dst_sn} + oc{s}[:, None] * {dst_sc}"
          f" + op{s}[None, :]\n")
        a(f"{i}tl.store(yp{s}, acc{s}.to({st['dbuf']}.dtype.element_ty),"
          f" mask=om{s}{dcg})\n")
    else:
        bp, cin, kh, kw = st["bp"], st["cin"], st["kh"], st["kw"]
        npb, nt = st["np"], st["nt"]
        even_p = p % bp == 0
        ph, pw = kh // 2, kw // 2
        a(f"    for t{s} in range(pid, {nt}, {grid}):\n")
        i = "        "
        if npb == 1:
            a(f"{i}nc{s} = t{s}\n{i}offs{s} = tl.arange(0, {bp})\n")
        else:
            a(f"{i}nc{s} = t{s} // {npb}\n")
            a(f"{i}offs{s} = (t{s} - nc{s} * {npb}) * {bp} + tl.arange(0, {bp})\n")
        if nt == cin * npb:
            a(f"{i}n{s} = 0\n{i}cc{s} = nc{s}\n")
        else:
            a(f"{i}n{s} = nc{s} // {cin}\n{i}cc{s} = nc{s} - n{s} * {cin}\n")
        a(f"{i}mp{s} = offs{s} < {p}\n")
        a(f"{i}oh{s} = offs{s} // {imw}\n")
        a(f"{i}ow{s} = offs{s} - oh{s} * {imw}\n")
        a(f"{i}xb{s} = {SRC} + n{s} * {src_sn} + cc{s} * {src_sc} + offs{s}\n")
        a(f"{i}wb{s} = WGT + {st['woff']} + cc{s} * {kh * kw}\n")
        a(f"{i}acc{s} = tl.zeros(({bp},), dtype=tl.float32)\n")
        for ii in range(kh):
            mh = f"(oh{s} - {ph - ii} >= 0) & (oh{s} - {ph - ii} < {imh})"
            if not even_p:
                mh += f" & mp{s}"
            a(f"{i}h{s}_{ii} = {mh}\n")
            for jj in range(kw):
                off = (ii - ph) * imw + (jj - pw)
                a(f"{i}v{s}_{ii}_{jj} = tl.load(xb{s} + ({off}), mask=h{s}_{ii}"
                  f" & (ow{s} - {pw - jj} >= 0) & (ow{s} - {pw - jj} < {imw}),"
                  f" other=0.0{scg})\n")
                a(f"{i}acc{s} += v{s}_{ii}_{jj}.to(tl.float32) *"
                  f" tl.load(wb{s} + {ii * kw + jj}).to(tl.float32)\n")
        if st["boff"] >= 0:
            a(f"{i}acc{s} += tl.load(BIA + {st['boff']} + cc{s}).to(tl.float32)\n")
        if st["act"]:
            a(f"{i}acc{s} *= tl.sigmoid(acc{s})\n")
        if st["has_res"]:
            a(f"{i}acc{s} += tl.load({RES} + n{s} * {res_sn} + cc{s} * {res_sc}"
              f" + offs{s}, mask=mp{s}, other=0.0,"
              f' cache_modifier=".cg").to(tl.float32)\n')
        a(f"{i}yp{s} = {DST} + n{s} * {dst_sn} + cc{s} * {dst_sc} + offs{s}\n")
        a(f"{i}tl.store(yp{s}, acc{s}.to({st['dbuf']}.dtype.element_ty),"
          f" mask=mp{s}{dcg})\n")
    return "".join(L)


_MEGA_CACHE: dict = {}


def _mega_compile(stages, grid, imh, imw, p, strides, lock_stride):
    """Generate, exec and JIT one block's megakernel; cached by source text.

    ``triton.jit`` reads a function's source with ``inspect.getsourcelines``, so
    the generated module is registered in ``linecache`` under a ``<...>``
    filename -- which ``inspect.findsource`` accepts precisely because it is not
    a real path -- rather than written to disk.
    """
    import hashlib
    import linecache
    body = _mega_src(stages, grid, imh, imw, p, strides, lock_stride, "_mega")
    key = hashlib.sha1(body.encode()).hexdigest()[:16]
    fn = _MEGA_CACHE.get(key)
    if fn is not None:
        return fn
    name = f"_mega_{key}"
    src = body.replace("def _mega(", f"def {name}(", 1)
    fname = f"<ako-mega-{key}>"
    linecache.cache[fname] = (len(src), None, src.splitlines(True), fname)
    ns: dict = {}
    exec(compile(src, fname, "exec"), ns)  # noqa: S102 - our own generated text
    fn = ns[name]
    _MEGA_CACHE[key] = fn
    return fn


# ---------------------------------------------------------------------------
# Megakernel plan: turn the flat list of ``_Conv`` steps into one stage table.
#
# Buffer layout.  A single workspace tensor ``WS`` of shape
# (N, (2+n)*c + 2*inner, H, W) holds everything the block needs, so every stage
# addresses its source, residual and destination as a channel offset into one
# tensor with one set of strides:
#
#   channels [0, 2c)                  cv1's output       (the first two chunks)
#   channels [(2+k)*c, (3+k)*c)       block k's output   (the concat buffer)
#   channels [(2+n)*c, ...)           two ping-pong scratch slabs for the
#                                     intermediates inside a block
#
# so ``cv2`` reads channels [0, (2+n)*c) as one contiguous tensor -- the same
# zero-copy concat the parent's graph path uses -- and the block's internal
# stages never touch it.  Two scratch slabs are enough because a block's stages
# form a chain: the CIB's five convs go WS[c:2c] -> A -> B -> A -> B -> WS slice,
# and the residual is always a *different* channel range from the destination,
# so nothing aliases.
# ---------------------------------------------------------------------------
_MEGA_KIND_DENSE = 0
_MEGA_KIND_DW = 1


def _mega_tile(p, cout, cin, k, n, padded, sms):
    """Tile shape for one megakernel stage.

    Unlike the per-conv path the grid is fixed at one CTA per SM and each CTA
    grid-strides over the stage's tiles, so ``_fill_machine``'s "manufacture more
    CTAs" logic does not apply; the swept per-conv tile is the starting point.

    ``_MEGA_TILE == "cap"`` additionally caps the k block at 128 and halves the
    pixel tile until ``BCO * BP <= 2048``, which keeps the fp32 accumulator and
    the A tile inside a 4-warp CTA's registers.  ``"raw"`` keeps the swept tile,
    which only fits at 8 warps.  Which pair wins is per shape and measured --
    see ``_mega_cfg``.
    """
    block_co, block_p, block_k, _, _ = _dense_cfg_base(p, cout, cin, k, n, padded)
    if _MEGA_TILE == "cap":
        block_k = min(block_k, 128)
        while block_co * block_p > 2048 and block_p > 32:
            block_p //= 2
    return block_co, block_p, block_k


def _mega_plan(cv1, blocks, cv2, c, cin, cout, n, imh, imw, dtype, device):
    """``(stages, weights, biases, ws_channels)`` or ``None`` if unsupported.

    Only fully-fused plans qualify: every step must have bound to either the
    dense implicit-GEMM path or the depthwise path, because those are the two
    stage bodies the megakernel implements.  Anything else (a strided conv, an
    fp32 tree, a conv that changes H/W) keeps the parent's per-conv graph path.
    """
    steps_flat = [cv1]
    for st, _ in blocks:
        steps_flat.extend(st)
    steps_flat.append(cv2)
    for st in steps_flat:
        if st.dense is None and st.dw is None:
            return None
        if st.w.dtype != dtype:
            return None

    nb_ch = (2 + len(blocks)) * c
    inner = 0
    for st, _ in blocks:
        for mid in st[:-1]:
            inner = max(inner, int(mid.w.shape[0]))
    ws_ch = nb_ch + 2 * inner
    scratch = (nb_ch, nb_ch + inner)

    wparts, bparts, stages = [], [], []
    cur = [0, 0]                   # running element counts in WGT / BIA
    work = [0]                     # tile work, for the mega-vs-graph choice
    p = imh * imw
    sms = _num_sms()

    def take(parts, which, t):
        """Append ``t`` to a packed buffer at a 16-byte-aligned offset, so the
        constexpr offset the kernel adds keeps Triton's widened loads aligned."""
        off = (cur[which] + 7) & ~7
        if off > cur[which]:
            parts.append(torch.zeros(off - cur[which], dtype=t.dtype, device=t.device))
        parts.append(t)
        cur[which] = off + t.numel()
        return off

    def emit(step, sb, soff, cin_, db, doff, roff):
        """Append one stage.  Every constant the generated kernel needs is
        computed here -- tile shape, tile counts, k blocks, element offsets --
        so the emitted source contains only literals."""
        boff = -1 if step.b is None else take(bparts, 1, step.b.reshape(-1))
        if step.dw is not None:
            woff = take(wparts, 0, step.dw[0].reshape(-1))
            kh, kw = step.dw[1], step.dw[2]
            co = cin_
            bp = min(512, triton.next_power_of_2(p))
            bco = bk = k = nkb = 0
            npb = -(-p // bp)
            nco = 1
            nt = n * cin_ * npb
            kind = _MEGA_KIND_DW
            npco = npb
            work[0] += nt * bp * kh * kw
        else:
            woff = take(wparts, 0, step.dense[0].reshape(-1))
            kh, kw = step.dense[2], step.dense[3]
            co = step.dense[1]
            k = cin_ * kh * kw
            bco, bp, bk = _mega_tile(p, co, cin_, k, n, kh * kw > 1, sms)
            nkb = -(-k // bk)
            npb = -(-p // bp)
            nco = -(-co // bco)
            npco = npb * nco
            nt = npco * n
            kind = _MEGA_KIND_DENSE
            work[0] += nt * bco * bp * k
        stages.append(dict(
            kind=kind, sb=sb, sbuf=("X" if sb == 0 else "WS"), soff=soff * p,
            db=db, dbuf=("WS" if db == 1 else "OUT"), doff=doff * p,
            roff=max(roff, 0) * p, has_res=roff >= 0,
            cin=cin_, cout=co, kh=kh, kw=kw, woff=woff, boff=boff,
            act=bool(step.act), bco=bco, bp=bp, bk=bk, k=k, nkb=nkb,
            np=npb, nco=nco, npco=npco, nt=nt,
            desc=f"{'dw' if kind else 'dense'} {kh}x{kw} {cin_}->{co} "
                 f"{'X' if sb == 0 else 'WS'}[{soff}] -> "
                 f"{'WS' if db == 1 else 'OUT'}[{doff}]"
                 f"{' +res WS[%d]' % roff if roff >= 0 else ''}"))
        return co

    # cv1: the block input -> channels [0, 2c)
    if emit(cv1, 0, 0, cin, 1, 0, -1) != 2 * c:
        return None
    prev = c                       # the second chunk is the first block's input
    for bi, (st, add) in enumerate(blocks):
        src, soff = 1, prev
        cin_ = c
        for i, mid in enumerate(st[:-1]):
            dst = scratch[i % 2]
            cin_ = emit(mid, src, soff, cin_, 1, dst, -1)
            src, soff = 1, dst
        dst = (2 + bi) * c
        if emit(st[-1], src, soff, cin_, 1, dst, prev if add else -1) != c:
            return None
        prev = dst
    # cv2: the whole concat buffer -> the output
    if emit(cv2, 1, 0, nb_ch, 2, 0, -1) != cout:
        return None

    weights = torch.cat([t.to(dtype) for t in wparts]) if wparts else None
    biases = torch.cat([t.to(dtype) for t in bparts]) if bparts else \
        torch.zeros(1, dtype=dtype, device=device)
    return tuple(stages), weights, biases, ws_ch, work[0]


def _dw_cfg(p):
    """(BLOCK, num_warps) for the depthwise kernel.

    Swept on the four CIB depthwise shapes: the landscape is flat to within
    0.2-0.3 us (the kernel is bound by issuing k*k masked gathers, not by the
    tile), and a 512-element tile at 8 warps is at or within noise of the best
    on all four.  One tile covers a whole 20x20 plane, so there is exactly one
    program per (n, c) and the taps all hit L1.
    """
    block = min(512, triton.next_power_of_2(p))
    return block, max(1, min(8, block // 64))


def _epilogue(src, bias, act, res, dst):
    """``dst = silu(src + bias) + res`` in one launch (dst allocated if None)."""
    if dst is None:
        dst = torch.empty_like(src)
    if triton is None or not _USE_TRITON_EPI or not src.is_cuda:
        return _epilogue_torch(src, bias, act, res, dst)
    n, c = src.shape[0], src.shape[1]
    p = src.shape[2] * src.shape[3]
    block = 1024 if p >= 1024 else triton.next_power_of_2(p)
    _epi_kernel[(n * c, (p + block - 1) // block)](
        src, bias, res, dst,
        p,
        src.stride(0), src.stride(1),
        0 if res is None else res.stride(0), 0 if res is None else res.stride(1),
        dst.stride(0), dst.stride(1),
        C=c, HAS_BIAS=bias is not None, ACT=act, HAS_RES=res is not None,
        BLOCK=block, EVEN_P=(p % block == 0), num_warps=4,
    )
    return dst


def _epilogue_torch(src, bias, act, res, dst):
    v = src if bias is None else src.add_(bias.reshape(1, -1, 1, 1))
    if act:
        v = F.silu(v, inplace=True)
    if res is not None:
        v = v.add_(res)
    return v if dst is v else dst.copy_(v)


# ---------------------------------------------------------------------------
# BatchNorm folding.
#
#   bn(conv(x)) = (conv(x) - mean) * g / sqrt(var + eps) + b
#               = conv(x, w * s) + (bias - mean) * s + b,   s = g / sqrt(var+eps)
#
# fp32 throughout, cast to the conv dtype at the end.
# ---------------------------------------------------------------------------
def _fold_bn(weight, bias, bn):
    dtype = weight.dtype
    scale = bn.weight.float() / torch.sqrt(bn.running_var.float() + bn.eps)
    b0 = torch.zeros_like(scale) if bias is None else bias.float()
    fw = (weight.float() * scale.reshape(-1, 1, 1, 1)).to(dtype)
    fb = ((b0 - bn.running_mean.float()) * scale + bn.bias.float()).to(dtype)
    return fw, fb


class _Conv:
    """One folded conv plus its fused epilogue, ready to issue."""

    __slots__ = ("w", "b", "stride", "padding", "groups", "act", "l1", "conv_bias",
                 "dw", "dense")

    def __init__(self, w, b, stride, padding, groups, act):
        self.w = w
        self.b = b
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.act = act
        self.l1 = None          # L1 Conv2d instance (NCHW fused 3x3) or None
        self.conv_bias = False  # True when the conv itself folds the bias in
        self.dw = None          # (weight[C,K*K], kh, kw, ph, pw) for the fused depthwise
        self.dense = None       # (wt, cout, kh, kw, ph, pw) for the fused dense conv

    def bind(self):
        """Pick the cheapest issue path for this conv, most fused first.

        Both Triton paths also require the folded weight to be on CUDA: the
        input always matches the module's device, and a CPU module that bound
        them would reach ``_dense_conv`` with a host pointer and raise
        ``Pointer argument (at 0) cannot be accessed from Triton``.

        1. **Depthwise** (``groups == cout``, stride 1, odd kernel,
           ``padding == k//2``) -> ``_dw_kernel``: conv and epilogue in one
           launch.  Measured against cudnn + silu + add on the CIB's shapes:
           7x7 [4,256,20,20] 20.1 -> 7.6 us, 3x3 [4,128,20,20] 9.7 -> 2.3 us.
        2. **Dense, shape-preserving, fp16/bf16** (every 1x1 and every dense
           3x3 here) -> ``_dense_kernel``: same one-launch deal.  Measured
           against cudnn + a separate epilogue, and against
           ``candidate/L1/conv2d.py``'s fused NCHW 3x3 + a separate epilogue,
           on all 18 dense convs the scored cases issue -- see ``_dense_cfg``.
           It wins on 17 of 18 and ties on the one where cudnn is already at
           its bandwidth floor.
        3. **L1's fused padded 3x3** -- kept as the path for anything the
           fused-dense kernel declines (``FK_C2F_DENSE=off``, or a dtype it
           does not take).  Exactly the pattern that file gates on and was
           tuned for; it folds the bias in for free and never round-trips the
           layout, and it falls back to ``F.conv2d`` internally outside its own
           FLOP bound, so binding it is safe even where it declines.
        4. Otherwise ``F.conv2d`` (bias unfused, so the epilogue applies it)
           -- which is where a strided, grouped or fp32 conv lands.
        """
        kh, kw = int(self.w.shape[2]), int(self.w.shape[3])
        if (triton is not None and _USE_DW and self.w.is_cuda
                and self.stride == _ONE and self.groups == int(self.w.shape[0])
                and int(self.w.shape[1]) == 1
                and kh % 2 == 1 and kw % 2 == 1
                and tuple(self.padding) == (kh // 2, kw // 2)
                and self.w.dtype in (torch.float16, torch.bfloat16, torch.float32)):
            self.dw = (self.w.reshape(self.groups, kh * kw).contiguous(),
                       kh, kw, kh // 2, kw // 2)
            return self
        if (self.groups == 1 and self.stride == _ONE and triton is not None
                and self.w.is_cuda
                and tuple(self.padding) == (kh // 2, kw // 2)
                and kh % 2 == 1 and kw % 2 == 1
                and self.w.dtype in (torch.float16, torch.bfloat16)
                and (_DENSE_MODE == "all" or (_DENSE_MODE == "1x1" and kh * kw == 1))):
            self.dense = (_dense_weight(self.w), int(self.w.shape[0]),
                          kh, kw, kh // 2, kw // 2)
            return self
        if (_L1Conv2d is None or not _USE_L1
                or self.groups != 1 or self.stride != _ONE
                or tuple(self.w.shape[2:]) != (3, 3) or tuple(self.padding) != _ONE
                or self.w.dtype not in (torch.float16, torch.bfloat16)):
            return self
        cout, cin = int(self.w.shape[0]), int(self.w.shape[1])
        conv = _L1Conv2d(cin, cout, 3, 1, 1, groups=1, bias=self.b is not None)
        conv.weight = nn.Parameter(self.w, requires_grad=False)
        if self.b is not None:
            conv.bias = nn.Parameter(self.b, requires_grad=False)
        self.l1 = conv
        self.conv_bias = self.b is not None
        return self

    def conv(self, x):
        if self.l1 is not None:
            return self.l1(x)
        return F.conv2d(x, self.w, None, self.stride, self.padding, _ONE, self.groups)

    def apply(self, x, res=None, dst=None):
        dw = self.dw
        if dw is not None:
            return _dw_conv(x, dw[0], self.b, self.act, res, dst, dw[1], dw[2], dw[3], dw[4])
        dn = self.dense
        if dn is not None:
            return _dense_conv(x, dn[0], self.b, self.act, res, dst,
                               dn[1], dn[2], dn[3], dn[4], dn[5])
        raw = self.conv(x)
        bias = None if self.conv_bias else self.b
        if bias is None and not self.act and res is None:
            return raw if dst is None else dst.copy_(raw)
        return _epilogue(raw, bias, self.act, res, dst)


def _conv_step(m, src):
    """Fold one ``YOLOConv`` (conv [+ bn] + act) into a single ``_Conv``."""
    conv = m.conv
    src.append(conv.weight)
    cbias = conv.bias
    if cbias is not None:
        src.append(cbias)
    bn = getattr(m, "bn", None)
    if bn is None or getattr(m, "_is_fused", False):
        w, b = conv.weight.detach(), None if cbias is None else cbias.detach()
    else:
        src += [bn.weight, bn.bias, bn.running_mean, bn.running_var]
        w, b = _fold_bn(conv.weight.detach(), None if cbias is None else cbias.detach(), bn)
    return _Conv(w, b, tuple(conv.stride), tuple(conv.padding), conv.groups,
                 not isinstance(m.act, nn.Identity))


def _repvgg_step(m, src):
    """Fold ``silu(conv7x7(x) + conv3x3(x))`` into one 7x7 depthwise conv."""
    big = _conv_step(m.conv, src)
    if not getattr(m, "_is_fused", False) and getattr(m, "conv1", None) is not None:
        small = _conv_step(m.conv1, src)
        pad = (big.w.shape[-1] - small.w.shape[-1]) // 2
        big.w = big.w + F.pad(small.w, [pad, pad, pad, pad])
        if small.b is not None:
            big.b = small.b if big.b is None else big.b + small.b
    big.act = True  # YOLORepVGGDW applies its own SiLU to the sum
    return big


def _block_steps(m, src):
    """``(steps, residual)`` for one ``self.m`` entry, or ``None`` if unknown."""
    if isinstance(m, YOLOBottleneck):
        seq = (m.cv1, m.cv2)
    elif isinstance(m, YOLOCIB):
        seq = tuple(m.cv1)
    else:
        return None
    steps = []
    for s in seq:
        if isinstance(s, YOLOConv):
            steps.append(_conv_step(s, src))
        elif isinstance(s, YOLORepVGGDW):
            steps.append(_repvgg_step(s, src))
        else:
            return None
    return steps, bool(m.add)


def _shape_preserving(step):
    """True when this conv keeps H and W (required to share the concat buffer)."""
    kh, kw = int(step.w.shape[2]), int(step.w.shape[3])
    return (step.stride == _ONE
            and tuple(step.padding) == (kh // 2, kw // 2)
            and kh % 2 == 1 and kw % 2 == 1)


class YOLOC2f(nn.Module):
    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = YOLOConv(c1, 2 * self.c, 1, 1)
        self.cv2 = YOLOConv((2 + n) * self.c, c2, 1, 1)
        self.m = nn.ModuleList(
            YOLOBottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n)
        )
        self._plan = None          # (cv1, blocks, cv2, shared_buffer)
        self._src: list = []       # tensors the fold was derived from
        self._sig = None           # (data_ptr, _version) guard over _src
        self._graphs: dict = {}    # (shape, dtype) -> (static_in, static_out, graph)
        self._mega: dict = {}      # (shape, dtype) -> megakernel entry or None
        self.register_load_state_dict_post_hook(_invalidate_hook)

    # -- plan construction / cache guard -----------------------------------
    def _signature(self):
        return tuple((t.data_ptr(), t._version) for t in self._src)

    def _build_plan(self):
        # The graphs bake in the folded weights' addresses, and a rebuild
        # replaces those tensors, so the captured graphs go with them.
        self._graphs.clear()
        self._mega.clear()
        src: list = []
        cv1 = _conv_step(self.cv1, src).bind()
        blocks = []
        shared = _shape_preserving(cv1)
        for m in self.m:
            step = _block_steps(m, src)
            if step is None:
                # Unrecognized tree: cache the verdict (guarded like any other
                # plan) so the reference path does not re-walk and re-fold the
                # whole module on every forward.
                self._plan = False
                self._src = src
                self._sig = self._signature()
                return False
            steps, add = step
            for s in steps:
                s.bind()
            # The shared buffer needs every block to preserve H/W and to end
            # with exactly self.c channels; otherwise fall back to torch.cat.
            if not all(_shape_preserving(s) for s in steps) or int(steps[-1].w.shape[0]) != self.c:
                shared = False
            blocks.append((tuple(steps), add))
        cv2 = _conv_step(self.cv2, src).bind()
        if int(cv1.w.shape[0]) != 2 * self.c:
            shared = False
        self._plan = (cv1, tuple(blocks), cv2, shared)
        self._src = src
        self._sig = self._signature()
        return self._plan

    def _get_plan(self):
        plan = self._plan
        if plan is not None and self._sig == self._signature():
            return plan  # a tuple, or False for a tree we do not recognize
        return self._build_plan()

    # -- forward paths ------------------------------------------------------
    def _reference(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def _fused(self, x, plan):
        cv1, blocks, cv2, shared = plan
        if not blocks:
            return cv2.apply(cv1.apply(x))
        c = self.c
        if shared:
            # One (2+n)*c-channel buffer: cv1 writes channels [0, 2c) and each
            # block writes its own c-wide slice, so cv2 reads a contiguous
            # tensor with no concat and no copy.  Every conv here is
            # shape-preserving (checked when the plan was built), and a block's
            # residual is always a *different* slice from its destination, so
            # there is no aliasing.
            buf = torch.empty((x.shape[0], (2 + len(blocks)) * c, x.shape[2], x.shape[3]),
                              dtype=torch.result_type(x, cv1.w), device=x.device)
            cv1.apply(x, dst=buf[:, :2 * c])
            prev = buf[:, c:2 * c]
            base = 2 * c
            for steps, add in blocks:
                o = prev
                for step in steps[:-1]:
                    o = step.apply(o)
                dst = buf[:, base:base + c]
                steps[-1].apply(o, res=prev if add else None, dst=dst)
                prev = dst
                base += c
            return cv2.apply(buf)
        y = list(cv1.apply(x).chunk(2, 1))
        prev = y[1]
        for steps, add in blocks:
            o = prev
            for step in steps[:-1]:
                o = step.apply(o)
            o = steps[-1].apply(o, res=prev if add else None)
            y.append(o)
            prev = o
        return cv2.apply(torch.cat(y, 1))

    def _capture(self, x, key, plan):
        """Capture ``_fused`` for this input shape; ``None`` if not capturable."""
        try:
            static_in = torch.empty(x.shape, dtype=x.dtype, device=x.device)
            static_in.copy_(x)
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):  # JIT the Triton kernels, size the workspaces
                    self._fused(static_in, plan)
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                static_out = self._fused(static_in, plan)
            entry = (static_in, static_out, graph)
        except Exception:  # noqa: BLE001 - uncapturable: stay on the eager path
            entry = None
        self._graphs[key] = entry
        return entry

    # -- megakernel: one launch for the whole block -------------------------
    def _mega_build(self, x, plan):
        """Build (or decline) the megakernel entry for this input shape."""
        cv1, blocks, cv2, shared = plan
        if (triton is None or _MEGA_MODE == "0" or not shared or not x.is_cuda
                or not x.is_contiguous() or x.dim() != 4
                or x.dtype not in (torch.float16, torch.bfloat16)):
            return None
        n, cin, imh, imw = (int(v) for v in x.shape)
        cout = int(cv2.w.shape[0])
        built = _mega_plan(cv1, blocks, cv2, self.c, cin, cout, n, imh, imw,
                           x.dtype, x.device)
        if built is None:
            return None
        stages, wgt, bia, ws_ch, work = built
        if _MEGA_MODE == "auto" and not _mega_wins(work):
            return None
        grid = _num_sms()
        p = imh * imw
        strides = {0: (cin * p, p), 1: (ws_ch * p, p), 2: (cout * p, p)}
        try:
            ws = torch.empty((n, ws_ch, imh, imw), dtype=x.dtype, device=x.device)
            lock = torch.zeros(len(stages) * _LOCK_STRIDE, dtype=torch.int32,
                               device=x.device)
            fn = _mega_compile(stages, grid, imh, imw, p, strides, _LOCK_STRIDE)
        except Exception:  # noqa: BLE001 - OOM, or a stage the generator declines
            return None
        return {"stages": stages, "wgt": wgt, "bia": bia, "ws": ws, "lock": lock,
                "grid": grid, "cout": cout, "ws_ch": ws_ch, "epoch": 0, "n": n,
                "imh": imh, "imw": imw, "cin": cin, "fn": fn}

    def _mega_run(self, x, e):
        imh, imw, n = e["imh"], e["imw"], e["n"]
        p = imh * imw
        grid = e["grid"]
        out = torch.empty((n, e["cout"], imh, imw), dtype=x.dtype, device=x.device)
        # The arrival counters are never reset; the target is launch_index * grid,
        # so wrap well before int32 overflow (once per ~7M forwards).
        epoch = e["epoch"] + 1
        if epoch * grid > (1 << 30):
            e["lock"].zero_()
            epoch = 1
        e["epoch"] = epoch
        e["fn"][(grid,)](x, e["wgt"], e["bia"], e["ws"], out, e["lock"],
                        epoch * grid, num_warps=_MEGA_WARPS, num_stages=1)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training or torch.is_grad_enabled():
            return self._reference(x)
        plan = self._get_plan()
        if plan is False:
            return self._reference(x)
        if _MEGA_MODE != "0" and x.is_cuda:
            key = (x.shape, x.dtype)
            e = self._mega.get(key, _MISSING)
            if e is _MISSING:
                e = None if len(self._mega) >= _MAX_GRAPHS else self._mega_build(x, plan)
                self._mega[key] = e
            if e is not None:
                return self._mega_run(x, e)
        if _USE_GRAPH and x.is_cuda:
            key = (x.shape, x.dtype)
            entry = self._graphs.get(key, _MISSING)
            if entry is _MISSING:
                # Bound what a caller sweeping shapes can pin, but only for
                # *new* shapes -- already-captured ones keep replaying.
                if len(self._graphs) >= _MAX_GRAPHS:
                    return self._fused(x, plan)
                entry = self._capture(x, key, plan)
            if entry is not None:
                static_in, static_out, graph = entry
                static_in.copy_(x)
                graph.replay()
                return static_out.clone()
        return self._fused(x, plan)


def _invalidate_hook(module, incompatible_keys):  # noqa: ARG001
    module._plan = None
    module._src = []
    module._sig = None
    module._graphs.clear()
    module._mega.clear()


class YOLOC2fCIB(YOLOC2f):
    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, lk: bool = False, g: int = 1, e: float = 0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(YOLOCIB(self.c, self.c, shortcut, e=1.0, lk=lk) for _ in range(n))


_MEGA_WORK_MAX = float(os.environ.get("FK_C2F_MEGA_WORK", "400e6"))


def _mega_wins(work):
    """Per-shape choice between the megakernel and the parent's graph path.

    The megakernel's grid is fixed at one CTA per SM and does not grow with the
    problem, while a per-conv launch sizes its grid to the work.  So the
    megakernel wins while per-node scheduling dominates and loses once there is
    enough work that the wider grid pays for itself.  ``work`` is the tile work
    the megakernel would issue -- ``sum(tiles * BCO * BP * K)`` over the dense
    stages plus ``sum(tiles * BP * KH * KW)`` over the depthwise ones, i.e. what
    the kernel actually executes including masked lanes.

    Measured on all seven scored cases (window us, median of 50, both paths in
    the same process; ``tools/mega.py``):

    | case | shape              | stages | work  | graph | mega  | ratio |
    |------|--------------------|-------:|------:|------:|------:|------:|
    | 4    | C2f  [1,192,40,40] |      4 | 197 M | 37.89 | 27.65 | 1.370 |
    | 3    | C2f  [1,128,40,40] |      6 | 315 M | 50.21 | 40.96 | 1.226 |
    | 6    | CIB  [1,384,20,20] |      7 | 125 M | 46.08 | 37.89 | 1.216 |
    | 0    | C2f  [1,256,20,20] |      4 | 196 M | 44.06 | 37.73 | 1.168 |
    | 2    | C2f  [1,32,160,160]|      4 | 183 M | 33.79 | 29.66 | 1.139 |
    | 5    | CIB  [4,384,20,20] |      7 | 500 M | 46.08 | 58.34 | 0.790 |
    | 1    | C2f  [4,192,80,80] |      4 | 944 M | 50.29 | 56.42 | 0.891 |

    Every win is at or below 315 M and both losses at or above 500 M, so the
    crossover is bracketed to a 1.6x-wide band and 400 M sits in the middle of
    it.  Two losing points is thin evidence for the exact value -- what the data
    supports firmly is the *sign*: the megakernel is a scheduling optimization,
    so it stops paying once the block is big enough to be work-bound.  Both
    losses are the batch-4 cases.
    """
    return work <= _MEGA_WORK_MAX
