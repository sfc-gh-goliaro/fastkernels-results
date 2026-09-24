"""YOLOv10 native neck.

Fast path: the whole neck is re-expressed as a fixed sequence of hand-written
Triton kernels over NHWC buffers, captured once into a CUDA graph.

* Conv+BN is folded into a biased convolution at build time; the bias add, the
  SiLU and the CIB residual add are epilogues of the conv kernel, so a whole
  Conv-BN-SiLU block is one kernel launch.
* The four ``Concat`` nodes and the two ``chunk`` splits never materialize.
  Every kernel takes a row stride + channel offset for its input, output and
  residual operand, so a producer writes straight into its channel slice of the
  consumer's buffer, and the pointwise kernel can read a concatenation as two
  source slices (two dots sharing one accumulator).
* Both nearest-2x upsamples are folded into the consumer as a row remap of its
  first source slice, so the upsampled tensors are never written out.

The neck is entirely launch/latency bound at the captured shapes (batch 1 / 4,
80x80 down to 20x20), so kernel count -- not arithmetic -- dominates: 23
launches (one eager NCHW->NHWC staging kernel plus a 22-node graph) replace the
~120 eager cuDNN / elementwise launches of the reference implementation.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn

from ..L1.interpolate import Interpolate
from ..L2.yolov10_c2f import YOLOC2f, YOLOC2fCIB
from ..L2.yolov10_concat import YOLOConcat
from ..L2.yolov10_conv import YOLOConv
from ..L2.yolov10_scdown import YOLOSCDown

try:
    import triton
    import triton.language as tl

    _HAVE_TRITON = True
except Exception:  # pragma: no cover - no Triton -> eager fallback
    _HAVE_TRITON = False


if _HAVE_TRITON:

    @triton.jit
    def _pw_conv(X1, X2, Wt, Bs, Y, R,
                 NROW: tl.constexpr, HO: tl.constexpr, WO: tl.constexpr,
                 C1: tl.constexpr, C2: tl.constexpr, CO: tl.constexpr,
                 S1: tl.constexpr, O1: tl.constexpr,
                 S2: tl.constexpr, O2: tl.constexpr,
                 YS: tl.constexpr, YO: tl.constexpr,
                 RS: tl.constexpr, RO: tl.constexpr,
                 UP1: tl.constexpr, YPAD: tl.constexpr,
                 ACT: tl.constexpr, RES: tl.constexpr, MASK: tl.constexpr,
                 MMA: tl.constexpr,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        """1x1 NHWC convolution over one or two concatenated source slices.

        Source ``k`` lives at ``row * Sk + Ok``; ``UP1`` remaps source 1's row
        to the nearest-2x-upsample source pixel.  ``C2 == 0`` -> single source.
        ``YPAD`` writes into the interior of a halo-padded output buffer, and
        ``MMA`` only tags the compile-cache entry (see ``_compile_ops``).
        """
        pid = tl.program_id(0)
        NB_N: tl.constexpr = CO // BN
        rm = (pid // NB_N) * BM + tl.arange(0, BM)
        rn = (pid % NB_N) * BN + tl.arange(0, BN)
        rk = tl.max_contiguous(tl.multiple_of(tl.arange(0, BK), BK), BK)
        if UP1 or YPAD:
            nb = rm // (HO * WO)
            hw = rm % (HO * WO)
            ho = hw // WO
            wo = hw % WO
        if UP1:
            r1 = (nb * (HO // 2) + ho // 2) * (WO // 2) + wo // 2
        else:
            r1 = rm
        if YPAD:
            ry = (nb * (HO + 2 * YPAD) + ho + YPAD) * (WO + 2 * YPAD) + wo + YPAD
        else:
            ry = rm
        if MASK:
            mr = (rm < NROW)[:, None]
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        xp = X1 + (r1 * S1 + O1)[:, None] + rk[None, :]
        wp = Wt + rk[:, None] * CO + rn[None, :]
        for _ in range(0, C1 // BK):
            if MASK:
                a = tl.load(xp, mask=mr, other=0.0)
            else:
                a = tl.load(xp)
            acc = tl.dot(a, tl.load(wp), acc)
            xp += BK
            wp += BK * CO
        if C2 > 0:
            xp2 = X2 + (rm * S2 + O2)[:, None] + rk[None, :]
            for _ in range(0, C2 // BK):
                if MASK:
                    a = tl.load(xp2, mask=mr, other=0.0)
                else:
                    a = tl.load(xp2)
                acc = tl.dot(a, tl.load(wp), acc)
                xp2 += BK
                wp += BK * CO
        acc += tl.load(Bs + rn)[None, :]
        if ACT:
            acc = acc * tl.sigmoid(acc)
        rp = R + (rm * RS + RO)[:, None] + rn[None, :]
        yp = Y + (ry * YS + YO)[:, None] + rn[None, :]
        if MASK:
            if RES:
                acc += tl.load(rp, mask=mr, other=0.0).to(tl.float32)
            tl.store(yp, acc.to(Y.dtype.element_ty), mask=mr)
        else:
            if RES:
                acc += tl.load(rp).to(tl.float32)
            tl.store(yp, acc.to(Y.dtype.element_ty))

    @triton.jit
    def _sp_conv(X, Wt, Bs, Y, R,
                 NROW: tl.constexpr, HI: tl.constexpr, WI: tl.constexpr,
                 HO: tl.constexpr, WO: tl.constexpr,
                 CI: tl.constexpr, CO: tl.constexpr,
                 XS: tl.constexpr, XO: tl.constexpr,
                 YS: tl.constexpr, YO: tl.constexpr,
                 RS: tl.constexpr, RO: tl.constexpr,
                 KH: tl.constexpr, KW: tl.constexpr,
                 ST: tl.constexpr, PD: tl.constexpr,
                 ACT: tl.constexpr, RES: tl.constexpr, MASK: tl.constexpr,
                 MMA: tl.constexpr,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        """KHxKW NHWC convolution (implicit GEMM, one flattened k loop).

        ``MMA`` only tags the compile-cache entry: 1 = lowered with Blackwell's
        tcgen05 MMA, 0 = lowered with ``mma.sync`` (see ``_compile_ops``).  The
        tcgen05 path costs ~3us of per-launch setup but is much faster per k
        step, so it only pays off for the kernels with a long k loop."""
        pid = tl.program_id(0)
        NB_N: tl.constexpr = CO // BN
        rm = (pid // NB_N) * BM + tl.arange(0, BM)
        rn = (pid % NB_N) * BN + tl.arange(0, BN)
        rk = tl.max_contiguous(tl.multiple_of(tl.arange(0, BK), BK), BK)
        KPT: tl.constexpr = CI // BK
        hw = rm % (HO * WO)
        ho = hw // WO
        wo = hw % WO
        base = (rm // (HO * WO)) * HI * WI * XS + XO
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        wp = Wt + rk[:, None] * CO + rn[None, :]
        for s in range(0, KH * KW * KPT):
            t = s // KPT
            hi = ho * ST - PD + t // KW
            wi = wo * ST - PD + t % KW
            sm = (hi >= 0) & (hi < HI) & (wi >= 0) & (wi < WI)
            if MASK:
                sm = sm & (rm < NROW)
            xp = base + (hi * WI + wi) * XS + (s % KPT) * BK
            a = tl.load(X + xp[:, None] + rk[None, :], mask=sm[:, None], other=0.0)
            acc = tl.dot(a, tl.load(wp), acc)
            wp += BK * CO
        acc += tl.load(Bs + rn)[None, :]
        if ACT:
            acc = acc * tl.sigmoid(acc)
        rp = R + (rm * RS + RO)[:, None] + rn[None, :]
        yp = Y + (rm * YS + YO)[:, None] + rn[None, :]
        if MASK:
            mr = (rm < NROW)[:, None]
            if RES:
                acc += tl.load(rp, mask=mr, other=0.0).to(tl.float32)
            tl.store(yp, acc.to(Y.dtype.element_ty), mask=mr)
        else:
            if RES:
                acc += tl.load(rp).to(tl.float32)
            tl.store(yp, acc.to(Y.dtype.element_ty))

    @triton.jit
    def _dw_conv(X, Wt, Bs, Y, R,
                 NROW: tl.constexpr, HI: tl.constexpr, WI: tl.constexpr,
                 HO: tl.constexpr, WO: tl.constexpr, C: tl.constexpr,
                 XS: tl.constexpr, XO: tl.constexpr,
                 YS: tl.constexpr, YO: tl.constexpr,
                 RS: tl.constexpr, RO: tl.constexpr,
                 KH: tl.constexpr, KW: tl.constexpr,
                 ST: tl.constexpr, PD: tl.constexpr,
                 ACT: tl.constexpr, RES: tl.constexpr, MASK: tl.constexpr,
                 XPAD: tl.constexpr, BM: tl.constexpr, BC: tl.constexpr):
        """Depthwise NHWC convolution, same slice / epilogue conventions.

        ``XPAD`` marks the input buffer as carrying a ``PD``-wide zero halo, so
        every tap is a compile-time offset from one base row and no per-tap
        bounds mask (nor its address arithmetic) is needed -- which is what the
        7x7 depthwise kernel is otherwise dominated by."""
        pid = tl.program_id(0)
        NB_C: tl.constexpr = C // BC
        rm = (pid // NB_C) * BM + tl.arange(0, BM)
        rc = (pid % NB_C) * BC + tl.arange(0, BC)
        rc = tl.max_contiguous(tl.multiple_of(rc, BC), BC)
        hw = rm % (HO * WO)
        ho = hw // WO
        wo = hw % WO
        nb = rm // (HO * WO)
        acc = tl.zeros((BM, BC), dtype=tl.float32)
        if XPAD:
            HP: tl.constexpr = HI + 2 * PD
            WP: tl.constexpr = WI + 2 * PD
            xb = ((nb * HP + ho * ST) * WP + wo * ST) * XS + XO
            for kh in tl.static_range(KH):
                for kw in tl.static_range(KW):
                    a = tl.load(X + xb[:, None] + (kh * WP + kw) * XS + rc[None, :])
                    w = tl.load(Wt + (kh * KW + kw) * C + rc)
                    # fp16 product, fp32 accumulate: same error as converting
                    # first (one rounding either way) at ~12% less ALU work,
                    # which is what the 7x7 depthwise is bound by.
                    acc += (a * w[None, :]).to(tl.float32)
        else:
            base = nb * HI * WI * XS + XO
            for kh in tl.static_range(KH):
                hi = ho * ST - PD + kh
                hm = (hi >= 0) & (hi < HI)
                for kw in tl.static_range(KW):
                    wi = wo * ST - PD + kw
                    sm = hm & (wi >= 0) & (wi < WI)
                    if MASK:
                        sm = sm & (rm < NROW)
                    xp = base + (hi * WI + wi) * XS
                    a = tl.load(X + xp[:, None] + rc[None, :], mask=sm[:, None], other=0.0)
                    w = tl.load(Wt + (kh * KW + kw) * C + rc)
                    acc += (a * w[None, :]).to(tl.float32)
        acc += tl.load(Bs + rc)[None, :]
        if ACT:
            acc = acc * tl.sigmoid(acc)
        rp = R + (rm * RS + RO)[:, None] + rc[None, :]
        yp = Y + (rm * YS + YO)[:, None] + rc[None, :]
        if MASK:
            mr = (rm < NROW)[:, None]
            if RES:
                acc += tl.load(rp, mask=mr, other=0.0).to(tl.float32)
            tl.store(yp, acc.to(Y.dtype.element_ty), mask=mr)
        else:
            if RES:
                acc += tl.load(rp).to(tl.float32)
            tl.store(yp, acc.to(Y.dtype.element_ty))

    @triton.jit
    def _tile_t(X, Y, pid, HW: tl.constexpr, C: tl.constexpr,
                BHW: tl.constexpr, BC: tl.constexpr):
        """One (BHW x BC) tile of an NCHW -> NHWC transpose."""
        NT_HW: tl.constexpr = (HW + BHW - 1) // BHW
        NT_C: tl.constexpr = C // BC
        b = pid // (NT_HW * NT_C)
        r = pid % (NT_HW * NT_C)
        rhw = (r // NT_C) * BHW + tl.arange(0, BHW)
        rc = (r % NT_C) * BC + tl.arange(0, BC)
        m = rhw < HW
        v = tl.load(X + b * C * HW + rc[:, None] * HW + rhw[None, :],
                    mask=m[None, :], other=0.0)
        tl.store(Y + (b * HW + rhw[:, None]) * C + rc[None, :], tl.trans(v),
                 mask=m[:, None])

    @triton.jit
    def _stage_in(P3, P4, P5, S3T, S4T, S5T,
                  HW3: tl.constexpr, HW4: tl.constexpr, HW5: tl.constexpr,
                  N0: tl.constexpr, N1: tl.constexpr, N2: tl.constexpr,
                  BHW: tl.constexpr, BC3: tl.constexpr, BC4: tl.constexpr,
                  BC5: tl.constexpr):
        """Stage the three backbone features NCHW -> NHWC in one launch."""
        pid = tl.program_id(0)
        if pid < N0:
            _tile_t(P3, S3T, pid, HW3, 64, BHW, BC3)
        elif pid < N0 + N1:
            _tile_t(P4, S4T, pid - N0, HW4, 128, BHW, BC4)
        else:
            _tile_t(P5, S5T, pid - N0 - N1, HW5, 256, BHW, BC5)


# Per-op launch configuration, keyed by (batch, position in the op list built
# below).  Tuned offline on B200 for the captured shapes; unlisted (batch, op)
# pairs fall back to the per-call defaults in ``_build``.  These layers are far
# too small to fill the GPU, so the tile shape that wins is the one that spreads
# the work over the most CTAs, not the one with the best arithmetic intensity --
# hence the measured table rather than a formula.
_CFG: dict[tuple, tuple] = {
    # (batch, op index): (BM, BN, BK, num_warps, num_stages, num_ctas, mma)
    (4, 0): (64, 32, 128, 4, 2, 1, 0),
    (4, 1): (64, 64, 32, 4, 4, 1, 0),
    (4, 2): (64, 64, 32, 4, 4, 1, 0),
    (4, 3): (32, 64, 64, 4, 3, 1, 0),
    (4, 4): (64, 64, 64, 4, 2, 1, 0),
    (4, 5): (64, 32, 32, 4, 3, 1, 0),
    (4, 6): (64, 32, 32, 4, 3, 1, 0),
    (4, 7): (64, 64, 32, 4, 2, 1, 0),
    (4, 8): (64, 32, 64, 4, 3, 1, 0),
    (4, 9): (32, 64, 64, 4, 3, 1, 0),
    (4, 10): (32, 32, 64, 4, 3, 1, 0),
    (4, 11): (32, 32, 64, 4, 3, 1, 0),
    (4, 12): (32, 64, 64, 4, 3, 1, 0),
    (4, 13): (32, 64, 32, 4, 3, 1, 0),
    (4, 14): (16, 32, None, 8, 1, 1, 0),
    (4, 15): (64, 64, 128, 4, 3, 1, 0),
    (4, 16): (16, 32, None, 8, 1, 1, 0),
    (4, 17): (16, 64, 128, 4, 2, 1, 0),
    (4, 18): (16, 64, None, 8, 1, 1, 0),
    (4, 19): (16, 64, 64, 4, 3, 1, 0),
    (4, 20): (16, 32, None, 8, 1, 1, 0),
    (4, 21): (64, 64, 128, 4, 3, 1, 0),
    (1, 0): (32, 64, 128, 8, 3, 1, 1),
    (1, 1): (32, 32, 64, 4, 4, 1, 1),
    (1, 2): (32, 32, 64, 4, 4, 1, 1),
    (1, 3): (16, 64, 64, 4, 3, 1, 1),
    (1, 4): (32, 32, 64, 4, 2, 1, 1),
    (1, 5): (32, 32, 32, 4, 3, 1, 1),
    (1, 6): (32, 32, 32, 4, 3, 1, 1),
    (1, 7): (16, 64, 32, 4, 3, 1, 1),
    (1, 8): (32, 32, 64, 4, 4, 1, 1),
    (1, 9): (16, 128, 64, 4, 3, 1, 1),
    (1, 10): (32, 32, 64, 4, 4, 1, 1),
    (1, 11): (32, 32, 64, 4, 4, 1, 1),
    (1, 12): (16, 128, 64, 4, 3, 1, 1),
    (1, 13): (16, 64, 128, 4, 2, 1, 0),
    (1, 14): (16, 32, None, 8, 1, 1, 0),
    (1, 15): (32, 32, 32, 4, 4, 1, 1),
    (1, 16): (16, 32, None, 8, 1, 1, 1),
    (1, 17): (16, 32, 128, 4, 3, 1, 1),
    (1, 18): (16, 32, None, 8, 1, 1, 1),
    (1, 19): (16, 32, 256, 4, 3, 1, 1),
    (1, 20): (16, 32, None, 8, 1, 1, 1),
    (1, 21): (16, 64, 128, 4, 3, 1, 1),
}


def _cfg(b: int, i: int, default: tuple) -> tuple:
    """(BM, BN, BK, num_warps, num_stages, num_ctas) for op *i* at batch *b*."""
    default = default + (1,) * (7 - len(default))
    got = _CFG.get((b, i), _CFG.get(i, ()))
    return tuple(g if g is not None else d for g, d in zip(got + (None,) * 7, default))


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


def _fit(block: int, total: int) -> int:
    """Largest power-of-two <= *block* that divides *total* (kernels index whole
    blocks, so a tile that does not divide the channel count would silently drop
    channels; tuned configs always divide, this only guards odd shapes)."""
    while block > 8 and total % block:
        block //= 2
    return block if total % block == 0 else _gcd(block, total)


def _fold_bn(m: YOLOConv):
    """Fused (weight fp32, bias fp32) of a YOLOConv's conv (+BN)."""
    conv = m.conv
    w = conv.weight.detach().float()
    b = (conv.bias.detach().float() if conv.bias is not None
         else torch.zeros(w.shape[0], device=w.device, dtype=torch.float32))
    bn = getattr(m, "bn", None)
    if bn is not None:
        scale = bn.weight.detach().float() / torch.sqrt(
            bn.running_var.detach().float() + bn.eps)
        w = w * scale.view(-1, 1, 1, 1)
        b = bn.bias.detach().float() + (b - bn.running_mean.detach().float()) * scale
    return w, b


class _Plan:
    __slots__ = ("ops", "graph", "stage", "outputs", "keep")


class YOLOv10Neck(nn.Module):
    def __init__(self):
        super().__init__()
        self._upsample = Interpolate()
        self.cat1 = YOLOConcat(1)
        self.c2f_p4 = YOLOC2f(384, 128, n=1, shortcut=False)
        self.cat2 = YOLOConcat(1)
        self.c2f_p3 = YOLOC2f(192, 64, n=1, shortcut=False)
        self.down_p3 = YOLOConv(64, 64, 3, 2)
        self.cat3 = YOLOConcat(1)
        self.c2f_n4 = YOLOC2f(192, 128, n=1, shortcut=False)
        self.down_n4 = YOLOSCDown(128, 128, 3, 2)
        self.cat4 = YOLOConcat(1)
        self.c2fcib_n5 = YOLOC2fCIB(384, 256, n=1, shortcut=True, lk=True)
        # Cached launch plans (folded weights + buffers + graph), keyed by input
        # shape/dtype/device; dropped whenever new weights arrive.
        self._plans: dict[tuple, object] = {}
        self.register_load_state_dict_post_hook(lambda mod, _out: mod._plans.clear())

    # ------------------------------------------------------------------ eager
    def _forward_eager(self, feats: dict[str, torch.Tensor]):
        p3_backbone = feats["p3_backbone"]
        p4_backbone = feats["p4_backbone"]
        p5_backbone = feats["p5_backbone"]

        x = self._upsample(p5_backbone, scale_factor=2.0, mode="nearest")
        x = self.cat1([x, p4_backbone])
        p4 = self.c2f_p4(x)

        x = self._upsample(p4, scale_factor=2.0, mode="nearest")
        x = self.cat2([x, p3_backbone])
        p3 = self.c2f_p3(x)

        x = self.down_p3(p3)
        x = self.cat3([x, p4])
        n4 = self.c2f_n4(x)

        x = self.down_n4(n4)
        x = self.cat4([x, p5_backbone])
        n5 = self.c2fcib_n5(x)
        return [p3, n4, n5]

    # ------------------------------------------------------------------- fast
    def _build(self, b: int, s3: int, device, dtype):
        """Fold the weights, allocate the NHWC buffers and list the launches.

        Buffers are named after the tensor they hold: ``sp{3,4,5}`` the staged
        backbone features, ``cat3`` / ``n5blk`` / ``p4blk`` / ``p3blk`` /
        ``n4blk`` the concat (and C2f chunk) buffers each producer writes its
        own channel slice of, ``t*`` scratch, ``o_*`` the three outputs.  A
        buffer feeding a depthwise conv is allocated with a zeroed halo so that
        kernel needs no bounds masks.  Every ``pw`` / ``sp_conv`` / ``dw`` call
        below mirrors one Conv-BN-Act of the reference module, in execution
        order; the ``(x, stride, offset, channels)`` groups are the row stride
        and channel offset of that operand inside its buffer.
        """
        s4, s5 = s3 // 2, s3 // 4
        plan = _Plan()
        keep = []

        def gemm_w(m):
            """Conv weight as an implicit-GEMM operand: (KH*KW*CI, CO)."""
            w, bias = _fold_bn(m)
            co, ci, kh, kw = w.shape
            wt = w.permute(2, 3, 1, 0).reshape(kh * kw * ci, co).contiguous().to(dtype)
            keep.extend((wt, bias))
            return wt, bias

        def dw_w(m, extra=None):
            w, bias = _fold_bn(m)
            if extra is not None:                      # RepVGGDW: 7x7 + padded 3x3
                w2, b2 = _fold_bn(extra)
                pad = (w.shape[-1] - w2.shape[-1]) // 2
                w = w + torch.nn.functional.pad(w2, [pad] * 4)
                bias = bias + b2
            c, _, kh, kw = w.shape
            wt = w.reshape(c, kh * kw).t().contiguous().to(dtype)
            keep.extend((wt, bias))
            return wt, bias

        def buf(sp, ch, pad=0):
            """NHWC activation buffer, optionally with a zeroed ``pad`` halo."""
            if pad:
                t = torch.zeros((b, sp + 2 * pad, sp + 2 * pad, ch),
                                device=device, dtype=dtype)
            else:
                t = torch.empty((b, sp, sp, ch), device=device, dtype=dtype)
            keep.append(t)
            return t

        sp3, sp4, sp5 = buf(s3, 64), buf(s4, 128), buf(s5, 256)
        p4blk, cat3, t4_64 = buf(s4, 192), buf(s4, 192), buf(s4, 64)
        p3blk, t3_32, o_p3 = buf(s3, 96), buf(s3, 32), buf(s3, 64)
        n4blk, o_n4 = buf(s4, 192), buf(s4, 128)
        t4_128 = buf(s4, 128, pad=1)          # dw 3x3 s2 input
        sc20, n5blk, o_n5 = buf(s5, 128), buf(s5, 384), buf(s5, 256)
        t5_128a, t5_256b = buf(s5, 128), buf(s5, 256)
        t5_256a = buf(s5, 256, pad=3)         # dw 7x7 input
        t5_128b = buf(s5, 128, pad=1)         # dw 3x3 input

        ops = []

        def pw(m, x1, s1, o1, c1, y, ys, yo, co, sp, x2=None, s2=0, o2=0, c2=0,
               up1=0, ypad=0, act=1, res=None, rs=0, ro=0, cfg=(64, None, 64, 4, 3)):
            wt, bias = gemm_w(m)
            bm, bn, bk, nw, ns, nc, mma = _cfg(b, len(ops), cfg)
            bn = _fit(bn or min(co, 128), co)
            bk = _fit(bk, c1 if not c2 else _gcd(c1, c2))
            nrow = b * sp * sp
            ops.append((_pw_conv, (-(-nrow // bm) * (co // bn),),
                        (x1, x2 if x2 is not None else x1, wt, bias, y,
                         res if res is not None else y, nrow, sp, sp, c1, c2, co,
                         s1, o1, s2, o2, ys, yo, rs, ro, up1, ypad, act,
                         0 if res is None else 1, int(nrow % bm != 0), mma,
                         bm, bn, bk),
                        {"num_warps": nw, "num_stages": ns, "num_ctas": nc}, mma))

        def sp_conv(m, x, xs, xo, ci, y, ys, yo, co, spi, spo, k, st, pd,
                    act=1, res=None, rs=0, ro=0, cfg=(64, None, None, 4, 3)):
            wt, bias = gemm_w(m)
            bm, bn, bk, nw, ns, nc, mma = _cfg(b, len(ops), cfg)
            bn = _fit(bn or min(co, 128), co)
            bk = _fit(bk or ci, ci)
            nrow = b * spo * spo
            ops.append((_sp_conv, (-(-nrow // bm) * (co // bn),),
                        (x, wt, bias, y, res if res is not None else y, nrow,
                         spi, spi, spo, spo, ci, co, xs, xo, ys, yo, rs, ro,
                         k, k, st, pd, act, 0 if res is None else 1,
                         int(nrow % bm != 0), mma, bm, bn, bk),
                        {"num_warps": nw, "num_stages": ns, "num_ctas": nc}, mma))

        def dw(m, x, xs, xo, c, y, ys, yo, spi, spo, k, st, pd, act=1,
               res=None, rs=0, ro=0, extra=None, xpad=0, cfg=(64, 64, None, 4, 1)):
            wt, bias = dw_w(m, extra)
            bm, bc, _unused, nw, ns, nc, _mma = _cfg(b, len(ops), cfg)
            bc = _fit(bc, c)
            nrow = b * spo * spo
            ops.append((_dw_conv, (-(-nrow // bm) * (c // bc),),
                        (x, wt, bias, y, res if res is not None else y, nrow,
                         spi, spi, spo, spo, c, xs, xo, ys, yo, rs, ro,
                         k, k, st, pd, act, 0 if res is None else 1,
                         int(nrow % bm != 0), xpad, bm, bc),
                        {"num_warps": nw, "num_stages": ns, "num_ctas": nc}, 1))

        # --- c2f_p4 on cat([up(p5), p4]) ---------------------------------
        pw(self.c2f_p4.cv1, sp5, 256, 0, 256, p4blk, 192, 0, 128, s4,
           x2=sp4, s2=128, o2=0, c2=128, up1=1)
        blk = self.c2f_p4.m[0]
        sp_conv(blk.cv1, p4blk, 192, 64, 64, t4_64, 64, 0, 64, s4, s4, 3, 1, 1)
        sp_conv(blk.cv2, t4_64, 64, 0, 64, p4blk, 192, 128, 64, s4, s4, 3, 1, 1)
        pw(self.c2f_p4.cv2, p4blk, 192, 0, 192, cat3, 192, 64, 128, s4)

        # --- c2f_p3 on cat([up(p4), p3]) ---------------------------------
        pw(self.c2f_p3.cv1, cat3, 192, 64, 128, p3blk, 96, 0, 64, s3,
           x2=sp3, s2=64, o2=0, c2=64, up1=1)
        blk = self.c2f_p3.m[0]
        sp_conv(blk.cv1, p3blk, 96, 32, 32, t3_32, 32, 0, 32, s3, s3, 3, 1, 1)
        sp_conv(blk.cv2, t3_32, 32, 0, 32, p3blk, 96, 64, 32, s3, s3, 3, 1, 1)
        pw(self.c2f_p3.cv2, p3blk, 96, 0, 96, o_p3, 64, 0, 64, s3)

        # --- down_p3 -> cat3, c2f_n4 -------------------------------------
        sp_conv(self.down_p3, o_p3, 64, 0, 64, cat3, 192, 0, 64, s3, s4, 3, 2, 1)
        pw(self.c2f_n4.cv1, cat3, 192, 0, 192, n4blk, 192, 0, 128, s4)
        blk = self.c2f_n4.m[0]
        sp_conv(blk.cv1, n4blk, 192, 64, 64, t4_64, 64, 0, 64, s4, s4, 3, 1, 1)
        sp_conv(blk.cv2, t4_64, 64, 0, 64, n4blk, 192, 128, 64, s4, s4, 3, 1, 1)
        pw(self.c2f_n4.cv2, n4blk, 192, 0, 192, o_n4, 128, 0, 128, s4)

        # --- down_n4 (SCDown) -> cat([sc, p5]), c2fcib_n5 -----------------
        pw(self.down_n4.cv1, o_n4, 128, 0, 128, t4_128, 128, 0, 128, s4, ypad=1)
        dw(self.down_n4.cv2, t4_128, 128, 0, 128, sc20, 128, 0, s4, s5, 3, 2, 1,
           act=0, xpad=1)
        pw(self.c2fcib_n5.cv1, sc20, 128, 0, 128, n5blk, 384, 0, 256, s5,
           x2=sp5, s2=256, o2=0, c2=256)
        cib = self.c2fcib_n5.m[0].cv1
        dw(cib[0], n5blk, 384, 128, 128, t5_128a, 128, 0, s5, s5, 3, 1, 1)
        pw(cib[1], t5_128a, 128, 0, 128, t5_256a, 256, 0, 256, s5, ypad=3)
        dw(cib[2].conv, t5_256a, 256, 0, 256, t5_256b, 256, 0, s5, s5, 7, 1, 3,
           extra=cib[2].conv1, xpad=1)
        pw(cib[3], t5_256b, 256, 0, 256, t5_128b, 128, 0, 128, s5, ypad=1)
        dw(cib[4], t5_128b, 128, 0, 128, n5blk, 384, 256, s5, s5, 3, 1, 1,
           res=n5blk, rs=384, ro=128, xpad=1)
        pw(self.c2fcib_n5.cv2, n5blk, 384, 0, 384, o_n5, 256, 0, 256, s5)

        # --- eager input staging (reads the live input tensors) -----------
        bhw, bcmax, _u, snw, _u2, _u3, _u4 = _cfg(b, -1, (32, 128, None, 4, 1))
        hw3, hw4, hw5 = s3 * s3, s4 * s4, s5 * s5
        bc3, bc4, bc5 = min(64, bcmax), min(128, bcmax), min(256, bcmax)

        def ntiles(nhw, c, bc):
            return ((nhw + bhw - 1) // bhw) * (c // bc) * b

        n0 = ntiles(hw3, 64, bc3)
        n1 = ntiles(hw4, 128, bc4)
        n2 = ntiles(hw5, 256, bc5)
        plan.stage = (_stage_in, (n0 + n1 + n2,),
                      (sp3, sp4, sp5, hw3, hw4, hw5, n0, n1, n2, bhw, bc3, bc4, bc5),
                      {"num_warps": snw})
        plan.ops = ops
        plan.keep = keep
        plan.outputs = [o_p3.permute(0, 3, 1, 2), o_n4.permute(0, 3, 1, 2),
                        o_n5.permute(0, 3, 1, 2)]
        plan.graph = None
        return plan

    @staticmethod
    def _run_ops(ops):
        for kernel, grid, args, kw, _mma in ops:
            kernel[grid](*args, **kw)

    @staticmethod
    def _compile_ops(ops):
        """First launch of every op, grouped by the MMA lowering it wants.

        ``DISABLE_MMA_V5`` is read by Triton while *compiling*, so the flag is
        toggled around each group's first launch; the ``MMA`` constexpr keeps
        the two lowerings in separate cache entries."""
        for off in (1, 0):
            sel = [o for o in ops if (o[4] == 0) == bool(off)]
            if not sel:
                continue
            prev = os.environ.get("DISABLE_MMA_V5")
            if off:
                os.environ["DISABLE_MMA_V5"] = "1"
            else:
                os.environ.pop("DISABLE_MMA_V5", None)
            try:
                for kernel, grid, args, kw, _mma in sel:
                    kernel[grid](*args, **kw)
            finally:
                if prev is None:
                    os.environ.pop("DISABLE_MMA_V5", None)
                else:
                    os.environ["DISABLE_MMA_V5"] = prev

    def _capture(self, plan, feats):
        kern, grid, args, skw = plan.stage
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            self._compile_ops(plan.ops)
            for _ in range(3):
                kern[grid](feats["p3_backbone"], feats["p4_backbone"],
                           feats["p5_backbone"], *args, **skw)
                self._run_ops(plan.ops)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._run_ops(plan.ops)
        plan.graph = graph

    def _plan_for(self, feats, p3, p4, p5):
        ok = (_HAVE_TRITON and p3.is_cuda
              and p3.dtype in (torch.float16, torch.bfloat16)
              and p3.dtype == p4.dtype == p5.dtype
              and p3.dim() == 4 and p3.shape[1] == 64 and p4.shape[1] == 128
              and p5.shape[1] == 256 and p3.shape[2] == p3.shape[3]
              and p3.shape[2] == 2 * p4.shape[2] == 4 * p5.shape[2]
              and p4.shape[2] == p4.shape[3] and p5.shape[2] == p5.shape[3]
              and p3.shape[2] % 8 == 0
              and p3.is_contiguous() and p4.is_contiguous() and p5.is_contiguous())
        if not ok:
            return False
        try:
            plan = self._build(p3.shape[0], p3.shape[2], p3.device, p3.dtype)
            self._capture(plan, feats)
            return plan
        except Exception:
            if os.environ.get("FK_NECK_DEBUG"):
                raise
            return False

    def forward(self, feats: dict[str, torch.Tensor]):
        if self.training:            # folded BN == eval semantics only
            return self._forward_eager(feats)
        p3 = feats["p3_backbone"]
        p4 = feats["p4_backbone"]
        p5 = feats["p5_backbone"]
        key = (p3.shape, p4.shape, p5.shape, p3.dtype, p3.device)
        plan = self._plans.get(key)
        if plan is None:
            plan = self._plans[key] = self._plan_for(feats, p3, p4, p5)
        if plan is False:
            return self._forward_eager(feats)
        kern, grid, args, skw = plan.stage
        kern[grid](p3, p4, p5, *args, **skw)
        plan.graph.replay()
        return plan.outputs
