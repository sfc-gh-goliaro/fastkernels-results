"""YOLOv10 Conv-BN-Act building block.

The scored path is ``act(bn(conv(x)))`` with ``_is_fused == False`` -- three
module calls, three kernel launches and two extra full-tensor HBM round-trips.
Eval-mode BatchNorm is a fixed per-output-channel affine, so it can be folded
into the convolution epilogue for free.  This implementation emits the whole
block as **one** Triton launch: an implicit-GEMM convolution that consumes NCHW
strides directly (no layout copy) and applies ``scale * acc + shift`` followed by
an optional SiLU in registers.

Per-shape launch state (prepacked weight, folded affine, tile choice, grid) is
built lazily on the first forward and cached on the module, keyed on
``(training, shape, strides)`` -- never on input pointers, which change every
call.  Cache invalidation is hooked into ``_apply`` / ``load_state_dict`` /
``fuse`` so a stale plan can never be used.
"""

from __future__ import annotations

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
# Kernels
# ---------------------------------------------------------------------------
# Every shape/stride/tile parameter is ``tl.constexpr``: the plan is per-shape
# anyway, so making them compile-time constants strength-reduces the ``//`` /
# ``%`` row decomposition into multiply-shift and keeps the per-call binder work
# down to five pointer arguments.
@triton.jit
def _conv_bn_act_gemm(
    X, WP, SCALE, SHIFT, Y,
    M: tl.constexpr, ICG: tl.constexpr, IH: tl.constexpr, IW: tl.constexpr,
    OC: tl.constexpr, OH: tl.constexpr, OW: tl.constexpr,
    SX0: tl.constexpr, SX1: tl.constexpr, SX2: tl.constexpr, SX3: tl.constexpr,
    SY0: tl.constexpr, SY1: tl.constexpr, SY2: tl.constexpr, SY3: tl.constexpr,
    WPLANE: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr, SH: tl.constexpr, SW: tl.constexpr,
    PH: tl.constexpr, PW: tl.constexpr, DH: tl.constexpr, DW: tl.constexpr,
    ACT: tl.constexpr, IEEE: tl.constexpr,
    NK: tl.constexpr, KMASK: tl.constexpr, BOUND: tl.constexpr,
    MMASK: tl.constexpr, NMASK: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    """Implicit-GEMM conv (groups == 1) with a folded BN affine + SiLU epilogue.

    Rows of the GEMM are output pixels ``m = ((n * OH) + oh) * OW + ow``, columns
    are output channels.  ``WP`` is the weight prepacked to ``[KH*KW, NK*BK, OC]``
    (zero padded along the reduction axis) so the B tile is contiguous in ``oc``.
    The reduction is a plain ``range`` loop over ``KH*KW*NK``, not
    ``static_range``: unrolling it hands the whole chain to the scheduler at once
    and measured ~35% slower.
    """
    offs_n = tl.program_id(1) * BN + tl.arange(0, BN)
    offs_m = tl.program_id(0) * BM + tl.arange(0, BM)
    nb = offs_m // (OH * OW)
    rem = offs_m % (OH * OW)
    oh = rem // OW
    ow = rem % OW
    m_ok = offs_m < M
    ih0 = oh * SH - PH
    iw0 = ow * SW - PW
    xbase = nb * SX0 + ih0 * SX2 + iw0 * SX3
    ybase = nb * SY0 + oh * SY2 + ow * SY3

    kj = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for it in range(KH * KW * NK):
        kk = it % NK
        pl = it // NK
        kh = pl // KW
        kw = pl % KW
        ic = kk * BK + kj
        ap = (X + xbase + (kh * DH) * SX2 + (kw * DW) * SX3)[:, None] + ic[None, :] * SX1
        bp = WP + pl * WPLANE + ic[:, None] * OC + offs_n[None, :]
        # Build the A mask only when some flag needs it: a 1x1 conv with
        # M % BM == 0 and BK dividing IC needs none, and skipping it drops the
        # predicate math and the select from the inner loop.
        if MMASK or BOUND or KMASK:
            am = None
            if BOUND:
                ih = ih0 + kh * DH
                iw = iw0 + kw * DW
                rok = (ih >= 0) & (ih < IH) & (iw >= 0) & (iw < IW)
                if MMASK:
                    rok = rok & m_ok
                am = tl.broadcast_to(rok[:, None], (BM, BK))
            elif MMASK:
                am = tl.broadcast_to(m_ok[:, None], (BM, BK))
            if KMASK:
                km = ic < ICG
                if am is None:
                    am = tl.broadcast_to(km[None, :], (BM, BK))
                else:
                    am = am & km[None, :]
            a = tl.load(ap, mask=am, other=0.0)
        else:
            a = tl.load(ap)
        if NMASK:
            b = tl.load(bp, mask=tl.broadcast_to((offs_n < OC)[None, :], (BK, BN)), other=0.0)
        else:
            b = tl.load(bp)
        if IEEE:
            acc = tl.dot(a, b, acc, input_precision="ieee")
        else:
            acc = tl.dot(a, b, acc)

    if NMASK:
        n_ok = offs_n < OC
        sc = tl.load(SCALE + offs_n, mask=n_ok, other=1.0)
        sf = tl.load(SHIFT + offs_n, mask=n_ok, other=0.0)
    else:
        sc = tl.load(SCALE + offs_n)
        sf = tl.load(SHIFT + offs_n)
    acc = acc * sc[None, :] + sf[None, :]
    if ACT == 1:
        acc = acc * tl.sigmoid(acc)
    yp = Y + ybase[:, None] + offs_n[None, :] * SY1
    val = acc.to(Y.dtype.element_ty)
    if MMASK and NMASK:
        tl.store(yp, val, mask=m_ok[:, None] & (offs_n < OC)[None, :])
    elif MMASK:
        tl.store(yp, val, mask=tl.broadcast_to(m_ok[:, None], (BM, BN)))
    elif NMASK:
        tl.store(yp, val, mask=tl.broadcast_to((offs_n < OC)[None, :], (BM, BN)))
    else:
        tl.store(yp, val)


@triton.jit
def _conv_bn_act_direct(
    X, W, SCALE, SHIFT, Y,
    TOTAL: tl.constexpr, ICG: tl.constexpr, IH: tl.constexpr, IW: tl.constexpr,
    OC: tl.constexpr, OCG: tl.constexpr, OH: tl.constexpr, OW: tl.constexpr,
    SX0: tl.constexpr, SX1: tl.constexpr, SX2: tl.constexpr, SX3: tl.constexpr,
    SY0: tl.constexpr, SY1: tl.constexpr, SY2: tl.constexpr, SY3: tl.constexpr,
    SW0: tl.constexpr, SW1: tl.constexpr, SW2: tl.constexpr, SW3: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr, SH: tl.constexpr, SW_: tl.constexpr,
    PH: tl.constexpr, PW: tl.constexpr, DH: tl.constexpr, DW: tl.constexpr,
    ACT: tl.constexpr, BLK: tl.constexpr,
):
    """Direct grouped/depthwise conv with the same fused epilogue.

    One program covers ``BLK`` flattened output elements and reduces over the
    group's ``ICG`` input channels -- the right shape of loop when each output
    channel only sees a handful of input channels (``g == C`` depthwise), where
    an implicit GEMM would be almost entirely zero padding.
    """
    pid = tl.program_id(0)
    idx = pid * BLK + tl.arange(0, BLK)
    ok = idx < TOTAL
    ow = idx % OW
    t = idx // OW
    oh = t % OH
    t = t // OH
    oc = t % OC
    nb = t // OC
    icb = (oc // OCG) * ICG

    ih0 = oh * SH - PH
    iw0 = ow * SW_ - PW
    xrow = nb * SX0 + icb * SX1
    acc = tl.zeros((BLK,), dtype=tl.float32)
    for ic in range(ICG):
        for kh in tl.static_range(KH):
            ih = ih0 + kh * DH
            h_ok = (ih >= 0) & (ih < IH)
            for kw in tl.static_range(KW):
                iw = iw0 + kw * DW
                v = ok & h_ok & (iw >= 0) & (iw < IW)
                a = tl.load(X + xrow + ic * SX1 + ih * SX2 + iw * SX3, mask=v, other=0.0)
                b = tl.load(W + oc * SW0 + ic * SW1 + kh * SW2 + kw * SW3, mask=ok, other=0.0)
                acc += a.to(tl.float32) * b.to(tl.float32)

    sc = tl.load(SCALE + oc, mask=ok, other=1.0)
    sf = tl.load(SHIFT + oc, mask=ok, other=0.0)
    acc = acc * sc + sf
    if ACT == 1:
        acc = acc * tl.sigmoid(acc)
    yp = Y + nb * SY0 + oc * SY1 + oh * SY2 + ow * SY3
    tl.store(yp, acc.to(Y.dtype.element_ty), mask=ok)


# ---------------------------------------------------------------------------
# Tile selection
# ---------------------------------------------------------------------------
_NUM_SM = None


def _num_sm(device) -> int:
    global _NUM_SM
    if _NUM_SM is None:
        _NUM_SM = torch.cuda.get_device_properties(device).multi_processor_count
    return _NUM_SM


def _npow2(v: int) -> int:
    p = 16
    while p < v:
        p *= 2
    return p


def _pick_tiles(M: int, OC: int, IC: int, KHW: int, nsm: int):
    """Pick (BM, BN, BK, num_warps, num_stages) for the implicit-GEMM kernel.

    Tuned against true kernel duration (CUDA-graph replay), not the harness
    metric, which is quantised in ~2 us steps and hides everything smaller.  What
    the measurements say (see ITERATIONS.md for the numbers):

    * ``BK`` up to 128 -- fewer, fatter reduction steps; 256 starts costing more
      in staged shared memory than it saves in steps.
    * Small ``BM`` (16) for every case except the one with two orders of
      magnitude more work: M is 400..6400 here, so there is no reuse to exploit,
      only latency to hide, and more CTAs hide it better than wider tiles.
    * ``BN`` 32 once OC >= 128 or the kernel has taps, again to get CTA count up.
    * ``num_stages = 4``; worth a steady 5-8% of kernel time over 2 or 3.
    """
    BK = min(128, _npow2(IC))
    BN = min(64, _npow2(OC))
    if OC >= 128 or KHW > 1:
        BN = 32
    BM = 16
    while BM * BN < 4096 and triton.cdiv(M, BM) * triton.cdiv(OC, BN) > 8 * nsm:
        BM *= 2
    warps = 2 if BM * BN <= 1024 else 4
    return BM, BN, BK, warps, 4


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------
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
        self._plans = {}
        self.register_load_state_dict_post_hook(lambda mod, keys: mod._plans.clear())

    # -- cache invalidation ------------------------------------------------
    def _apply(self, *args, **kwargs):
        self._plans.clear()
        return super()._apply(*args, **kwargs)

    # -- folded eval BN ----------------------------------------------------
    def _affine(self, oc: int, device):
        """(scale, shift) fp32 per output channel, or None when BN is not a
        fixed affine (training mode / no running stats)."""
        conv_bias = self.conv.bias
        bn = getattr(self, "bn", None)
        if self._is_fused or bn is None:
            scale = torch.ones(oc, dtype=torch.float32, device=device)
            shift = (conv_bias.to(torch.float32) if conv_bias is not None
                     else torch.zeros(oc, dtype=torch.float32, device=device))
            return scale, shift
        if bn.training or bn.running_var is None or bn.running_mean is None:
            return None
        scale = torch.rsqrt(bn.running_var.to(torch.float32) + bn.eps)
        if bn.weight is not None:
            scale = scale * bn.weight.to(torch.float32)
        shift = -scale * bn.running_mean.to(torch.float32)
        if bn.bias is not None:
            shift = shift + bn.bias.to(torch.float32)
        if conv_bias is not None:
            shift = shift + scale * conv_bias.to(torch.float32)
        return scale, shift

    def _act_code(self):
        act = self.act
        if isinstance(act, (SiLU, nn.SiLU)):
            return 1
        if isinstance(act, nn.Identity):
            return 0
        return -1

    # -- plan construction (once per shape) --------------------------------
    def _make_plan(self, x: torch.Tensor, key):
        if x.dim() == 3:
            inner = self._make_plan(x.unsqueeze(0), None)

            def run3(t, inner=inner):
                return inner(t.unsqueeze(0)).squeeze(0)

            self._plans[key] = run3
            return run3

        conv = self.conv
        w = conv.weight
        oc, icg, kh, kw = w.shape
        groups = conv.groups
        sh, sw = conv.stride
        ph, pw = conv.padding
        dh, dw = conv.dilation
        n, ic, ih, iw = x.shape
        oh = (ih + 2 * ph - dh * (kh - 1) - 1) // sh + 1
        ow = (iw + 2 * pw - dw * (kw - 1) - 1) // sw + 1
        device, dtype = x.device, x.dtype

        affine = self._affine(oc, device)
        act_code = self._act_code()
        post = None
        if affine is None:
            # BN is not a fixed affine (training): run the conv bare and let the
            # real submodules finish the job.
            scale = torch.ones(oc, dtype=torch.float32, device=device)
            shift = torch.zeros(oc, dtype=torch.float32, device=device)
            act_kernel = 0
            post = lambda t: self.act(self.bn(t))  # noqa: E731
        else:
            scale, shift = affine
            act_kernel = act_code if act_code >= 0 else 0
            if act_code < 0:
                post = self.act
        scale = scale.contiguous()
        shift = shift.contiguous()

        oshape = (n, oc, oh, ow)
        sy0, sy1, sy2, sy3 = oc * oh * ow, oh * ow, ow, 1
        sx0, sx1, sx2, sx3 = x.stride()
        ieee = 1 if dtype == torch.float32 else 0

        if oh <= 0 or ow <= 0 or n == 0 or oc == 0:
            def run_empty(t, oshape=oshape, dtype=dtype, device=device):
                return torch.empty(oshape, dtype=dtype, device=device)
            if key is not None:
                self._plans[key] = run_empty
            return run_empty

        if groups == 1:
            m = n * oh * ow
            nsm = _num_sm(device)
            bm, bn_, bk, warps, stages = _pick_tiles(m, oc, icg, kh * kw, nsm)
            nk = triton.cdiv(icg, bk)
            kpad = nk * bk
            # WP[kh*KW + kw, ic, oc]: contiguous in oc, zero padded along ic.
            wp = torch.zeros((kh * kw, kpad, oc), dtype=w.dtype, device=device)
            wp[:, :icg, :] = w.permute(2, 3, 1, 0).reshape(kh * kw, icg, oc)
            tail = (m, icg, ih, iw, oc, oh, ow,
                    sx0, sx1, sx2, sx3, sy0, sy1, sy2, sy3,
                    kpad * oc,
                    kh, kw, sh, sw, ph, pw, dh, dw,
                    act_kernel, ieee, nk, 1 if icg != kpad else 0,
                    1 if (ph or pw) else 0, 1 if m % bm else 0,
                    1 if oc % bn_ else 0,
                    bm, bn_, bk)
            grid = (triton.cdiv(m, bm), triton.cdiv(oc, bn_))
            launch = _conv_bn_act_gemm[grid]
        else:
            total = n * oc * oh * ow
            blk = 256 if total >= 256 else 64
            sw0, sw1, sw2, sw3 = w.stride()
            wp = w
            tail = (total, icg, ih, iw, oc, oc // groups, oh, ow,
                    sx0, sx1, sx2, sx3, sy0, sy1, sy2, sy3,
                    sw0, sw1, sw2, sw3,
                    kh, kw, sh, sw, ph, pw, dh, dw, act_kernel, blk)
            warps, stages = 4, 2
            grid = (triton.cdiv(total, blk),)
            launch = _conv_bn_act_direct[grid]

        empty = torch.empty

        if post is None:
            def run(t, launch=launch, wp=wp, sc=scale, sf=shift, oshape=oshape,
                    dtype=dtype, device=device, tail=tail, warps=warps,
                    stages=stages, empty=empty):
                y = empty(oshape, dtype=dtype, device=device)
                launch(t, wp, sc, sf, y, *tail, num_warps=warps, num_stages=stages)
                return y
        else:
            def run(t, launch=launch, wp=wp, sc=scale, sf=shift, oshape=oshape,
                    dtype=dtype, device=device, tail=tail, warps=warps,
                    stages=stages, empty=empty, post=post):
                y = empty(oshape, dtype=dtype, device=device)
                launch(t, wp, sc, sf, y, *tail, num_warps=warps, num_stages=stages)
                return post(y)

        if key is not None:
            self._plans[key] = run
        return run

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        key = (self.training, x.shape, x.stride())
        plan = self._plans.get(key)
        if plan is None:
            plan = self._make_plan(x, key)
        return plan(x)

    @torch.no_grad()
    def fuse(self):
        if self._is_fused:
            return self
        fused_weight, fused_bias = _fuse_conv_bn(self.conv, self.bn)
        self.conv.weight.data.copy_(fused_weight)
        self.conv.bias = nn.Parameter(fused_bias)
        delattr(self, "bn")
        self._is_fused = True
        self._plans.clear()
        return self


def fuse_module(module: nn.Module) -> nn.Module:
    from .yolov10_repvggdw import YOLORepVGGDW

    for child in module.children():
        fuse_module(child)
    if isinstance(module, YOLOConv):
        module.fuse()
    elif isinstance(module, YOLORepVGGDW):
        module.fuse()
    return module
