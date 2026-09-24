"""YOLOv10 CIB (Compact Inverted Block), fused.

The block is five Conv-BN-SiLU stages plus an optional residual. On a B200 the
eager version spends 89 us of GPU time inside a 240 us wall clock across 32 GPU
ops, so the cost is launch dispatch, not arithmetic (~257 MFLOP) or traffic
(~6 MB). Two things follow.

First, BatchNorm in eval is a per-channel affine, and a ``YOLORepVGGDW`` middle
stage collapses its 7x7 and 3x3 branches into a single 7x7 -- exactly what the
baseline's own ``fuse()`` methods do. That takes the block from 32 GPU ops to 4.

The affine is applied to the fp32 accumulator inside the kernels rather than
pre-multiplied into the stored weight. Pre-multiplying is the usual trick and it is
one operation cheaper, but ``gamma / sqrt(var + eps)`` is unbounded: a near-zero
running variance makes the folded weight overflow fp16 while the unfused baseline
stays finite, and the resulting NaN can only be detected by reading the weight back
to the host -- which is forbidden inside ``forward``. Keeping the weight
scale-normalized removes the failure instead of detecting it: what is stored is
bounded by the raw convolution weight, and the unbounded factor only ever meets an
fp32 accumulator. (A ``YOLORepVGGDW``'s two branches have different scales, so the
merged weight is normalized by their per-channel maximum, which becomes the stage's
scale.)

Second, the remaining work is four Triton launches: (dw3 + pw), (dw7), (pw),
(dw3 + residual). A depthwise stage needs a spatial halo, so nothing can fuse
*across* one; a 1x1 pointwise needs no halo and can ride on the depthwise feeding
it. That makes three launches the halo-free minimum -- but three is not the
fastest. A program that ends in a contraction has to hold every input channel, and
for the wide 7x7 stage that is unaffordable twice over: it pins the grid to
(images x pixel tiles), only 100 programs on a 148-SM part, and at the tile size
that grid needs it also drives the kernel into register spilling. Giving that stage
its own launch lets channel chunks go in the grid instead, and its pointwise then
follows separately. Measured with each grouping given its own best tile shape,
three launches cost 1.9x the GPU time and 1.7x the wall latency of four; five
launches match four on wall latency for ~12 us more host dispatch.

See ``profile/cib_v1_triton_4launch/REPORT.md`` for the measurements, including what
this does *not* remove: the host still needs ~65 us to issue a forward against
~39 us of GPU work, so dispatch, not arithmetic, bounds this operator even at 4 ops.

Folding cannot happen in ``__init__``: the benchmark constructs the module, then
loads the real weights into it, so anything derived at construction time would be
stale. Preparation is deferred to the first eval/no-grad forward and cached per
(compute dtype, device, residual state) -- per *compute* dtype, so one instance can
serve fp16 and bf16 calls from separate caches rather than baking one dtype in.

The cache is dropped by ``train()``, by ``_apply()`` (so any ``.to()`` /
``.half()`` / device move), by a load-state-dict post-hook, and by the public
``reset_fused_cache()``. Those cover every route the module itself is told about,
but they are hooks, not a freshness proof: writing to ``conv.weight.data`` in
place, calling ``train()`` on a *child* BatchNorm only, or swapping a submodule
leaves the cache alive and stale. That is the same exposure the baseline's own
``fuse()`` has -- it deletes ``bn`` outright and is equally blind to later mutation
-- and the alternative, hashing this module's 36 parameter and buffer tensors on
every call, was measured at 4.8 us against a 0.5 us guard, so it is not paid for on
a path this short. Callers that mutate weights in place must call
``reset_fused_cache()``. ``self.add`` is different: it is a plain public attribute
rather than a weight, so it is part of the cache key and re-checked every call.

Three tiers serve a call, in order of preference:

1. the fused Triton path, for eval + no-grad 4-D fp16/bf16 CUDA input whose module
   structure and channel count the guard recognized;
2. folded-weight eager ops, for eval + no-grad calls the kernels decline -- other
   dtypes, CPU, ranks and shapes outside the kernels' contract;
3. the unmodified ``self.cv1(x)`` module path, which is the correctness anchor.
   Training uses batch statistics and autograd needs the original parameters, so
   folded weights are simply the wrong computation there; the same path also covers
   any structure the guard does not recognize, and reproduces the baseline's own
   errors for inputs the operator does not accept.
"""

from __future__ import annotations

import dataclasses
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..L1.batch_norm2d import BatchNorm2d
from ..L1.conv2d import Conv2d
from ..L1.silu import SiLU
from .yolov10_conv import YOLOConv
from .yolov10_repvggdw import YOLORepVGGDW

try:
    import triton
    import triton.language as tl

    _HAVE_TRITON = True
except ImportError:  # pragma: no cover - Triton ships with the torch builds here
    _HAVE_TRITON = False


# Tile shapes and warp counts, tuned by measurement on the captured shapes and
# grouped here so a retune is a one-line change. BP is the number of flattened
# pixels a program owns, BH/BW a two-dimensional output tile, BC the channel chunk.
#
# BC also decides where the channel axis lives. A stage that ends in a contraction
# has to hold every input channel, so its channel chunking is a loop and its grid is
# only (images x pixel tiles); a stage that does not can put the channel chunk in
# the grid instead. That distinction matters far more than tile size here: 400
# pixels over 4 images caps a pixel-tiled grid at 100 programs on 148 SMs, and at
# that grid the wide stage also spills registers. Giving the 7x7 stage its own
# launch takes its grid to 100x16, its registers from 255 to 62 with no spilling,
# and its GPU time from ~98 us to ~19 us.
#
# The two contraction kernels need BP and their channel chunk at least 16 for
# `tl.dot`; the depthwise-only kernels have no such floor, so their chunks are free
# to be smaller and the sweep explores below 16.
_DW_PW_BP = 16
_DW_PW_BC = 128
_DW_PW_WARPS = 8
_PW_BP = 32
_PW_BC = 64
_PW_WARPS = 8

# Depthwise-only launches (stages 2 and 4). `_DW_USE_2D` selects between the
# flattened-pixel kernel and the two-dimensionally tiled one; both are correct, and
# the choice is a measured one recorded in the profile report.
_DW_USE_2D = False
_DW_BP = 16
_DW_BC = 16
_DW_WARPS = 4
_DW2D_BH = 4
_DW2D_BW = 32
_DW2D_BC = 8
_DW2D_WARPS = 4

# ``tl.arange`` extents must be powers of two, and ``tl.dot`` wants at least 16
# along every axis, so channel counts are rounded up and the surplus lanes are
# masked off rather than restricting the class to convenient channel counts.
_MIN_DOT_BLOCK = 16

# Offsets within one image are int32 in the kernels (see `_fused_forward`).
_INT32_MAX = 2 ** 31

# Compute dtypes the kernels serve, and the ones the fold may narrow between.
_HALF = (torch.float16, torch.bfloat16)


def _pow2(n: int) -> int:
    return 1 << max(0, n - 1).bit_length()


def _dot_block(n: int) -> int:
    """Block extent for an axis that feeds ``tl.dot``."""
    return max(_MIN_DOT_BLOCK, _pow2(n))


if _HAVE_TRITON:

    @triton.jit
    def _dw_silu_chunk(
        X, WDW, SDW, BDW, base, c0, prow, pcol, pmask,
        sc, sh, sw, H, W,
        C: tl.constexpr, BC: tl.constexpr,
        KH: tl.constexpr, KW: tl.constexpr,
        PH: tl.constexpr, PW: tl.constexpr, BP: tl.constexpr,
    ):
        """SiLU(depthwise KxK, then the BatchNorm affine) for one (channel chunk,
        pixel tile) block, returned as ``[BC, BP]`` fp32.

        The pixel axis is last so both the loads and the eventual store run along
        the fastest-varying axis of an NCHW tensor; the depthwise weight contributes
        one per-channel scalar per tap, broadcast over pixels.

        Input strides are arguments, never assumed: the benchmark's timed run feeds
        a tensor with gaps between batch elements while its correctness rounds feed
        a contiguous clone, so a contiguity assumption would pass the check and
        corrupt the measurement.
        """
        cin = c0 + tl.arange(0, BC)
        cmask = cin < C
        acc = tl.zeros([BC, BP], dtype=tl.float32)
        for kh in tl.static_range(KH):
            row = prow + (kh - PH)
            rmask = (row >= 0) & (row < H)
            for kw in tl.static_range(KW):
                col = pcol + (kw - PW)
                keep = pmask & rmask & (col >= 0) & (col < W)
                off = cin[:, None] * sc + row[None, :] * sh + col[None, :] * sw
                v = tl.load(X + base + off, mask=keep[None, :] & cmask[:, None], other=0.0)
                tap = tl.load(WDW + (kh * KW + kw) * C + cin, mask=cmask, other=0.0)
                acc += v.to(tl.float32) * tap.to(tl.float32)[:, None]
        z = (acc * tl.load(SDW + cin, mask=cmask, other=0.0)[:, None]
             + tl.load(BDW + cin, mask=cmask, other=0.0)[:, None])
        return z * tl.sigmoid(z)

    @triton.jit
    def _dw_pw_silu_kernel(
        X, WDW, SDW, BDW, WPW, SPW, BPW, OUT,
        x_sn, x_sc, x_sh, x_sw, H, W, HW, NT,
        CIN: tl.constexpr, CIN_P: tl.constexpr,
        COUT: tl.constexpr, COUT_P: tl.constexpr,
        KH: tl.constexpr, KW: tl.constexpr,
        PH: tl.constexpr, PW: tl.constexpr,
        BP: tl.constexpr, BC: tl.constexpr,
    ):
        """Depthwise KxK + SiLU, then pointwise CIN->COUT + SiLU, into ``OUT``.

        The pointwise contraction runs as an ordinary GEMM K-loop over channel
        chunks, and each chunk's depthwise activations are produced just in time to
        feed it. Only one ``[BC, BP]`` activation tile and one ``[COUT_P, BC]``
        weight tile are live at a time, which is what keeps this off the spill path
        -- the depthwise is per-channel, so chunking it costs nothing.
        """
        pid = tl.program_id(0)
        n = pid // NT
        p = (pid - n * NT) * BP + tl.arange(0, BP)
        pmask = p < HW
        prow = p // W
        pcol = p - prow * W
        base = n.to(tl.int64) * x_sn
        cout = tl.arange(0, COUT_P)
        omask = cout < COUT
        o = tl.zeros([COUT_P, BP], dtype=tl.float32)
        for c0 in range(0, CIN_P, BC):
            t = _dw_silu_chunk(
                X, WDW, SDW, BDW, base, c0, prow, pcol, pmask,
                x_sc, x_sh, x_sw, H, W, CIN, BC, KH, KW, PH, PW, BP,
            )
            cin = c0 + tl.arange(0, BC)
            w = tl.load(
                WPW + cout[:, None] * CIN + cin[None, :],
                mask=omask[:, None] & (cin < CIN)[None, :], other=0.0,
            )
            o = tl.dot(w, t.to(WPW.dtype.element_ty), o)
        o = (o * tl.load(SPW + cout, mask=omask, other=0.0)[:, None]
             + tl.load(BPW + cout, mask=omask, other=0.0)[:, None])
        o *= tl.sigmoid(o)
        tl.store(
            OUT + n.to(tl.int64) * COUT * HW + cout[:, None] * HW + p[None, :],
            o.to(OUT.dtype.element_ty), mask=omask[:, None] & pmask[None, :],
        )

    @triton.jit
    def _pw_silu_kernel(
        X, WPW, SPW, BPW, OUT, H, W, HW, NT,
        CIN: tl.constexpr, CIN_P: tl.constexpr,
        COUT: tl.constexpr, COUT_P: tl.constexpr,
        BP: tl.constexpr, BC: tl.constexpr,
    ):
        """Pointwise CIN->COUT + SiLU on a contiguous intermediate, into ``OUT``.

        Used for the pointwise stage that follows a depthwise the kernels keep
        separate. With no tap loop to interleave, the pixel tile can be wide enough
        to give ``tl.dot`` a reasonable N without the register pressure that made a
        wide tile unaffordable in the fused kernel.
        """
        pid = tl.program_id(0)
        n = pid // NT
        p = (pid - n * NT) * BP + tl.arange(0, BP)
        pmask = p < HW
        cout = tl.arange(0, COUT_P)
        omask = cout < COUT
        xbase = X + n.to(tl.int64) * CIN * HW
        o = tl.zeros([COUT_P, BP], dtype=tl.float32)
        for c0 in range(0, CIN_P, BC):
            cin = c0 + tl.arange(0, BC)
            cmask = cin < CIN
            t = tl.load(xbase + cin[:, None] * HW + p[None, :],
                        mask=cmask[:, None] & pmask[None, :], other=0.0)
            wp = tl.load(WPW + cout[:, None] * CIN + cin[None, :],
                         mask=omask[:, None] & cmask[None, :], other=0.0)
            o = tl.dot(wp, t, o)
        o = (o * tl.load(SPW + cout, mask=omask, other=0.0)[:, None]
             + tl.load(BPW + cout, mask=omask, other=0.0)[:, None])
        o *= tl.sigmoid(o)
        tl.store(
            OUT + n.to(tl.int64) * COUT * HW + cout[:, None] * HW + p[None, :],
            o.to(OUT.dtype.element_ty), mask=omask[:, None] & pmask[None, :],
        )

    @triton.jit
    def _dw_silu_res_kernel(
        X, WDW, SDW, BDW, R, OUT,
        x_sn, x_sc, x_sh, x_sw, r_sn, r_sc, r_sh, r_sw,
        H, W, HW, NT,
        C: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
        PH: tl.constexpr, PW: tl.constexpr,
        BP: tl.constexpr, BC: tl.constexpr, HAS_RES: tl.constexpr,
    ):
        """Depthwise KxK + SiLU, plus the block residual, into ``OUT``.

        With no contraction to serve, the channel chunk moves into the grid: one
        program per (image, pixel tile, channel chunk) spreads the tap loop's
        L1-resident traffic over more SMs, which matters more here than reuse.

        The residual re-reads the block's own input through its own strides, so the
        gapped layout is handled here too.
        """
        pid = tl.program_id(0)
        n = pid // NT
        p = (pid - n * NT) * BP + tl.arange(0, BP)
        c0 = tl.program_id(1) * BC
        pmask = p < HW
        prow = p // W
        pcol = p - prow * W
        o = _dw_silu_chunk(
            X, WDW, SDW, BDW, n.to(tl.int64) * x_sn, c0, prow, pcol, pmask,
            x_sc, x_sh, x_sw, H, W, C, BC, KH, KW, PH, PW, BP,
        )
        c = c0 + tl.arange(0, BC)
        keep = (c < C)[:, None] & pmask[None, :]
        if HAS_RES:
            roff = c[:, None] * r_sc + prow[None, :] * r_sh + pcol[None, :] * r_sw
            o += tl.load(
                R + n.to(tl.int64) * r_sn + roff, mask=keep, other=0.0
            ).to(tl.float32)
        tl.store(
            OUT + n.to(tl.int64) * C * HW + c[:, None] * HW + p[None, :],
            o.to(OUT.dtype.element_ty), mask=keep,
        )

    @triton.jit
    def _dw2d_silu_res_kernel(
        X, WDW, SDW, BDW, R, OUT,
        x_sn, x_sc, x_sh, x_sw, r_sn, r_sc, r_sh, r_sw,
        H, W, HW, NTH, NTW,
        C: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
        PH: tl.constexpr, PW: tl.constexpr,
        BH: tl.constexpr, BW: tl.constexpr, BC: tl.constexpr,
        HAS_RES: tl.constexpr,
    ):
        """The same depthwise stage over a two-dimensional output tile.

        The flattened-pixel kernel above recovers ``(row, col)`` from a 1-D tile, so
        a tile of 16 pixels straddles rows and each tap's addresses break into two
        short runs. Owning a ``[BH, BW]`` rectangle instead makes every tap read BH
        runs of BW contiguous elements, which is the widest request an NCHW row
        allows. The plan calls for this once the stage costs more than a few
        microseconds; which of the two ships is decided by measurement.

        It does not reduce the tap re-read: Triton cannot slice a shifted window out
        of a register tile, so the halo is re-read per tap from L1 (which it hits
        ~97% of the time) rather than loaded once.
        """
        pid = tl.program_id(0)
        tiles = NTH * NTW
        n = pid // tiles
        t = pid - n * tiles
        th = t // NTW
        tw = t - th * NTW
        c0 = tl.program_id(1) * BC

        rows = th * BH + tl.arange(0, BH)
        cols = tw * BW + tl.arange(0, BW)
        cin = c0 + tl.arange(0, BC)
        cmask = cin < C
        keep = cmask[:, None, None] & (rows < H)[None, :, None] & (cols < W)[None, None, :]

        base = n.to(tl.int64) * x_sn
        acc = tl.zeros([BC, BH, BW], dtype=tl.float32)
        for kh in tl.static_range(KH):
            rr = rows + (kh - PH)
            rmask = ((rr >= 0) & (rr < H))[None, :, None]
            for kw in tl.static_range(KW):
                cc = cols + (kw - PW)
                m = (cmask[:, None, None] & rmask
                     & ((cc >= 0) & (cc < W))[None, None, :])
                off = (cin[:, None, None] * x_sc + rr[None, :, None] * x_sh
                       + cc[None, None, :] * x_sw)
                v = tl.load(X + base + off, mask=m, other=0.0)
                tap = tl.load(WDW + (kh * KW + kw) * C + cin, mask=cmask, other=0.0)
                acc += v.to(tl.float32) * tap.to(tl.float32)[:, None, None]
        z = (acc * tl.load(SDW + cin, mask=cmask, other=0.0)[:, None, None]
             + tl.load(BDW + cin, mask=cmask, other=0.0)[:, None, None])
        o = z * tl.sigmoid(z)
        if HAS_RES:
            roff = (cin[:, None, None] * r_sc + rows[None, :, None] * r_sh
                    + cols[None, None, :] * r_sw)
            o += tl.load(
                R + n.to(tl.int64) * r_sn + roff, mask=keep, other=0.0
            ).to(tl.float32)
        ooff = (cin[:, None, None] * HW + rows[None, :, None] * W
                + cols[None, None, :])
        tl.store(OUT + n.to(tl.int64) * C * HW + ooff,
                 o.to(OUT.dtype.element_ty), mask=keep)


@dataclasses.dataclass(frozen=True)
class _Stage:
    """One stage's convolution weight plus the BatchNorm affine that follows it.

    ``weight``/``packed`` hold the *scale-normalized* weight -- bounded by the raw
    convolution weight, so representable in any compute dtype -- and ``scale`` is
    the per-output-channel factor the kernels apply to their fp32 accumulator. See
    the module docstring for why the scale is not folded into the weight.
    """

    depthwise: bool
    cin: int
    cout: int
    kh: int
    kw: int
    ph: int
    pw: int
    groups: int
    weight: torch.Tensor        # normalized, conv layout, compute dtype
    packed: torch.Tensor        # normalized, kernel layout: dw [KH*KW, C], pw [COUT, CIN]
    scale32: torch.Tensor       # [cout] fp32, for the kernels
    bias32: torch.Tensor        # [cout] fp32, for the kernels
    affine_scale: torch.Tensor  # [1, cout, 1, 1] compute dtype, for the eager tier
    affine_bias: torch.Tensor   # [1, cout, 1, 1] compute dtype, for the eager tier


@dataclasses.dataclass(frozen=True)
class _Plan:
    """Everything the fast paths need, valid for one (dtype, device, add) key.

    ``stages`` is empty when the module structure was not recognized, or when the
    weights cannot faithfully be folded into this input's dtype, which sends every
    call to the verbatim module path.
    """

    dtype: torch.dtype
    device: torch.device
    stages: tuple[_Stage, ...]
    fused: bool                 # the Triton path applies
    add: bool                   # residual state this plan's launch constants assume
    # Precomputed constexpr tails, one per launch, so a call only splats a tuple.
    c_dw_pw: tuple              # stages 0+1 -> _dw_pw_silu_kernel
    c_mid: tuple                # stage 2    -> the selected depthwise kernel
    c_pw: tuple                 # stage 3    -> _pw_silu_kernel
    c_res: tuple                # stage 4    -> the selected depthwise kernel, with residual
    mid_chunks: int             # channel chunks in the grid of the two depthwise launches
    res_chunks: int


def _fold(conv: Conv2d, bn: BatchNorm2d | None):
    """Split an eval Conv-BN into ``(weight, scale, bias)``.

    The stage computes ``conv(x, weight) * scale + bias``. ``weight`` is the raw
    convolution weight and ``scale`` is ``gamma / sqrt(var + eps)``, so nothing
    unbounded is ever stored in the weight; the algebra is otherwise the baseline's
    ``_fuse_conv_bn``. Scale and bias come back in fp32.
    """
    w = conv.weight.detach().float()
    n = w.shape[0]
    if bn is None:                      # already fused by the baseline's fuse()
        b = (conv.bias.detach().float() if conv.bias is not None
             else torch.zeros(n, dtype=torch.float32, device=w.device))
        return w, torch.ones(n, dtype=torch.float32, device=w.device), b
    scale = torch.rsqrt(bn.running_var.detach().float() + bn.eps)
    if bn.weight is not None:
        scale = scale * bn.weight.detach().float()
    b = -bn.running_mean.detach().float() * scale
    if bn.bias is not None:
        b = b + bn.bias.detach().float()
    if conv.bias is not None:
        b = b + conv.bias.detach().float() * scale
    return w, scale, b


def _conv_geometry(conv: Conv2d) -> tuple[int, int, int, int, int, int, int] | None:
    """``(cin, cout, kh, kw, ph, pw, groups)`` if the convolution preserves the
    spatial size at unit stride and dilation, else None."""
    if not isinstance(conv, Conv2d) or conv.weight.dim() != 4:
        return None
    if tuple(conv.stride) != (1, 1) or tuple(conv.dilation) != (1, 1):
        return None
    cout, cin_per_group, kh, kw = conv.weight.shape
    ph, pw = (int(v) for v in conv.padding)
    # Same-size output is what lets the stages chain and the residual line up.
    if 2 * ph != kh - 1 or 2 * pw != kw - 1:
        return None
    return cin_per_group * conv.groups, cout, kh, kw, ph, pw, conv.groups


def _folded_branch(yc: YOLOConv, want_act) -> tuple | None:
    """Fold one ``YOLOConv``, checking it is the shape of block this file knows.

    ``want_act`` is the activation type required after the convolution: ``SiLU`` for
    a normal stage, ``nn.Identity`` for a ``YOLORepVGGDW`` branch (which is built
    with ``act=False`` and activates only after the branches are summed).
    """
    if not isinstance(yc, YOLOConv) or not isinstance(yc.act, want_act):
        return None
    bn = getattr(yc, "bn", None)
    if bn is not None:
        # An eval BatchNorm only reduces to an affine when it is actually using its
        # running statistics.
        if not isinstance(bn, BatchNorm2d) or bn.training or not bn.track_running_stats:
            return None
        if bn.running_mean is None or bn.running_var is None:
            return None
    geom = _conv_geometry(yc.conv)
    if geom is None:
        return None
    return (geom, *_fold(yc.conv, bn))


def _stage_from_conv(yc: YOLOConv, dtype: torch.dtype, depthwise: bool) -> _Stage | None:
    got = _folded_branch(yc, SiLU)
    if got is None:
        return None
    geom, w, scale, bias = got
    return _make_stage(depthwise, geom, w, scale, bias, dtype)


def _stage_from_repvggdw(m: YOLORepVGGDW, dtype: torch.dtype) -> _Stage | None:
    """Collapse a ``YOLORepVGGDW`` middle stage into a single depthwise convolution.

    Mirrors ``YOLORepVGGDW.fuse()``: add the smaller kernel into the centre of the
    larger one and sum the biases. The two branches have different per-channel
    scales, so unlike a single-branch stage the merge has to happen in scaled space;
    the result is then normalized by the per-channel larger of the two scales, which
    keeps the stored weight bounded by ``|w_big| + |w_small|``.
    """
    if not isinstance(m, YOLORepVGGDW) or not isinstance(m.act, SiLU):
        return None
    big = _folded_branch(m.conv, nn.Identity)
    if big is None:
        return None
    geom, w, scale, bias = big
    small = getattr(m, "conv1", None)
    if small is not None:
        got = _folded_branch(small, nn.Identity)
        if got is None:
            return None
        geom_s, ws, scale_s, bias_s = got
        cin, cout, kh, kw, _, _, groups = geom
        cin_s, cout_s, kh_s, kw_s, _, _, groups_s = geom_s
        if (cin_s, cout_s, groups_s) != (cin, cout, groups):
            return None
        if kh_s > kh or kw_s > kw or (kh - kh_s) % 2 or (kw - kw_s) % 2:
            return None
        ref = torch.maximum(scale.abs(), scale_s.abs())
        ref = torch.where(ref > 0, ref, torch.ones_like(ref))
        dh, dwd = (kh - kh_s) // 2, (kw - kw_s) // 2
        shape = (-1, 1, 1, 1)
        w = (w * (scale / ref).reshape(shape)
             + F.pad(ws * (scale_s / ref).reshape(shape), [dwd, dwd, dh, dh]))
        scale = ref
        bias = bias + bias_s
    return _make_stage(True, geom, w, scale, bias, dtype)


def _make_stage(depthwise, geom, w, scale, bias, dtype) -> _Stage | None:
    cin, cout, kh, kw, ph, pw, groups = geom
    if depthwise:
        if groups != cin or cin != cout:
            return None
        packed = w.reshape(cout, kh * kw).t().contiguous().to(dtype)
    else:
        if groups != 1 or (kh, kw) != (1, 1):
            return None
        packed = w.reshape(cout, cin).contiguous().to(dtype)
    affine = (1, cout, 1, 1)
    return _Stage(
        depthwise=depthwise, cin=cin, cout=cout, kh=kh, kw=kw, ph=ph, pw=pw,
        groups=groups, weight=w.to(dtype), packed=packed,
        scale32=scale.contiguous(), bias32=bias.contiguous(),
        affine_scale=scale.reshape(affine).to(dtype),
        affine_bias=bias.reshape(affine).to(dtype),
    )


class YOLOCIB(nn.Module):
    def __init__(self, c1: int, c2: int, shortcut: bool = True, e: float = 0.5, lk: bool = False):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = nn.Sequential(
            YOLOConv(c1, c1, 3, g=c1),
            YOLOConv(c1, 2 * c_, 1),
            YOLOConv(2 * c_, 2 * c_, 3, g=2 * c_) if not lk else YOLORepVGGDW(2 * c_),
            YOLOConv(2 * c_, c2, 1),
            YOLOConv(c2, c2, 3, g=c2),
        )
        self.add = shortcut and c1 == c2
        # No derived tensors here: the real weights arrive after construction.
        self._plans: dict[tuple, _Plan] = {}
        self._plan: _Plan | None = None
        # An unbound method rather than a closure, so the module stays picklable.
        self.register_load_state_dict_post_hook(type(self)._reset_cache_hook)

    # -- cache lifecycle ---------------------------------------------------
    @staticmethod
    def _reset_cache_hook(module, incompatible_keys):  # noqa: ARG004
        module.reset_fused_cache()

    def reset_fused_cache(self) -> None:
        """Drop the folded weights. The next eval/no-grad forward rebuilds them."""
        self._plans.clear()
        self._plan = None

    def train(self, mode: bool = True):
        self.reset_fused_cache()
        return super().train(mode)

    def _apply(self, *args, **kwargs):
        out = super()._apply(*args, **kwargs)
        self.reset_fused_cache()
        return out

    # -- preparation -------------------------------------------------------
    def _foldable_for(self, x: torch.Tensor) -> bool:
        """Whether folded weights in ``x``'s dtype faithfully stand in for the tree.

        Every source tensor must already live on ``x``'s device. For a half compute
        dtype the fold may narrow from the other half format -- that is what lets
        one instance keep a separate cache for fp16 and for bf16 -- but nothing
        wider is narrowed silently: an fp32 parameter against an fp16 input keeps
        reaching the module path, which raises exactly as the baseline does. Buffers
        are exempt from the dtype rule because the benchmark's own recipe leaves
        BatchNorm statistics in fp32 while casting parameters to fp16, and the fold
        reads them in fp32 regardless.
        """
        params = list(self.cv1.parameters())
        for t in itertools.chain(params, self.cv1.buffers()):
            if t.device != x.device:
                return False
        kinds = {p.dtype for p in params if p.is_floating_point()}
        allowed = set(_HALF) if x.dtype in _HALF else {x.dtype}
        return kinds <= allowed

    def _build_plan(self, dtype: torch.dtype, device: torch.device) -> _Plan:
        """Fold the weights and evaluate every structural guard, once.

        Everything expensive or Python-heavy happens here, so a call on the fast
        path only has to check the module mode, the plan key and a flag. Nothing
        here reads a device tensor back to the host: the caller is ``forward``, and
        a host-blocking read there would serialize against the timed stream.
        """
        stages: tuple[_Stage, ...] = ()
        seq = self.cv1
        if isinstance(seq, nn.Sequential) and len(seq) == 5:
            built = [
                _stage_from_conv(seq[0], dtype, depthwise=True),
                _stage_from_conv(seq[1], dtype, depthwise=False),
                _stage_from_repvggdw(seq[2], dtype) if isinstance(seq[2], YOLORepVGGDW)
                else _stage_from_conv(seq[2], dtype, depthwise=True),
                _stage_from_conv(seq[3], dtype, depthwise=False),
                _stage_from_conv(seq[4], dtype, depthwise=True),
            ]
            if all(s is not None for s in built):
                chained = all(built[i].cout == built[i + 1].cin for i in range(4))
                # The residual only lines up when the block is channel-preserving.
                if chained and (not self.add or built[0].cin == built[4].cout):
                    stages = tuple(built)

        fused = bool(stages) and _HAVE_TRITON and device.type == "cuda" and dtype in _HALF
        c_dw_pw = c_mid = c_pw = c_res = ()
        mid_chunks = res_chunks = 0
        if fused:
            c_dw_pw = _dw_pw_consts(stages[0], stages[1])
            c_mid, mid_chunks = _dw_consts(stages[2], has_res=False)
            c_pw = _pw_consts(stages[3])
            c_res, res_chunks = _dw_consts(stages[4], has_res=self.add)
        plan = _Plan(dtype, device, stages, fused, self.add, c_dw_pw, c_mid, c_pw,
                     c_res, mid_chunks, res_chunks)
        self._plans[(dtype, device, self.add)] = plan
        self._plan = plan
        return plan

    def _plan_for(self, x: torch.Tensor) -> _Plan:
        key = (x.dtype, x.device, self.add)
        plan = self._plans.get(key)
        if plan is None:
            if self._foldable_for(x):
                plan = self._build_plan(x.dtype, x.device)
            else:
                plan = _Plan(x.dtype, x.device, (), False, self.add,
                             (), (), (), (), 0, 0)
                self._plans[key] = plan
        self._plan = plan
        return plan

    # -- dispatch ----------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # `plan.fused` already implies a CUDA plan, and a plan is only reused for a
        # matching device, so there is no separate `x.is_cuda` test to pay for. The
        # rank test is not an optimization: BatchNorm2d reads dim 1 as the channel
        # axis, so anything but NCHW is not this operator and has to reach the
        # module path, which rejects it exactly as the baseline does.
        if not (self.training or torch.is_grad_enabled() or torch.is_autocast_enabled()):
            plan = self._plan
            if (plan is None or x.dtype is not plan.dtype or x.device != plan.device
                    or plan.add is not self.add):
                plan = self._plan_for(x)
            if plan.fused and x.dim() == 4:
                out = _fused_forward(x, plan)
                if out is not None:
                    return out
            if plan.stages and x.dim() == 4 and not torch.is_autocast_enabled("cpu"):
                y = _folded_block(x, plan.stages)
                return x + y if self.add else y
        y = self.cv1(x)
        return x + y if self.add else y


def _dw_pw_consts(dw: _Stage, pw: _Stage) -> tuple:
    cin_p = _dot_block(dw.cin)
    return (
        dw.cin, cin_p, pw.cout, _dot_block(pw.cout),
        dw.kh, dw.kw, dw.ph, dw.pw, _DW_PW_BP, min(_DW_PW_BC, cin_p),
    )


def _pw_consts(pw: _Stage) -> tuple:
    cin_p = _dot_block(pw.cin)
    return (pw.cin, cin_p, pw.cout, _dot_block(pw.cout), _PW_BP, min(_PW_BC, cin_p))


def _dw_consts(dw: _Stage, has_res: bool) -> tuple[tuple, int]:
    """Constexpr tail plus the number of channel chunks the grid needs.

    Shaped for whichever depthwise kernel `_DW_USE_2D` selects. These kernels have
    no `tl.dot`, so the channel chunk is free to drop below 16.
    """
    if _DW_USE_2D:
        bc = min(_DW2D_BC, _pow2(dw.cin))
        consts = (dw.cin, dw.kh, dw.kw, dw.ph, dw.pw,
                  _DW2D_BH, _DW2D_BW, bc, has_res)
    else:
        bc = min(_DW_BC, _pow2(dw.cin))
        consts = (dw.cin, dw.kh, dw.kw, dw.ph, dw.pw, _DW_BP, bc, has_res)
    return consts, -(-dw.cin // bc)


def _folded_block(x: torch.Tensor, stages: tuple[_Stage, ...]) -> torch.Tensor:
    """The five stages as folded eager ops.

    Applies the BatchNorm affine after the convolution, the same way the kernels do,
    so the two fast tiers share one semantics -- including how they overflow, which
    is then the baseline's behaviour rather than a NaN from an overflowed weight.
    """
    y = x
    for s in stages:
        y = F.conv2d(y, s.weight, None, padding=(s.ph, s.pw), groups=s.groups)
        y = F.silu(torch.addcmul(s.affine_bias, y, s.affine_scale))
    return y


def _launch_dw(src, stage, res, out, src_strides, res_strides, n, h, w, hw,
               consts, chunks):
    """Issue a depthwise stage through whichever kernel `_DW_USE_2D` selects."""
    if _DW_USE_2D:
        nth = -(-h // _DW2D_BH)
        ntw = -(-w // _DW2D_BW)
        _dw2d_silu_res_kernel[(n * nth * ntw, chunks)](
            src, stage.packed, stage.scale32, stage.bias32, res, out,
            *src_strides, *res_strides, h, w, hw, nth, ntw, *consts,
            num_warps=_DW2D_WARPS,
        )
    else:
        nt = -(-hw // _DW_BP)
        _dw_silu_res_kernel[(n * nt, chunks)](
            src, stage.packed, stage.scale32, stage.bias32, res, out,
            *src_strides, *res_strides, h, w, hw, nt, *consts,
            num_warps=_DW_WARPS,
        )


def _fused_forward(x: torch.Tensor, plan: _Plan) -> torch.Tensor | None:
    """Four Triton launches over the prepared stages.

    Every buffer is allocated before the first launch so no framework operation
    lands between the launches.

    Returns None for a 4-D input the kernels must not touch, leaving the caller to
    fall back. Two cases matter and neither is hypothetical: a channel count other
    than the block's own would be silently truncated (too many) or read out of
    bounds (too few), and an offset span past int32 would wrap. Both are rejected
    here rather than in `forward`, so the per-call guard stays free for the shapes
    that do run, and the eager fallback reproduces the baseline's own error.
    """
    n, c, h, w = x.shape
    s0, s1, s2, s3, s4 = plan.stages
    if c != s0.cin or h <= 0 or w <= 0:
        return None
    xs = x.stride()
    xn, xc, xh, xw = xs
    # Per-image base offsets are computed in int64 in the kernels; the offsets
    # within one image are int32, so only this span has to fit.
    if (c - 1) * abs(xc) + (h - 1) * abs(xh) + (w - 1) * abs(xw) >= _INT32_MAX:
        return None
    hw = h * w
    dtype, device = x.dtype, x.device
    a1 = torch.empty((n, s1.cout, h, w), dtype=dtype, device=device)
    a2 = torch.empty((n, s2.cout, h, w), dtype=dtype, device=device)
    a3 = torch.empty((n, s3.cout, h, w), dtype=dtype, device=device)
    out = torch.empty((n, s4.cout, h, w), dtype=dtype, device=device)
    a1s = (s1.cout * hw, hw, w, 1)
    a3s = (s3.cout * hw, hw, w, 1)
    nt_dw_pw = -(-hw // _DW_PW_BP)
    nt_pw = -(-hw // _PW_BP)

    # stages 0+1: depthwise 3x3 + SiLU, then pointwise + SiLU, reading the input
    # through its own strides (it may be gapped).
    _dw_pw_silu_kernel[(n * nt_dw_pw,)](
        x, s0.packed, s0.scale32, s0.bias32, s1.packed, s1.scale32, s1.bias32, a1,
        xn, xc, xh, xw, h, w, hw, nt_dw_pw, *plan.c_dw_pw, num_warps=_DW_PW_WARPS,
    )
    # stage 2: the wide depthwise, channels spread across the grid.
    _launch_dw(a1, s2, a1, a2, a1s, a1s, n, h, w, hw, plan.c_mid, plan.mid_chunks)
    # stage 3: pointwise on a contiguous intermediate.
    _pw_silu_kernel[(n * nt_pw,)](
        a2, s3.packed, s3.scale32, s3.bias32, a3, h, w, hw, nt_pw,
        *plan.c_pw, num_warps=_PW_WARPS,
    )
    # stage 4: depthwise 3x3 + SiLU + the block residual, which re-reads the
    # original input through its own strides.
    _launch_dw(a3, s4, x, out, a3s, xs, n, h, w, hw, plan.c_res, plan.res_chunks)
    return out
