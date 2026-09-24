"""YOLOv10 C2f and C2fCIB blocks -- fused Triton implementation.

The reference block is a chain of ~14 (C2f, n=1) to ~26 (C2fCIB, lk=True)
eager ops: every ``YOLOConv`` is conv -> BatchNorm -> SiLU, plus a chunk/cat
round trip.  At the captured shapes every tensor is 0.1-5 MB, so essentially
none of that time is arithmetic -- it is per-op launch and drain (~14 us per
eager op on B200).  Everything here is aimed at kernel count and bytes moved.

Strategy
--------
* Fold every BatchNorm into its conv weight/bias once, lazily on the first
  forward (``load_state_dict`` runs after ``__init__``), and pre-fuse
  ``RepVGGDW``'s 7x7 + 3x3 branches into a single 7x7.
* Run the interior in NHWC (``[N*H*W, C]``) so 1x1 convs are plain GEMMs and
  the 3x3/depthwise convs are clean stencils.  The NCHW->NHWC transpose is
  folded into the first GEMM's A-operand addressing and the NHWC->NCHW
  transpose into the last GEMM's store, so neither costs a launch.
* Write each branch straight into its column slice of one ``[M, (2+n)c]``
  buffer, which removes the chunk/cat traffic entirely.
* Bias, SiLU and the shortcut add all live in the GEMM epilogue.
* Record the whole launch list once per (shape, dtype) and replay it as a CUDA
  graph, so no tile-picking, view-slicing or Triton dispatch happens per call.
  The first kernel stays outside the graph and reads the caller's tensor
  directly, which avoids a static-input copy.  Kernels 2..K use programmatic
  dependent launch so each one's index math overlaps its predecessor's drain.

That takes C2f(n=1) to 4 kernels, C2f(n=2) to 6, and C2fCIB(lk) to 7.

* For C2f, go one step further and run the whole block as **one persistent
  megakernel** whose stage boundaries are device-side grid barriers
  (:func:`_mk_c2f`).  The grid is capped by the driver's own occupancy answer so
  every program is co-resident, and the launch is cooperative so the driver --
  not us -- guarantees it.  Worth ~2 us on two of the five C2f shapes and a wash
  on the rest, because a grid barrier costs about what a kernel boundary costs
  (both measured: see ITERATIONS.md).  It loses on C2fCIB, which keeps the
  multi-kernel path; `FKC2F_MEGA=0` disables it everywhere and `=2` forces it on.

Parameter names/structure are identical to the baseline so
``load_state_dict(..., strict=False)`` transfers every weight.

Env knobs (all default to the tuned value) exist only for A/B measurement:
``FKC2F_NOGRAPH``, ``FKC2F_NOPDL``, ``FKC2F_EAGER1``, ``FKC2F_MEGA``,
``FKC2F_MK*`` and the tile overrides.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import triton
import triton.language as tl

try:  # Triton 3.6+: programmatic dependent launch
    from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait
    _HAVE_PDL = True
except ImportError:  # pragma: no cover
    _HAVE_PDL = False

from .yolov10_bottleneck import YOLOBottleneck
from .yolov10_cib import YOLOCIB
from .yolov10_conv import YOLOConv
from .yolov10_repvggdw import YOLORepVGGDW

_NO_GRAPH = bool(int(os.environ.get("FKC2F_NOGRAPH", "0")))
_NSM = 0
_DW_BM = int(os.environ.get("FKC2F_DWBM", "32"))
_DW_BC = int(os.environ.get("FKC2F_DWBC", "16"))
_NS_DW = int(os.environ.get("FKC2F_NSDW", "2"))
_NW_DW = int(os.environ.get("FKC2F_NWDW", "2"))
_USE_PDL = _HAVE_PDL and not int(os.environ.get("FKC2F_NOPDL", "0"))
_ROWPAD = int(os.environ.get("FKC2F_ROWPAD", "1"))
_GRID_MULT = float(os.environ.get("FKC2F_GRIDMULT", "4"))
_EAGER1 = bool(int(os.environ.get("FKC2F_EAGER1", "1")))
_NW_CONV = int(os.environ.get("FKC2F_NWCONV", "0"))  # 0 = pick from K
_GEMM_BN = int(os.environ.get("FKC2F_GBN", "64"))
_GEMM_BM = int(os.environ.get("FKC2F_GBM", "256"))
_GEMM_BM_MIN = int(os.environ.get("FKC2F_GBMMIN", "16"))
_NS_GEMM = int(os.environ.get("FKC2F_NSGEMM", "4"))
_NW_GEMM = int(os.environ.get("FKC2F_NWGEMM", "4"))

_NS_CONV = int(os.environ.get("FKC2F_NSCONV", "4"))
_CONV_BM = int(os.environ.get("FKC2F_CONVBM", "32"))
_CONV_BN = int(os.environ.get("FKC2F_CONVBN", "32"))

# Megakernel: 0 = off (multi-kernel CUDA-graph path), 1 = on where supported.
_MEGA = int(os.environ.get("FKC2F_MEGA", "1"))
_MK_NW = int(os.environ.get("FKC2F_MKNW", "0"))   # 0 = from _tile_conv's rule
_MK_NS = int(os.environ.get("FKC2F_MKNS", "4"))
_MK_WAVES = int(os.environ.get("FKC2F_MKWAVES", "0"))   # 0 = as many as are resident
_MK_COOP = int(os.environ.get("FKC2F_MKCOOP", "1"))



def _round_up(v, m):
    return v if m <= 1 else ((v + m - 1) // m) * m


def _nsm():
    """SM count, resolved on first use (import must not touch CUDA)."""
    global _NSM
    if _NSM == 0:
        try:
            _NSM = torch.cuda.get_device_properties(0).multi_processor_count
        except Exception:
            _NSM = 132
    return _NSM


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------
@triton.jit
def _silu(x):
    return x * tl.sigmoid(x)


@triton.jit
def _t_gemm(
    X, Wt, Bias, Y, Res, tile,
    M, K, N, HW,
    sxr, syr, srr,
    X_NCHW: tl.constexpr, Y_NCHW: tl.constexpr,
    HAS_RES: tl.constexpr, ACT: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    """One [BM, BN] output tile of a 1x1 conv + bias (+ residual) (+ SiLU).

    ``Wt`` is the transposed weight ``[K, N]``.  ``X_NCHW``/``Y_NCHW`` switch the
    A-operand / store addressing between NCHW (``[B, C, HW]``) and NHWC
    (``[M, C]`` with row stride ``sxr``/``syr``).  ``tile`` is a linear tile id,
    decomposed M-major so the ordering matches a ``(cdiv(M,BM), cdiv(N,BN))``
    grid -- the kernel wrapper and the megakernel therefore visit tiles in the
    same order.
    """
    gm = tl.cdiv(M, BM)
    pid_m = tile % gm
    pid_n = tile // gm
    offm = pid_m * BM + tl.arange(0, BM)
    offn = pid_n * BN + tl.arange(0, BN)
    mm = offm < M
    nm = offn < N

    b = offm // HW
    r = offm - b * HW
    if X_NCHW:
        xrow = b * (K * HW) + r
        xks = HW
    else:
        xrow = offm * sxr
        xks = 1

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in tl.range(0, K, BK):
        offk = k0 + tl.arange(0, BK)
        if EVEN_K:
            a = tl.load(X + xrow[:, None] + (offk * xks)[None, :],
                        mask=mm[:, None], other=0.0)
            w = tl.load(Wt + offk[:, None] * N + offn[None, :],
                        mask=nm[None, :], other=0.0)
        else:
            km = offk < K
            a = tl.load(X + xrow[:, None] + (offk * xks)[None, :],
                        mask=mm[:, None] & km[None, :], other=0.0)
            w = tl.load(Wt + offk[:, None] * N + offn[None, :],
                        mask=km[:, None] & nm[None, :], other=0.0)
        acc = tl.dot(a, w, acc)

    acc += tl.load(Bias + offn, mask=nm, other=0.0).to(tl.float32)[None, :]
    if ACT:
        acc = _silu(acc)
    if HAS_RES:
        acc += tl.load(Res + offm[:, None] * srr + offn[None, :],
                       mask=mm[:, None] & nm[None, :], other=0.0).to(tl.float32)
    out = acc.to(Y.dtype.element_ty)

    if Y_NCHW:
        yoff = b[:, None] * (N * HW) + offn[None, :] * HW + r[:, None]
    else:
        yoff = offm[:, None] * syr + offn[None, :]
    tl.store(Y + yoff, out, mask=mm[:, None] & nm[None, :])


@triton.jit
def _k_gemm1x1(
    X, Wt, Bias, Y, Res,
    M, K, N, HW,
    sxr, syr, srr,
    X_NCHW: tl.constexpr, Y_NCHW: tl.constexpr,
    HAS_RES: tl.constexpr, ACT: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    EVEN_K: tl.constexpr,
    PDL: tl.constexpr,
):
    """1x1 conv + bias (+ residual) (+ SiLU), one tile per program."""
    if PDL:
        gdc_wait()
    tile = tl.program_id(0) + tl.num_programs(0) * tl.program_id(1)
    _t_gemm(X, Wt, Bias, Y, Res, tile, M, K, N, HW, sxr, syr, srr,
            X_NCHW, Y_NCHW, HAS_RES, ACT, BM, BN, BK, EVEN_K)
    if PDL:
        gdc_launch_dependents()


@triton.jit
def _t_conv3x3(
    X, Wt, Bias, Y, Res, tile,
    M, K, N, H, W, HW,
    sxr, syr, srr,
    HAS_RES: tl.constexpr, ACT: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    HAS_KMASK: tl.constexpr,
):
    """One [BM, BN] output tile of a dense 3x3 conv (pad 1), NHWC in and out.

    The 9 taps are a single ``tl.range`` loop (not ``static_range``) with the
    whole ``K`` in one dot, so Triton software-pipelines the tap sequence and
    prefetches the next tap's tile while the current MMA runs.  That pipelining
    -- not the MMA shape -- is what these tiny latency-bound tiles are limited
    by: on B200 it is worth ~2x over the fully-unrolled form (re-measured in
    round 2 over a 27-config tile sweep: unrolled is 27% slower).  ``Wt`` is
    ``[9, K, N]`` (tap-major), so each weight tile is contiguous along N.
    """
    gm = tl.cdiv(M, BM)
    pid_m = tile % gm
    pid_n = tile // gm
    offm = pid_m * BM + tl.arange(0, BM)
    offn = pid_n * BN + tl.arange(0, BN)
    offk = tl.arange(0, BK)
    mm = offm < M
    nm = offn < N

    b = offm // HW
    r = offm - b * HW
    h = r // W
    w = r - h * W

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for tap in tl.range(0, 9):
        kh = tap // 3
        ih = h + kh - 1
        iw = w + (tap - kh * 3) - 1
        rowok = mm & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
        xrow = (b * HW + ih * W + iw) * sxr
        if HAS_KMASK:
            km = offk < K
            a = tl.load(X + xrow[:, None] + offk[None, :],
                        mask=rowok[:, None] & km[None, :], other=0.0)
            wv = tl.load(Wt + tap * (K * N) + offk[:, None] * N + offn[None, :],
                         mask=km[:, None] & nm[None, :], other=0.0)
        else:
            a = tl.load(X + xrow[:, None] + offk[None, :],
                        mask=rowok[:, None], other=0.0)
            wv = tl.load(Wt + tap * (K * N) + offk[:, None] * N + offn[None, :],
                         mask=nm[None, :], other=0.0)
        acc = tl.dot(a, wv, acc)

    acc += tl.load(Bias + offn, mask=nm, other=0.0).to(tl.float32)[None, :]
    if ACT:
        acc = _silu(acc)
    if HAS_RES:
        acc += tl.load(Res + offm[:, None] * srr + offn[None, :],
                       mask=mm[:, None] & nm[None, :], other=0.0).to(tl.float32)
    tl.store(Y + offm[:, None] * syr + offn[None, :],
             acc.to(Y.dtype.element_ty), mask=mm[:, None] & nm[None, :])


@triton.jit
def _k_conv3x3(
    X, Wt, Bias, Y, Res,
    M, K, N, H, W, HW,
    sxr, syr, srr,
    HAS_RES: tl.constexpr, ACT: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    HAS_KMASK: tl.constexpr,
    PDL: tl.constexpr,
):
    """Dense 3x3 conv (pad 1) + bias (+ SiLU) (+ residual), one tile per program."""
    if PDL:
        gdc_wait()
    tile = tl.program_id(0) + tl.num_programs(0) * tl.program_id(1)
    _t_conv3x3(X, Wt, Bias, Y, Res, tile, M, K, N, H, W, HW, sxr, syr, srr,
               HAS_RES, ACT, BM, BN, BK, HAS_KMASK)
    if PDL:
        gdc_launch_dependents()


@triton.jit
def _t_dw(
    X, Wt, Bias, Y, Res, tile,
    M, C, H, W, HW,
    sxr, syr, srr,
    KS: tl.constexpr, PAD: tl.constexpr,
    HAS_RES: tl.constexpr, ACT: tl.constexpr,
    BM: tl.constexpr, BC: tl.constexpr,
):
    """One [BM, BC] output tile of a depthwise KSxKS conv, NHWC in and out.

    Opposite of :func:`_t_conv3x3`: the tap loop is **fully unrolled**
    (``tl.static_range``).  There is no MMA here, so a tap is just a load plus an
    FMA and every tap's load is independent -- unrolled, all ``KS*KS`` of them can
    be in flight at once and the dependent-latency chain is one memory round trip
    instead of ``KS*KS/(num_stages-1)`` of them.  At KS=7 that is worth 1.15x on
    CIB ``[4,384,20,20]`` and 1.11x on ``[1,384,20,20]`` end to end; the same
    change applied to ``_t_conv3x3`` is 27% *slower*, because there the pipeliner
    is overlapping shared-memory staging for the dot and unrolling defeats it.
    ``Wt`` is ``[KS*KS, C]``.
    """
    gm = tl.cdiv(M, BM)
    pid_m = tile % gm
    pid_c = tile // gm
    offm = pid_m * BM + tl.arange(0, BM)
    offc = pid_c * BC + tl.arange(0, BC)
    mm = offm < M
    cm = offc < C

    b = offm // HW
    r = offm - b * HW
    h = r // W
    w = r - h * W

    acc = tl.zeros((BM, BC), dtype=tl.float32)
    for tap in tl.static_range(0, KS * KS):
        ih = h + (tap // KS) - PAD
        iw = w + (tap % KS) - PAD
        rowok = mm & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
        xrow = (b * HW + ih * W + iw) * sxr
        a = tl.load(X + xrow[:, None] + offc[None, :],
                    mask=rowok[:, None] & cm[None, :], other=0.0)
        wv = tl.load(Wt + tap * C + offc, mask=cm, other=0.0)
        acc += a.to(tl.float32) * wv.to(tl.float32)[None, :]

    acc += tl.load(Bias + offc, mask=cm, other=0.0).to(tl.float32)[None, :]
    if ACT:
        acc = _silu(acc)
    if HAS_RES:
        acc += tl.load(Res + offm[:, None] * srr + offc[None, :],
                       mask=mm[:, None] & cm[None, :], other=0.0).to(tl.float32)
    tl.store(Y + offm[:, None] * syr + offc[None, :],
             acc.to(Y.dtype.element_ty), mask=mm[:, None] & cm[None, :])


@triton.jit
def _k_dw(
    X, Wt, Bias, Y, Res,
    M, C, H, W, HW,
    sxr, syr, srr,
    KS: tl.constexpr, PAD: tl.constexpr,
    HAS_RES: tl.constexpr, ACT: tl.constexpr,
    BM: tl.constexpr, BC: tl.constexpr,
    PDL: tl.constexpr,
):
    """Depthwise KSxKS conv + bias (+ SiLU) (+ residual), one tile per program."""
    if PDL:
        gdc_wait()
    tile = tl.program_id(0) + tl.num_programs(0) * tl.program_id(1)
    _t_dw(X, Wt, Bias, Y, Res, tile, M, C, H, W, HW, sxr, syr, srr,
          KS, PAD, HAS_RES, ACT, BM, BC)
    if PDL:
        gdc_launch_dependents()


# ---------------------------------------------------------------------------
# BatchNorm folding / weight preparation
# ---------------------------------------------------------------------------
def _fold(conv_mod: YOLOConv):
    """Return ``(weight, bias)`` with the BatchNorm folded in, in fp32."""
    conv = conv_mod.conv
    w = conv.weight.detach().float()
    bn = getattr(conv_mod, "bn", None)
    if bn is None:
        b = conv.bias.detach().float() if conv.bias is not None else torch.zeros(
            w.shape[0], device=w.device, dtype=torch.float32)
        return w, b
    s = bn.weight.detach().float() / torch.sqrt(bn.running_var.detach().float() + bn.eps)
    b = bn.bias.detach().float() - bn.running_mean.detach().float() * s
    if conv.bias is not None:
        b = b + conv.bias.detach().float() * s
    return w * s.view(-1, 1, 1, 1), b


class _Plan:
    """A prepared, BN-folded weight for one Triton kernel invocation."""

    __slots__ = ("kind", "wt", "bias", "cin", "cout", "act", "res")

    def __init__(self, kind, wt, bias, cin, cout, act, res):
        self.kind = kind      # "1x1" | "3x3" | "dw3" | "dw7"
        self.wt = wt
        self.bias = bias
        self.cin = cin
        self.cout = cout
        self.act = act
        self.res = res        # residual source slot name or None


def _plan_1x1(mod: YOLOConv, dtype, act=True, res=None):
    w, b = _fold(mod)
    cout, cin = w.shape[0], w.shape[1]
    wt = w.reshape(cout, cin).t().contiguous().to(dtype)          # [K, N]
    return _Plan("1x1", wt, b.to(torch.float32), cin, cout, act, res)


def _plan_3x3(mod: YOLOConv, dtype, act=True, res=None):
    w, b = _fold(mod)
    cout, cin = w.shape[0], w.shape[1]
    wt = w.permute(2, 3, 1, 0).reshape(9, cin, cout).contiguous().to(dtype)  # [9, K, N]
    return _Plan("3x3", wt, b.to(torch.float32), cin, cout, act, res)


def _plan_dw(mod: YOLOConv, dtype, act=True, res=None):
    w, b = _fold(mod)
    cout, _, ks, _ = w.shape
    wt = w.reshape(cout, ks * ks).t().contiguous().to(dtype)      # [ks*ks, C]
    return _Plan("dw%d" % ks, wt, b.to(torch.float32), cout, cout, act, res)


def _plan_repvggdw(mod: YOLORepVGGDW, dtype, act=True, res=None):
    w7, b7 = _fold(mod.conv)
    w3, b3 = _fold(mod.conv1)
    w = w7 + torch.nn.functional.pad(w3, [2, 2, 2, 2])
    b = b7 + b3
    cout = w.shape[0]
    wt = w.reshape(cout, 49).t().contiguous().to(dtype)
    return _Plan("dw7", wt, b.to(torch.float32), cout, cout, act, res)



# ---------------------------------------------------------------------------
# Launch helpers
#
# Every block schedules its kernels through an ``emit`` callback: the dynamic
# path launches immediately, the static path records ``(kernel, grid, args,
# meta)`` tuples that are replayed verbatim (and captured into a CUDA graph),
# which keeps all tile-picking / view-slicing / index math out of the hot path.
# ---------------------------------------------------------------------------
def _tile_mn(M, N):
    """Pick (BM, BN, num_warps) for a 1x1-conv GEMM."""
    nsm = _nsm()
    bn = max(16, min(_GEMM_BN, N))
    bm = _GEMM_BM
    while bm > _GEMM_BM_MIN and triton.cdiv(M, bm) * triton.cdiv(N, bn) < nsm * _GRID_MULT:
        bm //= 2
    return bm, bn, _NW_GEMM


def _tile_conv(M, N, K):
    """Pick (BM, BN, num_warps) for the 3x3 conv.

    Swept on B200: the pipelined K loop likes many small tiles (BM=BN=32) far
    more than it likes fewer fat ones, and narrow-K shapes prefer 2 warps.
    """
    bn = max(16, min(_CONV_BN, N))
    bm = 64 if K <= 16 else _CONV_BM
    return bm, bn, _NW_CONV or (2 if K <= 32 else 4)


def _tile_dw(M, C):
    nsm = _nsm()
    bc = 16
    while bc < C and bc < _DW_BC:
        bc *= 2
    bm = _DW_BM
    while bm > 8 and triton.cdiv(M, bm) * triton.cdiv(C, bc) < nsm:
        bm //= 2
    return bm, bc, _NW_DW


def _launch(fn, grid, args, meta):
    fn[grid](*args, **meta)


def _em_1x1(emit, p, x, y, M, HW, sxr, syr, x_nchw, y_nchw, res=None, srr=0):
    BM, BN, nw = _tile_mn(M, p.cout)
    K = p.cin
    BK = 64 if K >= 64 else (32 if K >= 32 else 16)
    emit(_k_gemm1x1, (triton.cdiv(M, BM), triton.cdiv(p.cout, BN)),
         (x, p.wt, p.bias, y, res if res is not None else x,
          M, K, p.cout, HW, sxr, syr, srr),
         dict(X_NCHW=x_nchw, Y_NCHW=y_nchw, HAS_RES=res is not None, ACT=p.act,
              BM=BM, BN=BN, BK=BK, EVEN_K=(K % BK == 0), PDL=_USE_PDL,
              num_warps=nw, num_stages=_NS_GEMM, launch_pdl=_USE_PDL))


def _em_3x3(emit, p, x, y, M, H, W, sxr, syr, res=None, srr=0):
    K = p.cin
    BK = 16
    while BK < K:
        BK *= 2
    BM, BN, nw = _tile_conv(M, p.cout, K)
    emit(_k_conv3x3, (triton.cdiv(M, BM), triton.cdiv(p.cout, BN)),
         (x, p.wt, p.bias, y, res if res is not None else x,
          M, K, p.cout, H, W, H * W, sxr, syr, srr),
         dict(HAS_RES=res is not None, ACT=p.act,
              BM=BM, BN=BN, BK=BK, HAS_KMASK=(BK != K), PDL=_USE_PDL,
              num_warps=nw, num_stages=_NS_CONV, launch_pdl=_USE_PDL))


def _em_dw(emit, p, x, y, M, H, W, sxr, syr, res=None, srr=0):
    C = p.cout
    BM, BC, nw = _tile_dw(M, C)
    ks = 3 if p.kind == "dw3" else 7
    emit(_k_dw, (triton.cdiv(M, BM), triton.cdiv(C, BC)),
         (x, p.wt, p.bias, y, res if res is not None else x,
          M, C, H, W, H * W, sxr, syr, srr),
         dict(KS=ks, PAD=ks // 2, HAS_RES=res is not None, ACT=p.act,
              BM=BM, BC=BC, PDL=_USE_PDL,
              num_warps=nw, num_stages=_NS_DW, launch_pdl=_USE_PDL))




# ---------------------------------------------------------------------------
# Megakernel: the whole block as ONE persistent launch, stages separated by
# device-side grid barriers instead of kernel boundaries.
#
# The grid is exactly ``P`` programs, where ``P <= SMs * blocks_per_SM`` is taken
# from the driver's own occupancy query for the compiled kernel, so every program
# is co-resident and the barrier cannot deadlock without cooperative launch.
# Each program strides over the tiles of stage s, hits the barrier, then strides
# over the tiles of stage s+1.  Tiling, staging buffers and parallelism per stage
# are exactly what the multi-kernel path uses -- only the separator changes.
#
# Every shape scalar is a ``tl.constexpr`` so all the tile/index arithmetic and
# every buffer offset folds at compile time (one compile per distinct shape,
# cached on disk by Triton).
# ---------------------------------------------------------------------------
@triton.jit
def _mk_bar(BAR, ph: tl.constexpr, P):
    """Grid-wide barrier, phase ``ph``.  Two int32 slots per phase: an arrival
    counter and a release flag.  Counters are never reset inside a launch (so no
    sense-reversal is needed); the last phase's owner zeroes them all on the way
    out, when every program is provably past every barrier.

    The spin must be an ``acquire`` atomic: a ``volatile`` load gets hoisted and
    livelocks (measured -- 1e6 spins and still going).
    """
    tl.debug_barrier()
    old = tl.atomic_add(BAR + 2 * ph, 1, sem="release", scope="gpu")
    if old == P - 1:
        tl.atomic_xchg(BAR + 2 * ph + 1, 1, sem="release", scope="gpu")
    else:
        done = False
        while not done:
            done = tl.atomic_add(BAR + 2 * ph + 1, 0,
                                 sem="acquire", scope="gpu") != 0
    tl.debug_barrier()


@triton.jit
def _mk_c2f(
    X, OUT, BUF, TMP, W1, B1, W2, B2, WM, BM_, BAR, P,
    M: tl.constexpr, HW: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    C: tl.constexpr, C1: tl.constexpr, C2: tl.constexpr, WIDTH: tl.constexpr,
    NB: tl.constexpr, ADD: tl.constexpr,
    GBM1: tl.constexpr, GBN1: tl.constexpr, GBM2: tl.constexpr, GBN2: tl.constexpr,
    GBK1: tl.constexpr, GBK2: tl.constexpr, EK1: tl.constexpr, EK2: tl.constexpr,
    CBM: tl.constexpr, CBN: tl.constexpr, CBK: tl.constexpr, CKM: tl.constexpr,
    NPH: tl.constexpr,
):
    """cv1 | (conv3x3, conv3x3) x NB | cv2, as one launch."""
    pid = tl.program_id(0)

    nt: tl.constexpr = tl.cdiv(M, GBM1) * tl.cdiv(2 * C, GBN1)
    t = pid
    while t < nt:
        _t_gemm(X, W1, B1, BUF, X, t, M, C1, 2 * C, HW, 0, WIDTH, 0,
                True, False, False, True, GBM1, GBN1, GBK1, EK1)
        t += P

    ntc: tl.constexpr = tl.cdiv(M, CBM) * tl.cdiv(C, CBN)
    for i in tl.static_range(NB):
        _mk_bar(BAR, 2 * i, P)
        t = pid
        while t < ntc:
            _t_conv3x3(BUF + (1 + i) * C, WM + i * (18 * C * C), BM_ + i * (2 * C),
                       TMP + i * (M * C), BUF, t,
                       M, C, C, H, W, HW, WIDTH, C, 0,
                       False, True, CBM, CBN, CBK, CKM)
            t += P
        _mk_bar(BAR, 2 * i + 1, P)
        t = pid
        while t < ntc:
            _t_conv3x3(TMP + i * (M * C), WM + i * (18 * C * C) + 9 * C * C,
                       BM_ + i * (2 * C) + C,
                       BUF + (2 + i) * C, BUF + (1 + i) * C, t,
                       M, C, C, H, W, HW, C, WIDTH, WIDTH,
                       ADD, True, CBM, CBN, CBK, CKM)
            t += P

    _mk_bar(BAR, 2 * NB, P)
    nt2: tl.constexpr = tl.cdiv(M, GBM2) * tl.cdiv(C2, GBN2)
    t = pid
    while t < nt2:
        _t_gemm(BUF, W2, B2, OUT, BUF, t, M, (2 + NB) * C, C2, HW, WIDTH, 0, 0,
                False, True, False, True, GBM2, GBN2, GBK2, EK2)
        t += P
    if pid == 0:
        for j in tl.range(0, 2 * NPH):
            tl.store(BAR + j, 0)


@triton.jit
def _mk_cib(
    X, OUT, BUF, A1, A2, A3, A4,
    W1, B1, W2, B2, WD0, BD0, WG1, BG1, WD2, BD2, WG3, BG3, WD4, BD4,
    BAR, P,
    M: tl.constexpr, HW: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    C: tl.constexpr, C1: tl.constexpr, C2: tl.constexpr, WIDTH: tl.constexpr,
    NB: tl.constexpr, ADD: tl.constexpr, KS2: tl.constexpr,
    GBM1: tl.constexpr, GBN1: tl.constexpr, GBM2: tl.constexpr, GBN2: tl.constexpr,
    GBMA: tl.constexpr, GBNA: tl.constexpr, GBMB: tl.constexpr, GBNB: tl.constexpr,
    GBK1: tl.constexpr, GBK2: tl.constexpr, EK1: tl.constexpr, EK2: tl.constexpr,
    GBKA: tl.constexpr, GBKB: tl.constexpr, EKA: tl.constexpr, EKB: tl.constexpr,
    DBM: tl.constexpr, DBC: tl.constexpr,
    NPH: tl.constexpr,
):
    """cv1 | (dw3, 1x1, dwKS2, 1x1, dw3) x NB | cv2, as one launch."""
    pid = tl.program_id(0)

    nt: tl.constexpr = tl.cdiv(M, GBM1) * tl.cdiv(2 * C, GBN1)
    t = pid
    while t < nt:
        _t_gemm(X, W1, B1, BUF, X, t, M, C1, 2 * C, HW, 0, WIDTH, 0,
                True, False, False, True, GBM1, GBN1, GBK1, EK1)
        t += P

    ntd1: tl.constexpr = tl.cdiv(M, DBM) * tl.cdiv(C, DBC)
    ntd2: tl.constexpr = tl.cdiv(M, DBM) * tl.cdiv(2 * C, DBC)
    ntga: tl.constexpr = tl.cdiv(M, GBMA) * tl.cdiv(2 * C, GBNA)
    ntgb: tl.constexpr = tl.cdiv(M, GBMB) * tl.cdiv(C, GBNB)
    for i in tl.static_range(NB):
        _mk_bar(BAR, 5 * i, P)
        t = pid
        while t < ntd1:
            _t_dw(BUF + (1 + i) * C, WD0 + i * (9 * C), BD0 + i * C,
                  A1 + i * (M * C), BUF, t, M, C, H, W, HW, WIDTH, C, 0,
                  3, 1, False, True, DBM, DBC)
            t += P
        _mk_bar(BAR, 5 * i + 1, P)
        t = pid
        while t < ntga:
            _t_gemm(A1 + i * (M * C), WG1 + i * (2 * C * C), BG1 + i * (2 * C),
                    A2 + i * (2 * M * C), A1, t, M, C, 2 * C, HW, C, 2 * C, 0,
                    False, False, False, True, GBMA, GBNA, GBKA, EKA)
            t += P
        _mk_bar(BAR, 5 * i + 2, P)
        t = pid
        while t < ntd2:
            _t_dw(A2 + i * (2 * M * C), WD2 + i * (KS2 * KS2 * 2 * C),
                  BD2 + i * (2 * C), A3 + i * (2 * M * C), A2, t,
                  M, 2 * C, H, W, HW, 2 * C, 2 * C, 0,
                  KS2, KS2 // 2, False, True, DBM, DBC)
            t += P
        _mk_bar(BAR, 5 * i + 3, P)
        t = pid
        while t < ntgb:
            _t_gemm(A3 + i * (2 * M * C), WG3 + i * (2 * C * C), BG3 + i * C,
                    A4 + i * (M * C), A3, t, M, 2 * C, C, HW, 2 * C, C, 0,
                    False, False, False, True, GBMB, GBNB, GBKB, EKB)
            t += P
        _mk_bar(BAR, 5 * i + 4, P)
        t = pid
        while t < ntd1:
            _t_dw(A4 + i * (M * C), WD4 + i * (9 * C), BD4 + i * C,
                  BUF + (2 + i) * C, BUF + (1 + i) * C, t,
                  M, C, H, W, HW, C, WIDTH, WIDTH,
                  3, 1, ADD, True, DBM, DBC)
            t += P

    _mk_bar(BAR, 5 * NB, P)
    nt2: tl.constexpr = tl.cdiv(M, GBM2) * tl.cdiv(C2, GBN2)
    t = pid
    while t < nt2:
        _t_gemm(BUF, W2, B2, OUT, BUF, t, M, (2 + NB) * C, C2, HW, WIDTH, 0, 0,
                False, True, False, True, GBM2, GBN2, GBK2, EK2)
        t += P
    if pid == 0:
        for j in tl.range(0, 2 * NPH):
            tl.store(BAR + j, 0)


_CULIB = None


def _max_blocks_per_sm(ck):
    """``cuOccupancyMaxActiveBlocksPerMultiprocessor`` for a compiled Triton
    kernel -- the driver's own answer, so the persistent grid is guaranteed
    co-resident (a non-resident block would deadlock the barrier)."""
    global _CULIB
    import ctypes
    if _CULIB is None:
        _CULIB = ctypes.CDLL("libcuda.so")
    n = ctypes.c_int()
    rc = _CULIB.cuOccupancyMaxActiveBlocksPerMultiprocessor(
        ctypes.byref(n), ctypes.c_void_p(ck.function),
        ctypes.c_int(ck.metadata.num_warps * 32),
        ctypes.c_size_t(ck.metadata.shared))
    if rc != 0 or n.value < 1:
        raise RuntimeError("occupancy query failed (rc=%d)" % rc)
    return n.value


def _bk_for(K):
    bk = 64
    while bk > 16 and (K % bk or K < bk):
        bk //= 2
    return bk


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------
class _State:
    """Everything bound to one (shape, dtype): static buffers, recorded launch
    list, and (when capture succeeded) the CUDA graph that replays it."""

    __slots__ = ("shape", "dtype", "steps", "sin", "sout", "graph", "keep", "step0")

    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype
        self.steps = []
        self.sin = None
        self.sout = None
        self.graph = None
        self.keep = []
        self.step0 = None


class _MKState:
    """One (shape, dtype) bound to a single persistent megakernel launch."""

    __slots__ = ("shape", "dtype", "fn", "args", "meta", "P", "out", "keep")


class _FusedBlock(nn.Module):
    """Lazy BN folding + static-schedule / CUDA-graph dispatch."""

    def _init_fused(self):
        self._plans = None
        self._plan_key = None
        self._state = None
        self._mk = None      # None = not built, False = unavailable
        self._graph_off = _NO_GRAPH

    # -- megakernel ---------------------------------------------------------
    def _mk_build(self, x, plans, M, HW, H, W, buf, out, keep):
        """Subclass hook: return ``(fn, args, meta, nphases, tile_counts)``."""
        raise NotImplementedError

    def _mk_prepare(self, x):
        """Build the single-launch megakernel for this shape, or return None."""
        plans = self._plans_for(x)
        N, _, H, W = x.shape
        HW = H * W
        M = N * HW
        st = _MKState()
        st.shape, st.dtype = tuple(x.shape), x.dtype
        st.keep = keep = []
        buf = torch.empty((M, (2 + self.n) * self.c), dtype=x.dtype, device=x.device)
        out = torch.empty((N, self.c2, H, W), dtype=x.dtype, device=x.device)
        keep += [buf, out]
        fn, args, meta, nph, tiles = self._mk_build(
            x, plans, M, HW, H, W, buf, out, keep)
        bar = torch.zeros(2 * nph, dtype=torch.int32, device=x.device)
        keep.append(bar)
        args = list(args) + [bar]
        # One num_warps for the whole megakernel.  The stage that dominates picks
        # it: the 3x3 conv for C2f, the GEMMs for CIB -- which is exactly
        # ``_tile_conv``'s rule (2 warps for narrow K, else 4).
        nw = _MK_NW or self._mk_warps()
        meta = dict(meta, num_warps=nw, num_stages=_MK_NS)
        if _MK_COOP:
            meta["launch_cooperative_grid"] = True
        # One SM's worth of programs is always resident, so this launch is safe
        # whatever the occupancy turns out to be; it also compiles the kernel so
        # the driver can tell us how many blocks per SM really fit.
        ck = fn[(_nsm(),)](*args, _nsm(), **meta)
        cap = _nsm() * _max_blocks_per_sm(ck)
        if _MK_WAVES:
            cap = min(cap, _nsm() * _MK_WAVES)
        st.P = max(1, min(max(tiles), cap))
        st.fn, st.args, st.meta, st.out = fn, tuple(args), meta, out
        fn[(st.P,)](*args, st.P, **meta)    # prime with the real grid
        torch.cuda.synchronize()
        return st

    def _mk_ok(self):
        return False

    def _mk_warps(self):
        return 4

    # -- to be provided by subclasses ---------------------------------------
    def _build_plans(self, dtype):
        raise NotImplementedError

    def _schedule(self, emit, x, out, plans, N, H, W, alloc):
        raise NotImplementedError

    def _fallback(self):
        return False

    def _baseline_forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    # -- schedule construction ----------------------------------------------
    def _plans_for(self, x):
        key = (x.dtype, x.device)
        if self._plan_key != key:
            self._plans = self._build_plans(x.dtype)
            self._plan_key = key
        return self._plans

    def _run_dynamic(self, x):
        plans = self._plans_for(x)
        N, _, H, W = x.shape
        out = torch.empty((N, self.c2, H, W), dtype=x.dtype, device=x.device)
        keep = []

        def alloc(shape):
            t = torch.empty(shape, dtype=x.dtype, device=x.device)
            keep.append(t)
            return t

        # PDL is only safe kernel->kernel: the first launch must obey normal
        # stream ordering against whatever wrote ``x``.
        counter = [0]

        def emit(fn, grid, args, meta):
            if counter[0] == 0 and meta.get("launch_pdl"):
                meta = dict(meta, launch_pdl=False)
            counter[0] += 1
            fn[grid](*args, **meta)

        self._schedule(emit, x, out, plans, N, H, W, alloc)
        return out

    def _prepare(self, x):
        """Build static buffers + recorded launch list, then try to capture."""
        plans = self._plans_for(x)
        N, _, H, W = x.shape
        st = _State(tuple(x.shape), x.dtype)
        st.sin = torch.empty_like(x)
        st.sout = torch.empty((N, self.c2, H, W), dtype=x.dtype, device=x.device)

        def alloc(shape):
            t = torch.empty(shape, dtype=x.dtype, device=x.device)
            st.keep.append(t)
            return t

        steps = st.steps
        self._schedule(lambda *a: steps.append(a), st.sin, st.sout,
                       plans, N, H, W, alloc)
        if steps and steps[0][3].get("launch_pdl"):
            steps[0][3]["launch_pdl"] = False

        st.sin.copy_(x)
        _exec(steps)                      # JIT-compile + prime the allocator
        if self._graph_off:
            return st
        # Keeping the first kernel out of the graph lets it read the caller's
        # tensor directly, which removes the static-input ``copy_`` (a ~4 us
        # tax per forward at these sizes) at the cost of one eager launch.
        eager1 = _EAGER1 and len(steps) > 1
        tail = steps[1:] if eager1 else steps
        try:
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                _exec(steps)
                _exec(steps)
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                _exec(tail)
            st.graph = g
            if eager1:
                st.step0 = steps[0]
        except Exception:                 # capture unsupported -> static eager
            st.graph = None
            st.step0 = None
            torch.cuda.synchronize()
        return st

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4 or not x.is_cuda or self.training or self._fallback():
            # BN folding is only valid in eval; anything unexpected takes the
            # reference path rather than a wrong answer.
            return self._baseline_forward(x)
        if _MEGA and self._mk is not False and self._mk_ok() and x.is_contiguous():
            mk = self._mk
            if mk is None or mk.dtype is not x.dtype or mk.shape != tuple(x.shape):
                try:
                    self._mk = mk = self._mk_prepare(x)
                except Exception:
                    # Cooperative launch refused, occupancy query unavailable,
                    # compile failure: take the multi-kernel path for good rather
                    # than risk a barrier that cannot be satisfied.
                    self._mk = False
                    mk = None
            if mk is not None:
                mk.fn[(mk.P,)](x, *mk.args[1:], mk.P, **mk.meta)
                return mk.out
        st = self._state
        if st is None or st.dtype is not x.dtype or st.shape != x.shape:
            if not x.is_contiguous():
                return self._run_dynamic(x.contiguous())
            self._state = st = self._prepare(x)
        elif not x.is_contiguous():
            return self._run_dynamic(x.contiguous())
        if st.step0 is not None:
            # Re-bind only the X (and unused Res) operand of the first kernel to
            # the caller's tensor; everything downstream is baked into the graph.
            fn, grid, args, meta = st.step0
            fn[grid](x, *args[1:4], x, *args[5:], **meta)
            st.graph.replay()
        elif st.graph is not None:
            st.sin.copy_(x)
            st.graph.replay()
        else:
            st.sin.copy_(x)
            _exec(st.steps)
        # Static output buffer, reused across calls -- the same contract as
        # ``torch.cuda.make_graphed_callables``.  Every consumer in the model
        # reads it before this block runs again; clone if you need to keep it.
        return st.sout


def _exec(steps):
    for fn, grid, args, meta in steps:
        fn[grid](*args, **meta)


class YOLOC2f(_FusedBlock):
    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = YOLOConv(c1, 2 * self.c, 1, 1)
        self.cv2 = YOLOConv((2 + n) * self.c, c2, 1, 1)
        self.m = nn.ModuleList(
            YOLOBottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n)
        )
        self.n = n
        self.c2 = c2
        self._g = g
        self._init_fused()

    def _fallback(self):
        return self._g != 1

    def _build_plans(self, dtype):
        return {
            "cv1": _plan_1x1(self.cv1, dtype),
            "cv2": _plan_1x1(self.cv2, dtype),
            "m": [(_plan_3x3(b.cv1, dtype), _plan_3x3(b.cv2, dtype), bool(b.add))
                  for b in self.m],
        }

    def _schedule(self, emit, x, out, plans, N, H, W, alloc):
        HW = H * W
        M = N * HW
        c, n = self.c, self.n
        width = (2 + n) * c
        stride = _round_up(width, _ROWPAD)
        buf = alloc((M, stride))
        _em_1x1(emit, plans["cv1"], x, buf, M, HW, 0, stride, True, False)
        for i, (p1, p2, add) in enumerate(plans["m"]):
            src = buf[:, (1 + i) * c:]
            dst = buf[:, (2 + i) * c:]
            tmp = alloc((M, c))
            _em_3x3(emit, p1, src, tmp, M, H, W, stride, c)
            _em_3x3(emit, p2, tmp, dst, M, H, W, c, stride,
                    res=src if add else None, srr=stride)
        _em_1x1(emit, plans["cv2"], buf, out, M, HW, stride, 0, False, True)

    def _mk_ok(self):
        return self._g == 1 and self.n >= 1

    def _mk_warps(self):
        return 2 if self.c <= 32 else 4

    def _mk_build(self, x, plans, M, HW, H, W, buf, out, keep):
        c, n, c2, c1 = self.c, self.n, self.c2, self.cv1.conv.weight.shape[1]
        width = (2 + n) * c
        p1s = [p[0] for p in plans["m"]]
        p2s = [p[1] for p in plans["m"]]
        add = bool(plans["m"][0][2])
        WM = torch.stack([torch.stack([a.wt, b.wt]) for a, b in zip(p1s, p2s)]).contiguous()
        BMs = torch.stack([torch.stack([a.bias, b.bias]) for a, b in zip(p1s, p2s)]).contiguous()
        tmp = torch.empty(n * M * c, dtype=x.dtype, device=x.device)
        keep += [WM, BMs, tmp]
        cv1, cv2 = plans["cv1"], plans["cv2"]
        cbk = 16
        while cbk < c:
            cbk *= 2
        gbm1, gbn1, _ = _tile_mn(M, 2 * c)
        gbm2, gbn2, _ = _tile_mn(M, c2)
        cbm, cbn, _ = _tile_conv(M, c, c)
        args = (x, out, buf, tmp, cv1.wt, cv1.bias, cv2.wt, cv2.bias, WM, BMs)
        meta = dict(
            M=M, HW=HW, H=H, W=W, C=c, C1=c1, C2=c2, WIDTH=width, NB=n, ADD=add,
            GBM1=gbm1, GBN1=gbn1, GBM2=gbm2, GBN2=gbn2,
            GBK1=_bk_for(c1), GBK2=_bk_for(width),
            EK1=(c1 % _bk_for(c1) == 0), EK2=(width % _bk_for(width) == 0),
            CBM=cbm, CBN=cbn, CBK=cbk, CKM=(cbk != c),
            NPH=2 * n + 1)
        cd = triton.cdiv
        tiles = [cd(M, gbm1) * cd(2 * c, gbn1), cd(M, gbm2) * cd(c2, gbn2),
                 cd(M, cbm) * cd(c, cbn)]
        return _mk_c2f, args, meta, 2 * n + 1, tiles


class YOLOC2fCIB(YOLOC2f):
    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, lk: bool = False, g: int = 1, e: float = 0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(YOLOCIB(self.c, self.c, shortcut, e=1.0, lk=lk) for _ in range(n))
        self.lk = lk
        self._init_fused()

    def _fallback(self):
        return False

    def _build_plans(self, dtype):
        blocks = []
        for blk in self.m:
            seq = blk.cv1
            blocks.append((
                [_plan_dw(seq[0], dtype),
                 _plan_1x1(seq[1], dtype),
                 _plan_repvggdw(seq[2], dtype) if isinstance(seq[2], YOLORepVGGDW)
                 else _plan_dw(seq[2], dtype),
                 _plan_1x1(seq[3], dtype),
                 _plan_dw(seq[4], dtype)],
                bool(blk.add),
            ))
        return {"cv1": _plan_1x1(self.cv1, dtype),
                "cv2": _plan_1x1(self.cv2, dtype), "m": blocks}

    def _schedule(self, emit, x, out, plans, N, H, W, alloc):
        HW = H * W
        M = N * HW
        c, n = self.c, self.n
        width = (2 + n) * c
        stride = _round_up(width, _ROWPAD)
        buf = alloc((M, stride))
        _em_1x1(emit, plans["cv1"], x, buf, M, HW, 0, stride, True, False)
        for i, (stages, add) in enumerate(plans["m"]):
            src = buf[:, (1 + i) * c:]
            dst = buf[:, (2 + i) * c:]
            s0, s1, s2, s3, s4 = stages
            a1 = alloc((M, s0.cout))
            a2 = alloc((M, s1.cout))
            a3 = alloc((M, s2.cout))
            a4 = alloc((M, s3.cout))
            _em_dw(emit, s0, src, a1, M, H, W, stride, s0.cout)
            _em_1x1(emit, s1, a1, a2, M, HW, s0.cout, s1.cout, False, False)
            _em_dw(emit, s2, a2, a3, M, H, W, s1.cout, s2.cout)
            _em_1x1(emit, s3, a3, a4, M, HW, s2.cout, s3.cout, False, False)
            _em_dw(emit, s4, a4, dst, M, H, W, s3.cout, stride,
                   res=src if add else None, srr=stride)
        _em_1x1(emit, plans["cv2"], buf, out, M, HW, stride, 0, False, True)

    def _mk_ok(self):
        # The megakernel loses on CIB: 5 stage kinds with conflicting num_warps
        # and 6 barriers instead of 3.  Measured 44.2 -> 52.2 us and 36.0 -> 39.9.
        # `_MEGA=2` forces it on anyway, for A/B.
        return _MEGA >= 2 and self.n >= 1

    def _mk_warps(self):
        return 4

    def _mk_build(self, x, plans, M, HW, H, W, buf, out, keep):
        c, n, c2, c1 = self.c, self.n, self.c2, self.cv1.conv.weight.shape[1]
        width = (2 + n) * c
        stages = [b[0] for b in plans["m"]]
        add = bool(plans["m"][0][1])
        ks2 = 7 if stages[0][2].kind == "dw7" else 3
        W_ = [torch.stack([st[j].wt for st in stages]).contiguous() for j in range(5)]
        B_ = [torch.stack([st[j].bias for st in stages]).contiguous() for j in range(5)]
        A = [torch.empty(n * M * ch, dtype=x.dtype, device=x.device)
             for ch in (c, 2 * c, 2 * c, c)]
        keep += W_ + B_ + A
        cv1, cv2 = plans["cv1"], plans["cv2"]
        gbm1, gbn1, _ = _tile_mn(M, 2 * c)
        gbm2, gbn2, _ = _tile_mn(M, c2)
        gbma, gbna, _ = _tile_mn(M, 2 * c)
        gbmb, gbnb, _ = _tile_mn(M, c)
        dbm, dbc, _ = _tile_dw(M, c)
        args = (x, out, buf, A[0], A[1], A[2], A[3],
                cv1.wt, cv1.bias, cv2.wt, cv2.bias,
                W_[0], B_[0], W_[1], B_[1], W_[2], B_[2], W_[3], B_[3], W_[4], B_[4])
        meta = dict(
            M=M, HW=HW, H=H, W=W, C=c, C1=c1, C2=c2, WIDTH=width, NB=n, ADD=add,
            KS2=ks2,
            GBM1=gbm1, GBN1=gbn1, GBM2=gbm2, GBN2=gbn2,
            GBMA=gbma, GBNA=gbna, GBMB=gbmb, GBNB=gbnb,
            GBK1=_bk_for(c1), GBK2=_bk_for(width),
            EK1=(c1 % _bk_for(c1) == 0), EK2=(width % _bk_for(width) == 0),
            GBKA=_bk_for(c), GBKB=_bk_for(2 * c),
            EKA=(c % _bk_for(c) == 0), EKB=(2 * c % _bk_for(2 * c) == 0),
            DBM=dbm, DBC=dbc,
            NPH=5 * n + 1)
        cd = triton.cdiv
        tiles = [cd(M, gbm1) * cd(2 * c, gbn1), cd(M, gbm2) * cd(c2, gbn2),
                 cd(M, gbma) * cd(2 * c, gbna), cd(M, gbmb) * cd(c, gbnb),
                 cd(M, dbm) * cd(c, dbc), cd(M, dbm) * cd(2 * c, dbc)]
        return _mk_cib, args, meta, 5 * n + 1, tiles
