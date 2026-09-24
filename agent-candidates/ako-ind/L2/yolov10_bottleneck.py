"""YOLOv10 bottleneck block: BatchNorm folded once, both 3x3 convs written as
Triton implicit-GEMM kernels with bias / SiLU / residual folded into the epilogue,
and the two launches chained with programmatic dependent launch.

The eager block is conv1, bn1, silu1, conv2, bn2, silu2, add -- seven launches,
~40-93us of GPU work and ~120-130us of CPU dispatch at the captured shapes
(N in {1,4}, C in {16..128}, HW 160x160..20x20, fp16).  Here:

1. Each ``YOLOConv``'s eval-mode BatchNorm is folded into that conv's weight and
   bias exactly once and cached on the module, keyed on (dtype, shape, device) and
   invalidated by ``load_state_dict``, ``_apply`` and train/eval flips.  Two of the
   seven launches disappear outright.
2. Each conv is a single launch over a (pixel-tile x out-channel-block) grid that
   accumulates the nine taps in fp32 registers and applies folded bias + SiLU
   (+ the residual, for cv2) in the epilogue -- three more launches gone, and the
   c_-channel intermediate is written and read once instead of three times.
3. Tiles are held pixel-major, ``tl.dot(x[pixels, Cin], W[Cin, Cout])``, and the
   channel axis is made contiguous (NHWC) so the nine tap loads coalesce.  The
   apparently-natural alternative -- keeping NCHW and transposing the tile to
   ``tl.dot(W[Cout,Cin], x[Cin,P])`` so the *pixel* axis is the contiguous one --
   measures 2-6x slower on Triton 3.6/B200: MMA operand-layout matching dominates
   coalescing here.
4. The weight panels are stored **K-minor** (``[tap, Cout, K]``) and transposed in
   registers for the MMA, so a CTA's weight tile arrives as BCO rows of K
   contiguous fp16 -- one vector load per row instead of one per (k, Cout) pair.
   At the BCO=16 MMA minimum that is 1-2 KB per instruction instead of 32 bytes.
5. Both launches use **PDL** (``launch_pdl=True`` + ``gdc_wait()`` before the first
   producer-dependent load + ``gdc_launch_dependents()`` last), so the consumer
   grid starts while the producer's tail drains.  This operator is dominated by
   fixed per-launch cost -- ~2.05us of measured time per launch, enough that an
   *empty* kernel costs a full unit -- and PDL recovers essentially all of it.

Three tap schedules, chosen per shape class by a static table:

* ``_conv3x3``       nine K=Cin dots from an NHWC input.
* ``_conv3x3_k4``    three K=4*Cin dots from an NHWC input -- in NHWC the three
  horizontal taps of a row are a *contiguous* 3*C span, so they fold straight into
  the reduction dim.  Better MMA shape for a 4/3 padding overhead.
* ``_conv3x3_nchw_k4`` three K=4*Cin dots read straight from **NCHW**: there the
  three horizontal taps are consecutive *pixels*, so a [TP, Cin, 4] 3D block load
  is affine with a 4-wide contiguous minor axis (8-byte vectors instead of 2-byte
  scalars).  Used for cv1 so the NCHW->NHWC conversion launch can be skipped
  entirely, which is a win wherever the block is launch/latency-bound.

Any schedule may additionally split its reduction across ``SPLIT`` CTAs per output
tile.  That multiplies the CTA count at unchanged traffic *per CTA*, which is the
only way to buy parallelism for the shapes with too few pixels to fill the SMs; the
partials are combined without an extra launch by ``_reduce_splits`` (private fp32
slot per tile + arrival counter, last CTA runs the epilogue and re-zeros the
scratch).  The tuned table enables it only where it measured a win.

Anything the fast path does not cover -- training mode, grouped conv, kernel sizes
other than 3x3, non-unit stride/pad/dilation, a non-SiLU activation,
non-fp16/bf16 dtypes, CPU tensors -- falls back to the eager submodule path.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

from .yolov10_conv import YOLOConv

# Programmatic dependent launch.  The two convs form a chain, and this harness
# charges ~2.05us of measured time per launch; PDL lets the consumer grid start
# while the producer's tail drains, which recovers almost all of it (measured on
# chained trivial kernels: 2 launches 13.38us -> 9.22us, 1 launch 11.26us ->
# 7.14us, i.e. down to the 7.07us measurement floor).  Optional at import so the
# kernel still runs on a Triton build without the intrinsics.
try:
    from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait
    _HAS_PDL = True
except ImportError:                                            # pragma: no cover
    _HAS_PDL = False

    @triton.jit
    def gdc_wait():
        pass

    @triton.jit
    def gdc_launch_dependents():
        pass

# Output memory format.  False returns a channels-last *view* of the NHWC buffer
# the kernel already produced (identical shape / dtype / values, ~7-13% faster);
# True adds an in-epilogue transpose so the result is contiguous NCHW like the
# baseline's.
NCHW_OUT = False


@triton.jit
def _nchw_to_nhwc(X, Y, NPIX, C: tl.constexpr, C_R: tl.constexpr, TP: tl.constexpr,
                  PDL: tl.constexpr):
    """NCHW -> NHWC.  C is the power-of-two block width, C_R the real channel
    count (and therefore the NHWC row stride)."""
    pid = tl.program_id(0)
    n = tl.program_id(1)
    p = pid * TP + tl.arange(0, TP)
    ok = p < NPIX
    c = tl.arange(0, C)
    m = ok[:, None] & (c < C_R)[None, :]
    base = n * (C_R * NPIX)
    if PDL:
        gdc_wait()
    a = tl.load(X + base + c[None, :] * NPIX + tl.where(ok, p, 0)[:, None], mask=m, other=0.0)
    tl.store(Y + base + tl.where(ok, p * C_R, 0)[:, None] + c[None, :], a, mask=m)
    if PDL:
        gdc_launch_dependents()


@triton.jit
def _reduce_splits(acc, ACC, CNT, tid, SPLIT: tl.constexpr,
                   TP: tl.constexpr, BCO: tl.constexpr):
    """Cross-CTA split-K combine, no extra launch.

    Each tile owns a private ``[TP, BCO]`` fp32 slot, so the atomic add, the
    read-back and the reset are all fully coalesced and need no masking.  Every
    split CTA adds its partial with *release* semantics, then bumps the tile's
    arrival counter with *acq_rel*; the CTA that observes the last arrival has, by
    the release/acquire chain, all the other partials visible to it, reads the
    slot with a volatile (L1-bypassing) load and runs the epilogue.  It also
    resets the slot and the counter, so the buffers are back to zero for the next
    call and no zero-fill launch is ever needed (the plan allocates them zeroed).

    Returns (accumulated_value, this_CTA_owns_the_epilogue).
    """
    r = tl.arange(0, TP)
    cc = tl.arange(0, BCO)
    ao = ACC + tid * (TP * BCO) + r[:, None] * BCO + cc[None, :]
    tl.atomic_add(ao, acc, sem="release", scope="gpu")
    tl.debug_barrier()
    arrived = tl.atomic_add(CNT + tid, 1, sem="acq_rel", scope="gpu")
    last = arrived == SPLIT - 1
    tot = acc
    if last:
        tot = tl.load(ao, cache_modifier=".cv")
        tl.store(ao, tl.zeros([TP, BCO], tl.float32))
        tl.store(CNT + tid, 0)
    return tot, last


@triton.jit
def _epilogue(acc, B, R, Y, co, p, pok, cok, n, NPIX,
              COUT_R: tl.constexpr, TP: tl.constexpr, BCO: tl.constexpr,
              ADD: tl.constexpr, MASKC: tl.constexpr,
              RES_NCHW: tl.constexpr, TRANS_OUT: tl.constexpr, LAST=True):
    """folded bias -> SiLU -> optional residual -> store (NHWC, or NCHW via a
    register transpose so the strided store still coalesces along pixels)."""
    b = tl.load(B + co, mask=cok, other=0.0) if MASKC else tl.load(B + co)
    yv = acc + b[None, :]
    yv = yv * tl.sigmoid(yv)
    om = (pok[:, None] & cok[None, :]) if MASKC else tl.broadcast_to(pok[:, None], (TP, BCO))
    om = om & LAST
    if ADD:
        if RES_NCHW:
            yv += tl.load(R + n * (COUT_R * NPIX)
                          + tl.where(pok, p, 0)[:, None] + co[None, :] * NPIX,
                          mask=om, other=0.0)
        else:
            yv += tl.load(R + n * (NPIX * COUT_R)
                          + tl.where(pok, p * COUT_R, 0)[:, None] + co[None, :],
                          mask=om, other=0.0)
    if TRANS_OUT:
        omt = (cok[:, None] & pok[None, :]) if MASKC \
            else tl.broadcast_to(pok[None, :], (BCO, TP))
        tl.store(Y + n * (COUT_R * NPIX) + co[:, None] * NPIX
                 + tl.where(pok, p, 0)[None, :],
                 tl.trans(yv).to(Y.dtype.element_ty), mask=omt & LAST)
    else:
        tl.store(Y + n * (NPIX * COUT_R)
                 + tl.where(pok, p * COUT_R, 0)[:, None] + co[None, :],
                 yv.to(Y.dtype.element_ty), mask=om)


@triton.jit
def _conv3x3(X, R, Y, ACC, CNT, W, B, H, WD, NPIX,
             TP: tl.constexpr, CIN: tl.constexpr, BCO: tl.constexpr,
             CIN_R: tl.constexpr, COUT_R: tl.constexpr,
             ADD: tl.constexpr, MASKC: tl.constexpr,
             RES_NCHW: tl.constexpr, TRANS_OUT: tl.constexpr, WT: tl.constexpr,
             SPLIT: tl.constexpr, PDL: tl.constexpr):
    """NHWC input, nine K=CIN dots.  W is [9, Cin, Cout], tap index 3*kh + kw.
    ``SPLIT`` in {1, 3, 9} spreads the nine taps over that many CTAs per tile."""
    pid_p = tl.program_id(0)
    pid_c = tl.program_id(1)
    z = tl.program_id(2)
    n = z // SPLIT
    sp = z % SPLIT
    p = pid_p * TP + tl.arange(0, TP)
    ph = p // WD
    pw = p - ph * WD
    pok = p < NPIX
    co = pid_c * BCO + tl.arange(0, BCO)
    cin = tl.arange(0, CIN)
    cok = co < COUT_R
    ciok = cin < CIN_R
    xb = X + n * (NPIX * CIN_R)

    if PDL:
        gdc_wait()
    NT: tl.constexpr = 9 // SPLIT
    acc = tl.zeros([TP, BCO], dtype=tl.float32)
    for u in tl.static_range(NT):
        t = sp * NT + u
        ih = ph + (t // 3 - 1)
        iw = pw + (t % 3 - 1)
        ok = pok & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < WD)
        ap = xb + tl.where(ok, (ih * WD + iw) * CIN_R, 0)[:, None] + cin[None, :]
        if WT:
            wq = W + t * (CIN_R * COUT_R) + co[:, None] * CIN_R + cin[None, :]
            wt = tl.trans(tl.load(wq, mask=cok[:, None] & ciok[None, :], other=0.0) if MASKC
                          else tl.load(wq))
        else:
            wp = W + t * (CIN_R * COUT_R) + cin[:, None] * COUT_R + co[None, :]
            wt = tl.load(wp, mask=ciok[:, None] & cok[None, :], other=0.0) if MASKC \
                else tl.load(wp)
        a = tl.load(ap, mask=ok[:, None] & ciok[None, :], other=0.0) if MASKC \
            else tl.load(ap, mask=ok[:, None], other=0.0)
        acc = tl.dot(a, wt, acc)
    last = True
    if SPLIT > 1:
        tid = (n * tl.num_programs(0) + pid_p) * tl.num_programs(1) + pid_c
        acc, last = _reduce_splits(acc, ACC, CNT, tid, SPLIT, TP, BCO)
    _epilogue(acc, B, R, Y, co, p, pok, cok, n, NPIX, COUT_R, TP, BCO,
              ADD, MASKC, RES_NCHW, TRANS_OUT, last)
    if PDL:
        gdc_launch_dependents()


@triton.jit
def _conv3x3_k4(X, R, Y, ACC, CNT, W, B, H, WD, NPIX,
                TP: tl.constexpr, CIN: tl.constexpr, BCO: tl.constexpr,
                CIN_R: tl.constexpr, COUT_R: tl.constexpr,
                ADD: tl.constexpr, MASKC: tl.constexpr,
                RES_NCHW: tl.constexpr, TRANS_OUT: tl.constexpr, WT: tl.constexpr,
                SPLIT: tl.constexpr, PDL: tl.constexpr):
    """NHWC input, three K=4*CIN dots (the row's three horizontal taps are one
    contiguous 3*CIN span).  Needs CIN == CIN_R.  W is [3, 4*CIN, Cout] with the
    dw-th CIN-row block holding W[:, :, dh, dw]^T and the fourth block zero.
    ``SPLIT`` in {1, 3} spreads the three tap rows over that many CTAs per tile."""
    pid_p = tl.program_id(0)
    pid_c = tl.program_id(1)
    z = tl.program_id(2)
    n = z // SPLIT
    sp = z % SPLIT
    p = pid_p * TP + tl.arange(0, TP)
    ph = p // WD
    pw = p - ph * WD
    pok = p < NPIX
    K4: tl.constexpr = 4 * CIN
    j = tl.arange(0, K4)
    jt = j // CIN
    co = pid_c * BCO + tl.arange(0, BCO)
    cok = co < COUT_R
    xb = X + n * (NPIX * CIN_R)

    iwj = pw[:, None] - 1 + jt[None, :]
    cmask = (iwj >= 0) & (iwj < WD) & (jt < 3)[None, :]
    if PDL:
        gdc_wait()
    NT: tl.constexpr = 3 // SPLIT
    acc = tl.zeros([TP, BCO], dtype=tl.float32)
    for u in tl.static_range(NT):
        dh = sp * NT + u
        ih = ph + (dh - 1)
        rok = pok & (ih >= 0) & (ih < H)
        base = tl.where(rok, (ih * WD + pw - 1) * CIN_R, 0)
        a = tl.load(xb + base[:, None] + j[None, :], mask=rok[:, None] & cmask, other=0.0)
        if WT:
            wq = W + dh * (K4 * COUT_R) + co[:, None] * K4 + j[None, :]
            wt = tl.trans(tl.load(wq, mask=tl.broadcast_to(cok[:, None], (BCO, K4)), other=0.0)
                          if MASKC else tl.load(wq))
        else:
            wp = W + dh * (K4 * COUT_R) + j[:, None] * COUT_R + co[None, :]
            wt = tl.load(wp, mask=tl.broadcast_to(cok[None, :], (K4, BCO)), other=0.0) if MASKC \
                else tl.load(wp)
        acc = tl.dot(a, wt, acc)
    last = True
    if SPLIT > 1:
        tid = (n * tl.num_programs(0) + pid_p) * tl.num_programs(1) + pid_c
        acc, last = _reduce_splits(acc, ACC, CNT, tid, SPLIT, TP, BCO)
    _epilogue(acc, B, R, Y, co, p, pok, cok, n, NPIX, COUT_R, TP, BCO,
              ADD, MASKC, RES_NCHW, TRANS_OUT, last)
    if PDL:
        gdc_launch_dependents()


@triton.jit
def _conv3x3_nchw_k4(X, R, Y, ACC, CNT, W, B, H, WD, NPIX,
                     TP: tl.constexpr, CIN: tl.constexpr, BCO: tl.constexpr,
                     CIN_R: tl.constexpr, COUT_R: tl.constexpr,
                     ADD: tl.constexpr, MASKC: tl.constexpr,
                     RES_NCHW: tl.constexpr, TRANS_OUT: tl.constexpr, WT: tl.constexpr,
                     SPLIT: tl.constexpr, PDL: tl.constexpr):
    """**NCHW** input, NHWC output, three K=4*CIN dots.  In NCHW the three
    horizontal taps are consecutive pixels, so [TP, CIN, 4] is an affine 3D block
    whose minor axis is 4 contiguous elements -- 8-byte vector loads.  Needs
    CIN == CIN_R.  W is [3, CIN*4, Cout] with row c*4 + dw holding W[:, c, dh, dw].
    ``SPLIT`` in {1, 3} spreads the three tap rows over that many CTAs per tile."""
    pid_p = tl.program_id(0)
    pid_c = tl.program_id(1)
    z = tl.program_id(2)
    n = z // SPLIT
    sp = z % SPLIT
    p = pid_p * TP + tl.arange(0, TP)
    ph = p // WD
    pw = p - ph * WD
    pok = p < NPIX
    co = pid_c * BCO + tl.arange(0, BCO)
    cok = co < COUT_R
    cin = tl.arange(0, CIN)
    k = tl.arange(0, 4)
    K4: tl.constexpr = CIN * 4
    kk = tl.arange(0, K4)
    xb = X + n * (CIN_R * NPIX)

    iwk = pw[:, None] - 1 + k[None, :]
    okk = (iwk >= 0) & (iwk < WD) & (k < 3)[None, :]
    if PDL:
        gdc_wait()
    NT: tl.constexpr = 3 // SPLIT
    acc = tl.zeros([TP, BCO], dtype=tl.float32)
    for u in tl.static_range(NT):
        dh = sp * NT + u
        ih = ph + (dh - 1)
        rok = pok & (ih >= 0) & (ih < H)
        base = tl.where(rok, ih * WD + pw - 1, 0)
        a3 = tl.load(xb + base[:, None, None] + cin[None, :, None] * NPIX + k[None, None, :],
                     mask=rok[:, None, None] & okk[:, None, :], other=0.0)
        if WT:
            wq = W + dh * (K4 * COUT_R) + co[:, None] * K4 + kk[None, :]
            wt = tl.trans(tl.load(wq, mask=tl.broadcast_to(cok[:, None], (BCO, K4)), other=0.0)
                          if MASKC else tl.load(wq))
        else:
            wp = W + dh * (K4 * COUT_R) + kk[:, None] * COUT_R + co[None, :]
            wt = tl.load(wp, mask=tl.broadcast_to(cok[None, :], (K4, BCO)), other=0.0) if MASKC \
                else tl.load(wp)
        acc = tl.dot(tl.reshape(a3, (TP, K4)), wt, acc)
    last = True
    if SPLIT > 1:
        tid = (n * tl.num_programs(0) + pid_p) * tl.num_programs(1) + pid_c
        acc, last = _reduce_splits(acc, ACC, CNT, tid, SPLIT, TP, BCO)
    _epilogue(acc, B, R, Y, co, p, pok, cok, n, NPIX, COUT_R, TP, BCO,
              ADD, MASKC, RES_NCHW, TRANS_OUT, last)
    if PDL:
        gdc_launch_dependents()


_KERNELS = {"k1": _conv3x3, "k4": _conv3x3_k4, "nk4": _conv3x3_nchw_k4}


def _p2(v: int) -> int:
    return 1 << max(0, (int(v) - 1)).bit_length()


# (padded channel count, "does the whole batch have many pixels?") ->
#   (needs NCHW->NHWC pre-pass?, cv1 (schedule, TP, BCO, warps), cv2 (...))
#
# Tuned against the harness metric on B200 (see ITERATIONS.md).  Two things the
# sweep settled: (a) the pixel-count split is necessary -- at equal channel count
# the two batch sizes want different grids, because with few total pixels the
# kernel is parallelism-starved and wants a *finer* out-channel split (BCO=16,
# 2 warps) whose only cost is redundant L2-resident input reads; (b) skipping the
# NHWC pre-pass by reading cv1's input straight from NCHW wins everywhere except
# the (C=64, many pixels) class, where the wider strided loads cost more than the
# launch it saves.
_BIG_PIXELS = 4096
_CFG = {
    (16, True): (False, ("nk4", 32, 16, 2, 0, 1), ("k1", 64, 16, 2, 0, 1)),
    (16, False): (False, ("nk4", 32, 16, 2, 0, 1), ("k1", 32, 16, 2, 0, 1)),
    (32, True): (False, ("nk4", 16, 32, 4, 1, 1), ("k4", 16, 32, 2, 1, 1)),
    (32, False): (False, ("nk4", 16, 32, 4, 1, 1), ("k4", 16, 32, 2, 1, 1)),
    (64, True): (False, ("nk4", 16, 64, 2, 0, 1), ("k1", 32, 16, 2, 1, 1)),
    (64, False): (False, ("nk4", 16, 16, 2, 1, 1), ("k4", 16, 16, 2, 1, 1)),
    (128, True): (False, ("nk4", 16, 32, 2, 0, 3), ("k1", 16, 32, 4, 1, 9)),
    (128, False): (False, ("nk4", 16, 32, 2, 0, 3), ("k1", 16, 32, 4, 1, 9)),
}
_CFG_DEFAULT = (True, ("k1", 32, 32, 4, 0, 1), ("k1", 32, 32, 4, 0, 1))
_TRANSPOSE_TP, _TRANSPOSE_WARPS = 64, 4
PDL = _HAS_PDL
_PK = {"launch_pdl": True} if PDL else {}


@torch.no_grad()
def _fold(conv, bn):
    """conv (+ bias) followed by eval-mode BN, as a single weight/bias pair."""
    w = conv.weight.detach().float()
    b = (conv.bias.detach().float() if conv.bias is not None
         else torch.zeros(w.shape[0], device=w.device, dtype=torch.float32))
    if bn is not None:
        s = bn.weight.detach().float() / torch.sqrt(bn.running_var.detach().float() + bn.eps)
        w = w * s.view(-1, 1, 1, 1)
        b = b * s + bn.bias.detach().float() - bn.running_mean.detach().float() * s
    return w, b


def _pack(w: torch.Tensor, kind: str, cin: int, cout: int, dtype,
          wt: bool = False) -> torch.Tensor:
    """[Cout, Cin, 3, 3] -> the tap-major panel layout the chosen schedule reads.

    ``wt`` stores each tap panel K-minor ([tap, Cout, K] instead of [tap, K, Cout]),
    so a CTA's weight tile loads as BCO rows of K contiguous fp16 -- one vector load
    per row instead of one per (k, BCO) pair -- and is transposed in registers for
    the MMA.  At BCO=16 that is 32 bytes/instruction versus 1-2 KB.
    """
    if kind == "k1":
        r = w.permute(2, 3, 1, 0).contiguous().view(9, cin, cout).to(dtype)
        return r.transpose(1, 2).contiguous() if wt else r
    out = torch.zeros(3, 4 * cin, cout, device=w.device, dtype=dtype)
    for dh in range(3):
        for dw in range(3):
            panel = w[:, :, dh, dw].t().to(dtype)          # [Cin, Cout]
            if kind == "k4":                               # NHWC: rows dw*Cin + c
                out[dh, dw * cin:(dw + 1) * cin, :] = panel
            else:                                          # NCHW: rows c*4 + dw
                out[dh, dw::4, :] = panel
    return out.transpose(1, 2).contiguous() if wt else out


def _resolve(kind: str, cin_r: int, cin_p: int) -> str:
    """k4/nk4 fold the tap offset into the K index, which only lines up when the
    channel count needs no padding; fall back to the general schedule otherwise."""
    return "k1" if kind != "k1" and cin_r != cin_p else kind


def _drop_plan_hook(module, incompatible_keys):
    module._plan = None


class YOLOBottleneck(nn.Module):
    def __init__(self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k=(3, 3), e: float = 0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = YOLOConv(c1, c_, k[0], 1)
        self.cv2 = YOLOConv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2
        self._c1, self._c_, self._c2, self._g = c1, c_, c2, g
        self._plan = None
        self._pdtype = None
        self._pshape = ()
        self._pdev = None
        self.register_load_state_dict_post_hook(_drop_plan_hook)

    def _dummy(self, dev):
        d = self.__dict__.get("_dbuf")
        if d is None or d.device != dev:
            d = self._dbuf = torch.zeros(1, dtype=torch.float32, device=dev)
        return d

    # -- plan invalidation ---------------------------------------------------
    def _apply(self, *a, **kw):
        self._plan = None
        return super()._apply(*a, **kw)

    def train(self, mode: bool = True):
        self._plan = None
        return super().train(mode)

    # -- fast-path eligibility ----------------------------------------------
    def _eligible(self, x: torch.Tensor) -> bool:
        if self.training or not x.is_cuda or x.dim() != 4:
            return False
        if x.dtype not in (torch.float16, torch.bfloat16):
            return False
        if self._g != 1 or self._c_ < 1 or x.shape[1] != self._c1:
            return False
        act_t = type(YOLOConv.default_act)
        for cv, ci, co in ((self.cv1, self._c1, self._c_), (self.cv2, self._c_, self._c2)):
            cc = cv.conv
            if (tuple(cc.weight.shape) != (co, ci, 3, 3) or cc.stride != (1, 1)
                    or cc.padding != (1, 1) or cc.dilation != (1, 1) or cc.groups != 1):
                return False
            if not isinstance(cv.act, act_t):
                return False
        return True

    def _build_plan(self, x: torch.Tensor):
        n, _, h, w = x.shape
        dt = x.dtype
        c1, cm, c2 = self._c1, self._c_, self._c2
        npix = h * w
        p1 = max(16, _p2(c1))
        pm = max(16, _p2(cm))
        p2c = max(16, _p2(c2))

        use_t, s1, s2 = _CFG.get((max(p1, pm, p2c), n * npix >= _BIG_PIXELS), _CFG_DEFAULT)
        k1 = _resolve(s1[0], c1, p1)
        k2 = _resolve(s2[0], cm, pm)
        if k1 == "k1" and s1[0] == "nk4":
            use_t = True                      # the padded fallback needs NHWC input
        bco1 = min(s1[2], pm)
        bco2 = min(s2[2], p2c)
        sk1 = s1[5] if len(s1) > 5 else 1
        sk2 = s2[5] if len(s2) > 5 else 1
        if k1 == "k1" and 9 % sk1:
            sk1 = 1
        if k1 != "k1" and 3 % sk1:
            sk1 = 1
        if k2 == "k1" and 9 % sk2:
            sk2 = 1
        if k2 != "k1" and 3 % sk2:
            sk2 = 1
        mk1 = (c1 != p1) or (cm % bco1 != 0)
        mk2 = (cm != pm) or (c2 % bco2 != 0)
        nt1, nc1 = triton.cdiv(npix, s1[1]), triton.cdiv(cm, bco1)
        nt2, nc2 = triton.cdiv(npix, s2[1]), triton.cdiv(c2, bco2)
        g1 = (nt1, nc1, n * sk1)
        g2 = (nt2, nc2, n * sk2)
        wt1, wt2 = bool(s1[4]), bool(s2[4])
        dev = x.device
        # split-K scratch: one private [TP, BCO] fp32 slot per output tile, plus a
        # per-tile arrival counter.  Allocated zeroed once; every call's epilogue
        # resets the slots it consumed, so no zero-fill launch is ever needed.
        a1 = c1_ = a2 = c2_ = self._dummy(dev)
        if sk1 > 1:
            a1 = torch.zeros(n * nt1 * nc1 * s1[1] * bco1, dtype=torch.float32, device=dev)
            c1_ = torch.zeros(n * nt1 * nc1, dtype=torch.int32, device=dev)
        if sk2 > 1:
            a2 = torch.zeros(n * nt2 * nc2 * s2[1] * bco2, dtype=torch.float32, device=dev)
            c2_ = torch.zeros(n * nt2 * nc2, dtype=torch.int32, device=dev)

        w1, b1 = _fold(self.cv1.conv, getattr(self.cv1, "bn", None))
        w2, b2 = _fold(self.cv2.conv, getattr(self.cv2, "bn", None))
        plan = (
            use_t, (triton.cdiv(npix, _TRANSPOSE_TP), n),
            _KERNELS[k1], g1, s1[1], p1, bco1, s1[3], mk1,
            _pack(w1, k1, c1, cm, dt, wt1), wt1, sk1, a1, c1_,
            _KERNELS[k2], g2, s2[1], pm, bco2, s2[3], mk2,
            _pack(w2, k2, cm, c2, dt, wt2), wt2, sk2, a2, c2_,
            b1.contiguous(), b2.contiguous(), h, w, npix, c1, cm, c2, bool(self.add),
            (n, h, w, cm), (n, c2, h, w) if NCHW_OUT else (n, h, w, c2),
        )
        self._plan = plan
        self._pdtype = dt
        self._pshape = tuple(x.shape)
        self._pdev = x.device
        return plan

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        plan = self._plan
        if (plan is None or x.dtype is not self._pdtype or x.shape != self._pshape
                or x.device != self._pdev or self.training):
            if not self._eligible(x):
                y = self.cv2(self.cv1(x))
                return x + y if self.add else y
            plan = self._build_plan(x)
        if not x.is_contiguous():
            x = x.contiguous()

        (use_t, gt,
         K1, g1, tp1, cinp1, bco1, nw1, mk1, w1, wt1, sk1, ac1, cn1,
         K2, g2, tp2, cinp2, bco2, nw2, mk2, w2, wt2, sk2, ac2, cn2,
         b1, b2, h, w, npix, c1, cm, c2, add, mshape, oshape) = plan
        dev, dt = x.device, x.dtype

        if use_t:
            xin = torch.empty((x.shape[0], h, w, c1), dtype=dt, device=dev)
            _nchw_to_nhwc[gt](x, xin, npix, cinp1, c1, _TRANSPOSE_TP, PDL,
                              num_warps=_TRANSPOSE_WARPS, **_PK)
        else:
            xin = x                                   # cv1 reads NCHW directly
        mid = torch.empty(mshape, dtype=dt, device=dev)
        y = torch.empty(oshape, dtype=dt, device=dev)
        K1[g1](xin, xin, mid, ac1, cn1, w1, b1, h, w, npix, tp1, cinp1, bco1, c1, cm,
               False, mk1, False, False, wt1, sk1, PDL,
               num_warps=nw1, num_stages=1, **_PK)
        K2[g2](mid, xin, y, ac2, cn2, w2, b2, h, w, npix, tp2, cinp2, bco2, cm, c2,
               add, mk2, not use_t, NCHW_OUT, wt2, sk2, PDL,
               num_warps=nw2, num_stages=1, **_PK)
        return y if NCHW_OUT else y.permute(0, 3, 1, 2)
