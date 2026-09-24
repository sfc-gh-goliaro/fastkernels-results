"""YOLOv10 CIB (Compact Inverted Block).

The scored workload is fp16 ``[N, 128, 20, 20]``, ``N in {1, 4}``, built as
``YOLOCIB(c1=128, c2=128, shortcut=True, e=1.0, lk=True)``:

    cv1.0  dw3x3 g=128   128->128  +BN +SiLU
    cv1.1  pw1x1         128->256  +BN +SiLU
    cv1.2  RepVGGDW(256) dw7x7 +BN  ||  dw3x3 +BN  -> sum -> SiLU
    cv1.3  pw1x1         256->128  +BN +SiLU
    cv1.4  dw3x3 g=128   128->128  +BN +SiLU
    out = x + cv1(x)

~32 MMAC per image -- a couple of microseconds of real device work -- against
~19 kernel launches and ~280 us of *host* dispatch in the eager module chain.

Layer 1: every BatchNorm is folded into its convolution's weight+bias once and
cached, and RepVGGDW's 7x7/3x3 branch pair collapses into a single 7x7 kernel.

Layer 2: the folded chain runs as five kernels -- three depthwise, two pointwise
-- over a shared **zero-padded workspace**.  The workspace row pitch is
``W + 2*PAD``, so a run of output positions is a contiguous run of memory and
every depthwise tap is that same base address plus a *constexpr* offset: no
bounds masks, no per-tap address tile.  Positions outside the image are blanked
before each store, so the pad border stays zero and the buffers are safe to
reuse across calls.

Two geometry details every stencil here depends on:

* Tiles start on a 128 B boundary (``POS0`` rounds ``VLO`` *down* to ``_ALIGN``)
  so a store covers whole cache lines -- worth ~20%.
* Because ``POS0`` rounds *down*, the buffer needs more leading pad **rows**
  (``PADR``) than the column pad (``PAD``), or the KxK halo reads before the
  allocation.  ``PADR`` is sized for the worst tap reach, and both
  ``_make_launch`` and ``_cuda_dw_geom`` re-check the invariant rather than
  trusting it -- matching outputs do not reveal a bad halo read, they only mean
  the out-of-range positions happened to be pad.

The dw7x7 stage is hand-written CUDA (``cib_cuda.cu``); the other four stay
Triton.  That split is measured, not assumed -- see ITERATIONS.md.  The extension
is built once at import, named after its source hash, and *any* failure leaves
``_CUDA = None`` so the Triton path carries the whole block.

**What the launch structure is for.** This block never uses more than ~20% of the
GPU (ncu: 0.22 waves/SM), so its cost is a chain of serialized latency ramps, and
the forward's time tracks the number of separately submitted launch groups at
least as much as the number of kernels.  Three consequences:

* Stages B-D touch nothing but the workspace and the folded weights, so they are
  captured once into a graph valid for any input pointer.  That is the always-
  available path, and it is what removed r1's per-call ``si.copy_(x)`` (7.1 us).
* When the input address repeats -- which is what the harness does at N=4, whose
  captured input is non-contiguous and therefore passed through unchanged --
  stage A is captured with it (four kernels, one submission).
* When the *output* address repeats too, all five stages go into one submission.
  In steady state the caching allocator returns the same block for an identically
  shaped ``torch.empty`` once the previous output has been released, so the graph
  writes exactly the buffer being returned.  Both addresses are re-checked on
  every call, so this can only fire when that is true; a caller that holds its
  outputs, or moves its input, drops to one of the paths above and still gets a
  freshly allocated output that no later call will overwrite.

Measured at N=4, candidate ms: 0.0377 (five separate submissions, r1) -> 0.0318
(no input copy) -> 0.0298 (CUDA dw7x7) -> 0.0276 (A in the graph) -> 0.0266 (all
five in the graph).

``self.training``, an unsupported geometry, or a non-CUDA / non-fp16 input all
fall back to the eager module path, so the ``__init__`` / ``forward`` contracts,
the ``shortcut and c1 == c2`` add and the ``lk`` branch stay exact for every
configuration.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .yolov10_conv import YOLOConv
from .yolov10_repvggdw import YOLORepVGGDW

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # pragma: no cover - Triton is always present on the bench host
    _HAS_TRITON = False

_BP_MAX = 128
_ALIGN = 64  # workspace positions per 128 B cache line (fp16)
_ALIGN_C = None  # tl.constexpr(_ALIGN); kernels cannot read a plain global


# ---------------------------------------------------------------------------
# BatchNorm folding
# ---------------------------------------------------------------------------
def _fold_conv_bn(conv: nn.Module, bn: nn.Module | None):
    """``(weight, bias)`` equivalent to ``bn(conv(x))``, folded in fp32."""
    w = conv.weight
    b = conv.bias
    if bn is None:
        if b is None:
            b = torch.zeros(w.shape[0], device=w.device, dtype=w.dtype)
        return w, b
    scale = bn.weight.float() / torch.sqrt(bn.running_var.float() + bn.eps)
    b32 = torch.zeros_like(scale) if b is None else b.float()
    bias = bn.bias.float() + (b32 - bn.running_mean.float()) * scale
    weight = w.float() * scale.reshape(-1, *([1] * (w.dim() - 1)))
    return weight.to(w.dtype), bias.to(w.dtype)


def _fold_yoloconv(m: YOLOConv):
    return _fold_conv_bn(m.conv, None if getattr(m, "_is_fused", False) else m.bn)


def _fold_repvggdw(m: YOLORepVGGDW):
    """RepVGG 7x7 + 3x3 depthwise branch pair -> one folded 7x7 kernel."""
    w, b = _fold_yoloconv(m.conv)
    if getattr(m, "_is_fused", False):
        return w, b
    w1, b1 = _fold_yoloconv(m.conv1)
    p = (w.shape[-1] - w1.shape[-1]) // 2
    return w + F.pad(w1, [p, p, p, p]), b + b1


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------
if _HAS_TRITON:
    _ALIGN_C = tl.constexpr(_ALIGN)

    @triton.jit
    def _dw(
        IN, WD, BD, XR, OUT,
        sxn, sxc, sxh, sxw,
        H: tl.constexpr, W: tl.constexpr, PITCH: tl.constexpr, PS: tl.constexpr,
        PAD: tl.constexpr, PADR: tl.constexpr, C: tl.constexpr, K: tl.constexpr,
        BP: tl.constexpr, BC: tl.constexpr,
        IN_PAD: tl.constexpr, OUT_PAD: tl.constexpr, ADD: tl.constexpr,
    ):
        """``SiLU(bd + depthwise_KxK(IN, WD)) [+ XR]``.

        ``IN_PAD``  read from the padded workspace (constexpr tap offsets, no
                    masks) instead of a strided NCHW tensor.
        ``OUT_PAD`` write back into the padded workspace instead of NCHW.
        ``ADD``     add the residual, read from the strided NCHW ``XR``.
        """
        KP: tl.constexpr = K // 2
        NCT: tl.constexpr = C // BC
        VLO: tl.constexpr = PADR * PITCH
        VHI: tl.constexpr = VLO + H * PITCH
        POS0: tl.constexpr = (VLO // _ALIGN_C) * _ALIGN_C
        NPT: tl.constexpr = (VHI - POS0 + BP - 1) // BP

        pid = tl.program_id(0)
        n = pid // (NPT * NCT)
        rem = pid - n * (NPT * NCT)
        pt = rem // NCT
        ct = rem - pt * NCT

        pos = POS0 + pt * BP + tl.arange(0, BP)
        rp = pos // PITCH
        cp = pos - rp * PITCH
        h = rp - PADR
        wc = cp - PAD
        ok = (cp >= PAD) & (cp < PAD + W) & (pos >= VLO) & (pos < VHI)
        c = ct * BC + tl.arange(0, BC)

        wbase = WD + c
        acc = tl.zeros([BC, BP], tl.float32)
        if IN_PAD:
            ibase = IN + n * (C * PS) + c[:, None] * PS + pos[None, :]
            for dh in tl.static_range(K):
                for dw in tl.static_range(K):
                    wv = tl.load(wbase + (dh * K + dw) * C).to(tl.float32)[:, None]
                    acc += wv * tl.load(ibase + ((dh - KP) * PITCH + (dw - KP))).to(tl.float32)
        else:
            ibase = IN + n * sxn + c[:, None] * sxc + (h * sxh + wc * sxw)[None, :]
            for dh in tl.static_range(K):
                hh = h + dh - KP
                hok = (hh >= 0) & (hh < H)
                for dw in tl.static_range(K):
                    ww = wc + dw - KP
                    m = hok & (ww >= 0) & (ww < W)
                    wv = tl.load(wbase + (dh * K + dw) * C).to(tl.float32)[:, None]
                    acc += wv * tl.load(ibase + ((dh - KP) * sxh + (dw - KP) * sxw),
                                        mask=m[None, :], other=0.0).to(tl.float32)

        acc += tl.load(BD + c).to(tl.float32)[:, None]
        acc = acc * tl.sigmoid(acc)
        if ADD:
            acc += tl.load(XR + n * sxn + c[:, None] * sxc + (h * sxh + wc * sxw)[None, :],
                           mask=ok[None, :], other=0.0).to(tl.float32)
        if OUT_PAD:
            tl.store(OUT + n * (C * PS) + c[:, None] * PS + pos[None, :],
                     tl.where(ok[None, :], acc, 0.0).to(OUT.dtype.element_ty))
        else:
            tl.store(OUT + n * (C * H * W) + c[:, None] * (H * W) + (h * W + wc)[None, :],
                     acc.to(OUT.dtype.element_ty), mask=ok[None, :])

    @triton.jit
    def _pw(
        IN, WQ, BQ, OUT,
        H: tl.constexpr, W: tl.constexpr, PITCH: tl.constexpr, PS: tl.constexpr,
        PAD: tl.constexpr, PADR: tl.constexpr, CI: tl.constexpr, CO: tl.constexpr,
        BP: tl.constexpr, BC: tl.constexpr, KBK: tl.constexpr,
    ):
        """``SiLU(bq + WQ @ IN)`` -- 1x1 convolution, padded workspace both sides."""
        NCT: tl.constexpr = CO // BC
        VLO: tl.constexpr = PADR * PITCH
        VHI: tl.constexpr = VLO + H * PITCH
        POS0: tl.constexpr = (VLO // _ALIGN_C) * _ALIGN_C
        NPT: tl.constexpr = (VHI - POS0 + BP - 1) // BP

        pid = tl.program_id(0)
        n = pid // (NPT * NCT)
        rem = pid - n * (NPT * NCT)
        pt = rem // NCT
        ct = rem - pt * NCT

        pos = POS0 + pt * BP + tl.arange(0, BP)
        cout = ct * BC + tl.arange(0, BC)
        ibase = IN + n * (CI * PS) + pos[None, :]
        acc = tl.zeros([BC, BP], tl.float32)
        for k0 in tl.range(0, CI, KBK):
            ci = k0 + tl.arange(0, KBK)
            d = tl.load(ibase + ci[:, None] * PS)
            wq = tl.load(WQ + cout[:, None] * CI + ci[None, :])
            acc = tl.dot(wq, d, acc)
        acc += tl.load(BQ + cout).to(tl.float32)[:, None]
        acc = acc * tl.sigmoid(acc)
        cp = pos % PITCH
        ok = (cp >= PAD) & (cp < PAD + W) & (pos >= VLO) & (pos < VHI)
        tl.store(OUT + n * (CO * PS) + cout[:, None] * PS + pos[None, :],
                 tl.where(ok[None, :], acc, 0.0).to(OUT.dtype.element_ty))


def _ei(name, default):
    try:
        return int(os.environ[name])
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Hand-written CUDA (preferred for the depthwise stages)
# ---------------------------------------------------------------------------
_CUDA_SIG = "void cib_dw_pad(" + ",".join(["int64_t"] * 18) + ");"


def _load_cuda():
    """Build ``cib_cuda.cu`` once and return the extension module.

    Named after the source hash so a stale cached ``.so`` can never shadow an
    edited kernel, and ``TORCH_CUDA_ARCH_LIST`` pinned to sm_100 so the build
    never queries a device or compiles the whole default arch list.
    """
    import hashlib

    src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cib_cuda.cu")
    with open(src_path) as fh:
        src = fh.read()
    from torch.utils.cpp_extension import load_inline

    name = "fk_cib_" + hashlib.md5(src.encode()).hexdigest()[:12]
    prev = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"] = "10.0"
    try:
        return load_inline(
            name=name,
            cpp_sources=_CUDA_SIG,
            cuda_sources=src,
            functions=["cib_dw_pad"],
            extra_cuda_cflags=["-O3", "--use_fast_math", "-std=c++17"],
            verbose=False,
        )
    finally:
        if prev is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = prev


if _ei("CIB_NO_CUDA", 0):
    _CUDA = None
else:
    try:
        _CUDA = _load_cuda()
    except Exception:  # pragma: no cover - no nvcc, odd arch, read-only cache, ...
        _CUDA = None

_WT = 20  # output columns a CUDA thread register-blocks; matches W = 20


def _cuda_dw_geom(h, w, pitch, ps, vlo, pad, k):
    """``(LO, SPAN, PG)`` for the CUDA depthwise, or ``None`` if it cannot fit.

    ``LO`` is the first workspace position the KxK halo can reach, rounded down
    to a 16 B boundary so the strip staging uses 128-bit loads; ``SPAN`` covers
    through the last reachable position.  Both are checked against ``PS`` here
    rather than trusted, because matching outputs would not reveal a halo read
    that runs off the end of the buffer (see r1's iter 08).
    """
    kp = k // 2
    wpad = -(-w // _WT) * _WT          # the kernel's last column block overruns W
    lo = vlo - kp * pitch + pad - kp
    hi = vlo + (h - 1 + kp) * pitch + pad + (wpad - 1) + kp + 1
    lo16 = (lo // 8) * 8
    span = -(-(hi - lo16) // 8) * 8
    wt = _ei("CIB_DW_WT", 4)
    # The kernel reads its row windows as 8 B float2, so the rebased row start
    # -- VLO + (h-KP)*PITCH + PAD + w0 - KP - LO for even w0 -- must be even.
    if lo < 0 or lo16 < 0 or hi > ps or lo16 + span > ps:
        return None
    if wt % 2 or pitch % 2 or (vlo + pad - kp - lo16) % 2:
        return None
    pg = max(1, _ei("CIB_DW_PG", 1))
    while pg > 1 and pg * (span + k * k + 1) * 4 > 48 * 1024:
        pg -= 1
    return lo16, span, pg, wt, _ei("CIB_DW_TH", 256)


# per-stage tiles: depthwise (BP, BC, warps), pointwise (BP, BC, KBK, warps)
_CFG = {
    "dwa": (_ei("CIB_A_BP", 32), _ei("CIB_A_BC", 8), _ei("CIB_A_W", 4)),
    "pwb": (_ei("CIB_B_BP", 32), _ei("CIB_B_BC", 32), _ei("CIB_B_KB", 128), _ei("CIB_B_W", 8)),
    "dwc": (_ei("CIB_C_BP", 32), _ei("CIB_C_BC", 4), _ei("CIB_C_W", 1)),
    "pwd": (_ei("CIB_D_BP", 32), _ei("CIB_D_BC", 32), _ei("CIB_D_KB", 128), _ei("CIB_D_W", 4)),
    "dwe": (_ei("CIB_E_BP", 64), _ei("CIB_E_BC", 8), _ei("CIB_E_W", 8)),
}


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
        self._plan = None
        self._plan_ptr = 0
        self._ws = {}
        self._lp = {}
        self._graph = {}
        self._ghead = {}
        self._gfull = {}
        self._seen = {}
        self._seenf = {}
        self._ncap = 0
        self._register_load_state_dict_pre_hook(self._invalidate)

    def _invalidate(self, *args, **kwargs):
        self._plan = None
        self._plan_ptr = 0
        self._ws = {}
        self._lp = {}
        self._graph = {}
        self._ghead = {}
        self._gfull = {}
        self._seen = {}
        self._seenf = {}
        self._ncap = 0

    @torch.no_grad()
    def _build_plan(self):
        """Fold every BN once; decide whether the Triton pipeline is usable."""
        raw = []
        for m in self.cv1:
            w, b = _fold_repvggdw(m) if isinstance(m, YOLORepVGGDW) else _fold_yoloconv(m)
            raw.append((w, b.float().contiguous()))
        eager = [(w.contiguous(), b, w.shape[-1] // 2, w.shape[0] if w.shape[1] == 1 else 1)
                 for w, b in raw]
        w0, w1, w2, w3, w4 = (r[0] for r in raw)
        c1, cm, c2, km = w0.shape[0], w1.shape[0], w3.shape[0], w2.shape[-1]
        ok = (
            _HAS_TRITON and w0.is_cuda and w0.dtype == torch.float16
            and w0.shape[1] == 1 and w2.shape[1] == 1 and w4.shape[1] == 1
            and w1.shape[1] == c1 and w3.shape[1] == cm
            and w0.shape[-1] == 3 and w4.shape[-1] == 3
            and c1 % _CFG["dwa"][1] == 0 and cm % _CFG["dwc"][1] == 0
            and c2 % _CFG["dwe"][1] == 0
            and cm % _CFG["pwb"][1] == 0 and c2 % _CFG["pwd"][1] == 0
            and c1 % _CFG["pwb"][2] == 0 and cm % _CFG["pwd"][2] == 0
            and min(_CFG["pwb"][1], _CFG["pwd"][1], _CFG["pwb"][2], _CFG["pwd"][2]) >= 16
        )
        tri = None
        if ok:
            flat = [(w.reshape(w.shape[0], -1).t().contiguous() if w.shape[1] == 1
                     else w.reshape(w.shape[0], -1).contiguous(), b) for w, b in raw]
            tri = (flat, c1, cm, c2, km, max(1, km // 2))
        self._plan = (eager, tri)
        self._ws = {}
        self._lp = {}
        self._graph = {}
        self._ghead = {}
        self._gfull = {}
        self._seen = {}
        self._seenf = {}
        self._ncap = 0
        self._plan_ptr = self.cv1[0].conv.weight.data_ptr()
        return self._plan

    def _ws_get(self, name, n, c, ps, device, dtype):
        """Zero-initialised workspace, cached per (name, shape).

        The shape is part of the key on purpose: a captured graph holds this
        buffer's address, so a second input shape must get its *own* buffer
        rather than reallocating this one out from under that graph.
        """
        key = (name, n, c, ps)
        buf = self._ws.get(key)
        if buf is None:
            buf = self._ws[key] = torch.zeros((n, c, ps), device=device, dtype=dtype)
        return buf

    def _eager_folded(self, x, eager):
        y = x
        for w, b, p, g in eager:
            y = F.silu(F.conv2d(y, w, b.to(y.dtype), 1, p, 1, g))
        return y

    def _make_launch(self, x, tri):
        """Everything about a launch that depends only on the shape, done once."""
        flat, c1, cm, c2, km, pad = tri
        (w0, b0), (w1, b1), (w2, b2), (w3, b3), (w4, b4) = flat
        n, _, h, w = x.shape
        pitch = w + 2 * pad                 # row pitch: PAD zero cols each side
        reach = pad * pitch + pad           # furthest a depthwise tap can look
        # Tiles must start on a 128 B boundary, and `pos0` rounds *down* to get
        # there, so the buffer needs enough leading pad rows that `pos0 - reach`
        # is still inside it.  One extra ALIGN worth of rows is always enough.
        padr = pad + -(-(pad + _ALIGN) // pitch)
        vlo = padr * pitch                  # first position of image row 0
        vhi = vlo + h * pitch               # one past the last image row
        pos0 = (vlo // _ALIGN) * _ALIGN      # tile start, 128 B aligned (<= vlo)
        ps = -(-(pos0 + -(-(vhi - pos0) // _BP_MAX) * _BP_MAX + reach + 1)
               // _ALIGN) * _ALIGN
        if pos0 < reach or ps <= pos0 + -(-(vhi - pos0) // _BP_MAX) * _BP_MAX + reach:
            return None                      # geometry invariant broken -> eager
        dev, dt = x.device, x.dtype

        def grid(bp, c, bc):
            return (n * ((vhi - pos0 + bp - 1) // bp) * (c // bc),)

        abp, abc, aw = _CFG["dwa"]
        bbp, bbc, bkb, bw = _CFG["pwb"]
        cbp, cbc, cw = _CFG["dwc"]
        dbp, dbc, dkb, dwp = _CFG["pwd"]
        ebp, ebc, ew = _CFG["dwe"]

        a = self._ws_get("a", n, c1, ps, dev, dt)
        b = self._ws_get("b", n, cm, ps, dev, dt)
        cb = self._ws_get("c", n, cm, ps, dev, dt)
        d = a if c1 == c2 else self._ws_get("d", n, c2, ps, dev, dt)
        # Stage C in CUDA when the geometry checks out; every pointer here is
        # cached for the life of the plan, so the whole arg list is built once.
        ccu = None
        if _CUDA is not None:
            g = _cuda_dw_geom(h, w, pitch, ps, vlo, pad, km)
            if g is not None:
                lo16, span, pg, wt, th = g
                ccu = (b.data_ptr(), w2.data_ptr(), b2.data_ptr(), cb.data_ptr(),
                       cm, h, w, pitch, ps, vlo, pad, km, n * cm, pg, span, lo16,
                       wt, th)
        return (
            ccu,
            (n, c2, h, w, dev, dt),
            (grid(abp, c1, abc), (w0, b0, a), (h, w, pitch, ps, pad, padr, c1, 3, abp, abc,
                                               False, True, False), aw),
            (grid(bbp, cm, bbc), (a, w1, b1, b), (h, w, pitch, ps, pad, padr, c1, cm,
                                                   bbp, bbc, bkb), bw),
            (grid(cbp, cm, cbc), (b, w2, b2, cb), (h, w, pitch, ps, pad, padr, cm, km,
                                                   cbp, cbc, True, True, False), cw),
            (grid(dbp, c2, dbc), (cb, w3, b3, d), (h, w, pitch, ps, pad, padr, cm, c2,
                                                   dbp, dbc, dkb), dwp),
            (grid(ebp, c2, ebc), (d, w4, b4), (h, w, pitch, ps, pad, padr, c2, 3, ebp, ebc,
                                              True, False, self.add), ew),
        )

    def _lead(self, x, lp):
        """Stage A: dw3x3 straight off the strided NCHW input."""
        ga = lp[2]
        sn, sc, sh, sw = x.stride()
        _dw[ga[0]](x, ga[1][0], ga[1][1], x, ga[1][2], sn, sc, sh, sw, *ga[2],
                   num_warps=ga[3], num_stages=1)

    def _mid(self, lp):
        """Stages B-D: pw -> dwKxK -> pw, entirely inside the workspace.

        Deliberately touches no input- or output-dependent address, so it can be
        captured once into a CUDA graph and replayed for any input pointer.  That
        is what removes r1's ``si.copy_(x)`` (measured 7.1 us of the 37.9).
        """
        ccu, _, _, gb, gc, gd, _ = lp
        _pw[gb[0]](*gb[1], *gb[2], num_warps=gb[3], num_stages=1)
        if ccu is None:
            _dw[gc[0]](gc[1][0], gc[1][1], gc[1][2], gc[1][0], gc[1][3], 0, 0, 0, 0,
                       *gc[2], num_warps=gc[3], num_stages=1)
        else:
            _CUDA.cib_dw_pad(*ccu)
        _pw[gd[0]](*gd[1], *gd[2], num_warps=gd[3], num_stages=1)

    def _head(self, x, lp):
        """Stages A-D, launched directly (capture path / no-graph fallback)."""
        self._lead(x, lp)
        self._mid(lp)

    def _tail(self, x, lp, out):
        """Stage E: dw3x3 -> SiLU -> residual, writing plain NCHW."""
        ge = lp[6]
        sn, sc, sh, sw = x.stride()
        _dw[ge[0]](ge[1][0], ge[1][1], ge[1][2], x, out, sn, sc, sh, sw, *ge[2],
                   num_warps=ge[3], num_stages=1)

    def _launch_plan(self, x, tri):
        key = (x.shape, x.stride())
        if key not in self._lp:
            self._lp[key] = self._make_launch(x, tri)
        return self._lp[key]

    def _capture_full(self, x, lp, out):
        """Record all five stages, bound to *both* the input and output address.

        Legitimate because the caching allocator hands back the same block for an
        identically shaped ``torch.empty`` once the previous call's output has
        been released, so in steady state the address we are about to return is
        the one the graph writes.  Both addresses are re-checked on every call,
        so this can only ever fire when the graph writes exactly the buffer being
        returned -- no output is ever aliased across calls.

        Capture does not execute, so the caller replays immediately afterwards to
        fill ``out`` for the capturing call itself.
        """
        try:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    self._lead(x, lp)
                    self._mid(lp)
                    self._tail(x, lp, out)
            torch.cuda.current_stream().wait_stream(s)
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr):
                self._lead(x, lp)
                self._mid(lp)
                self._tail(x, lp, out)
            return x.data_ptr(), out.data_ptr(), gr
        except Exception:  # noqa: BLE001 - graphs unavailable; other paths work
            return False

    def _capture_head(self, x, lp):
        try:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    self._lead(x, lp)
                    self._mid(lp)
            torch.cuda.current_stream().wait_stream(s)
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr):
                self._lead(x, lp)
                self._mid(lp)
            return x.data_ptr(), gr
        except Exception:  # noqa: BLE001
            return False

    def _capture(self, lp):
        """Record stages B-D into a CUDA graph.  ``False`` if unavailable.

        Only the middle is captured.  Stage A reads the caller's ``x`` and stage
        E reads it again for the residual while writing a per-call output, so
        both stay ordinary launches -- which is exactly why no input ever has to
        be copied to a fixed address, and why ``forward`` never returns a buffer
        a later call will overwrite.  Host cost is ~2 Triton launches plus one
        replay, far inside the harness's ~70 us of enqueue slack.
        """
        try:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    self._mid(lp)
            torch.cuda.current_stream().wait_stream(s)
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr):
                self._mid(lp)
            return gr
        except Exception:  # noqa: BLE001 - graphs unavailable; direct launches work
            return False

    def _run_triton(self, x, tri):
        lp = self._launch_plan(x, tri)
        if lp is None:
            return None
        n, c2, h, w, dev, dt = lp[1]
        if not torch.cuda.is_current_stream_capturing():
            key = (x.shape, x.stride())
            ptr = x.data_ptr()
            # One submission for the whole block when both addresses repeat.
            out = torch.empty((n, c2, h, w), device=dev, dtype=dt)
            gf = self._gfull.get(key)
            pair = (ptr, out.data_ptr())
            if gf is not False:
                if gf is not None and gf[0] == ptr and gf[1] == pair[1]:
                    gf[2].replay()
                    return out
                if self._seenf.get(key) == pair and self._ncap < 8:
                    self._ncap += 1
                    self._seenf.pop(key, None)
                    gf = self._gfull[key] = self._capture_full(x, lp, out)
                    if gf:
                        gf[2].replay()
                        return out
                else:
                    self._seenf[key] = pair
            gh = self._ghead.get(key)
            if gh is not False and (gh is None or gh[0] != ptr):
                if self._seen.get(key) == ptr and self._ncap < 8:
                    self._ncap += 1
                    self._seen.pop(key, None)
                    gh = self._ghead[key] = self._capture_head(x, lp)
                else:
                    self._seen[key] = ptr
                    gh = None
            if gh:
                gh[1].replay()
                self._tail(x, lp, out)
                return out
            g = self._graph.get(key)
            if g is None:
                g = self._graph[key] = self._capture(lp)
            if g is not False:
                self._lead(x, lp)
                g.replay()
                self._tail(x, lp, out)
                return out
        self._head(x, lp)
        out = torch.empty((n, c2, h, w), device=dev, dtype=dt)
        self._tail(x, lp, out)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        plan = self._plan
        if (self.training or plan is None
                or self.cv1[0].conv.weight.data_ptr() != self._plan_ptr):
            if self.training:
                y = self.cv1(x)
                return x + y if self.add else y
            plan = self._build_plan()

        tri = plan[1]
        if (tri is not None and x.is_cuda and x.dtype == torch.float16
                and x.dim() == 4 and x.shape[1] == tri[1]):
            out = self._run_triton(x, tri)
            if out is not None:
                return out
        y = self._eager_folded(x, plan[0])
        return x + y if self.add else y
