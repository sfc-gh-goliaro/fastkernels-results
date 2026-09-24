"""YOLOv10 backbone: the eleven-block composition as ONE backbone-wide launch.

Every block here is a frozen L2 winner, imported as-is -- ``YOLOConv``,
``YOLOC2f``, ``YOLOSCDown``, ``YOLOSPPF``, ``YOLOPSA`` are each 2-6x over their
native form already and none of their kernels is touched.  What this level owns
is the *seam between them*, and the measurement that sets the whole strategy is
that the seams, not the kernels, are the window:

    n=4   whole-module window 344.5 us    host time to issue it 343.5 us
    n=1   whole-module window 308.6 us    host time to issue it 340.4 us

The window *is* the Python/dispatch cost.  Summed GPU kernel time for the same
forward is 297 us (n=4) and 171 us (n=1), so at batch 1 more than half the
window is the device sitting idle waiting for the host, and at batch 4 the two
costs merely happen to be similar.  Eleven blocks each independently revalidate
a fold signature, re-enter Python, and (``YOLOC2f``) copy into their own graph's
static input buffer and clone the static output back out -- 8 device-to-device
memcpys, 17.4 us of pure seam traffic at n=4.

So: capture the whole ``stem1..psa`` chain once per (shape, dtype, device) into
one CUDA graph and replay it.  One input copy, one graph launch, three output
views -- five host operations for the entire backbone, whatever it costs on the
device.

Choosing what to capture
------------------------
The blocks each ship several internal paths, and the one that is fastest
*standalone* is not the one to capture, because standalone each is paying its
own host cost and inside the outer graph none of them is.  Measured total GPU
kernel time per forward (``dev/p3_modes.py``, sum of self device time over 20
forwards):

    C2f path            n=1        n=4
    mega + graph      171.4 us   297.4 us     (as shipped, ``auto``)
    mega, no graph    171.4 us   273.3 us
    graph, no mega    163.1 us   298.2 us
    flat (neither)    150.9 us   272.7 us   <- captured

``YOLOC2f``'s megakernel is the right call at L2 -- one launch for a whole block
is worth a lot when every launch costs the host 13 us -- but its grid is pinned
at one CTA per SM and does not grow with the problem, so as *device* work it is
20.5-33.8 us per block against 84.6 us for all twenty flat per-conv launches at
n=1.  Once the outer graph has deleted the host cost the megakernel is only its
fixed grid, and it loses.  It is also not replayable: the grid-wide barrier
compares a never-reset arrival counter against a host-computed
``epoch * grid`` target passed as a kernel argument, so a capture bakes one
epoch's target in and every replay after the first would sail through the
barriers on stale counts.  The same goes for ``YOLOC2f``'s own whole-block
graph, which would nest a replay inside a capture.  Both are avoided by calling
``_get_plan()`` / ``_fused()`` directly rather than the block's ``forward``,
which is also what deletes the 8 seam memcpys: ``_fused`` allocates from the
outer graph's private pool, so p2/p3/p4/p5 and every intermediate flow
block-to-block with no copy-in and no clone-out.

``YOLOSCDown``'s raw ``cuLaunchKernel``, ``YOLOSPPF``'s pybind entry (three
PDL-chained ``cudaLaunchKernelEx``) and ``YOLOPSA``'s five direct Triton
launches (also PDL) all take ``getCurrentCUDAStream()`` and record cleanly into
the outer capture -- verified numerically against the baseline rather than
assumed, since a PDL edge that silently degrades to no dependency would be a
race, not an error.

What is left after that, and what was done about it
---------------------------------------------------
With the graph in place the host is 5-8x ahead of the device (32 us of Python
against 170-280 us of kernels) and the window is *entirely* device time.  Two
things follow, and both were measured rather than guessed:

* **Node count is not a lever.**  The chrome trace of one replay
  (``dev/p10_gaps.py``) puts the 31-kernel span at 143.20 us against a kernel
  sum of 146.24 us at n=1 -- the boundaries overlap, they do not cost.  What
  separates a replay's wall time from its kernel sum is the harness' 253 MiB
  ``l2.zero_()`` before every timed iteration: 165.8 us with the flush, 144.5 us
  without.  So there was no launch-boundary cost for seam fusion to recover, and
  none was attempted.
* **The frozen *gates* are mistuned here even though the frozen kernels are
  not.**  Both were swept on their own level's captured shapes, and this level
  feeds them shapes 4-64x larger.  Retuning only the constexpr bundle and launch
  config -- no kernel forked -- is worth 27.2 us at n=4 and 4.0 us at n=1 on the
  twenty ``YOLOC2f`` dense convs, and 1.1 / 3.6 us on the three stride-2 stems.
  See ``_DENSE_CFG`` and ``_STEM_CFG`` for the tables, the mechanisms, and which
  of them generalise.

That left one lever named and unopened, and this round took it.

Two new 3x3 kernels
-------------------
Every 3x3 here -- twelve dense convs inside the four ``YOLOC2f`` blocks and the
three stride-2 stems, together 56% of device time -- ran on an implicit GEMM that
gathers its A tile once PER TAP.  ncu says the cost of that is not L1 bandwidth
(DRAM is at 1%, and the four dense 3x3s sit at 33-70% of L1) but REGISTER
PRESSURE: ``ih``, ``iw`` and a four-way bounds mask are each a full
``[BLOCK_K, BLOCK_P]`` register tile, rebuilt per tap, which caps occupancy at
one to five CTAs per SM and leaves the kernel latency-bound at 0.3-2.2 waves.

``conv3x3.cu`` holds the replacements -- NVRTC-compiled per (shape, config) and
launched with ``cuLaunchKernel`` on ``current_stream()``, the same mechanics the
frozen ``YOLOSCDown`` uses and that round 1 verified record into the capture
above.  Both stage the input patch in shared memory once and take the nine taps
from there; that file's header carries the layout derivation, including the
measurement that rules out the obvious design (``ldmatrix`` needs 16-byte-aligned
row addresses and a 3x3's three horizontal taps are three consecutive column
offsets, so at most one can be aligned -- offsets 1, 2, 3, 7 and 9 fault on this
device).  ``KIND 0`` is a direct-FMA form for stem1, whose ``C = 3`` puts
``tl.dot`` at 19% MMA efficiency; ``KIND 1`` is ``mma.sync.m16n8k16`` against a
channel-pair-interleaved patch, parity-split at stride 2.

They are opt-in per shape behind ``_NEW_STEM`` / ``_NEW_DENSE``, swept offline, so
an unkeyed shape never reaches them and is never worse than round 1.  Worth
1.03-1.59x on the eleven shapes that are in and 0.2715 -> 0.2586 ms end to end at
n=4, 0.1772 -> 0.1670 at n=1.  ``FK_BB_NEW3X3=0`` turns both off.

What bounds all of this: an EMPTY Triton launch is 1.15 us of device time on this
device and a pure copy of these tensors is 1.55-1.94 us whether it moves 0.2 MB or
6.5 MB (``dev/p_floor.py``).  Every conv here has a DRAM roofline under 1 us, so
the roofline is not the target -- the launch floor is, and eleven blocks of it is
already 18 us of the 217.

Contracts
---------
Every parameter and buffer keeps its baseline name and nesting, so the harness'
``load_state_dict(baseline.state_dict(), strict=False)`` shares weights exactly.
``forward`` returns the ``p3_backbone``/``p4_backbone``/``p5_backbone`` dict.
The three values are the captured graph's own output tensors: they stay
allocated for the life of the module (the graph's private pool holds them) and
are overwritten by the *next* replay, which is the standard graph-inference
contract and what ``FK_BB_CLONE=1`` exists to opt out of.

Fallbacks, in order: replay -> capture -> the flat chain eagerly -> the literal
baseline composition, which is what handles training mode, grad-enabled
autograd, CPU tensors, fp32/bf16 and anything else the fused paths decline.

That last step is not simply "call each block's ``forward``".  Four breaks in the
frozen stack sit under it, all four inherited (``dev/p17_robust.py`` with
``R0=1`` runs the same matrix against the round-0 kernel) and all four fixed here
because ``candidate/L1`` and ``candidate/L2`` are frozen: a ``C = 3`` crash in
L1's padded conv that bf16 and training mode both fall onto (``_conv_ref``),
three blocks that fold running statistics without consulting ``self.training``
(``_c2f_train`` and ``_baseline``), ``YOLOSPPF`` caching a folded weight behind
no guard at all (``_sppf_reset``), and L1's ``Softmax`` launching Triton at a CPU
pointer (``_cpu_safe``).  Each has its own docstring below with the measurement
that found it.

One difference is left as inherited: under **eval with grad enabled**, ``YOLOPSA``
correctly declines its non-differentiable fused path and its ``_eager`` fallback
is 6.4e-02 relative against an fp32 reference where the native baseline is
1.25e-03.  It is bit-identical in round 0, it is inside a frozen block, and the
harness times under ``no_grad``; bypassing PSA's own grad guard would trade that
for a silently non-differentiable module.
"""

from __future__ import annotations

import contextlib
import ctypes
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except ImportError:  # noqa: BLE001
    triton = None
    tl = None

from ..L2 import yolov10_c2f as _c2fmod
from ..L2 import yolov10_conv as _convmod
from ..L2 import yolov10_sppf as _sppfmod
from ..L2.yolov10_c2f import YOLOC2f
from ..L2.yolov10_conv import YOLOConv
from ..L2.yolov10_psa import YOLOPSA
from ..L2.yolov10_scdown import YOLOSCDown
from ..L2.yolov10_sppf import YOLOSPPF

# Knobs, for attributing each lever in its own bench rather than for tuning.
_USE_GRAPH = os.environ.get("FK_BB_GRAPH", "1") != "0"
_USE_FLAT = os.environ.get("FK_BB_FLAT", "1") != "0"
# Return clones of the graph's outputs instead of the outputs themselves.
_CLONE_OUT = os.environ.get("FK_BB_CLONE", "0") != "0"
_MAX_GRAPHS = 8
_MISSING = object()
_OUT_KEYS = ("p3_backbone", "p4_backbone", "p5_backbone")


# ---------------------------------------------------------------------------
# The new 3x3: NVRTC compile + launch for ``conv3x3.cu``.
#
# Same mechanics the frozen ``YOLOSCDown`` uses and that round 1 verified
# records cleanly inside the whole-backbone capture: NVRTC once per
# (shape, config) at plan time, then ``cuLaunchKernel`` on
# ``current_stream()`` with a fixed argv whose cells are rewritten per call.
# No allocation, no sync and no host branch on a tensor *value* inside forward.
# ---------------------------------------------------------------------------
_C3_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conv3x3.cu")
_C3_KERNELS: dict = {}
_C3_LIBS: list = []
_C3_OFF = os.environ.get("FK_BB_NEW3X3", "1") == "0"


def _c3_libs():
    """(libnvrtc, libcuda) with ``cuLaunchKernel`` argtypes pinned."""
    if _C3_LIBS:
        return _C3_LIBS[0]
    major = str(torch.version.cuda or "12").split(".")[0]
    nvrtc = None
    for name in (f"libnvrtc.so.{major}", "libnvrtc.so"):
        try:
            nvrtc = ctypes.CDLL(name)
            break
        except OSError:
            continue
    if nvrtc is None:
        raise OSError("libnvrtc not found")
    cuda = ctypes.CDLL("libcuda.so.1")
    cuda.cuLaunchKernel.restype = ctypes.c_int
    cuda.cuLaunchKernel.argtypes = [
        ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
        ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ]
    _C3_LIBS.append((nvrtc, cuda))
    return _C3_LIBS[0]


def _c3_cubin(src: str, arch: str) -> bytes:
    nvrtc, _ = _c3_libs()
    prog = ctypes.c_void_p()
    if nvrtc.nvrtcCreateProgram(ctypes.byref(prog), src.encode(), b"conv3x3.cu",
                                0, None, None) != 0:
        raise RuntimeError("nvrtcCreateProgram failed")
    opts = [f"--gpu-architecture={arch}".encode(), b"-default-device",
            b"--std=c++17"]
    arr = (ctypes.c_char_p * len(opts))(*opts)
    if nvrtc.nvrtcCompileProgram(prog, len(opts), arr) != 0:
        n = ctypes.c_size_t()
        nvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(n))
        log = ctypes.create_string_buffer(n.value)
        nvrtc.nvrtcGetProgramLog(prog, log)
        raise RuntimeError("NVRTC failed:\n" + log.value.decode(errors="replace"))
    n = ctypes.c_size_t()
    nvrtc.nvrtcGetCUBINSize(prog, ctypes.byref(n))
    buf = ctypes.create_string_buffer(n.value)
    nvrtc.nvrtcGetCUBIN(prog, buf)
    nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))
    return buf.raw


class _C3Kernel:
    """A compiled ``conv3x3``, shared process-wide by every plan with this key."""

    __slots__ = ("fn", "mod", "nthread", "gx", "gy", "smem", "launch")

    def __init__(self, fn, mod, nthread, gx, gy, smem):
        self.fn, self.mod = fn, mod
        self.nthread, self.gx, self.gy, self.smem = nthread, gx, gy, smem
        self.launch = _c3_libs()[1].cuLaunchKernel


def _c3_compile(defs: dict, nthread: int, gx: int, gy: int, smem: int):
    key = tuple(sorted(defs.items()))
    ker = _C3_KERNELS.get(key, False)
    if ker is not False:
        return ker
    try:
        # int(), not str(): a Python bool renders as ``True``, which the
        # preprocessor treats as an undefined identifier -- so ``#if ACT`` would
        # silently evaluate to 0 and the activation would vanish with no error
        # anywhere.  ``dev/p_c2f.py`` caught exactly that (3.5e-02 against the
        # block's own reference) because ``_c2fmod._Conv.act`` is a bool.
        header = "".join(f"#define {k} {int(v)}\n" for k, v in defs.items())
        with open(_C3_SRC) as fh:
            src = header + fh.read()
        p = torch.cuda.get_device_properties(torch.cuda.current_device())
        arch = f"sm_{p.major}{p.minor}a" if p.major >= 9 else f"sm_{p.major}{p.minor}"
        _, cuda = _c3_libs()
        cubin = _c3_cubin(src, arch)
        mod = ctypes.c_void_p()
        if cuda.cuModuleLoadData(ctypes.byref(mod), cubin) != 0:
            raise RuntimeError("cuModuleLoadData failed")
        fn = ctypes.c_void_p()
        if cuda.cuModuleGetFunction(ctypes.byref(fn), mod, b"conv3x3") != 0:
            raise RuntimeError("cuModuleGetFunction failed")
        if smem > 48 * 1024:  # opt in to the >48 KB dynamic shared window
            if cuda.cuFuncSetAttribute(fn, 8, smem) != 0:
                raise RuntimeError("cuFuncSetAttribute failed")
        ker = _C3Kernel(fn, mod, nthread, gx, gy, smem)
    except Exception:  # noqa: BLE001 - unrealizable config: fall through
        if os.environ.get("FK_BB_C3DEBUG"):
            raise
        ker = None
    _C3_KERNELS[key] = ker
    return ker


def _c3_frag_weight(wt: torch.Tensor, c: int) -> torch.Tensor:
    """[COUT, 9*C] -> mma A-fragment order, so a lane's fragment is 16 B.

    ``mma.m16n8k16.row.col``'s A fragment gives lane *l* (gid = l >> 2,
    tg = l & 3) the four register pairs

        reg0 = A[gid  ][2*tg], A[gid  ][2*tg+1]
        reg1 = A[gid+8][2*tg], A[gid+8][2*tg+1]
        reg2 = A[gid  ][2*tg+8], A[gid  ][2*tg+9]
        reg3 = A[gid+8][2*tg+8], A[gid+8][2*tg+9]

    of the 16x16 tile A[m][k] = w[co_tile*16 + m][tap][k_tile*16 + k] -- read off
    the ``ldmatrix.x4`` path this replaces and confirmed numerically against it.
    Laying the weight out in that order on the host turns each A fragment into
    one 16 B contiguous read, so the whole warp loads 512 B coalesced with no
    shared memory and no barrier.  Done once per (weight, version) at plan time;
    the module's signature guard drops the cache when a weight changes.
    """
    cout = int(wt.shape[0])
    w = wt.reshape(cout // 16, 16, 9, c // 16, 16).permute(0, 2, 3, 1, 4)
    lane = torch.arange(32, device=wt.device)
    gid, tg = lane >> 2, lane & 3
    rows = torch.stack([gid, gid + 8, gid, gid + 8], 1).reshape(-1)
    cols = torch.stack([2 * tg, 2 * tg, 2 * tg + 8, 2 * tg + 8], 1).reshape(-1)
    lo = w[:, :, :, rows, cols]
    hi = w[:, :, :, rows, cols + 1]
    return torch.stack([lo, hi], -1).reshape(-1).contiguous()


class _C3Plan:
    """One call site's launch state: the kernel plus its nine argument cells."""

    __slots__ = ("ker", "cells", "argv", "out_shape", "n")

    def __init__(self, ker, n, out_shape):
        self.ker, self.n, self.out_shape = ker, n, out_shape
        self.cells = [ctypes.c_void_p(0) for _ in range(8)] + [ctypes.c_float(0.0)]
        self.argv = (ctypes.c_void_p * 9)(
            *[ctypes.cast(ctypes.byref(c), ctypes.c_void_p) for c in self.cells])

    def bind(self, wt, a0, a1, a2, a3, res, eps):
        self.cells[1].value = wt.data_ptr()
        self.cells[2].value = 0 if a0 is None else a0.data_ptr()
        self.cells[3].value = 0 if a1 is None else a1.data_ptr()
        self.cells[4].value = 0 if a2 is None else a2.data_ptr()
        self.cells[5].value = 0 if a3 is None else a3.data_ptr()
        self.cells[6].value = 0 if res is None else res.data_ptr()
        self.cells[8].value = float(eps)

    def run(self, x, y):
        k = self.ker
        self.cells[0].value = x.data_ptr()
        self.cells[7].value = y.data_ptr()
        k.launch(k.fn, k.gx, k.gy, self.n, k.nthread, 1, 1, k.smem,
                 torch.cuda.current_stream().cuda_stream, self.argv, None)
        return y


def _c3_align(*vals) -> int:
    """The widest staging chunk (in halves) every value below is a multiple of."""
    for chw in (8, 4, 2):
        if all(v % chw == 0 for v in vals):
            return chw
    return 1


def _aff_flag(t) -> int:
    return 1 if (t is not None and t.dtype is torch.float32) else 0


_C3_SMEM_CAP = 227 * 1024


def _c3_plan(n, c, cout, imh, imw, s, fold, act, has_res, xsn, xsc,
             rsn, rsc, ysn, ysc, aff, cfg, ptr_align):
    """A launch plan for the new 3x3, or None if *cfg* is not realizable here.

    Everything the kernel needs is decided here -- tile shape, staging chunk
    width, shared-memory budget, the epilogue's dtype flags -- so that nothing
    per call does any sizing, gating or dtype inspection.  That is what keeps
    the launch capturable, and it is why an unkeyed shape simply never gets here.
    """
    oh, ow = (imh + 2 - 3) // s + 1, (imw + 2 - 3) // s + 1
    if cfg[0] == 0:
        return _c3_plan_k0(n, c, cout, imh, imw, s, oh, ow, fold, act, has_res,
                           xsn, xsc, rsn, rsc, ysn, ysc, aff, cfg, ptr_align)
    if cfg[0] == 1:
        return _c3_plan_k1(n, c, cout, imh, imw, s, oh, ow, fold, act, has_res,
                           xsn, xsc, rsn, rsc, ysn, ysc, aff, cfg, ptr_align)
    return None


def _pad_mod(v, m, r):
    """Smallest value >= *v* that is congruent to *r* modulo *m*."""
    return v + ((r - v) % m)


def _c3_plan_k1(n, c, cout, imh, imw, s, oh, ow, fold, act, has_res,
                xsn, xsc, rsn, rsc, ysn, ysc, aff, cfg, ptr_align):
    _, th, bco, bc, mt, nt, nwm, nwn, wsmem, minblk = cfg
    if c % 8 or c % bc or bc % 16 or cout % bco or bco != nwm * mt * 16:
        return None
    if oh % th or ow % 2:
        return None
    npix = th * ow
    if npix % 8 or (npix // 8) != nwn * nt:
        return None
    nthread = 32 * nwm * nwn
    if nthread > 1024 or nt > 16 or mt > 8:
        return None
    # 8-half global chunks need the (batch, channel, row) strides all 8-aligned;
    # a 20-wide image is not, and drops to 2-half loads rather than losing the
    # kernel.  Every other shape here keeps the 16 B path.
    gvec = min(_c3_align(imw, xsc, xsn), ptr_align)
    if gvec < 2:
        return None
    ih_t = (th - 1) * s + 3
    if s == 1:
        ncol8 = -(-(ow + 9) // 8)
        xw = 8 * ncol8
    else:
        ncol8 = -(-(2 * ow + 8) // 8)
        xw = 4 * ncol8
    rs = s * xw * 2
    # CP == 16 (mod 32) halves is what puts the four channel-pair planes on four
    # distinct bank groups; a multiple of 32 collides two ways.
    cp = _pad_mod(ih_t * rs, 32, 16)
    smem = bc // 2 * cp * 2
    if wsmem:
        smem += bco * _pad_mod(9 * bc, 16, 8) * 2
    if smem > _C3_SMEM_CAP:
        return None
    defs = dict(KIND=1, C=c, COUT=cout, IMH=imh, IMW=imw, OH=oh, OW=ow, S=s,
                PADH=1, PADW=1, TH=th, NPIX=npix, BCO=bco, BC=bc, MT=mt,
                NT=nt, NWM=nwm, NWN=nwn, NTHREAD=nthread, P=s, XW=xw, CP=cp,
                IH_T=ih_t, NCOL8=ncol8, GVEC=gvec, WSMEM=wsmem,
                XSN=xsn, XSC=xsc, RSN=rsn, RSC=rsc, YSN=ysn, YSC=ysc,
                FOLD=fold, ACT=act, HAS_RES=has_res,
                A0F=aff[0], A1F=aff[1], A2F=aff[2], A3F=aff[3])
    if minblk:
        defs["MINBLK"] = minblk
    ker = _c3_compile(defs, nthread, oh // th, cout // bco, smem)
    if ker is None:
        return None
    return _C3Plan(ker, n, (n, cout, oh, ow))


def _c3_plan_k0(n, c, cout, imh, imw, s, oh, ow, fold, act, has_res,
                xsn, xsc, rsn, rsc, ysn, ysc, aff, cfg, ptr_align):
    _, th, tw, jw, co_t, bco, wf32, cu_, minblk = cfg
    npj = tw // jw
    npx = th * npj
    if oh % th or ow % tw or cout % bco or bco % co_t or tw % jw:
        return None
    if npx % 32 or npx & (npx - 1) or (jw * s) % 8:
        return None
    nthread = npx * (bco // co_t)
    if nthread > 1024:
        return None
    chw = min(_c3_align(imw, xsc, xsn, tw * s), ptr_align)
    if chw < 2:
        return None
    ih_t = (th - 1) * s + 3
    nld = -(-(7 + (jw - 1) * s + 3) // 8)
    xs = 8 * max(-(-((tw - 1) * s + 10) // 8), ((tw - jw) * s) // 8 + nld)
    nq = xs // chw
    smem = c * ih_t * xs * 2 + bco * 9 * c * (4 if wf32 else 2)
    if smem > _C3_SMEM_CAP:
        return None
    defs = dict(KIND=0, C=c, COUT=cout, IMH=imh, IMW=imw, OH=oh, OW=ow,
                SH=s, SW=s, PADH=1, PADW=1, TH=th, TW=tw, JW=jw, CO_T=co_t,
                BCO=bco, NTHREAD=nthread, XS=xs, IH_T=ih_t, NLD=nld, NQ=nq,
                CHW=chw, CU_=cu_, WF32=wf32, XSN=xsn, XSC=xsc, RSN=rsn,
                RSC=rsc, YSN=ysn, YSC=ysc, FOLD=fold, ACT=act,
                HAS_RES=has_res, A0F=aff[0], A1F=aff[1], A2F=aff[2],
                A3F=aff[3])
    if minblk:
        defs["MINBLK"] = minblk
    ker = _c3_compile(defs, nthread, (oh // th) * (ow // tw), cout // bco, smem)
    if ker is None:
        return None
    return _C3Plan(ker, n, (n, cout, oh, ow))



# ---------------------------------------------------------------------------
# The new 3x3, Triton form: a RECTANGULAR output tile, so the input address is
# affine and the bounds mask is one-dimensional.
#
# ``_dense_kernel``'s A tile is [BLOCK_K, BLOCK_P] over k = (tap, c) and p = a
# *linear* pixel index, so ``ih``, ``iw`` and a four-way in-bounds mask are all
# full two-dimensional register tiles, recomputed for every tap.  ncu says that
# is what costs: 90-204 registers per thread on the four dense 3x3s, occupancy
# capped at 1-5 blocks per SM, 6-27% of warps resident, and the whole kernel
# latency-bound at 0.3-2.2 waves with DRAM at 1% and L1 at 33-70%.
#
# Here the CTA owns a TH x TW *rectangle* of output, so a tap's input tile is
# ``c[:, None] * x_sc + row[None, :] * IMW + col[None, :]`` where ``row`` and
# ``col`` are one-dimensional [TH*TW] vectors built once, before the loops, and
# the tap only shifts them by a scalar.  The nine taps then reuse the same halo
# out of L1 with the addressing already resolved, the reduction blocks over
# channels alone (so ``BLOCK_C`` rather than ``BLOCK_K = 9C`` sets the tile
# width), and the register file holds accumulators instead of index arithmetic.
#
# The epilogue is the union of the two call sites' contracts, so this kernel can
# stand in for either: FOLD 2 folds bn.(weight, bias, running_mean, running_var)
# in fp32 the way ``_convmod._epilogue`` does, FOLD 1 adds an already-folded
# bias, ACT is SiLU, RES is the bottleneck's residual and (y_sn, y_sc) let the
# result land in a channel slice of the shared C2f buffer with no seam copy.
# ---------------------------------------------------------------------------
if triton is not None:

    @triton.jit
    def _hc3_kernel(X, WT, A0, A1, A2, A3, RES, Y, eps,
                    x_sn, x_sc, r_sn, r_sc, y_sn, y_sc,
                    C: tl.constexpr, IMH: tl.constexpr, IMW: tl.constexpr,
                    COUT: tl.constexpr, OH: tl.constexpr, OW: tl.constexpr,
                    S: tl.constexpr, NTW: tl.constexpr,
                    TH: tl.constexpr, TW: tl.constexpr, NP: tl.constexpr,
                    BLOCK_CO: tl.constexpr, BLOCK_C: tl.constexpr,
                    NUM_CB: tl.constexpr, EVEN_C: tl.constexpr,
                    EVEN_CO: tl.constexpr, SAFE_ROW: tl.constexpr,
                    SAFE_COL: tl.constexpr, FOLD: tl.constexpr,
                    ACT: tl.constexpr, HAS_RES: tl.constexpr):
        pid = tl.program_id(0)
        pid_co = tl.program_id(1)
        n = tl.program_id(2)
        oht = pid // NTW
        owt = pid - oht * NTW
        oh0 = oht * TH
        ow0 = owt * TW

        offs_co = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
        m_co = offs_co < COUT
        # The CTA's TH x TW rectangle, flattened.  ``pr`` / ``pc`` are the only
        # index vectors in the kernel and they are built once.
        p = tl.arange(0, NP)
        pr = p // TW
        pc = p - pr * TW
        ih0 = (oh0 + pr) * S - 1
        iw0 = (ow0 + pc) * S - 1
        opix = (oh0 + pr) * OW + (ow0 + pc)

        xn = X + n * x_sn
        acc = tl.zeros((BLOCK_CO, NP), dtype=tl.float32)
        for cb in range(NUM_CB):
            offs_c = cb * BLOCK_C + tl.arange(0, BLOCK_C)
            xc = xn + offs_c[:, None] * x_sc
            wc = WT + offs_co[:, None] * (9 * C) + offs_c[None, :]
            if EVEN_C:
                m_c = m_co[:, None] if not EVEN_CO else None
            else:
                m_c = (offs_c < C)[None, :] & m_co[:, None]
            for tap in tl.static_range(9):
                ky = tap // 3
                kx = tap % 3
                ih = ih0 + ky
                iw = iw0 + kx
                ok = None
                if not SAFE_ROW:
                    ok = (ih >= 0) & (ih < IMH)
                if not SAFE_COL:
                    cok = (iw >= 0) & (iw < IMW)
                    ok = cok if ok is None else ok & cok
                aoff = ih[None, :] * IMW + iw[None, :]
                if ok is None:
                    a = tl.load(xc + aoff)
                else:
                    a = tl.load(xc + aoff, mask=ok[None, :], other=0.0)
                if EVEN_C and EVEN_CO:
                    w = tl.load(wc + tap * C)
                elif EVEN_C:
                    w = tl.load(wc + tap * C, mask=m_co[:, None], other=0.0)
                else:
                    w = tl.load(wc + tap * C, mask=m_c, other=0.0)
                acc = tl.dot(w, a, acc=acc)

        if FOLD == 2:
            if EVEN_CO:
                g = tl.load(A0 + offs_co).to(tl.float32)
                bb = tl.load(A1 + offs_co).to(tl.float32)
                mu = tl.load(A2 + offs_co).to(tl.float32)
                var = tl.load(A3 + offs_co).to(tl.float32)
            else:
                g = tl.load(A0 + offs_co, mask=m_co, other=0.0).to(tl.float32)
                bb = tl.load(A1 + offs_co, mask=m_co, other=0.0).to(tl.float32)
                mu = tl.load(A2 + offs_co, mask=m_co, other=0.0).to(tl.float32)
                var = tl.load(A3 + offs_co, mask=m_co, other=1.0).to(tl.float32)
            sc = g * tl.rsqrt(var + eps)
            acc = acc * sc[:, None] + (bb - mu * sc)[:, None]
        elif FOLD == 1:
            if EVEN_CO:
                acc += tl.load(A0 + offs_co)[:, None].to(tl.float32)
            else:
                acc += tl.load(A0 + offs_co, mask=m_co,
                               other=0.0)[:, None].to(tl.float32)
        if ACT:
            acc *= tl.sigmoid(acc)
        if HAS_RES:
            acc += tl.load(RES + n * r_sn + offs_co[:, None] * r_sc
                           + opix[None, :],
                           mask=None if EVEN_CO else m_co[:, None],
                           other=0.0).to(tl.float32)
        o = Y + n * y_sn + offs_co[:, None] * y_sc + opix[None, :]
        if EVEN_CO:
            tl.store(o, acc.to(Y.dtype.element_ty))
        else:
            tl.store(o, acc.to(Y.dtype.element_ty), mask=m_co[:, None])


class _HC3Plan:
    """A resolved launch for ``_hc3_kernel``; nothing is decided per call."""

    __slots__ = ("grid", "consts", "cfg", "out_shape", "strides")

    def __init__(self, grid, consts, cfg, out_shape, strides):
        self.grid, self.consts, self.cfg = grid, consts, cfg
        self.out_shape, self.strides = out_shape, strides

    def run(self, x, wt, a0, a1, a2, a3, res, y, eps):
        _hc3_kernel[self.grid](x, wt, a0, a1, a2, a3, res, y, eps,
                               *self.strides, **self.consts, **self.cfg)
        return y


def _hc3_plan(n, c, cout, imh, imw, s, fold, act, has_res, strides, cfg):
    """Build the plan, or None when *cfg* does not tile this shape exactly."""
    if triton is None:
        return None
    th, tw, bco, bc, warps, stages = cfg
    oh, ow = (imh + 2 - 3) // s + 1, (imw + 2 - 3) // s + 1
    if oh % th or ow % tw:
        return None
    npx = th * tw
    if npx < 16 or npx > 512 or bco < 16 or bc < 16:
        return None
    if npx & (npx - 1):
        return None
    consts = dict(C=c, IMH=imh, IMW=imw, COUT=cout, OH=oh, OW=ow, S=s,
                  NTW=ow // tw, TH=th, TW=tw, NP=npx, BLOCK_CO=bco,
                  BLOCK_C=bc, NUM_CB=-(-c // bc), EVEN_C=(c % bc == 0),
                  EVEN_CO=(cout % bco == 0),
                  # The row span reaches (oh0+TH-1)*S+1 and the column span
                  # (ow0+TW-1)*S+1, so a tap can only leave the image at the
                  # first/last tile of each axis; when a tile is interior the
                  # mask folds away entirely.
                  SAFE_ROW=False, SAFE_COL=False, FOLD=fold, ACT=act,
                  HAS_RES=has_res)
    grid = ((oh // th) * (ow // tw), -(-cout // bco), n)
    return _HC3Plan(grid, consts, {"num_warps": warps, "num_stages": stages},
                    (n, cout, oh, ow), strides)


# ---------------------------------------------------------------------------
# Retuned launch plans.
#
# Both frozen kernels below are used exactly as they are -- imported, not
# forked.  What changes is the *constexpr bundle and launch config* the block
# picks for them, because both gates were swept on their own level's captured
# shapes and this level feeds them different ones.  Every entry here is a
# measured winner over a config grid on the shape it is keyed to
# (``dev/p12_stem2.py``, ``dev/p9_dense.py``, profiler self-time, +-0.05 us);
# any shape not in a table falls straight through to the frozen gate, so a
# different input size is never worse than the parent.
# ---------------------------------------------------------------------------

# The three 3x3-stride-2 convs (stem1, stem2, down3).
#   (n, c, imh, imw, cout) -> (kind, BLOCK_CO, BLOCK_P, BLOCK_C|BLOCK_K, warps, stages)
# ``kind`` picks which frozen kernel runs: "3x3s2" is the alignment kernel the
# gate routes stride 2 to, "3x3" is the general implicit-GEMM one, which is
# fully general in (SH, SW) and only ever *reached* at stride 1 because
# ``_plan`` says so.
#
# Measured (profiler self-time, dev/p12_stem2.py; frozen -> retuned):
#
#   [4,3,640,640]->16    23.02 -> 21.96 us  1.05x   general kernel at stride 2
#   [4,16,320,320]->32   13.11 -> 13.11 us  1.00x   frozen tile already best
#   [4,32,160,160]->64   11.79 -> 11.79 us  1.00x   frozen tile already best
#   [1,3,640,640]->16     7.74 ->  6.97 us  1.11x   general kernel at stride 2
#   [1,16,320,320]->32    7.03 ->  5.60 us  1.26x   BLOCK_P 64 -> 32, 2 -> 4 warps
#   [1,32,160,160]->64    8.10 ->  6.70 us  1.21x   BLOCK_P 64 -> 16, 2 -> 4 warps
#
# Two mechanisms, both of which are the frozen gate reading a regime it never
# saw:
#
# * ``_pick_3x3s2_cfg`` picks BLOCK_P from OW alone and never checks that it
#   *divides* OW.  At OW=160 (BLOCK_P 64) and OW=80 (BLOCK_P 32) it does not, so
#   ``SAFE_W`` is False and every input span takes the masked path, and the last
#   p tile of each row is half wasted.  Dividing tiles fix both.  This only
#   shows at n=1: at n=4 the same shapes are wave-rich enough that the extra
#   CTAs of a narrow tile cost as much as the masking saves (13.11 either way).
# * stem1 has C=3, and the alignment kernel blocks the channel axis at
#   BLOCK_C=16 (the ``tl.dot`` minimum), so 13 of every 16 rows of its nine
#   [BLOCK_CO,16]x[16,BLOCK_P] dots are masked-off padding -- 19% MMA
#   efficiency.  The *general* kernel orders K as (tap, c) and covers all
#   K=27 in one dot at BLOCK_K=32, i.e. 84%.  It pays for that with an
#   unvectorised stride-2 lane axis, which is why the win is only 1.05-1.11x
#   rather than the 4x the MMA arithmetic suggests, and why it flips sign at
#   C=16/32 where the alignment kernel's vectorised span dominates.
#
# What this does NOT fix: all three convs are still 3-6x off their bandwidth
# roofline (stem1 22 us against ~3.5 us for 23 MB of traffic).  No tile of
# either frozen kernel reaches it -- the slack is a 9x-amplified gather that
# needs a shared-memory halo, i.e. a different kernel.  See ITERATIONS.md.
_STEM_CFG: dict = {
    #  (n,  c, imh, imw, cout): (kind,      BCO,  BP, BC|BK, warps, stages)
    (4, 3, 640, 640, 16):       ("3x3",      16, 128,    32,     2, 1),
    (1, 3, 640, 640, 16):       ("3x3",      16,  64,    32,     4, 1),
    (1, 16, 320, 320, 32):      ("3x3s2",    32,  32,    16,     4, 1),
    (1, 32, 160, 160, 64):      ("3x3s2",    32,  16,    32,     4, 1),
}

# Every dense conv the four C2f blocks issue.
#   (n, p, cout, cin, k, padded) -> (BLOCK_CO, BLOCK_P, BLOCK_K, warps, stages)
#
# Measured with dev/p13_tune.py (profiler self-time, ~10 candidate configs per
# conv drawn from a 240-config coarse pass; frozen -> retuned us, and how many
# times the backbone issues that conv):
#
#   n=4                                frozen               retuned      x
#   p=25600  16<-16   3x3  x2   (16,64,128,4,1) 12.50  (16,64, 32,4,1) 10.85  2
#   p=25600  32<-48   1x1  x1   (32,64, 64,4,1)  5.60  (32,64, 64,2,1)  5.57  1
#   p= 6400  64<-64   1x1  x1   (64,64, 64,8,1)  4.92  (32,128,64,8,1)  3.95  1
#   p= 6400  64<-128  1x1  x1   (64,64,128,8,1)  5.53  (32,128,64,4,1)  4.90  1
#   p= 1600  64<-64   3x3  x4   (32,32,256,4,1) 11.31  (64,64,256,8,1)  7.64  4
#   p= 1600 128<-256  1x1  x1   (32,64,128,4,1)  4.92  (64,64,256,8,1)  4.83  1
#   p=  400 256<-256  1x1  x1   (32,64,128,4,1)  4.15  (32,128,128,4,1) 3.90  1
#   p=  400 128<-128  3x3  x2   (32,32,256,4,1) 12.37  (64,32,512,8,1)  8.94  2
#   p=  400 256<-384  1x1  x1   (32,64,128,4,1)  5.25  (32,128,128,4,1) 4.86  1
#   -> 27.2 us over the twenty launches
#
#   n=1                                frozen               retuned      x
#   p=25600  16<-16   3x3  x2   (16,64,128,4,1)  4.57  (16,64, 32,4,1)  4.15  2
#   p= 6400  64<-64   1x1  x1   (64,64, 64,8,1)  3.27  (16,64, 64,4,1)  2.40  1
#   p= 6400  32<-32   3x3  x4   (32,64,128,4,1)  4.39  (32,64,512,4,1)  4.32  4
#   p= 6400  64<-128  1x1  x1   (64,64,128,8,1)  3.54  (32,128,128,8,1) 2.86  1
#   p= 1600 128<-256  1x1  x1   (32,64,256,8,1)  3.00  (32,64,128,4,1)  2.99  1
#   p=  400 256<-256  1x1  x1   (32,32,256,4,1)  2.94  (32,32,256,8,1)  2.67  1
#   p=  400 128<-128  3x3  x2   (32,16,256,4,1)  5.88  (16,32,256,4,1)  5.80  2
#   p=  400 256<-384  1x1  x1   (32,32,256,4,1)  4.10  (16,64,128,4,1)  3.26  1
#   -> 4.0 us over the twenty launches
#
# One mechanism generalises and one did not.
#
# *Generalises*: for the padded (3x3) convs at n=4 the win is always to **widen**
# BLOCK_CO to the whole output-channel count and BLOCK_P to 64 -- exactly the
# opposite of ``_dense_cfg_base``'s padded branch, which caps them at (32, 32).
# A 3x3's A tile is a 9x-amplified gather and it is fetched once per BLOCK_CO
# output channels, so doubling BLOCK_CO halves the gather traffic; at n=4 the
# grid is wave-rich enough to pay for the halved CTA count (100 CTAs at
# [4,64,40,40] beats 400, 7.64 against 11.31 us) and at n=1 it is not, which is
# why the same conv keeps the frozen tile there.  The frozen rule was swept
# where its own scored 3x3s lived -- p=400 at batch 1, the few-CTA corner
# ``_fill_machine`` exists for -- and this level's 3x3s are 4-64x bigger.
#
# *Did not*: BLOCK_K.  k=144 wants 32 over 128, k=1152 wants 512 over 256, and
# k=288 and k=576 keep the frozen value.  Padded-K waste
# (``cdiv(k,BK)*BK/k``) predicts none of that -- 144 at BK=128 wastes 78% and
# does lose, but 288 at BK=128 wastes 33% and still wins over the 0%-waste
# BK=32, because five extra loop iterations cost more than the masked lanes.
# So BLOCK_K stays a swept quantity, and this stays a table.
#
# Anything not keyed here falls through to the frozen ``_dense_cfg``.
_DENSE_CFG: dict = {
    #  (n,     p, cout, cin,    k, padded): (BCO,  BP,  BK, warps, stages)
    (4, 25600, 16, 16, 144, True):           (16,  64,  32,     4, 1),
    (4, 25600, 32, 48, 48, False):           (32,  64,  64,     2, 1),
    (4, 6400, 64, 64, 64, False):            (32, 128,  64,     8, 1),
    (4, 6400, 64, 128, 128, False):          (32, 128,  64,     4, 1),
    (4, 1600, 64, 64, 576, True):            (64,  64, 256,     8, 1),
    (4, 1600, 128, 256, 256, False):         (64,  64, 256,     8, 1),
    (4, 400, 256, 256, 256, False):          (32, 128, 128,     4, 1),
    (4, 400, 128, 128, 1152, True):          (64,  32, 512,     8, 1),
    (4, 400, 256, 384, 384, False):          (32, 128, 128,     4, 1),
    (1, 25600, 16, 16, 144, True):           (16,  64,  32,     4, 1),
    (1, 6400, 64, 64, 64, False):            (16,  64,  64,     4, 1),
    (1, 6400, 32, 32, 288, True):            (32,  64, 512,     4, 1),
    (1, 6400, 64, 128, 128, False):          (32, 128, 128,     8, 1),
    (1, 1600, 128, 256, 256, False):         (32,  64, 128,     4, 1),
    (1, 400, 256, 256, 256, False):          (32,  32, 256,     8, 1),
    (1, 400, 128, 128, 1152, True):          (16,  32, 256,     4, 1),
    (1, 400, 256, 384, 384, False):          (16,  64, 128,     4, 1),
}



# ---------------------------------------------------------------------------
# Which shapes the NEW 3x3 kernels are opted in on.
#
# A static table, swept offline (``dev/t_k0.py``, ``dev/t_k1.py``, profiler
# self-time over 30 issues), keyed on the exact (n, c, imh, imw, cout) the
# backbone issues.  There is no in-forward A/B and nothing is inferred: a shape
# that is not keyed here never reaches the new kernels at all, so it keeps
# exactly the parent's behaviour and cannot be slower than it.
#
# ``cfg[0]`` selects the kernel: 0 = the direct-FMA form, 1 = the mma form.
#
# Measured, us of profiler self-time per launch, parent -> new (``ref`` is the
# parent, i.e. the retuned frozen tile from ``_STEM_CFG`` / ``_DENSE_CFG``):
#
#   n=4                                parent   new     x     kernel
#   [4,  3,640,640]->16  s2  stem1      21.98  15.00  1.47   direct FMA
#   [4, 32,160,160]->64  s2  down3      11.66   9.21  1.27   mma
#   [4, 16,320,320]->32  s2  stem2      12.94  11.01  1.18   mma
#   [4, 16,160,160]->16  s1  stage2 x2  10.56   6.65  1.59   mma
#   [4, 32, 80, 80]->32  s1  stage3 x4   7.78   6.13  1.27   mma
#   [4, 64, 40, 40]->64  s1  stage4 x4   7.63   6.91  1.10   mma
#
#   n=1                                parent   new     x     kernel
#   [1,  3,640,640]->16  s2  stem1       6.95   5.77  1.20   direct FMA
#   [1, 32,160,160]->64  s2  down3       6.73   5.31  1.27   mma
#   [1, 16,320,320]->32  s2  stem2       5.58   5.39  1.03   mma
#   [1, 16,160,160]->16  s1  stage2 x2   4.07   3.69  1.10   mma
#   [1, 32, 80, 80]->32  s1  stage3 x4   4.28   3.49  1.23   mma
#   [1, 64, 40, 40]->64  s1  stage4 x4   5.14   4.22  1.22   mma
#
# Weighted by launch count that is 26.9 us of the 262.8 us the graph spends on
# the device at n=4 and 10.4 us of 145.4 at n=1.
#
# One shape is deliberately absent: ``[n,128,20,20]->128`` (stage5, x2), where
# the new kernel measures 1.01x at n=4 and 0.87x at n=1.  It is the only shape
# here whose width is not a multiple of 8, so its (row, channel, batch) strides
# are only 4-byte aligned and the patch has to be staged with 2-half global
# loads instead of 16 B ones -- four times the load instructions for the one
# conv with the least work to hide them.  It keeps the parent's tile.
_NEW_STEM: dict = {
    #  (n,  c, imh, imw, cout): cfg
    #  KIND 0: (0, TH, TW, JW, CO_T, BCO, WF32, CU_, MINBLK)
    #  KIND 1: (1, TH, BCO, BC, MT, NT, NWM, NWN, WSMEM, MINBLK)
    (4, 3, 640, 640, 16):   (0, 8, 32, 8, 4, 16, 1, 3, 4),
    (1, 3, 640, 640, 16):   (0, 4, 64, 8, 4, 16, 1, 3, 0),
    (4, 16, 320, 320, 32):  (1, 1, 32, 16, 1, 4, 2, 5, 0, 0),
    (1, 16, 320, 320, 32):  (1, 1, 32, 16, 1, 4, 2, 5, 0, 0),
    (4, 32, 160, 160, 64):  (1, 1, 64, 32, 1, 5, 4, 2, 0, 0),
    (1, 32, 160, 160, 64):  (1, 1, 64, 32, 1, 5, 4, 2, 0, 0),
}

# The C2f dense 3x3s, keyed on (n, imh, imw, cin, cout).
_NEW_DENSE: dict = {
    (4, 160, 160, 16, 16):  (1, 2, 16, 16, 1, 5, 1, 8, 0, 0),
    (1, 160, 160, 16, 16):  (1, 2, 16, 16, 1, 4, 1, 10, 1, 0),
    (4, 80, 80, 32, 32):    (1, 1, 32, 32, 1, 5, 2, 2, 0, 0),
    (1, 80, 80, 32, 32):    (1, 1, 32, 32, 1, 2, 2, 5, 1, 0),
    (4, 40, 40, 64, 64):    (1, 2, 64, 64, 1, 5, 4, 2, 1, 0),
    (1, 40, 40, 64, 64):    (1, 2, 16, 64, 1, 1, 1, 10, 1, 0),
}


def _new_stem(m: YOLOConv, x: torch.Tensor, cache: dict):
    """The new 3x3 for a tabled stem, or None to leave it to the parent path.

    Resolved once per (module, shape, dtype) and cached, like every other plan
    here: the returned object carries the compiled kernel and its argument
    cells, and running it is one ``cuLaunchKernel`` on the current stream.
    """
    if _C3_OFF or x.dtype is not torch.float16:
        return None
    conv = m.conv
    w = conv.weight
    if (w.dtype is not torch.float16 or conv.groups != 1
            or conv.dilation != (1, 1) or tuple(conv.padding) != (1, 1)
            or tuple(w.shape[2:]) != (3, 3) or conv.stride[0] != conv.stride[1]
            or not w.is_contiguous() or not x.is_contiguous()):
        return None
    n, c, imh, imw = (int(v) for v in x.shape)
    cout = int(w.shape[0])
    s = int(conv.stride[0])
    cfg = _NEW_STEM.get((n, c, imh, imw, cout))
    if cfg is None:
        return None
    affine = m._affine()
    if affine is None:
        return None
    fold, a0, a1, a2, a3, eps = affine
    act = m._act_code()
    if act is None:
        return None
    oh, ow = (imh + 2 - 3) // s + 1, (imw + 2 - 3) // s + 1
    key = (id(m), n, c, imh, imw, cout, s, fold, act)
    ent = cache.get(key, False)
    if ent is False:
        plan = _c3_plan(n, c, cout, imh, imw, s,
                        2 if fold == 1 else 1, act, 0,
                        c * imh * imw, imh * imw, 0, 0,
                        cout * oh * ow, oh * ow,
                        (_aff_flag(a0), _aff_flag(a1), _aff_flag(a2),
                         _aff_flag(a3)), cfg,
                        _ptr_align(x))
        wt = None if plan is None else _c3_weight(m, cfg, w, c)
        ent = None if plan is None else (plan, wt)
        cache[key] = ent
    if ent is None:
        return None
    plan, wt = ent
    plan.bind(wt, a0, a1, a2, a3, None, eps)
    return plan


def _ptr_align(t: torch.Tensor) -> int:
    """Staging chunk (in halves) the tensor's own base address permits.

    Checked here, at plan time, rather than per call: every tensor these
    kernels see is either the graph pool's input copy or a channel slice of a
    ``YOLOC2f`` buffer, i.e. a torch allocation (512 B aligned) offset by a
    multiple of ``imh*imw*2`` bytes, so the value is a property of the shape and
    does not vary between the plan and the replays.
    """
    p = t.data_ptr()
    for chw in (8, 4, 2):
        if p % (2 * chw) == 0:
            return chw
    return 1


_C3_WCACHE: dict = {}


def _c3_weight(m, cfg, w: torch.Tensor, c: int) -> torch.Tensor:
    """The kernel's weight view: [COUT, 9, C] transposed, mma-permuted for KIND 1.

    Keyed on (data_ptr, _version) exactly as ``YOLOConv._packed_weight`` is, and
    dropped wholesale by ``_invalidate``, so a ``load_state_dict`` or a
    ``fuse()`` after the first forward cannot leave a stale copy behind.
    """
    frag = _c3_wants_frag(cfg)
    key = (id(m), frag, w.data_ptr(), w._version)
    got = _C3_WCACHE.get(key)
    if got is None:
        wt = _convmod._transposed_weight(w)
        if frag:
            wt = _c3_frag_weight(wt, c)
        _C3_WCACHE[key] = got = wt
    return got


def _c3_wants_frag(cfg) -> bool:
    """True when the kernel reads A fragments straight from global.

    Only the mma kernel with ``WSMEM = 0`` does; with ``WSMEM = 1`` it stages
    [BCO, 9, BC] itself and wants the plain [COUT, 9, C] transpose, and the
    direct-FMA kernel always does.  Getting this wrong is silent -- the
    permutation is a bijection, so the kernel runs and returns a confidently
    wrong answer (measured 1.0 relative, and 2.2e-02 end to end because only
    some tabled shapes take the staged path).  ``dev/t_res.py`` covers both.
    """
    return cfg[0] == 1 and cfg[8] == 0


def _stem_plan(m: YOLOConv, x: torch.Tensor, act: int, fold: int):
    """The frozen plan for a stem conv, with a retuned tile if one is tabled.

    ``_convmod._plan`` is called first and its verdict is respected: it owns the
    gate (fp16, groups 1, dilation 1, contiguous weight, 3x3/1x1 geometry, the
    FLOP bound), and a None from it means the fused path does not apply at all.
    Only the tile fields are then rewritten.
    """
    plan = _convmod._plan(x, m.conv, act, fold)
    if plan is None or plan["kind"] != "3x3s2":
        return plan
    n, c, imh, imw = (int(v) for v in x.shape)
    cout = int(m.conv.weight.shape[0])
    ent = _STEM_CFG.get((n, c, imh, imw, cout))
    if ent is None:
        return plan
    kind, bco, bp, bck, nw, ns = ent
    oh, ow = imh // 2, imw // 2
    p = oh * ow
    cdiv = (lambda a, b: -(-a // b))
    if kind == "3x3s2":
        consts = dict(plan["consts"])
        consts.update(BLOCK_CO=bco, BLOCK_P=bp, BLOCK_C=bck,
                      TPR=cdiv(ow, bp), NUM_CB=cdiv(c, bck), EVEN_C=c % bck == 0,
                      EVEN_CO=cout % bco == 0, EVEN_P=ow % bp == 0,
                      # The aligned span reaches 2*OW-1 and the kx=0 span
                      # 2*OW-2, so both stay inside the row exactly when the p
                      # tiles divide OW and 2*OW <= IMW.  It depends on BLOCK_P,
                      # so it has to be recomputed, not inherited.
                      SAFE_W=ow % bp == 0 and 2 * ow <= imw)
        grid = (consts["TPR"] * oh, cdiv(cout, bco), n)
    else:  # the general implicit-GEMM kernel, driven at stride 2
        k = c * 9
        consts = dict(C=c, IMH=imh, IMW=imw, COUT=cout, OW=ow, P=p, K=k, KW=3,
                      SH=2, SW=2, PH=1, PW=1, FOLD=fold, ACT=act,
                      BLOCK_CO=bco, BLOCK_P=bp, BLOCK_K=bck,
                      NUM_K=cdiv(k, bck), EVEN_K=k % bck == 0,
                      EVEN_CO=cout % bco == 0, EVEN_P=p % bp == 0)
        grid = (cdiv(p, bp), cdiv(cout, bco), n)
    return {"kind": kind, "out_shape": plan["out_shape"], "grid": grid,
            "cfg": {"num_warps": nw, "num_stages": ns}, "consts": consts}


def _stem_conv(m: YOLOConv, x: torch.Tensor, cache: dict) -> torch.Tensor:
    """``YOLOConv.forward`` with the new 3x3 where tabled, the retuned frozen
    plan otherwise, and the block's own ``forward`` behind both."""
    if not (x.is_cuda and m.conv.weight.is_cuda):
        return m(x)
    new = _new_stem(m, x, cache)
    if new is not None:
        return new.run(x, torch.empty(new.out_shape, dtype=x.dtype,
                                      device=x.device))
    act = m._act_code()
    if act is None:
        return m(x)
    affine = m._affine()
    if affine is None:
        return m(x)
    fold, a0, a1, a2, a3, eps = affine
    key = (id(m), tuple(x.shape), x.dtype, act, fold)
    plan = cache.get(key, False)
    if plan is False:
        plan = _stem_plan(m, x, act, fold)
        cache[key] = plan
    if plan is None:
        return m(x)
    return _convmod._run(x if x.is_contiguous() else x.contiguous(),
                         m._packed_weight(plan), a0, a1, a2, a3, eps, plan)


def _dense_apply(step, x, res, dst, n, cache=None):
    """``_c2fmod._Conv.apply``, with the new 3x3 where tabled and the retuned
    frozen tile otherwise."""
    dn = step.dense
    if dn is not None:
        wt, cout, kh, kw, ph, pw = dn
        cin = int(x.shape[1])
        if kh == 3 and kw == 3 and cache is not None:
            new = _new_dense(step, x, res, dst, n, cache)
            if new is not None:
                return new
        cfg = _DENSE_CFG.get((n, int(x.shape[2]) * int(x.shape[3]), cout, cin,
                              cin * kh * kw, kh * kw > 1))
        if cfg is not None:
            return _c2fmod._dense_conv(x, wt, step.b, step.act, res, dst,
                                       cout, kh, kw, ph, pw, cfg=cfg)
    return step.apply(x, res=res, dst=dst)


def _new_dense(step, x, res, dst, n, cache):
    """The new 3x3 for a tabled C2f dense conv, or None for the parent path.

    The epilogue contract is ``_c2fmod._dense_conv``'s exactly -- folded bias,
    SiLU, the optional residual, and a caller-supplied destination view, with
    the (batch, channel) strides passed through -- so a conv can read one
    channel slice of the shared concat buffer and write another with no copy.
    """
    if _C3_OFF or x.dtype is not torch.float16:
        return None
    wt, cout, kh, kw, ph, pw = step.dense
    n_, cin, imh, imw = (int(v) for v in x.shape)
    cfg = _NEW_DENSE.get((n, imh, imw, cin, cout))
    if cfg is None:
        return None
    b = step.b
    if b is None or b.dtype is not torch.float16:
        return None
    key = (id(step), n, cin, imh, imw, cout, res is not None,
           None if dst is None else (dst.stride(0), dst.stride(1)))
    ent = cache.get(key, False)
    if ent is False:
        rsn = 0 if res is None else int(res.stride(0))
        rsc = 0 if res is None else int(res.stride(1))
        ysn = cout * imh * imw if dst is None else int(dst.stride(0))
        ysc = imh * imw if dst is None else int(dst.stride(1))
        plan = _c3_plan(n, cin, cout, imh, imw, 1, 1, step.act,
                        0 if res is None else 1,
                        int(x.stride(0)), int(x.stride(1)), rsn, rsc,
                        ysn, ysc, (0, 0, 0, 0), cfg, _ptr_align(x))
        ent = None
        if plan is not None:
            fw = _c3_frag_dense(wt, cin) if _c3_wants_frag(cfg) else wt
            ent = (plan, fw)
        cache[key] = ent
    if ent is None:
        return None
    plan, fw = ent
    plan.bind(fw, b, b, b, b, res, 0.0)
    if dst is None:
        dst = torch.empty((n, cout, imh, imw), dtype=x.dtype, device=x.device)
    return plan.run(x, dst)


_C3_DWCACHE: dict = {}


def _c3_frag_dense(wt: torch.Tensor, cin: int) -> torch.Tensor:
    """``_c3_frag_weight`` for a ``_c2fmod._Conv``'s already-transposed weight."""
    key = (wt.data_ptr(), wt._version)
    got = _C3_DWCACHE.get(key)
    if got is None:
        _C3_DWCACHE[key] = got = _c3_frag_weight(wt, cin)
    return got



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
        # Derived state: plain attributes, never buffers/parameters, so they
        # stay out of ``state_dict`` and are not cast by ``.half()``.
        self._graphs: dict = {}     # (shape, dtype, device) -> entry | None
        self._sig_src: tuple | None = None
        self._sig = None
        self._stem_cache: dict = {}
        self.register_load_state_dict_post_hook(_invalidate_hook)

    # -- weight-change guard ------------------------------------------------
    #
    # A replay has every folded weight's *address* baked into it, so anything
    # that replaces or rewrites a parameter after capture has to drop the
    # graphs.  ``load_state_dict`` (the harness' own mechanism) and
    # ``Module._apply`` (``.half()``, ``.to()``, ``.cuda()``) are hooked
    # exactly; on top of that every forward re-derives a
    # ``(data_ptr, _version)`` tuple over the 108 parameters and 72 float
    # buffers, which catches a replaced or autograd-visible-mutated tensor.
    # (``num_batches_tracked`` is skipped: it is integral and nothing folds it.)
    #
    # The tensor list is built once and cached, because re-walking the module
    # tree every forward is not affordable: measured (``dev/p21_sigcost.py``)
    # 20 us for the signature over a cached list against 240 us for a
    # from-scratch walk, and the whole host budget is the 175-275 us the replay
    # spends on the device.  At 20 us the guard is free -- the host is already
    # 5-8x ahead -- and ``dev/p6_split.py`` shows it moving the window by 0.1 us.
    #
    # A cached list cannot see the parameter *set* change, which is what
    # ``fuse()`` does (it adds ``conv.bias`` and deletes ``bn``), so the fused
    # flag is read directly.  ``fuse_module`` recurses over every ``YOLOConv`` in
    # the tree, so ``stem1`` is fused whenever anything is; a hand-fused inner
    # block only is not covered.
    #
    # Two residual holes, both stated rather than papered over.  ``p.data.mul_()``
    # rewrites a parameter without bumping ``_version`` or moving it, which no
    # signature of this kind can see -- the frozen ``YOLOConv`` docstring makes
    # the same point, which is why *it* re-folds every call.  And an in-place
    # edit of the ``conv.bias`` that ``fuse()`` created is not in the cached
    # list until something else invalidates it.  For either, and for a caller who
    # wants the outputs' lifetime to be its own, ``FK_BB_GRAPH=0`` drops to the
    # eager chain.
    def _signature(self):
        src = self._sig_src
        if src is None:
            src = tuple(self.parameters()) + tuple(
                b for b in self.buffers() if b.is_floating_point())
            self._sig_src = src
        return (getattr(self.stem1, "_is_fused", False),
                tuple((t.data_ptr(), t._version) for t in src))

    def _invalidate(self):
        self._graphs.clear()
        self._sig_src = None
        self._sig = None
        self._stem_cache.clear()
        # The new 3x3 holds a permuted copy of the folded weight; a plan cache
        # keyed on (data_ptr, _version) cannot see ``fuse()``'s in-place
        # ``weight.data.copy_()``, so the copies go with the graphs.
        _C3_WCACHE.clear()
        _C3_DWCACHE.clear()
        _sppf_reset(self.sppf)

    def _apply(self, *args, **kwargs):
        self._invalidate()
        return super()._apply(*args, **kwargs)

    # -- the flat chain: what gets captured ---------------------------------
    #
    # Each block's fused fast path, reached without its own graph replay or
    # megakernel, so the capture records a flat stream of kernels and every
    # intermediate is an allocation from the outer graph's private pool.
    def _flat(self, x: torch.Tensor):
        sc = self._stem_cache
        x = _stem_conv(self.stem1, x, sc)
        x = _stem_conv(self.stem2, x, sc)
        p2 = _c2f(self.stage2, x, sc)
        x = _stem_conv(self.down3, p2, sc)
        p3 = _c2f(self.stage3, x, sc)
        x = self.down4(p3)
        p4 = _c2f(self.stage4, x, sc)
        x = self.down5(p4)
        p5 = _c2f(self.stage5, x, sc)
        p5 = self.sppf(p5)
        p5 = self.psa(p5)
        return p3, p4, p5

    def _baseline(self, x: torch.Tensor):
        """The literal composition -- every block's own ``forward``.

        The last step in the fallback chain, and the only path that can be
        correct in training mode: every fused path folds BN from *running*
        statistics, which is exact in eval and wrong in training.

        Three frozen blocks need help to be right there, and all three are
        frozen, so the fix lives here.  ``YOLOSCDown.forward`` and
        ``YOLOSPPF.forward`` never consult ``self.training`` -- they fold running
        statistics unconditionally -- so training mode goes through their own
        reference compositions (``cv2(cv1(x))`` and ``_reference``), whose
        ``YOLOConv`` children do check.  ``YOLOBottleneck`` gates on
        ``self._needs_grad and torch.is_grad_enabled()`` and likewise never on
        ``self.training``, which ``_c2f_train`` handles.  ``YOLOPSA`` gates
        correctly on its own.

        Measured under ``.train()`` inside ``no_grad`` at n=4, relative to the
        native composition (``dev/p17_robust.py``): 1.08 with none of this,
        0.94 with the SCDown/SPPF fix, 5.3e-03 with ``_c2f_train`` as well --
        i.e. fp16 noise.  The round-0 kernel raises instead of any of these,
        because in training mode stem1 falls out of ``YOLOConv``'s fused gate
        onto the L1 conv that cannot take ``C = 3`` (see ``_conv_ref``).
        """
        train = self.training
        on_cpu = not x.is_cuda
        c2f = _c2f_train if train else (lambda m, t: m(t))
        x = _leaf(self.stem1, x)
        x = _leaf(self.stem2, x)
        p2 = c2f(self.stage2, x)
        x = _leaf(self.down3, p2)
        p3 = c2f(self.stage3, x)
        x = self.down4.cv2(self.down4.cv1(p3)) if train else self.down4(p3)
        p4 = c2f(self.stage4, x)
        x = self.down5.cv2(self.down5.cv1(p4)) if train else self.down5(p4)
        p5 = c2f(self.stage5, x)
        p5 = self.sppf._reference(p5) if train else self.sppf(p5)
        if on_cpu:
            with _cpu_safe(self.psa):
                p5 = self.psa(p5)
        else:
            p5 = self.psa(p5)
        return p3, p4, p5

    # -- capture ------------------------------------------------------------
    def _capture(self, x: torch.Tensor, key):
        """Capture ``_flat`` for this (shape, dtype, device); None if it will not.

        The warm-up pass is what makes the capture cheap *and* possible: it JITs
        every Triton kernel, NVRTC-compiles SCDown, builds SPPF's plan and
        PSA's persistent buffers, and resolves all eleven blocks' fold plans, so
        none of that host work is attempted inside the capture (where an
        allocation outside the pool or a sync would abort it).
        """
        entry = None
        try:
            static_in = torch.empty_like(x, memory_format=torch.contiguous_format)
            static_in.copy_(x)
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    self._flat(static_in)
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                outs = self._flat(static_in)
            if any(not isinstance(t, torch.Tensor) for t in outs):
                raise RuntimeError("unexpected chain output")
            entry = (static_in, tuple(outs), graph)
        except Exception:  # noqa: BLE001 - uncapturable: stay on the eager path
            entry = None
        self._graphs[key] = entry
        return entry

    # -- forward ------------------------------------------------------------
    def forward(self, x: torch.Tensor):
        if self.training or torch.is_grad_enabled():
            return dict(zip(_OUT_KEYS, self._baseline(x)))

        if _USE_GRAPH and x.is_cuda:
            sig = self._signature()
            if sig != self._sig:
                # Everything derived from the weights goes, not just the graphs:
                # the retuned stem plans hold a packed weight and SPPF holds a
                # folded one behind no guard of its own.  ``_invalidate`` drops
                # ``_sig_src`` too, so the signature is re-derived after it.
                self._invalidate()
                self._sig = self._signature()
            key = (tuple(x.shape), x.dtype, x.device)
            entry = self._graphs.get(key, _MISSING)
            if entry is _MISSING:
                entry = (None if len(self._graphs) >= _MAX_GRAPHS
                         else self._capture(x, key))
            if entry is not None:
                static_in, outs, graph = entry
                static_in.copy_(x, non_blocking=True)
                graph.replay()
                if _CLONE_OUT:
                    outs = tuple(t.clone() for t in outs)
                return dict(zip(_OUT_KEYS, outs))

        if _USE_FLAT and x.is_cuda:
            try:
                return dict(zip(_OUT_KEYS, self._flat(x)))
            except Exception:  # noqa: BLE001
                pass
        return dict(zip(_OUT_KEYS, self._baseline(x)))


def _c2f(m: YOLOC2f, x: torch.Tensor, cache: dict | None = None) -> torch.Tensor:
    """``YOLOC2f``'s flat fused path: no inner graph replay, no megakernel.

    Both of those are wins at L2 and both are wrong here -- see the module
    docstring.  Falls back to the block's own ``forward`` for a module tree
    ``_build_plan`` does not recognize, which is also the only branch that can
    reach the reference composition.

    The body mirrors ``YOLOC2f._fused``'s shared-buffer arm exactly -- one
    ``(2+n)*c``-channel buffer, cv1 into channels [0, 2c), each bottleneck into
    its own c-wide slice, cv2 over the whole thing with no concat -- and differs
    only in going through ``_dense_apply`` so a retuned tile can be substituted.
    Anything that arm does not cover (no bottlenecks, or a plan whose convs are
    not all shape-preserving) is handed back to ``_fused`` untouched.
    """
    plan = m._get_plan()
    if plan is False:
        return m(x)
    cv1, blocks, cv2, shared = plan
    if not (shared and blocks):
        return m._fused(x, plan)
    n, c = int(x.shape[0]), m.c
    buf = torch.empty((n, (2 + len(blocks)) * c, x.shape[2], x.shape[3]),
                      dtype=torch.result_type(x, cv1.w), device=x.device)
    _dense_apply(cv1, x, None, buf[:, :2 * c], n, cache)
    prev = buf[:, c:2 * c]
    base = 2 * c
    for steps, add in blocks:
        o = prev
        for step in steps[:-1]:
            o = _dense_apply(step, o, None, None, n, cache)
        dst = buf[:, base:base + c]
        _dense_apply(steps[-1], o, prev if add else None, dst, n, cache)
        prev = dst
        base += c
    return _dense_apply(cv2, buf, None, None, n, cache)


def _conv_ref(m: YOLOConv, x: torch.Tensor) -> torch.Tensor:
    """``act(bn(conv(x)))`` in plain functional ops -- the last resort.

    Needed because the frozen stack has one hole this level sits on top of: at
    bf16 the three stride-2 stems fall out of ``YOLOConv``'s fp16-only fused
    gate onto the composed frozen ``L1.Conv2d``, whose padded implicit GEMM
    picks ``BLOCK_K = min(pow2(K), 64)`` and asserts ``K >= 16`` -- and stem1's
    ``C = 3`` is below that.  ``dev/p18_r0.py`` reproduces it on the *round-0*
    kernel (the literal ``baseline.py``) at both batch sizes, so it is inherited,
    not introduced; it is fixed here rather than in ``candidate/L1`` because
    those files are frozen.  Exact in eval and in training (``F.batch_norm``
    gets the same ``training`` flag the module would have used).
    """
    conv = m.conv
    y = F.conv2d(x, conv.weight, conv.bias, conv.stride, conv.padding,
                 conv.dilation, conv.groups)
    bn = getattr(m, "bn", None)
    if bn is not None:
        y = F.batch_norm(y, bn.running_mean, bn.running_var, bn.weight, bn.bias,
                         bn.training or not bn.track_running_stats,
                         bn.momentum if bn.momentum is not None else 0.0,
                         bn.eps)
    return m.act(y)


def _leaf(m: YOLOConv, x: torch.Tensor) -> torch.Tensor:
    """A top-level ``YOLOConv``, its own forward first and ``_conv_ref`` behind."""
    try:
        return m(x)
    except Exception:  # noqa: BLE001 - a frozen path that declines this dtype
        return _conv_ref(m, x)


def _c2f_train(m: YOLOC2f, x: torch.Tensor) -> torch.Tensor:
    """``YOLOC2f`` in training mode, down to the leaves.

    ``YOLOC2f.forward`` already routes training to ``_reference``, but
    ``_reference`` reaches its bottlenecks through ``YOLOBottleneck.forward``,
    and *that* block gates its folded fast path on
    ``self._needs_grad and torch.is_grad_enabled()`` and never on
    ``self.training``.  Under ``.train()`` inside ``torch.no_grad()`` -- which is
    exactly how an inference-shaped harness would exercise training mode -- the
    guard is False, so it folds *running* statistics into the weight and returns
    a confidently wrong answer (0.94x relative error end to end).  Composing the
    bottleneck from its own ``cv1``/``cv2`` ``YOLOConv`` children puts the
    decision back where it is made correctly: ``YOLOConv._affine`` returns None
    when ``bn.training``, so each conv takes the composed batch-statistics path.
    """
    y = list(m.cv1(x).chunk(2, 1))
    for b in m.m:
        inp = y[-1]
        o = b.cv2(b.cv1(inp))
        y.append(inp + o if getattr(b, "add", False) else o)
    return m.cv2(torch.cat(y, 1))


@contextlib.contextmanager
def _cpu_safe(psa: YOLOPSA):
    """Make the PSA reference path runnable on a CPU tensor.

    ``YOLOAttention`` declines its fused plan for fp32 (and for CPU) and falls
    back to ``self._softmax((q^T k) * scale)``, where ``_softmax`` is the frozen
    L1 ``Softmax`` -- which builds a Triton plan without ever checking the
    device, so on a CPU tensor it raises ``Pointer argument (at 0) cannot be
    accessed from Triton``.  It is otherwise a drop-in for ``nn.Softmax(dim=)``,
    so one is substituted for the duration of the call.  ``dev/p20_cpu.py``
    walks the eleven blocks and shows this is the *only* CPU hole in the stack;
    the round-0 kernel raises here too.
    """
    att = getattr(psa, "attn", None)
    sm = getattr(att, "_softmax", None) if att is not None else None
    if sm is None or isinstance(sm, nn.Softmax):
        yield
        return
    att._softmax = nn.Softmax(dim=getattr(sm, "dim", -1))
    try:
        yield
    finally:
        att._softmax = sm


def _sppf_reset(sppf: YOLOSPPF) -> None:
    """Drop ``YOLOSPPF``'s folded-weight plan so the next forward rebuilds it.

    ``YOLOSPPF`` is the one frozen block with no weight-change guard: ``_build``
    folds BN into a swizzled mma-fragment weight, hands it to
    ``sppf_make_plan`` and remembers only ``self._shape``, and ``forward``
    re-dispatches on shape alone.  So a ``load_state_dict`` after the first
    forward -- exactly what the harness does, and what ``dev/p19_which.py``
    isolates -- leaves it computing with the *old* weights: 9.2e-01 relative
    error at the sppf output while every other block stays at ~1e-3.  The
    round-0 kernel has the same bug (both measure 1.969e-01 end to end), and
    ``candidate/L2`` is frozen, so the reset is driven from here instead, off
    the same invalidation that drops the graphs.
    """
    plan = getattr(sppf, "_plan", 0)
    if plan:
        try:
            _sppfmod._EXT.sppf_free_plan(plan)
        except Exception:  # noqa: BLE001 - extension absent, or already freed
            pass
    sppf._plan = 0
    sppf._shape = None
    sppf._keep = None


def _invalidate_hook(module, incompatible_keys):  # noqa: ARG001
    module._invalidate()
