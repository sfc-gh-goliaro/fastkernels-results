"""YOLOv10 bottleneck as two fused Triton implicit-GEMM launches.

The baseline block is ``x -> [conv 3x3, BN, SiLU] -> [conv 3x3, BN, SiLU] -> +x``,
i.e. seven kernels and six full HBM round trips over the activations.  Every
captured workload is tiny (the largest, ``[4,16,160,160]`` fp16, is 3.3 MB and
944 MFLOP across both convs), so on a B200 the block is not bandwidth- or
FLOP-bound at all: measured, a single ``F.conv2d`` costs ~28 us of which ~22 us
is host-side dispatch, and the whole block costs ~120 us of which ~113 us is host
time.  For scale, an empty kernel launch on this box is 0.7-0.9 us of GPU and a
plain copy of the same bytes is 1.4-1.9 us, so seven launches cannot account for
more than a few us of real work.  The cost is *per-op overhead*, and the lever
that matters is collapsing ops.

This does that in two steps:

* **BatchNorm folding.**  ``bench.py`` runs the module in ``eval()`` and never
  calls ``fuse_module()`` (that is an L4 concern), so both BatchNorms execute at
  runtime even though in eval they are a fixed per-channel affine.  We fold them
  into the conv weights exactly as ``YOLOConv.fuse()`` would --- lazily on first
  forward, never in ``__init__``, because the harness shares weights via
  ``load_state_dict`` *after* construction.
* **Epilogue fusion.**  What is left is two convolutions whose epilogues are
  ``+bias, SiLU`` and ``+bias, SiLU, +x``.  Both run as one implicit GEMM each
  (``M = OH*OW``, ``N = Cout``, ``K = Cin*KH*KW``) built on the frozen L1 winner's
  ``_pad_k_step`` gather/K-loop, so the intermediate is written once and read
  once and nothing else touches HBM.

7 launches -> 2.  The launches themselves then matter, so the host path is kept
as short as it can be: the fold, the tiles, the grids and the ``CompiledKernel``
handles are all cached per shape, and steady-state calls go straight to
``CompiledKernel.run`` instead of back through ``JITFunction.run``'s binder
(measured 8.8 us -> 4.4 us per launch), which puts one ``forward`` at ~13 us of
host time against ~120 us for the baseline block.

The k-loop and A-tile gather are the L1 winner's ``_pad_k_step`` unchanged; only
the epilogue and the tile choice are new.  A stride-1 pad-1 rewrite of the gather
(A-tile addresses affine in the flat p index) was tried and *lost* once
``BLOCK_K`` was swept properly -- see ``_TILES``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

from .yolov10_conv import YOLOConv

# The frozen L1 winner's padded implicit-GEMM gather/K-loop.  Reused verbatim --
# only the epilogue below is new.  If the L1 candidate is absent the module still
# runs, just on the eager reference path.
try:
    from ..L1.conv2d import _pad_k_step, _pad_weight
    _HAVE_L1_PAD = True
except ImportError:  # pragma: no cover - candidate/L1 always present under bench
    _pad_k_step = _pad_weight = None
    _HAVE_L1_PAD = False

try:
    from torch._C import _cuda_getCurrentRawStream as _raw_stream
except ImportError:  # pragma: no cover
    def _raw_stream(idx):
        return torch.cuda.current_stream(idx).cuda_stream


# ---------------------------------------------------------------------------
# Fused conv + bias + SiLU (+ residual) implicit GEMM.
#
# Identical blocking to the L1 winner's ``_pad_conv_kernel`` -- M = OH*OW and
# Cout are tiled, the k loop is a static nest over ceil(K/BLOCK_K) chunks of the
# (tap, channel) axis, and the pad-1 halo is masked loads inside that loop -- and
# the k step itself *is* that kernel's ``_pad_k_step``.  Everything new is after
# the loop:
#
#   acc (fp32)  ->  + fused BN bias (fp32)  ->  SiLU  ->  + residual  ->  store
#
# The bias stays fp32 all the way into the epilogue: it absorbs BN's shift term,
# which has no reason to be representable in fp16, and rounding it would show up
# directly as drift against the reference's separate BN pass.
# ---------------------------------------------------------------------------
@triton.jit
def _bneck_conv_kernel(
    X, Wt, Bias, Res, Y,
    C: tl.constexpr,
    IMH: tl.constexpr,
    IMW: tl.constexpr,
    COUT: tl.constexpr,
    OW: tl.constexpr,
    P: tl.constexpr,            # OH*OW
    TPR: tl.constexpr,          # p tiles per output row (ROW_TILED only)
    K: tl.constexpr,            # C*KH*KW
    KW: tl.constexpr,
    SH: tl.constexpr,
    SW: tl.constexpr,
    PH: tl.constexpr,
    PW: tl.constexpr,
    RESIDUAL: tl.constexpr,     # add Res[n, co, oh, ow] (the block's shortcut)
    ROW_TILED: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_K: tl.constexpr,
    LOOP_STAGES: tl.constexpr,  # 0 = fully unroll the k loop, >=2 = pipeline it
    EVEN_K: tl.constexpr,
    EVEN_CO: tl.constexpr,
    EVEN_P: tl.constexpr,
):
    pid_p = tl.program_id(0)
    pid_co = tl.program_id(1)
    n = tl.program_id(2)

    offs_co = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    m_co = offs_co < COUT

    if ROW_TILED:
        oh = pid_p // TPR
        ow = (pid_p - oh * TPR) * BLOCK_P + tl.arange(0, BLOCK_P)
        offs_p = oh * OW + ow
        m_p = ow < OW
    else:
        offs_p = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
        oh = offs_p // OW
        ow = offs_p - oh * OW
        m_p = offs_p < P
    ih0 = oh * SH - PH
    iw0 = ow * SW - PW

    xn = X + n * (C * IMH * IMW)
    acc = tl.zeros((BLOCK_CO, BLOCK_P), dtype=tl.float32)
    if LOOP_STAGES == 0:
        for kb in tl.static_range(NUM_K):
            acc = _pad_k_step(acc, xn, Wt, kb, offs_co, m_co, m_p, ih0, iw0,
                              C, IMH, IMW, K, KW, ROW_TILED, BLOCK_K, BLOCK_P,
                              EVEN_K, EVEN_CO, EVEN_P)
    else:
        for kb in tl.range(NUM_K, num_stages=LOOP_STAGES):
            acc = _pad_k_step(acc, xn, Wt, kb, offs_co, m_co, m_p, ih0, iw0,
                              C, IMH, IMW, K, KW, ROW_TILED, BLOCK_K, BLOCK_P,
                              EVEN_K, EVEN_CO, EVEN_P)

    # --- fused epilogue -----------------------------------------------------
    if EVEN_CO:
        acc += tl.load(Bias + offs_co)[:, None]
    else:
        acc += tl.load(Bias + offs_co, mask=m_co, other=0.0)[:, None]
    acc = acc * tl.sigmoid(acc)

    y_off = n * (COUT * P) + offs_co[:, None] * P + offs_p[None, :]
    if EVEN_CO and EVEN_P:
        if RESIDUAL:
            acc += tl.load(Res + y_off).to(tl.float32)
        tl.store(Y + y_off, acc.to(Y.dtype.element_ty))
    else:
        keep = m_co[:, None] & m_p[None, :]
        if RESIDUAL:
            acc += tl.load(Res + y_off, mask=keep, other=0.0).to(tl.float32)
        tl.store(Y + y_off, acc.to(Y.dtype.element_ty), mask=keep)


# ---------------------------------------------------------------------------
# eval-mode BatchNorm folding.
# ---------------------------------------------------------------------------
@torch.no_grad()
def _fold_bn(conv: nn.Module, bn: nn.Module | None):
    """``(conv, eval BN) -> (weight [COUT, KH*KW, Cin] fp16, bias [COUT] fp32)``.

    In eval a BatchNorm is ``scale * y + shift`` with
    ``scale = w / sqrt(running_var + eps)`` and ``shift = b - scale * mean``, so
    it folds into the conv as ``w' = scale * w``, ``b' = scale * b + shift`` --
    the same algebra as ``YOLOConv.fuse()``.

    Two deliberate precision choices: the arithmetic runs in fp32 (the running
    stats are buffers and stay fp32 while ``weight``/``bias`` were cast to fp16),
    and the resulting bias is *kept* fp32 for the epilogue while the weight goes
    back to the conv's dtype so the k loop keeps its tensor-core path.
    """
    w = conv.weight
    if bn is None:
        bias = conv.bias
        bias32 = (torch.zeros(w.shape[0], dtype=torch.float32, device=w.device)
                  if bias is None else bias.detach().float())
        return _pad_weight(w.detach()), bias32.contiguous()

    scale = bn.weight.detach().float() / torch.sqrt(
        bn.running_var.detach().float() + bn.eps)
    shift = bn.bias.detach().float() - scale * bn.running_mean.detach().float()
    if conv.bias is not None:
        shift = shift + scale * conv.bias.detach().float()
    wf = (w.detach().float() * scale[:, None, None, None]).to(w.dtype)
    return _pad_weight(wf), shift.contiguous()


def _guard_tensors(conv: nn.Module, bn: nn.Module | None):
    """The five tensors a conv's fold depends on, always five so the version
    check below is a flat unrolled tuple rather than a loop."""
    w = conv.weight
    if bn is None:
        b = conv.bias if conv.bias is not None else w
        return (w, b, w, w, w)
    return (w, bn.weight, bn.bias, bn.running_mean, bn.running_var)


def _guard_vers(g):
    """Fold-cache validity token.

    ``load_state_dict`` copies in place (version bump) and a ``p.data = ...``
    reassignment moves the storage (pointer change), so between this and the
    ``load_state_dict`` post-hook a stale fold cannot survive a weight update.
    """
    return (g[0]._version, g[1]._version, g[2]._version, g[3]._version,
            g[4]._version, g[5]._version, g[6]._version, g[7]._version,
            g[8]._version, g[9]._version, g[0].data_ptr(), g[5].data_ptr())


# ---------------------------------------------------------------------------
# Tile selection.
#
# All five scored cases have e=1.0, so Cin == Cout == C and both convs of a block
# are the *same* GEMM; only (n, C, H) differ:
#
#   n=1 C=128 H=20   P=400    K=1152     n=4 C=16  H=160  P=25600  K=144
#   n=4 C=64  H=40   P=1600   K=576      n=1 C=64  H=40   P=1600   K=576
#   n=1 C=32  H=80   P=6400   K=288
#
# The shape is fixed at module construction, so this is a one-time decision per
# instance, not something the hot path pays for.
#
# ``BLOCK_K`` turned out to be the dimension that matters, and it is the one an
# earlier sweep held fixed at C (one 3x3 tap per k step).  K = 9C is never a
# power-of-two multiple of C beyond C itself, so a wider k block masks lanes
# off -- and wins anyway, because it collapses nine small ``tl.dot`` calls with
# nine sets of operand-layout conversions into one or two large ones.  Swept
# (CUDA-graph timing, so the host is out of the loop), per single fused conv:
#
#   C=128 P=400   n=1  BK 128 -> 512:  5.89 -> 5.56 us   (BK=C was 7.28)
#   C=16  P=25600 n=4  BK  16 ->  32: 11.22 -> 10.28 us
#   C=64  P=1600  n=4  BK  64 -> 128:  8.84 ->  8.38 us
#   C=64  P=1600  n=1  BK  64 -> 1024: 5.69 ->  4.85 us
#   C=32  P=6400  n=1  BK  32 -> 128:  5.21 ->  4.27 us
#
# Anchors from the same run: an empty kernel launch is 0.69-0.94 us and a plain
# copy of the same bytes 1.4-1.9 us, so these convs are 2.5-5x a memory-bound
# pass.  ``ROW_TILED`` lost every case and stays available but unused.
# ---------------------------------------------------------------------------

# (C, P, n) -> (BLOCK_P, BLOCK_CO, BLOCK_K, num_warps)
_TILES: dict[tuple[int, int, int], tuple[int, int, int, int]] = {
    #  (C,     P,   n): (BP,  BCO,   BK, warps)     swept us   runner-up
    (128,   400, 1): (32, 16, 512, 4),          # 5.56   BK256 5.64
    (16,  25600, 4): (64, 16, 32, 4),           # 10.28  BP128/BK32 10.32
    (64,   1600, 4): (128, 32, 128, 4),         # 8.38   BP64/BK64 8.51
    (64,   1600, 1): (32, 32, 1024, 4),         # 4.85   BP64/BCO16/BK128 4.97
    (32,   6400, 1): (64, 32, 128, 4),          # 4.27   BK256 4.27
}


def _num_sms() -> int:
    return torch.cuda.get_device_properties(
        torch.cuda.current_device()).multi_processor_count


def _pick_cfg(n: int, c: int, p: int, cout: int, k: int):
    """Swept entry when we have one, otherwise a deliberately narrow heuristic.

    The heuristic is bounded rather than adventurous because the fp32 accumulator
    is a cliff, not a gradient: ``BLOCK_P * BLOCK_CO`` above ~2048 spills, and
    spilling is not a 20% effect.  Measured on the 20x20 case against 5.56 us for
    the winner, BP128/BCO32 took 297 us at 4 warps and 19 us at 8, and
    BP256/BCO64 took 2365 us at 2 warps.  So the tile is capped and the warp count
    scales with it.  ``BLOCK_K`` defaults to 128, which was within 6% of the swept
    best on four of the five scored shapes.
    """
    tile = _TILES.get((c, p, n))
    if tile is not None:
        return tile
    block_co = min(triton.next_power_of_2(cout), 32)
    ctas32 = n * triton.cdiv(p, 32) * triton.cdiv(cout, block_co)
    block_p = 32 if ctas32 < 2 * _num_sms() else 64
    while block_p * block_co > 2048 and block_p > 16:
        block_p //= 2
    block_k = min(triton.next_power_of_2(k), 128)
    num_warps = 4 if block_p * block_co <= 1024 else 8
    return (block_p, block_co, block_k, num_warps)


# ---------------------------------------------------------------------------
# Host side: one fold per instance, one plan per (shape, dtype, device).
#
# Every host-side decision -- the fold, the tile constexprs, the grid, the
# compiled-kernel handle, even the fully assembled positional argument list for
# the launcher -- is computed once and reused.  With ~120 us of baseline dispatch
# against ~5 us of GPU work, this bookkeeping *is* the optimization target: a
# steady-state call costs one guard, one allocation, one stream query, five list
# stores and two launcher calls.
# ---------------------------------------------------------------------------

# Layout of a prebuilt launcher call list: the launcher's nine-entry fixed
# prologue (grid x3, stream, function, packed metadata, then the launch-metadata
# / enter-hook / exit-hook slots, all None because no profiling hook is
# installed), then the kernel's own arguments in declaration order.
_C_STREAM = 3
_C_X = 9
_C_RES = 12
_C_Y = 13


def _kernel_args(wt, bias, n, c, h, w, cout, residual, block_p, block_co,
                 block_k, num_warps, row_tiled=False):
    """Grid + positional args + launch options for one fused conv."""
    p = h * w
    k = c * 9
    tpr = triton.cdiv(w, block_p)
    grid = ((tpr * h) if row_tiled else triton.cdiv(p, block_p),
            triton.cdiv(cout, block_co), n)
    args = [None, wt, bias, None, None,
            c, h, w, cout, w, p, tpr, k, 3, 1, 1, 1, 1,
            bool(residual), row_tiled,
            block_co, block_p, block_k, triton.cdiv(k, block_k), 0,
            k % block_k == 0, cout % block_co == 0,
            (w % block_p == 0) if row_tiled else (p % block_p == 0)]
    return grid, args, {"num_warps": num_warps, "num_stages": 1}


class _Plan:
    """Cached launch state for one (shape, dtype, device) of one instance."""

    __slots__ = ("shape", "dtype", "device", "dev", "out_shape", "like",
                 "mid", "call1", "call2", "run1", "run2",
                 "opts1", "opts2", "fast", "guard", "vers")


def _invalidate(module, incompatible_keys=None):  # noqa: ARG001
    """``load_state_dict`` post-hook: drop the folded weights and every plan.

    This is the guard that matters in practice -- the harness shares weights by
    ``load_state_dict`` *after* construction, so a fold baked in any earlier
    would be a fold of the wrong weights.  The per-call version check below
    covers in-place edits on top of it.
    """
    module._fold = None
    module._plans = {}
    module._plan = None
    module._eligible = None


class YOLOBottleneck(nn.Module):
    def __init__(self, c1: int, c2: int, shortcut: bool = True, g: int = 1,
                 k=(3, 3), e: float = 0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = YOLOConv(c1, c_, k[0], 1)
        self.cv2 = YOLOConv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2
        self._c1, self._c2, self._c_ = c1, c2, c_
        self._fold = None          # (guard, vers, w1, b1, w2, b2)
        self._plans = {}
        self._plan = None
        self._eligible = None
        self._needs_grad = True
        self.register_load_state_dict_post_hook(_invalidate)

    # -- eligibility (decided once per instance) ----------------------------
    def _check_eligible(self) -> bool:
        """Only the measured pattern: two contiguous 3x3 stride-1 pad-1 dense
        convs, each followed by SiLU.  Grouped, dilated, other kernel sizes or a
        different activation stay on the eager reference path -- a pattern match
        alone is not evidence of a win."""
        if not _HAVE_L1_PAD:
            return False
        for cv in (self.cv1, self.cv2):
            conv = getattr(cv, "conv", None)
            if conv is None or type(getattr(cv, "act", None)).__name__ != "SiLU":
                return False
            w = getattr(conv, "weight", None)
            if (w is None or w.dim() != 4 or tuple(w.shape[2:]) != (3, 3)
                    or not w.is_contiguous()
                    or getattr(conv, "groups", 1) != 1
                    or tuple(conv.dilation) != (1, 1)
                    or tuple(conv.stride) != (1, 1)
                    or tuple(conv.padding) != (1, 1)):
                return False
            if not getattr(cv, "_is_fused", False) and getattr(cv, "bn", None) is None:
                return False
        return True

    def _bns(self):
        return (None if self.cv1._is_fused else self.cv1.bn,
                None if self.cv2._is_fused else self.cv2.bn)

    # -- the fold, cached per instance (shape-independent) ------------------
    def _get_fold(self):
        bn1, bn2 = self._bns()
        guard = (_guard_tensors(self.cv1.conv, bn1)
                 + _guard_tensors(self.cv2.conv, bn2))
        vers = _guard_vers(guard)
        fold = self._fold
        if fold is not None and fold[1] == vers:
            return fold
        w1, b1 = _fold_bn(self.cv1.conv, bn1)
        w2, b2 = _fold_bn(self.cv2.conv, bn2)
        fold = (guard, vers, w1, b1, w2, b2)
        self._fold = fold
        self._plans = {}
        self._plan = None
        return fold

    # -- plan construction -------------------------------------------------
    def _build_plan(self, x: torch.Tensor):
        if self._eligible is None:
            self._eligible = self._check_eligible()
        if not self._eligible or x.dtype not in (torch.float16, torch.bfloat16):
            return None
        if self.cv1.conv.weight.dtype is not x.dtype:
            return None
        n, c, h, w = (int(v) for v in x.shape)
        if c != self._c1 or h < 1 or w < 1:
            return None
        cfg1 = _pick_cfg(n, c, h * w, self._c_, c * 9)
        cfg2 = _pick_cfg(n, self._c_, h * w, self._c2, self._c_ * 9)

        guard, vers, w1, b1, w2, b2 = self._get_fold()
        self._needs_grad = any(t.requires_grad for t in self.parameters())

        p = _Plan()
        p.guard, p.vers = guard, vers
        p.shape = tuple(x.shape)
        p.dtype = x.dtype
        p.device = x.device
        p.dev = x.get_device()
        p.out_shape = (n, self._c2, h, w)
        p.like = self._c2 == c            # ``empty_like`` is ~1 us cheaper
        p.mid = torch.empty((n, self._c_, h, w), dtype=x.dtype, device=x.device)

        g1, a1, o1 = _kernel_args(w1, b1, n, c, h, w, self._c_, False, *cfg1)
        g2, a2, o2 = _kernel_args(w2, b2, n, self._c_, h, w, self._c2, self.add,
                                  *cfg2)
        a1[0] = x
        a1[3] = p.mid       # unused (RESIDUAL False) but must be a live pointer
        a1[4] = p.mid
        a2[0] = p.mid
        a2[3] = x
        out = torch.empty(p.out_shape, dtype=x.dtype, device=x.device)
        a2[4] = out
        # Compile through the normal path: that both produces the first (correct)
        # result and hands us the CompiledKernel to launch directly afterwards.
        ck1 = _bneck_conv_kernel[g1](*a1, **o1)
        ck2 = _bneck_conv_kernel[g2](*a2, **o2)
        p.opts1, p.opts2 = o1, o2
        p.call1 = [g1[0], g1[1], g1[2], 0, ck1.function, ck1.packed_metadata,
                   None, None, None] + a1
        p.call2 = [g2[0], g2[1], g2[2], 0, ck2.function, ck2.packed_metadata,
                   None, None, None] + a2
        p.run1, p.run2 = ck1.run, ck2.run
        p.fast = True
        # Validate the direct-launch arity once against this Triton build; a
        # launcher whose signature disagrees keeps the module on the normal path.
        try:
            p.run1(g1[0], g1[1], g1[2], _raw_stream(p.dev), ck1.function,
                   ck1.packed_metadata, None, None, None, *a1)
        except TypeError:
            p.fast = False
        return p, out

    # -- forward -----------------------------------------------------------
    def _reference(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y

    def _run(self, p: _Plan, x: torch.Tensor) -> torch.Tensor:
        out = (torch.empty_like(x) if p.like
               else torch.empty(p.out_shape, dtype=p.dtype, device=p.device))
        st = _raw_stream(p.dev)
        c1 = p.call1
        c1[_C_STREAM] = st
        c1[_C_X] = x
        c2 = p.call2
        c2[_C_STREAM] = st
        c2[_C_RES] = x
        c2[_C_Y] = out
        if p.fast:
            p.run1(*c1)
            p.run2(*c2)
        else:
            _bneck_conv_kernel[c1[0], c1[1], c1[2]](*c1[9:], **p.opts1)
            _bneck_conv_kernel[c2[0], c2[1], c2[2]](*c2[9:], **p.opts2)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self._plan
        if (p is None or x.shape != p.shape or x.dtype is not p.dtype
                or x.get_device() != p.dev or not x.is_contiguous()
                or _guard_vers(p.guard) != p.vers
                or (self._needs_grad and torch.is_grad_enabled())):
            return self._setup_forward(x)
        return self._run(p, x)

    def _setup_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Anything the hot path declines: first call for a shape, a weight
        update, a CPU tensor, grad enabled, or a shape the fused path rejects."""
        if not x.is_cuda or (self._needs_grad and torch.is_grad_enabled()):
            return self._reference(x)
        xc = x if x.is_contiguous() else x.contiguous()
        key = (tuple(xc.shape), xc.dtype, xc.get_device())
        fold = self._get_fold()          # may clear self._plans
        p = self._plans.get(key)
        if p is not None:
            if p is False:
                return self._reference(x)
            self._plan = p
            return self._run(p, xc)
        built = self._build_plan(xc)
        if built is None:
            self._plans[key] = False
            return self._reference(x)
        p, out = built
        self._plans[key] = p
        self._plan = p
        return out
