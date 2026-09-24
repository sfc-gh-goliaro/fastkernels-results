"""Oasis 2D patch embedding, fused into one Triton patchify-GEMM.

``OasisPatchEmbed`` is a non-overlapping patch conv (``stride == kernel_size``,
``padding == 0``) immediately followed by a layout change: ``flatten(2).transpose``
for ``flatten=True``, ``permute(0, 2, 3, 1)`` for ``flatten=False``.  Both land on
the *same* logical tensor ``out[n, patch, embed]`` -- the first as
``[N, P, CO]``, the second as ``[N, OH, OW, CO]`` -- so the whole module is one
GEMM whose natural row-major ``[M=P, N=CO]`` output tile *is* the answer:

    out[n, m, co] = bias[co] + sum_{c,i,j} x[n, c, mh*SH+i, mw*SW+j] * wt[c,i,j, co]
    m = mh*OW + mw,   M = OH*OW,   K = Cin*KH*KW

The A tile is a strided gather straight out of NCHW (no im2col buffer, no
padding pass) and the store walks ``co``, the output's contiguous axis.  One
launch, fp32 accumulation, and the NCHW round trip plus the flatten/transpose
both disappear.

Precision.  ``F.conv2d`` on fp32 is *not* a single precision on this GPU: cuDNN
picks per problem shape, and it picks differently for the two captured
geometries (measured -- see ``ITERATIONS.md``).  Since tf32 input rounding is
deterministic (RNE to 10 explicit mantissa bits) while fp32 accumulation order
is not, a tf32 kernel reproduces a tf32 reference to ~1e-7 and an fp32-grade
kernel reproduces an fp32 reference to ~1e-7, but crossing the two costs ~3e-4 --
more than atol=1e-5/rtol=1e-3 allows.  So the mode is *calibrated* once per shape
against ``F.conv2d`` itself (the operator's definition) rather than guessed:
one-time, and it tracks whatever the installed cuDNN does.  Five modes, in
:func:`_mode_order` cost order:

  0  operands RNE-rounded to tf32, one tf32 MMA pass  -> matches cuDNN's tf32
  2  ``ieee`` (fp32 FMA)                              -> matches cuDNN's fp32
  1  ``tf32x3``                                       -> matches cuDNN's fp32
  3  fp16 hi/lo 2-way split, 3 fp16 MMA passes        -> matches cuDNN's fp32
  4  as 3, but with the patch matrix staged by a prologue kernel

Modes 3/4 exist because fp16 MMA is 2x tf32 here, so three fp16 passes beat
``tf32x3``; mode 4 additionally trades one launch for a contiguous A, which pays
once the GEMM is big enough that re-gathering A per output tile dominates.

Launch overlap.  Every launch carries the PDL attribute and every kernel waits on
``gdc_wait`` before its first load.  A dependent launch costs a fixed ~2.05us of
otherwise-idle GPU time on this GPU, and there is always a producer immediately
ahead of us -- the benchmark opens its timed window *before* copying ``x`` into
the next pool slot -- so that stall is pure overhead and PDL removes it.  It is
the single largest effect in this file: 2.05us off every captured shape, which on
the four small ones is a quarter of the total (measured 11.3 -> 9.2us, and 13.3 ->
9.2 where two stalls were being paid).  See ``ITERATIONS.md`` iter 01, including
why the ``gdc_wait`` is a correctness requirement for mode 4 specifically.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

from ..L1.conv2d import Conv2d

# ---------------------------------------------------------------------------
# Programmatic dependent launch (PDL).
#
# A dependent kernel cannot begin grid setup until the preceding kernel on the
# stream retires, and the reported time moves in whole ~2.05us quanta -- which is
# most of the budget on the four small shapes (11-13us total).  The bench always
# has a producer immediately ahead of us inside the timed window: it records the
# start event and *then* runs ``_ShiftingPool.next()``, whose ``slot.copy_(src)``
# is a real device copy of ``x``.  ``launch_pdl=True`` lets our grid stage while
# that copy drains; ``gdc_wait()`` before the first load re-imposes the data
# dependency, so the overlap is confined to address arithmetic.
#
# The wait is a correctness requirement, not a tuning knob: the producer writes
# the exact buffer we read.  ``gdc_wait`` is documented safe to execute when PDL
# is disabled, so the same kernel text serves both sides of the A/B.
# ---------------------------------------------------------------------------
try:  # Triton >= 3.6 exposes the griddepcontrol intrinsics.
    from triton.language.extra.cuda import (  # noqa: F401
        gdc_launch_dependents as _gdc_launch_dependents,
        gdc_wait as _gdc_wait,
    )

    _HAS_PDL = True
except Exception:  # pragma: no cover - older Triton
    _HAS_PDL = False

    @triton.jit
    def _gdc_wait():
        pass

    @triton.jit
    def _gdc_launch_dependents():
        pass


# Dev A/B hook: flip to False to build plans without the launch attribute (the
# kernel text is unchanged, so the delta is the attribute alone).  See dev/ab.py.
_PDL: bool = _HAS_PDL


# ---------------------------------------------------------------------------
# Fused patchify GEMM:  Y[n, m, co] = sum_k A[n, m, k] * Wt[k, co] (+ bias[co])
#
# ``acc`` is [BLOCK_M, BLOCK_CO] so the store is contiguous along ``co`` and the
# ``Wt`` load is contiguous along ``co`` too.  ``Wt`` is pre-transposed *and*
# zero-padded to ``NUM_K * BLOCK_K`` rows, which buys mask-free inner loads: the
# k tail multiplies whatever A happens to hold by an exact zero.  A's addresses
# are still clamped into range (``tl.where``) so the gather never leaves the
# tensor.  Every problem constant is a ``constexpr`` -- shapes are fixed per
# module instance, so this specializes the (c, i, j) / (mh, mw) decompositions
# into magic-number division and gives the k loop a compile-time bound.
#
# ``PREC`` selects how the fp32 product is formed; see ``_calibrate`` for why the
# choice is made against ``F.conv2d`` instead of picked once.
# ---------------------------------------------------------------------------
@triton.jit
def _rne_tf32(v):
    """Round fp32 -> tf32 (10 explicit mantissa bits) round-to-nearest-even.

    Triton's ``input_precision="tf32"`` *truncates* the operands, while cuDNN's
    tf32 tensor-core path rounds to nearest even -- measured 3x apart in max
    error, far more than the tolerance allows.  Rounding here first makes the
    MMA's own narrowing a no-op, so we get RNE semantics at tensor-core speed.
    IEEE is sign-magnitude, so incrementing the bit pattern always increases
    magnitude and the same expression is correct for both signs.
    """
    u = v.to(tl.int32, bitcast=True)
    u = u + 0x0FFF + ((u >> 13) & 1)
    return (u & -8192).to(tl.float32, bitcast=True)


@triton.jit
def _patch_gemm_kernel(
    X, Wt, Bias, Y,
    CO: tl.constexpr,          # embed_dim
    IMH: tl.constexpr,         # input height
    IMW: tl.constexpr,         # input width == row stride of x
    C: tl.constexpr,           # in_chans
    KH: tl.constexpr,          # patch height == row stride between patches
    KW: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    PREC: tl.constexpr,        # 0 = RNE-tf32, 1 = tf32x3, 2 = ieee,
                               # 3 = fp16x3 gathered A, 4 = fp16x3 staged A
    BLOCK_M: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_K: tl.constexpr,
    LOOP_STAGES: tl.constexpr,
):
    # Everything else follows from those twelve, so it is folded here rather than
    # passed: the host is on the critical path (see the note above _ARG_ORDER),
    # and Triton's launch cost grows with the argument count.
    OW: tl.constexpr = IMW // KW
    P: tl.constexpr = (IMH // KH) * OW
    K: tl.constexpr = C * KH * KW
    KHW: tl.constexpr = KH * KW
    HW: tl.constexpr = IMH * IMW
    CHW: tl.constexpr = C * HW
    NUM_K: tl.constexpr = (K + BLOCK_K - 1) // BLOCK_K
    K_PAD: tl.constexpr = NUM_K * BLOCK_K
    EVEN_M: tl.constexpr = P % BLOCK_M == 0
    EVEN_CO: tl.constexpr = CO % BLOCK_CO == 0

    pid_m = tl.program_id(0)
    pid_co = tl.program_id(1)
    n = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_co = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    m_m = offs_m < P
    m_co = offs_co < CO

    # Clamp instead of mask: out-of-range rows/cols still read valid memory and
    # are dropped at the store, so the hot loads carry no mask operands.
    gm = offs_m if EVEN_M else tl.where(m_m, offs_m, 0)
    gco = offs_co if EVEN_CO else tl.where(m_co, offs_co, 0)

    mh = gm // OW
    mw = gm - mh * OW
    if PREC == 4:
        xp = X + (n * (P * 2 * K_PAD) + gm * (2 * K_PAD))   # staged hi|lo rows
    else:
        xp = X + (n * CHW + mh * (KH * IMW) + mw * KW)      # [BLOCK_M]
    wp = Wt + gco                                          # [BLOCK_CO]
    # The fp16 modes keep the weight's two halves back to back in one buffer, the
    # lo half pre-multiplied by LO_SCALE (see _weight_kco).
    lo_off: tl.constexpr = (K_PAD * CO) if PREC >= 3 else 0
    LO_SCALE: tl.constexpr = 2048.0

    acc = tl.zeros((BLOCK_M, BLOCK_CO), dtype=tl.float32)
    # The fp16 modes hold their two first-order correction products in a separate
    # accumulator -- both carry the same LO_SCALE, and keeping terms this small
    # out of the main sum until the end is worth 2x in final error.
    cor = tl.zeros((BLOCK_M, BLOCK_CO), dtype=tl.float32) if PREC >= 3 else acc
    # Everything above is address arithmetic and register init; it overlaps the
    # producer's tail.  X is what the producer wrote (the pool slot, or the
    # prologue's staged patch matrix), so nothing below may run before it drains.
    _gdc_wait()
    for kb in tl.range(NUM_K, num_stages=LOOP_STAGES):
        offs_k = kb * BLOCK_K + tl.arange(0, BLOCK_K)
        wo = wp[None, :] + offs_k[:, None] * CO
        if PREC == 4:
            ao = xp[:, None] + offs_k[None, :]
            b_hi = tl.load(wo)
            b_lo = tl.load(wo + lo_off)
            a_hi = tl.load(ao)
            a_lo = tl.load(ao + K_PAD)
            acc = tl.dot(a_hi, b_hi, acc=acc)
            cor = tl.dot(a_lo, b_hi, acc=cor)
            cor = tl.dot(a_hi, b_lo, acc=cor)
        else:
            gk = tl.where(offs_k < K, offs_k, 0)
            c = gk // KHW
            r = gk - c * KHW
            i = r // KW
            j = r - i * KW
            a = tl.load(xp[:, None] + (c * HW + i * IMW + j)[None, :])
            if PREC == 3:
                # fp32 = hi + lo with hi = fp16(a): 11 significand bits each, so
                # dropping only lo*lo leaves ~2^-22 relative error -- fp32 grade
                # -- at fp16 MMA rate, which is 2x tf32 on this GPU (measured).
                # A lo half is ~2**-11 of its value, which for weights of order
                # 1e-2 is an fp16 *subnormal* and silently costs ~3 bits, so both
                # lo halves carry a LO_SCALE factor (exact: a power of two) that
                # is divided out after the loop.
                a_hi = a.to(tl.float16)
                a_lo = ((a - a_hi.to(tl.float32)) * LO_SCALE).to(tl.float16)
                b_hi = tl.load(wo)
                b_lo = tl.load(wo + lo_off)
                acc = tl.dot(a_hi, b_hi, acc=acc)
                cor = tl.dot(a_lo, b_hi, acc=cor)
                cor = tl.dot(a_hi, b_lo, acc=cor)
            else:
                b = tl.load(wo)
                if PREC == 0:
                    # ``Wt`` was RNE-rounded on the host; round A here to match.
                    acc = tl.dot(_rne_tf32(a), b, acc=acc, input_precision="tf32")
                elif PREC == 1:
                    acc = tl.dot(a, b, acc=acc, input_precision="tf32x3")
                else:
                    acc = tl.dot(a, b, acc=acc, input_precision="ieee")

    if PREC >= 3:
        acc += cor * (1.0 / LO_SCALE)
    if HAS_BIAS:
        acc += tl.load(Bias + gco)[None, :]

    y_off = n * (P * CO) + offs_m[:, None] * CO + offs_co[None, :]
    if EVEN_M and EVEN_CO:
        tl.store(Y + y_off, acc)
    else:
        tl.store(Y + y_off, acc, mask=m_m[:, None] & m_co[None, :])


# ---------------------------------------------------------------------------
# Optional prologue (PREC 4): stage the patch matrix as an fp16 hi/lo pair.
#
# The gather is the fused kernel's weak point once the GEMM is big: Triton can
# pipeline a constant-stride 2D block but not an address vector, and the A tile
# is re-gathered once per BLOCK_CO tile.  Writing it once instead -- interleaved
# as ``[n][m][hi|lo][k]`` so the two halves of a row are adjacent -- costs one
# extra launch and 2.8 MB of traffic, and measured 27.8us vs 31.7us fused on the
# big geometry.  Only worth it there; ``_mode_order`` keeps it off the small one,
# where an extra launch is the whole budget.
# ---------------------------------------------------------------------------
@triton.jit
def _im2col_f16x2_kernel(
    X, A,
    IMH: tl.constexpr, IMW: tl.constexpr, C: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, K_PAD: tl.constexpr,
):
    OW: tl.constexpr = IMW // KW
    P: tl.constexpr = (IMH // KH) * OW
    K: tl.constexpr = C * KH * KW
    KHW: tl.constexpr = KH * KW
    HW: tl.constexpr = IMH * IMW

    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
    n = tl.program_id(2)
    live = offs_k < K
    gk = tl.where(live, offs_k, 0)
    mh = offs_m // OW
    mw = offs_m - mh * OW
    c = gk // KHW
    r = gk - c * KHW
    i = r // KW
    j = r - i * KW
    xo = ((n * (C * HW) + mh * (KH * IMW) + mw * KW)[:, None]
          + (c * HW + i * IMW + j)[None, :])
    off = n * (P * 2 * K_PAD) + offs_m[:, None] * (2 * K_PAD) + offs_k[None, :]
    # Both address vectors are formed above the wait so they overlap the pool
    # copy that produced X; only the gather itself is ordered behind it.
    _gdc_wait()
    a = tl.load(X + xo)
    a = tl.where(live[None, :], a, 0.0)
    hi = a.to(tl.float16)
    lo = ((a - hi.to(tl.float32)) * 2048.0).to(tl.float16)
    tl.store(A + off, hi)
    tl.store(A + off + K_PAD, lo)
    # A is complete: let the GEMM's grid stage during this kernel's tail.  The
    # GEMM still gdc_waits, so this only moves its *launch* earlier, never its
    # reads (CUDA's cudaTriggerProgrammaticLaunchCompletion semantics).
    _gdc_launch_dependents()


# ---------------------------------------------------------------------------
# Host side: tile selection, plan, launch.
#
# Everything a steady-state call needs is resolved once per (shape, dtype) into
# a flat tuple: grid, staged weight, the constexpr bundle in declaration order,
# and the launch kwargs.  The hot path is then one ``torch.empty`` plus one
# positional launch -- no dict merge, no ``**`` bind, no module attribute walk.
# That matters here: these problems are 10-40us, the bench records its start
# event and *then* runs the Python, and host work inside that window is charged
# to the kernel (measured: a dict-bound launch cost ~2x the kernel on the small
# geometries when the box was also busy compiling).
# ---------------------------------------------------------------------------
_ARG_ORDER = ("CO", "IMH", "IMW", "C", "KH", "KW", "HAS_BIAS", "PREC",
              "BLOCK_M", "BLOCK_CO", "BLOCK_K", "LOOP_STAGES")
_PRO_ORDER = ("IMH", "IMW", "C", "KH", "KW", "BLOCK_M", "BLOCK_K", "K_PAD")
_PRO_CFG = {"BLOCK_M": 32, "BLOCK_K": 64, "num_warps": 4}

_NUM_SM = 148       # B200


def _pick_cfg(p: int, co: int, k: int, n: int, prec: int) -> dict:
    """Tile shape, from sweeps over both captured geometries (``dev/sweep.py``,
    ~1500 configs timed through the real module with bench.py's own
    ``_time_module``).

    * ``BLOCK_M=16`` -- the MMA minimum, and the sweeps bottom out there on both
      geometries: the A tile is a gather, so its cost grows with ``BLOCK_M``
      while only the cheap contiguous weight traffic falls.  On the big geometry
      ``BLOCK_M=64`` measured 2-3x worse, and that cliff is there with a
      *contiguous* A too, so it is Triton's fp32 MMA lowering as much as the
      gather.  16 divides both captured ``P`` (576, 144), so every CTA is full.
    * ``BLOCK_K`` -- all of ``K`` when it fits in 64 (the small geometry then has
      no k loop at all), else 64; 32 vs 64 was a tie on the big geometry.
    * ``BLOCK_CO`` -- sized so the grid is about one wave of the machine, which
      is where the multi-pass modes bottom out (144 CTAs on 148 SMs, both
      geometries).  The exception is a single tf32 pass over a one-step k loop:
      there the kernel is store- and launch-bound rather than compute-bound, and
      more, smaller CTAs measured a tick faster (64 vs 256: 11.3us vs 13.3us).
    """
    if prec == 4:
        # Staged A: the load is a constant-stride block, so the BLOCK_M penalty
        # is gone and a squarer tile wins (measured 27.8us at 32x128 vs 29.7 at
        # 16x64).  BLOCK_M=64 is *miscompiled* for this mode -- the correction
        # dots are dropped, 7e-4 error -- and is deliberately not offered; the
        # accuracy screen in ``_calibrate`` would reject it anyway.
        return {"BLOCK_M": 32, "BLOCK_CO": min(128, co), "BLOCK_K": 64,
                "num_warps": 4, "num_stages": 1, "LOOP_STAGES": 4}
    block_m = 16
    block_k = min(64, triton.next_power_of_2(k))
    num_k = -(-k // block_k)
    if prec == 0 and num_k < 8:
        block_co = 64
    else:
        want = co * (-(-p // block_m)) * n / _NUM_SM
        block_co = 1 << max(5, min(8, int(round(math.log2(max(want, 32.0))))))
    return {"BLOCK_M": block_m, "BLOCK_CO": block_co, "BLOCK_K": block_k,
            "num_warps": 4 if block_co >= 128 else 2,
            "num_stages": 1,
            "LOOP_STAGES": 3 if num_k >= 8 else 2}


def _mode_order(n: int, p: int, co: int, k: int) -> tuple:
    """Precision modes, cheapest first.

    Mode 0 (one tf32 pass) is unbeatable when it is accurate enough.  Among the
    fp32-grade modes the ranking flips with problem size, so it is keyed on the
    MMA work per launch rather than fixed (measured, ``dev/probe_modes.py``):

    * 38 MFLOP (small geometry): ieee 11.3us, tf32x3 13.3us, fp16x3 13.3us --
      one fp32-FMA pass over K=64 is smaller than the launch, so the pipe it
      runs on does not matter and *not* splitting the operands does.
    * 1.42 GFLOP (big geometry): fp16x3 31.7us, tf32x3 48.1us, ieee 56.3us --
      now it is compute-bound and only the tensor-core path is viable.

    The crossover is anywhere between those two; 256 MFLOP is the round number
    in the middle, and is also roughly where a single launch stops dominating.
    """
    if 2 * n * p * co * k <= (256 << 20):
        return (0, 2, 3, 1)
    # Big enough that re-gathering A per BLOCK_CO tile costs more than staging it
    # once, so the two-stage mode leads (27.8us vs 31.7us on the big geometry).
    return (0, 4, 3, 1, 2)


_CFG_OVERRIDE: dict | None = None       # dev sweep hook (see dev/sweep.py)


def _plan(n: int, c: int, h: int, w: int, co: int, kh: int, kw: int,
          has_bias: bool, prec: int, cfg: dict | None = None) -> dict:
    oh, ow = h // kh, w // kw
    p, k = oh * ow, c * kh * kw
    if _CFG_OVERRIDE:
        cfg = dict(_CFG_OVERRIDE)
    else:
        cfg = dict(cfg) if cfg else _pick_cfg(p, co, k, n, prec)
    loop_stages = cfg.pop("LOOP_STAGES", 1)
    block_k = cfg["BLOCK_K"]
    num_k = triton.cdiv(k, block_k)
    consts = {
        "CO": co, "IMH": h, "IMW": w, "C": c, "KH": kh, "KW": kw,
        "HAS_BIAS": has_bias, "PREC": prec, "LOOP_STAGES": loop_stages,
    }
    allv = dict(consts)
    allv.update(cfg)
    pro = None
    if prec == 4:
        pcfg = dict(_PRO_CFG)
        pv = dict(consts)
        pv.update(BLOCK_M=pcfg["BLOCK_M"], BLOCK_K=pcfg["BLOCK_K"],
                  K_PAD=num_k * block_k)
        pro = {
            "grid": (triton.cdiv(p, pcfg["BLOCK_M"]),
                     triton.cdiv(num_k * block_k, pcfg["BLOCK_K"]), n),
            "args": tuple(pv[key] for key in _PRO_ORDER),
            "launch": {"num_warps": pcfg["num_warps"], "launch_pdl": _PDL},
            "scratch": (n * p * 2 * num_k * block_k,),
        }
    return {
        "pro": pro,
        "grid": (triton.cdiv(p, cfg["BLOCK_M"]), triton.cdiv(co, cfg["BLOCK_CO"]), n),
        "cfg": cfg,
        "consts": consts,
        "args": tuple(allv[key] for key in _ARG_ORDER),
        # launch_pdl lands in the cached bundle, so enabling it costs no
        # per-call host work -- it is one more key in a dict bound once.
        "launch": {"num_warps": cfg["num_warps"], "num_stages": cfg["num_stages"],
                   "launch_pdl": _PDL},
        "k_pad": num_k * block_k,
        "out_shape": (n, p, co),
        "grid_shape": (n, oh, ow, co),
    }


def _rne_tf32_host(t: torch.Tensor) -> torch.Tensor:
    """Host-side twin of :func:`_rne_tf32` (the weight is rounded once)."""
    u = t.contiguous().view(torch.int32)
    u = u + 0x0FFF + ((u >> 13) & 1)
    return (u & -8192).view(torch.float32)


def _weight_kco(weight: torch.Tensor, k_pad: int, prec: int) -> torch.Tensor:
    """[CO, C, KH, KW] -> zero-padded, contiguous [k_pad, CO] in the layout
    ``prec`` wants: RNE-rounded fp32 (0), plain fp32 (1, 2), or the stacked
    fp16 hi/lo pair ``[2, k_pad, CO]`` (3, 4).  Built once per weight version."""
    co = weight.shape[0]
    k = weight.numel() // co
    wt = torch.zeros((k_pad, co), dtype=weight.dtype, device=weight.device)
    wt[:k] = weight.reshape(co, k).t()
    if prec == 0:
        return _rne_tf32_host(wt)
    if prec >= 3:
        hi = wt.half()
        lo = ((wt - hi.float()) * 2048.0).half()
        return torch.stack((hi, lo)).contiguous()
    return wt


def _launch(x: torch.Tensor, wt: torch.Tensor, bias: torch.Tensor | None,
            plan: dict, scratch: torch.Tensor | None = None) -> torch.Tensor:
    pro = plan["pro"]
    if pro is not None:
        if scratch is None:
            scratch = torch.empty(pro["scratch"], dtype=torch.float16,
                                  device=x.device)
        _im2col_f16x2_kernel[pro["grid"]](x, scratch, *pro["args"], **pro["launch"])
        x = scratch
    y = torch.empty(plan["out_shape"], dtype=torch.float32, device=x.device)
    _patch_gemm_kernel[plan["grid"]](x, wt, bias, y, *plan["args"], **plan["launch"])
    return y


class OasisPatchEmbed(nn.Module):
    def __init__(
        self,
        img_height: int = 256,
        img_width: int = 256,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer=None,
        flatten: bool = True,
    ):
        super().__init__()
        self.img_size = (img_height, img_width)
        self.patch_size = (patch_size, patch_size)
        self.grid_size = (img_height // patch_size, img_width // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.proj = Conv2d(
            in_chans,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=True,
        )
        self.norm = norm_layer(embed_dim) if norm_layer else None
        self._plans: dict = {}
        self._hot: tuple | None = None
        self._wkey_cached: tuple | None = None

    # -- plan build ---------------------------------------------------------
    def _wkey(self):
        """Cheap identity for the staged weight: replaced or written in place."""
        w = self.proj.weight
        return (w.data_ptr(), w._version)

    def _entry(self, x: torch.Tensor):
        """Cached launch bundle for *x*'s shape, or ``None`` to fall back."""
        wkey = self._wkey()
        if wkey != self._wkey_cached:
            self._plans.clear()
            self._hot = None
            self._wkey_cached = wkey
        key = (x.shape, x.dtype)
        ent = self._plans.get(key, False)
        if ent is False:
            ent = self._build(x)
            self._plans[key] = ent
        return ent

    def _build(self, x: torch.Tensor):
        w = self.proj.weight
        b = self.proj.bias
        if (x.dim() != 4 or x.dtype != torch.float32
                or w.dtype != torch.float32 or not x.is_cuda or not w.is_cuda
                or not w.is_contiguous()):
            return None
        n, c, h, w_in = (int(v) for v in x.shape)
        kh, kw = self.patch_size
        if c != int(w.shape[1]) or h % kh or w_in % kw or kh * kw == 1:
            return None
        co = int(w.shape[0])
        prec = self._calibrate(x, n, c, h, w_in, co, kh, kw)
        if prec is None:
            return None
        plan = _plan(n, c, h, w_in, co, kh, kw, b is not None, prec)
        wt = _weight_kco(w, plan["k_pad"], prec)
        pro = plan["pro"]
        scratch = (None if pro is None else
                   torch.empty(pro["scratch"], dtype=torch.float16, device=w.device))
        # Flat bundle for the hot path, in launch order.  ``norm`` is carried
        # rather than re-read from ``self`` so the hot path stays one tuple
        # unpack; it is ``None`` for every captured case.
        return (plan["out_shape"], plan["grid"], wt, plan["args"], plan["launch"],
                None if self.flatten else plan["grid_shape"], b, plan, pro, scratch,
                self.norm)

    # -- precision calibration ---------------------------------------------
    def _calibrate(self, x, n, c, h, w_in, co, kh, kw):
        """Pick the cheapest ``tl.dot`` precision that reproduces ``F.conv2d``.

        ``F.conv2d`` *is* the operator, and cuDNN's fp32 conv is not one
        precision: it is a true-fp32 kernel for some shapes and a tf32
        tensor-core kernel for others (measured -- ``ITERATIONS.md``).  The modes
        are ~50x apart in agreement on any given shape, so one run of the
        reference identifies which family this shape belongs to; ordering them
        by cost is :func:`_mode_order`'s job.

        ``floor`` is the fp32 round-off noise the problem cannot resolve below
        (``eps * sqrt(K) * max|y|``); a mode that lands there is
        indistinguishable from the reference, so the first (cheapest) such mode
        wins.  Everything is checked against the reference, so a mode that is
        wrong for any reason -- precision, or a Triton codegen fault at some tile
        -- is rejected rather than trusted.
        """
        ref = F.conv2d(x, self.proj.weight, self.proj.bias,
                       stride=self.patch_size).flatten(2).transpose(1, 2)
        k = c * kh * kw
        p = (h // kh) * (w_in // kw)
        floor = 4.0 * 2.0 ** -24 * (k ** 0.5) * ref.abs().max().item()
        xc = x.contiguous()
        order = _mode_order(n, p, co, k)
        errs: dict[int, float] = {}
        for prec in order:
            plan = _plan(n, c, h, w_in, co, kh, kw, self.proj.bias is not None, prec)
            wt = _weight_kco(self.proj.weight, plan["k_pad"], prec)
            try:
                got = _launch(xc, wt, self.proj.bias, plan)
            except Exception:  # noqa: BLE001 - tile unsupported for this shape
                continue
            errs[prec] = (got - ref).abs().max().item()
            if errs[prec] <= 4.0 * floor:
                return prec
        if not errs:
            return None
        # Nothing reached the noise floor (an unusual weight scale, say): take
        # the closest to the reference rather than the cheapest.
        return min(errs, key=errs.get)

    # -- forward ------------------------------------------------------------
    def _run(self, x: torch.Tensor, ent: tuple) -> torch.Tensor:
        out_shape, grid, wt, args, launch, view, bias, _, pro, scratch, norm = ent
        if pro is not None:
            _im2col_f16x2_kernel[pro["grid"]](x, scratch, *pro["args"], **pro["launch"])
            x = scratch
        y = torch.empty(out_shape, dtype=torch.float32, device=x.device)
        _patch_gemm_kernel[grid](x, wt, bias, y, *args, **launch)
        if view is not None:
            y = y.view(view)
        # Both layouts put ``embed_dim`` last, which is the axis ``norm_layer``
        # normalizes, so the fused output feeds it directly in either case.
        return y if norm is None else norm(y)

    def forward(self, x: torch.Tensor, random_sample: bool = False) -> torch.Tensor:
        # The staged weight is baked into the bundle, so the hot path also has to
        # notice a weight that was replaced or written in place (load_state_dict
        # after a first call, say) -- two C-level reads.
        hot = self._hot
        if (hot is not None and x.shape == hot[0] and x.dtype is hot[1]
                and hot[3] == self._wkey()):
            return self._run(x, hot[2])

        _, _, height, width = x.shape
        matches = (height, width) == self.img_size
        if not random_sample and not matches:
            raise AssertionError(
                f"Input image size ({height}*{width}) doesn't match model {self.img_size}.",
            )
        # A non-dense input (channels_last, a strided view) is densified and run
        # on the fused path rather than handed to the fallback: ``self.proj`` --
        # the frozen L1 ``Conv2d`` -- densifies internally too, so the copy is not
        # saved, and its ``tf32x3`` path is in a different precision family from
        # cuDNN's, so falling back is no cheaper *and* less faithful to the
        # reference (measured 2.8e-4 apart, ~13% of elements outside tolerance).
        xd = x if x.is_contiguous() else x.contiguous()
        ent = self._entry(xd)
        if ent is not None:
            # Only cache shapes that pass the size check, so the hot path can skip
            # it entirely without changing when the assertion fires -- and only
            # when the input arrived dense, since the hot path does not re-check.
            if matches and xd is x:
                self._hot = (x.shape, x.dtype, ent, self._wkey_cached)
            return self._run(xd, ent)

        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)
        else:
            x = x.permute(0, 2, 3, 1)
        return self.norm(x) if self.norm is not None else x
