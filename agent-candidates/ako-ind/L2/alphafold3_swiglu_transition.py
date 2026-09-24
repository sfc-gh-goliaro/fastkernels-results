"""SwiGLU transition composites for AlphaFold3 (L2) -- fused Triton kernels.

SwiGLUTransition: LayerNorm -> SwiGLU -> Linear (AF3 Algorithm 11)
ConditionedTransitionBlock: AdaLN -> SwiGLU -> gated output (AF3 Algorithm 25)

Reference: openfold3/core/model/layers/transition.py SwiGLUTransition
           openfold3/core/model/layers/transition.py ConditionedTransitionBlock

Both composites are tiny -- M is 16..368 rows and the whole weight set is 3-9 MB
-- so the baseline's ~10 / ~18 operator launches behind ~20 nested
``nn.Module.__call__``s cost far more in host dispatch than the arithmetic costs
on the GPU. Each composite is collapsed into 2 (SwiGLUTransition) / 3
(ConditionedTransitionBlock) Triton launches, with the norms, the SiLU gate, the
output sigmoid gate and the mask multiply folded into GEMM prologues and
epilogues; the launches go through the compiled kernel's launcher directly, and
each dependent launch is overlapped with its producer via PDL.

The submodules (``layer_norm``, ``swiglu``, ``sigmoid``, ``linear_g``,
``linear_out`` and AdaLN's ``layer_norm_a`` / ``layer_norm_s`` / ``linear_g`` /
``linear_s``) are kept exactly as the baseline builds them and their weights are
read in place, so checkpoint keys still load. Anything the fused path cannot
handle (odd dtype, non-contiguous input, un-fusable broadcast, chunking) falls
back to the reference composition of those submodules.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

try:  # Programmatic Dependent Launch: lets a consumer kernel's grid start (and
    # prefetch the operands that do not come from its producer) while the
    # producer drains. Worth ~2 us per consumer here -- the same order as the
    # kernels themselves.
    from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait
    _HAS_PDL = True
except Exception:  # pragma: no cover - older Triton
    _HAS_PDL = False

    @triton.jit
    def gdc_launch_dependents():
        pass

    @triton.jit
    def gdc_wait():
        pass

from ..L1.sigmoid import Sigmoid
from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear
from .alphafold3_swiglu import AdaLN, SwiGLU

_FAST_DTYPES = (torch.bfloat16, torch.float16)

# Launch knobs, chosen by a cold-L2 sweep over all six captured shapes on B200
# (see ITERATIONS.md). Kept as named constants rather than @triton.autotune so
# the picked config cannot drift between runs, and so the plan cache can bake
# them into a direct-launch handle.
_LATE_WAIT = True        # down kernel: run the output-gate GEMM before gdc_wait
_W_SWIGLU_LN = 8         # swiglu kernel with the LayerNorm prologue folded in
_W_SWIGLU = 4            # swiglu kernel fed an already-normalized row (AdaLN)
_W_ADALN = 8
_W_DOWN = 4
_S_SWIGLU = 3
_S_ADALN = 3
_S_DOWN = 4
# Rows per CTA. A taller tile would re-read each weight slice fewer times, but
# measured strictly and monotonically worse (M=368: 23.6 / 25.6 / 27.7 / 33.8 us
# at BM = 16 / 32 / 64 / 128): with M this small the grid is the scarce resource,
# not bandwidth. Kept at the tl.dot minimum.
_BM_CAP = 16
_BN_CAP = 32             # see _blk_n
_BK_CAP = 256            # see _blk_k


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------
@triton.jit
def _swiglu_fwd(
    X, HOUT, LNW, LNB, WA, WB, M,
    C: tl.constexpr, HD: tl.constexpr, EPS: tl.constexpr,
    DO_LN: tl.constexpr, HAS_LNW: tl.constexpr, HAS_LNB: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    EVEN_M: tl.constexpr, EVEN_K: tl.constexpr, EVEN_N: tl.constexpr,
    PDL_W: tl.constexpr, PDL_T: tl.constexpr,
):
    """h = SiLU(u @ WA.T) * (u @ WB.T), with u = LayerNorm(X) folded into the
    prologue when DO_LN.

    One CTA owns a [BM rows x BN hidden] tile of h and reduces over all of C, so
    nothing crosses CTAs. Both projections are driven from the same normalized
    row inside one launch and one accumulator loop -- the reuse a concatenated
    ``[2*n*c_in, c_in]`` weight pack would buy, without the copy.

    The row statistics use the shifted one-pass form (sums of ``x - x[0]``) so the
    normalization costs one trip over the row rather than two, and shifting keeps
    ``E[d^2] - E[d]^2`` well-conditioned when the row mean is large.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    mm = offs_m >= 0 if EVEN_M else offs_m < M
    mn = offs_n >= 0 if EVEN_N else offs_n < HD
    dt = HOUT.dtype.element_ty
    xrow = X + offs_m[:, None] * C
    wa_row = WA + offs_n[:, None] * C
    wb_row = WB + offs_n[:, None] * C

    if PDL_W:
        gdc_wait()

    if DO_LN:
        x0 = tl.load(X + offs_m * C, mask=mm, other=0.0).to(tl.float32)
        s1 = tl.zeros((BM,), dtype=tl.float32)
        s2 = tl.zeros((BM,), dtype=tl.float32)
        for k0 in tl.range(0, C, BK):
            ok = k0 + tl.arange(0, BK)
            v_ok = (mm[:, None] & (ok[None, :] < C)) if not (EVEN_M and EVEN_K) \
                else tl.full((BM, BK), True, tl.int1)
            v = tl.load(xrow + ok[None, :], mask=v_ok, other=0.0)
            d = tl.where(v_ok, v.to(tl.float32) - x0[:, None], 0.0)
            s1 += tl.sum(d, axis=1)
            s2 += tl.sum(d * d, axis=1)
        md = s1 * (1.0 / C)
        mean = x0 + md
        rstd = 1.0 / tl.sqrt(s2 * (1.0 / C) - md * md + EPS)
    else:
        mean = tl.zeros((BM,), dtype=tl.float32)
        rstd = tl.zeros((BM,), dtype=tl.float32)

    acc_g = tl.zeros((BM, BN), dtype=tl.float32)
    acc_u = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in tl.range(0, C, BK):
        ok = k0 + tl.arange(0, BK)
        kk = ok < C
        if EVEN_K:
            xm = mm[:, None] & tl.full((1, BK), True, tl.int1)
            wm = mn[:, None] & tl.full((1, BK), True, tl.int1)
        else:
            xm = mm[:, None] & kk[None, :]
            wm = mn[:, None] & kk[None, :]
        x = tl.load(xrow + ok[None, :], mask=xm, other=0.0)
        if DO_LN:
            xn = (x.to(tl.float32) - mean[:, None]) * rstd[:, None]
            if HAS_LNW:
                xn = xn * tl.load(LNW + ok, mask=kk, other=0.0).to(tl.float32)[None, :]
            if HAS_LNB:
                xn = xn + tl.load(LNB + ok, mask=kk, other=0.0).to(tl.float32)[None, :]
            u = tl.where(xm, xn, 0.0).to(dt)
        else:
            u = x
        wa = tl.load(wa_row + ok[None, :], mask=wm, other=0.0)
        wb = tl.load(wb_row + ok[None, :], mask=wm, other=0.0)
        acc_g = tl.dot(u, tl.trans(wa), acc_g)
        acc_u = tl.dot(u, tl.trans(wb), acc_u)

    h = (acc_g * tl.sigmoid(acc_g) * acc_u).to(dt)
    tl.store(HOUT + offs_m[:, None] * HD + offs_n[None, :], h,
             mask=mm[:, None] & mn[None, :])
    if PDL_T:
        gdc_launch_dependents()


@triton.jit
def _adaln_fwd(
    A, S, U, LNW, WG, BG, WS, M,
    CA: tl.constexpr, CS: tl.constexpr, EPS: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BKA: tl.constexpr, BKS: tl.constexpr,
    NIT: tl.constexpr, EVEN_M: tl.constexpr, EVEN_KA: tl.constexpr,
    EVEN_KS: tl.constexpr, EVEN_N: tl.constexpr,
    PDL_W: tl.constexpr, PDL_T: tl.constexpr,
):
    """u = sigmoid(s_n @ WG.T + BG) * (LN_a(a) + s_n @ WS.T), s_n = LN_s(s).

    One CTA owns a [BM rows x BN channels] tile of u. ``linear_g`` and
    ``linear_s`` both consume the same normalized s row, so one normalization and
    one pass over s feeds both projections. ``layer_norm_a`` has no affine params
    and ``layer_norm_s`` has weight but no bias, exactly as AdaLN builds them.

    The a-row statistics need the whole row while the output only needs the BN
    slice, so the row is read once for the reduction and once for the slice; both
    are tiny and L2-resident after the first CTA touches them.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    mm = offs_m >= 0 if EVEN_M else offs_m < M
    mn = offs_n >= 0 if EVEN_N else offs_n < CA
    dt = U.dtype.element_ty
    arow = A + offs_m[:, None] * CA
    srow = S + offs_m[:, None] * CS
    wg_row = WG + offs_n[:, None] * CS
    ws_row = WS + offs_n[:, None] * CS

    if PDL_W:
        gdc_wait()

    # Everything that does not depend on a reduction result is issued first, and
    # the two row reductions share one unrolled loop, so the s-side and a-side
    # loads sit in the same window of outstanding requests instead of costing one
    # exposed HBM latency each. These kernels are latency-bound, not
    # bandwidth-bound: with M this small each CTA is alone on its SM, so the
    # number of *sequential* load phases is what the runtime tracks.
    nm = mm[:, None] & mn[None, :]
    aslice = tl.load(arow + offs_n[None, :], mask=nm, other=0.0)
    bgv = tl.load(BG + offs_n, mask=mn, other=0.0).to(tl.float32)
    s0 = tl.load(S + offs_m * CS, mask=mm, other=0.0).to(tl.float32)
    a0 = tl.load(A + offs_m * CA, mask=mm, other=0.0).to(tl.float32)
    s1 = tl.zeros((BM,), dtype=tl.float32)
    s2 = tl.zeros((BM,), dtype=tl.float32)
    a1 = tl.zeros((BM,), dtype=tl.float32)
    a2 = tl.zeros((BM,), dtype=tl.float32)
    for i in tl.static_range(NIT):
        if i * BKS < CS:
            oks = i * BKS + tl.arange(0, BKS)
            sok = mm[:, None] & ((oks[None, :] < CS) if not EVEN_KS
                                 else tl.full((1, BKS), True, tl.int1))
            sv0 = tl.load(srow + oks[None, :], mask=sok, other=0.0)
            sd = tl.where(sok, sv0.to(tl.float32) - s0[:, None], 0.0)
            s1 += tl.sum(sd, 1)
            s2 += tl.sum(sd * sd, 1)
        if i * BKA < CA:
            oka = i * BKA + tl.arange(0, BKA)
            aok = mm[:, None] & ((oka[None, :] < CA) if not EVEN_KA
                                 else tl.full((1, BKA), True, tl.int1))
            av0 = tl.load(arow + oka[None, :], mask=aok, other=0.0)
            ad = tl.where(aok, av0.to(tl.float32) - a0[:, None], 0.0)
            a1 += tl.sum(ad, 1)
            a2 += tl.sum(ad * ad, 1)
    smd = s1 * (1.0 / CS)
    s_mean = s0 + smd
    s_rstd = 1.0 / tl.sqrt(s2 * (1.0 / CS) - smd * smd + EPS)
    amd = a1 * (1.0 / CA)
    a_mean = a0 + amd
    a_rstd = 1.0 / tl.sqrt(a2 * (1.0 / CA) - amd * amd + EPS)

    acc_g = tl.zeros((BM, BN), dtype=tl.float32)
    acc_s = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in tl.range(0, CS, BKS):
        ok = k0 + tl.arange(0, BKS)
        kk = ok < CS
        if EVEN_KS:
            sm = mm[:, None] & tl.full((1, BKS), True, tl.int1)
            wm = mn[:, None] & tl.full((1, BKS), True, tl.int1)
        else:
            sm = mm[:, None] & kk[None, :]
            wm = mn[:, None] & kk[None, :]
        sv = tl.load(srow + ok[None, :], mask=sm, other=0.0)
        sn = (sv.to(tl.float32) - s_mean[:, None]) * s_rstd[:, None]
        sn = sn * tl.load(LNW + ok, mask=kk, other=0.0).to(tl.float32)[None, :]
        sn = tl.where(sm, sn, 0.0).to(dt)
        wg = tl.load(wg_row + ok[None, :], mask=wm, other=0.0)
        ws = tl.load(ws_row + ok[None, :], mask=wm, other=0.0)
        acc_g = tl.dot(sn, tl.trans(wg), acc_g)
        acc_s = tl.dot(sn, tl.trans(ws), acc_s)
    acc_g += bgv[None, :]

    an = (aslice.to(tl.float32) - a_mean[:, None]) * a_rstd[:, None]
    u = (tl.sigmoid(acc_g) * (an + acc_s)).to(dt)
    tl.store(U + offs_m[:, None] * CA + offs_n[None, :], u, mask=nm)
    if PDL_T:
        gdc_launch_dependents()


@triton.jit
def _down_fwd(
    OUT, SRAW, MASK, H, WO, WG, BG, M,
    C: tl.constexpr, HD: tl.constexpr, CS: tl.constexpr,
    HAS_GATE: tl.constexpr, HAS_MASK: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, BKS: tl.constexpr,
    EVEN_M: tl.constexpr, EVEN_K: tl.constexpr, EVEN_KS: tl.constexpr,
    EVEN_N: tl.constexpr, LATE_WAIT: tl.constexpr,
    PDL_W: tl.constexpr, PDL_T: tl.constexpr,
):
    """out = sigmoid(SRAW @ WG.T + BG) * (H @ WO.T) * MASK.

    The output gate produces exactly this kernel's [BM x BN] output tile, so its
    GEMM over CS rides along here instead of taking its own launch, and because
    none of its operands come from the producer it is computed *before* the
    grid-dependency wait -- its memory latency hides under the producer's tail.
    The mask is a per-row scalar and folds in at the same point; with no mask,
    ``HAS_MASK`` is off and nothing is read or multiplied.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    mm = offs_m >= 0 if EVEN_M else offs_m < M
    mn = offs_n >= 0 if EVEN_N else offs_n < C

    if PDL_W and not LATE_WAIT:
        gdc_wait()

    gacc = tl.zeros((BM, BN), dtype=tl.float32)
    if HAS_GATE:
        srow = SRAW + offs_m[:, None] * CS
        wg_row = WG + offs_n[:, None] * CS
        for k0 in tl.range(0, CS, BKS):
            ok = k0 + tl.arange(0, BKS)
            if EVEN_KS:
                sm = mm[:, None] & tl.full((1, BKS), True, tl.int1)
                wm = mn[:, None] & tl.full((1, BKS), True, tl.int1)
            else:
                kk = ok[None, :] < CS
                sm = mm[:, None] & kk
                wm = mn[:, None] & kk
            sv = tl.load(srow + ok[None, :], mask=sm, other=0.0)
            wg = tl.load(wg_row + ok[None, :], mask=wm, other=0.0)
            gacc = tl.dot(sv, tl.trans(wg), gacc)
        gacc += tl.load(BG + offs_n, mask=mn, other=0.0).to(tl.float32)[None, :]
    if HAS_MASK:
        mv = tl.load(MASK + offs_m, mask=mm, other=0.0).to(tl.float32)

    if PDL_W and LATE_WAIT:
        gdc_wait()

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    hrow = H + offs_m[:, None] * HD
    wo_row = WO + offs_n[:, None] * HD
    for k0 in tl.range(0, HD, BK):
        ok = k0 + tl.arange(0, BK)
        if EVEN_K:
            hm = mm[:, None] & tl.full((1, BK), True, tl.int1)
            wom = mn[:, None] & tl.full((1, BK), True, tl.int1)
        else:
            kok = ok[None, :] < HD
            hm = mm[:, None] & kok
            wom = mn[:, None] & kok
        h = tl.load(hrow + ok[None, :], mask=hm, other=0.0)
        wo = tl.load(wo_row + ok[None, :], mask=wom, other=0.0)
        acc = tl.dot(h, tl.trans(wo), acc)

    if HAS_GATE:
        acc = acc * tl.sigmoid(gacc)
    if HAS_MASK:
        acc = acc * mv[:, None]

    tl.store(OUT + offs_m[:, None] * C + offs_n[None, :],
             acc.to(OUT.dtype.element_ty), mask=mm[:, None] & mn[None, :])
    if PDL_T:
        gdc_launch_dependents()


# ---------------------------------------------------------------------------
# Launch-configuration helpers (run once per shape, cached on the module)
# ---------------------------------------------------------------------------
_SMS = {}

try:
    _raw_stream = torch._C._cuda_getCurrentRawStream
except AttributeError:  # pragma: no cover - very old torch
    _raw_stream = None


def _direct(jitfn, grid, args, **kw):
    """Compile *jitfn* once and return a handle for launching it directly.

    Triton's ``kernel[grid](...)`` path re-derives the specialization key and
    re-binds every argument on each call: ~17 us of host time for these argument
    counts, which is the dominant cost when the kernels themselves run in a few
    microseconds. ``warmup`` compiles the exact same specialization, after which
    the compiled kernel's launcher can be invoked directly for ~7 us. Returns
    None (keep the ordinary path) if anything about that is unavailable.

    The handle is only valid while the specialization holds: same constexprs,
    same value of the runtime ``M``, same dtypes, and 16-byte-aligned pointers.
    The first three are pinned by the plan cache key; the caller checks the
    alignment of the pointers it does not own itself.
    """
    if _raw_stream is None:
        return None
    try:
        k = jitfn.warmup(*args, grid=grid, **kw)
        run = k.run  # property: loads the cubin and builds the launcher
        return (run, int(grid[0]), int(grid[1]) if len(grid) > 1 else 1,
                k.function, k.packed_metadata)
    except Exception:  # noqa: BLE001 - any Triton-internal change: use the JIT path
        return None


def _num_sms(device: torch.device) -> int:
    n = _SMS.get(device.index)
    if n is None:
        n = torch.cuda.get_device_properties(device).multi_processor_count
        _SMS[device.index] = n
    return n


def _blk_k(k: int, cap: int = 0):
    """Largest power-of-two <= cap that divides k (so the K loop needs no mask)."""
    cap = cap or _BK_CAP
    b = cap
    while b > 16 and k % b != 0:
        b //= 2
    if k % b == 0:
        return b, True
    return min(triton.next_power_of_2(k), cap), False


def _blk_m(m: int):
    """Rows per CTA: the largest power of two <= min(M, _BM_CAP), at least 16."""
    bm = 16
    while bm * 2 <= _BM_CAP and bm * 2 <= m:
        bm *= 2
    return bm, m % bm == 0


def _blk_n(n: int, m_tiles: int, sms: int):
    """Tile width over the N (output) axis.

    Measured on B200 across all six captured shapes: a 32-wide tile is the best
    or tied-best everywhere, even where it halves the CTA count. The narrower
    16-wide tile buys occupancy these shapes cannot use -- M is 16-368, so the
    grid is small either way -- while adding a pass over the activation row per
    tile. Widen past 32 only if it would otherwise produce a very deep grid.
    """
    if n <= 16:
        return 16, n % 16 == 0
    bn = min(_BN_CAP, triton.next_power_of_2(n))
    while bn < 256 and m_tiles * triton.cdiv(n, bn) > 4 * sms:
        bn *= 2
    return bn, n % bn == 0


def _same_sig(a, b) -> bool:
    """Element-wise identity comparison of two plan signatures.

    Tuple equality already short-circuits on per-element identity, so the common
    (unchanged) case is a C-level pointer scan -- cheap enough to run on every
    forward. The fallback is only reached when an element genuinely differs, at
    which point a tensor element's ``__eq__`` would return a tensor and raise on
    ``bool()``; that means "replaced", so it answers False.
    """
    if a is b:
        return True
    try:
        return bool(a == b)
    except RuntimeError:
        return False


def _tensors_ok(*ts) -> bool:
    for t in ts:
        if t is not None and not t.is_contiguous():
            return False
    return True


class _Cfg:
    __slots__ = ("fast", "M", "C", "grid1", "tail1", "kw1", "grid2", "tail2",
                 "kw2", "grid0", "tail0", "kw0", "buf_h", "buf_u", "out_shape",
                 "d0", "d1", "d2", "dev", "direct", "sig")

    def __init__(self):
        self.fast = False
        self.d0 = self.d1 = self.d2 = None
        self.direct = False
        self.sig = None


# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------
class SwiGLUTransition(nn.Module):
    """AF3 Algorithm 11: SwiGLU-based transition.

    Args:
        c_in: Input channel dimension
        n: Factor multiplied to c_in for hidden dimension
    """

    def __init__(self, c_in: int, n: int):
        super().__init__()
        self.c_in = c_in
        self.n = n

        self.layer_norm = LayerNorm(c_in)
        self.swiglu = SwiGLU(c_in, n * c_in)
        self.linear_out = Linear(n * c_in, c_in, bias=False)
        self._plans: dict = {}

    def _sig(self):
        """Identity of every submodule and parameter a plan bakes in.

        A plan stores the parameter *tensors*, so an in-place update (a plain
        ``load_state_dict``) is picked up for free; what it cannot survive is one
        of these objects being *replaced*. Read through ``_modules`` /
        ``_parameters`` rather than attribute access: ``nn.Module.__getattr__``
        costs ~0.3 us a hop, and at ~20 us of host time per forward that was
        enough to push two of the six shapes over a measurable boundary.
        """
        m = self._modules
        ln, sw, lo = m["layer_norm"], m["swiglu"], m["linear_out"]
        sm = sw._modules
        la, lb = sm["linear_a"], sm["linear_b"]
        return (ln, sw, lo, la, lb, la._parameters["weight"],
                lb._parameters["weight"], lo._parameters["weight"],
                ln._parameters["weight"], ln._parameters["bias"])

    def _apply(self, fn, *args, **kwargs):
        # ``.to()`` / ``.cuda()`` / ``.half()`` all funnel through here. A plan
        # holds compiled kernels bound to one device and one dtype plus
        # references to the parameter tensors, none of which survive a move or a
        # re-type, so drop it. This is the cheap place to notice: it runs on
        # mutation instead of on every forward.
        plans = self.__dict__.get("_plans")
        if plans:
            plans.clear()
        return super()._apply(fn, *args, **kwargs)

    # -- reference composition, used whenever the fused path does not apply --
    def _ref(self, x, mask):
        if mask is None:
            mask = x.new_ones(x.shape[:-1])
        mask = mask.unsqueeze(-1)
        x = self.layer_norm(x)
        x = self.swiglu(x)
        x = self.linear_out(x)
        return x * mask

    def _build(self, x, mask):
        cfg = _Cfg()
        ln, sw = self.layer_norm, self.swiglu
        wa, wb, wo = sw.linear_a.weight, sw.linear_b.weight, self.linear_out.weight
        lnw, lnb = ln.weight, ln.bias
        C = x.shape[-1]
        HD = wa.shape[0]
        if (not x.is_cuda or x.dtype not in _FAST_DTYPES or x.dim() < 2
                or C < 16 or HD < 16 or x.numel() >= 2 ** 31
                or not x.is_contiguous()
                or wa.shape[1] != C or wb.shape != wa.shape
                or wo.shape != (C, HD) or ln.normalized_shape != (C,)
                or not ln.promote_fp32
                or wa.dtype != x.dtype or wb.dtype != x.dtype or wo.dtype != x.dtype
                or not _tensors_ok(wa, wb, wo, lnw, lnb)):
            return cfg
        M = x.numel() // C
        has_mask = False
        if mask is not None:
            mu = mask.unsqueeze(-1)
            if (mask.dtype != x.dtype or not mask.is_contiguous()
                    or mask.numel() != M
                    or torch.broadcast_shapes(x.shape, mu.shape) != x.shape):
                return cfg
            has_mask = True

        dev, sms = x.device, _num_sms(x.device)
        BM, evm = _blk_m(M)
        m_tiles = triton.cdiv(M, BM)
        BN1, evn1 = _blk_n(HD, m_tiles, sms)
        BK1, evk1 = _blk_k(C)
        BN2, evn2 = _blk_n(C, m_tiles, sms)
        BK2, evk2 = _blk_k(HD)

        cfg.buf_h = torch.empty((M, HD), dtype=x.dtype, device=dev)
        cfg.M, cfg.C, cfg.out_shape = M, C, x.shape
        cfg.grid1 = (m_tiles, triton.cdiv(HD, BN1))
        cfg.tail1 = (cfg.buf_h, lnw, lnb, wa, wb, M, C, HD, float(ln.eps),
                     True, lnw is not None, lnb is not None, BM, BN1, BK1,
                     evm, evk1, evn1, False, _HAS_PDL)
        cfg.kw1 = {"num_warps": _W_SWIGLU_LN, "num_stages": _S_SWIGLU,
                   "launch_pdl": False}
        cfg.grid2 = (m_tiles, triton.cdiv(C, BN2))
        cfg.tail2 = (cfg.buf_h, wo, cfg.buf_h, cfg.buf_h, M, C, HD, 16,
                     False, has_mask, BM, BN2, BK2, 16, evm, evk2,
                     True, evn2, _LATE_WAIT, _HAS_PDL, False)
        cfg.kw2 = {"num_warps": _W_DOWN, "num_stages": _S_DOWN,
                   "launch_pdl": _HAS_PDL}
        cfg.dev = dev.index if dev.index is not None else torch.cuda.current_device()
        dummy = torch.empty(1, dtype=x.dtype, device=dev)
        cfg.d1 = _direct(_swiglu_fwd, cfg.grid1, (dummy,) + cfg.tail1, **cfg.kw1)
        cfg.d2 = _direct(_down_fwd, cfg.grid2, (dummy, dummy, dummy) + cfg.tail2,
                         **cfg.kw2)
        # All or nothing: a partially-available direct path would leave forward
        # indexing a None handle.
        cfg.direct = cfg.d1 is not None and cfg.d2 is not None
        cfg.sig = self._sig()
        cfg.fast = True
        return cfg

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        chunk_size: int | None = None,
        ckpt_chunk_size: int | None = None,
    ) -> torch.Tensor:
        key = (x.shape, x.dtype, x.device, None if mask is None else mask.shape,
               None if mask is None else mask.dtype)
        cfg = self._plans.get(key)
        if cfg is None:
            if len(self._plans) >= 64:
                self._plans.clear()  # bound the pinned intermediate buffers
            cfg = self._build(x, mask)
            self._plans[key] = cfg
        elif cfg.sig is not None and not _same_sig(cfg.sig, self._sig()):
            cfg = self._build(x, mask)   # a submodule or parameter was replaced
            self._plans[key] = cfg
        if not cfg.fast:
            return self._ref(x, mask)
        M, C = cfg.M, cfg.C
        out = torch.empty_like(x)
        x2 = x.view(M, C)
        m2 = cfg.buf_h if mask is None else mask.view(M)
        d1, d2 = cfg.d1, cfg.d2
        if (cfg.direct and x.data_ptr() % 16 == 0
                and (mask is None or mask.data_ptr() % 16 == 0)):
            st = _raw_stream(cfg.dev)
            d1[0](d1[1], d1[2], 1, st, d1[3], d1[4], None, None, None,
                  x2, *cfg.tail1)
            d2[0](d2[1], d2[2], 1, st, d2[3], d2[4], None, None, None,
                  out.view(M, C), cfg.buf_h, m2, *cfg.tail2)
        else:
            _swiglu_fwd[cfg.grid1](x2, *cfg.tail1, **cfg.kw1)
            _down_fwd[cfg.grid2](out.view(M, C), cfg.buf_h, m2,
                                 *cfg.tail2, **cfg.kw2)
        return out


class ConditionedTransitionBlock(nn.Module):
    """AF3 Algorithm 25: SwiGLU transition with AdaLN-Zero conditioning.

    Submodule names match the reference checkpoint:
    - layer_norm: AdaLN
    - swiglu: SwiGLU
    - linear_g: output gate Linear(c_s, c_a, bias=True)
    - linear_out: Linear(n*c_a, c_a, bias=False)

    Reference: openfold3/core/model/layers/transition.py ConditionedTransitionBlock

    Args:
        c_a: Activation channel dimension
        c_s: Conditioning channel dimension
        n: Factor for hidden dimension
    """

    def __init__(self, c_a: int, c_s: int, n: int):
        super().__init__()
        self.layer_norm = AdaLN(c_a=c_a, c_s=c_s)
        self.swiglu = SwiGLU(c_a, n * c_a)
        self.sigmoid = Sigmoid()
        self.linear_g = Linear(c_s, c_a, bias=True)
        self.linear_out = Linear(n * c_a, c_a, bias=False)
        self._plans: dict = {}

    def _sig(self):
        """See SwiGLUTransition._sig."""
        m = self._modules
        ada, sw, lo, lg = m["layer_norm"], m["swiglu"], m["linear_out"], m["linear_g"]
        am, sm = ada._modules, sw._modules
        ag, asx, lns = am["linear_g"], am["linear_s"], am["layer_norm_s"]
        la, lb = sm["linear_a"], sm["linear_b"]
        return (ada, sw, lo, lg, ag, asx, lns, la, lb,
                la._parameters["weight"], lb._parameters["weight"],
                lo._parameters["weight"], lg._parameters["weight"],
                lg._parameters["bias"], ag._parameters["weight"],
                ag._parameters["bias"], asx._parameters["weight"],
                lns._parameters["weight"])

    def _apply(self, fn, *args, **kwargs):
        # ``.to()`` / ``.cuda()`` / ``.half()`` all funnel through here. A plan
        # holds compiled kernels bound to one device and one dtype plus
        # references to the parameter tensors, none of which survive a move or a
        # re-type, so drop it. This is the cheap place to notice: it runs on
        # mutation instead of on every forward.
        plans = self.__dict__.get("_plans")
        if plans:
            plans.clear()
        return super()._apply(fn, *args, **kwargs)

    def _ref(self, a, s, mask):
        if mask is None:
            mask = a.new_ones(a.shape[:-1])
        mask = mask.unsqueeze(-1)
        a = self.layer_norm(a, s)
        b = self.swiglu(a)
        a = self.sigmoid(self.linear_g(s)) * self.linear_out(b)
        return a * mask

    def _build(self, a, s, mask):
        cfg = _Cfg()
        ada, sw = self.layer_norm, self.swiglu
        lna, lns = ada.layer_norm_a, ada.layer_norm_s
        wg_a, bg_a, ws_a = ada.linear_g.weight, ada.linear_g.bias, ada.linear_s.weight
        wa, wb = sw.linear_a.weight, sw.linear_b.weight
        wo = self.linear_out.weight
        wg_o, bg_o = self.linear_g.weight, self.linear_g.bias
        lnsw = lns.weight
        CA = a.shape[-1]
        CS = s.shape[-1]
        HD = wa.shape[0]
        if (not a.is_cuda or a.dtype not in _FAST_DTYPES or a.dim() < 2
                or s.dim() < 2 or CA < 16 or CS < 16 or HD < 16
                or a.numel() >= 2 ** 31 or s.numel() >= 2 ** 31
                or not a.is_contiguous() or not s.is_contiguous()
                or s.dtype != a.dtype
                or lna.normalized_shape != (CA,) or lns.normalized_shape != (CS,)
                or not lna.promote_fp32 or not lns.promote_fp32
                or lna.weight is not None or lna.bias is not None
                or lnsw is None or lns.bias is not None
                or wg_a.shape != (CA, CS) or ws_a.shape != (CA, CS)
                or bg_a is None or wa.shape != (HD, CA) or wb.shape != (HD, CA)
                or wo.shape != (CA, HD) or wg_o.shape != (CA, CS) or bg_o is None
                or any(t.dtype != a.dtype for t in
                       (wg_a, bg_a, ws_a, wa, wb, wo, wg_o, bg_o, lnsw))
                or not _tensors_ok(wg_a, bg_a, ws_a, wa, wb, wo, wg_o, bg_o, lnsw)):
            return cfg
        M = a.numel() // CA
        if (s.numel() // CS != M
                or torch.broadcast_shapes(a.shape, s.shape[:-1] + (CA,)) != a.shape):
            return cfg
        has_mask = False
        if mask is not None:
            mu = mask.unsqueeze(-1)
            if (mask.dtype != a.dtype or not mask.is_contiguous()
                    or mask.numel() != M
                    or torch.broadcast_shapes(a.shape, mu.shape) != a.shape):
                return cfg
            has_mask = True

        dev, sms = a.device, _num_sms(a.device)
        BM, evm = _blk_m(M)
        m_tiles = triton.cdiv(M, BM)
        BN0, evn0 = _blk_n(CA, m_tiles, sms)
        BKA, evka = _blk_k(CA)
        BKS, evks = _blk_k(CS)
        BN1, evn1 = _blk_n(HD, m_tiles, sms)
        BK1, evk1 = _blk_k(CA)
        BN2, evn2 = _blk_n(CA, m_tiles, sms)
        BK2, evk2 = _blk_k(HD)

        cfg.buf_u = torch.empty((M, CA), dtype=a.dtype, device=dev)
        cfg.buf_h = torch.empty((M, HD), dtype=a.dtype, device=dev)
        cfg.M, cfg.C = M, CA
        cfg.grid0 = (m_tiles, triton.cdiv(CA, BN0))
        cfg.tail0 = (cfg.buf_u, lnsw, wg_a, bg_a, ws_a, M, CA, CS, float(lns.eps),
                     BM, BN0, BKA, BKS,
                     max(triton.cdiv(CA, BKA), triton.cdiv(CS, BKS)),
                     evm, evka, evks, evn0, False, _HAS_PDL)
        cfg.kw0 = {"num_warps": _W_ADALN, "num_stages": _S_ADALN,
                   "launch_pdl": False}
        cfg.grid1 = (m_tiles, triton.cdiv(HD, BN1))
        cfg.tail1 = (cfg.buf_h, lnsw, lnsw, wa, wb, M, CA, HD, float(lna.eps),
                     False, False, False, BM, BN1, BK1, evm, evk1, evn1,
                     _HAS_PDL, _HAS_PDL)
        cfg.kw1 = {"num_warps": _W_SWIGLU, "num_stages": _S_SWIGLU,
                   "launch_pdl": _HAS_PDL}
        cfg.grid2 = (m_tiles, triton.cdiv(CA, BN2))
        cfg.tail2 = (cfg.buf_h, wo, wg_o, bg_o, M, CA, HD, CS,
                     True, has_mask, BM, BN2, BK2, BKS, evm, evk2, evks, evn2,
                     _LATE_WAIT, _HAS_PDL, False)
        cfg.kw2 = {"num_warps": _W_DOWN, "num_stages": _S_DOWN,
                   "launch_pdl": _HAS_PDL}
        cfg.dev = dev.index if dev.index is not None else torch.cuda.current_device()
        dummy = torch.empty(1, dtype=a.dtype, device=dev)
        cfg.d0 = _direct(_adaln_fwd, cfg.grid0, (dummy, dummy) + cfg.tail0, **cfg.kw0)
        cfg.d1 = _direct(_swiglu_fwd, cfg.grid1, (dummy,) + cfg.tail1, **cfg.kw1)
        cfg.d2 = _direct(_down_fwd, cfg.grid2, (dummy, dummy, dummy) + cfg.tail2,
                         **cfg.kw2)
        # All or nothing: a partially-available direct path would leave forward
        # indexing a None handle.
        cfg.direct = (cfg.d0 is not None and cfg.d1 is not None
                      and cfg.d2 is not None)
        cfg.sig = self._sig()
        cfg.fast = True
        return cfg

    def forward(
        self,
        a: torch.Tensor,
        s: torch.Tensor,
        mask: torch.Tensor | None = None,
        chunk_size: int | None = None,
    ) -> torch.Tensor:
        key = (a.shape, s.shape, a.dtype, a.device,
               None if mask is None else mask.shape,
               None if mask is None else mask.dtype)
        cfg = self._plans.get(key)
        if cfg is None:
            if len(self._plans) >= 64:
                self._plans.clear()  # bound the pinned intermediate buffers
            cfg = self._build(a, s, mask)
            self._plans[key] = cfg
        elif cfg.sig is not None and not _same_sig(cfg.sig, self._sig()):
            cfg = self._build(a, s, mask)   # a submodule or parameter was replaced
            self._plans[key] = cfg
        if not cfg.fast:
            return self._ref(a, s, mask)
        M, CA = cfg.M, cfg.C
        s2d = s.view(M, s.shape[-1])
        out = torch.empty_like(a)
        a2 = a.view(M, CA)
        m2 = cfg.buf_h if mask is None else mask.view(M)
        d0, d1, d2 = cfg.d0, cfg.d1, cfg.d2
        if (cfg.direct and (a.data_ptr() | s.data_ptr()) % 16 == 0
                and (mask is None or mask.data_ptr() % 16 == 0)):
            st = _raw_stream(cfg.dev)
            d0[0](d0[1], d0[2], 1, st, d0[3], d0[4], None, None, None,
                  a2, s2d, *cfg.tail0)
            d1[0](d1[1], d1[2], 1, st, d1[3], d1[4], None, None, None,
                  cfg.buf_u, *cfg.tail1)
            d2[0](d2[1], d2[2], 1, st, d2[3], d2[4], None, None, None,
                  out.view(M, CA), s2d, m2, *cfg.tail2)
        else:
            _adaln_fwd[cfg.grid0](a2, s2d, *cfg.tail0, **cfg.kw0)
            _swiglu_fwd[cfg.grid1](cfg.buf_u, *cfg.tail1, **cfg.kw1)
            _down_fwd[cfg.grid2](out.view(M, CA), s2d, m2, *cfg.tail2, **cfg.kw2)
        return out
