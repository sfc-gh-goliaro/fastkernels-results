"""YOLOv10 spatial attention block -- fused, three launches, no score matrix.

The captured shapes are tiny: ``x`` is fp16[4,128,20,20] / [1,128,20,20] and the
capture pins ``__init__`` to ``dim=128, num_heads=2, attn_ratio=0.5`` -- *not*
the class defaults -- so head_dim=64, key_dim=32 and n = h*w = 400 tokens.  Two
measurements shape everything below (B200, this harness):

* **The block is 100% host-dispatch bound.**  For the reference, wall-clock host
  enqueue time and amortized device time agree to 0.1% (201.5 vs 201.0 us at
  b=4), and b=1 costs the same as b=4.  Per-call *enqueue* costs here are
  ``F.batch_norm`` 29.7 us, ``F.conv2d`` 1x1 15.8 us, depthwise-3x3 8.7 us, a
  Triton launch ~7 us + ~0.45 us per argument, ``a+b`` 4.4 us, ``torch.empty``
  2.4 us.  The reference issues ~13 ops, so *op count* is the primary cost and
  *argument count* is the secondary one.
* **The materialized score matrix is real but secondary.**  ``attn`` is
  [b,heads,400,400] fp16 written by QK^T, read+written by softmax and read again
  by AV -- ~10 MB against ~1.2 MB for everything else -- but removing it bought
  1.27x on its own.  Removing the *launches* around it bought the rest.

So the block is reorganized into exactly three Triton launches with nothing
between them:

1. ``_qkv_kernel`` -- the qkv 1x1 conv as a GEMM over the pixel axis, with the
   BatchNorm folded into the weight on the host.
2. ``_attn_kernel`` -- QK^T, scale, softmax and AV in one program, so no score
   ever reaches memory.  Q/K/V are read with strided pointers straight out of
   the [b, heads, 2*key_dim+head_dim, n] qkv output, whose (d, n) layout is
   already the layout ``tl.dot`` wants, and the result is written directly in
   [b, c, h, w] order.
3. ``_pe_proj_kernel`` -- the depthwise-3x3 ``pe`` branch (reading ``v`` with
   strided pointers out of the same qkv output, so the reference's
   ``v.reshape`` copy disappears), the residual add and the ``proj`` 1x1 conv,
   all in one program, written in place over the attention output.  Every tile
   here is indexed so the compiler can see its last axis is contiguous; the
   obvious ``(row+di)*ww + (col+dj)`` spelling of the 3x3 taps hides that and
   costs 2.3 us of device time in scalar loads.

All three are chained with programmatic dependent launch, so each consumer's
grid is staged while its producer drains.  That is still the single largest
speedup here: turning it off costs 20.5 -> 27.6 us (b=4) / 18.6 -> 25.3 (b=1).

Both BatchNorm folding and every launch parameter are resolved once into a plan
cached on (shape, dtype, training, weight-version), so a steady-state forward is
a dict hit, two allocations and three launches.  Anything the fast path cannot
express -- an unexpected head geometry, a token count too large, training mode,
a conv that is not 1x1/3x3 -- falls back to the reference formulation.

Two things dominated this round, both of them consequences of the *tile config
and the address arithmetic* rather than the algorithm:

* Kernel 3 addressed its 3x3 taps as ``(row+di)*ww + (col+dj)``.  That is the
  same integer as ``offs_p + di*ww + dj``, but it hides the tile's contiguity
  from the compiler, which then emits scalar 2-byte loads: 87% excessive
  sectors, 3.9 of every 32 B sector used.  Rewriting the offset (and keeping
  row/col only for the bounds mask) is 7.25 -> 4.50 us of device time.
* Kernel 2's tile config had been tuned at the class-default ``num_heads=8``.
  At the captured ``num_heads=2`` the fp32 accumulator is 4x larger and
  ``num_warps=2`` **spills** (255 registers/thread, 100% spill overhead).
  Retuning is 7.93 -> 5.77 us.

Warm device time is now qkv 2.61 + attention 5.77 + pe/proj 4.50 = 12.88 us at
b=4 (10.77 at b=1), against a harness measurement floor of 7.10 us (the timed
window includes the benchmark's own input copy), so the 20.5 / 18.5 us it
reports is ~13 us of kernel and ~7 us of harness.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl
from triton.language.extra.cuda.gdc import gdc_launch_dependents, gdc_wait

from ..L1.softmax import Softmax
from .yolov10_conv import YOLOConv

_LOG2E = tl.constexpr(1.4426950408889634)
_NEG_INF = tl.constexpr(float("-inf"))

# Programmatic dependent launch (Hopper+).  The three kernels form a chain, and
# each is small enough that the CTA-launch ramp is a real fraction of its
# runtime, so staging the consumer's grid while the producer drains is worth
# ~4 us (b=4) / ~2 us (b=1) of the ~25 us total -- see ITERATIONS.md iter 06.
# ``gdc_wait()`` in the consumer is what makes the early launch safe: it blocks
# until every store from the whole producer grid is visible.
_PDL: bool | None = None


def _pdl_ok(device) -> bool:
    global _PDL
    if _PDL is None:
        try:
            _PDL = torch.cuda.get_device_capability(device)[0] >= 9
        except Exception:
            _PDL = False
    return _PDL


# Guards for the fast path.
_MAX_N = 4096
_MAX_ELEMS = 1 << 30          # keep every offset inside int32

# --- tile shapes -------------------------------------------------------------
# All three were picked by sweeping true device time per call -- each config
# replayed from a 40-launch CUDA graph so the host is never the limit.  The
# cliffs are steep and not where occupancy heuristics suggest, so these are
# measurements, not guesses.  Two of the three sweeps below were redone at the
# *captured* head geometry (num_heads=2, head_dim=64); the qkv one did not need
# it, because cout = dim + 2*key_dim*heads is 256 whatever num_heads is, so this
# kernel's shape does not depend on it.  Re-checked anyway over
# BLOCK_P x BLOCK_OC x warps (64 configs): 2.85 us best against 2.96 here, i.e.
# already at its floor.
#
# qkv 1x1 conv (BLOCK_P x BLOCK_OC x warps, us at b=4 / b=1):
#   32x64x2 **2.99 / 2.48** | 64x64x2 2.89 / 2.75 | 16x64x2 3.10 / 2.48
#   128x128x2 153.72 (spill) | 128x256x2 352.75 (spill)
_QKV_BLOCK_P = 32
_QKV_BLOCK_OC = 64
_QKV_WARPS = 2

# attention (BLOCK_M x BLOCK_N x warps), isolated device us at b=4 / b=1.  The
# score block is [BLOCK_M, BLOCK_N] fp32 and the accumulator [BLOCK_M, HEAD_DIM]
# fp32, so ``num_warps`` is what decides whether they fit: at 2 warps this kernel
# spills (ncu: 255 registers/thread, 14976 local-memory spill requests at 100%
# overhead, 3.1% achieved occupancy).  Raising it is most of the win here.
#   BM=32 BN=512 w=8  **6.19 / 6.12**   BM=16 BN=512 w=4  6.50 / **4.59**
#   BM=32 BN=512 w=4    6.21 / 6.07     BM=16 BN=256 w=4  6.64 / 4.98
#   BM=32 BN=256 w=4    6.36 / 6.26     BM=16 BN=256 w=2  6.57 / 5.41
#   BM=32 BN=256 w=2    8.36 / 8.17  <- spills; this was the parent's choice
#   BM=64 BN=64  w=4    8.53 / 8.14     BM=128 anything     >= 16 (spills hard)
# Two facts here reverse the parent's conclusions, both because the parent tuned
# at the class-default num_heads=8 (head_dim=16) and the benchmark runs
# num_heads=2 (head_dim=64), which makes the fp32 accumulator 4x larger:
#   * BLOCK_N=512 > n=400 means the key loop runs once, i.e. all keys resident.
#     That now *wins*; with head_dim=16 tiling the key axis won by 1.34x.
#   * num_warps=2 was optimal there and spills here.
# BLOCK_M is the one axis where the two benched shapes disagree (b=4 wants 32,
# b=1 wants 16), so it is chosen per shape in ``_build_plan`` -- see the note
# there for how far that is calibrated.
_ATTN_CFG_SMALL = (16, 512, 4)      # (BLOCK_M, BLOCK_N, num_warps)
_ATTN_CFG_LARGE = (32, 512, 8)

# pe + add + proj (BLOCK_P x warps).  Latency bound -- b=1 and b=4 cost the same
# -- and *more* pixel tiles beats fatter ones, because with only 25*b programs
# the kernel never fills a wave (0.34 waves/SM measured), so CTA count is the
# binding constraint, not per-CTA arithmetic intensity.  Isolated device us at
# b=4, all with the contiguous tap offsets below:
#   BP=16 w=4 **5.06** | BP=16 w=8 5.62 | BP=16 w=2 8.44 | BP=16 w=1 20.3
#   BP=8  w=4 7.21     | BP=32 w=4 6.91 | BP=32 w=8 6.67 | BP=64 w=8 10.0
# ``num_stages`` is worth exactly nothing here (5.06/5.06/5.09/5.06 for 1/2/3/4):
# the tap loop is a ``static_range``, so there is no loop for Triton to pipeline.
_PE_BLOCK_P = 16
_PE_WARPS = 4


@triton.jit
def _qkv_kernel(X, WB, O, n,
                CIN: tl.constexpr, CPAD: tl.constexpr, COUT: tl.constexpr,
                BLOCK_P: tl.constexpr, BLOCK_OC: tl.constexpr,
                EXACT_P: tl.constexpr, EXACT_C: tl.constexpr,
                EXACT_OC: tl.constexpr, GDC: tl.constexpr):
    """qkv 1x1 conv as ``W[COUT, CIN] @ x[CIN, n]`` + bias, per pixel tile.

    ``WB`` is the folded weight followed by the folded bias in one buffer, and
    both batch strides fall out of the constexpr channel counts, so the launch
    carries 13 arguments instead of 15.  ``F.conv2d`` costs 23.1 us of host time
    and 5.5 us of device time at these shapes; this costs 11.7 and 3.0.
    """
    pid_p = tl.program_id(0)
    pid_oc = tl.program_id(1)
    b = tl.program_id(2)
    offs_p = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
    offs_oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    ic = tl.arange(0, CPAD)
    # `| EXACT_*` folds the matching predicate away at compile time.
    pm = (offs_p < n) | EXACT_P
    cm = (ic < CIN) | EXACT_C
    om = (offs_oc < COUT) | EXACT_OC

    if GDC:
        # Whatever produced ``x`` may itself have been launched with PDL; the
        # wait guarantees its stores are visible before the first load.  Free
        # when the producer never triggers (it returns immediately).
        gdc_wait()
    x_ptrs = X + b * (CIN * n) + ic[:, None] * n + offs_p[None, :]
    w_ptrs = WB + offs_oc[:, None] * CIN + ic[None, :]
    if EXACT_P and EXACT_C:
        x = tl.load(x_ptrs)
    else:
        x = tl.load(x_ptrs, mask=cm[:, None] & pm[None, :], other=0.0)
    if EXACT_C and EXACT_OC:
        w = tl.load(w_ptrs)
    else:
        w = tl.load(w_ptrs, mask=om[:, None] & cm[None, :], other=0.0)

    acc = tl.dot(w, x, out_dtype=tl.float32)
    bias = WB + COUT * CIN
    if EXACT_OC:
        acc += tl.load(bias + offs_oc).to(tl.float32)[:, None]
    else:
        acc += tl.load(bias + offs_oc, mask=om, other=0.0).to(tl.float32)[:, None]

    o_ptrs = O + b * (COUT * n) + offs_oc[:, None] * n + offs_p[None, :]
    if EXACT_P and EXACT_OC:
        tl.store(o_ptrs, acc.to(O.dtype.element_ty))
    else:
        tl.store(o_ptrs, acc.to(O.dtype.element_ty),
                 mask=om[:, None] & pm[None, :])
    if GDC:
        gdc_launch_dependents()


@triton.jit
def _attn_kernel(QKV, O, n, scale,
                 HEAD_DIM: tl.constexpr, KEY_DIM: tl.constexpr,
                 NUM_HEADS: tl.constexpr, BLOCK_M: tl.constexpr,
                 BLOCK_N: tl.constexpr, DPAD: tl.constexpr,
                 GDC: tl.constexpr):
    """QK^T, scale, softmax and AV for one (query tile, batch, head).

    The score block never leaves registers, so the [b, h, n, n] intermediate is
    never written.  ``DPAD`` pads key_dim=8 up to the fp16 MMA's K minimum so
    both products stay on tensor cores instead of falling back to scalar FMA;
    the pad rows of K are zeroed, which is why Q may be loaded unmasked along
    ``d``.  The key axis is tiled with an online rescale -- not because n is
    large (it is 400, and all of K and V would fit) but because the smaller
    score block is what this kernel is actually limited by.
    """
    tot: tl.constexpr = 2 * KEY_DIM + HEAD_DIM
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // NUM_HEADS
    h = pid_bh % NUM_HEADS
    base = QKV + b * (NUM_HEADS * tot * n) + h * (tot * n)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_dk = tl.arange(0, DPAD)
    offs_dv = tl.arange(0, HEAD_DIM)
    m_ok = offs_m < n
    dk_ok = offs_dk < KEY_DIM

    if GDC:
        gdc_wait()
    q = tl.load(base + offs_dk[None, :] * n + offs_m[:, None],
                mask=m_ok[:, None], other=0.0)
    run_max = tl.full((BLOCK_M,), _NEG_INF, tl.float32)
    run_sum = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)

    for start in range(0, n, BLOCK_N):
        offs_n = start + tl.arange(0, BLOCK_N)
        n_ok = offs_n < n
        k = tl.load(base + KEY_DIM * n + offs_dk[:, None] * n + offs_n[None, :],
                    mask=dk_ok[:, None] & n_ok[None, :], other=0.0)
        v = tl.load(base + 2 * KEY_DIM * n + offs_dv[None, :] * n
                    + offs_n[:, None], mask=n_ok[:, None], other=0.0)
        s = tl.dot(q, k, out_dtype=tl.float32) * scale
        s = tl.where(n_ok[None, :], s, _NEG_INF)
        new_max = tl.maximum(run_max, tl.max(s, 1))
        alpha = tl.exp2((run_max - new_max) * _LOG2E)
        p = tl.exp2((s - new_max[:, None]) * _LOG2E)
        run_sum = run_sum * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v,
                                            out_dtype=tl.float32)
        run_max = new_max

    acc = acc / run_sum[:, None]
    tl.store(O + b * (NUM_HEADS * HEAD_DIM * n)
             + (h * HEAD_DIM + offs_dv)[None, :] * n + offs_m[:, None],
             acc.to(O.dtype.element_ty), mask=m_ok[:, None])
    if GDC:
        gdc_launch_dependents()


@triton.jit
def _pe_proj_kernel(QKV, AV, PW, OUT, hh, ww,
                    HEAD_DIM: tl.constexpr, KEY_DIM: tl.constexpr,
                    CH: tl.constexpr, COUT: tl.constexpr,
                    BLOCK_P: tl.constexpr, EXACT_P: tl.constexpr,
                    GDC: tl.constexpr):
    """pe (depthwise 3x3) + residual add + proj (1x1) for one pixel tile.

    A pixel tile needs the whole channel axis anyway -- proj reduces over it --
    so pe, the add and proj's GEMM all fit in one program with no intermediate
    reaching memory.  ``v`` is read straight out of the qkv output: it is
    channels ``[2*KEY_DIM, 2*KEY_DIM+HEAD_DIM)`` of each head, so logical
    channel ``c`` sits on qkv row ``(c // HEAD_DIM) * tot + 2*KEY_DIM +
    c % HEAD_DIM``.  ``PW`` packs pe's weight and bias and proj's weight and
    bias into one buffer (offsets are constexpr), which keeps the launch at 13
    arguments instead of 17 -- host launch cost is ~7 us plus ~0.45 us per
    argument.

    This deliberately stays one CTA per pixel tile holding all of COUT, after
    re-tiling it as a proper GEMM (a [BLOCK_OC, BLOCK_P] accumulator with an
    inner loop over proj's input channels) and sweeping 96 configs found nothing
    that beats it -- see ITERATIONS.md for the whole grid.  The reason is that
    this kernel is latency bound at 0.23 waves per SM: total CTA count, not
    per-CTA arithmetic intensity, is what limits it, and BLOCK_P trades the
    second for the first while BLOCK_OC buys CTAs only by recomputing the pe
    stage once per output-channel tile.  ``row`` and ``col`` below exist purely
    for the neighbour mask; the addresses go through ``offs_p``.
    """
    tot: tl.constexpr = 2 * KEY_DIM + HEAD_DIM
    bpe_off: tl.constexpr = CH * 9
    wpj_off: tl.constexpr = CH * 9 + CH
    bpj_off: tl.constexpr = CH * 9 + CH + CH * CH
    pid_p = tl.program_id(0)
    b = tl.program_id(1)
    n = hh * ww

    offs_p = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
    row = offs_p // ww
    col = offs_p % ww
    cs = tl.arange(0, CH)
    p_ok = (offs_p < n) | EXACT_P

    v_base = (QKV + b * (CH // HEAD_DIM * tot * n)
              + ((cs // HEAD_DIM) * tot + 2 * KEY_DIM + cs % HEAD_DIM)[:, None] * n)

    if GDC:
        gdc_wait()
    acc = tl.zeros((CH, BLOCK_P), tl.float32)
    for di in tl.static_range(-1, 2):
        for dj in tl.static_range(-1, 2):
            rr = row + di
            cc = col + dj
            ok = p_ok & (rr >= 0) & (rr < hh) & (cc >= 0) & (cc < ww)
            # ``rr * ww + cc`` and ``offs_p + di * ww + dj`` are the same
            # integer, but only the second form lets the compiler see that the
            # tile's last axis is contiguous.  Via row/col it cannot, and emits
            # scalar 2-byte loads: 87% excessive sectors, 3.9 of every 32 B
            # sector used.  Same addresses, 8.07 -> 5.07 us.
            val = tl.load(v_base + (offs_p + di * ww + dj)[None, :],
                          mask=ok[None, :], other=0.0)
            wt = tl.load(PW + cs * 9 + (di + 1) * 3 + (dj + 1))
            acc += val.to(tl.float32) * wt.to(tl.float32)[:, None]

    av_ptrs = AV + b * (CH * n) + cs[:, None] * n + offs_p[None, :]
    if EXACT_P:
        av = tl.load(av_ptrs)
    else:
        av = tl.load(av_ptrs, mask=p_ok[None, :], other=0.0)
    acc += av.to(tl.float32) + tl.load(PW + bpe_off + cs).to(tl.float32)[:, None]

    oc = tl.arange(0, COUT)
    wpj = tl.load(PW + wpj_off + oc[:, None] * CH + cs[None, :])
    out = tl.dot(wpj, acc.to(av.dtype), out_dtype=tl.float32)
    out += tl.load(PW + bpj_off + oc).to(tl.float32)[:, None]

    out_ptrs = OUT + b * (COUT * n) + oc[:, None] * n + offs_p[None, :]
    if EXACT_P:
        tl.store(out_ptrs, out.to(OUT.dtype.element_ty))
    else:
        tl.store(out_ptrs, out.to(OUT.dtype.element_ty), mask=p_ok[None, :])


def _pow2(x: int) -> int:
    return 1 << max(0, (x - 1).bit_length())


def _folded_wb(block):
    """``(weight, bias)`` for a YOLOConv with its eval-mode BatchNorm folded in.

    With fixed running stats a BatchNorm is a per-output-channel affine map, so
    it folds into the conv weight exactly the way ``YOLOConv.fuse()`` folds it
    -- and it is worth 29.7 us of host time per call not to issue.  Cached on
    the block, keyed by the version counters of every tensor that feeds the
    fold, so a later ``load_state_dict`` is picked up.
    """
    conv = block.conv
    w = conv.weight
    bn = getattr(block, "bn", None)
    if bn is None:
        return w, conv.bias
    ver = (w._version, bn.weight._version, bn.bias._version,
           bn.running_mean._version, bn.running_var._version)
    cached = getattr(block, "_ako_fold", None)
    if cached is not None and cached[0] == ver:
        return cached[1], cached[2]
    scale = bn.weight.float() / torch.sqrt(bn.running_var.float() + bn.eps)
    fw = (w.float() * scale.view(-1, *([1] * (w.dim() - 1)))).to(w.dtype)
    fb = bn.bias.float() - bn.running_mean.float() * scale
    if conv.bias is not None:
        fb = fb + conv.bias.float() * scale
    fb = fb.to(w.dtype)
    block._ako_fold = (ver, fw, fb)
    return fw, fb


class YOLOAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, attn_ratio: float = 0.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim ** -0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = YOLOConv(dim, h, 1, act=False)
        self.proj = YOLOConv(dim, dim, 1, act=False)
        self.pe = YOLOConv(dim, dim, 3, 1, g=dim, act=False)
        self._softmax = Softmax(dim=-1)
        self._plan: dict = {}
        # Every tensor the folded weights derive from; their version counters
        # are the plan cache's invalidation key.
        self._watched = tuple(
            t for blk in (self.qkv, self.pe, self.proj)
            for t in (blk.conv.weight, blk.bn.weight, blk.bn.bias,
                      blk.bn.running_mean, blk.bn.running_var))

    # -- planning -------------------------------------------------------------
    def _build_plan(self, x: torch.Tensor, b: int, c: int, hh: int, ww: int):
        """Folded + packed weights and every launch parameter, or None.

        Everything that does not depend on the values in ``x`` is resolved here
        once per (shape, dtype, weight version).
        """
        n = hh * ww
        nh, hd, kd = self.num_heads, self.head_dim, self.key_dim
        # ``c < 16`` is below the fp16 MMA's K minimum, so both the qkv GEMM
        # (K = c) and proj's GEMM (K = c) would fail to compile; the reference
        # path covers it.  ``tl.arange`` needs power-of-two extents, and
        # head_dim / c are used as tile extents directly (padding them would cost
        # more than the rare non-power-of-two geometry is worth).
        if (self.training or n > _MAX_N or c != nh * hd or kd > 64 or c < 16
                or _pow2(hd) != hd or _pow2(c) != c
                or x.dtype not in (torch.float16, torch.bfloat16)):
            return None
        qw, qb = _folded_wb(self.qkv)
        pw, pb = _folded_wb(self.pe)
        ow, ob = _folded_wb(self.proj)
        if qb is None or pb is None or ob is None:
            return None
        cout, cin = int(qw.shape[0]), int(qw.shape[1])
        cpad = _pow2(cin)
        if (cout != nh * (2 * kd + hd) or cin != c or cpad > 512
                or tuple(qw.shape[-2:]) != (1, 1)
                or tuple(ow.shape[-2:]) != (1, 1)
                or tuple(pw.shape[-2:]) != (3, 3)
                or int(pw.shape[1]) != 1 or int(ow.shape[0]) != c
                or b * cout * n >= _MAX_ELEMS):
            return None
        # BLOCK_M: with only a couple of (batch, head) pairs the grid is so short
        # that halving the query tile to double the CTA count wins; with more
        # pairs the extra K/V re-reads that costs are worse than the shortfall.
        # Measured at b=1 (2 pairs: 4.59 us at BM=16 vs 6.12 at BM=32) and b=4
        # (8 pairs: 6.50 vs 6.19).  Only those two points pin the crossover, so
        # the wider config is the default and the narrow one is the exception.
        abm, abn, awarps = (_ATTN_CFG_SMALL if b * nh <= 2 else _ATTN_CFG_LARGE)
        # One buffer per launch instead of one per tensor: -0.45 us of host time
        # per argument dropped.
        qkv_w = torch.cat((qw.reshape(-1), qb))
        pe_proj_w = torch.cat((pw.reshape(-1), pb, ow.reshape(-1), ob))
        return (
            qkv_w, pe_proj_w, cin, cpad, cout,
            ((n + _QKV_BLOCK_P - 1) // _QKV_BLOCK_P,
             (cout + _QKV_BLOCK_OC - 1) // _QKV_BLOCK_OC, b),
            (n % _QKV_BLOCK_P == 0, cpad == cin, cout % _QKV_BLOCK_OC == 0),
            ((n + abm - 1) // abm, b * nh),
            (max(16, _pow2(kd)), abm, abn, awarps),   # (DPAD, BM, BN, warps)
            ((n + _PE_BLOCK_P - 1) // _PE_BLOCK_P, b),
            n % _PE_BLOCK_P == 0,
            _pdl_ok(x.device),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        # ``training`` is part of the key: the folded weights are only valid
        # while BatchNorm is using its running stats, and toggling train()/eval()
        # changes neither the shape nor any weight version counter.
        key = (b, c, h, w, x.dtype, self.training)
        cache = self._plan
        entry = cache.get(key)
        ver = tuple(t._version for t in self._watched)
        if entry is None or entry[0] != ver:
            entry = (ver, self._build_plan(x, b, c, h, w))
            cache[key] = entry
        plan = entry[1]
        if plan is None:
            return self._reference(x)
        (qkv_w, pe_proj_w, cin, cpad, cout, qgrid, qexact, agrid, acfg,
         pgrid, pexact, pdl) = plan

        n = h * w
        qkv = torch.empty((b, cout, n), dtype=x.dtype, device=x.device)
        _qkv_kernel[qgrid](
            x, qkv_w, qkv, n,
            CIN=cin, CPAD=cpad, COUT=cout,
            BLOCK_P=_QKV_BLOCK_P, BLOCK_OC=_QKV_BLOCK_OC,
            EXACT_P=qexact[0], EXACT_C=qexact[1], EXACT_OC=qexact[2],
            GDC=pdl, num_warps=_QKV_WARPS, num_stages=2, launch_pdl=pdl)

        o = torch.empty((b, c, n), dtype=x.dtype, device=x.device)
        _attn_kernel[agrid](
            qkv, o, n, self.scale,
            HEAD_DIM=self.head_dim, KEY_DIM=self.key_dim,
            NUM_HEADS=self.num_heads, BLOCK_M=acfg[1],
            BLOCK_N=acfg[2], DPAD=acfg[0],
            GDC=pdl, num_warps=acfg[3], num_stages=2, launch_pdl=pdl)

        # In place over the attention output: a program reads only the pixels it
        # then overwrites, and that read feeds the proj GEMM whose result is
        # what gets stored, so no program can observe a half-written tile.
        _pe_proj_kernel[pgrid](
            qkv, o, pe_proj_w, o, h, w,
            HEAD_DIM=self.head_dim, KEY_DIM=self.key_dim, CH=c, COUT=c,
            BLOCK_P=_PE_BLOCK_P, EXACT_P=pexact,
            GDC=pdl, num_warps=_PE_WARPS, num_stages=2, launch_pdl=pdl)
        return o.view(b, c, h, w)

    def _reference(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        n = h * w
        qkv = self.qkv(x)
        q, k, v = qkv.view(
            b, self.num_heads, self.key_dim * 2 + self.head_dim, n).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2)
        attn = self._softmax((q.transpose(-2, -1) @ k) * self.scale)
        av = (v @ attn.transpose(-2, -1)).view(b, c, h, w)
        return self.proj(av + self.pe(v.reshape(b, c, h, w)))
