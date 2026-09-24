"""YOLOv10 native neck -- the whole neck as one fused Triton pipeline.

The baseline evaluates the neck as twelve module calls (two nearest upsamples,
four concats, three C2f blocks, a C2fCIB block, a plain Conv and an SCDown).
Composed out of the frozen L2 winners that is already ~5x the eager baseline,
but it is still 28 kernels for 7 GFLOP, and at these sizes the neck is bound on
how many kernels run and how much memory each one touches, not on arithmetic.
Three things are unavailable when the blocks are optimised one at a time:

* **The upsamples and the concats are not work.**  ``cat`` exists only so the
  next ``1x1`` conv can read one tensor, and a nearest ``2x`` upsample only
  renames indices.  ``cat3`` disappears by letting ``c2f_p4.cv2`` write its
  output straight into the slot of ``c2f_n4``'s input buffer that ``cat3`` would
  have copied it to, and ``cat4`` folds into ``c2fcib_n5.cv1`` as a second
  source with its own channel offset.  For the two upsampled concats,
  ``conv1x1(cat(up2(A), B)) == up2(conv1x1(A, Wa)) + conv1x1(B, Wb)``, so the
  ``A`` half is evaluated at the lower resolution -- a quarter of the multiplies
  -- into an fp32 buffer that the other half's epilogue adds through the ``2x``
  index map.  Six kernels and ~19 MB of copy traffic go away.
* **BatchNorm, SiLU, the residual add and the C2f ``chunk``/``cat`` are
  epilogues, not kernels.**  With frozen statistics BN is a per-channel affine
  map, so every conv carries one ``(scale, shift)`` pair applied in fp32 before
  the activation; ``cv1`` writes the head of the ``(2+n)*c`` concat buffer and
  the bottleneck writes the tail, so ``cv2`` reads one contiguous tensor.
* **RepVGGDW is one convolution.**  Its ``7x7`` and ``3x3`` depthwise branches
  each carry their own BN, and ``silu(bn7(c7 x) + bn3(c3 x))`` is
  ``silu(conv7(x, s7*w7 + pad(s3*w3)) + (b7 + b3))``: one fp32 ``7x7`` weight.

What is left is 24 convolutions over a fixed buffer graph.  Each is an implicit
GEMM ``C[Cout, PQ] = W[Cout, K] @ X[K, PQ]`` per image: the weight tile is
contiguous along ``K`` and the activation tile is contiguous along the flat
spatial index.  Keeping that second fact true is the whole game, because a tile
whose addresses Triton cannot prove affine turns into a per-element gather and
costs an order of magnitude more than the arithmetic:

* stride-1 same-padded im2col is a pure *shift* of the flat index
  (``n + (r-1)*W + (s-1)``), so the ``3x3`` path stages exactly like the ``1x1``
  path;
* the one stride-2 conv is tiled a row at a time, which makes the input row a
  scalar and the wanted columns every other element of one contiguous run -- so
  it loads the whole run (twice the bytes, fully coalesced) and drops the odd
  half with ``tl.split``;
* bounds tests are single unsigned compares, and vanish altogether where the
  tile width divides the extent.

The 24 launches are captured into a CUDA graph.  The pipeline is a chain, so the
graph overlaps nothing -- it exists because the bench's L2 flush only hides
~30 us of host time and 24 Triton launches need more than that.  The live inputs
are staged into the graph's fixed buffers with one ``_foreach_copy_``; all
routing stays on the GPU.

Per-kernel efficiency here is far more sensitive to the SM clock than a chain of
module calls is (a good part of the latter's wall time is host dispatch), so
which of the two is faster depends on the state of the machine.  The first
forward for a shape therefore times both, the way the bench will, and keeps the
winner.  Anything the kernels do not cover -- a non-fp16 or non-CUDA input,
unexpected channel counts, no Triton, a capture failure -- also falls back to
the eager module composition unchanged.
"""


from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.interpolate import Interpolate
from ..L2.yolov10_c2f import YOLOC2f, YOLOC2fCIB
from ..L2.yolov10_concat import YOLOConcat
from ..L2.yolov10_conv import YOLOConv
from ..L2.yolov10_scdown import YOLOSCDown

try:
    import triton
    import triton.language as tl

    _HAVE_TRITON = True
except Exception:  # pragma: no cover - Triton missing
    _HAVE_TRITON = False


if _HAVE_TRITON:

    @triton.jit
    def _k_gemm(
        out_ptr, w_ptr, sc_ptr, sh_ptr, x0_ptr, x1_ptr, res_ptr, ar_ptr,
        # output geometry
        COUT: tl.constexpr, PQ: tl.constexpr, WD: tl.constexpr,
        OIMG: tl.constexpr, OCOFF: tl.constexpr,
        # source 0
        C0: tl.constexpr, PQ0: tl.constexpr, IMG0: tl.constexpr,
        COFF0: tl.constexpr,
        # source 1
        C1: tl.constexpr, PQ1: tl.constexpr, IMG1: tl.constexpr,
        COFF1: tl.constexpr,
        # residual / replicated fp32 addend
        RIMG: tl.constexpr, RPQ: tl.constexpr, RCOFF: tl.constexpr,
        ADDREP: tl.constexpr, ARIMG: tl.constexpr, ARPQ: tl.constexpr,
        ARW: tl.constexpr, ARCOFF: tl.constexpr,
        # conv shape / epilogue
        R: tl.constexpr, STRIDE: tl.constexpr, PAD: tl.constexpr,
        HIN: tl.constexpr, WIN: tl.constexpr,
        NSRC: tl.constexpr, ACT: tl.constexpr, HAS_RES: tl.constexpr,
        ROWT: tl.constexpr, NCH: tl.constexpr, NBLK: tl.constexpr,
        EXACT: tl.constexpr, UNROLL: tl.constexpr, BM: tl.constexpr,
        BN: tl.constexpr, BK: tl.constexpr,
    ):
        """Implicit-GEMM conv: out[Cout, PQ] = W[Cout, K] @ X[K, PQ] per image.

        ``R == 1`` takes one or two sources (the folded ``cat``), each optionally
        read through a nearest ``2x`` index map (the folded upsample); ``R > 1``
        takes a single source and walks ``(r, s)`` outside the channel loop, so a
        k-tile is one *shift* of the flat spatial index and the activation tile
        stays contiguous instead of gathering per element.

        Two spatial tilings, picked per op:

        * ``ROWT == 0`` -- ``BN`` consecutive flat pixels.  No wasted lanes, and
          for stride 1 the shift keeps the tile contiguous even where it crosses
          an image row.
        * ``ROWT == 1`` -- ``BN`` pixels *within one output row*.  Costs
          ``ceil(W/BN)*BN/W`` wasted lanes but makes the row index a scalar, so a
          stride-2 gather becomes ``2*arange`` (a strided access) rather than an
          unanalysable one.  This is what makes the one stride-2 conv affordable.

        Batch always rides on the n grid dimension, which is what keeps the
        pixel index an affine function of a single ``tl.arange``.
        """
        pid_m = tl.program_id(0)
        pid = tl.program_id(1)
        b = pid // NBLK
        blk = pid % NBLK
        m = pid_m * BM + tl.arange(0, BM)
        if ROWT:
            p = blk // NCH
            q = (blk % NCH) * BN + tl.arange(0, BN)
            pqm = (q < WD) if not EXACT else (q == q)
            pq = p * WD + q
        else:
            pq = blk * BN + tl.arange(0, BN)
            pqm = (pq < PQ) if not EXACT else (pq == pq)
            p = pq // WD
            q = pq - p * WD
        kk = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), dtype=tl.float32)

        KTOT: tl.constexpr = C0 + C1

        if R == 1:
            xb0 = x0_ptr + (b * IMG0 + COFF0 * PQ0) + pq
            for k0 in tl.range(0, C0, BK):
                a = tl.load(w_ptr + m[:, None] * KTOT + (k0 + kk)[None, :])
                x = tl.load(xb0[None, :] + (k0 + kk)[:, None] * PQ0,
                            mask=pqm[None, :], other=0.0)
                acc = tl.dot(a, x, acc)
            if NSRC > 1:
                xb1 = x1_ptr + (b * IMG1 + COFF1 * PQ1) + pq
                for k1 in tl.range(0, C1, BK):
                    a = tl.load(w_ptr + m[:, None] * KTOT + (C0 + k1 + kk)[None, :])
                    x = tl.load(xb1[None, :] + (k1 + kk)[:, None] * PQ1,
                                mask=pqm[None, :], other=0.0)
                    acc = tl.dot(a, x, acc)
        else:
            CPT: tl.constexpr = C0 // BK
            KT: tl.constexpr = R * R * C0
            xbase = x0_ptr + (b * IMG0 + COFF0 * PQ0)
            for t in tl.range(0, R * R * CPT, loop_unroll_factor=UNROLL):
                rs = t // CPT
                r = rs // R
                s = rs - r * R
                k0 = (t - rs * CPT) * BK
                hh = p * STRIDE + r - PAD
                a = tl.load(w_ptr + m[:, None] * KT + (t * BK + kk)[None, :])
                if STRIDE == 2:
                    # Row-tiled, so the input row is a scalar and the columns this
                    # tile wants are every *other* element of one contiguous run.
                    # Load the whole run (2x the bytes, fully coalesced) and drop
                    # the odd half -- a strided gather here costs an order of
                    # magnitude more than the wasted bandwidth.
                    col = (blk % NCH) * (2 * BN) + s - PAD + tl.arange(0, 2 * BN)
                    ok2 = (hh.to(tl.uint32, bitcast=True) < HIN) & \
                        (col.to(tl.uint32, bitcast=True) < WIN)
                    xw = tl.load(xbase + (hh * WIN + col)[None, :]
                                 + (k0 + kk)[:, None] * PQ0,
                                 mask=ok2[None, :], other=0.0)
                    x, _ = tl.split(tl.reshape(xw, (BK, BN, 2)))
                else:
                    ww = q + s - PAD
                    ok = (hh.to(tl.uint32, bitcast=True) < HIN) & \
                        (ww.to(tl.uint32, bitcast=True) < WIN) & pqm
                    if ROWT:
                        off = hh * WIN + ww
                    else:
                        off = pq + (r - PAD) * WIN + (s - PAD)
                    x = tl.load(xbase + off[None, :] + (k0 + kk)[:, None] * PQ0,
                                mask=ok[None, :], other=0.0)
                acc = tl.dot(a, x, acc)

        if ADDREP:
            acc += tl.load(ar_ptr + (b * ARIMG + ARCOFF * ARPQ) + m[:, None] * ARPQ
                           + ((p // 2) * ARW + q // 2)[None, :],
                           mask=pqm[None, :], other=0.0)
        sc = tl.load(sc_ptr + m).to(tl.float32)
        sh = tl.load(sh_ptr + m).to(tl.float32)
        y = acc * sc[:, None] + sh[:, None]
        if ACT:
            y = y * tl.sigmoid(y)
        if HAS_RES:
            rv = tl.load(res_ptr + (b * RIMG + RCOFF * RPQ) + m[:, None] * RPQ
                         + pq[None, :], mask=pqm[None, :], other=0.0)
            y = y + rv.to(tl.float32)
        tl.store(out_ptr + (b * OIMG + OCOFF * PQ) + m[:, None] * PQ
                 + pq[None, :], y.to(out_ptr.dtype.element_ty),
                 mask=pqm[None, :])

    @triton.jit
    def _k_dw(
        out_ptr, w_ptr, sc_ptr, sh_ptr, x_ptr, res_ptr,
        PQ: tl.constexpr, WD: tl.constexpr, OIMG: tl.constexpr,
        OCOFF: tl.constexpr, C: tl.constexpr,
        PQ0: tl.constexpr, IMG0: tl.constexpr, COFF0: tl.constexpr,
        RIMG: tl.constexpr, RPQ: tl.constexpr, RCOFF: tl.constexpr,
        K: tl.constexpr, STRIDE: tl.constexpr, PAD: tl.constexpr,
        HIN: tl.constexpr, WIN: tl.constexpr,
        ACT: tl.constexpr, HAS_RES: tl.constexpr,
        ROWT: tl.constexpr, NCH: tl.constexpr, NBLK: tl.constexpr,
        EXACT: tl.constexpr, BC: tl.constexpr, BN: tl.constexpr,
    ):
        """Depthwise KxK over a (BC channel) x (BN pixel) tile.

        Weights are fp32, which is what lets RepVGGDW's two BN'd branches arrive
        here already collapsed into a single 7x7 kernel.  The K*K taps are a
        static loop: with one tile per tap and nothing to pipeline, unrolling
        removes the per-tap index division and lets the bounds tests -- which
        cost more than the arithmetic at 3x3, let alone 7x7 -- be hoisted and
        shared.
        """
        pid = tl.program_id(0)
        cg = pid // NBLK
        blk = pid % NBLK
        b = tl.program_id(1)
        c = cg * BC + tl.arange(0, BC)
        if ROWT:
            p = blk // NCH
            q = (blk % NCH) * BN + tl.arange(0, BN)
            pqm = (q < WD) if not EXACT else (q == q)
            pq = p * WD + q
        else:
            pq = blk * BN + tl.arange(0, BN)
            pqm = (pq < PQ) if not EXACT else (pq == pq)
            p = pq // WD
            q = pq - p * WD
        xb = x_ptr + (b * IMG0 + COFF0 * PQ0) + c[:, None] * PQ0
        acc = tl.zeros((BC, BN), dtype=tl.float32)
        for rs in tl.static_range(0, K * K):
            r = rs // K
            sx = rs % K
            hh = p * STRIDE + (r - PAD)
            wv = tl.load(w_ptr + c * (K * K) + rs)
            if STRIDE == 2:
                col = (blk % NCH) * (2 * BN) + (sx - PAD) + tl.arange(0, 2 * BN)
                ok2 = (hh.to(tl.uint32, bitcast=True) < HIN) & \
                    (col.to(tl.uint32, bitcast=True) < WIN)
                xw = tl.load(xb + (hh * WIN + col)[None, :], mask=ok2[None, :],
                             other=0.0)
                xv, _ = tl.split(tl.reshape(xw, (BC, BN, 2)))
            else:
                ww = q + (sx - PAD)
                ok = (hh.to(tl.uint32, bitcast=True) < HIN) & \
                    (ww.to(tl.uint32, bitcast=True) < WIN) & pqm
                if ROWT:
                    off = hh * WIN + ww
                else:
                    off = pq + (r - PAD) * WIN + (sx - PAD)
                xv = tl.load(xb + off[None, :], mask=ok[None, :], other=0.0)
            acc += wv[:, None].to(tl.float32) * xv.to(tl.float32)
        y = acc * tl.load(sc_ptr + c).to(tl.float32)[:, None] \
            + tl.load(sh_ptr + c).to(tl.float32)[:, None]
        if ACT:
            y = y * tl.sigmoid(y)
        msk = pqm[None, :] & (c < C)[:, None]
        if HAS_RES:
            rv = tl.load(res_ptr + (b * RIMG + RCOFF * RPQ) + c[:, None] * RPQ
                         + pq[None, :], mask=msk, other=0.0)
            y = y + rv.to(tl.float32)
        tl.store(out_ptr + (b * OIMG + OCOFF * PQ) + c[:, None] * PQ
                 + pq[None, :], y.to(out_ptr.dtype.element_ty), mask=msk)


# ---------------------------------------------------------------------------
# Weight lowering
# ---------------------------------------------------------------------------
def _affine(mod) -> tuple[torch.Tensor, torch.Tensor]:
    """A YOLOConv's eval-time per-output-channel (scale, shift) in fp32.

    ``silu(bn(conv(x)))`` with frozen statistics is ``silu(conv(x)*scale + shift)``;
    the convolution weight itself is left bit-exact.
    """
    conv = mod.conv
    bn = getattr(mod, "bn", None)
    if bn is None:
        co = conv.weight.shape[0]
        one = torch.ones(co, device=conv.weight.device, dtype=torch.float32)
        b = (conv.bias.detach().float() if conv.bias is not None
             else torch.zeros_like(one))
        return one, b
    scale = bn.weight.detach().float() / torch.sqrt(
        bn.running_var.detach().float() + bn.eps)
    shift = bn.bias.detach().float() - bn.running_mean.detach().float() * scale
    if conv.bias is not None:
        shift = shift + conv.bias.detach().float() * scale
    return scale.contiguous(), shift.contiguous()


def _w_gemm(mod) -> torch.Tensor:
    """Conv weight as the GEMM's A operand, ``[Cout][R*S][Cin]`` (fp16)."""
    w = mod.conv.weight.detach()
    co, ci, r, s = w.shape
    if r == 1 and s == 1:
        return w.reshape(co, ci).contiguous()
    return w.permute(0, 2, 3, 1).reshape(co, r * s * ci).contiguous()


def _w_dw(mod) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Depthwise weight (fp32 ``[C][K*K]``) plus its epilogue affine."""
    w = mod.conv.weight.detach().float()
    c = w.shape[0]
    scale, shift = _affine(mod)
    return w.reshape(c, -1).contiguous(), scale, shift


def _w_repvggdw(mod) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """RepVGGDW's two BN'd depthwise branches as one fp32 7x7 weight + bias.

    ``silu(bn7(c7(x)) + bn3(c3(x)))`` is ``silu(conv7(x, s7*w7 + pad(s3*w3)) +
    (b7 + b3))``, so the block costs one depthwise launch instead of three.
    """
    s7, b7 = _affine(mod.conv)
    w7 = mod.conv.conv.weight.detach().float() * s7.view(-1, 1, 1, 1)
    if getattr(mod, "_is_fused", False) or not hasattr(mod, "conv1"):
        w, b = w7, b7
    else:
        s3, b3 = _affine(mod.conv1)
        w3 = mod.conv1.conv.weight.detach().float() * s3.view(-1, 1, 1, 1)
        w = w7 + torch.nn.functional.pad(w3, [2, 2, 2, 2])
        b = b7 + b3
    c = w.shape[0]
    one = torch.ones(c, device=w.device, dtype=torch.float32)
    return w.reshape(c, -1).contiguous(), one, b.contiguous()


class _Unsupported(Exception):
    pass


# ---------------------------------------------------------------------------
# The plan: buffers + the fixed 24-launch schedule
# ---------------------------------------------------------------------------
class _Buf:
    """One activation buffer: ``(B, C, H, W)`` fp16, addressed by channel slice."""

    __slots__ = ("t", "C", "H", "W", "PQ", "img")

    def __init__(self, B, C, H, W, device, dtype=torch.float16):
        self.t = torch.empty((B, C, H, W), dtype=dtype, device=device)
        self.C, self.H, self.W = C, H, W
        self.PQ = H * W
        self.img = C * H * W


class _Src:
    """A channel slice ``[coff, coff+C)`` of one buffer, as a conv input."""

    __slots__ = ("buf", "coff", "C")

    def __init__(self, buf, coff, C):
        self.buf, self.coff, self.C = buf, coff, C


# Tile configs measured on this GPU by a sweep over (BM, BN, BK, warps,
# stages, row-tiling, unroll) per op -- the space is not separable, so a
# coordinate descent lands 40% off the joint optimum.  Anything not listed
# falls back to the shape-driven guess below.
_CFG: dict = {
    # ("g", Cout, K, R, stride, PQ, batch): (BM, BN, BK, warps, stages, rowtile, unroll)
    ('g', 32, 32, 3, 1, 6400, 1): (16, 32, 32, 4, 4, False, 3),
    ('g', 64, 64, 1, 1, 6400, 1): (16, 64, 64, 4, 3, False, 3),
    ('g', 64, 64, 3, 1, 1600, 1): (32, 32, 64, 8, 5, False, 1),
    ('g', 64, 64, 3, 2, 1600, 1): (32, 16, 64, 4, 4, True, 1),
    ('g', 64, 96, 1, 1, 6400, 1): (16, 64, 32, 2, 3, False, 9),
    ('g', 64, 128, 1, 1, 1600, 1): (16, 64, 128, 8, 2, False, 9),
    ('g', 128, 128, 1, 1, 1600, 1): (16, 32, 128, 4, 3, False, 9),
    ('g', 128, 192, 1, 1, 1600, 1): (16, 128, 64, 4, 3, False, 1),
    ('g', 128, 256, 1, 1, 400, 1): (16, 32, 64, 4, 5, False, 1),
    ('g', 256, 128, 1, 1, 400, 1): (32, 32, 128, 8, 4, False, 9),
    ('g', 256, 384, 1, 1, 400, 1): (32, 32, 64, 4, 3, False, 9),
    ('g', 32, 32, 3, 1, 6400, 4): (32, 64, 32, 4, 3, False, 9),
    ('g', 64, 64, 1, 1, 6400, 4): (16, 64, 64, 4, 3, False, 1),
    ('g', 64, 64, 3, 1, 1600, 4): (32, 32, 64, 4, 4, False, 1),
    ('g', 64, 64, 3, 2, 1600, 4): (64, 16, 64, 2, 2, True, 1),
    ('g', 64, 96, 1, 1, 6400, 4): (32, 128, 32, 4, 3, False, 9),
    ('g', 64, 128, 1, 1, 1600, 4): (32, 128, 128, 4, 3, False, 1),
    ('g', 128, 128, 1, 1, 1600, 4): (16, 128, 128, 8, 2, False, 9),
    ('g', 128, 192, 1, 1, 1600, 4): (32, 64, 32, 2, 4, False, 3),
    ('g', 128, 256, 1, 1, 400, 4): (32, 64, 64, 2, 4, False, 3),
    ('g', 256, 128, 1, 1, 400, 4): (32, 32, 128, 4, 5, False, 9),
    ('g', 256, 384, 1, 1, 400, 4): (32, 128, 32, 4, 4, False, 1),
    # ("d", C, K, stride, PQ, batch): (BC, BN, warps, stages, rowtile)
    ('d', 128, 3, 1, 400, 1): (16, 32, 8, 1, False),
    ('d', 128, 3, 2, 400, 1): (4, 32, 4, 3, True),
    ('d', 256, 7, 1, 400, 1): (4, 64, 4, 2, False),
    ('d', 128, 3, 1, 400, 4): (2, 128, 4, 3, False),
    ('d', 128, 3, 2, 400, 4): (8, 32, 4, 3, True),
    ('d', 256, 7, 1, 400, 4): (4, 64, 4, 1, False),
}


def _gemm_cfg(cout, k, r, stride, pq, b):
    c = _CFG.get(("g", cout, k, r, stride, pq, b))
    if c is not None:
        return c
    if stride == 2:
        return (min(64, cout), 16, min(64, k), 4, 4, True)
    return (min(64, cout), 64, min(64, k), 4, 4, False)


def _dw_cfg(c_, k, stride, pq, b):
    cfg = _CFG.get(("d", c_, k, stride, pq, b))
    if cfg is not None:
        return cfg
    if stride == 2:
        return (1, 32, 4, 2, True)
    return (1, 256, 4, 2, False)


class _Op:
    """One launch in the plan, with its grid and its fully-static geometry."""

    def __init__(self, kind, **kw):
        self.kind = kind
        self.__dict__.update(kw)

    def launch(self):
        if self.kind == "gemm":
            _k_gemm[self.grid](self.out, self.w, self.sc, self.sh, self.x0,
                               self.x1, self.res, self.ar, **self.kw)
        else:
            _k_dw[self.grid](self.out, self.w, self.sc, self.sh, self.x0,
                             self.res, **self.kw)


class _Plan:
    def __init__(self, neck, B, device):
        self.B = B
        self.ops = []
        self.keep = []  # keeps lowered weights alive
        n = neck

        b4, b3, b5 = 40, 80, 20
        CB1 = _Buf(B, 192, b4, b4, device)
        T1 = _Buf(B, 64, b4, b4, device)
        CB3 = _Buf(B, 192, b4, b4, device)
        CB2 = _Buf(B, 96, b3, b3, device)
        T2 = _Buf(B, 32, b3, b3, device)
        P3 = _Buf(B, 64, b3, b3, device)
        CB4 = _Buf(B, 192, b4, b4, device)
        T3 = _Buf(B, 64, b4, b4, device)
        N4 = _Buf(B, 128, b4, b4, device)
        D1 = _Buf(B, 128, b4, b4, device)
        D2 = _Buf(B, 128, b5, b5, device)
        CB6 = _Buf(B, 384, b5, b5, device)
        TC1 = _Buf(B, 128, b5, b5, device)
        TC2 = _Buf(B, 256, b5, b5, device)
        TC3 = _Buf(B, 256, b5, b5, device)
        TC4 = _Buf(B, 128, b5, b5, device)
        N5 = _Buf(B, 256, b5, b5, device)
        IN3 = _Buf(B, 64, b3, b3, device)
        IN4 = _Buf(B, 128, b4, b4, device)
        IN5 = _Buf(B, 256, b5, b5, device)
        G0 = _Buf(B, 128, b5, b5, device, torch.float32)
        G1 = _Buf(B, 64, b4, b4, device, torch.float32)
        self.inputs = [IN3.t, IN4.t, IN5.t]
        self.outputs = [P3.t, N4.t, N5.t]

        g, dw = self._gemm, self._dw
        # p4 branch. cv1 reads cat(up2(p5), p4); the up2'd half is evaluated at
        # 20x20 into G0 and added back through the 2x index map in the epilogue.
        g(n.c2f_p4.cv1, [_Src(IN5, 0, 256)], G0, 0, act=False, raw=True,
          kslice=(0, 256))
        g(n.c2f_p4.cv1, [_Src(IN4, 0, 128)], CB1, 0, kslice=(256, 384),
          addrep=_Src(G0, 0, 128))
        g(n.c2f_p4.m[0].cv1, [_Src(CB1, 64, 64)], T1, 0)
        g(n.c2f_p4.m[0].cv2, [_Src(T1, 0, 64)], CB1, 128)
        g(n.c2f_p4.cv2, [_Src(CB1, 0, 192)], CB3, 64)  # == P4, in cat3's slot
        # p3 branch, same split: the up2(p4) half goes through G1 at 40x40.
        g(n.c2f_p3.cv1, [_Src(CB3, 64, 128)], G1, 0, act=False, raw=True,
          kslice=(0, 128))
        g(n.c2f_p3.cv1, [_Src(IN3, 0, 64)], CB2, 0, kslice=(128, 192),
          addrep=_Src(G1, 0, 64))
        g(n.c2f_p3.m[0].cv1, [_Src(CB2, 32, 32)], T2, 0)
        g(n.c2f_p3.m[0].cv2, [_Src(T2, 0, 32)], CB2, 64)
        g(n.c2f_p3.cv2, [_Src(CB2, 0, 96)], P3, 0)
        # n4 branch
        g(n.down_p3, [_Src(P3, 0, 64)], CB3, 0, stride=2)
        g(n.c2f_n4.cv1, [_Src(CB3, 0, 192)], CB4, 0)
        g(n.c2f_n4.m[0].cv1, [_Src(CB4, 64, 64)], T3, 0)
        g(n.c2f_n4.m[0].cv2, [_Src(T3, 0, 64)], CB4, 128)
        g(n.c2f_n4.cv2, [_Src(CB4, 0, 192)], N4, 0)
        # n5 branch
        g(n.down_n4.cv1, [_Src(N4, 0, 128)], D1, 0)
        dw(n.down_n4.cv2, _Src(D1, 0, 128), D2, 0, stride=2, act=False)
        g(n.c2fcib_n5.cv1, [_Src(D2, 0, 128), _Src(IN5, 0, 256)], CB6, 0)
        cib = n.c2fcib_n5.m[0].cv1
        dw(cib[0], _Src(CB6, 128, 128), TC1, 0)
        g(cib[1], [_Src(TC1, 0, 128)], TC2, 0)
        dw(cib[2], _Src(TC2, 0, 256), TC3, 0, rep=True)
        g(cib[3], [_Src(TC3, 0, 256)], TC4, 0)
        dw(cib[4], _Src(TC4, 0, 128), CB6, 256, res=_Src(CB6, 128, 128))
        g(n.c2fcib_n5.cv2, [_Src(CB6, 0, 384)], N5, 0)

    # -- plan builders -----------------------------------------------------
    def _gemm(self, mod, srcs, out, ocoff, stride=1, act=True, res=None,
              kslice=None, raw=False, addrep=None):
        w = _w_gemm(mod)
        if kslice is not None:
            w = w[:, kslice[0]:kslice[1]].contiguous()
        if raw:
            one = torch.ones(w.shape[0], device=w.device, dtype=torch.float32)
            sc, sh = one, torch.zeros_like(one)
        else:
            sc, sh = _affine(mod)
        self.keep += [w, sc, sh]
        cout = w.shape[0]
        r = mod.conv.weight.shape[2]
        if r not in (1, 3) or mod.conv.groups != 1:
            raise _Unsupported("conv shape")
        if (r == 1 and len(srcs) > 2) or (r > 1 and len(srcs) != 1):
            raise _Unsupported("source count")
        s0 = srcs[0]
        s1 = srcs[1] if len(srcs) > 1 else s0
        ktot = sum(s.C for s in srcs)
        if w.shape[1] != ktot * r * r:
            raise _Unsupported("weight/source channel mismatch")
        if addrep is not None and addrep.C != cout:
            raise _Unsupported("addrep channel mismatch")
        gc = _gemm_cfg(cout, ktot, r, stride, out.PQ, self.B)
        bm, bn, bk, nw, ns, rowt = gc[:6]
        unroll = int(gc[6]) if len(gc) > 6 else 1
        if stride == 2:
            rowt = True
        # A cached config comes from a sweep on one shape family; shrink rather
        # than reject if it does not divide the shape in front of us.
        bm = min(bm, cout)
        while bm > 16 and cout % bm:
            bm //= 2
        cmin = min(s.C for s in srcs)
        while bk > 16 and (bk > cmin or cmin % bk):
            bk //= 2
        if cout % bm or cmin % bk:
            raise _Unsupported("channel count not tileable")
        nch = (out.W + bn - 1) // bn
        nblk = nch * out.H if rowt else (out.PQ + bn - 1) // bn
        self.ops.append(_Op(
            "gemm", out=out.t, w=w, sc=sc, sh=sh, x0=s0.buf.t, x1=s1.buf.t,
            res=(res.buf.t if res is not None else out.t),
            ar=(addrep.buf.t if addrep is not None else out.t),
            grid=(cout // bm, self.B * nblk),
            kw=dict(
                COUT=cout, PQ=out.PQ, WD=out.W, OIMG=out.img, OCOFF=ocoff,
                C0=s0.C, PQ0=s0.buf.PQ, IMG0=s0.buf.img, COFF0=s0.coff,
                C1=(s1.C if len(srcs) > 1 else 0), PQ1=s1.buf.PQ,
                IMG1=s1.buf.img, COFF1=s1.coff,
                RIMG=(res.buf.img if res else 0), RPQ=(res.buf.PQ if res else 0),
                RCOFF=(res.coff if res else 0),
                ADDREP=addrep is not None,
                ARIMG=(addrep.buf.img if addrep else 0),
                ARPQ=(addrep.buf.PQ if addrep else 0),
                ARW=(addrep.buf.W if addrep else 0),
                ARCOFF=(addrep.coff if addrep else 0),
                R=r, STRIDE=stride, PAD=(r // 2), HIN=s0.buf.H, WIN=s0.buf.W,
                NSRC=len(srcs), ACT=act, HAS_RES=res is not None,
                ROWT=rowt, NCH=nch, NBLK=nblk,
                EXACT=((out.W % bn == 0) if rowt else (out.PQ % bn == 0)),
                UNROLL=(unroll or 1),
                BM=bm, BN=bn, BK=bk, num_warps=nw, num_stages=ns,
            )))

    def _dw(self, mod, src, out, ocoff, stride=1, act=True, res=None, rep=False):
        if rep:
            w, sc, sh = _w_repvggdw(mod)
            k = 7
        else:
            w, sc, sh = _w_dw(mod)
            k = mod.conv.weight.shape[2]
            if mod.conv.groups != mod.conv.weight.shape[0]:
                raise _Unsupported("not depthwise")
        self.keep += [w, sc, sh]
        c = w.shape[0]
        bc, bn, nw, ns, rowt = _dw_cfg(c, k, stride, out.PQ, self.B)
        if stride == 2:
            rowt = True
        bc = min(bc, c)
        while bc > 1 and c % bc:
            bc //= 2
        nch = (out.W + bn - 1) // bn
        nblk = nch * out.H if rowt else (out.PQ + bn - 1) // bn
        self.ops.append(_Op(
            "dw", out=out.t, w=w, sc=sc, sh=sh, x0=src.buf.t,
            res=(res.buf.t if res is not None else out.t),
            grid=((c + bc - 1) // bc * nblk, self.B),
            kw=dict(
                PQ=out.PQ, WD=out.W, OIMG=out.img, OCOFF=ocoff, C=c,
                PQ0=src.buf.PQ, IMG0=src.buf.img, COFF0=src.coff,
                RIMG=(res.buf.img if res else 0), RPQ=(res.buf.PQ if res else 0),
                RCOFF=(res.coff if res else 0),
                K=k, STRIDE=stride, PAD=(k // 2), HIN=src.buf.H, WIN=src.buf.W,
                ACT=act, HAS_RES=res is not None,
                ROWT=rowt, NCH=nch, NBLK=nblk,
                EXACT=((out.W % bn == 0) if rowt else (out.PQ % bn == 0)),
                BC=bc, BN=bn, num_warps=nw, num_stages=ns,
            )))

    # -- execution ---------------------------------------------------------
    def run(self):
        for op in self.ops:
            op.launch()


def _timed(fn, warmup: int = 4, iters: int = 15) -> float:
    """Median latency of ``fn`` under the bench's own timing regime: an L2 flush
    before every timed iteration, CUDA-event brackets, median of the samples."""
    import statistics

    dev = torch.cuda.current_device()
    try:
        l2 = int(torch.cuda.get_device_properties(dev).L2_cache_size)
    except Exception:
        l2 = 50 << 20
    flush = torch.empty(2 * l2, dtype=torch.int8, device=f"cuda:{dev}")
    try:
        with torch.no_grad():
            for _ in range(warmup):
                flush.zero_()
                fn()
            torch.cuda.synchronize()
            st = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
            en = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
            for i in range(iters):
                flush.zero_()
                st[i].record()
                fn()
                en[i].record()
            torch.cuda.synchronize()
        return statistics.median(a.elapsed_time(b) for a, b in zip(st, en))
    finally:
        del flush


class YOLOv10Neck(nn.Module):
    def __init__(self):
        super().__init__()
        self._upsample = Interpolate()
        self.cat1 = YOLOConcat(1)
        self.c2f_p4 = YOLOC2f(384, 128, n=1, shortcut=False)
        self.cat2 = YOLOConcat(1)
        self.c2f_p3 = YOLOC2f(192, 64, n=1, shortcut=False)
        self.down_p3 = YOLOConv(64, 64, 3, 2)
        self.cat3 = YOLOConcat(1)
        self.c2f_n4 = YOLOC2f(192, 128, n=1, shortcut=False)
        self.down_n4 = YOLOSCDown(128, 128, 3, 2)
        self.cat4 = YOLOConcat(1)
        self.c2fcib_n5 = YOLOC2fCIB(384, 256, n=1, shortcut=True, lk=True)
        self._plans: dict | None = {}

    # -- the fused path ----------------------------------------------------
    def _key(self, feats):
        p3, p4, p5 = (feats["p3_backbone"], feats["p4_backbone"],
                      feats["p5_backbone"])
        if not _HAVE_TRITON or self._plans is None:
            return None
        for t in (p3, p4, p5):
            if (not t.is_cuda) or t.dtype != torch.float16 or not t.is_contiguous():
                return None
        B = p3.shape[0]
        if (tuple(p3.shape) != (B, 64, 80, 80)
                or tuple(p4.shape) != (B, 128, 40, 40)
                or tuple(p5.shape) != (B, 256, 20, 20)):
            return None
        return (B, p3.device.index)

    def _entry(self, key, feats):
        entry = self._plans.get(key)
        if entry is not None:
            return entry
        plan = _Plan(self, key[0], feats["p3_backbone"].device)
        srcs = [feats["p3_backbone"], feats["p4_backbone"], feats["p5_backbone"]]
        torch._foreach_copy_(plan.inputs, srcs)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                plan.run()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            plan.run()
        entry = (plan, graph)

        def fused():
            torch._foreach_copy_(plan.inputs, srcs)
            graph.replay()
            return plan.outputs

        if _timed(fused) > 1.02 * _timed(lambda: self._eager(feats)):
            raise _Unsupported("eager composition is faster on this machine")
        self._plans[key] = entry
        return entry

    def forward(self, feats: dict[str, torch.Tensor]):
        key = self._key(feats)
        if key is not None:
            try:
                plan, graph = self._entry(key, feats)
            except Exception:
                self._plans = None
            else:
                torch._foreach_copy_(
                    plan.inputs, [feats["p3_backbone"], feats["p4_backbone"],
                                  feats["p5_backbone"]])
                graph.replay()
                return plan.outputs
        return self._eager(feats)

    # -- reference composition --------------------------------------------
    def _eager(self, feats: dict[str, torch.Tensor]):
        p3_backbone = feats["p3_backbone"]
        p4_backbone = feats["p4_backbone"]
        p5_backbone = feats["p5_backbone"]

        x = self._upsample(p5_backbone, scale_factor=2.0, mode="nearest")
        x = self.cat1([x, p4_backbone])
        p4 = self.c2f_p4(x)

        x = self._upsample(p4, scale_factor=2.0, mode="nearest")
        x = self.cat2([x, p3_backbone])
        p3 = self.c2f_p3(x)

        x = self.down_p3(p3)
        x = self.cat3([x, p4])
        n4 = self.c2f_n4(x)

        x = self.down_n4(n4)
        x = self.cat4([x, p5_backbone])
        n5 = self.c2fcib_n5(x)
        return [p3, n4, n5]
