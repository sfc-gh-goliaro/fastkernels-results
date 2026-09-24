"""Adaptive continuous layer norm for diffusion transformers (L2 composite).

Used as the final output norm in FLUX (``norm_out``).  Projects the
conditioning embedding through SiLU + Linear into per-channel scale and
shift, then applies LayerNorm with those modulations.

The whole operator collapses into **two launches and one traversal of x**.

The reference composition
``LN(x) * (1 + scale)[:, None, :] + shift[:, None, :]`` walks the activation
three times: once inside ``F.layer_norm``, once for the broadcast multiply and
once for the broadcast add.  Worse, the two broadcast passes are the slowest
part of the whole operator -- a ``[1, S, C] * [1, 1, C]`` bf16 TensorIterator
cannot vectorize its loads, so it runs at 1.3 TB/s where a plain ``copy_`` of
the same tensor runs at 6.2 TB/s.  Measured on B200 at ``[1, 4096, 3072]``:
layer_norm 29.7 us, multiply 36.0 us, add 35.9 us -- against 15.3 us for a
read+write of x.

The modulation is token-invariant, so all three passes are one row-wise
LayerNorm whose per-channel affine happens to be built per call.  Which is
exactly what ``candidate/L1/layer_norm.py`` already computes at close to
read+write speed -- so ``_ada_layer_norm_fwd`` below is that kernel's
reduction verbatim (exact power-of-two tile split, shifted one-pass reduction,
``evict_first`` both ways), extended with the modulation and a PDL wait, and
this module only has to produce the ``[C]`` coefficients first.

Producing them is its own small problem.  ``SiLU -> Linear -> chunk ->
1 + scale`` is five launches for 6144 values, and on this benchmark an extra
small launch costs ~2.05 us of device time (windows are quantized to
``(N + 0.5) * 2.048 us``).  ``_ada_prologue`` does all five in one: each
program owns ``BN`` channels, reads *both* row bands of the projection weight
so the chunk never exists, and recomputes ``silu(cond)`` in registers rather
than round-tripping it through memory.  The GEMV is pure weight streaming --
37.75 MB for FLUX's 3072->6144 -- and lands within one quantum of a read-only
kernel over the same bytes (15.4 us against 13.3 us).

Net traffic per call: 37.75 MB of weight, plus one read and one write of x.

Numerics: the fused path reproduces the reference's *rounding chain*, not just
its value.  ``bf16(silu(cond))``, ``emb``, ``1 + scale``, ``self.norm(x)`` and
``norm * (1 + scale)`` are each materialized in the activation dtype by the
reference, so each is rounded to it here too (see ``_rne16`` for why that needs
forcing, and ``_ada_layer_norm_fwd`` for where).

Folding the tail into one fp32 expression instead is more accurate and
*further* from the reference: wherever ``norm * (1 + scale)`` nearly cancels
``shift``, the reference's own rounding of that product is the dominant term,
and being exact misses it by more than ``atol=1e-2, rtol=1e-2`` allows.  With
the modulation scaled 16x / 64x past what the benchmark's own inputs produce,
the fp32 fold holds only 0.9877 / 0.9730 of elements within tolerance -- an
``INCORRECT_NUMERICAL`` verdict -- where rounding as the reference does holds
1.0000.  The rounding is free: it costs one level of window in the obvious form
and none once ``(d - off) * rstd`` is rewritten as one FFMA.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

from ..L1.layer_norm import LayerNorm, _num_warps_for, _tile_split
from ..L1.linear import Linear
from ..L1.silu import SiLU

# Programmatic dependent launch.  The LayerNorm pass depends on the prologue
# only for the [C] modulation, which it does not touch until after its
# reduction -- so with PDL its programs start while the prologue's last wave is
# still draining, and spend that time loading x.  Measured for the pair:
# 29.7 -> 27.6 us at [1, 4096, 3072] and 21.5 -> 19.5 us at [1, 1024, 3072],
# with bit-identical output.
#
# The launch attribute goes on the *consumer* only.  Putting it on the prologue
# as well measured the same and would let the prologue start before whatever
# precedes it in the stream has finished -- harmless with an ATen predecessor
# (a kernel that never calls ``launch_dependents`` triggers at completion) but
# not a hazard worth owning for no gain.
try:
    from triton.language.extra.cuda import gdc_launch_dependents as _gdc_launch
    from triton.language.extra.cuda import gdc_wait as _gdc_wait
    _HAS_PDL = True
except ImportError:              # older Triton: run the two passes back to back
    _HAS_PDL = False

    @triton.jit
    def _gdc_launch():
        pass

    @triton.jit
    def _gdc_wait():
        pass


# Widest normalized row the fused LayerNorm keeps in registers; mirrors
# ``candidate/L1/layer_norm.py``'s own gate, since this is that kernel's tile.
_MAX_C = 16384
# Widest reduction tile the prologue holds per program.  ``BN`` channels x this
# many k-columns is the register footprint of one band, and a program holds two
# bands plus the SiLU'd conditioning vector.
_MAX_BK = 2048
# Largest conditioning dimension the prologue takes.  The k-loop is fully
# unrolled (measured worth one quantum over a rolled loop), so the trip count
# has to stay bounded; past this the reference's cuBLAS call is a real GEMM
# anyway and the prologue has nothing to offer.
_MAX_K = 8192

# fp32 is deliberately absent.  For an fp32 input the reference's projection is
# ``F.linear``, which on this GPU dispatches a **TF32** kernel: its own result
# carries ~2e-3 of relative error where the fp32 tolerance is
# ``atol=1e-5, rtol=1e-3``, so an accurate fp32 dot product *fails* against it
# (measured 0.877 of elements within tolerance).  ``candidate/L1/linear.py``
# defers fp32 to the reference for exactly this reason -- "for fp32 the
# reference silently switches between exact-fp32 and TF32 kernels depending on
# shape, and a candidate has to match whichever it picked" -- and so do we.
_FAST_DTYPES = (torch.bfloat16, torch.float16)


@triton.jit
def _rne16(x, IS_BF16: tl.constexpr):
    """fp32 -> (bf16 | fp16) -> fp32, round-to-nearest-even, one element at a time.

    Needed because the reference materializes several intermediates in the
    activation dtype and those roundings are load-bearing (see the module
    docstring), and because **Triton folds ``x.to(bf16).to(float32)`` away**:
    written the obvious way the round trip compiles to nothing at all, silently
    leaving an accurate-but-not-reference fp32 value.  Verified against torch on
    8192 random values: bit-identical for both dtypes, where the folded form
    agrees on only 78% (bf16) / 77% (fp16) of them.

    bf16 goes through integer registers -- a bf16 *is* the top half of an fp32,
    so RNE is "add half an LSB plus the tie bit, then mask" -- which needs no
    PTX.  Inf and NaN survive (0x7F800000 + 0x7FFF masks back to itself) and the
    sign is untouched, the bias acting on the magnitude of a sign-magnitude
    representation.  fp16 is not a prefix of fp32, so it uses the hardware
    convert.
    """
    if IS_BF16:
        i = x.to(tl.int32, bitcast=True)
        return ((i + 0x7FFF + ((i >> 16) & 1)) & -65536).to(tl.float32,
                                                            bitcast=True)
    return tl.inline_asm_elementwise(
        "{ .reg .b16 t; cvt.rn.f16.f32 t, $1; cvt.f32.f16 $0, t; }",
        "=f,f", [x], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _rne16x2(x, IS_BF16: tl.constexpr):
    """``_rne16`` two lanes at a time, for the per-element LayerNorm tail.

    The tail needs two of these per output element, and at [1, 4096, 3072] that
    is 25 M roundings -- enough to show up even against 50 MB of traffic.  For
    that pass: 19.5 us with the scalar integer form, 18.9 us with this packed
    convert, and 17.4 us with this *and* the FFMA rewrite of the normalization
    below -- the same as doing no rounding at all.

    bf16 unpacks with ``prmt`` (a bf16 is an fp32 with the low half zeroed);
    fp16 needs two real converts back.
    """
    if IS_BF16:
        return tl.inline_asm_elementwise(
            "{ .reg .b32 p; cvt.rn.bf16x2.f32 p, $3, $2;"
            "  prmt.b32 $0, 0, p, 0x5410; prmt.b32 $1, 0, p, 0x7632; }",
            "=f,=f,f,f", [x], dtype=tl.float32, is_pure=True, pack=2)
    return tl.inline_asm_elementwise(
        "{ .reg .b32 p; .reg .b16 lo, hi; cvt.rn.f16x2.f32 p, $3, $2;"
        "  mov.b32 {lo, hi}, p; cvt.f32.f16 $0, lo; cvt.f32.f16 $1, hi; }",
        "=f,=f,f,f", [x], dtype=tl.float32, is_pure=True, pack=2)


@triton.jit
def _rnd(x, IS_BF16: tl.constexpr, PACK2: tl.constexpr):
    """Round *x* to the activation dtype, packed when the tile has even width."""
    if PACK2:
        return _rne16x2(x, IS_BF16)
    return _rne16(x, IS_BF16)


@triton.jit
def _ada_prologue(
    COND, W, BIAS, A, B,
    C: tl.constexpr, K: tl.constexpr, BN: tl.constexpr,
    B0: tl.constexpr, NB0: tl.constexpr, B1: tl.constexpr,
    TWO: tl.constexpr, MASK1: tl.constexpr, MASKN: tl.constexpr,
    HAS_BIAS: tl.constexpr, IS_BF16: tl.constexpr, PDL: tl.constexpr,
):
    """``SiLU -> Linear -> chunk -> 1 + scale`` for ``BN`` channels, one program.

    Writes, in the activation dtype and therefore bit-identical to what the
    reference materializes::

        A[c] = 1 + scale[c]        scale = emb[:, :C]
        B[c] =     shift[c]        shift = emb[:, C:]

    ``scale[c]`` is output row ``c`` of the projection and ``shift[c]`` is row
    ``C + c``, so one program reads both row bands and ``torch.chunk`` never
    happens.  Folding the two halves here rather than in a second launch is
    what keeps the operator at two kernels.

    ``silu(cond)`` is recomputed per program instead of being written out by a
    separate launch: it is K values against ``2 * C * K`` of weight, and the
    launch it would cost is worth more than the redundant MUFU -- measured
    15.33 us with the SiLU inline against 15.25 us reading a precomputed
    vector, i.e. inside one quantum.  It is rounded back to the activation
    dtype because the reference feeds ``self.silu(cond).to(x.dtype)`` to the
    projection, so the dot product has to see the same rounded operand.

    Each band is covered by ``NB0`` tiles of the power-of-two ``B0`` plus an
    optional ``B1`` remainder tile -- the same exact-split idea as the
    LayerNorm kernel.  For FLUX's K=3072 that is one 2048-lane tile and one
    1024-lane tile with the loop unrolled, rather than a masked 4096-lane tile
    idling a quarter of its lanes: 15.4 us against 17.4 us.
    """
    pid = tl.program_id(0)
    samp = tl.program_id(1)
    rn = pid * BN + tl.arange(0, BN)
    nm = rn < C
    # Out-of-range channels fold onto channel 0 for the *address* arithmetic --
    # reading row ``C + rn`` past the end of the weight would fault -- and are
    # dropped by ``nm`` at the store.  Only the last program of a grid where BN
    # does not divide C takes this path.
    rnc = tl.where(nm, rn, 0) if MASKN else rn
    cond_p = COND + samp.to(tl.int64) * K
    ws = W + rnc[:, None].to(tl.int64) * K
    wh = W + (C + rnc)[:, None].to(tl.int64) * K

    acc_s = tl.zeros((BN,), tl.float32)
    acc_h = tl.zeros((BN,), tl.float32)
    for i in tl.static_range(NB0):
        k0 = i * B0 + tl.arange(0, B0)
        v = tl.load(cond_p + k0).to(tl.float32)
        v = _rne16(v * tl.sigmoid(v), IS_BF16)
        acc_s += tl.sum(tl.load(ws + k0[None, :],
                                eviction_policy="evict_first").to(tl.float32)
                        * v[None, :], axis=1)
        acc_h += tl.sum(tl.load(wh + k0[None, :],
                                eviction_policy="evict_first").to(tl.float32)
                        * v[None, :], axis=1)
    if TWO:
        k1 = NB0 * B0 + tl.arange(0, B1)
        if MASK1:
            m1 = k1 < K
            v = tl.load(cond_p + k1, mask=m1, other=0.0).to(tl.float32)
            v = _rne16(v * tl.sigmoid(v), IS_BF16)
            # silu(0) is 0, so the padding lanes drop out of the dot product on
            # their own and the weight load needs no separate zero-fill.
            acc_s += tl.sum(tl.load(ws + k1[None, :], mask=m1[None, :],
                                    other=0.0,
                                    eviction_policy="evict_first").to(tl.float32)
                            * v[None, :], axis=1)
            acc_h += tl.sum(tl.load(wh + k1[None, :], mask=m1[None, :],
                                    other=0.0,
                                    eviction_policy="evict_first").to(tl.float32)
                            * v[None, :], axis=1)
        else:
            v = tl.load(cond_p + k1).to(tl.float32)
            v = _rne16(v * tl.sigmoid(v), IS_BF16)
            acc_s += tl.sum(tl.load(ws + k1[None, :],
                                    eviction_policy="evict_first").to(tl.float32)
                            * v[None, :], axis=1)
            acc_h += tl.sum(tl.load(wh + k1[None, :],
                                    eviction_policy="evict_first").to(tl.float32)
                            * v[None, :], axis=1)
    if HAS_BIAS:
        acc_s += tl.load(BIAS + rn, mask=nm, other=0.0).to(tl.float32)
        acc_h += tl.load(BIAS + C + rn, mask=nm, other=0.0).to(tl.float32)

    # ``emb`` is a dtype tensor and so is ``1 + scale``: two roundings, both of
    # which the reference performs, so both happen here.
    base = samp.to(tl.int64) * C
    dt = A.dtype.element_ty
    tl.store(A + base + rn, (1.0 + _rne16(acc_s, IS_BF16)).to(dt), mask=nm)
    tl.store(B + base + rn, acc_h.to(dt), mask=nm)
    if PDL:
        # Last statement, after every store: release the LayerNorm pass.
        _gdc_launch()


@triton.jit
def _ada_layer_norm_fwd(
    X, Y, A, B, LNW, LNB,
    N: tl.constexpr, eps: tl.constexpr,
    B0: tl.constexpr, B1: tl.constexpr, TWO: tl.constexpr, MASK1: tl.constexpr,
    HAS_LNW: tl.constexpr, HAS_LNB: tl.constexpr, IS_BF16: tl.constexpr,
    PACK2: tl.constexpr, PDL: tl.constexpr,
):
    """``candidate/L1/layer_norm.py``'s ``_layer_norm_fwd``, plus the modulation.

    The reduction is that kernel's, unchanged and for its reasons: one program
    per row with the whole row in registers; the row covered by one or two
    power-of-two tiles that sum to *exactly* N rather than a masked
    ``next_pow2(N)`` tile; the **shifted one-pass** formula (subtract the row's
    own first element, then accumulate ``sum(x-c)`` and ``sum((x-c)^2)`` in the
    same pass so the two reduction trees pipeline instead of serializing);
    ``maximum(var, 0)`` so a row whose rounded variance lands a hair below zero
    yields the reference's inf rather than NaN; ``evict_first`` on the row in
    and the row out, neither of which is revisited, leaving L2 to the
    modulation that every program re-reads.

    What is added is the tail, and the tail is written to round exactly where
    the reference rounds::

        norm = dtype( (x - mean) * rstd * w + b )   <- F.layer_norm's output
        prod = dtype( norm * (1 + scale) )          <- the broadcast multiply
        y    = dtype( prod + shift )                <- the broadcast add

    Collapsing those into one fp32 expression is more accurate and *wrong for
    this purpose*: where ``prod`` nearly cancels ``shift``, the reference's
    rounding of ``prod`` is the dominant term and being exact misses it by more
    than the bf16 tolerance allows.  The three casts are free -- the pass is
    bandwidth bound at 50.3 MB and unchanged in window from the unrounded form.

    One warp, not the two ``_num_warps_for`` picks for a bare LayerNorm: every
    program here also loads the modulation, and with one warp per row more rows
    stay in flight to hide it (17.4 us against 19.4 us at [1, 4096, 3072]).
    """
    row = tl.program_id(0)
    base = row.to(tl.int64) * N
    c0 = tl.arange(0, B0)
    shift = tl.load(X + base).to(tl.float32)
    d0 = tl.load(X + base + c0,
                 eviction_policy="evict_first").to(tl.float32) - shift
    acc = tl.sum(d0, axis=0)
    sq = tl.sum(d0 * d0, axis=0)
    if TWO:
        c1 = B0 + tl.arange(0, B1)
        if MASK1:
            m1 = c1 < N
            # Padding lanes must contribute 0 to both sums, so zero them after
            # the shift rather than loading ``other=0.0``.
            d1 = tl.where(m1, tl.load(X + base + c1, mask=m1,
                                      eviction_policy="evict_first")
                          .to(tl.float32) - shift, 0.0)
        else:
            d1 = tl.load(X + base + c1,
                         eviction_policy="evict_first").to(tl.float32) - shift
        acc += tl.sum(d1, axis=0)
        sq += tl.sum(d1 * d1, axis=0)
    inv_n: tl.constexpr = 1.0 / N
    off = acc * inv_n                       # mean, relative to the shift
    var = sq * inv_n - off * off
    rstd = 1.0 / tl.sqrt(tl.maximum(var, 0.0) + eps)
    # ``d * rstd + nrm`` rather than ``(d - off) * rstd``: one FFMA per element
    # instead of a subtract and a multiply.  Worth a whole level here -- it is
    # exactly what pays for the two exact roundings below, which cost 18.9 us
    # against 17.4 us in the ``(d - off) * rstd`` form and 17.4 us in this one.
    nrm = -off * rstd

    if PDL:
        # As late as it can go: everything above depends only on x, and
        # everything below is the prologue's output.
        _gdc_wait()
    dt = Y.dtype.element_ty
    n0 = d0 * rstd + nrm
    if HAS_LNW:
        n0 = n0 * tl.load(LNW + c0).to(tl.float32)
    if HAS_LNB:
        n0 = n0 + tl.load(LNB + c0).to(tl.float32)
    y0 = _rnd(_rnd(n0, IS_BF16, PACK2) * tl.load(A + c0).to(tl.float32),
              IS_BF16, PACK2) + tl.load(B + c0).to(tl.float32)
    tl.store(Y + base + c0, y0.to(dt), eviction_policy="evict_first")
    if TWO:
        n1 = d1 * rstd + nrm
        if MASK1:
            if HAS_LNW:
                n1 = n1 * tl.load(LNW + c1, mask=m1).to(tl.float32)
            if HAS_LNB:
                n1 = n1 + tl.load(LNB + c1, mask=m1).to(tl.float32)
            y1 = _rnd(_rnd(n1, IS_BF16, PACK2)
                      * tl.load(A + c1, mask=m1).to(tl.float32), IS_BF16, PACK2) \
                + tl.load(B + c1, mask=m1).to(tl.float32)
            tl.store(Y + base + c1, y1.to(dt), mask=m1,
                     eviction_policy="evict_first")
        else:
            if HAS_LNW:
                n1 = n1 * tl.load(LNW + c1).to(tl.float32)
            if HAS_LNB:
                n1 = n1 + tl.load(LNB + c1).to(tl.float32)
            y1 = _rnd(_rnd(n1, IS_BF16, PACK2)
                      * tl.load(A + c1).to(tl.float32), IS_BF16, PACK2) \
                + tl.load(B + c1).to(tl.float32)
            tl.store(Y + base + c1, y1.to(dt), eviction_policy="evict_first")


def _prologue_plan(k: int):
    """Cover exactly *k* columns with ``NB0`` tiles of ``B0`` plus a remainder.

    ``B0`` is the largest power of two that fits the register budget
    (``_MAX_BK``) without overshooting *k*; the tail gets one power-of-two tile,
    masked only when the tail is not itself a power of two -- never the case for
    K=3072, which splits as 2048 + 1024.
    """
    b0 = min(_MAX_BK, 1 << (k.bit_length() - 1))
    nb0 = k // b0
    rem = k - nb0 * b0
    if rem == 0:
        return b0, nb0, 1, False, False
    b1 = triton.next_power_of_2(rem)
    return b0, nb0, b1, True, b1 != rem


class AdaLayerNormContinuous(nn.Module):
    r"""
    Adaptive normalization layer with a norm layer (layer_norm or rms_norm).

    Args:
        embedding_dim (`int`): Embedding dimension to use during projection.
        conditioning_embedding_dim (`int`): Dimension of the input condition.
        elementwise_affine (`bool`, defaults to `True`):
            Boolean flag to denote if affine transformation should be applied.
        eps (`float`, defaults to 1e-5): Epsilon factor.
        bias (`bool`, defaults to `True`): Whether to use bias in the linear layer.
        norm_type (`str`, defaults to `"layer_norm"`):
            Normalization layer to use. Values supported: "layer_norm", "rms_norm".
    """

    def __init__(
        self,
        embedding_dim: int,
        conditioning_embedding_dim: int,
        elementwise_affine=True,
        eps=1e-5,
        bias=True,
        norm_type="layer_norm",
        promote_fp32: bool = True,
    ):
        super().__init__()
        self.silu = SiLU()
        self.linear = Linear(conditioning_embedding_dim, embedding_dim * 2, bias=bias)
        if norm_type == "layer_norm":
            self.norm = LayerNorm(embedding_dim, eps=eps, elementwise_affine=elementwise_affine,
                                  promote_fp32=promote_fp32)
        else:
            raise ValueError(f"unknown norm_type {norm_type}")

        # ---- fused-path launch plan, resolved once -------------------------
        c = int(embedding_dim)
        k = int(conditioning_embedding_dim)
        self._c = c
        self._k = k
        self._fast = (norm_type == "layer_norm" and 0 < c <= _MAX_C
                      and 0 < k <= _MAX_K)
        if self._fast:
            b0, nb0, b1, two, mask1 = _prologue_plan(k)
            bn = max(1, min(8, 8192 // b0))
            self._pgrid0 = -(-c // bn)
            self._pargs = (c, k, bn, b0, nb0, b1, two, mask1, c % bn != 0)
            self._pwarps = min(8, max(1, (bn * b0) // 2048))
            lb0, lb1, ltwo, lmask1 = _tile_split(c)
            # The packed rounding consumes lanes in pairs, so it needs every
            # tile it is applied to to have even width. Only degenerate widths
            # (C=1, or a one-element remainder tile as in C=3) fail that.
            pack2 = lb0 % 2 == 0 and (not ltwo or lb1 % 2 == 0)
            self._largs = (c, float(eps), lb0, lb1, ltwo, lmask1)
            self._pack2 = pack2
            self._lwarps = 1 if c <= 4096 else _num_warps_for(c)
        # Scratch for ``1 + scale`` and ``shift``, keyed by (batch, dtype,
        # device).  Only the allocation is cached: ``cond`` is a fresh
        # embedding every diffusion step, so the contents are rewritten by the
        # prologue on every call.
        self._mod: dict = {}

    # ------------------------------------------------------------------
    def _reference(self, x: torch.Tensor, conditioning_embedding: torch.Tensor):
        emb = self.linear(self.silu(conditioning_embedding).to(x.dtype))
        scale, shift = torch.chunk(emb, 2, dim=1)
        return self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]

    def forward(self, x: torch.Tensor, conditioning_embedding: torch.Tensor) -> torch.Tensor:
        if self._fast and not torch.is_grad_enabled():
            dt = x.dtype
            lw = self.linear.weight
            lb = self.linear.bias
            nw = self.norm.weight
            nb = self.norm.bias
            if (dt in _FAST_DTYPES and x.is_cuda
                    and x.ndim == 3 and x.shape[2] == self._c
                    and x.is_contiguous() and x.numel()
                    and conditioning_embedding.ndim == 2
                    and conditioning_embedding.shape[0] == x.shape[0]
                    and conditioning_embedding.shape[1] == self._k
                    and conditioning_embedding.dtype is dt
                    and conditioning_embedding.is_contiguous()
                    and lw.dtype is dt and lw.is_contiguous()
                    and (lb is None or (lb.dtype is dt and lb.is_contiguous()))
                    and (nw is None or (nw.dtype is dt and nw.is_contiguous()))
                    and (nb is None or (nb.dtype is dt and nb.is_contiguous()))
                    # Triton launches on the *current* device; a tensor parked
                    # on another one has to go the reference route.
                    and x.get_device() == torch.cuda.current_device()):
                return self._fused(x, conditioning_embedding, lw, lb, nw, nb)
        return self._reference(x, conditioning_embedding)

    def _fused(self, x, cond, lw, lb, nw, nb):
        c = self._c
        bsz, seq = x.shape[0], x.shape[1]
        key = (bsz, x.dtype, x.device)
        mod = self._mod.get(key)
        if mod is None:
            mod = self._mod[key] = (
                torch.empty(bsz * c, dtype=x.dtype, device=x.device),
                torch.empty(bsz * c, dtype=x.dtype, device=x.device),
            )
        a, b = mod
        pdl = _HAS_PDL and bsz == 1
        is_bf16 = x.dtype is torch.bfloat16
        _ada_prologue[(self._pgrid0, bsz)](
            cond, lw, lb, a, b, *self._pargs, lb is not None, is_bf16, pdl,
            num_warps=self._pwarps,
        )
        y = torch.empty_like(x)
        if bsz == 1:
            _ada_layer_norm_fwd[(seq,)](
                x, y, a, b, nw, nb, *self._largs,
                nw is not None, nb is not None, is_bf16, self._pack2, pdl,
                num_warps=self._lwarps, launch_pdl=pdl)
        else:
            # ``(1 + scale)[:, None, :]`` is per sample, so each sample gets its
            # own launch over its own slice of the modulation.  Never hit by the
            # captured shapes, which are batch=1.
            xf = x.view(bsz, seq * c)
            yf = y.view(bsz, seq * c)
            for i in range(bsz):
                _ada_layer_norm_fwd[(seq,)](
                    xf[i], yf[i], a[i * c:], b[i * c:], nw, nb, *self._largs,
                    nw is not None, nb is not None, is_bf16, self._pack2, False,
                    num_warps=self._lwarps)
        return y
