"""Oasis final DiT projection layer, fused for B200 / sm_100.

The baseline composes ten ATen kernels for what is, in work terms, nothing: the
whole operator moves ~6.2 MB and does 138 MFLOP, which is under a microsecond of
B200 bandwidth. Measured at T=6 those ten kernels occupy 38.8 us of device time
inside a 78.4 us scored window (``profile/p1-baseline-probe/``), so half the
window is host-side gap and the *number* of dispatches is the entire cost. The
window also barely moves with problem size, which is the signature of a
fixed-cost problem rather than a work problem.

So this module collapses the ten dispatches into one Python-level ``forward``
issuing two kernels:

* ``adaln_gain_shift`` -- SiLU, the ``K -> 2K`` projection, its bias, the
  ``chunk``, and the ``1 + scale`` -- writing ``shift`` and ``gain``; and
* ``ln_modulate_project`` -- LayerNorm, the modulation, and the ``K -> N``
  output projection with its bias.

Triton is the language because its launch overhead was measured equal to a
hand-written ``TORCH_LIBRARY`` operator's on this machine (5.18 vs 5.34 us for
one launch, 7.20 vs 7.18 us for two, ``profile/p1-baseline-probe/``), so its
convenience costs nothing here.

The input layout drives both mappings. ``x`` arrives channel-major -- at T=6,
``shape (1,6,9,16,1024)`` with ``stride (884736,147456,16,1,144)`` -- so the
hidden axis the LayerNorm reduces along is strided by 144 elements while the
*spatial* axes are the contiguous ones. A row-per-warp mapping along the hidden
axis would touch a separate 32-byte sector per element. Both kernels therefore
tile over spatial positions and loop over the hidden axis, and nothing here calls
``x.contiguous()`` -- that copy is one of the ten kernels being removed.

Numerics reproduce the reference's rounding boundaries rather than merely landing
inside its tolerance: ``shift``/``scale`` round to fp16 at the adaLN projection's
output, ``gain = fp16(1 + fp16(scale))``, the normalized value rounds to fp16
before the modulation, and the modulating multiply and add each round to fp16.
Both reductions and both matrix products accumulate in fp32, and LayerNorm uses
the biased variance with eps inside the root.

That variance is computed as the mean of squared deviations, in a pass of its own,
rather than as ``E[x^2] - E[x]^2``. The moment form is one sweep cheaper and is
accurate on the captured standard-normal inputs, but it cancels catastrophically
whenever a row's mean dominates its spread: on a row of 1023 fp16 ``16.0`` values
plus one ``16.015625`` both of its terms round to the same fp32 number, so the
variance comes out *exactly zero* against a true 2.38e-7 and ``rstd`` is 1000
instead of 899 -- putting 85% of the output past the comparison bound. Clamping at
zero stops that becoming a NaN and recovers none of the accuracy. Deviations from
the mean have no such failure mode, which is also why ATen reduces with Welford.

Anything the fused path is not written for -- another dtype, a layout whose index
mapping does not collapse, a width the kernels are not specialized for, a row
count past the padded tile, autograd, a build failure -- delegates to this
module's own submodules, which reproduce the baseline body.
"""

from __future__ import annotations

import sys

import torch
import torch.nn as nn

from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear
from ..L1.silu import SiLU

# ---------------------------------------------------------------------------
# Tile configuration, chosen on the end-to-end window rather than on either
# kernel in isolation (profile/p1-fused-probe/config_compare.py -- per-kernel
# figures there are noisy and do not add up to a total, because host enqueue and
# device execution overlap differently in each). The scored window quantises in
# ~2.03 us steps, so differences below that are the same reading.
#
# What the sweep found, over BLOCK_N in {16,32,64,128} x BLOCK_K in {32..256} x
# warps in {1,2,4,8} x stages in {2,3,4} for the adaLN kernel and the analogous
# grid for the projection: 61.4 us at the first guess (16/128 both, 4 warps),
# 47.0 us here. Everything at 47.0 ties; everything else is 49 us or worse.
#
# NCU (profile/p1-fused-kernels-v1/) explains why the tiles matter and the block
# count does not: both kernels run at ~6.2% warp occupancy and 0.03-0.09 waves
# per SM, with DRAM at 2.6% of peak and the tensor pipes at 2.4%. There is not
# enough work here to fill the machine by any mapping, so the only lever inside a
# kernel is instruction-level parallelism per warp -- which is what the wider
# BLOCK_K buys, and why BLOCK_N 64 and 128 measure identically despite 32 versus
# 16 blocks. The adaLN kernel needs stages=3 (61.4 us at stages=2): with
# BLOCK_K=256 its hidden loop runs four times, so there is something to pipeline.
#
# ``_ROW_TILE`` is ``tl.dot``'s minimum M. The adaLN kernel's row axis is the
# captured T (2..6), far below it, so the tile is padded and masked; the waste
# lands on a 25 MFLOP product whose real cost is streaming 4.19 MB of weight.
#
# ``_PROJ_BLOCK_M = 16`` is the narrowest legal ``tl.dot`` M and keeps every
# strided ``x`` access 32-byte aligned and fully used: lane ``m`` at step ``k``
# sits at byte ``(s0 + m + 144k) * 2`` and ``144 = 9 * 16``, so a 16-wide tile
# starting at a multiple of 16 never straddles a sector. NCU confirms it --
# 113,040 load sectors against a 112,900-sector minimum, so no amplification at
# all on a layout that would cost 16x under a row-per-warp mapping.
# ---------------------------------------------------------------------------
_ROW_TILE = 16
_ADALN_BLOCK_N = 64
_ADALN_BLOCK_K = 256
_ADALN_WARPS = 4
_ADALN_STAGES = 3
_PROJ_BLOCK_M = 16
_PROJ_BLOCK_K = 512
_PROJ_WARPS = 4
_PROJ_STAGES = 3

# The projection width the output kernel is specialized for: ``patch_size**2 *
# out_channels`` is 64 in the captured configuration, and 64 is also a legal
# ``tl.dot`` N. A module built with any other width delegates.
_PROJ_N = 64

# 128-bit accesses want 16-byte bases. Torch allocations are far more aligned
# than this and the harness's shifting pool steps by 256 bytes, so the captured
# operands always qualify; the check is here so a caller that hands us a
# narrowly-offset view delegates instead of running a kernel whose access width
# assumption no longer holds.
_ALIGN_BYTES = 16

# fp16 ``tl.dot`` with fp32 accumulate. Measured on sm_100.
_MIN_CAPABILITY = (8, 0)

# Triton keys its kernel cache on each stride being ``== 1`` or divisible by 16, and
# the projection kernel keeps those specializations because ``stride_s == 1`` is what
# makes the lane-axis load vectorise. So the admitted stride classes have to be a
# finite set that ``_warm`` can compile at import: every stride is either exactly 1
# or a multiple of ``_STRIDE_CLASS``. That covers the captured layout
# (147456, 1, 144), a contiguous ``x`` (K*S, K, 1) and the rows-1 collapse (1, 1, K),
# and rejects the rest -- a clause that exists purely because of the compiler, which
# is why it names the compiler.
_STRIDE_CLASS = 16

# The compiled launchers, or ``None``. ``_BUILD_STATE`` is "ready" once both are
# warmed, "failed" after a genuine compile error (latched: retrying per call would
# put the compiler in the timed window), or "pending" when CUDA was simply absent --
# which must *not* latch, or a candidate imported before its device exists would run
# the fallback for the whole bench.
#
# ``_WARMED_DEVICES`` records where the warm actually happened. Triton's JIT cache is
# per device, so a call on a device that was never warmed would compile inside
# ``forward``; the predicate rejects those instead.
_ADALN_KERNEL = None
_PROJ_KERNEL = None
_BUILD_STATE = "pending"
_WARMED_DEVICES: set[int] = set()


def _define_kernels():
    """Define both Triton kernels. Imported and JIT-decorated here rather than at
    module scope so a missing or broken Triton degrades to delegation instead of
    making the module unimportable."""
    import triton
    import triton.language as tl

    # ``do_not_specialize`` keeps Triton from keying its cache on ``rows`` and
    # ``spatial`` being ``== 1`` or divisible by 16. Those two are plain scalar
    # multipliers, and freeing them measured free: 25.57 us against 25.54 us fully
    # specialized on the captured layout, 19.47 against 19.42 contiguous
    # (profile/p1-fused-probe/specialization_ab.py). Freeing them removes ``rows``
    # 1/16 and any ``spatial`` as sources of extra binaries.
    #
    # The three strides and ``k_width`` stay specialized because there their
    # specialization is load-bearing, not cosmetic. ``stride_s`` is the lane axis:
    # its ``== 1`` is what lets Triton prove the innermost access is contiguous and
    # emit a vectorised load. Freeing everything cost the whole win -- 44.13 us,
    # i.e. below the baseline -- and freeing just ``stride_s`` still cost 31.71.
    # ``_warm`` therefore compiles the lattice of stride classes the predicate
    # admits, and the predicate rejects any stride outside it.
    @triton.jit(do_not_specialize=["rows"], do_not_specialize_on_alignment=["rows"])
    def adaln_gain_shift(
        c_ptr, weight_ptr, bias_ptr, out_ptr,
        rows, k_width,
        BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, ROW_TILE: tl.constexpr,
    ):
        """``SiLU(c) @ weight.T + bias``, split into ``shift`` and ``1 + scale``.

        One block owns ``BLOCK_N`` output columns for *all* rows, so the grid is
        over columns alone and the weight is streamed exactly once. Mapping rows
        onto a grid axis instead would re-read all 4.19 MB of it per row -- 25 MB
        at T=6, which is four times the whole operator's DRAM floor.

        ``out`` is ``[2, rows, k_width]``: plane 0 is ``shift``, plane 1 is
        ``gain``. Because ``k_width`` is a multiple of ``BLOCK_N``, a block falls
        entirely inside one half of the concatenated projection, so a single
        uniform branch on the block's first column replaces a per-element select.
        """
        col0 = tl.program_id(0) * BLOCK_N
        cols = col0 + tl.arange(0, BLOCK_N)
        row_ids = tl.arange(0, ROW_TILE)
        row_ok = row_ids < rows

        acc = tl.zeros((ROW_TILE, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, k_width, BLOCK_K):
            ks = k0 + tl.arange(0, BLOCK_K)
            cv = tl.load(
                c_ptr + row_ids[:, None] * k_width + ks[None, :],
                mask=row_ok[:, None], other=0.0,
            ).to(tl.float32)
            # ATen's SiLU is x / (1 + exp(-x)) evaluated in fp32 and rounded
            # once, which is what the scored reference runs.
            act = (cv / (1.0 + tl.exp(-cv))).to(tl.float16)
            # The weight tile is loaded k-contiguous (its natural layout) and
            # transposed in registers; loading it transposed would gather.
            wt = tl.load(weight_ptr + cols[:, None] * k_width + ks[None, :])
            acc = tl.dot(act, tl.trans(wt), acc)

        # The reference's projection emits fp16, so shift and scale round here.
        val = (acc + tl.load(bias_ptr + cols).to(tl.float32)[None, :]).to(tl.float16)
        if col0 < k_width:
            tl.store(
                out_ptr + row_ids[:, None] * k_width + cols[None, :],
                val, mask=row_ok[:, None],
            )
        else:
            gain = (1.0 + val.to(tl.float32)).to(tl.float16)
            tl.store(
                out_ptr + rows * k_width
                + row_ids[:, None] * k_width + (cols[None, :] - k_width),
                gain, mask=row_ok[:, None],
            )

    @triton.jit(do_not_specialize=["rows", "spatial"],
                do_not_specialize_on_alignment=["rows", "spatial"])
    def ln_modulate_project(
        x_ptr, mod_ptr, weight_ptr, bias_ptr, y_ptr,
        rows, spatial, k_width, eps,
        stride_r, stride_s, stride_k,
        BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, N: tl.constexpr,
    ):
        """LayerNorm over the hidden axis, the adaLN modulation, then ``K -> N``.

        ``x`` is addressed as ``[rows, spatial, k_width]`` through explicit
        strides, so the channel-major captured layout is read in place. Holding
        the row constant per block makes ``gain``/``shift`` plain vectors and
        keeps all integer arithmetic out of the inner loop.

        Three passes over ``x``: the mean, then the squared deviations from it,
        then the projection. Only the first read is cold; the rest are L2-resident
        (one frame is 288 KB against a >100 MB L2), and keeping a 1024-wide row in
        registers alongside the ``[BLOCK_M, N]`` accumulator would spill.

        The separate deviation pass is what makes the variance trustworthy. The
        one-pass moment form ``E[x^2] - E[x]^2`` cancels catastrophically when the
        row's mean dominates its spread: on a row of 1023 fp16 ``16.0`` values plus
        one ``16.015625``, both terms round to the same fp32 number and the
        variance comes out *exactly zero* against a true 2.38e-7, making ``rstd``
        1000 instead of 899 and putting 85% of the output outside the comparison
        bound. Clamping at zero prevents the NaN but recovers none of the lost
        variance. Deviations from the mean have no such failure mode, and ATen's
        Welford recurrence does not either -- which is the form being matched.
        """
        row = tl.program_id(1)
        s0 = tl.program_id(0) * BLOCK_M
        # Lane axis is the spatial one, which is the contiguous axis of x.
        x_base = x_ptr + row * stride_r + (s0 + tl.arange(0, BLOCK_M))[None, :] * stride_s

        total = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for k0 in range(0, k_width, BLOCK_K):
            ks = k0 + tl.arange(0, BLOCK_K)
            total += tl.sum(tl.load(x_base + ks[:, None] * stride_k).to(tl.float32), 0)
        mean = total / k_width

        deviation_sq = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for k0 in range(0, k_width, BLOCK_K):
            ks = k0 + tl.arange(0, BLOCK_K)
            dev = tl.load(x_base + ks[:, None] * stride_k).to(tl.float32) - mean[None, :]
            deviation_sq += tl.sum(dev * dev, 0)
        # Biased variance, eps inside the root -- ATen's form. The clamp costs
        # nothing and keeps a negative value unrepresentable even if a future
        # reduction order made one possible; a NaN here is a hard failure rather
        # than a rounding loss.
        var = tl.maximum(deviation_sq / k_width, 0.0)
        rstd = tl.rsqrt(var + eps)

        cols = tl.arange(0, N)
        shift_ptr = mod_ptr + row * k_width
        gain_ptr = mod_ptr + rows * k_width + row * k_width
        acc = tl.zeros((BLOCK_M, N), dtype=tl.float32)
        for k0 in range(0, k_width, BLOCK_K):
            ks = k0 + tl.arange(0, BLOCK_K)
            tile = tl.load(x_base + ks[:, None] * stride_k).to(tl.float32)
            # Each of these three roundings is one the reference performs: the
            # normalized value becomes an fp16 tensor, then the fp16 multiply by
            # the gain, then the fp16 add of the shift.
            nrm = ((tile - mean[None, :]) * rstd[None, :]).to(tl.float16)
            gain = tl.load(gain_ptr + ks).to(tl.float32)
            shift = tl.load(shift_ptr + ks).to(tl.float32)
            val = (nrm.to(tl.float32) * gain[:, None]).to(tl.float16)
            val = (val.to(tl.float32) + shift[:, None]).to(tl.float16)
            wt = tl.load(weight_ptr + cols[:, None] * k_width + ks[None, :])
            acc = tl.dot(tl.trans(val), tl.trans(wt), acc)

        out = (acc + tl.load(bias_ptr + cols).to(tl.float32)[None, :]).to(tl.float16)
        y_rows = (row * spatial + s0 + tl.arange(0, BLOCK_M))
        tl.store(y_ptr + y_rows[:, None] * N + cols[None, :], out)

    return adaln_gain_shift, ln_modulate_project


def _warm(adaln, proj, device) -> None:
    """Compile every specialization class the predicate admits, at import.

    ``rows`` and ``spatial`` are unspecialized, so they contribute no classes. What
    remains is the three strides, each of which the predicate constrains to be 1 or a
    multiple of ``_STRIDE_CLASS`` -- eight combinations, compiled here with
    representative values. The captured layout is ``(16, 1, 16)`` in those terms, a
    contiguous ``x`` is ``(16, 16, 1)``, and a rows-1 collapse is ``(1, 1, 16)``.

    Pointer arguments are specialized on 16-byte alignment too. Every operand the
    predicate admits is 16-byte aligned by its own clause, and these warm buffers are
    fresh allocations, so all of them share the aligned class.

    The captured strided layout is compiled last so that its binary is the most
    recently touched, and for every captured row count -- which costs nothing now that
    ``rows`` is unspecialized, and demonstrates it.
    """
    k_width, spatial = 1024, 144
    half = dict(device=device, dtype=torch.float16)
    weight = torch.zeros((_PROJ_N, k_width), **half)
    bias = torch.zeros((_PROJ_N,), **half)
    ada_weight = torch.zeros((2 * k_width, k_width), **half)
    ada_bias = torch.zeros((2 * k_width,), **half)

    adaln[(2 * k_width // _ADALN_BLOCK_N,)](
        torch.zeros((_ROW_TILE, k_width), **half), ada_weight, ada_bias,
        torch.empty((2, _ROW_TILE, k_width), **half), 2, k_width,
        BLOCK_N=_ADALN_BLOCK_N, BLOCK_K=_ADALN_BLOCK_K, ROW_TILE=_ROW_TILE,
        num_warps=_ADALN_WARPS, num_stages=_ADALN_STAGES,
    )

    def launch_proj(x, rows, spatial_extent, stride_r, stride_s, stride_k):
        proj[(spatial_extent // _PROJ_BLOCK_M, rows)](
            x, torch.empty((2, max(rows, 1), k_width), **half), weight, bias,
            torch.empty((rows, spatial_extent, _PROJ_N), **half),
            rows, spatial_extent, k_width, 1e-6, stride_r, stride_s, stride_k,
            BLOCK_M=_PROJ_BLOCK_M, BLOCK_K=_PROJ_BLOCK_K, N=_PROJ_N,
            num_warps=_PROJ_WARPS, num_stages=_PROJ_STAGES,
        )

    # One buffer large enough for any stride triple drawn from the lattice.
    lattice = (1, _STRIDE_CLASS)
    scratch = torch.zeros(
        1 + max(lattice) * (_PROJ_BLOCK_M + k_width), **half)
    for stride_r in lattice:
        for stride_s in lattice:
            for stride_k in lattice:
                launch_proj(scratch, 1, _PROJ_BLOCK_M, stride_r, stride_s, stride_k)

    for rows in range(2, 7):
        x = torch.empty_strided(
            (1, rows, 9, 16, k_width),
            (rows * 147456, 147456, 16, 1, 144), **half,
        ).zero_()
        launch_proj(x, rows, spatial, 147456, 1, 144)
    torch.cuda.synchronize()


def _warm_device(index: int) -> bool:
    """Compile both kernels for one device, and record that it is warm.

    Separate from ``_build`` so a caller that genuinely wants a second device can ask
    for it explicitly, outside any timed region. Nothing on the call path invokes
    this: the predicate rejects unwarmed devices rather than warming them, because
    warming is exactly the compiler work that must not happen inside ``forward``.
    """
    global _ADALN_KERNEL, _PROJ_KERNEL, _BUILD_STATE
    if _BUILD_STATE == "failed":
        return False
    if not torch.cuda.is_available():
        return False                       # stays "pending"
    try:
        if _ADALN_KERNEL is None or _PROJ_KERNEL is None:
            _ADALN_KERNEL, _PROJ_KERNEL = _define_kernels()
        device = torch.device("cuda", index)
        with torch.cuda.device(device):
            _warm(_ADALN_KERNEL, _PROJ_KERNEL, device)
    except Exception as exc:  # a Triton / ptxas failure, not a wrong answer
        _ADALN_KERNEL = _PROJ_KERNEL = None
        _BUILD_STATE = "failed"
        # One line, to the per-operator bench log, so a build failure cannot
        # hide behind a silent 1.00x.
        print(f"oasis_final_layer: Triton build failed, delegating ({exc!r})",
              file=sys.stderr, flush=True)
        return False
    _WARMED_DEVICES.add(index)
    _BUILD_STATE = "ready"
    return True


def _build() -> bool:
    """Warm the current device, or latch the fast path off. Idempotent.

    A missing CUDA device is *not* a build failure and must not latch: a candidate
    imported before its device exists has to be able to warm later, or it would run
    the fallback for an entire bench. Only a genuine Triton/ptxas error sets
    ``failed`` and prints its one line.
    """
    if _BUILD_STATE == "ready":
        return True
    if _BUILD_STATE == "failed":
        return False
    if not torch.cuda.is_available():
        return False                       # stays "pending"
    return _warm_device(torch.cuda.current_device())


# The bench worker imports the candidate before it touches CUDA, on the device it
# will benchmark, so building here puts Triton's compiler -- which spawns
# subprocesses and threads the harness would otherwise see appear mid-timing --
# entirely outside the measured window.
_build()


def _collapsed_stride(sizes, strides):
    """The stride of a dimension group flattened to a single axis, or ``None`` if
    no such flattening exists.

    A group ``d_0..d_n`` collapses iff ``stride[i] == stride[i+1] * size[i+1]``
    for every adjacent pair, in which case the flattened stride is the innermost
    one. Size-1 dimensions are skipped: their stride is unobservable, so
    requiring anything of it would reject layouts the kernel handles. An empty
    group has length 1, so its stride is never used; 1 is returned as a
    placeholder.
    """
    step = 1
    unit = None
    for size, stride in zip(reversed(sizes), reversed(strides)):
        if size == 1:
            continue
        if unit is None:
            unit = stride
        elif stride != step:
            return None
        step = stride * size
    return 1 if unit is None else unit


class OasisFinalLayer(nn.Module):
    """adaLN-modulated final projection, same contract as ``baseline.py``.

    Holds exactly the baseline's submodules, hence exactly its four parameters
    under exactly its ``state_dict`` names, and derives nothing from them: the
    harness moves the module, casts its parameters, and only *then* shares the
    baseline's weights, so anything precomputed in ``__init__`` would be stale.
    """

    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.linear = Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.ModuleList(
            [
                SiLU(),
                Linear(hidden_size, 2 * hidden_size, bias=True),
            ]
        )
        # Plain host-side int: distinguishes "the fused path ran and tied" from
        # "the fused path never ran". No threads, no device sync, nothing the
        # harness's integrity guards watch.
        self.fastpath_calls = 0

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        geometry = self._fused_geometry(x, c)
        if geometry is None:
            return self._reference_forward(x, c)
        self.fastpath_calls += 1
        return self._fused_forward(x, c, geometry)

    def _fused_geometry(self, x: torch.Tensor, c: torch.Tensor):
        """``(rows, spatial, k_width, stride_r, stride_s, stride_k)`` if the fused
        path can run this call, else ``None``.

        Every clause guards something the kernels rely on; none of them encodes a
        captured shape. A plain contiguous ``x`` is admitted and correct, just
        uncoalesced.
        """
        # Ordering matters as much as the clauses. Everything cheap and local runs
        # first, so an ineligible call never reaches the build attempt: a CPU call on
        # a module imported without a device must not drag Triton's compiler in.
        #
        # The kernels emit no autograd graph.
        if torch.is_grad_enabled():
            return None

        # Rank before any width access: ``x.shape[-1]`` on a rank-0 tensor raises,
        # and a predicate must reject rather than throw.
        if x.dim() < 1 or c.dim() < 1:
            return None

        half = torch.float16
        if x.dtype is not half or c.dtype is not half:
            return None
        if not x.is_cuda or x.numel() == 0:
            return None

        weight, bias = self.linear.weight, self.linear.bias
        ada = self.adaLN_modulation[1]
        ada_weight, ada_bias = ada.weight, ada.bias
        if bias is None or ada_bias is None:
            return None
        params = (weight, bias, ada_weight, ada_bias)
        device = x.device
        if c.device != device or any(p.device != device for p in params):
            return None
        if any(p.dtype is not half for p in params):
            return None

        # Lazy negation and conjugation live in tensor metadata, not in storage.
        # The kernels take a raw pointer and never see the bit, so a negated view
        # would be computed with the opposite sign from what PyTorch would produce.
        # (``torch._neg_view(t)`` keeps ``t``'s data pointer, stays contiguous and
        # keeps its dtype, so it clears every other clause here.)
        if x.is_neg() or c.is_neg() or x.is_conj() or c.is_conj():
            return None
        if any(p.is_neg() or p.is_conj() for p in params):
            return None

        # Both kernels address their weights as 2-D row-major arrays, so the rank has
        # to be checked before the shape is indexed -- the same lesson as the operand
        # rank above, applied to the parameters. Reading ``weight.shape[1]`` on a
        # rank-1 parameter raises out of the predicate, and a contiguous
        # ``[N, K, 1]`` parameter would otherwise satisfy every clause below and take
        # the fast path while the baseline body rejects it outright.
        if weight.dim() != 2 or ada_weight.dim() != 2:
            return None
        if bias.dim() != 1 or ada_bias.dim() != 1:
            return None

        # Widths. ``k_width`` is read from the weight rather than from an
        # ``__init__`` argument, so a reshaped module cannot silently disagree
        # with its own parameters.
        k_width = weight.shape[1]
        if weight.shape[0] != _PROJ_N or bias.shape != (_PROJ_N,):
            return None
        # The concatenated projection is split by halving its width, so that
        # width has to be exactly twice the hidden size.
        if tuple(ada_weight.shape) != (2 * k_width, k_width):
            return None
        if ada_bias.shape != (2 * k_width,):
            return None
        if x.shape[-1] != k_width or c.shape[-1] != k_width:
            return None
        # Neither kernel masks its hidden-axis loop, and the adaLN kernel needs a
        # block to fall wholly inside one half of the concatenated output. Each
        # clause names its own kernel's requirement, so changing one tile constant
        # cannot silently invalidate the other kernel's guard.
        if k_width % _PROJ_BLOCK_K or k_width % _ADALN_BLOCK_K or k_width % _ADALN_BLOCK_N:
            return None

        # The LayerNorm the kernel reproduces is the parameterless, fp32-promoted
        # one over the hidden axis.
        norm = self.norm_final
        if norm.weight is not None or norm.bias is not None:
            return None
        if not norm.promote_fp32 or norm.normalized_shape != (k_width,):
            return None

        # Index mapping: x is addressed as [rows, spatial, k_width].
        lead = c.dim() - 1
        if lead > x.dim() - 1:
            return None
        if tuple(x.shape[:lead]) != tuple(c.shape[:-1]):
            return None
        rows = 1
        for size in c.shape[:-1]:
            rows *= size
        # The adaLN kernel pads its row axis to one fixed tile and masks; past
        # that, rows would be silently dropped.
        if rows > _ROW_TILE:
            return None
        spatial = 1
        for size in x.shape[lead:-1]:
            spatial *= size
        # The projection kernel carries no tail mask on its lane axis.
        if spatial % _PROJ_BLOCK_M:
            return None

        stride_r = _collapsed_stride(x.shape[:lead], x.stride()[:lead])
        stride_s = _collapsed_stride(x.shape[lead:-1], x.stride()[lead:-1])
        if stride_r is None or stride_s is None:
            return None

        # Every stride must land in a class ``_warm`` compiled; see ``_STRIDE_CLASS``.
        # Without this, a stride like 3 would be a binary nobody warmed, and Triton
        # would compile it inside ``forward``.
        if not all(abs(stride) == 1 or abs(stride) % _STRIDE_CLASS == 0
                   for stride in (stride_r, stride_s, x.stride()[-1])):
            return None

        # Triton does index arithmetic in int32, so every element offset either
        # kernel forms has to fit there. The captured cases peak at 884,735; these
        # bounds only bite on shapes orders of magnitude larger, and such a shape
        # would otherwise wrap silently to a wrong address rather than fail.
        stride_k = x.stride()[-1]
        widest = max(
            2 * k_width * k_width,                  # adaLN weight
            2 * rows * k_width,                     # the shift/gain scratch
            (rows - 1) * abs(stride_r)
            + (spatial - 1) * abs(stride_s)
            + (k_width - 1) * abs(stride_k),        # the strided x
            rows * spatial * _PROJ_N,               # the output
        )
        if widest >= 2 ** 31:
            return None

        # 128-bit accesses; ``c`` and the parameters are read as flat arrays.
        if not c.is_contiguous() or any(not p.is_contiguous() for p in params):
            return None
        if x.data_ptr() % _ALIGN_BYTES or c.data_ptr() % _ALIGN_BYTES:
            return None
        if any(p.data_ptr() % _ALIGN_BYTES for p in params):
            return None
        if torch.cuda.get_device_capability(device) < _MIN_CAPABILITY:
            return None

        # Last, because it is the only expensive clause. ``_build`` is a no-op once
        # warmed; it does real work only on the recovery path where CUDA was absent
        # at import, which the bench never produces.
        if _BUILD_STATE != "ready" and not _build():
            return None
        # Triton caches compiled kernels per device, so a call on a device the warm
        # never reached would compile here, inside ``forward``. Delegate instead.
        if device.index not in _WARMED_DEVICES:
            return None

        return rows, spatial, k_width, stride_r, stride_s, stride_k

    def _fused_forward(self, x: torch.Tensor, c: torch.Tensor, geometry) -> torch.Tensor:
        rows, spatial, k_width, stride_r, stride_s, stride_k = geometry
        ada = self.adaLN_modulation[1]
        device = x.device
        # Triton launches on the *current* device and stream, not on one inferred
        # from its arguments, so operands on cuda:0 would launch on cuda:1 whenever
        # cuda:1 happens to be current. Bind both the allocations and the launches.
        with torch.cuda.device(device):
            # Fresh every call: reusing a module-level scratch across calls would be
            # a stream-safety and reentrancy hazard, and would tie correctness to the
            # harness's pointer behaviour.
            mod = torch.empty((2, rows, k_width), device=device, dtype=torch.float16)
            y = torch.empty((rows, spatial, _PROJ_N), device=device, dtype=torch.float16)
            _ADALN_KERNEL[(2 * k_width // _ADALN_BLOCK_N,)](
                c, ada.weight, ada.bias, mod, rows, k_width,
                BLOCK_N=_ADALN_BLOCK_N, BLOCK_K=_ADALN_BLOCK_K, ROW_TILE=_ROW_TILE,
                num_warps=_ADALN_WARPS, num_stages=_ADALN_STAGES,
            )
            _PROJ_KERNEL[(spatial // _PROJ_BLOCK_M, rows)](
                x, mod, self.linear.weight, self.linear.bias, y,
                rows, spatial, k_width, self.norm_final.eps,
                stride_r, stride_s, stride_k,
                BLOCK_M=_PROJ_BLOCK_M, BLOCK_K=_PROJ_BLOCK_K, N=_PROJ_N,
                num_warps=_PROJ_WARPS, num_stages=_PROJ_STAGES,
            )
        # ``view`` rather than ``reshape``: if the output ever stopped being
        # contiguous this should fail loudly, not copy silently.
        return y.view(*x.shape[:-1], _PROJ_N)

    def _reference_forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """The baseline body, for every call the fused path does not accept.

        Lazy negation and conjugation are resolved first. They live in tensor
        metadata rather than in storage, and the frozen ``L1`` LayerNorm this body
        delegates to takes a raw pointer, so it would read the *un*-negated values:
        measured 6.55 absolute error against the scored baseline, which handles the
        bit correctly. Resolving here costs nothing on a tensor without the bit set
        -- ``resolve_neg`` returns ``self`` -- and keeps the fallback faithful to
        what the baseline computes rather than to what its submodules happen to do.
        """
        x = x.resolve_conj().resolve_neg()
        c = c.resolve_conj().resolve_neg()
        modulation = c
        for layer in self.adaLN_modulation:
            modulation = layer(modulation)
        shift, scale = modulation.chunk(2, dim=-1)
        while shift.dim() < x.dim():
            shift = shift.unsqueeze(-2)
            scale = scale.unsqueeze(-2)
        x = self.norm_final(x) * (1 + scale) + shift
        return self.linear(x)
