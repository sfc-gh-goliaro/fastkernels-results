"""Oasis 2D patch embedding, served by one fused channels-last kernel.

The captured configurations make ``self.proj`` a *non-overlapping* patch embed --
kernel equals stride, no padding, unit dilation, one group -- so the convolution
is exactly a GEMM over the flattened patch::

    y[n, p, o] = sum_k W[o, k] * X[n, p, k] + b[o]
        k = (c, kh, kw),  p = (oh, ow),  X[n, p, k] = x[n, c, oh*P+kh, ow*P+kw]

and the baseline's layout change after it is a pure view, not a copy. That fixes
the shape of the win: there is no transpose to fuse away, only a convolution to
replace and a launch count to hold at one. So one Triton program computes a
``[patches, out_channels]`` tile straight into a channels-last buffer, with the
bias folded in, and both of the baseline's logical shapes are views of it.

"Correct" here means "agrees with cuDNN", not "is accurate", and at fp32's
``atol=1e-5, rtol=1e-3`` the reference's own plan choice is observable. Measured
on this device (``docs/measurements/cudnn_plan_sweep.txt``, three draws per row,
against an exact float64 convolution and against a TF32 im2col GEMM):

    16ch p2 18x32     batch 1-2  exact fp32 (1.000 matched, 2e-07)
    16ch p2 18x32     batch 3-8  TF32      (exact fp32 only 0.869 matched)
    3ch p20 360x640   batch 1    exact fp32 (1.000 matched, 5e-06)

An error budget explains the split rather than merely fitting it. With
``w ~ N(0, 0.02)`` and ``x ~ N(0, 1)`` the median ``|y|`` is ``0.674*0.02*sqrt(K)``
-- 0.108 for K=64 and 0.468 for K=1200, both matching the measured medians -- so
the tolerance band is 1.2e-4 and 4.8e-4 wide. Single-pass TF32 rounds operands to
a 10-bit mantissa, giving ``0.02 * 2^-11 * sqrt(K)`` of error: 7.8e-5 and 3.4e-4.
Held against those bands a Gaussian error leaves 13% and 17% of elements outside,
which is the measured 0.869 and 0.830. So from batch 3 up on the small
configuration the reference *is* the imprecise answer, and an exact kernel is
scored wrong there. ``_MEASURED_CLASSES`` therefore names, per input class, the
arithmetic that was measured to agree, and an input class absent from it takes
the reference path -- the same discipline as ``Tile.validated_inputs`` in the
frozen ``L1/conv2d.py``.

Reproducing the reference's TF32 needs one thing the budget did not predict.
``tl.dot(input_precision="tf32")`` *truncates* the low 13 mantissa bits, while
cuBLAS and cuDNN round to nearest, and the difference is a full extra factor of
two of operand error -- measured at 8.0e-4 against a 1.2e-4 band, so single-pass
``tf32`` agrees with the reference on only 0.63 of elements, worse than an exact
kernel's 0.87 (``docs/measurements/tf32_rounding_mode.txt``, where the hardware
``tf32`` path is bit-identical to an explicit truncation). Converting both
operands with ``cvt.rn.tf32.f32`` first makes the subsequent hardware truncation
a no-op and the whole dot *bit-identical* to a torch TF32 GEMM, at K=64 and
K=256. That is what ``"tf32rn"`` below means, and it is why the batch-3-and-up
classes are available to this kernel at all.

Everything the per-call guard rejects goes down the baseline's own code path
through ``self.proj``, so an input or a module state the fused kernel cannot
reproduce gets the reference result, including the error the reference raises.

Three divergences survive that, and all are inherited rather than introduced. The
guard here rejects the fused path for each; the loss then happens one level down,
because ``self.proj`` is the frozen ``L1.conv2d`` winner and its own guard checks
fewer things than this one. Under forward-mode AD it takes its Triton route and drops
the tangent; given a ``__torch_dispatch__`` parameter subclass it reads the raw
storage; and with cuDNN disabled it computes exact fp32 while the reference has moved
to another backend. The timed baseline's ``F.conv2d`` wrapper follows all three.
Measured in ``tests/test_semantics.py`` by the three tests named
``test_*_divergence_is_inherited`` and ``test_cudnn_disabled``, which pin *where*
each divergence comes from so it cannot silently change. In every case the frozen
module declines its own route for a batch outside its validated inputs and the two
sides then agree, so this is its Triton path specifically rather than the fallback as
such. Fixing any of them would mean editing a frozen file or not calling
``self.proj`` at all, and the bench exercises none of the three, so all are recorded
rather than worked around.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.autograd.forward_ad as forward_ad
import torch.nn as nn
import triton
import triton.language as tl
from torch.nn.modules import module as _module_hooks

from ..L1.conv2d import Conv2d

# Largest element offset the kernel forms in 32-bit arithmetic. Offsets stay
# narrow deliberately (see the kernel docstring); a batch whose spans could
# exceed this is dropped at construction rather than checked per call.
_MAX_INT32_OFFSET = 2 ** 31 - 1

# The compute capability every entry in ``_MEASURED_CLASSES`` was measured on.
# Two separate things pin it: ``cvt.rn.tf32.f32`` needs a recent architecture, and
# more importantly cuDNN's plan choice -- which is what "correct" means here -- was
# only ever observed on this one. Another architecture may answer the same
# configuration with a different plan, so it takes the reference path.
_MEASURED_CAPABILITY = (10, 0)

# The cuDNN convolution precision the arithmetic table was measured under. This is
# not a flag the module sets; it is one it *reads*, because the table's whole
# content depends on it. With cuDNN allowed to use TF32 -- the default here -- the
# small configuration is answered exactly at batch 1-2 and in TF32 from batch 3 up,
# which is what ``"tf32rn"`` reproduces. Turn TF32 off and cuDNN answers those
# batches exactly instead, so the fused path would then be the imprecise side and
# agree on about 0.87 of elements. A call made under a different precision
# therefore takes the reference path rather than a table entry measured under this
# one.
#
# The new-API sub-flag is read rather than the legacy ``cudnn.allow_tf32`` alias
# for two reasons: it is the one that governs convolutions specifically, and the
# legacy alias *raises* when the sub-flags disagree with each other ("a mix of the
# legacy and new APIs"), which would turn a guard into an exception the baseline
# does not throw. Setting the legacy alias still moves this one -- measured,
# ``allow_tf32 = False`` leaves it reading ``"none"`` -- so nothing is lost.
_MEASURED_CUDNN_CONV_PRECISION = "tf32"

# And with cuDNN *enabled*, which is not the same condition. Disabling it sends
# ``F.conv2d`` to a different backend with different arithmetic: measured, the small
# configuration at batch 2 goes from exactly matching an exact fp32 convolution to
# 0.869 of elements, i.e. from the class ``"tf32x3"`` reproduces to the class
# ``"tf32rn"`` would. Every table entry names an arithmetic for one backend state.
_MEASURED_CUDNN_ENABLED = True

# Tensor types whose logical value is the bytes in their storage. A ``Parameter``
# belongs here -- it is a subclass, but it adds only autograd bookkeeping and no
# dispatch behaviour -- while any other subclass may implement
# ``__torch_dispatch__`` and compute its value from something else entirely.
# Measured: a weight subclass whose dispatch doubled it was served from raw storage
# and matched the reference on 0.0001 of elements.
_STORAGE_FAITHFUL_TYPES = (torch.Tensor, nn.Parameter)


class Tile(NamedTuple):
    """How one measured input class is mapped onto the fused kernel."""

    block_m: int        # patches per program
    block_n: int        # output channels per program; must divide embed_dim
    block_k: int        # slice of the flat C*P*P reduction held at once
    num_warps: int
    num_stages: int
    # Which arithmetic reproduces the reference for this class: "ieee" (true
    # fp32), "tf32x3" (three tf32 passes, ~2^-22 of operand mantissa), or
    # "tf32rn" (round-to-nearest operands, then one tf32 pass -- bit-identical to
    # cuBLAS TF32). Plain "tf32" is deliberately absent: its truncating operand
    # conversion agrees with no reference plan measured here.
    precision: str


# Input classes whose agreement with the reference was measured, keyed by
# ``(in_chans, embed_dim, patch, img_height, img_width)`` and then by batch. This
# is a literal: nothing inserts into it, and no admission decision reads weight
# or input values. A configuration or a batch that is not written here takes the
# reference path.
#
# The precisions come from ``docs/measurements/cudnn_plan_sweep.txt`` (which
# arithmetic cuDNN returns) confirmed against this kernel in
# ``docs/measurements/kernel_agreement.txt`` (whether this kernel reproduces it).
# The tile shapes come from the sweep in ``docs/measurements/tile_sweep.txt``.
_MEASURED_CLASSES: dict[tuple[int, int, int, int, int], dict[int, Tile]] = {
    # Small configuration, K = 16*2*2 = 64: one reduction chunk, so the whole
    # GEMM is a single unmasked dot. cuDNN answers batch 1-2 exactly and batch 3
    # up in TF32, so the arithmetic switches with the batch and nothing else.
    # These batches are close to launch-bound rather than arithmetic-bound: most
    # tiles land at the same 13.3 us that one Triton launch plus one allocation
    # costs in this loop, so the tile matches the large configuration's rather
    # than being tuned per batch. Two batches are an exception worth naming --
    # batch 3 reaches 11.3 us and batch 4 12.3 us with block_n=64, reproduced
    # across two independent sweeps, while batches 5 and 6 do not go below 13.3
    # despite doing more work. That is a real 1-2 us left on the table for a
    # reason not yet explained, recorded in tile_sweep.txt and queued rather than
    # taken: a per-batch tile split with no account of why block_n=64 helps only
    # there is worse than a uniform one that is understood.
    (16, 1024, 2, 18, 32): {
        1: Tile(16, 128, 64, 4, 2, "tf32x3"),
        2: Tile(16, 128, 64, 4, 2, "tf32x3"),
        3: Tile(16, 128, 64, 4, 2, "tf32rn"),
        4: Tile(16, 128, 64, 4, 2, "tf32rn"),
        5: Tile(16, 128, 64, 4, 2, "tf32rn"),
        6: Tile(16, 128, 64, 4, 2, "tf32rn"),
        7: Tile(16, 128, 64, 4, 2, "tf32rn"),
        8: Tile(16, 128, 64, 4, 2, "tf32rn"),
    },
    # Large configuration, K = 3*20*20 = 1200. cuDNN is exact here, so the
    # arithmetic has to be better than one tf32 pass; "tf32x3" is what pays for
    # that on tensor cores. True "ieee" gives the identical answer and measured
    # 2.77x slower on this class -- 138.3 us against 50.0 us. (On the small
    # configuration the same comparison is only 1.16x, so this is a property of
    # K=1200, not of the two arithmetics in general.)
    #
    # This is the one class where the tile is worth a real margin, and the thing
    # it buys is occupancy rather than reuse. A reuse-shaped tile
    # (block_m 64, block_n 64, block_k 128) needs 196 KB of shared memory, which
    # fits one CTA per SM; with only 144 CTAs in the grid that leaves each SM
    # running four warps against a 19-step dependent reduction, and it measured
    # 0.89x -- slower than cuDNN. Sixteen rows instead of sixty-four costs
    # weight re-reads and wins anyway, by 2.4x, because 288 smaller CTAs is the
    # only source of latency hiding this kernel has.
    (3, 1024, 20, 360, 640): {
        1: Tile(16, 128, 64, 4, 3, "tf32x3"),
    },
}


class _Launch(NamedTuple):
    """Everything one admitted batch needs, resolved at construction.

    ``forward`` must not compute a grid, a chunk count or an output shape: the
    measured window is tens of microseconds wide and a large fraction of it is
    Python, so every admitted batch gets its launch arguments built once here.
    """

    grid: tuple[int, int]
    consts: tuple            # the kernel's constexpr arguments, in order
    shape: tuple[int, ...]   # output shape when self.flatten is False
    flat_shape: tuple[int, ...]  # output shape when self.flatten is True
    numel: int
    num_warps: int
    num_stages: int


def _reads_as_stored(t: torch.Tensor) -> bool:
    """Whether a tensor's logical values equal the bytes in its storage.

    ``neg`` and ``conj`` views carry a lazy flag that ATen applies when it reads
    them, so the values a PyTorch op sees are not the values in memory. A kernel
    that dereferences the pointer never sees that flag, so such a view must reach
    the reference path instead.
    """
    return not t.is_neg() and not t.is_conj()


# ``precision`` -> (tl.dot input_precision, round operands to nearest tf32 first).
_ARITHMETIC = {
    "ieee": ("ieee", False),
    "tf32x3": ("tf32x3", False),
    "tf32rn": ("tf32", True),
}


@triton.jit
def _to_nearest_tf32(v):
    """Round an fp32 block to the nearest tf32 value, ties to even.

    ``tl.dot``'s own tf32 conversion truncates, so a dot fed pre-rounded operands
    is what reproduces a vendor TF32 plan; the truncation it then applies is the
    identity. Done with the hardware instruction rather than integer arithmetic on
    the bit pattern because that also keeps NaN and infinity intact -- adding a
    rounding bias to a NaN's mantissa carries into its exponent and yields
    infinity, which the reference would not do.
    """
    return tl.inline_asm_elementwise("cvt.rn.tf32.f32 $0, $1;", "=r,r", [v],
                                     dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _patch_embed_nhwc(
    x_ptr, w_ptr, bias_ptr, y_ptr,
    PATCHES: tl.constexpr,             # batch * PATCHES_PER_SAMPLE
    PATCHES_PER_SAMPLE: tl.constexpr,  # grid_h * grid_w
    GRID_W: tl.constexpr,
    PATCH: tl.constexpr,
    CHANNELS: tl.constexpr,
    OUT_CHANNELS: tl.constexpr,
    IN_H: tl.constexpr, IN_W: tl.constexpr,
    REDUCTION: tl.constexpr,           # CHANNELS * PATCH * PATCH
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    K_CHUNKS: tl.constexpr, K_EXACT: tl.constexpr,
    PRECISION: tl.constexpr, ROUND_TF32: tl.constexpr,
):
    """One ``[BLOCK_M patches, BLOCK_N channels]`` tile of channels-last output.

    The accumulator is patch-major and channel-fast, which is the opposite
    orientation to the frozen ``L1`` implicit-GEMM kernel and the one substantive
    difference between them: that kernel must write NCHW, this one writes the
    layout the caller actually wants, so the store's fastest axis is the
    1024-wide channel axis and the baseline's permute costs nothing here either.

    The reduction runs over the flattened ``C*P*P`` extent, decoding ``(c, kh, kw)``
    from the flat index, because a flat reduction index *is* the offset within a
    packed filter -- so the weight tile is read contiguously along ``k`` and
    transposed for the MMA, rather than read strided by ``REDUCTION`` and wasting
    28 of every 32 bytes fetched.

    Every extent is a compile-time constant: the input is known contiguous (the
    guard requires it) with a shape the guard has already matched, so the strides
    are derived here rather than passed, the ``//`` and ``%`` decodes strength-reduce,
    and the only per-call work left is the pointer arguments. One specialization
    is compiled per admitted (configuration, batch), all of them during the
    bench's correctness rounds and warmup rather than inside a timed iteration.

    Offsets are 32-bit element counts added to a scalar base pointer; the address
    block is the largest register consumer in a gather kernel this shallow, and
    construction refuses any batch whose spans would not fit. The weight offsets
    are rebuilt per chunk rather than hoisted as a loop-invariant block plus a
    scalar: the hoisted form was measured in-process against this one and came out
    1.6% faster on the large configuration, inside the noise, while raising the
    register count from 220 to 251 -- four short of the spill cliff. ptxas already
    strength-reduces the multiply, which is why there was nothing to win.
    """
    m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    o = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    m_ok = m < PATCHES

    # Flat offset of each patch's top-left source pixel in channel 0.
    within = m % PATCHES_PER_SAMPLE
    x_row = ((m // PATCHES_PER_SAMPLE) * (CHANNELS * IN_H * IN_W)
             + (within // GRID_W) * (PATCH * IN_W)
             + (within % GRID_W) * PATCH)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for chunk in range(K_CHUNKS):
        k = chunk * BLOCK_K + tl.arange(0, BLOCK_K)
        tap = k % (PATCH * PATCH)
        # Offset of one reduction element relative to a patch's top-left pixel.
        x_col = ((k // (PATCH * PATCH)) * (IN_H * IN_W)
                 + (tap // PATCH) * IN_W + (tap % PATCH))
        x_off = x_row[:, None] + x_col[None, :]
        w_off = o[:, None] * REDUCTION + k[None, :]
        if K_EXACT:
            xv = tl.load(x_ptr + x_off, mask=m_ok[:, None], other=0.0)
            wv = tl.load(w_ptr + w_off)
        else:
            k_ok = k < REDUCTION
            xv = tl.load(x_ptr + x_off, mask=m_ok[:, None] & k_ok[None, :],
                         other=0.0)
            wv = tl.load(w_ptr + w_off, mask=k_ok[None, :], other=0.0)
        if ROUND_TF32:
            xv, wv = _to_nearest_tf32(xv), _to_nearest_tf32(wv)
        acc = tl.dot(xv, tl.trans(wv), acc, input_precision=PRECISION)

    acc += tl.load(bias_ptr + o)[None, :]
    # No mask on the channel axis: construction only admits a tile whose
    # BLOCK_N divides embed_dim.
    tl.store(y_ptr + m[:, None] * OUT_CHANNELS + o[None, :], acc,
             mask=m_ok[:, None])


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

        # Admission is decided from the constructor arguments alone. The harness
        # fills the weight *after* construction, so nothing here may depend on
        # what the weight holds -- not even its shape, which is derived from the
        # arguments rather than read off a parameter of uninitialized memory.
        self._plans = self._build_plans(
            img_height, img_width, patch_size, in_chans, embed_dim)
        self._in_chans = in_chans
        self._embed_dim = embed_dim
        self._in_h = img_height
        self._in_w = img_width
        self._weight_shape = (embed_dim, in_chans, patch_size, patch_size)
        # Only what ``F.conv2d`` itself reads. It takes the kernel extent and the
        # channel counts from the weight's shape, which is checked separately, so
        # ``in_channels`` / ``out_channels`` / ``kernel_size`` are deliberately
        # absent: reassigning one changes neither side's result, and the frozen
        # L1 ``Conv2d`` is the only implementation that even defines them -- the
        # baseline wrapper this module also has to work against does not.
        self._conv_config = ((patch_size, patch_size), (0, 0), (1, 1), 1)
        # Which CUDA devices carry the capability the table was measured on.
        # Resolved once here rather than per call: ``get_device_capability`` costs
        # about 1.7 us, which is a real fraction of this module's per-call Python,
        # while comparing a device index against a short tuple costs tens of
        # nanoseconds. Every visible device is resolved, so moving the module
        # between them is still handled.
        self._measured_devices: tuple[int, ...] = ()
        if torch.cuda.is_available():
            self._measured_devices = tuple(
                i for i in range(torch.cuda.device_count())
                if torch.cuda.get_device_capability(i) == _MEASURED_CAPABILITY)

        # Which *kind* of module the plans were built for. A caller may replace
        # ``self.proj`` outright, and the replacement need not even have a
        # ``stride`` to compare -- so the class is checked before anything is read
        # off it. Storing the class rather than a reference to the instance keeps
        # ``self.proj`` from being registered a second time (which would add
        # state-dict keys), survives ``deepcopy`` and pickling, and admits a fresh
        # instance of the same class, whose ``forward`` is this convolution by
        # construction once the configuration and parameters below check out.
        self._proj_type = type(self.proj)
        # The functions the child's call path resolved to at construction, and the
        # code object inside each. Both are needed and neither implies the other:
        # rebinding ``Conv2d.forward`` changes which function is found while the class
        # object stays the same, and assigning to that function's ``__code__`` changes
        # what it executes while the function object itself stays the same. Measured:
        # a ``__code__`` swap to an equivalent convolution plus one was served by the
        # fused path and agreed with the reference on 0.0 of elements.
        self._proj_forward = self._proj_type.forward
        self._proj_call_impl = self._proj_type._call_impl
        self._proj_dunder_call = self._proj_type.__call__
        self._proj_code = self._resolved_child_code()

    def _resolved_child_code(self) -> tuple:
        """The code objects currently inside the child's call-path functions."""
        return (self._proj_forward.__code__, self._proj_call_impl.__code__,
                self._proj_dunder_call.__code__)

    # Code objects cannot be pickled, and this module is imported by higher-level
    # baselines that a caller may reasonably ``deepcopy`` or ``torch.save``. The
    # tuple is derived from the class, not owned by the instance, so it is dropped
    # on the way out and re-resolved on the way in -- which is also the right
    # semantics: a fresh copy should trust whatever the class holds now, exactly as
    # a fresh construction would.
    def __getstate__(self):
        state = super().__getstate__()
        if isinstance(state, dict):
            state.pop("_proj_code", None)
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self._proj_code = self._resolved_child_code()

    def _build_plans(self, height: int, width: int, patch: int, in_chans: int,
                     embed_dim: int) -> dict[int, _Launch] | None:
        """Launch arguments per admitted batch, or ``None`` if none is admitted."""
        if patch <= 0 or height % patch or width % patch:
            return None
        tiles = _MEASURED_CLASSES.get((in_chans, embed_dim, patch, height, width))
        if tiles is None:
            return None
        grid_h, grid_w = height // patch, width // patch
        per_sample = grid_h * grid_w
        reduction = in_chans * patch * patch
        plans: dict[int, _Launch] = {}
        for batch, tile in tiles.items():
            if embed_dim % tile.block_n:
                # The store leaves the channel axis unmasked.
                continue
            patches = batch * per_sample
            if max(batch * in_chans * height * width, patches * embed_dim,
                   embed_dim * reduction) > _MAX_INT32_OFFSET:
                continue
            plans[batch] = _Launch(
                grid=(triton.cdiv(patches, tile.block_m),
                      embed_dim // tile.block_n),
                consts=(patches, per_sample, grid_w, patch, in_chans, embed_dim,
                        height, width, reduction, tile.block_m, tile.block_n,
                        tile.block_k, triton.cdiv(reduction, tile.block_k),
                        reduction % tile.block_k == 0)
                       + _ARITHMETIC[tile.precision],
                shape=(batch, grid_h, grid_w, embed_dim),
                flat_shape=(batch, per_sample, embed_dim),
                numel=patches * embed_dim,
                num_warps=tile.num_warps,
                num_stages=tile.num_stages,
            )
        return plans or None

    def _admitted_plan(self, x: torch.Tensor, height: int, width: int):
        """The launch plan for *this* call, or ``None`` to use the reference path.

        Ordered so the checks that reject most cheaply come first.
        ``docs/measurements/guard_cost.txt`` measures that rather than asserting it,
        and holds the current figures; they are not restated here because they move
        with the machine and a stale copy in the source is worse than a pointer to a
        fresh one. The shape of the result is what matters: the cheapest rejection is
        the batch table lookup, the most expensive rejection is an order of magnitude
        under the accepted chain, and the accepted chain is itself a small fraction of
        an accepted ``forward`` once allocation and launch are counted.

        No predicate here allocates a tensor, walks strides in a Python loop, or
        formats a string, and the same probe shows CUDA allocation unchanged across
        32000 evaluations. Transient Python objects are another matter and the probe
        counts them rather than claiming none: reading ``.shape`` yields a
        ``torch.Size`` and ``.device`` a ``torch.device``, three of each, plus two
        small tuples for the configuration and bias-shape comparisons.

        ``x`` is known four-dimensional: ``forward`` unpacks ``x.shape`` into four
        names first, which raises the ``ValueError`` the baseline raises for any
        other rank before this is reached.

        Not checked, because the fused path already reproduces the baseline for
        them: ``self.flatten`` is read live and only selects which view of the
        same buffer is returned; ``self.patch_size``, ``self.grid_size`` and
        ``self.num_patches`` are never read by the baseline's ``forward``, so
        reassigning one changes neither side; and ``self.img_size`` drives the
        spatial assertion, which ``forward`` evaluates against the live value
        while this compares the input against the extents the kernel was
        compiled for, so a reassignment either raises exactly as the baseline
        does or falls out here.

        What *is* checked about ``self.proj`` covers the whole call path rather than
        just ``forward``, because no one link implies the others: the class, so a
        replacement module cannot be read at all; the functions ``forward``,
        ``_call_impl`` and ``__call__`` resolved to on that class, because replacing
        any of them leaves the class object identical; the instance dictionary, which
        can shadow all three; and ``_compiled_call_impl``, which ``torch.compile``
        sets and which takes precedence over every one of them.
        """
        plans = self._plans
        if plans is None:
            return None
        # The exact-type check leads, before this function reads any other attribute
        # of ``x``. A subclass is free to make ``dtype`` or ``shape`` raise, and the
        # reference would still serve it, so reading either first would turn this
        # guard into an exception the baseline does not throw. (``forward`` does read
        # ``x.shape`` before calling this, which is deliberate: the baseline unpacks
        # it there too, so both sides raise together on a bad rank.) Leading with it
        # costs nothing, since it is also the cheapest predicate here: a fake,
        # functional or meta tensor, or any subclass whose logical value comes from a
        # dispatch rule rather than its storage, cannot be read through a raw pointer.
        if type(x) not in _STORAGE_FAITHFUL_TYPES:
            return None
        # Then the admission table, the narrowest predicate and the one the
        # captured-but-unmeasured classes fail on.
        plan = plans.get(x.shape[0])
        if plan is None:
            return None
        if (x.dtype is not torch.float32 or height != self._in_h
                or width != self._in_w or x.shape[1] != self._in_chans):
            return None
        # The kernel derives its strides from the extents above, so a layout that
        # is not packed would be read as though it were.
        if not x.is_cuda or not x.is_contiguous() or not _reads_as_stored(x):
            return None
        if x.device.index not in self._measured_devices:
            return None
        if self.norm is not None:
            return None
        # The fused path is forward-only and reaches memory by pointer, below the
        # dispatcher that implements graph building, autocast, forward-mode
        # tangents and any active dispatch or function mode, so a call using any
        # of them would silently lose it. ``no_grad`` only turns off the reverse
        # mode, hence the separate forward-mode check.
        if torch.is_grad_enabled() or torch.is_autocast_enabled("cuda"):
            return None
        if (forward_ad._current_level >= 0
                or torch._C._len_torch_dispatch_stack()
                or torch._C._len_torch_function_stack()):
            return None
        # The reference computes its arithmetic on whichever backend cuDNN's state
        # selects, and the table says which arithmetic that is only for the state it
        # was measured under -- both which precision cuDNN may use, and whether it
        # is used at all.
        if (torch.backends.cudnn.conv.fp32_precision
                != _MEASURED_CUDNN_CONV_PRECISION
                or torch.backends.cudnn.enabled is not _MEASURED_CUDNN_ENABLED):
            return None
        proj = self.proj
        if type(proj) is not self._proj_type:
            return None
        # The fused path does not call ``self.proj``, so anything that would run as
        # part of calling it must send this call to the reference path instead. The
        # baseline's own hooks are not a concern -- both modules run their own --
        # but ``proj``'s are, and this asymmetry exists only because this module
        # replaces its child's work rather than its own.
        #
        # Calling a module is ``__call__`` -> ``_call_impl`` (or
        # ``_compiled_call_impl``) -> hooks -> ``forward``, and every link is
        # replaceable. Each check below stands for a substitution that was measured to
        # change what the reference computes while leaving the class identity above
        # intact: a hook on the instance; a hook registered globally for every module
        # (a global forward hook ran on the reference path and not here, and the two
        # agreed on 0.00006 of elements); a ``torch.compile`` wrapper in
        # ``_compiled_call_impl``, which precedes all of the rest; ``forward`` or
        # ``_call_impl`` shadowed on the instance; any of the three functions rebound
        # on the class; and any of their ``__code__`` objects reassigned in place,
        # which leaves every function identity intact and agreed on 0.0 of elements.
        #
        # ``__call__`` is checked on the class only, and deliberately not on the
        # instance: ``proj(x)`` resolves it through the type, so an instance attribute
        # of that name is never consulted -- verified, and an earlier version of this
        # guard rejected on it for a hazard that does not exist.
        #
        # Backward hooks are absent because they change what the reference builds for
        # the backward pass, not the values it returns, and this path is only entered
        # with grad disabled. Rebinding a name inside the frozen module's own globals
        # is not covered and is accepted: the globals dict keeps its identity, so
        # nothing short of comparing its contents would see it.
        if (proj._forward_hooks or proj._forward_pre_hooks
                or _module_hooks._global_forward_hooks
                or _module_hooks._global_forward_pre_hooks
                or proj._compiled_call_impl is not None
                or "forward" in proj.__dict__
                or "_call_impl" in proj.__dict__
                or self._proj_type.forward is not self._proj_forward
                or self._proj_type._call_impl is not self._proj_call_impl
                or self._proj_type.__call__ is not self._proj_dunder_call
                or self._resolved_child_code() != self._proj_code):
            return None
        # Same object, but every part of its configuration is a plain attribute
        # a caller can reassign, and the reference reads all of these per call.
        # Only a unit-dilation, unpadded, single-group convolution whose stride
        # equals its kernel is the GEMM the kernel implements.
        if (proj.stride, proj.padding, proj.dilation,
                proj.groups) != self._conv_config:
            return None
        # The parameters are read as raw memory. Contiguity is load-bearing: a
        # flat reduction index only equals the offset within the filter when the
        # filter is packed.
        weight = proj.weight
        if (type(weight) not in _STORAGE_FAITHFUL_TYPES
                or weight.shape != self._weight_shape
                or weight.dtype is not torch.float32
                or weight.device != x.device
                or not weight.is_contiguous()
                or not _reads_as_stored(weight)):
            return None
        bias = proj.bias
        if (bias is None
                or type(bias) not in _STORAGE_FAITHFUL_TYPES
                or bias.shape != (self._embed_dim,)
                or bias.dtype is not torch.float32
                or bias.device != x.device
                or not bias.is_contiguous()
                or not _reads_as_stored(bias)):
            return None
        return plan

    def _fused(self, x: torch.Tensor, plan: _Launch) -> torch.Tensor:
        """One launch into one flat buffer, returned as the requested view.

        The buffer is allocated flat, exactly ``batch*grid_h*grid_w*embed_dim``
        elements, which is how the kernel addresses it; both of the baseline's
        logical shapes are then views of that one allocation, which is the whole
        reason a single kernel can serve both layout branches. The baseline instead
        returns a non-contiguous view of an NCHW buffer, so the values and the
        shape match while the strides do not -- a deliberate divergence, and the
        reason the store is fully coalesced.
        """
        proj = self.proj
        y = torch.empty(plan.numel, dtype=x.dtype, device=x.device)
        _patch_embed_nhwc[plan.grid](
            x, proj.weight, proj.bias, y, *plan.consts,
            num_warps=plan.num_warps, num_stages=plan.num_stages,
        )
        return y.view(plan.flat_shape if self.flatten else plan.shape)

    def forward(self, x: torch.Tensor, random_sample: bool = False) -> torch.Tensor:
        _, _, height, width = x.shape
        if not random_sample and (height, width) != self.img_size:
            raise AssertionError(
                f"Input image size ({height}*{width}) doesn't match model {self.img_size}.",
            )
        plan = self._admitted_plan(x, height, width)
        if plan is not None:
            return self._fused(x, plan)
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)
        else:
            x = x.permute(0, 2, 3, 1)
        return self.norm(x) if self.norm is not None else x
