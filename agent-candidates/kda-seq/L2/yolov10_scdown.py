"""SCDown: the whole ``1x1 -> BN -> SiLU -> depthwise 3x3 s2 -> BN`` block in one kernel.

Every number quoted here comes from a committed script; the script is named where the
number is.

The captured workload is dispatch-bound at both ends, which is what the design is shaped
around. On the device side, the block's minimum traffic for the largest benched case is
3.3 MB read and 1.6 MB written -- arithmetic from the benched shapes, not a measurement --
while the five stages the reference issues cost 10-30 us each
(``profile/p1_baseline_probe/probe_pool.py``, which re-derives them with the benchmark's
shifting pool in place). On the *host* side,
this machine charges ~22 us of Python and dispatch for a single ``F.conv2d`` call and ~24 us
for a single ``F.batch_norm``, so the reference's forward costs ~98 us of host time before
the GPU is asked to do anything, while a bare Triton launch costs 8.8 us with one argument
and 16.8 us with twenty-one (``tests/host_cost.py``). The benchmark records a CUDA-event
span around each iteration, so whichever of the two chains is longer is what it measures.
This module's whole forward costs ~48 us of host time against the reference's ~98 us. How much
of the measured speedup that accounts for is *not* decomposed: the window also contains the
input copy (9.2 us) and the kernels themselves, and on every case the measured wall sits
10-20 us above ``copy + kernel``, a gap that tracks the number of launches. So host cost is
material and worth cutting, which is why the code below does, but the split between the two
causes is not something these measurements establish.

Two consequences run through the whole file. Collapsing five launches into one is the larger
lever than making any stage faster, and reducing the count part-way is not obviously enough:
``probe.py`` measured a folded four-op torch path *losing* to the five-op baseline at batch 4
(105.6 us vs 77.7 us), and the two-kernel control below lost as well. Neither shows that one
launch is the only count that works -- they show that the two intermediate counts tried did
not. And the per-call Python has to stay lean: the guard below is
written to be complete without being chatty, the tile rule is integer arithmetic rather
than ``triton.cdiv`` calls, and the kernel takes the smallest set of arguments its work
needs, deriving the rest.

Two optimized routes exist and were measured against each other on the benchmark's own
recipe -- shifting input pool, 253 MiB L2 flush, median of 50 CUDA-event spans
(``tests/timing.py``):

* ``_FUSED`` -- one kernel. A program owns one ``(sample, channel block, output row,
  run of output columns)`` and never materializes the activation.
* ``_SPLIT`` -- the control: two kernels, the first writing ``SiLU(BN1(1x1(x)))`` and the
  second reducing the 3x3 window out of it. It evaluates each activation position exactly
  once instead of the fused route's ~1.5x, and pays a second launch and the
  write-and-reread of the intermediate for it.

The fused route ships because it measured faster on all four benched cases -- narrowly on B.
Against the reference it is 1.40x / 1.07x / 2.00x / 1.72x, while the split control is
0.67x / 0.58x / 0.52x / 0.48x. Its depthwise kernel is the dominant device-side cost -- 51 us
against 6 us for the pointwise half -- on top of which it pays a second launch and the
intermediate's write and reread. ``benchmark.csv`` carries both.

How the fused kernel reads ``x`` is the part worth explaining, because the obvious
formulation is much slower and was measured to be. Gathering each of the nine taps
separately gives each program nine ``[channels, positions]`` tiles whose position axis has
memory stride 2 and whose reduction axis has stride ``in_h * in_w``; ncu put that at 2.2 of
every 32 bytes per sector, 255 registers per thread, 553k local-memory spill requests and
12.5% occupancy (``profile/p1_fused_v1_baseline/REPORT.md``). Loading a *contiguous run*
of input columns instead and separating the even and odd columns out of the register tile
costs the same arithmetic, needs two dots per row tap instead of three, and brings two
thirds of the bytes in at stride 1. Comparing each formulation at its own best tile, that is
46.8 us against 27.4 us on case A and 29.1 us against 14.1 us on case D
(``tests/compare_formulations.py``).

Two things about the block are easy to get wrong and are worth stating where the code is:

* The zero padding belongs to ``cv2``'s *input*, i.e. to the activation, not to ``x``. An
  out-of-bounds tap must contribute exactly ``0``, whereas an in-bounds position whose
  pointwise sum happens to be zero contributes ``SiLU(shift1)``, which is not zero. So the
  bounds mask is applied after the activation and the gather is free to load zeros.
  Getting this wrong moves every border output, and it survives the benchmark's own
  tolerance on the benchmark's own weights -- 2.1e-3 against an ``atol`` of 1e-2 -- which
  is why ``tests/check_correctness.py`` carries a weight recipe where the same error is
  1.4e+01 instead.
* BN is read raw on every call and folded inside the kernel. Caching a folded weight would
  need a validity key over ten tensors to survive ``load_state_dict`` and in-place weight
  edits -- Python work inside a budget the launch floor already fills, and a staleness bug
  waiting to happen. Folding costs one divide and one multiply-add per channel per program.

The route *tag* comes from constructor arguments alone; weight values never enter the
decision, which is what keeps it stable across ``load_state_dict`` (and is checked by
constructing on the meta device). Whether that route applies to a given call is a separate
question answered by ``_fused_route_applies``. Everything it checks is there because this
module must behave exactly like ``baseline.py``, which reads its attributes and parameters
on every call and dispatches through ATen. Anything the guard rejects gets the reference
result -- built from the same frozen L1 primitives the baseline composes -- including the
error the reference would raise.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn
import triton
import triton.language as tl

from ..L1.batch_norm2d import BatchNorm2d
from ..L1.conv2d import Conv2d
from ..L1.silu import SiLU

# The process-wide hook registries ``nn.Module.__call__`` consults on the way into *every*
# module. Bound once here because the guard reads them on every call; they are the same dict
# objects for the process's lifetime, so binding the names cannot go stale.
_GLOBAL_FORWARD_HOOKS = nn.modules.module._global_forward_hooks
_GLOBAL_FORWARD_PRE_HOOKS = nn.modules.module._global_forward_pre_hooks
_GLOBAL_BACKWARD_HOOKS = nn.modules.module._global_backward_hooks
_GLOBAL_BACKWARD_PRE_HOOKS = nn.modules.module._global_backward_pre_hooks

# Route tags. Small ints so the per-call dispatch is an integer compare.
_REFERENCE = 0
_FUSED = 1
_SPLIT = 2

# Which optimized route ships, from the interleaved A/B in ``tests/timing.py`` rather than
# from the argument that one launch must beat two.
_OPTIMIZED_ROUTE = _FUSED

# Largest element offset either kernel forms in 32-bit arithmetic. Offsets are kept narrow
# deliberately -- the address block is the biggest register consumer in a gather this
# shallow -- so anything that could exceed this takes the reference path rather than
# silently wrapping.
_MAX_INT32_OFFSET = 2 ** 31 - 1

# Output columns per program in the fused route; see ``_tile_for`` for the sweep behind it.
_BLOCK_W = 16
_MIN_BLOCK_C = 32
_MAX_BLOCK_C = 128

# The dtypes BN can arrive in, and they are not the same set for the affine terms as for the
# running statistics. ``F.batch_norm`` with an fp16 input accepts ``weight`` and ``bias`` only
# when the two agree and are both fp16 or both fp32 -- every other combination raises, bf16
# included -- while ``running_mean`` and ``running_var`` are accepted in fp16, bf16 or fp32.
# That matrix is measured, not assumed: ``tests/check_correctness.py`` walks all twelve
# single-tensor cases and all nine affine pairs, and requires this module to agree with the
# reference on each, which for the rejected combinations means raising what it raises.
# Frozen sets so the guard's membership tests do not rebuild a tuple per call.
# The rule is *joint*, not per-tensor: which running-statistic dtypes are accepted depends on
# the affine dtype. Measured over the full 27-case cross product (three dtypes for the affine
# pair by three for each buffer), an fp16 affine pair accepts running statistics in any of the
# three, while an fp32 affine pair requires both buffers fp32 -- eight of the nine fp32-affine
# combinations raise. Checking each tensor against its own set admitted all eight.
_BN_BUFFERS_FOR_AFFINE = {
    torch.float16: frozenset((torch.float16, torch.bfloat16, torch.float32)),
    torch.float32: frozenset((torch.float32,)),
}

# CUDA's grid limits. The x axis is effectively unbounded here; y and z are not, and the fused
# launch puts output rows on y and the batch on z, both of which a caller can exceed with a
# shape the reference handles fine.
_MAX_GRID_X = 2 ** 31 - 1
_MAX_GRID_YZ = 65535

# Largest reduction block the fused kernel compiles at. Measured, not assumed: 512 lowers and
# agrees with the reference, while 1024 and 2048 fail in Triton's sm_100 pipeline
# (``PassManager::run failed``) and a 4096-wide ``w1`` tile asks for 256 KB of shared memory.
# ``c1`` above this takes the reference path.
_MAX_BLOCK_K = 512

_FP16_OR_FP32 = frozenset((torch.float16, torch.float32))

# Positions per program in the split control's two kernels. Named because the int32 offset
# bound has to account for its tail.
_SPLIT_BLOCK_P = 64


class Tile(NamedTuple):
    """How one call is mapped onto the kernels."""

    block_c: int          # output channels per program
    block_w: int          # output columns per program
    num_warps: int
    num_stages: int


def _tile_for(channels: int) -> Tile:
    """Pick the tile. It depends on the channel extent and nothing else.

    The numbers below are matched rows of ``tests/timing.py --tiles`` on the benchmark recipe
    (``profile/p1_measurements/timing_tiles.txt``), at ``block_w = 16`` and equal warp counts,
    because comparing across two axes at once is how a tile sweep talks itself into the wrong
    answer.

    Channel block, at the shipped 4 warps -- 128 / 64 / 32 microseconds per case::

        A [4,64,80,80]    57.3   61.4   94.2      128 best
        B [4,128,40,40]   61.5   71.5  101.4      128 best
        C [1,64,80,80]    30.7   28.7   36.8      64 best by one timer tick
        D [1,128,40,40]   32.8   36.8   47.1      128 best

    So the widest block wins on three of four cases and is one tick behind on the fourth, and
    narrowing to 32 is much worse everywhere. 128 is taken as the single choice; the cost is
    that one tick on case C, and the alternative is a table keyed on the captured shapes, which
    Phase 3 is where such a thing belongs. Note this is the opposite of what a program count
    alone would suggest -- cases C and D produce 120 and 80 programs against 148 SMs at
    ``block_c = 128`` -- because every extra channel block re-reads the same input columns.
    Row strips already supply programs along two axes, so the device's SM count does not enter;
    an earlier version consulted it and measured worse.

    Warps: 4 rather than 8 on the strength of case A, where it is 57.3 against 69.6 (six
    ticks). B is tied at 61.5; C and D prefer 8 by two ticks and one tick respectively, which
    A's margin outweighs.

    Output columns per program: 16. The run this kernel loads is twice that wide and both are
    ``tl.dot`` extents, so 16 is the floor, and it is also the optimum -- 57.3 us at 16 against
    129.1 us at 32 on case A -- because the run tile grows with it and the register file is what
    this kernel runs out of first.

    Spatial *shape*, not just width: giving a program one output row is the most elongated tile
    there is, and it was measured against square and intermediate ones on this same kernel with
    the row index lifted into a compile-time loop (``tests/compare_geometry.py``). Every geometry
    is validated against the baseline on all four cases and against this kernel's own output
    bit-for-bit before it is timed, so the comparison is geometry and not arithmetic. Case A,
    microseconds on the benchmark recipe::

        1x16 (shipped)   55.3      4x16  157.7      8x8   168.0
        2x32            180.3      1x64  237.6     16x16  643.1

    The one-row tile wins by 2.9-11.6x and the ordering holds on B, C and D, so it is what
    ships. (Absolute figures move between runs on this shared machine; the ordering does not.)

    That is the whole of what the experiment shows, and it is worth saying what it does not. It
    does not adjudicate the recompute argument that first motivated a one-row tile: the benched
    outputs are 40 and 20 columns wide, so a ``1x64`` program has only 40 or 20 live lanes, and
    the equal-work premise that derivation needs does not hold at these shapes. It also does not
    identify a cause -- no per-geometry register or spill figure was collected. The reason to
    prefer 16 columns over 32 in the width sweep above is likewise a measurement, not a
    mechanism.

    The channel block is the channel count rounded *up* to a power of two, capped at 128 and
    floored at 32, so ``c2 = 96`` gives 128 and one masked block rather than 64 and two: the
    mask wastes lanes once, while a second channel block would re-read every input column.
    """
    rounded = 1 << (max(channels, 1) - 1).bit_length()
    block_c = min(_MAX_BLOCK_C, max(_MIN_BLOCK_C, rounded))
    return Tile(block_c=block_c, block_w=_BLOCK_W, num_warps=4, num_stages=2)


def _reads_as_stored(t: torch.Tensor) -> bool:
    """Whether a tensor's logical values equal the bytes in its storage.

    ``neg`` and ``conj`` views carry a lazy flag ATen applies when it reads them, so the
    values a PyTorch op sees are not the values in memory. A kernel that dereferences the
    pointer never sees that flag -- measured: a ``torch._neg_view`` input on a route that
    admits it agrees with the reference on 0.73 of its elements -- so such a view must
    reach the reference path.
    """
    return not t.is_neg() and not t.is_conj()


@triton.jit
def _fold_bn(weight_ptr, bias_ptr, mean_ptr, var_ptr, c, c_ok, eps):
    """BN in inference mode as an affine map: ``scale * v + shift``.

    Everything is promoted to fp32 because the four tensors do not arrive in one dtype:
    the benchmark casts ``parameters()`` to the run dtype and leaves buffers alone, so
    ``weight`` and ``bias`` are fp16 while ``running_mean`` and ``running_var`` stay fp32.
    ``sqrt`` and a divide rather than ``rsqrt``, both IEEE-rounded -- this is one divide
    per channel per program.
    """
    gamma = tl.load(weight_ptr + c, mask=c_ok, other=0.0).to(tl.float32)
    beta = tl.load(bias_ptr + c, mask=c_ok, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=c_ok, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + c, mask=c_ok, other=0.0).to(tl.float32)
    scale = gamma / tl.sqrt(var + eps)
    return scale, beta - mean * scale


@triton.jit
def _activate(acc, scale, shift):
    """``fp16(SiLU(fp16(BN1(fp16(acc)))))`` -- the value the depthwise conv reads.

    The rounding points mirror the reference exactly: cuDNN rounds the pointwise conv's
    output to fp16, ``F.batch_norm`` reads and writes fp16 while computing in fp32, and
    ``F.silu`` does the same. Keeping the intermediate rounding is not about scraping past
    a tolerance -- the margin is wide -- it reduces the risk of the two paths drifting apart
    on weights nobody tested, and it is checked rather than assumed: on every tested case the
    fused output's distance from a float64 reference comes out *equal* to the reference's own
    distance from it, and ``tests/check_correctness.py`` asserts that equality. Dropping any
    one of the four roundings breaks it while moving the result by only ~1e-5, which no
    tolerance here could see. That is evidence about the tested recipes, not a proof over all
    weights.
    """
    z = acc.to(tl.float16).to(tl.float32)
    z = (z * scale[:, None] + shift[:, None]).to(tl.float16).to(tl.float32)
    return (z / (1.0 + tl.exp(-z))).to(tl.float16).to(tl.float32)


@triton.jit
def _scdown_fused(
    x_ptr, w1_ptr, g1_ptr, b1_ptr, m1_ptr, v1_ptr,
    w2_ptr, g2_ptr, b2_ptr, m2_ptr, v2_ptr, out_ptr,
    in_h, in_w, out_h, out_w, channels, eps1, eps2,
    IN_CHANNELS: tl.constexpr, BLOCK_K: tl.constexpr, K_EXACT: tl.constexpr,
    BLOCK_C: tl.constexpr, BLOCK_W: tl.constexpr, C_BLOCKS: tl.constexpr,
):
    """One ``[BLOCK_C, BLOCK_W]`` output row strip of the whole block per program.

    The block written over the output index, which is what makes a single kernel possible
    at all -- because ``cv2`` is depthwise, output channel ``c`` reads only channel ``c``
    of the activation, so a program owning a channel block never needs a cross-channel
    exchange::

        Y[n,c,ih,iw]   = SiLU(BN1(sum_k w1[c,k] * x[n,k,ih,iw]))  in bounds, else 0
        out[n,c,oh,ow] = BN2(sum_{i,j} w2[c,i,j] * Y[n,c, 2*oh-1+i, 2*ow-1+j])

    A strip of ``BLOCK_W`` output columns in row ``oh`` reads, for row tap ``i``, input
    columns ``2*ow0-1 .. 2*ow0+2*BLOCK_W`` of row ``2*oh-1+i``. Those are *contiguous*, so
    one load brings the whole run in at stride 1 and one ``tl.dot`` evaluates the
    activation for all of it. Splitting that register tile into its even and odd columns
    hands over two of the three column taps directly -- even columns are ``j=0``, odd are
    ``j=1`` -- and the third tap sits two columns right of the first, which is a second and
    narrower load.

    That is the formulation the measurement chose. Gathering the nine taps one at a time
    instead -- the same arithmetic with three dots per row tap and every load at stride 2 --
    profiles at 2.2 of every 32 bytes per sector and spills, and at its own best tile runs
    46.8 us against this kernel's 27.4 us on case A (``tests/compare_formulations.py``,
    ``profile/p1_fused_v1_baseline/REPORT.md``).

    Offsets are 32-bit element counts added to a 64-bit sample base: the sample stride is
    the only span that can be large, and a block of 64-bit addresses would cost more
    registers than the accumulator. The strides themselves are derived here rather than
    passed, because at this size every kernel argument is ~0.4 us of host time.
    """
    tile_w = tl.program_id(0)
    row_and_channels = tl.program_id(1)
    n = tl.program_id(2)
    oh = row_and_channels // C_BLOCKS
    tile_c = row_and_channels % C_BLOCKS

    in_hw = in_h * in_w
    positions = out_h * out_w
    ow0 = tile_w * BLOCK_W
    ow = ow0 + tl.arange(0, BLOCK_W)
    ow_ok = ow < out_w

    c = tile_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_ok = c < channels
    k = tl.arange(0, BLOCK_K)
    k_ok = k < IN_CHANNELS

    scale1, shift1 = _fold_bn(g1_ptr, b1_ptr, m1_ptr, v1_ptr, c, c_ok, eps1)
    scale2, shift2 = _fold_bn(g2_ptr, b2_ptr, m2_ptr, v2_ptr, c, c_ok, eps2)
    # A 1x1 filter already *is* the matrix tl.dot wants, so it is consumed as stored.
    w1 = tl.load(w1_ptr + (c[:, None] * IN_CHANNELS + k[None, :]),
                 mask=c_ok[:, None] & k_ok[None, :], other=0.0)

    x_base = x_ptr + n.to(tl.int64) * (IN_CHANNELS * in_hw)
    k_span = k[:, None] * in_hw
    left = 2 * ow0 - 1                 # leftmost column any tap of this strip reads
    run = tl.arange(0, 2 * BLOCK_W)
    step = tl.arange(0, BLOCK_W)

    acc = tl.zeros((BLOCK_C, BLOCK_W), dtype=tl.float32)
    for i in tl.static_range(3):
        ih = 2 * oh - 1 + i
        row_ok = (ih >= 0) & (ih < in_h)
        row_off = ih * in_w

        # Taps j=0 and j=1, from one contiguous run of 2*BLOCK_W columns.
        iw = left + run
        run_ok = row_ok & (iw >= 0) & (iw < in_w)
        # The 2-D mask is the expensive one -- BLOCK_K by 2*BLOCK_W lanes -- so the
        # reduction term is dropped when the block covers the channels exactly, which is
        # the case for every benched configuration.
        gather_ok = run_ok[None, :] if K_EXACT else (k_ok[:, None] & run_ok[None, :])
        run_y = _activate(
            tl.dot(w1, tl.load(x_base + k_span + (row_off + iw)[None, :],
                               mask=gather_ok, other=0.0)),
            scale1, shift1)
        # Masked *after* the activation: a padded tap contributes 0, while an in-bounds tap
        # whose pointwise sum is 0 contributes SiLU(shift1), which is not 0.
        run_y = tl.where(run_ok[None, :], run_y, 0.0)
        even_y, odd_y = tl.split(tl.reshape(run_y, (BLOCK_C, BLOCK_W, 2)))
        acc += tl.load(w2_ptr + c * 9 + (i * 3), mask=c_ok,
                       other=0.0).to(tl.float32)[:, None] * even_y
        acc += tl.load(w2_ptr + c * 9 + (i * 3 + 1), mask=c_ok,
                       other=0.0).to(tl.float32)[:, None] * odd_y

        # Tap j=2: two columns right of the j=0 tap, so it is the one value the run above
        # does not already hold. Stride 2 and only BLOCK_W wide.
        iw = left + 2 + 2 * step
        tap_ok = row_ok & (iw >= 0) & (iw < in_w) & ow_ok
        gather_ok = tap_ok[None, :] if K_EXACT else (k_ok[:, None] & tap_ok[None, :])
        tap_y = _activate(
            tl.dot(w1, tl.load(x_base + k_span + (row_off + iw)[None, :],
                               mask=gather_ok, other=0.0)),
            scale1, shift1)
        acc += tl.load(w2_ptr + c * 9 + (i * 3 + 2), mask=c_ok,
                       other=0.0).to(tl.float32)[:, None] * tl.where(
                           tap_ok[None, :], tap_y, 0.0)

    # The depthwise conv rounds its own output to fp16 before BN2 reads it.
    out = acc.to(tl.float16).to(tl.float32) * scale2[:, None] + shift2[:, None]
    tl.store(out_ptr + n.to(tl.int64) * (channels * positions)
             + (c[:, None] * positions + (oh * out_w + ow)[None, :]),
             out.to(out_ptr.dtype.element_ty),
             mask=c_ok[:, None] & ow_ok[None, :])


@triton.jit
def _pointwise_bn_act(
    x_ptr, w1_ptr, g1_ptr, b1_ptr, m1_ptr, v1_ptr, y_ptr,
    positions, channels, eps1,
    IN_CHANNELS: tl.constexpr, BLOCK_K: tl.constexpr, K_EXACT: tl.constexpr,
    BLOCK_C: tl.constexpr, BLOCK_P: tl.constexpr, C_BLOCKS: tl.constexpr,
):
    """First half of the split control: ``y = SiLU(BN1(1x1(x)))``, materialized.

    Every position is evaluated exactly once, which is the split route's whole advantage
    over the fused one. Spatial extent never enters: with unit stride and no padding the
    pointwise mix is a matmul over flattened positions, and the position axis is stride 1, so
    this half is the cheap one: 6.3 us for case A against the fused kernel's whole 27.4 us.
    The route's device cost is dominated by its *second* kernel, which measures 51.4 us
    (``tests/compare_formulations.py``); the extra launch and the intermediate's traffic are
    on top of that.
    """
    tile_p = tl.program_id(0)
    sample_and_channels = tl.program_id(1)
    n = sample_and_channels // C_BLOCKS
    tile_c = sample_and_channels % C_BLOCKS

    pos = tile_p * BLOCK_P + tl.arange(0, BLOCK_P)
    pos_ok = pos < positions
    c = tile_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_ok = c < channels
    k = tl.arange(0, BLOCK_K)
    k_ok = k < IN_CHANNELS

    scale1, shift1 = _fold_bn(g1_ptr, b1_ptr, m1_ptr, v1_ptr, c, c_ok, eps1)
    w1 = tl.load(w1_ptr + (c[:, None] * IN_CHANNELS + k[None, :]),
                 mask=c_ok[:, None] & k_ok[None, :], other=0.0)

    gather_ok = pos_ok[None, :] if K_EXACT else (k_ok[:, None] & pos_ok[None, :])
    xv = tl.load(x_ptr + n.to(tl.int64) * (IN_CHANNELS * positions)
                 + (k[:, None] * positions + pos[None, :]), mask=gather_ok, other=0.0)
    y = _activate(tl.dot(w1, xv), scale1, shift1)
    tl.store(y_ptr + n.to(tl.int64) * (channels * positions)
             + (c[:, None] * positions + pos[None, :]),
             y.to(y_ptr.dtype.element_ty), mask=c_ok[:, None] & pos_ok[None, :])


@triton.jit
def _depthwise_bn(
    y_ptr, w2_ptr, g2_ptr, b2_ptr, m2_ptr, v2_ptr, out_ptr,
    in_h, in_w, out_h, out_w, channels, eps2,
    BLOCK_C: tl.constexpr, BLOCK_P: tl.constexpr, C_BLOCKS: tl.constexpr,
):
    """Second half of the split control: ``out = BN2(depthwise 3x3 s2 pad 1 of y)``.

    No ``tl.dot``: depthwise means one channel per accumulator lane, so this is nine masked
    gathers and a running fp32 sum. The padding mask is the load mask here, which is
    correct in this route precisely because the activation has already been materialized --
    ``y`` really is zero outside the image.
    """
    tile_p = tl.program_id(0)
    sample_and_channels = tl.program_id(1)
    n = sample_and_channels // C_BLOCKS
    tile_c = sample_and_channels % C_BLOCKS

    in_hw = in_h * in_w
    positions = out_h * out_w
    pos = tile_p * BLOCK_P + tl.arange(0, BLOCK_P)
    pos_ok = pos < positions
    oh = pos // out_w
    ow = pos - oh * out_w
    c = tile_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_ok = c < channels

    scale2, shift2 = _fold_bn(g2_ptr, b2_ptr, m2_ptr, v2_ptr, c, c_ok, eps2)
    y_base = y_ptr + n.to(tl.int64) * (channels * in_hw)
    c_span = c[:, None] * in_hw

    acc = tl.zeros((BLOCK_C, BLOCK_P), dtype=tl.float32)
    for i in tl.static_range(3):
        ih = 2 * oh - 1 + i
        row_ok = (ih >= 0) & (ih < in_h) & pos_ok
        for j in tl.static_range(3):
            iw = 2 * ow - 1 + j
            tap_ok = row_ok & (iw >= 0) & (iw < in_w)
            yv = tl.load(y_base + c_span + (ih * in_w + iw)[None, :],
                         mask=c_ok[:, None] & tap_ok[None, :], other=0.0)
            acc += tl.load(w2_ptr + c * 9 + (i * 3 + j), mask=c_ok,
                           other=0.0).to(tl.float32)[:, None] * yv.to(tl.float32)

    out = acc.to(tl.float16).to(tl.float32) * scale2[:, None] + shift2[:, None]
    tl.store(out_ptr + n.to(tl.int64) * (channels * positions)
             + (c[:, None] * positions + pos[None, :]),
             out.to(out_ptr.dtype.element_ty),
             mask=c_ok[:, None] & pos_ok[None, :])


def _autopad(k: int) -> int:
    return k // 2


class _ConvBN(nn.Module):
    """``conv -> bn -> act``, holding the parameter names the baseline's block holds.

    ``load_state_dict(..., strict=False)`` is how the benchmark shares weights, and it
    silently leaves any key this module fails to name at this module's own random value.
    So the two-level ``cv{1,2}.conv.weight`` / ``cv{1,2}.bn.*`` surface is reproduced
    exactly, out of the same frozen L1 primitives the baseline's block composes, rather
    than flattened into something more convenient.

    Deliberately not importing the baseline's own ``YOLOConv`` wrapper: the benchmark
    imports candidates non-standalone, so that name resolves to whichever L2 candidate file
    happens to exist, whose internals -- BN folded at construction, renamed attributes --
    this module would then be reading.
    """

    def __init__(self, c1: int, c2: int, k: int, s: int, groups: int, act: bool):
        super().__init__()
        self.conv = Conv2d(c1, c2, k, s, _autopad(k), groups=groups, bias=False)
        self.bn = BatchNorm2d(c2, eps=1e-3, momentum=0.03)
        self.act = SiLU() if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class YOLOSCDown(nn.Module):
    def __init__(self, c1: int, c2: int, k: int, s: int):
        super().__init__()
        self.cv1 = _ConvBN(c1, c2, 1, 1, groups=1, act=True)
        self.cv2 = _ConvBN(c2, c2, k, s, groups=c2, act=False)

        self.c1 = c1
        self.c2 = c2
        self.k = k
        self.s = s
        # The weights are filled *after* construction, so the route may depend only on the
        # configuration above -- never on what a parameter holds. Checked by constructing
        # this class on the meta device, where there are no values to read.
        self.route = self._select_route()
        self._route_config = (c1, c2, k, s)
        self._pad = _autopad(k)
        # Everything the guard and the launch would otherwise recompute per call. The
        # reduction block is a power of two at least 16 because it is a ``tl.dot`` extent;
        # when it equals ``c1`` the kernels drop the reduction mask entirely.
        self._block_k = max(16, triton.next_power_of_2(c1))
        self._k_exact = self._block_k == c1
        self._cv1_expected = (c1, c2, (1, 1), (1, 1), (0, 0), (1, 1), 1)
        self._cv2_expected = (c2, c2, (k, k), (s, s), (self._pad, self._pad), (1, 1), c2)
        self._w1_shape = (c2, c1, 1, 1)
        self._w2_shape = (c2, 1, k, k)
        self._bn_shape = (c2,)
        # Observable by tests only, and plain Python attributes so nothing about the
        # process is patched: an "all green" run must not be able to mean the guard quietly
        # fell back on every case.
        self.last_route = _REFERENCE
        self.calls = 0

    # -- routing ----------------------------------------------------------
    def _select_route(self) -> int:
        # The kernels hard-code a 3x3 window at stride 2 with the padding autopad gives it,
        # because that is what makes the tap structure a compile-time constant and the
        # even/odd column split of the row run mean what it means.
        # The reduction block is a power of two at least 16, and above ``_MAX_BLOCK_K`` the
        # kernel does not compile at all, so a wide input channel count is the reference's.
        if (self.k == 3 and self.s == 2 and self.c1 >= 1 and self.c2 >= 1
                and max(16, triton.next_power_of_2(self.c1)) <= _MAX_BLOCK_K):
            return _OPTIMIZED_ROUTE
        return _REFERENCE

    def _reference(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv2(self.cv1(x))

    def _out_extent(self, in_h: int, in_w: int) -> tuple[int, int]:
        """``floor((extent + 2*pad - k)/s) + 1``, the same formula F.conv2d uses.

        Not ``extent // s``: those agree only for even extents, and an odd one is where
        they part -- a 39-row input gives 20 output rows, not 19.
        """
        slack = 2 * self._pad - self.k
        return (in_h + slack) // self.s + 1, (in_w + slack) // self.s + 1

    # -- guards -----------------------------------------------------------
    def _holder_is_as_built(self, holder: _ConvBN, expected, weight_shape,
                            act_type: type, device: torch.device) -> bool:
        """Whether one ``conv -> bn -> act`` holder still matches what it was built as.

        Every attribute here is one a caller can reassign after construction, and the
        reference reads all of them on every call, so the kernels' compile-time constants
        are only valid while they still hold. Parameter shapes are compared against the
        configuration rather than against each other, so a replacement parameter of the
        wrong extent cannot be indexed past its storage.
        """
        # Every one of these is a compile-time constant inside the kernels -- the tap
        # structure, the output extent, the reduction width. Reassigning any of them leaves
        # the kernel computing the geometry it was compiled for on data that no longer has
        # it, which is a wrong answer rather than an error.
        conv = holder.conv
        if (conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride,
                conv.padding, conv.dilation, conv.groups) != expected:
            return False
        # A bias would be added by the reference and by nothing here.
        if conv.bias is not None:
            return False
        # The kernels reach the filter as raw memory, so a replacement the reference would
        # accept can still be unreadable here: a differing dtype or device is a wrong
        # pointer, a differing shape indexes past the storage, and a non-contiguous or
        # lazily-negated view has values that are not the bytes at that pointer.
        weight = conv.weight
        if (weight.dtype is not torch.float16 or weight.device != device
                or weight.shape != weight_shape
                or not weight.is_contiguous() or not _reads_as_stored(weight)):
            return False
        bn = holder.bn
        # Training-mode BN normalizes by *batch* statistics, and so does BN with
        # ``track_running_stats`` off; both are a different function from the affine map
        # the kernels fold, so neither may reach them.
        if (bn.training or not bn.track_running_stats or not bn.affine
                or bn.num_features != self.c2):
            return False
        # Same reasoning for the four BN vectors, which the kernel folds by pointer. Two dtype
        # sets rather than one, because the reference does not treat them alike: an affine
        # pair the reference refuses -- bf16, or the two disagreeing -- must raise here too
        # rather than be served, which is what a single permissive set got wrong.
        bn_shape = self._bn_shape
        affine = bn.weight
        if affine is None or bn.bias is None or affine.dtype is not bn.bias.dtype:
            return False
        # The affine dtype selects which buffer dtypes the reference will accept; an affine
        # dtype outside the table is one it never accepts.
        buffer_dtypes = _BN_BUFFERS_FOR_AFFINE.get(affine.dtype)
        if buffer_dtypes is None:
            return False
        for t, allowed in ((affine, _FP16_OR_FP32), (bn.bias, _FP16_OR_FP32),
                           (bn.running_mean, buffer_dtypes),
                           (bn.running_var, buffer_dtypes)):
            if (t is None or t.dtype not in allowed or t.device != device
                    or t.shape != bn_shape or not t.is_contiguous()
                    or not _reads_as_stored(t)):
                return False
        # The activation is baked into the kernels, so a replaced one must not be skipped. The
        # other three types are checked for the same reason one step up: the kernel reproduces
        # what *these* classes compute, so a subclass or a stand-in that overrides ``forward``
        # is a different function. An instance-level ``forward`` is the cheap way to swap an
        # implementation without changing the type, and it is invisible to every other clause
        # here -- measured: assigning ``cv1.forward`` left the route fused and disagreed with
        # this module's own reference path by 0.216, with 13% of elements inside tolerance.
        act = holder.act
        if (type(holder) is not _ConvBN or type(conv) is not Conv2d
                or type(bn) is not BatchNorm2d or type(act) is not act_type):
            return False
        for child in (holder, conv, bn, act):
            if "forward" in vars(child):
                return False
        # A hook on any of these children is called by ``nn.Module.__call__`` on the way
        # through the reference path, and the kernel calls none of them -- measured: a
        # forward hook on ``cv1`` returning ``out + 1`` left the fused route active and
        # disagreed with the reference path by 0.138, which is the ``admits_child_hooks``
        # mutant in ``tests/check_correctness.py``. Hooks on *this* module still fire,
        # because ``__call__`` wraps ``forward`` either way; it is only the children the
        # kernel replaces.
        for child in (holder, conv, bn, act):
            if (child._forward_pre_hooks or child._forward_hooks
                    or child._backward_hooks or child._backward_pre_hooks):
                return False
        return True

    @staticmethod
    def _no_global_hooks() -> bool:
        """Whether any *process-wide* module hook is registered.

        Separate from the per-child scan above and just as load-bearing:
        ``register_module_forward_hook`` installs into a global registry that
        ``nn.Module.__call__`` consults for every module it enters, so the reference path
        applies it once per fused-away child while the kernel applies it never. Measured: a
        global forward hook returning ``out + 1`` on a route that admits it disagreed with the
        reference by 4.58.
        """
        return not (_GLOBAL_FORWARD_HOOKS or _GLOBAL_FORWARD_PRE_HOOKS
                    or _GLOBAL_BACKWARD_HOOKS or _GLOBAL_BACKWARD_PRE_HOOKS)

    def _offsets_fit(self, in_h: int, in_w: int, out_h: int, out_w: int) -> bool:
        """Does every element offset either kernel forms fit in 32 bits?

        Each term is the **largest index the expression forms, plus one** -- not the logical
        extent. That distinction is the whole point of this function: the pointwise filter is
        addressed as ``c * IN_CHANNELS + k`` with ``k`` padded to ``BLOCK_K``, so when
        ``BLOCK_K > c1`` the extent ``channels * c1`` is *smaller* than the last address formed,
        and a configuration whose real last offset sat 111 past int32 was admitted. Channel and
        reduction lanes are padded to their blocks because a masked lane still has its address
        computed; spatial terms carry their block's overrun for the same reason.

        Kept separate from ``_grid_fits`` so each term can be made binding on its own in a test.
        Several of these terms are in fact unreachable while the grid bound holds -- it caps the
        padded channel count at ``65535 * block_c`` -- but they are checked here rather than
        argued away, because that coupling is not a property either function states.

        The sample term is promoted to 64 bits inside the kernels and so is excluded; that
        promotion is asserted structurally by the test suite, because the input that would wrap
        it cannot be allocated.
        """
        c1, c2 = self.c1, self.c2
        in_hw = in_h * in_w
        positions = out_h * out_w
        block_k = self._block_k
        tile = _tile_for(c2)
        block_c, block_w = tile.block_c, tile.block_w
        channels = -(-c2 // block_c) * block_c      # padded channel lanes
        last_c = channels - 1
        last_k = block_k - 1
        last_ih = 2 * (out_h - 1) + 1               # tap overrun past the last output row
        last_iw = 2 * (out_w - 1) - 1 + 2 * block_w  # run overrun past the last output column
        last_pos = positions - 1 + _SPLIT_BLOCK_P

        spans = (
            last_c * c1 + last_k + 1,                                   # w1: c*IN_CHANNELS + k
            last_c * 9 + 8 + 1,                                         # w2: c*9 + tap
            channels,                                                   # the four BN vectors
            last_k * in_hw + last_ih * in_w + last_iw + 1,               # fused x gather
            last_c * positions + (out_h - 1) * out_w
            + (out_w - 1 + block_w) + 1,                                # fused output store
            last_k * in_hw + last_pos + 1,                              # split x gather
            last_c * in_hw + last_pos + 1,                              # split y store
            last_c * in_hw + last_ih * in_w + 2 * (out_w - 1) + 2 + 1,   # split y load
            last_c * positions + last_pos + 1,                          # split output store
        )
        return max(spans) <= _MAX_INT32_OFFSET

    def _grid_fits(self, batch: int, in_h: int, in_w: int, out_h: int, out_w: int) -> bool:
        """Are both launches' grids inside CUDA's per-axis limits?

        Not an arithmetic question and not covered by the offset bound. CUDA bounds the y and z
        axes at 65535; the fused launch puts ``out_h * c_blocks`` on y and the batch on z, and the
        split launch puts ``batch * c_blocks`` on y. A caller can exceed either with a shape the
        reference handles without complaint -- a batch of 65536, or a 131073-row input.
        """
        tile = _tile_for(self.c2)
        c_blocks = -(-self.c2 // tile.block_c)
        positions = out_h * out_w
        fused = (-(-out_w // tile.block_w), out_h * c_blocks, batch)
        split = (max(-(-(in_h * in_w) // _SPLIT_BLOCK_P),
                     -(-positions // _SPLIT_BLOCK_P)), batch * c_blocks)
        return (fused[0] <= _MAX_GRID_X and fused[1] <= _MAX_GRID_YZ
                and fused[2] <= _MAX_GRID_YZ
                and split[0] <= _MAX_GRID_X and split[1] <= _MAX_GRID_YZ)

    def _fused_route_applies(self, x: torch.Tensor) -> bool:
        """Whether the configuration-derived route is valid for *this* call.

        Anything not covered here reaches the reference path, so an input or a module state
        the kernels cannot serve gets the reference result -- including the error the
        reference would raise. Ordered cheapest-first: this runs inside a host budget the
        launch floor already half fills.
        """
        # The kernels compute in fp16 and store fp16, and tl.dot needs both operands in one
        # dtype; every other input dtype is the reference's business. A non-CUDA tensor has
        # no device pointer to launch against at all.
        if x.dtype is not torch.float16 or not x.is_cuda:
            return False
        # Forward-only: building a graph is left to the reference so backward() keeps
        # working, and under autocast the reference casts its operands and returns the
        # autocast dtype -- a pointer read sits below the dispatcher that implements that,
        # so it would return the input's dtype instead, or succeed where the reference
        # raises.
        # ``x.requires_grad`` is included even though the fused route was measured to agree
        # with the reference under ``no_grad`` on such an input (both return a tensor with
        # ``requires_grad`` false and the same values). It is a conservative clause matching
        # the contract's enumeration rather than a measured divergence, and it costs one
        # attribute read; the equivalence it forgoes is recorded in the tests.
        if (torch.is_grad_enabled() or x.requires_grad
                or torch.is_autocast_enabled("cuda") or self.training):
            return False
        # A process-wide module hook is applied by the reference to every child the kernel
        # fuses away.
        if not self._no_global_hooks():
            return False
        # The route was chosen for one configuration; any part of it may have been
        # reassigned since, and the kernels take k, s and the channel extents as
        # compile-time constants.
        if (self.c1, self.c2, self.k, self.s) != self._route_config:
            return False
        # The kernels read exactly c1 channels from a packed NCHW buffer. is_contiguous
        # with an explicit memory format rather than a stride(-1) test, because the kernels
        # derive the channel and sample spans from the shape, so a channels_last or
        # otherwise padded layout would be read as though it were packed. An empty input is
        # left to the reference: at least one grid dimension would be zero, which this
        # Triton treats as a silent no-op, so a zero spatial extent would return an empty
        # tensor where the reference raises.
        if (x.dim() != 4 or x.shape[1] != self.c1 or x.numel() == 0
                or not x.is_contiguous(memory_format=torch.contiguous_format)
                or not _reads_as_stored(x)):
            return False
        device = x.device
        return (self._holder_is_as_built(self.cv1, self._cv1_expected, self._w1_shape,
                                         SiLU, device)
                and self._holder_is_as_built(self.cv2, self._cv2_expected,
                                             self._w2_shape, nn.Identity, device))

    # -- optimized routes -------------------------------------------------
    def _launch_fused(self, x: torch.Tensor, out_h: int, out_w: int,
                      tile: Tile | None = None) -> torch.Tensor:
        n, _, in_h, in_w = x.shape
        c1, c2 = self.c1, self.c2
        cv1, cv2 = self.cv1, self.cv2
        bn1, bn2 = cv1.bn, cv2.bn
        if tile is None:
            tile = _tile_for(c2)
        c_blocks = -(-c2 // tile.block_c)
        out = torch.empty((n, c2, out_h, out_w), dtype=torch.float16, device=x.device)
        _scdown_fused[(-(-out_w // tile.block_w), out_h * c_blocks, n)](
            x, cv1.conv.weight, bn1.weight, bn1.bias, bn1.running_mean, bn1.running_var,
            cv2.conv.weight, bn2.weight, bn2.bias, bn2.running_mean, bn2.running_var, out,
            in_h, in_w, out_h, out_w, c2, bn1.eps, bn2.eps,
            IN_CHANNELS=c1, BLOCK_K=self._block_k, K_EXACT=self._k_exact,
            BLOCK_C=tile.block_c, BLOCK_W=tile.block_w, C_BLOCKS=c_blocks,
            num_warps=tile.num_warps, num_stages=tile.num_stages,
        )
        return out

    def _launch_split(self, x: torch.Tensor, out_h: int, out_w: int,
                      tile: Tile | None = None) -> torch.Tensor:
        n, _, in_h, in_w = x.shape
        c1, c2 = self.c1, self.c2
        cv1, cv2 = self.cv1, self.cv2
        bn1, bn2 = cv1.bn, cv2.bn
        w1, w2 = cv1.conv.weight, cv2.conv.weight
        if tile is None:
            tile = _tile_for(c2)
        block_c = tile.block_c
        c_blocks = -(-c2 // block_c)
        out = torch.empty((n, c2, out_h, out_w), dtype=torch.float16, device=x.device)
        in_positions = in_h * in_w
        y = torch.empty((n, c2, in_h, in_w), dtype=torch.float16, device=x.device)
        block_p = 64
        _pointwise_bn_act[(-(-in_positions // block_p), n * c_blocks)](
            x, w1, bn1.weight, bn1.bias, bn1.running_mean, bn1.running_var, y,
            in_positions, c2, bn1.eps,
            IN_CHANNELS=c1, BLOCK_K=self._block_k, K_EXACT=self._k_exact,
            BLOCK_C=block_c, BLOCK_P=block_p, C_BLOCKS=c_blocks,
            num_warps=tile.num_warps, num_stages=tile.num_stages,
        )
        _depthwise_bn[(-(-(out_h * out_w) // block_p), n * c_blocks)](
            y, w2, bn2.weight, bn2.bias, bn2.running_mean, bn2.running_var, out,
            in_h, in_w, out_h, out_w, c2, bn2.eps,
            BLOCK_C=block_c, BLOCK_P=block_p, C_BLOCKS=c_blocks,
            num_warps=tile.num_warps, num_stages=tile.num_stages,
        )
        return out

    def forward_via(self, x: torch.Tensor, route: int,
                    tile: Tile | None = None) -> torch.Tensor:
        """``forward`` with the optimized route and geometry overridden.

        Only the timing driver and the tests use this -- it is how the split control was
        measured against the fused kernel while both were correct, and how the tile rule's
        constants were chosen. ``forward`` dispatches on its own route tag so the scored
        path carries no extra indirection.
        """
        if route != _REFERENCE and self._fused_route_applies(x):
            in_h, in_w = x.shape[2], x.shape[3]
            out_h, out_w = self._out_extent(in_h, in_w)
            if (self._offsets_fit(in_h, in_w, out_h, out_w)
                    and self._grid_fits(x.shape[0], in_h, in_w, out_h, out_w)):
                self.last_route = route
                self.calls += 1
                launch = self._launch_fused if route == _FUSED else self._launch_split
                return launch(x, out_h, out_w, tile)
        self.last_route = _REFERENCE
        self.calls += 1
        return self._reference(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The route tag is read once and compared once: an integer test that selects the
        # launcher, with no dict lookup or isinstance on the scored path. The guard below
        # answers a different question -- whether that route is valid for this call -- and
        # deliberately does not re-examine the tag.
        route = self.route
        if route != _REFERENCE and self._fused_route_applies(x):
            in_h, in_w = x.shape[2], x.shape[3]
            out_h, out_w = self._out_extent(in_h, in_w)
            if (self._offsets_fit(in_h, in_w, out_h, out_w)
                    and self._grid_fits(x.shape[0], in_h, in_w, out_h, out_w)):
                self.last_route = route
                self.calls += 1
                if route == _FUSED:
                    return self._launch_fused(x, out_h, out_w)
                return self._launch_split(x, out_h, out_w)
        self.last_route = _REFERENCE
        self.calls += 1
        return self._reference(x)
