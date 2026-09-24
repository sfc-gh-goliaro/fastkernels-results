"""Conv-BN-Act with the BatchNorm affine transform and the activation fused into
the convolution's epilogue.

The captured workload is launch-bound, not compute- or bandwidth-bound. In the
harness's own timing loop an ``nn.Identity`` module measures 7.2 us -- that is the
shifting-pool input copy plus event overhead, an irreducible floor -- and one extra
kernel launch costs about 4 us. The baseline issues five kernels on a 1x1 case
(matmul, a device-to-device copy, an elementwise op, ``batch_norm_transform_input``,
SiLU) and seven on the small 3x3 case, where cuDNN spends more time in
``nchwToNhwcKernel``/``nhwcToNchwKernel`` than in the convolution itself. So the
lever is the number of kernels launched, not FLOPs.

Every route therefore collapses launches by evaluating the BatchNorm affine
transform and the activation where the convolution's accumulator already lives,
in fp32, rounding once on the store (KernelWiki ``technique-epilogue-fusion``,
``wiki/techniques/epilogue-fusion.md``: fusing the post-MMA scale/bias/activation
"avoids a separate kernel launch and an extra global memory round-trip"). Triton is
the tool rather than CuTe DSL because these shapes are nowhere near compute-bound
and Triton 3.6 is the first release with native SM100 ``tcgen05``/TMEM lowering for
``tl.dot`` (KernelWiki ``wiki/languages/triton-blackwell.md`` and
``sources/docs/triton-3.6-blackwell.md``, which reserve CuTe-DSL/CUTLASS for
"peak-performance compute-bound matmul" and name Triton's lane as prototyping and
memory-bound work).

Four routes, tagged once at construction time from constructor arguments alone:

* ``_POINTWISE`` -- ``1x1``, unit stride, no padding, no dilation, one group: the
  convolution *is* a batched GEMM and NCHW is already the right layout, so one
  Triton kernel does the GEMM, the affine transform and the activation.
* ``_IMPLICIT_GEMM`` -- an explicitly enumerated set of ``k>1, g=1``
  configurations, on an NCHW implicit-GEMM kernel carrying the same epilogue.
* ``_POST_CONV`` -- everything else: the frozen L1 ``Conv2d`` (itself fast for many
  configurations) followed by one standalone epilogue kernel. Two launches against
  the baseline's three to seven, for depthwise, grouped and dilated configurations
  and for any ``k>1`` shape no fused tile was measured for.
* ``_REFERENCE`` -- literally ``self.act(self.bn(self.conv(x)))``, what the guard
  falls back to.

The route *tag* never depends on weight values, which is what keeps it stable
across the harness's ``load_state_dict`` and across in-place weight edits without a
cache keyed on weight identity. Taking that route on a given call is a separate
question, answered by ``_fast_plan``: everything it checks is there because
this module must behave exactly like ``baseline.py``, which re-reads its attributes
and its children on every call.

No derived state is cached anywhere -- not the folded weight, not ``scale``/``shift``,
not a reference to a child module. Three separate reasons, all of them measured
rather than assumed:

* the harness rewrites parameters after construction (``p.data = p.data.to(fp16)``
  replaces the tensor object, then ``load_state_dict`` copies in place), so anything
  derived at ``__init__`` time is stale before the first forward;
* ``YOLORepVGGDW.fuse()`` calls ``YOLOConv.fuse()`` on two children and *then*
  writes ``conv.conv.weight.data`` and ``conv.conv.bias.data`` in place, so anything
  derived at ``fuse()`` time is stale too;
* a registered buffer for derived state would add a ``state_dict`` key, and the
  harness loads with ``strict=False``, so that key would silently keep
  candidate-local values instead of the shared ones. Assigning a child module to a
  helper attribute is worse: it duplicates every BatchNorm key under the helper
  name and those keys survive ``delattr(self, "bn")``, so ``fuse()`` would leave
  BatchNorm state in the ``state_dict`` (measured: eight keys instead of four, four
  of them surviving the ``delattr``).

Instead the kernels take ``gamma, beta, mean, var, eps`` as arguments and compute
``scale = gamma / sqrt(var + eps)`` and ``shift = beta - mean * scale`` in fp32, a
handful of scalar loads and flops per program. The activation is recognized by
exact type, so a reassigned ``self.act`` is handled by value semantics rather than
by an identity check against a stored reference.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.autograd.forward_ad as _forward_ad
import torch.nn.modules.module as _module_hooks
import triton
import triton.language as tl

from ..L1.batch_norm2d import BatchNorm2d
from ..L1.conv2d import Conv2d
from ..L1.silu import SiLU

# Route tags. Small ints so the per-call dispatch is an integer compare.
_REFERENCE = 0
_POST_CONV = 1
_POINTWISE = 2
_IMPLICIT_GEMM = 3

# What the epilogue owes the convolution's accumulator before the activation. A
# fused convolution that computed its own bias needs neither, which the two-kernel
# route expresses by passing ``apply_bn=False`` rather than by a third tag.
_EPI_BN = 0      # unfused: apply the BatchNorm affine transform
_EPI_BIAS = 1    # fused: add the fused bias

# Activation folded into the epilogue.
_ACT_NONE = 0
_ACT_SILU = 1
# A module the epilogue does not recognize: the kernel runs with ``_ACT_NONE`` and
# the activation is then called as a module, so its own hooks still fire.
_ACT_EXTERNAL = 2

# Largest element offset any kernel here forms in 32-bit arithmetic. The batch term
# is promoted to 64 bits; anything that could exceed this takes the reference path
# rather than silently wrapping.
_MAX_INT32_OFFSET = 2 ** 31 - 1

# CUDA's limit on the second and third grid dimensions. A launch past it is a
# hard error rather than a slow kernel, and the reference has to answer instead.
_MAX_GRID_YZ = 65535

# Dtypes the kernels accept for the data. fp32 is deliberately absent: at fp32's
# atol=1e-5 the reference's own choice of convolution algorithm becomes observable,
# so agreeing with cuDNN rather than being accurate would be the requirement.
_FAST_DTYPES = (torch.float16, torch.bfloat16)
# Dtypes the per-channel BatchNorm operands may have. The harness casts parameters
# only, so ``bn.weight``/``bn.bias`` arrive fp16 while ``running_mean``/``running_var``
# stay fp32; the kernels promote everything to fp32 rather than assume one dtype.
_PARAM_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


class PointwiseTile(NamedTuple):
    """How one ``1x1`` configuration is mapped onto ``_pointwise_bn_act``."""

    block_p: int         # output pixels per program
    block_oc: int        # output channels per program
    block_k: int         # slice of the channel reduction held at once
    num_warps: int
    num_stages: int


class GemmTile(NamedTuple):
    """How one ``k>1`` configuration is mapped onto ``_implicit_gemm_bn_act``."""

    block_oh: int        # output rows per program
    block_ow: int        # output columns per program (a power of two)
    block_oc: int        # output channels per program
    block_k: int         # slice of the flat C*KH*KW reduction held at once
    num_warps: int
    num_stages: int


# Tile shapes for the ``1x1`` configurations, keyed by ``(in_channels,
# out_channels)`` alone so the table cannot grow at run time. Every entry is the
# winner of the offline sweep in ``scratch/sweep_tiles2.py``, which scores a tile by
# the geometric mean of its latency across *all* the input shapes that configuration
# is captured with -- a tile frozen against one shape is not a tile keyed on the
# configuration -- and only reports it when it beats the two-kernel route on every
# one of those shapes. The comments record the measured speedup against the baseline
# per captured shape, hottest first.
#
# ``block_k=256`` dominates almost everywhere, which is the one result worth stating
# plainly: the reduction is only ``C`` long here (32 to 512), so a large ``block_k``
# collapses it to a single chunk and the loop disappears. The earlier sweep capped
# ``block_k`` at 64 and concluded that six of these twelve configurations should stay
# on the two-kernel route; all twelve win once the cap is lifted.
_POINTWISE_TILES: dict[tuple[int, int], PointwiseTile] = {
    (32, 32): PointwiseTile(128, 32, 64, 4, 1),    # 2.88x / 3.18x
    (48, 32): PointwiseTile(32, 32, 256, 4, 1),    # 2.65x / 3.51x
    (64, 64): PointwiseTile(128, 32, 256, 8, 1),   # 2.23x / 2.47x
    (96, 64): PointwiseTile(32, 32, 128, 4, 1),    # 1.84x / 2.25x
    (128, 64): PointwiseTile(128, 32, 256, 4, 1),  # 2.20x / 2.18x
    (128, 256): PointwiseTile(64, 32, 256, 8, 1),  # 2.28x / 1.93x
    (192, 64): PointwiseTile(64, 64, 256, 8, 1),   # 1.83x / 2.08x
    (192, 128): PointwiseTile(64, 32, 128, 4, 1),  # 2.24x / 1.93x
    (256, 128): PointwiseTile(64, 32, 256, 4, 1),  # 1.96x / 1.96x / 1.94x / 1.93x
    (384, 128): PointwiseTile(64, 32, 256, 4, 1),  # 1.43x / 1.76x
    (384, 256): PointwiseTile(64, 32, 256, 4, 1),  # 1.67x / 1.71x
    (512, 256): PointwiseTile(32, 64, 256, 8, 1),  # 1.80x / 1.67x
}

# A ``1x1`` configuration the sweep never saw. Every captured one is in the table
# above, and the twelve entries agree closely enough on shape -- ``block_oc=32`` or 64,
# ``block_k`` covering the whole channel reduction -- that this generalizes rather than
# extrapolates, so an unseen pointwise configuration gets the fused kernel too instead
# of paying an extra launch. It is a starting geometry, not a measured winner, which is
# why it is written separately from the table.
#
# This is deliberately asymmetric with ``_GEMM_TILES``, which admits *only* enumerated
# configurations. The asymmetry is the plan's: the pointwise route is specified as a
# family (``k=1, s=1, p=0, d=1, g=1``) because with ``k=1`` the convolution simply *is* a
# GEMM and the kernel has no geometry-specific reasoning in it, whereas the
# implicit-GEMM route is specified as an enumerated set so that its much larger surface
# -- padding masks, stride, tap decoding -- stays narrow. Set this to ``None`` to make
# unmeasured pointwise configurations take the two-kernel route instead.
_POINTWISE_DEFAULT: PointwiseTile | None = PointwiseTile(64, 32, 256, 4, 1)

# Tile shapes for the enumerated ``k>1, g=1`` configurations, keyed by the full
# convolution geometry, so this route is admitted only where it was measured rather
# than for a generic ``k>1`` family. Winners of the same offline sweep, then
# confirmed against the two-kernel route by the randomized interleaved paired
# comparison in ``scratch/ab_compare.py`` -- absolute latencies on this box drift far
# enough between runs (the same shape read 38.9 us and 104.5 us minutes apart) that a
# ship decision taken from two medians measured at different times is not a decision
# about the kernel.
_GEMM_TILES: dict[tuple, GemmTile] = {
    (3, 16, 3, 3, 2, 2, 1, 1): GemmTile(4, 32, 32, 16, 4, 1),
    (16, 16, 3, 3, 1, 1, 1, 1): GemmTile(4, 32, 32, 256, 4, 1),
    # The weakest entry in either table, and the only one whose margin is worth
    # stating: a balanced ABBA ablation over four separate bench invocations moved the
    # scored case from 2.51x to 2.61x, winning 4 of 4 blocks with a median paired
    # latency of -2.10 us (56.3 -> 53.4 us). That clears one event-timer quantum
    # (~2.05 us) but not 5% of the two-kernel latency. A proportional threshold is the
    # wrong yardstick here: collapsing two launches into one is worth a fixed ~4 us, so
    # on the one scored case whose convolution genuinely costs ~43 us of GPU time the
    # same absolute saving is a small percentage. It never loses, so it ships -- but
    # re-tuning this tile for a decisive margin is the first thing to revisit.
    (16, 32, 3, 3, 2, 2, 1, 1): GemmTile(2, 32, 32, 16, 4, 1),
    (32, 32, 3, 3, 1, 1, 1, 1): GemmTile(8, 16, 32, 64, 4, 1),
    (64, 64, 3, 3, 1, 1, 1, 1): GemmTile(2, 8, 64, 256, 8, 2),
}
# ``128 -> 128 k3 s1 p1`` is deliberately absent even though the sweep found a tile for
# it: its only captured input is non-contiguous, so the route would be demoted to the
# two-kernel route on every call the captures contain and the tile would never actually
# run. Shipping it would be shipping an unmeasured launch geometry.


def autopad(k: int | tuple[int, int], p=None, d: int = 1):
    if isinstance(k, tuple):
        if d > 1:
            k = tuple(d * (x - 1) + 1 for x in k)
        if p is None:
            return tuple(x // 2 for x in k)
        return p
    if d > 1:
        k = d * (k - 1) + 1
    return k // 2 if p is None else p


def _fuse_conv_bn(conv: Conv2d, bn: BatchNorm2d) -> tuple[torch.Tensor, torch.Tensor]:
    w_conv = conv.weight.clone().view(conv.weight.shape[0], -1)
    w_bn = torch.diag(
        bn.weight.to(dtype=conv.weight.dtype).div(
            torch.sqrt(bn.eps + bn.running_var.to(dtype=conv.weight.dtype))
        )
    )
    fused_weight = torch.mm(w_bn, w_conv).view_as(conv.weight)

    conv_bias = conv.bias
    if conv_bias is None:
        conv_bias = torch.zeros(conv.weight.shape[0], device=conv.weight.device, dtype=conv.weight.dtype)
    b_bn = (
        bn.bias.to(dtype=conv.weight.dtype)
        - bn.weight.to(dtype=conv.weight.dtype)
        .mul(bn.running_mean.to(dtype=conv.weight.dtype))
        .div(torch.sqrt(bn.running_var.to(dtype=conv.weight.dtype) + bn.eps))
    )
    fused_bias = torch.mm(w_bn, conv_bias.reshape(-1, 1)).reshape(-1) + b_bn
    return fused_weight, fused_bias


def _pair(value: int | tuple[int, int]) -> tuple[int, int]:
    return (value, value) if isinstance(value, int) else tuple(value)


# ``neg`` carries a lazy flag that ATen applies when it reads a tensor, so the values
# a PyTorch op sees are not the values in memory; a kernel that dereferences the
# pointer never sees it (the frozen L1 winner measured a sign-flipped result from a
# negative-bit input), so such a view has to reach the reference path. Every guard
# below therefore checks ``is_neg()``.
#
# Its companion ``conj`` flag is deliberately *not* checked, which keeps two calls per
# guarded tensor out of a path whose whole budget is a few microseconds. The bit only
# exists for complex dtypes and every guard establishes a real dtype first, and the
# three ways a real-dtyped view could plausibly inherit it were all checked directly:
# ``torch._conj`` on an fp16 CUDA tensor raises an internal assert; ``real.conj()``
# returns the tensor itself with the bit clear; ``view_as_real`` refuses an unresolved
# conjugated tensor outright ("view_as_real doesn't work on unresolved conjugated
# tensors"); and ``.real``/``.imag`` of a conjugated complex tensor come back with
# ``is_conj()`` false -- ``.imag`` instead carries ``is_neg()``, and both are strided by
# two, so the ``is_neg()`` and contiguity checks already reject them.


def _reduction_blocks(reduction: int, requested: int) -> tuple[int, int]:
    """Reduction block size (a power of two, at least 16 for the MMA shape) and the
    number of chunks needed to cover *reduction*."""
    block = min(max(16, triton.next_power_of_2(requested)),
                max(16, triton.next_power_of_2(reduction)))
    return block, triton.cdiv(reduction, block)


@triton.jit
def _bn_act_epilogue(acc, gamma, beta, mean, var, bias, eps,
                     HAS_BN: tl.constexpr, HAS_BIAS: tl.constexpr,
                     SILU: tl.constexpr):
    """The affine transform and the activation, in fp32, shared by every kernel.

    The caller supplies the per-channel operands already loaded in whatever shape
    broadcasts against its accumulator, because the channel indexing differs
    between a GEMM tile and an elementwise block. The arithmetic -- the part that
    has to be identical everywhere -- lives only here. No rounding happens in this
    function: the caller rounds once, on its store.

    ``HAS_BN`` and ``HAS_BIAS`` are mutually exclusive and both false when the
    convolution already applied its fused bias. They are booleans rather than one
    enum because a Triton kernel may not read a plain module-level int.
    """
    if HAS_BN:
        scale = gamma / tl.sqrt(var + eps)
        y = acc * scale + (beta - mean * scale)
    elif HAS_BIAS:
        y = acc + bias
    else:
        y = acc
    if SILU:
        y = y * tl.sigmoid(y)
    return y


@triton.jit
def _bn_act(y_ptr, gamma_ptr, beta_ptr, mean_ptr, var_ptr, bias_ptr,
            plane, eps,
            CHANNELS: tl.constexpr, BLOCK_P: tl.constexpr,
            HAS_BN: tl.constexpr, HAS_BIAS: tl.constexpr, SILU: tl.constexpr):
    """The epilogue alone, in place over an already-computed convolution result.

    One program owns ``BLOCK_P`` pixels of one ``(sample, channel)`` plane, so the
    channel index is a scalar rather than something decoded per element: an
    integer division per element would be the most expensive arithmetic in a pass
    this shallow. In place because the convolution's output is freshly allocated
    and read exactly once, which saves an allocation and a second traversal.
    """
    tile_p = tl.program_id(0)
    sample_chan = tl.program_id(1)
    chan = sample_chan % CHANNELS
    sample = sample_chan // CHANNELS

    p = tile_p * BLOCK_P + tl.arange(0, BLOCK_P)
    ok = p < plane
    # The sample stride follows from the layout the guard already established, so
    # it does not have to be passed: every argument costs CPU time at launch.
    base = y_ptr + sample.to(tl.int64) * (CHANNELS * plane) + chan * plane
    acc = tl.load(base + p, mask=ok, other=0.0).to(tl.float32)

    # A one-element vector, so the same epilogue body serves this kernel and the
    # GEMM kernels without a second code path.
    ci = chan + tl.arange(0, 1)
    if HAS_BN:
        y = _bn_act_epilogue(acc,
                             tl.load(gamma_ptr + ci).to(tl.float32),
                             tl.load(beta_ptr + ci).to(tl.float32),
                             tl.load(mean_ptr + ci).to(tl.float32),
                             tl.load(var_ptr + ci).to(tl.float32),
                             acc, eps, HAS_BN, HAS_BIAS, SILU)
    elif HAS_BIAS:
        y = _bn_act_epilogue(acc, acc, acc, acc, acc,
                             tl.load(bias_ptr + ci).to(tl.float32),
                             eps, HAS_BN, HAS_BIAS, SILU)
    else:
        y = _bn_act_epilogue(acc, acc, acc, acc, acc, acc, eps,
                             HAS_BN, HAS_BIAS, SILU)

    tl.store(base + p, y.to(y_ptr.dtype.element_ty), mask=ok)


@triton.jit
def _pointwise_bn_act(x_ptr, w_ptr, gamma_ptr, beta_ptr, mean_ptr, var_ptr,
                      bias_ptr, y_ptr,
                      pixels, eps,
                      CHANNELS: tl.constexpr, OUT_CHANNELS: tl.constexpr,
                      BLOCK_P: tl.constexpr, BLOCK_OC: tl.constexpr,
                      BLOCK_K: tl.constexpr, K_CHUNKS: tl.constexpr,
                      HAS_BN: tl.constexpr, HAS_BIAS: tl.constexpr,
                      SILU: tl.constexpr):
    """``y[n,oc,p] = act(bn(sum_c w[oc,c] * x[n,c,p]))`` in one kernel.

    With ``k=1`` the convolution *is* a GEMM and NCHW is already the right layout:
    for each sample, ``y[n] = W @ x[n]`` with ``W`` a free ``[OC, C]`` view of the
    weight and ``x[n]`` a ``[C, P]`` block whose ``P = H*W`` axis is contiguous. No
    transpose, no im2col, no weight repack -- which is why this beats routing the
    same shape through a matmul plus a separate epilogue launch.

    The accumulator is channel-major so its fastest axis is the pixel, the
    innermost NCHW axis for both the gather and the store.
    """
    tile_p = tl.program_id(0)
    tile_oc = tl.program_id(1)
    sample = tl.program_id(2)

    p = tile_p * BLOCK_P + tl.arange(0, BLOCK_P)
    p_ok = p < pixels
    oc = tile_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_ok = oc < OUT_CHANNELS

    # Sample strides follow from the contiguous NCHW layout the guard established.
    x_base = x_ptr + sample.to(tl.int64) * (CHANNELS * pixels)
    acc = tl.zeros((BLOCK_OC, BLOCK_P), dtype=tl.float32)
    for chunk in tl.static_range(K_CHUNKS):
        k = chunk * BLOCK_K + tl.arange(0, BLOCK_K)
        k_ok = k < CHANNELS
        # Contiguous along the reduction: the weight of a 1x1 filter is packed
        # ``[OC, C]``, so ``oc * C + k`` is exactly the offset inside it.
        wv = tl.load(w_ptr + (oc[:, None] * CHANNELS + k[None, :]),
                     mask=oc_ok[:, None] & k_ok[None, :], other=0.0)
        xv = tl.load(x_base + (k[:, None] * pixels + p[None, :]),
                     mask=k_ok[:, None] & p_ok[None, :], other=0.0)
        acc = tl.dot(wv, xv, acc, input_precision="tf32")

    ci = oc[:, None]
    ci_ok = oc_ok[:, None]
    if HAS_BN:
        y = _bn_act_epilogue(acc,
                             tl.load(gamma_ptr + ci, mask=ci_ok, other=0.0).to(tl.float32),
                             tl.load(beta_ptr + ci, mask=ci_ok, other=0.0).to(tl.float32),
                             tl.load(mean_ptr + ci, mask=ci_ok, other=0.0).to(tl.float32),
                             tl.load(var_ptr + ci, mask=ci_ok, other=1.0).to(tl.float32),
                             acc, eps, HAS_BN, HAS_BIAS, SILU)
    elif HAS_BIAS:
        y = _bn_act_epilogue(acc, acc, acc, acc, acc,
                             tl.load(bias_ptr + ci, mask=ci_ok, other=0.0).to(tl.float32),
                             eps, HAS_BN, HAS_BIAS, SILU)
    else:
        y = _bn_act_epilogue(acc, acc, acc, acc, acc, acc, eps,
                             HAS_BN, HAS_BIAS, SILU)

    y_base = y_ptr + sample.to(tl.int64) * (OUT_CHANNELS * pixels)
    tl.store(y_base + (oc[:, None] * pixels + p[None, :]),
             y.to(y_ptr.dtype.element_ty),
             mask=oc_ok[:, None] & p_ok[None, :])


@triton.jit
def _implicit_gemm_bn_act(x_ptr, w_ptr, gamma_ptr, beta_ptr, mean_ptr, var_ptr,
                          bias_ptr, y_ptr,
                          in_h, in_w, out_h, out_w, eps,
                          CHANNELS: tl.constexpr, OUT_CHANNELS: tl.constexpr,
                          KH: tl.constexpr, KW: tl.constexpr,
                          SH: tl.constexpr, SW: tl.constexpr,
                          PH: tl.constexpr, PW: tl.constexpr,
                          BLOCK_OH: tl.constexpr, BLOCK_OW: tl.constexpr,
                          BLOCK_OC: tl.constexpr, BLOCK_K: tl.constexpr,
                          K_CHUNKS: tl.constexpr, REDUCTION: tl.constexpr,
                          OC_BLOCKS: tl.constexpr,
                          HAS_BN: tl.constexpr, HAS_BIAS: tl.constexpr,
                          SILU: tl.constexpr):
    """One output tile of ``[BLOCK_OC, BLOCK_OH * BLOCK_OW]``, epilogue included.

    The convolution is the frozen L1 winner's shape: the reduction runs over the
    flattened ``C*KH*KW`` extent in power-of-two chunks with a masked tail,
    decoding ``(c, kh, kw)`` from the flat index. That form beat a tap-outer loop
    decisively on the large sliding-window configuration (31.7 us against 52.3 us)
    because a flat reduction index *is* the offset within a packed filter, so the
    weight is read contiguously along the reduction; the tap-outer form reads it
    strided by ``KH*KW``. What this kernel adds is the epilogue, which the L1
    winner cannot have because it does not know about the BatchNorm.

    Offsets are 32-bit element counts added to a scalar base pointer rather than
    blocks of 64-bit pointers: the address block is the largest register consumer
    in a gather kernel this shallow.
    """
    tile_ow = tl.program_id(0)
    tile_oh = tl.program_id(1)
    batch_oc = tl.program_id(2)
    sample = batch_oc // OC_BLOCKS
    tile_oc = batch_oc % OC_BLOCKS

    pos = tl.arange(0, BLOCK_OH * BLOCK_OW)
    oh = tile_oh * BLOCK_OH + pos // BLOCK_OW
    ow = tile_ow * BLOCK_OW + pos % BLOCK_OW
    in_tile = (oh < out_h) & (ow < out_w)

    oc = tile_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_ok = oc < OUT_CHANNELS

    ih0 = oh * SH - PH
    iw0 = ow * SW - PW
    # Every stride follows from the shapes: the guard establishes that the input is
    # contiguous NCHW and the output is this kernel's own fresh allocation. Passing
    # nine stride arguments instead would cost more CPU at launch than the whole
    # epilogue costs on the GPU.
    x_plane = in_h * in_w
    y_plane = out_h * out_w
    x_base = x_ptr + sample.to(tl.int64) * (CHANNELS * x_plane)
    y_base = y_ptr + sample.to(tl.int64) * (OUT_CHANNELS * y_plane)

    acc = tl.zeros((BLOCK_OC, BLOCK_OH * BLOCK_OW), dtype=tl.float32)
    for chunk in tl.static_range(K_CHUNKS):
        k = chunk * BLOCK_K + tl.arange(0, BLOCK_K)
        k_ok = k < REDUCTION
        c = k // (KH * KW)
        tap = k % (KH * KW)
        kh = tap // KW
        kw = tap % KW
        ih = ih0[None, :] + kh[:, None]
        iw = iw0[None, :] + kw[:, None]
        if PH == 0 and PW == 0:
            # Zero padding with an exact output extent: every tap of an in-range
            # output position is in range, so bounds follow from ``in_tile``.
            ok = k_ok[:, None] & in_tile[None, :]
        else:
            ok = (k_ok[:, None] & in_tile[None, :]
                  & (ih >= 0) & (ih < in_h) & (iw >= 0) & (iw < in_w))
        xv = tl.load(x_base + (c[:, None] * x_plane + ih * in_w + iw),
                     mask=ok, other=0.0)
        wv = tl.load(w_ptr + (oc[:, None] * REDUCTION + k[None, :]),
                     mask=oc_ok[:, None] & k_ok[None, :], other=0.0)
        acc = tl.dot(wv, xv, acc, input_precision="tf32")

    ci = oc[:, None]
    ci_ok = oc_ok[:, None]
    if HAS_BN:
        y = _bn_act_epilogue(acc,
                             tl.load(gamma_ptr + ci, mask=ci_ok, other=0.0).to(tl.float32),
                             tl.load(beta_ptr + ci, mask=ci_ok, other=0.0).to(tl.float32),
                             tl.load(mean_ptr + ci, mask=ci_ok, other=0.0).to(tl.float32),
                             tl.load(var_ptr + ci, mask=ci_ok, other=1.0).to(tl.float32),
                             acc, eps, HAS_BN, HAS_BIAS, SILU)
    elif HAS_BIAS:
        y = _bn_act_epilogue(acc, acc, acc, acc, acc,
                             tl.load(bias_ptr + ci, mask=ci_ok, other=0.0).to(tl.float32),
                             eps, HAS_BN, HAS_BIAS, SILU)
    else:
        y = _bn_act_epilogue(acc, acc, acc, acc, acc, acc, eps,
                             HAS_BN, HAS_BIAS, SILU)

    y_off = oc[:, None] * y_plane + (oh[None, :] * out_w + ow[None, :])
    tl.store(y_base + y_off, y.to(y_ptr.dtype.element_ty),
             mask=oc_ok[:, None] & in_tile[None, :])


class YOLOConv(nn.Module):
    default_act = SiLU()

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 1,
        s: int = 1,
        p=None,
        g: int = 1,
        d: int = 1,
        act=True,
    ):
        super().__init__()
        self.conv = Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = BatchNorm2d(c2, eps=1e-3, momentum=0.03)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self._is_fused = False

        # Routing reads the convolution's normalized configuration rather than the
        # raw arguments, so a tuple ``k`` and an int ``k`` land in the same place.
        conv = self.conv
        self._c1 = c1
        self._c2 = c2
        # The configuration the route was chosen for. Every one of these is a plain
        # attribute a caller can reassign afterwards and the reference reads all of
        # them on every call, so the route is only valid while they still hold.
        self._route_config = (conv.in_channels, conv.out_channels, conv.kernel_size,
                              conv.stride, conv.padding, conv.dilation, conv.groups)
        # Derived from the arguments, not read off the parameter, which holds
        # uninitialized memory at this point.
        self._weight_shape = torch.Size((c2, c1 // g) + conv.kernel_size)
        self._bias_shape = torch.Size((c2,))
        self.route, self.tile = self._select_route()

    # -- routing ----------------------------------------------------------
    def _select_route(self):
        conv = self.conv
        kh, kw = conv.kernel_size
        sh, sw = conv.stride
        ph, pw = conv.padding
        if conv.groups == 1 and conv.dilation == (1, 1):
            if (kh, kw) == (1, 1) and (sh, sw) == (1, 1) and (ph, pw) == (0, 0):
                tile = _POINTWISE_TILES.get((conv.in_channels, conv.out_channels),
                                            _POINTWISE_DEFAULT)
                if tile is not None:
                    return _POINTWISE, tile
            else:
                tile = _GEMM_TILES.get((conv.in_channels, conv.out_channels,
                                        kh, kw, sh, sw, ph, pw))
                if tile is not None:
                    return _IMPLICIT_GEMM, tile
        # Two launches instead of the baseline's three or more, for every
        # configuration no fused tile was measured for.
        return _POST_CONV, None

    # -- the reference expression -----------------------------------------
    def _reference(self, x: torch.Tensor) -> torch.Tensor:
        """Exactly what ``baseline.py`` computes, including the error it raises."""
        if self._is_fused:
            return self.act(self.conv(x))
        return self.act(self.bn(self.conv(x)))

    # -- the per-call guard ------------------------------------------------
    def _fast_plan(self, x: torch.Tensor):
        """``(mode, act_kind, bn)`` if a fused route may serve this call, else ``None``.

        One function rather than two, and written as a flat sequence of attribute
        reads, because the whole fast path's CPU budget is about 20 us: the harness's
        L2 flush hides roughly 30 us of enqueue latency per iteration and past that
        the measured window is CPU-starved, so a guard that costs 6 us instead of 3
        is directly visible in the score. Every condition is here because the
        baseline reads that piece of state on every call; anything not covered
        reaches ``_reference``, which is the baseline expression and raises whatever
        the baseline raises.

        What is deliberately *not* checked, because ``BatchNorm2d.forward`` and
        ``Conv2d.forward`` demonstrably do not read it: ``bn.affine`` (with it false the
        parameters are ``None``, which the loop below rejects anyway), ``bn.num_features``
        (the shape checks are stronger), ``num_batches_tracked`` (only touched when
        ``bn.training``, already rejected), and this module's own hooks (``__call__``
        fires them whichever route ``forward`` takes). Backward hooks are not checked
        either: grad is disabled on every admitted call, so the ``BackwardHook``
        machinery is an identity pass-through.
        """
        # Forward-only and dispatcher-free. Building a graph is left to the
        # reference so ``backward()`` keeps working; under autocast the reference
        # casts its operands and returns the autocast dtype, which a kernel reading
        # raw pointers below the dispatcher would not; and forward-mode AD is *not*
        # disabled by ``no_grad``, so a dual tensor would silently lose its tangent.
        #
        # The dual test is the level counter, not ``torch._C._is_fwd_grad_enabled()``:
        # that flag is unconditionally true even with no dual level open, so using it
        # here disabled every fast route while leaving every correctness test green.
        # ``_current_level`` is -1 outside ``dual_level()`` and >= 0 inside it.
        if (torch.is_grad_enabled() or torch.is_autocast_enabled("cuda")
                or _forward_ad._current_level >= 0):
            return None
        # A subclass (a Parameter, a FakeTensor, a tensor-subclass wrapper) carries
        # semantics that live above the pointer the kernels read.
        if type(x) is not torch.Tensor:
            return None
        if x.dim() != 4 or not x.is_cuda or x.numel() == 0:
            return None
        if x.dtype not in _FAST_DTYPES or x.shape[1] != self._c1:
            return None
        # The input's layout is deliberately *not* checked here, because it does not
        # decide whether a fast route may run -- only which one. ``forward`` demotes a
        # fused route to the two-kernel route for an input the kernels cannot read as
        # raw memory. Four of the fifty captured pairs are non-contiguous, which is how
        # this was found.
        device = x.device

        conv = self.conv
        # The exact type matters, not just the attributes: a look-alike with matching
        # ``stride``/``padding``/``weight`` would be bypassed entirely by the fused
        # routes, and on the two-kernel route could return an output whose channel
        # count disagrees with the BatchNorm vectors the epilogue indexes.
        if type(conv) is not Conv2d:
            return None
        # The route was chosen for one configuration and every part of it is a plain
        # attribute a caller can reassign. ``in_channels``/``out_channels`` are in
        # here because the kernels take them as their reduction and store extents:
        # reassigning one while leaving the weight alone would otherwise admit a
        # launch whose extents disagree with the weight.
        if (conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride,
                conv.padding, conv.dilation, conv.groups) != self._route_config:
            return None
        weight = conv.weight
        # The kernels reach the weight as raw memory, so a replacement parameter the
        # reference would reject must not get that far. Contiguity is load-bearing
        # for the kernels too: a flat reduction index only equals the offset within
        # the filter when the filter is packed, and the strides the kernels use are
        # derived from the shape rather than read off the tensor.
        if (weight.shape != self._weight_shape or weight.dtype != x.dtype
                or weight.device != device or not weight.is_contiguous()
                or weight.is_neg()):
            return None

        act = self.act
        act_type = type(act)
        if act_type is SiLU:
            act_kind = _ACT_SILU
        elif act_type is nn.Identity:
            act_kind = _ACT_NONE
        else:
            # A module the epilogue does not recognize: the kernel leaves the
            # activation out and it is then called as a module, so it keeps its own
            # semantics and its own hooks -- which is also why its hooks are not
            # checked on this branch. A plain callable is not a module at all and
            # would not even have hook dictionaries to read.
            if not isinstance(act, nn.Module):
                return None
            act_kind = _ACT_EXTERNAL

        # The fused routes read ``conv`` and (for a recognized activation) ``act`` as
        # raw memory and never invoke their ``__call__``, so a hook registered on one
        # of them -- or globally -- would silently not fire where the baseline fires
        # it. On the two-kernel route a ``conv`` forward hook matters for a second
        # reason: it can retain the convolution's output, and the epilogue then
        # rewrites that retained tensor in place, which the baseline never does.
        if (conv._forward_pre_hooks or conv._forward_hooks
                or _module_hooks._global_forward_pre_hooks
                or _module_hooks._global_forward_hooks):
            return None
        if act_kind != _ACT_EXTERNAL and (act._forward_pre_hooks or act._forward_hooks):
            return None

        if self._is_fused:
            bias = conv.bias
            # ``_is_fused`` is a plain attribute and proves nothing on its own: a
            # caller can set it while ``conv.bias`` is still ``None``, and the
            # reference reproduces whatever that does.
            if (bias is None or bias.shape != self._bias_shape
                    or bias.dtype != x.dtype or bias.device != device
                    or not bias.is_contiguous() or bias.is_neg()):
                return None
            return _EPI_BIAS, act_kind, None

        # An unfused convolution carrying a bias is a configuration this module never
        # builds, and the fused kernels' BatchNorm epilogue has nowhere to add it --
        # the baseline's ``Conv2d`` would. Rejecting outright is both cheaper and
        # safer than making one route handle it.
        if conv.bias is not None:
            return None
        # ``bn`` deleted without fusing is the reference path's business, so that the
        # AttributeError it raises is the baseline's.
        bn = getattr(self, "bn", None)
        if bn is None or type(bn) is not BatchNorm2d or bn.training:
            return None
        if not bn.track_running_stats:
            # ``training or not track_running_stats`` is what the baseline passes as
            # ``F.batch_norm``'s training flag, so this would compute batch statistics.
            return None
        # ``F.batch_norm`` rejects a non-positive eps outright ("batch_norm eps must be
        # positive"), and its ``momentum`` argument is typed, so a non-float raises at
        # the dispatch boundary even though eval-mode BatchNorm never reads the value.
        eps = bn.eps
        if (type(eps) is not float or not math.isfinite(eps) or eps <= 0.0
                or type(bn.momentum) is not float):
            return None
        if bn._forward_pre_hooks or bn._forward_hooks:
            return None
        # Measured on this box: CUDA ``F.batch_norm`` requires the affine pair to share
        # one dtype, which must be the input's or fp32 -- an fp16 input with bf16
        # affine parameters raises, and so does an fp16 weight beside an fp32 bias --
        # while the Triton epilogue would happily promote any of them. The running
        # statistics are checked the same way. The captured layout is fp16 affine
        # parameters beside fp32 statistics, because the harness casts parameters only.
        gamma, beta = bn.weight, bn.bias
        mean, var = bn.running_mean, bn.running_var
        if gamma is None or beta is None or mean is None or var is None:
            return None
        affine_dtype = gamma.dtype
        stat_dtype = mean.dtype
        if (beta.dtype is not affine_dtype or var.dtype is not stat_dtype
                or (affine_dtype is not x.dtype and affine_dtype is not torch.float32)
                or (stat_dtype is not x.dtype and stat_dtype is not torch.float32)):
            return None
        shape = self._bias_shape
        for t in (gamma, beta, mean, var):
            if (t.shape != shape or t.device != device
                    or not t.is_contiguous() or t.is_neg()):
                return None
        return _EPI_BN, act_kind, bn

    # -- fused routes ------------------------------------------------------
    def _bn_act_inplace(self, y: torch.Tensor, apply_bn, kernel_act, bn) -> torch.Tensor:
        """Apply the epilogue in place over an already-computed convolution result.

        ``apply_bn`` is whether the BatchNorm affine transform is still owed;
        ``kernel_act`` is the activation this kernel should fold in, which is
        ``_ACT_NONE`` when the activation will be called as a module afterwards. Only
        those two, rather than the full epilogue mode, because a fused convolution
        has already applied its own bias by the time it gets here.

        In place because the convolution's output is freshly allocated and read
        exactly once, which saves an allocation and a second traversal. That is safe
        for the frozen L1 ``Conv2d`` specifically -- ``F.conv2d``, ``mm`` and ``matmul``
        are all out-of-place, and the pointwise route's trailing ``view`` aliases the
        fresh matmul result rather than the input -- and the guard's exact-type check
        on ``conv`` is what keeps it true.
        """
        n, channels, out_h, out_w = y.shape
        plane = out_h * out_w
        if (channels != self._c2 or not y.is_contiguous()
                or y.numel() > _MAX_INT32_OFFSET
                or channels * n > _MAX_GRID_YZ):
            # Finish with the baseline's own operators on the result the convolution
            # already produced, in the baseline's order.
            if apply_bn:
                y = bn(y)
            if kernel_act == _ACT_SILU:
                y = self.act(y)
            return y
        if not apply_bn and kernel_act == _ACT_NONE:
            return y  # nothing left to do; do not pay for a copy
        block_p = 1024 if plane >= 1024 else max(64, triton.next_power_of_2(plane))
        filler = self.conv.weight
        _bn_act[(triton.cdiv(plane, block_p), channels * n)](
            y,
            bn.weight if apply_bn else filler,
            bn.bias if apply_bn else filler,
            bn.running_mean if apply_bn else filler,
            bn.running_var if apply_bn else filler,
            filler,
            plane, bn.eps if apply_bn else 0.0,
            CHANNELS=channels, BLOCK_P=block_p,
            HAS_BN=apply_bn, HAS_BIAS=False,
            SILU=(kernel_act == _ACT_SILU),
            num_warps=4, num_stages=1,
        )
        return y

    def _post_conv(self, x: torch.Tensor, mode, act_kind, bn) -> torch.Tensor:
        # The convolution applies its own fused bias when there is one, so all the
        # epilogue can still owe is the BatchNorm transform and the activation.
        y = self.conv(x)
        y = self._bn_act_inplace(
            y, mode == _EPI_BN,
            _ACT_NONE if act_kind == _ACT_EXTERNAL else act_kind, bn)
        if act_kind == _ACT_EXTERNAL:
            return self.act(y)
        return y

    def _pointwise(self, x: torch.Tensor, mode, act_kind, bn) -> torch.Tensor:
        n, channels, in_h, in_w = x.shape
        out_channels = self._c2
        pixels = in_h * in_w
        tile = self.tile
        if (max(channels * pixels, out_channels * pixels,
                out_channels * channels) > _MAX_INT32_OFFSET
                or n > _MAX_GRID_YZ
                or triton.cdiv(out_channels, tile.block_oc) > _MAX_GRID_YZ):
            return self._reference(x)
        block_k, k_chunks = _reduction_blocks(channels, tile.block_k)
        kernel_act = _ACT_NONE if act_kind == _ACT_EXTERNAL else act_kind
        y = torch.empty((n, out_channels, in_h, in_w), dtype=x.dtype, device=x.device)
        conv = self.conv
        filler = conv.weight
        _pointwise_bn_act[(triton.cdiv(pixels, tile.block_p),
                           triton.cdiv(out_channels, tile.block_oc), n)](
            x, conv.weight,
            bn.weight if mode == _EPI_BN else filler,
            bn.bias if mode == _EPI_BN else filler,
            bn.running_mean if mode == _EPI_BN else filler,
            bn.running_var if mode == _EPI_BN else filler,
            conv.bias if mode == _EPI_BIAS else filler,
            y,
            pixels, bn.eps if mode == _EPI_BN else 0.0,
            CHANNELS=channels, OUT_CHANNELS=out_channels,
            BLOCK_P=tile.block_p, BLOCK_OC=tile.block_oc,
            BLOCK_K=block_k, K_CHUNKS=k_chunks,
            HAS_BN=(mode == _EPI_BN), HAS_BIAS=(mode == _EPI_BIAS),
            SILU=(kernel_act == _ACT_SILU),
            num_warps=tile.num_warps, num_stages=tile.num_stages,
        )
        if act_kind == _ACT_EXTERNAL:
            return self.act(y)
        return y

    def _fits_narrow_offsets(self, x: torch.Tensor, out_h: int, out_w: int) -> bool:
        """Can every offset the implicit-GEMM kernel forms be held in 32 bits?

        The batch term is promoted to 64 bits inside the kernel, so only the
        within-sample spans matter. Spans come from the tensors' real strides,
        because a replacement parameter may be laid out differently from a freshly
        constructed one.
        """
        kh, kw = self.conv.kernel_size
        # The guard established a contiguous input and the output is this kernel's
        # own allocation, so the spans follow from the shapes. ``+ kh``/``+ kw`` cover
        # the padded gather reaching one tap past the last row and column.
        span_in = self._c1 * (x.shape[2] + kh) * (x.shape[3] + kw)
        span_out = self._c2 * out_h * out_w
        span_w = self._c2 * self._c1 * kh * kw
        return max(span_in, span_out, span_w) <= _MAX_INT32_OFFSET

    def _implicit_gemm(self, x: torch.Tensor, mode, act_kind, bn) -> torch.Tensor:
        conv = self.conv
        n, _, in_h, in_w = x.shape
        kh, kw = conv.kernel_size
        sh, sw = conv.stride
        ph, pw = conv.padding
        out_h = (in_h + 2 * ph - kh) // sh + 1
        out_w = (in_w + 2 * pw - kw) // sw + 1
        if out_h <= 0 or out_w <= 0 or not self._fits_narrow_offsets(x, out_h, out_w):
            # A degenerate result is the reference path's business, so that
            # whatever it raises is what the baseline raises.
            return self._reference(x)
        tile = self.tile
        out_channels = self._c2
        reduction = self._c1 * kh * kw
        block_k, k_chunks = _reduction_blocks(reduction, tile.block_k)
        oc_blocks = triton.cdiv(out_channels, tile.block_oc)
        if (n * oc_blocks > _MAX_GRID_YZ
                or triton.cdiv(out_h, tile.block_oh) > _MAX_GRID_YZ):
            return self._reference(x)
        kernel_act = _ACT_NONE if act_kind == _ACT_EXTERNAL else act_kind
        y = torch.empty((n, out_channels, out_h, out_w), dtype=x.dtype, device=x.device)
        filler = conv.weight
        _implicit_gemm_bn_act[(triton.cdiv(out_w, tile.block_ow),
                               triton.cdiv(out_h, tile.block_oh),
                               n * oc_blocks)](
            x, conv.weight,
            bn.weight if mode == _EPI_BN else filler,
            bn.bias if mode == _EPI_BN else filler,
            bn.running_mean if mode == _EPI_BN else filler,
            bn.running_var if mode == _EPI_BN else filler,
            conv.bias if mode == _EPI_BIAS else filler,
            y,
            in_h, in_w, out_h, out_w, bn.eps if mode == _EPI_BN else 0.0,
            CHANNELS=self._c1, OUT_CHANNELS=out_channels,
            KH=kh, KW=kw, SH=sh, SW=sw, PH=ph, PW=pw,
            BLOCK_OH=tile.block_oh, BLOCK_OW=tile.block_ow,
            BLOCK_OC=tile.block_oc, BLOCK_K=block_k, K_CHUNKS=k_chunks,
            REDUCTION=reduction, OC_BLOCKS=oc_blocks,
            HAS_BN=(mode == _EPI_BN), HAS_BIAS=(mode == _EPI_BIAS),
            SILU=(kernel_act == _ACT_SILU),
            num_warps=tile.num_warps, num_stages=tile.num_stages,
        )
        if act_kind == _ACT_EXTERNAL:
            return self.act(y)
        return y

    # -- forward -----------------------------------------------------------
    def _effective_route(self, x: torch.Tensor) -> int:
        """The route this call can take, which is not always the one it was tagged with.

        The two fused kernels index ``x`` as raw memory, so they need a contiguous input
        whose stored bytes are its values. The two-kernel route does not: it hands ``x``
        to the convolution and its epilogue touches only the convolution's output. A
        non-contiguous or negated input therefore gets demoted one step rather than
        dropping all the way to the reference -- worth doing, because four of the fifty
        captured pairs arrive non-contiguous.
        """
        route = self.route
        if route != _POST_CONV and (not x.is_contiguous() or x.is_neg()):
            return _POST_CONV
        return route

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        plan = self._fast_plan(x)
        if plan is None:
            return self._reference(x)
        mode, act_kind, bn = plan
        route = self._effective_route(x)
        if route == _POINTWISE:
            return self._pointwise(x, mode, act_kind, bn)
        if route == _IMPLICIT_GEMM:
            return self._implicit_gemm(x, mode, act_kind, bn)
        return self._post_conv(x, mode, act_kind, bn)

    @torch.no_grad()
    def fuse(self):
        if self._is_fused:
            return self
        fused_weight, fused_bias = _fuse_conv_bn(self.conv, self.bn)
        self.conv.weight.data.copy_(fused_weight)
        self.conv.bias = nn.Parameter(fused_bias)
        delattr(self, "bn")
        self._is_fused = True
        return self


def fuse_module(module: nn.Module) -> nn.Module:
    # The import stays inside the function body: ``yolov10_repvggdw`` imports
    # ``YOLOConv`` from here, so hoisting it to module scope makes that a circular
    # import.
    from .yolov10_repvggdw import YOLORepVGGDW

    for child in module.children():
        fuse_module(child)
    if isinstance(module, YOLOConv):
        module.fuse()
    elif isinstance(module, YOLORepVGGDW):
        module.fuse()
    return module
