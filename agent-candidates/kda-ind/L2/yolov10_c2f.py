"""YOLOv10 C2f / C2fCIB lowered to a straight-line chain of fused Triton kernels.

Why this shape of solution
--------------------------
The operator is launch-bound, not arithmetic-bound. On a B200 the reference chain issues
22-45 kernels per call, which costs the host 0.31-0.53 ms while the GPU has only 66-128 us of
work, so measured latency equals dispatch cost. Of that GPU time only ~30% is convolution
math: bias-add + SiLU + `cat` + residual are ~45% and cuDNN's internal NCHW<->NHWC transposes
another ~25%. The largest benchmarked case is 0.944 GMAC, well under a microsecond of fp16
tensor-core time. Numbers in `docs/measurements.md`.

So the wins come from removing launches and round-trips, in this order:

1.  BatchNorm is folded into a convolution bias, and `YOLORepVGGDW`'s 7x7 + 3x3 pair collapses
    into a single depthwise 7x7 (6 kernels -> 1). Folding alone measured 1.7-2.2x.
2.  bias, SiLU and the residual add become epilogues of the convolution that produces the
    value, so each result is written exactly once.
3.  `chunk`/`cat` disappear: every intermediate is a channel slice of one pre-allocated NHWC
    buffer, so concatenation is an offset and the copy is gone.
4.  A CUDA graph per shape, enabled only for chains long enough to pay for it. See below.

Layout: the input arrives NCHW-contiguous and the result leaves NCHW-contiguous, but the
interior is NHWC. No standalone transpose is ever paid -- a standalone NCHW->NHWC copy costs
23.6 us on the largest case, more than every convolution in the chain combined. Instead the
first and last steps are always 1x1 convolutions, i.e. GEMMs, and they absorb the layout
change into their own loads and stores.

Contract and known limitations
------------------------------
*   Weights are treated as frozen after loading. The folded plan is rebuilt on
    `load_state_dict` and on `_apply` (`.to()`, `.half()`, `.cuda()`), and a dtype/device/
    identity stamp on `cv1.conv.weight` is rechecked every call. A bare in-place
    `weight.data.copy_()` or a mutated BatchNorm buffer between two forwards fires no hook and
    is not detected: the next call still reads the new input and recomputes, but it does so with
    the *stale folded coefficients*, so the result reflects the old weights. The stamp only
    watches `cv1.conv.weight`, so a mutation to any other parameter or buffer is invisible to it
    as well. Detecting either would mean re-folding or hashing every weight on every call, which
    costs more than the entire kernel budget.
*   The fast path handles fp16, CUDA, 4-D, contiguous input in eval mode with grad disabled,
    dense convolutions with `groups == 1` and depthwise convolutions with `groups == channels`.
    Everything else -- including grouped convolutions, other dtypes, CPU tensors, training
    mode and non-contiguous input -- runs the retained submodules, so it is the reference
    computation exactly rather than an approximation of it.
*   `forward` returns a freshly allocated NCHW-contiguous `torch.Tensor` every call. The
    persistent graph-static output buffer is never handed out, because returning it would
    silently overwrite a result the caller still holds.
*   One module instance owns one concatenation buffer and one set of scratch buffers, so calls
    on the *same* instance must be serialized. Two host threads, or two calls enqueued on
    independent CUDA streams, would overwrite each other's intermediates -- stream ordering only
    serializes work on one stream. Separate instances are independent. This is the usual
    consequence of holding scratch and matches how the module is driven here (one instance, one
    thread, one stream), but it is a real limitation rather than an oversight.

When the CUDA graph pays, and when it does not
---------------------------------------------
Graph capture was expected to be the largest single lever: the reference chain issues 22-45
launches, and replacing that dispatch with a 2 us replay is worth 1.7-2.5x *on that chain*. But
by the time the fold, the fused epilogues and the concatenation buffer are in place the chain is
only 4 to 7 launches, and a graph has to pay for its fixed pointers with two copies the ungraphed
path does not need: `static_in.copy_(x)` on the way in, because replay reads a fixed address, and
a clone on the way out, because the static output buffer must not be handed to the caller.

Those two costs are set by the tensor sizes and do not shrink; the host dispatch a graph saves is
proportional to the number of launches. So there is a crossover. Ungraphed / graphed with
identical kernels, three independent runs of `tools/probe_ablation.py` (this host is shared, so
the spread between runs is contention, not measurement error -- see `docs/measurements.md`):

    steps  case                  run 1   run 2   run 3
    4      C2f [1,256,20,20]     0.94    0.97    0.98
    4      C2f [4,192,80,80]     0.88    0.89    0.87
    4      C2f [1,32,160,160]    0.93    0.99    0.94
    4      C2f [1,192,40,40]     0.88    0.95    0.95
    6      C2f [1,128,40,40]     1.00    1.12    1.02
    7      CIB [4,384,20,20]     1.19    1.30    1.03
    7      CIB [1,384,20,20]     1.12    1.25    1.00

The robust half of that is the loss: every one of the twelve measurements of a four-step chain is
below 1.0, between 0.87 and 0.99. The gain on longer chains is real but load-dependent, ranging
from 1.00 to 1.30 and never dropping below 1.00. Both halves point the same way, so capture is
enabled from `_GRAPH_MIN_STEPS` launches upward and skipped below it, and summed over the seven
scored cases that threshold beat both always-capturing and never-capturing in the run where all
three were measured together. This is a property of the lowered chain rather than of the
benchmarked shapes: any configuration deep enough to amortize the two copies gets a graph. Below
the threshold nothing is copied at all, because the ungraphed path binds the caller's input and a
freshly allocated output straight into the first and last kernels.

`FK_C2F_FAST=0` disables the fast path entirely, and `FK_C2F_GRAPH` overrides the threshold with
`1` (always capture) or `0` (never), so the kernel-side and host-side effects can be measured
apart from each other.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .yolov10_bottleneck import YOLOBottleneck
from .yolov10_cib import YOLOCIB
from .yolov10_conv import YOLOConv
from .yolov10_repvggdw import YOLORepVGGDW

try:
    import triton
    import triton.language as tl

    _HAVE_TRITON = True
except Exception:  # pragma: no cover - Triton is present in the target environment
    _HAVE_TRITON = False


_FAST_ENABLED = os.environ.get("FK_C2F_FAST", "1") != "0"

# Chains of this many launches or more are captured; shorter ones are not, because the graph's
# mandatory input copy and output clone cost more than the dispatch they save. Measured crossover
# is between 4 and 6 launches; see the module docstring.
_GRAPH_MIN_STEPS = 6
# "auto" follows the threshold; "1" always captures and "0" never does, for ablation.
_GRAPH_OVERRIDE = os.environ.get("FK_C2F_GRAPH", "auto")

# How a 1x1 step addresses an NCHW boundary. Mode 0 is the NHWC interior, where channels are
# contiguous and nothing special is needed. Modes 1 and 2 both read (or write) NCHW and differ
# only in which tile axis Triton maps lanes onto: mode 1 puts lanes on the channel axis, whose
# stride is H*W, so a warp can touch 32 sectors for 64 useful bytes; mode 2 puts lanes on the
# unit-stride pixel axis and pays a register-level transpose instead. Which one wins is a
# property of Triton's lowering, not of the arithmetic, so both are kept and chosen by
# measurement (`tools/sweep_tiles.py`).
_LANES_ON_CHANNELS = 1
_LANES_ON_PIXELS = 2

_PLANAR_SRC_MODE = int(os.environ.get("FK_C2F_PLANAR_SRC", _LANES_ON_CHANNELS))
_PLANAR_DST_MODE = int(os.environ.get("FK_C2F_PLANAR_DST", _LANES_ON_PIXELS))

# The kernels index with 32-bit arithmetic. Named so the guard below can be exercised by a test
# without allocating a multi-gigabyte tensor.
_MAX_ELEMENT_INDEX = 0x7FFFFFFF

_KIND_CONV1X1 = 0
_KIND_CONVKXK = 1
_KIND_DWCONV = 2


if _HAVE_TRITON:

    @triton.jit
    def _conv1x1_kernel(
        src_ptr,
        wgt_ptr,
        bias_ptr,
        dst_ptr,
        res_ptr,
        n_pix,
        n_in,
        n_out,
        src_img_stride,
        src_pix_stride,
        src_chan_stride,
        dst_img_stride,
        dst_pix_stride,
        dst_chan_stride,
        res_pix_stride,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        SRC_MODE: tl.constexpr,
        DST_MODE: tl.constexpr,
        APPLY_SILU: tl.constexpr,
        ADD_RESIDUAL: tl.constexpr,
    ):
        """1x1 convolution as a GEMM, with bias / SiLU / residual folded into the epilogue.

        The image index is the third grid axis rather than part of a flattened row index, so a
        tile never straddles two images and no integer division appears in the addressing.
        """
        pix0 = tl.program_id(0) * BLOCK_M
        out0 = tl.program_id(1) * BLOCK_N
        img = tl.program_id(2)

        offs_m = pix0 + tl.arange(0, BLOCK_M)
        offs_n = out0 + tl.arange(0, BLOCK_N)
        mask_m = offs_m < n_pix
        mask_n = offs_n < n_out

        src_base = src_ptr + img * src_img_stride
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k0 in range(0, tl.cdiv(n_in, BLOCK_K)):
            offs_k = k0 * BLOCK_K + tl.arange(0, BLOCK_K)
            mask_k = offs_k < n_in

            if SRC_MODE == 0:
                # NHWC: channels are contiguous, so the fastest-varying tile axis is already
                # the unit-stride one and the load vectorizes as-is.
                tile = tl.load(
                    src_base + offs_m[:, None] * src_pix_stride + offs_k[None, :],
                    mask=mask_m[:, None] & mask_k[None, :],
                    other=0.0,
                )
            elif SRC_MODE == 1:
                tile = tl.load(
                    src_base + offs_m[:, None] + offs_k[None, :] * src_chan_stride,
                    mask=mask_m[:, None] & mask_k[None, :],
                    other=0.0,
                )
            else:
                # NCHW read with pixels as the fastest-varying tile axis, so each lane group
                # walks unit-stride memory; the [K, M] -> [M, K] fix-up is a register-level
                # layout conversion instead of a strided global read.
                tile = tl.trans(
                    tl.load(
                        src_base + offs_k[:, None] * src_chan_stride + offs_m[None, :],
                        mask=mask_k[:, None] & mask_m[None, :],
                        other=0.0,
                    )
                )

            wgt = tl.load(
                wgt_ptr + offs_k[:, None] * n_out + offs_n[None, :],
                mask=mask_k[:, None] & mask_n[None, :],
                other=0.0,
            )
            acc += tl.dot(tile, wgt)

        acc += tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)[None, :]
        if APPLY_SILU:
            # x * sigmoid(x) written as x / (1 + exp(-x)): exact for the same reason and it
            # saturates the right way at both ends without a guard.
            acc = acc / (1.0 + tl.exp(-acc))
        if ADD_RESIDUAL:
            acc += tl.load(
                res_ptr + img * (n_pix * res_pix_stride) + offs_m[:, None] * res_pix_stride + offs_n[None, :],
                mask=mask_m[:, None] & mask_n[None, :],
                other=0.0,
            ).to(tl.float32)

        dst_base = dst_ptr + img * dst_img_stride
        out = acc.to(dst_ptr.dtype.element_ty)
        if DST_MODE == 0:
            tl.store(
                dst_base + offs_m[:, None] * dst_pix_stride + offs_n[None, :],
                out,
                mask=mask_m[:, None] & mask_n[None, :],
            )
        elif DST_MODE == 1:
            tl.store(
                dst_base + offs_n[None, :] * dst_chan_stride + offs_m[:, None],
                out,
                mask=mask_m[:, None] & mask_n[None, :],
            )
        else:
            # Same addresses as mode 1, but lanes walk the unit-stride pixel axis.
            tl.store(
                dst_base + offs_n[:, None] * dst_chan_stride + offs_m[None, :],
                tl.trans(out),
                mask=mask_n[:, None] & mask_m[None, :],
            )

    @triton.jit
    def _convkxk_kernel(
        src_ptr,
        wgt_ptr,
        bias_ptr,
        dst_ptr,
        res_ptr,
        n_pix,
        n_in,
        n_out,
        height,
        width,
        src_img_stride,
        src_pix_stride,
        dst_img_stride,
        dst_pix_stride,
        res_pix_stride,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        KSIZE: tl.constexpr,
        PAD: tl.constexpr,
        APPLY_SILU: tl.constexpr,
        ADD_RESIDUAL: tl.constexpr,
    ):
        """Dense k x k convolution as an implicit GEMM over NHWC, stride 1, `groups == 1`.

        The (r, s) loops are unrolled and the halo is handled by masking the gathered rows, so
        no padded copy of the source is ever materialized.
        """
        pix0 = tl.program_id(0) * BLOCK_M
        out0 = tl.program_id(1) * BLOCK_N
        img = tl.program_id(2)

        offs_m = pix0 + tl.arange(0, BLOCK_M)
        offs_n = out0 + tl.arange(0, BLOCK_N)
        mask_m = offs_m < n_pix
        mask_n = offs_n < n_out

        out_h = offs_m // width
        out_w = offs_m % width

        src_base = src_ptr + img * src_img_stride
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for r in tl.static_range(KSIZE):
            in_h = out_h + r - PAD
            row_ok = (in_h >= 0) & (in_h < height) & mask_m
            for s in tl.static_range(KSIZE):
                in_w = out_w + s - PAD
                tap_ok = row_ok & (in_w >= 0) & (in_w < width)
                row_off = (in_h * width + in_w) * src_pix_stride
                tap_base = (r * KSIZE + s) * n_in
                for k0 in range(0, tl.cdiv(n_in, BLOCK_K)):
                    offs_k = k0 * BLOCK_K + tl.arange(0, BLOCK_K)
                    mask_k = offs_k < n_in
                    tile = tl.load(
                        src_base + row_off[:, None] + offs_k[None, :],
                        mask=tap_ok[:, None] & mask_k[None, :],
                        other=0.0,
                    )
                    wgt = tl.load(
                        wgt_ptr + (tap_base + offs_k)[:, None] * n_out + offs_n[None, :],
                        mask=mask_k[:, None] & mask_n[None, :],
                        other=0.0,
                    )
                    acc += tl.dot(tile, wgt)

        acc += tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)[None, :]
        if APPLY_SILU:
            acc = acc / (1.0 + tl.exp(-acc))
        if ADD_RESIDUAL:
            acc += tl.load(
                res_ptr + img * (n_pix * res_pix_stride) + offs_m[:, None] * res_pix_stride + offs_n[None, :],
                mask=mask_m[:, None] & mask_n[None, :],
                other=0.0,
            ).to(tl.float32)

        tl.store(
            dst_ptr + img * dst_img_stride + offs_m[:, None] * dst_pix_stride + offs_n[None, :],
            acc.to(dst_ptr.dtype.element_ty),
            mask=mask_m[:, None] & mask_n[None, :],
        )

    @triton.jit
    def _dwconv_kernel(
        src_ptr,
        wgt_ptr,
        bias_ptr,
        dst_ptr,
        res_ptr,
        n_pix,
        n_chan,
        height,
        width,
        src_img_stride,
        src_pix_stride,
        dst_img_stride,
        dst_pix_stride,
        res_pix_stride,
        BLOCK_M: tl.constexpr,
        BLOCK_C: tl.constexpr,
        KSIZE: tl.constexpr,
        PAD: tl.constexpr,
        APPLY_SILU: tl.constexpr,
        ADD_RESIDUAL: tl.constexpr,
    ):
        """Depthwise k x k convolution over NHWC, stride 1, `groups == channels`.

        There is no reduction across channels, so this is a masked weighted sum of KSIZE**2
        taps rather than a GEMM. The taps of neighbouring output pixels overlap heavily, which
        is what keeps the repeated loads in cache.
        """
        pix0 = tl.program_id(0) * BLOCK_M
        chan0 = tl.program_id(1) * BLOCK_C
        img = tl.program_id(2)

        offs_m = pix0 + tl.arange(0, BLOCK_M)
        offs_c = chan0 + tl.arange(0, BLOCK_C)
        mask_m = offs_m < n_pix
        mask_c = offs_c < n_chan

        out_h = offs_m // width
        out_w = offs_m % width

        src_base = src_ptr + img * src_img_stride
        acc = tl.zeros((BLOCK_M, BLOCK_C), dtype=tl.float32)

        for r in tl.static_range(KSIZE):
            in_h = out_h + r - PAD
            row_ok = (in_h >= 0) & (in_h < height) & mask_m
            for s in tl.static_range(KSIZE):
                in_w = out_w + s - PAD
                tap_ok = row_ok & (in_w >= 0) & (in_w < width)
                vals = tl.load(
                    src_base + ((in_h * width + in_w) * src_pix_stride)[:, None] + offs_c[None, :],
                    mask=tap_ok[:, None] & mask_c[None, :],
                    other=0.0,
                )
                taps = tl.load(wgt_ptr + (r * KSIZE + s) * n_chan + offs_c, mask=mask_c, other=0.0)
                acc += vals.to(tl.float32) * taps.to(tl.float32)[None, :]

        acc += tl.load(bias_ptr + offs_c, mask=mask_c, other=0.0).to(tl.float32)[None, :]
        if APPLY_SILU:
            acc = acc / (1.0 + tl.exp(-acc))
        if ADD_RESIDUAL:
            acc += tl.load(
                res_ptr + img * (n_pix * res_pix_stride) + offs_m[:, None] * res_pix_stride + offs_c[None, :],
                mask=mask_m[:, None] & mask_c[None, :],
                other=0.0,
            ).to(tl.float32)

        tl.store(
            dst_ptr + img * dst_img_stride + offs_m[:, None] * dst_pix_stride + offs_c[None, :],
            acc.to(dst_ptr.dtype.element_ty),
            mask=mask_m[:, None] & mask_c[None, :],
        )


# ---------------------------------------------------------------------------------------
# Tile selection
# ---------------------------------------------------------------------------------------
#
# These problems are latency-bound, not throughput-bound: the smallest benchmarked case is 400
# pixels, which at a 128-row tile fills 4 CTAs of 148 SMs. Tiles are therefore chosen for grid
# breadth first and arithmetic efficiency second, but "more CTAs" is not free -- shrinking the
# N tile re-reads the source tile once per extra column block -- so the table below is filled
# from a measured sweep (`tools/sweep_tiles.py`) and the heuristic only covers geometries the
# sweep never visited.

# Tile selection is a closed-form function of the kernel geometry, not a runtime autotuner:
# autotuning synchronizes, so it cannot happen under graph capture, and it would have to happen
# on the first call, which is a correctness round.
#
# It is also not a lookup table. `tools/sweep_tiles.py` measured 28 configurations per step
# against every geometry the scored cases produce, and the resulting per-geometry table was then
# priced against the closed-form rule below by timing both interleaved in one process
# (`tools/probe_ab.py`). The table lost, 0.990x. The reason is that the per-step differences it
# was built from -- around 2 us -- sit inside the resolution of a full-chain graph-replay
# measurement, so the table had largely encoded noise, while the rule distilled from the shape
# of the winners generalizes. So the rule is all that ships, and it applies to any geometry
# rather than only to the ones that were measured.


def _graph_wanted(n_steps: int) -> bool:
    """Capture only chains long enough to amortize the graph's input copy and output clone."""
    if _GRAPH_OVERRIDE == "1":
        return True
    if _GRAPH_OVERRIDE == "0":
        return False
    return n_steps >= _GRAPH_MIN_STEPS


def _pow2_at_most(value: int, cap: int) -> int:
    out = 16
    while out * 2 <= min(value, cap):
        out *= 2
    return out


def _tiles(n_pix: int, n_in: int, n_out: int):
    """Short M tile, wide N and K, four warps.

    An earlier rule picked the smallest tile that still covered one wave of the machine, on the
    theory that these problems are too small to fill a B200 (the smallest is 400 pixels). The
    sweep showed that reasoning fails in both directions: wave-filling chose 128x128 for the
    batch-4 cases, where batch already multiplies the CTA count, and 32x16 for batch-1, and the
    measured optimum beat those by 18 us and 4 us respectively. Grid breadth does matter, but
    not at the price of a 16-wide N tile that re-reads the whole source tile for every column
    block. A 32-row tile gives enough CTAs at every scored size; only the two 25600-pixel
    geometries have enough rows to prefer 64.
    """
    block_m = 64 if n_pix >= 16384 else 32
    block_n = 64 if n_out >= 64 else _pow2_at_most(max(n_out, 16), 32)
    return block_m, block_n, _pow2_at_most(max(n_in, 16), 64), 4, 2


def _conv1x1_tiles(n_pix: int, n_in: int, n_out: int):
    return _tiles(n_pix, n_in, n_out)


def _kxk_tiles(n_pix: int, n_in: int, n_out: int):
    return _tiles(n_pix, n_in, n_out)


def _dw_tiles(n_pix: int, n_chan: int):
    """Depthwise has no reduction across channels, so there is no K axis to widen."""
    block_m = 64 if n_pix >= 16384 else 32
    return block_m, _pow2_at_most(max(n_chan, 16), 32), 4, 2


# ---------------------------------------------------------------------------------------
# Folding
# ---------------------------------------------------------------------------------------


def _folded_conv_bn(conv: nn.Module, bn: nn.Module | None) -> tuple[torch.Tensor, torch.Tensor]:
    """Fold BatchNorm into a convolution weight and bias, computing in fp32.

    The benchmark never calls `YOLOConv.fuse`, so the reference is the *unfused* BatchNorm
    path. That path evaluates the normalization against fp32 running statistics, so folding in
    fp32 and casting the result once is closer to it than folding in the weight dtype would be.
    """
    weight = conv.weight.detach().float()
    bias = conv.bias.detach().float() if conv.bias is not None else torch.zeros(
        weight.shape[0], dtype=torch.float32, device=weight.device
    )
    if bn is None:
        return weight, bias
    scale = bn.weight.detach().float() / torch.sqrt(bn.running_var.detach().float() + bn.eps)
    folded_w = weight * scale.view(-1, *([1] * (weight.dim() - 1)))
    folded_b = (bias - bn.running_mean.detach().float()) * scale + bn.bias.detach().float()
    return folded_w, folded_b


def _folded_yolo_conv(block: YOLOConv) -> tuple[torch.Tensor, torch.Tensor]:
    return _folded_conv_bn(block.conv, None if getattr(block, "_is_fused", False) else block.bn)


def _folded_repvggdw(block: YOLORepVGGDW) -> tuple[torch.Tensor, torch.Tensor]:
    """Collapse the 7x7 + 3x3 depthwise pair into one depthwise 7x7.

    Both branches are linear and share an input, so their sum is a single convolution whose
    weight is the wide kernel plus the narrow one padded into its centre. Padding into a corner
    instead would shift the narrow branch's receptive field and is the easy way to get this
    wrong. `act=False` on both branches means the SiLU belongs to the collapsed step.
    """
    wide_w, wide_b = _folded_yolo_conv(block.conv)
    if getattr(block, "_is_fused", False):
        return wide_w, wide_b
    narrow_w, narrow_b = _folded_yolo_conv(block.conv1)
    pad = (wide_w.shape[-1] - narrow_w.shape[-1]) // 2
    return wide_w + F.pad(narrow_w, (pad, pad, pad, pad)), wide_b + narrow_b


class _LeafOp:
    """One convolution of the lowered chain, with its weight already in the kernel's layout."""

    __slots__ = ("kind", "n_in", "n_out", "ksize", "pad", "weight", "bias", "silu")

    def __init__(self, kind, n_in, n_out, ksize, pad, weight, bias, silu):
        self.kind = kind
        self.n_in = n_in
        self.n_out = n_out
        self.ksize = ksize
        self.pad = pad
        self.weight = weight
        self.bias = bias
        self.silu = silu


def _classify(block: YOLOConv, dtype: torch.dtype) -> _LeafOp | None:
    """Fold one `YOLOConv` and put its weight in the layout its kernel wants, or reject it."""
    conv = block.conv
    ksize = conv.weight.shape[-1]
    n_out, per_group = conv.weight.shape[0], conv.weight.shape[1]
    n_in = per_group * conv.groups
    if conv.weight.shape[-1] != conv.weight.shape[-2]:
        return None
    if tuple(conv.stride) != (1, 1) or tuple(conv.dilation) != (1, 1):
        return None
    if tuple(conv.padding) != (ksize // 2, ksize // 2):
        return None
    silu = not isinstance(block.act, nn.Identity)
    if silu and not isinstance(block.act, type(YOLOConv.default_act)):
        return None

    weight, bias = _folded_yolo_conv(block)
    weight, bias = weight.to(dtype), bias.to(dtype)
    if not (torch.isfinite(weight).all() and torch.isfinite(bias).all()):
        # Folding multiplies the weight by gamma/sqrt(var + eps). A small running variance can
        # push that product outside fp16 range even where the unfused convolution-then-normalize
        # sequence stays finite, so representability is checked rather than assumed.
        return None

    if conv.groups == 1 and ksize == 1:
        # [Cout, Cin, 1, 1] -> [Cin, Cout], the GEMM's B matrix.
        return _LeafOp(_KIND_CONV1X1, n_in, n_out, 1, 0, weight.reshape(n_out, n_in).t().contiguous(), bias, silu)
    if conv.groups == 1 and ksize in (3, 5, 7):
        # [Cout, Cin, R, S] -> [R*S, Cin, Cout], one B matrix per tap.
        packed = weight.permute(2, 3, 1, 0).reshape(ksize * ksize, n_in, n_out).contiguous()
        return _LeafOp(_KIND_CONVKXK, n_in, n_out, ksize, ksize // 2, packed, bias, silu)
    if conv.groups == n_in == n_out and ksize in (3, 5, 7):
        # [C, 1, R, S] -> [R*S, C].
        packed = weight.reshape(n_out, ksize * ksize).t().contiguous()
        return _LeafOp(_KIND_DWCONV, n_in, n_out, ksize, ksize // 2, packed, bias, silu)
    return None


def _plain_depthwise(conv: nn.Module) -> bool:
    """True if `conv` is a stride-1, dilation-1, centre-padded depthwise convolution."""
    ksize = conv.weight.shape[-1]
    return (
        conv.weight.shape[-1] == conv.weight.shape[-2]
        and conv.groups == conv.weight.shape[0]
        and tuple(conv.stride) == (1, 1)
        and tuple(conv.dilation) == (1, 1)
        and tuple(conv.padding) == (ksize // 2, ksize // 2)
    )


def _classify_repvggdw(block: YOLORepVGGDW, dtype: torch.dtype) -> _LeafOp | None:
    inner = block.conv
    if not _plain_depthwise(inner.conv):
        return None
    # The narrow branch has to be checked too. Dilating it preserves its output shape, so it
    # would pass a wide-branch-only check while being collapsed as an undilated centred kernel,
    # which is a different convolution.
    if not getattr(block, "_is_fused", False):
        if not _plain_depthwise(block.conv1.conv):
            return None
        if block.conv1.conv.weight.shape[0] != inner.conv.weight.shape[0]:
            return None
        if block.conv1.conv.weight.shape[-1] > inner.conv.weight.shape[-1]:
            return None
        if (inner.conv.weight.shape[-1] - block.conv1.conv.weight.shape[-1]) % 2 != 0:
            return None  # cannot be centred
        if not isinstance(inner.act, nn.Identity) or not isinstance(block.conv1.act, nn.Identity):
            return None  # both branches must be linear for the sum to be one convolution
    weight, bias = _folded_repvggdw(block)
    ksize = weight.shape[-1]
    if ksize not in (3, 5, 7):
        return None
    n_chan = weight.shape[0]
    weight, bias = weight.to(dtype), bias.to(dtype)
    if not (torch.isfinite(weight).all() and torch.isfinite(bias).all()):
        return None
    packed = weight.reshape(n_chan, ksize * ksize).t().contiguous()
    return _LeafOp(_KIND_DWCONV, n_chan, n_chan, ksize, ksize // 2, packed, bias, True)


def _leaf_ops(module: nn.Module, dtype: torch.dtype) -> list[_LeafOp] | None:
    if isinstance(module, YOLORepVGGDW):
        op = _classify_repvggdw(module, dtype)
        return None if op is None else [op]
    if isinstance(module, YOLOConv):
        op = _classify(module, dtype)
        return None if op is None else [op]
    if isinstance(module, (nn.Sequential, nn.ModuleList)):
        out: list[_LeafOp] = []
        for child in module:
            got = _leaf_ops(child, dtype)
            if got is None:
                return None
            out.extend(got)
        return out
    return None


def _block_chain(block: nn.Module, dtype: torch.dtype) -> tuple[list[_LeafOp], bool] | None:
    """Flatten one C2f block into its convolution chain plus whether it adds a residual."""
    if isinstance(block, YOLOBottleneck):
        ops = _leaf_ops(nn.Sequential(block.cv1, block.cv2), dtype)
    elif isinstance(block, YOLOCIB):
        ops = _leaf_ops(block.cv1, dtype)
    else:
        return None
    if not ops:
        return None
    return ops, bool(block.add)


# ---------------------------------------------------------------------------------------
# Execution plan
# ---------------------------------------------------------------------------------------


class _Step:
    """A single kernel launch with its grid and arguments resolved ahead of time."""

    __slots__ = ("kernel", "grid", "args", "meta", "label")

    def __init__(self, kernel, grid, args, meta, label):
        self.kernel = kernel
        self.grid = grid
        self.args = args
        self.meta = meta
        self.label = label

    def launch(self) -> None:
        self.kernel[self.grid](*self.args, **self.meta)

    def launch_with(self, index: int, value: torch.Tensor) -> None:
        """Launch with one argument substituted, without mutating the shared argument list.

        The ungraphed path has to point the first step at the caller's input and the last step at
        a fresh output on every call. Rebinding `self.args` in place would work, but it would
        leave the caller's input reachable from the plan after the call returned, and would make
        the endpoints depend on call order. Substituting on a copy avoids both. Copying a
        15-element list costs far less than the launch it precedes.
        """
        args = list(self.args)
        args[index] = value
        self.kernel[self.grid](*args, **self.meta)


class _Plan:
    """The lowered chain for one input shape, plus its optional captured graph.

    `buffers` is held only to keep the concatenation buffer and the scratch alive: the steps
    address them through raw pointers baked into `args`.
    """

    __slots__ = ("steps", "buffers", "static_in", "static_out", "graph", "out_shape", "out_kwargs", "key")

    def __init__(self, steps, buffers, static_in, static_out, out_shape, out_kwargs, key):
        self.steps = steps
        self.buffers = buffers
        self.static_in = static_in
        self.static_out = static_out
        self.graph = None
        self.out_shape = out_shape
        self.out_kwargs = out_kwargs
        self.key = key

    def bind(self, src: torch.Tensor, dst: torch.Tensor) -> None:
        """Fix the chain's endpoints permanently, for capture."""
        self.steps[0].args[0] = src
        self.steps[-1].args[3] = dst

    def run_eager(self, x: torch.Tensor) -> torch.Tensor:
        # The caller's input and a fresh output are read and written directly by the first and
        # last kernels, so this path performs no copies at all. `cv1` and `cv2` are always
        # distinct steps, so the two substitutions never collide.
        out = torch.empty(self.out_shape, **self.out_kwargs)
        last = len(self.steps) - 1
        for i, step in enumerate(self.steps):
            if i == 0:
                step.launch_with(0, x)
            elif i == last:
                step.launch_with(3, out)
            else:
                step.launch()
        return out

    def run_graph(self, x: torch.Tensor) -> torch.Tensor:
        self.static_in.copy_(x)
        self.graph.replay()
        return self.static_out.clone()


class YOLOC2f(nn.Module):
    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = YOLOConv(c1, 2 * self.c, 1, 1)
        self.cv2 = YOLOConv((2 + n) * self.c, c2, 1, 1)
        self.m = nn.ModuleList(
            YOLOBottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n)
        )
        self._init_fast_state()

    # -- fast-path state -------------------------------------------------------------

    def _init_fast_state(self) -> None:
        # Plain attributes, never parameters or buffers, so the state-dict key set stays
        # exactly the baseline's before and after the fast path initializes.
        self._fk_chain = None
        self._fk_plan = None
        self._fk_stamp = None
        self._fk_widest_weight = 0
        self._fk_unsupported = False
        self._fk_graph_ok = True
        self.register_load_state_dict_post_hook(lambda mod, _keys: mod._invalidate_fast_state())

    def _invalidate_fast_state(self) -> None:
        self._fk_chain = None
        self._fk_plan = None
        self._fk_stamp = None
        self._fk_widest_weight = 0
        self._fk_unsupported = False
        self._fk_graph_ok = True

    def _apply(self, *args, **kwargs):
        # .to(), .half(), .cuda() all funnel through here and move or retype the weights the
        # folded plan was built from.
        self._invalidate_fast_state()
        return super()._apply(*args, **kwargs)

    # -- reference path --------------------------------------------------------------

    def _reference_forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    # -- lowering -------------------------------------------------------------------

    def _build_chain(self, dtype: torch.dtype):
        """Fold the whole module into (head, per-block chains, tail), or None if unsupported."""
        head = _leaf_ops(self.cv1, dtype)
        tail = _leaf_ops(self.cv2, dtype)
        if head is None or tail is None or len(head) != 1 or len(tail) != 1:
            return None
        if head[0].kind != _KIND_CONV1X1 or tail[0].kind != _KIND_CONV1X1:
            return None
        if head[0].n_out % 2 != 0:
            return None

        blocks = []
        width = head[0].n_out
        carry = head[0].n_out // 2
        for block in self.m:
            got = _block_chain(block, dtype)
            if got is None:
                return None
            ops, add = got
            if ops[0].n_in != carry:
                return None
            for prev, nxt in zip(ops, ops[1:]):
                if prev.n_out != nxt.n_in:
                    return None
            if ops[-1].n_out != carry:
                # The block's output becomes the next block's input and one concatenated
                # piece, so a block that changes width cannot be laid out this way.
                return None
            blocks.append((ops, add))
            width += ops[-1].n_out
        if tail[0].n_in != width:
            return None
        return head[0], blocks, tail[0], width, carry

    def _build_plan(self, x: torch.Tensor, chain) -> _Plan:
        head, blocks, tail, width, carry = chain
        batch, _, height, w = x.shape
        n_pix = height * w
        dtype, device = x.dtype, x.device
        kwargs = dict(dtype=dtype, device=device)

        # One NHWC buffer holds every concatenated piece, so `chunk` is an offset and `cat`
        # does not exist. Rows are pixels of one image; images are `n_pix * width` apart.
        concat = torch.empty((batch * n_pix, width), **kwargs)
        buffers = [concat]

        # Scratch for values that live only inside a block. Slot parity guarantees a step
        # never reads and writes the same buffer; a slot is reused two steps later, by which
        # point its previous value has already been consumed.
        scratch: dict[tuple[int, int], torch.Tensor] = {}

        def slot(depth: int, cols: int) -> torch.Tensor:
            key = (depth % 2, cols)
            buf = scratch.get(key)
            if buf is None:
                buf = torch.empty((batch * n_pix, cols), **kwargs)
                scratch[key] = buf
                buffers.append(buf)
            return buf

        # Only the graphed path needs fixed endpoints; the ungraphed path binds the caller's
        # input and a fresh output directly, which is why it copies nothing.
        # The chain length is known before anything is allocated, so the graph decision -- and
        # therefore whether the static endpoints are needed at all -- is made up front.
        n_steps = 2 + sum(len(ops) for ops, _ in blocks)
        want_graph = self._fk_graph_ok and _graph_wanted(n_steps)
        static_in = torch.empty_like(x) if want_graph else None
        static_out = torch.empty((batch, tail.n_out, height, w), **kwargs) if want_graph else None
        steps: list[_Step] = []

        # Both NCHW boundaries are rebound every call on the ungraphed path, so `emit` takes
        # their stride triples directly rather than a tensor to read strides off: nothing has to
        # be allocated just to describe a layout. An interleaved buffer passes `None` and its
        # strides are read from the view itself.
        def emit(op: _LeafOp, src, dst, res, src_planar=None, dst_planar=None):
            grid_n = op.n_out
            if op.kind == _KIND_DWCONV:
                block_m, block_c, warps, stages = _dw_tiles(n_pix, op.n_out)
                args = [
                    src, op.weight, op.bias, dst, dst if res is None else res,
                    n_pix, op.n_out, height, w,
                    n_pix * src.stride(0), src.stride(0),
                    n_pix * dst.stride(0), dst.stride(0),
                    0 if res is None else res.stride(0),
                ]  # depthwise steps are interior-only, so both ends are interleaved
                meta = dict(
                    BLOCK_M=block_m, BLOCK_C=block_c, KSIZE=op.ksize, PAD=op.pad,
                    APPLY_SILU=op.silu, ADD_RESIDUAL=res is not None,
                    num_warps=warps, num_stages=stages,
                )
                grid = (-(-n_pix // block_m), -(-grid_n // block_c), batch)
                steps.append(_Step(_dwconv_kernel, grid, args, meta, f"dw{op.ksize}x{op.ksize}"))
                return
            if op.kind == _KIND_CONVKXK:
                block_m, block_n, block_k, warps, stages = _kxk_tiles(n_pix, op.n_in, op.n_out)
                args = [
                    src, op.weight, op.bias, dst, dst if res is None else res,
                    n_pix, op.n_in, op.n_out, height, w,
                    n_pix * src.stride(0), src.stride(0),
                    n_pix * dst.stride(0), dst.stride(0),
                    0 if res is None else res.stride(0),
                ]
                meta = dict(
                    BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, KSIZE=op.ksize, PAD=op.pad,
                    APPLY_SILU=op.silu, ADD_RESIDUAL=res is not None,
                    num_warps=warps, num_stages=stages,
                )
                grid = (-(-n_pix // block_m), -(-grid_n // block_n), batch)
                steps.append(_Step(_convkxk_kernel, grid, args, meta, f"conv{op.ksize}x{op.ksize}"))
                return

            if src_planar is not None:
                src_img, src_pix, src_chan = src_planar
                src_mode = _PLANAR_SRC_MODE
            else:
                src_img, src_pix, src_chan = n_pix * src.stride(0), src.stride(0), 1
                src_mode = 0
            if dst_planar is not None:
                dst_img, dst_pix, dst_chan = dst_planar
                dst_mode = _PLANAR_DST_MODE
            else:
                dst_img, dst_pix, dst_chan = n_pix * dst.stride(0), dst.stride(0), 1
                dst_mode = 0
            block_m, block_n, block_k, warps, stages = _conv1x1_tiles(n_pix, op.n_in, op.n_out)
            args = [
                src, op.weight, op.bias, dst, dst if res is None else res,
                n_pix, op.n_in, op.n_out,
                src_img, src_pix, src_chan,
                dst_img, dst_pix, dst_chan,
                0 if res is None else res.stride(0),
            ]
            meta = dict(
                BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
                SRC_MODE=src_mode, DST_MODE=dst_mode,
                APPLY_SILU=op.silu, ADD_RESIDUAL=res is not None,
                num_warps=warps, num_stages=stages,
            )
            grid = (-(-n_pix // block_m), -(-grid_n // block_n), batch)
            steps.append(_Step(_conv1x1_kernel, grid, args, meta, "conv1x1"))

        # Head: NCHW in, writes the first two concatenated pieces.
        in_strides = (head.n_in * n_pix, 1, n_pix)
        out_strides = (tail.n_out * n_pix, 1, n_pix)
        emit(head, static_in, concat[:, : head.n_out], None, src_planar=in_strides)

        # Blocks: piece `i` reads piece `i + 1` and appends piece `i + 2`.
        for i, (ops, add) in enumerate(blocks):
            src_view = concat[:, (1 + i) * carry : (2 + i) * carry]
            dst_view = concat[:, (2 + i) * carry : (3 + i) * carry]
            cursor = src_view
            for depth, op in enumerate(ops):
                last = depth == len(ops) - 1
                out_view = dst_view if last else slot(depth, op.n_out)
                emit(op, cursor, out_view, src_view if (last and add) else None)
                cursor = out_view

        # Tail: reads the whole buffer as one matrix, NCHW out.
        emit(tail, concat, static_out, None, dst_planar=out_strides)

        return _Plan(steps, buffers, static_in, static_out,
                     (batch, tail.n_out, height, w), kwargs, self._plan_key(x))

    def _capture(self, plan: _Plan) -> None:
        """Capture the chain, after warming it up eagerly so nothing JITs under capture."""
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            plan.bind(plan.static_in, plan.static_out)
            for _ in range(3):
                for step in plan.steps:
                    step.launch()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for step in plan.steps:
                step.launch()
        plan.graph = graph

    # -- dispatch -------------------------------------------------------------------

    def _plan_key(self, x: torch.Tensor):
        return (tuple(x.shape), x.dtype, x.device.index)

    def _fast_path_possible(self, x: torch.Tensor) -> bool:
        """Everything that can be decided without calling into CUDA."""
        if self._fk_unsupported or not _FAST_ENABLED or not _HAVE_TRITON:
            return False
        if self.training or torch.is_grad_enabled():
            return False
        if x.dtype is not torch.float16 or not x.is_cuda or x.dim() != 4 or not x.is_contiguous():
            return False
        if x.numel() == 0:
            # A zero-sized dimension would give an empty grid and an empty result, while the
            # reference path raises for a zero spatial extent. Defer to it either way.
            return False
        if x.shape[0] > 65535:
            # Batch is the third grid axis, which CUDA caps at 65535.
            return False
        return True

    def _fast_plan(self, x: torch.Tensor, capturing: bool) -> _Plan | None:

        weight = self.cv1.conv.weight
        if weight.dtype is not x.dtype or weight.device != x.device:
            return None
        if x.shape[1] != weight.shape[1]:
            return None
        stamp = (weight.dtype, weight.device, weight.data_ptr(), weight.shape)
        if stamp != self._fk_stamp:
            # A `.to()` that bypassed `_apply`, or a first call: rebuild from current weights.
            self._fk_chain = None
            self._fk_plan = None
            self._fk_stamp = stamp

        if self._fk_chain is None:
            chain = self._build_chain(x.dtype)
            if chain is None:
                # A property of the module, not of this input, so it will never be supported.
                self._fk_unsupported = True
                return None
            self._fk_chain = chain
            self._fk_widest_weight = max(
                (op.ksize * op.ksize * op.n_in * op.n_out
                 for op in [chain[0], chain[2]] + [o for ops, _ in chain[1] for o in ops]),
                default=0,
            )

        # The kernels index with 32-bit arithmetic, so a tensor large enough to overflow a signed
        # 32-bit element offset has to take the reference path. Nothing in the captured set comes
        # close -- the largest is 4.9 M elements against a 2.1 G limit -- but the bound is a
        # property of the addressing, not of the shapes that happen to be benchmarked, so it is
        # checked rather than assumed. This depends on the input, so it must not latch
        # `_fk_unsupported`.
        head, blocks, tail, width, _ = self._fk_chain
        _, _, height, w = x.shape
        n_pix = height * w
        widest = x.shape[0] * n_pix * max(width, x.shape[1], tail.n_out)
        if widest > _MAX_ELEMENT_INDEX:
            return None
        if self._fk_widest_weight > _MAX_ELEMENT_INDEX:
            # Weights are indexed as `tap * n_in * n_out + k * n_out + n`, which the activation
            # bound above does not cover.
            return None

        plan = self._fk_plan
        if plan is None or plan.key != self._plan_key(x):
            if capturing:
                # Building a plan means compiling and first-launching Triton kernels, which
                # must not happen inside somebody else's capture. Use the reference path for
                # this call rather than risking the enclosing graph.
                return None
            # Single-entry cache: a new shape replaces the old plan rather than accumulating
            # buffers and graphs, so memory stays bounded however many shapes are seen.
            self._fk_plan = None
            plan = self._build_plan(x, self._fk_chain)
            # `_build_plan` only allocates the static endpoints when it decided the chain is long
            # enough to be worth capturing, so their presence is the decision.
            if plan.static_in is not None:
                try:
                    self._capture(plan)
                except Exception:
                    # Capture is an optimization, never a requirement: fall back permanently
                    # to the ungraphed chain, which computes exactly the same values.
                    plan.graph = None
                    self._fk_graph_ok = False
            if plan.graph is None:
                # Warm the kernels outside any capture so a later call made under an
                # enclosing capture only has to launch them.
                plan.run_eager(x)
            self._fk_plan = plan
        return plan

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._fast_path_possible(x):
            # Checked before anything touches a CUDA API: on a CUDA build with no visible
            # device, even querying the capture state raises, and a CPU input is supposed to
            # reach the reference path rather than an error.
            return self._reference_forward(x)
        # Consulted on every call, not just at capture time: a call made inside an enclosing
        # capture must add its launches to that graph rather than replaying a nested one.
        capturing = torch.cuda.is_current_stream_capturing()
        plan = self._fast_plan(x, capturing)
        if plan is None:
            return self._reference_forward(x)
        if plan.graph is not None and not capturing:
            return plan.run_graph(x)
        return plan.run_eager(x)


class YOLOC2fCIB(YOLOC2f):
    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, lk: bool = False, g: int = 1, e: float = 0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(YOLOCIB(self.c, self.c, shortcut, e=1.0, lk=lk) for _ in range(n))
