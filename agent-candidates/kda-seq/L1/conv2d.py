"""Conv2d that dispatches on configuration.

The captured workload is latency-bound rather than throughput-bound: the benched
inputs are small enough that the number of kernels launched inside the measured
window dominates the cost. So each configuration is routed once, at construction
time, to the cheapest formulation that is exact for it:

* ``1x1``, unit stride, no padding, no dilation, one group, no bias becomes a
  single cuBLAS matmul -- the convolution *is* a GEMM there, and both the input
  reshape and the weight reshape are free views.
* A small set of measured sliding-window and patch-embed configurations goes to
  a fused implicit-GEMM Triton kernel that reads NCHW directly, so there is no
  im2col buffer and no weight repack.
* Everything else goes to ``F.conv2d`` unchanged.

The route *tag* comes from constructor arguments alone. Weight values never enter
the decision, which is what keeps the tag stable across ``load_state_dict`` and
in-place weight edits without a cache keyed on weight identity.

Taking that route on a given call is a separate question, answered by
``_optimized_route_applies``. Everything it checks is there because the module must
behave exactly like ``baseline.py``, which reads its attributes and parameters on
every call and dispatches through ATen:

* the configuration may have been reassigned since construction, and the
  parameters may have been replaced with something of another shape, dtype, device
  or layout;
* the optimized routes are forward-only and reach memory by pointer, so a call
  that builds a graph or runs under autocast is handed back to ``F.conv2d``, which
  reproduces those modes by construction;
* one configuration additionally restricts itself to input shapes whose agreement
  with the reference was measured -- see ``Tile.validated_inputs``.

Anything the guard rejects gets the reference result, including the error the
reference would raise.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# Route tags. Kept as small ints so the per-call dispatch is an integer compare.
_FALLBACK = 0
_POINTWISE = 1
_IMPLICIT_GEMM = 2

# Largest element offset the kernel forms in 32-bit arithmetic. Offsets are kept
# narrow deliberately (see the kernel docstring); anything that could exceed this
# takes the reference path instead of silently wrapping.
_MAX_INT32_OFFSET = 2 ** 31 - 1


class Tile(NamedTuple):
    """How one configuration is mapped onto the implicit-GEMM kernel."""

    block_oh: int          # output rows per program
    block_ow: int          # output columns per program (a power of two)
    block_oc: int          # output channels per program
    block_k: int           # slice of the flat C*KH*KW reduction held at once
    num_warps: int
    num_stages: int
    precision: str         # tl.dot input_precision; only meaningful for fp32
    # Input shapes whose agreement with the reference has actually been measured.
    # ``None`` means every input the route can serve. A fixed tuple written here
    # is not a cache and cannot grow: it exists because for a tight tolerance the
    # reference's own choice of algorithm becomes observable, so a route may only
    # be taken where that agreement was checked (see the fp32 note below).
    validated_inputs: tuple[tuple[int, int, int, int], ...] | None = None


# Tile shapes for the configurations that were measured to beat ``F.conv2d`` on
# this device. Keyed by the configuration alone -- never by input shape -- so the
# table cannot grow at run time. ``block_oh * block_ow`` must be a power of two
# because it forms one axis of the tile.
_TILES: dict[tuple, dict[torch.dtype, Tile]] = {
    # 16 -> 1024, 2x2 stride 2, no padding, with bias: a non-overlapping patch
    # embed. Computed in true IEEE fp32, which on this kernel is bit-identical to
    # the cuDNN reference and still 1.27x faster than it. (An earlier tap-outer
    # kernel could only reach 1.000x under `ieee`, which is why the flat reduction
    # matters here and not just for fp16.)
    #
    # This is the one configuration with an explicit validated-input list, and the
    # reason is the reference rather than the kernel. At fp32's atol=1e-5 the
    # reference's own algorithm choice is observable in the comparison: cuDNN
    # answers this configuration with an exact fp32 plan at batch 1-2 but a TF32
    # plan from batch 3 up (measured 3.0e-4 from float64, only 0.869 of elements
    # inside tolerance). Since "correct" here means "agrees with cuDNN", an
    # accurate kernel is judged wrong exactly where cuDNN is imprecise. So the
    # custom path is admitted only for the input classes whose agreement was
    # measured, and the captured batch-3..6 variants take the reference path.
    (16, 1024, 2, 2, 2, 2, 0, 0, True): {
        torch.float32: Tile(1, 16, 128, 32, 2, 3, "ieee",
                            validated_inputs=((1, 16, 18, 32), (2, 16, 18, 32)))},
    # 16 -> 32, 3x3 stride 2 pad 1: the large sliding-window case.
    (16, 32, 3, 3, 2, 2, 1, 1, False): {
        torch.float16: Tile(8, 32, 32, 16, 4, 1, "tf32")},
    # 64 -> 64, 3x3 stride 1 pad 1: the small sliding-window case.
    (64, 64, 3, 3, 1, 1, 1, 1, False): {
        torch.float16: Tile(4, 8, 32, 256, 8, 2, "tf32")},
}
# The fp16 entries carry no validated-input list because fp16's atol=1e-2 is far
# wider than any algorithm-choice difference: every captured shape of both
# configurations agrees with the reference, which the full-capture sweep checks.


def _pair(value: int | tuple[int, int]) -> tuple[int, int]:
    return (value, value) if isinstance(value, int) else value


def _reads_as_stored(t: torch.Tensor) -> bool:
    """Whether a tensor's logical values equal the bytes in its storage.

    ``neg`` and ``conj`` views carry a lazy flag that ATen applies when it reads
    them, so the values a PyTorch op sees are not the values in memory. A kernel
    that dereferences the pointer itself never sees that flag -- measured: a
    negative-bit input produced a sign-flipped result -- so such a view must reach
    the reference path instead.
    """
    return not t.is_neg() and not t.is_conj()


def _reduction_blocks(reduction: int, requested: int) -> tuple[int, int]:
    """Reduction block size (a power of two, at least 16 for the MMA shape) and
    the number of chunks needed to cover *reduction*."""
    block = min(max(16, triton.next_power_of_2(requested)),
                max(16, triton.next_power_of_2(reduction)))
    return block, triton.cdiv(reduction, block)


@triton.jit
def _implicit_gemm_nchw(
    x_ptr, w_ptr, bias_ptr, y_ptr,
    in_h, in_w, out_h, out_w,
    x_sn, x_sc, x_sh, x_sw,
    y_sn, y_sc, y_sh, y_sw,
    w_so,
    CHANNELS: tl.constexpr, OUT_CHANNELS: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr,
    SH: tl.constexpr, SW: tl.constexpr,
    PH: tl.constexpr, PW: tl.constexpr,
    BLOCK_OH: tl.constexpr, BLOCK_OW: tl.constexpr, BLOCK_OC: tl.constexpr,
    BLOCK_K: tl.constexpr, K_CHUNKS: tl.constexpr, REDUCTION: tl.constexpr,
    HAS_BIAS: tl.constexpr, PRECISION: tl.constexpr, OC_BLOCKS: tl.constexpr,
):
    """One output tile of ``[BLOCK_OC, BLOCK_OH * BLOCK_OW]`` per program.

    The reduction runs over the flattened ``C*KH*KW`` extent in power-of-two
    chunks with a masked tail, decoding ``(c, kh, kw)`` from the flat index. The
    alternative -- looping the ``(kh, kw)`` taps outside a channel-blocked
    reduction, so the tap offsets are compile-time constants -- was implemented and
    measured against this one at matched tiles (``tests/compare_reductions.py``).
    Flattening won on the patch embed (13.3us vs 15.3us) and decisively on the
    large sliding-window case (31.7us vs 52.3us), and tied on the small one, so it
    is what ships.

    The reason is the weight operand. A flat reduction index *is* the offset within
    a packed filter, so ``w`` is read contiguously along the reduction; the
    tap-outer form instead reads it strided by ``KH*KW``. That outweighs the extra
    per-lane index arithmetic the flat form pays, which is why this had to be
    measured rather than argued -- the arithmetic cost is the visible one.

    The accumulator is channel-major so its fastest axis is the output column
    ``ow``, the innermost NCHW axis for both the gather and the store. Offsets are
    32-bit element counts added to a scalar base pointer rather than blocks of
    64-bit pointers: the address block is the largest register consumer in a gather
    kernel this shallow.
    """
    tile_ow = tl.program_id(0)
    tile_oh = tl.program_id(1)
    batch_oc = tl.program_id(2)
    n = batch_oc // OC_BLOCKS
    tile_oc = batch_oc % OC_BLOCKS

    pos = tl.arange(0, BLOCK_OH * BLOCK_OW)
    oh = tile_oh * BLOCK_OH + pos // BLOCK_OW
    ow = tile_ow * BLOCK_OW + pos % BLOCK_OW
    in_tile = (oh < out_h) & (ow < out_w)

    oc = tile_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_ok = oc < OUT_CHANNELS

    # Top-left input pixel of each output position, before the tap offset.
    ih0 = oh * SH - PH
    iw0 = ow * SW - PW
    x_base = x_ptr + n.to(tl.int64) * x_sn
    y_base = y_ptr + n.to(tl.int64) * y_sn

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
            # output position is in range, so bounds are implied by ``in_tile``.
            ok = k_ok[:, None] & in_tile[None, :]
        else:
            ok = (k_ok[:, None] & in_tile[None, :]
                  & (ih >= 0) & (ih < in_h) & (iw >= 0) & (iw < in_w))
        xv = tl.load(x_base + (c[:, None] * x_sc + ih * x_sh + iw * x_sw),
                     mask=ok, other=0.0)
        # Contiguous along the reduction: this is the whole point of flattening.
        wv = tl.load(w_ptr + (oc[:, None] * w_so + k[None, :]),
                     mask=oc_ok[:, None] & k_ok[None, :], other=0.0)
        acc = tl.dot(wv, xv, acc, input_precision=PRECISION)

    if HAS_BIAS:
        acc += tl.load(bias_ptr + oc, mask=oc_ok, other=0.0).to(tl.float32)[:, None]

    y_off = oc[:, None] * y_sc + (oh[None, :] * y_sh + ow[None, :] * y_sw)
    tl.store(y_base + y_off, acc.to(y_ptr.dtype.element_ty),
             mask=oc_ok[:, None] & in_tile[None, :])


class Conv2d(nn.Module):
    """Parametric 2D convolution: stores weight and bias internally."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        groups: int = 1,
        dilation: int | tuple[int, int] = 1,
        bias: bool = True,
    ):
        super().__init__()
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.dilation = dilation

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, *kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        # The harness fills the weight *after* construction, so the route may
        # depend only on the configuration above -- never on what weight holds.
        self.route, self.tiles = self._select_route(bias)
        # The configuration the route was chosen for. Every one of these is a
        # plain attribute a caller can reassign afterwards, and the reference
        # reads all of them on every call, so the route is only valid while they
        # still hold. Shape is derived from the arguments rather than read off
        # the parameter, which holds uninitialized memory at this point.
        self._route_config = (in_channels, out_channels, kernel_size, stride,
                              padding, dilation, groups, bias)
        self._weight_shape = (out_channels, in_channels // groups) + kernel_size

    # -- routing ----------------------------------------------------------
    def _select_route(self, has_bias: bool):
        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding
        if self.groups == 1 and self.dilation == (1, 1):
            if (kh, kw) == (1, 1) and (sh, sw) == (1, 1) and (ph, pw) == (0, 0) \
                    and not has_bias:
                # Bias would need a second kernel here, so a biased 1x1 stays on
                # the fallback rather than paying an extra launch.
                return _POINTWISE, None
            tiles = _TILES.get(
                (self.in_channels, self.out_channels, kh, kw, sh, sw, ph, pw,
                 has_bias))
            if tiles is not None:
                return _IMPLICIT_GEMM, tiles
        return _FALLBACK, None

    def _conv2d(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            x,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

    # -- optimized routes -------------------------------------------------
    def _pointwise(self, x: torch.Tensor) -> torch.Tensor:
        """``out[n,o,p] = sum_c weight[o,c] * x[n,c,p]`` as a single matmul."""
        n, c, h, w = x.shape
        x_sn, x_sc, x_sh, x_sw = x.stride()
        hw = h * w
        # Merging (h, w) is only a view when the spatial block is packed. The
        # batch stride may still be padded, which is how the captured
        # non-contiguous inputs are laid out.
        if x_sw != 1 or x_sh != w or x_sc != hw:
            return self._conv2d(x)
        flat = x.as_strided((n, c, hw), (x_sn, x_sc, 1))
        # A weight still carrying ``requires_grad`` sends the 2-D by 3-D matmul
        # down a path that materializes the broadcast operand, costing two extra
        # elementwise kernels. Detaching is safe here only because this route is
        # entered with grad disabled.
        weight = self.weight.detach().view(self.out_channels, c)
        if n == 1:
            out = torch.mm(weight, flat[0])
        else:
            # A 2-D operand broadcasts across the batch without materializing an
            # expanded copy, so this stays one kernel.
            out = torch.matmul(weight, flat)
        return out.view(n, self.out_channels, h, w)

    def _fits_narrow_offsets(self, x: torch.Tensor, out_h: int, out_w: int) -> bool:
        """Can every element offset the kernel forms be held in 32 bits?

        The batch term is promoted to 64 bits inside the kernel, so only the
        within-sample spans matter. Weight and input spans come from the tensors'
        real strides, because a replacement parameter may be laid out differently
        from a freshly constructed one.
        """
        _, x_sc, x_sh, x_sw = (abs(v) for v in x.stride())
        kh, kw = self.kernel_size
        span_in = ((self.in_channels - 1) * x_sc + (x.shape[2] + kh) * x_sh
                   + (x.shape[3] + kw) * x_sw)
        span_out = self.out_channels * out_h * out_w
        span_w = 1 + sum((d - 1) * abs(v)
                         for d, v in zip(self.weight.shape, self.weight.stride()))
        return max(span_in, span_out, span_w) <= _MAX_INT32_OFFSET

    def _implicit_gemm(self, x: torch.Tensor, tile: Tile) -> torch.Tensor:
        n, _, in_h, in_w = x.shape
        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding
        out_h = (in_h + 2 * ph - kh) // sh + 1
        out_w = (in_w + 2 * pw - kw) // sw + 1
        if out_h <= 0 or out_w <= 0 or not self._fits_narrow_offsets(x, out_h, out_w):
            # An empty or degenerate result is the reference path's business, so
            # that whatever it raises is what the baseline raises.
            return self._conv2d(x)
        reduction = self.in_channels * kh * kw
        block_k, k_chunks = _reduction_blocks(reduction, tile.block_k)
        oc_blocks = triton.cdiv(self.out_channels, tile.block_oc)
        y = torch.empty((n, self.out_channels, out_h, out_w),
                        dtype=x.dtype, device=x.device)
        grid = (triton.cdiv(out_w, tile.block_ow), triton.cdiv(out_h, tile.block_oh),
                n * oc_blocks)
        _implicit_gemm_nchw[grid](
            x, self.weight, self.bias if self.bias is not None else self.weight, y,
            in_h, in_w, out_h, out_w,
            *x.stride(), *y.stride(), self.weight.stride(0),
            CHANNELS=self.in_channels, OUT_CHANNELS=self.out_channels,
            KH=kh, KW=kw, SH=sh, SW=sw, PH=ph, PW=pw,
            BLOCK_OH=tile.block_oh, BLOCK_OW=tile.block_ow,
            BLOCK_OC=tile.block_oc, BLOCK_K=block_k, K_CHUNKS=k_chunks,
            REDUCTION=reduction,
            HAS_BIAS=self.bias is not None,
            # Single-pass TF32 is not accurate enough for the fp32 tolerance
            # (~6.6e-4 absolute against atol=1e-5), so an fp32 configuration
            # names the precision it was validated with.
            PRECISION=tile.precision, OC_BLOCKS=oc_blocks,
            num_warps=tile.num_warps, num_stages=tile.num_stages,
        )
        return y

    def _optimized_route_applies(self, x: torch.Tensor) -> bool:
        """Whether the configuration-derived route is valid for *this* call.

        Anything not covered here reaches ``F.conv2d``, so an input or a module
        state the optimized routes cannot serve gets the reference result --
        including the error the reference would raise.
        """
        if self.route == _FALLBACK or x.dim() != 4 or not x.is_cuda:
            return False
        # Both optimized routes are forward-only. Building a graph is left to
        # F.conv2d so that ``backward()`` keeps working.
        if torch.is_grad_enabled():
            return False
        # Under autocast the reference casts its operands and returns the autocast
        # dtype. The Triton route reaches memory by pointer, below the dispatcher
        # that implements that, so it would return the input's dtype instead --
        # measured: a reference returning float16 against a route returning
        # float32. Handing autocast back to F.conv2d keeps the dtype the
        # baseline's.
        if torch.is_autocast_enabled("cuda"):
            return False
        # The route was chosen for one configuration; a caller may have
        # reassigned any part of it since. ``in_channels`` and ``out_channels`` are
        # in here because the kernel takes them as its reduction extent and its
        # store extent: reassigning one while leaving the weight alone would
        # otherwise admit a launch whose extents disagree with the weight.
        if (self.in_channels, self.out_channels, self.kernel_size, self.stride,
                self.padding, self.dilation, self.groups,
                self.bias is not None) != self._route_config:
            return False
        # The routes read exactly ``in_channels`` channels; a disagreeing input
        # must not be indexed past its logical extent.
        if x.shape[1] != self.in_channels:
            return False
        # A lazily negated or conjugated view would be read as its raw storage.
        if not _reads_as_stored(x):
            return False
        # Both routes reach the parameters as raw memory -- the Triton kernel by
        # pointer, the pointwise route through a reshape -- so a replacement
        # parameter that the reference would reject must not get that far. The
        # contiguity requirement is load-bearing for the kernel too: a flat
        # reduction index only equals the offset within the filter when it is
        # packed.
        weight = self.weight
        if (tuple(weight.shape) != self._weight_shape
                or weight.dtype != x.dtype
                or weight.device != x.device
                or not weight.is_contiguous()
                or not _reads_as_stored(weight)):
            return False
        bias = self.bias
        if bias is not None and (
                tuple(bias.shape) != (self.out_channels,)
                or bias.dtype != x.dtype
                or bias.device != x.device
                or not bias.is_contiguous()
                or not _reads_as_stored(bias)):
            return False
        return True

    @staticmethod
    def _tile_admits(tile: Tile, x: torch.Tensor) -> bool:
        """Whether this tile was validated for an input of exactly this shape."""
        return (tile.validated_inputs is None
                or tuple(x.shape) in tile.validated_inputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._optimized_route_applies(x):
            if self.route == _POINTWISE:
                return self._pointwise(x)
            tile = self.tiles.get(x.dtype)
            # A unit innermost stride is what keeps the gather on the fast axis;
            # every other layout keeps the reference path.
            if (tile is not None and x.stride(-1) == 1
                    and self._tile_admits(tile, x)):
                return self._implicit_gemm(x, tile)
        return self._conv2d(x)
