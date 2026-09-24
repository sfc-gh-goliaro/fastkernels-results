"""YOLOv10 bottleneck block, fused into two kernels.

The baseline composes two ``YOLOConv`` blocks -- each ``conv3x3 -> BatchNorm ->
SiLU`` -- and adds the block input back when the shortcut applies. Measured on a
B200 that composition costs 17 kernel launches and 106-167 us per forward
(``profile/out_baseline.txt``), of which only 41-94 us is GPU time and only a
quarter to a third of *that* is convolution. The rest is NCHW<->NHWC layout
conversion for cuDNN, BatchNorm, SiLU, and the residual add. CPU-only time
matches wall time to within a few percent, so the host is on the critical path
and the launch count *is* the cost.

So this module attacks kernel count and layout traffic rather than convolution
throughput. In eval mode BatchNorm is an affine map from the running statistics,
which makes it foldable, and everything after the convolution is elementwise --
which makes it register work in a convolution epilogue. Each ``YOLOConv`` becomes
one Triton implicit-GEMM kernel that reads and writes NCHW directly, applies the
folded per-channel affine, SiLU, and the optional residual in registers, and
never materializes an intermediate. Seventeen launches become two.

Two launches under the bench's own timing protocol floor at 12.3 us against the
baseline's 106-167 us (``profile/out_launch_floor.txt``: ``triton x2: 12.3 us``,
next to ``triton x7: 32.8 us``), so the launch budget is what a two-kernel design
is worth here and the epilogue fusion is what keeps it at two.

The fast path is guarded, and everything the guard rejects reaches the baseline
composition -- the same ``YOLOConv`` submodules, called the same way -- so an
input or a module state the kernel cannot serve gets the reference result,
including the error the reference would raise.

Two properties this design depends on are invisible to the harness's own
correctness gate, and both are covered by ``tests/test_bottleneck.py`` instead:

* The correctness rounds compare a *cloned* input, and cloning a padded-batch-
  stride tensor returns a contiguous one, while the timing loop passes the
  original through unchanged. The largest benched case is
  ``empty_strided((4,16,160,160), (819200,25600,160,1))``, so a stride bug
  corrupts only the timed region -- whose output is never compared. This is why
  the residual carries its own four strides rather than reusing the output's.
* The BatchNorm state the bench runs is degenerate: it casts parameters to the run
  dtype but leaves buffers alone and resamples only parameters, so
  ``running_mean == 0`` and ``running_var == 1`` exactly and ``bn.weight == 1``,
  making the fold scale a uniform 0.9995004 on every channel. A fold that drops
  the mean term or indexes the statistics wrongly still passes the bench.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn
import triton
import triton.language as tl

from .yolov10_conv import YOLOConv

# Route tags, kept as small ints so per-call dispatch is an integer compare.
_FALLBACK = 0
_FUSED = 1

# Largest element offset the kernel forms in 32-bit arithmetic. The batch term is
# promoted to 64-bit on a scalar base, so only within-sample spans are checked
# against this; anything wider takes the reference path instead of wrapping.
_MAX_INT32_OFFSET = 2 ** 31 - 1


class Tile(NamedTuple):
    """How one convolution configuration is mapped onto the fused kernel."""

    block_oh: int      # output rows per program
    block_ow: int      # output columns per program (a power of two)
    block_oc: int      # output channels per program
    block_k: int       # slice of the flat C*KH*KW reduction held at once
    num_warps: int
    num_stages: int


# Tile shapes measured on this device with the bench's own timing protocol; the
# sweep and its output live in ``profile/probe_triton_tiles.py`` and
# ``profile/out_triton_tiles.{txt,csv}``. Keyed by the convolution configuration
# alone -- never by input shape -- so the table cannot grow at run time.
#
# The reduction extent is ``C*KH*KW``: 144, 288, 576, or 1152. None of those is a
# power of two, so ``block_k`` trades masked waste against chunk count and has to
# be swept per channel count rather than shared. ``block_oc`` is 16 at ``C = 16``
# because that is the smallest legal ``tl.dot`` M extent and 32 would leave half
# the M work masked off.
_TILES: dict[tuple, dict[torch.dtype, Tile]] = {
    # Ranked by the geometric mean of the two captured batch sizes rather than by
    # the benched one alone, because the two disagree. At C=128 the tile that is
    # fastest at N=1 is BLOCK_OC=16 (59.4 us, 1.86x) and it is a *regression* at
    # N=4 (155.5 us against a 112.6 us baseline, 0.72x): a small output-channel
    # block raises program count, which helps a 400-pixel grid that cannot fill 148
    # SMs, but each block re-reads the whole input tile -- eight times over at
    # BLOCK_OC=16. The frozen L1 kernel shows the same flip independently
    # (profile/out_triton_tiles.txt: BLOCK_OC=16 wins at N=1 and does not reach the
    # top four at N=4). So the entries below cost some of the benched N=1 case to
    # avoid being slower than the baseline on a captured one.
    (16, 16, 3, 3): {torch.float16: Tile(8, 32, 16, 256, 8, 1)},     # 5.93x / 3.31x
    (32, 32, 3, 3): {torch.float16: Tile(4, 16, 32, 128, 8, 1)},     # 3.69x / 2.22x
    (64, 64, 3, 3): {torch.float16: Tile(4, 8, 64, 256, 8, 3)},      # 2.74x / 2.07x
    (128, 128, 3, 3): {torch.float16: Tile(4, 8, 32, 128, 8, 1)},    # 1.36x / 1.37x
}
# The two speedups on each line are N=1 and N=4 at that channel count, against the
# baseline composition under the bench's timing protocol; the sweep that produced
# them is profile/probe_candidate_tiles.py and its output is
# profile/out_candidate_tiles.{txt,csv}. BLOCK_OC is 16 at C=16 because that is the
# smallest legal tl.dot M extent and 32 would leave half the M work masked off.
# There is no fp32 or bf16 entry: those dtypes take the reference path.


def _pair(value) -> tuple[int, int]:
    return (value, value) if isinstance(value, int) else tuple(value)


def _reads_as_stored(t: torch.Tensor) -> bool:
    """Whether a tensor's logical values equal the bytes in its storage.

    ``neg`` and ``conj`` views carry a lazy flag that ATen applies when it reads
    them, so the values a PyTorch op sees are not the values in memory. A kernel
    that dereferences the pointer never sees that flag, so such a view has to
    reach the reference path.
    """
    return not t.is_neg() and not t.is_conj()


def _unhooked(m) -> bool:
    """Whether *m* would do nothing but its own ``forward``.

    The fused path replaces the submodule calls rather than making them, so a hook
    registered on any of them would silently not run. Hooks cannot be detected by a
    tensor stamp, so this flag goes into the cache key and the miss path rejects on
    it. Hooks installed globally through ``nn.modules.module.register_module_*``
    apply to every module in the process and are not checked here.
    """
    return not m._forward_hooks and not m._forward_pre_hooks and not m._backward_hooks


def _reduction_blocks(reduction: int, requested: int) -> tuple[int, int]:
    """Reduction block size (a power of two, at least 16 for the MMA shape) and
    the number of chunks needed to cover *reduction*."""
    block = min(max(16, triton.next_power_of_2(requested)),
                max(16, triton.next_power_of_2(reduction)))
    return block, triton.cdiv(reduction, block)


def _stamp(t):
    """A cheap change stamp for one tensor a derived value depends on.

    Three reads, chosen so that everything a caller can actually do to a parameter
    or buffer changes at least one of them:

    * ``data_ptr`` catches new storage. This is the load-bearing one: the harness
      prepares a module with ``p.data = p.data.to(dtype)``, which installs a *new*
      storage on the same Parameter object and restarts the version counter at zero,
      so a stamp holding version 0 from the first call would otherwise match the
      stale entry.
    * ``_version`` catches an in-place write that keeps the storage --
      ``load_state_dict`` copying into place, ``running_var.mul_(3.0)``.
    * ``id`` catches replacement by a different tensor object that happens to land
      on a recycled address.

    Shape, stride, dtype and device are in here too, cheaply, because ``.data``
    assignment can rebind them while leaving all three of the above alone:
    ``w.data = w.data.view(torch.bfloat16)`` keeps the pointer, the identity, the
    version and the shape while changing the dtype, and for a square kernel
    ``w.data = w.data.transpose(2, 3)`` keeps the shape while changing the stride.
    Both would otherwise reach a cache *hit*, and the full metadata validation runs
    only on a miss -- which is also why ``device`` belongs here rather than being left
    to the input's own device check: the miss path is the only thing that compares a
    source tensor's device against the input's, so the device has to participate in
    the key that decides whether the miss path runs at all.

    What remains undetectable, and is not supported: writing to the storage through
    an aliasing view, and rewriting the running statistics by calling
    ``F.batch_norm`` on the buffers directly (or through a second BatchNorm module
    sharing them) rather than through this module's own submodule -- neither touches
    this tensor's version counter, and neither increments the
    ``num_batches_tracked`` this key relies on as its tripwire. Detecting either
    would mean hashing values on every call, which is not affordable on a host path
    this operator's whole design exists to shorten.
    """
    return None if t is None else (id(t), t.data_ptr(), t._version, t.dtype,
                                   t.device, t.shape, t.stride())


@triton.jit
def _fused_conv_affine_silu_nchw(
    x_ptr, w_ptr, shift_ptr, scale_ptr, res_ptr, y_ptr,
    in_h, in_w,
    x_sn, x_sc, x_sh, x_sw,
    y_sn, y_sc, y_sh, y_sw,
    r_sn, r_sc, r_sh, r_sw,
    w_so,
    CHANNELS: tl.constexpr, OUT_CHANNELS: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr, PH: tl.constexpr, PW: tl.constexpr,
    BLOCK_OH: tl.constexpr, BLOCK_OW: tl.constexpr, BLOCK_OC: tl.constexpr,
    BLOCK_K: tl.constexpr, K_CHUNKS: tl.constexpr, REDUCTION: tl.constexpr,
    HAS_SCALE: tl.constexpr, HAS_RESIDUAL: tl.constexpr, OC_BLOCKS: tl.constexpr,
):
    """One output tile of ``[BLOCK_OC, BLOCK_OH * BLOCK_OW]`` per program:
    ``silu(conv3x3(x) * scale + shift) (+ residual)``, all after the reduction and
    all in registers.

    The reduction runs over the flattened ``C*KH*KW`` extent in power-of-two
    chunks with a masked tail, decoding ``(c, kh, kw)`` from the flat index. The
    alternative -- looping the ``(kh, kw)`` taps outside a channel-blocked
    reduction, so tap offsets are compile-time constants -- was measured against
    this form in the frozen L1 conv2d winner and lost decisively on a 3x3
    sliding window (31.7 us against 52.3 us), because a flat reduction index *is*
    the offset within a packed filter, so the weight operand is read contiguously
    along the reduction instead of strided by ``KH*KW``.

    The accumulator is channel-major, so its fastest axis is the output column --
    the innermost NCHW axis for the gather, the store, and the residual alike.
    Offsets are 32-bit element counts added to a scalar base rather than blocks of
    64-bit pointers, because the address block is the largest register consumer in
    a gather kernel this shallow.

    The residual takes its own four strides. It is the *block input*, not a
    sibling of the output: the output is freshly allocated and contiguous, while
    the largest benched input has a batch stride of 819200 against the output's
    409600. Reading the residual through the output's batch stride would fetch the
    wrong data for every batch element after the first, and the bench compares
    only cloned -- hence contiguous -- inputs, so it would not notice.

    Stride is 1 on both axes, which is what ``YOLOConv`` is constructed with here,
    so the output extent equals the input extent and no stride multiply appears in
    the index arithmetic.
    """
    tile_ow = tl.program_id(0)
    tile_oh = tl.program_id(1)
    batch_oc = tl.program_id(2)
    n = batch_oc // OC_BLOCKS
    tile_oc = batch_oc % OC_BLOCKS

    pos = tl.arange(0, BLOCK_OH * BLOCK_OW)
    oh = tile_oh * BLOCK_OH + pos // BLOCK_OW
    ow = tile_ow * BLOCK_OW + pos % BLOCK_OW
    in_tile = (oh < in_h) & (ow < in_w)

    oc = tile_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_ok = oc < OUT_CHANNELS

    # Top-left input pixel of each output position, before the tap offset.
    ih0 = oh - PH
    iw0 = ow - PW
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
        ok = (k_ok[:, None] & in_tile[None, :]
              & (ih >= 0) & (ih < in_h) & (iw >= 0) & (iw < in_w))
        xv = tl.load(x_base + (c[:, None] * x_sc + ih * x_sh + iw * x_sw),
                     mask=ok, other=0.0)
        # Contiguous along the reduction: this is the whole point of flattening.
        wv = tl.load(w_ptr + (oc[:, None] * w_so + k[None, :]),
                     mask=oc_ok[:, None] & k_ok[None, :], other=0.0)
        acc = tl.dot(wv, xv, acc, input_precision="tf32")

    shift = tl.load(shift_ptr + oc, mask=oc_ok, other=0.0).to(tl.float32)[:, None]
    if HAS_SCALE:
        # BatchNorm applied to the fp32 accumulator, which is the order the
        # unfused reference uses and the comparison is made against.
        scale = tl.load(scale_ptr + oc, mask=oc_ok, other=0.0).to(tl.float32)[:, None]
        acc = acc * scale + shift
    else:
        acc = acc + shift
    acc = acc * tl.sigmoid(acc)
    if HAS_RESIDUAL:
        r_base = res_ptr + n.to(tl.int64) * r_sn
        r_off = oc[:, None] * r_sc + (oh[None, :] * r_sh + ow[None, :] * r_sw)
        acc += tl.load(r_base + r_off, mask=oc_ok[:, None] & in_tile[None, :],
                       other=0.0).to(tl.float32)

    y_off = oc[:, None] * y_sc + (oh[None, :] * y_sh + ow[None, :] * y_sw)
    tl.store(y_base + y_off, acc.to(y_ptr.dtype.element_ty),
             mask=oc_ok[:, None] & in_tile[None, :])


class YOLOBottleneck(nn.Module):
    """Two ``YOLOConv`` blocks and an optional residual, in two kernels.

    The submodules are built exactly as the baseline builds them, with the same
    arguments and therefore the same state-dict keys, so
    ``load_state_dict(baseline.state_dict())`` shares weights instead of silently
    leaving this module on its own initialization -- and so the reference path is
    the baseline composition itself rather than a reimplementation of it.
    """

    def __init__(self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k=(3, 3), e: float = 0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = YOLOConv(c1, c_, k[0], 1)
        self.cv2 = YOLOConv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

        # The harness fills weights *after* construction, so the route may depend
        # only on the configuration -- never on what the weights hold.
        kh, kw = _pair(k[0]), _pair(k[1])
        self.route, self._tiles = self._select_route(c1, c_, c2, kh, kw, g)
        # The configuration the route was chosen for. Every part of it is a plain
        # attribute a caller can reassign, and the reference reads all of them on
        # every call, so the route is only valid while they still hold.
        self._route_config = (c1, c_, c2, kh, kw, g, self.add)
        # What each submodule's convolution must still look like for the route to
        # hold: weight shape, stride, padding, dilation, groups. The channel counts
        # and kernel extent come from the weight's *shape* rather than from an
        # attribute, because the ``Conv2d`` behind ``YOLOConv`` records only stride,
        # padding, dilation, and groups -- and shape is configuration, not value.
        self._conv_configs = (
            (torch.Size((c_, c1) + kh), (kh[0] // 2, kh[1] // 2), 1),
            (torch.Size((c2, c_ // g) + kw), (kw[0] // 2, kw[1] // 2), g),
        )
        # Validated derived state per submodule, with the stamp it was derived at.
        self._state_cache: list[tuple | None] = [None, None]
        # Launch quantities per (shape, strides). Never keyed on ``data_ptr``:
        # the bench's shifting pool moves the input pointer every iteration, so a
        # pointer-keyed memo would miss on every timed call.
        self._launch_memo: dict[tuple, tuple | None] = {}

    # -- routing ----------------------------------------------------------
    @staticmethod
    def _select_route(c1, c_, c2, kh, kw, g):
        """A tag from the constructor arguments alone.

        Both convolutions must have a measured tile. Grouped convolution is not
        served: the kernel reduces over all ``CHANNELS`` for every output channel,
        which is only the convolution when there is one group.
        """
        if g != 1:
            return _FALLBACK, None
        first = _TILES.get((c1, c_) + kh)
        second = _TILES.get((c_, c2) + kw)
        if first is None or second is None:
            return _FALLBACK, None
        return _FUSED, (first, second)

    # -- reference path ---------------------------------------------------
    def _reference(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y

    # -- derived BatchNorm affine -----------------------------------------
    def _submodule_state(self, index: int, cv, x: torch.Tensor):
        """``(weight, shift, scale)`` for one submodule, or ``None`` if the fused
        path cannot serve it.

        This is both the derived-value cache and half the guard. Splitting them
        would mean reading the same six tensors twice per call, and the host is what
        this whole design is trying to shorten -- so the metadata checks a guard
        would repeat every call (shape, dtype, device, contiguity, lazy views) live
        on the miss path instead, behind a stamp that changes whenever any of them
        could have.

        ``scale`` is ``None`` once BatchNorm has been folded away, either by an
        earlier ``YOLOConv.fuse()`` -- which is what the end-to-end model path does,
        so it is the normal case rather than an exotic one -- or because there is
        nothing left to fold.

        Attributes are read through ``nn.Module``'s own ``_parameters``,
        ``_buffers``, and ``_modules`` dicts rather than by attribute access.
        ``nn.Module.__getattr__`` is a Python-level function that runs on every
        submodule, parameter, and buffer read, and there are twenty of those per
        call here; measured, going through it cost about 6 us of the 12.5 us the
        guard used to take. ``__setattr__`` keeps these dicts authoritative for
        exactly these names -- assigning a tensor to a registered parameter or
        buffer name routes into them, and assigning a non-module to a registered
        submodule name raises -- so this reads what attribute access would return.
        """
        conv = cv._modules["conv"]
        weight = conv._parameters["weight"]
        # ``bias`` is absent from ``_parameters`` while the convolution is unbiased,
        # because assigning ``None`` before registration lands in ``__dict__``.
        bias = conv._parameters.get("bias")
        bn = cv._modules.get("bn")
        act = cv._modules["act"]
        clean = _unhooked(cv) and _unhooked(conv) and _unhooked(act)
        if bn is None or cv.__dict__.get("_is_fused"):
            key = (x.dtype, x.device, weight.shape, conv.stride, conv.padding,
                   conv.dilation, conv.groups, act.__class__, clean,
                   _stamp(weight), _stamp(bias), bn is None,
                   bool(cv.__dict__.get("_is_fused")))
        else:
            bnd = bn.__dict__
            bnp, bnb = bn._parameters, bn._buffers
            gamma, beta = bnp.get("weight"), bnp.get("bias")
            mean, var = bnb.get("running_mean"), bnb.get("running_var")
            key = (x.dtype, x.device, weight.shape, conv.stride, conv.padding,
                   conv.dilation, conv.groups, act.__class__,
                   clean and _unhooked(bn), _stamp(weight), _stamp(bias), False, False,
                   _stamp(gamma), _stamp(beta), _stamp(mean), _stamp(var),
                   # A train-mode forward rewrites the running statistics in place,
                   # and -- measured -- ``F.batch_norm`` does that below the version
                   # counter: the buffers' contents change while their pointer,
                   # identity, and ``_version`` all stay put, so the stamps above
                   # cannot see it. ``BatchNorm2d.forward`` increments this counter on
                   # every training call, and ``add_`` does bump a version, so it is
                   # the tripwire for "these statistics have moved since". Without it
                   # a train()/eval() round trip serves an eval result derived from
                   # the pre-training statistics.
                   _stamp(bnb.get("num_batches_tracked")),
                   bnd["eps"], bnd["training"], bnd["track_running_stats"])
        cached = self._state_cache[index]
        if cached is not None and cached[0] == key:
            return cached[1]
        value = self._derive(index, cv, conv, weight, bias, bn, x)
        self._state_cache[index] = (key, value)
        return value

    def _derive(self, index: int, cv, conv, weight, bias, bn, x: torch.Tensor):
        """Validate one submodule and compute its derived affine. Runs on a stamp
        change only, so it may be as thorough as it likes."""
        shape, pad, groups = self._conv_configs[index]
        if ((weight.shape, conv.stride, conv.padding, conv.dilation, conv.groups)
                != (shape, (1, 1), pad, (1, 1), groups)):
            return None
        # The kernel reaches the weight as raw memory, so a replacement parameter the
        # reference would reject must not get that far. Contiguity is load-bearing
        # besides: a flat reduction index only equals the offset within a filter when
        # the filter is packed.
        if (weight.dtype != x.dtype or weight.device != x.device
                or not weight.is_contiguous() or not _reads_as_stored(weight)):
            return None
        # The activation is applied in the epilogue as SiLU, so a submodule whose
        # activation was replaced has to reach the reference path. Identity rather
        # than ``isinstance``: the name behind ``from .yolov10_conv import YOLOConv``
        # resolves into the baseline package, so the class the submodules actually
        # hold is the baseline's, not any candidate's.
        conv_act = cv._modules["act"]
        if conv_act.__class__ is not type(YOLOConv.default_act):
            return None
        # The fused path replaces the submodule calls rather than making them, so a
        # hook on any of them would silently not run.
        if not (_unhooked(cv) and _unhooked(conv) and _unhooked(conv_act)):
            return None
        channels = shape[0]
        # The bias dtype is checked because ``F.conv2d`` rejects a bias that does not
        # match the weight, while the kernel would happily widen whatever it is
        # handed -- so without this the fused path would succeed where the reference
        # raises.
        if bias is not None and (bias.shape != (channels,)
                                 or bias.dtype != x.dtype
                                 or bias.device != x.device
                                 or not bias.is_contiguous()
                                 or not _reads_as_stored(bias)):
            return None
        if bn is None:
            # ``fuse()`` deletes ``bn`` and sets ``_is_fused``. If the flag has since
            # been cleared, the reference path will try to call a BatchNorm that is
            # not there; serving that from the fused path would answer where the
            # baseline raises.
            if not cv.__dict__.get("_is_fused") or bias is None:
                return None
            return weight, bias, None
        if cv.__dict__.get("_is_fused"):
            # Folded, with ``bn`` still attached: the reference ignores it too.
            if bias is None:
                return None
            return weight, bias, None
        if not (_unhooked(bn) and bn.track_running_stats):
            return None
        # The fold is only valid against the running statistics. Grad being disabled
        # says nothing about this: a module left in train mode has to reach the
        # reference path, which will use batch statistics.
        if bn.training:
            return None
        gamma, beta = bn._parameters.get("weight"), bn._parameters.get("bias")
        mean, var = bn._buffers.get("running_mean"), bn._buffers.get("running_var")
        if gamma is None or beta is None or mean is None or var is None:
            return None
        # Only the dtype combinations whose agreement with ``F.batch_norm`` was
        # checked are served. The harness leaves a heterogeneous one behind -- it
        # casts parameters to the run dtype and leaves buffers in fp32 -- so this
        # cannot demand a single dtype; but it also must not fold a combination
        # ``F.batch_norm`` would have rejected. Anything else takes the reference
        # path, which answers exactly as the baseline does.
        allowed = (x.dtype, torch.float32)
        for t in (gamma, beta, mean, var):
            if (t.shape != (channels,) or t.device != x.device
                    or t.dtype not in allowed
                    or not t.is_contiguous() or not _reads_as_stored(t)):
                return None
        # Derived in fp32 with a single rounding at the end, which is closer to the
        # unfused reference than the reference fuse helper's fp16 division and square
        # root. The vectors stay fp32: they are 2C floats, the kernel widens them to
        # fp32 anyway, and rounding them to the parameter dtype would only move the
        # result away from what the comparison is made against.
        with torch.no_grad():
            scale = gamma.float() * torch.rsqrt(var.float() + bn.eps)
            shift = beta.float() - scale * mean.float()
            if bias is not None:
                shift = shift + scale * bias.float()
        return weight, shift.contiguous(), scale.contiguous()

    # -- launch -----------------------------------------------------------
    @staticmethod
    def _span(shape, strides, channels, kh, kw) -> int:
        """Largest within-sample element offset a gather over *shape* can form.

        The batch term is promoted to 64-bit on a scalar base inside the kernel, so
        only this part has to fit in 32 bits. Strides come from the tensor rather
        than from the shape, because a padded layout and a replacement parameter
        both lay out differently from a freshly allocated one.
        """
        _, c_stride, h_stride, w_stride = (abs(v) for v in strides)
        return ((channels - 1) * c_stride + (shape[2] + kh) * h_stride
                + (shape[3] + kw) * w_stride)

    def _launch_plan(self, shape, strides, out_channels: int, tile: Tile,
                     in_channels: int, kh: int, kw: int, res_span: int = 0):
        """Grid and reduction blocking for one convolution on this input.

        Returns ``None`` when any offset the kernel would form is too wide for its
        32-bit arithmetic, so the call takes the reference path rather than
        producing a wrapped-offset result.
        """
        n, _, in_h, in_w = shape
        span_out = out_channels * in_h * in_w
        span_w = out_channels * in_channels * kh * kw
        if max(self._span(shape, strides, in_channels, kh, kw),
               span_out, span_w, res_span) > _MAX_INT32_OFFSET:
            return None
        reduction = in_channels * kh * kw
        block_k, k_chunks = _reduction_blocks(reduction, tile.block_k)
        oc_blocks = triton.cdiv(out_channels, tile.block_oc)
        grid = (triton.cdiv(in_w, tile.block_ow), triton.cdiv(in_h, tile.block_oh),
                n * oc_blocks)
        return grid, block_k, k_chunks, reduction, oc_blocks

    def _plans(self, x: torch.Tensor):
        """Both launch plans, memoized on shape and strides.

        Never on ``data_ptr``: the bench's shifting pool moves the input pointer on
        every timed iteration, so a pointer-keyed memo would miss every time and
        would grow without bound besides.
        """
        shape, strides = tuple(x.shape), tuple(x.stride())
        memo_key = (shape, strides)
        plans = self._launch_memo.get(memo_key, ())
        if plans != ():
            return plans
        c1, c_, c2, kh, kw, _g, add = self._route_config
        tile1, tile2 = self._tiles[0][x.dtype], self._tiles[1][x.dtype]
        n, _, in_h, in_w = shape
        # The intermediate is freshly allocated and contiguous, so its strides are
        # known here rather than read off a tensor that does not exist yet.
        mid_shape = (n, c_, in_h, in_w)
        mid_strides = (c_ * in_h * in_w, in_h * in_w, in_w, 1)
        # The residual is the *block input*, not a sibling of the output: it carries
        # its own strides, and its span has to be inside the budget too.
        res_span = self._span(shape, strides, c2, kw[0], kw[1]) if add else 0
        p1 = self._launch_plan(shape, strides, c_, tile1, c1, kh[0], kh[1])
        p2 = self._launch_plan(mid_shape, mid_strides, c2, tile2, c_, kw[0], kw[1],
                               res_span=res_span)
        plans = None if p1 is None or p2 is None else (p1, p2)
        self._launch_memo[memo_key] = plans
        return plans

    def _fused(self, x: torch.Tensor, state1, state2, plans) -> torch.Tensor:
        _c1, c_, c2, kh, kw, _g, add = self._route_config
        tile1, tile2 = self._tiles[0][x.dtype], self._tiles[1][x.dtype]
        w1, shift1, scale1 = state1
        w2, shift2, scale2 = state2
        p1, p2 = plans
        n, c1, in_h, in_w = x.shape
        mid = torch.empty((n, c_, in_h, in_w), dtype=x.dtype, device=x.device)
        out = torch.empty((n, c2, in_h, in_w), dtype=x.dtype, device=x.device)

        grid, block_k, k_chunks, reduction, oc_blocks = p1
        _fused_conv_affine_silu_nchw[grid](
            x, w1, shift1, scale1 if scale1 is not None else shift1, x, mid,
            in_h, in_w,
            *x.stride(), *mid.stride(), *x.stride(),
            w1.stride(0),
            CHANNELS=c1, OUT_CHANNELS=c_, KH=kh[0], KW=kh[1],
            PH=kh[0] // 2, PW=kh[1] // 2,
            BLOCK_OH=tile1.block_oh, BLOCK_OW=tile1.block_ow,
            BLOCK_OC=tile1.block_oc, BLOCK_K=block_k, K_CHUNKS=k_chunks,
            REDUCTION=reduction, HAS_SCALE=scale1 is not None,
            HAS_RESIDUAL=False, OC_BLOCKS=oc_blocks,
            num_warps=tile1.num_warps, num_stages=tile1.num_stages,
        )
        grid, block_k, k_chunks, reduction, oc_blocks = p2
        _fused_conv_affine_silu_nchw[grid](
            mid, w2, shift2, scale2 if scale2 is not None else shift2, x, out,
            in_h, in_w,
            *mid.stride(), *out.stride(), *x.stride(),
            w2.stride(0),
            CHANNELS=c_, OUT_CHANNELS=c2, KH=kw[0], KW=kw[1],
            PH=kw[0] // 2, PW=kw[1] // 2,
            BLOCK_OH=tile2.block_oh, BLOCK_OW=tile2.block_ow,
            BLOCK_OC=tile2.block_oc, BLOCK_K=block_k, K_CHUNKS=k_chunks,
            REDUCTION=reduction, HAS_SCALE=scale2 is not None,
            HAS_RESIDUAL=add, OC_BLOCKS=oc_blocks,
            num_warps=tile2.num_warps, num_stages=tile2.num_stages,
        )
        return out

    # -- guard ------------------------------------------------------------
    def _admits(self, x: torch.Tensor) -> bool:
        """Everything about the *call* the fused path needs, in eight reads.

        The module's own state is checked by ``_submodule_state``, which has to read
        those tensors anyway. What is left here depends on the input and on the
        ambient mode, so none of it can be cached across calls.

        One mode is knowingly not covered: forward-mode automatic differentiation.
        ``torch.is_grad_enabled()`` is a reverse-mode flag, and a dual tensor is not
        cheaply distinguishable from an ordinary one -- ``_is_fwd_grad_enabled()``
        reads true outside any dual level, and ``unpack_dual`` costs more than the
        whole guard. A dual input would therefore reach the kernel and lose its
        tangent. Nothing in this workload uses forward-mode AD.
        """
        if self.route == _FALLBACK or x.dim() != 4 or not x.is_cuda:
            return False
        # A unit innermost stride is what keeps the gather on the fast axis.
        # Contiguity is deliberately *not* required: the largest benched input has a
        # padded batch stride, and the kernel takes all four strides as runtime
        # arguments, so requiring contiguity would forfeit that case.
        if x.stride(-1) != 1:
            return False
        # Forward-only, and below the dispatcher that implements autocast: a call
        # that builds a graph or runs under autocast reaches the baseline
        # composition, which reproduces those modes by construction.
        if torch.is_grad_enabled() or torch.is_autocast_enabled("cuda"):
            return False
        # The kernel reduces over exactly ``c1`` channels, and the residual is added
        # over ``c2`` of them; a disagreeing input must not be indexed past its
        # logical extent.
        if x.shape[1] != self._route_config[0]:
            return False
        # ``self.add`` decides whether the residual is an epilogue term, and it is a
        # plain attribute the reference path reads on every call -- so a caller that
        # reassigned it since construction would move the baseline's answer and not
        # the kernel's.
        if self.add is not self._route_config[6]:
            return False
        # A lazily negated or conjugated view would be read as its raw storage.
        return _reads_as_stored(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._admits(x) and x.dtype in self._tiles[0] and x.dtype in self._tiles[1]:
            state1 = self._submodule_state(0, self.cv1, x)
            if state1 is not None:
                state2 = self._submodule_state(1, self.cv2, x)
                if state2 is not None:
                    plans = self._plans(x)
                    if plans is not None:
                        return self._fused(x, state1, state2, plans)
        return self._reference(x)
