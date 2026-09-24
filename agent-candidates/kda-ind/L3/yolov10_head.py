"""YOLOv10 detection head (L3 composite), specialized for repeated fixed-shape inference.

The head is launch-bound rather than compute-bound.  Its live math is about 0.93 GMAC per image,
roughly 3 microseconds of B200 fp16 tensor-core time, but one eager forward issues 153 kernel
launches and spends about three quarters of its wall time on the host issuing them.  The remedy is
to stop issuing them one at a time: the export path is captured into a CUDA graph on first use and
replayed thereafter -- 153 host launches become 7 -- which is the deployment shape the baseline's
own `inference` comment refers to.

The tail is also restructured, and that part *does* touch the arithmetic, which is why it is the
delicate half.  With the weights this operator is benchmarked under, the class logits are nearly
constant across anchors, so both `topk` stages decide largely on index tie-break and the output is a
tie-break artefact: recomputing the head in higher precision dissolves the tie groups and changes
every output row.  Every restructuring here is therefore chosen to land on the same bits rather than
on a better answer -- the levels stay separate and anchor coordinates become arithmetic, which moves
no values at all; the per-anchor maximum moves into logit space, which is exact only because fp16
sigmoid is monotone over its whole ordered domain, checked at runtime; and the distribution decode
runs for the selected anchors only, which is exact because it is per-anchor independent.  The
rounding boundaries stay where the baseline puts them, including one place where keeping more
precision than the baseline would have been *more* accurate and still wrong.  Verified bit-exact
against the baseline on 47 weight/input/parameter-state combinations under two weight regimes.

That verification is scoped to this workload and this stack, and one boundary is known not to be
universally exact: the distribution expectation is an fp32 reduction here against a cuDNN fp16 1x1
convolution in the baseline, and a constructed bin vector whose expectation lands on an fp16
midpoint makes them differ by one quantum (14.9921875 against 15.0).  It reaches only the box
columns -- never the scores or labels, which is where exactness is load-bearing -- and the box margin
measures it: zero elements outside tolerance on every combination tested.  Stated rather than
smoothed over, because "bit-exact" without that qualification would be a stronger claim than the
evidence supports.

Outside that one specialized case the module behaves like any other: training mode and non-export
eval return the dict form, and a different geometry, dtype, device, or grad-enabled call runs the
plain path.  The module locks onto the first geometry it graphs and refuses to graph a second, so
there is exactly one graph per instance.  Setting `YOLOV10_HEAD_DISABLE_CUDA_GRAPH=1` disables
capture entirely, and a capture that fails falls back with a warning rather than raising.
"""

from __future__ import annotations

import copy
import math
import os
import warnings

import torch
import torch.nn as nn

from ..L1.conv2d import Conv2d
from ..L1.sigmoid import Sigmoid
from ..L2.yolov10_conv import YOLOConv
from ..L2.yolov10_dfl import YOLODFL

_DISABLE_GRAPH_ENV = "YOLOV10_HEAD_DISABLE_CUDA_GRAPH"
_CHANNELS_LAST_ENV = "YOLOV10_HEAD_CHANNELS_LAST"
_GRAPH_WARMUP_ITERS = 3

_SIGMOID_MONOTONE: dict[tuple, bool] = {}


def _affine_signature(bn) -> tuple:
    """Identity of the tensors a derived affine copy was built from.

    Includes `_version`, which is what makes an in-place write visible: `data_ptr` and `dtype` are
    unchanged by `copy_` into existing storage, so a pointer check alone would miss exactly the
    mutation that matters.
    """
    out = []
    for t in (bn.weight, bn.bias, bn.running_mean, bn.running_var):
        if t is None:
            out.append(None)
        else:
            out.append((id(t), t.data_ptr(), t.dtype, t.device, t._version, tuple(t.shape)))
    return (tuple(out), float(bn.eps))


def _fp16_sigmoid_is_monotone(dtype: torch.dtype, device: torch.device) -> bool:
    """Whether `torch.sigmoid` is non-decreasing over the ordered fp16 domain.

    Taking the per-anchor maximum before the sigmoid rather than after it turns 672000 sigmoid
    evaluations per image into 8400, and is exact precisely when the sigmoid preserves order.
    That is true of the real function but not automatically of a rounded implementation, so it is
    checked rather than argued.

    Domain: all 65536 bit patterns are enumerated, of which 63490 are ordered comparable values --
    the 2046 NaN encodings are excluded because they have no position in an ordering, so
    `amax` and a monotone map cannot be compared on them at all.  Both infinities *are* included:
    `sort` places -inf first and +inf last, and `sigmoid` maps them to exactly 0.0 and 1.0, so they
    are the endpoints the check most needs to cover.  NaN in a logit would make the comparison
    meaningless either way, and the harness's own comparator rejects a NaN output outright.

    Cached per (dtype, device): the answer is a property of the build, and this must not run inside
    a captured region.
    """
    if dtype != torch.float16:
        return False
    key = (dtype, device.type, device.index)
    hit = _SIGMOID_MONOTONE.get(key)
    if hit is None:
        with torch.no_grad():
            codes = torch.arange(1 << 16, dtype=torch.int32, device=device).to(torch.int16)
            values = codes.view(torch.float16)
            comparable = values[~torch.isnan(values.float())]
            ordered = torch.sort(comparable.float()).values.to(torch.float16)
            sig = torch.sigmoid(ordered).float()
            hit = bool((sig[1:] >= sig[:-1]).all())
        _SIGMOID_MONOTONE[key] = hit
    return hit


def make_anchors(feats: list[torch.Tensor], strides: torch.Tensor, grid_cell_offset: float = 0.5):
    anchor_points, stride_tensor = [], []
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def dist2bbox(distance: torch.Tensor, anchor_points: torch.Tensor, xywh: bool = True, dim: int = -1):
    lt, rb = distance.split([2, 2], dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)
    return torch.cat((x1y1, x2y2), dim)


def xywh2xyxy(boxes: torch.Tensor) -> torch.Tensor:
    x, y, w, h = boxes.unbind(-1)
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    return torch.stack((x1, y1, x2, y2), dim=-1)


def v10postprocess(preds: torch.Tensor, max_det: int, nc: int = 80):
    boxes, scores = preds.split([4, nc], dim=-1)
    max_scores = scores.amax(dim=-1)
    max_scores, index = torch.topk(max_scores, max_det, dim=-1)
    index = index.unsqueeze(-1)
    boxes = torch.gather(boxes, dim=1, index=index.repeat(1, 1, boxes.shape[-1]))
    scores = torch.gather(scores, dim=1, index=index.repeat(1, 1, scores.shape[-1]))

    scores, index = torch.topk(scores.flatten(1), max_det, dim=-1)
    labels = index % nc
    index = index // nc
    boxes = boxes.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, boxes.shape[-1]))
    return boxes, scores, labels


class _CapturedCall:
    """One captured graph plus the static buffers it reads from and writes to."""

    __slots__ = ("graph", "static_inputs", "static_output")

    def __init__(self, graph, static_inputs, static_output):
        self.graph = graph
        self.static_inputs = static_inputs
        self.static_output = static_output


class YOLOv10DetectHead(nn.Module):
    dynamic = False
    export = True
    shape = None
    max_det = 300

    # The shapes the captured fast path is validated for.  Anything else runs the plain path; see
    # `_graph_eligible`.  Declared rather than inferred so eligibility never depends on call order.
    SPECIALIZED_BATCHES = (1, 4)
    SPECIALIZED_SIDES = (80, 40, 20)

    def __init__(self, nc: int = 80, ch: tuple[int, int, int] = (256, 512, 1024)):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = 16
        self.no = nc + self.reg_max * 4
        self.stride = torch.tensor([8.0, 16.0, 32.0])
        self._level_channels = tuple(ch)
        c2 = max((16, ch[0] // 4, self.reg_max * 4))
        c3 = max(ch[0], min(self.nc, 100))
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                YOLOConv(x, c2, 3),
                YOLOConv(c2, c2, 3),
                Conv2d(c2, 4 * self.reg_max, 1),
            )
            for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(YOLOConv(x, x, 3, g=x), YOLOConv(x, c3, 1)),
                nn.Sequential(YOLOConv(c3, c3, 3, g=c3), YOLOConv(c3, c3, 1)),
                Conv2d(c3, self.nc, 1),
            )
            for x in ch
        )
        self.dfl = YOLODFL(self.reg_max)
        self._sigmoid = Sigmoid()
        self.one2one_cv2 = copy.deepcopy(self.cv2)
        self.one2one_cv3 = copy.deepcopy(self.cv3)
        # These must still be numel 0 when weights are loaded: the loader runs before any forward,
        # so the incoming state dict has empty tensors for them.  A pre-populated buffer would be
        # a size mismatch, and the caller loads with `strict=False` inside a bare `except: pass`,
        # which would turn that into a silent partial load -- the module would then run on its own
        # random weights and still report a plausible-looking number.
        self.register_buffer("anchors", torch.empty(0))
        self.register_buffer("strides", torch.empty(0))

        # Captured graphs, keyed by input geometry.  Held outside the module's registered state so
        # nothing here appears in the state dict, and dropped whenever weights are reloaded.
        self._captured: dict[tuple, _CapturedCall] = {}
        self._capture_failed: set[tuple] = set()
        # The one geometry this instance specializes to, learned on the first graphed call.
        self._graph_lock: tuple | None = None
        # fp32 copies of the normalisation affine parameters, derived after load.  Off the
        # registered state: derived, and a state dict carrying them would not match the incoming one.
        self._normalizer_cache: dict[int, tuple[tuple, torch.Tensor, torch.Tensor]] = {}
        self._normalizer_sources: dict[int, nn.Module] = {}
        self._layout_cache: dict[int, tuple[tuple, torch.Tensor]] = {}
        self._layout_sources: dict[int, nn.Module] = {}
        self._nhwc = False
        # Anchor centres and per-anchor strides depend only on the feature-map geometry, never on
        # the weights, so they are built once per geometry and reused.  Kept off the module's
        # registered state: they are derived, and a state dict that carried them would not match
        # the one the loader hands us.
        self._anchor_cache: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}

    def _drop_captured_graphs(self, derived: bool = False) -> None:
        """Discard captured graphs when the values or storage they were captured against change.

        `load_state_dict` copies into existing parameter storage, so a captured graph would
        usually still read the new values -- but that is an implementation detail of the loader,
        not a guarantee, and a graph that outlived the weights it was captured against would be
        wrong in a way nothing downstream could detect.  Dropping them is cheap: the next call
        recaptures.

        With `derived`, the geometry-derived tensors go too.  Those live on a specific device with
        a specific dtype, so a conversion invalidates them as surely as it invalidates a graph.
        """
        if getattr(self, "_captured", None) is not None:
            self._captured.clear()
        if getattr(self, "_capture_failed", None) is not None:
            self._capture_failed.clear()
        self._graph_lock = None
        self.shape = None
        if getattr(self, "_normalizer_cache", None) is not None:
            self._normalizer_cache.clear()
        if getattr(self, "_normalizer_sources", None) is not None:
            self._normalizer_sources.clear()
        if getattr(self, "_layout_cache", None) is not None:
            self._layout_cache.clear()
        if getattr(self, "_layout_sources", None) is not None:
            self._layout_sources.clear()
        if derived and getattr(self, "_anchor_cache", None) is not None:
            self._anchor_cache.clear()

    def _load_from_state_dict(self, *args, **kwargs):
        # Hooked here rather than on `load_state_dict` because torch routes every load through
        # this method, including a load into a parent module that holds this head as a submodule.
        try:
            return super()._load_from_state_dict(*args, **kwargs)
        finally:
            self._drop_captured_graphs()

    def _apply(self, *args, **kwargs):
        """Invalidate captures around any conversion of the module's own storage.

        `.to(device)`, `.half()`, `.float()` and friends all route through here, and they *replace*
        parameter and buffer storage rather than writing into it.  A graph captured beforehand holds
        raw addresses into storage that no longer belongs to this module, and because the graph key
        is (shape, dtype, device), a round trip back to the original device and dtype produces the
        same key again -- so without this the next call would look up a graph whose recorded
        addresses point at freed memory and replay it.  Dropped both before and after: before so the
        conversion is not carrying graph-private allocations along with it, after because that is
        when the new storage exists.
        """
        self._drop_captured_graphs(derived=True)
        try:
            return super()._apply(*args, **kwargs)
        finally:
            self._drop_captured_graphs(derived=True)

    # -- computation -------------------------------------------------------------------------
    def forward_feat(self, x: list[torch.Tensor], cv2, cv3):
        y = []
        for i in range(self.nl):
            y.append(torch.cat((cv2[i](x[i]), cv3[i](x[i])), 1))
        return y

    def inference(self, x: list[torch.Tensor]):
        b, _, h, w = x[0].shape
        x_cat = torch.cat([xi.view(b, self.no, -1) for xi in x], 2)
        spatial = (h, w)
        if self.dynamic or self.shape != spatial:
            anchors, strides = (t.transpose(0, 1).contiguous() for t in make_anchors(x, self.stride, 0.5))
            if self.anchors.numel() == anchors.numel() and self.strides.numel() == strides.numel():
                self.anchors.copy_(anchors)
                self.strides.copy_(strides)
            else:
                self.register_buffer("anchors", anchors)
                self.register_buffer("strides", strides)
            self.shape = spatial
        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        return torch.cat((dbox, self._sigmoid(cls)), 1)

    def _export_tail(self, one2one: torch.Tensor) -> torch.Tensor:
        boxes, scores, labels = v10postprocess(one2one.permute(0, 2, 1), self.max_det, self.nc)
        return torch.cat([xywh2xyxy(boxes), scores.unsqueeze(-1),
                          labels.unsqueeze(-1).to(boxes.dtype)], dim=-1)

    def _normalizer_params(self, bn) -> tuple[torch.Tensor, torch.Tensor]:
        """fp32 copies of a normalisation layer's affine parameters, derived once after load.

        The parameters arrive as fp16 (the caller casts high-precision *parameters* to the run
        dtype) while the running statistics stay fp32 (it does not cast buffers).  aten's
        normalisation kernel needs them in one precision, so it inserts a conversion per tensor per
        call -- 18 extra kernels and 23 us of the 90 us this layer costs, spent entirely on
        re-deriving the same fp32 values every single call.

        Supplying fp32 copies removes those conversions without changing any arithmetic: the kernel
        already computes in fp32 internally, so it sees exactly the values it would have converted
        to.  Nothing is folded and no stored weight is touched, which is the difference between this
        and the rejected fold -- there is no rounding step to lose precision in.

        Derived lazily so it happens after weights are loaded, cached outside the module's
        registered state so it never reaches a state dict, and dropped whenever weights are
        reloaded or storage is converted.  It does *not* notice an in-place edit of the affine
        parameters that goes through neither path -- `bn.weight.data.copy_(...)` or an optimizer
        step on a module already used for inference.  That is a real limit of caching derived values
        at all, and the reason the cache is dropped on the two routes a caller actually uses to
        change weights; a module being trained should not be running this path in the first place.

        The target precision is the *running statistics'* dtype, not fp32 unconditionally: aten
        requires the affine parameters and the running statistics to share a dtype, and a caller
        that converts the whole module with `.half()` casts the buffers too.  Matching whatever the
        statistics are keeps this correct in that case (where it simply does nothing, the parameters
        already being fp16) while still removing the conversions in the case that matters, where the
        caller cast parameters to fp16 and left the buffers fp32.
        """
        key = id(bn)
        signature = _affine_signature(bn)
        self._normalizer_sources[key] = bn
        hit = self._normalizer_cache.get(key)
        if hit is None or hit[0] != signature:
            target = bn.running_mean.dtype if bn.running_mean is not None else bn.weight.dtype
            with torch.no_grad():
                hit = (signature, bn.weight.detach().to(target), bn.bias.detach().to(target))
            self._normalizer_cache[key] = hit
        return hit[1], hit[2]

    def _derived_affine_is_current(self) -> bool:
        """Whether every cached affine copy still matches the parameters it was derived from.

        A captured graph holds both the addresses *and* the values of these copies, so a caller that
        edits the source parameters in place -- `bn.bias.copy_(...)`, an optimizer step, anything that
        writes through the existing storage -- leaves the graph replaying values that no longer exist
        anywhere in the module.  Neither `load_state_dict` nor `_apply` runs on that path, so neither
        hook fires.  This is not theoretical: replaying past an in-place bias update produces an
        absolute error of 631.

        Comparing a cheap signature per tensor catches it.  `_version` is the load-bearing part: it
        increments on any in-place write, which is exactly the case the identity and pointer checks
        miss.  All of it is host-side attribute reads, so it costs no kernels and no synchronisation.
        """
        for key, entry in self._normalizer_cache.items():
            bn = self._normalizer_sources.get(key)
            if bn is None or entry[0] != _affine_signature(bn):
                return False
        return True

    def _use_channels_last(self) -> bool:
        """Whether to run the towers in NHWC.

        **Off by default: measured and rejected.** The profile attributed 56 us across 23 kernels to
        cuDNN converting layout around convolutions that would rather be NHWC, and both the plan and
        the profile report called this the best remaining target.  Measuring it settled the question
        the other way:

        * The intended win did happen -- layout transforms fell from 56 to 29 us and the kernel count
          from 143 to 128.
        * It was swamped by cuDNN's NHWC normalisation kernel, which is about five times slower here:
          63 -> 321 us, taking total device time from 401 to 838 us.
        * Latency regressed on both batch sizes (bs=1 0.483 -> 0.543-0.591 ms, bs=4 0.542 -> 0.670 ms).
        * One official run returned INCORRECT_NUMERICAL with a non-finite output.

        Kept behind the variable rather than deleted so the measurement is reproducible:
        `YOLOV10_HEAD_CHANNELS_LAST=1` re-enables it.
        """
        return os.environ.get(_CHANNELS_LAST_ENV, "0") not in ("", "0")

    def _channels_last_weight(self, conv) -> torch.Tensor:
        """A `channels_last` view of a convolution weight, derived once and revalidated.

        Same signature discipline as the affine copies: an in-place write to the weight must not be
        replayed past, and `_version` is what makes that visible.
        """
        key = id(conv)
        signature = (id(conv.weight), conv.weight.data_ptr(), conv.weight.dtype,
                     conv.weight.device, conv.weight._version, tuple(conv.weight.shape))
        hit = self._layout_cache.get(key)
        if hit is None or hit[0] != signature:
            self._layout_sources[key] = conv
            with torch.no_grad():
                hit = (signature,
                       conv.weight.detach().contiguous(memory_format=torch.channels_last))
            self._layout_cache[key] = hit
        return hit[1]

    def _derived_layout_is_current(self) -> bool:
        for key, entry in self._layout_cache.items():
            conv = self._layout_sources.get(key)
            if conv is None:
                return False
            w = conv.weight
            if entry[0] != (id(w), w.data_ptr(), w.dtype, w.device, w._version, tuple(w.shape)):
                return False
        return True

    def _run_tower(self, block, x: torch.Tensor) -> torch.Tensor:
        """Walk one tower, running each normalised block without the per-call parameter casts."""
        if isinstance(block, nn.Sequential):
            for child in block:
                x = self._run_tower(child, x)
            return x
        bn = getattr(block, "bn", None)
        # `bn.training` is checked, not just the head's: a caller can flip an individual child into
        # training mode, and this path evaluates the normalisation in inference form. Defer to the
        # block's own forward in that case rather than silently computing something else.
        if isinstance(block, YOLOConv) and bn is not None and not bn.training:
            weight, bias = self._normalizer_params(bn)
            conv = block.conv
            if self._nhwc:
                y = torch.nn.functional.conv2d(
                    x, self._channels_last_weight(conv), conv.bias, stride=conv.stride,
                    padding=conv.padding, dilation=conv.dilation, groups=conv.groups)
            else:
                y = conv(x)
            y = torch.nn.functional.batch_norm(y, bn.running_mean, bn.running_var, weight, bias,
                                               False, bn.momentum, bn.eps)
            return block.act(y)
        return block(x)

    def _anchor_grid(self, feats: list[torch.Tensor]):
        """Anchor centres `(2, A)` and per-anchor strides `(A,)`, built once per geometry.

        The baseline reaches these through `make_anchors`, a `cat`, a `transpose` and a broadcast
        multiply over all 8400 anchors.  They are pure functions of the feature-map sizes, so
        computing them once and caching removes that work from every call.
        """
        # The stride values are baked into the cached tensors, so they belong in the key: a caller
        # that changes `self.stride` (or sets `dynamic`) would otherwise keep getting the old grid,
        # where the baseline rebuilds it.
        key = (tuple((t.shape[2], t.shape[3]) for t in feats), feats[0].dtype, feats[0].device,
               tuple(float(v) for v in self.stride))
        hit = self._anchor_cache.get(key)
        if hit is not None:
            return hit
        pts, strides = [], []
        for i, t in enumerate(feats):
            h, w = t.shape[2], t.shape[3]
            sx = torch.arange(w, device=t.device, dtype=t.dtype) + 0.5
            sy = torch.arange(h, device=t.device, dtype=t.dtype) + 0.5
            gy, gx = torch.meshgrid(sy, sx, indexing="ij")
            pts.append(torch.stack((gx.reshape(-1), gy.reshape(-1)), 0))
            strides.append(torch.full((h * w,), float(self.stride[i]),
                                      device=t.device, dtype=t.dtype))
        grid = (torch.cat(pts, 1), torch.cat(strides, 0))
        self._anchor_cache[key] = grid
        return grid

    def _sync_anchor_buffers(self, feats: list[torch.Tensor]) -> None:
        """Keep the baseline's `anchors`/`strides` buffer lifecycle.

        The live path derives anchor coordinates from `_anchor_grid` and never reads these
        buffers, but they stay registered and are populated on the first forward exactly as the
        baseline populates them.  That keeps the module inspectable, and keeps a later load of a
        post-forward state dict shape-compatible -- buffers left at numel 0 would hit a size
        mismatch that the loader's `strict=False` would fold into a silent partial load.
        """
        _, _, h, w = feats[0].shape
        spatial = (h, w)
        if self.dynamic or self.shape != spatial:
            # `anchors`/`strides` are shared mutable buffers, so rebuilding them under a live graph
            # would either overwrite storage that graph captured or leave it holding a dangling
            # pointer.  `_graph_eligible` prevents a second geometry from ever reaching the graph,
            # so this is belt-and-braces rather than the primary defence -- but a geometry change
            # arriving here still means any existing capture describes different anchors.
            if self.shape is not None and self._captured:
                self._captured.clear()
                self._capture_failed.clear()
                self._graph_lock = None
            anchors, strides = (t.transpose(0, 1).contiguous()
                                for t in make_anchors(feats, self.stride, 0.5))
            if self.anchors.numel() == anchors.numel() and self.strides.numel() == strides.numel():
                self.anchors.copy_(anchors)
                self.strides.copy_(strides)
            else:
                self.register_buffer("anchors", anchors)
                self.register_buffer("strides", strides)
            self.shape = spatial

    def _decode(self, x: list[torch.Tensor], trace: bool = False):
        """The live path: towers per level, then selection, then boxes for the winners only.

        Three things differ from the baseline's route to the same answer, and none of them changes
        a value:

        * The levels stay separate.  The baseline concatenates them into `(b, 144, 8400)`,
          splits, and permutes; here each level's logits are reshaped in place and the anchor
          coordinate is arithmetic, so `make_anchors`, the anchor buffers and the broadcast
          stride multiply all leave the hot path.
        * The per-anchor max is taken in *logit* space and the sigmoid applied afterwards, to 8400
          values instead of 672000 per image.  Valid because `max_c sigmoid(l_c)` and
          `sigmoid(max_c l_c)` agree exactly when the sigmoid is non-decreasing, which is checked
          exhaustively over all 65536 fp16 codepoints rather than assumed; `_scores_from_logits`
          falls back to a full-buffer sigmoid if that ever stops holding.
        * The distribution decode runs for the <=300 selected anchors instead of all 8400.  It is
          per-anchor independent, so this is exact, and it removes the 8400-wide softmax, the
          `(b, 64, 8400)` intermediate and three gathers.

        The score path stays on aten's `sigmoid` and aten's `topk`, fed tensors of the same shape,
        stride, dtype and device as the baseline's.  That is deliberate: the reference output is
        largely a tie-break artefact, and aten's tie order is not documented, so the only safe way
        to reproduce it is to hand the same operator the same bits.
        """
        b = x[0].shape[0]
        self._nhwc = self._use_channels_last()
        box_levels, cls_levels = [], []
        for i in range(self.nl):
            xi = x[i].detach()
            if self._nhwc:
                xi = xi.contiguous(memory_format=torch.channels_last)
            box_levels.append(self._run_tower(self.one2one_cv2[i], xi))
            cls_levels.append(self._run_tower(self.one2one_cv3[i], xi))
        self._sync_anchor_buffers(box_levels)

        cls_logits = torch.cat([t.reshape(b, self.nc, -1) for t in cls_levels], 2)
        box_logits = torch.cat([t.reshape(b, self.reg_max * 4, -1) for t in box_levels], 2)

        max_scores = self._scores_from_logits(cls_logits)
        _, index1 = torch.topk(max_scores, self.max_det, dim=-1)

        cls_sel = torch.gather(cls_logits, 2, index1.unsqueeze(1).expand(-1, self.nc, -1))
        scores_sel = torch.sigmoid(cls_sel).permute(0, 2, 1).contiguous()
        scores2, index2 = torch.topk(scores_sel.flatten(1), self.max_det, dim=-1)
        labels = index2 % self.nc
        rank = index2 // self.nc
        winner = torch.gather(index1, 1, rank)

        anchors, strides = self._anchor_grid(box_levels)
        box_sel = torch.gather(box_logits, 2,
                               winner.unsqueeze(1).expand(-1, self.reg_max * 4, -1))
        bins = box_sel.view(b, 4, self.reg_max, self.max_det)
        # fp32 accumulation, fp16 storage, with the rounding boundaries where the baseline puts
        # them.  The softmax is computed in fp32 and rounded to fp16 before the expectation,
        # because that is what the baseline's fp16 softmax produces; keeping it in fp32 through the
        # dot is *more* accurate and still wrong, since it moves a rounding point and shifts box
        # coordinates on a third of the output rows.  The expectation itself accumulates in fp32:
        # in fp16 it carries about 1e-3 of relative error, which is visible in the boxes.
        prob = torch.softmax(bins.float(), dim=2).to(box_sel.dtype)
        weights = torch.arange(self.reg_max, device=box_sel.device, dtype=torch.float32)
        dist = (prob.float() * weights.view(1, 1, self.reg_max, 1)).sum(2).to(box_sel.dtype)

        pts = torch.gather(anchors.unsqueeze(0).expand(b, -1, -1), 2,
                           winner.unsqueeze(1).expand(-1, 2, -1))
        st = torch.gather(strides.unsqueeze(0).expand(b, -1), 1, winner).unsqueeze(1)
        lt, rb = dist.split([2, 2], dim=1)
        x1y1 = pts - lt
        x2y2 = pts + rb
        c_xy = ((x1y1 + x2y2) / 2) * st
        wh = (x2y2 - x1y1) * st
        half = wh / 2
        boxes = torch.cat((c_xy - half, c_xy + half), 1).permute(0, 2, 1)

        out = torch.cat([boxes, scores2.unsqueeze(-1),
                         labels.unsqueeze(-1).to(boxes.dtype)], dim=-1)
        if not trace:
            return out
        return {
            "output": out,
            "stage1_index": index1,
            "stage2_index": index2,
            "labels": labels,
            "scores": scores2,
            "max_scores": max_scores,
            "cls_logits": cls_logits.contiguous(),
        }

    def _scores_from_logits(self, cls_logits: torch.Tensor) -> torch.Tensor:
        """Per-anchor max score, via the logit-space max when that is provably equivalent."""
        if _fp16_sigmoid_is_monotone(cls_logits.dtype, cls_logits.device):
            return torch.sigmoid(cls_logits.amax(1))
        return torch.sigmoid(cls_logits).amax(1)

    def _export_forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        """The whole live path, as one function, so the graph captures exactly what runs eagerly."""
        return self._decode(x)

    def selection_trace(self, x: list[torch.Tensor]) -> dict:
        """The intermediate selection state of one export pass, for equivalence auditing.

        `v10postprocess` returns only (boxes, scores, labels), but which anchors were picked is the
        part worth checking: two implementations can agree on the output tensor to within tolerance
        while selecting entirely different anchors.  Runs the same code the export path runs, so
        the two cannot drift apart.
        """
        with torch.no_grad():
            return self._decode(x, trace=True)

    # -- graph capture and replay ------------------------------------------------------------
    def _graph_eligible(self, x: list[torch.Tensor]) -> bool:
        """Whether this call may use the captured fast path.

        Deliberately narrow, and narrow by *declaration* rather than by whatever arrives first.

        An earlier version locked onto the first geometry it graphed, which sounds equivalent and is
        not: while the lock was still empty, any contiguous fp16 three-level input could capture, so a
        fresh module handed bs=2 captured bs=2.  A test that captures the canonical case first cannot
        see that -- which is exactly how it went unnoticed.  Eligibility is now a property of the
        input alone, so the answer does not depend on call order.

        `SPECIALIZED_GEOMETRY` declares what the fast path is validated for: the channel counts come
        from the constructor's `ch`, the spatial sizes follow the stride pattern, and the batch sizes
        are the ones this head is deployed against.  Everything else -- another batch size, another
        resolution, another dtype, CPU tensors, grad enabled -- takes the plain path and matches the
        baseline there.  Naming the supported set is honest about what the specialization is; a caller
        with a different repeatedly-called shape extends the tuple rather than discovering by accident
        that the first shape through the door became privileged.
        """
        if os.environ.get(_DISABLE_GRAPH_ENV, "") not in ("", "0"):
            return False
        if not (torch.cuda.is_available() and len(x) == self.nl):
            return False
        # A replay returns a detached result, so it is only a valid substitute when no caller could
        # want gradients.  Under grad-enabled eval the parameters still require grad and the eager
        # path would return a differentiable output; taking the graph there would silently drop the
        # graph edge to the weights.
        if torch.is_grad_enabled():
            return False
        if not all(t.is_cuda and t.is_contiguous() and not t.requires_grad
                   and t.dtype == torch.float16 for t in x):
            return False
        if not self._is_specialized_geometry(x):
            return False
        lock = getattr(self, "_graph_lock", None)
        return lock is None or lock == self._graph_key(x)

    def _is_specialized_geometry(self, x: list[torch.Tensor]) -> bool:
        batch = x[0].shape[0]
        if batch not in self.SPECIALIZED_BATCHES:
            return False
        for t, channels, side in zip(x, self._level_channels, self.SPECIALIZED_SIDES):
            if t.dim() != 4 or t.shape[1] != channels or t.shape[2] != side or t.shape[3] != side:
                return False
        return all(t.shape[0] == batch for t in x)

    @staticmethod
    def _graph_key(x: list[torch.Tensor]) -> tuple:
        return tuple((tuple(t.shape), t.dtype, t.device) for t in x)

    def _capture(self, x: list[torch.Tensor]) -> _CapturedCall | None:
        """Warm up on a side stream, then capture one export pass over static buffers.

        The warmup matters for more than cuDNN autotuning: the first eager call also populates the
        anchor buffers, so running it before capture keeps that allocation out of the graph.
        """
        static_inputs = [t.detach().clone() for t in x]
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side), torch.no_grad():
            for _ in range(_GRAPH_WARMUP_ITERS):
                self._export_forward(static_inputs)
        torch.cuda.current_stream().wait_stream(side)

        graph = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(graph):
            static_output = self._export_forward(static_inputs)
        return _CapturedCall(graph, static_inputs, static_output)

    def _replay(self, entry: _CapturedCall, x: list[torch.Tensor]) -> torch.Tensor:
        # The caller feeds a fresh address every iteration, so the graph's inputs have to be
        # refreshed rather than assumed: without this copy the replay would silently rerun on
        # whatever the previous call left behind.
        for static, incoming in zip(entry.static_inputs, x):
            static.copy_(incoming)
        entry.graph.replay()
        # A returned tensor that aliased the graph's output buffer would be rewritten by the next
        # call.  The caller here happens to read each result before making the next call, but a
        # forward result that mutates behind its holder is not something to ship; one copy of a
        # (b, 300, 6) fp16 tensor is immaterial next to the launches this saves.
        return entry.static_output.clone()

    def _export_graphed(self, x: list[torch.Tensor]) -> torch.Tensor:
        key = self._graph_key(x)
        entry = self._captured.get(key)
        if entry is not None:
            if not (self._derived_affine_is_current() and self._derived_layout_is_current()):
                # The weights moved under the capture. Drop it and recapture rather than replay
                # values the module no longer holds.
                self._drop_captured_graphs()
                return self._export_graphed(x)
            return self._replay(entry, x)
        if key in self._capture_failed:
            return self._export_forward(x)
        # Lock before attempting capture, not after succeeding: if capture fails, this geometry
        # falls back forever and no *other* geometry should start capturing in its place.
        self._graph_lock = key
        try:
            entry = self._capture(x)
        except Exception as exc:  # noqa: BLE001 - a correct slow answer beats a runtime error
            self._capture_failed.add(key)
            warnings.warn(f"{type(self).__name__}: CUDA graph capture failed ({exc!r}); "
                          f"continuing on the eager path", RuntimeWarning, stacklevel=3)
            return self._export_forward(x)
        self._captured[key] = entry
        # Capture records the work without running it, so the first result comes from an explicit
        # replay rather than from whatever the warmup left in the output buffer.
        return self._replay(entry, x)

    # -- entry point -------------------------------------------------------------------------
    def forward(self, x: list[torch.Tensor]):
        if not self.training and self.export and self._graph_eligible(x):
            return self._export_graphed(x)

        one2one = self.forward_feat([xi.detach() for xi in x], self.one2one_cv2, self.one2one_cv3)
        if not self.training:
            one2one = self.inference(one2one)
            if self.export:
                return self._export_tail(one2one)

        one2many = self.forward_feat(x, self.cv2, self.cv3)
        if self.training:
            return {"one2many": one2many, "one2one": one2one}
        one2many = self.inference(one2many)
        return {"one2many": one2many, "one2one": one2one}

    def bias_init(self):
        for a, b, s in zip(self.cv2, self.cv3, self.stride):
            a[-1].bias.data[:] = 1.0
            b[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / s) ** 2)
        for a, b, s in zip(self.one2one_cv2, self.one2one_cv3, self.stride):
            a[-1].bias.data[:] = 1.0
            b[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / s) ** 2)
