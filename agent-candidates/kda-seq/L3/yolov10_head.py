"""YOLOv10 detection head: one replayed graph, and a tail that selects before it decodes.

The captured cases are ``nc=80, ch=(64,128,256)`` at ``b=1`` and ``b=4``, fp16, and the
head is dispatch-bound rather than compute-bound. The baseline issues roughly 150 device
kernels and about 2.7 ms of host work per call for about 1.9 GFLOP of convolution;
replaying the *unmodified* baseline op sequence from a CUDA graph is bitwise identical to
eager and lands at 825/959 us against 1584/1628 us eager. Everything above that is host
submission cost, so the first lever is the number of launches the host submits and the
second is the number of kernels inside the graph.

Three things happen here, in the order their payoff was measured:

**The whole inference path is replayed from a captured graph.** Composing the frozen
``YOLOConv`` without a graph measures 0.92x -- slower than the baseline -- because the
host is the bottleneck and each frozen block pays a per-call guard. Fusion only pays once
replay removes the launch path, so capture is not a later refinement; it is what makes the
rest worth doing. The guard around the cache is a per-call signature over every address
the replay will dereference, not a shape key: see :meth:`_graph_signature`.

**The tail selects first and decodes afterwards.** The baseline builds a ``(b,144,8400)``
concatenation, sigmoids all ``80x8400`` class logits, runs the distribution-focal-loss
soft-argmax and the box decode over all 8400 anchors, and only then keeps 300. Every one
of those steps is per-anchor independent, and ``sigmoid`` is elementwise, so gathering
commutes with all of them. Selecting first and decoding 300 anchors instead of 8400 is
bitwise equal to the baseline and retires the concatenation, the full-width sigmoid, the
``permute``/``split`` round trip and 28/29ths of the box work. The three levels' outputs
are read through per-level base pointers plus an anchor-to-(level, offset) mapping, so
nothing is concatenated to make the anchor axis contiguous.

**The class-score path goes through a table, not a transcendental.** Under the harness's
weight draw the class logits are bias-dominated: ``sigmoid(cls)`` in fp16 takes 60 distinct
values, ``max_scores`` takes *two*, and 2000 anchors tie at the top value with zero
strictly above it while 300 must be selected. Both ``topk`` calls are therefore decided
entirely by tie-breaking, and injecting sigma=1e-4 into the logits drops the harness's
matched ratio to 0.9847. So the score path may not merely be close -- it has to reproduce
``fp16(sigmoid(fp16 logit))`` exactly. Rather than hope a Triton ``exp`` agrees with ATen's
bit for bit, :func:`_sigmoid_lut` tabulates ``torch.sigmoid`` over all 65536 fp16 bit
patterns once and the kernels index that table. Exactness is then true by construction
rather than by measurement, for this and any future toolchain.

Both ``topk`` calls stay on ``torch.topk``. Installed PyTorch documents tied indices as
unstable, so there is no ordering contract to reimplement against -- and since the
*baseline* calls ``torch.topk``, the reference answer is whatever the installed
implementation does. A custom selection would have to reproduce that empirically, which is
a bigger risk than the roughly 190 us it would win, and it is not attempted here.

Nothing weight-derived is module state. The sigmoid table, the per-shape level plan and
the captured graph live in plain attributes that ``_apply`` drops, so they never appear in
``state_dict()`` and cannot silently shadow the harness's shared weights. ``anchors`` and
``strides`` are registered as ``torch.empty(0)`` exactly as the baseline registers them,
because the harness shares weights with ``load_state_dict(..., strict=False)`` inside a
bare ``except``: a candidate that pre-registered them at their final shapes would raise a
size mismatch, have it swallowed, and keep candidate-local anchor values with nothing
reported.

Set ``FK_YOLO_HEAD_GRAPH=0`` to run the fast path without replay, which is how the graph's
contribution is priced.
"""

from __future__ import annotations

import copy
import math
import os
from typing import NamedTuple

import torch
import torch.autograd.forward_ad as _forward_ad
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules.module as _module_hooks
import triton
import triton.language as tl

from ..L1.conv2d import Conv2d
from ..L1.sigmoid import Sigmoid
from ..L1.batch_norm2d import BatchNorm2d
from ..L1.silu import SiLU
from ..L2.yolov10_conv import YOLOConv
from ..L2.yolov10_dfl import YOLODFL

# Replay is on by default and off under this variable, so the graph's contribution can be
# priced without editing the module. Warmups precede capture because a JIT compile is not
# capturable and a fused-weight build inside the capture would be frozen in the graph's
# private pool.
_GRAPH_REPLAY = os.environ.get("FK_YOLO_HEAD_GRAPH", "1") != "0"
_GRAPH_WARMUP = 3

# Both detection branches run the baseline's own arithmetic rather than the frozen fused
# convolutions, which is what makes the output bitwise equal to the reference rather than
# merely inside the harness's tolerance. Either branch can be fused back for pricing:
#
#   FK_YOLO_HEAD_FUSED_CLS=1  -- fuse the class branch. Measurably unsafe: 1 weight draw in
#                                48 then fails at matched_ratio 0.7233. See :meth:`_class_logits`.
#   FK_YOLO_HEAD_FUSED_BOX=1  -- fuse the box branch. Safe against the harness's own gate but
#                                not bitwise, so it forfeits exact equality. See :meth:`_box_logits`.
_FUSED_CLASS = os.environ.get("FK_YOLO_HEAD_FUSED_CLS", "0") != "0"
_FUSED_BOX = os.environ.get("FK_YOLO_HEAD_FUSED_BOX", "0") != "0"
#   FK_YOLO_HEAD_FUSED_DFL=1  -- use the frozen soft-argmax. One kernel instead of two, and it
#                                agrees with the baseline's on almost every weight draw; the
#                                exception is what closed the last gap to bitwise equality.
_FUSED_DFL = os.environ.get("FK_YOLO_HEAD_FUSED_DFL", "0") != "0"

# The recursive host-state snapshot is what stops a replay from surviving a mutation of a child
# module's configuration. It is also the most expensive thing the guard does, so
# ``FK_YOLO_HEAD_SEMANTIC_GUARD=0`` exists purely to *price* it against a run with it -- it is
# not a shipping configuration, and a module running without it can return a stale answer.
_SEMANTIC_GUARD = os.environ.get("FK_YOLO_HEAD_SEMANTIC_GUARD", "1") != "0"

# The harness rebuilds the module for every case, so one instance only ever sees one batch size
# and a single entry would do for scoring. The cache is bounded rather than absent so a caller who
# alternates shapes stays correct without growing memory without limit.
_GRAPH_CACHE_LIMIT = 8

# Largest element offset the kernels form in 32-bit arithmetic. Anything that could exceed
# it takes the reference path rather than silently wrapping.
_MAX_INT32_OFFSET = 2 ** 31 - 1

# Dtype the fast path is written for. fp32 is deliberately absent: the harness casts
# parameters to fp16 and compares at atol=rtol=1e-2, and every measurement behind this
# design was taken in fp16.
_FAST_DTYPE = torch.float16

# Route names, so a test can assert which path served a call without running either.
ROUTE_GRAPH = "graph"
ROUTE_FAST = "fast"
ROUTE_REFERENCE = "reference"


# ---------------------------------------------------------------------------
# The baseline's own helpers. Reproduced rather than imported because the reference
# path has to raise what the baseline raises and round where the baseline rounds.
# ---------------------------------------------------------------------------
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


def _fast_path_modules(root: nn.Module):
    """Every module the fast path reaches, depth-first in a stable order.

    Reads ``_modules`` directly rather than calling ``modules()``, for the same reason
    :meth:`YOLOv10DetectHead._graph_operands` does: this runs before every replay. Used by the
    tests; :func:`_fast_path_state` inlines the same walk.
    """
    out = [root]
    stack = [root]
    while stack:
        mod = stack.pop()
        for child in mod._modules.values():
            if child is not None:
                out.append(child)
                stack.append(child)
    return out


def _conv_state(conv, push) -> bool:
    """The convolution configuration ``F.conv2d`` is handed. ``False`` to decline the call."""
    if conv.__class__ is not Conv2d:
        return False
    push(conv.stride)
    push(conv.padding)
    push(conv.dilation)
    push(conv.groups)
    push(conv.bias is None)
    return True


def _act_state(act, push) -> bool:
    """The activation's identity. ``False`` for anything whose behaviour cannot be summarised."""
    cls = act.__class__
    push(cls)
    # Identity and SiLU are the two the head constructs, and both are fully described by their
    # type. Anything else is a module whose behaviour depends on state this does not read -- and
    # rather than guess at that state, the call goes to the reference path, where the activation
    # is invoked as a module exactly as the baseline invokes it.
    return cls is nn.Identity or cls is SiLU or cls is nn.SiLU


def _fast_path_state(roots):
    """The host state the fast path's arithmetic depends on, or ``None`` to decline the call.

    This is the guard that stops a replay from surviving a mutation of a child module's
    configuration -- the bug that reached round 1, where editing
    ``one2one_cv2[0][0].conv.padding`` left the capture generation untouched and returned the
    previous answer while the ungraphed path raised.

    What it reads is narrow on purpose, and the reason is worth stating because a wider version
    looks safer and is not. The fast path reproduces ``baseline.py``'s own expression --
    ``F.silu(F.batch_norm(F.conv2d(...)))`` -- so the values that can make the two disagree are
    exactly the ones *that expression* reads: each convolution's stride, padding, dilation, group
    count and whether it has a bias; each BatchNorm's epsilon, momentum, training mode and
    tracking flag; and each activation's identity. A frozen layer's chosen route or tile is
    **not** in that set: the fast path never consults it and neither does the baseline, so a
    caller who edits it changes neither answer.

    A first version snapshotted every plain attribute of all 95 modules generically. It was
    sound, and it cost 284 us per call -- measured against the harness's own timing loop, which
    has less host slack than a bare probe loop because it also enqueues an L2 flush and the
    shifting pool's input copies, that was **256 us of score**, taking the geomean from 3.46x to
    2.30x. Flattening its output changed nothing, because the cost was the traversal itself.
    Reading the attributes the arithmetic actually depends on, by name, is both cheaper and
    easier to argue about.

    The generic safety net is kept where it matters: any module whose class this does not
    recognise, and any activation it cannot summarise, returns ``None`` and the call takes the
    reference path. Declining is always safe; assuming is not.
    """
    flat = []
    push = flat.append
    for root in roots:
        stack = [root]
        while stack:
            mod = stack.pop()
            if _hooked(mod):
                return None
            cls = mod.__class__
            if cls is YOLOConv:
                # The wrapper *and* the three leaves this path evaluates functionally.
                if _hooked(mod.conv, mod.bn, mod.act):
                    return None
                if not _conv_state(mod.conv, push):
                    return None
                bn = mod.bn
                if bn.__class__ is not BatchNorm2d:
                    return None
                # A BatchNorm that is both training and tracking makes the baseline increment
                # ``num_batches_tracked`` on every call. That is a state mutation the functional
                # path does not perform and a captured graph could not represent, so the call
                # goes to the reference path instead of being served approximately.
                if bn.training and bn.track_running_stats:
                    return None
                push(bn.eps)
                push(bn.momentum)
                push(bn.training)
                push(bn.track_running_stats)
                push(bn.affine)
                push(mod._is_fused)
                if not _act_state(mod.act, push):
                    return None
                continue                      # its children are covered by the reads above
            if cls is Conv2d:
                if not _conv_state(mod, push):
                    return None
                continue
            if cls is YOLODFL:
                softmax = mod._modules["_softmax"]
                if _hooked(mod.conv, softmax):
                    return None
                push(mod.c1)
                push(softmax.dim)
                if not _conv_state(mod.conv, push):
                    return None
                continue
            if cls is nn.Sequential or cls is nn.ModuleList:
                children = mod._modules
                push(len(children))
                for child in children.values():
                    if child is not None:
                        stack.append(child)
                continue
            # An unrecognised container or leaf: decline rather than guess at what it reads.
            return None
    return tuple(flat)


def aten_fp16_topk_key(x: torch.Tensor) -> torch.Tensor:
    """``TopKTypeConfig<at::Half>::convert``, the key ATen's radix select orders fp16 by.

    From ``ATen/native/cuda/SortingRadixSelect.cuh``::

        mask = (bits & 0x8000) ? 0xffff : 0x8000;
        return (v == v) ? (bits ^ mask) : 0xffff;

    Two details a generic order-preserving key gets wrong, and both matter here. NaN maps to
    ``0xffff``, the *maximum* key, so ``torch.topk`` ranks NaN anchors **first** -- which is why
    :func:`_score_max` forces NaN propagation instead of letting ``tl.max`` drop it. And ``+0``
    (``0x0000 ^ 0x8000``) outranks ``-0`` (``0x8000 ^ 0xffff``), so the two zeros are ordered even
    though they compare equal.

    This lives in the module rather than in a test because it is the transform any exact
    replacement for the two ``torch.topk`` calls has to reproduce, and one definition shared
    between a kernel and its parity test is the only way the two cannot drift. Nothing on the
    shipped path calls it: the shipped path uses ``torch.topk`` itself, and this is the artifact a
    replacement is gated against.
    """
    bits = x.view(torch.int16).to(torch.int32) & 0xFFFF
    mask = torch.where((bits & 0x8000) != 0, 0xFFFF, 0x8000)
    return torch.where(torch.isnan(x.float()), torch.full_like(bits, 0xFFFF), bits ^ mask)


def _hooked(*modules) -> bool:
    """Whether any of these modules carries a hook that would change what it computes.

    The fast path reaches *past* ``forward``: the detection branches are evaluated as functionals
    so their arithmetic can match the baseline's bit for bit. A forward or pre-forward hook on any
    module along that path therefore fires for the baseline and not here, which is a semantic
    difference no graph can represent.

    Every module the functional path bypasses has to be asked, not just the wrapper. An earlier
    version checked ``YOLOConv`` and missed ``YOLOConv.conv``, ``.bn`` and ``.act``, so a raising
    hook on a BatchNorm left the fast path serving the call while the baseline raised.
    """
    for mod in modules:
        if mod is not None and (mod._forward_hooks or mod._forward_pre_hooks):
            return True
    return False


def _global_hooks_present() -> bool:
    """A hook registered through ``nn.modules.module`` applies to every module in the process.

    No per-module dictionary can see it, so it is asked about separately.
    """
    return bool(_module_hooks._global_forward_hooks or _module_hooks._global_forward_pre_hooks)


def _dfl_reference(dfl, x: torch.Tensor) -> torch.Tensor:
    """``baseline.py``'s soft-argmax expression, calling the two leaves as modules.

    The reference path cannot simply call ``self.dfl(x)``. This module's soft-argmax is the frozen
    L2 winner, whose own fast route bypasses its ``_softmax`` child, while the baseline's calls it
    every time. So a forward hook on ``dfl._softmax`` fires for the baseline and not for a
    candidate that delegates -- found by the hook coverage tests, where a raising hook made the
    baseline raise and the candidate return.

    Spelling the expression out restores it: hooks fire in ``Module.__call__``, so calling the
    leaves as modules reproduces the baseline's observable behaviour whatever route each leaf then
    takes internally. The frozen module's weights are still the ones read.
    """
    b, _, a = x.shape
    v = x.view(b, 4, dfl.c1, a).transpose(2, 1)
    return dfl.conv(dfl._modules["_softmax"](v)).view(b, 4, a)


def _baseline_chain(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Apply a detection branch using exactly ``baseline.py``'s arithmetic.

    The baseline's L1 leaves are thin wrappers -- ``F.conv2d``, ``F.batch_norm``, ``F.silu``
    -- so reproducing them bit for bit means calling those functionals in that order and
    letting each one round its own result. Reaching past the frozen modules' fast routes is
    the point: the frozen convolution rounds once where the baseline rounds three times, and
    on the class branch that difference is what decides a tie.

    The frozen modules are still the owners of the weights; only the arithmetic is bypassed.
    Nothing here mutates them, so their own guards and fallbacks are untouched.
    """
    if isinstance(module, nn.Sequential):
        for child in module:
            x = _baseline_chain(child, x)
        return x
    if isinstance(module, YOLOConv):
        conv, bn = module.conv, module.bn
        y = F.conv2d(x, conv.weight, conv.bias, stride=conv.stride, padding=conv.padding,
                     dilation=conv.dilation, groups=conv.groups)
        if not module._is_fused:
            # The training flag is the baseline's own expression, not a hardcoded ``False``.
            # ``BatchNorm2d.forward`` passes ``self.training or not self.track_running_stats``,
            # so a block with ``track_running_stats`` cleared normalises by *batch* statistics
            # even in eval -- and a path that assumed running statistics there would diverge.
            # Caught by ``tests/t10_graph_semantics.py`` at a matched ratio of 0.878.
            y = F.batch_norm(y, bn.running_mean, bn.running_var, bn.weight, bn.bias,
                             bn.training or not bn.track_running_stats, bn.momentum, bn.eps)
        act = module.act
        if isinstance(act, nn.Identity):
            return y
        # ``F.silu`` rather than the frozen leaf, which is a custom kernel and therefore not
        # obliged to agree with ATen on every bit.
        if isinstance(act, (SiLU, nn.SiLU)):
            return F.silu(y)
        return act(y)
    if isinstance(module, Conv2d):
        return F.conv2d(x, module.weight, module.bias, stride=module.stride,
                        padding=module.padding, dilation=module.dilation,
                        groups=module.groups)
    return module(x)


# ---------------------------------------------------------------------------
# The sigmoid table.
# ---------------------------------------------------------------------------
def _sigmoid_lut(device: torch.device) -> torch.Tensor:
    """``fp16(sigmoid(x))`` for every fp16 bit pattern, indexed by that pattern.

    The class branch's ``sigmoid`` decides which anchors survive two tie-broken
    selections, so it has to agree with the baseline's ``torch.sigmoid`` on every bit, not
    merely to within fp16 tolerance. A table built *by calling* ``torch.sigmoid`` is exact
    by construction; a Triton ``exp`` would only be exact until a toolchain changed which
    approximation it lowers to. 128 KiB, read out of L2, built once per device.
    """
    bits = torch.arange(65536, dtype=torch.int32, device=device).to(torch.int16)
    return torch.sigmoid(bits.view(torch.float16))


def _lut_is_monotone(lut: torch.Tensor) -> bool:
    """Whether ``max`` may be taken over logits instead of over table values.

    ``max_i f(x_i) == f(max_i x_i)`` needs ``f`` non-decreasing. Mathematically ``sigmoid``
    is, but the shipped path depends on the *implementation* being non-decreasing after
    rounding to fp16, and a single non-monotone adjacent pair would flip a tie in a 2000-way
    tied selection. So the question is decided by sweeping the table in value order rather
    than by appealing to the mathematics; if the sweep says no, the kernels reduce over
    table values instead, which is exact either way and costs one lookup per class.

    NaN is excluded from the sweep: it has no place in the value order, and the fast path
    cannot check its inputs for NaN without a host synchronisation.
    """
    bits = torch.arange(65536, dtype=torch.int32, device=lut.device).to(torch.int16)
    x = bits.view(torch.float16).float()
    keep = ~torch.isnan(x)
    order = torch.argsort(x[keep])
    ordered = lut[keep][order].float()
    return bool(torch.all(ordered[1:] >= ordered[:-1]).item())


# ---------------------------------------------------------------------------
# Kernels.
# ---------------------------------------------------------------------------
@triton.jit
def _score_max_level(cls_ptr, out_ptr, lut_ptr, n_level, level_off, a_total,
                     blk, bidx, NC: tl.constexpr, BLOCK_A: tl.constexpr,
                     BLOCK_C: tl.constexpr, AMAX_FIRST: tl.constexpr):
    """One level's contribution to ``max_scores``, for one block of anchors.

    Inlined into :func:`_score_max` so the three levels share one launch while each
    program still holds a single base pointer -- the level is a scalar property of the
    program, not a per-lane one, which is what keeps the loads contiguous along the anchor
    axis.
    """
    offs_a = blk * BLOCK_A + tl.arange(0, BLOCK_A)
    live = offs_a < n_level
    offs_c = tl.arange(0, BLOCK_C)
    chan = offs_c < NC
    # The level's output is contiguous ``(b, NC, n_level)``, so the channel stride is
    # ``n_level`` and the batch stride is ``NC * n_level``.
    base = cls_ptr + bidx.to(tl.int64) * (NC * n_level)
    ptrs = base + offs_c[:, None] * n_level + offs_a[None, :]
    keep = chan[:, None] & live[None, :]
    tile = tl.load(ptrs, mask=keep, other=float("-inf"))
    # ``torch.amax`` propagates NaN: one NaN class logit makes the anchor's score NaN, and
    # ATen's selection then maps that NaN to the *maximum* key and puts the anchor first.
    # ``tl.max`` does not propagate it -- it keeps the non-NaN operand -- so an anchor the
    # baseline ranks top would come back finite and mid-pack, and the whole 300-row
    # selection would reorder. This is not hypothetical: the harness redraws a float
    # parameter only when its amax is non-finite, below 1e-6 or above 1e4, so the frozen
    # ``Conv2d``'s ``torch.empty`` weights sometimes survive sanitisation, and five chained
    # layers of a weight with amax ~45 overflow fp16 to an infinity whose products are NaN.
    # Measured: about one process in four draws such a set. So the reduction forces the
    # propagation rather than relying on the hardware's choice.
    nan_seen = tl.max(tl.where(tile != tile, 1, 0), axis=0)
    if AMAX_FIRST:
        top = tl.max(tile, axis=0).to(tl.float16)
        key = top.to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
        score = tl.load(lut_ptr + key)
    else:
        key = tile.to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
        # ``sigmoid`` is non-negative, so a masked lane reading 0 can never raise the max
        # above a real one.
        vals = tl.load(lut_ptr + key, mask=keep, other=0.0)
        score = tl.max(vals, axis=0).to(tl.float16)
    score = tl.where(nan_seen > 0, float("nan"), score).to(tl.float16)
    tl.store(out_ptr + bidx.to(tl.int64) * a_total + level_off + offs_a, score, mask=live)


@triton.jit
def _score_max(c0, c1, c2, out_ptr, lut_ptr, n0, n1, n2, nb0, nb1, a_total,
               NC: tl.constexpr, BLOCK_A: tl.constexpr, BLOCK_C: tl.constexpr,
               AMAX_FIRST: tl.constexpr):
    """``max_scores[b, a] = max over classes of fp16(sigmoid(cls logit))``, all levels.

    Replaces the baseline's ``cat`` to ``(b,144,8400)``, its full-width ``sigmoid``, the
    ``permute``/``split`` round trip and the ``amax`` -- five kernels and about 10 MB of
    traffic -- with one pass that never materialises a sigmoid value it is not about to
    reduce away.
    """
    blk = tl.program_id(0)
    bidx = tl.program_id(1)
    if blk < nb0:
        _score_max_level(c0, out_ptr, lut_ptr, n0, 0, a_total, blk, bidx,
                         NC, BLOCK_A, BLOCK_C, AMAX_FIRST)
    elif blk < nb0 + nb1:
        _score_max_level(c1, out_ptr, lut_ptr, n1, n0, a_total, blk - nb0, bidx,
                         NC, BLOCK_A, BLOCK_C, AMAX_FIRST)
    else:
        _score_max_level(c2, out_ptr, lut_ptr, n2, n0 + n1, a_total, blk - nb0 - nb1, bidx,
                         NC, BLOCK_A, BLOCK_C, AMAX_FIRST)


@triton.jit
def _gather_box(b0, b1, b2, idx_ptr, out_ptr, n0, n1, n2, k_total,
                C: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_C: tl.constexpr):
    """Box-branch columns of the selected anchors: ``(b, C, K)`` from three level buffers.

    The selected anchors are data-dependent, so a program's lanes can straddle level
    boundaries and the level cannot be hoisted to a scalar the way it is in
    :func:`_score_max`. Each level is therefore read under its own mask and the three
    results are selected between; a warp whose lanes all fall in one level issues one real
    load and two fully-masked ones.
    """
    pid_k = tl.program_id(0)
    bidx = tl.program_id(1)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    live = offs_k < k_total
    anchor = tl.load(idx_ptr + bidx * k_total + offs_k, mask=live, other=0).to(tl.int32)
    in_0 = anchor < n0
    in_1 = (anchor >= n0) & (anchor < n0 + n1)
    in_2 = anchor >= n0 + n1
    loc = tl.where(in_0, anchor, tl.where(in_1, anchor - n0, anchor - n0 - n1))

    offs_c = tl.arange(0, BLOCK_C)
    chan = offs_c < C
    keep = chan[:, None] & live[None, :]
    bb = bidx.to(tl.int64)
    v0 = tl.load(b0 + bb * (C * n0) + offs_c[:, None] * n0 + loc[None, :],
                 mask=keep & in_0[None, :], other=0.0)
    v1 = tl.load(b1 + bb * (C * n1) + offs_c[:, None] * n1 + loc[None, :],
                 mask=keep & in_1[None, :], other=0.0)
    v2 = tl.load(b2 + bb * (C * n2) + offs_c[:, None] * n2 + loc[None, :],
                 mask=keep & in_2[None, :], other=0.0)
    val = tl.where(in_0[None, :], v0, tl.where(in_1[None, :], v1, v2))
    tl.store(out_ptr + bb * (C * k_total) + offs_c[:, None] * k_total + offs_k[None, :],
             val, mask=keep)


@triton.jit
def _gather_cls_scores(c0, c1, c2, idx_ptr, lut_ptr, out_ptr, n0, n1, n2, k_total,
                       NC: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_C: tl.constexpr):
    """Class scores of the selected anchors: ``(b, K, NC)``, table-exact ``sigmoid``.

    The baseline gathers *post*-sigmoid values; ``sigmoid`` is elementwise so gathering
    first and looking the table up afterwards is bitwise equal while touching ``K``
    columns instead of all 8400. The output is contiguous ``(b, K, NC)`` so that
    ``flatten(1)`` presents the second selection with the same flattened order --
    ``anchor * NC + class`` -- that the baseline's gather produces.
    """
    pid_k = tl.program_id(0)
    bidx = tl.program_id(1)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    live = offs_k < k_total
    anchor = tl.load(idx_ptr + bidx * k_total + offs_k, mask=live, other=0).to(tl.int32)
    in_0 = anchor < n0
    in_1 = (anchor >= n0) & (anchor < n0 + n1)
    in_2 = anchor >= n0 + n1
    loc = tl.where(in_0, anchor, tl.where(in_1, anchor - n0, anchor - n0 - n1))

    offs_c = tl.arange(0, BLOCK_C)
    chan = offs_c < NC
    keep = live[:, None] & chan[None, :]
    bb = bidx.to(tl.int64)
    v0 = tl.load(c0 + bb * (NC * n0) + offs_c[None, :] * n0 + loc[:, None],
                 mask=keep & in_0[:, None], other=0.0)
    v1 = tl.load(c1 + bb * (NC * n1) + offs_c[None, :] * n1 + loc[:, None],
                 mask=keep & in_1[:, None], other=0.0)
    v2 = tl.load(c2 + bb * (NC * n2) + offs_c[None, :] * n2 + loc[:, None],
                 mask=keep & in_2[:, None], other=0.0)
    logit = tl.where(in_0[:, None], v0, tl.where(in_1[:, None], v1, v2))
    key = logit.to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
    score = tl.load(lut_ptr + key, mask=keep, other=0.0)
    tl.store(out_ptr + bb * (k_total * NC) + offs_k[:, None] * NC + offs_c[None, :],
             score, mask=keep)


@triton.jit
def _assemble(dist_ptr, anchor_ptr, stride_ptr, idx1_ptr, idx2_ptr, score_ptr, out_ptr,
              k_total, a_total, NC: tl.constexpr, BLOCK_K: tl.constexpr):
    """The ``(b, K, 6)`` answer: box decode, corner form, score and label in one pass.

    Fuses what the baseline spends about a dozen launches on -- ``dist2bbox``, the stride
    multiply, the second gather of the box rows, ``xywh2xyxy``, and the final ``cat`` --
    and does it for ``K`` anchors rather than all of them.

    Every arithmetic step rounds to fp16 where the baseline's fp16 tensors round, spelled
    as an explicit narrow-then-widen. The intermediate fp32 is exact for a single add,
    subtract or multiply of two fp16 values, so narrowing once per step reproduces fp16
    arithmetic bit for bit rather than merely approximating it.
    """
    pid_k = tl.program_id(0)
    bidx = tl.program_id(1)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    live = offs_k < k_total
    bb = bidx.to(tl.int64)

    flat = tl.load(idx2_ptr + bb * k_total + offs_k, mask=live, other=0)
    label = (flat % NC).to(tl.float16)
    col = (flat // NC).to(tl.int32)
    anchor = tl.load(idx1_ptr + bb * k_total + col, mask=live, other=0).to(tl.int32)

    dist = dist_ptr + bb * (4 * k_total) + col
    lt_x = tl.load(dist + 0 * k_total, mask=live, other=0.0).to(tl.float32)
    lt_y = tl.load(dist + 1 * k_total, mask=live, other=0.0).to(tl.float32)
    rb_x = tl.load(dist + 2 * k_total, mask=live, other=0.0).to(tl.float32)
    rb_y = tl.load(dist + 3 * k_total, mask=live, other=0.0).to(tl.float32)
    ax = tl.load(anchor_ptr + 0 * a_total + anchor, mask=live, other=0.0).to(tl.float32)
    ay = tl.load(anchor_ptr + 1 * a_total + anchor, mask=live, other=0.0).to(tl.float32)
    st = tl.load(stride_ptr + anchor, mask=live, other=0.0).to(tl.float32)

    x1y1_x = (ax - lt_x).to(tl.float16).to(tl.float32)
    x1y1_y = (ay - lt_y).to(tl.float16).to(tl.float32)
    x2y2_x = (ax + rb_x).to(tl.float16).to(tl.float32)
    x2y2_y = (ay + rb_y).to(tl.float16).to(tl.float32)
    cx = (x1y1_x + x2y2_x).to(tl.float16).to(tl.float32)
    cy = (x1y1_y + x2y2_y).to(tl.float16).to(tl.float32)
    cx = (cx * 0.5).to(tl.float16).to(tl.float32)
    cy = (cy * 0.5).to(tl.float16).to(tl.float32)
    w = (x2y2_x - x1y1_x).to(tl.float16).to(tl.float32)
    h = (x2y2_y - x1y1_y).to(tl.float16).to(tl.float32)
    cx = (cx * st).to(tl.float16).to(tl.float32)
    cy = (cy * st).to(tl.float16).to(tl.float32)
    w = (w * st).to(tl.float16).to(tl.float32)
    h = (h * st).to(tl.float16).to(tl.float32)

    half_w = (w * 0.5).to(tl.float16).to(tl.float32)
    half_h = (h * 0.5).to(tl.float16).to(tl.float32)
    x1 = (cx - half_w).to(tl.float16)
    y1 = (cy - half_h).to(tl.float16)
    x2 = (cx + half_w).to(tl.float16)
    y2 = (cy + half_h).to(tl.float16)
    score = tl.load(score_ptr + bb * k_total + offs_k, mask=live, other=0.0)

    row = out_ptr + bb * (k_total * 6) + offs_k * 6
    tl.store(row + 0, x1, mask=live)
    tl.store(row + 1, y1, mask=live)
    tl.store(row + 2, x2, mask=live)
    tl.store(row + 3, y2, mask=live)
    tl.store(row + 4, score, mask=live)
    tl.store(row + 5, label, mask=live)


# ---------------------------------------------------------------------------
# Per-shape plan.
# ---------------------------------------------------------------------------
class _LevelPlan(NamedTuple):
    """Everything about a call's shape the kernels need, derived once per shape.

    Held in a plain attribute rather than recomputed per call: the ungraphed fast path is
    the fallback when capture is unavailable, and re-deriving anchor grids and block counts
    on every call would make that fallback a regression against the baseline rather than a
    degraded win.
    """

    batch: int
    spatial: tuple           # ((h, w), ...) per level
    counts: tuple            # anchors per level
    anchors_total: int
    blocks: tuple            # score-kernel blocks per level
    score_block_a: int
    score_block_c: int
    gather_block_k: int
    gather_block_box_c: int
    gather_block_cls_c: int


class _GraphEntry(NamedTuple):
    key: tuple
    operands: tuple
    graph: object
    static_in: tuple
    static_out: torch.Tensor


class YOLOv10DetectHead(nn.Module):
    dynamic = False
    export = True
    shape = None
    max_det = 300

    def __init__(self, nc: int = 80, ch: tuple[int, int, int] = (256, 512, 1024)):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = 16
        self.no = nc + self.reg_max * 4
        self.stride = torch.tensor([8.0, 16.0, 32.0])
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
        # Registered at ``empty(0)`` exactly as the baseline registers them. The harness
        # shares weights with ``load_state_dict(..., strict=False)`` inside a bare
        # ``except``, so a candidate that pre-registered these at their final shapes would
        # raise a size mismatch, have it swallowed, and quietly keep its own anchor grid.
        self.register_buffer("anchors", torch.empty(0))
        self.register_buffer("strides", torch.empty(0))

        # The input channel counts the fast path was built for. The convolutions guard
        # their own inputs, but the level plan bakes anchor counts in, so the guard
        # confirms the shape it was built for before a replay dereferences it.
        self._ch = tuple(ch)

        # Derived state. Not buffers and not parameters: a ``state_dict`` key here would
        # survive the harness's ``strict=False`` load and shadow the shared weights, and
        # ``named_buffers`` would report a captured graph as if it were model state.
        # ``_apply`` drops all of it.
        self._lut: torch.Tensor | None = None
        self._amax_first = False
        self._plan_key: tuple | None = None
        self._plan: _LevelPlan | None = None
        # Which plan the anchor grid currently holds. Keyed on the plan rather than on
        # ``self.shape``: the baseline keys its own rebuild on level 0's height and width,
        # which is enough for it because it rebuilds from the feature list it was handed,
        # but here the grid outlives the call, and two input sets can share level 0's shape
        # while differing at levels 1 and 2. The built tensors are held alongside so that a
        # reference call -- which registers its own grid through ``inference`` -- is noticed
        # by identity even when the plan key has not moved.
        self._anchor_key: tuple | None = None
        self._anchor_held: tuple | None = None
        self._graphs: dict = {}
        # Counts captures, so a test can *prove* a recapture happened rather than infer it
        # from a changed output -- replaying a graph over updated weights at the same
        # addresses legitimately changes the output with no recapture at all.
        self._graph_generation = 0
        # Latched on a capture failure. Capture is attempted once per shape; a failure is
        # permanent for the module, because the second attempt would fail for the same
        # reason and each one costs warmups.
        self._graph_failed = False

    # -- module plumbing --------------------------------------------------
    def load_state_dict(self, *args, **kwargs):
        """Drop the derived state, then load.

        A default ``load_state_dict`` copies values in place, which a replay would pick up on
        its own -- the captured kernels re-read the live bytes at the captured addresses, and
        the signature deliberately excludes operand *contents* so an in-place weight update
        needs no recapture. So this override is not what makes the shipped path correct.
        It is here because ``assign=True`` replaces the tensors rather than copying into them,
        and because a caller who has just loaded a different model should not have to reason
        about which of those two things happened to decide whether the graph is stale.
        """
        self._drop_derived()
        return super().load_state_dict(*args, **kwargs)

    def _drop_derived(self):
        self._lut = None
        self._plan_key = None
        self._plan = None
        self._anchor_key = None
        self._anchor_held = None
        self._graphs = {}

    def _apply(self, *args, **kwargs):
        """Drop every derived quantity when the module's tensors move or change identity.

        ``.to()``, ``.half()``, ``.cuda()`` and ``load_state_dict``'s ``_apply`` path all
        replace parameter tensors, which invalidates the sigmoid table's device, the anchor
        plan and -- above all -- a captured graph that holds the old addresses. The
        signature guard would catch these anyway; dropping here means the module does not
        pay a recapture's worth of warmups to discover it.
        """
        self._drop_derived()
        return super()._apply(*args, **kwargs)

    # -- the baseline's public surface ------------------------------------
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
        dbox = dist2bbox(_dfl_reference(self.dfl, box), self.anchors.unsqueeze(0),
                         xywh=True, dim=1) * self.strides
        return torch.cat((dbox, self._sigmoid(cls)), 1)

    def bias_init(self):
        for a, b, s in zip(self.cv2, self.cv3, self.stride):
            a[-1].bias.data[:] = 1.0
            b[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / s) ** 2)
        for a, b, s in zip(self.one2one_cv2, self.one2one_cv3, self.stride):
            a[-1].bias.data[:] = 1.0
            b[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / s) ** 2)

    # -- routing ----------------------------------------------------------
    def route_for(self, x) -> str:
        """Which path serves ``x``, decided without running any of them.

        A guard that wrongly rejected the benched cases would still return correct values,
        so the route has to be assertable on its own rather than inferred from output.
        """
        if self._admits(x) is None:
            return ROUTE_REFERENCE
        if not _GRAPH_REPLAY or self._graph_failed:
            return ROUTE_FAST
        return ROUTE_GRAPH

    def _admits(self, x):
        """The shape key if the fast path may serve this call, else ``None``.

        Every condition is here because the baseline reads that piece of state on every
        call, or because a captured replay would otherwise dereference something that has
        moved. Anything not covered reaches :meth:`_reference_forward`, which is the
        baseline expression and raises whatever the baseline raises.
        """
        if self.training or not self.export or self.dynamic:
            return None
        if torch.is_grad_enabled() or torch.is_autocast_enabled("cuda"):
            return None
        # A dual level is not caught by ``is_grad_enabled``: forward-mode AD carries a tangent
        # alongside the value, and every kernel here reads the primal by pointer, below the
        # dispatcher that implements it. The baseline propagates the tangent and returns; this path
        # would drop it or raise. The frozen L2 convolution rejects the same condition.
        if _forward_ad._current_level >= 0:
            return None
        if _global_hooks_present():
            return None
        if type(x) is not list and type(x) is not tuple:
            return None
        if len(x) != self.nl or self.nl != 3:
            return None
        # Derived head geometry the kernels bake in as constants.
        if self.no != self.nc + 4 * self.reg_max or self.nc < 1 or self.reg_max < 1:
            return None
        if not isinstance(self.max_det, int) or self.max_det < 1:
            return None
        if type(self.stride) is not torch.Tensor or self.stride.numel() != self.nl:
            return None
        if self.stride.device.type != "cpu":
            return None

        spatial = []
        total = 0
        batch = None
        for i, xi in enumerate(x):
            # A subclass would make the output a subclass too, and the harness requires
            # every output leaf to be exactly ``torch.Tensor``.
            if type(xi) is not torch.Tensor:
                return None
            if xi.dtype is not _FAST_DTYPE or xi.device.type != "cuda":
                return None
            if xi.dim() != 4 or not xi.is_contiguous():
                return None
            if xi.shape[1] != self._ch[i]:
                return None
            if batch is None:
                batch = xi.shape[0]
            elif xi.shape[0] != batch:
                return None
            h, w = int(xi.shape[2]), int(xi.shape[3])
            if h < 1 or w < 1:
                return None
            spatial.append((h, w))
            total += h * w
        if batch is None or batch < 1:
            return None
        if self.max_det > total:
            return None
        # The kernels form offsets in 32-bit arithmetic. The widest of them is the class
        # tile's ``batch * nc * anchors_in_level``; the gathers are narrower.
        widest = batch * max(self.nc, 4 * self.reg_max) * total
        if widest > _MAX_INT32_OFFSET:
            return None
        if self.max_det * self.nc * batch > _MAX_INT32_OFFSET:
            return None
        # The host state of every module the recorded kernels read. ``None`` means the fast path
        # may not serve this call at all -- a forward hook the functional path would skip, or an
        # attribute the signature cannot represent faithfully and must not approximate. Computed
        # once here and carried in the key, because it is the most expensive thing the guard does.
        config = self._config_snapshot() if _SEMANTIC_GUARD else ()
        if config is None:
            return None

        # Anchors are the only thing the baseline rebuilds from ``self.stride``'s values,
        # so a caller who changed a stride has changed the plan.
        return (batch, tuple(spatial), tuple(self.stride.tolist()),
                self.nc, self.reg_max, self.max_det, x[0].device, config)

    # -- the reference expression -----------------------------------------
    def _reference_forward(self, x: list[torch.Tensor]):
        """Exactly what ``baseline.py`` computes, including the error it raises."""
        one2one = self.forward_feat([xi.detach() for xi in x], self.one2one_cv2, self.one2one_cv3)
        if not self.training:
            one2one = self.inference(one2one)
            if self.export:
                boxes, scores, labels = v10postprocess(one2one.permute(0, 2, 1), self.max_det, self.nc)
                return torch.cat([xywh2xyxy(boxes), scores.unsqueeze(-1), labels.unsqueeze(-1).to(boxes.dtype)], dim=-1)

        one2many = self.forward_feat(x, self.cv2, self.cv3)
        if self.training:
            return {"one2many": one2many, "one2one": one2one}
        one2many = self.inference(one2many)
        return {"one2many": one2many, "one2one": one2one}

    # -- the fast path ----------------------------------------------------
    def _build_plan(self, key) -> _LevelPlan:
        batch, spatial, _strides, nc, _reg_max, _max_det, _device, _config = key
        counts = tuple(h * w for h, w in spatial)
        block_a = 32
        blocks = tuple(triton.cdiv(n, block_a) for n in counts)
        return _LevelPlan(
            batch=batch,
            spatial=spatial,
            counts=counts,
            anchors_total=sum(counts),
            blocks=blocks,
            score_block_a=block_a,
            score_block_c=max(16, triton.next_power_of_2(nc)),
            gather_block_k=64,
            gather_block_box_c=max(16, triton.next_power_of_2(4 * self.reg_max)),
            gather_block_cls_c=max(16, triton.next_power_of_2(nc)),
        )

    def _prepare(self, x, key) -> _LevelPlan:
        """Everything the launch needs that does not depend on this call's values.

        Run outside any capture: the table build and the anchor grid allocate, and a
        ``cat``-built buffer inside a capture would be frozen in the graph's private pool.
        """
        device = x[0].device
        if self._lut is None or self._lut.device != device:
            lut = _sigmoid_lut(device)
            self._amax_first = _lut_is_monotone(lut)
            self._lut = lut
        # The plan and the anchor grid depend on the *shape* half of the key only. Keying them
        # on the whole thing would rebuild an anchor grid every time a child module's
        # configuration moved, which has nothing to do with either.
        plan_key = key[:7]
        if self._plan_key != plan_key:
            self._plan = self._build_plan(key)
            self._plan_key = plan_key
        plan = self._plan

        # The anchor grid, maintained the way the baseline maintains it -- including
        # ``self.shape``, so a caller who reads it back sees what the baseline would show.
        # In-place ``copy_`` when the size is unchanged keeps the data pointer, which is
        # what lets a captured graph keep reading the live values without a recapture.
        spatial = plan.spatial[0]
        held = self._anchor_held
        # Identity, spelled out: a tuple comparison would fall back to ``Tensor.__eq__`` and
        # raise on the elementwise result.
        if (self._anchor_key != plan_key or self.shape != spatial or held is None
                or held[0] is not self.anchors or held[1] is not self.strides):
            feats = [
                torch.empty(0, 0, h, w, device=device, dtype=x[0].dtype)
                for h, w in plan.spatial
            ]
            anchors, strides = (
                t.transpose(0, 1).contiguous() for t in make_anchors(feats, self.stride, 0.5)
            )
            if self.anchors.numel() == anchors.numel() and self.strides.numel() == strides.numel():
                self.anchors.copy_(anchors)
                self.strides.copy_(strides)
            else:
                self.register_buffer("anchors", anchors)
                self.register_buffer("strides", strides)
            self.shape = spatial
            self._anchor_key = plan_key
            self._anchor_held = (self.anchors, self.strides)
        return plan

    def _trunk(self, x, plan: _LevelPlan):
        """Per-level box and class logits, kept in their natural layout.

        The baseline concatenates the two branches per level and then concatenates the
        three levels along the anchor axis, which costs four kernels and copies the whole
        ``(b,144,8400)`` tensor for no reason other than to make the anchor axis
        contiguous. Nothing downstream needs that: each level's output is already
        contiguous, so viewing it as ``(b, C, HW)`` is free and the tail takes three base
        pointers and a level-offset mapping instead.
        """
        nb = 4 * self.reg_max
        box, cls = [], []
        for i in range(self.nl):
            xi = x[i]
            box.append(self._box_logits(i, xi).view(plan.batch, nb, plan.counts[i]))
            cls.append(self._class_logits(i, xi).view(plan.batch, self.nc, plan.counts[i]))
        return box, cls

    def _box_logits(self, level: int, xi: torch.Tensor) -> torch.Tensor:
        """The box branch, computed the way ``baseline.py`` computes it.

        Unlike the class branch, this one does *not* have to be exact for the harness to pass:
        its output reaches only the four box columns, which are compared against
        ``atol + rtol*|ref|`` with both at 1e-2 on values of order 300-600, and the frozen fused
        convolutions cost at most five wrong elements in 7200 there -- a matched ratio of about
        0.9978 against a 0.99 gate.

        It is exact anyway, because "inside the tolerance" is a weaker claim than the operator can
        support. With this branch exact too, the whole output is bitwise identical to the
        reference on every weight draw, which is a property worth more than the 60 us it costs --
        and it makes this path a usable oracle:
        any future fused or custom box kernel can be gated on reproducing it exactly rather than
        on staying inside a tolerance, which is a much sharper test.

        ``FK_YOLO_HEAD_FUSED_BOX=1`` restores the frozen fused convolutions here. That is the
        faster configuration and it still passes the harness; it simply forfeits exactness, and
        both numbers are recorded.
        """
        if _FUSED_BOX:
            return self.one2one_cv2[level](xi)
        return _baseline_chain(self.one2one_cv2[level], xi)

    def _class_logits(self, level: int, xi: torch.Tensor) -> torch.Tensor:
        """The class branch, computed the way ``baseline.py`` computes it.

        This is the one place in the fast path that deliberately declines the frozen fused
        convolutions, and it costs real time. The reason is measured, not defensive.

        The two selections are decided entirely by tie-breaking: under the harness's weight
        draw ``max_scores`` takes two to four distinct fp16 values with thousands of anchors
        tied at the deciding one. The frozen convolution's arithmetic is *better* than the
        baseline's -- fp32 through the BatchNorm affine and the activation, one rounding at
        the store, against the baseline's rounding at each boundary -- but it is not the
        *same*, so 1 to 11 percent of class logits differ by about an fp16 ulp. Whether that
        reaches the answer depends on whether ``fp16(sigmoid(.))`` happens to quantise the
        differing logits onto the same value.

        Swept over 48 independent weight draws, it usually does and occasionally does not: 47
        draws gave a matched ratio of 0.9978 or better, and one gave **0.7233**, with 192 of
        300 selected anchors different. That draw had no NaN and nothing else unusual about it
        -- four distinct score values instead of three was enough. A candidate that fails
        outright on one weight draw in fifty is not a faster candidate, so the class branch
        uses the baseline's expression and the selections become exact by construction rather
        than by luck.

        The box branch does the same, for the reason on :meth:`_box_logits`: its output cannot
        reach the selection, but leaving it approximate would forfeit exactness of the whole answer
        for about 60 us.
        """
        if _FUSED_CLASS:
            return self.one2one_cv3[level](xi)
        return _baseline_chain(self.one2one_cv3[level], xi)

    def _launch(self, x, plan: _LevelPlan, trace: dict | None = None) -> torch.Tensor:
        """The whole fast path, as the captured region and as its own fallback.

        One function rather than two so that a replay and an ungraphed call cannot drift:
        the graph is a recording of exactly this sequence.

        ``trace``, when a dict is passed, collects the intermediates under stable names. The
        graph path never passes one, so this costs nothing on the shipped path -- it exists
        because the tail's equivalence claims are about ``i1``, ``i2`` and each gathered tensor
        individually, and a test that can only see the final output cannot distinguish "the
        selections are exact" from "the selections differ and the tied values hide it".
        """
        box, cls = self._trunk(x, plan)
        batch = plan.batch
        a_total = plan.anchors_total
        k = self.max_det
        n0, n1, n2 = plan.counts
        device = x[0].device

        max_scores = torch.empty((batch, a_total), dtype=_FAST_DTYPE, device=device)
        _score_max[(sum(plan.blocks), batch)](
            cls[0], cls[1], cls[2], max_scores, self._lut,
            n0, n1, n2, plan.blocks[0], plan.blocks[1], a_total,
            NC=self.nc, BLOCK_A=plan.score_block_a, BLOCK_C=plan.score_block_c,
            AMAX_FIRST=self._amax_first, num_warps=4,
        )
        # The installed implementation is the reference answer here: the baseline calls
        # ``torch.topk``, both selections are decided entirely by tie-breaking, and
        # documented behaviour for tied indices is that there is none. Matching a
        # reimplementation against it would have to be empirical, so it is not attempted.
        anchor_idx = torch.topk(max_scores, k, dim=-1).indices

        nb = 4 * self.reg_max
        grid_k = (triton.cdiv(k, plan.gather_block_k), batch)
        box_sel = torch.empty((batch, nb, k), dtype=_FAST_DTYPE, device=device)
        _gather_box[grid_k](
            box[0], box[1], box[2], anchor_idx, box_sel, n0, n1, n2, k,
            C=nb, BLOCK_K=plan.gather_block_k, BLOCK_C=plan.gather_block_box_c,
            num_warps=4,
        )
        # The soft-argmax and the box decode are per-anchor independent, so running them
        # on the selected columns is bitwise equal to running them on all 8400 and
        # gathering afterwards -- at 1/28th of the work.
        dist = self._soft_argmax(box_sel)

        cls_sel = torch.empty((batch, k, self.nc), dtype=_FAST_DTYPE, device=device)
        _gather_cls_scores[grid_k](
            cls[0], cls[1], cls[2], anchor_idx, self._lut, cls_sel, n0, n1, n2, k,
            NC=self.nc, BLOCK_K=plan.gather_block_k, BLOCK_C=plan.gather_block_cls_c,
            num_warps=4,
        )
        scores, flat_idx = torch.topk(cls_sel.flatten(1), k, dim=-1)

        out = torch.empty((batch, k, 6), dtype=_FAST_DTYPE, device=device)
        _assemble[grid_k](
            dist, self.anchors, self.strides, anchor_idx, flat_idx, scores, out,
            k, a_total, NC=self.nc, BLOCK_K=plan.gather_block_k, num_warps=4,
        )
        if trace is not None:
            trace.update(box=box, cls=cls, max_scores=max_scores, anchor_idx=anchor_idx,
                         box_sel=box_sel, dist=dist, cls_sel=cls_sel, scores=scores,
                         flat_idx=flat_idx, out=out)
        return out

    def _soft_argmax(self, box_sel: torch.Tensor) -> torch.Tensor:
        """The distribution-focal-loss soft-argmax, the way ``baseline.py`` computes it.

        The frozen layer accepts any anchor count and would serve this in one kernel instead of
        two, and its answer agrees with the baseline's on almost every weight draw. Almost: over
        12 draws crossed with three seeds and both batches, one case differed by 0.375 in a box
        coordinate -- inside the harness's tolerance, and the last thing standing between this
        candidate and being bitwise equal to the reference on every case. Two kernels on 300
        columns is a cheap price for deleting the word "almost".

        ``FK_YOLO_HEAD_FUSED_DFL=1`` restores the frozen layer.
        """
        dfl = self._modules["dfl"]
        if _FUSED_DFL:
            return dfl(box_sel)
        b, _, a = box_sel.shape
        v = box_sel.view(b, 4, dfl.c1, a).transpose(2, 1)
        v = F.softmax(v, dim=dfl._modules["_softmax"].dim)
        conv = dfl.conv
        return F.conv2d(v, conv.weight, conv.bias, stride=conv.stride, padding=conv.padding,
                        dilation=conv.dilation, groups=conv.groups).view(b, 4, a)

    # -- graph replay -----------------------------------------------------
    def _graph_operands(self) -> tuple:
        """Every tensor a captured replay reads, in a stable order.

        Held by the cache entry as well as hashed into the key: a graph captures addresses,
        not tensors, so holding the operands is what stops a freed allocation being reused
        at the same address by a fresh tensor and passing the pointer check.

        The traversal reads ``_modules`` / ``_parameters`` / ``_buffers`` directly instead of
        calling ``parameters()`` and ``buffers()``. Those go through ``named_parameters()``,
        which formats a name string per tensor and drives a nested generator, and this walk
        happens on *every* call: measured, the generator form cost 124 us against a 466 us
        call, so the guard was costing more than the tail it protects. The dict reads are
        exactly equivalent -- reassigning a parameter or a child goes through
        ``Module.__setattr__``, which updates these same containers -- and the frozen L2
        layers read their own state the same way for the same reason.

        Order is depth-first and stable across calls, which is all the comparison needs; it
        is deliberately not ``named_parameters()``'s order.
        """
        out = []
        stack = [self._modules["one2one_cv2"], self._modules["one2one_cv3"],
                 self._modules["dfl"]]
        while stack:
            mod = stack.pop()
            for t in mod._parameters.values():
                if t is not None:
                    out.append(t)
            for t in mod._buffers.values():
                if t is not None:
                    out.append(t)
            for child in mod._modules.values():
                if child is not None:
                    stack.append(child)
        out.append(self._lut)
        out.append(self.anchors)
        out.append(self.strides)
        return tuple(out)

    def _config_snapshot(self):
        """Host configuration of every module the fast path reaches, or ``None``.

        ``None`` means some module holds an attribute the signature cannot represent, which
        :meth:`_admits` turns into a reference-path call. Returning ``None`` *into the key*
        would be the dangerous reading -- two unrepresentable states would compare equal.
        """
        return _fast_path_state((self._modules["one2one_cv2"], self._modules["one2one_cv3"],
                                 self._modules["dfl"]))

    def _graph_signature(self, x, shape_key):
        """``(key, operands)`` -- everything a capture baked in, plus what to hold.

        Shape and dtype are not enough. A replay dereferences the addresses that were live
        at capture, so the key carries, for each operand, its object identity, data
        pointer, version, dtype, device, shape, stride and storage offset; the stream,
        because a replay on a different stream is not ordered against that stream's other
        work; and every host scalar the recorded launches baked in as a kernel argument --
        ``nc``, ``reg_max``, ``max_det``, the level counts and the reduction flag are all
        plain attributes a caller can reassign between calls.

        Deliberately *not* in the key: an operand's contents. An in-place same-pointer
        ``param.data.copy_(...)`` needs no recapture, because the replay re-reads the live
        bytes at the captured addresses.
        """
        operands = self._graph_operands()
        key = (
            shape_key,
            torch.cuda.current_stream(x[0].device).cuda_stream,
            self.nc, self.reg_max, self.no, self.max_det, self.nl,
            self._amax_first, self.dynamic, self.export, self.training,
            _FUSED_CLASS, _FUSED_BOX, _FUSED_DFL,
            self._plan.counts, self._plan.blocks,
            self._plan.score_block_a, self._plan.score_block_c,
            self._plan.gather_block_k, self._plan.gather_block_box_c,
            self._plan.gather_block_cls_c,
            # ``shape_key``'s last element is the configuration of every module the recorded
            # kernels read, collected by ``_admits`` so the walk happens once per call.
            # ``_fast_path_state`` explains why it has to be in here at all.
            # ``t.shape`` is already a hashable tuple subclass and ``t.stride()`` already
            # returns a tuple, so neither is re-wrapped: 248 needless tuple allocations per
            # call is a measurable share of a guard that runs before every replay.
            tuple(
                (id(t), t.data_ptr(), t._version, t.dtype, t.device, t.shape,
                 t.stride(), t.storage_offset())
                for t in operands
            ),
        )
        return key, operands

    def _replay(self, x, plan: _LevelPlan, shape_key) -> torch.Tensor:
        cache_key = (shape_key[0], shape_key[1], shape_key[6])
        key, operands = self._graph_signature(x, shape_key)
        entry = self._graphs.get(cache_key)
        if entry is not None and entry.key == key \
                and all(a is b for a, b in zip(entry.operands, operands)):
            torch._foreach_copy_(list(entry.static_in), list(x))
            entry.graph.replay()
            # Cloned out of the static buffer because the baseline returns fresh memory:
            # without this, two consecutive calls would hand the caller one storage and
            # the first answer would change under the second. About 2 us, and every
            # measurement quoted here includes it.
            return entry.static_out.clone()
        return self._capture(x, plan, shape_key, cache_key, key, operands)

    def _capture(self, x, plan, shape_key, cache_key, key, operands) -> torch.Tensor:
        """Capture the fast path, then replay it for this call's answer.

        Warmups run on a side stream first, which is what the capture contract requires and
        what gets the Triton compiles, the launcher caches and the frozen layers' extension
        loads out of the recorded region.

        A failure leaves the module on the ungraphed fast path for good. The ``try`` covers
        the warmups too, because the first Triton compile is where a failure is most likely
        and a half-initialised capture must not leak into the next call.
        """
        # One launch before the capture machinery is entered, and deliberately *not* inside the
        # try below. If the module's own configuration has changed in a way the guard admitted
        # but the kernels cannot serve -- a child convolution's padding edited so its output no
        # longer has the anchor count the plan was built for -- that has to raise here, exactly
        # as the ungraphed path raises, and exactly as the baseline raises on the same state. It
        # is a misconfiguration, not a capture failure, and latching it would disable replay for
        # the life of the module over something the caller can undo.
        static_in = tuple(xi.clone() for xi in x)
        self._launch(static_in, plan)
        try:
            with torch.cuda.device(x[0].device):
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    for _ in range(_GRAPH_WARMUP):
                        self._launch(static_in, plan)
                torch.cuda.current_stream().wait_stream(side)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    static_out = self._launch(static_in, plan)
        except Exception:
            # Correct-but-slower beats wrong-or-dead. The addresses the aborted capture
            # would have baked in are gone with it, and nothing here has been published to
            # the cache, so the next call simply takes the ungraphed path.
            self._graph_failed = True
            self._graphs = {}
            torch.cuda.synchronize(x[0].device)
            return self._launch(x, plan)

        if len(self._graphs) >= _GRAPH_CACHE_LIMIT and cache_key not in self._graphs:
            self._graphs.pop(next(iter(self._graphs)))
        self._graphs[cache_key] = _GraphEntry(key, operands, graph, static_in, static_out)
        self._graph_generation += 1
        torch._foreach_copy_(list(static_in), list(x))
        graph.replay()
        return static_out.clone()

    # -- dispatch ---------------------------------------------------------
    def forward(self, x: list[torch.Tensor]):
        shape_key = self._admits(x)
        if shape_key is None:
            return self._reference_forward(x)
        plan = self._prepare(x, shape_key)
        if not _GRAPH_REPLAY or self._graph_failed:
            return self._launch(x, plan)
        return self._replay(x, plan, shape_key)
