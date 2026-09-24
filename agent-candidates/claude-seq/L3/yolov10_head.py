"""YOLOv10 detection head (L3 composite) -- the whole head in three launches.

The baseline runs ~100 eager ops here: 24 convolutions each split into
conv + BatchNorm + SiLU, then the inference tail (DFL softmax, dist2bbox,
sigmoid) and ``v10postprocess``, whose two ``torch.topk`` calls alone cost ~145us
of the ~1.7ms total.  Only ~0.9ms of that is kernel time -- the rest is the CPU
never getting ahead of the GPU.

So the candidate hands the entire head to one extension call (see
``yolov10_head_fk.cu``).  Everything downstream of the NCHW inputs lives in
NHWC, which makes the 1x1 convs plain GEMMs, the 3x3 im2col gather one
contiguous ``Cin``-half run per tap, and the per-anchor tail (4x16 DFL softmax,
80-way class max) a single row read.  BatchNorm folds into a per-channel
scale/shift applied in the mma epilogue with SiLU riding along, the inference
tail rides in the last conv's epilogue of each branch, and the 3 levels x
{cv2, cv3} chains run on side streams -- captured once into a CUDA graph, since
at ~5us of host launch cost per kernel the launches outweighed the work.

The fp16 roundings the baseline performs *between* conv, BN and SiLU are
reproduced, so the class logits land on the same fp16 grid.  That is not
cosmetic: with the bench's random weights the class scores collapse onto three
distinct fp16 values, so which 300 anchors survive the top-k is decided almost
entirely by tie-breaking.  The custom top-k reproduces ``torch.topk``'s rule
(value descending, ties by ascending index) exactly, and the per-anchor maxima
it selects on come out bit-identical to the baseline's.

The packed weights / folded BN pairs are built once and dropped again by every
mutation torch routes through the module (``train()``, ``_apply``, a state-dict
load), so a cached fold can never go stale.
"""

from __future__ import annotations

import math
import copy

import torch
import torch.nn as nn

from ....infra.cuda_ext import lazy_op
from ..L1.conv2d import Conv2d
from ..L1.sigmoid import Sigmoid
from ..L2.yolov10_conv import YOLOConv
from ..L2.yolov10_dfl import YOLODFL

_C = lazy_op("yolov10_head_fk", "yolov10_head_fk.cu")

_LSTRIDE = 40
_PLAN_LEN = 128


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


# ---------------------------------------------------------------------------
# weight packing
# ---------------------------------------------------------------------------
def _fold_bn(bn: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    """``(scale, shift)`` in fp32: ``y = scale * conv + shift`` before the act."""
    var = bn.running_var.detach().to(torch.float32)
    mean = bn.running_mean.detach().to(torch.float32)
    w = bn.weight.detach().to(torch.float32)
    b = bn.bias.detach().to(torch.float32)
    scale = w * torch.rsqrt(var + bn.eps)
    return scale, b - mean * scale


def _pack_dense(conv: nn.Module) -> torch.Tensor:
    """Conv weight as ``[Cout][R*S][Cin]`` (k contiguous for the implicit GEMM)."""
    w = conv.weight.detach()
    cout = w.shape[0]
    return w.permute(0, 2, 3, 1).contiguous().view(cout, -1)


def _pack_dw(conv: nn.Module) -> torch.Tensor:
    """Depthwise weight as ``[R*S][C]``."""
    w = conv.weight.detach()
    c = w.shape[0]
    return w.view(c, -1).t().contiguous()


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
        self.register_buffer("anchors", torch.empty(0))
        self.register_buffer("strides", torch.empty(0))
        self._ch = tuple(ch)
        self._c2 = c2
        self._c3 = c3
        self._fk = None          # packed weights / plan / workspace
        self._fk_ok = None       # None = untested, False = unsupported here
        self._ext = None

    # -- cache invalidation on every mutation torch routes through ----------
    def _drop_fk(self):
        self._fk = None

    def train(self, mode: bool = True):
        self._drop_fk()
        return super().train(mode)

    def _apply(self, *args, **kwargs):
        self._drop_fk()
        return super()._apply(*args, **kwargs)

    def _load_from_state_dict(self, *args, **kwargs):
        self._drop_fk()
        return super()._load_from_state_dict(*args, **kwargs)

    # -- baseline path ------------------------------------------------------
    def forward_feat(self, x: list[torch.Tensor], cv2, cv3):
        y = []
        for i in range(self.nl):
            y.append(torch.cat((cv2[i](x[i]), cv3[i](x[i])), 1))
        return y

    def _set_anchors(self, feats):
        b, _, h, w = feats[0].shape
        spatial = (h, w)
        if self.dynamic or self.shape != spatial:
            anchors, strides = (t.transpose(0, 1).contiguous() for t in make_anchors(feats, self.stride, 0.5))
            if self.anchors.numel() == anchors.numel() and self.strides.numel() == strides.numel():
                self.anchors.copy_(anchors)
                self.strides.copy_(strides)
            else:
                self.register_buffer("anchors", anchors)
                self.register_buffer("strides", strides)
            self.shape = spatial

    def inference(self, x: list[torch.Tensor]):
        b, _, h, w = x[0].shape
        x_cat = torch.cat([xi.view(b, self.no, -1) for xi in x], 2)
        self._set_anchors(x)
        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        return torch.cat((dbox, self._sigmoid(cls)), 1)

    def _forward_ref(self, x: list[torch.Tensor]):
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

    # -- fast path ----------------------------------------------------------
    def _set_anchors_from(self, x):
        """Anchors depend only on the feature-map HW, which the (stride-1) head
        preserves -- so the inputs' own spatial dims are enough."""
        self._set_anchors([t[:, :1] for t in x])

    def _structure_ok(self) -> bool:
        if self.nl != 3 or self.nc != 80 or self.reg_max != 16:
            return False
        if self._c2 != 64 or self._c3 != 80 or self.max_det != 300:
            return False
        if any(c not in (64, 128, 256) for c in self._ch):
            return False
        for m in list(self.one2one_cv2.modules()) + list(self.one2one_cv3.modules()):
            if not isinstance(m, YOLOConv):
                continue
            if getattr(m, "_is_fused", False) or not hasattr(m, "bn"):
                return False
            bn = m.bn
            if not getattr(bn, "track_running_stats", False) or bn.running_mean is None:
                return False
            if bn.weight is None or bn.bias is None:
                return False
            if not isinstance(m.act, type(YOLOConv.default_act)):
                return False
            c = m.conv
            if c.stride != (1, 1) or c.dilation != (1, 1) or c.bias is not None:
                return False
            k = c.weight.shape[2]
            if k != c.weight.shape[3] or c.padding != (k // 2, k // 2):
                return False
            if c.groups != 1 and c.groups != c.weight.shape[0]:
                return False
        return True

    def _inputs_ok(self, x) -> bool:
        if not isinstance(x, (list, tuple)) or len(x) != 3:
            return False
        b = None
        for i, t in enumerate(x):
            if not (isinstance(t, torch.Tensor) and t.is_cuda and t.dim() == 4):
                return False
            if t.dtype != torch.float16 or not t.is_contiguous():
                return False
            if t.shape[1] != self._ch[i]:
                return False
            if t.shape[3] != t.shape[2] or int(t.shape[2]) < 1:
                return False
            if b is None:
                b = int(t.shape[0])
            elif int(t.shape[0]) != b:
                return False
        return True

    @torch.no_grad()
    def _build_fk(self, x):
        dev = x[0].device
        nimg = int(x[0].shape[0])
        A = 0
        specs = []
        for i in range(3):
            S = int(x[i].shape[2])
            specs.append((self._ch[i], S, S * S, A))
            A += S * S

        wparts, pparts = [], []
        wlen = plen = 0

        def add_w(t):
            nonlocal wlen
            off = wlen
            t = t.reshape(-1).to(torch.float16)
            wparts.append(t)
            wlen += (t.numel() + 7) & ~7
            pad = wlen - off - t.numel()
            if pad:
                wparts.append(torch.zeros(pad, device=t.device, dtype=torch.float16))
            return off

        def add_p(v):
            nonlocal plen
            off = plen
            v = v.reshape(-1).to(torch.float32)
            pparts.append(v)
            plen += v.numel()
            return off

        plan = [0] * _PLAN_LEN
        for lv in range(3):
            C, S, HW, aoff = specs[lv]
            cv2, cv3 = self.one2one_cv2[lv], self.one2one_cv3[lv]
            layers = [
                ("d", cv2[0]), ("d", cv2[1]), ("b", cv2[2]),
                ("w", cv3[0][0]), ("d", cv3[0][1]),
                ("w", cv3[1][0]), ("d", cv3[1][1]), ("b", cv3[2]),
            ]
            L = lv * _LSTRIDE
            plan[L + 0], plan[L + 1], plan[L + 2], plan[L + 3] = C, S, HW, aoff
            for j, (kind, mod) in enumerate(layers):
                if kind == "b":
                    w = _pack_dense(mod)
                    cout = w.shape[0]
                    sc = torch.ones(cout, device=dev, dtype=torch.float32)
                    sh = (mod.bias.detach().to(torch.float32) if mod.bias is not None
                          else torch.zeros(cout, device=dev, dtype=torch.float32))
                elif kind == "d":
                    w = _pack_dense(mod.conv)
                    sc, sh = _fold_bn(mod.bn)
                else:
                    w = _pack_dw(mod.conv)
                    sc, sh = _fold_bn(mod.bn)
                plan[L + 4 + j] = add_w(w)
                plan[L + 12 + j] = add_p(sc)
                plan[L + 20 + j] = add_p(sh)

        wbuf = torch.cat(wparts).contiguous()
        pbuf = torch.cat(pparts).contiguous()

        total = 0

        def alloc(n):
            nonlocal total
            off = total
            total += (n + 7) & ~7
            return off

        for lv in range(3):
            C, S, HW, aoff = specs[lv]
            L = lv * _LSTRIDE
            plan[L + 28] = alloc(nimg * HW * C)
            plan[L + 29] = alloc(nimg * HW * 64)
            plan[L + 30] = alloc(nimg * HW * 64)
            plan[L + 31] = alloc(nimg * HW * C)
            plan[L + 32] = alloc(nimg * HW * 80)
            plan[L + 33] = alloc(nimg * HW * 80)
        plan[120] = alloc(nimg * A * 64)
        plan[121] = alloc(nimg * A * self.nc)
        plan[122] = alloc(nimg * A * 4)
        plan[123] = alloc(nimg * A * self.nc)
        plan[124] = alloc(nimg * A)

        ws = torch.empty(total, device=dev, dtype=torch.float16)
        plan_t = torch.tensor(plan, dtype=torch.int64)
        dflw = self.dfl.conv.weight.detach().reshape(-1).to(torch.float16).contiguous()
        self._set_anchors_from(x)
        return {
            "nimg": nimg,
            "shapes": tuple(int(t.shape[2]) for t in x),
            "args": (wbuf, pbuf, self.anchors, self.strides, dflw, plan_t, ws,
                     nimg, A, self.nc, self.max_det),
        }

    def forward(self, x: list[torch.Tensor]):
        fk = self._fk
        if fk is not None:
            x0, x1, x2 = x
            if (x0.shape[0] == fk["nimg"] and x0.dtype is torch.float16
                    and x0.is_contiguous() and x1.is_contiguous()
                    and x2.is_contiguous()
                    and (x0.shape[2], x1.shape[2], x2.shape[2]) == fk["shapes"]):
                return self._ext.head_forward(x0, x1, x2, *fk["args"])
        return self._forward_setup(x)

    def _forward_setup(self, x):
        if (self._fk_ok is False or self.training or not self.export
                or self.dynamic or not self._structure_ok()
                or not self._inputs_ok(x)):
            return self._forward_ref(x)
        try:
            self._ext = _C._load()
        except Exception:
            self._fk_ok = False
            return self._forward_ref(x)
        self._fk_ok = True
        fk = self._build_fk(x)
        self._fk = fk
        return self._ext.head_forward(x[0], x[1], x[2], *fk["args"])

    def bias_init(self):
        for a, b, s in zip(self.cv2, self.cv3, self.stride):
            a[-1].bias.data[:] = 1.0
            b[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / s) ** 2)
        for a, b, s in zip(self.one2one_cv2, self.one2one_cv3, self.stride):
            a[-1].bias.data[:] = 1.0
            b[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / s) ** 2)
