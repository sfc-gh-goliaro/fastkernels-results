"""YOLOv10 detection head (L3 composite)."""

from __future__ import annotations

import math
import copy

import torch
import torch.nn as nn
import triton
import triton.language as tl

from ..L1.conv2d import Conv2d
from ..L1.sigmoid import Sigmoid
from ..L2.yolov10_conv import YOLOConv, fuse_module
from ..L2.yolov10_dfl import YOLODFL


@triton.jit
def _load_dfl_side(
    x0,
    x1,
    x2,
    batch,
    bins,
    a0,
    a1,
    a2,
    m0,
    m1,
    m2,
    N0: tl.constexpr,
    N1: tl.constexpr,
    N2: tl.constexpr,
    SIDE: tl.constexpr,
):
    channel = SIDE * 16 + bins
    value = tl.load(
        x0 + batch * 64 * N0 + channel * N0 + a0,
        mask=m0,
        other=0.0,
    )
    value += tl.load(
        x1 + batch * 64 * N1 + channel * N1 + a1,
        mask=m1,
        other=0.0,
    )
    value += tl.load(
        x2 + batch * 64 * N2 + channel * N2 + a2,
        mask=m2,
        other=0.0,
    )
    value = value.to(tl.float32)
    value = tl.exp(value - tl.max(value, axis=0))
    return tl.sum(value * bins, axis=0) / tl.sum(value, axis=0)


@triton.jit
def _dfl_decode_xyxy(
    x0,
    x1,
    x2,
    anchors,
    strides,
    out,
    N: tl.constexpr,
    N0: tl.constexpr,
    N1: tl.constexpr,
    N2: tl.constexpr,
    OUT_CHANNELS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    a = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    batch = tl.program_id(1)
    bins = tl.arange(0, 16)[:, None]
    mask = a[None, :] < N
    m0 = mask & (a[None, :] < N0)
    m1 = mask & (a[None, :] >= N0) & (a[None, :] < N0 + N1)
    m2 = mask & (a[None, :] >= N0 + N1)
    a0 = a[None, :]
    a1 = a[None, :] - N0
    a2 = a[None, :] - N0 - N1

    d0 = _load_dfl_side(x0, x1, x2, batch, bins, a0, a1, a2, m0, m1, m2, N0, N1, N2, 0)
    d1 = _load_dfl_side(x0, x1, x2, batch, bins, a0, a1, a2, m0, m1, m2, N0, N1, N2, 1)
    d2 = _load_dfl_side(x0, x1, x2, batch, bins, a0, a1, a2, m0, m1, m2, N0, N1, N2, 2)
    d3 = _load_dfl_side(x0, x1, x2, batch, bins, a0, a1, a2, m0, m1, m2, N0, N1, N2, 3)

    anchor_x = tl.load(anchors + a, mask=a < N)
    anchor_y = tl.load(anchors + N + a, mask=a < N)
    stride = tl.load(strides + a, mask=a < N)
    d0 = d0.to(tl.float16)
    d1 = d1.to(tl.float16)
    d2 = d2.to(tl.float16)
    d3 = d3.to(tl.float16)

    left = (anchor_x - d0).to(tl.float16)
    top = (anchor_y - d1).to(tl.float16)
    right = (anchor_x + d2).to(tl.float16)
    bottom = (anchor_y + d3).to(tl.float16)
    center_x = ((left + right).to(tl.float16) * 0.5).to(tl.float16)
    center_y = ((top + bottom).to(tl.float16) * 0.5).to(tl.float16)
    width = (right - left).to(tl.float16)
    height = (bottom - top).to(tl.float16)
    center_x = (center_x * stride).to(tl.float16)
    center_y = (center_y * stride).to(tl.float16)
    width = (width * stride).to(tl.float16)
    height = (height * stride).to(tl.float16)
    half_width = (width * 0.5).to(tl.float16)
    half_height = (height * 0.5).to(tl.float16)
    out_base = batch * OUT_CHANNELS * N + a
    tl.store(out + out_base, center_x - half_width, mask=a < N)
    tl.store(out + out_base + N, center_y - half_height, mask=a < N)
    tl.store(out + out_base + 2 * N, center_x + half_width, mask=a < N)
    tl.store(out + out_base + 3 * N, center_y + half_height, mask=a < N)


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
        self._one2one_fused = False
        for module in self.one2one_cv3.modules():
            if isinstance(module, YOLOConv):
                module.act = nn.SiLU()

    def forward_feat(self, x: list[torch.Tensor], cv2, cv3):
        y = []
        for i in range(self.nl):
            y.append(torch.cat((cv2[i](x[i]), cv3[i](x[i])), 1))
        return y

    def inference_export(self, box: list[torch.Tensor], cls: list[torch.Tensor]):
        b, _, h, w = box[0].shape
        spatial = (h, w)
        if self.dynamic or self.shape != spatial:
            anchors, strides = (t.transpose(0, 1).contiguous() for t in make_anchors(box, self.stride, 0.5))
            self.anchors = anchors
            self.strides = strides
            self.shape = spatial

        box = [xi.flatten(2) for xi in box]
        cls = torch.cat([xi.flatten(2) for xi in cls], 2)
        n0, n1, n2 = (xi.shape[2] for xi in box)
        n = n0 + n1 + n2
        boxes = torch.empty((b, 4, n), device=box[0].device, dtype=box[0].dtype)
        _dfl_decode_xyxy[(triton.cdiv(n, 32), b)](
            box[0],
            box[1],
            box[2],
            self.anchors,
            self.strides,
            boxes,
            N=n,
            N0=n0,
            N1=n1,
            N2=n2,
            OUT_CHANNELS=4,
            BLOCK=32,
            num_warps=2,
            num_stages=2,
        )
        preds = torch.cat((boxes, torch.sigmoid(cls)), 1)
        boxes, scores, labels = v10postprocess(preds.permute(0, 2, 1), self.max_det, self.nc)
        return torch.cat(
            (boxes, scores.unsqueeze(-1), labels.unsqueeze(-1).to(boxes.dtype)), dim=-1
        )

    def inference(self, x: list[torch.Tensor]):
        b, _, h, w = x[0].shape
        x_cat = torch.cat([xi.view(b, self.no, -1) for xi in x], 2)
        # Anchors depend only on feature-map HW. Keying on full BCHW rebuilt
        # them on every batch-size change and freed the buffers Hopper CUDA
        # graphs had captured.
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
        dbox = dist2bbox(self.dfl(box.contiguous()), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        return torch.cat((dbox, torch.sigmoid(cls)), 1)

    def forward(self, x: list[torch.Tensor]):
        if not self.training and not self._one2one_fused:
            fuse_module(self.one2one_cv2)
            self._one2one_fused = True

        if not self.training and self.export:
            box, cls = [], []
            for i in range(self.nl):
                box.append(self.one2one_cv2[i](x[i].detach()))
                cls.append(self.one2one_cv3[i](x[i].detach()))
            return self.inference_export(box, cls)

        one2one = self.forward_feat([xi.detach() for xi in x], self.one2one_cv2, self.one2one_cv3)
        if not self.training:
            one2one = self.inference(one2one)

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
