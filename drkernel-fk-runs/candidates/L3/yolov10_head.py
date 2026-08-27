import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import Triton; provide a soft-fallback if unavailable.
try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False


def _next_power_of_2(x: int) -> int:
    return 1 if x <= 1 else 1 << (x - 1).bit_length()


if _HAS_TRITON:
    @triton.jit
    def _softmax_conv1x1_fused_kernel(
        x_ptr,            # *f16/f32/bf16
        w_ptr,            # *f16/f32/bf16 (1D of length C)
        y_ptr,            # *f16/f32/bf16
        C: tl.constexpr,  # int: channels
        stride_b: tl.constexpr,
        stride_k: tl.constexpr,
        stride_c: tl.constexpr,
        stride_a: tl.constexpr,
        out_stride_b: tl.constexpr,
        out_stride_k: tl.constexpr,
        out_stride_a: tl.constexpr,
        BLOCK: tl.constexpr,  # power-of-two block
    ):
        # One program per (b, k, a) row
        pid = tl.program_id(0)

        total_k = 4
        # grid = B * total_k * A
        R = total_k * A
        b = pid // R
        r = pid % R
        k = r % total_k
        a = r // total_k

        # Base offset for (b, k, a=0)
        base_bka = b * stride_b + k * stride_k
        c = tl.arange(0, BLOCK)
        mask = c < C

        # Load x[b, k, c, a] as float32
        x_ptrs = x_ptr + base_bka + c * stride_c + a * stride_a
        x = tl.load(x_ptrs, mask=mask, other=-float('inf')).to(tl.float32)

        # Stable softmax
        x_max = tl.max(x, axis=0)
        x = x - x_max
        num = tl.exp(x)
        den = tl.sum(num, axis=0)
        soft = num / den  # [BLOCK]

        # Load weights w[c] as float32
        w = tl.load(w_ptr + c, mask=mask, other=0.0).to(tl.float32)

        # Fused multiply-reduce
        out_val = tl.sum(soft * w, axis=0)  # scalar float32

        # Store y[b, k, a]
        y_ptr_out = y_ptr + b * out_stride_b + k * out_stride_k + a * out_stride_a
        tl.store(y_ptr_out, out_val)


def _softmax_conv1x1_fused(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    Fused softmax (over dim=1) + 1x1 conv (dot over dim=1) using Triton.

    Args:
      x: shape (B, 4, C, A), floating dtype.
      w: shape (1, C, 1, 1) or 1D of length C, same/device dtype.

    Returns:
      y: shape (B, 4, A), same dtype as x.
    """
    assert x.is_cuda, "Triton fused kernel requires CUDA tensor"
    assert x.ndim == 4, f"Expected 4D tensor (B,4,C,A), got shape {tuple(x.shape)}"
    B, K, C, A = x.shape
    assert K == 4, f"Expected dim=1 size 4, got {K}"

    # Prepare weight as 1D vector
    if w.ndim == 4:
        assert w.shape == (1, C, 1, 1), f"Weight shape must be (1,{C},1,1), got {tuple(w.shape)}"
        w_1d = w.view(C).contiguous()
    else:
        assert w.ndim == 1 and w.shape[0] == C, f"Weight must be 1D of length {C}, got shape {tuple(w.shape)}"
        w_1d = w.contiguous()

    # Output
    y = torch.empty((B, K, A), device=x.device, dtype=x.dtype)

    # Block size
    BLOCK = _next_power_of_2(C)
    BLOCK = min(BLOCK, 1024)

    # Strides in elements
    sb, sk, sc, sa = x.stride()
    y_sb, y_sk, y_sa = y.stride()  # y has shape (B,K,A)

    # Grid: one program per (b,k,a)
    grid = (B * K * A,)

    # Heuristics
    num_warps = 1 if C <= 64 else 2

    _softmax_conv1x1_fused_kernel[grid](
        x, w_1d, y,
        C,
        sb, sk, sc, sa,
        y_sb, y_sk, y_sa,
        BLOCK,
        num_warps=num_warps,
        num_stages=1,
    )
    return y


class YOLODFL(nn.Module):
    def __init__(self, c1: int = 16):
        super().__init__()
        self.conv = Conv2d(c1, 1, 1, bias=False)
        self.conv.requires_grad_(False)
        # Initialize weight to arange [0..c1-1]
        x = torch.arange(c1, dtype=torch.float32)
        self.conv.weight.data.copy_(x.view(1, c1, 1, 1))
        self.c1 = c1
        self._softmax = Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute y = conv(softmax(x, dim=1)) where x shape is (B,4,C1,A).
        Returns y shape (B,4,A).
        Uses Triton fused kernel on CUDA, falls back to PyTorch otherwise.
        """
        # Expect shape (B,4,C1,A)
        assert x.ndim == 4 and x.shape[1] == 4 and x.shape[2] == self.c1, \
            f"Expected x shape (B,4,{self.c1},A), got {tuple(x.shape)}"

        if not _HAS_TRITON or not x.is_cuda:
            sx = self._softmax(x)  # (B,4,C1,A)
            return self.conv(sx).view(-1, 4, x.shape[-1])  # (B,4,A)

        # Run fused kernel; compute in fp32 for stability, cast back
        in_dtype = x.dtype
        x32 = x if x.dtype == torch.float32 else x.float()
        w32 = self.conv.weight if self.conv.weight.dtype == torch.float32 else self.conv.weight.float()
        y32 = _softmax_conv1x1_fused(x32, w32)  # (B,4,A) float32
        y = y32 if in_dtype == torch.float32 else y32.to(in_dtype)
        return y.view(-1, 4, x.shape[-1])


# Keep the rest of the code identical for drop-in replacement.

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
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.dilation = dilation

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, *kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            x,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


class Softmax(nn.Module):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(x, dim=self.dim)


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


class BatchNorm2d(nn.Module):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        if track_running_stats:
            self.register_buffer("running_mean", torch.zeros(num_features))
            self.register_buffer("running_var", torch.ones(num_features))
            self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.track_running_stats and self.num_batches_tracked is not None:
            self.num_batches_tracked.add_(1)
        return F.batch_norm(
            x,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            self.training or not self.track_running_stats,
            self.momentum,
            self.eps,
        )


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


class SiLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x)


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._is_fused:
            return self.act(self.conv(x))
        return self.act(self.bn(self.conv(x)))

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


class Sigmoid(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x)


# YOLOv10 detection head (L3 composite).
from __future__ import annotations

import math
import copy

import torch
import torch.nn as nn

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


class ModelNew(nn.Module):
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
        self.dfl = YOLODFL(self.reg_max)  # will use Triton fused forward
        self._sigmoid = Sigmoid()
        self.one2one_cv2 = copy.deepcopy(self.cv2)
        self.one2one_cv3 = copy.deepcopy(self.cv3)
        self.register_buffer("anchors", torch.empty(0))
        self.register_buffer("strides", torch.empty(0))

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

    def forward(self, x: list[torch.Tensor]):
        # one2one uses a deep copy; it won't use our Triton path (still correct)
        one2one = self.forward_feat([xi.detach() for xi in x], self.one2one_cv2, self.one2one_cv3)
        if not self.training:
            one2one = self.inference(one2one)
            if self.export:
                boxes, scores, labels = v10postprocess(one2one.permute(0, 2, 1), self.max_det, self.nc)
                return torch.cat([xywh2xyxy(boxes), scores.unsqueeze(-1), labels.unsqueeze(-1).to(boxes.dtype)], dim=-1)

        # one2many is the working path; YOLODFL here will use Triton
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

YOLOv10DetectHead = ModelNew
