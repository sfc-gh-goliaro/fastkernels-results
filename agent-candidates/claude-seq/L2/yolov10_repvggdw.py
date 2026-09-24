"""YOLOv10 RepVGG depthwise block.

At inference ``silu(BN(dw7x7(x)) + BN(dw3x3(x)))`` is one depthwise 7x7
convolution with a per-channel bias followed by SiLU: both BNs are affine in
eval mode, and the 3x3 kernel is the centre of a 7x7 one.  The baseline spends
six launches on it (2 convs, 2 batch norms, add, silu), each latency-bound on a
0.8 MB tensor.  Here the BN folding and the kernel merge happen once, on the
first forward, and a single custom kernel does conv + bias + SiLU.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ....infra.cuda_ext import lazy_op
from ..L1.silu import SiLU
from ..L1.tensor_ops import Pad
from .yolov10_conv import YOLOConv

_C = lazy_op("yolov10_repvggdw_fk", "yolov10_repvggdw_fk.cu")

_FAST_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _fold_bn(yc: YOLOConv):
    """``(weight_fp32, bias_fp32)`` for one YOLOConv with its BN folded in, or
    ``None`` if it is not a stride-1 'same'-padded depthwise conv."""
    conv = yc.conv
    if conv.stride != (1, 1) or conv.dilation != (1, 1):
        return None
    w = conv.weight.data
    if w.dim() != 4 or w.size(1) != 1 or conv.groups != w.size(0):
        return None  # not depthwise
    k = w.size(2)
    if w.size(3) != k or k % 2 == 0 or conv.padding != (k // 2, k // 2):
        return None  # not an odd 'same' kernel
    if not isinstance(yc.act, nn.Identity):
        return None
    wf = w.float()
    bf = conv.bias.data.float() if conv.bias is not None else None
    bn = getattr(yc, "bn", None)
    if bn is None:  # already fused by YOLOConv.fuse()
        return wf, bf if bf is not None else wf.new_zeros(wf.size(0))
    if bn.training or not bn.affine or not bn.track_running_stats:
        return None
    scale = bn.weight.data.float() / torch.sqrt(bn.running_var.data.float() + bn.eps)
    bias = bn.bias.data.float() - bn.running_mean.data.float() * scale
    if bf is not None:
        bias = bias + bf * scale
    return wf * scale.view(-1, 1, 1, 1), bias


class YOLORepVGGDW(nn.Module):
    def __init__(self, ed: int):
        super().__init__()
        self.conv = YOLOConv(ed, ed, 7, 1, 3, g=ed, act=False)
        self.conv1 = YOLOConv(ed, ed, 3, 1, 1, g=ed, act=False)
        self.act = SiLU()
        self._pad = Pad()
        self._is_fused = False
        self._merged = None  # (weight_fp32, bias_fp32) or False if unsupported

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        merged = self._merged
        if merged is None:
            merged = self._merge()
        if merged is False:
            return self._eager(x)
        w, b = merged
        if (x.dim() == 4 and x.size(1) == w.size(0) and x.is_cuda
                and x.dtype in _FAST_DTYPES and x.is_contiguous()):
            if x.size(2) == 20 and x.size(3) == 20:
                out = torch.empty_like(x)
                _C.dw_conv_bias_silu(x, w, b, out)
                return out
            # Same fused math, without the shape-specialised kernel.
            return F.silu(F.conv2d(x, w.to(x.dtype), b.to(x.dtype), 1,
                                   w.size(2) // 2, 1, w.size(0)))
        return self._eager(x)

    def _eager(self, x: torch.Tensor) -> torch.Tensor:
        if self._is_fused:
            return self.act(self.conv(x))
        return self.act(self.conv(x) + self.conv1(x))

    @torch.no_grad()
    def _merge(self):
        """Fold both BNs and add the zero-padded 3x3 kernel into the 7x7 one."""
        self._merged = False
        try:
            if not isinstance(self.act, SiLU):
                return False
            parts = [_fold_bn(self.conv)]
            if not self._is_fused:
                parts.append(_fold_bn(self.conv1))
            if any(p is None for p in parts):
                return False
            ks = max(p[0].size(2) for p in parts)
            w = b = None
            for pw, pb in parts:
                pad = (ks - pw.size(2)) // 2
                if pad:
                    pw = F.pad(pw, [pad, pad, pad, pad])
                w = pw if w is None else w + pw
                b = pb if b is None else b + pb
            if not w.is_cuda:
                return False
            self._merged = (w.contiguous(), b.contiguous())
        except Exception:  # noqa: BLE001 - any surprise falls back to eager
            self._merged = False
        return self._merged

    # The merged weights are derived from the parameters, so drop them whenever
    # those can have changed.
    def _invalidate(self):
        self._merged = None

    def _apply(self, *args, **kwargs):
        self._invalidate()
        return super()._apply(*args, **kwargs)

    def _load_from_state_dict(self, *args, **kwargs):
        self._invalidate()
        return super()._load_from_state_dict(*args, **kwargs)

    def train(self, mode: bool = True):
        self._invalidate()
        return super().train(mode)

    @torch.no_grad()
    def fuse(self):
        if self._is_fused:
            return self
        self.conv.fuse()
        self.conv1.fuse()
        final_conv_w = self.conv.conv.weight.data + self._pad(self.conv1.conv.weight.data, [2, 2, 2, 2])
        final_conv_b = self.conv.conv.bias.data + self.conv1.conv.bias.data
        self.conv.conv.weight.data.copy_(final_conv_w)
        self.conv.conv.bias.data.copy_(final_conv_b)
        delattr(self, "conv1")
        self._is_fused = True
        self._invalidate()
        return self
