"""YOLOv10 RepVGG depthwise block -- one fused launch.

The block is ``silu(bn7(dw7x7(x)) + bn3(dw3x3(x)))``.  Both branches are
depthwise over the same input, so the 3x3 taps sit inside the 7x7 support: each
BatchNorm folds into its conv weight, the zero-padded 3x3 kernel adds into the
7x7 kernel, and the two biases add.  What is left is a single depthwise 7x7
convolution + bias + SiLU, which ``repvggdw.cu`` computes in one launch that
reads ``x`` exactly once.  The folded taps are derived on the first forward and
cached; the cache is dropped on ``fuse`` / ``_apply`` / ``load_state_dict``.

Two tap layouts are prepared, because the kernel has two forms.  The fast one
gives each CTA *two adjacent channels* and packs them into ``__half2`` lanes, so
one shared load and one HFMA2 serve both planes -- half the instructions per
output pixel.  It needs the taps as channel-pair-packed fp16 (also halving the
53 KB that misses L2 on every call), and it is only used when the taps are small
enough for fp16 accumulation to be numerically free (see ``_PAIR_MAX_L1``);
otherwise the fp32 kernel runs off the fp32 ``[C, 52]`` layout.

Measured on a B200 the benched window is ~7.1 us of harness floor (the shifting
input-pool copy plus the event pair, with *zero* GPU work) plus the kernel's own
duration, and it lands on a ~2.048 us grid: 7.14, 9.22, 11.26, 13.31 us.  An
empty kernel on this grid already costs ~1.0 us, which puts every possible
single-launch solution at 11.26 us or above -- and that is where both captured
shapes now sit.  See ITERATIONS.md.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..L1.silu import SiLU
from ..L1.tensor_ops import Pad
from .yolov10_conv import YOLOConv

# --------------------------------------------------------------------------
# Extension: compiled once at import (cached in ~/.cache/torch_extensions).
# --------------------------------------------------------------------------
_EXT = None


def _load_ext():
    global _EXT
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "repvggdw.cu")
    if not os.path.exists(src):
        return
    try:
        from torch.utils.cpp_extension import load

        _EXT = load(
            name="fk_yolov10_repvggdw_v2",
            sources=[src],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False,
        )
    except Exception:  # pragma: no cover -- fall back to the torch path
        _EXT = None


_load_ext()
_TAP_STRIDE = _EXT.tap_stride() if _EXT is not None else 0
_TAP_STRIDE_P = _EXT.pair_tap_stride() if _EXT is not None else 0

# The pair kernel accumulates in fp16 -- that is the point: one HFMA2 advances
# two channels at once.  How much that costs depends only on the size of the
# partial sums, i.e. on the largest per-channel tap L1 norm.  Measured against
# the fp32-accumulating kernel on the same weights (dev/tol.py):
#
#   tap L1 max   1.1     3.0     5.6    10.7    21.4    43.0    86.0   172.0
#   fp16 match  1.0     1.0     1.0     1.0     1.0   0.99997 0.99906 0.99521
#   fp32 match  1.0     1.0     1.0     1.0     1.0    1.0     1.0    0.99999
#
# so fp16 accumulation is indistinguishable from fp32 up to L1 ~= 21 and only
# then starts to lose match ratio.  Default-initialized weights put this at
# ~1.1, so the bound below leaves ~15x of headroom; past it the fp32 kernel
# takes over (~30% slower, still correct), which also means uninitialized
# garbage weights can never drive a partial sum into an fp16 inf.
_PAIR_MAX_L1 = 16.0


def _branch_wb(m: YOLOConv):
    """Folded (weight, bias) of one Conv[-BN] branch, in fp32."""
    w = m.conv.weight.detach().float()
    cb = m.conv.bias
    b = (
        torch.zeros(w.shape[0], device=w.device, dtype=w.dtype)
        if cb is None
        else cb.detach().float()
    )
    bn = getattr(m, "bn", None)
    if bn is not None:
        if bn.running_var is None or bn.running_mean is None:
            raise RuntimeError("batchnorm without running stats")
        s = torch.rsqrt(bn.running_var.detach().float() + bn.eps)
        if bn.weight is not None:
            s = s * bn.weight.detach().float()
        b = (b - bn.running_mean.detach().float()) * s
        if bn.bias is not None:
            b = b + bn.bias.detach().float()
        w = w * s.reshape(-1, 1, 1, 1)
    return w, b


class YOLORepVGGDW(nn.Module):
    def __init__(self, ed: int):
        super().__init__()
        self.conv = YOLOConv(ed, ed, 7, 1, 3, g=ed, act=False)
        self.conv1 = YOLOConv(ed, ed, 3, 1, 1, g=ed, act=False)
        self.act = SiLU()
        self._pad = Pad()
        self._is_fused = False
        # Plain attributes (not buffers): the folded-tap cache stays out of
        # state_dict, and instance-dict reads keep ``forward`` cheap.
        self._fkw = None     # [C, TAP_STRIDE] fp32 taps (row-padded), or [C,k*k]
        self._fkb = None     # [C] fp32 bias
        self._fkw2 = None    # [C/2, TAP_STRIDE_P, 2] fp16 taps, channel-pair packed
        self._fkb2 = None    # [C/2, 2] fp16 bias, channel-pair packed
        self._fkk = 0        # kernel size
        self._fkfwd = None   # bound extension entry point when usable
        self._fktried = False

    # ---- folded-tap cache ----------------------------------------------
    def _fk_reset(self):
        self._fkw = None
        self._fkb = None
        self._fkw2 = None
        self._fkb2 = None
        self._fkfwd = None
        self._fktried = False

    def _fk_prepare(self):
        self._fktried = True
        branches = [self.conv]
        if not self._is_fused and hasattr(self, "conv1"):
            branches.append(self.conv1)
        for m in branches:
            if not isinstance(m.act, nn.Identity):
                raise RuntimeError("branch activation is not Identity")
            if m.conv.stride != (1, 1) or m.conv.dilation != (1, 1):
                raise RuntimeError("unsupported conv geometry")
        w, b = _branch_wb(branches[0])
        c, k = w.shape[0], w.shape[2]
        if w.shape[1] != 1 or w.shape[2] != w.shape[3] or self.conv.conv.groups != c:
            raise RuntimeError("not a depthwise conv")
        wf = w.reshape(c, k, k)
        for m in branches[1:]:
            w1, b1 = _branch_wb(m)
            k1 = w1.shape[2]
            if w1.shape[1] != 1 or k1 > k or (k - k1) % 2:
                raise RuntimeError("branch kernels do not nest")
            p = (k - k1) // 2
            wf = wf + F.pad(w1.reshape(c, k1, k1), [p, p, p, p])
            b = b + b1
        self._fkk = k
        self._fkb = b.contiguous()
        taps = wf.reshape(c, k * k)
        if _EXT is not None and k == 7 and _TAP_STRIDE >= 49:
            # Pad each channel's taps to a 16B-aligned row so the kernel can
            # pull them in as 128-bit loads.
            padded = taps.new_zeros(c, _TAP_STRIDE)
            padded[:, :49] = taps
            self._fkw = padded
            # Channel-pair-packed fp16 copy for the half2 kernel: element
            # [q, k, i] is channel (2q+i)'s tap k, so a 32-bit load is the
            # __half2 (tap of channel 2q, tap of channel 2q+1).
            if c % 2 == 0 and float(taps.abs().sum(1).max()) <= _PAIR_MAX_L1:
                pw = taps.new_zeros(c, _TAP_STRIDE_P)
                pw[:, :49] = taps
                self._fkw2 = (
                    pw.reshape(c // 2, 2, _TAP_STRIDE_P)
                    .permute(0, 2, 1)
                    .contiguous()
                    .half()
                )
                self._fkb2 = b.reshape(c // 2, 2).contiguous().half()
            else:
                self._fkw2 = taps.new_empty(0, dtype=torch.float16)
                self._fkb2 = self._fkw2
            self._fkfwd = _EXT.forward
        else:
            self._fkw = taps.contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self._fkfwd
        if f is not None:
            try:
                return f(x, self._fkw, self._fkb, self._fkw2, self._fkb2)
            except RuntimeError:  # shape/dtype outside the fast path
                return self._forward_torch(x)
        if not self._fktried:
            try:
                self._fk_prepare()
            except Exception:
                self._fk_reset()
                self._fktried = True
            if self._fkfwd is not None:
                return self.forward(x)
        return self._forward_torch(x)

    def _forward_torch(self, x: torch.Tensor) -> torch.Tensor:
        """Folded single-conv path (any shape/dtype), else the literal block."""
        w = self._fkw
        if w is not None:
            c, k = w.shape[0], self._fkk
            return F.silu(
                F.conv2d(
                    x,
                    w[:, : k * k].reshape(c, 1, k, k).to(x.dtype),
                    self._fkb.to(x.dtype),
                    padding=k // 2,
                    groups=c,
                )
            )
        if self._is_fused:
            return self.act(self.conv(x))
        return self.act(self.conv(x) + self.conv1(x))

    # ---- cache invalidation --------------------------------------------
    def _apply(self, *args, **kwargs):
        self._fk_reset()
        return super()._apply(*args, **kwargs)

    def _load_from_state_dict(self, *args, **kwargs):
        self._fk_reset()
        return super()._load_from_state_dict(*args, **kwargs)

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
        self._fk_reset()
        return self
