"""YOLOv10 neck -- one fused static-shape inference path.

The eager neck issues ~75 ops (22 Conv2d + 22 BatchNorm2d + 21 SiLU + 8 cat +
4 chunk + 2 upsample + 1 add) for 6.9 GFLOP of actual work, so it is bound by
per-op dispatch, not by arithmetic.  This module collapses it into a single host
call (``neck.cu``) that issues 24 kernels:

* every BatchNorm2d is folded into the weight/bias of the conv in front of it,
  and ``YOLORepVGGDW``'s 7x7 + 3x3 depthwise pair collapses to one 7x7 -- both
  done once, the first time ``forward`` runs (which is the earliest point the
  harness's ``load_state_dict`` has delivered the real weights);
* bias, SiLU, the CIB residual add and the *placement of the result inside a
  concat buffer* are all conv epilogues, so no concat, chunk or activation
  tensor is ever materialized on its own;
* the 2 nearest-2x upsamples and the 3 concat sources that no conv produces are
  batched into 2 fill kernels;
* all weight packing/permutation and the whole intermediate workspace are
  hoisted out of forward.

Layout stays NCHW end to end, matching the harness inputs and outputs, so there
is no transpose at either end.  If the extension cannot be built the module
falls back to an eager path that still folds the BatchNorms.

With 22 launches for ~80 us of real arithmetic the path was then bound by
*per-launch* cost -- ~150 us of CPU issue and ~37 us of GPU inter-kernel gap.
So the whole sequence is recorded into a **CUDA graph** on the first call of a
shape and replayed afterwards.  The harness hands a fresh address for each of
the 3 inputs on every iteration and the 3 outputs must be freshly allocated
tensors, which a recording cannot bake in, so the 8 kernels that touch harness
memory read their base pointer out of a 6-entry device array instead; each call
allocates its outputs, stores the 6 addresses into that array (one 48-byte H2D
copy) and replays.  No copy-in of the ~5.7 MB of input is needed and every call
recomputes from its own inputs.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..L1.interpolate import Interpolate
from ..L2.yolov10_c2f import YOLOC2f, YOLOC2fCIB
from ..L2.yolov10_concat import YOLOConcat
from ..L2.yolov10_conv import YOLOConv
from ..L2.yolov10_scdown import YOLOSCDown

# --------------------------------------------------------------------------
# Extension build (once per process, at import).
# --------------------------------------------------------------------------
_EXT = None
_EXT_ERR = None
_NSLOT = 512   # pinned host ring depth (harness runs 60 timed iterations)


def _load_ext():
    global _EXT, _EXT_ERR
    if _EXT is not None or _EXT_ERR is not None:
        return _EXT
    src = Path(__file__).resolve().parent / "neck.cu"
    if not src.is_file():
        _EXT_ERR = FileNotFoundError(src)
        return None
    try:
        from torch.utils.cpp_extension import load
        major, minor = torch.cuda.get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
        _EXT = load(
            name=f"yolov10_neck_graph_sm{major}{minor}",
            sources=[str(src)],
            extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
            verbose=False,
        )
    except Exception as exc:  # noqa: BLE001 - fall back to the eager path
        _EXT_ERR = exc
        import warnings
        warnings.warn(f"yolov10_neck: CUDA extension build failed ({exc!r}); "
                      "falling back to the eager folded path")
    return _EXT


# --------------------------------------------------------------------------
# Inference-only conv/BN folding (same algebra as YOLOConv._fuse_conv_bn,
# which the eager path never invokes).
# --------------------------------------------------------------------------
def _fold_conv_bn(conv, bn):
    """Return (weight, bias) of the single conv equivalent to conv -> bn."""
    w = conv.weight.detach().float()
    b = (conv.bias.detach().float() if conv.bias is not None
         else torch.zeros(w.shape[0], device=w.device, dtype=torch.float32))
    if bn is not None:
        s = bn.weight.detach().float() / torch.sqrt(
            bn.running_var.detach().float() + bn.eps)
        w = w * s.view(-1, 1, 1, 1)
        b = (b - bn.running_mean.detach().float()) * s + bn.bias.detach().float()
    return w, b


def _fold_yconv(m):
    return _fold_conv_bn(m.conv, getattr(m, "bn", None))


def _fold_repvggdw(rep):
    """7x7 + 3x3 depthwise -> one 7x7 (both branches are act=False)."""
    w7, b7 = _fold_yconv(rep.conv)
    w3, b3 = _fold_yconv(rep.conv1)
    return w7 + F.pad(w3, [2, 2, 2, 2]), b7 + b3


class _EagerConv:
    """Folded conv, eager fallback path."""

    __slots__ = ("w", "b", "stride", "padding", "groups", "act")

    def __init__(self, w, b, stride, padding, groups, act, dtype):
        self.w = w.to(dtype).contiguous()
        self.b = b.to(dtype).contiguous()
        self.stride, self.padding, self.groups, self.act = stride, padding, groups, act

    def __call__(self, x):
        y = F.conv2d(x, self.w, self.b, self.stride, self.padding, (1, 1), self.groups)
        return F.silu(y) if self.act else y


class YOLOv10Neck(nn.Module):
    def __init__(self):
        super().__init__()
        self._upsample = Interpolate()
        self.cat1 = YOLOConcat(1)
        self.c2f_p4 = YOLOC2f(384, 128, n=1, shortcut=False)
        self.cat2 = YOLOConcat(1)
        self.c2f_p3 = YOLOC2f(192, 64, n=1, shortcut=False)
        self.down_p3 = YOLOConv(64, 64, 3, 2)
        self.cat3 = YOLOConcat(1)
        self.c2f_n4 = YOLOC2f(192, 128, n=1, shortcut=False)
        self.down_n4 = YOLOSCDown(128, 128, 3, 2)
        self.cat4 = YOLOConcat(1)
        self.c2fcib_n5 = YOLOC2fCIB(384, 256, n=1, shortcut=True, lk=True)
        self._reset()
        try:  # a fresh load_state_dict invalidates the folded weights
            self.register_load_state_dict_post_hook(
                lambda mod, incompatible_keys: mod._reset())
        except AttributeError:
            pass

    def _reset(self):
        self._packed = None   # fused path: one fp16 tensor of all weights+biases
        self._ws = None       # fused path: intermediate workspace
        self._ws_bn = -1
        self._io = None       # device: the 6 harness-facing addresses
        self._io_h = None     # pinned host ring `publish` copies from
        self._graph = None    # recorded 22-kernel path for self._ws_bn
        self._no_graph = False  # capture or its verification failed -> direct
        self._eager = None    # fallback path
        self._budget = -1     # debug: cap the number of kernels launched

    @torch.no_grad()
    def _alloc(self, ext, bn, dtype, device):
        """Per-batch workspace + the pointer plumbing a replay needs.

        The pinned host side is a ring: the harness enqueues all 60 timed
        iterations without synchronising, so the CPU runs far ahead of the GPU
        and the slot a still-queued copy has to read must not be overwritten.
        One lap is much longer than the driver's pending-launch depth.
        """
        nio = ext.io_slots()
        self._ws = torch.empty(ext.ws_elems(bn), dtype=dtype, device=device)
        self._io = torch.zeros(nio, dtype=torch.int64, device=device)
        self._io_h = torch.zeros(nio * _NSLOT, dtype=torch.int64,
                                 pin_memory=True)
        self._graph = None    # the workspace address is baked into a recording
        self._ws_bn = bn

    @torch.no_grad()
    def _capture(self, ext, bn, p3b, p4b, p5b, ref):
        """Record the 22-kernel path, then prove a replay reproduces ``ref``.

        Capture is deliberately *after* this call's real work, so the call that
        triggers it still returns its own correct outputs.  The replay is then
        checked bit-for-bit against them before it is ever used, and any
        failure pins the module to the direct path.
        """
        graph = None
        try:
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):  # capture requires a warmed-up sequence
                    ext.capture_body(self._packed, self._ws, self._io, bn, -1)
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                ext.capture_body(self._packed, self._ws, self._io, bn, -1)
            outs = ext.publish(p3b, p4b, p5b, self._io, self._io_h)
            graph.replay()
            torch.cuda.synchronize()
            ok = all(torch.equal(a, b) for a, b in zip(outs, ref))
        except Exception:  # noqa: BLE001 - any capture problem -> direct path
            ok = False
        if ok:
            self._graph = graph
        else:
            self._no_graph = True

    # ---------------------------------------------------------------- setup
    def _folded(self):
        """The 22 folded convs, in the order neck.cu's ConvId enum declares."""
        cib = self.c2fcib_n5.m[0].cv1
        yc = [self.c2f_p4.cv1, self.c2f_p4.m[0].cv1, self.c2f_p4.m[0].cv2,
              self.c2f_p4.cv2,
              self.c2f_p3.cv1, self.c2f_p3.m[0].cv1, self.c2f_p3.m[0].cv2,
              self.c2f_p3.cv2,
              self.down_p3,
              self.c2f_n4.cv1, self.c2f_n4.m[0].cv1, self.c2f_n4.m[0].cv2,
              self.c2f_n4.cv2,
              self.down_n4.cv1, self.down_n4.cv2,
              self.c2fcib_n5.cv1,
              cib[0], cib[1], None, cib[3], cib[4],
              self.c2fcib_n5.cv2]
        out = []
        for i, m in enumerate(yc):
            if m is None:  # the RepVGGDW slot
                w, b = _fold_repvggdw(cib[2])
                out.append((w, b, (1, 1), (3, 3), 256, True))
            else:
                w, b = _fold_yconv(m)
                out.append((w, b, m.conv.stride, m.conv.padding, m.conv.groups,
                            not isinstance(m.act, nn.Identity)))
        return out

    @torch.no_grad()
    def _build_fused(self, ext, dtype, device):
        convs = self._folded()
        L = ext.weight_layout()
        packed = torch.zeros(L[-1], dtype=dtype, device=device)
        for i, (w, b, _s, _p, groups, _a) in enumerate(convs):
            woff, wn, boff, bn_ = L[4 * i], L[4 * i + 1], L[4 * i + 2], L[4 * i + 3]
            if groups == 1:
                # dense: (co, ci, kh, kw) -> (co, kh, kw, ci) so each (kh, kw)
                # slice is a contiguous Cin run of the implicit-GEMM K axis
                wp = w.permute(0, 2, 3, 1).contiguous()
            else:
                wp = w.reshape(w.shape[0], -1)
            assert wp.numel() == wn, (i, wp.numel(), wn)
            assert b.numel() == bn_, (i, b.numel(), bn_)
            packed[woff:woff + wn] = wp.reshape(-1).to(dtype)
            packed[boff:boff + bn_] = b.to(dtype)
        self._packed = packed

    @torch.no_grad()
    def _build_eager(self, dtype):
        self._eager = tuple(_EagerConv(w, b, s, p, g, a, dtype)
                            for w, b, s, p, g, a in self._folded())

    # -------------------------------------------------------------- forward
    def forward(self, feats: dict[str, torch.Tensor]):
        p3_backbone = feats["p3_backbone"]
        p4_backbone = feats["p4_backbone"]
        p5_backbone = feats["p5_backbone"]

        # The fused path is compiled for this neck's captured static geometry;
        # anything else (dtype, device, spatial size, layout) takes the eager
        # folded path rather than silently reading the wrong strides.
        ext = _load_ext()
        if ext is not None and self._fusable(p3_backbone, p4_backbone, p5_backbone):
            if self._packed is None:
                self._build_fused(ext, p3_backbone.dtype, p3_backbone.device)
            bn = p3_backbone.shape[0]
            if self._ws_bn != bn:
                self._alloc(ext, bn, p3_backbone.dtype, p3_backbone.device)
            if self._graph is not None:
                # 3 allocations + 6 host stores + a 48-byte H2D + one replay.
                outs = ext.publish(p3_backbone, p4_backbone, p5_backbone,
                                   self._io, self._io_h)
                self._graph.replay()
                return outs
            outs = ext.forward(p3_backbone, p4_backbone, p5_backbone,
                               self._packed, self._ws, self._io, self._io_h,
                               self._budget)
            if not self._no_graph and self._budget < 0:
                self._capture(ext, bn, p3_backbone, p4_backbone, p5_backbone,
                              outs)
            return outs
        return self._forward_eager(p3_backbone, p4_backbone, p5_backbone)

    @staticmethod
    def _fusable(p3, p4, p5):
        b = p3.shape[0]
        return (p3.dtype == torch.float16 and p3.is_cuda
                and p4.dtype == torch.float16 and p5.dtype == torch.float16
                and tuple(p3.shape) == (b, 64, 80, 80)
                and tuple(p4.shape) == (b, 128, 40, 40)
                and tuple(p5.shape) == (b, 256, 20, 20)
                and p3.is_contiguous() and p4.is_contiguous()
                and p5.is_contiguous())

    # ---------------------------------------------------- eager fallback
    def _forward_eager(self, p3_backbone, p4_backbone, p5_backbone):
        if self._eager is None:
            self._build_eager(p3_backbone.dtype)
        (a_cv1, a_b1, a_b2, a_cv2,
         b_cv1, b_b1, b_b2, b_cv2,
         dp3,
         d_cv1, d_b1, d_b2, d_cv2,
         s_cv1, s_cv2,
         e_cv1, i0, i1, i2, i3, i4, e_cv2) = self._eager

        x = torch.cat([F.interpolate(p5_backbone, scale_factor=2.0, mode="nearest"),
                       p4_backbone], 1)
        y0, y1 = a_cv1(x).chunk(2, 1)
        p4 = a_cv2(torch.cat([y0, y1, a_b2(a_b1(y1))], 1))

        x = torch.cat([F.interpolate(p4, scale_factor=2.0, mode="nearest"),
                       p3_backbone], 1)
        y0, y1 = b_cv1(x).chunk(2, 1)
        p3 = b_cv2(torch.cat([y0, y1, b_b2(b_b1(y1))], 1))

        x = torch.cat([dp3(p3), p4], 1)
        y0, y1 = d_cv1(x).chunk(2, 1)
        n4 = d_cv2(torch.cat([y0, y1, d_b2(d_b1(y1))], 1))

        x = torch.cat([s_cv2(s_cv1(n4)), p5_backbone], 1)
        y0, y1 = e_cv1(x).chunk(2, 1)
        n5 = e_cv2(torch.cat([y0, y1, y1 + i4(i3(i2(i1(i0(y1)))))], 1))

        return [p3, n4, n5]
