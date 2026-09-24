"""YOLOv10 Spatial Pyramid Pooling - Fast.

One fused CUDA implementation of the whole block, reached through a single
pybind call.  See ``sppf_cuda.cu`` for the kernel design; the short version:

* This op is pure launch latency.  The working set is ~1 MB (fp16
  [4,256,20,20]) and the arithmetic is 0.52 GFLOP -- about 1 us of mma.sync on
  a B200 -- yet the baseline spends ~9 ATen launches on it (cv1 conv + BN +
  SiLU, three serially dependent max-pools, ``torch.cat``, cv2 conv + BN +
  SiLU).  Measured on this harness an ATen op costs 4-20 us of Python/dispatch
  and a kernel launch ~2 us of the timed window, so the baseline's ~91 us is
  almost entirely overhead.  Everything therefore collapses into three
  PDL-chained launches issued from one pybind call.
* The max-pool cascade is not serially dependent: maxpool(5,s1,p2) composed
  with itself is exactly maxpool(9,s1,p4) and three-deep exactly
  maxpool(13,s1,p6), so all three levels come out of a single on-chip pass over
  cv1's output.
* ``torch.cat`` never happens.  The four levels are just where the k index
  lands in cv2's reduction, so the concatenated tensor is only ever the k axis
  of an implicit GEMM.
* BatchNorm is folded into (weight, bias) on the host and the bias + SiLU are
  folded into the GEMM epilogues, so cv1/cv2 are one kernel each rather than
  three.

Weight preparation (BN folding + swizzling the weights into mma fragment
order) happens on the first forward, not in ``__init__``, because the harness
overwrites the weights with ``load_state_dict`` after construction.  Anything
that is not fp16 NCHW [N,256,20,20] with c1=c2=256, k=5 falls back to the
reference composition.
"""

from __future__ import annotations

import hashlib
import os

import torch
import torch.nn as nn

from ..L1.max_pool2d import MaxPool2d
from .yolov10_conv import YOLOConv

# ---------------------------------------------------------------------------
# CUDA extension (built once at import, named after the source hash so a stale
# cached .so can never shadow an edited kernel)
# ---------------------------------------------------------------------------
_EXT = None


# Pool rows-per-warp for the fused kernel.  Set here rather than as the C++
# default because changing the .cu would invalidate the cached build for a tuning
# choice; R=10 measured 0.08us faster than R=5 at both captured shapes, across two
# interleaved A/B runs.  Everything in sppf_cuda.cu reads its knobs through
# getenv on the first forward, so an outer setting still wins.
os.environ.setdefault("SPPF_FUSE_R", "10")


def _load_ext():
    src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sppf_cuda.cu")
    with open(src_path) as fh:
        src = fh.read()
    from torch.utils.cpp_extension import load_inline

    prev = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"] = "10.0"  # B200; never query a device
    try:
        return load_inline(
            name="fk_sppf_" + hashlib.md5(src.encode()).hexdigest()[:12],
            cpp_sources=(
                "int64_t sppf_make_plan(at::Tensor, at::Tensor, at::Tensor, at::Tensor,"
                " at::Tensor, int64_t, int64_t);\n"
                "void sppf_free_plan(int64_t);\n"
                "at::Tensor sppf_run(int64_t, at::Tensor);\n"
                "void sppf_bench_phase(int64_t, at::Tensor, at::Tensor, int64_t,"
                " int64_t);\n"
                "void sppf_set_cfg(int64_t, int64_t, int64_t, int64_t, int64_t,"
                " int64_t, int64_t);\n"
            ),
            cuda_sources=src,
            functions=["sppf_make_plan", "sppf_free_plan", "sppf_run",
                       "sppf_bench_phase", "sppf_set_cfg"],
            extra_cuda_cflags=["-O3", "--use_fast_math", "-std=c++17"],
            verbose=False,
        )
    finally:
        if prev is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = prev


try:
    _EXT = _load_ext()
except Exception:  # pragma: no cover - no nvcc, odd arch, read-only cache, ...
    _EXT = None


# ---------------------------------------------------------------------------
# host-side weight preparation
# ---------------------------------------------------------------------------
def _fused_weight_bias(conv_block: YOLOConv):
    """(weight, bias) of ``conv_block`` with BatchNorm folded in, fp32."""
    w = conv_block.conv.weight
    cb = conv_block.conv.bias
    if getattr(conv_block, "_is_fused", False):
        bias = (torch.zeros(w.shape[0], device=w.device, dtype=torch.float32)
                if cb is None else cb.float())
        return w.float(), bias
    bn = conv_block.bn
    scale = bn.weight.float() / torch.sqrt(bn.running_var.float() + bn.eps)
    bias = bn.bias.float() - bn.running_mean.float() * scale
    if cb is not None:
        bias = bias + cb.float() * scale
    return w.float() * scale.view(-1, 1, 1, 1), bias


def _swizzle_a(w: torch.Tensor) -> torch.Tensor:
    """[Cout, K] fp16 -> mma.m16n8k16 A-fragment order, [Cout/16, K/16, 32, 4] int32.

    Lane ``l`` of the warp (gid = l/4, tig = l%4) holds, in its four .b32
    registers, A[gid][2tig..2tig+1], A[gid+8][2tig..2tig+1],
    A[gid][2tig+8..2tig+9], A[gid+8][2tig+8..2tig+9] -- so the whole operand is
    one 16-byte load per lane per k-step.
    """
    co, k = w.shape
    mt, ks = co // 16, k // 16
    t = w.half().contiguous().view(mt, 16, ks, 16).permute(0, 2, 1, 3)
    lane = torch.arange(32, device=w.device)
    gid, tig = lane // 4, lane % 4
    rows = torch.stack([gid, gid, gid + 8, gid + 8, gid, gid, gid + 8, gid + 8], 1)
    cols = torch.stack(
        [2 * tig, 2 * tig + 1, 2 * tig, 2 * tig + 1,
         2 * tig + 8, 2 * tig + 9, 2 * tig + 8, 2 * tig + 9], 1)
    frag = t[:, :, rows, cols].contiguous()          # [mt][ks][32][8] fp16
    return frag.view(torch.int32).contiguous()       # [mt][ks][32][4]


class YOLOSPPF(nn.Module):
    def __init__(self, c1: int, c2: int, k: int = 5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = YOLOConv(c1, c_, 1, 1)
        self.cv2 = YOLOConv(c_ * 4, c2, 1, 1)
        self.m = MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self._c1, self._c2, self._k = c1, c2, k
        self._plan = 0
        self._shape = None
        self._keep = None

    # -- reference composition (fallback, and what the fast path reproduces) --
    def _reference(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape == self._shape and x.dtype == torch.float16 and x.is_contiguous():
            return _EXT.sppf_run(self._plan, x)
        return self._cold(x)

    # -- first call for a given shape: build the plan, else fall back ---------
    def _cold(self, x: torch.Tensor) -> torch.Tensor:
        if self._shape is not None or not self._eligible(x):
            return self._reference(x)
        try:
            self._build(x)
        except Exception:
            self._shape = None
            return self._reference(x)
        return _EXT.sppf_run(self._plan, x)

    def _eligible(self, x: torch.Tensor) -> bool:
        if _EXT is None or not x.is_cuda or x.dtype != torch.float16:
            return False
        if x.dim() != 4 or not x.is_contiguous():
            return False
        n, c, h, w = x.shape
        if not (c == 256 and h == 20 and w == 20):
            return False
        if not (self._c1 == 256 and self._c2 == 256 and self._k == 5):
            return False
        for blk in (self.cv1, self.cv2):
            if type(blk.act).__name__ != "SiLU":
                return False
            cv = blk.conv
            if tuple(cv.stride) != (1, 1) or tuple(cv.padding) != (0, 0):
                return False
            if tuple(cv.dilation) != (1, 1) or cv.groups != 1:
                return False
            if tuple(cv.weight.shape[2:]) != (1, 1):
                return False
        return n > 0

    def _build(self, x: torch.Tensor) -> None:
        w1, b1 = _fused_weight_bias(self.cv1)
        w2, b2 = _fused_weight_bias(self.cv2)
        aw1 = _swizzle_a(w1.reshape(128, 256))
        aw2 = _swizzle_a(w2.reshape(256, 512))
        b1 = b1.contiguous()
        b2 = b2.contiguous()
        n = int(x.shape[0])
        cat = torch.empty((n, 512, 20, 20), device=x.device, dtype=torch.float16)
        plan = _EXT.sppf_make_plan(aw1, b1, aw2, b2, cat, n, 256)
        self._keep = (aw1, b1, aw2, b2, cat)
        self._plan = plan
        self._shape = (n, 256, 20, 20)

    def __del__(self):
        try:
            if self._plan:
                _EXT.sppf_free_plan(self._plan)
        except Exception:
            pass
