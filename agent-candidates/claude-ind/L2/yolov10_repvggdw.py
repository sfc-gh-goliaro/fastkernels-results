"""YOLOv10 RepVGG depthwise block.

The captured workload runs this block in *inference* (eval) mode without
``fuse()`` having been called, so the eager path is

    silu( bn7(dwconv7x7(x)) + bn3(dwconv3x3(x)) )

which torch executes as six separate kernels.  Both branches are depthwise
convolutions of the same input at the same stride, so the two conv+BN pairs
collapse analytically into a *single* depthwise 7x7 convolution plus a bias --
the same algebra ``fuse()`` performs.  We fold them once (cached on first
inference call), then evaluate conv + bias + SiLU in one custom CUDA kernel.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn

from ..L1.silu import SiLU
from ..L1.tensor_ops import Pad
from .yolov10_conv import YOLOConv

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>

#define KS 7
#define KR 3

// One block per (n, channel-pair): the 20x20 plane of two adjacent channels is
// staged in shared memory as __half2 with a zeroed 3-wide halo, so every tap
// index is a compile-time constant.  Packing the channel pair into the __half2
// lanes makes one hfma2 advance two outputs at once.  threadIdx.x/y tile the
// plane COLS x ROWS per thread; the kw-outer loop order keeps the input column
// strip in registers so each staged word and each weight is read exactly once
// per ROWS*COLS accumulators.
template <int H, int W, int ROWS, int COLS>
__global__ __launch_bounds__((W / COLS) * (H / ROWS)) void dw7(
    const __half* __restrict__ x, const __half2* __restrict__ wp,
    const __half2* __restrict__ bp, __half* __restrict__ y, int ppi) {
  constexpr int SW = W + 2 * KR;
  constexpr int SH = H + 2 * KR;
  constexpr int HW = H * W;
  constexpr int NX = W / COLS;
  constexpr int NTHR = NX * (H / ROWS);

  __shared__ __half2 sp[SH * SW];
  __shared__ __half2 sw[KS * KS];

  const int t = threadIdx.y * NX + threadIdx.x;
  const int gp = blockIdx.x;
  const int pk = gp % ppi;
  const __half* __restrict__ xp = x + (size_t)gp * (2 * HW);
  __half* __restrict__ yp = y + (size_t)gp * (2 * HW);

  const __half2 zero = __float2half2_rn(0.f);
  for (int i = t; i < KR * SW; i += NTHR) {
    sp[i] = zero;
    sp[(KR + H) * SW + i] = zero;
  }
  for (int i = t; i < H * 2 * KR; i += NTHR) {
    const int r = i / (2 * KR), k = i - r * (2 * KR);
    sp[(r + KR) * SW + (k < KR ? k : k + W)] = zero;
  }
  // Two spatially adjacent elements of both planes per iteration; the byte
  // permutes turn (c,c) / (c+1,c+1) pairs into the (c, c+1) interleave.
  for (int u = t; u < HW / 2; u += NTHR) {
    const int i = u * 2, r = i / W, c = i - r * W;
    const __half2 a = *(const __half2*)(xp + i);
    const __half2 b = *(const __half2*)(xp + HW + i);
    __half2* d = &sp[(r + KR) * SW + c + KR];
    d[0] = __lows2half2(a, b);
    d[1] = __highs2half2(a, b);
  }
  for (int i = t; i < KS * KS; i += NTHR) sw[i] = wp[pk * (KS * KS) + i];
  __syncthreads();

  const int oh0 = threadIdx.y * ROWS, ow0 = threadIdx.x * COLS;
  __half2 acc[ROWS][COLS];
#pragma unroll
  for (int r = 0; r < ROWS; ++r)
#pragma unroll
    for (int c = 0; c < COLS; ++c) acc[r][c] = zero;

  const __half2* __restrict__ sb = sp + oh0 * SW + ow0;
#pragma unroll
  for (int kw = 0; kw < KS; ++kw) {
    __half2 in[ROWS + 2 * KR][COLS];
#pragma unroll
    for (int j = 0; j < ROWS + 2 * KR; ++j)
#pragma unroll
      for (int c = 0; c < COLS; ++c) in[j][c] = sb[j * SW + c + kw];
#pragma unroll
    for (int kh = 0; kh < KS; ++kh) {
      const __half2 wv = sw[kh * KS + kw];
#pragma unroll
      for (int r = 0; r < ROWS; ++r)
#pragma unroll
        for (int c = 0; c < COLS; ++c) acc[r][c] = __hfma2(in[r + kh][c], wv, acc[r][c]);
    }
  }

  const __half2 bias = bp[pk];
#pragma unroll
  for (int r = 0; r < ROWS; ++r)
#pragma unroll
    for (int c = 0; c < COLS; ++c) {
      const __half2 v = __hadd2(bias, acc[r][c]);
      const int o = (oh0 + r) * W + ow0 + c;
      const float a0 = __low2float(v), a1 = __high2float(v);
      yp[o] = __float2half(a0 * __frcp_rn(1.f + __expf(-a0)));
      yp[HW + o] = __float2half(a1 * __frcp_rn(1.f + __expf(-a1)));
    }
}

torch::Tensor dw7_bias_silu(torch::Tensor x, torch::Tensor w, torch::Tensor b) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kHalf && x.dim() == 4);
  auto xc = x.is_contiguous() ? x : x.contiguous();
  const int64_t N = xc.size(0), C = xc.size(1), H = xc.size(2), W = xc.size(3);
  TORCH_CHECK(H == 20 && W == 20 && C % 2 == 0 && C == 2 * w.size(0));
  auto y = at::empty({N, C, H, W}, xc.options());
  const __half* xd = (const __half*)xc.data_ptr<at::Half>();
  __half* yd = (__half*)y.data_ptr<at::Half>();
  TORCH_CHECK((((uintptr_t)xd | (uintptr_t)yd) & 3) == 0, "planes must be 4B aligned");
  dw7<20, 20, 2, 1><<<N * C / 2, dim3(20, 10), 0, at::cuda::getCurrentCUDAStream()>>>(
      xd, (const __half2*)w.data_ptr<at::Half>(), (const __half2*)b.data_ptr<at::Half>(),
      yd, (int)(C / 2));
  return y;
}
"""

_CPP_SRC = "torch::Tensor dw7_bias_silu(torch::Tensor x, torch::Tensor w, torch::Tensor b);"

_EXT = None


def _ext():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load_inline

        if "TORCH_CUDA_ARCH_LIST" not in os.environ:
            major, minor = torch.cuda.get_device_capability()
            os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}{'a' if major in (9, 10, 12) else ''}"
        _EXT = load_inline(
            name="fk_yolov10_repvggdw_dw7",
            cpp_sources=_CPP_SRC,
            cuda_sources=_CUDA_SRC,
            functions=["dw7_bias_silu"],
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_HALF2_OPERATORS__",
            ],
            verbose=False,
        )
    return _EXT


def _bn_scale_shift(bn):
    inv = bn.weight.data.float() / torch.sqrt(bn.running_var.data.float() + bn.eps)
    return inv, bn.bias.data.float() - bn.running_mean.data.float() * inv


class YOLORepVGGDW(nn.Module):
    def __init__(self, ed: int):
        super().__init__()
        self.conv = YOLOConv(ed, ed, 7, 1, 3, g=ed, act=False)
        self.conv1 = YOLOConv(ed, ed, 3, 1, 1, g=ed, act=False)
        self.act = SiLU()
        self._pad = Pad()
        self._is_fused = False
        self._packed = None

    # ---- fused-weight cache ------------------------------------------------
    def _build_packed(self):
        ed = self.conv.conv.weight.shape[0]
        w = self.conv.conv.weight.data.float().reshape(ed, 7, 7)
        cb = self.conv.conv.bias
        b = cb.data.float() if cb is not None else w.new_zeros(ed)
        if not self._is_fused:
            inv, shift = _bn_scale_shift(self.conv.bn)
            w = w * inv[:, None, None]
            b = b * inv + shift
            w1 = self.conv1.conv.weight.data.float().reshape(ed, 3, 3)
            cb1 = self.conv1.conv.bias
            b1 = cb1.data.float() if cb1 is not None else w.new_zeros(ed)
            inv1, shift1 = _bn_scale_shift(self.conv1.bn)
            w[:, 2:5, 2:5] += w1 * inv1[:, None, None]
            b = b + b1 * inv1 + shift1
        # Interleave adjacent channels so each __half2 lane holds (c, c+1).
        wp = w.reshape(ed // 2, 2, 49).permute(0, 2, 1).contiguous().half()
        bp = b.reshape(ed // 2, 2).contiguous().half()
        self._packed = (wp, bp)
        return self._packed

    def load_state_dict(self, *args, **kwargs):
        self._packed = None
        return super().load_state_dict(*args, **kwargs)

    def _apply(self, *args, **kwargs):
        self._packed = None
        return super()._apply(*args, **kwargs)

    def train(self, mode: bool = True):
        # A training pass moves the BN running stats the fold depends on.
        self._packed = None
        return super().train(mode)

    def _eager(self, x: torch.Tensor) -> torch.Tensor:
        if self._is_fused:
            return self.act(self.conv(x))
        return self.act(self.conv(x) + self.conv1(x))

    def _slow(self, x: torch.Tensor) -> torch.Tensor:
        """First inference call (or anything the kernel does not cover)."""
        if self.training or self._packed is not None or not (
            x.is_cuda
            and x.dtype is torch.float16
            and x.dim() == 4
            and x.shape[1] % 2 == 0
            and x.shape[2] == 20
            and x.shape[3] == 20
        ):
            return self._eager(x)
        try:
            _ext()
        except Exception:
            return self._eager(x)
        p = self._build_packed()
        return _EXT.dw7_bias_silu(x, p[0], p[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self._packed
        if (
            p is not None
            and not self.training
            and x.dtype is torch.float16
            and x.shape[2] == 20
            and x.shape[3] == 20
        ):
            return _EXT.dw7_bias_silu(x, p[0], p[1])
        return self._slow(x)

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
        self._packed = None
        return self
