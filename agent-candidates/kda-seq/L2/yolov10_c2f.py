"""YOLOv10 C2f / C2fCIB for B200 (sm_100): the whole block from one Python call.

This operator is dispatch-bound, not arithmetic-bound, and that decides the
design. Measured on a leased B200 with harness-identical timing
(``scratch/micro1.py``), the CPU-side enqueue cost of one baseline forward is
0.204-0.404 ms against 0.176-0.358 ms of reported latency -- the same magnitude,
and larger on six of the seven benchmarked cases -- while the identical kernel
sequence replayed from a CUDA graph costs 0.113-0.219 ms (``scratch/micro3.py``).
The GPU idles through most of the measured window. Six of seven cases are
batch-1 or 20x20, a few hundred KB of activations; case 2 is the only one with
real arithmetic (~1.4 GFLOP).

The baseline issues ~14 (C2f n=1), ~21 (C2f n=2) or ~26 (C2fCIB lk=True) ATen
ops through ~40 ``nn.Module.__call__`` frames at 7-25 us of CPU each, because
every ``YOLOConv`` is three separate ops (``Conv2d(bias=False)`` ->
``BatchNorm2d(eps=1e-3)`` -> ``SiLU``) and the block adds a ``chunk``, a ``cat``
and the residual adds on top. So the lever is the number of things dispatched,
not the arithmetic:

1. ``BatchNorm2d`` in eval mode is a per-channel affine, so it folds into the
   convolution's epilogue. That deletes one op per convolution and, on its own,
   measures 1.46-2.92x (geomean 1.89) with no custom kernel at all.
2. The whole block then runs from one Python call into a CUDA extension that
   issues 4 (C2f n=1), 6 (C2f n=2) or 7 (C2fCIB lk=True) kernels, addressing
   channel slices of one output-layout buffer so ``chunk`` and ``cat`` disappear
   structurally rather than being replaced by copies.

Three routes are built and each configuration takes the one that measures best:

``fused``
    One call into the extension; the block's whole step program runs host-side
    in C++, so there is one Python frame and one tensor argument per forward.
``folded``
    The same folded constants executed with ``F.conv2d``/``F.silu``. This is the
    permanent middle route: it is always correct and always faster than the
    baseline, so no configuration ever has to fall back all the way down.
``reference``
    The baseline's own three lines over the baseline's own submodules. Every
    input or module state the folded constants cannot serve exactly lands here,
    so this module can never be more wrong than the baseline.

The fold is built lazily on the first eligible forward and never in
``__init__``: the benchmark harness constructs the module, sanitizes its
parameters and only then loads weights into it, so anything derived from weight
*values* at construction time would be derived from garbage. It is dropped again
whenever those values or their storage can move (``load_state_dict``,
``_apply``). Nothing derived is registered as a buffer or parameter, so
``state_dict()`` stays key-identical to the baseline's -- which matters because
the harness shares weights with ``strict=False``, and a candidate whose key
names drift silently keeps its own random weights and fails numerically.

``FK_YOLOV10_C2F_ROUTE`` forces a route (``fused``/``folded``/``reference``) for
A/B measurement; unset takes the per-configuration choice.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

from .yolov10_bottleneck import YOLOBottleneck
from .yolov10_cib import YOLOCIB
from .yolov10_conv import YOLOConv
from .yolov10_repvggdw import YOLORepVGGDW

ROUTE_FUSED = "fused"      # the kernels in this file, one Python call
ROUTE_ATEN = "aten"        # cuDNN's convolutions, also from one Python call
ROUTE_FOLDED = "folded"    # the same folded arithmetic, driven from Python
ROUTE_REFERENCE = "reference"
_ROUTE_ENV = "FK_YOLOV10_C2F_ROUTE"

# ``act=True`` in YOLOConv resolves to the single shared ``default_act``
# instance, so identity -- not isinstance -- is the exact test for "this stage
# ends in SiLU". Comparing types would be wrong here: the class this module
# imports and the class the constructed submodules hold can differ, because
# candidate imports resolve to the frozen L1 winner while the submodule tree is
# built from the baseline's own YOLOConv.
_SILU_ACT = YOLOConv.default_act

# Kernel families in the extension's step program.
_KIND_PW = 0      # 1x1, dense: a GEMM over the channel axis
_KIND_CONV3 = 1   # 3x3, dense: 2-D spatial tile, reduction over (c, tap)
_KIND_DW = 2      # KxK depthwise, K in {3, 7}: one channel per block row

# Element offsets are computed in int32 inside the kernels; every buffer the
# plan addresses must stay inside that range.
_INT32_MAX = 2**31 - 1

# Buffer slots a step can name. Slot 0 and up are the arena's own slots, so the
# three special values are negative -- and "no shortcut" needs its own sentinel
# rather than reusing SLOT_OUT, or a stage with no residual reads the output
# tensor's uninitialised memory.
# Iterations per route in the one-time selection measurement.
_ROUTE_TRIALS = 20

_SLOT_OUT = -1
_SLOT_INPUT = -2
_SLOT_NONE = -3


# ---------------------------------------------------------------------------
# The fused route's CUDA extension.
#
# One entry point per forward: ``plan_run(handle, x)`` walks a host-side step
# program and issues 4 (C2f n=1), 6 (C2f n=2) or 7 (C2fCIB lk=True) kernels.
# Three families cover every stage, all fp16 in and out with fp32 accumulation,
# all ending in the same epilogue, all reading and writing channel slices of a
# caller-supplied buffer so the concatenation is addressing rather than copying.
#
# Plain FFMA accumulation rather than warp-level MMA, deliberately: the largest
# case is ~1.4 GFLOP and the smallest reduces over 144 elements with 16 output
# channels, at grid sizes of tens of blocks against 148 SMs. The metric here is
# single-wave latency, not throughput, and MMA's TMEM/cluster machinery buys
# nothing at these sizes. It is the next lever if a profile says otherwise.
# ---------------------------------------------------------------------------
_EXTENSION_NAME = "fk_yolov10_c2f_sm100_v1"
_CUDA_ARCH = "10.0"

_CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include <cstdlib>
#include <vector>

// ex2.approx.f32 + div.approx.f32, the form the frozen candidate/L1/silu.py
// established against ATen: it drops expf's range reduction and div.rn's
// Newton-Raphson fixup. Verified here over all 65536 finite fp16 encodings
// (scratch/silu_sweep.py) rather than inherited from that file's bfloat16 sweep.
__device__ __forceinline__ float silu_f32(const float x) {
  return __fdividef(x, 1.0f + __expf(-x));
}

// The baseline rounds to fp16 twice on the way out of a YOLOConv: once when
// BatchNorm2d stores its result, once when SiLU stores its own. Reproducing both
// rounding points keeps this within a few fp16 ULP of the unfused baseline.
// Explicit intrinsics only -- torch's nvcc line adds -D__CUDA_NO_HALF_OPERATORS__
// and -D__CUDA_NO_HALF_CONVERSIONS__, so operator syntax does not compile.
__device__ __forceinline__ __half affine_silu(const float acc, const float s, const float b) {
  const __half h = __float2half(fmaf(s, acc, b));
  return __float2half(silu_f32(__half2float(h)));
}

struct StageArgs {
  const __half* __restrict__ in;
  __half* __restrict__ out;
  const __half* __restrict__ res;   // nullptr when the stage has no shortcut
  const __half* __restrict__ w;
  const float* __restrict__ s;
  const float* __restrict__ b;
  long long in_bs, out_bs, res_bs;  // element stride between batch items
  int C, O, H, W, HW;
};

// --- 1x1 dense: a GEMM over the channel axis --------------------------------
// out[o, p] = silu(s[o] * sum_c w[o, c] * in[c, p] + b[o]) (+ res[o, p])
// The batch rides a grid axis because the gap between batch items rules out
// folding n into the pixel axis.
template <int BO, int BP, int BK, int TO, int TP>
__global__ void pw_kernel(const StageArgs a) {
  constexpr int PT = BP / TP;
  constexpr int NT = (BO / TO) * PT;
  __shared__ float xs[BK][BP];
  __shared__ float ws[BK][BO];

  const int tid = threadIdx.x;
  const int tp = tid % PT;
  const int to = tid / PT;
  const int p0 = blockIdx.x * BP;
  const int o0 = blockIdx.y * BO;
  const __half* __restrict__ in = a.in + (long long)blockIdx.z * a.in_bs;

  float acc[TO][TP];
#pragma unroll
  for (int i = 0; i < TO; ++i)
#pragma unroll
    for (int j = 0; j < TP; ++j) acc[i][j] = 0.0f;

  for (int k0 = 0; k0 < a.C; k0 += BK) {
    for (int idx = tid; idx < BK * BP; idx += NT) {
      const int kk = idx / BP;
      const int pp = idx - kk * BP;
      const int c = k0 + kk, p = p0 + pp;
      xs[kk][pp] = (c < a.C && p < a.HW) ? __half2float(in[(long long)c * a.HW + p]) : 0.0f;
    }
    for (int idx = tid; idx < BK * BO; idx += NT) {
      const int oo = idx / BK;
      const int kk = idx - oo * BK;
      const int o = o0 + oo, c = k0 + kk;
      ws[kk][oo] = (o < a.O && c < a.C) ? __half2float(a.w[(long long)o * a.C + c]) : 0.0f;
    }
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < BK; ++kk) {
      float wv[TO], xv[TP];
#pragma unroll
      for (int i = 0; i < TO; ++i) wv[i] = ws[kk][to * TO + i];
#pragma unroll
      for (int j = 0; j < TP; ++j) xv[j] = xs[kk][tp * TP + j];
#pragma unroll
      for (int i = 0; i < TO; ++i)
#pragma unroll
        for (int j = 0; j < TP; ++j) acc[i][j] = fmaf(wv[i], xv[j], acc[i][j]);
    }
    __syncthreads();
  }

  __half* __restrict__ out = a.out + (long long)blockIdx.z * a.out_bs;
  const __half* __restrict__ res = a.res ? a.res + (long long)blockIdx.z * a.res_bs : nullptr;
#pragma unroll
  for (int i = 0; i < TO; ++i) {
    const int o = o0 + to * TO + i;
    if (o >= a.O) continue;
    const float sc = a.s[o], bi = a.b[o];
#pragma unroll
    for (int j = 0; j < TP; ++j) {
      const int p = p0 + tp * TP + j;
      if (p >= a.HW) continue;
      const long long e = (long long)o * a.HW + p;
      __half v = affine_silu(acc[i][j], sc, bi);
      if (res) v = __hadd(v, res[e]);
      out[e] = v;
    }
  }
}

// --- 1x1 dense, operands from the cache hierarchy ---------------------------
// The shared-memory tile above is bandwidth-bound at these sizes, and there is
// no way out of that by tuning it: the load-to-multiply ratio only improves with
// larger register tiles, and a larger register tile means fewer threads, while
// the smallest stage here has 51200 outputs in total -- already too few to fill
// 148 SMs. Measured on case 1's cv1 (O=256, C=384, HW=400): 36 us for the
// shared-memory tile against an 8 us shared-bandwidth bound.
//
// So this variant skips shared memory. Each thread owns TO output channels by
// four consecutive pixels, and both operands come straight from L1: the weight
// row is warp-uniform (every thread in the warp wants the same w[o][c], so it
// broadcasts) and eight of them arrive in one 16-byte load because c is
// contiguous, while the pixel vector is four halves in one 8-byte load. That is
// 64 multiply-adds per 10 load instructions at TO=2, against 16 per 8 for the
// shared-memory tile, with no barrier and no bank conflict.
//
// Requires HW % 4 == 0 (so a pixel group is aligned and never straddles the end
// of a row) and C % 8 == 0 (so the weight octet is 16-byte aligned). Every
// benched configuration satisfies both; anything else takes the tile above.
__device__ __forceinline__ void unpack4(const uint2 raw, float* v) {
  const __half2 lo = *reinterpret_cast<const __half2*>(&raw.x);
  const __half2 hi = *reinterpret_cast<const __half2*>(&raw.y);
  v[0] = __low2float(lo);
  v[1] = __high2float(lo);
  v[2] = __low2float(hi);
  v[3] = __high2float(hi);
}

__device__ __forceinline__ void unpack8(const uint4 raw, float* v) {
  unpack4(make_uint2(raw.x, raw.y), v);
  unpack4(make_uint2(raw.z, raw.w), v + 4);
}

// TP consecutive halves in one instruction where the alignment allows it.
template <int TP>
__device__ __forceinline__ void load_pixels(const __half* __restrict__ p, float* v) {
  if constexpr (TP == 4) {
    unpack4(*reinterpret_cast<const uint2*>(p), v);
  } else if constexpr (TP == 2) {
    const __half2 h = *reinterpret_cast<const __half2*>(p);
    v[0] = __low2float(h);
    v[1] = __high2float(h);
  } else {
    v[0] = __half2float(*p);
  }
}

template <int TP>
__device__ __forceinline__ void store_pixels(__half* __restrict__ dst, __half* val,
                                             const __half* __restrict__ res) {
  if (res) {
    __half r[TP];
    if constexpr (TP == 4) {
      *reinterpret_cast<uint2*>(r) = *reinterpret_cast<const uint2*>(res);
    } else if constexpr (TP == 2) {
      *reinterpret_cast<__half2*>(r) = *reinterpret_cast<const __half2*>(res);
    } else {
      r[0] = *res;
    }
#pragma unroll
    for (int j = 0; j < TP; ++j) val[j] = __hadd(val[j], r[j]);
  }
  if constexpr (TP == 4) {
    *reinterpret_cast<uint2*>(dst) = *reinterpret_cast<const uint2*>(val);
  } else if constexpr (TP == 2) {
    *reinterpret_cast<__half2*>(dst) = *reinterpret_cast<const __half2*>(val);
  } else {
    *dst = val[0];
  }
}

template <int TO, int TP, int BLK>
__global__ void pw_cached_kernel(const StageArgs a) {
  const int p0 = (blockIdx.x * BLK + threadIdx.x) * TP;
  const int o0 = blockIdx.y * TO;
  if (p0 >= a.HW) return;
  const __half* __restrict__ in = a.in + (long long)blockIdx.z * a.in_bs;

  float acc[TO][TP];
#pragma unroll
  for (int i = 0; i < TO; ++i)
#pragma unroll
    for (int j = 0; j < TP; ++j) acc[i][j] = 0.0f;

  for (int c = 0; c < a.C; c += 8) {
    float xv[8][TP];
#pragma unroll
    for (int k = 0; k < 8; ++k)
      load_pixels<TP>(in + (long long)(c + k) * a.HW + p0, xv[k]);
#pragma unroll
    for (int i = 0; i < TO; ++i) {
      if (o0 + i >= a.O) continue;
      float wv[8];
      unpack8(*reinterpret_cast<const uint4*>(a.w + (long long)(o0 + i) * a.C + c), wv);
#pragma unroll
      for (int k = 0; k < 8; ++k)
#pragma unroll
        for (int j = 0; j < TP; ++j) acc[i][j] = fmaf(wv[k], xv[k][j], acc[i][j]);
    }
  }

  __half* __restrict__ out = a.out + (long long)blockIdx.z * a.out_bs;
  const __half* __restrict__ res = a.res ? a.res + (long long)blockIdx.z * a.res_bs : nullptr;
#pragma unroll
  for (int i = 0; i < TO; ++i) {
    const int o = o0 + i;
    if (o >= a.O) continue;
    const float sc = a.s[o], bi = a.b[o];
    __half val[TP];
#pragma unroll
    for (int j = 0; j < TP; ++j) val[j] = affine_silu(acc[i][j], sc, bi);
    const long long e = (long long)o * a.HW + p0;
    store_pixels<TP>(out + e, val, res ? res + e : nullptr);
  }
}

// --- 3x3 dense --------------------------------------------------------------
// A 2-D spatial tile, so the pixel index never has to be divided by W, with the
// reduction running over (channel, tap). Each thread reloads one shared row per
// tap row and reuses it across the three taps, which is what keeps the FFMA to
// shared-load ratio near 3.
template <int BO, int TH, int TWX, int BK, int TO, int TP>
__global__ void conv3_kernel(const StageArgs a, const int N) {
  constexpr int PH = TH + 2;
  constexpr int PW = TWX + 2;
  constexpr int XT = TWX / TP;
  constexpr int NT = (BO / TO) * XT * TH;
  __shared__ float xs[BK][PH][PW];
  __shared__ float ws[BK][9][BO];

  const int tid = threadIdx.x;
  const int tx = tid % XT;
  const int ty = (tid / XT) % TH;
  const int to = tid / (XT * TH);
  const int nb = blockIdx.z % N;
  const int o0 = (blockIdx.z / N) * BO;
  const int x0 = blockIdx.x * TWX;
  const int y0 = blockIdx.y * TH;
  const __half* __restrict__ in = a.in + (long long)nb * a.in_bs;

  float acc[TO][TP];
#pragma unroll
  for (int i = 0; i < TO; ++i)
#pragma unroll
    for (int j = 0; j < TP; ++j) acc[i][j] = 0.0f;

  for (int k0 = 0; k0 < a.C; k0 += BK) {
    for (int idx = tid; idx < BK * PH * PW; idx += NT) {
      const int kk = idx / (PH * PW);
      const int r = idx - kk * (PH * PW);
      const int ry = r / PW;
      const int rx = r - ry * PW;
      const int c = k0 + kk;
      const int yy = y0 + ry - 1, xx = x0 + rx - 1;
      float v = 0.0f;
      if (c < a.C && yy >= 0 && yy < a.H && xx >= 0 && xx < a.W)
        v = __half2float(in[((long long)c * a.H + yy) * a.W + xx]);
      xs[kk][ry][rx] = v;
    }
    for (int idx = tid; idx < BK * 9 * BO; idx += NT) {
      const int oo = idx / (BK * 9);
      const int r = idx - oo * (BK * 9);
      const int kk = r / 9;
      const int t = r - kk * 9;
      const int o = o0 + oo, c = k0 + kk;
      ws[kk][t][oo] = (o < a.O && c < a.C)
                          ? __half2float(a.w[((long long)o * a.C + c) * 9 + t]) : 0.0f;
    }
    __syncthreads();
#pragma unroll 1
    for (int kk = 0; kk < BK; ++kk) {
#pragma unroll
      for (int ky = 0; ky < 3; ++ky) {
        float xr[TP + 2];
#pragma unroll
        for (int q = 0; q < TP + 2; ++q) xr[q] = xs[kk][ty + ky][tx * TP + q];
#pragma unroll
        for (int kx = 0; kx < 3; ++kx) {
          float wv[TO];
#pragma unroll
          for (int i = 0; i < TO; ++i) wv[i] = ws[kk][ky * 3 + kx][to * TO + i];
#pragma unroll
          for (int i = 0; i < TO; ++i)
#pragma unroll
            for (int j = 0; j < TP; ++j) acc[i][j] = fmaf(wv[i], xr[kx + j], acc[i][j]);
        }
      }
    }
    __syncthreads();
  }

  const int y = y0 + ty;
  if (y >= a.H) return;
  __half* __restrict__ out = a.out + (long long)nb * a.out_bs;
  const __half* __restrict__ res = a.res ? a.res + (long long)nb * a.res_bs : nullptr;
#pragma unroll
  for (int i = 0; i < TO; ++i) {
    const int o = o0 + to * TO + i;
    if (o >= a.O) continue;
    const float sc = a.s[o], bi = a.b[o];
#pragma unroll
    for (int j = 0; j < TP; ++j) {
      const int x = x0 + tx * TP + j;
      if (x >= a.W) continue;
      const long long e = ((long long)o * a.H + y) * a.W + x;
      __half v = affine_silu(acc[i][j], sc, bi);
      if (res) v = __hadd(v, res[e]);
      out[e] = v;
    }
  }
}

// --- KxK depthwise, K in {3, 7} ---------------------------------------------
// One channel per block row: there is no channel reduction, so the taps are the
// whole inner loop. Sliding the K+TP-1 shared values for a tap row into
// registers turns K*TP fused multiply-adds into K+TP-1 shared loads.
template <int K, int TH, int TWX, int TP>
__global__ void dw_kernel(const StageArgs a, const int C) {
  constexpr int R = K / 2;
  constexpr int PH = TH + K - 1;
  constexpr int PW = TWX + K - 1;
  constexpr int XT = TWX / TP;
  constexpr int NT = XT * TH;
  __shared__ float xs[PH][PW];
  __shared__ float wsh[K][K];

  const int tid = threadIdx.x;
  const int tx = tid % XT;
  const int ty = tid / XT;
  const int c = blockIdx.z % C;
  const int nb = blockIdx.z / C;
  const int x0 = blockIdx.x * TWX;
  const int y0 = blockIdx.y * TH;
  const __half* __restrict__ in = a.in + (long long)nb * a.in_bs + (long long)c * a.HW;

  for (int idx = tid; idx < PH * PW; idx += NT) {
    const int ry = idx / PW;
    const int rx = idx - ry * PW;
    const int yy = y0 + ry - R, xx = x0 + rx - R;
    xs[ry][rx] = (yy >= 0 && yy < a.H && xx >= 0 && xx < a.W)
                     ? __half2float(in[(long long)yy * a.W + xx]) : 0.0f;
  }
  for (int idx = tid; idx < K * K; idx += NT) {
    wsh[idx / K][idx % K] = __half2float(a.w[(long long)c * (K * K) + idx]);
  }
  __syncthreads();

  float acc[TP];
#pragma unroll
  for (int j = 0; j < TP; ++j) acc[j] = 0.0f;
#pragma unroll
  for (int ky = 0; ky < K; ++ky) {
    float xr[TP + K - 1];
#pragma unroll
    for (int q = 0; q < TP + K - 1; ++q) xr[q] = xs[ty + ky][tx * TP + q];
#pragma unroll
    for (int kx = 0; kx < K; ++kx) {
      const float wv = wsh[ky][kx];
#pragma unroll
      for (int j = 0; j < TP; ++j) acc[j] = fmaf(wv, xr[kx + j], acc[j]);
    }
  }

  const int y = y0 + ty;
  if (y >= a.H) return;
  const float sc = a.s[c], bi = a.b[c];
  __half* __restrict__ out = a.out + (long long)nb * a.out_bs + (long long)c * a.HW;
  const __half* __restrict__ res =
      a.res ? a.res + (long long)nb * a.res_bs + (long long)c * a.HW : nullptr;
#pragma unroll
  for (int j = 0; j < TP; ++j) {
    const int x = x0 + tx * TP + j;
    if (x >= a.W) continue;
    const long long e = (long long)y * a.W + x;
    __half v = affine_silu(acc[j], sc, bi);
    if (res) v = __hadd(v, res[e]);
    out[e] = v;
  }
}

// --- the step program -------------------------------------------------------
// Slot -2 is the caller's x, -1 is the block's own output tensor, 0 is the
// concatenation-layout buffer Y, and 1.. are per-stage temporaries. Y and the
// temporaries share one allocation, so a forward performs exactly two.
namespace {

constexpr int kFields = 10;
constexpr int kKindPw = 0, kKindConv3 = 1, kKindDw = 2;
constexpr int kSlotOut = -1, kSlotInput = -2, kSlotNone = -3;

struct Step {
  int kind, cin, cout, ksize;
  int in_slot, in_off, out_slot, out_off, res_slot, res_off;
  const __half* w;
  const float* s;
  const float* b;
};

class BlockPlan {
 public:
  BlockPlan(std::vector<at::Tensor> ws, std::vector<at::Tensor> ss, std::vector<at::Tensor> bs,
            std::vector<at::Tensor> fws, std::vector<at::Tensor> fbs,
            const std::vector<int64_t>& ints, std::vector<int64_t> slot_widths,
            std::vector<int64_t> block_lens, std::vector<int64_t> block_add,
            int64_t in_channels, int64_t out_channels, int64_t split)
      : ws_(std::move(ws)), ss_(std::move(ss)), bs_(std::move(bs)),
        fws_(std::move(fws)), fbs_(std::move(fbs)),
        slot_widths_(std::move(slot_widths)), block_lens_(std::move(block_lens)),
        block_add_(std::move(block_add)),
        in_channels_(in_channels), out_channels_(out_channels), split_(split) {
    TORCH_CHECK(ints.size() % kFields == 0, "step program is not a multiple of the field count");
    const size_t nsteps = ints.size() / kFields;
    TORCH_CHECK(nsteps > 0 && nsteps == ws_.size() && nsteps == ss_.size() && nsteps == bs_.size(),
                "step program and weight lists disagree on the step count");
    TORCH_CHECK(!slot_widths_.empty(), "the step program needs at least the Y slot");
    steps_.resize(nsteps);
    for (size_t i = 0; i < nsteps; ++i) {
      const int64_t* f = ints.data() + i * kFields;
      Step& st = steps_[i];
      st.kind = static_cast<int>(f[0]);
      st.cin = static_cast<int>(f[1]);
      st.cout = static_cast<int>(f[2]);
      st.ksize = static_cast<int>(f[3]);
      st.in_slot = static_cast<int>(f[4]);
      st.in_off = static_cast<int>(f[5]);
      st.out_slot = static_cast<int>(f[6]);
      st.out_off = static_cast<int>(f[7]);
      st.res_slot = static_cast<int>(f[8]);
      st.res_off = static_cast<int>(f[9]);
      TORCH_CHECK(ws_[i].is_cuda() && ws_[i].scalar_type() == at::kHalf && ws_[i].is_contiguous(),
                  "stage weight must be a contiguous fp16 CUDA tensor");
      TORCH_CHECK(ss_[i].is_cuda() && ss_[i].scalar_type() == at::kFloat && ss_[i].is_contiguous()
                      && bs_[i].is_cuda() && bs_[i].scalar_type() == at::kFloat
                      && bs_[i].is_contiguous(),
                  "stage affine must be contiguous fp32 CUDA tensors");
      st.w = reinterpret_cast<const __half*>(ws_[i].data_ptr<at::Half>());
      st.s = ss_[i].data_ptr<float>();
      st.b = bs_[i].data_ptr<float>();
    }
    slot_base_.resize(slot_widths_.size());
  }

  at::Tensor run(const at::Tensor& x) {
    TORCH_CHECK(x.dim() == 4 && x.is_cuda() && x.scalar_type() == at::kHalf && x.is_contiguous(),
                "the fused route needs a contiguous 4-D fp16 CUDA tensor");
    TORCH_CHECK(x.size(1) == in_channels_, "input channel count does not match the plan");
    const int64_t N = x.size(0), H = x.size(2), W = x.size(3);
    const int64_t HW = H * W;

    int64_t total = 0;
    for (size_t s = 0; s < slot_widths_.size(); ++s) {
      slot_base_[s] = total;
      total += N * slot_widths_[s] * HW;
    }
    at::Tensor buf = at::empty({total}, x.options());
    at::Tensor out = at::empty({N, out_channels_, H, W}, x.options());

    __half* bufp = reinterpret_cast<__half*>(buf.data_ptr<at::Half>());
    __half* outp = reinterpret_cast<__half*>(out.data_ptr<at::Half>());
    const __half* xp = reinterpret_cast<const __half*>(x.data_ptr<at::Half>());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    for (const Step& st : steps_) {
      StageArgs a;
      a.w = st.w;
      a.s = st.s;
      a.b = st.b;
      a.C = st.cin;
      a.O = st.cout;
      a.H = static_cast<int>(H);
      a.W = static_cast<int>(W);
      a.HW = static_cast<int>(HW);
      a.in = const_cast<const __half*>(slot_ptr(st.in_slot, st.in_off, bufp, outp,
                                                const_cast<__half*>(xp), HW, &a.in_bs));
      a.out = slot_ptr(st.out_slot, st.out_off, bufp, outp, const_cast<__half*>(xp), HW,
                       &a.out_bs);
      if (st.res_slot != kSlotNone) {
        a.res = slot_ptr(st.res_slot, st.res_off, bufp, outp, const_cast<__half*>(xp), HW,
                         &a.res_bs);
      } else {
        a.res = nullptr;
        a.res_bs = 0;
      }
      launch(st, a, static_cast<int>(N), stream);
    }
    return out;
  }

  // The same folded arithmetic, but with cuDNN's convolutions instead of the
  // kernels above and with the whole chain issued from C++.
  //
  // Both halves of that matter. cuDNN reaches these shapes with tensor-core
  // implicit-GEMM kernels (cutlass3x_sm100_tensorop / nvjet), and the kernels
  // above accumulate on the FFMA pipe, so on the stages with real reduction
  // depth cuDNN is several times faster on the device -- measured 8.4 us against
  // 89 us for case 1's 3x3 stage. Meanwhile the same chain driven from Python
  // spends 140-310 us of CPU against 85-148 us of device time, so it is the
  // dispatch that costs, not the arithmetic. Issuing it from here keeps one
  // Python frame per forward and lets the device work set the pace.
  at::Tensor run_aten(const at::Tensor& x) {
    TORCH_CHECK(x.dim() == 4 && x.is_cuda() && x.scalar_type() == at::kHalf && x.is_contiguous(),
                "the folded-in-C++ route needs a contiguous 4-D fp16 CUDA tensor");
    TORCH_CHECK(x.size(1) == in_channels_, "input channel count does not match the plan");
    size_t si = 0;
    at::Tensor head = stage_apply(si++, x);
    std::vector<at::Tensor> parts;
    parts.reserve(block_lens_.size() + 1);
    parts.push_back(head);
    at::Tensor prev = head.slice(1, split_, 2 * split_);
    for (size_t b = 0; b < block_lens_.size(); ++b) {
      at::Tensor h = prev;
      for (int64_t k = 0; k < block_lens_[b]; ++k) h = stage_apply(si++, h);
      if (block_add_[b]) h = h.add_(prev);
      parts.push_back(h);
      prev = h;
    }
    at::Tensor joined = parts.size() == 1 ? parts[0] : at::cat(parts, 1);
    return stage_apply(si, joined);
  }

  at::Tensor stage_apply(size_t i, const at::Tensor& in) {
    const Step& st = steps_[i];
    const int64_t pad = st.ksize / 2;
    const int64_t groups = (st.kind == kKindDw) ? st.cout : 1;
    at::Tensor y = at::conv2d(in, fws_[i], fbs_[i], {1, 1}, {pad, pad}, {1, 1}, groups);
    return at::silu_(y);
  }

 private:
  __half* slot_ptr(int slot, int off, __half* bufp, __half* outp, __half* xp, int64_t HW,
                   long long* batch_stride) const {
    if (slot == kSlotInput) {
      *batch_stride = in_channels_ * HW;
      return xp + (long long)off * HW;
    }
    if (slot == kSlotOut) {
      *batch_stride = out_channels_ * HW;
      return outp + (long long)off * HW;
    }
    *batch_stride = slot_widths_[slot] * HW;
    return bufp + slot_base_[slot] + (long long)off * HW;
  }

  // Register tiles are chosen by the number of outputs the launch has to
  // produce, because that is what decides whether the GPU can be filled at all.
  // A 4x4 register tile gives the best fused-multiply-add to shared-load ratio,
  // but it also means 16 outputs per thread: the smallest stage here produces
  // 51200 outputs, which is 3200 threads -- a fifth of the SMs holding one block
  // of four warps each, with nothing to hide shared-load latency behind
  // (measured 147 us for that stage, against 5.7 us for the thin tile that
  // spreads the same work over 80 blocks of 512 threads). Above the threshold
  // the ratio wins instead, so both tiles are compiled and selected per launch.
  // FK_YOLOV10_C2F_TILES forces one (0 thin, 1 wide) for sweeping.
  // Sizing rule, from the profile rather than from a model. The register tile
  // that gives the best load-to-multiply ratio also decides how many threads
  // exist at all, and these stages are small: cib#7's cv1 produces 102400
  // outputs in total, so a 2x4 tile is 12800 threads -- ncu measured
  // launch__waves_per_multiprocessor 0.096, achieved occupancy 5.4% and
  // issue-active 11.4%, with no throughput counter above 11% of peak. Nothing
  // was saturated; there was simply nothing resident to hide latency behind, and
  // the stage took 63.6 us. So the tile is chosen to keep roughly 100k threads
  // in flight, which for the small stages means one output per thread and gives
  // up the ratio deliberately. FK_YOLOV10_C2F_TILES pins a tier for sweeping.
  static int tile_policy(int kind) {
    static const int v[3] = {
        [] { const char* s = std::getenv("FK_YOLOV10_C2F_TILES_PW"); return s ? std::atoi(s) : -1; }(),
        [] { const char* s = std::getenv("FK_YOLOV10_C2F_TILES_CONV3"); return s ? std::atoi(s) : -1; }(),
        [] { const char* s = std::getenv("FK_YOLOV10_C2F_TILES_DW"); return s ? std::atoi(s) : -1; }(),
    };
    return v[kind];
  }

  // 0 = one output per thread, 3 = the widest tile.
  static int tier(int kind, long long outputs, const long long (&bounds)[3]) {
    const int forced = tile_policy(kind);
    if (forced >= 0) return forced > 3 ? 3 : forced;
    for (int t = 3; t >= 1; --t)
      if (outputs >= bounds[t - 1]) return t;
    return 0;
  }

  static void launch(const Step& st, const StageArgs& a, int N, cudaStream_t stream) {
    const long long outputs = (long long)a.O * a.HW * N;
    if (st.kind == kKindPw) {
      constexpr int BLK = 64;
      // The vectorised loads need an aligned pixel group and an aligned weight
      // octet; anything else takes the shared-memory tile, which has no such
      // requirement. Every benched configuration meets both.
      if (a.C % 8 == 0) {
        static constexpr long long kBounds[3] = {150000, 300000, 300000};
        const int t = tier(0, outputs, kBounds);
        if (t >= 3 && a.HW % 4 == 0) {
          const dim3 grid((a.HW / 4 + BLK - 1) / BLK, (a.O + 1) / 2, N);
          pw_cached_kernel<2, 4, BLK><<<grid, BLK, 0, stream>>>(a);
          return;
        }
        if (t >= 2 && a.HW % 2 == 0) {
          const dim3 grid((a.HW / 2 + BLK - 1) / BLK, (a.O + 1) / 2, N);
          pw_cached_kernel<2, 2, BLK><<<grid, BLK, 0, stream>>>(a);
          return;
        }
        if (t == 1 && a.HW % 2 == 0) {
          const dim3 grid((a.HW / 2 + BLK - 1) / BLK, a.O, N);
          pw_cached_kernel<1, 2, BLK><<<grid, BLK, 0, stream>>>(a);
          return;
        }
        const dim3 grid((a.HW + BLK - 1) / BLK, a.O, N);
        pw_cached_kernel<1, 1, BLK><<<grid, BLK, 0, stream>>>(a);
        return;
      }
      constexpr int BO = 32, BP = 64, BK = 16, TO = 4, TP = 4;
      const dim3 grid((a.HW + BP - 1) / BP, (a.O + BO - 1) / BO, N);
      pw_kernel<BO, BP, BK, TO, TP><<<grid, (BO / TO) * (BP / TP), 0, stream>>>(a);
    } else if (st.kind == kKindConv3) {
      static constexpr long long kBounds[3] = {0, 200000, 800000};
      constexpr int TH = 4, TWX = 32, BK = 8;
      const int nx = (a.W + TWX - 1) / TWX, ny = (a.H + TH - 1) / TH;
      const int t = tier(1, outputs, kBounds);
      if (t >= 3) {
        constexpr int BO = 32, TO = 4, TP = 4;
        const dim3 grid(nx, ny, N * ((a.O + BO - 1) / BO));
        conv3_kernel<BO, TH, TWX, BK, TO, TP>
            <<<grid, (BO / TO) * (TWX / TP) * TH, 0, stream>>>(a, N);
      } else if (t == 2) {
        constexpr int BO = 16, TO = 2, TP = 4;
        const dim3 grid(nx, ny, N * ((a.O + BO - 1) / BO));
        conv3_kernel<BO, TH, TWX, BK, TO, TP>
            <<<grid, (BO / TO) * (TWX / TP) * TH, 0, stream>>>(a, N);
      } else if (t == 1) {
        constexpr int BO = 8, TO = 1, TP = 2;
        const dim3 grid(nx, ny, N * ((a.O + BO - 1) / BO));
        conv3_kernel<BO, TH, TWX, BK, TO, TP>
            <<<grid, (BO / TO) * (TWX / TP) * TH, 0, stream>>>(a, N);
      } else {
        constexpr int BO = 8, TO = 1, TP = 1;
        const dim3 grid(nx, ny, N * ((a.O + BO - 1) / BO));
        conv3_kernel<BO, TH, TWX, BK, TO, TP>
            <<<grid, (BO / TO) * (TWX / TP) * TH, 0, stream>>>(a, N);
      }
    } else {
      static constexpr long long kBounds[3] = {100000, 300000, 1LL << 62};
      constexpr int TH = 8, TWX = 32;
      const dim3 grid((a.W + TWX - 1) / TWX, (a.H + TH - 1) / TH, N * a.O);
      const int t = tier(2, outputs, kBounds);
      if (st.ksize == 7) {
        if (t >= 2) dw_kernel<7, TH, TWX, 4><<<grid, (TWX / 4) * TH, 0, stream>>>(a, a.O);
        else if (t == 1) dw_kernel<7, TH, TWX, 2><<<grid, (TWX / 2) * TH, 0, stream>>>(a, a.O);
        else dw_kernel<7, TH, TWX, 1><<<grid, TWX * TH, 0, stream>>>(a, a.O);
      } else {
        if (t >= 2) dw_kernel<3, TH, TWX, 4><<<grid, (TWX / 4) * TH, 0, stream>>>(a, a.O);
        else if (t == 1) dw_kernel<3, TH, TWX, 2><<<grid, (TWX / 2) * TH, 0, stream>>>(a, a.O);
        else dw_kernel<3, TH, TWX, 1><<<grid, TWX * TH, 0, stream>>>(a, a.O);
      }
    }
  }

  std::vector<at::Tensor> ws_, ss_, bs_, fws_, fbs_;
  std::vector<int64_t> slot_widths_, block_lens_, block_add_;
  std::vector<int64_t> slot_base_;
  std::vector<Step> steps_;
  int64_t in_channels_, out_channels_, split_;
};

}  // namespace

int64_t plan_create(std::vector<at::Tensor> ws, std::vector<at::Tensor> ss,
                    std::vector<at::Tensor> bs, std::vector<at::Tensor> fws,
                    std::vector<at::Tensor> fbs, std::vector<int64_t> ints,
                    std::vector<int64_t> slot_widths, std::vector<int64_t> block_lens,
                    std::vector<int64_t> block_add, int64_t in_channels,
                    int64_t out_channels, int64_t split) {
  auto* plan = new BlockPlan(std::move(ws), std::move(ss), std::move(bs), std::move(fws),
                             std::move(fbs), ints, std::move(slot_widths),
                             std::move(block_lens), std::move(block_add), in_channels,
                             out_channels, split);
  return reinterpret_cast<int64_t>(plan);
}

at::Tensor plan_run(int64_t handle, const at::Tensor& x) {
  return reinterpret_cast<BlockPlan*>(handle)->run(x);
}

at::Tensor plan_run_aten(int64_t handle, const at::Tensor& x) {
  return reinterpret_cast<BlockPlan*>(handle)->run_aten(x);
}

void plan_destroy(int64_t handle) {
  delete reinterpret_cast<BlockPlan*>(handle);
}
"""

# Declarations must live in cpp_sources: load_inline generates the pybind11
# module into main.cpp, which cannot see anything declared only in the .cu
# (observed directly -- "was not declared in this scope", scratch/micro2.log).
_CPP_SOURCE = r"""
#include <torch/extension.h>
#include <vector>

int64_t plan_create(std::vector<at::Tensor> ws, std::vector<at::Tensor> ss,
                    std::vector<at::Tensor> bs, std::vector<at::Tensor> fws,
                    std::vector<at::Tensor> fbs, std::vector<int64_t> ints,
                    std::vector<int64_t> slot_widths, std::vector<int64_t> block_lens,
                    std::vector<int64_t> block_add, int64_t in_channels,
                    int64_t out_channels, int64_t split);
at::Tensor plan_run(int64_t handle, const at::Tensor& x);
at::Tensor plan_run_aten(int64_t handle, const at::Tensor& x);
void plan_destroy(int64_t handle);
"""


def _build_directory() -> str | None:
    """Workspace-local build cache, so a warm import never invokes nvcc."""
    try:
        path = Path(__file__).resolve().parents[2] / ".torch_extensions" / _EXTENSION_NAME
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    return str(path)


def _load_extension(arch: str | None = _CUDA_ARCH):
    # Pinning the arch list is load-bearing twice over. This workspace's shell
    # exports TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 9.0 10.0 12.0+PTX", so without
    # the pin the same source compiles six times; and wherever the variable is
    # unset or "native", that branch of _get_cuda_arch_flags indexes
    # torch.cuda.device_count() and raises IndexError in a process with no
    # visible GPU -- which is exactly the process this module has to import in.
    previous = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if arch is None:
        os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
    else:
        os.environ["TORCH_CUDA_ARCH_LIST"] = arch
    try:
        return load_inline(
            name=_EXTENSION_NAME,
            cpp_sources=_CPP_SOURCE,
            cuda_sources=_CUDA_SOURCE,
            functions=["plan_create", "plan_run", "plan_run_aten", "plan_destroy"],
            # -lineinfo lets ncu attribute SASS to source. --use_fast_math is
            # deliberately absent: every approximation here is an explicit,
            # auditable intrinsic, the discipline the frozen L1 winners set.
            extra_cuda_cflags=["-O3", "-lineinfo"],
            build_directory=_build_directory(),
            verbose=False,
        )
    finally:
        if previous is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = previous


# Built at import, never lazily inside forward: ninja spawns subprocesses, and
# the harness fails any candidate whose thread count grows during its timing
# window. EXTENSION_STATUS is the empty string exactly when the fused route is
# live and carries the build error otherwise, in which case the folded route
# serves every configuration and this module still imports and still passes.
EXTENSION_STATUS = ""
try:
    _EXT = _load_extension()
except Exception as exc:  # noqa: BLE001 - a build failure must stay importable
    _EXT = None
    EXTENSION_STATUS = f"{type(exc).__name__}: {exc}"
    print(f"[yolov10_c2f] CUDA extension unavailable, folded route only: {EXTENSION_STATUS}",
          file=sys.stderr, flush=True)


class _FusedStepProgram:
    """Owns the C++ plan's lifetime; the hot path calls through it directly."""

    __slots__ = ("handle", "run", "run_aten")

    def __init__(self, weights, scales, biases, folded_weights, folded_biases, fields,
                 slot_widths, block_lens, block_add, cin, cout, split):
        self.handle = _EXT.plan_create(weights, scales, biases, folded_weights, folded_biases,
                                       fields, list(slot_widths), list(block_lens),
                                       list(block_add), int(cin), int(cout), int(split))
        self.run = _EXT.plan_run
        self.run_aten = _EXT.plan_run_aten

    def __del__(self):
        handle, self.handle = getattr(self, "handle", 0), 0
        if handle and _EXT is not None:
            _EXT.plan_destroy(handle)


class _Stage:
    """One folded convolution: weight, per-channel affine, SiLU, maybe residual.

    ``weight``/``scale``/``bias`` serve the fused route, which applies the affine
    in its fp32 epilogue and reads the convolution weight exactly as stored.
    ``folded_weight``/``folded_bias`` serve the folded route, where the scale has
    to ride in the weight because ``F.conv2d`` applies nothing but a bias.

    Slots and channel offsets address the fused route's buffers: slot 0 is the
    concatenation-layout buffer ``Y``, slots >= 1 are per-stage temporaries, and
    slot -1 is the block's own output tensor.
    """

    __slots__ = ("kind", "ksize", "pad", "groups", "cin", "cout",
                 "weight", "scale", "bias", "folded_weight", "folded_bias",
                 "in_slot", "in_off", "out_slot", "out_off", "res_slot", "res_off")

    def __init__(self, kind, ksize, groups, cin, cout, weight, scale, bias):
        self.kind = kind
        self.ksize = ksize
        self.pad = ksize // 2
        self.groups = groups
        self.cin = cin
        self.cout = cout
        self.weight = weight
        self.scale = scale
        self.bias = bias
        # (w * s) re-rounds every weight element (~5e-4 relative); the fused
        # route avoids it by keeping s in the fp32 epilogue it already runs.
        self.folded_weight = (weight.float() * scale.view(-1, 1, 1, 1)).to(weight.dtype)
        self.folded_bias = bias.to(weight.dtype)
        self.in_slot = self.in_off = 0
        self.out_slot = self.out_off = 0
        self.res_slot = _SLOT_NONE
        self.res_off = 0


class _Plan:
    """Folded constants plus the step program, valid for one weight generation."""

    __slots__ = ("cin", "cout", "c", "n", "dtype", "dev_index", "stages", "blocks",
                 "slot_widths", "y_width", "max_buffer_channels",
                 "fused", "fused_run", "fused_handle", "aten_run", "route")

    def __init__(self):
        self.stages = []          # flat, in execution order (fused route)
        self.blocks = []          # [(stages, add), ...] per bottleneck/CIB (folded route)
        self.slot_widths = []     # channel width of slot 0 (Y) then each temporary
        self.fused = None         # owns the C++ plan
        self.fused_run = None     # the extension entry points, hoisted out of forward
        self.aten_run = None
        self.fused_handle = 0
        self.route = ROUTE_FOLDED


def _bn_affine(bn):
    """BN in eval mode as a per-channel affine, read from the live buffers.

    Under the harness ``bn.weight`` is exactly 1, ``running_mean`` 0 and
    ``running_var`` 1 (the sanitizer rewrites all-zero *parameters*; BN's
    running stats are buffers and stay fp32), but reading all four rather than
    assuming those values is what makes this correct for any other caller.
    """
    if bn.weight is None or bn.bias is None or bn.running_var is None or bn.running_mean is None:
        return None
    var = bn.running_var.detach().float()
    scale = bn.weight.detach().float() / torch.sqrt(var + bn.eps)
    bias = bn.bias.detach().float() - scale * bn.running_mean.detach().float()
    return scale, bias


def _conv_ok(conv, dtype, device):
    return (conv.bias is None
            and tuple(conv.stride) == (1, 1)
            and tuple(conv.dilation) == (1, 1)
            and conv.weight.dtype is dtype
            and conv.weight.device == device
            and conv.weight.dim() == 4
            and conv.weight.shape[2] == conv.weight.shape[3])


def _stage_from_conv(mod, dtype, device):
    """Fold one YOLOConv into a stage, or return None if it is not serveable."""
    if not isinstance(mod, YOLOConv) or getattr(mod, "_is_fused", False):
        return None
    if mod.act is not _SILU_ACT:
        return None
    bn = getattr(mod, "bn", None)
    if bn is None or bn.training:
        return None
    conv = mod.conv
    if not _conv_ok(conv, dtype, device):
        return None
    k = int(conv.weight.shape[2])
    cout = int(conv.weight.shape[0])
    groups = int(conv.groups)
    cin = int(conv.weight.shape[1]) * groups
    if tuple(conv.padding) != (k // 2, k // 2) or bn.num_features != cout:
        return None
    if groups == 1:
        kind = _KIND_PW if k == 1 else _KIND_CONV3
        if k not in (1, 3):
            return None
    elif groups == cin == cout and k in (3, 7):
        kind = _KIND_DW
    else:
        return None
    affine = _bn_affine(bn)
    if affine is None:
        return None
    scale, bias = affine
    weight = conv.weight.detach()
    if not weight.is_contiguous():
        weight = weight.contiguous()
    return _Stage(kind, k, groups, cin, cout, weight, scale, bias)


def _stage_from_repvggdw(mod, dtype, device):
    """Fold RepVGGDW's two depthwise branches into one 7x7 depthwise stage.

    The branches carry different BN scales, so here -- unlike everywhere else --
    the fold has to reach the weights: ``W = pad(w3*s3) + w7*s7`` and
    ``B = b3 + b7``, computed in fp32 and stored once as a single fp16 filter,
    exactly as the baseline's own ``fuse()`` does.
    """
    if not isinstance(mod, YOLORepVGGDW) or getattr(mod, "_is_fused", False):
        return None
    # RepVGGDW holds its own SiLU instance rather than YOLOConv's shared one.
    if isinstance(mod.act, nn.Identity) or type(mod.act).__name__ != "SiLU":
        return None
    parts = []
    for branch, ksize in ((mod.conv, 7), (mod.conv1, 3)):
        if not isinstance(branch, YOLOConv) or not isinstance(branch.act, nn.Identity):
            return None
        bn = getattr(branch, "bn", None)
        if bn is None or bn.training:
            return None
        conv = branch.conv
        if not _conv_ok(conv, dtype, device):
            return None
        k = int(conv.weight.shape[2])
        ed = int(conv.weight.shape[0])
        if (k != ksize or int(conv.groups) != ed or int(conv.weight.shape[1]) != 1
                or tuple(conv.padding) != (k // 2, k // 2) or bn.num_features != ed):
            return None
        affine = _bn_affine(bn)
        if affine is None:
            return None
        parts.append((conv.weight.detach().float(), affine[0], affine[1], k))
    (w7, s7, b7, _), (w3, s3, b3, _) = parts
    if w7.shape[0] != w3.shape[0]:
        return None
    weight = w7 * s7.view(-1, 1, 1, 1) + F.pad(w3 * s3.view(-1, 1, 1, 1), (2, 2, 2, 2))
    ed = int(weight.shape[0])
    stage = _Stage(_KIND_DW, 7, ed, ed, ed, weight.to(dtype),
                   torch.ones(ed, dtype=torch.float32, device=device), b3 + b7)
    # The scale is already inside the weight, so the fused route's epilogue must
    # not apply it twice; folded_weight is then the same filter.
    stage.folded_weight = stage.weight
    return stage


def _block_stages(block, dtype, device):
    """The stages of one bottleneck / CIB, in order, plus its residual flag."""
    if isinstance(block, YOLOBottleneck):
        mods = [block.cv1, block.cv2]
    elif isinstance(block, YOLOCIB):
        if not isinstance(block.cv1, nn.Sequential):
            return None
        mods = list(block.cv1)
    else:
        return None
    stages = []
    for mod in mods:
        stage = (_stage_from_repvggdw(mod, dtype, device) if isinstance(mod, YOLORepVGGDW)
                 else _stage_from_conv(mod, dtype, device))
        if stage is None:
            return None
        stages.append(stage)
    for prev, nxt in zip(stages, stages[1:]):
        if prev.cout != nxt.cin:
            return None
    return stages, bool(block.add)


class YOLOC2f(nn.Module):
    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = YOLOConv(c1, 2 * self.c, 1, 1)
        self.cv2 = YOLOConv((2 + n) * self.c, c2, 1, 1)
        self.m = nn.ModuleList(
            YOLOBottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n)
        )
        # Plain attributes, never register_buffer: anything in state_dict would
        # break the key-for-key identity the harness's strict=False load needs.
        self._plan = None
        self._plan_failed = False
        self.register_load_state_dict_post_hook(_drop_plan)

    # -- routing ----------------------------------------------------------
    def _apply(self, *args, **kwargs):
        # .to()/.half()/.float()/.cuda()/.cpu() all funnel through here, and all
        # of them invalidate constants derived from parameter values.
        self._plan = None
        self._plan_failed = False
        return super()._apply(*args, **kwargs)

    def _fast_path_applies(self, x, plan) -> bool:
        """Only what can change between calls; everything else is plan-time."""
        return (x.dim() == 4
                and x.shape[1] == plan.cin
                and x.dtype is plan.dtype
                and x.is_cuda
                and x.get_device() == plan.dev_index
                and x.is_contiguous()
                and x.numel() > 0
                and not self.training
                and not torch.is_grad_enabled()
                and not torch.is_autocast_enabled("cuda")
                and x.shape[0] * plan.max_buffer_channels * x.shape[2] * x.shape[3] <= _INT32_MAX)

    def build_plan(self):
        """Fold the module's live weights, or return None if it is not serveable.

        Called on the first eligible forward -- never from ``__init__``, where
        the weights are still whatever ``torch.empty`` left behind.
        """
        cv1, cv2 = self.cv1, self.cv2
        if not isinstance(cv1, YOLOConv) or not isinstance(cv2, YOLOConv):
            return None
        dtype = cv1.conv.weight.dtype
        device = cv1.conv.weight.device
        if dtype is not torch.float16 or device.type != "cuda":
            return None
        for p in self.parameters():
            if p.device != device or (p.is_floating_point() and p.dtype is not dtype):
                return None
        head = _stage_from_conv(cv1, dtype, device)
        tail = _stage_from_conv(cv2, dtype, device)
        if head is None or tail is None or head.kind != _KIND_PW or tail.kind != _KIND_PW:
            return None
        c = int(self.c)
        n = len(self.m)
        if n < 1 or head.cout != 2 * c or tail.cin != (2 + n) * c:
            return None

        plan = _Plan()
        plan.c = c
        plan.n = n
        plan.cin = head.cin
        plan.cout = tail.cout
        plan.dtype = dtype
        plan.dev_index = device.index if device.index is not None else torch.cuda.current_device()
        plan.y_width = (2 + n) * c

        # Y holds the concatenation the baseline builds with chunk + cat:
        #   cv1 -> Y[:, 0:2c]; block i reads Y[:, (1+i)c:(2+i)c] and writes the
        #   next slice, which is also where the residual reads from. Reads and
        #   writes are disjoint by construction -- a stage never writes over a
        #   slice it is still reading -- and the channel order is exactly cv1
        #   chunk 0, cv1 chunk 1, block 1, ..., block n.
        head.in_slot, head.in_off = _SLOT_INPUT, 0
        head.out_slot, head.out_off = 0, 0
        plan.stages.append(head)
        plan.slot_widths = [plan.y_width]

        for i, block in enumerate(self.m):
            found = _block_stages(block, dtype, device)
            if found is None:
                return None
            stages, add = found
            if stages[0].cin != c or stages[-1].cout != c:
                return None
            in_off = (1 + i) * c
            for j, stage in enumerate(stages):
                stage.in_slot, stage.in_off = (0, in_off) if j == 0 else (
                    len(plan.slot_widths) - 1, 0)
                if j == len(stages) - 1:
                    stage.out_slot, stage.out_off = 0, (2 + i) * c
                    if add:
                        stage.res_slot, stage.res_off = 0, in_off
                else:
                    plan.slot_widths.append(stage.cout)
                    stage.out_slot, stage.out_off = len(plan.slot_widths) - 1, 0
                plan.stages.append(stage)
            plan.blocks.append((stages, add))

        tail.in_slot, tail.in_off = 0, 0
        tail.out_slot, tail.out_off = _SLOT_OUT, 0
        plan.stages.append(tail)
        plan.max_buffer_channels = max(max(plan.slot_widths), plan.cout, plan.cin)
        _attach_step_program(plan)
        return plan

    def _run_folded(self, plan, x):
        """Folded constants through ATen: BN and its op per convolution are gone.

        ``F.conv2d`` has no ``out=``, so this route cannot write into slices of
        one buffer the way the fused route does -- a ``copy_`` into Y would cost
        more dispatch than the single ``cat`` it replaces. The Y layout is
        therefore the fused route's alone.
        """
        head, tail = plan.stages[0], plan.stages[-1]
        y0 = F.conv2d(x, head.folded_weight, head.folded_bias)
        F.silu(y0, inplace=True)
        parts = [y0]
        prev = y0[:, plan.c:]
        for stages, add in plan.blocks:
            h = prev
            for stage in stages:
                h = F.conv2d(h, stage.folded_weight, stage.folded_bias,
                             padding=stage.pad, groups=stage.groups)
                F.silu(h, inplace=True)
            if add:
                h = h.add_(prev)
            parts.append(h)
            prev = h
        cat = parts[0] if len(parts) == 1 else torch.cat(parts, 1)
        out = F.conv2d(cat, tail.folded_weight, tail.folded_bias)
        return F.silu(out, inplace=True)

    def _measure_route(self, plan, x) -> None:
        """Take the fastest of the correct fast routes, once, at plan-build time.

        Which route wins is a property of the configuration, not of the input
        values, and it genuinely varies: the kernels in this file beat cuDNN on
        the shallow stages (case 3's 16-channel block, the CIB's pointwise chain)
        and lose on the ones with real reduction depth, where cuDNN reaches the
        tensor cores. Rather than freeze a table of guesses, the routes are timed
        against each other here -- on the first eligible forward, which is a
        warm-up call, never a timed one -- so no configuration can end up on a
        route slower than the one below it.
        """
        options = [(ROUTE_FOLDED, lambda: self._run_folded(plan, x))]
        if plan.aten_run is not None:
            options.append((ROUTE_ATEN, lambda: plan.aten_run(plan.fused_handle, x)))
            options.append((ROUTE_FUSED, lambda: plan.fused_run(plan.fused_handle, x)))
        best, best_ms = ROUTE_FOLDED, float("inf")
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        for route, call in options:
            try:
                for _ in range(3):
                    call()
                torch.cuda.synchronize()
                start.record()
                for _ in range(_ROUTE_TRIALS):
                    call()
                end.record()
                torch.cuda.synchronize()
                ms = start.elapsed_time(end)
            except Exception:  # noqa: BLE001 - a route that cannot run cannot win
                continue
            if ms < best_ms:
                best, best_ms = route, ms
        plan.route = best

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        route = _ROUTE_OVERRIDE
        if route is not ROUTE_REFERENCE:
            plan = self._plan
            if plan is None and not self._plan_failed:
                plan = self.build_plan()
                self._plan_failed = plan is None
                if plan is not None and self._fast_path_applies(x, plan):
                    self._measure_route(plan, x)
                self._plan = plan
            if plan is not None and self._fast_path_applies(x, plan):
                taken = route or plan.route
                if taken is ROUTE_FUSED and plan.fused_run is not None:
                    return plan.fused_run(plan.fused_handle, x)
                if taken is ROUTE_ATEN and plan.aten_run is not None:
                    return plan.aten_run(plan.fused_handle, x)
                return self._run_folded(plan, x)
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class YOLOC2fCIB(YOLOC2f):
    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, lk: bool = False, g: int = 1, e: float = 0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(YOLOCIB(self.c, self.c, shortcut, e=1.0, lk=lk) for _ in range(n))


def _attach_step_program(plan) -> None:
    """Hand the folded stages to the extension as one flat step program."""
    if _EXT is None:
        return
    fields, weights, scales, biases = [], [], [], []
    folded_weights, folded_biases = [], []
    for stage in plan.stages:
        fields.extend((stage.kind, stage.cin, stage.cout, stage.ksize,
                       stage.in_slot, stage.in_off, stage.out_slot, stage.out_off,
                       stage.res_slot, stage.res_off))
        weights.append(stage.weight)
        scales.append(stage.scale)
        biases.append(stage.bias)
        folded_weights.append(stage.folded_weight)
        folded_biases.append(stage.folded_bias)
    try:
        plan.fused = _FusedStepProgram(
            weights, scales, biases, folded_weights, folded_biases, fields,
            plan.slot_widths, [len(st) for st, _ in plan.blocks],
            [int(add) for _, add in plan.blocks], plan.cin, plan.cout, plan.c)
    except Exception as exc:  # noqa: BLE001 - the folded route still serves this
        print(f"[yolov10_c2f] step program rejected, folded route only: {exc}",
              file=sys.stderr, flush=True)
        return
    plan.fused_run = plan.fused.run
    plan.aten_run = plan.fused.run_aten
    plan.fused_handle = plan.fused.handle


def _drop_plan(module, incompatible_keys):
    """load_state_dict post-hook: new weight values, so the fold is stale."""
    module._plan = None
    module._plan_failed = False


# Normalised through this table so the comparisons in forward() are identity
# checks against this module's own constants rather than string equality.
_ROUTES = {ROUTE_FUSED: ROUTE_FUSED, ROUTE_ATEN: ROUTE_ATEN, ROUTE_FOLDED: ROUTE_FOLDED,
           ROUTE_REFERENCE: ROUTE_REFERENCE}
_ROUTE_OVERRIDE = _ROUTES.get(os.environ.get(_ROUTE_ENV, "").strip().lower())
