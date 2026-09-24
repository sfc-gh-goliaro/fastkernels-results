"""YOLOv10 RepVGG depthwise block, collapsed into a single fused CUDA kernel.

In eval mode each ``BatchNorm2d`` is a per-channel affine map, and both branches
of the block are shape-preserving depthwise convolutions.  The block therefore
reduces exactly to one depthwise 7x7 convolution with a per-channel bias
followed by SiLU -- the same algebra the reference block's own ``fuse()``
performs, but folded once and cached instead of re-dispatched::

    s_k[c] = bn_k.weight[c] / sqrt(bn_k.running_var[c] + bn_k.eps)
    t_k[c] = bn_k.bias[c]  - s_k[c] * bn_k.running_mean[c]        for k in {7, 3}

    Wf[c,0,i,j] = s7[c]*W7[c,0,i,j] + (s3[c]*W3[c,0,i-2,j-2] if 2 <= i,j <= 4 else 0)
    Bf[c]       = t7[c] + t3[c]

    out = SiLU(dwconv7x7(x, Wf, pad=3) + Bf)

Six dispatches (conv, bn, conv, bn, add, silu) become one launch, which is what
matters here: the operator is ~20 M MAC on ~1.6 MB of traffic, so its latency is
dominated by per-call overhead rather than by throughput.  Anything the fused
path cannot serve -- a missing extension, training mode, an odd dtype, layout or
rank -- runs the reference algebra on the module's own submodules instead.

The cached fold is derived state, not weights: it lives in ``__dict__`` (so it
never reaches ``state_dict()`` and never costs an ``nn.Module.__getattr__`` on
the hot path) and is dropped whenever anything it was derived from can have
moved.  ``train()``, ``_apply`` (``.to()`` / ``.half()`` / ``.cuda()``),
``load_state_dict`` and ``fuse()`` invalidate it outright, and each call re-reads
the module tree's own dicts and the handful of scalars the fold depends on -- see
``_Fold`` for exactly what and why, and for the one mutation that is documented
as unsupported.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import torch
import torch.nn as nn

from ..L1.silu import SiLU
from ..L1.tensor_ops import Pad
from .yolov10_conv import YOLOConv

__all__ = ["YOLORepVGGDW"]

# ---------------------------------------------------------------------------
# Fused kernel: one block per (n, c) plane, one launch per forward.
#
# The staged variant copies the padded input plane into shared memory as fp32 so
# the 7x7 stencil reads it 49 times without re-converting, and keeps the 49
# folded weights alongside it (every thread reads identical addresses, so those
# loads broadcast).  Every element of the padded tile is written exactly once --
# halo to zero, interior to the converted input -- so one barrier separates
# staging from the sweep and there is no write-after-write hazard.
#
# The direct variant skips staging and reads the input from global memory with
# bounds checks, relying on L1/L2: the harness writes the input inside the
# measured window, so it is cache-hot when the kernel runs and staging is not
# obviously a win.  Which variant is faster is a measurement, not an assumption.
# ---------------------------------------------------------------------------

_CUDA_SOURCE = r"""
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>

namespace {

constexpr int KS = 7;   // fused kernel extent
constexpr int PAD = 3;  // zero padding that keeps the plane shape

__device__ __forceinline__ float silu(float a) {
  return a / (1.0f + __expf(-a));
}

// Padded plane staged in shared memory, then swept 7x7 per output pixel.
template <typename scalar_t>
__global__ void repvggdw_silu_smem_kernel(
    const scalar_t* __restrict__ x,
    const float* __restrict__ wf,
    const float* __restrict__ bf,
    scalar_t* __restrict__ out,
    const int C, const int H, const int W,
    const int HW, const int PW, const int tile_elems) {
  extern __shared__ float smem[];
  float* tile = smem;            // tile_elems floats: (H + 6) rows of pitch PW
  float* w = smem + tile_elems;  // KS*KS folded weights

  const int plane = blockIdx.x;
  const int c = plane % C;
  const long long base = static_cast<long long>(plane) * HW;
  const scalar_t* xp = x + base;

  // Single assignment per element the sweep can read -- halo to zero, interior to
  // the converted input.  Only (H+6) x (W+6) is meaningful; the pitch may be
  // wider, and the slack between rows is never read.
  const int TW = W + 2 * PAD;
  for (int i = threadIdx.x; i < (H + 2 * PAD) * TW; i += blockDim.x) {
    const int ph = i / TW;
    const int pw = i - ph * TW;
    const int ih = ph - PAD;
    const int iw = pw - PAD;
    tile[ph * PW + pw] = (ih >= 0 && ih < H && iw >= 0 && iw < W)
                             ? static_cast<float>(xp[ih * W + iw])
                             : 0.0f;
  }
  for (int i = threadIdx.x; i < KS * KS; i += blockDim.x) {
    w[i] = wf[c * (KS * KS) + i];
  }
  const float b = bf[c];
  __syncthreads();

  scalar_t* op = out + base;
  for (int idx = threadIdx.x; idx < HW; idx += blockDim.x) {
    const int oh = idx / W;
    const int ow = idx - oh * W;
    float acc = b;
#pragma unroll
    for (int i = 0; i < KS; ++i) {
      const float* row = tile + (oh + i) * PW + ow;
#pragma unroll
      for (int j = 0; j < KS; ++j) {
        acc = fmaf(row[j], w[i * KS + j], acc);
      }
    }
    op[idx] = static_cast<scalar_t>(silu(acc));
  }
}

// No staging: the stencil reads the input straight from global memory.
template <typename scalar_t>
__global__ void repvggdw_silu_direct_kernel(
    const scalar_t* __restrict__ x,
    const float* __restrict__ wf,
    const float* __restrict__ bf,
    scalar_t* __restrict__ out,
    const int C, const int H, const int W, const int HW) {
  __shared__ float w[KS * KS];

  const int plane = blockIdx.x;
  const int c = plane % C;
  const long long base = static_cast<long long>(plane) * HW;
  const scalar_t* xp = x + base;

  for (int i = threadIdx.x; i < KS * KS; i += blockDim.x) {
    w[i] = wf[c * (KS * KS) + i];
  }
  const float b = bf[c];
  __syncthreads();

  scalar_t* op = out + base;
  for (int idx = threadIdx.x; idx < HW; idx += blockDim.x) {
    const int oh = idx / W;
    const int ow = idx - oh * W;
    float acc = b;
#pragma unroll
    for (int i = 0; i < KS; ++i) {
      const int ih = oh + i - PAD;
      if (ih < 0 || ih >= H) continue;
      const scalar_t* row = xp + ih * W;
      const float* wr = w + i * KS;
#pragma unroll
      for (int j = 0; j < KS; ++j) {
        const int iw = ow + j - PAD;
        if (iw < 0 || iw >= W) continue;
        acc = fmaf(static_cast<float>(row[iw]), wr[j], acc);
      }
    }
    op[idx] = static_cast<scalar_t>(silu(acc));
  }
}


// Each block owns a horizontal band of one (n, c) plane, and each thread owns a
// column strip of ROWS vertically adjacent outputs inside it.
//
// The strip is what cuts work: those ROWS outputs' 7x7 windows overlap in 6 of 7
// rows, so one tile-row read feeds up to ROWS accumulators and (ROWS + 6) * 7
// shared loads buy ROWS outputs instead of ROWS * 49.  The arithmetic is
// unchanged; only the load count falls.
//
// The band is what supplies parallelism: one block per plane leaves the grid far
// smaller than the machine (256 blocks over 148 SMs at N=1), so every block is
// resident at once and the kernel lasts as long as a single block does.  Cutting
// each block's work and raising the block count instead shortens that critical
// path.  ``bands`` of 1 reduces exactly to one block per plane.
template <typename scalar_t, int ROWS>
__global__ void repvggdw_silu_rows_kernel(
    const scalar_t* __restrict__ x,
    const float* __restrict__ wf,
    const float* __restrict__ bf,
    scalar_t* __restrict__ out,
    const int C, const int H, const int W,
    const int HW, const int PW, const int rows_per_band, const int bands) {
  extern __shared__ float smem[];

  const int band = blockIdx.x % bands;
  const int plane = blockIdx.x / bands;
  const int c = plane % C;
  const long long base = static_cast<long long>(plane) * HW;
  const scalar_t* xp = x + base;

  const int h0 = band * rows_per_band;
  const int h1 = min(H, h0 + rows_per_band);
  const int band_rows = h1 - h0;
  const int tile_rows = band_rows + 2 * PAD;

  float* tile = smem;                    // tile_rows rows of pitch PW
  float* w = smem + tile_rows * PW;      // KS*KS folded weights

  // Single assignment per element the sweep can read.  Tile row ``tr`` holds
  // input row ``h0 - PAD + tr``; rows outside the plane are the zero halo.
  const int TW = W + 2 * PAD;
  for (int i = threadIdx.x; i < tile_rows * TW; i += blockDim.x) {
    const int tr = i / TW;
    const int pw = i - tr * TW;
    const int ih = h0 - PAD + tr;
    const int iw = pw - PAD;
    tile[tr * PW + pw] = (ih >= 0 && ih < H && iw >= 0 && iw < W)
                             ? static_cast<float>(xp[ih * W + iw])
                             : 0.0f;
  }
  for (int i = threadIdx.x; i < KS * KS; i += blockDim.x) {
    w[i] = wf[c * (KS * KS) + i];
  }
  const float b = bf[c];
  __syncthreads();

  scalar_t* op = out + base;
  const int strips = ((band_rows + ROWS - 1) / ROWS) * W;
  for (int s = threadIdx.x; s < strips; s += blockDim.x) {
    const int sh = s / W;
    const int ow = s - sh * W;
    const int lr0 = sh * ROWS;  // first output row of this strip, band-relative

    float acc[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) acc[r] = b;

#pragma unroll
    for (int i = 0; i < ROWS + KS - 1; ++i) {
      const int tr = lr0 + i;
      if (tr >= tile_rows) break;
      const float* row = tile + tr * PW + ow;
      float v[KS];
#pragma unroll
      for (int j = 0; j < KS; ++j) v[j] = row[j];
#pragma unroll
      for (int r = 0; r < ROWS; ++r) {
        const int k = i - r;
        if (k >= 0 && k < KS) {
          const float* wr = w + k * KS;
#pragma unroll
          for (int j = 0; j < KS; ++j) acc[r] = fmaf(v[j], wr[j], acc[r]);
        }
      }
    }

#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      const int lr = lr0 + r;
      if (lr < band_rows) op[(h0 + lr) * W + ow] = static_cast<scalar_t>(silu(acc[r]));
    }
  }
}

// ---------------------------------------------------------------------------
// fp16 specialization: half shared tile, adjacent output columns paired in
// half2, one __hfma2 per tap per pair.
//
// Two things get cheaper at once.  Arithmetic halves, because a tap now retires
// two outputs per instruction instead of one.  Shared traffic halves again,
// because a strip of ROWS output rows x 2 output columns needs tile columns
// ow..ow+7 per tile row -- exactly four aligned half2 loads, from which the seven
// shifted tap pairs are built with three byte permutes and no further loads.
//
// The tile is zeroed and then filled, separated by two barriers, rather than
// assigned once: the fill reads the plane through 16-byte uint4 loads over the
// flat contiguous plane, whose element-to-row mapping does not line up with the
// halo, so single-assignment would cost the vectorization.
//
// PLANES > 1 gives each plane its own thread cohort and its own disjoint tile and
// weight region, cutting the block count for a given amount of work.
// ---------------------------------------------------------------------------

__device__ __forceinline__ __half2 shift_pair(__half2 lo, __half2 hi) {
  // (lo.y, hi.x) -- the odd-offset pair, without an unaligned shared access.
  unsigned a, b, r;
  memcpy(&a, &lo, sizeof(a));
  memcpy(&b, &hi, sizeof(b));
  r = __byte_perm(a, b, 0x5432);
  __half2 packed;
  memcpy(&packed, &r, sizeof(packed));
  return packed;
}

template <int ROWS, int PLANES>
__global__ void repvggdw_silu_h2_kernel(
    const __half* __restrict__ x,
    const __half* __restrict__ wh,
    const __half* __restrict__ bh,
    __half* __restrict__ out,
    const int C, const int H, const int W,
    const int HW, const int PW, const int rows_per_band, const int bands,
    const int planes_total, const int tile_halves) {
  extern __shared__ __half2 smem_h2[];

  const int cohort_threads = blockDim.x / PLANES;
  const int cohort = threadIdx.x / cohort_threads;
  const int lane = threadIdx.x - cohort * cohort_threads;

  __half* tile = reinterpret_cast<__half*>(smem_h2) + cohort * tile_halves;
  __half2* w2 = smem_h2 + (PLANES * tile_halves) / 2 + cohort * (KS * KS);

  const int slot = blockIdx.x * PLANES + cohort;
  const int band = slot % bands;
  const int plane = slot / bands;
  const bool live = plane < planes_total && cohort < PLANES;

  const int h0 = live ? band * rows_per_band : 0;
  const int h1 = live ? min(H, h0 + rows_per_band) : 0;
  const int band_rows = h1 - h0;
  const int tile_rows = band_rows + 2 * PAD;

  // Zero the whole tile, then fill the interior: two barriers, no overlap.
  if (live) {
    __half2* t2 = reinterpret_cast<__half2*>(tile);
    const int zero_h2 = (tile_rows * PW) >> 1;
    const __half2 zero = __floats2half2_rn(0.f, 0.f);
    for (int i = lane; i < zero_h2; i += cohort_threads) t2[i] = zero;
  }
  __syncthreads();

  if (live) {
    // The tensor base is 16-byte aligned, but a plane base only inherits that when
    // HW is a multiple of eight -- otherwise plane 1 starts 8 bytes in.  So the fill
    // is head / middle / tail: copy scalars up to the first globally 8-element
    // aligned index, vector-load the aligned middle, and copy the remainder
    // scalarly.  Every plane length is then legal, with no eligibility restriction.
    const long long gbase = static_cast<long long>(plane) * HW;
    const __half* xp = x + gbase;
    const int head = min(static_cast<int>((8 - (gbase & 7)) & 7), HW);
    const int chunks = (HW - head) >> 3;
    const int vec_end = head + (chunks << 3);

    for (int f = lane; f < head; f += cohort_threads) {
      const int ih = f / W;
      const int iw = f - ih * W;
      const int tr = ih - h0 + PAD;
      if (tr >= 0 && tr < tile_rows) tile[tr * PW + iw + PAD] = xp[f];
    }
    for (int t = lane; t < chunks; t += cohort_threads) {
      const uint4 v = *reinterpret_cast<const uint4*>(xp + head + (t << 3));
      const __half* hv = reinterpret_cast<const __half*>(&v);
      int f = head + (t << 3);
#pragma unroll
      for (int k = 0; k < 8; ++k, ++f) {
        const int ih = f / W;
        const int iw = f - ih * W;
        const int tr = ih - h0 + PAD;
        if (tr >= 0 && tr < tile_rows) tile[tr * PW + iw + PAD] = hv[k];
      }
    }
    for (int f = vec_end + lane; f < HW; f += cohort_threads) {
      const int ih = f / W;
      const int iw = f - ih * W;
      const int tr = ih - h0 + PAD;
      if (tr >= 0 && tr < tile_rows) tile[tr * PW + iw + PAD] = xp[f];
    }
    const __half* wp = wh + plane % C * (KS * KS);
    for (int i = lane; i < KS * KS; i += cohort_threads) w2[i] = __half2half2(wp[i]);
  }
  __syncthreads();
  if (!live) return;

  const float bias_f = __half2float(bh[plane % C]);
  __half* op = out + static_cast<long long>(plane) * HW;
  const int pairs = W >> 1;
  const int strips = ((band_rows + ROWS - 1) / ROWS) * pairs;

  for (int s = lane; s < strips; s += cohort_threads) {
    const int sh = s / pairs;
    const int ow = (s - sh * pairs) << 1;
    const int lr0 = sh * ROWS;

    // One kernel row at a time in fp16, reduced across rows in fp32.  Summing all
    // 49 taps in fp16 costs about 1.6e-2 of absolute error on non-degenerate
    // BatchNorm statistics, over the harness's 1e-2 atol; reducing seven 7-tap
    // partials in fp32 keeps the halved arithmetic and brings the error back under
    // it, for one conversion and two adds per kernel row.
    float2 acc[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) acc[r] = make_float2(bias_f, bias_f);

#pragma unroll
    for (int i = 0; i < ROWS + KS - 1; ++i) {
      const int tr = lr0 + i;
      if (tr >= tile_rows) break;
      // Tile columns ow..ow+7: four aligned half2, seven tap pairs.
      const __half2* b = reinterpret_cast<const __half2*>(tile + tr * PW + ow);
      const __half2 b0 = b[0], b1 = b[1], b2 = b[2], b3 = b[3];
      __half2 p[KS];
      p[0] = b0;
      p[1] = shift_pair(b0, b1);
      p[2] = b1;
      p[3] = shift_pair(b1, b2);
      p[4] = b2;
      p[5] = shift_pair(b2, b3);
      p[6] = b3;
#pragma unroll
      for (int r = 0; r < ROWS; ++r) {
        const int k = i - r;
        if (k >= 0 && k < KS) {
          const __half2* wr = w2 + k * KS;
          __half2 part = __hmul2(p[0], wr[0]);
#pragma unroll
          for (int j = 1; j < KS; ++j) part = __hfma2(p[j], wr[j], part);
          const float2 pf = __half22float2(part);
          acc[r].x += pf.x;
          acc[r].y += pf.y;
        }
      }
    }

#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      const int lr = lr0 + r;
      if (lr < band_rows) {
        __half* dst = op + (h0 + lr) * W + ow;
        dst[0] = __float2half(silu(acc[r].x));
        dst[1] = __float2half(silu(acc[r].y));
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Diagnostic pair: same launch geometry, block size, shared footprint and output
// write as the shipped kernel, but with one half of the work removed each.
//
// PHASE 0 stages the tile exactly as the shipped kernel does, then writes a fixed
// value per output.  The write happens after the barrier and reads one staged
// element, so nothing can be dead-code eliminated.
// PHASE 1 skips staging and sweeps a tile whose contents it does not fill,
// touching the same shared addresses in the same order for the same count.
//
// Neither is correct arithmetic; they exist only to split the shipped kernel's
// time between filling the tile and sweeping it.  They are unreachable from
// forward() -- the module never selects them.
// ---------------------------------------------------------------------------
template <typename scalar_t, int ROWS, int PHASE>
__global__ void repvggdw_silu_probe_kernel(
    const scalar_t* __restrict__ x,
    const float* __restrict__ wf,
    const float* __restrict__ bf,
    scalar_t* __restrict__ out,
    const int C, const int H, const int W,
    const int HW, const int PW, const int rows_per_band, const int bands) {
  extern __shared__ float smem[];

  const int band = blockIdx.x % bands;
  const int plane = blockIdx.x / bands;
  const int c = plane % C;
  const long long base = static_cast<long long>(plane) * HW;
  const scalar_t* xp = x + base;

  const int h0 = band * rows_per_band;
  const int h1 = min(H, h0 + rows_per_band);
  const int band_rows = h1 - h0;
  const int tile_rows = band_rows + 2 * PAD;

  float* tile = smem;
  float* w = smem + tile_rows * PW;

  if (PHASE == 0) {
    const int TW = W + 2 * PAD;
    for (int i = threadIdx.x; i < tile_rows * TW; i += blockDim.x) {
      const int tr = i / TW;
      const int pw = i - tr * TW;
      const int ih = h0 - PAD + tr;
      const int iw = pw - PAD;
      tile[tr * PW + pw] = (ih >= 0 && ih < H && iw >= 0 && iw < W)
                               ? static_cast<float>(xp[ih * W + iw])
                               : 0.0f;
    }
  }
  for (int i = threadIdx.x; i < KS * KS; i += blockDim.x) {
    w[i] = wf[c * (KS * KS) + i];
  }
  const float b = bf[c];
  __syncthreads();

  scalar_t* op = out + base;
  const int strips = ((band_rows + ROWS - 1) / ROWS) * W;

  if (PHASE == 0) {
    // Fill measured; sweep removed.  One staged read per output keeps the fill
    // live without doing any of the stencil work.
    for (int s = threadIdx.x; s < strips; s += blockDim.x) {
      const int sh = s / W;
      const int ow = s - sh * W;
      const int lr0 = sh * ROWS;
#pragma unroll
      for (int r = 0; r < ROWS; ++r) {
        const int lr = lr0 + r;
        if (lr < band_rows) {
          op[(h0 + lr) * W + ow] =
              static_cast<scalar_t>(tile[(lr + PAD) * PW + ow + PAD] + b);
        }
      }
    }
  } else {
    // Sweep measured; fill removed.  Same shared addresses, same order, same
    // count as the shipped kernel.
    for (int s = threadIdx.x; s < strips; s += blockDim.x) {
      const int sh = s / W;
      const int ow = s - sh * W;
      const int lr0 = sh * ROWS;
      float acc[ROWS];
#pragma unroll
      for (int r = 0; r < ROWS; ++r) acc[r] = b;
#pragma unroll
      for (int i = 0; i < ROWS + KS - 1; ++i) {
        const int tr = lr0 + i;
        if (tr >= tile_rows) break;
        const float* row = tile + tr * PW + ow;
        float v[KS];
#pragma unroll
        for (int j = 0; j < KS; ++j) v[j] = row[j];
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
          const int k = i - r;
          if (k >= 0 && k < KS) {
            const float* wr = w + k * KS;
#pragma unroll
            for (int j = 0; j < KS; ++j) acc[r] = fmaf(v[j], wr[j], acc[r]);
          }
        }
      }
#pragma unroll
      for (int r = 0; r < ROWS; ++r) {
        const int lr = lr0 + r;
        if (lr < band_rows) {
          op[(h0 + lr) * W + ow] = static_cast<scalar_t>(silu(acc[r]));
        }
      }
    }
  }
}

// A fully unrolled 7x7 sweep is register-hungry, so a large requested block can
// exceed the per-block register file and fail to launch.  Cache each
// instantiation's own ceiling -- the kernel is a non-type template argument, so
// every kernel gets its own static -- and clamp the request to it.
template <auto KERNEL>
inline int max_threads() {
  static const int cap = []() {
    cudaFuncAttributes attr{};
    C10_CUDA_CHECK(cudaFuncGetAttributes(
        &attr, reinterpret_cast<const void*>(KERNEL)));
    return attr.maxThreadsPerBlock;
  }();
  return cap;
}

inline int round_block(int requested, int cap) {
  int t = requested < 32 ? 32 : (requested > cap ? cap : requested);
  t = (t / 32) * 32;
  return t < 32 ? 32 : t;
}

}  // namespace

// Only the three high-precision dtypes this block is ever benched or trained in;
// routing anything else belongs to the caller, which has a reference path.
#define REPVGGDW_DISPATCH(TYPE, NAME, ...)                                     \
  switch (TYPE) {                                                              \
    case at::kHalf: { using scalar_t = at::Half; __VA_ARGS__; break; }         \
    case at::kBFloat16: { using scalar_t = at::BFloat16; __VA_ARGS__; break; }  \
    case at::kFloat: { using scalar_t = float; __VA_ARGS__; break; }           \
    default: TORCH_CHECK(false, NAME ": unsupported dtype ", TYPE);            \
  }

at::Tensor repvggdw_silu(const at::Tensor& x, const at::Tensor& wf,
                         const at::Tensor& bf, const at::Tensor& wh,
                         const at::Tensor& bh, int64_t variant, int64_t block,
                         int64_t bands_arg) {
  TORCH_CHECK(x.is_cuda() && wf.is_cuda() && bf.is_cuda(),
              "repvggdw_silu: all tensors must be CUDA");
  TORCH_CHECK(x.dim() == 4, "repvggdw_silu: expected a 4D NCHW input");
  TORCH_CHECK(x.is_contiguous(), "repvggdw_silu: input must be contiguous");
  TORCH_CHECK(wf.scalar_type() == at::kFloat && bf.scalar_type() == at::kFloat,
              "repvggdw_silu: folded weight and bias must be float32");
  TORCH_CHECK(wf.is_contiguous() && bf.is_contiguous(),
              "repvggdw_silu: folded weight and bias must be contiguous");
  TORCH_CHECK(wf.dim() == 4 && wf.size(1) == 1 && wf.size(2) == KS && wf.size(3) == KS,
              "repvggdw_silu: folded weight must be [C,1,7,7]");
  TORCH_CHECK(wf.size(0) == x.size(1) && bf.numel() == x.size(1),
              "repvggdw_silu: channel count mismatch");
  TORCH_CHECK(wf.device() == x.device() && bf.device() == x.device(),
              "repvggdw_silu: tensors must share a device");

  const at::cuda::OptionalCUDAGuard device_guard(at::device_of(x));
  at::Tensor out = at::empty(x.sizes(), x.options());
  if (x.numel() == 0) return out;

  const int N = static_cast<int>(x.size(0));
  const int C = static_cast<int>(x.size(1));
  const int H = static_cast<int>(x.size(2));
  const int W = static_cast<int>(x.size(3));
  const int HW = H * W;
  // Bank behaviour of the sweep is set by the shared row pitch.  For the
  // one-output-per-thread kernel, threads walk output index ``idx`` and a tap
  // lands on bank ``(oh*PW + ow + const) % 32``; choosing ``PW % 32 == W % 32``
  // makes that exactly ``idx % 32``, so a warp's 32 consecutive idx values cover
  // all 32 banks once each.  The smallest such pitch that still holds the halo is
  // W + 32.  Variant 0 keeps the minimal pitch so the two can be compared.
  const int PW = (variant == 0) ? (W + 2 * PAD) : (W + 32);
  (void)PW;
  const int requested = static_cast<int>(block);
  const auto stream = at::cuda::getCurrentCUDAStream();
  const size_t smem_limit = static_cast<size_t>(
      at::cuda::getCurrentDeviceProperties()->sharedMemPerBlock);

  // The half2 kernels need an even plane width (columns are processed in pairs),
  // an even shared pitch (aligned half2 tile reads), a 16-byte-aligned plane base
  // so the vectorized fill's first load is legal, and an fp16 folded weight.  A
  // plane length that is not a multiple of eight is fine: the fill finishes the
  // remainder scalarly.
  const int planes_total = N * C;
  const bool h2_ok =
      (variant >= 6) && x.scalar_type() == at::kHalf && (W % 2 == 0) &&
      wh.defined() && bh.defined() &&
      wh.scalar_type() == at::kHalf && bh.scalar_type() == at::kHalf &&
      wh.is_contiguous() && bh.is_contiguous() && wh.numel() == C * KS * KS &&
      bh.numel() == C && wh.device() == x.device() && bh.device() == x.device() &&
      (reinterpret_cast<uintptr_t>(x.const_data_ptr()) % 16 == 0);
  // Variant 8 asks for the fp16 kernel where it applies and the fp32 row-blocked
  // kernel everywhere else, so one entry point covers every dtype and geometry
  // without a second Python-side predicate.  Variants 6 and 7 name the fp16
  // kernel explicitly and must therefore qualify.
  TORCH_CHECK(variant < 9 || x.scalar_type() == at::kHalf,
              "repvggdw_silu: the fill/sweep diagnostics are fp16 only");
  TORCH_CHECK((variant != 6 && variant != 7) || h2_ok,
              "repvggdw_silu: fp16 half2 kernel requested for an unsupported case");
  const int eff = (variant == 8) ? (h2_ok ? 6 : 4) : static_cast<int>(variant);

  if (eff == 6 || eff == 7) {
    int bands = static_cast<int>(bands_arg);
    bands = bands < 1 ? 1 : (bands > H ? H : bands);
    const int rows_per_band = (H + bands - 1) / bands;
    bands = (H + rows_per_band - 1) / rows_per_band;
    int pitch = W + 32;
    pitch += (pitch & 1);  // aligned half2 tile reads
    const int tile_halves = (rows_per_band + 2 * PAD) * pitch;
    const int planes_per_block = (eff == 7) ? 2 : 1;
    const size_t smem = static_cast<size_t>(planes_per_block) *
                        (static_cast<size_t>(tile_halves) * sizeof(__half) +
                         KS * KS * sizeof(__half2));
    TORCH_CHECK(smem <= smem_limit, "repvggdw_silu: fp16 tile needs ", smem,
                " bytes of shared memory, over the per-block limit");
    const int slots = planes_total * bands;
    const dim3 h2_grid((slots + planes_per_block - 1) / planes_per_block);
    const auto* xp = reinterpret_cast<const __half*>(x.const_data_ptr<at::Half>());
    const auto* whp = reinterpret_cast<const __half*>(wh.const_data_ptr<at::Half>());
    const auto* bhp = reinterpret_cast<const __half*>(bh.const_data_ptr<at::Half>());
    auto* outp = reinterpret_cast<__half*>(out.mutable_data_ptr<at::Half>());
    if (planes_per_block == 2) {
      const int threads = round_block(
          requested, max_threads<repvggdw_silu_h2_kernel<4, 2>>()) & ~63;
      repvggdw_silu_h2_kernel<4, 2><<<h2_grid, threads < 64 ? 64 : threads, smem,
                                      stream>>>(
          xp, whp, bhp, outp, C, H, W, HW, pitch, rows_per_band, bands,
          planes_total, tile_halves);
    } else {
      repvggdw_silu_h2_kernel<4, 1>
          <<<h2_grid,
             round_block(requested, max_threads<repvggdw_silu_h2_kernel<4, 1>>()),
             smem, stream>>>(
              xp, whp, bhp, outp, C, H, W, HW, pitch, rows_per_band, bands,
              planes_total, tile_halves);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
  }

  if (eff == 1) {
    const dim3 grid(static_cast<unsigned>(N) * static_cast<unsigned>(C));
    REPVGGDW_DISPATCH(x.scalar_type(), "repvggdw_silu",
        repvggdw_silu_direct_kernel<scalar_t>
            <<<grid,
               round_block(requested,
                           max_threads<repvggdw_silu_direct_kernel<scalar_t>>()),
               0, stream>>>(
            x.const_data_ptr<scalar_t>(), wf.const_data_ptr<float>(),
            bf.const_data_ptr<float>(), out.mutable_data_ptr<scalar_t>(),
            C, H, W, HW))
  } else if (eff == 0 || eff == 2) {
    const dim3 grid(static_cast<unsigned>(N) * static_cast<unsigned>(C));
    const int tile_elems = (H + 2 * PAD) * PW;
    const size_t smem = static_cast<size_t>(tile_elems + KS * KS) * sizeof(float);
    TORCH_CHECK(smem <= smem_limit, "repvggdw_silu: staged tile needs ", smem,
                " bytes of shared memory, over the per-block limit");
    REPVGGDW_DISPATCH(x.scalar_type(), "repvggdw_silu",
        repvggdw_silu_smem_kernel<scalar_t>
            <<<grid,
               round_block(requested,
                           max_threads<repvggdw_silu_smem_kernel<scalar_t>>()),
               smem, stream>>>(
            x.const_data_ptr<scalar_t>(), wf.const_data_ptr<float>(),
            bf.const_data_ptr<float>(), out.mutable_data_ptr<scalar_t>(),
            C, H, W, HW, PW, tile_elems))
  } else {
    // Split each plane into ``bands`` horizontal bands, each its own block.  The
    // halo means bands overlap on input rows but never on output rows, so no
    // block writes another's pixels.
    int bands = static_cast<int>(bands_arg);
    bands = bands < 1 ? 1 : (bands > H ? H : bands);
    const int rows_per_band = (H + bands - 1) / bands;
    // A band beyond the last row would be an empty block; drop those.
    bands = (H + rows_per_band - 1) / rows_per_band;
    const size_t smem =
        static_cast<size_t>((rows_per_band + 2 * PAD) * PW + KS * KS) * sizeof(float);
    TORCH_CHECK(smem <= smem_limit, "repvggdw_silu: staged band needs ", smem,
                " bytes of shared memory, over the per-block limit");
    const dim3 grid(static_cast<unsigned>(N) * static_cast<unsigned>(C) *
                    static_cast<unsigned>(bands));

#define REPVGGDW_LAUNCH_ROWS(KERNEL)                                          \
  REPVGGDW_DISPATCH(x.scalar_type(), "repvggdw_silu",                         \
      KERNEL<<<grid, round_block(requested, max_threads<KERNEL>()),           \
               smem, stream>>>(                                               \
          x.const_data_ptr<scalar_t>(), wf.const_data_ptr<float>(),           \
          bf.const_data_ptr<float>(), out.mutable_data_ptr<scalar_t>(),       \
          C, H, W, HW, PW, rows_per_band, bands))

    switch (eff) {
      case 3: REPVGGDW_LAUNCH_ROWS((repvggdw_silu_rows_kernel<scalar_t, 2>)); break;
      case 5: REPVGGDW_LAUNCH_ROWS((repvggdw_silu_rows_kernel<scalar_t, 5>)); break;
      // 9 and 10 are the fill-only / sweep-only diagnostics; they are not
      // arithmetically correct and forward() never selects them.
      case 9: REPVGGDW_LAUNCH_ROWS((repvggdw_silu_probe_kernel<scalar_t, 4, 0>)); break;
      case 10: REPVGGDW_LAUNCH_ROWS((repvggdw_silu_probe_kernel<scalar_t, 4, 1>)); break;
      default: REPVGGDW_LAUNCH_ROWS((repvggdw_silu_rows_kernel<scalar_t, 4>)); break;
    }
#undef REPVGGDW_LAUNCH_ROWS
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
"""

_CPP_SOURCE = r"""
#include <ATen/ATen.h>

at::Tensor repvggdw_silu(const at::Tensor& x, const at::Tensor& wf,
                         const at::Tensor& bf, const at::Tensor& wh,
                         const at::Tensor& bh, int64_t variant, int64_t block,
                         int64_t bands);
"""

# Which kernel and block size to use.  Both are measured choices, swept under the
# benchmark's own timing recipe; see docs/tuning.md for the numbers.  Four
# vertically blocked output rows per thread at 128 threads per block is the best
# configuration measured on both captured shapes.
_STAGED_TIGHT = 0   # staged, minimal shared footprint, two-way bank conflicts
_DIRECT = 1         # no staging, reads through L1/L2
_STAGED_PADDED = 2  # staged at a conflict-free row pitch, one output per pass
_ROWS_2 = 3         # staged, 2 output rows per thread
_ROWS_4 = 4         # staged, 4 output rows per thread
_ROWS_5 = 5         # staged, 5 output rows per thread
_H2 = 6             # fp16 only: half2 column pairs, 4 output rows per thread
_H2_2PLANE = 7      # same, two planes per block
_AUTO = 8           # the fp16 kernel where it applies, rows-4 everywhere else
# Diagnostics, not implementations: same launch shape as _ROWS_4 with one half of
# the work removed each, to split its time between staging and sweeping.  Their
# output is deliberately not the operator's, and forward() never selects them.
_PROBE_FILL_ONLY = 9
_PROBE_SWEEP_ONLY = 10
# The fp32 row-blocked kernel ships.  The fp16 half2 kernel (_H2 / _H2_2PLANE, and
# _AUTO which selects it where eligible) is correct, sanitizer-clean and retained as
# a selectable variant, but it is not adopted: paired, interleaved A/Bs
# (profile/repvggdw_variant_isolation/analysis/paired_ab.json) tie with this kernel
# in the uncontended regime the official runs measure -- 15.36 vs 15.36 us at N=4
# with the fp16 arm ahead in 3 of 9 trials, 13.31 vs 13.31 at N=1 ahead in 5 of 9 --
# and three official runs each way report the same 0.0154 / 0.0133 ms.  It does win
# 9 of 9 when the device is loaded, which is a real effect but not the one the
# benchmark scores.  On a tie the fp32 kernel is also the better choice: it rounds
# less (max_abs 9.8e-4 against 1.3e-3) and needs no second dispatch.
_FAST_VARIANT = _ROWS_4
_FAST_BLOCK = 128
_FAST_BANDS = 1  # horizontal bands per plane; 1 == one block per plane
# Splitting a plane across several blocks was swept over {1, 2, 4, 5, 10, 20} to
# attack the "grid too small" signal ncu reports (0.17 waves per SM at N=1).  It
# bought nothing measurable at either shape, so one block per plane stands.

# Workspace-local build directory, named for this operator so it cannot collide
# with another operator's extension in a shared cache.
_EXT_NAME = "fk_yolov10_repvggdw_fused"
_BUILD_DIR = Path(
    os.environ.get(
        "FK_REPVGGDW_BUILD_DIR",
        str(Path(__file__).resolve().parents[2] / ".torch_extensions" / _EXT_NAME),
    )
)


def _build_extension():
    """Compile the fused kernel, or load it from the workspace build cache."""
    from torch.utils.cpp_extension import load_inline

    _BUILD_DIR.mkdir(parents=True, exist_ok=True)
    arch_var = "TORCH_CUDA_ARCH_LIST"
    prev_arch = os.environ.get(arch_var)
    if torch.cuda.is_available():
        # Compiling for every architecture in the ambient list costs minutes and
        # buys nothing: the kernel only ever runs on the device in front of us.
        major, minor = torch.cuda.get_device_capability()
        os.environ[arch_var] = f"{major}.{minor}"
    try:
        return load_inline(
            name=_EXT_NAME,
            cpp_sources=_CPP_SOURCE,
            cuda_sources=_CUDA_SOURCE,
            functions=["repvggdw_silu"],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "-lineinfo"],
            build_directory=str(_BUILD_DIR),
            verbose=False,
        )
    finally:
        if prev_arch is None:
            os.environ.pop(arch_var, None)
        else:
            os.environ[arch_var] = prev_arch


_EXT = None
_EXT_ERROR: str | None = None

if os.environ.get("FK_REPVGGDW_DISABLE_EXT") == "1":
    _EXT_ERROR = "fused extension disabled by FK_REPVGGDW_DISABLE_EXT"
    if os.environ.get("FK_REPVGGDW_REQUIRE_EXT") == "1":
        raise RuntimeError(_EXT_ERROR)
    warnings.warn(f"YOLORepVGGDW: {_EXT_ERROR}; using the reference path", stacklevel=2)
else:
    try:
        _EXT = _build_extension()
    except Exception as exc:  # noqa: BLE001 - a broken toolchain must not kill the import
        _EXT_ERROR = f"{type(exc).__name__}: {exc}"
        print(
            f"YOLORepVGGDW: fused extension unavailable ({_EXT_ERROR}); "
            "falling back to the reference path",
            file=sys.stderr,
            flush=True,
        )
        # Falling back keeps the answer right, but a *recorded* latency produced
        # by the reference path would be a misattributed measurement.  Runs whose
        # numbers get written down set this so a build failure is fatal instead.
        if os.environ.get("FK_REPVGGDW_REQUIRE_EXT") == "1":
            raise

_FUSED_OP = None if _EXT is None else _EXT.repvggdw_silu


def extension_available() -> bool:
    """Whether the fused kernel is loaded.  Measurement runs assert on this."""
    return _EXT is not None


def extension_error() -> str | None:
    """Why the fused kernel is unavailable, or ``None`` if it loaded."""
    return _EXT_ERROR


_FAST_DTYPES = frozenset((torch.float16, torch.bfloat16, torch.float32))


class _Fold:
    """A folded 7x7 weight and bias, plus everything that would invalidate them.

    ``wf is None`` records that the fused path cannot serve this module at all,
    so the decision is reached once rather than re-derived on every call.

    Staleness is checked from three lists, all reachable without
    ``nn.Module.__getattr__``:

    - ``slots``: ``(container, key, obj)`` over the owning modules' ``_modules`` /
      ``_parameters`` / ``_buffers`` dicts.  Those dicts live in the module's own
      ``__dict__`` and are mutated in place by ``setattr``, ``_apply`` and
      ``load_state_dict``, so re-reading them catches a replaced submodule, a
      replaced parameter or buffer, and a bias appearing where there was none
      (the entry is watched even when it holds ``None``).
    - ``tensors``: ``(tensor, version, data_ptr)``.  The version counter catches
      ordinary in-place writes; the storage pointer catches a rebind through
      ``.data`` and a leaf ``_apply``, neither of which touches the counter.
    - ``scalars``: ``(obj, attr, value)`` for the plain attributes the fold's
      validity depends on -- the BatchNorm epsilons and per-module ``training``
      flags, each branch's fused flag, and the convolution geometry.

    Measurement says all of this is free: under the benchmark's timing recipe the
    guarded module and a guard-free wrapper around the same kernel are
    indistinguishable, because the L2 flush enqueued before the start event lets
    the CPU run ahead of the GPU.  See docs/tuning.md.

    One path remains uncovered by construction: an in-place write *through*
    ``.data`` into the same storage (``p.data.copy_(t)``, or an in-place op on
    ``p.data``) changes the values while leaving object identity, the parameter's
    version counter and the storage pointer all untouched.  Mutate parameters
    directly (``p.copy_(t)``) or go through ``load_state_dict`` instead.
    """

    __slots__ = ("wf", "bf", "wh", "bh", "device", "channels", "max_tile",
                 "slots", "tensors", "scalars")

    def __init__(self, wf=None, bf=None, slots=(), tensors=(), scalars=(), max_tile=0):
        self.wf = wf
        self.bf = bf
        # fp16 copies for the half2 kernel, derived from the fp32 master so they
        # share its lifetime and its invalidation exactly.
        self.wh = None if wf is None else wf.half().contiguous()
        self.bh = None if bf is None else bf.half().contiguous()
        self.device = None if wf is None else wf.device
        self.channels = -1 if wf is None else wf.size(0)
        self.max_tile = max_tile
        self.slots = slots
        self.tensors = tensors
        self.scalars = scalars

    def current(self) -> bool:
        for container, key, obj in self.slots:
            if container.get(key) is not obj:
                return False
        for tensor, version, ptr in self.tensors:
            if tensor._version != version or tensor.data_ptr() != ptr:
                return False
        for obj, attr, value in self.scalars:
            if getattr(obj, attr) != value:
                return False
        return True


def _watch(container, key, slots: list, tensors: list) -> None:
    """Watch one ``_parameters`` / ``_buffers`` entry, present or not."""
    obj = container.get(key)
    slots.append((container, key, obj))
    if obj is not None:
        tensors.append((obj, obj._version, obj.data_ptr()))


def _branch_affine(branch: YOLOConv):
    """The branch reduced to fp32 ``(weight, bias, slots, tensors, scalars)``.

    Returns ``None`` when the branch is not a unit-stride, shape-preserving
    depthwise convolution whose BatchNorm (if any) can be folded into it.
    """
    conv = branch._modules.get("conv")
    if conv is None:
        return None
    weight = conv._parameters.get("weight")
    if weight is None or weight.dim() != 4:
        return None
    channels = weight.size(0)
    kh, kw = weight.size(2), weight.size(3)
    if (
        weight.size(1) != 1
        or conv.groups != channels
        or kh % 2 == 0
        or kw % 2 == 0
        or tuple(conv.stride) != (1, 1)
        or tuple(conv.dilation) != (1, 1)
        or tuple(conv.padding) != (kh // 2, kw // 2)
    ):
        return None

    w = weight.detach().float()
    bias = conv._parameters.get("bias")
    b = (
        bias.detach().float()
        if bias is not None
        else torch.zeros(channels, dtype=torch.float32, device=w.device)
    )

    # The branch applies its own activation after the convolution and BatchNorm.
    # Folding two branches into one pre-activation sum is only valid while both of
    # those are the identity; anything else has to run the reference forward.
    branch_act = branch._modules.get("act")
    if type(branch_act) is not nn.Identity:
        return None

    slots: list = [
        (branch._modules, "conv", conv),
        (branch._modules, "act", branch_act),
    ]
    tensors: list = []
    _watch(conv._parameters, "weight", slots, tensors)
    _watch(conv._parameters, "bias", slots, tensors)
    # A submodule-level fuse() rewrites the weight through .data, adds a bias and
    # deletes the BatchNorm, so the fused flag and the bn slot both have to be
    # watched or that path leaves the fold stale.
    scalars: list = [
        (branch, "_is_fused", branch._is_fused),
        (conv, "groups", conv.groups),
        (conv, "stride", conv.stride),
        (conv, "padding", conv.padding),
        (conv, "dilation", conv.dilation),
    ]

    bn = branch._modules.get("bn")
    slots.append((branch._modules, "bn", bn))
    if bn is not None and not branch._is_fused:
        # ``BatchNorm2d`` passes ``training or not track_running_stats`` to
        # ``F.batch_norm``, so either flag being set the wrong way makes it
        # normalize by *batch* statistics -- which a running-statistics fold does
        # not model, even in eval.  Detecting a change is not enough: the rebuild
        # has to decline too, or it just produces the same wrong fold again.
        if bn.training or not bn.track_running_stats:
            return None
        var = bn._buffers.get("running_var")
        mean = bn._buffers.get("running_mean")
        if var is None or mean is None:
            return None  # without running statistics BN needs the batch, not a fold
        scale = torch.rsqrt(var.detach().float() + bn.eps)
        bn_weight = bn._parameters.get("weight")
        bn_bias = bn._parameters.get("bias")
        if bn_weight is not None:
            scale = scale * bn_weight.detach().float()
        shift = bn_bias.detach().float() if bn_bias is not None else torch.zeros_like(scale)
        shift = shift - scale * mean.detach().float()
        w = w * scale.view(-1, 1, 1, 1)
        b = b * scale + shift
        for name in ("weight", "bias"):
            _watch(bn._parameters, name, slots, tensors)
        for name in ("running_mean", "running_var"):
            _watch(bn._buffers, name, slots, tensors)
        scalars += [
            (bn, "eps", bn.eps),
            (bn, "training", bn.training),
            (bn, "track_running_stats", bn.track_running_stats),
        ]

    return w, b, slots, tensors, scalars


class YOLORepVGGDW(nn.Module):
    """RepVGG depthwise block whose eval-mode forward is a single fused kernel.

    The submodule tree matches the reference block exactly, so
    ``load_state_dict(reference.state_dict(), strict=False)`` shares weights with
    no missing or unexpected keys.
    """

    def __init__(self, ed: int):
        super().__init__()
        self.conv = YOLOConv(ed, ed, 7, 1, 3, g=ed, act=False)
        self.conv1 = YOLOConv(ed, ed, 3, 1, 1, g=ed, act=False)
        self.act = SiLU()
        self._pad = Pad()
        self._is_fused = False
        # Free invalidation for what a version counter cannot see:
        # ``load_state_dict(assign=True)`` rebinds parameter objects, and a
        # plain ``load_state_dict`` may report incompatible keys.
        self.register_load_state_dict_post_hook(_drop_fold)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fold = self.__dict__.get("_fold")
        if fold is None or not fold.current():
            fold = self._refresh_fold()
        wf = fold.wf
        if wf is not None and (
            x.dim() == 4
            and x.dtype in _FAST_DTYPES
            and x.size(1) == fold.channels
            and x.device == fold.device
            and x.is_contiguous()
            and (x.size(2) + 6) * (x.size(3) + 32) <= fold.max_tile
            # The launch is a plain function with no autograd registration, so a
            # fused output would be a leaf where the reference returns a connected
            # tensor.  Inference callers are under no_grad and pay one C-level
            # predicate for this.
            and not (torch.is_grad_enabled() and _needs_grad(x, self))
        ):
            return _FUSED_OP(x, wf, fold.bf, fold.wh, fold.bh,
                             _FAST_VARIANT, _FAST_BLOCK, _FAST_BANDS)
        return self._reference_forward(x)

    def _reference_forward(self, x: torch.Tensor) -> torch.Tensor:
        """The reference block's own algebra, for whatever the kernel cannot run."""
        if self._is_fused:
            return self.act(self.conv(x))
        return self.act(self.conv(x) + self.conv1(x))

    @torch.no_grad()
    def _refresh_fold(self) -> _Fold:
        """Rebuild the cached fold, or cache the fact that there is no fast path.

        Built lazily rather than in ``__init__`` because the benchmark harness
        shares weights *after* construction, so a fold computed at construction
        time would describe weights nobody goes on to use.
        """
        fold = self._compute_fold()
        self.__dict__["_fold"] = fold
        return fold

    def _compute_fold(self) -> _Fold:
        if _EXT is None or self.training:
            # Training-mode BatchNorm normalizes by batch statistics and advances
            # ``num_batches_tracked``; a running-statistics fold is not that.
            return _Fold()
        # The kernel bakes in SiLU, so a different activation is not this fold.
        # Exact type, not isinstance: a SiLU subclass may override forward and
        # would otherwise silently receive the built-in activation.
        act = self._modules.get("act")
        if type(act) is not SiLU:
            return _Fold()

        primary = _branch_affine(self.conv)
        if primary is None:
            return _Fold()
        wf, bf, slots, tensors, scalars = primary
        if wf.size(2) != 7 or wf.size(3) != 7 or not wf.is_cuda:
            return _Fold()

        residual = self._modules.get("conv1")
        if self._is_fused:
            if residual is not None:
                return _Fold()  # a fused block should have no residual branch left
        else:
            if residual is None:
                return _Fold()
            secondary = _branch_affine(residual)
            if secondary is None:
                return _Fold()
            w3, b3, slots3, tensors3, scalars3 = secondary
            kh, kw = w3.size(2), w3.size(3)
            if kh > 7 or kw > 7 or w3.size(0) != wf.size(0):
                return _Fold()
            top, left = (7 - kh) // 2, (7 - kw) // 2
            wf = wf.clone()
            wf[:, :, top : top + kh, left : left + kw] += w3
            bf = bf + b3
            slots = slots + slots3
            tensors = tensors + tensors3
            scalars = scalars + scalars3

        # Watch the block's own topology too, so replacing a branch or the
        # activation, or flipping the fused flag by hand, drops the fold.
        slots += [
            (self._modules, "conv", self._modules.get("conv")),
            (self._modules, "conv1", residual),
            (self._modules, "act", act),
        ]
        scalars += [(self, "_is_fused", self._is_fused), (self, "training", False)]

        # Largest staged plane that fits, in elements.  Measured against the
        # widest pitch any variant may ask for, so the envelope does not depend
        # on which one is selected.
        smem_floats = torch.cuda.get_device_properties(wf.device).shared_memory_per_block // 4
        return _Fold(wf.contiguous(), bf.contiguous(), tuple(slots), tuple(tensors),
                     tuple(scalars), smem_floats - 49)

    def train(self, mode: bool = True):
        # Batch statistics and ``num_batches_tracked`` move under training, and
        # so do the running statistics the fold was derived from.
        self.__dict__.pop("_fold", None)
        return super().train(mode)

    def _apply(self, *args, **kwargs):
        # ``_apply`` replaces buffer objects outright and rewrites parameter
        # storage, neither of which bumps a parameter's version counter.
        self.__dict__.pop("_fold", None)
        return super()._apply(*args, **kwargs)

    @torch.no_grad()
    def fuse(self):
        if self._is_fused:
            return self
        self.__dict__.pop("_fold", None)
        self.conv.fuse()
        self.conv1.fuse()
        final_conv_w = self.conv.conv.weight.data + self._pad(
            self.conv1.conv.weight.data, [2, 2, 2, 2]
        )
        final_conv_b = self.conv.conv.bias.data + self.conv1.conv.bias.data
        self.conv.conv.weight.data.copy_(final_conv_w)
        self.conv.conv.bias.data.copy_(final_conv_b)
        delattr(self, "conv1")
        self._is_fused = True
        return self


def _needs_grad(x: torch.Tensor, module: nn.Module) -> bool:
    """Whether a backward pass through this call could be wanted."""
    if x.requires_grad:
        return True
    return any(p.requires_grad for p in module.parameters(recurse=True))


def _drop_fold(module: nn.Module, incompatible_keys) -> None:  # noqa: ARG001
    module.__dict__.pop("_fold", None)
