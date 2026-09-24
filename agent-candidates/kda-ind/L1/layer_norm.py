"""Fused LayerNorm for B200 (sm_100) with the same contract as ``baseline.py``.

The baseline calls ``F.layer_norm``, and when ``promote_fp32`` is set it brackets
that call with an explicit fp32 round trip. On the small captured shapes this
costs three kernels and two fp32 temporaries, where the kernel *count* -- not the
arithmetic -- is the whole cost. On the wide ``[*, 4608]`` shapes the single ATen
kernel is issue-bound on narrow memory instructions rather than bandwidth-bound:
an NCU record of it (``profile/p1-baseline-torch-ln/``) shows 51.26 M L1TEX
sectors moving 298 MB of useful data, with only 12.8 of every 32 bytes per sector
actually used.

Both are replaced by one kernel that reads each row once with 128-bit accesses,
keeps the row packed in registers across both reduction passes so the variance
pass costs no global traffic, accumulates in fp32, and rounds once on the store.

``promote_fp32`` therefore does not select a code path. ATen's low-precision
LayerNorm already accumulates in fp32 with the affine parameters upcast
in-register, and bf16/fp16 -> fp32 is exact, so one native-dtype kernel with fp32
accumulation reproduces both modes. The baseline's ``_w32``/``_b32`` parameter
cache has no analogue here and is deliberately not ported.

Anything the kernel does not cover -- another dtype, a non-contiguous or
misaligned input, a row width that is not a multiple of the 16-byte vector, a CPU
tensor, a call that needs gradients -- reproduces the baseline *formula* exactly,
fp32 promotion included, rather than merely landing inside tolerance.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Unique to this file so a second import under a different module name cannot
# double-register the operator; registration happens once at ``.so`` load and
# Python's module cache makes any later import a no-op.
_LIBRARY_NAME = "fk_ln_cand"

# The mapping for the widest captured row is exposed as two ``-D`` macros with
# fixed defaults, so ``ab_row_mapping.py`` can A/B it by recompiling rather than
# by adding a runtime switch here. 192x3 was measured fastest of the three
# mappings that divide 576 vectors exactly (192x3 / 288x2 / 576x1).
_CUDA_SOURCE = r"""
#include <ATen/ATen.h>
#include <ATen/TensorSubclassLikeUtils.h>
#include <ATen/autocast_mode.h>
#include <ATen/core/grad_mode.h>
#include <torch/library.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <optional>

#ifndef FK_LN_ROW_BLOCK
#define FK_LN_ROW_BLOCK 192
#endif
#ifndef FK_LN_ROW_VPT
#define FK_LN_ROW_VPT 3
#endif
// Rows sharing one CTA on the warp-per-row path. Measured a wash across 1/2/4/8
// on every scored case, so this is set for the narrow-and-tall captured shapes
// (N=16 over 49152 rows), where one CTA per row would be all launch overhead.
#ifndef FK_LN_WARP_ROWS
#define FK_LN_WARP_ROWS 4
#endif
// Widest row, in 16-byte vectors, still given to the warp-per-row kernel. 32
// keeps that path to one vector per lane; 64 extends it to two, which is the
// one-warp mapping for N=384; 0 disables it so the narrow rows go to the
// block-per-row kernel instead. Every value is a real dispatch difference --
// ``ab_row_mapping.py`` refuses to pass an override this source does not consume.
#ifndef FK_LN_WARP_MAX_VECS
#define FK_LN_WARP_MAX_VECS 32
#endif
// Block size for the narrowest block-per-row rung.
#ifndef FK_LN_NARROW_BLOCK
#define FK_LN_NARROW_BLOCK 64
#endif

namespace {

constexpr int kWarpSize = 32;
constexpr unsigned kFullMask = 0xffffffffu;

constexpr int kWarpRows = FK_LN_WARP_ROWS;
constexpr int kWarpMaxVecs = FK_LN_WARP_MAX_VECS;
constexpr int kNarrowBlock = FK_LN_NARROW_BLOCK;
constexpr int kRowBlock = FK_LN_ROW_BLOCK;
constexpr int kRowVpt = FK_LN_ROW_VPT;
constexpr int kTunedVecs = kRowBlock * kRowVpt;
// Widest row the block-per-row ladder covers, in 16-byte vectors. Anything
// wider than this (and not exactly kTunedVecs) takes the exact fallback.
constexpr int kMaxLadderVecs = 512;
static_assert(kWarpMaxVecs <= 2 * kWarpSize,
              "the warp-per-row kernel holds at most two vectors per lane");

// Row widths that are not a multiple of the vector go to a scalar kernel
// instead. Two rungs cover every odd width in the captured envelope (267, 833).
constexpr int kScalarBlock = 256;
constexpr int kMaxScalarElems = 4 * kScalarBlock;

// ---------------------------------------------------------------------------
// 16 bytes is the widest single global access the SM offers, so the packed
// element count follows from the element size. Rows stay in registers in this
// packed form -- 4 registers per vector instead of the 8 an fp32 unpack would
// need, which is what keeps CTA residency up on the 4608-wide rows.
// ---------------------------------------------------------------------------
template <typename T>
struct Packed;

template <>
struct Packed<__nv_bfloat16> {
  using Elem = __nv_bfloat16;
  __device__ __forceinline__ static float scalar_to_float(Elem v) {
    return __bfloat162float(v);
  }
  __device__ __forceinline__ static Elem scalar_from_float(float v) {
    return __float2bfloat16_rn(v);
  }
  static constexpr int kElems = 8;
  __device__ __forceinline__ static void to_float(const uint4& v, float* out) {
    const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float2 f = __bfloat1622float2(p[j]);
      out[2 * j] = f.x;
      out[2 * j + 1] = f.y;
    }
  }
  __device__ __forceinline__ static uint4 from_float(const float* in) {
    uint4 v;
    __nv_bfloat162* p = reinterpret_cast<__nv_bfloat162*>(&v);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      p[j] = __floats2bfloat162_rn(in[2 * j], in[2 * j + 1]);
    }
    return v;
  }
};

template <>
struct Packed<__half> {
  using Elem = __half;
  __device__ __forceinline__ static float scalar_to_float(Elem v) {
    return __half2float(v);
  }
  __device__ __forceinline__ static Elem scalar_from_float(float v) {
    return __float2half_rn(v);
  }
  static constexpr int kElems = 8;
  __device__ __forceinline__ static void to_float(const uint4& v, float* out) {
    const __half2* p = reinterpret_cast<const __half2*>(&v);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float2 f = __half22float2(p[j]);
      out[2 * j] = f.x;
      out[2 * j + 1] = f.y;
    }
  }
  __device__ __forceinline__ static uint4 from_float(const float* in) {
    uint4 v;
    __half2* p = reinterpret_cast<__half2*>(&v);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      p[j] = __floats2half2_rn(in[2 * j], in[2 * j + 1]);
    }
    return v;
  }
};

template <>
struct Packed<float> {
  using Elem = float;
  __device__ __forceinline__ static float scalar_to_float(Elem v) {
    return v;
  }
  __device__ __forceinline__ static Elem scalar_from_float(float v) {
    return v;
  }
  static constexpr int kElems = 4;
  __device__ __forceinline__ static void to_float(const uint4& v, float* out) {
    const float* p = reinterpret_cast<const float*>(&v);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      out[j] = p[j];
    }
  }
  __device__ __forceinline__ static uint4 from_float(const float* in) {
    uint4 v;
    float* p = reinterpret_cast<float*>(&v);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      p[j] = in[j];
    }
    return v;
  }
};

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    v += __shfl_xor_sync(kFullMask, v, offset);
  }
  return v;
}

// Sum across a whole CTA. ``stage`` needs kWarps + 1 floats; the mean and the
// variance reduction are handed disjoint regions so neither has to guard the
// other's broadcast slot with an extra __syncthreads.
template <int kBlockThreads>
__device__ __forceinline__ float block_reduce_sum(float v, float* stage) {
  constexpr int kWarps = kBlockThreads / kWarpSize;
  v = warp_reduce_sum(v);
  if constexpr (kWarps == 1) {
    return v;
  } else {
    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp = threadIdx.x >> 5;
    if (lane == 0) {
      stage[warp] = v;
    }
    __syncthreads();
    // Every warp repeats the final reduction over the same kWarps values, with
    // the lanes past kWarps reading a zero. Cheaper than the divergence of
    // restricting it to warp 0, and it keeps the broadcast a single store.
    float t = (threadIdx.x < kWarps) ? stage[threadIdx.x] : 0.0f;
    t = warp_reduce_sum(t);
    if (threadIdx.x == 0) {
      stage[kWarps] = t;
    }
    __syncthreads();
    return stage[kWarps];
  }
}

// Sum of one packed vector, and its sum of squared deviations from the mean.
// Both unpack from the register copy, so neither touches global memory.
template <typename T>
__device__ __forceinline__ float vector_sum(const uint4& packed) {
  float e[Packed<T>::kElems];
  Packed<T>::to_float(packed, e);
  float s = 0.0f;
#pragma unroll
  for (int j = 0; j < Packed<T>::kElems; ++j) {
    s += e[j];
  }
  return s;
}

template <typename T>
__device__ __forceinline__ float vector_sq_dev(const uint4& packed, float mean) {
  float e[Packed<T>::kElems];
  Packed<T>::to_float(packed, e);
  float s = 0.0f;
#pragma unroll
  for (int j = 0; j < Packed<T>::kElems; ++j) {
    const float d = e[j] - mean;
    s += d * d;
  }
  return s;
}

// Apply the normalisation and the affine transform in fp32, then round once.
// ``w``/``b`` are tested against nullptr rather than templated: the branch is
// grid-uniform, and reading each parameter vector inside its own guard keeps the
// unpacked floats out of the register budget when the parameter is absent.
template <typename T>
__device__ __forceinline__ void write_normalized(
    const uint4& packed, uint4* __restrict__ out,
    const uint4* __restrict__ w, const uint4* __restrict__ b,
    int idx, float mean, float rstd) {
  constexpr int kElems = Packed<T>::kElems;
  float e[kElems];
  Packed<T>::to_float(packed, e);
#pragma unroll
  for (int j = 0; j < kElems; ++j) {
    e[j] = (e[j] - mean) * rstd;
  }
  if (w != nullptr) {
    // Copy into a local before unpacking. Handing ``to_float`` the global
    // location directly makes it take that address and read the four halves
    // separately: SASS showed 4x LDG.E (32-bit) per parameter vector instead of
    // one LDG.E.128, which put nine load instructions on the L1TEX pipe per
    // vector group where three would do.
    const uint4 wp = w[idx];
    float wf[kElems];
    Packed<T>::to_float(wp, wf);
#pragma unroll
    for (int j = 0; j < kElems; ++j) {
      e[j] *= wf[j];
    }
  }
  if (b != nullptr) {
    const uint4 bp = b[idx];
    float bf[kElems];
    Packed<T>::to_float(bp, bf);
#pragma unroll
    for (int j = 0; j < kElems; ++j) {
      e[j] += bf[j];
    }
  }
  out[idx] = Packed<T>::from_float(e);
}

// ---------------------------------------------------------------------------
// Warp-per-row: one warp owns a row, holding kVecsPerLane 16-byte vectors per
// lane. The reduction is shuffle-only -- no shared memory, no __syncthreads --
// and several rows share a CTA so a short row does not spend a whole CTA launch
// on one warp's work.
// ---------------------------------------------------------------------------
template <typename T, int kVecsPerLane, int kRowsPerBlock>
__global__ void __launch_bounds__(kRowsPerBlock * kWarpSize)
layer_norm_warp_kernel(const T* __restrict__ x, T* __restrict__ y,
                       const T* __restrict__ weight, const T* __restrict__ bias,
                       int64_t rows, int vecs_per_row, float inv_n, float eps) {
  const int lane = threadIdx.x & (kWarpSize - 1);
  // Uniform across the warp, so an out-of-range warp exits fully converged and
  // the shuffles below always see all 32 lanes.
  const int64_t row =
      static_cast<int64_t>(blockIdx.x) * kRowsPerBlock + (threadIdx.x >> 5);
  if (row >= rows) {
    return;
  }

  const uint4* __restrict__ xv =
      reinterpret_cast<const uint4*>(x) + row * vecs_per_row;
  uint4* __restrict__ yv = reinterpret_cast<uint4*>(y) + row * vecs_per_row;
  const uint4* __restrict__ wv = reinterpret_cast<const uint4*>(weight);
  const uint4* __restrict__ bv = reinterpret_cast<const uint4*>(bias);

  uint4 packed[kVecsPerLane];
  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < kVecsPerLane; ++i) {
    const int idx = lane + i * kWarpSize;
    if (idx < vecs_per_row) {
      packed[i] = xv[idx];
      sum += vector_sum<T>(packed[i]);
    }
  }
  const float mean = warp_reduce_sum(sum) * inv_n;

  float sq = 0.0f;
#pragma unroll
  for (int i = 0; i < kVecsPerLane; ++i) {
    const int idx = lane + i * kWarpSize;
    if (idx < vecs_per_row) {
      sq += vector_sq_dev<T>(packed[i], mean);
    }
  }
  const float rstd = rsqrtf(warp_reduce_sum(sq) * inv_n + eps);

#pragma unroll
  for (int i = 0; i < kVecsPerLane; ++i) {
    const int idx = lane + i * kWarpSize;
    if (idx < vecs_per_row) {
      write_normalized<T>(packed[i], yv, wv, bv, idx, mean, rstd);
    }
  }
}

// ---------------------------------------------------------------------------
// Block-per-row for rows too wide for one warp. Consecutive threads take
// consecutive vectors within each of the kVecsPerThread passes, so every pass is
// a fully coalesced 128-bit access.
// ---------------------------------------------------------------------------
template <typename T, int kBlockThreads, int kVecsPerThread>
__global__ void __launch_bounds__(kBlockThreads)
layer_norm_block_kernel(const T* __restrict__ x, T* __restrict__ y,
                        const T* __restrict__ weight,
                        const T* __restrict__ bias,
                        int vecs_per_row, float inv_n, float eps) {
  constexpr int kWarps = kBlockThreads / kWarpSize;
  __shared__ float stage[2 * (kWarps + 1)];

  const int64_t row = blockIdx.x;
  const uint4* __restrict__ xv =
      reinterpret_cast<const uint4*>(x) + row * vecs_per_row;
  uint4* __restrict__ yv = reinterpret_cast<uint4*>(y) + row * vecs_per_row;
  const uint4* __restrict__ wv = reinterpret_cast<const uint4*>(weight);
  const uint4* __restrict__ bv = reinterpret_cast<const uint4*>(bias);

  uint4 packed[kVecsPerThread];
  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < kVecsPerThread; ++i) {
    const int idx = threadIdx.x + i * kBlockThreads;
    if (idx < vecs_per_row) {
      packed[i] = xv[idx];
      sum += vector_sum<T>(packed[i]);
    }
  }
  const float mean = block_reduce_sum<kBlockThreads>(sum, stage) * inv_n;

  float sq = 0.0f;
#pragma unroll
  for (int i = 0; i < kVecsPerThread; ++i) {
    const int idx = threadIdx.x + i * kBlockThreads;
    if (idx < vecs_per_row) {
      sq += vector_sq_dev<T>(packed[i], mean);
    }
  }
  const float var = block_reduce_sum<kBlockThreads>(sq, stage + kWarps + 1);
  const float rstd = rsqrtf(var * inv_n + eps);

#pragma unroll
  for (int i = 0; i < kVecsPerThread; ++i) {
    const int idx = threadIdx.x + i * kBlockThreads;
    if (idx < vecs_per_row) {
      write_normalized<T>(packed[i], yv, wv, bv, idx, mean, rstd);
    }
  }
}

// ---------------------------------------------------------------------------
// Row widths that are not a multiple of the 16-byte vector -- 267 and 833 in the
// captured envelope. Scalar loads, but the row still stays in registers across
// both reduction passes, so the variance pass costs no global traffic and the
// arithmetic is identical to the vectorised path. Only the access width differs.
// ---------------------------------------------------------------------------
template <typename T, int kBlockThreads, int kElemsPerThread>
__global__ void __launch_bounds__(kBlockThreads)
layer_norm_scalar_kernel(const T* __restrict__ x, T* __restrict__ y,
                         const T* __restrict__ weight,
                         const T* __restrict__ bias,
                         int n, float inv_n, float eps) {
  constexpr int kWarps = kBlockThreads / kWarpSize;
  __shared__ float stage[2 * (kWarps + 1)];

  const int64_t row = blockIdx.x;
  const T* __restrict__ xr = x + row * n;
  T* __restrict__ yr = y + row * n;

  T held[kElemsPerThread];
  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < kElemsPerThread; ++i) {
    const int idx = threadIdx.x + i * kBlockThreads;
    if (idx < n) {
      held[i] = xr[idx];
      sum += Packed<T>::scalar_to_float(held[i]);
    }
  }
  const float mean = block_reduce_sum<kBlockThreads>(sum, stage) * inv_n;

  float sq = 0.0f;
#pragma unroll
  for (int i = 0; i < kElemsPerThread; ++i) {
    const int idx = threadIdx.x + i * kBlockThreads;
    if (idx < n) {
      const float d = Packed<T>::scalar_to_float(held[i]) - mean;
      sq += d * d;
    }
  }
  const float var = block_reduce_sum<kBlockThreads>(sq, stage + kWarps + 1);
  const float rstd = rsqrtf(var * inv_n + eps);

#pragma unroll
  for (int i = 0; i < kElemsPerThread; ++i) {
    const int idx = threadIdx.x + i * kBlockThreads;
    if (idx < n) {
      float v = (Packed<T>::scalar_to_float(held[i]) - mean) * rstd;
      if (weight != nullptr) {
        v *= Packed<T>::scalar_to_float(weight[idx]);
      }
      if (bias != nullptr) {
        v += Packed<T>::scalar_to_float(bias[idx]);
      }
      yr[idx] = Packed<T>::scalar_from_float(v);
    }
  }
}

// ---------------------------------------------------------------------------
// Host side: the mapping ladder, the eligibility predicate that mirrors it, and
// the exact-baseline fallback.
// ---------------------------------------------------------------------------

// Which row shapes have an implemented mapping. Kept adjacent to the launchers
// so a predicate and its dispatch cannot drift apart.
inline bool has_vector_mapping(int vecs_per_row) {
  return vecs_per_row >= 1 &&
         (vecs_per_row <= kMaxLadderVecs || vecs_per_row == kTunedVecs);
}

inline bool has_scalar_mapping(int64_t n) {
  return n >= 1 && n <= kMaxScalarElems;
}

// How a row reaches the kernel. Decided once, on host, before any pointer is
// dereferenced, and returned so the caller cannot pick a different launcher than
// the predicate approved.
enum class Path { kFallback, kEmpty, kVector, kScalar };

template <typename T>
void launch_vector(const T* x, T* y, const T* w, const T* b, int64_t rows,
                   int vecs_per_row, float inv_n, float eps,
                   cudaStream_t stream) {
  const unsigned row_grid = static_cast<unsigned>(rows);

#define FK_LN_LAUNCH_BLOCK(BLK, VPT)                                     \
  do {                                                                   \
    layer_norm_block_kernel<T, (BLK), (VPT)>                             \
        <<<row_grid, (BLK), 0, stream>>>(x, y, w, b, vecs_per_row, inv_n, \
                                         eps);                           \
    return;                                                              \
  } while (0)
#define FK_LN_LAUNCH_WARP(VPL)                                           \
  do {                                                                   \
    layer_norm_warp_kernel<T, (VPL), kWarpRows>                          \
        <<<static_cast<unsigned>((rows + kWarpRows - 1) / kWarpRows),     \
           kWarpRows * kWarpSize, 0, stream>>>(x, y, w, b, rows,          \
                                               vecs_per_row, inv_n, eps); \
    return;                                                              \
  } while (0)

  // The tuned width is tested first so overriding it cannot be shadowed by a
  // generic rung.
  if (vecs_per_row == kTunedVecs) FK_LN_LAUNCH_BLOCK(kRowBlock, kRowVpt);
  // ``if constexpr`` so a build that disables or narrows the warp path does not
  // instantiate a kernel it can never reach.
  if constexpr (kWarpMaxVecs >= 1) {
    if (vecs_per_row <= kWarpMaxVecs) {
      if (vecs_per_row <= kWarpSize) FK_LN_LAUNCH_WARP(1);
      if constexpr (kWarpMaxVecs > kWarpSize) FK_LN_LAUNCH_WARP(2);
    }
  }
  if (vecs_per_row <= kNarrowBlock) FK_LN_LAUNCH_BLOCK(kNarrowBlock, 1);
  if (vecs_per_row <= 128) FK_LN_LAUNCH_BLOCK(128, 1);
  if (vecs_per_row <= 256) FK_LN_LAUNCH_BLOCK(256, 1);
  FK_LN_LAUNCH_BLOCK(256, 2);

#undef FK_LN_LAUNCH_WARP
#undef FK_LN_LAUNCH_BLOCK
}

template <typename T>
void launch_scalar(const T* x, T* y, const T* w, const T* b, int64_t rows,
                   int n, float inv_n, float eps, cudaStream_t stream) {
  const unsigned row_grid = static_cast<unsigned>(rows);
  if (n <= 2 * kScalarBlock) {
    layer_norm_scalar_kernel<T, kScalarBlock, 2>
        <<<row_grid, kScalarBlock, 0, stream>>>(x, y, w, b, n, inv_n, eps);
    return;
  }
  layer_norm_scalar_kernel<T, kScalarBlock, 4>
      <<<row_grid, kScalarBlock, 0, stream>>>(x, y, w, b, n, inv_n, eps);
}

inline bool is_aligned16(const void* p) {
  return (reinterpret_cast<uintptr_t>(p) & 15u) == 0;
}

// A parameter that is absent is fine; a present one is validated on its own
// merits, because nothing guarantees it travelled with ``x``. Alignment is a
// separate question, asked only of the path that issues vector loads.
inline bool param_shape_ok(const at::Tensor& p, const at::Tensor& x, int64_t n) {
  if (!p.defined()) {
    return true;
  }
  return p.device() == x.device() && p.scalar_type() == x.scalar_type() &&
         p.dim() == 1 && p.size(0) == n && p.is_contiguous();
}

inline bool param_vector_aligned(const at::Tensor& p) {
  return !p.defined() || is_aligned16(p.const_data_ptr());
}

// Every check here precedes any pointer dereference by a kernel. Shape
// validation deliberately comes before the empty-input question, so a genuinely
// invalid shape still reaches ATen and raises what the baseline would raise.
inline Path choose_path(const at::Tensor& x, const at::Tensor& w,
                        const at::Tensor& b, int64_t n) {
  if (!x.defined() || !x.is_cuda()) {
    return Path::kFallback;
  }
  // Functorch duals, functional/fake tensors and Python subclasses have no
  // ordinary storage to take a pointer to, and reach here without ever setting
  // requires_grad -- torch.func.jvp is the concrete case.
  if (at::isTensorSubclassLike(x) || at::isTensorSubclassLike(w) ||
      at::isTensorSubclassLike(b)) {
    return Path::kFallback;
  }
  // Autocast rewrites LayerNorm's output dtype, and this operator has no
  // autocast registration of its own; the baseline formula re-dispatches through
  // at::layer_norm and so picks up that policy exactly.
  if (at::autocast::is_autocast_enabled(x.device().type())) {
    return Path::kFallback;
  }
  const at::ScalarType dtype = x.scalar_type();
  if (dtype != at::kBFloat16 && dtype != at::kHalf && dtype != at::kFloat) {
    return Path::kFallback;
  }
  if (n <= 0 || x.dim() < 1 || x.size(-1) != n) {
    return Path::kFallback;
  }
  const int64_t total = x.numel();
  if (total % n != 0) {
    return Path::kFallback;
  }
  if (total / n > static_cast<int64_t>(INT32_MAX)) {
    return Path::kFallback;
  }
  if (!x.is_contiguous()) {
    return Path::kFallback;
  }
  if (!param_shape_ok(w, x, n) || !param_shape_ok(b, x, n)) {
    return Path::kFallback;
  }
  // Only *after* the shape and affine-parameter checks above, so an input that
  // the baseline would reject still reaches ATen and raises the same error. There
  // is no row to normalise, so the answer is an empty tensor of the same shape and
  // dtype -- and reaching it must cost no kernel launch, which the fallback would
  // not manage: promoting nonempty affine parameters to fp32 costs two copies.
  if (total == 0) {
    return Path::kEmpty;
  }

  const int elems = 16 / static_cast<int>(x.element_size());
  if (n % elems == 0) {
    // Bound before narrowing: a row of more than INT32_MAX vectors would wrap to
    // a small count that has_vector_mapping accepts, and the kernel would then
    // normalise a prefix of the row and leave the rest of the output
    // uninitialised.
    const int64_t vecs = n / elems;
    if (vecs > static_cast<int64_t>(INT32_MAX) ||
        !has_vector_mapping(static_cast<int>(vecs))) {
      return Path::kFallback;
    }
    if (!is_aligned16(x.const_data_ptr()) || !param_vector_aligned(w) ||
        !param_vector_aligned(b)) {
      return Path::kFallback;
    }
    return Path::kVector;
  }
  // Not a multiple of the vector: the scalar kernel needs no 16-byte alignment,
  // only the element alignment every tensor already has.
  return has_scalar_mapping(n) ? Path::kScalar : Path::kFallback;
}

at::Tensor run_fused(const at::Tensor& x, const at::Tensor& w,
                     const at::Tensor& b, int64_t n, double eps, Path path) {
  const c10::cuda::CUDAGuard device_guard(x.device());
  at::Tensor y = at::empty(x.sizes(), x.options());

  const int elems = 16 / static_cast<int>(x.element_size());
  const int64_t rows = x.numel() / n;
  const float inv_n = 1.0f / static_cast<float>(n);
  const float epsf = static_cast<float>(eps);
  const auto stream = c10::cuda::getCurrentCUDAStream();

#define FK_LN_LAUNCH_DTYPE(CUDA_T)                                           \
  do {                                                                       \
    const CUDA_T* xp = reinterpret_cast<const CUDA_T*>(x.const_data_ptr());  \
    CUDA_T* yp = reinterpret_cast<CUDA_T*>(y.mutable_data_ptr());            \
    const CUDA_T* wp =                                                       \
        w.defined() ? reinterpret_cast<const CUDA_T*>(w.const_data_ptr())    \
                    : nullptr;                                               \
    const CUDA_T* bp =                                                       \
        b.defined() ? reinterpret_cast<const CUDA_T*>(b.const_data_ptr())    \
                    : nullptr;                                               \
    if (path == Path::kVector) {                                             \
      launch_vector<CUDA_T>(xp, yp, wp, bp, rows,                            \
                            static_cast<int>(n / elems), inv_n, epsf,        \
                            stream);                                         \
    } else {                                                                 \
      launch_scalar<CUDA_T>(xp, yp, wp, bp, rows, static_cast<int>(n), inv_n, \
                            epsf, stream);                                   \
    }                                                                        \
  } while (0)

  // Exhaustive over the dtypes ``choose_path`` admits: the last arm needs no
  // test because it is the only one left. ``path`` here is kVector or kScalar --
  // kEmpty and kFallback are handled by the caller before any allocation.
  if (x.scalar_type() == at::kBFloat16) {
    FK_LN_LAUNCH_DTYPE(__nv_bfloat16);
  } else if (x.scalar_type() == at::kHalf) {
    FK_LN_LAUNCH_DTYPE(__half);
  } else {
    FK_LN_LAUNCH_DTYPE(float);
  }
#undef FK_LN_LAUNCH_DTYPE
  return y;
}

// Exactly what ``baseline.py`` computes, promotion included -- an equality, not
// a tolerance argument. ATen's native-dtype LayerNorm already accumulates in
// fp32, but that is not the same function as the baseline's fp32 round trip.
at::Tensor baseline_formula(const at::Tensor& x, const at::Tensor& w,
                            const at::Tensor& b, int64_t n, double eps,
                            bool promote_fp32) {
  const std::optional<at::Tensor> wo =
      w.defined() ? std::optional<at::Tensor>(w) : std::nullopt;
  const std::optional<at::Tensor> bo =
      b.defined() ? std::optional<at::Tensor>(b) : std::nullopt;
  if (!promote_fp32) {
    return at::layer_norm(x, {n}, wo, bo, eps);
  }
  const at::ScalarType orig = x.scalar_type();
  const std::optional<at::Tensor> wf =
      (w.defined() && w.scalar_type() != at::kFloat)
          ? std::optional<at::Tensor>(w.to(at::kFloat))
          : wo;
  const std::optional<at::Tensor> bf =
      (b.defined() && b.scalar_type() != at::kFloat)
          ? std::optional<at::Tensor>(b.to(at::kFloat))
          : bo;
  return at::layer_norm(x.to(at::kFloat), {n}, wf, bf, eps).to(orig);
}

at::Tensor layer_norm(const at::Tensor& x,
                      const std::optional<at::Tensor>& weight,
                      const std::optional<at::Tensor>& bias, int64_t n,
                      double eps, bool promote_fp32) {
  const at::Tensor w = weight.has_value() ? *weight : at::Tensor();
  const at::Tensor b = bias.has_value() ? *bias : at::Tensor();
  // The fused paths allocate with at::empty and launch a raw kernel, so they
  // record nothing for autograd. Grad mode being *enabled* is the whole test, not
  // whether some tensor currently requires grad: a caller who has not entered
  // no_grad may attach requires_grad later in the same graph, or wrap this in
  // checkpointing that replays the forward under different requires_grad state.
  // Gradient-enabled calls therefore take the baseline formula, whose ATen ops
  // this operator's CompositeImplicitAutograd registration traces through. The
  // cost is that the fused kernels run only under no_grad, which is where every
  // inference path -- including the benchmark harness -- already puts them.
  if (at::GradMode::is_enabled()) {
    return baseline_formula(x, w, b, n, eps, promote_fp32);
  }
  const Path path = choose_path(x, w, b, n);
  if (path == Path::kEmpty) {
    return at::empty_like(x);
  }
  if (path != Path::kFallback) {
    return run_fused(x, w, b, n, eps, path);
  }
  return baseline_formula(x, w, b, n, eps, promote_fp32);
}

}  // namespace

TORCH_LIBRARY(fk_ln_cand, m) {
  m.def(
      "layer_norm(Tensor x, Tensor? weight, Tensor? bias, int n, float eps, "
      "bool promote_fp32) -> Tensor",
      &layer_norm);
}
"""


def _load_fused_op():
    """Build and register the fused operator, returning its callable.

    Compilation happens here, at import, so nothing is deferred into a timed
    ``forward``. The includes are lean on purpose: ``<torch/extension.h>``
    through nvcc dominates the build, and this operator is registered with
    ``TORCH_LIBRARY`` rather than pybind, so none of it is needed.
    """
    import os

    from torch.utils.cpp_extension import load_inline

    # Without this, cpp_extension honours the ambient TORCH_CUDA_ARCH_LIST, which
    # in this environment names six architectures -- six nvcc passes over every
    # instantiation, for five targets that will never run the kernel. Narrowing it
    # to the device actually present cuts the cold build several-fold, which is
    # what keeps it comfortable inside the harness's wall-clock cap. Derived from
    # the live device rather than hardcoded, so it can never name the wrong arch,
    # and restored afterwards so no later build in this process is affected.
    previous_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
    try:
        load_inline(
            name=_LIBRARY_NAME,
            cpp_sources="",
            cuda_sources=_CUDA_SOURCE,
            extra_cuda_cflags=["-O3"],
            is_python_module=False,
            no_implicit_headers=True,
        )
    finally:
        if previous_arch_list is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = previous_arch_list
    # Bind the overload, not the packet: the packet re-resolves overloads from
    # the argument types on every call, and forward is launch-latency bound on
    # the small shapes.
    return getattr(torch.ops, _LIBRARY_NAME).layer_norm.default


try:
    _fused_layer_norm = _load_fused_op()
except Exception:  # noqa: BLE001 - a build that cannot happen must degrade, not
    # take the module down with it: an import failure costs every case at once.
    _fused_layer_norm = None


class LayerNorm(nn.Module):
    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        create_scale: bool = True,
        create_offset: bool = True,
        promote_fp32: bool = True,
    ):
        super().__init__()
        self.normalized_shape = (normalized_shape,)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.promote_fp32 = promote_fp32

        if elementwise_affine and create_scale:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
        else:
            self.register_parameter("weight", None)

        if elementwise_affine and create_offset:
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter("bias", None)

        # The kernel wants the row width as a plain int; ``normalized_shape``
        # stays a tuple to mirror the baseline's attribute exactly.
        self._n = int(normalized_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if _fused_layer_norm is not None:
            # One dispatch: eligibility and the fallback both live inside the
            # operator, so nothing here has to inspect the tensor's metadata.
            return _fused_layer_norm(
                x, self.weight, self.bias, self._n, self.eps, self.promote_fp32,
            )

        if not self.promote_fp32:
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps,
            )
        orig_dtype = x.dtype
        weight, bias = self.weight, self.bias
        if weight is not None and weight.dtype != torch.float32:
            weight = weight.float()
        if bias is not None and bias.dtype != torch.float32:
            bias = bias.float()
        return F.layer_norm(
            x.float(), self.normalized_shape, weight, bias, self.eps,
        ).to(orig_dtype)
