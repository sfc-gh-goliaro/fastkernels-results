"""Softmax / LogSoftmax activations, with a fused CUDA softmax for strided inputs.

`F.softmax` materializes a contiguous copy of any non-contiguous input before
reducing, which costs a second kernel launch, and it reduces a non-final dimension
with a separate three-pass spatial kernel. Both show up on the captured shapes.

This module replaces that with a single kernel launch per call. A C++ dispatcher
normalizes any `(dim, shape, stride)` into a logical `(rows, R)` row set and picks
one of three kernels: a warp-per-row kernel when the reduction runs along contiguous
memory, a thread-per-row kernel when it is strided, and a block-per-row kernel when
the reduction is longer than either of those can hold in registers. Anything outside
the supported set goes to `at::softmax`, so correctness never depends on the
extension.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_EXT_NAME = "fk_l1_softmax"
_BUILD_DIR = Path(__file__).resolve().parent / ".torch_extensions"

# Reduction dim is the innermost in memory -> one warp per row, row in registers.
ROUTE_ROW = 1
# Reduction dim is strided -> one thread per row, R values in registers.
ROUTE_COLUMN = 2
# Reduction longer than either register budget -> one block per row, threads
# striding the reduction. Unbounded R, still one launch.
ROUTE_BLOCK_ROW = 3
# Outside the supported set -> at::softmax.
ROUTE_FALLBACK = 0

ROUTE_NAMES = {
    ROUTE_FALLBACK: "aten_fallback",
    ROUTE_ROW: "row_kernel",
    ROUTE_COLUMN: "column_kernel",
    ROUTE_BLOCK_ROW: "block_row_kernel",
}

_CPP_SOURCE = r"""
at::Tensor softmax_forward(const at::Tensor& x, int64_t dim);
at::Tensor softmax_plan(const at::Tensor& x, int64_t dim);
"""

_CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <limits>
#include <cuda_bf16.h>

namespace {

constexpr int kWarp = 32;
constexpr int kRowWarps = 4;                    // warps per block, row kernel
constexpr int kRowBlock = kRowWarps * kWarp;
constexpr int kColBlock = 128;                  // threads per block, column kernel
constexpr int kMaxRowR = 1024;                  // register budget, row kernel
constexpr int kMaxColR = 32;                    // register budget, column kernel
constexpr int kMaxLevels = 3;                   // non-reduced dims mapped to grid x/y/z
constexpr int kMaxRank = 8;
// CUDA launch bounds. These are architecture-independent minimums guaranteed by
// the programming model, so planning does not have to query the device.
constexpr int64_t kMaxGridX = 2147483647;       // 2^31 - 1
constexpr int64_t kMaxGridYZ = 65535;
constexpr int64_t kTargetBlocks = 148;  // SMs on this device

constexpr int64_t kRouteFallback = 0;
constexpr int64_t kRouteRow = 1;
constexpr int64_t kRouteColumn = 2;
constexpr int64_t kRouteBlockRow = 3;

// ---------------------------------------------------------------------------
// Scalar conversions. `-D__CUDA_NO_HALF_CONVERSIONS__` is on, so every narrow
// type goes through an explicit intrinsic rather than an implicit cast.
// ---------------------------------------------------------------------------
template <typename T> struct Num;

template <> struct Num<float> {
  static __device__ __forceinline__ float to(float v) { return v; }
  static __device__ __forceinline__ float from(float v) { return v; }
};

template <> struct Num<__half> {
  static __device__ __forceinline__ float to(__half v) { return __half2float(v); }
  static __device__ __forceinline__ __half from(float v) { return __float2half_rn(v); }
};

template <> struct Num<__nv_bfloat16> {
  static __device__ __forceinline__ float to(__nv_bfloat16 v) {
    return __bfloat162float(v);
  }
  static __device__ __forceinline__ __nv_bfloat16 from(float v) {
    return __float2bfloat16_rn(v);
  }
};

// ---------------------------------------------------------------------------
// 16 B vector access. Packing goes through `*_as_ushort` bit moves rather than
// taking the address of a local, so nothing needs a stack slot.
// ---------------------------------------------------------------------------
template <typename T, int W> struct Vec;

template <> struct Vec<float, 4> {
  static __device__ __forceinline__ void load(const float* p, float* d) {
    const float4 r = *reinterpret_cast<const float4*>(p);
    d[0] = r.x; d[1] = r.y; d[2] = r.z; d[3] = r.w;
  }
  static __device__ __forceinline__ void store(float* p, const float* s) {
    float4 r;
    r.x = s[0]; r.y = s[1]; r.z = s[2]; r.w = s[3];
    *reinterpret_cast<float4*>(p) = r;
  }
};

template <> struct Vec<__half, 8> {
  static __device__ __forceinline__ void unpack(unsigned w, float* d) {
    d[0] = __half2float(__ushort_as_half((unsigned short)(w & 0xffffu)));
    d[1] = __half2float(__ushort_as_half((unsigned short)(w >> 16)));
  }
  static __device__ __forceinline__ unsigned pack(float a, float b) {
    return (unsigned)(unsigned short)__half_as_ushort(__float2half_rn(a))
         | ((unsigned)(unsigned short)__half_as_ushort(__float2half_rn(b)) << 16);
  }
  static __device__ __forceinline__ void load(const __half* p, float* d) {
    const uint4 r = *reinterpret_cast<const uint4*>(p);
    unpack(r.x, d); unpack(r.y, d + 2); unpack(r.z, d + 4); unpack(r.w, d + 6);
  }
  static __device__ __forceinline__ void store(__half* p, const float* s) {
    uint4 r;
    r.x = pack(s[0], s[1]); r.y = pack(s[2], s[3]);
    r.z = pack(s[4], s[5]); r.w = pack(s[6], s[7]);
    *reinterpret_cast<uint4*>(p) = r;
  }
};

template <> struct Vec<__nv_bfloat16, 8> {
  static __device__ __forceinline__ void unpack(unsigned w, float* d) {
    d[0] = __bfloat162float(__ushort_as_bfloat16((unsigned short)(w & 0xffffu)));
    d[1] = __bfloat162float(__ushort_as_bfloat16((unsigned short)(w >> 16)));
  }
  static __device__ __forceinline__ unsigned pack(float a, float b) {
    return (unsigned)(unsigned short)__bfloat16_as_ushort(__float2bfloat16_rn(a))
         | ((unsigned)(unsigned short)__bfloat16_as_ushort(__float2bfloat16_rn(b)) << 16);
  }
  static __device__ __forceinline__ void load(const __nv_bfloat16* p, float* d) {
    const uint4 r = *reinterpret_cast<const uint4*>(p);
    unpack(r.x, d); unpack(r.y, d + 2); unpack(r.z, d + 4); unpack(r.w, d + 6);
  }
  static __device__ __forceinline__ void store(__nv_bfloat16* p, const float* s) {
    uint4 r;
    r.x = pack(s[0], s[1]); r.y = pack(s[2], s[3]);
    r.z = pack(s[4], s[5]); r.w = pack(s[6], s[7]);
    *reinterpret_cast<uint4*>(p) = r;
  }
};

__device__ __forceinline__ float warp_max(float v) {
  #pragma unroll
  for (int off = kWarp / 2; off > 0; off >>= 1) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
  }
  return v;
}

__device__ __forceinline__ float warp_sum(float v) {
  #pragma unroll
  for (int off = kWarp / 2; off > 0; off >>= 1) {
    v += __shfl_xor_sync(0xffffffffu, v, off);
  }
  return v;
}

// Reduces one value per thread across a whole block. Every thread returns the
// same answer, so no broadcast round trip is needed: the per-warp partials are
// small enough (at most 32) for each thread to fold them itself, in a fixed
// order so all threads agree bit for bit. The trailing barrier is what makes the
// shared buffer safe to reuse for a second reduction in the same kernel.
template <int BLOCK, bool MAXOP>
__device__ __forceinline__ float block_reduce(float v, float* smem) {
  constexpr int kWarps = BLOCK / kWarp;
  v = MAXOP ? warp_max(v) : warp_sum(v);
  if constexpr (kWarps == 1) {
    return v;
  } else {
    if ((threadIdx.x & (kWarp - 1)) == 0) smem[threadIdx.x >> 5] = v;
    __syncthreads();
    float acc = smem[0];
    #pragma unroll
    for (int i = 1; i < kWarps; ++i) {
      acc = MAXOP ? fmaxf(acc, smem[i]) : acc + smem[i];
    }
    __syncthreads();
    return acc;
  }
}

// One (size, in_stride, out_stride) per non-reduced level, innermost first.
struct Levels {
  int64_t size[kMaxLevels];
  int64_t in_stride[kMaxLevels];
  int64_t out_stride[kMaxLevels];
};

// ---------------------------------------------------------------------------
// Row kernel: the reduction runs along contiguous input memory (in_stride == 1).
// One warp owns one row; the row lives in registers, so the input is read once.
//
// ITEMS is the compile-time per-lane count -- a runtime bound here would put the
// array in local memory and the single-read property would quietly become false.
// An item is one scalar when W == 1 and one 16 B vector otherwise.
// ---------------------------------------------------------------------------
template <typename T, int W, int ITEMS, bool EXACT>
__global__ __launch_bounds__(kRowBlock) void softmax_rows(
    const T* __restrict__ in, T* __restrict__ out, Levels lv,
    int n_items, int64_t out_reduce_stride) {
  const int lane = threadIdx.x & (kWarp - 1);
  const int64_t row = (int64_t)blockIdx.x * kRowWarps + (threadIdx.x >> 5);
  if (row >= lv.size[0]) return;

  const int64_t in_base = row * lv.in_stride[0]
                        + (int64_t)blockIdx.y * lv.in_stride[1]
                        + (int64_t)blockIdx.z * lv.in_stride[2];
  const int64_t out_base = row * lv.out_stride[0]
                         + (int64_t)blockIdx.y * lv.out_stride[1]
                         + (int64_t)blockIdx.z * lv.out_stride[2];

  float v[ITEMS * W];
  #pragma unroll
  for (int j = 0; j < ITEMS; ++j) {
    const int item = lane + j * kWarp;
    // Clamp rather than predicate the address: a speculated load then still
    // lands inside the tensor, and the value is masked out below.
    const int safe = EXACT ? item : (item < n_items ? item : 0);
    const bool live = EXACT || item < n_items;
    if constexpr (W == 1) {
      const float raw = Num<T>::to(in[in_base + safe]);
      v[j] = live ? raw : -INFINITY;
    } else {
      float tmp[W];
      Vec<T, W>::load(in + in_base + (int64_t)safe * W, tmp);
      #pragma unroll
      for (int k = 0; k < W; ++k) v[j * W + k] = live ? tmp[k] : -INFINITY;
    }
  }

  float m = v[0];
  #pragma unroll
  for (int k = 1; k < ITEMS * W; ++k) m = fmaxf(m, v[k]);
  m = warp_max(m);

  float total = 0.f;
  #pragma unroll
  for (int k = 0; k < ITEMS * W; ++k) {
    v[k] = expf(v[k] - m);
    total += v[k];
  }
  const float scale = 1.f / warp_sum(total);

  #pragma unroll
  for (int j = 0; j < ITEMS; ++j) {
    const int item = lane + j * kWarp;
    if (!EXACT && item >= n_items) continue;
    if constexpr (W == 1) {
      out[out_base + (int64_t)item * out_reduce_stride] =
          Num<T>::from(v[j] * scale);
    } else {
      float tmp[W];
      #pragma unroll
      for (int k = 0; k < W; ++k) tmp[k] = v[j * W + k] * scale;
      Vec<T, W>::store(out + out_base + (int64_t)item * W, tmp);
    }
  }
}

// ---------------------------------------------------------------------------
// Column kernel: the reduction is strided (in_stride != 1). One thread owns one
// row so that consecutive threads walk the fastest-varying non-reduced level --
// stride 1 in the output for the dim=1 capture, which makes both the loads and
// the stores coalesced. A warp per row would put lanes a full reduction stride
// apart instead.
// ---------------------------------------------------------------------------
template <typename T, int R, bool EXACT>
__global__ __launch_bounds__(kColBlock) void softmax_columns(
    const T* __restrict__ in, T* __restrict__ out, Levels lv,
    int n_reduce, int64_t in_reduce_stride, int64_t out_reduce_stride) {
  const int64_t i0 = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t i1 = (int64_t)blockIdx.y * blockDim.y + threadIdx.y;
  if (i0 >= lv.size[0] || i1 >= lv.size[1]) return;

  const int64_t in_base = i0 * lv.in_stride[0] + i1 * lv.in_stride[1]
                        + (int64_t)blockIdx.z * lv.in_stride[2];
  const int64_t out_base = i0 * lv.out_stride[0] + i1 * lv.out_stride[1]
                         + (int64_t)blockIdx.z * lv.out_stride[2];

  float v[R];
  #pragma unroll
  for (int r = 0; r < R; ++r) {
    const int safe = EXACT ? r : (r < n_reduce ? r : 0);
    const float raw = Num<T>::to(in[in_base + (int64_t)safe * in_reduce_stride]);
    v[r] = (EXACT || r < n_reduce) ? raw : -INFINITY;
  }

  float m = v[0];
  #pragma unroll
  for (int r = 1; r < R; ++r) m = fmaxf(m, v[r]);

  float total = 0.f;
  #pragma unroll
  for (int r = 0; r < R; ++r) {
    v[r] = expf(v[r] - m);
    total += v[r];
  }
  const float scale = 1.f / total;

  #pragma unroll
  for (int r = 0; r < R; ++r) {
    if (!EXACT && r >= n_reduce) continue;
    out[out_base + (int64_t)r * out_reduce_stride] = Num<T>::from(v[r] * scale);
  }
}

// ---------------------------------------------------------------------------
// Block kernel: one block per logical row, threads striding the reduction. This
// is what lifts the register-budget limits of the other two -- nothing is held
// per thread across the whole row, so the reduction extent is unbounded and both
// contiguous and strided reductions of any length are served here rather than
// handed to ATen.
//
// Still one launch: the exponentials are staged into the output the row will
// occupy anyway, then normalized in place after the block sum is known. The cost
// is re-reading the input once and the staged output once, which is why this is
// the outside-the-budget path and not the path the captured shapes take.
// ---------------------------------------------------------------------------
template <typename T, int BLOCK>
__global__ __launch_bounds__(BLOCK) void softmax_block_rows(
    const T* __restrict__ in, T* __restrict__ out, Levels lv,
    int64_t n_reduce, int64_t in_reduce_stride, int64_t out_reduce_stride) {
  __shared__ float smem[BLOCK / kWarp];
  // Block-uniform, so either the whole block leaves or none of it does and the
  // barriers inside block_reduce stay well formed.
  const int64_t row = blockIdx.x;
  if (row >= lv.size[0]) return;

  const int64_t in_base = row * lv.in_stride[0]
                        + (int64_t)blockIdx.y * lv.in_stride[1]
                        + (int64_t)blockIdx.z * lv.in_stride[2];
  const int64_t out_base = row * lv.out_stride[0]
                         + (int64_t)blockIdx.y * lv.out_stride[1]
                         + (int64_t)blockIdx.z * lv.out_stride[2];

  float m = -INFINITY;
  for (int64_t r = threadIdx.x; r < n_reduce; r += BLOCK) {
    m = fmaxf(m, Num<T>::to(in[in_base + r * in_reduce_stride]));
  }
  m = block_reduce<BLOCK, true>(m, smem);

  float total = 0.f;
  for (int64_t r = threadIdx.x; r < n_reduce; r += BLOCK) {
    const float e = expf(Num<T>::to(in[in_base + r * in_reduce_stride]) - m);
    out[out_base + r * out_reduce_stride] = Num<T>::from(e);
    total += e;
  }
  // Summing the pre-rounding fp32 exponentials, not the stored ones, so the
  // normalizer is not itself degraded by the staging dtype.
  total = block_reduce<BLOCK, false>(total, smem);
  const float scale = 1.f / total;

  for (int64_t r = threadIdx.x; r < n_reduce; r += BLOCK) {
    const int64_t o = out_base + r * out_reduce_stride;
    out[o] = Num<T>::from(Num<T>::to(out[o]) * scale);
  }
}

// ---------------------------------------------------------------------------
// Host-side normalization. See docs/normalization.md for the contract; this is
// the only place that decides row vs column vs fallback.
// ---------------------------------------------------------------------------
struct Plan {
  int64_t route = kRouteFallback;
  int64_t reduce = 0;             // R
  int64_t in_reduce_stride = 0;
  int64_t out_reduce_stride = 0;
  int64_t n_levels = 0;
  int64_t width = 1;              // 1 = scalar, else 16 / element size
  int64_t bucket = 0;             // per-lane items (row) or R bucket (column)
  int64_t exact = 0;              // bucket matches the true extent, no masking
  int64_t n_items = 0;            // R / width, row kernel only
  Levels levels{};
  bool input_aligned_16 = false;  // every input pointer/stride permits 16 B access
};

// Per-lane item counts the row kernel is instantiated for.
constexpr int kRowBuckets[] = {1, 2, 3, 4, 6, 8, 16, 32};
// Reduction extents the column kernel is instantiated for.
constexpr int kColBuckets[] = {1, 2, 4, 8, 16, 32};

// Threads per block for the block kernel: enough to cover the reduction in a
// few strides without paying for warps that would idle on a short row.
int64_t block_threads_for(int64_t reduce) {
  if (reduce <= 128) return 128;
  if (reduce <= 256) return 256;
  if (reduce <= 512) return 512;
  return 1024;
}

// The launch shape a resolved plan implies. `make_plan` and every launcher get
// this from `launch_geometry`, so a route can never be planned with a grid the
// launch cannot express.
struct Geometry {
  int64_t grid_x, grid_y, grid_z;
  int64_t block_x, block_y;
};

int64_t pick_bucket(int64_t need, const int* table, int n) {
  for (int i = 0; i < n; ++i) {
    if (table[i] >= need) return table[i];
  }
  return 0;
}

// Fills in the vectorization-dependent fields. Called once with the input-side
// alignment verdict, and again if the freshly allocated output turns out not to
// be 16 B aligned (the caching allocator makes that impossible in practice, but
// the kernel's correctness must not rest on an allocator detail).
void resolve_row_width(Plan& p, int64_t esize, bool out_aligned) {
  const int64_t full = 16 / esize;
  const bool vector_ok = p.input_aligned_16 && out_aligned
                      && p.out_reduce_stride == 1
                      && (p.reduce * esize) % 16 == 0;
  p.width = vector_ok ? full : 1;
  p.n_items = p.reduce / p.width;
  const int64_t need = (p.n_items + kWarp - 1) / kWarp;
  const int64_t cap = kWarp / p.width;   // register budget at R <= kMaxRowR
  p.bucket = need <= cap
      ? pick_bucket(need, kRowBuckets, (int)(sizeof(kRowBuckets) / sizeof(int)))
      : 0;
  p.exact = (p.bucket != 0 && p.n_items == p.bucket * kWarp) ? 1 : 0;
  if (p.bucket == 0) p.route = kRouteFallback;
}

// Derives the launch shape for a resolved route and reports whether CUDA can
// express it. Returning false is what turns an otherwise-supported input into an
// `at::softmax` fallback: an expanded view can carry a logical extent far past
// the grid limits on a few kilobytes of storage, and narrowing that to `unsigned`
// would silently produce the wrong grid.
bool launch_geometry(const Plan& p, Geometry& g) {
  const int64_t s0 = p.levels.size[0];
  const int64_t s1 = p.levels.size[1];
  const int64_t s2 = p.levels.size[2];
  if (p.route == kRouteRow) {
    g.block_x = kRowBlock;
    g.block_y = 1;
    g.grid_x = (s0 + kRowWarps - 1) / kRowWarps;   // one warp per row
    g.grid_y = s1;
    g.grid_z = s2;
  } else if (p.route == kRouteColumn) {
    // Spread the fastest-varying level across threadIdx.x so a warp reads
    // consecutive addresses, then cap the block so the rows land on several SMs
    // where there are enough of them to go round. Filling one block first leaves
    // a small case on a single SM, whose latency then depends on what else that
    // one SM is doing.
    int64_t bx = 1;
    while (bx < s0 && bx < kColBlock) bx <<= 1;
    const __int128 rows = (__int128)s0 * s1 * s2;
    int64_t cap = kColBlock;
    while (cap > kWarp && rows / cap < kTargetBlocks) cap >>= 1;
    if (cap < bx) cap = bx;
    int64_t by = 1;
    while (bx * by * 2 <= cap && by < s1) by <<= 1;
    g.block_x = bx;
    g.block_y = by;
    g.grid_x = (s0 + bx - 1) / bx;
    g.grid_y = (s1 + by - 1) / by;
    g.grid_z = s2;
  } else if (p.route == kRouteBlockRow) {
    g.block_x = p.bucket;                          // one block per row
    g.block_y = 1;
    g.grid_x = s0;
    g.grid_y = s1;
    g.grid_z = s2;
  } else {
    return false;
  }
  return g.grid_x >= 1 && g.grid_x <= kMaxGridX
      && g.grid_y >= 1 && g.grid_y <= kMaxGridYZ
      && g.grid_z >= 1 && g.grid_z <= kMaxGridYZ;
}

Plan make_plan(const at::Tensor& x, int64_t dim) {
  Plan p;
  if (!x.is_cuda() || x.numel() == 0 || x.requires_grad()) return p;
  if (x.is_neg() || x.is_conj()) return p;   // data_ptr() would expose raw values
  const auto st = x.scalar_type();
  if (st != at::kFloat && st != at::kHalf && st != at::kBFloat16) return p;

  const int64_t rank = x.dim();
  if (rank <= 0 || rank > kMaxRank) return p;
  const int64_t d = dim < 0 ? dim + rank : dim;
  if (d < 0 || d >= rank) return p;          // let at::softmax raise

  int64_t contig[kMaxRank];
  int64_t acc = 1;
  for (int64_t k = rank - 1; k >= 0; --k) {
    contig[k] = acc;
    acc *= x.size(k);
  }

  p.reduce = x.size(d);
  p.in_reduce_stride = x.stride(d);
  p.out_reduce_stride = contig[d];
  if (p.in_reduce_stride < 0) return p;

  // Non-reduced levels, size-1 dropped, outermost first for now.
  int64_t sz[kMaxRank], is[kMaxRank], os[kMaxRank];
  int n = 0;
  for (int64_t k = 0; k < rank; ++k) {
    if (k == d || x.size(k) == 1) continue;
    if (x.stride(k) < 0) return p;
    sz[n] = x.size(k); is[n] = x.stride(k); os[n] = contig[k];
    ++n;
  }

  // Collapse innermost-first: an outer level joins the group it precedes only
  // when the input AND the output strides both say the two are one flat run.
  // Testing the input alone would merge across the reduction dim and scatter
  // the output.
  int64_t msz[kMaxRank], mis[kMaxRank], mos[kMaxRank];
  int m = 0;
  for (int k = n - 1; k >= 0; --k) {
    if (m > 0 && is[k] == msz[m - 1] * mis[m - 1]
              && os[k] == msz[m - 1] * mos[m - 1]) {
      msz[m - 1] *= sz[k];
    } else {
      msz[m] = sz[k]; mis[m] = is[k]; mos[m] = os[k];
      ++m;
    }
  }
  if (m > kMaxLevels) return p;

  for (int i = 0; i < kMaxLevels; ++i) {
    p.levels.size[i] = i < m ? msz[i] : 1;
    p.levels.in_stride[i] = i < m ? mis[i] : 0;
    p.levels.out_stride[i] = i < m ? mos[i] : 0;
  }
  p.n_levels = m;

  const int64_t esize = x.element_size();
  // Widest element offset either kernel can form. Accumulated in __int128 so
  // the check cannot be defeated by the check's own arithmetic overflowing.
  __int128 span = (__int128)(p.reduce - 1) * p.in_reduce_stride;
  for (int i = 0; i < m; ++i) span += (__int128)(msz[i] - 1) * mis[i];
  if (span < 0 || span > (__int128)(std::numeric_limits<int64_t>::max() / esize)) return p;

  // A reduction too long to hold per thread goes to the block kernel rather
  // than to ATen: it has no register-budget ceiling, so this is a capability
  // boundary rather than a support boundary.
  const bool over_budget = p.in_reduce_stride == 1 ? p.reduce > kMaxRowR
                                                   : p.reduce > kMaxColR;
  if (over_budget) {
    p.route = kRouteBlockRow;
    p.bucket = block_threads_for(p.reduce);
  } else if (p.in_reduce_stride == 1) {
    p.route = kRouteRow;
    bool aligned = (reinterpret_cast<uintptr_t>(x.data_ptr()) % 16) == 0;
    for (int i = 0; i < m && aligned; ++i) {
      aligned = ((mis[i] * esize) % 16 == 0) && ((mos[i] * esize) % 16 == 0);
    }
    p.input_aligned_16 = aligned;
    resolve_row_width(p, esize, true);
  } else {
    p.route = kRouteColumn;
    p.bucket = pick_bucket(p.reduce, kColBuckets,
                           (int)(sizeof(kColBuckets) / sizeof(int)));
    if (p.bucket == 0) return Plan{};
    p.exact = (p.bucket == p.reduce) ? 1 : 0;
  }

  // Last gate, and the only one that knows the grid: a route is real only if its
  // launch shape fits CUDA's bounds. `resolve_row_width` can also have given up
  // here, which this catches too.
  Geometry g;
  if (p.route == kRouteFallback || !launch_geometry(p, g)) return Plan{};
  return p;
}

// ---------------------------------------------------------------------------
// Launch
// ---------------------------------------------------------------------------
template <typename T, int W>
void launch_rows(const Plan& p, const Geometry& g, const T* in, T* out,
                 cudaStream_t stream) {
  const dim3 grid((unsigned)g.grid_x, (unsigned)g.grid_y, (unsigned)g.grid_z);
  const dim3 block((unsigned)g.block_x, (unsigned)g.block_y);
  const int items = (int)p.n_items;
#define FK_ROW_CASE(N)                                                         \
  case N:                                                                      \
    if constexpr (N <= kWarp / W) {                                            \
      if (p.exact) {                                                           \
        softmax_rows<T, W, N, true><<<grid, block, 0, stream>>>(                \
            in, out, p.levels, items, p.out_reduce_stride);                     \
      } else {                                                                 \
        softmax_rows<T, W, N, false><<<grid, block, 0, stream>>>(               \
            in, out, p.levels, items, p.out_reduce_stride);                     \
      }                                                                        \
    } else {                                                                   \
      TORCH_CHECK(false, "row bucket ", N, " exceeds the register budget");     \
    }                                                                          \
    break;
  switch (p.bucket) {
    FK_ROW_CASE(1) FK_ROW_CASE(2) FK_ROW_CASE(3) FK_ROW_CASE(4)
    FK_ROW_CASE(6) FK_ROW_CASE(8) FK_ROW_CASE(16) FK_ROW_CASE(32)
    default: TORCH_CHECK(false, "unhandled row bucket ", p.bucket);
  }
#undef FK_ROW_CASE
}

template <typename T>
void launch_columns(const Plan& p, const Geometry& g, const T* in, T* out,
                    cudaStream_t stream) {
  const dim3 block((unsigned)g.block_x, (unsigned)g.block_y);
  const dim3 grid((unsigned)g.grid_x, (unsigned)g.grid_y, (unsigned)g.grid_z);
  const int reduce = (int)p.reduce;
#define FK_COL_CASE(N)                                                         \
  case N:                                                                      \
    if (p.exact) {                                                             \
      softmax_columns<T, N, true><<<grid, block, 0, stream>>>(                  \
          in, out, p.levels, reduce, p.in_reduce_stride, p.out_reduce_stride);  \
    } else {                                                                   \
      softmax_columns<T, N, false><<<grid, block, 0, stream>>>(                 \
          in, out, p.levels, reduce, p.in_reduce_stride, p.out_reduce_stride);  \
    }                                                                          \
    break;
  switch (p.bucket) {
    FK_COL_CASE(1) FK_COL_CASE(2) FK_COL_CASE(4)
    FK_COL_CASE(8) FK_COL_CASE(16) FK_COL_CASE(32)
    default: TORCH_CHECK(false, "unhandled column bucket ", p.bucket);
  }
#undef FK_COL_CASE
}

template <typename T>
void launch_block_rows(const Plan& p, const Geometry& g, const T* in, T* out,
                       cudaStream_t stream) {
  // One block per row of the fastest-varying level; the other two levels stay
  // on grid y and z exactly as in the other kernels.
  const dim3 grid((unsigned)g.grid_x, (unsigned)g.grid_y, (unsigned)g.grid_z);
#define FK_BLOCK_CASE(N)                                                       \
  case N:                                                                      \
    softmax_block_rows<T, N><<<grid, (unsigned)g.block_x, 0, stream>>>(         \
        in, out, p.levels, p.reduce, p.in_reduce_stride, p.out_reduce_stride);  \
    break;
  switch (p.bucket) {
    FK_BLOCK_CASE(128) FK_BLOCK_CASE(256) FK_BLOCK_CASE(512) FK_BLOCK_CASE(1024)
    default: TORCH_CHECK(false, "unhandled block size ", p.bucket);
  }
#undef FK_BLOCK_CASE
}

template <typename T>
void launch(const Plan& p, const at::Tensor& x, at::Tensor& y,
            cudaStream_t stream) {
  const T* in = reinterpret_cast<const T*>(x.data_ptr());
  T* out = reinterpret_cast<T*>(y.data_ptr());
  // The same derivation `make_plan` gated on, so the two cannot disagree.
  Geometry g;
  TORCH_CHECK(launch_geometry(p, g),
              "launch geometry outside CUDA's grid bounds for route ", p.route,
              "; make_plan should have routed this to at::softmax");
  if (p.route == kRouteBlockRow) {
    launch_block_rows<T>(p, g, in, out, stream);
  } else if (p.route == kRouteRow) {
    if (p.width == 1) {
      launch_rows<T, 1>(p, g, in, out, stream);
    } else {
      launch_rows<T, 16 / (int)sizeof(T)>(p, g, in, out, stream);
    }
  } else {
    launch_columns<T>(p, g, in, out, stream);
  }
}

}  // namespace

at::Tensor softmax_forward(const at::Tensor& x, int64_t dim) {
  Plan p = make_plan(x, dim);
  if (p.route == kRouteFallback) return at::softmax(x, dim);

  const c10::cuda::CUDAGuard device_guard(x.device());
  auto y = at::empty(x.sizes(), x.options());
  if (p.route == kRouteRow && p.width > 1
      && (reinterpret_cast<uintptr_t>(y.data_ptr()) % 16) != 0) {
    resolve_row_width(p, x.element_size(), false);
    if (p.route == kRouteFallback) return at::softmax(x, dim);
  }

  auto stream = at::cuda::getCurrentCUDAStream();
  switch (x.scalar_type()) {
    case at::kFloat: launch<float>(p, x, y, stream); break;
    case at::kHalf: launch<__half>(p, x, y, stream); break;
    case at::kBFloat16: launch<__nv_bfloat16>(p, x, y, stream); break;
    default: TORCH_CHECK(false, "unreachable dtype ", x.scalar_type());
  }
  AT_CUDA_CHECK(cudaGetLastError());
  return y;
}

// Reports the routing decision without running it, so a test can assert that a
// given input really is served by the kernel it is credited to. Slot meanings
// are documented by `PLAN_FIELDS` on the Python side.
at::Tensor softmax_plan(const at::Tensor& x, int64_t dim) {
  const Plan p = make_plan(x, dim);
  auto t = at::zeros({12}, at::TensorOptions().dtype(at::kLong));
  auto a = t.accessor<int64_t, 1>();
  a[0] = p.route;
  a[1] = p.reduce;
  a[2] = p.in_reduce_stride;
  a[3] = p.out_reduce_stride;
  a[4] = p.n_levels;
  a[5] = p.width;
  a[6] = p.bucket;
  a[7] = p.exact;
  a[8] = p.n_items;
  a[9] = p.levels.size[0];
  a[10] = p.levels.size[1];
  a[11] = p.levels.size[2];
  return t;
}
"""

PLAN_FIELDS = (
    "route", "reduce", "in_reduce_stride", "out_reduce_stride", "n_levels",
    "width", "bucket", "exact", "n_items", "size0", "size1", "size2",
)


def _load_extension():
    """Build (or reuse) the extension in a workspace-local, sm_100-only cache.

    A build failure must not escape: correctness is available through
    `F.softmax` either way, and an import error would cost every captured case.
    """
    from torch.utils.cpp_extension import load_inline

    _BUILD_DIR.mkdir(parents=True, exist_ok=True)
    previous = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"] = "10.0"
    try:
        return load_inline(
            name=_EXT_NAME,
            cpp_sources=_CPP_SOURCE,
            cuda_sources=_CUDA_SOURCE,
            functions=["softmax_forward", "softmax_plan"],
            extra_cuda_cflags=["-O3", "-std=c++17", "--generate-line-info",
                               "--expt-relaxed-constexpr"],
            build_directory=str(_BUILD_DIR),
            verbose=False,
        )
    finally:
        if previous is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = previous


if os.environ.get("FK_SOFTMAX_DISABLE_EXT"):
    _EXT, _EXT_ERROR = None, "disabled by FK_SOFTMAX_DISABLE_EXT"
else:
    try:
        _EXT, _EXT_ERROR = _load_extension(), None
    except Exception as exc:  # noqa: BLE001 - a build failure must degrade, not raise
        _EXT, _EXT_ERROR = None, f"{type(exc).__name__}: {exc}"


def extension_loaded() -> bool:
    """Whether the fused kernels are available in this process."""
    return _EXT is not None


def extension_error() -> str | None:
    """Why the extension is unavailable, or None if it loaded."""
    return _EXT_ERROR


def plan_for(x: torch.Tensor, dim: int) -> dict[str, int]:
    """The routing decision `forward` would make for `(x, dim)`.

    `route` is one of `ROUTE_ROW`, `ROUTE_COLUMN` or `ROUTE_FALLBACK`; the rest
    describes the launch. Raises if the extension is not loaded, so a caller
    cannot mistake a build failure for a routing answer.
    """
    if _EXT is None:
        raise RuntimeError(f"extension not loaded: {_EXT_ERROR}")
    values = _EXT.softmax_plan(x, dim).tolist()
    return dict(zip(PLAN_FIELDS, values))


class Softmax(nn.Module):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim
        # Bind the implementation once so `forward` has no branch, no tensor
        # property query and no `.contiguous()` -- every decision is in C++.
        self._softmax = _EXT.softmax_forward if _EXT is not None else F.softmax

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._softmax(x, self.dim)


class LogSoftmax(nn.Module):
    """Numerically-stable log-softmax. Used by the TTT-E2E inner-loop CE loss.

    No capture report contains it, so it is never benchmarked; it stays a
    passthrough so the module matches the baseline's contract.
    """

    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(x, dim=self.dim)
