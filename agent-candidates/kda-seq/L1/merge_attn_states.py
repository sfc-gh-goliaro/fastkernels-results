"""Online softmax merge of two attention partitions, tuned for sm_100.

Semantically identical to the vendored vLLM kernel, including the parts random
inputs never reach: ``isinf(x) ? -inf : x`` folds +inf to -inf as well, ``fmaxf``
keeps the non-NaN operand, and a row whose two log-sum-exps are both infinite is
copied from ``prefix_output`` **bitwise** rather than computed, so -0.0 survives
and a NaN or Inf suffix cannot contaminate it.

The vendored kernel is instruction-bound rather than bandwidth-bound: it spends
two runtime integer divisions per thread on index math, and the sixteen threads
that cover one ``(token, head)`` row each recompute that row's softmax scales, so
both ``expf`` calls and both divisions happen sixteen times over. This version
keeps the vendored kernel's perfectly coalesced one-pack-per-thread access and
removes the arithmetic around it:

* the head count and head size are compile-time constants for the shape this
  operator actually sees, so both divisions become a shift and a mask;
* one lane per row computes that row's scales and degenerate flag and broadcasts
  them across the row with warp shuffles, so the transcendental work drops by the
  number of lanes per row;
* ``output_lse`` present and absent are separate specialisations, since most
  calls do not ask for it;
* the two 16-byte loads are issued before the scale phase, so the leader's
  log-sum-exp fetch and the data fetches overlap instead of forming two dependent
  memory round trips.

Carrying several rows per thread is available (``FK_MERGE_ROWS``) but measured
slower at every block size: this grid already runs about fourteen waves per SM, so
there is ample thread-level parallelism to hide latency with and the extra
registers only cost occupancy.

Everything the fast path cannot index correctly -- a padded head stride that is
not a whole number of packs, a non-contiguous log-sum-exp tensor, an outer stride
that is not ``num_heads * head_stride``, an unaligned pointer -- is routed to a
generic CUDA kernel or, failing that, to a PyTorch implementation of the same
semantics.

The CUDA extension is compiled from the string below on the first ``forward``, so
this file is the whole deliverable.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn

_EXTENSION_NAME = "merge_attn_states_fk"

# Launch shapes the specialised kernel is instantiated for. The defaults are the
# measured winners; FK_MERGE_THREADS / FK_MERGE_ROWS select another instantiation
# for sweeps and are ignored when they name a combination that was not compiled.
_THREAD_CHOICES = (128, 256, 512)
_ROW_CHOICES = (1, 2, 4, 8)
_VEC_CHOICES = (16, 32)
_DEFAULT_THREADS = 128
_DEFAULT_ROWS = 1
# 16-byte access with no cache hints; both alternatives are compiled in and
# selectable, and both measured slower here (see solutions.jsonl).
_DEFAULT_VEC = 16
# 0 plain, 1 L1::no_allocate on the two streaming reads, 2 also non-coherent.
# Chosen per call from the working-set size rather than fixed: see _stream_hint.
# Level 2 buys nothing measurable over level 1 and constrains aliasing, so it is
# selectable but never chosen by default.
_HINT_CHOICES = (0, 1, 2)
_DEFAULT_HINT = None
# 0 leader-broadcast flat kernel (the default), 1 fixed-head warp tile,
# 2 grid-stride over packs, 3 flattened-row warp tile. The alternatives are
# compiled in and selectable with FK_MERGE_KERNEL so they can be measured; all
# three lost (see solutions.jsonl).
_KERNEL_CHOICES = (0, 1, 2, 3)
_DEFAULT_KERNEL = 0
_DEFAULT_WAVES = 0
# Write policy for the output stream: 0 plain, 1 L1::evict_first.
_STORE_CHOICES = (0, 1)
_DEFAULT_STORE = 0
# Minimum resident CTAs per SM, forced through __launch_bounds__. 1 is the value
# the compiler already targets, so it is the no-constraint case; higher values cap
# registers to fit that many blocks. The largest value per block size is the one
# that fills sm_100's 2048 threads per SM, and so the only one whose ~32-register
# budget can actually bind this kernel.
_MIN_CTAS_CHOICES = (1, 2, 4, 8, 16)
_DEFAULT_MIN_CTAS = 1

_CUDA_SOURCE = r"""
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

namespace fk_merge {

// Element conversions matching the vendored kernel's: cvt.f32.bf16 /
// cvt.f32.f16 in, round-to-nearest-even out.
template <typename T>
struct Convert;

template <>
struct Convert<__nv_bfloat16> {
  static __device__ __forceinline__ float to_float(__nv_bfloat16 v) {
    return __bfloat162float(v);
  }
  static __device__ __forceinline__ __nv_bfloat16 from_float(float v) {
    return __float2bfloat16(v);
  }
};

template <>
struct Convert<__half> {
  static __device__ __forceinline__ float to_float(__half v) {
    return __half2float(v);
  }
  static __device__ __forceinline__ __half from_float(float v) {
    return __float2half(v);
  }
};

template <>
struct Convert<float> {
  static __device__ __forceinline__ float to_float(float v) { return v; }
  static __device__ __forceinline__ float from_float(float v) { return v; }
};

// One row's softmax weights, or a flag saying the row is copied verbatim.
struct RowScale {
  float prefix_weight;
  float suffix_weight;
  bool degenerate;
};

// The vendored kernel's scalar arithmetic, unchanged: the isinf folding maps
// +inf to -inf too, fmaxf keeps the non-NaN operand, and the two full-precision
// divisions are kept rather than replaced by a reciprocal so the result is bit
// for bit what the vendored kernel produces. Computing this once per row instead
// of once per lane is what makes the divisions affordable.
__device__ __forceinline__ RowScale row_scale(float p_lse, float s_lse,
                                              float* merged_lse) {
  p_lse = isinf(p_lse) ? -INFINITY : p_lse;
  s_lse = isinf(s_lse) ? -INFINITY : s_lse;
  const float max_lse = fmaxf(p_lse, s_lse);

  RowScale scale;
  scale.degenerate = isinf(max_lse);
  if (scale.degenerate) {
    scale.prefix_weight = 0.0f;
    scale.suffix_weight = 0.0f;
    if (merged_lse != nullptr) *merged_lse = max_lse;
    return scale;
  }
  // expf, never __expf: the fast intrinsic does not reproduce the Inf and NaN
  // behaviour this contract exposes.
  const float p_se = expf(p_lse - max_lse);
  const float s_se = expf(s_lse - max_lse);
  const float out_se = p_se + s_se;
  scale.prefix_weight = p_se / out_se;
  scale.suffix_weight = s_se / out_se;
  if (merged_lse != nullptr) *merged_lse = logf(out_se) + max_lse;
  return scale;
}

// One vector access. 16 bytes is ld.global.v4.u32; 32 bytes is ld.global.v4.u64,
// which is the only construct that lowers to a genuine 256-bit transaction on
// sm_100 rather than a pair of 128-bit ones.
//
// STORE selects the write policy: 0 is a plain store, 1 is `L1::evict_first`, which
// tells L1 to drop the line as soon as it is written -- `output` is never read back
// inside the kernel, so nothing is lost by not keeping it.
//
// HINT is a level rather than a flag because the two qualifiers carry different
// obligations. L1::no_allocate (level 1) only says a line that is never revisited
// should not evict one that is, which is unconditionally true here. `nc` (level 2)
// additionally routes the read through the non-coherent path, which is only sound
// if nothing writes the same memory during the launch -- so the wrapper refuses
// the fast path when `output` overlaps either input.
struct __align__(32) Vec32 {
  uint4 lo;
  uint4 hi;
};

template <int BYTES, int HINT, int STORE>
struct VecIO;

template <int HINT, int STORE>
struct VecIO<16, HINT, STORE> {
  using Type = uint4;
  static __device__ __forceinline__ uint4 load(const void* p) {
    uint4 v;
    if (HINT == 2) {
      asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                   : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                   : "l"(p));
    } else if (HINT == 1) {
      asm volatile("ld.global.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                   : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                   : "l"(p));
    } else {
      v = *reinterpret_cast<const uint4*>(p);
    }
    return v;
  }
  static __device__ __forceinline__ void store(void* p, const uint4& v) {
    if (STORE == 1) {
      asm volatile("st.global.L1::evict_first.v4.u32 [%0], {%1,%2,%3,%4};"
                   :
                   : "l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
                   : "memory");
    } else {
      *reinterpret_cast<uint4*>(p) = v;
    }
  }
};

template <int HINT, int STORE>
struct VecIO<32, HINT, STORE> {
  using Type = Vec32;
  static __device__ __forceinline__ Vec32 load(const void* p) {
    Vec32 v;
    uint64_t a, b, c, d;
    if (HINT == 2) {
      asm volatile("ld.global.nc.L1::no_allocate.v4.u64 {%0,%1,%2,%3}, [%4];"
                   : "=l"(a), "=l"(b), "=l"(c), "=l"(d)
                   : "l"(p));
    } else if (HINT == 1) {
      asm volatile("ld.global.L1::no_allocate.v4.u64 {%0,%1,%2,%3}, [%4];"
                   : "=l"(a), "=l"(b), "=l"(c), "=l"(d)
                   : "l"(p));
    } else {
      asm volatile("ld.global.v4.u64 {%0,%1,%2,%3}, [%4];"
                   : "=l"(a), "=l"(b), "=l"(c), "=l"(d)
                   : "l"(p));
    }
    v.lo.x = (uint32_t)a; v.lo.y = (uint32_t)(a >> 32);
    v.lo.z = (uint32_t)b; v.lo.w = (uint32_t)(b >> 32);
    v.hi.x = (uint32_t)c; v.hi.y = (uint32_t)(c >> 32);
    v.hi.z = (uint32_t)d; v.hi.w = (uint32_t)(d >> 32);
    return v;
  }
  static __device__ __forceinline__ void store(void* p, const Vec32& v) {
    const uint64_t a = (uint64_t)v.lo.x | ((uint64_t)v.lo.y << 32);
    const uint64_t b = (uint64_t)v.lo.z | ((uint64_t)v.lo.w << 32);
    const uint64_t c = (uint64_t)v.hi.x | ((uint64_t)v.hi.y << 32);
    const uint64_t d = (uint64_t)v.hi.z | ((uint64_t)v.hi.w << 32);
    if (STORE == 1) {
      asm volatile("st.global.L1::evict_first.v4.u64 [%0], {%1,%2,%3,%4};"
                   :
                   : "l"(p), "l"(a), "l"(b), "l"(c), "l"(d)
                   : "memory");
    } else {
      asm volatile("st.global.v4.u64 [%0], {%1,%2,%3,%4};"
                   :
                   : "l"(p), "l"(a), "l"(b), "l"(c), "l"(d)
                   : "memory");
    }
  }
};

// Weighted sum of one pack, in fp32, in the vendored kernel's operand order so
// nvcc contracts it into the same FMA.
template <typename T, int PACK, typename Vec>
__device__ __forceinline__ Vec merge_pack(const Vec& prefix, const Vec& suffix,
                                          float p_scale, float s_scale) {
  Vec merged;
  const T* p = reinterpret_cast<const T*>(&prefix);
  const T* s = reinterpret_cast<const T*>(&suffix);
  T* o = reinterpret_cast<T*>(&merged);
#pragma unroll
  for (int i = 0; i < PACK; ++i) {
    const float pf = Convert<T>::to_float(p[i]);
    const float sf = Convert<T>::to_float(s[i]);
    o[i] = Convert<T>::from_float(pf * p_scale + (sf * s_scale));
  }
  return merged;
}

// ---------------------------------------------------------------------------
// Specialised kernel: compile-time head count and head size, one leader lane per
// row broadcasting that row's scales, ROWS rows per thread with their loads
// issued before any of them is consumed.
//
// Lane assignment is the vendored kernel's: PACKS_PER_ROW consecutive lanes cover
// one row, so a block covers THREADS / PACKS_PER_ROW consecutive rows per pass
// and every access is a fully coalesced run of 16-byte packs.
// ---------------------------------------------------------------------------
template <typename T, int HEADS, int HEAD_SIZE, bool HAS_LSE, int THREADS, int ROWS,
          int VEC_BYTES = 16, int HINT = 0, int STORE = 0, int MIN_CTAS = 1>
__launch_bounds__(THREADS, MIN_CTAS) __global__ void merge_leader_broadcast(
    T* __restrict__ output, float* __restrict__ output_lse,
    const T* __restrict__ prefix_output, const float* __restrict__ prefix_lse,
    const T* __restrict__ suffix_output, const float* __restrict__ suffix_lse,
    const int num_tokens, const int in_head_stride, const int out_head_stride) {
  using IO = VecIO<VEC_BYTES, HINT, STORE>;
  using Vec = typename IO::Type;
  constexpr int PACK = VEC_BYTES / sizeof(T);
  constexpr int PACKS_PER_ROW = HEAD_SIZE / PACK;
  constexpr int ROWS_PER_PASS = THREADS / PACKS_PER_ROW;
  static_assert(HEAD_SIZE % PACK == 0, "head size must be a whole number of packs");
  static_assert(PACKS_PER_ROW > 0 && PACKS_PER_ROW <= 32,
                "a row must fit inside one warp");
  static_assert((PACKS_PER_ROW & (PACKS_PER_ROW - 1)) == 0,
                "lane-to-row mapping assumes a power-of-two pack count");
  static_assert(THREADS % PACKS_PER_ROW == 0 && THREADS % 32 == 0,
                "block must hold a whole number of rows and warps");
  static_assert((HEADS & (HEADS - 1)) == 0, "token/head split assumes a power of two");

  const int total_rows = num_tokens * HEADS;
  const int pack = threadIdx.x % PACKS_PER_ROW;          // mask
  const int row_in_block = threadIdx.x / PACKS_PER_ROW;  // shift
  const int first_row =
      blockIdx.x * (ROWS_PER_PASS * ROWS) + row_in_block;

  const int lane = threadIdx.x & 31;
  const int leader_lane = lane & ~(PACKS_PER_ROW - 1);
  const bool is_leader = lane == leader_lane;

  // Data phase first. Nothing here depends on the scales, and __shfl_sync is a
  // scheduling barrier the compiler will not hoist a global load across -- so
  // issuing the packs before the scale phase is what keeps the data loads and
  // the leader's log-sum-exp load in flight together instead of turning them
  // into two dependent memory round trips. Both packs are loaded
  // unconditionally: the suffix address is valid whether or not the row turns
  // out to be degenerate.
  Vec prefix_pack[ROWS];
  Vec suffix_pack[ROWS];
  bool active[ROWS];
#pragma unroll
  for (int i = 0; i < ROWS; ++i) {
    const int row = first_row + i * ROWS_PER_PASS;
    active[i] = row < total_rows;
    if (active[i]) {
      const int src = row * in_head_stride + pack * PACK;
      prefix_pack[i] = IO::load(prefix_output + src);
      suffix_pack[i] = IO::load(suffix_output + src);
    }
  }

  // Scale phase. Every lane of the warp reaches every shuffle -- an early exit
  // here would hand the surviving lanes undefined scales, which is exactly how a
  // token count that is not a whole number of warps would break.
  float prefix_weight[ROWS];
  float suffix_weight[ROWS];
  bool degenerate[ROWS];
#pragma unroll
  for (int i = 0; i < ROWS; ++i) {
    const int row = first_row + i * ROWS_PER_PASS;
    RowScale scale = {0.0f, 0.0f, false};
    if (is_leader && row < total_rows) {
      const int token = row / HEADS;  // shift
      const int head = row % HEADS;   // mask
      const int lse_idx = head * num_tokens + token;
      float* merged = HAS_LSE ? output_lse + lse_idx : nullptr;
      scale = row_scale(prefix_lse[lse_idx], suffix_lse[lse_idx], merged);
    }
    prefix_weight[i] = __shfl_sync(0xffffffffu, scale.prefix_weight, leader_lane);
    suffix_weight[i] = __shfl_sync(0xffffffffu, scale.suffix_weight, leader_lane);
    degenerate[i] =
        __shfl_sync(0xffffffffu, static_cast<int>(scale.degenerate), leader_lane) != 0;
  }
#pragma unroll
  for (int i = 0; i < ROWS; ++i) {
    if (!active[i]) continue;
    const int dst =
        (first_row + i * ROWS_PER_PASS) * out_head_stride + pack * PACK;
    // The degenerate row stays a branch and moves raw bits: weights of (1, 0)
    // would turn -0.0 into +0.0 and let a NaN or Inf suffix contaminate a row the
    // contract copies verbatim.
    const Vec value =
        degenerate[i] ? prefix_pack[i]
                      : merge_pack<T, PACK, Vec>(prefix_pack[i], suffix_pack[i],
                                                 prefix_weight[i], suffix_weight[i]);
    IO::store(output + dst, value);
  }
}

// ---------------------------------------------------------------------------
// Fixed-head warp tile: one warp owns TILE consecutive tokens at a single head.
//
// The point is the log-sum-exp side. In the flat kernel a block's rows differ in
// head, so their LSE entries sit num_tokens floats apart and each 4-byte read
// pulls its own sector; here lane i reads token t0+i at a fixed head, so the
// whole warp's scale phase is one contiguous 128-byte line per tensor and the
// output_lse stores are contiguous too, which is what the lse-present case pays
// for in the flat kernel.
//
// The data side is unchanged in cost: the two half-warp rows are num_heads head
// strides apart, but each is its own 256-byte aligned segment, so the line count
// per instruction is the same.
// ---------------------------------------------------------------------------
template <typename T, int HEADS, int HEAD_SIZE, bool HAS_LSE, int THREADS, int TILE>
__launch_bounds__(THREADS) __global__ void merge_head_tile(
    T* __restrict__ output, float* __restrict__ output_lse,
    const T* __restrict__ prefix_output, const float* __restrict__ prefix_lse,
    const T* __restrict__ suffix_output, const float* __restrict__ suffix_lse,
    const int num_tokens, const int in_head_stride, const int out_head_stride,
    const int tiles_per_head) {
  constexpr int PACK = 16 / sizeof(T);
  constexpr int PACKS_PER_ROW = HEAD_SIZE / PACK;
  constexpr int TOKENS_PER_ITER = 32 / PACKS_PER_ROW;
  constexpr int ITERS = TILE / TOKENS_PER_ITER;
  static_assert(TILE == 32, "the scale phase maps one token to one lane");
  static_assert(PACKS_PER_ROW > 0 && PACKS_PER_ROW <= 32, "a row must fit a warp");
  static_assert(TILE % TOKENS_PER_ITER == 0, "tile must divide into whole passes");

  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x & 31;
  const int tile = blockIdx.x * (THREADS / 32) + warp;
  if (tile >= tiles_per_head) return;  // uniform across the warp
  const int head = blockIdx.y;
  const int first_token = tile * TILE;

  const int pack = lane % PACKS_PER_ROW;
  const int lane_token = lane / PACKS_PER_ROW;

  // The first iteration's packs are issued before the scale phase. __shfl_sync is
  // a scheduling barrier the compiler will not hoist a global load across, so
  // without this the whole tile waits on the leader's log-sum-exp fetch before it
  // requests a single byte of data.
  const int first_sub = lane_token;
  const int first_t = first_token + first_sub;
  const bool first_valid = first_t < num_tokens;
  const int first_src = first_valid
                            ? (first_t * HEADS + head) * in_head_stride + pack * PACK
                            : 0;
  uint4 pending_prefix, pending_suffix;
  if (first_valid) {
    pending_prefix = *reinterpret_cast<const uint4*>(prefix_output + first_src);
    pending_suffix = *reinterpret_cast<const uint4*>(suffix_output + first_src);
  }

  // Scale phase: lane i owns token first_token + i, so both log-sum-exp reads
  // are one coalesced line and the merged write is one coalesced line.
  const int token = first_token + lane;
  float prefix_weight = 0.0f, suffix_weight = 0.0f;
  bool degenerate = false;
  if (token < num_tokens) {
    const int lse_idx = head * num_tokens + token;
    float* merged = HAS_LSE ? output_lse + lse_idx : nullptr;
    const RowScale scale =
        row_scale(prefix_lse[lse_idx], suffix_lse[lse_idx], merged);
    prefix_weight = scale.prefix_weight;
    suffix_weight = scale.suffix_weight;
    degenerate = scale.degenerate;
  }

  // Data phase. The shuffles run unconditionally -- a lane that steps past the
  // token count must still reach them, or the lanes that did not would read
  // undefined scales, which is how a token count that is not a whole number of
  // warps breaks.
#pragma unroll
  for (int j = 0; j < ITERS; ++j) {
    const int sub = j * TOKENS_PER_ITER + lane_token;
    const float p_w = __shfl_sync(0xffffffffu, prefix_weight, sub);
    const float s_w = __shfl_sync(0xffffffffu, suffix_weight, sub);
    const bool degen =
        __shfl_sync(0xffffffffu, static_cast<int>(degenerate), sub) != 0;
    const int t = first_token + sub;
    if (t >= num_tokens) continue;
    const int row = t * HEADS + head;
    const int src = row * in_head_stride + pack * PACK;
    const uint4 prefix_pack =
        j == 0 ? pending_prefix
               : *reinterpret_cast<const uint4*>(prefix_output + src);
    const uint4 suffix_pack =
        j == 0 ? pending_suffix
               : *reinterpret_cast<const uint4*>(suffix_output + src);
    const uint4 value =
        degen ? prefix_pack
              : merge_pack<T, PACK, uint4>(prefix_pack, suffix_pack, p_w, s_w);
    *reinterpret_cast<uint4*>(output + row * out_head_stride + pack * PACK) = value;
  }
}

// ---------------------------------------------------------------------------
// Flattened-row tile: one warp owns TILE consecutive *flattened* (token, head)
// rows, the alternative the plan asks to be measured against the fixed-head tile.
//
// It trades the two sides of the fixed-head tile. The data side improves: rows are
// consecutive in memory, so a warp's TILE rows are one contiguous run rather than
// TILE segments num_heads head strides apart. The log-sum-exp side gets worse:
// consecutive flattened rows step the head index, so a warp's 32 rows are two adjacent
// tokens at each of HEADS heads. The two entries one head needs are neighbours and
// share a 32-byte sector, so the scale phase costs HEADS sectors per instruction
// against the fixed-head tile's four -- a 4x over-fetch, measured and reconciled in
// profile/p1_leader_broadcast/analysis/tile_lse_overfetch.txt. The output_lse stores
// scatter the same way.
// ---------------------------------------------------------------------------
template <typename T, int HEADS, int HEAD_SIZE, bool HAS_LSE, int THREADS, int TILE>
__launch_bounds__(THREADS) __global__ void merge_flat_tile(
    T* __restrict__ output, float* __restrict__ output_lse,
    const T* __restrict__ prefix_output, const float* __restrict__ prefix_lse,
    const T* __restrict__ suffix_output, const float* __restrict__ suffix_lse,
    const int num_tokens, const int in_head_stride, const int out_head_stride) {
  constexpr int PACK = 16 / sizeof(T);
  constexpr int PACKS_PER_ROW = HEAD_SIZE / PACK;
  constexpr int ROWS_PER_ITER = 32 / PACKS_PER_ROW;
  constexpr int ITERS = TILE / ROWS_PER_ITER;
  static_assert(TILE == 32, "the scale phase maps one row to one lane");
  static_assert(PACKS_PER_ROW > 0 && PACKS_PER_ROW <= 32, "a row must fit a warp");
  static_assert(TILE % ROWS_PER_ITER == 0, "tile must divide into whole passes");
  static_assert((HEADS & (HEADS - 1)) == 0, "token/head split assumes a power of two");

  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x & 31;
  const int total_rows = num_tokens * HEADS;
  const int first_row = (blockIdx.x * (THREADS / 32) + warp) * TILE;
  if (first_row >= total_rows) return;  // uniform across the warp

  const int pack = lane % PACKS_PER_ROW;
  const int lane_row = lane / PACKS_PER_ROW;

  // First iteration's packs before the scale phase, for the same reason as above.
  const int first_target = first_row + lane_row;
  const bool first_valid = first_target < total_rows;
  const int first_src =
      first_valid ? first_target * in_head_stride + pack * PACK : 0;
  uint4 pending_prefix, pending_suffix;
  if (first_valid) {
    pending_prefix = *reinterpret_cast<const uint4*>(prefix_output + first_src);
    pending_suffix = *reinterpret_cast<const uint4*>(suffix_output + first_src);
  }

  // Scale phase: lane i owns flattened row first_row + i. Its log-sum-exp index is
  // head * num_tokens + token, and consecutive rows differ in head, so these reads
  // are strided rather than contiguous -- the cost this mapping pays.
  const int my_row = first_row + lane;
  float prefix_weight = 0.0f, suffix_weight = 0.0f;
  bool degenerate = false;
  if (my_row < total_rows) {
    const int token = my_row / HEADS;  // shift
    const int head = my_row % HEADS;   // mask
    const int lse_idx = head * num_tokens + token;
    float* merged = HAS_LSE ? output_lse + lse_idx : nullptr;
    const RowScale scale =
        row_scale(prefix_lse[lse_idx], suffix_lse[lse_idx], merged);
    prefix_weight = scale.prefix_weight;
    suffix_weight = scale.suffix_weight;
    degenerate = scale.degenerate;
  }

#pragma unroll
  for (int j = 0; j < ITERS; ++j) {
    const int sub = j * ROWS_PER_ITER + lane_row;
    const float p_w = __shfl_sync(0xffffffffu, prefix_weight, sub);
    const float s_w = __shfl_sync(0xffffffffu, suffix_weight, sub);
    const bool degen =
        __shfl_sync(0xffffffffu, static_cast<int>(degenerate), sub) != 0;
    const int row = first_row + sub;
    if (row >= total_rows) continue;
    const int src = row * in_head_stride + pack * PACK;
    const uint4 prefix_pack =
        j == 0 ? pending_prefix
               : *reinterpret_cast<const uint4*>(prefix_output + src);
    const uint4 suffix_pack =
        j == 0 ? pending_suffix
               : *reinterpret_cast<const uint4*>(suffix_output + src);
    const uint4 value =
        degen ? prefix_pack
              : merge_pack<T, PACK, uint4>(prefix_pack, suffix_pack, p_w, s_w);
    *reinterpret_cast<uint4*>(output + row * out_head_stride + pack * PACK) = value;
  }
}

// ---------------------------------------------------------------------------
// Grid-stride over packs, kept as a measured comparison point rather than as a
// candidate. Each thread walks several packs with a grid stride and computes its
// own scales, which is what the flat mapping looks like without the leader: the
// iterations land in different rows, so U packs per thread means U independent
// scale computations and the transcendental work per byte is unchanged.
// ---------------------------------------------------------------------------
template <typename T, int HEADS, int HEAD_SIZE, bool HAS_LSE, int THREADS>
__launch_bounds__(THREADS) __global__ void merge_grid_stride(
    T* __restrict__ output, float* __restrict__ output_lse,
    const T* __restrict__ prefix_output, const float* __restrict__ prefix_lse,
    const T* __restrict__ suffix_output, const float* __restrict__ suffix_lse,
    const int num_tokens, const int in_head_stride, const int out_head_stride) {
  constexpr int PACK = 16 / sizeof(T);
  constexpr int PACKS_PER_ROW = HEAD_SIZE / PACK;
  const int total = num_tokens * HEADS * PACKS_PER_ROW;
  const int stride = THREADS * gridDim.x;
  for (int idx = blockIdx.x * THREADS + threadIdx.x; idx < total; idx += stride) {
    const int row = idx / PACKS_PER_ROW;  // shift
    const int pack = idx % PACKS_PER_ROW;  // mask
    const int token = row / HEADS;
    const int head = row % HEADS;
    const int lse_idx = head * num_tokens + token;
    float* merged = HAS_LSE && pack == 0 ? output_lse + lse_idx : nullptr;
    const RowScale scale =
        row_scale(prefix_lse[lse_idx], suffix_lse[lse_idx], merged);
    const int src = row * in_head_stride + pack * PACK;
    const uint4 prefix_pack = *reinterpret_cast<const uint4*>(prefix_output + src);
    const uint4 suffix_pack = *reinterpret_cast<const uint4*>(suffix_output + src);
    const uint4 value =
        scale.degenerate
            ? prefix_pack
            : merge_pack<T, PACK, uint4>(prefix_pack, suffix_pack,
                                         scale.prefix_weight, scale.suffix_weight);
    *reinterpret_cast<uint4*>(output + row * out_head_stride + pack * PACK) = value;
  }
}

// ---------------------------------------------------------------------------
// Generic kernel: runtime head count, head size and strides, one thread per
// (row, pack) with a grid-stride loop. Reached by dtypes and shapes outside the
// specialisation, where being correct matters and being fast does not.
// ---------------------------------------------------------------------------
template <typename T>
__global__ void merge_generic(
    T* __restrict__ output, float* __restrict__ output_lse,
    const T* __restrict__ prefix_output, const float* __restrict__ prefix_lse,
    const T* __restrict__ suffix_output, const float* __restrict__ suffix_lse,
    const int num_tokens, const int num_heads, const int packs_per_row,
    const int in_head_stride, const int out_head_stride,
    const int64_t total_threads) {
  constexpr int PACK = 16 / sizeof(T);
  for (int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
       idx < total_threads; idx += (int64_t)blockDim.x * gridDim.x) {
    const int64_t row = idx / packs_per_row;
    const int pack = static_cast<int>(idx % packs_per_row);
    const int token = static_cast<int>(row / num_heads);
    const int head = static_cast<int>(row % num_heads);
    const int lse_idx = head * num_tokens + token;

    float* merged = output_lse != nullptr && pack == 0 ? output_lse + lse_idx : nullptr;
    const RowScale scale =
        row_scale(prefix_lse[lse_idx], suffix_lse[lse_idx], merged);

    const int64_t src = row * in_head_stride + (int64_t)pack * PACK;
    const int64_t dst = row * out_head_stride + (int64_t)pack * PACK;
    const uint4 prefix_pack = *reinterpret_cast<const uint4*>(prefix_output + src);
    if (scale.degenerate) {
      *reinterpret_cast<uint4*>(output + dst) = prefix_pack;
    } else {
      const uint4 suffix_pack = *reinterpret_cast<const uint4*>(suffix_output + src);
      *reinterpret_cast<uint4*>(output + dst) = merge_pack<T, PACK>(
          prefix_pack, suffix_pack, scale.prefix_weight, scale.suffix_weight);
    }
  }
}

// ---------------------------------------------------------------------------
// Launch.
// ---------------------------------------------------------------------------
constexpr int SPECIALISED_HEADS = 16;
constexpr int SPECIALISED_HEAD_SIZE = 128;

struct LaunchArgs {
  void* output;
  float* output_lse;
  const void* prefix_output;
  const float* prefix_lse;
  const void* suffix_output;
  const float* suffix_lse;
  int num_tokens;
  int in_head_stride;
  int out_head_stride;
  int64_t total_rows;
  cudaStream_t stream;
};

template <bool HAS_LSE, int THREADS, int ROWS, int VEC_BYTES, int HINT, int STORE = 0,
          int MIN_CTAS = 1>
void launch_specialised_bf16(const LaunchArgs& a) {
  using T = __nv_bfloat16;
  constexpr int PACKS_PER_ROW = SPECIALISED_HEAD_SIZE / (VEC_BYTES / sizeof(T));
  constexpr int ROWS_PER_BLOCK = (THREADS / PACKS_PER_ROW) * ROWS;
  const int blocks =
      static_cast<int>((a.total_rows + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK);
  merge_leader_broadcast<T, SPECIALISED_HEADS, SPECIALISED_HEAD_SIZE, HAS_LSE,
                         THREADS, ROWS, VEC_BYTES, HINT, STORE, MIN_CTAS>
      <<<blocks, THREADS, 0, a.stream>>>(
          static_cast<T*>(a.output), a.output_lse,
          static_cast<const T*>(a.prefix_output), a.prefix_lse,
          static_cast<const T*>(a.suffix_output), a.suffix_lse, a.num_tokens,
          a.in_head_stride, a.out_head_stride);
}

// Every launch shape the Python side may select. Compiling the whole set once
// keeps the sweep to an environment variable rather than a rebuild; an
// unrecognised pair returns false and takes the generic kernel instead.
template <bool HAS_LSE, int THREADS, int VEC_BYTES, int HINT>
bool dispatch_rows(const LaunchArgs& a, int rows) {
  switch (rows) {
    case 1: launch_specialised_bf16<HAS_LSE, THREADS, 1, VEC_BYTES, HINT>(a); return true;
    case 2: launch_specialised_bf16<HAS_LSE, THREADS, 2, VEC_BYTES, HINT>(a); return true;
    case 4: launch_specialised_bf16<HAS_LSE, THREADS, 4, VEC_BYTES, HINT>(a); return true;
    case 8: launch_specialised_bf16<HAS_LSE, THREADS, 8, VEC_BYTES, HINT>(a); return true;
    default: return false;
  }
}

template <bool HAS_LSE, int THREADS>
void launch_head_tile_bf16(const LaunchArgs& a, int num_heads) {
  using T = __nv_bfloat16;
  constexpr int TILE = 32;
  const int tiles_per_head = (a.num_tokens + TILE - 1) / TILE;
  const int warps = THREADS / 32;
  const dim3 grid((tiles_per_head + warps - 1) / warps, num_heads);
  merge_head_tile<T, SPECIALISED_HEADS, SPECIALISED_HEAD_SIZE, HAS_LSE, THREADS,
                  TILE><<<grid, THREADS, 0, a.stream>>>(
      static_cast<T*>(a.output), a.output_lse,
      static_cast<const T*>(a.prefix_output), a.prefix_lse,
      static_cast<const T*>(a.suffix_output), a.suffix_lse, a.num_tokens,
      a.in_head_stride, a.out_head_stride, tiles_per_head);
}

template <bool HAS_LSE, int THREADS>
void launch_flat_tile_bf16(const LaunchArgs& a) {
  using T = __nv_bfloat16;
  constexpr int TILE = 32;
  constexpr int WARPS = THREADS / 32;
  const int64_t tiles = (a.total_rows + TILE - 1) / TILE;
  const int blocks = static_cast<int>((tiles + WARPS - 1) / WARPS);
  merge_flat_tile<T, SPECIALISED_HEADS, SPECIALISED_HEAD_SIZE, HAS_LSE, THREADS, TILE>
      <<<blocks, THREADS, 0, a.stream>>>(
          static_cast<T*>(a.output), a.output_lse,
          static_cast<const T*>(a.prefix_output), a.prefix_lse,
          static_cast<const T*>(a.suffix_output), a.suffix_lse, a.num_tokens,
          a.in_head_stride, a.out_head_stride);
}

template <bool HAS_LSE, int THREADS>
void launch_grid_stride_bf16(const LaunchArgs& a, int waves) {
  using T = __nv_bfloat16;
  constexpr int PACKS_PER_ROW = SPECIALISED_HEAD_SIZE / (16 / sizeof(T));
  const int64_t total = a.total_rows * PACKS_PER_ROW;
  const int64_t full = (total + THREADS - 1) / THREADS;
  int blocks = static_cast<int>(full);
  if (waves > 0) {
    // Resident-CTA form: enough blocks for `waves` per SM and no more, so each
    // thread walks several packs instead of one.
    int sms = 0;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    const int want = sms * waves;
    blocks = want < blocks ? want : blocks;
  }
  merge_grid_stride<T, SPECIALISED_HEADS, SPECIALISED_HEAD_SIZE, HAS_LSE, THREADS>
      <<<blocks, THREADS, 0, a.stream>>>(
          static_cast<T*>(a.output), a.output_lse,
          static_cast<const T*>(a.prefix_output), a.prefix_lse,
          static_cast<const T*>(a.suffix_output), a.suffix_lse, a.num_tokens,
          a.in_head_stride, a.out_head_stride);
}

template <bool HAS_LSE>
bool dispatch_alternate(const LaunchArgs& a, int kernel, int threads, int waves,
                        int num_heads) {
  if (kernel == 1) {  // fixed-head warp tile
    switch (threads) {
      case 128: launch_head_tile_bf16<HAS_LSE, 128>(a, num_heads); return true;
      case 256: launch_head_tile_bf16<HAS_LSE, 256>(a, num_heads); return true;
      case 512: launch_head_tile_bf16<HAS_LSE, 512>(a, num_heads); return true;
      default: return false;
    }
  }
  if (kernel == 3) {  // flattened-row tile
    switch (threads) {
      case 128: launch_flat_tile_bf16<HAS_LSE, 128>(a); return true;
      case 256: launch_flat_tile_bf16<HAS_LSE, 256>(a); return true;
      case 512: launch_flat_tile_bf16<HAS_LSE, 512>(a); return true;
      default: return false;
    }
  }
  if (kernel == 2) {  // grid-stride over packs
    switch (threads) {
      case 128: launch_grid_stride_bf16<HAS_LSE, 128>(a, waves); return true;
      case 256: launch_grid_stride_bf16<HAS_LSE, 256>(a, waves); return true;
      case 512: launch_grid_stride_bf16<HAS_LSE, 512>(a, waves); return true;
      default: return false;
    }
  }
  return false;
}

template <bool HAS_LSE, int VEC_BYTES, int HINT>
bool dispatch_threads(const LaunchArgs& a, int threads, int rows) {
  switch (threads) {
    case 128: return dispatch_rows<HAS_LSE, 128, VEC_BYTES, HINT>(a, rows);
    case 256: return dispatch_rows<HAS_LSE, 256, VEC_BYTES, HINT>(a, rows);
    case 512: return dispatch_rows<HAS_LSE, 512, VEC_BYTES, HINT>(a, rows);
    default: return false;
  }
}

template <bool HAS_LSE, int VEC_BYTES>
bool dispatch_hint(const LaunchArgs& a, int threads, int rows, int hint) {
  switch (hint) {
    case 0: return dispatch_threads<HAS_LSE, VEC_BYTES, 0>(a, threads, rows);
    case 1: return dispatch_threads<HAS_LSE, VEC_BYTES, 1>(a, threads, rows);
    case 2: return dispatch_threads<HAS_LSE, VEC_BYTES, 2>(a, threads, rows);
    default: return false;
  }
}

// The store-policy and residency experiments are dispatched separately rather than
// added to the matrix above: crossing five dimensions would multiply the compiled
// kernel count into the hundreds, and each experiment only needs the read policy
// that actually ships (L1::no_allocate) held fixed.
template <bool HAS_LSE, int VEC_BYTES>
bool dispatch_store_policy(const LaunchArgs& a, int threads) {
  switch (threads) {
    case 128:
      launch_specialised_bf16<HAS_LSE, 128, 1, VEC_BYTES, 1, 1>(a);
      return true;
    case 256:
      launch_specialised_bf16<HAS_LSE, 256, 1, VEC_BYTES, 1, 1>(a);
      return true;
    default:
      return false;
  }
}

// Residency points that fit sm_100's 2048 threads per SM.
#define FK_RESIDENCY_ROWS(THREADS, MIN_CTAS)                                   \
  switch (rows) {                                                              \
    case 1: launch_specialised_bf16<HAS_LSE, THREADS, 1, 16, 1, 0, MIN_CTAS>(a); return true; \
    case 2: launch_specialised_bf16<HAS_LSE, THREADS, 2, 16, 1, 0, MIN_CTAS>(a); return true; \
    case 4: launch_specialised_bf16<HAS_LSE, THREADS, 4, 16, 1, 0, MIN_CTAS>(a); return true; \
    case 8: launch_specialised_bf16<HAS_LSE, THREADS, 8, 16, 1, 0, MIN_CTAS>(a); return true; \
    default: return false;                                                     \
  }

template <bool HAS_LSE>
bool dispatch_residency(const LaunchArgs& a, int threads, int rows, int min_ctas) {
  if (threads == 128) {
    switch (min_ctas) {
      case 1: FK_RESIDENCY_ROWS(128, 1)
      case 2: FK_RESIDENCY_ROWS(128, 2)
      case 4: FK_RESIDENCY_ROWS(128, 4)
      case 8: FK_RESIDENCY_ROWS(128, 8)
      case 16: FK_RESIDENCY_ROWS(128, 16)  // 2048 threads/SM, ~32-register budget
      default: return false;
    }
  }
  if (threads == 256) {
    switch (min_ctas) {
      case 1: FK_RESIDENCY_ROWS(256, 1)
      case 2: FK_RESIDENCY_ROWS(256, 2)
      case 4: FK_RESIDENCY_ROWS(256, 4)
      case 8: FK_RESIDENCY_ROWS(256, 8)  // 2048 threads/SM
      default: return false;
    }
  }
  if (threads == 512) {
    switch (min_ctas) {
      case 1: FK_RESIDENCY_ROWS(512, 1)
      case 2: FK_RESIDENCY_ROWS(512, 2)
      case 4: FK_RESIDENCY_ROWS(512, 4)  // 2048 threads/SM
      default: return false;
    }
  }
  return false;
}

template <bool HAS_LSE>
bool dispatch_access(const LaunchArgs& a, int threads, int rows, int vec_bytes,
                     int hint) {
  if (vec_bytes == 32) return dispatch_hint<HAS_LSE, 32>(a, threads, rows, hint);
  if (vec_bytes == 16) return dispatch_hint<HAS_LSE, 16>(a, threads, rows, hint);
  return false;
}

template <typename T>
void launch_generic(const LaunchArgs& a, int num_heads, int packs_per_row,
                    int64_t total_threads) {
  constexpr int THREADS = 256;
  const int64_t want = (total_threads + THREADS - 1) / THREADS;
  const int blocks = static_cast<int>(want < 65535 ? want : 65535);
  merge_generic<T><<<blocks, THREADS, 0, a.stream>>>(
      static_cast<T*>(a.output), a.output_lse,
      static_cast<const T*>(a.prefix_output), a.prefix_lse,
      static_cast<const T*>(a.suffix_output), a.suffix_lse, a.num_tokens, num_heads,
      packs_per_row, a.in_head_stride, a.out_head_stride, total_threads);
}

void merge_attn_states_fk(at::Tensor output, std::optional<at::Tensor> output_lse,
                          at::Tensor prefix_output, at::Tensor prefix_lse,
                          at::Tensor suffix_output, at::Tensor suffix_lse,
                          int64_t threads_per_block, int64_t rows_per_thread,
                          int64_t vec_bytes, int64_t hint_level, int64_t kernel_id,
                          int64_t waves, int64_t store_policy, int64_t min_ctas) {
  const int num_tokens = static_cast<int>(output.size(0));
  const int num_heads = static_cast<int>(output.size(1));
  const int head_size = static_cast<int>(output.size(2));

  const at::cuda::CUDAGuard device_guard(prefix_output.device());

  LaunchArgs a;
  a.output = output.data_ptr();
  a.output_lse = output_lse.has_value() ? output_lse->data_ptr<float>() : nullptr;
  a.prefix_output = prefix_output.data_ptr();
  a.prefix_lse = prefix_lse.data_ptr<float>();
  a.suffix_output = suffix_output.data_ptr();
  a.suffix_lse = suffix_lse.data_ptr<float>();
  a.num_tokens = num_tokens;
  a.in_head_stride = static_cast<int>(prefix_output.stride(1));
  a.out_head_stride = static_cast<int>(output.stride(1));
  a.total_rows = (int64_t)num_tokens * num_heads;
  a.stream = at::cuda::getCurrentCUDAStream();

  const bool specialised = num_heads == SPECIALISED_HEADS &&
                           head_size == SPECIALISED_HEAD_SIZE &&
                           output.scalar_type() == at::kBFloat16;
  if (specialised) {
    const int threads = static_cast<int>(threads_per_block);
    const int rows = static_cast<int>(rows_per_thread);
    const int vec = static_cast<int>(vec_bytes);
    const int kernel = static_cast<int>(kernel_id);
    if (kernel != 0) {
      const bool alt =
          a.output_lse != nullptr
              ? dispatch_alternate<true>(a, kernel, threads,
                                         static_cast<int>(waves), num_heads)
              : dispatch_alternate<false>(a, kernel, threads,
                                          static_cast<int>(waves), num_heads);
      if (alt) return;
    }
    if (min_ctas > 1) {
      const int mc = static_cast<int>(min_ctas);
      const bool res = a.output_lse != nullptr
                           ? dispatch_residency<true>(a, threads, rows, mc)
                           : dispatch_residency<false>(a, threads, rows, mc);
      if (res) return;
    }
    if (store_policy == 1) {
      const bool st =
          vec == 32
              ? (a.output_lse != nullptr
                     ? dispatch_store_policy<true, 32>(a, threads)
                     : dispatch_store_policy<false, 32>(a, threads))
              : (a.output_lse != nullptr
                     ? dispatch_store_policy<true, 16>(a, threads)
                     : dispatch_store_policy<false, 16>(a, threads));
      if (st) return;
    }
    const bool launched =
        a.output_lse != nullptr
            ? dispatch_access<true>(a, threads, rows, vec,
                                    static_cast<int>(hint_level))
            : dispatch_access<false>(a, threads, rows, vec,
                                     static_cast<int>(hint_level));
    if (launched) return;
  }

  const int pack = 16 / static_cast<int>(output.element_size());
  const int packs_per_row = head_size / pack;
  const int64_t total_threads = a.total_rows * packs_per_row;
  switch (output.scalar_type()) {
    case at::kBFloat16:
      launch_generic<__nv_bfloat16>(a, num_heads, packs_per_row, total_threads);
      break;
    case at::kHalf:
      launch_generic<__half>(a, num_heads, packs_per_row, total_threads);
      break;
    case at::kFloat:
      launch_generic<float>(a, num_heads, packs_per_row, total_threads);
      break;
    default:
      TORCH_CHECK(false, "merge_attn_states_fk: unsupported dtype ",
                  output.scalar_type());
  }
}

}  // namespace fk_merge

void merge_attn_states_fk(at::Tensor output, std::optional<at::Tensor> output_lse,
                          at::Tensor prefix_output, at::Tensor prefix_lse,
                          at::Tensor suffix_output, at::Tensor suffix_lse,
                          int64_t threads_per_block, int64_t rows_per_thread,
                          int64_t vec_bytes, int64_t hint_level, int64_t kernel_id,
                          int64_t waves, int64_t store_policy, int64_t min_ctas) {
  fk_merge::merge_attn_states_fk(output, output_lse, prefix_output, prefix_lse,
                                 suffix_output, suffix_lse, threads_per_block,
                                 rows_per_thread, vec_bytes, hint_level, kernel_id,
                                 waves, store_policy, min_ctas);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""

_CPP_SOURCE = r"""
#include <torch/extension.h>

void merge_attn_states_fk(at::Tensor output, std::optional<at::Tensor> output_lse,
                          at::Tensor prefix_output, at::Tensor prefix_lse,
                          at::Tensor suffix_output, at::Tensor suffix_lse,
                          int64_t threads_per_block, int64_t rows_per_thread,
                          int64_t vec_bytes, int64_t hint_level, int64_t kernel_id,
                          int64_t waves, int64_t store_policy, int64_t min_ctas);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("merge_attn_states_fk", &merge_attn_states_fk,
        "merge_attn_states (leader-broadcast CUDA kernel)");
}
"""


_l2_bytes: dict[torch.device, int] = {}


def _stream_hint(prefix_output: torch.Tensor) -> int:
    """Whether to tell L1 not to allocate lines for the two streaming reads.

    Worth it only when the data genuinely streams. The three data tensors are read
    or written exactly once each, so when they do not fit in L2 a line is never
    revisited and keeping it only evicts something useful -- that is the captured
    16384- and 14128-token shapes, and the hint is worth about 5% there. When the
    whole working set does fit in L2, though, a second call really can find its
    inputs resident (the attention kernels that produced them ran moments before),
    and bypassing L1 then costs a re-fetch per access: measured 19.6us -> 24.6us
    with a warm L2 on the 7359-token shape, which is the one that fits.
    """
    device = prefix_output.device
    l2 = _l2_bytes.get(device)
    if l2 is None:
        # Keyed by device: this machine's GPUs need not share an L2 size, and a
        # single cached value would carry one device's threshold onto another.
        l2 = torch.cuda.get_device_properties(device).L2_cache_size
        _l2_bytes[device] = l2
    working_set = 3 * prefix_output.numel() * prefix_output.element_size()
    return 1 if working_set > l2 else 0


def _local_arch() -> str:
    """The running device's architecture, in the 'a' variant sm_100 features need."""
    major, minor = torch.cuda.get_device_capability()
    suffix = "a" if major in (9, 10, 12) else ""
    return f"{major}.{minor}{suffix}"


_extension = None


def _ext():
    """JIT-compile on first use, so nothing is built at import time."""
    global _extension
    if _extension is None:
        from torch.utils.cpp_extension import load_inline

        # Mirror the project loader's flags, and keep line information so the
        # candidate can be attributed in a profile.
        os.environ["TORCH_CUDA_ARCH_LIST"] = _local_arch()
        _extension = load_inline(
            name=_EXTENSION_NAME,
            cpp_sources=_CPP_SOURCE,
            cuda_sources=_CUDA_SOURCE,
            extra_cflags=["-O3"],
            extra_cuda_cflags=[
                "-O3",
                "-lineinfo",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "-U__CUDA_NO_HALF2_OPERATORS__",
                "--expt-extended-lambda",
                "--expt-relaxed-constexpr",
            ],
            verbose=False,
        )
    return _extension


def _launch_shape() -> tuple[int, int, int, int, int, int, int, int]:
    threads = int(os.environ.get("FK_MERGE_THREADS", _DEFAULT_THREADS))
    rows = int(os.environ.get("FK_MERGE_ROWS", _DEFAULT_ROWS))
    vec = int(os.environ.get("FK_MERGE_VEC", _DEFAULT_VEC))
    forced = os.environ.get("FK_MERGE_HINT")
    hint = int(forced) if forced is not None else -1
    kernel = int(os.environ.get("FK_MERGE_KERNEL", _DEFAULT_KERNEL))
    waves = int(os.environ.get("FK_MERGE_WAVES", _DEFAULT_WAVES))
    store = int(os.environ.get("FK_MERGE_STORE", _DEFAULT_STORE))
    min_ctas = int(os.environ.get("FK_MERGE_MINCTAS", _DEFAULT_MIN_CTAS))
    if threads not in _THREAD_CHOICES:
        threads = _DEFAULT_THREADS
    if rows not in _ROW_CHOICES:
        rows = _DEFAULT_ROWS
    if vec not in _VEC_CHOICES:
        vec = _DEFAULT_VEC
    if hint not in _HINT_CHOICES:
        hint = -1  # -1 means "decide from the working-set size"
    if kernel not in _KERNEL_CHOICES:
        kernel = _DEFAULT_KERNEL
    if store not in _STORE_CHOICES:
        store = _DEFAULT_STORE
    if min_ctas not in _MIN_CTAS_CHOICES:
        min_ctas = _DEFAULT_MIN_CTAS
    return threads, rows, vec, hint, kernel, max(0, waves), store, min_ctas


_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


def _route_key(
    output: torch.Tensor,
    prefix_output: torch.Tensor,
    prefix_lse: torch.Tensor,
    suffix_output: torch.Tensor,
    suffix_lse: torch.Tensor,
    output_lse: torch.Tensor | None,
) -> tuple:
    """Every property :func:`_cuda_path_ok` reads, so a cache hit cannot be stale.

    Keying on the data tensors alone would be wrong: a later call carrying the
    same shapes and strides but a non-contiguous, mis-shaped or off-device
    log-sum-exp tensor would reuse the earlier decision and let the kernel index
    it linearly. Pointer alignment and storage overlap are deliberately absent --
    they are properties of this call's allocations rather than of its layout, so
    they are re-checked on every call instead.
    """

    def layout(t: torch.Tensor | None) -> tuple | None:
        if t is None:
            return None
        return (t.shape, t.dtype, t.stride(), t.device)

    return (
        layout(output),
        layout(prefix_output),
        layout(suffix_output),
        layout(prefix_lse),
        layout(suffix_lse),
        layout(output_lse),
    )


def _cuda_path_ok(
    output: torch.Tensor,
    prefix_output: torch.Tensor,
    prefix_lse: torch.Tensor,
    suffix_output: torch.Tensor,
    suffix_lse: torch.Tensor,
    output_lse: torch.Tensor | None,
) -> bool:
    """Whether the CUDA kernels can index these tensors correctly.

    Reads only what :func:`_route_key` records, so the decision is cacheable.
    """
    data = (output, prefix_output, suffix_output)
    if not all(t.is_cuda for t in data) or not prefix_lse.is_cuda or not suffix_lse.is_cuda:
        return False
    device = prefix_output.device
    if any(t.device != device for t in (output, suffix_output, prefix_lse, suffix_lse)):
        return False
    if output_lse is not None and (not output_lse.is_cuda or output_lse.device != device):
        return False

    dtype = prefix_output.dtype
    if dtype not in _SUPPORTED_DTYPES:
        return False
    if output.dtype != dtype or suffix_output.dtype != dtype:
        return False

    if prefix_output.dim() != 3:
        return False
    shape = prefix_output.shape
    if output.shape != shape or suffix_output.shape != shape:
        return False
    num_tokens, num_heads, head_size = (int(s) for s in shape)
    if num_tokens == 0 or num_heads == 0 or head_size == 0:
        return False

    pack = 16 // prefix_output.element_size()
    if head_size % pack:
        return False

    # Both kernels address a row as `row * head_stride`, which is only the right
    # element for a token when the outer stride is exactly num_heads head strides.
    # Requiring the head stride to be at least the head size on top of that makes
    # the rows disjoint: a smaller or zero stride satisfies the two relations
    # while mapping several logical rows onto the same storage, which for the
    # output means concurrent writes to one address.
    for t in data:
        if t.stride(2) != 1:
            return False
        if t.stride(1) < head_size:
            return False
        if t.stride(0) != num_heads * t.stride(1):
            return False
    if prefix_output.stride(1) != suffix_output.stride(1):
        return False
    # A head stride that is not a whole number of packs makes the 16-byte access
    # misaligned on odd heads.
    if prefix_output.stride(1) % pack or output.stride(1) % pack:
        return False

    for lse in (prefix_lse, suffix_lse, output_lse):
        if lse is None:
            continue
        if lse.dtype != torch.float32 or not lse.is_contiguous():
            return False
        if tuple(lse.shape) != (num_heads, num_tokens):
            return False

    # The specialised kernel indexes in 32-bit, so the largest element offset it
    # forms has to fit. Head strides are positive by the check above, so the
    # extreme offset is the last row's last element. Evaluated in Python, before
    # any 32-bit multiply happens in the kernel.
    rows = num_tokens * num_heads
    for stride in (prefix_output.stride(1), output.stride(1)):
        if (rows - 1) * stride + head_size - 1 >= 2**31:
            return False
    return True


def _aligned(*tensors: torch.Tensor | None) -> bool:
    """16-byte alignment of every data pointer, re-checked on every call."""
    return all(t is None or t.data_ptr() % 16 == 0 for t in tensors)


def _reachable_bytes(t: torch.Tensor) -> tuple[int, int]:
    """The half-open byte range *t* can touch, as ``(first, last_exclusive)``.

    ``numel * element_size`` is the span only for contiguous storage. A padded
    view reaches ``sum((size_i - 1) * stride_i) + 1`` elements past its base,
    which for a head-strided tensor is nearly twice as far -- so using the
    contiguous span lets two shifted views of one storage look disjoint when they
    are not. Strides are positive here: this is only consulted for a layout that
    already passed `_cuda_path_ok`, which requires ``stride(1) >= head_size`` and
    ``stride(2) == 1``.
    """
    span = sum((size - 1) * stride for size, stride in zip(t.shape, t.stride())) + 1
    start = t.data_ptr()
    return start, start + span * t.element_size()


def _overlaps(
    writes: tuple[torch.Tensor | None, ...], reads: tuple[torch.Tensor | None, ...]
) -> bool:
    """Whether any pair of the kernel's pointers shares storage.

    Every pointer is declared ``__restrict__``, so an overlap is undefined behaviour
    even where the addresses would happen to line up. That covers three kinds of
    pair, not one: a write against a read, and also **write against write** --
    ``output`` and ``output_lse`` are both written, and if they shared storage the
    two stores would race. Reads may freely overlap each other.

    Overlapping calls take the PyTorch fallback, which reads its inputs into
    temporaries before writing anything.
    """
    spans = [(_reachable_bytes(t), True) for t in writes if t is not None]
    spans += [(_reachable_bytes(t), False) for t in reads if t is not None]
    for i, ((a_start, a_end), a_write) in enumerate(spans):
        for (b_start, b_end), b_write in spans[i + 1 :]:
            if not (a_write or b_write):
                continue  # two reads may share storage
            if a_start < b_end and b_start < a_end:
                return True
    return False


def _wide_access_ok(*tensors: torch.Tensor) -> bool:
    """32-byte alignment, which a 256-bit access needs on pointer and row start."""
    for t in tensors:
        if t.data_ptr() % 32:
            return False
        if (t.stride(1) * t.element_size()) % 32:
            return False
        if t.shape[2] % (32 // t.element_size()):
            return False
    return True


def _torch_fallback(
    output: torch.Tensor,
    prefix_output: torch.Tensor,
    prefix_lse: torch.Tensor,
    suffix_output: torch.Tensor,
    suffix_lse: torch.Tensor,
    output_lse: torch.Tensor | None,
) -> None:
    """The same semantics in PyTorch, for layouts the kernels cannot index."""
    neg_inf = float("-inf")
    p_lse = prefix_lse.to(torch.float32)
    s_lse = suffix_lse.to(torch.float32)
    p_lse = torch.where(torch.isinf(p_lse), torch.full_like(p_lse, neg_inf), p_lse)
    s_lse = torch.where(torch.isinf(s_lse), torch.full_like(s_lse, neg_inf), s_lse)
    max_lse = torch.fmax(p_lse, s_lse)
    degenerate = torch.isinf(max_lse)

    p_se = torch.exp(p_lse - max_lse)
    s_se = torch.exp(s_lse - max_lse)
    out_se = p_se + s_se
    p_w = (p_se / out_se).transpose(0, 1).unsqueeze(-1)
    s_w = (s_se / out_se).transpose(0, 1).unsqueeze(-1)

    merged = (
        prefix_output.to(torch.float32) * p_w + suffix_output.to(torch.float32) * s_w
    ).to(output.dtype)

    # Verbatim rows move raw bits, so -0.0 and NaN payloads survive.
    int_view = {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[
        output.element_size()
    ]
    rows = degenerate.transpose(0, 1).unsqueeze(-1).expand_as(merged)
    merged = torch.where(
        rows,
        prefix_output.contiguous().view(int_view),
        merged.contiguous().view(int_view),
    ).view(output.dtype)
    output.copy_(merged)

    if output_lse is not None:
        output_lse.copy_(torch.where(degenerate, max_lse, torch.log(out_se) + max_lse))


class MergeAttnStates(nn.Module):
    """Online softmax merge of two attention partitions."""

    def __init__(self) -> None:
        super().__init__()
        # (shape, dtype, strides, output_lse presence) -> which path to take.
        self._route: dict[tuple, bool] = {}

    def forward(
        self,
        output: torch.Tensor,
        prefix_output: torch.Tensor,
        prefix_lse: torch.Tensor,
        suffix_output: torch.Tensor,
        suffix_lse: torch.Tensor,
        output_lse: torch.Tensor | None = None,
    ) -> None:
        # Both the CUDA and Triton kernels derive the suffix head stride from
        # prefix_output, so suffix_output must share the same head stride.
        assert prefix_output.stride(1) == suffix_output.stride(1), (
            "merge_attn_states requires prefix_output and suffix_output to have "
            f"matching head strides, got {prefix_output.stride(1)} and "
            f"{suffix_output.stride(1)}"
        )

        key = _route_key(
            output, prefix_output, prefix_lse, suffix_output, suffix_lse, output_lse
        )
        route = self._route.get(key)
        if route is None:
            route = _cuda_path_ok(
                output, prefix_output, prefix_lse, suffix_output, suffix_lse, output_lse
            )
            self._route[key] = route

        # Alignment and overlap are re-checked every call: the cached decision
        # above says nothing about where this call's tensors happen to live. The
        # kernels declare their pointers __restrict__, so an `output` that shares
        # storage with an input goes to the PyTorch fallback, which reads both
        # inputs before writing anything.
        if (
            route
            and _aligned(output, prefix_output, suffix_output)
            and not _overlaps(
                (output, output_lse),
                (prefix_output, suffix_output, prefix_lse, suffix_lse),
            )
        ):
            (
                threads,
                rows,
                vec,
                hint,
                kernel,
                waves,
                store,
                min_ctas,
            ) = _launch_shape()
            if hint < 0:
                hint = _stream_hint(prefix_output)
            if vec == 32 and not _wide_access_ok(
                output, prefix_output, suffix_output
            ):
                # A 256-bit access needs 32-byte alignment on the base pointer
                # and on every row start; drop to 16 bytes rather than fault.
                vec = 16
            _ext().merge_attn_states_fk(
                output,
                output_lse,
                prefix_output,
                prefix_lse,
                suffix_output,
                suffix_lse,
                threads,
                rows,
                vec,
                hint,
                kernel,
                waves,
                store,
                min_ctas,
            )
            return

        _torch_fallback(
            output, prefix_output, prefix_lse, suffix_output, suffix_lse, output_lse
        )
