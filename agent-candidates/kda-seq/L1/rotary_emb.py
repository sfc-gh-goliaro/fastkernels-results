"""Rotary position embeddings (RoPE), with optional Llama 3.1-style frequency scaling.

Same module contract as the baseline, with two per-call costs removed.

The first is not in the kernel at all.  ``cos_sin_cache`` is a ``float32``
buffer while activations are ``bfloat16``, and nothing in the surrounding stack
casts a non-persistent buffer, so the baseline's ``cache.dtype != query.dtype``
branch is taken on *every* forward and re-materializes the whole
``[max_position, rot_dim]`` table each time.  Rather than caching that
conversion -- which would then have to be invalidated whenever the buffer
changed -- the kernel reads the live table and rounds each cos/sin value through
the activation dtype itself.  That is bit-for-bit what ``cache.to(query.dtype)``
produces, so no table is derived, nothing can go stale, and a subclass may
rebind or overwrite ``cos_sin_cache`` however it likes.

The second is the kernel.  The vendored kernel moves one 2-byte element per
instruction.  It is replaced by a single fused kernel covering query *and* key in
one launch with 128-bit loads and stores.  That path is taken only when the
launch provably writes each address from exactly one thread: every address
16-byte aligned, token rows and heads disjoint, and the query and key ranges not
overlapping each other.  Anything else -- an unaligned view, an odd stride,
overlapping rows, ``query is key``, the interleaved GPT-J layout -- falls through
to a scalar kernel that reproduces the vendored index arithmetic element for
element.  So the module matches the reference on any shape, stride, dtype,
``rot_dim`` or rotary style the reference itself is defined on.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# The baseline already owns the extension name ``rotary_emb`` and the
# ``torch.library`` namespace ``fastkernels_rope`` in this process -- the harness
# imports it first, as the reference -- so both names have to differ here.
_EXT_NAME = "fk_rope_cand"
_OP_NAMESPACE = "fastkernels_rope_cand"

# The CUDA source is embedded rather than shipped as a sidecar ``.cu`` so that
# this file is the whole candidate: the harness resolves exactly
# ``candidate/L{level}/{stem}.py`` and loads it by path, and nothing guarantees
# an adjacent file travels with it.
_CPP_SOURCE = r"""
#include <torch/extension.h>

#include <optional>

void fk_rope_forward(torch::Tensor positions, torch::Tensor query,
                     std::optional<torch::Tensor> key, int64_t head_size,
                     torch::Tensor cos_sin_cache, bool is_neox,
                     bool round_cache_to_query);

int64_t fk_rope_last_path();
"""

_CUDA_SOURCE = r"""
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <type_traits>

namespace fk_rope {

// A 16-byte pack of `kLen` elements.  Loading and storing through this type is
// what turns eight 2-byte bf16 accesses into one `ld.global.v4.u32` /
// `st.global.v4.u32`: the alignment attribute is the compiler's licence to use
// a 128-bit instruction.  Mirrors the `vec_n_t` idiom used by the in-tree
// elementwise kernels.
template <typename T>
struct __align__(16) Pack16 {
  static constexpr int kLen = 16 / sizeof(T);
  T val[kLen];
};

// Rounding an angle through the activation dtype reproduces the reference's
// Python-side `cache.to(query.dtype)`: `static_cast<scalar_t>` is the same
// conversion ATen's `.to()` performs, which is what lets the live table be read
// directly with no derived copy to keep in sync.  The vectorized path passes a
// compile-time constant here, so it folds to one branchless form.
template <typename scalar_t, typename cache_t>
__device__ __forceinline__ float angle_to_float(cache_t v, bool round_to_query) {
  return round_to_query ? static_cast<float>(static_cast<scalar_t>(v))
                        : static_cast<float>(v);
}

// ---------------------------------------------------------------------------
// Scalar path.  Correct for any shape, stride, rot_dim, cache dtype and rotary
// style, and its index arithmetic and fp32 operation order are the reference's,
// so anything the vectorized predicate rejects lands here without changing a
// single rounding decision.
// ---------------------------------------------------------------------------
template <typename scalar_t, typename cache_t, bool IS_NEOX>
__device__ __forceinline__ void apply_token_rope(
    scalar_t* __restrict__ arr, const cache_t* __restrict__ cos_ptr,
    const cache_t* __restrict__ sin_ptr, int rot_offset, int embed_dim,
    int64_t angle_stride, bool round_to_query) {
  int x_index, y_index;
  if (IS_NEOX) {
    // Rotate element i against element i + embed_dim.
    x_index = rot_offset;
    y_index = embed_dim + rot_offset;
  } else {
    // GPT-J: rotate adjacent even/odd elements against each other.
    x_index = 2 * rot_offset;
    y_index = 2 * rot_offset + 1;
  }
  // Both layouts index cos/sin by angle, i.e. by rot_offset.  The reference
  // spells the interleaved case `x_index / 2`, which is the same value.
  const int64_t angle = static_cast<int64_t>(rot_offset) * angle_stride;
  const float cos_f =
      angle_to_float<scalar_t, cache_t>(cos_ptr[angle], round_to_query);
  const float sin_f =
      angle_to_float<scalar_t, cache_t>(sin_ptr[angle], round_to_query);
  const float x_f = static_cast<float>(arr[x_index]);
  const float y_f = static_cast<float>(arr[y_index]);
  arr[x_index] = static_cast<scalar_t>(x_f * cos_f - y_f * sin_f);
  arr[y_index] = static_cast<scalar_t>(y_f * cos_f + x_f * sin_f);
}

template <typename scalar_t, typename cache_t, bool IS_NEOX>
__global__ void scalar_rope_kernel(
    const int64_t* __restrict__ positions, scalar_t* __restrict__ query,
    scalar_t* __restrict__ key, const cache_t* __restrict__ cos_sin_cache,
    const int rot_dim, const int64_t query_stride, const int64_t key_stride,
    const int64_t head_stride, const int num_heads, const int num_kv_heads,
    const int64_t cache_row_stride, const int64_t cache_angle_stride,
    const bool round_to_query) {
  const int64_t token_idx = blockIdx.x;
  // The row pitch and the angle pitch are given rather than assumed, so this
  // kernel can read a table the reference would have compacted (see the host).
  const cache_t* cache_ptr = cos_sin_cache + positions[token_idx] * cache_row_stride;
  const int embed_dim = rot_dim / 2;
  const cache_t* cos_ptr = cache_ptr;
  const cache_t* sin_ptr = cache_ptr + embed_dim * cache_angle_stride;

  const int nq = num_heads * embed_dim;
  for (int i = threadIdx.x; i < nq; i += blockDim.x) {
    const int64_t token_head = token_idx * query_stride +
                               static_cast<int64_t>(i / embed_dim) * head_stride;
    apply_token_rope<scalar_t, cache_t, IS_NEOX>(
        query + token_head, cos_ptr, sin_ptr, i % embed_dim, embed_dim,
        cache_angle_stride, round_to_query);
  }
  if (key != nullptr) {
    const int nk = num_kv_heads * embed_dim;
    for (int i = threadIdx.x; i < nk; i += blockDim.x) {
      const int64_t token_head = token_idx * key_stride +
                                 static_cast<int64_t>(i / embed_dim) * head_stride;
      apply_token_rope<scalar_t, cache_t, IS_NEOX>(
          key + token_head, cos_ptr, sin_ptr, i % embed_dim, embed_dim,
          cache_angle_stride, round_to_query);
    }
  }
}

// ---------------------------------------------------------------------------
// Vectorized path: one block per token, query and key in the same launch.
//
// A thread owns one (head slot, 16-byte chunk) pair, where head slots run over
// query's heads first and then key's.  For the captured 32 query / 8 key heads
// and rot_dim 128 that is 40 slots x 8 chunks = 320 threads, i.e. 10 full warps
// with the query/key split landing exactly on the warp-8 boundary, so the branch
// is warp-uniform and no warp is partial.
//
// Every element is loaded once and stored once by exactly one thread.  That is a
// property of the *addresses*, not just of the (slot, chunk) indices, and it is
// the host predicate that establishes it: token rows are disjoint, heads within
// a row are disjoint, and the query and key ranges do not overlap.  Anything the
// predicate cannot prove goes to the scalar kernel instead.
// ---------------------------------------------------------------------------
template <typename scalar_t, typename cache_t, bool ROUND_TO_QUERY, bool USE_PDL>
__global__ void fused_rope_vec_kernel(
    const int64_t* __restrict__ positions, scalar_t* __restrict__ query,
    scalar_t* __restrict__ key, const cache_t* __restrict__ cos_sin_cache,
    const int rot_dim, const int64_t query_stride, const int64_t key_stride,
    const int64_t head_stride, const int num_heads, const int chunks_per_head,
    const int items_per_token) {
  using vec_t = Pack16<scalar_t>;
  using cache_pack_t = Pack16<cache_t>;
  constexpr int kVec = vec_t::kLen;
  // The table is normally *wider* than the activations - an fp32 table against
  // bf16 q/k - so one thread's kVec angles arrive as several 16-byte packs.
  constexpr int kSrcPerPack = cache_pack_t::kLen;
  constexpr int kSrcPacks = kVec / kSrcPerPack;
  static_assert(kSrcPacks >= 1 && kSrcPacks * kSrcPerPack == kVec,
                "cache element size must be a whole multiple of the activation's");

  // Programmatic dependent launch: the grid may be set up while the kernel that
  // produced these tensors is still running, and only the first *read* has to
  // wait for it.  The harness copies its inputs into a fresh pool slot inside
  // the timed region, so on a small grid that setup is worth overlapping -- but
  // it costs more than it saves once the grid is large enough to hide its own
  // launch, so the host turns it off there and the branch below skips it.
  // Every input may have been written by the predecessor, so the wait precedes
  // all of them.
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  if constexpr (USE_PDL) {
    asm volatile("griddepcontrol.wait;");
  }
#endif
  const int64_t token_idx = blockIdx.x;
  const int embed_dim = rot_dim >> 1;
  // The whole 16-byte-aligned cos/sin row is read once per token here and
  // broadcast from L1 to every head slot in the block.
  const cache_t* __restrict__ row =
      cos_sin_cache + positions[token_idx] * static_cast<int64_t>(rot_dim);
  const int64_t query_base = token_idx * query_stride;
  const int64_t key_base = token_idx * key_stride;

  for (int item = threadIdx.x; item < items_per_token; item += blockDim.x) {
    const int head_slot = item / chunks_per_head;
    const int offset = (item - head_slot * chunks_per_head) * kVec;
    const bool is_query = head_slot < num_heads;
    scalar_t* base =
        is_query ? query + query_base + static_cast<int64_t>(head_slot) * head_stride
                 : key + key_base +
                       static_cast<int64_t>(head_slot - num_heads) * head_stride;

    vec_t x_v = *reinterpret_cast<const vec_t*>(base + offset);
    vec_t y_v = *reinterpret_cast<const vec_t*>(base + embed_dim + offset);

    // One source pack of cos and of sin is consumed before the next is read.
    // Holding all of them at once costs registers, and registers are what bound
    // this kernel: at 320 threads per block, 34 per thread is the difference
    // between 6 and 4 resident blocks per SM, which halves the eligible warps.
#pragma unroll
    for (int p = 0; p < kSrcPacks; ++p) {
      const int lane = p * kSrcPerPack;
      const cache_pack_t cos_p =
          *reinterpret_cast<const cache_pack_t*>(row + offset + lane);
      const cache_pack_t sin_p =
          *reinterpret_cast<const cache_pack_t*>(row + embed_dim + offset + lane);
#pragma unroll
      for (int e = 0; e < kSrcPerPack; ++e) {
        const int j = lane + e;
        const float cos_f =
            angle_to_float<scalar_t, cache_t>(cos_p.val[e], ROUND_TO_QUERY);
        const float sin_f =
            angle_to_float<scalar_t, cache_t>(sin_p.val[e], ROUND_TO_QUERY);
        const float x_f = static_cast<float>(x_v.val[j]);
        const float y_f = static_cast<float>(y_v.val[j]);
        x_v.val[j] = static_cast<scalar_t>(x_f * cos_f - y_f * sin_f);
        y_v.val[j] = static_cast<scalar_t>(y_f * cos_f + x_f * sin_f);
      }
    }

    *reinterpret_cast<vec_t*>(base + offset) = x_v;
    *reinterpret_cast<vec_t*>(base + embed_dim + offset) = y_v;
  }
}

// Which kernel the last launch used (0 scalar, 1 vectorized).  A diagnostic for
// the test suite only: it is a plain global with no synchronization, so it is
// meaningless under concurrent forwards and must not be read as anything but a
// single-threaded probe.  Nothing in the kernels' output depends on it.
int64_t g_last_path = -1;

inline bool aligned16(const void* p) {
  return (reinterpret_cast<std::uintptr_t>(p) & 15) == 0;
}

// The extent of `count` blocks of `width` elements laid out `stride` apart:
// (count - 1) * stride + width.  Returns false rather than a wrapped value.
// Used twice - for the heads inside one token's row, and for the token rows
// inside the whole tensor.
inline bool strided_extent(int64_t count, int64_t stride, int64_t width,
                           int64_t* out) {
  int64_t span = 0;
  if (__builtin_mul_overflow(count - 1, stride, &span)) return false;
  return !__builtin_add_overflow(span, width, out);
}

// Everything the vectorized path needs from one mutated tensor.  On success
// *footprint* is the per-row element count and *span* the whole accessed extent,
// both of which the overlap test below consumes.
//
// Every address the kernel forms is
//   data_ptr + token*row_stride + head*head_stride + {0, embed_dim} + chunk*kVec
// elements, so all of them are 16-byte aligned exactly when the base pointer and
// each of those terms is, measured in bytes.  `data_ptr()` already folds in
// `storage_offset()`, so a view sliced to an odd element offset is rejected here
// instead of issuing a misaligned 128-bit access.  `16 / esz` rather than
// `stride * esz % 16` keeps the test free of overflow on a pathological stride.
//
// Disjointness is a separate requirement and is what makes one-writer-per-element
// true in addresses: heads must not overlap each other (`head_stride >= rot_dim`)
// and rows must not overlap each other (`row_stride >= footprint`).  A positive
// row stride alone does not give the latter - an aligned [n, 4096] view with
// stride 256 has 4096-element rows 256 elements apart.
inline bool vec_addressable(const torch::Tensor& t, int64_t row_stride,
                            int64_t head_stride, int64_t heads, int64_t rot_dim,
                            int64_t embed_dim, int64_t num_tokens,
                            int64_t* footprint, int64_t* span) {
  const int64_t esz = t.element_size();
  const int64_t per_pack = 16 / esz;
  if (!aligned16(t.const_data_ptr()) || row_stride <= 0 ||
      head_stride < rot_dim || row_stride % per_pack != 0 ||
      head_stride % per_pack != 0 || embed_dim % per_pack != 0) {
    return false;
  }
  if (!strided_extent(heads, head_stride, rot_dim, footprint)) return false;
  if (row_stride < *footprint) return false;
  return strided_extent(num_tokens, row_stride, *footprint, span);
}

// The cos/sin table is read-only and indexed as
//   data_ptr + position*rot_dim + {0, embed_dim} + chunk*kVec,
// so it needs alignment but none of the disjointness the mutated tensors do.
// The per-thread angles span `sizeof(cache_t)/sizeof(scalar_t)` packs, all at
// multiples of 16 bytes from the row base, so aligning the row is sufficient.
inline bool cache_addressable(const torch::Tensor& t, int64_t rot_dim,
                              int64_t embed_dim) {
  const int64_t per_pack = 16 / t.element_size();
  return aligned16(t.const_data_ptr()) && rot_dim % per_pack == 0 &&
         embed_dim % per_pack == 0;
}

// Are the query and key ranges the launch touches provably disjoint?
//
// This predicate is what licenses the vectorized decomposition, so it is written
// as arithmetic on `uintptr_t` rather than on pointers: relationally comparing or
// subtracting pointers into *different* allocations is undefined in C++, and this
// is exactly the case that has to be decided.  Every product and sum below is
// overflow-checked, and any overflow rejects the vectorized path.
//
// Two ways to prove it, and deliberately not just the first: the captured inputs
// are two slices of one fused QKV tensor whose bounding intervals interleave, so
// a bounding-interval test alone would reject the captured layout and give up the
// fast path entirely.
//
//   * the half-open byte intervals do not meet - separate allocations, which is
//     what the harness itself produces; or
//   * both tensors advance by the same row stride, so their footprints repeat
//     with one shared period, and the two windows inside that period do not meet.
//
// Anything else is treated as possibly overlapping. That matters because the
// reference applies query's rotation and then key's *sequentially within one
// thread*, so on a genuinely overlapping pair it rotates twice; splitting the two
// across threads would race and rotate once. The scalar kernel keeps the
// reference's semantics.
inline bool checked_mul(uintptr_t a, uintptr_t b, uintptr_t* out) {
  return !__builtin_mul_overflow(a, b, out);
}

inline bool ranges_disjoint(const void* q_base, int64_t q_span, int64_t q_footprint,
                            int64_t q_row_stride, const void* k_base,
                            int64_t k_span, int64_t k_footprint,
                            int64_t k_row_stride, int64_t esz) {
  // vec_addressable has already established that every count here is positive.
  const uintptr_t q0 = reinterpret_cast<uintptr_t>(q_base);
  const uintptr_t k0 = reinterpret_cast<uintptr_t>(k_base);
  const uintptr_t width = static_cast<uintptr_t>(esz);
  uintptr_t q_bytes = 0, k_bytes = 0, q_end = 0, k_end = 0;
  if (!checked_mul(static_cast<uintptr_t>(q_span), width, &q_bytes)) return false;
  if (!checked_mul(static_cast<uintptr_t>(k_span), width, &k_bytes)) return false;
  if (__builtin_add_overflow(q0, q_bytes, &q_end)) return false;
  if (__builtin_add_overflow(k0, k_bytes, &k_end)) return false;
  if (q_end <= k0 || k_end <= q0) return true;

  if (q_row_stride != k_row_stride) return false;
  uintptr_t period = 0, q_width = 0, k_width = 0;
  if (!checked_mul(static_cast<uintptr_t>(q_row_stride), width, &period)) return false;
  if (!checked_mul(static_cast<uintptr_t>(q_footprint), width, &q_width)) return false;
  if (!checked_mul(static_cast<uintptr_t>(k_footprint), width, &k_width)) return false;
  if (period == 0) return false;
  // Key's byte offset within one shared period, derived from an unsigned
  // difference in whichever direction is non-negative, so no signed pointer
  // subtraction is involved.
  uintptr_t offset = 0;
  if (k0 >= q0) {
    offset = (k0 - q0) % period;
  } else {
    const uintptr_t behind = (q0 - k0) % period;
    offset = behind == 0 ? 0 : period - behind;
  }
  uintptr_t k_hi = 0;
  if (__builtin_add_overflow(offset, k_width, &k_hi)) return false;
  return offset >= q_width && k_hi <= period;
}

// The reference launcher's derived geometry, resolved once on the host.
struct Geometry {
  int64_t num_tokens;
  int rot_dim;
  int embed_dim;
  int num_heads;
  int num_kv_heads;
  int64_t query_stride;
  int64_t key_stride;
  int64_t head_stride;
  int64_t cache_row_stride;
  int64_t cache_angle_stride;
  bool is_neox;
  bool round_to_query;
  bool writes_disjoint;
  bool dense_cache;
  bool use_pdl;
};

// Kept a template so that `if constexpr` genuinely suppresses instantiation:
// outside a templated entity a discarded branch is still instantiated, and the
// cos/sin pack arithmetic only divides evenly when the cache element size is a
// whole multiple of the activation's.
template <typename scalar_t, typename cache_t>
void launch_rope(const Geometry& g, const int64_t* positions, scalar_t* query,
                 scalar_t* key, const cache_t* cos_sin_cache,
                 cudaStream_t stream) {
  if constexpr (sizeof(cache_t) >= sizeof(scalar_t) &&
                sizeof(cache_t) % sizeof(scalar_t) == 0) {
    constexpr int kVec = Pack16<scalar_t>::kLen;
    // The interleaved layout pairs adjacent elements, which a contiguous pack
    // cannot express, so it stays on the scalar path.
    // `dense_cache` is what lets the vectorized kernel keep its hard-coded
    // rot_dim row pitch and contiguous 16-byte angle packs.
    if (g.is_neox && g.writes_disjoint && g.dense_cache &&
        g.rot_dim % (2 * kVec) == 0 && g.embed_dim >= kVec) {
      const int chunks_per_head = g.embed_dim / kVec;
      // 64-bit while the geometry is being multiplied out: a head count large
      // enough to overflow int is absurd, but silent wraparound here would
      // launch a wrong-sized grid rather than fail.
      const int64_t head_slots =
          g.num_heads + (key != nullptr ? g.num_kv_heads : 0);
      const int64_t items = head_slots * chunks_per_head;
      TORCH_CHECK(items > 0 && items <= std::numeric_limits<int32_t>::max(),
                  "vectorized rotary geometry does not fit in 32 bits");
      // Round the block up to whole warps and cap it; the in-kernel stride loop
      // covers the remainder for exotic head counts, while the captured 320
      // items are exactly ten full warps.
      const int block_size =
          static_cast<int>(std::min<int64_t>(1024, ((items + 31) / 32) * 32));
      cudaLaunchConfig_t config{};
      config.gridDim = dim3(g.num_tokens);
      config.blockDim = dim3(block_size);
      config.dynamicSmemBytes = 0;
      config.stream = stream;
      cudaLaunchAttribute attrs[1];
      attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
      attrs[0].val.programmaticStreamSerializationAllowed = 1;
      config.numAttrs = g.use_pdl ? 1 : 0;
      config.attrs = attrs;
      // PDL is a template parameter rather than a branch: the `griddepcontrol`
      // branch cost 16 registers per thread, which dropped the large-n grid from
      // 6 resident blocks per SM to 4.  Templating confines that cost to the
      // small-n instantiation, where the grid is far too small for occupancy to
      // bind anyway.
      auto launch = [&](auto round_tag, auto pdl_tag) {
        constexpr bool kRound = decltype(round_tag)::value;
        constexpr bool kPdl = decltype(pdl_tag)::value;
        C10_CUDA_CHECK(cudaLaunchKernelEx(
            &config, fused_rope_vec_kernel<scalar_t, cache_t, kRound, kPdl>,
            positions, query, key, cos_sin_cache, g.rot_dim, g.query_stride,
            g.key_stride, g.head_stride, g.num_heads, chunks_per_head,
            static_cast<int>(items)));
      };
      if (g.round_to_query) {
        if (g.use_pdl) launch(std::true_type{}, std::true_type{});
        else launch(std::true_type{}, std::false_type{});
      } else {
        if (g.use_pdl) launch(std::false_type{}, std::true_type{});
        else launch(std::false_type{}, std::false_type{});
      }
      g_last_path = 1;
      return;
    }
  }

  const dim3 grid(g.num_tokens);
  const dim3 block(
      std::min<int64_t>(static_cast<int64_t>(g.num_heads) * g.rot_dim / 2, 512));
  if (g.is_neox) {
    scalar_rope_kernel<scalar_t, cache_t, true><<<grid, block, 0, stream>>>(
        positions, query, key, cos_sin_cache, g.rot_dim, g.query_stride,
        g.key_stride, g.head_stride, g.num_heads, g.num_kv_heads,
        g.cache_row_stride, g.cache_angle_stride, g.round_to_query);
  } else {
    scalar_rope_kernel<scalar_t, cache_t, false><<<grid, block, 0, stream>>>(
        positions, query, key, cos_sin_cache, g.rot_dim, g.query_stride,
        g.key_stride, g.head_stride, g.num_heads, g.num_kv_heads,
        g.cache_row_stride, g.cache_angle_stride, g.round_to_query);
  }
  g_last_path = 0;
}

}  // namespace fk_rope

void fk_rope_forward(torch::Tensor positions, torch::Tensor query,
                     std::optional<torch::Tensor> key, int64_t head_size,
                     torch::Tensor cos_sin_cache, bool is_neox,
                     bool round_cache_to_query) {
  // Everything down to the launch mirrors the reference launcher, so the scalar
  // kernel is bit-identical to it and the vectorized kernel differs only in how
  // many elements one thread carries.
  const int64_t num_tokens = positions.numel();
  const int positions_ndim = positions.dim();
  TORCH_CHECK(positions_ndim == 1 || positions_ndim == 2,
              "positions must have shape [num_tokens] or [batch_size, seq_len]");
  TORCH_CHECK(positions.scalar_type() == at::kLong, "positions must be int64");
  TORCH_CHECK(query.is_cuda() && positions.is_cuda() && cos_sin_cache.is_cuda(),
              "positions, query and cos_sin_cache must be CUDA tensors");
  TORCH_CHECK(!key.has_value() || key->is_cuda(), "key must be a CUDA tensor");
  TORCH_CHECK(!key.has_value() || key->scalar_type() == query.scalar_type(),
              "query and key must share a dtype");
  if (positions_ndim == 1) {
    TORCH_CHECK(query.size(0) == positions.size(0) &&
                    (!key.has_value() || key->size(0) == positions.size(0)),
                "query, key and positions must have the same number of tokens");
  } else {
    TORCH_CHECK(query.size(0) == positions.size(0) &&
                    (!key.has_value() || key->size(0) == positions.size(0)) &&
                    query.size(1) == positions.size(1) &&
                    (!key.has_value() || key->size(1) == positions.size(1)),
                "query, key and positions must have the same batch_size and seq_len");
  }
  if (num_tokens == 0) {
    return;
  }

  // Head counts come from numel, not from strides: on the fused-QKV views the
  // captures use, query.stride(0) spans q+k+v and would report 48 heads, not 32.
  const int64_t query_hidden_size = query.numel() / num_tokens;
  const int64_t key_hidden_size = key.has_value() ? key->numel() / num_tokens : 0;
  TORCH_CHECK(query_hidden_size % head_size == 0,
              "query hidden size must be a multiple of head_size");
  TORCH_CHECK(key_hidden_size % head_size == 0,
              "key hidden size must be a multiple of head_size");
  const int num_heads = static_cast<int>(query_hidden_size / head_size);
  const int num_kv_heads =
      key.has_value() ? static_cast<int>(key_hidden_size / head_size) : num_heads;
  TORCH_CHECK(num_kv_heads > 0 && num_heads % num_kv_heads == 0,
              "num_heads must be a multiple of num_kv_heads");

  TORCH_CHECK(cos_sin_cache.dim() == 2, "cos_sin_cache must be [max_position, rot_dim]");
  const int rot_dim = static_cast<int>(cos_sin_cache.size(1));
  TORCH_CHECK(rot_dim > 0, "rot_dim must be positive");
  TORCH_CHECK(rot_dim <= head_size, "rot_dim must not exceed head_size");
  // Deliberately not requiring an even rot_dim: the reference takes the floor
  // here and leaves the unpaired trailing cache entry unused, so requiring
  // evenness would reject an input the reference accepts.
  const int embed_dim = rot_dim / 2;
  // grid.x is unsigned 32-bit, so a larger token count would silently wrap.
  TORCH_CHECK(num_tokens <= std::numeric_limits<int32_t>::max(),
              "num_tokens exceeds the maximum CUDA grid dimension");

  const int seq_dim_idx = positions_ndim - 1;
  const int64_t query_stride = query.stride(seq_dim_idx);
  const int64_t key_stride = key.has_value() ? key->stride(seq_dim_idx) : 0;
  // [*, heads, head_size] carries an explicit head stride; a flat
  // [*, heads*head_size] has contiguous head blocks of head_size.  The
  // reference applies query's head stride to key as well, so this does too.
  const int64_t head_stride =
      (query.dim() == positions_ndim + 2) ? query.stride(-2) : head_size;

  // The predicate resolves before the dtype dispatch: it needs only element
  // sizes and pointers, both available without knowing the scalar type.  16
  // bytes is the pack width for every supported dtype, which is why one modulus
  // covers all of them.
  int64_t q_footprint = 0;
  int64_t q_span = 0;
  int64_t k_footprint = 0;
  int64_t k_span = 0;
  bool writes_disjoint =
      fk_rope::vec_addressable(query, query_stride, head_stride, num_heads,
                               rot_dim, embed_dim, num_tokens, &q_footprint,
                               &q_span) &&
      fk_rope::cache_addressable(cos_sin_cache, rot_dim, embed_dim);
  if (writes_disjoint && key.has_value()) {
    writes_disjoint =
        fk_rope::vec_addressable(*key, key_stride, head_stride, num_kv_heads,
                                 rot_dim, embed_dim, num_tokens, &k_footprint,
                                 &k_span) &&
        fk_rope::ranges_disjoint(query.const_data_ptr(), q_span, q_footprint,
                                 query_stride, key->const_data_ptr(), k_span,
                                 k_footprint, key_stride, query.element_size());
  }

  fk_rope::Geometry geo{};
  geo.num_tokens = num_tokens;
  geo.rot_dim = rot_dim;
  geo.embed_dim = embed_dim;
  geo.num_heads = num_heads;
  geo.num_kv_heads = num_kv_heads;
  geo.query_stride = query_stride;
  geo.key_stride = key_stride;
  geo.head_stride = head_stride;
  // `cache.to(query.dtype)` in the reference chooses a *layout* as well as a
  // dtype, and the vendored kernel then reads the result with a hard-coded dense
  // row pitch.  Two cases follow, and both have to be reproduced:
  //
  //  * the cast compacts - the source is not non-overlapping-and-dense, so
  //    `empty_like(..., MemoryFormat::Preserve)` falls back to contiguous. The
  //    reference reads *logical* values, so the table is indexed by its own
  //    strides here and the launch goes to the scalar kernel.
  //  * the cast preserves strides, or no cast happens at all because the dtypes
  //    already match. The reference reads the buffer's *physical* order with
  //    pitch rot_dim, which is what a dense-linear read reproduces.
  const bool logical_cache = round_cache_to_query &&
                             cos_sin_cache.scalar_type() != query.scalar_type() &&
                             !cos_sin_cache.is_non_overlapping_and_dense();
  geo.cache_row_stride = logical_cache ? cos_sin_cache.stride(0) : rot_dim;
  geo.cache_angle_stride = logical_cache ? cos_sin_cache.stride(1) : 1;
  geo.dense_cache = !logical_cache;
  geo.is_neox = is_neox;
  geo.round_to_query = round_cache_to_query;
  geo.writes_disjoint = writes_disjoint;
  // Overlapping the grid setup with the preceding kernel pays off only while the
  // grid is too small to hide its own launch.  Measured on B200: it saves ~1.9 us
  // at 1-414 tokens and costs ~6 us at 16384, so the switch is a few waves of
  // blocks rather than a fixed token count.
  geo.use_pdl =
      num_tokens <= 8 * at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

  const c10::cuda::CUDAGuard device_guard(query.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, query.scalar_type(),
      "fk_rope_forward", [&] {
        using query_t = scalar_t;
        query_t* q_ptr = query.data_ptr<query_t>();
        query_t* k_ptr = key.has_value() ? key->data_ptr<query_t>() : nullptr;
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16,
            cos_sin_cache.scalar_type(), "fk_rope_forward_cache", [&] {
              fk_rope::launch_rope<query_t, scalar_t>(
                  geo, positions.const_data_ptr<int64_t>(), q_ptr, k_ptr,
                  cos_sin_cache.const_data_ptr<scalar_t>(), stream);
            });
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

int64_t fk_rope_last_path() { return fk_rope::g_last_path; }
"""

_ext = None


def _extension():
    """JIT-build the embedded CUDA source once, then return the handle.

    Deliberately no ``--use_fast_math``: the fp32 multiply/add sequence has to
    stay the reference's for the results to match bit for bit.  ``-lineinfo`` is
    cheap and keeps source attribution available to a profiler.
    """
    global _ext
    if _ext is None:
        _ext = load_inline(
            name=_EXT_NAME,
            cpp_sources=[_CPP_SOURCE],
            cuda_sources=[_CUDA_SOURCE],
            functions=["fk_rope_forward", "fk_rope_last_path"],
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
    return _ext


# ---------------------------------------------------------------------------
# Register the in-place rotary op for torch.compile compatibility, using
# Tensor(a!) annotations so Inductor auto-functionalizes correctly.  The guard
# matters because this module is imported by file path: loading it twice under
# two module names would otherwise raise on a duplicate definition.
#
# ``round_cache_to_query`` defaults to off, which is the vendored op's own
# behaviour when it is handed a cos/sin table in some other dtype.  The module
# passes it on, because the *module* it replaces casts the table first.
# ---------------------------------------------------------------------------
def _op_already_defined() -> bool:
    try:
        return hasattr(getattr(torch.ops, _OP_NAMESPACE), "rotary_embedding")
    except Exception:
        return False


if not _op_already_defined():
    _lib = torch.library.Library(_OP_NAMESPACE, "DEF")

    _lib.define(
        "rotary_embedding(Tensor positions, Tensor(a!) query, "
        "Tensor(b!)? key, int head_size, Tensor cos_sin_cache, "
        "bool is_neox, bool round_cache_to_query=False) -> ()"
    )

    def _rotary_embedding_impl(
        positions, query, key, head_size, cos_sin_cache, is_neox,
        round_cache_to_query=False,
    ):
        _extension().fk_rope_forward(
            positions, query, key, head_size, cos_sin_cache, is_neox,
            round_cache_to_query,
        )

    _lib.impl("rotary_embedding", _rotary_embedding_impl, "CUDA")

    @torch.library.impl(_lib, "rotary_embedding", "Meta")
    def _rotary_embedding_meta(
        positions, query, key, head_size, cos_sin_cache, is_neox,
        round_cache_to_query=False,
    ):
        pass


_rope_op = getattr(torch.ops, _OP_NAMESPACE).rotary_embedding


def _compute_scaled_inv_freq(
    inv_freq: torch.Tensor,
    scaling_factor: float,
    low_freq_factor: float,
    high_freq_factor: float,
    original_max_position_embeddings: int,
) -> torch.Tensor:
    low_wl = original_max_position_embeddings / low_freq_factor
    high_wl = original_max_position_embeddings / high_freq_factor
    wl = 2 * math.pi / inv_freq
    if low_freq_factor != high_freq_factor:
        smooth = (original_max_position_embeddings / wl - low_freq_factor) / (
            high_freq_factor - low_freq_factor
        )
    else:
        smooth = torch.zeros_like(inv_freq)
    return torch.where(
        wl < high_wl,
        inv_freq,
        torch.where(
            wl > low_wl,
            inv_freq / scaling_factor,
            (1 - smooth) * inv_freq / scaling_factor + smooth * inv_freq,
        ),
    )


class RotaryEmbedding(nn.Module):
    """RoPE with optional Llama 3.1-style frequency scaling.

    When rope_scaling_factor == 1.0 (default), behaves as standard RoPE.
    When rope_scaling_factor != 1.0, applies the Llama 3.1 piecewise
    frequency scaling controlled by low/high freq factors.
    """

    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int,
        rope_theta: float,
        rope_scaling_factor: float = 1.0,
        rope_low_freq_factor: float = 1.0,
        rope_high_freq_factor: float = 1.0,
        rope_original_max_position_embeddings: int | None = None,
        is_neox_style: bool = True,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.is_neox_style = is_neox_style
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim))

        if rope_scaling_factor != 1.0 and rope_original_max_position_embeddings is not None:
            inv_freq = _compute_scaled_inv_freq(
                inv_freq,
                rope_scaling_factor,
                rope_low_freq_factor,
                rope_high_freq_factor,
                rope_original_max_position_embeddings,
            )

        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1).float()
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @staticmethod
    def forward_native(positions, query, key, head_dim, cos_sin_cache):
        """Pure PyTorch NeOX-style RoPE matching the CUDA kernel.

        The cache stores [cos, sin] each with embed_dim = head_dim/2 entries.
        Rotation pairs elements (i, i + embed_dim) across the full head,
        exactly matching the CUDA kernel's IS_NEOX=true path:
          out[i]            = x[i]*cos[i] - x[i+embed_dim]*sin[i]
          out[i+embed_dim]  = x[i+embed_dim]*cos[i] + x[i]*sin[i]
        """
        cos_sin = cos_sin_cache[positions]
        embed_dim = cos_sin.shape[-1] // 2
        cos = cos_sin[..., :embed_dim]
        sin = cos_sin[..., embed_dim:]

        q_shape = query.shape
        k_shape = key.shape
        q = query.view(q_shape[0], -1, head_dim)
        k = key.view(k_shape[0], -1, head_dim)

        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

        q1, q2 = q[..., :embed_dim], q[..., embed_dim:]
        k1, k2 = k[..., :embed_dim], k[..., embed_dim:]

        query = torch.cat([q1 * cos - q2 * sin,
                           q2 * cos + q1 * sin], dim=-1).view(q_shape)
        key = torch.cat([k1 * cos - k2 * sin,
                         k2 * cos + k1 * sin], dim=-1).view(k_shape)
        return query, key

    @staticmethod
    def forward_native_interleaved(positions, query, key, head_dim, cos_sin_cache):
        """Pure PyTorch GPT-J/interleaved RoPE matching CUDA IS_NEOX=false."""
        cos_sin = cos_sin_cache[positions]
        embed_dim = cos_sin.shape[-1] // 2
        cos = cos_sin[..., :embed_dim].unsqueeze(1)
        sin = cos_sin[..., embed_dim:].unsqueeze(1)

        q_shape = query.shape
        k_shape = key.shape
        q = query.view(q_shape[0], -1, head_dim)
        k = key.view(k_shape[0], -1, head_dim)

        q_even, q_odd = q[..., 0::2], q[..., 1::2]
        k_even, k_odd = k[..., 0::2], k[..., 1::2]

        q_rot = torch.stack(
            (q_even * cos - q_odd * sin,
             q_odd * cos + q_even * sin),
            dim=-1,
        ).flatten(-2)
        k_rot = torch.stack(
            (k_even * cos - k_odd * sin,
             k_odd * cos + k_even * sin),
            dim=-1,
        ).flatten(-2)
        return q_rot.view(q_shape), k_rot.view(k_shape)

    def forward_cuda(self, positions, query, key):
        """CUDA kernel path for eager mode.

        The live buffer goes straight to the kernel, which rounds each cos/sin
        value through ``query``'s dtype -- bit-for-bit what the reference's
        ``cache.to(query.dtype)`` produces.  Nothing is derived from the buffer,
        so nothing can be stale when a subclass rebinds or overwrites it.
        """
        _rope_op(
            positions, query, key, self.head_dim, self.cos_sin_cache,
            self.is_neox_style, True,
        )
        return query, key

    def forward(self, positions, query, key):
        if torch.compiler.is_compiling():
            cache = self.cos_sin_cache
            if cache.dtype != query.dtype:
                cache = cache.to(query.dtype)
            if self.is_neox_style:
                return self.forward_native(
                    positions, query, key, self.head_dim, cache,
                )
            return self.forward_native_interleaved(
                positions, query, key, self.head_dim, cache,
            )
        return self.forward_cuda(positions, query, key)


class Gemma4ProportionalRotaryEmbedding(RotaryEmbedding):
    """Gemma4 proportional RoPE.

    Gemma4 full-attention layers use a partial rotary factor, but the
    frequency exponents are divided by the full head dimension and the
    non-rotated angle pairs are represented as identity rotation.  This
    matches HF/vLLM's proportional RoPE instead of rotating a compact
    leading slice with ``rotary_dim`` as the denominator.
    """

    def __init__(
        self,
        head_dim: int,
        rotary_dim: int,
        max_position_embeddings: int,
        rope_theta: float,
    ):
        nn.Module.__init__(self)
        self.head_dim = head_dim
        self.is_neox_style = True
        rope_angles = rotary_dim // 2
        nope_angles = (head_dim // 2) - rope_angles

        inv_freq = 1.0 / (
            rope_theta ** (
                torch.arange(0, 2 * rope_angles, 2, dtype=torch.float) / head_dim
            )
        )
        if nope_angles > 0:
            inv_freq = torch.cat(
                [inv_freq, torch.zeros(nope_angles, dtype=torch.float)],
            )

        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1).float()
        self.register_buffer("cos_sin_cache", cache, persistent=False)
