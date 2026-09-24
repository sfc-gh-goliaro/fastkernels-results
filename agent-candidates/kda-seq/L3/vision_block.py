"""Qwen vision transformer block with the same contract as ``baseline.py``.

The baseline's dataflow is already the right one -- pre-norm, two residual adds --
and the attention and MLP underneath it are frozen winners whose internals are not
this module's business. What this file changes is the *information* it passes down
and the *rate* at which the block's own elementwise work runs.

``max_seqlen`` is the information. The frozen L2 ``VisionAttention`` admits its
no-copy strided q/k/v path only when ``max_seqlen`` is already a value, and routes
every rejected call to the L1 winner's *baseline* attention body rather than its
tuned FA4 launch, so forwarding a ``None`` costs both. Every captured call in this
workspace supplies a real ``max_seqlen``, so the scored path never needs the
derivation -- but a caller who omits it would otherwise pay for a value the
attention is about to compute one level down anyway, from the same ``cu_seqlens``,
with the same single host synchronisation. Deriving it here is therefore free where
it fires and absent where it does not.

The guard matters more than the derivation. ``cu_seqlens`` reaches this method
unvalidated, and the baseline hands a malformed one straight to the attention,
which raises there. Deriving from it unconditionally would move that error into
this block -- a different exception from a different line -- or, worse, succeed on
something that is not a cumulative-length vector. So the value is derived only for
a shape the attention would itself have derived from, and anything else is
forwarded untouched.

The rate is the four elementwise passes. ``norm1``, ``x + attn_out``, ``norm2`` and
``x1 + mlp_out`` are 17% of the call, and the frozen L1 LayerNorm is *issue*-bound
at this row width rather than bandwidth-bound: its dispatch ladder picks the
256-thread, one-vector-per-thread rung for ``n=1152``, but a 1152-wide bf16 row is
144 sixteen-byte vectors, so 112 of those 256 threads have no vector to load. Two
leases at different clocks separate the two kernel classes cleanly -- ``torch.add``
moved 1.21x with the core clock while the norm moved 1.54x, at less than half the
add's bandwidth on two thirds of the bytes. That is a mapping mismatch, not a
defect: the frozen kernel is tuned for the 4608-wide rows it was frozen on.

So two kernels replace three of the four passes:

    norm1              -> layer_norm(x, w1, b1, eps)
    x + attn_out,
    norm2              -> add_layer_norm(x, a, w2, b2, eps) -> (residual, normed)
    x1 + mlp_out       -> torch.add, unchanged

Folding the first add into ``norm2`` drops a whole read of the residual, taking the
elementwise traffic from ten row-passes to nine, and one launch disappears. The last
add stays ``torch.add``: it already runs at the DRAM rate a custom kernel would aim
for, and folding it into fc2's epilogue is a measured dead end -- ATen's ``addmm``
copies the full ``[M, N]`` input into the output and then runs a beta-GEMM, so the
elementwise add becomes a device-to-device copy of the same size.

Admission is two tiers, because a C++ operator cannot see a Python module. The
operator itself asks every tensor-level question -- dtype, device, contiguity, row
width, an implemented lane mapping, alignment, aliasing, grad mode -- and reproduces
the baseline *formula* when any of them fails, which is an equality rather than an
approximation. This method asks the handful of module-level questions C++ cannot:
that the extension built at all, that the norm really is the frozen ``LayerNorm``
rather than something assigned over it, that its affine is enabled and its
``promote_fp32`` is off, and that no forward hook is attached whose observation of
``norm1(x)`` bypassing it would be a silent behaviour change. Parametrizations,
autocast on a non-CUDA device, and forward-mode AD are outside this file's scope and
are named here rather than half-handled; the tensor-level clauses in the operator
turn away the subclass and autocast cases those would arrive as.

Neither kernel writes an input. The harness's shifting pool copies the pristine
source into a fresh slot every iteration and its correctness path hands the
candidate a clone, so an in-place write on ``x`` would be invisible on iteration one,
wrong from iteration two, and detected by nothing.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn

from ..L1.layer_norm import LayerNorm
from ..L1.quickgelu import QuickGELU
from ..L2.vision_attention import VisionAttention
from ..L2.vision_mlp import VisionMLP

# Unique to this file so a second import under a different module name cannot
# double-register the operators; registration happens once at ``.so`` load and
# Python's module cache makes any later import a no-op.
_LIBRARY_NAME = "fk_vb_norm"

_CUDA_SOURCE = r"""

#include <ATen/ATen.h>
#include <ATen/TensorSubclassLikeUtils.h>
#include <ATen/autocast_mode.h>
#include <ATen/core/grad_mode.h>
#include <torch/library.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>

#include <atomic>
#include <cstdint>
#include <optional>

// The mapping, as -D macros whose defaults are the ones that measured fastest.
// profile/p1-mapping/ab_mapping.py records all nine variants on harness geomean,
// which is the only figure a mapping may be chosen on -- the frozen L2 MLP's notes
// record a variant that won in isolation and lost in the harness.
//
// 1152 bf16 is 144 sixteen-byte vectors. The obvious mapping is the one that divides
// exactly, 16 lanes x 9 vectors over 8 rows per CTA, and it does fix the defect being
// answered: the frozen L1 ladder picks its 256-thread one-vector-per-thread rung for
// this width and leaves 112 of those threads idle. But it lost. A full warp per row
// with 5 vectors per lane -- 160 slots for 144 vectors, so the last is predicated --
// wins by 1.4121x to 1.3999x, because it needs 53 registers where the exact mapping
// needs 80, and because 4 rows per CTA launches twice the blocks. Grid parallelism
// turns out to matter more here than the idle-lane count: at the smallest scored
// shape the exact mapping issues only ceil(1760/8) = 220 CTAs over 148 SMs.
//
// The cache-policy split measured neutral and is off by default. See FK_VB_CACHE_HINTS.
#ifndef FK_VB_LANES_PER_ROW
#define FK_VB_LANES_PER_ROW 32
#endif
#ifndef FK_VB_VECS_PER_LANE
#define FK_VB_VECS_PER_LANE 5
#endif
#ifndef FK_VB_ROWS_PER_BLOCK
#define FK_VB_ROWS_PER_BLOCK 4
#endif
// Bytes per global access: 8 (uint2), 16 (uint4, the default) or 32 (two uint4s,
// issued as a 256-bit pair). Wider means fewer instructions and more registers.
#ifndef FK_VB_VEC_BYTES
#define FK_VB_VEC_BYTES 16
#endif
// Cache-policy differentiation on the loads: the row is read exactly once and should
// not be admitted to L1, while w/b are re-read by every row and should stay.
// KernelWiki's technique-vectorized-loads records 1.44x from exactly this split on a
// streaming/reused pair, but that was an NVFP4 GEMV, and it does not transfer here:
// on the winning mapping the split measured 1.4121x with the hints against 1.4126x
// without, a 0.04% difference against per-shape run-to-run variance of several
// percent. On the exact mapping it was worth +0.25% on geomean while *costing* 9% at
// the largest scored shape, whose working set is far past L2. So the tie is broken
// towards the plainer loads and the hints ship off, with the A/B kept as the record
// of why rather than deleted.
#ifndef FK_VB_CACHE_HINTS
#define FK_VB_CACHE_HINTS 0
#endif

namespace {

constexpr int kWarpSize = 32;
constexpr unsigned kFullMask = 0xffffffffu;

constexpr int kLanesPerRow = FK_VB_LANES_PER_ROW;
constexpr int kVecsPerLane = FK_VB_VECS_PER_LANE;
constexpr int kRowsPerBlock = FK_VB_ROWS_PER_BLOCK;
constexpr int kVecBytes = FK_VB_VEC_BYTES;
constexpr int kBlockThreads = kLanesPerRow * kRowsPerBlock;

static_assert(kLanesPerRow == 8 || kLanesPerRow == 16 || kLanesPerRow == 32,
              "the reduction is a power-of-two shuffle tree confined to one row's "
              "lane group, so the group must tile a warp");
static_assert(kVecBytes == 8 || kVecBytes == 16 || kVecBytes == 32,
              "one access is uint2, uint4, or a 256-bit uint4 pair");
static_assert(kBlockThreads % kWarpSize == 0 && kBlockThreads <= 1024,
              "a row's lane group must not straddle two warps");
static_assert(kVecsPerLane >= 1, "each lane owns at least one vector");

// bf16 only. Every scored call is bf16, and an untested dtype belongs on the
// baseline formula rather than on a kernel whose rounding was never checked for it.
constexpr int kElemsPerVec = kVecBytes / 2;

// ---------------------------------------------------------------------------
// One global access, as a register-resident chunk of bf16 pairs. Packed rather
// than unpacked to fp32: the row is held across both reduction passes, and four
// registers per 16 bytes instead of eight is what keeps that affordable.
// ---------------------------------------------------------------------------
struct Chunk {
#if FK_VB_VEC_BYTES == 8
  uint2 raw;
#elif FK_VB_VEC_BYTES == 16
  uint4 raw;
#else
  uint4 lo;
  uint4 hi;
#endif
};

__device__ __forceinline__ void chunk_zero(Chunk& c) {
#if FK_VB_VEC_BYTES == 8
  c.raw = make_uint2(0u, 0u);
#elif FK_VB_VEC_BYTES == 16
  c.raw = make_uint4(0u, 0u, 0u, 0u);
#else
  c.lo = make_uint4(0u, 0u, 0u, 0u);
  c.hi = make_uint4(0u, 0u, 0u, 0u);
#endif
}

// ``streaming`` picks the cache policy: the activation row is read once and is not
// worth an L1 line, while w/b are read again by every row in the grid. KernelWiki's
// technique-vectorized-loads records 1.44x from exactly this split on a
// streaming/reused pair (GPU Mode NVFP4 hackathon, a different workload -- a reason
// to measure it, not a prediction). ``nc`` marks the data read-only for the whole
// kernel, which every pointer here is.
template <bool kStreaming>
__device__ __forceinline__ Chunk chunk_load(const void* p) {
  Chunk c;
#if FK_VB_CACHE_HINTS
#if FK_VB_VEC_BYTES == 8
  if constexpr (kStreaming) {
    asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0,%1}, [%2];"
                 : "=r"(c.raw.x), "=r"(c.raw.y) : "l"(p));
  } else {
    asm volatile("ld.global.nc.L1::evict_last.v2.u32 {%0,%1}, [%2];"
                 : "=r"(c.raw.x), "=r"(c.raw.y) : "l"(p));
  }
#elif FK_VB_VEC_BYTES == 16
  if constexpr (kStreaming) {
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(c.raw.x), "=r"(c.raw.y), "=r"(c.raw.z), "=r"(c.raw.w)
                 : "l"(p));
  } else {
    asm volatile("ld.global.nc.L1::evict_last.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(c.raw.x), "=r"(c.raw.y), "=r"(c.raw.z), "=r"(c.raw.w)
                 : "l"(p));
  }
#else
  const uint4* q = reinterpret_cast<const uint4*>(p);
  if constexpr (kStreaming) {
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(c.lo.x), "=r"(c.lo.y), "=r"(c.lo.z), "=r"(c.lo.w)
                 : "l"(q));
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(c.hi.x), "=r"(c.hi.y), "=r"(c.hi.z), "=r"(c.hi.w)
                 : "l"(q + 1));
  } else {
    asm volatile("ld.global.nc.L1::evict_last.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(c.lo.x), "=r"(c.lo.y), "=r"(c.lo.z), "=r"(c.lo.w)
                 : "l"(q));
    asm volatile("ld.global.nc.L1::evict_last.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(c.hi.x), "=r"(c.hi.y), "=r"(c.hi.z), "=r"(c.hi.w)
                 : "l"(q + 1));
  }
#endif
#else
#if FK_VB_VEC_BYTES == 32
  const uint4* q = reinterpret_cast<const uint4*>(p);
  c.lo = q[0];
  c.hi = q[1];
#else
  c.raw = *reinterpret_cast<const decltype(c.raw)*>(p);
#endif
#endif
  return c;
}

__device__ __forceinline__ void chunk_store(void* p, const Chunk& c) {
#if FK_VB_VEC_BYTES == 32
  uint4* q = reinterpret_cast<uint4*>(p);
  q[0] = c.lo;
  q[1] = c.hi;
#else
  *reinterpret_cast<decltype(c.raw)*>(p) = c.raw;
#endif
}

// bf16 -> fp32 is exact, so the unpack is not a rounding decision. ``__bfloat1622-
// float2`` on the pair rather than two scalar converts, for the same reason the
// frozen L1 winner does it: one instruction per pair.
__device__ __forceinline__ void chunk_to_float(const Chunk& c, float* out) {
#if FK_VB_VEC_BYTES == 32
  const __nv_bfloat162* lo = reinterpret_cast<const __nv_bfloat162*>(&c.lo);
  const __nv_bfloat162* hi = reinterpret_cast<const __nv_bfloat162*>(&c.hi);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float2 f = __bfloat1622float2(lo[j]);
    out[2 * j] = f.x;
    out[2 * j + 1] = f.y;
  }
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float2 f = __bfloat1622float2(hi[j]);
    out[8 + 2 * j] = f.x;
    out[8 + 2 * j + 1] = f.y;
  }
#else
  const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&c.raw);
#pragma unroll
  for (int j = 0; j < kElemsPerVec / 2; ++j) {
    const float2 f = __bfloat1622float2(p[j]);
    out[2 * j] = f.x;
    out[2 * j + 1] = f.y;
  }
#endif
}

__device__ __forceinline__ Chunk chunk_from_float(const float* in) {
  Chunk c;
#if FK_VB_VEC_BYTES == 32
  __nv_bfloat162* lo = reinterpret_cast<__nv_bfloat162*>(&c.lo);
  __nv_bfloat162* hi = reinterpret_cast<__nv_bfloat162*>(&c.hi);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    lo[j] = __floats2bfloat162_rn(in[2 * j], in[2 * j + 1]);
  }
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    hi[j] = __floats2bfloat162_rn(in[8 + 2 * j], in[8 + 2 * j + 1]);
  }
#else
  __nv_bfloat162* p = reinterpret_cast<__nv_bfloat162*>(&c.raw);
#pragma unroll
  for (int j = 0; j < kElemsPerVec / 2; ++j) {
    p[j] = __floats2bfloat162_rn(in[2 * j], in[2 * j + 1]);
  }
#endif
  return c;
}

// Sum across one row's lane group. The xor tree with offsets below kLanesPerRow
// confines every exchange to the group, and the mask is the full warp because every
// lane of the warp is still executing -- an out-of-range row is predicated, never
// exited. Exiting a sub-warp group would leave a sibling group naming lanes that are
// gone, which is undefined; the frozen L1 winner may exit only because its row is
// warp-uniform.
__device__ __forceinline__ float row_reduce_sum(float v) {
#pragma unroll
  for (int offset = kLanesPerRow / 2; offset > 0; offset >>= 1) {
    v += __shfl_xor_sync(kFullMask, v, offset);
  }
  return v;
}

// ---------------------------------------------------------------------------
// The kernel. ``kFuseAdd`` folds a residual add into the same pass: the sum is
// rounded once to bf16, stored as the residual, and the statistics are taken from
// that rounded value -- what the baseline's separate ``x + a`` then LayerNorm
// computes. ``kExact`` drops the intra-row bounds test for the mapping that divides
// the row exactly, which is the one the defaults select.
// ---------------------------------------------------------------------------
template <bool kFuseAdd, bool kExact>
__global__ void __launch_bounds__(kBlockThreads)
fused_norm_kernel(const __nv_bfloat16* __restrict__ x,
                  const __nv_bfloat16* __restrict__ a,
                  __nv_bfloat16* __restrict__ residual,
                  __nv_bfloat16* __restrict__ y,
                  const __nv_bfloat16* __restrict__ weight,
                  const __nv_bfloat16* __restrict__ bias,
                  int64_t rows, int vecs_per_row, float inv_n, float eps) {
  const int group = threadIdx.x / kLanesPerRow;
  const int lane_in_row = threadIdx.x % kLanesPerRow;
  const int64_t row = static_cast<int64_t>(blockIdx.x) * kRowsPerBlock + group;
  // Predication, not exit: see row_reduce_sum.
  const bool active = row < rows;

  const int64_t row_offset =
      (active ? row : 0) * vecs_per_row * kVecBytes;
  const char* xr = reinterpret_cast<const char*>(x) + row_offset;
  char* yr = reinterpret_cast<char*>(y) + row_offset;
  // Formed only where they exist. ``a`` and ``residual`` are null in the norm-only
  // instantiation, and offsetting a null pointer is undefined even when the result
  // is never dereferenced -- the dead code the compiler removes is still UB in the
  // source it removes it from.
  const char* ar = nullptr;
  char* rr = nullptr;
  if constexpr (kFuseAdd) {
    ar = reinterpret_cast<const char*>(a) + row_offset;
    rr = reinterpret_cast<char*>(residual) + row_offset;
  }

  // Consecutive lanes take consecutive vectors within each pass, so every access
  // is one fully coalesced kVecBytes-wide transaction across the group. The
  // alternative -- lane_in_row * kVecsPerLane + j -- gives each lane a contiguous
  // private span and strides the warp, which is the same bytes in many more
  // sectors.
  Chunk held[kVecsPerLane];
  float sum = 0.0f;
#pragma unroll
  for (int j = 0; j < kVecsPerLane; ++j) {
    const int vec = j * kLanesPerRow + lane_in_row;
    const bool take = active && (kExact || vec < vecs_per_row);
    if (take) {
      held[j] = chunk_load<true>(xr + static_cast<size_t>(vec) * kVecBytes);
      if constexpr (kFuseAdd) {
        const Chunk av = chunk_load<true>(ar + static_cast<size_t>(vec) * kVecBytes);
        float xf[kElemsPerVec], af[kElemsPerVec];
        chunk_to_float(held[j], xf);
        chunk_to_float(av, af);
#pragma unroll
        for (int e = 0; e < kElemsPerVec; ++e) {
          // One fp32 add, one round. ATen's bf16 add promotes to fp32, adds, and
          // converts back with the same cvt.rn.bf16.f32, so this residual is the
          // baseline's bit for bit -- and the statistics below must see the
          // *rounded* value, because that is the tensor the baseline's norm2 reads.
          xf[e] = xf[e] + af[e];
        }
        held[j] = chunk_from_float(xf);
        chunk_store(rr + static_cast<size_t>(vec) * kVecBytes, held[j]);
      }
      float e[kElemsPerVec];
      chunk_to_float(held[j], e);
#pragma unroll
      for (int k = 0; k < kElemsPerVec; ++k) {
        sum += e[k];
      }
    } else {
      chunk_zero(held[j]);
    }
  }
  const float mean = row_reduce_sum(sum) * inv_n;

  float sq = 0.0f;
#pragma unroll
  for (int j = 0; j < kVecsPerLane; ++j) {
    const int vec = j * kLanesPerRow + lane_in_row;
    if (active && (kExact || vec < vecs_per_row)) {
      float e[kElemsPerVec];
      chunk_to_float(held[j], e);
#pragma unroll
      for (int k = 0; k < kElemsPerVec; ++k) {
        const float d = e[k] - mean;
        sq += d * d;
      }
    }
  }
  // rsqrtf(var + eps), not 1/(sqrt(var) + eps): at zero variance the first gives a
  // scale of 1/sqrt(eps) = 1000 and the second 1/(0 + eps) = 1e6, and the reference
  // computes the first. Population variance (divide by n), as ATen does.
  const float rstd = rsqrtf(row_reduce_sum(sq) * inv_n + eps);

#pragma unroll
  for (int j = 0; j < kVecsPerLane; ++j) {
    const int vec = j * kLanesPerRow + lane_in_row;
    if (active && (kExact || vec < vecs_per_row)) {
      float e[kElemsPerVec];
      chunk_to_float(held[j], e);
      // The frozen L1 winner's operation order, copied rather than rederived:
      // centre and scale, then multiply by w, then add b, each its own step. That
      // formulation is the one measured against ATen at this width under plain -O3.
#pragma unroll
      for (int k = 0; k < kElemsPerVec; ++k) {
        e[k] = (e[k] - mean) * rstd;
      }
      {
        // Copied into a local before unpacking: handing the unpack a global address
        // makes it read the halves separately, which the frozen winner measured as
        // four 32-bit loads per vector instead of one wide one.
        const Chunk wp = chunk_load<false>(
            reinterpret_cast<const char*>(weight)
            + static_cast<size_t>(vec) * kVecBytes);
        float wf[kElemsPerVec];
        chunk_to_float(wp, wf);
#pragma unroll
        for (int k = 0; k < kElemsPerVec; ++k) {
          e[k] *= wf[k];
        }
      }
      {
        const Chunk bp = chunk_load<false>(
            reinterpret_cast<const char*>(bias)
            + static_cast<size_t>(vec) * kVecBytes);
        float bf[kElemsPerVec];
        chunk_to_float(bp, bf);
#pragma unroll
        for (int k = 0; k < kElemsPerVec; ++k) {
          e[k] += bf[k];
        }
      }
      chunk_store(yr + static_cast<size_t>(vec) * kVecBytes, chunk_from_float(e));
    }
  }
}

// ---------------------------------------------------------------------------
// Host side: the eligibility predicate, the route counters that make a fused claim
// checkable, and the exact-baseline fallback.
// ---------------------------------------------------------------------------

// Which row widths have a mapping. Adjacent to the launcher so the predicate and
// the dispatch cannot drift apart -- and deliberately not ``n % 8 == 0``, which
// would admit widths this mapping has no lane assignment for.
inline bool has_vector_mapping(int vecs_per_row) {
  return vecs_per_row >= 1 && vecs_per_row <= kLanesPerRow * kVecsPerLane;
}

// A fused claim has to be falsifiable: a build that silently fell back must not be
// recordable as a win. Relaxed because these are only ever read after a
// synchronising call, and the cost is one uncontended increment against a kernel
// that moves tens of megabytes.
std::atomic<int64_t> g_fused_calls{0};
std::atomic<int64_t> g_fallback_calls{0};

inline bool is_aligned(const void* p, int bytes) {
  return (reinterpret_cast<uintptr_t>(p) % static_cast<uintptr_t>(bytes)) == 0;
}

// Byte ranges, not data_ptr equality: ``a`` may be a different view into the same
// storage, and the kernel's __restrict__ promises the compiler that cannot happen.
inline bool overlaps(const at::Tensor& p, const at::Tensor& q) {
  if (!p.defined() || !q.defined() || p.device() != q.device()) {
    return false;
  }
  const char* pb = reinterpret_cast<const char*>(p.const_data_ptr());
  const char* qb = reinterpret_cast<const char*>(q.const_data_ptr());
  const char* pe = pb + p.numel() * p.element_size();
  const char* qe = qb + q.numel() * q.element_size();
  return pb < qe && qb < pe;
}

inline bool param_ok(const at::Tensor& p, const at::Tensor& x, int64_t n) {
  return p.defined() && p.device() == x.device() &&
         p.scalar_type() == at::kBFloat16 && p.dim() == 1 && p.size(0) == n &&
         p.is_contiguous() && is_aligned(p.const_data_ptr(), kVecBytes);
}

// Every clause is asked before a pointer reaches a kernel, and the shape questions
// come before the empty-input question so an input the baseline would reject still
// reaches ATen and raises what ATen raises.
inline bool admits(const at::Tensor& x, const at::Tensor& a, const at::Tensor& w,
                   const at::Tensor& b, int64_t n, bool fuse_add) {
  if (!x.defined() || !x.is_cuda() || x.scalar_type() != at::kBFloat16) {
    return false;
  }
  // Functorch duals, fake and functional tensors and Python subclasses have no
  // ordinary storage to take a pointer to, and reach here without requires_grad
  // ever being set.
  if (at::isTensorSubclassLike(x) || at::isTensorSubclassLike(w) ||
      at::isTensorSubclassLike(b) || (fuse_add && at::isTensorSubclassLike(a))) {
    return false;
  }
  // Forward-mode AD is not reverse mode: a dual is an ordinary tensor with
  // requires_grad clear, and ``no_grad`` deliberately does not disable it, so
  // neither the grad-mode test nor the subclass test above sees it. A raw launch
  // into a fresh ``at::empty`` would return a correct primal carrying no tangent,
  // which is silently wrong rather than an error.
  // ``_fw_grad`` on the tensor rather than ``at::isFwGradDefined``: the latter is
  // declared in torch/csrc/autograd, outside the lean include set this source keeps
  // for build time, while the method is on TensorBase itself. Level 0 is the only
  // level ``dual_level()`` creates.
  if (x._fw_grad(0).defined() || w._fw_grad(0).defined() ||
      b._fw_grad(0).defined() ||
      (fuse_add && a.defined() && a._fw_grad(0).defined())) {
    return false;
  }
  // Autocast rewrites LayerNorm's output dtype and this operator has no autocast
  // registration; the baseline formula re-dispatches and picks the policy up exactly.
  if (at::autocast::is_autocast_enabled(x.device().type())) {
    return false;
  }
  // Only the *activations* are asked this, never ``w``/``b``. An ordinary
  // ``nn.Parameter`` has requires_grad set whether or not anyone will ever
  // differentiate it, so asking it of the affine would refuse every real module and
  // leave the fast path unreachable -- which is exactly what it did until the
  // whole-block route assertion in profile/p1-parity/ caught it. Grad *mode* being
  // disabled is what makes the raw launch safe, and that is tested above; the
  // activation clauses below are belt and braces for a caller who reaches here with
  // a graph leaf.
  if (x.requires_grad() || (fuse_add && a.requires_grad())) {
    return false;
  }
  if (n <= 0 || x.dim() < 2 || x.size(-1) != n || !x.is_contiguous()) {
    return false;
  }
  const int64_t total = x.numel();
  if (total == 0 || total % n != 0) {
    return false;
  }
  const int64_t rows = total / n;
  if (rows > static_cast<int64_t>(INT32_MAX)) {
    return false;
  }
  if (!param_ok(w, x, n) || !param_ok(b, x, n)) {
    return false;
  }
  // The mapping is stated in whole vectors, and the row pitch must be a whole
  // number of them or a row would start mid-vector.
  if (n % (kVecBytes / 2) != 0) {
    return false;
  }
  const int64_t vecs = n / (kVecBytes / 2);
  if (vecs > static_cast<int64_t>(INT32_MAX) ||
      !has_vector_mapping(static_cast<int>(vecs))) {
    return false;
  }
  if (!is_aligned(x.const_data_ptr(), kVecBytes)) {
    return false;
  }
  if (fuse_add) {
    if (!a.defined() || a.scalar_type() != at::kBFloat16 || !a.is_cuda() ||
        a.device() != x.device() || a.sizes() != x.sizes() ||
        !a.is_contiguous() || !is_aligned(a.const_data_ptr(), kVecBytes)) {
      return false;
    }
    // __restrict__ on both inputs promises they do not alias. ``a is x`` is a
    // legitimate pointwise x + x, so it is turned away rather than mis-executed.
    if (overlaps(a, x)) {
      return false;
    }
  }
  if (overlaps(w, x) || overlaps(b, x) || (fuse_add && (overlaps(w, a) ||
                                                       overlaps(b, a)))) {
    return false;
  }
  return true;
}

// Exactly what the baseline computes. ``promote_fp32`` is false on both norms in
// this block, so this is at::layer_norm on the native dtype -- an equality with the
// baseline expression, not a tolerance argument.
at::Tensor baseline_norm(const at::Tensor& x, const at::Tensor& w,
                         const at::Tensor& b, int64_t n, double eps) {
  const std::optional<at::Tensor> wo =
      w.defined() ? std::optional<at::Tensor>(w) : std::nullopt;
  const std::optional<at::Tensor> bo =
      b.defined() ? std::optional<at::Tensor>(b) : std::nullopt;
  return at::layer_norm(x, {n}, wo, bo, eps);
}

template <bool kFuseAdd>
void launch(const at::Tensor& x, const at::Tensor& a, at::Tensor& residual,
            at::Tensor& y, const at::Tensor& w, const at::Tensor& b, int64_t n,
            double eps) {
  const int64_t rows = x.numel() / n;
  const int vecs = static_cast<int>(n / (kVecBytes / 2));
  const unsigned grid =
      static_cast<unsigned>((rows + kRowsPerBlock - 1) / kRowsPerBlock);
  const auto stream = c10::cuda::getCurrentCUDAStream();
  const auto* xp = reinterpret_cast<const __nv_bfloat16*>(x.const_data_ptr());
  const auto* ap = kFuseAdd
      ? reinterpret_cast<const __nv_bfloat16*>(a.const_data_ptr()) : nullptr;
  auto* rp = kFuseAdd
      ? reinterpret_cast<__nv_bfloat16*>(residual.mutable_data_ptr()) : nullptr;
  auto* yp = reinterpret_cast<__nv_bfloat16*>(y.mutable_data_ptr());
  const auto* wp = reinterpret_cast<const __nv_bfloat16*>(w.const_data_ptr());
  const auto* bp = reinterpret_cast<const __nv_bfloat16*>(b.const_data_ptr());
  const float inv_n = 1.0f / static_cast<float>(n);
  const float epsf = static_cast<float>(eps);

  if (vecs == kLanesPerRow * kVecsPerLane) {
    fused_norm_kernel<kFuseAdd, true><<<grid, kBlockThreads, 0, stream>>>(
        xp, ap, rp, yp, wp, bp, rows, vecs, inv_n, epsf);
  } else {
    fused_norm_kernel<kFuseAdd, false><<<grid, kBlockThreads, 0, stream>>>(
        xp, ap, rp, yp, wp, bp, rows, vecs, inv_n, epsf);
  }
}

at::Tensor layer_norm(const at::Tensor& x, const std::optional<at::Tensor>& weight,
                      const std::optional<at::Tensor>& bias, int64_t n,
                      double eps) {
  const at::Tensor w = weight.has_value() ? *weight : at::Tensor();
  const at::Tensor b = bias.has_value() ? *bias : at::Tensor();
  // Grad mode being enabled is the whole test, not whether some tensor currently
  // requires grad: the fused path allocates with at::empty and launches raw, so it
  // records nothing, and a caller who has not entered no_grad may attach
  // requires_grad later in the same graph.
  if (at::GradMode::is_enabled() ||
      !admits(x, at::Tensor(), w, b, n, /*fuse_add=*/false)) {
    g_fallback_calls.fetch_add(1, std::memory_order_relaxed);
    return baseline_norm(x, w, b, n, eps);
  }
  const c10::cuda::CUDAGuard guard(x.device());
  at::Tensor y = at::empty(x.sizes(), x.options());
  at::Tensor none;
  launch<false>(x, none, none, y, w, b, n, eps);
  g_fused_calls.fetch_add(1, std::memory_order_relaxed);
  return y;
}

std::tuple<at::Tensor, at::Tensor> add_layer_norm(
    const at::Tensor& x, const at::Tensor& a,
    const std::optional<at::Tensor>& weight,
    const std::optional<at::Tensor>& bias, int64_t n, double eps) {
  const at::Tensor w = weight.has_value() ? *weight : at::Tensor();
  const at::Tensor b = bias.has_value() ? *bias : at::Tensor();
  if (at::GradMode::is_enabled() || !admits(x, a, w, b, n, /*fuse_add=*/true)) {
    g_fallback_calls.fetch_add(1, std::memory_order_relaxed);
    // The baseline's two expressions, in its order: the residual is a real tensor
    // the block goes on to use, so it is returned rather than recomputed.
    at::Tensor r = at::add(x, a);
    return std::make_tuple(r, baseline_norm(r, w, b, n, eps));
  }
  const c10::cuda::CUDAGuard guard(x.device());
  // Two fresh allocations. Writing the residual into ``x`` would be invisible on
  // the harness's first iteration and wrong from the second, and its correctness
  // path hands the candidate a clone -- so it would not be detected at all.
  at::Tensor residual = at::empty(x.sizes(), x.options());
  at::Tensor y = at::empty(x.sizes(), x.options());
  launch<true>(x, a, residual, y, w, b, n, eps);
  g_fused_calls.fetch_add(1, std::memory_order_relaxed);
  return std::make_tuple(residual, y);
}

// (fused, fallback) since the last reset. Read by profile/p1-parity/parity.py to
// assert that a recorded fused result was actually fused.
std::tuple<int64_t, int64_t> route_counts() {
  return std::make_tuple(g_fused_calls.load(std::memory_order_relaxed),
                         g_fallback_calls.load(std::memory_order_relaxed));
}

void reset_route_counts() {
  g_fused_calls.store(0, std::memory_order_relaxed);
  g_fallback_calls.store(0, std::memory_order_relaxed);
}

// The mapping the build actually compiled, so a recorded A/B row cannot name a
// variant the binary does not contain.
std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t> mapping() {
  return std::make_tuple(kLanesPerRow, kVecsPerLane, kRowsPerBlock, kVecBytes,
                         FK_VB_CACHE_HINTS);
}

}  // namespace

TORCH_LIBRARY(fk_vb_norm, m) {
  m.def("layer_norm(Tensor x, Tensor? weight, Tensor? bias, int n, float eps) "
        "-> Tensor",
        &layer_norm);
  m.def("add_layer_norm(Tensor x, Tensor a, Tensor? weight, Tensor? bias, int n, "
        "float eps) -> (Tensor, Tensor)",
        &add_layer_norm);
  m.def("route_counts() -> (int, int)", &route_counts);
  m.def("reset_route_counts() -> ()", &reset_route_counts);
  m.def("mapping() -> (int, int, int, int, int)", &mapping);
}
"""


def _load_fused_ops():
    """Build and register both operators, returning their bound overloads.

    Compilation happens here, at import, so nothing is deferred into a timed
    ``forward`` -- a lazy build would land inside the first timed iteration and, on
    the small shape, would dominate it. The includes are lean because
    ``<torch/extension.h>`` through nvcc dominates the build and none of it is
    needed: these are ``TORCH_LIBRARY`` registrations, not pybind.
    """
    import os

    from torch.utils.cpp_extension import load_inline

    # Without this, cpp_extension honours the ambient TORCH_CUDA_ARCH_LIST, which in
    # this environment names six architectures -- six nvcc passes over every
    # instantiation, for five targets that will never run the kernel. Derived from
    # the live device rather than hardcoded, and restored afterwards so no later
    # build in this process inherits it.
    previous_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
    try:
        load_inline(
            name=_LIBRARY_NAME,
            cpp_sources="",
            cuda_sources=_CUDA_SOURCE,
            # No fast-math and no -fmad=false: the frozen L1 winner's measured
            # agreement with ATen at this width was established under plain -O3, and
            # this file copies its operation order precisely so that the measurement
            # transfers. Changing the flags would forfeit that.
            extra_cuda_cflags=["-O3"],
            is_python_module=False,
            no_implicit_headers=True,
        )
    finally:
        if previous_arch_list is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = previous_arch_list
    lib = getattr(torch.ops, _LIBRARY_NAME)
    # Bind the overloads, not the packets: a packet re-resolves overloads from the
    # argument types on every call, and the small shape is launch-latency bound.
    return lib.layer_norm.default, lib.add_layer_norm.default


try:
    _fused_layer_norm, _fused_add_layer_norm = _load_fused_ops()
except Exception:  # noqa: BLE001 - a build that cannot happen must degrade, not
    # take the operator down with it. The frozen submodules then run unchanged and
    # the result is correct but unaccelerated, which is the whole point of staging
    # the composition floor first.
    _fused_layer_norm = _fused_add_layer_norm = None


def _install_route_report() -> None:
    """Report the compiled mapping and the route taken, if asked to, at exit.

    The route counters live in the ``.so``, so a parity script that reads them proves
    something about *its own* process and nothing about the benchmark's. Those are
    separate processes, and between them the build can fail, a cache can be stale, or
    a different mapping can be loaded -- after which the ``try`` above degrades
    silently and a fused row could be recorded for a run that fused nothing.

    So when ``FK_VB_ROUTE_LOG`` names a file, every process that imports this module
    appends one JSON line saying which mapping it compiled and how many calls took
    each path. ``profile/record.py`` refuses to write a fused row without it. Inert
    and free unless the variable is set, and wrapped so that a reporting failure can
    never take the operator down.
    """
    import atexit
    import json
    import os

    path = os.environ.get("FK_VB_ROUTE_LOG")
    if not path:
        return

    def _report():
        try:
            if _fused_layer_norm is None:
                row = {"pid": os.getpid(), "loaded": False}
            else:
                lib = getattr(torch.ops, _LIBRARY_NAME)
                fused, fallback = lib.route_counts.default()
                lanes, vecs, rows, vec_bytes, hints = lib.mapping.default()
                row = {"pid": os.getpid(), "loaded": True,
                       "fused": fused, "fallback": fallback,
                       "mapping": [lanes, vecs, rows, vec_bytes, hints]}
            with open(path, "a") as fh:
                fh.write(json.dumps(row) + "\n")
        except Exception:  # noqa: BLE001 - observability must not break the operator
            pass

    atexit.register(_report)


try:
    _install_route_report()
except Exception:  # noqa: BLE001
    pass

# The module-level hook registries are consulted by reference, not copied: they are
# mutated in place by ``register_module_forward_hook``, so a snapshot taken at import
# would go stale.
_GLOBAL_FORWARD_PRE_HOOKS = nn.modules.module._global_forward_pre_hooks
_GLOBAL_FORWARD_HOOKS = nn.modules.module._global_forward_hooks
_GLOBAL_BACKWARD_PRE_HOOKS = nn.modules.module._global_backward_pre_hooks
_GLOBAL_BACKWARD_HOOKS = nn.modules.module._global_backward_hooks


def _derivable_cu_seqlens(cu_seqlens: object) -> bool:
    """Whether ``max_seqlen`` may be derived from this ``cu_seqlens``.

    An allowlist of exactly what the frozen attention's own derivation needs:
    ``(cu_seqlens[1:] - cu_seqlens[:-1]).max().item()`` requires at least two
    entries to have a segment at all, one dimension for the slices to mean what
    they say, an integer dtype so the subtraction is exact, and CUDA residence so
    the sync is the one the attention would have paid rather than an extra
    host-side round trip. ``type(...) is torch.Tensor`` rather than ``isinstance``:
    a subclass with only ``__torch_function__`` is unwrapped on the way into ATen,
    so it has to be screened in Python or not at all.
    """
    return (type(cu_seqlens) is torch.Tensor
            and cu_seqlens.is_cuda
            and cu_seqlens.dim() == 1
            and cu_seqlens.numel() >= 2
            and not cu_seqlens.is_floating_point()
            and not cu_seqlens.is_complex())


def _fusable_norm(norm: nn.Module) -> int:
    """The row width to fuse over, or 0 if this norm must run its own ``forward``.

    Only the module-level facts, all of them things the operator cannot see from its
    tensors. ``type(...) is`` rather than ``isinstance``: a subclass may override
    ``forward``, and reading the parameters off it while skipping that override would
    compute something the caller did not ask for. A forward hook is the same problem
    one level out -- it was registered to observe or rewrite this norm's call, and
    bypassing the module would silently stop it firing.
    """
    if _fused_layer_norm is None or type(norm) is not LayerNorm:
        return 0
    if not norm.elementwise_affine or norm.promote_fp32:
        return 0
    if norm.weight is None or norm.bias is None:
        return 0
    if (norm._forward_pre_hooks or norm._forward_hooks
            or norm._backward_pre_hooks or norm._backward_hooks
            or _GLOBAL_FORWARD_PRE_HOOKS or _GLOBAL_FORWARD_HOOKS
            or _GLOBAL_BACKWARD_PRE_HOOKS or _GLOBAL_BACKWARD_HOOKS):
        return 0
    # ``type(...) is LayerNorm`` above does not cover a ``forward`` assigned onto the
    # *instance*, which shadows the class's method without changing its type, nor the
    # wrapper ``torch.compile`` installs. Both would be bypassed by calling the
    # operator directly, so both retire the fast path.
    if "forward" in vars(norm) or norm._compiled_call_impl is not None:
        return 0
    shape = norm.normalized_shape
    # Read per call, never cached: ``normalized_shape``, ``eps``, ``weight`` and
    # ``bias`` are public mutable attributes, and a cached copy would disagree with
    # the baseline the moment a caller wrote to one of them.
    if len(shape) != 1:
        return 0
    # ``int(...)`` rather than a type test would silently accept a float width and
    # normalise over a rounded row, where ``F.layer_norm`` rejects the shape outright.
    n = shape[0]
    if type(n) is not int or n <= 0:
        return 0
    return n


class VisionBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int,
                 mlp_hidden_dim: int,
                 act_fn: Callable[[torch.Tensor], torch.Tensor] = QuickGELU(),
                 norm_eps: float = 1e-6):
        super().__init__()
        # Constructed exactly as the baseline constructs them, names included: the
        # harness shares weights with ``load_state_dict(..., strict=False)``, where a
        # renamed submodule is silently left at its random init rather than raising.
        # Nothing here reads a parameter value -- the state dict arrives afterwards.
        self.norm1 = LayerNorm(embed_dim, eps=norm_eps, promote_fp32=False)
        self.norm2 = LayerNorm(embed_dim, eps=norm_eps, promote_fp32=False)
        self.attn = VisionAttention(embed_dim, num_heads)
        self.mlp = VisionMLP(embed_dim, mlp_hidden_dim, act_fn=act_fn)

    def forward(
        self, x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        if max_seqlen is None and _derivable_cu_seqlens(cu_seqlens):
            # The value the attention would derive itself, computed once. Its own
            # ``if max_seqlen is None`` branch is then not taken, so this is the
            # same single synchronisation rather than a second one. Issued before
            # ``norm1`` because that ordering measured 1.0069x faster in one lease
            # than issuing it after (profile/p1-contract/derive_position.py).
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()

        n1 = _fusable_norm(self.norm1)
        h = (_fused_layer_norm(x, self.norm1.weight, self.norm1.bias, n1,
                               self.norm1.eps)
             if n1 else self.norm1(x))

        a = self.attn(
            h, cu_seqlens, rotary_pos_emb_cos, rotary_pos_emb_sin, max_seqlen,
        )

        n2 = _fusable_norm(self.norm2)
        if n2:
            x, h2 = _fused_add_layer_norm(x, a, self.norm2.weight,
                                          self.norm2.bias, n2, self.norm2.eps)
        else:
            x = x + a
            h2 = self.norm2(x)

        return x + self.mlp(h2)
