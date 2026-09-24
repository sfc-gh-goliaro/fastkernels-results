"""Patch merger for Qwen vision encoders.

Merges spatial_merge_size^2 patches into one via norm + 2-layer MLP.

Unified across Qwen2-VL and Qwen3-VL:
  - use_postshuffle_norm: Qwen3 DeepStack mergers norm after spatial reshape.

The module tree is the baseline's, exactly: ``self.norm`` / ``self.fc1`` /
``self.act`` / ``self.fc2``, same parameter names, shapes and dtypes. The bench
shares weights with ``load_state_dict(baseline.state_dict(), strict=False)``
inside a bare ``try/except Exception: pass``, so a renamed or reshaped parameter
would not raise -- it would leave that parameter holding the harness's random
re-roll and quietly compare two different functions. Keeping the tree identical
is what makes the comparison mean anything.

Keeping ``fc1`` / ``act`` / ``fc2`` as module *forwards* rather than open-coding
``F.linear`` is the same argument one level up: the TP split, the FP8 branch, the
rank-0-only bias and the all-reduce all stay correct by delegation. The bench
happens to run this operator at ``tp_size() == 1`` with no quantization, but
nothing here depends on that.

Two things are different from the baseline, and only two.

**The imports.** ``from ..L1.layer_norm import LayerNorm`` and
``from ..L1.gelu import GELU`` resolve to the frozen L1 winners under the
candidate finder, while the baseline reference keeps ATen's ``F.layer_norm`` and
``F.gelu``. That alone measures about 1.11x geomean on the scored shapes.
``from .parallel_linear import ...`` has no candidate file, so it aliases to the
baseline's plain-``F.linear`` implementation and both sides run the same cuBLAS
GEMMs.

**One kernel.** ``self.norm(x).view(-1, self.hidden_size)`` is replaced, on the
non-postshuffle path only, by a single kernel that LayerNorms each ``C``-wide row
and writes it into the merged ``[M, hidden]`` buffer. Note what that is *not*: the
``.view()`` launches no kernel and moves no byte, so there is no fusion gain here.
The whole win is a LayerNorm mapping built for this row width.

The frozen L1 LayerNorm is excellent on the 4608-wide rows of its own capture mix
and mismatched to a 1152-wide one. 1152 bf16 elements are exactly 144 16-byte
vectors, and the frozen dispatch ladder hands 144 vectors to a block-per-row rung
with 256 threads at one vector per thread: 112 of 256 threads idle, one 16-byte
load in flight per thread, and an eight-warp block reduction with a
``__syncthreads`` per pass.

The kernel here gives each row to one warp -- 32 lanes, 8 rows per CTA -- so the
reduction is five shuffles with no barrier, every lane is busy, and five loads are in
flight per lane. Measured at N=20680, on logical traffic (read the input once, write the
output once): the frozen kernel reaches 1.48 TB/s, this one 2.41 TB/s, and a bare
``clone()`` of the same bytes 3.10 TB/s -- so roughly 78% of a copy. NCU's *physical*
DRAM counters read far lower (0.85 and 1.37 TB/s) because only 4.7-6.7 MB of the 47.6 MB
output ever reaches DRAM; the rest is still dirty in a 133 MB L2 when the kernel ends.
The instruction count is the other half of the story: 29.8M for the frozen kernel against
16.7M here, for the same bytes.

The mapping was chosen by a measured A/B over lanes x rows with every loser still
reachable by macro. See the notes on FK_VPM_LANES below, because the mapping that wins
the norm in isolation is *not* the one that wins the operator, and picking on the
isolated measurement would have shipped a mapping that regresses the smallest scored
shape.

Everything else in this operator was measured and left alone: the two cuBLAS GEMMs
are 78% of the op at 1440-1511 TFLOPS, and the frozen GELU is already at the copy
roofline. Three fusions were measured and *lost* -- a cuBLASLt bias+GELU epilogue,
owning ``fc1`` with Triton, and M-chunking for L2 reuse -- and are recorded in
``docs/plan.md`` so they are not re-litigated.

Anything the kernel does not cover -- the postshuffle path, fp32, a
non-contiguous or misaligned input, a row width that is not a multiple of the
16-byte vector, a CPU tensor, a call that needs gradients, an empty input -- takes
the baseline formula. The eligibility predicate is evaluated on the host, inside
the operator, before any pointer is dereferenced: a launch that starts has to be
a launch that is correct.

Environment hooks, all off by default:

``FK_VPM_EXT=0``          disable the extension without breaking its build
``FK_VPM_BREAK_BUILD=1``  append an invalid nvcc flag, so the failed-build path
                          is exercised for real rather than mocked
``FK_VPM_LANES``          lanes per row: 16 or 32 (default, measured)
``FK_VPM_ROWS``           rows per CTA: 2, 4 or 8 (default, measured)
``FK_VPM_CHUNK``          16-byte vectors per lane per step: 1 (default) or 2,
                          i.e. a 128-bit or a 256-bit access tile
``FK_VPM_MAX_WAVES``      cap the grid at this many waves and grid-stride over
                          rows; 0 (default) is uncapped, one CTA per row group
``FK_VPM_CACHE_HINTS=1``  L1::no_allocate on the streamed x, L1::evict_last on
                          the row-invariant gamma/beta
``FK_VPM_LD256=1``        issue each 32-byte tile as one 256-bit
                          ``ld.global.v4.u64`` instead of two 128-bit loads;
                          needs ``FK_VPM_CHUNK=2``
``FK_VPM_ONEPASS_VAR=1``  E[x^2]-mu^2 instead of a second pass: a negative
                          control for the numerics suite, never shipped
``FK_VPM_BF16_ACC=1``     round the reduction accumulator to the storage dtype
                          on every add: negative control, never shipped
``FK_VPM_BF16_AFFINE=1``  apply gamma/beta in the storage dtype instead of
                          upcast: negative control, never shipped
``FK_VPM_SKIP_ALIGN_CHECK=1``   drop the 16-byte alignment check from the
                          predicate: negative control, never shipped
``FK_VPM_SKIP_CONTIG_CHECK=1``  accept any dense tensor instead of a contiguous
                          one: negative control, never shipped
``FK_VPM_IGNORE_GRAD=1``  take the fused path under grad mode: negative control,
                          never shipped
``FK_VPM_PTXAS_V=1``      -Xptxas -v, for register counts
``FK_VPM_BUILD_DIR``      override the workspace-local build directory
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch
import torch.nn as nn

from ..L1.gelu import GELU
from ..L1.layer_norm import LayerNorm
from .parallel_linear import ColumnParallelLinear, RowParallelLinear

# ---------------------------------------------------------------------------
# The kernel.
#
# Registration is pybind under a content-hashed extension name rather than
# TORCH_LIBRARY under a fixed namespace, for one concrete reason: the mapping A/B
# and the numerics suite both need two builds of this source alive in one process
# (the shipping mapping against a challenger, the two-pass variance against the
# one-pass control), and a fixed TORCH_LIBRARY namespace makes the second load a
# duplicate-registration error. A hashed pybind module is per-build unique by
# construction, and cannot collide with the frozen L1 LayerNorm's ``fk_ln_cand``
# namespace or with any vendored extension name.
#
# Loading two builds at once has one trap, and it is silent. The three entry
# points below have external linkage and are compiled ``-fPIC``, so calls to them
# go through the PLT and are *preemptible*: once the first extension is in the
# process, a second one's pybind glue resolves ``norm_merge`` to the *first*
# library's definition, and every variant measures the shipping kernel while
# reporting the variant's name. Measured, before the fix: three separately
# compiled controls -- one-pass variance, a rounded accumulator, a low-precision
# affine -- returned bit-identical results to the shipping build on three
# different cases in three different dtypes, which is not a numerical coincidence.
# ``-fvisibility=hidden`` keeps the symbols out of the dynamic table so each
# library binds its own, and the numerics controls are what would catch a
# regression here: if two builds ever agree bit-for-bit again, they fail.
# ---------------------------------------------------------------------------
_CPP_SOURCE = r"""
#include <ATen/ATen.h>
#include <cstdint>
#include <string>

at::Tensor norm_merge(const at::Tensor& x, const at::Tensor& weight,
                      const at::Tensor& bias, int64_t n, int64_t hidden,
                      double eps);
int64_t last_path();
void reset_last_path();
std::string config();
"""

_CUDA_SOURCE = r"""
#include <ATen/ATen.h>
#include <ATen/TensorSubclassLikeUtils.h>
#include <ATen/autocast_mode.h>
#include <ATen/core/grad_mode.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cmath>
#include <cstdint>
#include <optional>
#include <sstream>
#include <string>

// Lanes that share one row. 16 and 32 are the only values that keep the whole
// reduction inside a single warp: 64 or 128 would reintroduce the eight-warp
// block barrier that is the frozen kernel's problem, and 144 vectors is not
// divisible by either anyway.
//
// 32 ships, and it is *not* what an isolated measurement of the norm picks. Timed on
// its own with an L2 flush, 16 lanes x 8 rows is the fastest norm by a clear margin
// (geomean 1.579x over the frozen L1 kernel against 1.540x for 32 x 4), which is also
// the prior this workspace's frozen kernels established -- fewer threads per row,
// more vectors per thread (l2_norm.py: (128, 2) at 13.28 us against (256, 1) at
// 15.31 us). Inside the operator the ranking inverts: whole-op medians of five
// validate.py runs are 1.186x for 32 x 4 against 1.183x for 16 x 4 and 1.178x for
// 16 x 8.
//
// What separates them is the smallest scored shape, N=1760. Every 16-lane mapping is
// 0.93-0.94x of the frozen kernel there and costs the whole op about 2%; the 32-lane
// mappings are 1.00-1.07x and gain about 2%. 1760 rows at 16 lanes x 8 rows is 220
// CTAs of 128 threads on 148 SMs -- one and a half waves, and a row count too small
// to cover the mapping's lower thread count. t5_layer_norm.py records the same
// failure mode from the other side: a 32-lanes-per-row order "on its own costs most
// of the kernel's parallelism -- 512 rows would be 16k threads instead of 131k".
// This operator's captured envelope reaches down to 1760 rows, so it is inside that
// regime, and the mapping that wins the average is not the one that wins everywhere.
//
// A row-count-dependent dispatch -- 32 lanes for few rows, 16 for many -- would take
// both. It needs two lane counts instantiated and selected at runtime, which is a
// larger change than a tuning knob; recorded in docs/phase1-results.md rather than
// attempted here.
#ifndef FK_VPM_LANES
#define FK_VPM_LANES 32
#endif
// Rows per CTA. 8 ships, chosen by the rule declared in
// docs/selection-rule-round2.md before the deciding measurement was taken.
//
// Measured whole-op on the corrected kernel, five interleaved repeats of the floor and
// every stage-qualified contender in one session (median geomean, and delta against the
// floor per shape):
//
//   32x8   1.20198   +0.10 +5.02 +13.45 +2.63 +6.40   ahead on 5/5   <- ships
//   16x8   1.19794   +0.05 +1.09 +12.44 +2.38 +7.64   ahead on 5/5
//   32x4   1.18533   +0.08 -0.09  +3.93 +4.27 +3.49   ahead on 4/5
//   16x8+waves=8  1.18238  +0.07 +9.09 +11.45 +4.56 +4.77  ahead on 5/5
//
// Every range overlaps every other, so the geomean alone does not separate them; the
// rule's tie-break does. 32x8 is ahead of the floor on all five shapes where 32x4 is
// ahead on four, and among the 5/5 group it carries 64 registers per thread against the
// 16-lane mappings' 95 -- the property most likely to matter on a shape outside the
// capture. 4 and 8 both own a contiguous output region, since four source rows merge into
// one output row; 2 is behind both.
#ifndef FK_VPM_ROWS
#define FK_VPM_ROWS 8
#endif
// 16-byte vectors per lane per step. 1 is a 128-bit tile (144 vectors per row);
// 2 is a 256-bit tile (72). The workspace's own frozen kernels disagree about
// this -- rms_norm_gated.py and silu_and_mul_kernels.cu both ship 32-byte tiles
// -- so it is a measured axis, not an argument.
#ifndef FK_VPM_CHUNK
#define FK_VPM_CHUNK 1
#endif
// Cap the grid at this many waves so each CTA grid-strides over row groups and
// amortizes its staged gamma/beta. 0 is uncapped. Genuinely unsettled on this
// machine: rms_norm_kernels.cu measured capping worth 826 -> 700 us on its widest
// shape, rms_norm_gated.py found it never won, and silu_and_mul_kernels.cu
// measured it *costing* 4.7%.
#ifndef FK_VPM_MAX_WAVES
#define FK_VPM_MAX_WAVES 0
#endif
// Differentiated L1 policies: no_allocate for the streamed x, evict_last for the
// gamma/beta that every row re-reads. KernelWiki reports 1.44x from this on a
// bandwidth-bound GEMV; gamma and beta are only 2.3 KB each here, so the payoff
// is expected to be far smaller. Measured on its own axis.
#ifndef FK_VPM_CACHE_HINTS
#define FK_VPM_CACHE_HINTS 0
#endif
// Issue each 32-byte tile as one 256-bit ld.global.v4.u64 rather than two
// 128-bit LDG.128s. Requires FK_VPM_CHUNK=2, and a 32-byte-aligned row: the
// predicate only guarantees 16, so the kernel takes a grid-uniform flag from the
// host and falls back to the pair of 128-bit loads when the row is not aligned.
// Without this, "256-bit access width" would only ever mean "a 32-byte tile made
// of two 128-bit loads", which is what the frozen kernels ship and is a different
// measurement.
#ifndef FK_VPM_LD256
#define FK_VPM_LD256 0
#endif
// The negative control for the numerics suite: E[x^2]-mu^2 in one pass instead of
// a true second pass over the row. Catastrophic cancellation on a large-mean,
// tiny-variance row is the failure it is built to demonstrate. Never shipped.
#ifndef FK_VPM_ONEPASS_VAR
#define FK_VPM_ONEPASS_VAR 0
#endif
// Two more negative controls, for the same reason: a claim that fp32 accumulation
// and in-register affine upcast are load-bearing is only worth making if breaking
// them is measurably worse. Neither is ever shipped.
#ifndef FK_VPM_BF16_ACC
#define FK_VPM_BF16_ACC 0
#endif
#ifndef FK_VPM_BF16_AFFINE
#define FK_VPM_BF16_AFFINE 0
#endif
// Three more negative controls, on the *predicate* rather than the arithmetic. A
// guard is only worth having if removing it breaks something, so each of these
// drops one check and is expected to produce a wrong answer, a fault, or a
// non-differentiable result. None is ever shipped, and each is built into its own
// hashed library so it cannot be confused with the shipping one.
#ifndef FK_VPM_SKIP_ALIGN_CHECK
#define FK_VPM_SKIP_ALIGN_CHECK 0
#endif
#ifndef FK_VPM_SKIP_CONTIG_CHECK
#define FK_VPM_SKIP_CONTIG_CHECK 0
#endif
#ifndef FK_VPM_IGNORE_GRAD
#define FK_VPM_IGNORE_GRAD 0
#endif

namespace {

constexpr int kWarpSize = 32;
constexpr int kLanesPerRow = FK_VPM_LANES;
constexpr int kRowsPerCta = FK_VPM_ROWS;
constexpr int kChunk = FK_VPM_CHUNK;
constexpr int kBlockThreads = kLanesPerRow * kRowsPerCta;
// The shuffle mask names exactly the lanes that exist. A CTA of 16 threads is
// half a warp, and a full mask there would name lanes that never arrive.
constexpr unsigned kShflMask =
    (kBlockThreads >= kWarpSize) ? 0xffffffffu
                                 : ((1u << kBlockThreads) - 1u);

static_assert(kLanesPerRow == 16 || kLanesPerRow == 32,
              "the row reduction must stay inside one warp");
static_assert(kRowsPerCta == 1 || kRowsPerCta == 2 || kRowsPerCta == 4 ||
                  kRowsPerCta == 8,
              "rows per CTA is a measured axis over {2,4,8}, with 1 for probing");
static_assert(kChunk == 1 || kChunk == 2, "128-bit or 256-bit access tile");
static_assert(kBlockThreads % kWarpSize == 0 || kBlockThreads < kWarpSize,
              "a CTA is whole warps, or a single sub-warp");

// ---------------------------------------------------------------------------
// 16 bytes is the widest single global access the SM offers. The row is held in
// registers in this *packed* form across all three passes -- 4 registers per
// vector instead of the 8 an fp32 unpack would need -- and unpacked on the fly
// each pass. Nine uint4s is 36 registers; nine unpacked vectors would be 72
// before the affine temporaries, and this kernel has ALU to spare and no
// registers to spare.
// ---------------------------------------------------------------------------
template <typename T>
struct Packed;

template <>
struct Packed<__nv_bfloat16> {
  static constexpr int kElems = 8;
  // Round trip through the storage dtype. Only the low-precision negative
  // controls use it; the shipping path rounds exactly once, on the store.
  __device__ __forceinline__ static float round_storage(float v) {
    return __bfloat162float(__float2bfloat16_rn(v));
  }
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
  static constexpr int kElems = 8;
  __device__ __forceinline__ static float round_storage(float v) {
    return __half2float(__float2half_rn(v));
  }
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

// ---------------------------------------------------------------------------
// Loads. The default is a plain dereference, which nvcc turns into LDG.E.128.
// The hinted variants are a separately measured axis, not a default: x is
// streamed once and never re-read, so keeping it out of L1 leaves the whole
// cache for gamma and beta, which every row wants.
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint4 load_x(const uint4* __restrict__ p) {
#if FK_VPM_CACHE_HINTS && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p)
               : "memory");
  return v;
#else
  return *p;
#endif
}

#if FK_VPM_LD256
// One 256-bit load into two uint4s. The pair has to be adjacent in memory, which
// is exactly what a chunk of two consecutive vectors is.
__device__ __forceinline__ void load_x_256(const uint4* __restrict__ p, uint4* out) {
  unsigned long long a, b, c, d;
  asm volatile("ld.global.nc.v4.u64 {%0, %1, %2, %3}, [%4];"
               : "=l"(a), "=l"(b), "=l"(c), "=l"(d)
               : "l"(p)
               : "memory");
  out[0] = make_uint4(static_cast<unsigned>(a), static_cast<unsigned>(a >> 32),
                      static_cast<unsigned>(b), static_cast<unsigned>(b >> 32));
  out[1] = make_uint4(static_cast<unsigned>(c), static_cast<unsigned>(c >> 32),
                      static_cast<unsigned>(d), static_cast<unsigned>(d >> 32));
}
#endif

__device__ __forceinline__ uint4 load_affine(const uint4* __restrict__ p) {
#if FK_VPM_CACHE_HINTS && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  uint4 v;
  asm volatile("ld.global.nc.L1::evict_last.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p)
               : "memory");
  return v;
#else
  return *p;
#endif
}

// Butterfly over exactly kLanesPerRow lanes. With 16 lanes the xor offsets stay
// below 16, so each 16-aligned half of the warp reduces independently and every
// lane ends with its own row's sum -- no shared memory, no barrier, and no
// broadcast from a single lane. The full mask is valid because no lane exits
// early: out-of-range rows are *masked*, not returned from.
__device__ __forceinline__ float row_reduce_sum(float v) {
#pragma unroll
  for (int off = kLanesPerRow / 2; off > 0; off >>= 1) {
    v += __shfl_xor_sync(kShflMask, v, off);
  }
  return v;
}

// Reinterpret an affine array as 16-byte vectors. A named helper only so the
// kernel body reads the same way for gamma and beta.
template <typename T>
__device__ __forceinline__ const uint4* gv_of(const T* p) {
  return reinterpret_cast<const uint4*>(p);
}

__device__ __forceinline__ float row_reduce_max(float v) {
#pragma unroll
  for (int off = kLanesPerRow / 2; off > 0; off >>= 1) {
    v = fmaxf(v, __shfl_xor_sync(kShflMask, v, off));
  }
  return v;
}

// Largest |x| for which an fp32 sum of squares over a row cannot overflow. The
// binding constraint is the *variance* accumulator: the widest row this kernel
// admits is 1536 vectors (the 48 KiB shared-memory limit on the staged affine
// arrays), so 12288 elements, and 12288 * (4.7e17)^2 is about FLT_MAX. 1e16 leaves
// two orders of margin.
//
// This threshold exists because bf16 reaches 3.39e38 and fp16 65504, and both are
// finite inputs a correct LayerNorm has to handle. Above it the row is normalised
// by a power of two before any accumulation -- see the kernel.
constexpr float kScaleSafeMax = 1.0e16f;

// ---------------------------------------------------------------------------
// One kernel: LayerNorm each C-wide row of x and write it into the merged
// output buffer. The merge is a reinterpretation of the same bytes -- source row
// r lands at flat offset r*C in an output shaped [rows*C/hidden, hidden] -- so it
// costs no kernel, no copy, and does not appear in the indexing below at all.
//
// kChunksPerLane is the compile-time bound on how many access tiles a lane owns.
// The host picks the instantiation from the row width, so the loop is fully
// unrolled and every load is issued before the first reduction consumes any of
// them.
// ---------------------------------------------------------------------------
template <typename T, int kChunksPerLane>
__global__ void __launch_bounds__(kBlockThreads)
norm_merge_kernel(const T* __restrict__ x, T* __restrict__ y,
                  const T* __restrict__ gamma, const T* __restrict__ beta,
                  int64_t rows, int vecs_per_row, float n, float eps,
                  bool wide_ok) {
  constexpr int kElems = Packed<T>::kElems;
  constexpr int kVecsPerLane = kChunk * kChunksPerLane;

  const int lane = static_cast<int>(threadIdx.x) % kLanesPerRow;
  const int row_in_cta = static_cast<int>(threadIdx.x) / kLanesPerRow;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * kRowsPerCta;
  int64_t base = static_cast<int64_t>(blockIdx.x) * kRowsPerCta;

  uint4 packed[kVecsPerLane];

  // The one and only read of each row. Issued as a whole group before anything
  // consumes it, so nvcc emits all kVecsPerLane LDG.128s up front instead of
  // interleaving a load with the reduction that eats it. This memory-level
  // parallelism is what the frozen kernel's one-vector-per-thread mapping cannot
  // have.
  auto load_group = [&](int64_t group_base) {
    const int64_t row = group_base + row_in_cta;
    const bool in_row = row < rows;
    const uint4* __restrict__ xv =
        reinterpret_cast<const uint4*>(x) +
        (in_row ? row : 0) * static_cast<int64_t>(vecs_per_row);
#pragma unroll
    for (int c = 0; c < kChunksPerLane; ++c) {
      const int chunk = lane + c * kLanesPerRow;
      const int base_idx = chunk * kChunk;
#if FK_VPM_LD256
      if (kChunk == 2 && wide_ok && in_row && base_idx + 1 < vecs_per_row) {
        load_x_256(xv + base_idx, &packed[c * kChunk]);
        continue;
      }
#endif
#pragma unroll
      for (int k = 0; k < kChunk; ++k) {
        const int idx = base_idx + k;
        packed[c * kChunk + k] = (in_row && idx < vecs_per_row)
                                     ? load_x(xv + idx)
                                     : make_uint4(0u, 0u, 0u, 0u);
      }
    }
  };

  // Order matters here, and it is the whole point of this arrangement: the first
  // row group's loads are issued *first*, and gamma/beta are staged into shared
  // memory while those loads are still in flight. An earlier version staged the
  // affine arrays and crossed the barrier before issuing any x load, which is the
  // same instructions with none of the overlap.
  //
  // Staging at all is settled by measurement rather than by argument, in this
  // workspace: gemma_rms_norm.py records reading the affine row down in the
  // scaling loop as a flat ~2 us penalty (it puts a second *dependent* global
  // round trip on the critical path -- load x, reduce, load affine, store), and
  // holding it in registers as pushing VPT=8 from 48 to 80 registers, giving back
  // most of the win on the largest shape. Shared memory gets the early issue
  // without the register cost, and one copy per CTA means the CTA's warps share
  // the load instead of each repeating it. Under a capped grid it is also
  // amortized over every row group the CTA walks.
  load_group(base);

  extern __shared__ uint4 smem[];
  uint4* __restrict__ smem_gamma = smem;
  uint4* __restrict__ smem_beta = smem + vecs_per_row;
  for (int v = static_cast<int>(threadIdx.x); v < vecs_per_row;
       v += kBlockThreads) {
    smem_gamma[v] = load_affine(gv_of(gamma) + v);
    smem_beta[v] = load_affine(gv_of(beta) + v);
  }
  __syncthreads();

  // `base` is CTA-uniform, so every thread runs the same number of iterations and
  // the shuffles below always see the full lane set. Rows past the end are masked
  // at the store, not returned from.
  while (base < rows) {
    const int64_t row = base + row_in_cta;
    const bool active = row < rows;
    uint4* __restrict__ yv =
        reinterpret_cast<uint4*>(y) +
        (active ? row : 0) * static_cast<int64_t>(vecs_per_row);

    // Pass 0: the row's magnitude, and from it a scale that makes the accumulators
    // safe. bf16 reaches 3.39e38 and fp16 65504; an fp32 sum of 1152 squares
    // overflows well below either, so an unscaled kernel returns NaN on finite
    // input that ATen handles. The scale is a power of two, so multiplying by its
    // reciprocal is *exact* and the ordinary path (scale == 1) is bit-for-bit what
    // it was before this pass existed.
    float amax = 0.0f;
#pragma unroll
    for (int c = 0; c < kChunksPerLane; ++c) {
      const int chunk = lane + c * kLanesPerRow;
#pragma unroll
      for (int k = 0; k < kChunk; ++k) {
        const int idx = chunk * kChunk + k;
        if (idx >= vecs_per_row) {
          continue;
        }
        float e[kElems];
        Packed<T>::to_float(packed[c * kChunk + k], e);
#pragma unroll
        for (int j = 0; j < kElems; ++j) {
          amax = fmaxf(amax, fabsf(e[j]));
        }
      }
    }
    amax = row_reduce_max(amax);

    float inv_s = 1.0f;
    float eps_s = eps;
    bool rescaled = false;
    if (amax > kScaleSafeMax) {
      rescaled = true;
      int exponent = 0;
      frexpf(amax, &exponent);       // amax = m * 2^exponent, m in [0.5, 1)
      inv_s = ldexpf(1.0f, -exponent);
      // eps scales as 1/s^2 because it sits alongside a variance. For the scales
      // this branch sees (>= 1e16) that underflows to zero, which is the right
      // answer: eps is thirty-eight decades below the variance there.
      eps_s = eps * inv_s * inv_s;
    }

    // The reduction passes iterate over the same (chunk, k) indexing and skip the
    // out-of-range slots rather than relying on their zero fill. Zeros are
    // harmless in the mean, but in the variance pass a zeroed slot would
    // contribute (0 - mean)^2 per element -- a ragged mapping (144 vectors over 32
    // lanes) would then compute the wrong variance on every row.
    float sum = 0.0f;
#if FK_VPM_ONEPASS_VAR
    float sumsq = 0.0f;
#endif
#pragma unroll
    for (int c = 0; c < kChunksPerLane; ++c) {
      const int chunk = lane + c * kLanesPerRow;
#pragma unroll
      for (int k = 0; k < kChunk; ++k) {
        const int idx = chunk * kChunk + k;
        if (idx >= vecs_per_row) {
          continue;
        }
        float e[kElems];
        Packed<T>::to_float(packed[c * kChunk + k], e);
#pragma unroll
        for (int j = 0; j < kElems; ++j) {
          const float v = e[j] * inv_s;
#if FK_VPM_BF16_ACC
          sum = Packed<T>::round_storage(sum + v);
#if FK_VPM_ONEPASS_VAR
          sumsq = Packed<T>::round_storage(sumsq + v * v);
#endif
#else
          sum += v;
#if FK_VPM_ONEPASS_VAR
          sumsq += v * v;
#endif
#endif
        }
      }
    }
    // Divide rather than multiply by a precomputed reciprocal. 1/1152 is not
    // exact in fp32, so a reciprocal leaves a constant row with a mean a few
    // ulps off its own value: every deviation is then nonzero, and multiplied by
    // rsqrt(eps) = 1000 it produces a visibly wrong answer where the exact one is
    // zero. Two divisions per row per lane is nothing on a bandwidth-bound
    // kernel, and it is what makes the all-equal row exact.
    const float mean = row_reduce_sum(sum) / n;
#if FK_VPM_ONEPASS_VAR
    const float var = row_reduce_sum(sumsq) / n - mean * mean;
#else
    // A true second pass over the row. It costs no global traffic -- the row is
    // still in registers -- and it is the whole reason this kernel is no worse
    // than ATen on a large-mean, tiny-variance row, where E[x^2]-mu^2 cancels
    // catastrophically.
    float sq = 0.0f;
#pragma unroll
    for (int c = 0; c < kChunksPerLane; ++c) {
      const int chunk = lane + c * kLanesPerRow;
#pragma unroll
      for (int k = 0; k < kChunk; ++k) {
        const int idx = chunk * kChunk + k;
        if (idx >= vecs_per_row) {
          continue;
        }
        float e[kElems];
        Packed<T>::to_float(packed[c * kChunk + k], e);
#pragma unroll
        for (int j = 0; j < kElems; ++j) {
          const float d = e[j] * inv_s - mean;
#if FK_VPM_BF16_ACC
          sq = Packed<T>::round_storage(sq + d * d);
#else
          sq += d * d;
#endif
        }
      }
    }
    const float var = row_reduce_sum(sq) / n;
#endif
    // rsqrtf(var + eps), with one exception that is confined to the case that
    // creates it.
    //
    // After an extreme-value rescale, eps_s = eps / s^2 underflows to zero (s is at
    // least 1e16 there), so a constant row leaves denom == 0 exactly. rsqrtf(0) is
    // +inf and 0 * inf is NaN, where the exact arithmetic gives beta -- so that one
    // case is answered directly. The guard is gated on `rescaled` rather than on
    // `denom > 0` alone, because an *unrescaled* zero denominator is a different
    // situation: it can only arise from an eps that is zero, negative or NaN, and
    // there LayerNorm's answer is the non-finite one that rsqrtf produces. An
    // earlier version tested only `denom > 0` and so returned beta for `eps = 0` on
    // an ordinary constant row, silently changing the operator's contract for a
    // value a caller can pass. The host predicate now rejects such an eps outright,
    // which makes this branch unreachable except after a rescale; the condition is
    // written out anyway so the invariant survives a later edit to the predicate.
    const float denom = var + eps_s;
    const float rstd = (rescaled && !(denom > 0.0f)) ? 0.0f : rsqrtf(denom);

    if (active) {
#pragma unroll
      for (int c = 0; c < kChunksPerLane; ++c) {
        const int chunk = lane + c * kLanesPerRow;
#pragma unroll
        for (int k = 0; k < kChunk; ++k) {
          const int idx = chunk * kChunk + k;
          if (idx >= vecs_per_row) {
            continue;
          }
          float e[kElems];
          float g[kElems];
          float b[kElems];
          Packed<T>::to_float(packed[c * kChunk + k], e);
          Packed<T>::to_float(smem_gamma[idx], g);
          Packed<T>::to_float(smem_beta[idx], b);
#pragma unroll
          for (int j = 0; j < kElems; ++j) {
#if FK_VPM_BF16_AFFINE
            // The control: scale and shift in the storage dtype instead, so every
            // step rounds. gamma and beta are already exact in T.
            e[j] = Packed<T>::round_storage(
                Packed<T>::round_storage(
                    Packed<T>::round_storage((e[j] * inv_s - mean) * rstd) * g[j]) +
                b[j]);
#else
            // Affine parameters upcast in-register and applied in fp32; one
            // rounding, on the store.
            e[j] = (e[j] * inv_s - mean) * rstd * g[j] + b[j];
#endif
          }
          yv[idx] = Packed<T>::from_float(e);
        }
      }
    }

    base += stride;
    if (base >= rows) {
      break;
    }
    load_group(base);
  }
}

// ---------------------------------------------------------------------------
// Host side: the eligibility predicate, the mapping ladder it mirrors, and the
// exact baseline fallback.
// ---------------------------------------------------------------------------

// -1 not dispatched, 0 baseline formula, 1 fused kernel, 2 empty input. Read by
// the test suite so every delegation case can assert *which* path ran: a suite
// that only compared values would pass even if the fused kernel never executed.
int g_last_path = -1;

// Chunks-per-lane values with an instantiated kernel. Bounded on purpose -- each
// entry is two more kernels to compile, and this operator's captured width is a
// single number. 9 covers 144 vectors at 16 lanes (the scored case at a 128-bit
// tile); 5 covers 144 at 32 lanes and 72 at 16; 3 covers 72 at 32.
inline bool has_mapping(int chunks_per_lane) {
  switch (chunks_per_lane) {
    case 1: case 2: case 3: case 4: case 5: case 6: case 8: case 9:
      return true;
    default:
      return false;
  }
}

inline bool is_aligned16(const void* p) {
  return (reinterpret_cast<uintptr_t>(p) & 15u) == 0;
}

inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

enum class Path { kFallback, kEmpty, kFused };

// Every check here precedes any pointer dereference by a kernel. Nothing is
// caught around a launch: a launch that starts has to be a launch that is
// correct.
inline Path choose_path(const at::Tensor& x, const at::Tensor& g,
                        const at::Tensor& b, int64_t n, int64_t hidden, double eps,
                        int* chunks_per_lane_out) {
  // eps is a public constructor parameter, so a caller can pass zero, a negative
  // value, or NaN, and LayerNorm's contract for those is whatever `rsqrt(var + eps)`
  // does -- inf for a zero denominator, NaN for a negative or NaN one. This kernel
  // cannot reproduce that and also keep the rescale's underflow guard below (the two
  // want opposite answers for denom == 0), so anything outside "positive and finite"
  // is handed to ATen, which is the definition of the behaviour rather than an
  // approximation of it. The benchmark's eps is 1e-6, so this costs nothing there.
  if (!(eps > 0.0) || !std::isfinite(eps)) {
    return Path::kFallback;
  }
  if (!x.defined() || !g.defined() || !b.defined() || !x.is_cuda()) {
    return Path::kFallback;
  }
  // Functorch duals, functional/fake tensors and Python subclasses have no
  // ordinary storage to take a pointer to.
  if (at::isTensorSubclassLike(x) || at::isTensorSubclassLike(g) ||
      at::isTensorSubclassLike(b)) {
    return Path::kFallback;
  }
  // Autocast rewrites LayerNorm's output dtype and this has no autocast
  // registration of its own; the baseline formula re-dispatches through
  // at::layer_norm and so picks that policy up exactly.
  if (at::autocast::is_autocast_enabled(x.device().type())) {
    return Path::kFallback;
  }
  const at::ScalarType dtype = x.scalar_type();
  // bf16 and fp16 only. Both hold 8 elements in a 16-byte vector, so C=1152 is
  // 144 vectors either way and it is one extra instantiation. fp32's vector holds
  // 4, which would need a different mapping for no captured shape.
  if (dtype != at::kBFloat16 && dtype != at::kHalf) {
    return Path::kFallback;
  }
  if (n <= 0 || hidden <= 0 || x.dim() < 1 || x.size(-1) != n) {
    return Path::kFallback;
  }
  // Plain is_contiguous(), not "non-overlapping-and-dense with row stride == n":
  // the merged write treats the storage as one flat row-major buffer, and a
  // dense-but-permuted tensor would come out with its rows reordered.
#if FK_VPM_SKIP_CONTIG_CHECK
  // Control: accept any dense tensor. A permuted-but-dense input then comes out
  // with its rows in the wrong order, which is the point.
  if (!x.is_non_overlapping_and_dense()) {
    return Path::kFallback;
  }
#else
  if (!x.is_contiguous()) {
    return Path::kFallback;
  }
#endif
  const int64_t total = x.numel();
  if (total % n != 0 || total % hidden != 0) {
    return Path::kFallback;
  }
  if (g.device() != x.device() || b.device() != x.device() ||
      g.scalar_type() != dtype || b.scalar_type() != dtype || g.dim() != 1 ||
      b.dim() != 1 || g.size(0) != n || b.size(0) != n || !g.is_contiguous() ||
      !b.is_contiguous()) {
    return Path::kFallback;
  }
  // Shape validation first, so a genuinely invalid shape still reaches ATen and
  // raises what the baseline would raise. There is no row to normalise, and
  // reaching the answer must cost no launch.
  if (total == 0) {
    return Path::kEmpty;
  }
  if (total / n > static_cast<int64_t>(INT32_MAX)) {
    return Path::kFallback;
  }
  const int elems = 16 / static_cast<int>(x.element_size());
  if (n % elems != 0) {
    return Path::kFallback;
  }
  // Bound before narrowing: a row of more than INT32_MAX vectors would wrap to a
  // small count the ladder accepts, and the kernel would then normalise a prefix
  // of the row and leave the rest of the output uninitialised.
  const int64_t vecs = n / elems;
  if (vecs > static_cast<int64_t>(INT32_MAX)) {
    return Path::kFallback;
  }
  const int vecs_per_row = static_cast<int>(vecs);
  const int chunks_per_row = ceil_div(vecs_per_row, kChunk);
  const int chunks_per_lane = ceil_div(chunks_per_row, kLanesPerRow);
  if (!has_mapping(chunks_per_lane)) {
    return Path::kFallback;
  }
  // gamma and beta are staged in shared memory, so the row has to fit the
  // default 48 KiB limit.
  if (static_cast<size_t>(2 * vecs_per_row) * sizeof(uint4) > 48u * 1024u) {
    return Path::kFallback;
  }
#if !FK_VPM_SKIP_ALIGN_CHECK
  if (!is_aligned16(x.const_data_ptr()) || !is_aligned16(g.const_data_ptr()) ||
      !is_aligned16(b.const_data_ptr())) {
    return Path::kFallback;
  }
#endif
  *chunks_per_lane_out = chunks_per_lane;
  return Path::kFused;
}

template <typename T>
void launch(const T* x, T* y, const T* g, const T* b, int64_t rows,
            int vecs_per_row, int chunks_per_lane, float n, float eps,
            bool wide_ok, cudaStream_t stream) {
  const int64_t row_groups = (rows + kRowsPerCta - 1) / kRowsPerCta;
  unsigned grid = static_cast<unsigned>(row_groups);
#if FK_VPM_MAX_WAVES > 0
  {
    int device = 0;
    cudaGetDevice(&device);
    int sms = 0;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
    const int64_t cap = static_cast<int64_t>(sms) * FK_VPM_MAX_WAVES;
    if (row_groups > cap) {
      grid = static_cast<unsigned>(cap);
    }
  }
#endif
  const size_t smem = static_cast<size_t>(2 * vecs_per_row) * sizeof(uint4);

#define FK_VPM_LAUNCH(CPL)                                                  \
  case (CPL):                                                               \
    norm_merge_kernel<T, (CPL)><<<grid, kBlockThreads, smem, stream>>>(      \
        x, y, g, b, rows, vecs_per_row, n, eps, wide_ok);                      \
    return;

  switch (chunks_per_lane) {
    FK_VPM_LAUNCH(1)
    FK_VPM_LAUNCH(2)
    FK_VPM_LAUNCH(3)
    FK_VPM_LAUNCH(4)
    FK_VPM_LAUNCH(5)
    FK_VPM_LAUNCH(6)
    FK_VPM_LAUNCH(8)
    FK_VPM_LAUNCH(9)
    default:
      // Unreachable: has_mapping() gates this, and the two live next to each
      // other so a predicate and its dispatch cannot drift apart.
      TORCH_CHECK(false, "no instantiated mapping for chunks_per_lane=",
                  chunks_per_lane);
  }
#undef FK_VPM_LAUNCH
}

at::Tensor run_fused(const at::Tensor& x, const at::Tensor& g,
                     const at::Tensor& b, int64_t n, int64_t hidden, double eps,
                     int chunks_per_lane) {
  const c10::cuda::CUDAGuard device_guard(x.device());
  const int64_t total = x.numel();
  at::Tensor y = at::empty({total / hidden, hidden}, x.options());

  const int elems = 16 / static_cast<int>(x.element_size());
  const int vecs_per_row = static_cast<int>(n / elems);
  const int64_t rows = total / n;
  const float nf = static_cast<float>(n);
  // Every row starts at a multiple of the row width, so one check on the base
  // pointer and the row stride settles it for the whole tensor.
  const bool wide_ok =
      ((reinterpret_cast<uintptr_t>(x.const_data_ptr()) & 31u) == 0) &&
      ((static_cast<int64_t>(n) * x.element_size()) % 32 == 0);
  const float epsf = static_cast<float>(eps);
  const auto stream = c10::cuda::getCurrentCUDAStream();

  if (x.scalar_type() == at::kBFloat16) {
    launch<__nv_bfloat16>(
        reinterpret_cast<const __nv_bfloat16*>(x.const_data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(y.mutable_data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(g.const_data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(b.const_data_ptr()), rows,
        vecs_per_row, chunks_per_lane, nf, epsf, wide_ok, stream);
  } else {
    launch<__half>(reinterpret_cast<const __half*>(x.const_data_ptr()),
                   reinterpret_cast<__half*>(y.mutable_data_ptr()),
                   reinterpret_cast<const __half*>(g.const_data_ptr()),
                   reinterpret_cast<const __half*>(b.const_data_ptr()), rows,
                   vecs_per_row, chunks_per_lane, nf, epsf, wide_ok, stream);
  }
  return y;
}

// What the baseline computes, for the merged shape: LayerNorm(x) then the view.
// promote_fp32 is False in this operator's construction, so the baseline norm is
// native-dtype at::layer_norm and there is no fp32 round trip to reproduce.
at::Tensor baseline_formula(const at::Tensor& x, const at::Tensor& g,
                           const at::Tensor& b, int64_t n, int64_t hidden,
                           double eps) {
  const std::optional<at::Tensor> go =
      g.defined() ? std::optional<at::Tensor>(g) : std::nullopt;
  const std::optional<at::Tensor> bo =
      b.defined() ? std::optional<at::Tensor>(b) : std::nullopt;
  return at::layer_norm(x, {n}, go, bo, eps).reshape({-1, hidden});
}

}  // namespace

at::Tensor norm_merge(const at::Tensor& x, const at::Tensor& weight,
                      const at::Tensor& bias, int64_t n, int64_t hidden,
                      double eps) {
  // The fused path allocates with at::empty and launches a raw kernel, so it
  // records nothing for autograd. Grad mode being *enabled* is the whole test,
  // not whether some tensor currently requires grad: a caller who has not
  // entered no_grad may attach requires_grad later in the same graph, or wrap
  // this in checkpointing that replays the forward under different requires_grad
  // state. Gradient-enabled calls take the baseline formula, whose ATen ops build
  // an ordinary graph. The cost is that the kernel runs only under no_grad, which
  // is where every inference path -- including the benchmark harness -- puts it.
#if !FK_VPM_IGNORE_GRAD
  if (at::GradMode::is_enabled()) {
    g_last_path = 0;
    return baseline_formula(x, weight, bias, n, hidden, eps);
  }
#endif
  int chunks_per_lane = 0;
  const Path path = choose_path(x, weight, bias, n, hidden, eps, &chunks_per_lane);
  if (path == Path::kEmpty) {
    g_last_path = 2;
    return at::empty({0, hidden}, x.options());
  }
  if (path == Path::kFused) {
    g_last_path = 1;
    return run_fused(x, weight, bias, n, hidden, eps, chunks_per_lane);
  }
  g_last_path = 0;
  return baseline_formula(x, weight, bias, n, hidden, eps);
}

int64_t last_path() { return g_last_path; }

void reset_last_path() { g_last_path = -1; }

// The compile-time configuration this library was actually built with.
//
// Not decoration: the A/B measures several builds in one process, and the only
// thing that distinguishes them is a set of macros that leaves no trace in the
// output values -- a mapping change alters the reduction order, not the answer. A
// variant that silently ran a different library would look like a correct
// measurement. So every timing run asserts this string first.
std::string config() {
  std::ostringstream os;
  os << "lanes=" << FK_VPM_LANES << ",rows=" << FK_VPM_ROWS
     << ",chunk=" << FK_VPM_CHUNK << ",waves=" << FK_VPM_MAX_WAVES
     << ",hints=" << FK_VPM_CACHE_HINTS << ",ld256=" << FK_VPM_LD256
     << ",onepass=" << FK_VPM_ONEPASS_VAR << ",acc=" << FK_VPM_BF16_ACC
     << ",affine=" << FK_VPM_BF16_AFFINE
     << ",noalign=" << FK_VPM_SKIP_ALIGN_CHECK
     << ",nocontig=" << FK_VPM_SKIP_CONTIG_CHECK
     << ",nograd=" << FK_VPM_IGNORE_GRAD;
  return os.str();
}
"""

_ENTRY_POINTS = ("norm_merge", "last_path", "reset_last_path", "config")


def _local_arch() -> str | None:
    """The single compute capability to build for, without initializing CUDA.

    The ambient ``TORCH_CUDA_ARCH_LIST`` in this environment names six
    architectures, which multiplies a cold compile by six for a kernel that only
    ever runs on one device -- and this operator already pays for three extension
    builds at import.

    ``nvidia-smi`` is asked *first*, deliberately. This runs at import, before the
    module has any reason to touch the GPU, and ``torch.cuda.is_available()``
    initializes a CUDA context as a side effect; doing that from an import means a
    process that only wanted to read the module's source pays for a context.
    ``candidate/L1/l2_norm.py`` uses the same probe for the same reason.

    The tradeoff is that ``nvidia-smi`` reports every GPU on the host regardless of
    the visible-device set, so on a mixed-architecture machine it could answer for
    a device this process cannot see. That is handled by only trusting it when every
    GPU on the host reports the *same* capability, and falling back to torch (and
    accepting the context) when they differ. Returning ``None`` means "leave the
    ambient value alone", which is right only when neither source can answer.
    """
    cap = None
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL, timeout=20)
        caps = sorted({line.strip() for line in out.splitlines() if line.strip()})
        cap = caps[0] if len(caps) == 1 else None
    except Exception:  # noqa: BLE001 - no nvidia-smi is a valid answer, not an error
        cap = None
    if cap is None:
        # Either nvidia-smi is absent or the host is mixed-architecture. Asking
        # torch costs a CUDA context, which is why it is the fallback.
        try:
            if torch.cuda.is_available():
                major, minor = torch.cuda.get_device_capability()
                cap = f"{major}.{minor}"
        except Exception:  # noqa: BLE001
            cap = None
    if cap is None:
        return None
    major = cap.split(".")[0]
    # Blackwell and Hopper want the architecture-specific variant.
    return f"{cap}a" if major in ("9", "10", "12") and not cap.endswith("a") else cap


def _build_root() -> Path:
    """A build directory this workspace owns.

    The default extension cache lives under ``$HOME``, which sibling agent
    workspaces share, and it is keyed only by extension name with no stale-lock
    recovery -- so a concurrent build elsewhere could be imported in place of this
    one, or its abandoned lock could block this import forever.
    """
    options = []
    override = os.environ.get("FK_VPM_BUILD_DIR")
    if override:
        options.append(Path(override))
    try:
        options.append(Path(__file__).resolve().parents[2] / ".torch_extensions")
    except (IndexError, OSError):
        pass
    options.append(Path(tempfile.gettempdir()) / f"fk_vpm_build_{os.getuid()}")
    for root in options:
        try:
            root.mkdir(parents=True, exist_ok=True)
            probe = root / ".writable"
            probe.touch()
            probe.unlink()
            return root
        except OSError:
            continue
    raise RuntimeError("no writable build directory for the norm-merge extension")


# A build killed between creating the loader's lock and releasing it leaves the
# file behind, and the loader has no timeout and no stale-lock recovery: the next
# import would wait on it forever rather than degrade. A cold build here is well
# under a minute, so anything this old is abandoned.
_STALE_LOCK_SECONDS = 600


def _clear_stale_lock(build_dir: Path) -> None:
    lock = build_dir / "lock"
    try:
        age = time.time() - lock.stat().st_mtime
    except OSError:
        return
    if age > _STALE_LOCK_SECONDS:
        print(f"[vision_patch_merger] removing an abandoned build lock "
              f"({age:.0f}s old): {lock}", flush=True)
        try:
            lock.unlink()
        except OSError:
            pass


def _defines_from_env() -> tuple[str, ...]:
    """The A/B axes, read from the environment into ``-D`` flags.

    Every axis is compile-time so a losing variant stays reachable for
    re-measurement instead of being deleted and trusted. Each value is validated
    here rather than passed through, so a typo cannot silently select the default
    and be recorded as a measurement of something else.
    """
    defines = []

    def pick(env: str, macro: str, allowed: tuple[int, ...]) -> None:
        raw = os.environ.get(env, "")
        if raw == "":
            return
        value = int(raw)
        if value not in allowed:
            raise ValueError(f"{env}={raw} is not one of {allowed}")
        defines.append(f"-D{macro}={value}")

    pick("FK_VPM_LANES", "FK_VPM_LANES", (16, 32))
    pick("FK_VPM_ROWS", "FK_VPM_ROWS", (1, 2, 4, 8))
    pick("FK_VPM_CHUNK", "FK_VPM_CHUNK", (1, 2))
    pick("FK_VPM_MAX_WAVES", "FK_VPM_MAX_WAVES", (0, 1, 2, 4, 8, 16, 32, 64, 96, 128, 256))
    pick("FK_VPM_CACHE_HINTS", "FK_VPM_CACHE_HINTS", (0, 1))
    pick("FK_VPM_LD256", "FK_VPM_LD256", (0, 1))
    pick("FK_VPM_ONEPASS_VAR", "FK_VPM_ONEPASS_VAR", (0, 1))
    pick("FK_VPM_BF16_ACC", "FK_VPM_BF16_ACC", (0, 1))
    pick("FK_VPM_BF16_AFFINE", "FK_VPM_BF16_AFFINE", (0, 1))
    pick("FK_VPM_SKIP_ALIGN_CHECK", "FK_VPM_SKIP_ALIGN_CHECK", (0, 1))
    pick("FK_VPM_SKIP_CONTIG_CHECK", "FK_VPM_SKIP_CONTIG_CHECK", (0, 1))
    pick("FK_VPM_IGNORE_GRAD", "FK_VPM_IGNORE_GRAD", (0, 1))
    return tuple(defines)


def build_extension(defines: tuple[str, ...] = (), *, verbose: bool | None = None):
    """Compile and import the embedded extension.

    The extension name carries a hash of the source, the flags and the
    architecture, so a variant build can never be confused with the shipping one
    -- in this process or a later one -- and two variants can be alive at once,
    which the mapping A/B and the numerics negative control both need. Keying on
    the name alone would be worse than useless: torch rebuilds whenever a source
    is newer than the ``.so``, so two sources under one name invalidate each
    other's build on every import.
    """
    from torch.utils.cpp_extension import load_inline

    # -fvisibility=hidden is load-bearing, not hygiene: see the note above the
    # source. Without it, a second build of this extension in the same process
    # silently runs the first one's kernel.
    host_flags = ["-O3", "-fvisibility=hidden"]
    cuda_flags = ["-O3", "-lineinfo", "-Xcompiler", "-fvisibility=hidden", *defines]
    if os.environ.get("FK_VPM_PTXAS_V") == "1":
        cuda_flags += ["-Xptxas", "-v"]
    broken = os.environ.get("FK_VPM_BREAK_BUILD") == "1"
    if broken:
        # Test hook: make nvcc reject the translation unit, so the failed-build
        # path is exercised for real rather than mocked.
        cuda_flags.append("--this-flag-does-not-exist")
    arch = _local_arch()
    # The architecture is part of the key, not just of the flags, so a binary
    # built for another GPU can never be picked up under this name.
    key = hashlib.sha256(
        "\x00".join([_CPP_SOURCE, _CUDA_SOURCE, *host_flags, *cuda_flags,
                      arch or "ambient"]).encode()
    ).hexdigest()[:16]
    name = f"fk_vpm_{key}"

    build_dir = _build_root() / name
    build_dir.mkdir(parents=True, exist_ok=True)
    _clear_stale_lock(build_dir)
    cold = not (build_dir / f"{name}.so").exists()
    if cold:
        # Keep the log moving: this operator compiles three extensions at import
        # (the frozen L1 LayerNorm, the frozen L1 GELU and this one), and a silent
        # compile can trip a no-output watchdog.
        print(f"[vision_patch_merger] compiling {name} for arch {arch} "
              f"(cold cache){' with a deliberately broken flag' if broken else ''}",
              flush=True)

    previous = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if arch:
        os.environ["TORCH_CUDA_ARCH_LIST"] = arch
    try:
        return load_inline(
            name=name,
            cpp_sources=[_CPP_SOURCE],
            cuda_sources=[_CUDA_SOURCE],
            functions=list(_ENTRY_POINTS),
            extra_cflags=host_flags,
            extra_cuda_cflags=cuda_flags,
            build_directory=str(build_dir),
            verbose=cold if verbose is None else verbose,
        )
    finally:
        if arch:
            if previous is None:
                os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
            else:
                os.environ["TORCH_CUDA_ARCH_LIST"] = previous


# Built at import, never inside a timed forward.
_EXT = None
_EXT_ERROR: str | None = None
if os.environ.get("FK_VPM_EXT") == "0":
    _EXT_ERROR = "disabled by FK_VPM_EXT=0"
else:
    try:
        _EXT = build_extension(_defines_from_env())
    except Exception as exc:  # noqa: BLE001 - a build that cannot happen must
        # degrade, not take the module down: an import failure costs every case at
        # once, and a silent one would later report a meaningless ~1.0x.
        _EXT_ERROR = f"{type(exc).__name__}: {exc}"
        print(f"[vision_patch_merger] extension unavailable, every case takes the "
              f"baseline formula ({_EXT_ERROR})", file=sys.stderr, flush=True)

_NORM_MERGE = _EXT.norm_merge if _EXT is not None else None

_PATH_NAMES = {-1: "not-dispatched", 0: "baseline", 1: "fused", 2: "empty"}


def config() -> str | None:
    """The macro values the loaded library was compiled with, or None."""
    return _EXT.config() if _EXT is not None else None


def extension_loaded() -> bool:
    """Whether the fused kernel is available in this process."""
    return _EXT is not None


def extension_error() -> str | None:
    """Why the extension is unavailable, or None if it loaded."""
    return _EXT_ERROR


def last_path() -> str:
    """Which path the most recent fused dispatch took.

    ``not-dispatched`` means the operator was never called -- the postshuffle
    branch delegates to ``self.norm`` in Python and never reaches C++, which is a
    stronger statement than "it returned the right answer".
    """
    if _EXT is None:
        return "no-extension"
    return _PATH_NAMES.get(_EXT.last_path(), "unknown")


def reset_last_path() -> None:
    if _EXT is not None:
        _EXT.reset_last_path()


class VisionPatchMerger(nn.Module):
    """Patch merger: norm -> flatten -> MLP(fc1, GELU, fc2).

    Qwen3 DeepStack mergers set use_postshuffle_norm=True to norm after reshape.
    """

    def __init__(self, d_model: int, context_dim: int,
                 spatial_merge_size: int = 2,
                 use_postshuffle_norm: bool = False,
                 eps: float = 1e-6):
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size ** 2)
        self.use_postshuffle_norm = use_postshuffle_norm
        norm_dim = self.hidden_size if use_postshuffle_norm else context_dim
        # See VisionBlock: vLLM's vision path uses plain nn.LayerNorm on
        # bf16, and our fp32 promotion costs two full-tensor copies here.
        self.norm = LayerNorm(norm_dim, eps=eps, promote_fp32=False)
        self.fc1 = ColumnParallelLinear(self.hidden_size, self.hidden_size, bias=True)
        self.act = GELU()
        self.fc2 = RowParallelLinear(self.hidden_size, d_model, bias=True)
        # The kernel wants plain ints; norm.normalized_shape stays the tuple the
        # baseline exposes.
        self._norm_dim = int(norm_dim)
        # Read once, here, rather than per forward: the postshuffle rows are 4608
        # wide, which is 576 vectors -- precisely the width the frozen L1 kernel is
        # tuned for. There is nothing to win there, and it is not a scored case.
        # The postshuffle path delegates, and so does a norm without affine
        # parameters: the kernel's predicate requires both, and the baseline
        # always creates them, so this is a guard rather than a live case.
        affine = self.norm.weight is not None and self.norm.bias is not None
        self._fused = _NORM_MERGE if (affine and not use_postshuffle_norm) else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._fused is None:
            if self.use_postshuffle_norm:
                x = self.norm(x.view(-1, self.hidden_size))
            else:
                x = self.norm(x).view(-1, self.hidden_size)
        else:
            # One dispatch: eligibility and the baseline fallback both live inside
            # the operator, so nothing here inspects the tensor's metadata.
            x = self._fused(x, self.norm.weight, self.norm.bias, self._norm_dim,
                            self.hidden_size, self.norm.eps)
        x = self.fc2(self.act(self.fc1(x)))
        return x
