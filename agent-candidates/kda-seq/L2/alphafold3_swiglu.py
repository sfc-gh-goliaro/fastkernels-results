"""Fused SwiGLU and AdaLN for B200 (sm_100), same contract as ``baseline.py``.

The baseline composes frozen L1 primitives, and on the captured shapes the cost
is the *number of kernels*, not the arithmetic. Written out:

    SwiGLU: F.linear x2, F.silu, mul                                    4 kernels
    AdaLN:  2 x (to(fp32), layer_norm, to(bf16)) + F.linear x2
            + sigmoid + add + mul                                      11 kernels

The heaviest scored case moves 4.7 MB and does 50.3 M MACs -- under ~1.5 us of
real GPU work against measured windows of 19.5-113.8 us. A calibration ladder
replaying the harness's own timing loop over modules issuing N trivial kernels
(``profile/p1-baseline-survey/survey.json``) reads a ~6.08 us window floor at
N = 0 and 9.20 us at N = 1. So one kernel per forward is both the whole prize and
the floor worth aiming at, and that is what these two kernels do.

Both fusions are natural rather than forced. Each class multiplies the *same*
left operand by two weights of the same shape, so one pass over that operand
produces both accumulators; every activation (LayerNorm, sigmoid, SiLU, add,
multiply) is then an epilogue on data already in registers, and no intermediate
reaches global memory. `wmma` at 16x16x16 is the arithmetic: every captured M,
K and N is a multiple of 16, so the fragment geometry lands exactly on the
problem. tcgen05/TMEM accumulators, CLC and the 2-SM cooperative path are the
wrong tools at ~0.05-0.8 us of arithmetic per case -- they buy peak on
compute-bound shapes three orders larger and would only add launch-side setup.

Numerics mirror the baseline's *rounding* ladder point for point rather than
aiming at the tolerance, which is nearly free here (one `__float2bfloat16_rn` on
a value already in a register) and removes tolerance risk as a category. Two fp32
reassociations remain and are deliberate, both far below bf16's 4e-3 quantum: the
row reduction is a two-pass sum of squared deviations rather than ATen's Welford
order, and the tensor-core accumulation (with the k-split, a shared-memory sum of
fp32 partials) is not cuBLAS's order. Every *rounding* is in the same place:

    F.layer_norm(x.float(), ...)  -> fp32 two-pass mean / sum-of-squared-
                                     deviations over the bf16 row (bf16 -> fp32
                                     is exact), biased variance, rsqrtf(var+eps)
    .to(bf16) on the norm result  -> round s_norm / a_norm to bf16 before use
    F.linear (bf16, fp32 epilogue)-> fp32 accumulate, bias added in fp32,
                                     rounded once to bf16
    sigmoid / silu on bf16        -> computed in fp32 from the bf16-rounded
                                     input, result rounded to bf16
    add, mul on bf16              -> fp32 arithmetic on bf16 operands, one
                                     rounding per op

That the roundings really are in those places is checked three ways, all against the
*baseline composition* -- the thing this module has to agree with, rather than against
a mathematical ideal that neither implementation computes. Over the ten scored cases
under three weight/input seeds plus 22 randomized shapes -- 52 case-runs -- every case has
`matched_ratio == 1.0` at the bench tolerance, and max-abs against the composition is
at most 7.81e-3 against the bench's own 1e-2 + 1e-2|y| gate. Most cases are
bit-exact.

That 7.81e-3 is not the kernel's rounding: a per-stage probe
(`profile/p4-stage-attribution/`) feeds the reference its own intermediates and finds
its LayerNorm bit-identical to the two-pass fp32 formula used here (0.000e+00 on
every scored case) and its epilogue bit-identical, with the entire difference being
cuBLAS's GEMM accumulation order -- 1.953e-3 on the AdaLN GEMMs and 7.812e-3 on
SwiGLU's, against an exact fp32 matmul of identical operands. Two later bf16
roundings carry that to the output, which is why one output quantum is not reachable
against this reference by any implementation that does not reproduce cuBLAS's own
kernel choice.

Rounding-point placement itself is checked against an fp32 mirror of the same ladder
(`tools/exact_ladder.py`): removing a single rounding leaves max-abs indistinguishable
(7.812e-3 against 7.810e-3) but moves the RMS deviation from the mirror by 20x to
1900x, so that is the statistic the negative control asserts on.

Anything the fast path does not admit -- a build failure, an unexpected dtype,
rank, layout, alignment, or a shape outside the kernels' resource envelope --
returns the baseline composition evaluated through the frozen L1 winners. The
operator signals that by returning ``None``, which costs one ``is None`` test on
the fast path and keeps every heavy predicate in C++ behind the single dispatch.
Rejection is always a fallback and never a throw, so a surprise input costs
speed, never correctness. That generality is not polish: this module is imported
by ``alphafold3_swiglu_transition`` and ``alphafold3_attention_pair_bias``, so it
will be called with shapes and layouts well outside the ten scored cases.
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear
from ..L1.sigmoid import Sigmoid
from ..L1.silu import SiLU

# Unique to this file so importing it a second time under a different module name
# cannot double-register the operators; registration happens once at ``.so`` load
# and Python's module cache makes any later import a no-op.
_LIBRARY_NAME = "fk_af3_swiglu_l2"

_CUDA_SOURCE = r"""
#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>
#include <torch/library.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <mma.h>

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <optional>

namespace {

using nvcuda::wmma::col_major;
using nvcuda::wmma::row_major;
namespace wmma = nvcuda::wmma;

constexpr int kWarpSize = 32;
constexpr unsigned kFullMask = 0xffffffffu;

// The one tile shape every scored extent is a multiple of.
constexpr int kTile = 16;

// `wmma`'s matrix-pointer contract is 256-bit, so every base handed to
// load_matrix_sync / store_matrix_sync must be 32-byte aligned -- not merely
// 16. Shared-memory rows are padded by a whole multiple of this in elements so
// that every *row* start inherits the alignment, and the pad doubles as bank-
// conflict relief (a row stride of exactly c_in bf16 puts every row on the same
// bank for the widths captured here).
constexpr int kAlign = 32;
constexpr int kSmemPadElems = 16;   // 32 bytes of bf16

// Upper bound on any channel extent the fast path will look at.
//
// The staging arithmetic is deliberately `int`: a padded leading dimension is
// `c + kSmemPadElems` and the staging loop bound is `16 * (c / 8)`, so an extent
// anywhere near INT32_MAX wraps *before* the shared-memory budget is checked --
// a negative byte count then passes a `> ceiling` test and the kernel stages
// nothing while the mma loop reads uninitialised shared memory. Bounding the
// extent up front kills the whole class instead of hardening each expression.
//
// The bound is far above anything admissible anyway: the shared-memory ceiling
// already caps a staged row at roughly 7 thousand channels, so nothing real is
// rejected by this and everything unreal is rejected before it can wrap.
constexpr int64_t kMaxChannels = 1 << 24;

// Leading dimension, in floats, of the 16x16 fp32 staging tiles the epilogues
// read accumulators back through. store_matrix_sync wants a multiple of 4
// floats; 20 rather than 16 shifts consecutive rows by 16 bytes so the indexed
// walk in AdaLN's epilogue does not serialise on one bank.
// How many k-steps of B fragments to keep in flight per warp.
//
// The two M = 16 cases are latency-bound with an almost-idle machine: NCU
// (`profile/p1-mtile16-ncu/`) reads long_scoreboard at 14.6 warps stalled per
// issue-active against 0.020 waves per SM, 1.56% of peak DRAM, and 0.89% of peak
// SM throughput. The total warp count is fixed at one per 16x16 output tile
// (96 for c_out = 1536), so no block or grid arrangement changes it -- which is
// why the 1/2/4/8 warp sweep barely moved those two cases. The only per-warp
// lever is how many independent loads are outstanding, and by Little's law that
// is the whole story: 96 warps x one k-step of ~1 KB is ~98 KB outstanding, which
// at ~600 ns of latency is the ~120 GB/s actually measured.
//
// `#pragma unroll` alone bought ~10%: the mma consumes the fragment the load just
// produced, so the compiler has little freedom to hoist. This is instead an
// explicit prefetch pipeline -- the loads for step i + kStages are issued before
// step i is consumed -- which puts kStages k-steps in flight by construction
// rather than by hoping. The slot index has to be a compile-time constant or the
// fragment array lands in local memory and spills, hence the unrolled inner loop
// over the stage rather than a `% kStages` subscript.
// Exposed as a -D macro with a fixed default so `tools/ab_stages.py` can A/B it
// by recompiling, rather than by adding a runtime switch to the shipped path.
#ifndef FK_AF3_STAGES
#define FK_AF3_STAGES 6
#endif
// Compute capability this translation unit was built for, injected by the loader
// from the live device. Zero would make `device_arch_matches` reject everything,
// which is the safe direction if the define ever goes missing.
#ifndef FK_AF3_BUILT_SM
#define FK_AF3_BUILT_SM 0
#endif
constexpr int kStages = FK_AF3_STAGES;
static_assert(kStages >= 1, "the pipeline needs at least one stage");

// An explicitly 16-byte-aligned eight-wide bf16 bundle, so the one coalesced
// store per lane is a single aligned vector store by construction. Punning a
// plain `__nv_bfloat16[8]` through `uint4*` happens to work with this compiler
// but the source guarantees neither the alignment nor the aliasing.
struct __align__(16) Bf16x8 {
  __nv_bfloat16 v[8];
};

constexpr int kAccLd = 20;
constexpr int kAccFloats = kTile * kAccLd;

// Round a shared-memory cursor up to the wmma pointer requirement.
__device__ __forceinline__ char* align_up(char* p) {
  return reinterpret_cast<char*>(
      (reinterpret_cast<uintptr_t>(p) + (kAlign - 1)) & ~static_cast<uintptr_t>(kAlign - 1));
}

// ---------------------------------------------------------------------------
// The activation spellings the frozen L1 winners already use and that already
// passed this bench at this tolerance. Both are approximate: ex2.approx.f32
// plus a rounded reciprocal / approximate divide. Their ~2 ulp fp32 error is
// three orders below bf16's 4e-3 quantum, so it is invisible in the output, and
// both saturate cleanly (a large negative argument gives +inf in the
// denominator, hence 0, never NaN).
// ---------------------------------------------------------------------------
__device__ __forceinline__ float sigmoid_f(float x) {
  return __frcp_rn(1.0f + __expf(-x));
}

__device__ __forceinline__ float silu_f(float x) {
  return __fdividef(x, 1.0f + __expf(-x));
}

__device__ __forceinline__ float round_bf16(float x) {
  return __bfloat162float(__float2bfloat16_rn(x));
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    v += __shfl_xor_sync(kFullMask, v, offset);
  }
  return v;
}

// One warp reduces one row of `n` bf16 values held in shared memory, twice: the
// mean, then the sum of squared deviations from it. Two passes rather than
// E[x^2] - E[x]^2 because that is the shape of the reduction ATen performs, and
// re-reading shared memory costs nothing next to getting the variance's
// cancellation behaviour right.
__device__ __forceinline__ void row_stats(const __nv_bfloat16* row, int n, int lane,
                                          float inv_n, float eps,
                                          float* mean_out, float* rstd_out) {
  float sum = 0.0f;
  for (int j = lane; j < n; j += kWarpSize) {
    sum += __bfloat162float(row[j]);
  }
  const float mean = warp_reduce_sum(sum) * inv_n;
  float sq = 0.0f;
  for (int j = lane; j < n; j += kWarpSize) {
    const float d = __bfloat162float(row[j]) - mean;
    sq += d * d;
  }
  const float var = warp_reduce_sum(sq) * inv_n;
  *mean_out = mean;
  *rstd_out = rsqrtf(var + eps);
}

// Copy `rows` x `n` bf16 from a row-major global block into a padded shared tile,
// eight elements (one 16-byte access) at a time. Rows past `valid` are zeroed
// rather than read, which is what makes a partial last m-tile legal instead of
// an out-of-bounds read -- so any M is admissible, not only multiples of 16.
__device__ __forceinline__ void stage_tile(const __nv_bfloat16* __restrict__ src,
                                           int64_t src_row0, int n, int64_t ld_src,
                                           __nv_bfloat16* dst, int ld_dst,
                                           int64_t valid, int tid, int nthreads) {
  const int vecs = n >> 3;
  const int total = kTile * vecs;
  const uint4 zero = make_uint4(0u, 0u, 0u, 0u);
  for (int i = tid; i < total; i += nthreads) {
    const int r = i / vecs;
    const int v = i - r * vecs;
    uint4 val = zero;
    const int64_t row = src_row0 + r;
    if (row < valid) {
      val = *reinterpret_cast<const uint4*>(src + row * ld_src + (v << 3));
    }
    *reinterpret_cast<uint4*>(dst + static_cast<int64_t>(r) * ld_dst + (v << 3)) = val;
  }
}

// ---------------------------------------------------------------------------
// SwiGLU:  out[m, n] = silu(dot(x[m, :], Wa[n, :])) * dot(x[m, :], Wb[n, :])
//
// Both weights are [c_out, c_in] row-major, hence k-contiguous, which *is* the
// col_major B-fragment layout with ldm = c_in: load_matrix_sync(b, Wa + n0*c_in
// + k, c_in) addresses Wa[(n0+n)*c_in + k+k'] with no relayout and coalesced
// reads along k. x is k-contiguous too, so the A fragment is row_major.
//
// grid = (m_tiles, n_tiles). m on x so a large M cannot exceed the 65535 limit
// the y and z extents carry, and so consecutive blocks share an n-slice -- i.e.
// the same weight columns, which is the traffic that dominates.
// ---------------------------------------------------------------------------
template <int NCOL, int NSPLIT>
__global__ void __launch_bounds__(kWarpSize* NCOL* NSPLIT)
swiglu_kernel(const __nv_bfloat16* __restrict__ x,
              const __nv_bfloat16* __restrict__ wa,
              const __nv_bfloat16* __restrict__ wb,
              __nv_bfloat16* __restrict__ out,
              int64_t rows, int c_in, int c_out, int ld_x) {
  extern __shared__ char smem_raw[];
  constexpr int kWarps = NCOL * NSPLIT;
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid >> 5;
  const int lane = tid & (kWarpSize - 1);
  // Warp w owns column group w / NSPLIT and k-slice w % NSPLIT. At NSPLIT == 1
  // this is one warp per 16x16 output tile, as before.
  const int col_group = warp / NSPLIT;
  const int split = warp % NSPLIT;
  const int64_t m0 = static_cast<int64_t>(blockIdx.x) * kTile;
  const int n0 = (static_cast<int>(blockIdx.y) * NCOL + col_group) * kTile;

  // One region, used twice: the x tile while the mma loop runs, then the fp32
  // accumulator staging tiles once x is dead. Sized as the max of the two by the
  // launcher, so the widest c_in does not also pay for the epilogue.
  char* base = align_up(smem_raw);
  __nv_bfloat16* xs = reinterpret_cast<__nv_bfloat16*>(base);

  stage_tile(x, m0, c_in, c_in, xs, ld_x, rows, tid, kWarpSize * kWarps);
  __syncthreads();

  // Warp-uniform: the trailing n-tile of a grid whose c_out is not a multiple of
  // 16*NCOL leaves some warps with no columns. They still have to reach every
  // __syncthreads below.
  const bool active = n0 < c_out;

  const int steps = c_in / kTile;
  const int per_split = (steps + NSPLIT - 1) / NSPLIT;
  const int begin = split * per_split;
  const int end = begin + per_split < steps ? begin + per_split : steps;

  wmma::fragment<wmma::accumulator, kTile, kTile, kTile, float> acc_a, acc_b;
  wmma::fill_fragment(acc_a, 0.0f);
  wmma::fill_fragment(acc_b, 0.0f);
  if (active) {
    wmma::fragment<wmma::matrix_a, kTile, kTile, kTile, __nv_bfloat16, row_major> af;
    wmma::fragment<wmma::matrix_b, kTile, kTile, kTile, __nv_bfloat16, col_major>
        bf_a[kStages], bf_b[kStages];
    const __nv_bfloat16* pa = wa + static_cast<int64_t>(n0) * c_in;
    const __nv_bfloat16* pb = wb + static_cast<int64_t>(n0) * c_in;
#pragma unroll
    for (int s = 0; s < kStages; ++s) {
      if (begin + s < end) {
        wmma::load_matrix_sync(bf_a[s], pa + (begin + s) * kTile, c_in);
        wmma::load_matrix_sync(bf_b[s], pb + (begin + s) * kTile, c_in);
      }
    }
    for (int step0 = begin; step0 < end; step0 += kStages) {
#pragma unroll
      for (int s = 0; s < kStages; ++s) {
        const int step = step0 + s;
        if (step < end) {
          wmma::load_matrix_sync(af, xs + step * kTile, ld_x);
          wmma::mma_sync(acc_a, af, bf_a[s], acc_a);
          wmma::mma_sync(acc_b, af, bf_b[s], acc_b);
          const int ahead = step + kStages;
          if (ahead < end) {
            wmma::load_matrix_sync(bf_a[s], pa + ahead * kTile, c_in);
            wmma::load_matrix_sync(bf_b[s], pb + ahead * kTile, c_in);
          }
        }
      }
    }
  }

  __syncthreads();  // x tile is dead; the staging tiles overlay it
  float* acc_smem = reinterpret_cast<float*>(base) + warp * 2 * kAccFloats;
  if (active) {
    // The partials go to shared memory in fp32 and are summed there. Rounding a
    // partial to bf16 would introduce a second rounding inside what the baseline
    // computes as one fp32 accumulation over the whole k range.
    wmma::store_matrix_sync(acc_smem, acc_a, kAccLd, wmma::mem_row_major);
    wmma::store_matrix_sync(acc_smem + kAccFloats, acc_b, kAccLd, wmma::mem_row_major);
  }
  __syncthreads();

  // One 8-wide chunk per lane, so each lane makes exactly one 16-byte store.
  // c_out and n0 are multiples of 16 and the half offset is 8, so the address is
  // 16-byte aligned. Chunks are handed out across the whole block because at
  // NSPLIT > 1 most warps have no output tile of their own to finish.
  const int chunks = NCOL * (kTile * 2);
  for (int chunk = tid; chunk < chunks; chunk += kWarpSize * kWarps) {
    const int group = chunk / (kTile * 2);
    const int within = chunk - group * (kTile * 2);
    const int r = within >> 1;
    const int c = (within & 1) << 3;
    const int gn0 = (static_cast<int>(blockIdx.y) * NCOL + group) * kTile;
    const int64_t row = m0 + r;
    if (gn0 >= c_out || row >= rows) {
      continue;
    }
    const float* part = reinterpret_cast<float*>(base) +
                        group * NSPLIT * 2 * kAccFloats + r * kAccLd + c;
    Bf16x8 packed;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      // Pairwise tree rather than a running sum: ((p0+p1)+(p2+p3)) at four ways,
      // (p0+p1) at two. The association is then fixed by the shape of the
      // reduction instead of by the order a loop happens to visit the partials,
      // which is one fewer thing that can differ between builds.
      float partial[NSPLIT];
#pragma unroll
      for (int p = 0; p < NSPLIT; ++p) {
        partial[p] = part[p * 2 * kAccFloats + j];
      }
#pragma unroll
      for (int width = NSPLIT / 2; width >= 1; width >>= 1) {
#pragma unroll
        for (int p = 0; p < width; ++p) {
          partial[p] += partial[p + width];
        }
      }
      const float sum_a = partial[0];
      float partial_b[NSPLIT];
#pragma unroll
      for (int p = 0; p < NSPLIT; ++p) {
        partial_b[p] = part[p * 2 * kAccFloats + kAccFloats + j];
      }
#pragma unroll
      for (int width = NSPLIT / 2; width >= 1; width >>= 1) {
#pragma unroll
        for (int p = 0; p < width; ++p) {
          partial_b[p] += partial_b[p + width];
        }
      }
      const float sum_b = partial_b[0];
      // Only now, after the full k range, does anything round to bf16.
      const float lhs = round_bf16(sum_a);
      const float rhs = round_bf16(sum_b);
      packed.v[j] = __float2bfloat16_rn(round_bf16(silu_f(lhs)) * rhs);
    }
    *reinterpret_cast<Bf16x8*>(out + row * c_out + gn0 + c) = packed;
  }
}

// ---------------------------------------------------------------------------
// AdaLN:  out = sigmoid(linear_g(s_norm)) * (a_norm + linear_s(s_norm))
//
// The same skeleton, with one LayerNorm folded in on the way in and the other
// folded into the epilogue. Four sequential regions of one kernel:
//
//   stage/normalise s -> s_norm in shared memory, shared by both GEMMs
//   stage/normalise a in place (layer_norm_a has no affine parameters at all)
//   accumulate both GEMMs against the shared s_norm tile
//   epilogue, indexed: bias add, sigmoid gate, the a_norm add, one store
//
// Unlike SwiGLU's, this epilogue is *not* elementwise in the accumulator --
// a_norm and bias_g depend on output indices -- so the accumulators go through
// shared memory and the epilogue walks the tile with (row, col) in hand.
//
// layer_norm_a normalises over the whole c_a row while a block owns only
// 16*WARPS of its columns, so when there is more than one n-tile each block
// stages the whole row block just to reduce it. The redundancy is
// c_a / (16*WARPS) extra reads of a, all L2-resident after the first block
// touches them, and it is why WARPS prefers 16*WARPS == c_a when the shape
// allows.
// ---------------------------------------------------------------------------
template <int NCOL, int NSPLIT>
__global__ void __launch_bounds__(kWarpSize* NCOL* NSPLIT)
adaln_kernel(const __nv_bfloat16* __restrict__ a,
             const __nv_bfloat16* __restrict__ s,
             const __nv_bfloat16* __restrict__ ln_s_w,
             const __nv_bfloat16* __restrict__ wg,
             const __nv_bfloat16* __restrict__ bias_g,
             const __nv_bfloat16* __restrict__ ws,
             __nv_bfloat16* __restrict__ out,
             int64_t rows, int c_a, int c_s, int ld_a, int ld_s,
             float inv_ca, float inv_cs, float eps_a, float eps_s) {
  extern __shared__ char smem_raw[];
  constexpr int kWarps = NCOL * NSPLIT;
  const int nthreads = kWarpSize * kWarps;
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid >> 5;
  const int lane = tid & (kWarpSize - 1);
  const int col_group = warp / NSPLIT;
  const int split = warp % NSPLIT;
  const int64_t m0 = static_cast<int64_t>(blockIdx.x) * kTile;
  const int n0 = (static_cast<int>(blockIdx.y) * NCOL + col_group) * kTile;

  // [a tile][ s_norm | accumulator staging ]. s_norm dies once every warp has
  // finished its mma loop, so the two staging tiles overlay it; the a tile lives
  // until the epilogue and cannot. Neither row's statistics need a shared slot:
  // both tiles are normalised in place, so the epilogue reads a_norm and the mma
  // loop reads s_norm straight out of shared memory.
  char* cursor = align_up(smem_raw);
  __nv_bfloat16* as = reinterpret_cast<__nv_bfloat16*>(cursor);
  cursor = align_up(cursor + sizeof(__nv_bfloat16) * kTile * ld_a);
  __nv_bfloat16* sn = reinterpret_cast<__nv_bfloat16*>(cursor);
  float* acc_smem = reinterpret_cast<float*>(cursor) + warp * 2 * kAccFloats;
  float* acc_base = reinterpret_cast<float*>(cursor);

  stage_tile(s, m0, c_s, c_s, sn, ld_s, rows, tid, nthreads);
  stage_tile(a, m0, c_a, c_a, as, ld_a, rows, tid, nthreads);
  __syncthreads();

  // Normalise both tiles in place, one warp per row, so the epilogue reads
  // a_norm straight out of shared memory and the mma loop reads s_norm. A row
  // past `rows` was zero-filled: its mean and variance are 0, rstd is
  // 1/sqrt(eps), and its output is masked off at the store.
  for (int r = warp; r < kTile; r += kWarps) {
    __nv_bfloat16* srow = sn + static_cast<int64_t>(r) * ld_s;
    float mean, rstd;
    row_stats(srow, c_s, lane, inv_cs, eps_s, &mean, &rstd);
    for (int j = lane; j < c_s; j += kWarpSize) {
      // The affine weight is applied in fp32, before the single rounding, and is
      // read from the live parameter: it is torch.ones under this bench, but
      // that is a property of the bench and not of the operator.
      float v = (__bfloat162float(srow[j]) - mean) * rstd;
      if (ln_s_w != nullptr) {
        v *= __bfloat162float(ln_s_w[j]);
      }
      srow[j] = __float2bfloat16_rn(v);
    }
    // layer_norm_a has neither a scale nor an offset, so this is the pure
    // normalisation, rounded once.
    __nv_bfloat16* arow = as + static_cast<int64_t>(r) * ld_a;
    row_stats(arow, c_a, lane, inv_ca, eps_a, &mean, &rstd);
    for (int j = lane; j < c_a; j += kWarpSize) {
      arow[j] = __float2bfloat16_rn((__bfloat162float(arow[j]) - mean) * rstd);
    }
  }
  __syncthreads();

  const bool active = n0 < c_a;
  const int steps = c_s / kTile;
  const int per_split = (steps + NSPLIT - 1) / NSPLIT;
  const int begin = split * per_split;
  const int end = begin + per_split < steps ? begin + per_split : steps;

  wmma::fragment<wmma::accumulator, kTile, kTile, kTile, float> acc_g, acc_s;
  wmma::fill_fragment(acc_g, 0.0f);
  wmma::fill_fragment(acc_s, 0.0f);
  if (active) {
    wmma::fragment<wmma::matrix_a, kTile, kTile, kTile, __nv_bfloat16, row_major> af;
    wmma::fragment<wmma::matrix_b, kTile, kTile, kTile, __nv_bfloat16, col_major>
        bf_g[kStages], bf_s[kStages];
    const __nv_bfloat16* pg = wg + static_cast<int64_t>(n0) * c_s;
    const __nv_bfloat16* ps = ws + static_cast<int64_t>(n0) * c_s;
#pragma unroll
    for (int s = 0; s < kStages; ++s) {
      if (begin + s < end) {
        wmma::load_matrix_sync(bf_g[s], pg + (begin + s) * kTile, c_s);
        wmma::load_matrix_sync(bf_s[s], ps + (begin + s) * kTile, c_s);
      }
    }
    for (int base_step = begin; base_step < end; base_step += kStages) {
#pragma unroll
      for (int s = 0; s < kStages; ++s) {
        const int step = base_step + s;
        if (step < end) {
          wmma::load_matrix_sync(af, sn + step * kTile, ld_s);
          wmma::mma_sync(acc_g, af, bf_g[s], acc_g);
          wmma::mma_sync(acc_s, af, bf_s[s], acc_s);
          const int ahead = step + kStages;
          if (ahead < end) {
            wmma::load_matrix_sync(bf_g[s], pg + ahead * kTile, c_s);
            wmma::load_matrix_sync(bf_s[s], ps + ahead * kTile, c_s);
          }
        }
      }
    }
  }

  __syncthreads();  // s_norm is dead; the staging tiles overlay it
  if (active) {
    // fp32 partials. Rounding one to bf16 here would put a second rounding
    // inside what the baseline computes as a single fp32 accumulation.
    wmma::store_matrix_sync(acc_smem, acc_g, kAccLd, wmma::mem_row_major);
    wmma::store_matrix_sync(acc_smem + kAccFloats, acc_s, kAccLd, wmma::mem_row_major);
  }
  __syncthreads();

  const int chunks = NCOL * (kTile * 2);
  for (int chunk = tid; chunk < chunks; chunk += nthreads) {
    const int group = chunk / (kTile * 2);
    const int within = chunk - group * (kTile * 2);
    const int r = within >> 1;
    const int c = (within & 1) << 3;
    const int gn0 = (static_cast<int>(blockIdx.y) * NCOL + group) * kTile;
    const int64_t row = m0 + r;
    if (gn0 >= c_a || row >= rows) {
      continue;
    }
    const float* part = acc_base + group * NSPLIT * 2 * kAccFloats + r * kAccLd + c;
    const __nv_bfloat16* an = as + static_cast<int64_t>(r) * ld_a + gn0 + c;
    const __nv_bfloat16* bg = bias_g + gn0 + c;
    Bf16x8 packed;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      // Pairwise tree, as in SwiGLU's epilogue: the association is fixed by the
      // shape of the reduction rather than by loop order.
      float partial_g[NSPLIT];
      float partial_s[NSPLIT];
#pragma unroll
      for (int p = 0; p < NSPLIT; ++p) {
        partial_g[p] = part[p * 2 * kAccFloats + j];
        partial_s[p] = part[p * 2 * kAccFloats + kAccFloats + j];
      }
#pragma unroll
      for (int width = NSPLIT / 2; width >= 1; width >>= 1) {
#pragma unroll
        for (int p = 0; p < width; ++p) {
          partial_g[p] += partial_g[p + width];
          partial_s[p] += partial_s[p + width];
        }
      }
      const float sum_g = partial_g[0];
      const float sum_s = partial_s[0];
      // F.linear's fp32 epilogue: the bias joins the *complete* accumulation
      // before the one rounding to bf16, and sigmoid then reads that bf16 value.
      const float gate =
          round_bf16(sigmoid_f(round_bf16(sum_g + __bfloat162float(bg[j]))));
      const float t = round_bf16(__bfloat162float(an[j]) + round_bf16(sum_s));
      packed.v[j] = __float2bfloat16_rn(gate * t);
    }
    *reinterpret_cast<Bf16x8*>(out + row * c_a + gn0 + c) = packed;
  }
}

// ---------------------------------------------------------------------------
// Launch geometry.
// ---------------------------------------------------------------------------

// Device attributes, cached per device rather than once.
//
// A cache keyed on nothing would answer for whichever device happened to be
// current at the first call, which on a multi-GPU host is not necessarily the
// device the launch goes to. `CUDAGuard` has already switched by the time these
// are consulted, so the query is cheap and the answer has to follow the device.
constexpr int kMaxDevices = 16;

int device_attr(cudaDeviceAttr attr, int fallback, int slot) {
  // Atomic because two threads can reach a cold cell at once. Relaxed is enough:
  // the cell only ever moves from 0 to one particular value, so a racing pair
  // computes the same answer twice and stores it twice, and no reader can observe
  // anything but 0 or that value.
  static std::atomic<int> cached[kMaxDevices][4];
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess || device < 0 ||
      device >= kMaxDevices) {
    return fallback;
  }
  std::atomic<int>& cell = cached[device][slot];
  int seen = cell.load(std::memory_order_relaxed);
  if (seen == 0) {
    int value = 0;
    seen = (cudaDeviceGetAttribute(&value, attr, device) == cudaSuccess && value > 0)
               ? value : fallback;
    cell.store(seen, std::memory_order_relaxed);
  }
  return seen;
}

// Blackwell offers far more than the 48 KB a kernel gets by default, and the
// widest admitted (c_a, c_s) pair needs it, so the request is opted into per
// kernel and the predicate compares against what the device actually allows
// rather than a hardcoded number.
int device_smem_ceiling() {
  return device_attr(cudaDevAttrMaxSharedMemoryPerBlockOptin, 48 * 1024, 0);
}

int max_grid_y() { return device_attr(cudaDevAttrMaxGridDimY, 65535, 1); }

int sm_count() { return device_attr(cudaDevAttrMultiProcessorCount, 148, 2); }

// The translation unit is compiled for exactly the device present at import
// (TORCH_CUDA_ARCH_LIST is narrowed to it, which is what keeps the cold build
// inside the harness's wall-clock cap), so a launch onto a different
// architecture would fail with "no kernel image available" -- an exception out of
// the operator rather than the fallback this file promises. On a heterogeneous
// host that is a real call, so it is a predicate.
bool device_arch_matches() {
  const int major = device_attr(cudaDevAttrComputeCapabilityMajor, 0, 3);
  int minor_value = 0;
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess ||
      cudaDeviceGetAttribute(&minor_value, cudaDevAttrComputeCapabilityMinor,
                             device) != cudaSuccess) {
    return false;
  }
  return major * 10 + minor_value == FK_AF3_BUILT_SM;
}

// A/B override, read once from the environment at load. Present so a warp-count
// comparison is a re-run rather than a recompile; unset means the rule below
// decides, which is what ships.
int warps_override(const char* name) {
  const char* raw = std::getenv(name);
  if (raw == nullptr) {
    return 0;
  }
  const int value = std::atoi(raw);
  return (value == 1 || value == 2 || value == 4 || value == 8) ? value : 0;
}

// Largest power of two that is at most both `v` and `cap`, and at least 1.
//
// The launch dispatches on a *compile-time* column-group count, so the value the
// grid is computed from must be one the switch actually instantiates. Returning
// something like 3 would size the grid for three column groups and then launch
// the one-group kernel, leaving two thirds of the output never written -- which
// is exactly what a c_out of 48 did before this existed. Flooring to a power of
// two keeps the host arithmetic and the template parameter in step by
// construction rather than by the switch happening to have the right case.
int pow2_at_most(int64_t v, int cap) {
  int64_t limit = v < cap ? v : cap;
  int result = 1;
  while (result * 2 <= limit) {
    result *= 2;
  }
  return result;
}

int swiglu_warps_env() {
  static int cached = warps_override("FK_AF3_L2_SWIGLU_WARPS");
  return cached;
}

int adaln_warps_env() {
  static int cached = warps_override("FK_AF3_L2_ADALN_WARPS");
  return cached;
}

// Warps per block, hence 16*WARPS output columns per block.
//
// The obvious rule -- take the narrowest block that puts a block on every SM --
// was measured and is wrong, decisively for AdaLN and mildly for SwiGLU
// (`profile/p1-warp-sweep/warp_sweep.json`, the ten scored cases swept over
// 1/2/4/8 through the harness's own timing loop). Block count is not what limits
// these shapes, and chasing it costs on both sides: every extra n-tile
// re-executes the whole per-block prologue, and a narrow block runs that
// prologue with fewer threads.
//
// AdaLN, microseconds, warps 1 / 2 / 4 / 8:
//
//     a[1,1,16,768] s[1,16,384]   58.4  50.2  42.0  40.0
//     a[1,1,12,128,128]           38.0  29.8  25.6  23.5
//     a[1,12,32,128]              37.9  29.7  25.6  23.6
//     a[1,1,368,128]              37.9  29.7  25.6  23.6
//
// Monotone in the warp count on every case, because AdaLN's prologue is the
// expensive part: two 16-row tiles staged and four row reductions run over them,
// duplicated across every n-tile that shares the m-tile. So take the widest
// block the column count supports. At c_a = 128 that lands on 16*WARPS == c_a
// exactly -- one n-tile, no duplication at all -- which is the right way round,
// since those are also the cases where `a` is large.
int adaln_warps(int64_t col_tiles) {
  const int env = adaln_warps_env();
  return pow2_at_most(col_tiles, env != 0 ? env : 8);
}

int adaln_nsplit_env() {
  static int cached = warps_override("FK_AF3_L2_ADALN_NSPLIT");
  return cached;
}

// AdaLN's k-split, on the same evidence as SwiGLU's and with one difference.
//
// The single-m-tile case (c_a = 768, c_s = 384, 48 output tiles) leaves the
// machine as idle as SwiGLU's did, so splitting k the same way multiplies the
// resident warps the same way. But SwiGLU can afford one column group per block,
// and AdaLN cannot: its prologue stages two 16-row tiles and reduces four rows
// over them, and that work is repeated by every block sharing the m-tile. The
// warp sweep is what says so -- AdaLN improved monotonically from 1 to 8 warps
// precisely because wider blocks meant fewer of them.
//
// So the block stays eight warps wide and the split is taken *out of* the column
// groups rather than added on top: NSPLIT = 4 means two column groups, not eight.
// Block count goes up, prologue thread count stays at 256, and the number of
// prologues rises only as far as the split factor.
int adaln_nsplit(int64_t col_tiles, int64_t m_tiles, int steps) {
  const int env = adaln_nsplit_env() > 4 ? 4 : adaln_nsplit_env();
  if (env != 0) {
    return steps >= env ? env : 1;
  }
  if (col_tiles * m_tiles >= sm_count()) {
    return 1;
  }
  for (int split = 4; split >= 2; split >>= 1) {
    if (steps / split >= 6 && col_tiles >= split) {
      return split;
    }
  }
  return 1;
}

// SwiGLU, same sweep:
//
//     x[1,16,384]     (M=16)      21.5  23.5  23.5  37.9
//     x[1,1,368,128]  (M=368)     13.3  13.3  13.3  17.4
//     x[1,16,16,128]  (M=256)     13.3  13.3  13.3  17.4
//     x[1,1,16,768]   (M=16)      33.8  35.8  33.8  66.7
//     x[1,8,16,64]    (M=128)     11.3  13.3  11.3  13.3
//
// Flat from 1 to 4 and worse at 8: the prologue here is a single x tile, so
// widening the block buys nothing and eventually costs -- 8 warps is 16*8 = 128
// columns per block, which over-subscribes the 256-column cases and leaves half
// the warps idle on the trailing n-tile. Four is the widest value that is never
// worse, and it keeps the block wide enough that the x tile is staged quickly.
//
// The two M = 16 cases sit far off the window floor at every warp count, which
// says their limit is elsewhere -- see `swiglu_ksplit` below.
int swiglu_warps(int64_t col_tiles) {
  const int env = swiglu_warps_env();
  return pow2_at_most(col_tiles, env != 0 ? env : 4);
}

int swiglu_nsplit_env() {
  static int cached = warps_override("FK_AF3_L2_SWIGLU_NSPLIT");
  return cached;
}

// How many warps share one output tile by splitting the k range.
//
// One warp per 16x16 output tile caps the warp population at c_out/16 * m_tiles,
// which for the two M = 16 cases is 96 -- well under the 148 SMs. Measurement
// says that ceiling, and not per-warp load depth, is what binds: sweeping the
// register pipeline from 1 to 8 stages moved these cases by under 4% with no
// spilling at any depth (`profile/p1-warp-sweep/stage_sweep.json`, registers
// 54-64 at one stage against 96-116 at eight), and the warp sweep was flat
// because 96 blocks of one warp and 24 blocks of four warps have the same
// aggregate: the first spreads over 4x the SMs with 1/4 the warps each to keep
// their L1 busy.
//
// Splitting k is the one arrangement that raises the product, and it does, by a
// lot. Measured on the two single-m-tile cases (speedup against the baseline in
// the same process, `profile/p1-warp-sweep/nsplit*.json`):
//
//     NSPLIT                        1     2     4     8
//     x[1,1,16,768]  c_in=768    0.57  0.92  1.24  0.54
//     x[1,16,384]    c_in=384    0.83  1.12  1.13  0.83
//
// Four is best on both and eight is worse than not splitting at all. The cases
// that already fill the machine are untouched at every setting, which is what the
// tile-count guard below is for.
//
// The ceiling on the split is the length of the k-slice each warp is left with:
// at c_in = 768 a split of eight leaves six 16-wide steps and loses 2.3x, because
// the prologue, the shared-memory reduction and the ragged tail stop being
// amortised. Six steps is therefore the floor, and four the cap -- eight is never
// selected on any shape.
int swiglu_nsplit(int64_t col_tiles, int64_t m_tiles, int steps) {
  // Only 1, 2 and 4 are instantiated; clamping here rather than letting the
  // dispatch fall through keeps the knob honest -- a request for 8 would
  // otherwise run unsplit while claiming to be split.
  const int env = swiglu_nsplit_env() > 4 ? 4 : swiglu_nsplit_env();
  if (env != 0) {
    return steps >= env ? env : 1;
  }
  if (col_tiles * m_tiles >= sm_count()) {
    return 1;  // the grid already fills the machine; splitting only adds overhead
  }
  for (int split = 4; split >= 2; split >>= 1) {
    if (steps / split >= 6) {
      return split;
    }
  }
  return 1;
}

int smem_swiglu(int c_in, int warps, int* ld_x) {
  *ld_x = c_in + kSmemPadElems;
  const int64_t x_bytes = static_cast<int64_t>(sizeof(__nv_bfloat16)) * kTile * *ld_x;
  const int64_t acc_bytes = static_cast<int64_t>(sizeof(float)) * warps * 2 * kAccFloats;
  const int64_t need = (x_bytes > acc_bytes ? x_bytes : acc_bytes) + kAlign;
  return need > INT32_MAX ? -1 : static_cast<int>(need);
}

int smem_adaln(int c_a, int c_s, int warps, int* ld_a, int* ld_s) {
  *ld_a = c_a + kSmemPadElems;
  *ld_s = c_s + kSmemPadElems;
  const int64_t a_bytes = static_cast<int64_t>(sizeof(__nv_bfloat16)) * kTile * *ld_a;
  const int64_t s_bytes = static_cast<int64_t>(sizeof(__nv_bfloat16)) * kTile * *ld_s;
  const int64_t acc_bytes = static_cast<int64_t>(sizeof(float)) * warps * 2 * kAccFloats;
  const int64_t tail = s_bytes > acc_bytes ? s_bytes : acc_bytes;
  // One kAlign of slack per align_up the kernel performs on the way through.
  const int64_t need = a_bytes + tail + 2 * kAlign;
  return need > INT32_MAX ? -1 : static_cast<int>(need);
}

template <typename Fn>
bool opt_in_smem(Fn kernel, int bytes) {
  // Idempotent per instantiation: the attribute is a property of the function,
  // so one successful call covers every later launch. Failure is a rejection,
  // not an error -- the caller takes the fallback.
  if (bytes <= 48 * 1024) {
    return true;
  }
  if (bytes > device_smem_ceiling()) {
    return false;
  }
  return cudaFuncSetAttribute(reinterpret_cast<const void*>(kernel),
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              bytes) == cudaSuccess;
}

// ---------------------------------------------------------------------------
// Predicates. Every one of these guards something a kernel relies on; failing
// any of them returns nullopt and the caller evaluates the baseline composition.
// ---------------------------------------------------------------------------
bool aligned(const at::Tensor& t) {
  return (reinterpret_cast<uintptr_t>(t.const_data_ptr()) % kAlign) == 0;
}

bool bf16_cuda_contig(const at::Tensor& t, const at::Device& device) {
  return t.defined() && t.scalar_type() == at::kBFloat16 && t.is_cuda() &&
         t.device() == device && t.is_contiguous();
}

// A [c_out, c_in] weight the B fragment can read in place: k-contiguous, both
// extents on the tile, and a base the 256-bit matrix pointer accepts.
bool weight_ok(const at::Tensor& w, const at::Device& device, int64_t rows_expected,
               int64_t cols_expected) {
  return bf16_cuda_contig(w, device) && w.dim() == 2 &&
         w.size(0) == rows_expected && w.size(1) == cols_expected && aligned(w);
}

bool vector_ok(const at::Tensor& v, const at::Device& device, int64_t numel) {
  return bf16_cuda_contig(v, device) && v.dim() == 1 && v.size(0) == numel &&
         aligned(v);
}

// Grid extents the launch itself has to fit, given m goes on x (effectively
// unbounded) and n on y.
bool grid_ok(int64_t m_tiles, int64_t col_tiles, int warps) {
  const int64_t n_blocks = (col_tiles + warps - 1) / warps;
  return m_tiles > 0 && m_tiles <= INT32_MAX && n_blocks >= 1 &&
         n_blocks <= max_grid_y();
}

// ---------------------------------------------------------------------------
// SwiGLU entry point.
// ---------------------------------------------------------------------------
std::optional<at::Tensor> swiglu(const at::Tensor& x, const at::Tensor& wa,
                                 const at::Tensor& wb) {
  // The kernels allocate with at::empty and launch raw, so they record nothing
  // for autograd. Grad mode being *enabled* is the test, not whether some tensor
  // currently requires grad: a caller who has not entered no_grad may attach
  // requires_grad later in the same graph, or replay this forward under
  // checkpointing with different requires_grad state.
  if (at::GradMode::is_enabled()) {
    return std::nullopt;
  }
  if (!x.defined() || !wa.defined() || !wb.defined()) {
    return std::nullopt;
  }
  const at::Device device = x.device();
  if (!device.is_cuda()) {
    return std::nullopt;
  }
  // Establishing the device first: CUDAGuard rejects a CPU device by throwing,
  // and this operator never throws. The guard is what makes the attribute
  // queries below answer for the device the launch will go to.
  const c10::cuda::CUDAGuard arch_guard(device);
  if (!device_arch_matches()) {
    return std::nullopt;
  }
  if (!bf16_cuda_contig(x, device) || x.dim() < 1 || !aligned(x)) {
    return std::nullopt;
  }
  if (wa.dim() != 2) {
    return std::nullopt;
  }
  const int64_t c_out = wa.size(0);
  const int64_t c_in = wa.size(1);
  if (!weight_ok(wa, device, c_out, c_in) || !weight_ok(wb, device, c_out, c_in)) {
    return std::nullopt;
  }
  // Both extents on the tile: k must be exact for the mma loop and n for the
  // B fragment's 16 columns; c_in doubles as the B ldm, which wmma needs to be a
  // multiple of 8 elements, so the 16 here is the stricter of the two.
  if (c_in <= 0 || c_out <= 0 || (c_in % kTile) != 0 || (c_out % kTile) != 0 ||
      c_in > kMaxChannels || c_out > kMaxChannels) {
    return std::nullopt;
  }
  if (x.size(-1) != c_in) {
    return std::nullopt;
  }
  const int64_t rows = x.numel() / c_in;
  if (rows <= 0) {
    return std::nullopt;
  }

  const int64_t m_tiles = (rows + kTile - 1) / kTile;
  const int64_t col_tiles = c_out / kTile;
  const int nsplit = swiglu_nsplit(col_tiles, m_tiles, static_cast<int>(c_in / kTile));
  // With a k-split, one column group per block is what puts the extra warps on
  // extra SMs rather than stacking them onto the same few.
  const int ncol = nsplit > 1 ? 1 : swiglu_warps(col_tiles);
  const int warps = ncol * nsplit;
  if (!grid_ok(m_tiles, col_tiles, ncol)) {
    return std::nullopt;
  }
  int ld_x = 0;
  const int smem = smem_swiglu(static_cast<int>(c_in), warps, &ld_x);
  if (smem < 0 || smem > device_smem_ceiling()) {
    return std::nullopt;
  }

  std::vector<int64_t> out_shape(x.sizes().begin(), x.sizes().end());
  out_shape.back() = c_out;
  at::Tensor out = at::empty(out_shape, x.options());

  const dim3 grid(static_cast<unsigned>(m_tiles),
                  static_cast<unsigned>((col_tiles + ncol - 1) / ncol));
  const auto stream = c10::cuda::getCurrentCUDAStream();
  const __nv_bfloat16* xp = reinterpret_cast<const __nv_bfloat16*>(x.const_data_ptr());
  const __nv_bfloat16* ap = reinterpret_cast<const __nv_bfloat16*>(wa.const_data_ptr());
  const __nv_bfloat16* bp = reinterpret_cast<const __nv_bfloat16*>(wb.const_data_ptr());
  __nv_bfloat16* op = reinterpret_cast<__nv_bfloat16*>(out.mutable_data_ptr());

#define FK_AF3_SWIGLU_LAUNCH(NC, NS)                                             \
  do {                                                                           \
    if (!opt_in_smem(swiglu_kernel<NC, NS>, smem)) {                             \
      return std::nullopt;                                                       \
    }                                                                            \
    swiglu_kernel<NC, NS><<<grid, kWarpSize * (NC) * (NS), smem, stream>>>(      \
        xp, ap, bp, op, rows, static_cast<int>(c_in), static_cast<int>(c_out),    \
        ld_x);                                                                   \
  } while (0)

  if (nsplit == 4) {
    FK_AF3_SWIGLU_LAUNCH(1, 4);
  } else if (nsplit == 2) {
    FK_AF3_SWIGLU_LAUNCH(1, 2);
  } else {
    switch (ncol) {
      case 8: FK_AF3_SWIGLU_LAUNCH(8, 1); break;
      case 4: FK_AF3_SWIGLU_LAUNCH(4, 1); break;
      case 2: FK_AF3_SWIGLU_LAUNCH(2, 1); break;
      default: FK_AF3_SWIGLU_LAUNCH(1, 1); break;
    }
  }
#undef FK_AF3_SWIGLU_LAUNCH
  return out;
}

// ---------------------------------------------------------------------------
// AdaLN entry point.
// ---------------------------------------------------------------------------
std::optional<at::Tensor> adaln(const at::Tensor& a, const at::Tensor& s,
                                const std::optional<at::Tensor>& ln_s_weight,
                                const at::Tensor& wg, const at::Tensor& bias_g,
                                const at::Tensor& ws, double eps_a, double eps_s) {
  if (at::GradMode::is_enabled()) {
    return std::nullopt;
  }
  if (!a.defined() || !s.defined() || !wg.defined() || !ws.defined() ||
      !bias_g.defined()) {
    return std::nullopt;
  }
  const at::Device device = a.device();
  if (!device.is_cuda()) {
    return std::nullopt;
  }
  const c10::cuda::CUDAGuard arch_guard(device);
  if (!device_arch_matches()) {
    return std::nullopt;
  }
  if (!bf16_cuda_contig(a, device) || !bf16_cuda_contig(s, device) ||
      a.dim() < 1 || s.dim() < 1 || !aligned(a) || !aligned(s)) {
    return std::nullopt;
  }
  if (wg.dim() != 2) {
    return std::nullopt;
  }
  const int64_t c_a = wg.size(0);
  const int64_t c_s = wg.size(1);
  if (!weight_ok(wg, device, c_a, c_s) || !weight_ok(ws, device, c_a, c_s)) {
    return std::nullopt;
  }
  if (!vector_ok(bias_g, device, c_a)) {
    return std::nullopt;
  }
  if (c_a <= 0 || c_s <= 0 || (c_a % kTile) != 0 || (c_s % kTile) != 0 ||
      c_a > kMaxChannels || c_s > kMaxChannels) {
    return std::nullopt;
  }
  const __nv_bfloat16* wp = nullptr;
  if (ln_s_weight.has_value() && ln_s_weight->defined()) {
    if (!vector_ok(*ln_s_weight, device, c_s)) {
      return std::nullopt;
    }
    wp = reinterpret_cast<const __nv_bfloat16*>(ln_s_weight->const_data_ptr());
  }
  if (a.size(-1) != c_a || s.size(-1) != c_s) {
    return std::nullopt;
  }
  const int64_t rows = a.numel() / c_a;
  if (rows <= 0 || rows != s.numel() / c_s) {
    return std::nullopt;
  }
  // The baseline's output is the broadcast of a_norm (shape a.shape) with the
  // two linears (shape s.shape[:-1] + (c_a,)). Pairing row m of a with row m of
  // s is only legal when that broadcast changes nothing about the row ordering,
  // which is exactly: the broadcast equals a.shape, and the two row counts
  // agree (checked above). Anything that needs a genuine expansion -- a of
  // [1, 4, 16, 768] against s of [1, 16, 384] -- is rejected here rather than
  // silently computed row-for-row.
  {
    std::vector<int64_t> rhs(s.sizes().begin(), s.sizes().end());
    rhs.back() = c_a;
    const int64_t ra = a.dim();
    const int64_t rb = static_cast<int64_t>(rhs.size());
    if (rb > ra) {
      return std::nullopt;  // would broadcast to more dimensions than a has
    }
    for (int64_t i = 0; i < rb; ++i) {
      const int64_t da = a.size(ra - 1 - i);
      const int64_t db = rhs[rb - 1 - i];
      if (db != da && db != 1) {
        return std::nullopt;
      }
    }
  }

  const int64_t m_tiles = (rows + kTile - 1) / kTile;
  const int64_t col_tiles = c_a / kTile;
  const int nsplit = adaln_nsplit(col_tiles, m_tiles, static_cast<int>(c_s / kTile));
  // The split comes out of the column groups, so the block stays eight warps wide.
  const int ncol = nsplit > 1 ? pow2_at_most(col_tiles, 8 / nsplit)
                              : adaln_warps(col_tiles);
  const int warps = ncol * nsplit;
  if (!grid_ok(m_tiles, col_tiles, ncol)) {
    return std::nullopt;
  }
  int ld_a = 0, ld_s = 0;
  const int smem = smem_adaln(static_cast<int>(c_a), static_cast<int>(c_s), warps,
                              &ld_a, &ld_s);
  if (smem < 0 || smem > device_smem_ceiling()) {
    return std::nullopt;
  }

  at::Tensor out = at::empty(a.sizes(), a.options());

  const dim3 grid(static_cast<unsigned>(m_tiles),
                  static_cast<unsigned>((col_tiles + ncol - 1) / ncol));
  const auto stream = c10::cuda::getCurrentCUDAStream();
  const __nv_bfloat16* ap = reinterpret_cast<const __nv_bfloat16*>(a.const_data_ptr());
  const __nv_bfloat16* sp = reinterpret_cast<const __nv_bfloat16*>(s.const_data_ptr());
  const __nv_bfloat16* gp = reinterpret_cast<const __nv_bfloat16*>(wg.const_data_ptr());
  const __nv_bfloat16* bp = reinterpret_cast<const __nv_bfloat16*>(bias_g.const_data_ptr());
  const __nv_bfloat16* lp = reinterpret_cast<const __nv_bfloat16*>(ws.const_data_ptr());
  __nv_bfloat16* op = reinterpret_cast<__nv_bfloat16*>(out.mutable_data_ptr());
  const float inv_ca = 1.0f / static_cast<float>(c_a);
  const float inv_cs = 1.0f / static_cast<float>(c_s);

#define FK_AF3_ADALN_LAUNCH(NC, NS)                                              \
  do {                                                                           \
    if (!opt_in_smem(adaln_kernel<NC, NS>, smem)) {                              \
      return std::nullopt;                                                       \
    }                                                                            \
    adaln_kernel<NC, NS><<<grid, kWarpSize * (NC) * (NS), smem, stream>>>(       \
        ap, sp, wp, gp, bp, lp, op, rows, static_cast<int>(c_a),                  \
        static_cast<int>(c_s), ld_a, ld_s, inv_ca, inv_cs,                        \
        static_cast<float>(eps_a), static_cast<float>(eps_s));                   \
  } while (0)

  if (nsplit == 4) {
    if (ncol == 2) {
      FK_AF3_ADALN_LAUNCH(2, 4);
    } else {
      FK_AF3_ADALN_LAUNCH(1, 4);
    }
  } else if (nsplit == 2) {
    if (ncol == 4) {
      FK_AF3_ADALN_LAUNCH(4, 2);
    } else if (ncol == 2) {
      FK_AF3_ADALN_LAUNCH(2, 2);
    } else {
      FK_AF3_ADALN_LAUNCH(1, 2);
    }
  } else {
    switch (ncol) {
      case 8: FK_AF3_ADALN_LAUNCH(8, 1); break;
      case 4: FK_AF3_ADALN_LAUNCH(4, 1); break;
      case 2: FK_AF3_ADALN_LAUNCH(2, 1); break;
      default: FK_AF3_ADALN_LAUNCH(1, 1); break;
    }
  }
#undef FK_AF3_ADALN_LAUNCH
  return out;
}

}  // namespace

TORCH_LIBRARY(fk_af3_swiglu_l2, m) {
  m.def("swiglu(Tensor x, Tensor wa, Tensor wb) -> Tensor?", &swiglu);
  m.def(
      "adaln(Tensor a, Tensor s, Tensor? ln_s_weight, Tensor wg, Tensor bias_g, "
      "Tensor ws, float eps_a, float eps_s) -> Tensor?",
      &adaln);
}
"""


def _extra_cuda_flags() -> list:
    """Optional build-time overrides, for A/B only.

    ``FK_AF3_L2_CUDA_FLAGS`` is whitespace-split and appended to the nvcc command
    line, which is how ``tools/ab_stages.py`` recompiles the pipeline depth
    without a runtime switch in the shipped path. Unset in every normal run.
    """
    raw = os.environ.get("FK_AF3_L2_CUDA_FLAGS", "").split()
    return [flag for flag in raw if flag.startswith("-D")]


def _load_fused_ops():
    """Build and register both operators, returning their bound overloads.

    Compilation happens here, at import, so nothing is deferred into a timed
    ``forward``: the harness snapshots ``threading.active_count()`` around the
    candidate's timing and treats a new thread as tampering, and a first-call
    build would also stall the worker's output past its watchdog. The includes
    are lean on purpose -- ``<torch/extension.h>`` through nvcc dominates the
    build, and these ops are registered with ``TORCH_LIBRARY`` rather than
    pybind, so none of it is needed.
    """
    from torch.utils.cpp_extension import load_inline

    # Without this, cpp_extension honours the ambient TORCH_CUDA_ARCH_LIST, which
    # in this environment names six architectures -- six nvcc passes over every
    # template instantiation, for five targets that will never run these kernels.
    # Narrowing it to the device actually present cuts the cold build
    # several-fold, which is what keeps it inside the harness's wall-clock cap.
    # Derived from the live device rather than hardcoded, so it can never name the
    # wrong arch, and restored afterwards so no later build in this process is
    # affected.
    previous_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST")
    built_sm = 0
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
        built_sm = major * 10 + minor
    try:
        load_inline(
            name=_LIBRARY_NAME,
            cpp_sources="",
            cuda_sources=_CUDA_SOURCE,
            # The operators compare a call's device against this, so a launch onto
            # a different architecture falls back instead of raising "no kernel
            # image available" out of the fast path.
            extra_cuda_cflags=["-O3", f"-DFK_AF3_BUILT_SM={built_sm}"]
            + _extra_cuda_flags(),
            is_python_module=False,
            no_implicit_headers=True,
        )
    finally:
        if previous_arch_list is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = previous_arch_list
    # Bind the overloads, not the packets: a packet re-resolves the overload from
    # the argument types on every call, and forward is launch-latency bound.
    library = getattr(torch.ops, _LIBRARY_NAME)
    return library.swiglu.default, library.adaln.default


try:
    _fused_swiglu, _fused_adaln = _load_fused_ops()
except Exception as exc:  # noqa: BLE001
    # A build that cannot happen must degrade, not take the module down with it:
    # an import failure costs every case at once, and every level above this one.
    # One line, at import, on stderr -- the bench worker routes it to the
    # per-operator log, so a swallowed build failure stays visible instead of
    # hiding behind a silent 1.00x.
    _fused_swiglu = _fused_adaln = None
    print(f"[candidate L2/alphafold3_swiglu] fused kernels unavailable, "
          f"delegating to the baseline composition: {type(exc).__name__}: {exc}",
          file=sys.stderr, flush=True)


# Fast-path entries per class. Plain ints incremented on the host: no threads, no
# device sync, nothing the harness's integrity guards watch. This is what
# separates "the fused kernel ran and tied" from "the fused kernel never ran",
# and it is what the rejection tests assert on.
_FASTPATH_HITS = {"SwiGLU": 0, "AdaLN": 0}


def fastpath_hits() -> dict:
    """A copy of the per-class fused-launch counters."""
    return dict(_FASTPATH_HITS)


def reset_fastpath_hits() -> None:
    for key in _FASTPATH_HITS:
        _FASTPATH_HITS[key] = 0


class SwiGLU(nn.Module):
    """SwiGLU activation: SiLU(Wa x) * Wb x.

    Registers exactly the baseline's submodules and derives nothing from them:
    the harness moves and casts the module and only *then* loads the baseline's
    state dict, so anything precomputed in ``__init__`` -- a transposed or
    concatenated weight above all -- would be stale. The submodules are the
    frozen L1 winners, imported the way the baseline imports them, and they are
    what the fallback runs.

    Args:
        c_in: Number of input channels
        c_out: Number of output channels
    """

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.silu = SiLU()
        self.linear_a = Linear(c_in, c_out, bias=False)
        self.linear_b = Linear(c_in, c_out, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if _fused_swiglu is not None:
            linear_a, linear_b = self.linear_a, self.linear_b
            # The only checks that cannot live in C++ without shipping the
            # parameters there just to prove they are absent. A bias would change
            # the formula, so its presence has to route to the baseline.
            if linear_a.bias is None and linear_b.bias is None:
                out = _fused_swiglu(x, linear_a.weight, linear_b.weight)
                if out is not None:
                    _FASTPATH_HITS["SwiGLU"] += 1
                    return out
        return self.silu(self.linear_a(x)) * self.linear_b(x)


class AdaLN(nn.Module):
    """Adaptive Layer Normalization matching the reference AdaLN.

    Submodule structure matches checkpoint keys:
    - layer_norm_s: LayerNorm(c_s), weight-only
    - linear_g: Linear(c_s, c_a, bias=True) — gating
    - linear_s: Linear(c_s, c_a, bias=False) — additive conditioning

    Reference: openfold3/core/model/primitives/normalization.py AdaLN

    Args:
        c_a: Activation channel dimension
        c_s: Conditioning channel dimension
    """

    def __init__(self, c_a: int, c_s: int):
        super().__init__()
        self.c_a = c_a
        self.c_s = c_s

        self.layer_norm_a = LayerNorm(c_a, create_scale=False, create_offset=False)
        self.layer_norm_s = LayerNorm(c_s, create_offset=False)
        self.sigmoid = Sigmoid()
        self.linear_g = Linear(c_s, c_a, bias=True)
        self.linear_s = Linear(c_s, c_a, bias=False)

    def forward(self, a: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        if _fused_adaln is not None:
            ln_a, ln_s = self.layer_norm_a, self.layer_norm_s
            linear_g = self.linear_g
            # The affine shape of both norms, and the fp32 promotion both perform,
            # decide *which formula* the baseline computes, so they are read from
            # the live submodules on every call rather than cached in __init__.
            # The kernel folds a weight-only layer_norm_s and a parameterless
            # layer_norm_a; anything else is the baseline's business.
            # The affine shape of both norms, the axis each one reduces, and the
            # fp32 promotion they perform all decide *which formula* the baseline
            # computes, so each is read from the live submodule per call. A
            # normalized_shape of more than one dimension is a different reduction
            # than the kernel's per-row one and has to reach the baseline.
            if (ln_s.bias is None and ln_a.weight is None and ln_a.bias is None
                    and ln_s.promote_fp32 and ln_a.promote_fp32
                    and ln_a.normalized_shape == (self.c_a,)
                    and ln_s.normalized_shape == (self.c_s,)
                    and linear_g.bias is not None and self.linear_s.bias is None):
                out = _fused_adaln(a, s, ln_s.weight, linear_g.weight, linear_g.bias,
                                   self.linear_s.weight, ln_a.eps, ln_s.eps)
                if out is not None:
                    _FASTPATH_HITS["AdaLN"] += 1
                    return out
        s_norm = self.layer_norm_s(s)
        g = self.sigmoid(self.linear_g(s_norm))
        a_norm = self.layer_norm_a(a)
        return g * (a_norm + self.linear_s(s_norm))
