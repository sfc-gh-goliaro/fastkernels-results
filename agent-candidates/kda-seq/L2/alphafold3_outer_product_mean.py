"""Fused outer product mean for AlphaFold3 (L2) on B200 / sm_100.

The baseline (AF3 Algorithm 9) is an eager chain of thirteen kernels over a problem
that is 35.6 M MAC and 344 KB -- about 32 ns of tensor-core math and 43 ns of HBM
traffic on this device. Measured decomposition
(``profile/p1-baseline-characterisation/``) puts the cost almost entirely in launches
and eager dispatch:

    measured_us ~= 12.2 (harness floor) + ~4.5 * launches + device_time

so the target is the launch *count*, and the reachable ideal is one. This file replaces
the chain with a single custom operator holding a single kernel that recomputes the
LayerNorm and both ``64 -> 32`` projections inside every CTA rather than communicating
them through global memory: the redundant arithmetic costs less than the launch it
removes.

The kernel is latency-bound rather than compute- or bandwidth-bound, and it cannot be
otherwise: one output element per thread pins the grid to ``N_res^2 * c_z = 32768``
threads, which over 148 SMs is ~6.9 warps per SM for *any* tile shape, so the schedulers
issue on about 45% of cycles no matter how the work is arranged. Every tuning knob was
measured against that ceiling rather than against an arithmetic roofline; the record is
``profile/p2-fused-kernel/`` and the sizing derivation that preceded the kernel is
``docs/tile-derivation.md``.

Eligibility and the exact fallback both live in C++, so one Python-level dispatch covers
the fast path and everything it does not admit. Registration is a schema-only
``m.def``, which makes the operator ``CompositeImplicitAutograd``: a gradient-enabled
call decomposes through the fallback's ATen ops instead of silently losing its graph.

Rounding points are the baseline's, not the mathematically best ones. In particular
``norm`` is rounded to bf16 *after* the ``eps`` add, exactly as the baseline's bf16
einsum and bf16 scalar add do. That matters only where no sequence pair is co-valid --
there the output is ``linear_out.bias / norm``, and the baseline's ``norm`` is
``bf16(0 + 1e-3) = 0.0009765625`` rather than ``0.001``, a ~2.4% shift that would exceed
the bf16 comparison bound at that magnitude. Reproducing it costs one instruction and
has no effect on the scored all-ones-mask case.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# The baseline builds its LayerNorm with the default eps and leaves ``promote_fp32``
# on, so the reduction and the affine parameters are fp32 and the result is rounded
# once on the way out. Both are properties of the baseline call rather than of this
# operator's signature, which is why they are named here instead of being plumbed
# through ``__init__``.
_LN_EPS = 1e-5

# Unique to this file so importing it under a second module name cannot double-register
# the operator; registration happens once at ``.so`` load and Python's module cache
# makes any later import a no-op.
_LIBRARY_NAME = "fk_af3_opm_cand"

# Tile, chosen by the measured sweep in ``profile/p2-fused-kernel/analysis/sweep.md``.
# PAIR_SIDE x PAIR_SIDE residue pairs by ZT channels of c_z per CTA, one output per
# thread, so the grid is (R/PAIR_SIDE)^2 * (c_z/ZT) * batch = 128 CTAs at the captured
# shape -- one per SM across 86% of the 148 SMs, measured directly by reading %smid
# (profile/p2-fused-kernel/harness/smid.cu), against the 32 CTAs (21.6%) that the draft's
# first-cut 4x4/64 tile would have launched.
_PAIR_SIDE = 4
_ZT = 16
# Largest MSA depth the kernel's shared-memory budget is sized for. Deeper inputs take
# the exact fallback rather than a second code path. The captured MSA depth is 8.
_S_MAX = 8
# How the c_z slice of w_out reaches shared memory (1 = cp.async), where its wait sits
# (1 = immediately before the first fragment load, so the transfer overlaps the front of
# the kernel), and the CTA width. All three were chosen by the measured sweep in
# profile/p2-fused-kernel/analysis/sweep.md.
_STAGE = 1
_OVERLAP = 1
_THREADS = 256

_CUDA_SOURCE = r"""
#include <ATen/ATen.h>
#include <ATen/TensorSubclassLikeUtils.h>
#include <ATen/autocast_mode.h>
#include <ATen/core/grad_mode.h>
#include <torch/library.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>

#include <cstdint>
#include <optional>
#include <vector>

#ifndef FK_OPM_PAIR_SIDE
#define FK_OPM_PAIR_SIDE 4
#endif
#ifndef FK_OPM_ZT
#define FK_OPM_ZT 16
#endif
#ifndef FK_OPM_S_MAX
#define FK_OPM_S_MAX 8
#endif
// How the c_z slice of w_out reaches shared memory. 0 stages it through registers, 1
// issues cp.async straight from global to shared and waits once. Measured in
// profile/p2-fused-kernel/analysis/sweep.md.
#ifndef FK_OPM_STAGE
#define FK_OPM_STAGE 1
#endif
// Where the cp.async group is waited on. 1 waits immediately before the first fragment
// load, which is the only read of the staged slice, so the transfer overlaps the LayerNorm,
// both projections and the outer-product build. 0 waits at the issue site, which is what
// round 1 shipped by mistake and is kept only so the sweep can measure the difference.
#ifndef FK_OPM_OVERLAP
#define FK_OPM_OVERLAP 1
#endif
// Threads per CTA. Once the contraction moved to mma this stopped being tied to the output
// count -- the epilogue walks the CTA's outputs -- so it is a free parameter, and more warps
// is the only lever on an issue rate that sits at 34.5% with two warps per scheduler.
#ifndef FK_OPM_THREADS
#define FK_OPM_THREADS 256
#endif

namespace {

// The kernel is specialised to the captured projection geometry. Anything else is a
// different kernel, not a slower path through this one, so the predicate rejects it and
// the ATen fallback answers instead.
// FK_OPM_DEVICE_BEGIN
constexpr int kCm = 64;             // c_m, the MSA channel width
constexpr int kC = 32;              // c_hidden
constexpr int kK = kC * kC;         // the linear_out contraction depth, 1024
constexpr int kWarpSize = 32;
constexpr unsigned kFullMask = 0xffffffffu;
constexpr int kThreads = FK_OPM_THREADS;

// The tensor-core tile. mma.m16n8k16 consumes bf16 operands directly and accumulates in
// fp32, which is what the baseline's rounding chain wants anyway.
constexpr int kMmaM = 16;
constexpr int kMmaN = 8;
constexpr int kMmaK = 16;

// Row pads. Every shared buffer an mma fragment is read from uses a stride whose value in
// 4-byte words is 4 (mod 32): a fragment load has lane = 4 * groupId + tidInGroup reading
// word offset groupId * stride_words + tidInGroup, so stride_words % 32 == 4 sends the 32
// lanes to 32 distinct banks. 1024 + 8 bf16 = 516 words and 64 + 8 bf16 = 36 words both
// satisfy it, and both are multiples of 8 elements so 16-byte accesses stay aligned.
constexpr int kRowPadK = 8;                    // for buffers indexed by the contraction K
constexpr int kRowPadCm = 8;                   // for buffers indexed by c_m
constexpr int kKRow = kK + kRowPadK;           // 1032
constexpr int kCmRow = kCm + kRowPadCm;        // 72

// bf16 -> fp32 is exactly a 16-bit left shift, so the whole conversion is one integer op
// with no rounding decision to make. ``__bfloat162float`` does not compile to this -- an
// NCU source attribution of an earlier version put 19% of all instructions inside
// cuda_bf16.hpp -- so every hot conversion goes through these.
__device__ __forceinline__ float lo_to_f32(const unsigned bits) {
  return __uint_as_float(bits << 16);
}

__device__ __forceinline__ float hi_to_f32(const unsigned bits) {
  return __uint_as_float(bits & 0xffff0000u);
}

__device__ __forceinline__ float bf16_to_f32(const __nv_bfloat16 v) {
  return lo_to_f32(static_cast<unsigned>(__bfloat16_as_ushort(v)));
}

// Copy 16 bytes global -> shared without passing through a register, so the staging loop
// holds no live values across its latency and the wait is a single barrier at the end.
__device__ __forceinline__ void cp_async_16(void* dst, const void* src) {
  const unsigned smem_addr =
      static_cast<unsigned>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
               :: "r"(smem_addr), "l"(src) : "memory");
}

__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;");
}

// Waits for every committed group of *this thread*; a __syncthreads() after it is what
// makes the whole CTA's copies visible to every thread.
__device__ __forceinline__ void cp_async_wait_all() {
  asm volatile("cp.async.wait_group 0;" ::: "memory");
}

__device__ __forceinline__ unsigned pack_bf16x2(const float lo, const float hi) {
  const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<const unsigned*>(&v);
}

// Fragments are named-member structs rather than arrays: an indexed local array of this
// size defeated nvcc's register allocation once already and landed in local memory.
struct FragA {              // 16x16 bf16, row-major, 8 bf16 per lane
  unsigned r0, r1, r2, r3;
};

struct FragB {              // 16x8 bf16, column-major, 4 bf16 per lane
  unsigned r0, r1;
};

struct Acc {                // 16x8 fp32, 4 per lane
  float x0, x1, x2, x3;
};

__device__ __forceinline__ void mma(Acc& d, const FragA& a, const FragB& b) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d.x0), "+f"(d.x1), "+f"(d.x2), "+f"(d.x3)
      : "r"(a.r0), "r"(a.r1), "r"(a.r2), "r"(a.r3), "r"(b.r0), "r"(b.r1));
}

// The m16n8k16 operand layouts, once, so no call site re-derives them.
//   lane = 4 * group + slot, group in [0,8), slot in [0,4)
//   A: rows {group, group+8} x cols {2*slot, 2*slot+1} and the same shifted by 8 columns
//   B: col {group} x rows {2*slot, 2*slot+1} and the same shifted by 8 rows
//   D: rows {group, group+8} x cols {2*slot, 2*slot+1}
__device__ __forceinline__ FragA load_frag_a(const __nv_bfloat16* base, int stride,
                                            int group, int slot) {
  FragA a;
  a.r0 = *reinterpret_cast<const unsigned*>(base + group * stride + 2 * slot);
  a.r1 = *reinterpret_cast<const unsigned*>(base + (group + 8) * stride + 2 * slot);
  a.r2 = *reinterpret_cast<const unsigned*>(base + group * stride + 2 * slot + 8);
  a.r3 = *reinterpret_cast<const unsigned*>(base + (group + 8) * stride + 2 * slot + 8);
  return a;
}

__device__ __forceinline__ FragB load_frag_b(const __nv_bfloat16* base, int stride,
                                            int group, int slot) {
  FragB b;
  b.r0 = *reinterpret_cast<const unsigned*>(base + group * stride + 2 * slot);
  b.r1 = *reinterpret_cast<const unsigned*>(base + group * stride + 2 * slot + 8);
  return b;
}

// Sum over the eight lanes that share one row of m. Three butterfly steps rather than
// five, because a row of c_m = 64 is held by eight lanes at eight elements each.
__device__ __forceinline__ float row_reduce_sum(float v) {
#pragma unroll
  for (int offset = 4; offset > 0; offset >>= 1) {
    v += __shfl_xor_sync(kFullMask, v, offset);
  }
  return v;
}

// ---------------------------------------------------------------------------
// The fused kernel.
//
// One CTA owns a PAIR_SIDE x PAIR_SIDE block of residue pairs and ZT channels of c_z.
// All three matrix products run on the tensor cores: the two c_m -> c_hidden projections,
// and the c_hidden^2 -> c_z contraction. Only the outer product itself stays scalar,
// because its reduction depth is the MSA depth (8) rather than a multiple of the mma K.
//
// LayerNorm and both projections are recomputed here rather than read back from a global
// intermediate: a square pair block needs only 2 * PAIR_SIDE distinct residues, so the
// recompute is small and it buys away an entire launch.
// ---------------------------------------------------------------------------
template <int PAIR_SIDE, int ZT, int S_MAX>
__global__ void __launch_bounds__(kThreads) opm_fused_kernel(
    const __nv_bfloat16* __restrict__ m,
    const __nv_bfloat16* __restrict__ mask,  // nullptr means an all-ones mask
    const __nv_bfloat16* __restrict__ ln_weight,
    const __nv_bfloat16* __restrict__ ln_bias,
    const __nv_bfloat16* __restrict__ w1,
    const __nv_bfloat16* __restrict__ w2,
    const __nv_bfloat16* __restrict__ w_out,
    const __nv_bfloat16* __restrict__ b_out,
    __nv_bfloat16* __restrict__ out,
    const int S, const int R, const int Z, const float ln_eps, const float eps) {
  constexpr int kPairs = PAIR_SIDE * PAIR_SIDE;
  constexpr int kWarps = kThreads / kWarpSize;
  // The contraction's N dimension is the pair index, padded up to whole mma tiles. With
  // PAIR_SIDE = 2 that means four real pairs in a tile of eight; the padding columns are
  // computed and discarded, which is far cheaper than a narrower instruction.
  constexpr int kOuterRows = ((kPairs + kMmaN - 1) / kMmaN) * kMmaN;
  constexpr int kNTiles = kOuterRows / kMmaN;
  constexpr int kZTiles = ZT / kMmaM;
  // The projections' M dimension is (residue, sequence) rows for one side of the pair,
  // padded up to whole mma tiles so a shallow MSA or a small pair block still maps.
  constexpr int kProjRowsUsed = PAIR_SIDE * S_MAX;
  constexpr int kProjRows = ((kProjRowsUsed + kMmaM - 1) / kMmaM) * kMmaM;
  constexpr int kProjMTiles = kProjRows / kMmaM;
  constexpr int kProjNTiles = kC / kMmaN;
  constexpr int kKSteps = kK / kMmaK;

  static_assert(ZT % kMmaM == 0, "c_z tile must be whole mma tiles");
  static_assert(kC % kMmaN == 0, "c_hidden must be whole mma tiles");
  static_assert(kZTiles * kNTiles <= kWarps, "one warp per output tile at least");
  static_assert(kWarps % (kZTiles * kNTiles) == 0,
                "the warps sharing an output tile must divide the CTA evenly");
  static_assert(kKSteps % (kWarps / (kZTiles * kNTiles)) == 0,
                "the contraction K must split evenly across the warps sharing a tile");
  static_assert(kCm == 8 * 8, "one lane holds eight elements of a row of m");

  // Every large buffer lives in one dynamic allocation so the launch needs a single
  // opt-in, issued once when the extension loads.
  extern __shared__ char smem[];
  auto* const w_s = reinterpret_cast<__nv_bfloat16*>(smem);              // [ZT][kKRow]
  auto* const outer_s = w_s + ZT * kKRow;                     // [kOuterRows][kKRow]
  auto* const ln_s = outer_s + kOuterRows * kKRow;            // [2][kProjRows][kCmRow]
  auto* const w1_s = ln_s + 2 * kProjRows * kCmRow;                      // [kC][kCmRow]
  auto* const w2_s = w1_s + kC * kCmRow;                                 // [kC][kCmRow]
  auto* const ab_s = w2_s + kC * kCmRow;                    // [2][PAIR_SIDE][S_MAX][kC]
  auto* const acc_s = reinterpret_cast<float*>(
      ab_s + 2 * PAIR_SIDE * S_MAX * kC);       // [kSplits][kZTiles][kNTiles][kMmaM*kMmaN]

  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid % kWarpSize;
  const int warp = tid / kWarpSize;
  const int group = lane / 4;    // mma fragment row/column group
  const int slot = lane % 4;     // mma fragment element pair within the group

  const int p0 = static_cast<int>(blockIdx.x) * PAIR_SIDE;
  const int q0 = static_cast<int>(blockIdx.y) * PAIR_SIDE;
  const int zblocks = Z / ZT;
  const int z0 = (static_cast<int>(blockIdx.z) % zblocks) * ZT;
  const int batch = static_cast<int>(blockIdx.z) / zblocks;

  const __nv_bfloat16* const m_batch = m + static_cast<int64_t>(batch) * S * R * kCm;
  const __nv_bfloat16* const mask_batch =
      mask == nullptr ? nullptr : mask + static_cast<int64_t>(batch) * S * R;

  // ------------------------------------------------------------------
  // Stage w_out and both projection weights, and clear the buffers whose padding the
  // tensor cores will read. Every staging loop strides over the CTA rather than assuming
  // one element per thread: a tile that made the CTA smaller than a buffer would
  // otherwise leave its tail holding whatever was in shared memory, which is a wrong
  // answer rather than a slow one.
  // ------------------------------------------------------------------
  // The rows of m this warp will normalise, issued before the w_out staging so their
  // latency overlaps it. Every one is cold: the harness flushes L2 before each timed
  // iteration. Held as uint4 because that is one lane's eight elements of a row.
  constexpr int kLanesPerRow = kCm / 8;                     // 8
  constexpr int kRowsPerWarp = kWarpSize / kLanesPerRow;    // 4
  constexpr int kRowPasses =
      (2 * PAIR_SIDE * S_MAX + kWarps * kRowsPerWarp - 1) / (kWarps * kRowsPerWarp);
  const int lane_chunk = (lane % kLanesPerRow) * 8;
  const int row_in_warp = lane / kLanesPerRow;
  const int total_rows = 2 * PAIR_SIDE * S;
  uint4 m_pre[kRowPasses];
#pragma unroll
  for (int pass = 0; pass < kRowPasses; ++pass) {
    const int task = (warp + pass * kWarps) * kRowsPerWarp + row_in_warp;
    if (task < total_rows) {
      const int side = task / (PAIR_SIDE * S);
      const int rem = task - side * (PAIR_SIDE * S);
      const int r = (side == 0 ? p0 : q0) + rem / S;
      m_pre[pass] = *reinterpret_cast<const uint4*>(
          m_batch + (static_cast<int64_t>(rem % S) * R + r) * kCm + lane_chunk);
    }
  }

  {
    // The whole c_z slice of w_out in one pass: the harness flushes L2 before every timed
    // iteration, so this is the one cold read in the kernel and it is worth paying its
    // latency exactly once. Global reads are 16 bytes and fully coalesced along k.
    constexpr int kVecsPerRow = kK / 8;
    for (int idx = tid; idx < ZT * kVecsPerRow; idx += kThreads) {
      const int zz = idx / kVecsPerRow;
      const int kk = (idx % kVecsPerRow) * 8;
#if FK_OPM_STAGE
      cp_async_16(w_s + zz * kKRow + kk,
                  w_out + static_cast<int64_t>(z0 + zz) * kK + kk);
#else
      *reinterpret_cast<uint4*>(w_s + zz * kKRow + kk) =
          *reinterpret_cast<const uint4*>(
              w_out + static_cast<int64_t>(z0 + zz) * kK + kk);
#endif
    }
    for (int idx = tid; idx < kC * (kCm / 8); idx += kThreads) {
      const int c = idx / (kCm / 8);
      const int j = (idx % (kCm / 8)) * 8;
      *reinterpret_cast<uint4*>(w1_s + c * kCmRow + j) =
          *reinterpret_cast<const uint4*>(w1 + c * kCm + j);
      *reinterpret_cast<uint4*>(w2_s + c * kCmRow + j) =
          *reinterpret_cast<const uint4*>(w2 + c * kCm + j);
    }
    // The pair padding of outer_s, and the sequence rows of ln_s beyond S, are read by
    // mma but never written with data. Zeroing them keeps their discarded outputs finite.
    const uint4 zero = make_uint4(0u, 0u, 0u, 0u);
    for (int idx = tid; idx < (kOuterRows - kPairs) * (kK / 8); idx += kThreads) {
      const int row = kPairs + idx / (kK / 8);
      *reinterpret_cast<uint4*>(outer_s + row * kKRow + (idx % (kK / 8)) * 8) = zero;
    }
    for (int idx = tid; idx < 2 * kProjRows * (kCm / 8); idx += kThreads) {
      const int row = idx / (kCm / 8);
      *reinterpret_cast<uint4*>(ln_s + row * kCmRow + (idx % (kCm / 8)) * 8) = zero;
    }
  }
#if FK_OPM_STAGE
  // Commit the group and leave it outstanding. Nothing below touches w_s until the
  // contraction, so the transfer runs underneath the LayerNorm, both projections and the
  // outer-product build; the matching wait sits immediately before the first fragment
  // load. This barrier is for the *ordinary* shared stores above -- the projection
  // weights and the cleared padding -- which the LayerNorm and projections do read.
  cp_async_commit();
#if !FK_OPM_OVERLAP
  cp_async_wait_all();
#endif
#endif
  __syncthreads();

  // ------------------------------------------------------------------
  // LayerNorm, eight lanes per row so a warp normalises four rows at once and their
  // butterfly reductions overlap instead of queueing. Doing one row at a time was 2 us of
  // pure exposed shuffle latency at two warps per scheduler.
  // ------------------------------------------------------------------
  {
    const int chunk = lane_chunk;
    const uint4 gamma = *reinterpret_cast<const uint4*>(ln_weight + chunk);
    const uint4 beta = *reinterpret_cast<const uint4*>(ln_bias + chunk);

#pragma unroll
    for (int pass = 0; pass < kRowPasses; ++pass) {
      const int task = (warp + pass * kWarps) * kRowsPerWarp + row_in_warp;
      if (task >= total_rows) {
        break;
      }
      const int side = task / (PAIR_SIDE * S);
      const int rem = task - side * (PAIR_SIDE * S);
      const int i = rem / S;
      const int s = rem - i * S;

      const uint4 raw = m_pre[pass];
      const unsigned* const rp = reinterpret_cast<const unsigned*>(&raw);
      float x[8];
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        x[2 * e] = lo_to_f32(rp[e]);
        x[2 * e + 1] = hi_to_f32(rp[e]);
      }
      float local = 0.0f;
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        local += x[e];
      }
      const float mean = row_reduce_sum(local) * (1.0f / kCm);
      float dev = 0.0f;
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        x[e] -= mean;
        dev = fmaf(x[e], x[e], dev);
      }
      const float rstd = rsqrtf(row_reduce_sum(dev) * (1.0f / kCm) + ln_eps);

      const unsigned* const gp = reinterpret_cast<const unsigned*>(&gamma);
      const unsigned* const bp = reinterpret_cast<const unsigned*>(&beta);
      uint4 packed;
      unsigned* const pk = reinterpret_cast<unsigned*>(&packed);
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        pk[e] = pack_bf16x2(fmaf(x[2 * e] * rstd, lo_to_f32(gp[e]), lo_to_f32(bp[e])),
                            fmaf(x[2 * e + 1] * rstd, hi_to_f32(gp[e]), hi_to_f32(bp[e])));
      }
      *reinterpret_cast<uint4*>(
          ln_s + (side * kProjRows + i * S_MAX + s) * kCmRow + chunk) = packed;
    }
  }
  __syncthreads();

  // ------------------------------------------------------------------
  // Both projections on the tensor cores: ln[rows][c_m] times w[c_hidden][c_m] transposed.
  // One warp owns one (side, row tile, channel tile), so each warp reduces the whole c_m
  // itself and no cross-warp reduction is needed.
  // ------------------------------------------------------------------
  constexpr int kProjTiles = 2 * kProjMTiles * kProjNTiles;
#pragma unroll
  for (int t = warp; t < kProjTiles; t += kWarps) {
    const int side = t / (kProjMTiles * kProjNTiles);
    const int rest = t % (kProjMTiles * kProjNTiles);
    const int mt = rest / kProjNTiles;
    const int nt = rest % kProjNTiles;
    const __nv_bfloat16* const a_base = ln_s + (side * kProjRows + mt * kMmaM) * kCmRow;
    const __nv_bfloat16* const b_base =
        (side == 0 ? w1_s : w2_s) + nt * kMmaN * kCmRow;

    Acc acc = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int k = 0; k < kCm; k += kMmaK) {
      // B is w[c][j] read as the transpose: column n = c, row k = j.
      mma(acc, load_frag_a(a_base + k, kCmRow, group, slot),
          load_frag_b(b_base + k, kCmRow, group, slot));
    }

    // The baseline rounds the projection to bf16 and only then multiplies by the mask,
    // rounding again. Both roundings are reproduced. The accumulator holds rows
    // {group, group+8} and columns {2*slot, 2*slot+1} of this tile.
    const int c0 = nt * kMmaN + 2 * slot;
    const float vals[4] = {acc.x0, acc.x1, acc.x2, acc.x3};
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int row = mt * kMmaM + group + 8 * half;
      const int i = row / S_MAX;
      const int s = row - i * S_MAX;
      if (s >= S) {
        continue;
      }
      const int r = (side == 0 ? p0 : q0) + i;
      const float mv = mask_batch == nullptr
                           ? 1.0f
                           : bf16_to_f32(mask_batch[static_cast<int64_t>(s) * R + r]);
      __nv_bfloat16* const dst =
          ab_s + ((side * PAIR_SIDE + i) * S_MAX + s) * kC + c0;
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const float rounded = bf16_to_f32(__float2bfloat16_rn(vals[2 * half + e]));
        dst[e] = __float2bfloat16_rn(rounded * mv);
      }
    }
  }
  __syncthreads();

  // ------------------------------------------------------------------
  // The outer product, rounded to bf16 as the baseline's einsum is. Its reduction depth is
  // the MSA depth, not a multiple of the mma K, so this stays scalar; each thread owns an
  // 8x8 block of (c, e) for one pair, which is the blocking that keeps its shared traffic
  // at one 16-byte load per eight multiply-adds.
  // ------------------------------------------------------------------
  {
    constexpr int kBlk = 8;
    constexpr int kBlocksPerPair = (kC / kBlk) * (kC / kBlk);
    constexpr int kBlocksPerThread = (kPairs * kBlocksPerPair + kThreads - 1) / kThreads;
    for (int b = 0; b < kBlocksPerThread; ++b) {
      const int idx = b * kThreads + tid;
      if (idx >= kPairs * kBlocksPerPair) {
        break;
      }
      const int pair = idx / kBlocksPerPair;
      const int blk = idx % kBlocksPerPair;
      const int c0 = (blk / (kC / kBlk)) * kBlk;
      const int e0 = (blk % (kC / kBlk)) * kBlk;
      const int i = pair / PAIR_SIDE;
      const int j = pair % PAIR_SIDE;

      float acc[kBlk][kBlk];
#pragma unroll
      for (int u = 0; u < kBlk; ++u) {
#pragma unroll
        for (int v = 0; v < kBlk; ++v) {
          acc[u][v] = 0.0f;
        }
      }
      for (int s = 0; s < S; ++s) {
        const uint4 av = *reinterpret_cast<const uint4*>(
            ab_s + ((0 * PAIR_SIDE + i) * S_MAX + s) * kC + c0);
        const uint4 bv = *reinterpret_cast<const uint4*>(
            ab_s + ((1 * PAIR_SIDE + j) * S_MAX + s) * kC + e0);
        const unsigned* const ap = reinterpret_cast<const unsigned*>(&av);
        const unsigned* const bp = reinterpret_cast<const unsigned*>(&bv);
#pragma unroll
        for (int u = 0; u < kBlk; ++u) {
          const float a = (u & 1) ? hi_to_f32(ap[u / 2]) : lo_to_f32(ap[u / 2]);
#pragma unroll
          for (int v = 0; v < kBlk; ++v) {
            const float bb = (v & 1) ? hi_to_f32(bp[v / 2]) : lo_to_f32(bp[v / 2]);
            acc[u][v] = fmaf(a, bb, acc[u][v]);
          }
        }
      }
#pragma unroll
      for (int u = 0; u < kBlk; ++u) {
        uint4 packed;
        unsigned* const pk = reinterpret_cast<unsigned*>(&packed);
#pragma unroll
        for (int v = 0; v < 4; ++v) {
          pk[v] = pack_bf16x2(acc[u][2 * v], acc[u][2 * v + 1]);
        }
        *reinterpret_cast<uint4*>(outer_s + pair * kKRow + (c0 + u) * kC + e0) = packed;
      }
    }
  }
  __syncthreads();

  // ------------------------------------------------------------------
  // The linear_out contraction on the tensor cores: out[z][pair] = sum_k w_out[z][k] *
  // outer[pair][k]. A is the staged w_out tile, B is outer read as its transpose. The
  // contraction K is split across the warps that share an output tile and the partials are
  // summed in the epilogue.
  // ------------------------------------------------------------------
  constexpr int kSplits = kWarps / (kZTiles * kNTiles);
#if FK_OPM_STAGE && FK_OPM_OVERLAP
  // The first read of w_s in the whole kernel is the fragment load below, so this is where
  // the transfer has to have landed -- and everything between the issue and here has been
  // free to run while it was in flight.
  cp_async_wait_all();
  __syncthreads();
#endif
  {
    const int tile = warp % (kZTiles * kNTiles);
    const int split = warp / (kZTiles * kNTiles);
    const int zt = tile / kNTiles;
    const int nt = tile % kNTiles;
    const __nv_bfloat16* const a_base = w_s + zt * kMmaM * kKRow;
    const __nv_bfloat16* const b_base = outer_s + nt * kMmaN * kKRow;

    Acc acc = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll 4
    for (int step = split; step < kKSteps; step += kSplits) {
      const int k = step * kMmaK;
      mma(acc, load_frag_a(a_base + k, kKRow, group, slot),
          load_frag_b(b_base + k, kKRow, group, slot));
    }
    float* const dst = acc_s + ((split * kZTiles + zt) * kNTiles + nt) * kMmaM * kMmaN;
    dst[(group) * kMmaN + 2 * slot] = acc.x0;
    dst[(group) * kMmaN + 2 * slot + 1] = acc.x1;
    dst[(group + 8) * kMmaN + 2 * slot] = acc.x2;
    dst[(group + 8) * kMmaN + 2 * slot + 1] = acc.x3;
  }
  __syncthreads();

  // ------------------------------------------------------------------
  // Epilogue. norm is formed in registers straight from mask -- 256 bytes, with nothing to
  // gain from precomputing it -- and rounded to bf16 twice, once for the einsum and once
  // after the eps add, which is what the baseline does.
  // ------------------------------------------------------------------
  for (int o = tid; o < kPairs * ZT; o += kThreads) {
    const int pair = o / ZT;
    const int zl = o % ZT;
    const int p = p0 + pair / PAIR_SIDE;
    const int q = q0 + pair % PAIR_SIDE;
    const int z = z0 + zl;

    float linear = bf16_to_f32(b_out[z]);
#pragma unroll
    for (int split = 0; split < kSplits; ++split) {
      linear += acc_s[((split * kZTiles + zl / kMmaM) * kNTiles + pair / kMmaN)
                          * kMmaM * kMmaN
                      + (zl % kMmaM) * kMmaN + pair % kMmaN];
    }
    const float rounded = bf16_to_f32(__float2bfloat16_rn(linear));

    float co_valid;
    if (mask_batch == nullptr) {
      co_valid = static_cast<float>(S);
    } else {
      co_valid = 0.0f;
      for (int s = 0; s < S; ++s) {
        const int64_t base = static_cast<int64_t>(s) * R;
        co_valid = fmaf(bf16_to_f32(mask_batch[base + p]),
                        bf16_to_f32(mask_batch[base + q]), co_valid);
      }
    }
    const float norm = bf16_to_f32(
        __float2bfloat16_rn(bf16_to_f32(__float2bfloat16_rn(co_valid)) + eps));

    out[((static_cast<int64_t>(batch) * R + p) * R + q) * Z + z] =
        __float2bfloat16_rn(rounded / norm);
  }
}

// The dynamic shared-memory footprint of one CTA, so the host can size the opt-in.
template <int PAIR_SIDE, int ZT, int S_MAX>
constexpr size_t opm_smem_bytes() {
  constexpr int kPairs = PAIR_SIDE * PAIR_SIDE;
  constexpr int kOuterRows = ((kPairs + kMmaN - 1) / kMmaN) * kMmaN;
  constexpr int kZTiles = ZT / kMmaM;
  constexpr int kNTiles = kOuterRows / kMmaN;
  constexpr int kSplits = (kThreads / kWarpSize) / (kZTiles * kNTiles);
  constexpr int kProjRowsUsed = PAIR_SIDE * S_MAX;
  constexpr int kProjRows = ((kProjRowsUsed + kMmaM - 1) / kMmaM) * kMmaM;
  return sizeof(__nv_bfloat16) *
             (static_cast<size_t>(ZT) * kKRow + static_cast<size_t>(kOuterRows) * kKRow +
              2u * kProjRows * kCmRow + 2u * kC * kCmRow +
              2u * PAIR_SIDE * S_MAX * kC) +
         sizeof(float) * static_cast<size_t>(kSplits) * kZTiles * kNTiles * kMmaM * kMmaN;
}

// FK_OPM_DEVICE_END

// ---------------------------------------------------------------------------
// Eligibility. Every predicate guards something the kernel relies on; nothing here
// dereferences device memory or synchronises.
// ---------------------------------------------------------------------------
constexpr int kPairSide = FK_OPM_PAIR_SIDE;
constexpr int kZT = FK_OPM_ZT;
constexpr int kSMax = FK_OPM_S_MAX;

using FusedKernel = void (*)(const __nv_bfloat16*, const __nv_bfloat16*,
                             const __nv_bfloat16*, const __nv_bfloat16*,
                             const __nv_bfloat16*, const __nv_bfloat16*,
                             const __nv_bfloat16*, const __nv_bfloat16*,
                             __nv_bfloat16*, int, int, int, float, float);
constexpr FusedKernel kFusedKernel = &opm_fused_kernel<kPairSide, kZT, kSMax>;
constexpr size_t kSmemBytes = opm_smem_bytes<kPairSide, kZT, kSMax>();

// The shared footprint is well above the 48 KiB a block gets by default, so the kernel
// needs the large-dynamic-shared-memory opt-in. It is issued exactly once, from a
// namespace-scope initializer that runs when the .so loads -- never from the operator and
// never from ``forward``, so no per-call driver call is added. A failure here leaves
// ``kSmemReady`` false and every call takes the exact ATen fallback.
const bool kSmemReady = [] {
  return cudaFuncSetAttribute(reinterpret_cast<const void*>(kFusedKernel),
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              static_cast<int>(kSmemBytes)) == cudaSuccess;
}();

inline bool is_aligned16(const void* p) {
  return (reinterpret_cast<uintptr_t>(p) & 15u) == 0;
}

inline bool weight_ok(const at::Tensor& t, const at::Tensor& ref, int64_t rows,
                      int64_t cols) {
  return t.defined() && t.device() == ref.device() &&
         t.scalar_type() == at::kBFloat16 && t.dim() == 2 && t.size(0) == rows &&
         t.size(1) == cols && t.is_contiguous() && is_aligned16(t.const_data_ptr());
}

inline bool vector_ok(const at::Tensor& t, const at::Tensor& ref, int64_t n) {
  return t.defined() && t.device() == ref.device() &&
         t.scalar_type() == at::kBFloat16 && t.dim() == 1 && t.size(0) == n &&
         t.is_contiguous() && is_aligned16(t.const_data_ptr());
}

bool fast_path_ok(const at::Tensor& m, const at::Tensor& mask,
                  const at::Tensor& ln_weight, const at::Tensor& ln_bias,
                  const at::Tensor& w1, const at::Tensor& w2,
                  const at::Tensor& w_out, const at::Tensor& b_out, int64_t c_m,
                  int64_t c_z, int64_t c_hidden) {
  // The shared-memory opt-in is a precondition of the launch, not of the maths.
  if (!kSmemReady) {
    return false;
  }
  if (!m.defined() || !m.is_cuda() || m.scalar_type() != at::kBFloat16) {
    return false;
  }
  // Functorch duals, functional/fake tensors and Python subclasses have no ordinary
  // storage to take a pointer to, and can arrive without ever setting requires_grad.
  if (at::isTensorSubclassLike(m) || at::isTensorSubclassLike(mask) ||
      at::isTensorSubclassLike(ln_weight) || at::isTensorSubclassLike(ln_bias) ||
      at::isTensorSubclassLike(w1) || at::isTensorSubclassLike(w2) ||
      at::isTensorSubclassLike(w_out) || at::isTensorSubclassLike(b_out)) {
    return false;
  }
  // Autocast rewrites the dtype of the ATen ops the fallback is built from; the
  // fallback re-dispatches through them and so picks up that policy exactly, which a
  // raw kernel writing bf16 would not.
  if (at::autocast::is_autocast_enabled(m.device().type())) {
    return false;
  }
  // The projection geometry the kernel is specialised to.
  if (c_m != kCm || c_hidden != kC) {
    return false;
  }
  // [batch, seq, res, c_m] exactly: other leading ranks are a different index
  // calculation, not a slower one.
  if (m.dim() != 4 || m.size(3) != kCm || !m.is_contiguous() || m.numel() == 0) {
    return false;
  }
  const int64_t S = m.size(1);
  const int64_t R = m.size(2);
  if (S <= 0 || S > kSMax || R <= 0 || R % kPairSide != 0) {
    return false;
  }
  if (c_z <= 0 || c_z % kZT != 0) {
    return false;
  }
  if (!is_aligned16(m.const_data_ptr())) {
    return false;
  }
  if (!vector_ok(ln_weight, m, kCm) || !vector_ok(ln_bias, m, kCm)) {
    return false;
  }
  if (!weight_ok(w1, m, kC, kCm) || !weight_ok(w2, m, kC, kCm)) {
    return false;
  }
  // linear_out consumes the flattened c_hidden^2 outer product.
  if (!weight_ok(w_out, m, c_z, kK) || !vector_ok(b_out, m, c_z)) {
    return false;
  }
  if (mask.defined()) {
    if (mask.device() != m.device() || mask.scalar_type() != at::kBFloat16 ||
        mask.dim() != 3 || mask.size(0) != m.size(0) || mask.size(1) != S ||
        mask.size(2) != R || !mask.is_contiguous()) {
      return false;
    }
  }
  // The grid is R/PAIR_SIDE by R/PAIR_SIDE by (c_z/ZT * batch); keep every extent
  // inside a 32-bit launch configuration.
  const int64_t pair_blocks = R / kPairSide;
  if (pair_blocks > INT32_MAX || (c_z / kZT) * m.size(0) > INT32_MAX ||
      m.numel() > INT32_MAX) {
    return false;
  }
  return true;
}

at::Tensor run_fused(const at::Tensor& m, const at::Tensor& mask,
                     const at::Tensor& ln_weight, const at::Tensor& ln_bias,
                     const at::Tensor& w1, const at::Tensor& w2,
                     const at::Tensor& w_out, const at::Tensor& b_out, int64_t c_z,
                     double eps, double ln_eps) {
  const c10::cuda::CUDAGuard device_guard(m.device());
  const int64_t B = m.size(0);
  const int64_t S = m.size(1);
  const int64_t R = m.size(2);
  at::Tensor out = at::empty({B, R, R, c_z}, m.options());

  const int64_t pair_blocks = R / kPairSide;
  // Two grid dimensions for the pair block, so the kernel reads p0 and q0 directly
  // instead of dividing a flat index by a runtime extent.
  const dim3 grid(static_cast<unsigned>(pair_blocks),
                  static_cast<unsigned>(pair_blocks),
                  static_cast<unsigned>((c_z / kZT) * B));
  const dim3 block(kThreads);
  const auto stream = c10::cuda::getCurrentCUDAStream();

  kFusedKernel<<<grid, block, kSmemBytes, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(m.const_data_ptr()),
      mask.defined() ? reinterpret_cast<const __nv_bfloat16*>(mask.const_data_ptr())
                     : nullptr,
      reinterpret_cast<const __nv_bfloat16*>(ln_weight.const_data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(ln_bias.const_data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(w1.const_data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(w2.const_data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(w_out.const_data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(b_out.const_data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(out.mutable_data_ptr()),
      static_cast<int>(S), static_cast<int>(R), static_cast<int>(c_z),
      static_cast<float>(ln_eps), static_cast<float>(eps));
  return out;
}

// Exactly what ``baseline.py`` computes, op for op -- an equality, not a tolerance
// argument. This is one dispatch but several launches; it is the answer for every
// input the fused kernel does not admit, and the autograd formula for all of them,
// since the operator is registered CompositeImplicitAutograd.
at::Tensor baseline_formula(const at::Tensor& m, const at::Tensor& mask,
                           const at::Tensor& ln_weight, const at::Tensor& ln_bias,
                           const at::Tensor& w1, const at::Tensor& w2,
                           const at::Tensor& w_out, const at::Tensor& b_out,
                           int64_t c_m, double eps, double ln_eps) {
  at::Tensor msk = mask;
  if (!msk.defined()) {
    msk = at::ones(m.sizes().slice(0, m.dim() - 1), m.options());
  }
  const at::ScalarType orig = m.scalar_type();
  const std::optional<at::Tensor> gamma =
      ln_weight.defined() && ln_weight.scalar_type() != at::kFloat
          ? std::optional<at::Tensor>(ln_weight.to(at::kFloat))
          : (ln_weight.defined() ? std::optional<at::Tensor>(ln_weight)
                                 : std::nullopt);
  const std::optional<at::Tensor> beta =
      ln_bias.defined() && ln_bias.scalar_type() != at::kFloat
          ? std::optional<at::Tensor>(ln_bias.to(at::kFloat))
          : (ln_bias.defined() ? std::optional<at::Tensor>(ln_bias) : std::nullopt);
  at::Tensor ln =
      at::layer_norm(m.to(at::kFloat), {c_m}, gamma, beta, ln_eps).to(orig);

  const at::Tensor msk_u = msk.unsqueeze(-1);
  at::Tensor a = at::linear(ln, w1) * msk_u;
  at::Tensor b = at::linear(ln, w2) * msk_u;
  ln.reset();

  a = a.transpose(-2, -3);
  b = b.transpose(-2, -3);

  at::Tensor outer = at::einsum("...bac,...dae->...bdce", {a, b});
  std::vector<int64_t> flat(outer.sizes().begin(), outer.sizes().end() - 2);
  flat.push_back(-1);
  outer = at::linear(outer.reshape(flat), w_out, b_out);

  // The scalar add stays a scalar add: bf16 + a double Scalar keeps the tensor's dtype
  // and computes in fp32 opmath, so norm is rounded to bf16 after eps, not before.
  const at::Tensor norm =
      at::add(at::einsum("...abc,...adc->...bdc", {msk_u, msk_u}), eps);
  return outer / norm;
}

at::Tensor outer_product_mean(const at::Tensor& m,
                              const std::optional<at::Tensor>& mask,
                              const at::Tensor& ln_weight,
                              const at::Tensor& ln_bias, const at::Tensor& w1,
                              const at::Tensor& w2, const at::Tensor& w_out,
                              const at::Tensor& b_out, int64_t c_m, int64_t c_z,
                              int64_t c_hidden, double eps, double ln_eps) {
  const at::Tensor msk = mask.has_value() ? *mask : at::Tensor();
  // The fused path allocates with at::empty and launches a raw kernel, so it records
  // nothing for autograd. Grad mode being *enabled* is the test, not whether some
  // tensor currently requires grad: a caller who has not entered no_grad may attach
  // requires_grad later in the same graph, or replay this forward under checkpointing
  // with different requires_grad state.
  if (at::GradMode::is_enabled() ||
      !fast_path_ok(m, msk, ln_weight, ln_bias, w1, w2, w_out, b_out, c_m, c_z,
                    c_hidden)) {
    return baseline_formula(m, msk, ln_weight, ln_bias, w1, w2, w_out, b_out, c_m,
                            eps, ln_eps);
  }
  return run_fused(m, msk, ln_weight, ln_bias, w1, w2, w_out, b_out, c_z, eps,
                   ln_eps);
}

}  // namespace

// Schema only, with no dispatch-key m.impl: that registers the operator as
// CompositeImplicitAutograd, so autograd decomposes a grad-enabled call through the
// ATen ops of the fallback above instead of finding no formula and silently detaching.
TORCH_LIBRARY(fk_af3_opm_cand, m) {
  m.def(
      "outer_product_mean(Tensor m, Tensor? mask, Tensor ln_weight, Tensor ln_bias, "
      "Tensor w1, Tensor w2, Tensor w_out, Tensor b_out, int c_m, int c_z, "
      "int c_hidden, float eps, float ln_eps) -> Tensor",
      &outer_product_mean);
}
"""


def _load_fused_op():
    """Build and register the fused operator, returning its callable.

    Compilation happens here, at import, so nothing is deferred into a timed
    ``forward`` and no thread appears between the harness's ``active_count``
    snapshots. The includes are lean deliberately: ``<torch/extension.h>`` through
    nvcc dominates the build, and ``TORCH_LIBRARY`` needs none of it.
    """
    import os

    from torch.utils.cpp_extension import load_inline

    # cpp_extension otherwise honours the ambient TORCH_CUDA_ARCH_LIST, which in this
    # environment names several architectures -- one nvcc pass each, for targets that
    # will never run this kernel. Narrowing it to the device actually present is what
    # keeps the cold build comfortably inside the harness's wall-clock cap. Derived
    # from the live device so it cannot name the wrong arch, and restored afterwards so
    # no later build in this process is affected.
    previous_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
    try:
        load_inline(
            name=_LIBRARY_NAME,
            cpp_sources="",
            cuda_sources=_CUDA_SOURCE,
            extra_cuda_cflags=[
                "-O3",
                f"-DFK_OPM_PAIR_SIDE={_PAIR_SIDE}",
                f"-DFK_OPM_ZT={_ZT}",
                f"-DFK_OPM_S_MAX={_S_MAX}",
                f"-DFK_OPM_STAGE={_STAGE}",
                f"-DFK_OPM_OVERLAP={_OVERLAP}",
                f"-DFK_OPM_THREADS={_THREADS}",
            ],
            is_python_module=False,
            no_implicit_headers=True,
        )
    finally:
        if previous_arch_list is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = previous_arch_list
    # Bind the overload, not the packet: the packet re-resolves overloads from the
    # argument types on every call, and this forward is launch-latency bound.
    return getattr(torch.ops, _LIBRARY_NAME).outer_product_mean.default


try:
    _fused_outer_product_mean = _load_fused_op()
except Exception:  # noqa: BLE001 - a build that cannot happen must degrade rather than
    # take the module down with it: an import failure would cost every case at once.
    _fused_outer_product_mean = None


class _AffineNormParams(nn.Module):
    """The affine parameters of the baseline's LayerNorm, under its two keys."""

    def __init__(self, normalized_shape: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))


class _ProjectionParams(nn.Module):
    """The parameters of one of the baseline's Linear submodules.

    ``torch.empty`` matches the baseline's own initialization, which the harness
    detects as uninitialized and replaces with ``N(0, 0.02)`` on both modules before
    sharing weights. Nothing is derived from either tensor here: the harness casts the
    module to bf16 and only *then* loads the state dict, so a value computed in
    ``__init__`` would be stale -- which is also why the kernel takes ``w1`` and ``w2``
    from separate pointers instead of a concatenation.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None


class OuterProductMean(nn.Module):
    """AF3 Algorithm 9: Outer product mean.

    Args:
        c_m: MSA embedding channel dimension
        c_z: Pair embedding channel dimension
        c_hidden: Hidden channel dimension
        eps: Epsilon for numerical stability
    """

    def __init__(self, c_m: int, c_z: int, c_hidden: int, eps: float = 1e-3):
        super().__init__()
        self.c_m = c_m
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.eps = eps

        self.layer_norm = _AffineNormParams(c_m)
        self.linear_1 = _ProjectionParams(c_m, c_hidden, bias=False)
        self.linear_2 = _ProjectionParams(c_m, c_hidden, bias=False)
        self.linear_out = _ProjectionParams(c_hidden ** 2, c_z, bias=True)

    def forward(
        self,
        m: torch.Tensor,
        mask: torch.Tensor | None = None,
        chunk_size: int | None = None,
        inplace_safe: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            m:    [*, N_seq, N_res, C_m] MSA embedding
            mask: [*, N_seq, N_res] MSA mask

        Returns:
            [*, N_res, N_res, C_z] pair embedding update
        """
        if _fused_outer_product_mean is not None:
            # One dispatch. Eligibility, the grad guard and the exact fallback all live
            # inside the operator, so nothing here inspects a tensor's metadata.
            return _fused_outer_product_mean(
                m, mask,
                self.layer_norm.weight, self.layer_norm.bias,
                self.linear_1.weight, self.linear_2.weight,
                self.linear_out.weight, self.linear_out.bias,
                self.c_m, self.c_z, self.c_hidden, self.eps, _LN_EPS,
            )
        return self._baseline_formula(m, mask)

    def _baseline_formula(
        self, m: torch.Tensor, mask: torch.Tensor | None
    ) -> torch.Tensor:
        """The baseline's own op chain, for a process where the build did not happen."""
        if mask is None:
            mask = m.new_ones(m.shape[:-1])
        ln = F.layer_norm(
            m.float(), (self.c_m,),
            self.layer_norm.weight.float(), self.layer_norm.bias.float(), _LN_EPS,
        ).to(m.dtype)
        mask = mask.unsqueeze(-1)
        a = F.linear(ln, self.linear_1.weight) * mask
        b = F.linear(ln, self.linear_2.weight) * mask
        del ln
        a = a.transpose(-2, -3)
        b = b.transpose(-2, -3)
        outer = torch.einsum("...bac,...dae->...bdce", a, b)
        outer = outer.reshape(outer.shape[:-2] + (-1,))
        outer = F.linear(outer, self.linear_out.weight, self.linear_out.bias)
        norm = torch.einsum("...abc,...adc->...bdc", mask, mask)
        norm = norm + self.eps
        return outer / norm
