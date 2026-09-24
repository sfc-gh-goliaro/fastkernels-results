// Device code for the fused AlphaFold3 SwiGLU transition kernels (B200, sm_100).
//
// Split out of the extension translation unit so it depends on nothing but the
// CUDA bf16 headers. That lets a standalone nvcc harness compile the exact same
// kernels with -lineinfo for Nsight Compute, which is the only way to get
// per-source-line stall attribution out of a load_inline extension. See
// profile/p1-tiles/ and profile/p1-ncu-*/ for the harnesses that use it.
//
// Two operators are registered by the including translation unit, one per class,
// and between them they use three kernels:
//
//   adaln       CTB stage 1: LayerNorm(s) and LayerNorm(a) folded in, then
//               a1 = sigmoid(sn@Wag^T + bag) * (an + sn@Was^T)
//   up_swiglu   shared: hh = silu(A@Wa^T) * (A@Wb^T), with LayerNorm(x) folded
//               in for SwiGLUTransition and A = a1 for CTB
//   down        shared: out = (hh@Wout^T) * mask, and for CTB the outer gate
//               sigmoid(s@Wg^T + bg) folded into the same epilogue
//
// So SwiGLUTransition costs two launches and ConditionedTransitionBlock three,
// which is the budget the measured cost model allows (~4 us per launch inside
// one op call, against a ~15 us window floor).
//
// Two decisions here came from measurement rather than from the shape of the
// problem, and both are worth stating because the obvious alternatives are worse.
//
// *Operands come straight from global into registers.* There is no shared-memory
// staging of A or B and no __syncthreads inside a K loop. An earlier version
// staged the A tile, which turned the K=1536 down-projection into 24 serialized
// stage/sync iterations and cost 48 us of GPU time for a kernel that moves 1.2 MB.
// Issuing the loads straight from the K loop lets the compiler keep many in
// flight. The LayerNorm prologue survives the change because normalising
// in-register needs only the row's mean and rstd.
//
// *Warps split K, not N.* One CTA owns one n8 output tile, and its warps divide
// the reduction extent between them and combine their fp32 partials through
// shared memory at the end. This is the opposite of the usual arrangement, and
// the reason is that these shapes have no other parallelism left. Nsight Compute
// on the c_a=768 case (profile/ctb-c768-adaln-down-v1/REPORT.md) measured the
// N-splitting version at 1.53% and 1.56% achieved occupancy -- 0.98 and 1.00
// active warps per SM -- with 84.6% and 95.6% of scheduler slots finding no
// eligible warp and ~70-79% of the stall cycles on `long_scoreboard`, against
// 1-2% compute throughput and under 1.2% DRAM throughput. The kernels were
// latency-bound with nothing to switch to. With M=16 there is exactly one BM tile
// and N/8 is only 96 or 192 CTAs, so K was the sole remaining axis: splitting it
// sixteen ways multiplies resident warps by sixteen at constant CTA count. It needs
// one barrier per accumulator group -- so two in each of the dual-accumulator
// kernels -- and no semaphore, atomic, or grid-wide synchronisation. Measured on the
// delivered code (profile/final-ksplit-ncu/REPORT.md): achieved occupancy 1.53% ->
// 20.38% for adaln and 1.56% -> 20.28% for the gated down-projection, with duration
// more than halved in both.
//
// The partial sums make the reduction tree wider than a single warp's would be.
// That is still an fp32 tree, so it stays inside the accumulation-order argument
// the numerics rest on -- it is not a precision change.
//
// The GEMM shape is the same in all three: small M, K contiguous in both
// operands, one or two B operands, fused prologue and epilogue. One warp-level
// primitive covers it, in two interchangeable forms selected per call --
// `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`, and an explicit fp32
// fused-multiply-add inner product over the *same* accumulator layout. The second
// exists to be the numerics oracle for the first: it is the fp32 identity the
// correctness argument claims, differing from the mma path only in reduction
// order, so a disagreement between them localises a fragment-mapping error
// rather than leaving it hidden inside the harness tolerance.

#pragma once

#include <cuda_bf16.h>

#include <cstdint>

namespace fk_af3 {


constexpr int kWarpSize = 32;
constexpr unsigned kFullMask = 0xffffffffu;

// M tile. 16 is one mma m-tile and the expectation is that it wins even at
// M = 368: with BM=16 and BN=32 the up-projection grid is 184 CTAs against 148
// SMs, where BM=64 would drop it to 48. A larger M tile removes parallelism
// exactly where the grid is already starved, and nothing here is math-bound. It
// is a knob so the claim is tested rather than asserted.
#ifndef FK_AF3_BM
#define FK_AF3_BM 16
#endif
constexpr int kBM = FK_AF3_BM;
constexpr int kMTiles = kBM / 16;
static_assert(kBM % 16 == 0, "M tiles are whole m16 mma tiles");

// Cap on rows per call, mirrored on the Python side.
constexpr int kMaxRows = 4096;

// ---------------------------------------------------------------------------
// Scalar helpers.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float bf2f(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ __nv_bfloat16 f2bf(float v) { return __float2bfloat16_rn(v); }

// One bf16 rounding, value returned in fp32. Every call site is a point at which
// the baseline materialises a bf16 tensor; between two of them everything stays
// fp32. Making the rounding explicit rather than implicit in a store is what lets
// the epilogues chain several boundaries without a round trip to memory.
__device__ __forceinline__ float round_bf16(float v) { return bf2f(f2bf(v)); }

// expf, not __expf. ATen computes sigmoid and SiLU in fp32 opmath for bf16 inputs
// using the accurate libdevice call; the fast intrinsic would be a second,
// avoidable difference on top of the reduction-order one. (The build also passes
// no -use_fast_math, which would substitute it silently.)
__device__ __forceinline__ float sigmoid_f32(float x) {
  return 1.0f / (1.0f + expf(-x));
}
__device__ __forceinline__ float silu_f32(float x) { return x * sigmoid_f32(x); }

// Butterfly sum across a group of `kWidth` lanes (a power of two, <= 32, aligned
// inside the warp). Leaves the total in every lane of the group, so no broadcast
// step is needed. All 32 lanes must reach it, which is why the callers contribute
// zeros for out-of-range rows instead of returning early: a warp holds several
// row groups, and one group leaving would make the others' shuffle a partial one.
template <int kWidth>
__device__ __forceinline__ float group_sum(float v) {
#pragma unroll
  for (int off = kWidth / 2; off > 0; off >>= 1) {
    v += __shfl_xor_sync(kFullMask, v, off);
  }
  return v;
}

__device__ __forceinline__ uint32_t load_u32(const __nv_bfloat16* p) {
  return *reinterpret_cast<const uint32_t*>(p);
}

// ---------------------------------------------------------------------------
// Warp-level GEMM primitive.
//
// Both forms accumulate C[16 x 8] over K in the *mma C fragment layout*: lane `l`
// owns rows {gid, gid+8} x cols {2*tig, 2*tig+1}, where gid = l/4 and tig = l%4.
// Sharing the layout is what makes the two interchangeable behind one epilogue,
// and what makes an mma mis-mapping show up as a value difference rather than a
// shape difference.
//
// Every operand index is clamped into range rather than branched on, and the
// *store* carries the bounds check. An out-of-range lane then computes a
// perfectly well-formed wrong number that is never written, which keeps the K
// loop free of per-lane control flow.
// ---------------------------------------------------------------------------
__device__ __forceinline__ int frag_gid() { return (threadIdx.x % kWarpSize) >> 2; }
__device__ __forceinline__ int frag_tig() { return (threadIdx.x % kWarpSize) & 3; }

// The two A rows this lane reads, plus their LayerNorm statistics. Row pointers
// are pre-offset by the lane's `2*tig` so the mma fragment loads are plain
// `[kg]` and `[kg+8]` indexing; the fp32 reference path subtracts that offset
// back out because it walks k contiguously.
struct ARows {
  const __nv_bfloat16* r0;  // row gid,     already advanced by 2*tig
  const __nv_bfloat16* r1;  // row gid + 8, already advanced by 2*tig
  float mu0, rs0;
  float mu1, rs1;
};

// Apply (x - mu) * rstd, then the affine, in fp32, and round once -- the
// LayerNorm boundary, computed in registers rather than read back from a staged
// tile. `w`/`b` hold two bf16 each, matching the two elements of `raw`.
__device__ __forceinline__ uint32_t norm_pair(uint32_t raw, float mu, float rs,
                                              const __nv_bfloat16* w,
                                              const __nv_bfloat16* b, int k) {
  const __nv_bfloat162 rv = *reinterpret_cast<const __nv_bfloat162*>(&raw);
  float2 f = __bfloat1622float2(rv);
  f.x = (f.x - mu) * rs;
  f.y = (f.y - mu) * rs;
  if (w != nullptr) {
    const float2 wf = __bfloat1622float2(
        *reinterpret_cast<const __nv_bfloat162*>(w + k));
    f.x *= wf.x;
    f.y *= wf.y;
  }
  if (b != nullptr) {
    const float2 bf = __bfloat1622float2(
        *reinterpret_cast<const __nv_bfloat162*>(b + k));
    f.x += bf.x;
    f.y += bf.y;
  }
  const __nv_bfloat162 out = __floats2bfloat162_rn(f.x, f.y);
  return *reinterpret_cast<const uint32_t*>(&out);
}

// A fragment for one m16 x k16 tile, straight from global.
//   a[0]: row gid,     k = kg + 2*tig + {0,1}
//   a[1]: row gid + 8, k = kg + 2*tig + {0,1}
//   a[2]: row gid,     k = kg + 2*tig + {8,9}
//   a[3]: row gid + 8, k = kg + 2*tig + {8,9}
template <bool kNormalize>
__device__ __forceinline__ void load_a_frag(const ARows& a, int kg, int tig2,
                                            const __nv_bfloat16* w,
                                            const __nv_bfloat16* b,
                                            uint32_t out[4]) {
  const uint32_t r0lo = load_u32(a.r0 + kg);
  const uint32_t r0hi = load_u32(a.r0 + kg + 8);
  const uint32_t r1lo = load_u32(a.r1 + kg);
  const uint32_t r1hi = load_u32(a.r1 + kg + 8);
  if (kNormalize) {
    out[0] = norm_pair(r0lo, a.mu0, a.rs0, w, b, kg + tig2);
    out[1] = norm_pair(r1lo, a.mu1, a.rs1, w, b, kg + tig2);
    out[2] = norm_pair(r0hi, a.mu0, a.rs0, w, b, kg + tig2 + 8);
    out[3] = norm_pair(r1hi, a.mu1, a.rs1, w, b, kg + tig2 + 8);
  } else {
    out[0] = r0lo;
    out[1] = r1lo;
    out[2] = r0hi;
    out[3] = r1hi;
  }
}

// B fragment for one k16 x n8 tile, straight from global.
//   b[0]: n = n_base + gid, k = kg + 2*tig + {0,1}
//   b[1]: n = n_base + gid, k = kg + 2*tig + {8,9}
// W is [N][K] row-major, which is already the `.col` layout the mma B operand
// wants (n-major with k contiguous), so no transpose and no staging.
//
// These two loads use **half of every sector they fetch**. Per-instruction NCU
// counters put both at 0.50x sector efficiency -- 16.0 of every 32 bytes
// (profile/final-ksplit-ncu/analysis/per_line_sectors.txt), because b[0] and b[1]
// are separate 16-byte accesses at kg and kg+8 and the fragment wants those k halves
// in different registers, so they cannot be merged. An earlier comment here claimed
// the pair consumes one full sector; that was a derivation, and it was wrong.
//
// It is also not where the sectors are. In the gated down-projection the A-fragment
// loads are 53% of all global-load sectors, also at 0.50x, against 7.7% for this
// pair. Neither is the binding constraint at 20% occupancy, and neither was changed.
__device__ __forceinline__ void load_b_frag(const __nv_bfloat16* bp, int kg,
                                            uint32_t out[2]) {
  out[0] = load_u32(bp + kg);
  out[1] = load_u32(bp + kg + 8);
}

__device__ __forceinline__ void mma_m16n8k16(float acc[4], const uint32_t a[4],
                                             const uint32_t b[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// The staging oracle: an explicit fp32 fused-multiply-add inner product over the
// same 16x16 A tile and 16x8 B tile, writing the same accumulator layout. Each
// lane re-reads two A rows and two B columns per k, so it moves 8x the B traffic
// the mma path does -- irrelevant for its purpose, which is to be an
// independent, obviously-fp32 reference for the mma fragment mapping.
template <bool kNormalize>
__device__ __forceinline__ void fma_m16n8k16(float acc[4], const ARows& a, int kg,
                                             int tig2, const __nv_bfloat16* lnw,
                                             const __nv_bfloat16* lnb,
                                             const __nv_bfloat16* w0,
                                             const __nv_bfloat16* w1) {
  // Undo the 2*tig skew baked into the row pointers: this path walks k
  // contiguously rather than in the fragment's interleaved pairs.
  const __nv_bfloat16* ar0 = a.r0 - tig2;
  const __nv_bfloat16* ar1 = a.r1 - tig2;
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    const int k = kg + j;
    float a0 = bf2f(ar0[k]);
    float a1 = bf2f(ar1[k]);
    if (kNormalize) {
      a0 = (a0 - a.mu0) * a.rs0;
      a1 = (a1 - a.mu1) * a.rs1;
      if (lnw != nullptr) {
        const float wv = bf2f(lnw[k]);
        a0 *= wv;
        a1 *= wv;
      }
      if (lnb != nullptr) {
        const float bv = bf2f(lnb[k]);
        a0 += bv;
        a1 += bv;
      }
      // The staged tile rounded here, so the reference path must too.
      a0 = round_bf16(a0);
      a1 = round_bf16(a1);
    }
    const float b0 = bf2f(w0[k]);
    const float b1 = bf2f(w1[k]);
    acc[0] = fmaf(a0, b0, acc[0]);
    acc[1] = fmaf(a0, b1, acc[1]);
    acc[2] = fmaf(a1, b0, acc[2]);
    acc[3] = fmaf(a1, b1, acc[3]);
  }
}

// ---------------------------------------------------------------------------
// Two-pass LayerNorm statistics over rows [m0, m0+kBM).
//
// An fp32 mean, then an fp32 sum of squared deviations from that mean. Not the
// single-pass E[x^2] - E[x]^2 identity: that is a different reduction, it is what
// the frozen L1 winner deliberately does not use, and it loses conditioning on
// the nearly-constant rows this operator sees after a previous normalisation.
//
// The row is read from global twice (the second time out of L2), plus once more
// per k16 step by the fragment loads. That is the price of folding the prologue
// into the GEMM instead of launching a kernel for it, and against a ~4 us launch
// it is not close.
// ---------------------------------------------------------------------------
template <int kThreads>
__device__ void row_stats(const __nv_bfloat16* __restrict__ src, int64_t rows,
                          int n, int m0, float eps, float* __restrict__ mean_out,
                          float* __restrict__ rstd_out) {
  constexpr int kTpr = kThreads / kBM;
  static_assert(kTpr >= 1 && kTpr <= kWarpSize, "row group must fit in a warp");
  static_assert((kTpr & (kTpr - 1)) == 0, "row group must be a power of two");
  static_assert(kTpr * kBM == kThreads, "threads split evenly across the M tile");

  const int r = static_cast<int>(threadIdx.x) / kTpr;
  const int t = static_cast<int>(threadIdx.x) % kTpr;
  const int64_t row = static_cast<int64_t>(m0) + r;
  const int vecs = n / 8;  // K % 16 == 0 is a predicate, so this divides exactly
  const float inv_n = 1.0f / static_cast<float>(n);
  const bool live = row < rows;
  const uint4* base =
      live ? reinterpret_cast<const uint4*>(src + row * n) : nullptr;

  float sum = 0.0f;
  for (int v = live ? t : vecs; v < vecs; v += kTpr) {
    const uint4 packed = base[v];
    const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&packed);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float2 f = __bfloat1622float2(p[j]);
      sum += f.x + f.y;
    }
  }
  const float mean = group_sum<kTpr>(sum) * inv_n;

  float sq = 0.0f;
  for (int v = live ? t : vecs; v < vecs; v += kTpr) {
    const uint4 packed = base[v];
    const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&packed);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float2 f = __bfloat1622float2(p[j]);
      const float dx = f.x - mean;
      const float dy = f.y - mean;
      sq += dx * dx + dy * dy;
    }
  }
  const float rstd = rsqrtf(group_sum<kTpr>(sq) * inv_n + eps);

  if (t == 0) {
    mean_out[r] = mean;
    rstd_out[r] = rstd;
  }
}

// Build this lane's A row view for m-tile `mt`. Rows past the end are clamped to
// a valid row and masked at the store.
__device__ __forceinline__ ARows a_rows(const __nv_bfloat16* src, int64_t rows,
                                       int n, int m0, int mt, int tig2,
                                       const float* mean, const float* rstd) {
  const int gid = frag_gid();
  const int l0 = mt * 16 + gid;
  const int l1 = l0 + 8;
  const int64_t g0 = min(static_cast<int64_t>(m0) + l0, rows - 1);
  const int64_t g1 = min(static_cast<int64_t>(m0) + l1, rows - 1);
  ARows a;
  a.r0 = src + g0 * n + tig2;
  a.r1 = src + g1 * n + tig2;
  if (mean != nullptr) {
    const int c0 = min(l0, kBM - 1);
    const int c1 = min(l1, kBM - 1);
    a.mu0 = mean[c0];
    a.rs0 = rstd[c0];
    a.mu1 = mean[c1];
    a.rs1 = rstd[c1];
  } else {
    a.mu0 = a.rs0 = a.mu1 = a.rs1 = 0.0f;
  }
  return a;
}

// One B operand, as the three row pointers the two inner-product forms need.
// Every one is clamped into range: an out-of-range lane reads a valid row and
// computes a well-formed wrong number that the store then drops, which keeps the
// K loop free of per-lane control flow.
//
//   `frag` is for the mma B fragment: n = n_base + gid, pre-advanced by 2*tig
//          so the loads are plain [kg] and [kg+8].
//   `f0` / `f1` are for the fp32 reference path, which owns columns
//          n_base + 2*tig and + 1 and walks k contiguously.
struct BView {
  const __nv_bfloat16* frag;
  const __nv_bfloat16* f0;
  const __nv_bfloat16* f1;
};

__device__ __forceinline__ BView b_view(const __nv_bfloat16* w, int64_t ldb,
                                        int n_base, int n_limit, int tig2) {
  const int nf = min(n_base + frag_gid(), n_limit - 1);
  const int n0 = min(n_base + tig2, n_limit - 1);
  const int n1 = min(n_base + tig2 + 1, n_limit - 1);
  BView v;
  v.frag = w + static_cast<int64_t>(nf) * ldb + tig2;
  v.f0 = w + static_cast<int64_t>(n0) * ldb;
  v.f1 = w + static_cast<int64_t>(n1) * ldb;
  return v;
}

// One full reduction over K against one or two B operands sharing the A
// fragment. The dual form is the gated dual-GEMM structure the up-projection
// needs: `Wa` and `Wb` run against the same A fragment with two register
// accumulators, so the A loads and the normalisation are paid once.
template <bool kUseMma, bool kNormalize, bool kDual>
__device__ __forceinline__ void gemm_k(float acc0[4], float acc1[4],
                                       const ARows& a, int k, int k_begin,
                                       int k_stride, const __nv_bfloat16* lnw,
                                       const __nv_bfloat16* lnb, const BView& b0,
                                       const BView& b1, int tig2) {
  // `k_begin`/`k_stride` are this warp's slice of the reduction: warp w covers
  // k16 tiles w, w + warps, w + 2*warps, ... A strided slice rather than a
  // contiguous block so every warp issues its first load immediately, and so an
  // extent that is not a multiple of the warp count spreads its remainder one
  // tile per warp instead of piling it on the last.
#pragma unroll 4
  for (int kg = k_begin; kg < k; kg += k_stride) {
    if (kUseMma) {
      uint32_t af[4];
      load_a_frag<kNormalize>(a, kg, tig2, lnw, lnb, af);
      uint32_t bf[2];
      load_b_frag(b0.frag, kg, bf);
      mma_m16n8k16(acc0, af, bf);
      if (kDual) {
        load_b_frag(b1.frag, kg, bf);
        mma_m16n8k16(acc1, af, bf);
      }
    } else {
      fma_m16n8k16<kNormalize>(acc0, a, kg, tig2, lnw, lnb, b0.f0, b0.f1);
      if (kDual) {
        fma_m16n8k16<kNormalize>(acc1, a, kg, tig2, lnw, lnb, b1.f0, b1.f1);
      }
    }
  }
}

// Combine the warps' fp32 partial accumulators for one n8 tile.
//
// Every warp in the CTA holds a partial sum for the same four (row, col)
// positions, in the same lane, so the reduction is a plain elementwise sum down a
// [warps][32][4] shared-memory array -- no shuffles and no lane remapping. Warp 0
// ends up with the total and is the only warp that stores.
//
// One barrier per accumulator group, which is the whole synchronisation cost of
// the K split. `stage` must hold kNWarps * 32 * 4 floats per accumulator.
template <int kNWarps, int kAccs>
__device__ __forceinline__ void reduce_warp_partials(float* __restrict__ stage,
                                                     float acc[kAccs][4]) {
  if (kNWarps == 1) {
    return;
  }
  const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
  const int lane = static_cast<int>(threadIdx.x) % kWarpSize;
#pragma unroll
  for (int m = 0; m < kAccs; ++m) {
    float* slot = stage + (static_cast<size_t>(m) * kNWarps + warp) * kWarpSize * 4;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      slot[lane * 4 + j] = acc[m][j];
    }
  }
  __syncthreads();
  if (warp != 0) {
    return;
  }
#pragma unroll
  for (int m = 0; m < kAccs; ++m) {
    const float* base = stage + static_cast<size_t>(m) * kNWarps * kWarpSize * 4;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float sum = 0.0f;
      for (int w = 0; w < kNWarps; ++w) {
        sum += base[(static_cast<size_t>(w) * kWarpSize + lane) * 4 + j];
      }
      acc[m][j] = sum;
    }
  }
}

// The accumulator's four (row, col) positions for this lane.
__device__ __forceinline__ void acc_positions(int m0, int mt, int n_base,
                                              int row[4], int col[4]) {
  const int gid = frag_gid();
  const int tig = frag_tig();
  row[0] = row[1] = m0 + mt * 16 + gid;
  row[2] = row[3] = m0 + mt * 16 + gid + 8;
  col[0] = col[2] = n_base + 2 * tig;
  col[1] = col[3] = n_base + 2 * tig + 1;
}

// Floats of shared staging one accumulator group needs for the cross-warp
// reduction: one [32][4] block per warp. The `max(..., 1)` keeps the array
// non-empty in the kNWarps==1 case, where reduce_warp_partials is a no-op.
#define FK_AF3_STAGE(WARPS, GROUPS) ((WARPS) > 1 ? (WARPS) * (GROUPS) * 32 * 4 : 1)

#define FK_AF3_ZERO_ACC(acc)         \
  _Pragma("unroll") for (int m = 0; m < kMTiles; ++m) {  \
    _Pragma("unroll") for (int j = 0; j < 4; ++j) { acc[m][j] = 0.0f; } \
  }

// ---------------------------------------------------------------------------
// Kernel 1: AdaLN.
//
//   sn = LayerNorm_s(s)                      weight-only affine, eps_s
//   an = LayerNorm_a(a)                      no affine, eps_a
//   a1 = sigmoid(sn @ Wag^T + bag) * (an + sn @ Was^T)
//
// `s` is the reduction operand of two GEMMs and is normalised in-register on
// every fragment load. `a` is needed only elementwise, in this CTA's own BN
// columns, so its statistics are reduced from global and only the columns the
// epilogue touches are read again.
// ---------------------------------------------------------------------------
template <int kNWarps, bool kUseMma>
__global__ __launch_bounds__(kNWarps * kWarpSize) void adaln_kernel(
    const __nv_bfloat16* __restrict__ a, const __nv_bfloat16* __restrict__ s,
    const __nv_bfloat16* __restrict__ ln_s_w,
    const __nv_bfloat16* __restrict__ wag, const __nv_bfloat16* __restrict__ bag,
    const __nv_bfloat16* __restrict__ was, __nv_bfloat16* __restrict__ out,
    int64_t rows, int c_a, int c_s, float eps_a, float eps_s) {
  constexpr int kThreads = kNWarps * kWarpSize;
  __shared__ float s_mean[kBM], s_rstd[kBM], a_mean[kBM], a_rstd[kBM];
  __shared__ float stage[FK_AF3_STAGE(kNWarps, 2 * kMTiles)];

  const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
  const int m0 = static_cast<int>(blockIdx.x) * kBM;
  const int n_base = static_cast<int>(blockIdx.y) * 8;
  const int tig2 = 2 * frag_tig();

  row_stats<kThreads>(s, rows, c_s, m0, eps_s, s_mean, s_rstd);
  row_stats<kThreads>(a, rows, c_a, m0, eps_a, a_mean, a_rstd);
  __syncthreads();

  float acc_g[kMTiles][4], acc_s[kMTiles][4];
  FK_AF3_ZERO_ACC(acc_g)
  FK_AF3_ZERO_ACC(acc_s)

  const BView bv_g = b_view(wag, c_s, n_base, c_a, tig2);
  const BView bv_s = b_view(was, c_s, n_base, c_a, tig2);
  const int k0 = 16 * warp;
  const int kstep = 16 * kNWarps;

#pragma unroll
  for (int mt = 0; mt < kMTiles; ++mt) {
    const ARows av = a_rows(s, rows, c_s, m0, mt, tig2, s_mean, s_rstd);
    gemm_k<kUseMma, true, true>(acc_g[mt], acc_s[mt], av, c_s, k0, kstep, ln_s_w,
                                nullptr, bv_g, bv_s, tig2);
  }
  reduce_warp_partials<kNWarps, kMTiles>(stage, acc_g);
  reduce_warp_partials<kNWarps, kMTiles>(
      stage + FK_AF3_STAGE(kNWarps, kMTiles), acc_s);
  if (kNWarps > 1 && warp != 0) {
    return;
  }

#pragma unroll
  for (int mt = 0; mt < kMTiles; ++mt) {
    int row[4], col[4];
    acc_positions(m0, mt, n_base, row, col);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      if (row[j] >= rows || col[j] >= c_a) {
        continue;
      }
      // sigmoid(sn @ Wag^T + bag): the bias joins in fp32 so the sum rounds
      // once, then sigmoid rounds again -- two boundaries, matching F.linear
      // followed by torch.sigmoid.
      const float g1 =
          round_bf16(sigmoid_f32(round_bf16(acc_g[mt][j] + bf2f(bag[col[j]]))));
      const float ls = round_bf16(acc_s[mt][j]);
      const int lr = row[j] - m0;
      const float an = round_bf16(
          (bf2f(a[static_cast<int64_t>(row[j]) * c_a + col[j]]) - a_mean[lr]) *
          a_rstd[lr]);
      out[static_cast<int64_t>(row[j]) * c_a + col[j]] = f2bf(g1 * round_bf16(an + ls));
    }
  }
}

// ---------------------------------------------------------------------------
// Kernel 2: up-projection with the SwiGLU gate.
//
//   hh = silu(A @ Wa^T) * (A @ Wb^T)
//
// with LayerNorm(x) folded into the A fragment loads when this is
// SwiGLUTransition's first kernel, and A = a1 straight through for CTB.
// ---------------------------------------------------------------------------
template <int kNWarps, bool kUseMma, bool kNormalize>
__global__ __launch_bounds__(kNWarps * kWarpSize) void up_swiglu_kernel(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ ln_w,
    const __nv_bfloat16* __restrict__ ln_b, const __nv_bfloat16* __restrict__ wa,
    const __nv_bfloat16* __restrict__ wb, __nv_bfloat16* __restrict__ hh,
    int64_t rows, int k, int h, float eps) {
  constexpr int kThreads = kNWarps * kWarpSize;
  __shared__ float x_mean[kBM], x_rstd[kBM];
  __shared__ float stage[FK_AF3_STAGE(kNWarps, 2 * kMTiles)];

  const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
  const int m0 = static_cast<int>(blockIdx.x) * kBM;
  const int n_base = static_cast<int>(blockIdx.y) * 8;
  const int tig2 = 2 * frag_tig();

  if (kNormalize) {
    row_stats<kThreads>(x, rows, k, m0, eps, x_mean, x_rstd);
    __syncthreads();
  }

  float acc_a[kMTiles][4], acc_b[kMTiles][4];
  FK_AF3_ZERO_ACC(acc_a)
  FK_AF3_ZERO_ACC(acc_b)

  const BView bv_a = b_view(wa, k, n_base, h, tig2);
  const BView bv_b = b_view(wb, k, n_base, h, tig2);
  const int k0 = 16 * warp;
  const int kstep = 16 * kNWarps;

#pragma unroll
  for (int mt = 0; mt < kMTiles; ++mt) {
    const ARows av = a_rows(x, rows, k, m0, mt, tig2,
                            kNormalize ? x_mean : nullptr,
                            kNormalize ? x_rstd : nullptr);
    gemm_k<kUseMma, kNormalize, true>(acc_a[mt], acc_b[mt], av, k, k0, kstep,
                                      ln_w, ln_b, bv_a, bv_b, tig2);
  }
  reduce_warp_partials<kNWarps, kMTiles>(stage, acc_a);
  reduce_warp_partials<kNWarps, kMTiles>(
      stage + FK_AF3_STAGE(kNWarps, kMTiles), acc_b);
  if (kNWarps > 1 && warp != 0) {
    return;
  }

#pragma unroll
  for (int mt = 0; mt < kMTiles; ++mt) {
    int row[4], col[4];
    acc_positions(m0, mt, n_base, row, col);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      if (row[j] >= rows || col[j] >= h) {
        continue;
      }
      // ha rounds, silu(ha) rounds, hb rounds, and the product rounds: four
      // boundaries, matching F.linear / F.silu / F.linear / bf16 mul.
      const float ha = round_bf16(acc_a[mt][j]);
      const float sa = round_bf16(silu_f32(ha));
      const float hb = round_bf16(acc_b[mt][j]);
      hh[static_cast<int64_t>(row[j]) * h + col[j]] = f2bf(sa * hb);
    }
  }
}

// ---------------------------------------------------------------------------
// Kernel 3: down-projection, with the outer gate and the mask in the epilogue.
//
//   out = (hh @ Wout^T) * mask                              SwiGLUTransition
//   out = sigmoid(s @ Wg^T + bg) * (hh @ Wout^T) * mask      CTB
//
// The gate is folded rather than materialised because it is an elementwise factor
// of the output: this CTA owns exactly the [BM, BN] output tile, so it needs
// exactly the [BM, BN] gate tile, at the cost of one extra K = c_s reduction and
// the saving of an [M, c_a] intermediate plus its launch. The two A operands have
// different reduction extents (h and c_s), so they are two separate K loops
// against two accumulators.
// ---------------------------------------------------------------------------
template <int kNWarps, bool kUseMma, bool kGate>
__global__ __launch_bounds__(kNWarps * kWarpSize) void down_kernel(
    const __nv_bfloat16* __restrict__ hh, const __nv_bfloat16* __restrict__ wout,
    const __nv_bfloat16* __restrict__ s, const __nv_bfloat16* __restrict__ wg,
    const __nv_bfloat16* __restrict__ bg, const __nv_bfloat16* __restrict__ mask,
    __nv_bfloat16* __restrict__ out, int64_t rows, int h, int c_out, int c_s) {
  __shared__ float stage[FK_AF3_STAGE(kNWarps, 2 * kMTiles)];

  const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
  const int m0 = static_cast<int>(blockIdx.x) * kBM;
  const int n_base = static_cast<int>(blockIdx.y) * 8;
  const int tig2 = 2 * frag_tig();

  float acc_d[kMTiles][4], acc_g[kMTiles][4];
  FK_AF3_ZERO_ACC(acc_d)
  FK_AF3_ZERO_ACC(acc_g)

  const BView bv_d = b_view(wout, h, n_base, c_out, tig2);
  const int k0 = 16 * warp;
  const int kstep = 16 * kNWarps;

#pragma unroll
  for (int mt = 0; mt < kMTiles; ++mt) {
    const ARows av = a_rows(hh, rows, h, m0, mt, tig2, nullptr, nullptr);
    gemm_k<kUseMma, false, false>(acc_d[mt], acc_d[mt], av, h, k0, kstep, nullptr,
                                  nullptr, bv_d, bv_d, tig2);
  }

  if (kGate) {
    const BView bv_g = b_view(wg, c_s, n_base, c_out, tig2);
#pragma unroll
    for (int mt = 0; mt < kMTiles; ++mt) {
      const ARows av = a_rows(s, rows, c_s, m0, mt, tig2, nullptr, nullptr);
      gemm_k<kUseMma, false, false>(acc_g[mt], acc_g[mt], av, c_s, k0, kstep,
                                    nullptr, nullptr, bv_g, bv_g, tig2);
    }
  }

  reduce_warp_partials<kNWarps, kMTiles>(stage, acc_d);
  if (kGate) {
    reduce_warp_partials<kNWarps, kMTiles>(
        stage + FK_AF3_STAGE(kNWarps, kMTiles), acc_g);
  }
  if (kNWarps > 1 && warp != 0) {
    return;
  }

#pragma unroll
  for (int mt = 0; mt < kMTiles; ++mt) {
    int row[4], col[4];
    acc_positions(m0, mt, n_base, row, col);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      if (row[j] >= rows || col[j] >= c_out) {
        continue;
      }
      float v = round_bf16(acc_d[mt][j]);
      if (kGate) {
        const float g =
            round_bf16(sigmoid_f32(round_bf16(acc_g[mt][j] + bf2f(bg[col[j]]))));
        // Rounds here whether or not a mask follows. Under the harness's
        // all-ones mask the extra rounding is invisible, but the declared
        // fast-path domain accepts other masks and for those the orders differ.
        v = round_bf16(g * v);
      }
      if (mask != nullptr) {
        v = v * bf2f(mask[row[j]]);
      }
      out[static_cast<int64_t>(row[j]) * c_out + col[j]] = f2bf(v);
    }
  }
}


}  // namespace fk_af3
