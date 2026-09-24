// Fused AlphaFold3 pair-stack block for B200 (sm_100).
//
// The block is TriMulOut -> TriMulIn -> TriAttStart -> TriAttEnd ->
// SwiGLUTransition, each in a residual add. Eager it issues 109 kernels whose
// total device self-time is 431 us while the host spends 1324 us enqueueing
// them, so the cost is launch overhead twice over: once on the host at ~5.0 us
// per dispatch, once on the device at 2.0-3.8 us per dependent launch (measured,
// profile/p1-baseline/results.md). Everything here exists to collapse both
// counts: ten kernels reached through one host dispatch.
//
// Two algebraic facts make the collapse possible.
//
//  * Every LayerNorm in the block is immediately followed by a matmul that
//    contracts exactly the normalised axis, so a CTA owning a row block of that
//    matmul already holds the rows' full contraction extent and the row
//    statistics cost no extra global traffic. With LN(x)_k = xhat_k*w_k + b_k,
//    y_c = sum_k (W[c,k]*w_k)*xhat_k + sum_k W[c,k]*b_k -- the affine folds into
//    a pre-scaled weight and an offset vector, both built once on the host.
//
//  * Every consumer of a LayerNorm output consumes it through several
//    independent projections of the same input, so those concatenate into one
//    GEMM: 27 `mm` calls become 5. The 1/sqrt(c_hidden) query scale folds into
//    the q rows of the concatenated attention weight.
//
// Column order inside a concatenated weight is chosen so that the values an
// epilogue has to combine land in the same thread. The triangle-multiplication
// weight interleaves (a_p, a_g, b_p, b_g) per hidden channel and the transition
// weight interleaves (a, b) per hidden channel, because a column tile is 64 wide
// while the hidden dim is 128/512 -- without the interleave the gate and the
// value being gated would sit in different CTAs.
//
// Decomposition. Kernel count is a measured Pareto variable here, not a target.
// A first cut fused each stage's wide output projection (linear_z, linear_o,
// linear_out) into the kernel that produced its input, giving ten kernels. An NCU
// record of that version (profile/p1-fused-v1/) showed why that was wrong: the
// combine and attention kernels ran 64 CTAs -- 0.43 waves over 148 SMs -- and
// spent 16.6 and 12.7 stall cycles per issue-active on long_scoreboard, because a
// 128-iteration scalar walk over a [128,128] weight has nothing to hide it at four
// warps per SM. FMA pipe utilization was 2-4.5% and L2 under 1% of peak, so the
// problem was neither arithmetic nor bandwidth.
//
// So the three output projections are their own properly-tiled kernel instead
// (`gemm_residual`, shared by all three, with an optional gate, optional mask and
// optional transposed write). That is fourteen kernels rather than ten, which is
// the right trade in this direction: each saved launch costs 2.0-3.8 us while the
// over-fused kernels cost tens of microseconds.
//
// Numerics: fp32 accumulation throughout, with FK_PB_FAITHFUL reinstating a bf16
// round at each point the eager baseline rounds (after every LayerNorm, after
// every GEMM, after the softmax, after each residual add). The residual chain
// itself is bf16 in both modes because the baseline's `z = z + stage(z)` is a
// bf16 add. LayerNorm uses a two-pass reduction, never E[x^2]-E[x]^2: the
// same catastrophic-cancellation argument that rules out folding the mean out
// of the GEMM rules that out too, and normalising explicitly in registers is
// free here.

#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>
#include <torch/library.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>

#include <algorithm>
#include <cstdint>

// Reinstate the eager baseline's bf16 rounding points. Off by default: at the
// harness's own weight scale the fp32-throughout path holds matched_ratio
// 0.99976 against a 0.99 gate, and dropping the rounds is what buys the fusion.
#ifndef FK_PB_FAITHFUL
#define FK_PB_FAITHFUL 0
#endif

// Mapping choices, all measured rather than deduced. Exposed as macros so
// tools/ab_mapping.py can re-measure them by recompiling instead of by adding a
// runtime switch on the timed path. Outputs per thread is the important one: with
// 148 SMs and a block this small, warps in flight is work/(TM*TN*32), so raising
// the register tile buys arithmetic intensity and costs the only latency hiding
// available.
// Measured by tools/ab_mapping.py, pinned to one GPU: (8,64,1,2,128) at 146.5 us beat
// (16,64,2,2,128) 150.5, (8,32,1,2,128) 151.2, (16,64,1,2,128) 156.5,
// (16,64,1,4,128) 161.8 and (32,64,1,4,128) 173.1. Two conclusions, both the same one:
// fewer outputs per thread wins (warps in flight is work/(TM*TN*32), and at 148 SMs on
// a problem this small the warp population is worth more than arithmetic intensity or
// a float4 weight read), and KT at the full contraction extent wins because it costs
// one barrier pair instead of K/KT and leaves the inner trip count a compile-time
// constant that nvcc can unroll.
#ifndef FK_PB_PROJ_BM
#define FK_PB_PROJ_BM 8
#endif
#ifndef FK_PB_PROJ_BN
#define FK_PB_PROJ_BN 64
#endif
#ifndef FK_PB_PROJ_TM
#define FK_PB_PROJ_TM 1
#endif
#ifndef FK_PB_PROJ_TN
#define FK_PB_PROJ_TN 2
#endif
#ifndef FK_PB_PROJ_KT
#define FK_PB_PROJ_KT 128
#endif
// Pair rows per CTA in the triangle einsum. One row per CTA gives 256 CTAs of
// four warps; more rows per CTA trades CTA count for reuse of the b operand.
#ifndef FK_PB_EINSUM_RPB
#define FK_PB_EINSUM_RPB 1
#endif
// Queries per CTA in the attention kernel. Each (query, head) pair owns a warp, so
// this is also warps per CTA divided by no_heads_pair.
#ifndef FK_PB_ATTN_QPB
#define FK_PB_ATTN_QPB 4
#endif
// Keys whose k-vectors are loaded before any of their dot products is reduced.
// Without this the score loop is N dependent global loads in series, which is what
// put 12.7 long_scoreboard cycles per issue on the first version.
#ifndef FK_PB_ATTN_KKT
#define FK_PB_ATTN_KKT 8
#endif
// Measured by tools/ab_mapping.py, pinned to one GPU (the devices on this node differ
// by 1.5x in clock, and an unpinned sweep reads that as a mapping difference -- it
// ranked TN=4 first). Pinned: outputs per thread 4 -> 2 -> 1 costs 193.6 -> 180.6 ->
// 161.0 us. One output per thread wins outright here even though it gives up the
// float4 weight read, because this product is only 256x128: warps in flight is
// work/(TM*TN*32), so TN=1 quadruples the warp population to ~1024 and that is worth
// more than vectorisation on a kernel whose stall profile is dominated by memory
// latency at four warps per SM.
#ifndef FK_PB_GEMM_BM
#define FK_PB_GEMM_BM 16
#endif
#ifndef FK_PB_GEMM_BN
#define FK_PB_GEMM_BN 32
#endif
#ifndef FK_PB_GEMM_TM
#define FK_PB_GEMM_TM 1
#endif
#ifndef FK_PB_GEMM_TN
#define FK_PB_GEMM_TN 1
#endif
#ifndef FK_PB_GEMM_KT
#define FK_PB_GEMM_KT 128
#endif

namespace {

constexpr int kWarp = 32;
constexpr unsigned kFull = 0xffffffffu;
// Every concatenated weight is padded out to this many columns so a column tile
// is always fully in bounds and the k loop needs no predicate. The padding is
// zero-filled and its beta is zero, so the padded columns compute exact zeros.
constexpr int kColPad = 64;
// Shared-memory row padding: an unpadded [rows][C] tile puts every row's element
// k in the same bank, which serialises the GEMM's inner reads.
constexpr int kSmemPad = 4;
// Widest single global access the SM offers, in bf16 elements.
constexpr int kBf16Vec = 8;

__host__ __device__ __forceinline__ int round_up(int v, int m) {
  return ((v + m - 1) / m) * m;
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = kWarp / 2; o > 0; o >>= 1) v += __shfl_xor_sync(kFull, v, o);
  return v;
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = kWarp / 2; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(kFull, v, o));
  return v;
}

// bf16 round, or identity in the fp32-throughout mode.
__device__ __forceinline__ float rnd(float v) {
#if FK_PB_FAITHFUL
  return __bfloat162float(__float2bfloat16_rn(v));
#else
  return v;
#endif
}

__device__ __forceinline__ float sigmoidf(float x) {
  return 1.0f / (1.0f + __expf(-x));
}

__device__ __forceinline__ float siluf(float x) { return x * sigmoidf(x); }

__device__ __forceinline__ void unpack_bf16x8(const uint4& v, float* out) {
  const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float2 f = __bfloat1622float2(p[j]);
    out[2 * j] = f.x;
    out[2 * j + 1] = f.y;
  }
}

// ===========================================================================
// Packed-weight layout.
//
// One fp32 tensor holds every fused weight, so `forward` passes a single tensor
// and the kernels reach their sections by offsets computed from the config. The
// host builds the buffer from the offsets this function returns, so there is one
// definition of the layout rather than two that can drift.
// ===========================================================================
enum Section {
  // Triangle multiplication, outgoing then incoming.
  kW1 = 0,      // [C, p1p]  concatenated (a_p,a_g,b_p,b_g) interleaved, then g
  kB1,          // [p1p]
  kLnwIn,       // [C]       layer_norm_in affine, used only when FK_PB_FAITHFUL
  kLnbIn,       // [C]
  kWz,          // [M, C]    linear_z with layer_norm_out folded
  kBz,          // [C]
  kLnwOut,      // [M]
  kLnbOut,      // [M]
  kSecPerTriMul,
};

// Section ids for the attention and transition blocks continue past the two
// triangle-multiplication groups.
enum {
  kAttW5 = 2 * kSecPerTriMul,  // [C, p5p]  (q/sqrt(d), k, v, g, linear_z)
  kAttB5,                      // [p5p]
  kAttLnw,                     // [C]
  kAttLnb,                     // [C]
  kAttWo,                      // [Hd, C]   linear_o
  kSecPerAtt = kAttWo + 1 - kAttW5,
};
enum {
  kTrW9 = 2 * kSecPerTriMul + 2 * kSecPerAtt,  // [C, p9p]  (a, b) interleaved
  kTrB9,                                       // [p9p]
  kTrLnw,                                      // [C]
  kTrLnb,                                      // [C]
  kTrW10,                                      // [F, C]    linear_out
  kNumSections,
};

struct Layout {
  int C, M, Dh, Hn, Hd, F;
  int p1p, p5p, p9p, np4;
  int64_t off[kNumSections];
  int64_t num[kNumSections];
  int nsec;
  int64_t total;
};

Layout make_layout(int c_z, int c_hidden_mul, int c_hidden_pair_att,
                   int no_heads_pair, int transition_n) {
  Layout L{};
  L.C = c_z;
  L.M = c_hidden_mul;
  L.Dh = c_hidden_pair_att;
  L.Hn = no_heads_pair;
  L.Hd = c_hidden_pair_att * no_heads_pair;
  L.F = transition_n * c_z;
  L.p1p = round_up(4 * L.M + L.C, kColPad);
  L.p5p = round_up(4 * L.Hd + L.Hn, kColPad);
  L.p9p = round_up(2 * L.F, kColPad);
  // The attention projection buffer holds only q, k, v and the gate; the
  // no_heads-wide triangle-bias columns go to their own [no_heads, rows] buffer,
  // where the attention kernel's per-key read is contiguous instead of striding
  // by a whole projection row.
  L.np4 = 4 * L.Hd;
  L.nsec = kNumSections;

  int64_t num[kNumSections];
  for (int t = 0; t < 2; ++t) {
    const int b = t * kSecPerTriMul;
    num[b + kW1] = (int64_t)L.C * L.p1p;
    num[b + kB1] = L.p1p;
    num[b + kLnwIn] = L.C;
    num[b + kLnbIn] = L.C;
    num[b + kWz] = (int64_t)L.M * L.C;
    num[b + kBz] = L.C;
    num[b + kLnwOut] = L.M;
    num[b + kLnbOut] = L.M;
  }
  for (int t = 0; t < 2; ++t) {
    const int b = kAttW5 + t * kSecPerAtt;
    num[b + 0] = (int64_t)L.C * L.p5p;
    num[b + 1] = L.p5p;
    num[b + 2] = L.C;
    num[b + 3] = L.C;
    num[b + 4] = (int64_t)L.Hd * L.C;
  }
  num[kTrW9] = (int64_t)L.C * L.p9p;
  num[kTrB9] = L.p9p;
  num[kTrLnw] = L.C;
  num[kTrLnb] = L.C;
  num[kTrW10] = (int64_t)L.F * L.C;

  int64_t at = 0;
  for (int s = 0; s < kNumSections; ++s) {
    at = ((at + 3) / 4) * 4;  // 16-byte aligned, so sections can be read as float4
    L.off[s] = at;
    L.num[s] = num[s];
    at += num[s];
  }
  L.total = at;
  return L;
}

// One k step of the register-tiled product, shared by the projection tile and the
// output-projection kernel. Kept as a function so both callers can present it with
// a compile-time trip count.
template <int BM, int BN, int TM, int TN>
__device__ __forceinline__ void mac_step(const float* __restrict__ xs,
                                         const float* __restrict__ ws,
                                         float (&acc)[TM][TN], int xstride,
                                         int ry, int cx, int xk, int wk) {
  float a[TM];
#pragma unroll
  for (int m = 0; m < TM; ++m) a[m] = xs[(ry * TM + m) * xstride + xk];
  float w[TN];
  if constexpr (TN == 4) {
    const float4 v = *reinterpret_cast<const float4*>(&ws[wk * BN + cx * TN]);
    w[0] = v.x; w[1] = v.y; w[2] = v.z; w[3] = v.w;
  } else {
#pragma unroll
    for (int n = 0; n < TN; ++n) w[n] = ws[wk * BN + cx * TN + n];
  }
#pragma unroll
  for (int m = 0; m < TM; ++m)
#pragma unroll
    for (int n = 0; n < TN; ++n) acc[m][n] += a[m] * w[n];
}

// ===========================================================================
// LayerNorm + concatenated GEMM tile.
//
// Shared layout: xs[BM][C+kSmemPad] holds the tile's normalised rows, ws[KT][BN]
// stages the weight, st[2*BM] the row statistics.
// ===========================================================================
template <int BM, int BN, int TM, int TN, int KT>
__device__ __forceinline__ void ln_gemm_tile(
    const __nv_bfloat16* __restrict__ src, const float* __restrict__ wt,
    const float* __restrict__ beta, const float* __restrict__ lnw,
    const float* __restrict__ lnb, int S1, int C, int NP, int N, int transpose,
    float eps, int m0, int n0, float (&acc)[TM][TN], float* smem) {
  constexpr int kThreads = (BM / TM) * (BN / TN);
  constexpr int kNCol = BN / TN;
  constexpr int kWVec = BN / 4;                 // float4 per staged weight row
  constexpr int kWSpan = kWVec < kThreads ? kWVec : kThreads;
  constexpr int kWRows = kThreads / kWSpan;
  const int tid = threadIdx.x;
  const int cx = tid % kNCol;
  const int ry = tid / kNCol;
  const int xstride = C + kSmemPad;
  float* xs = smem;
  float* ws = xs + BM * xstride;
  float* st = ws + KT * BN;

  // Vectorised row load: 16 bytes per thread, so BM rows are covered in a couple
  // of passes of independent loads rather than BM dependent scalar ones.
  const int xvec = C / kBf16Vec;
  const int xv_span = min(kThreads, xvec);
  const int xv_col0 = tid % xv_span, xv_row = tid / xv_span;
  const int xv_rows = max(1, kThreads / xv_span);
  for (int r = xv_row; r < BM; r += xv_rows) {
    const int lr = m0 + r;
    int sr = lr;
    if (transpose && lr < S1) {
      const int i = lr / N;
      sr = (lr - i * N) * N + i;
    }
    for (int v = xv_col0; v < xvec; v += xv_span) {
      float vals[kBf16Vec] = {};
      if (lr < S1) {
        const uint4 raw = *reinterpret_cast<const uint4*>(
            src + (size_t)sr * C + v * kBf16Vec);
        unpack_bf16x8(raw, vals);
      }
#pragma unroll
      for (int e = 0; e < kBf16Vec; ++e) {
        xs[r * xstride + v * kBf16Vec + e] = vals[e];
      }
    }
  }
  __syncthreads();

  constexpr int kNWarp = kThreads / kWarp;
  const int wid = tid / kWarp, lane = tid % kWarp;
  for (int r = wid; r < BM; r += kNWarp) {
    float s = 0.0f;
    for (int c = lane; c < C; c += kWarp) s += xs[r * xstride + c];
    const float mu = warp_sum(s) / (float)C;
    float v = 0.0f;
    for (int c = lane; c < C; c += kWarp) {
      const float d = xs[r * xstride + c] - mu;
      v += d * d;
    }
    v = warp_sum(v);
    if (lane == 0) {
      st[2 * r] = mu;
      st[2 * r + 1] = rsqrtf(v / (float)C + eps);
    }
  }
  __syncthreads();

  for (int r = 0; r < BM; ++r) {
    const float mu = st[2 * r], rs = st[2 * r + 1];
    for (int c = tid; c < C; c += kThreads) {
      float h = (xs[r * xstride + c] - mu) * rs;
#if FK_PB_FAITHFUL
      // The affine cannot be folded into the weight when the baseline's round
      // between LayerNorm and the GEMM has to be reproduced, so this mode is
      // handed the raw weight and applies the affine here instead.
      h = rnd(h * lnw[c] + lnb[c]);
#endif
      xs[r * xstride + c] = h;
    }
  }
  __syncthreads();

#pragma unroll
  for (int m = 0; m < TM; ++m)
#pragma unroll
    for (int n = 0; n < TN; ++n) acc[m][n] = 0.0f;

  const int wv_col0 = tid % kWSpan, wv_row = tid / kWSpan;
  for (int k0 = 0; k0 < C; k0 += KT) {
    for (int kk = wv_row; kk < KT; kk += kWRows) {
      const int k = k0 + kk;
      for (int vc = wv_col0; vc < kWVec; vc += kWSpan) {
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (k < C) {
          v = *reinterpret_cast<const float4*>(wt + (size_t)k * NP + n0 + vc * 4);
        }
        *reinterpret_cast<float4*>(&ws[kk * BN + vc * 4]) = v;
      }
    }
    __syncthreads();
    const int kmax = min(KT, C - k0);
    if (kmax == KT) {
#pragma unroll 8
      for (int kk = 0; kk < KT; ++kk) {
        mac_step<BM, BN, TM, TN>(xs, ws, acc, xstride, ry, cx, k0 + kk, kk);
      }
    } else {
      for (int kk = 0; kk < kmax; ++kk) {
        mac_step<BM, BN, TM, TN>(xs, ws, acc, xstride, ry, cx, k0 + kk, kk);
      }
    }
    __syncthreads();
  }

  const int nbase = n0 + cx * TN;
#pragma unroll
  for (int m = 0; m < TM; ++m)
#pragma unroll
    for (int n = 0; n < TN; ++n) acc[m][n] = rnd(acc[m][n] + beta[nbase + n]);
}

inline int proj_smem_floats(int BM, int BN, int KT, int C) {
  return BM * (C + kSmemPad) + KT * BN + 2 * BM;
}

// ---------------------------------------------------------------------------
// TriMul projections: LayerNorm, the five concatenated projections, three
// sigmoids and the two mask multiplies, producing a, b and the output gate.
// ---------------------------------------------------------------------------
template <int BM, int BN, int TM, int TN, int KT>
__global__ void trimul_proj_kernel(
    const __nv_bfloat16* __restrict__ src, const float* __restrict__ packed,
    const __nv_bfloat16* __restrict__ mask, float* __restrict__ a_out,
    float* __restrict__ b_out, float* __restrict__ g_out, int64_t o_w1,
    int64_t o_b1, int64_t o_lnw, int64_t o_lnb, int S1, int C, int Mh, int NP,
    int N, float eps) {
  extern __shared__ float smem[];
  const int b = blockIdx.z;
  const int m0 = blockIdx.y * BM;
  const int n0 = blockIdx.x * BN;
  src += (size_t)b * S1 * C;
  mask += (size_t)b * S1;
  a_out += (size_t)b * S1 * Mh;
  b_out += (size_t)b * S1 * Mh;
  g_out += (size_t)b * S1 * C;

  float acc[TM][TN];
  ln_gemm_tile<BM, BN, TM, TN, KT>(src, packed + o_w1, packed + o_b1,
                                   packed + o_lnw, packed + o_lnb, S1, C, NP, N,
                                   /*transpose=*/0, eps, m0, n0, acc, smem);

  constexpr int kNCol = BN / TN;
  const int cx = threadIdx.x % kNCol;
  const int ry = threadIdx.x / kNCol;
  const int nbase = n0 + cx * TN;
  const int gate_base = 4 * Mh;

  // Interleaved (value, gate) per hidden channel, a's pairs then b's, so a gate and
  // the value it gates always land in the same thread. Pairs rather than quads,
  // because the pair is the smallest grouping that keeps them together and the tile
  // width is what buys warps in flight.
  static_assert(TN % 2 == 0, "the interleaved (value, gate) pair must fit one tile");
  if (nbase < gate_base) {
    const bool second = nbase >= 2 * Mh;
    float* dst = second ? b_out : a_out;
    const int chan0 = (nbase - (second ? 2 * Mh : 0)) >> 1;
#pragma unroll
    for (int m = 0; m < TM; ++m) {
      const int lr = m0 + ry * TM + m;
      if (lr >= S1) continue;
      const float mv = __bfloat162float(mask[lr]);
#pragma unroll
      for (int q = 0; q < TN / 2; ++q) {
        dst[(size_t)lr * Mh + chan0 + q] =
            rnd(rnd(mv * sigmoidf(acc[m][2 * q + 1])) * acc[m][2 * q + 0]);
      }
    }
  } else if (nbase < gate_base + C) {
    const int c0 = nbase - gate_base;
#pragma unroll
    for (int m = 0; m < TM; ++m) {
      const int lr = m0 + ry * TM + m;
      if (lr >= S1) continue;
#pragma unroll
      for (int n = 0; n < TN; ++n) {
        if (c0 + n < C) g_out[(size_t)lr * C + c0 + n] = sigmoidf(acc[m][n]);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// TriAttention projections: LayerNorm, q (pre-scaled by 1/sqrt(c_hidden)), k, v
// and the gate (sigmoid applied here) into the projection buffer, plus the
// no_heads-wide triangle-bias projection into its own [no_heads, rows] buffer.
// ---------------------------------------------------------------------------
template <int BM, int BN, int TM, int TN, int KT>
__global__ void triatt_proj_kernel(
    const __nv_bfloat16* __restrict__ src, const float* __restrict__ packed,
    float* __restrict__ proj, float* __restrict__ tbias, int64_t o_w5,
    int64_t o_b5, int64_t o_lnw, int64_t o_lnb, int S1, int C, int Hd, int Hn,
    int NP, int N, float eps, int transpose) {
  extern __shared__ float smem[];
  const int b = blockIdx.z;
  const int m0 = blockIdx.y * BM;
  const int n0 = blockIdx.x * BN;
  src += (size_t)b * S1 * C;
  proj += (size_t)b * S1 * 4 * Hd;
  tbias += (size_t)b * Hn * S1;

  float acc[TM][TN];
  ln_gemm_tile<BM, BN, TM, TN, KT>(src, packed + o_w5, packed + o_b5,
                                   packed + o_lnw, packed + o_lnb, S1, C, NP, N,
                                   transpose, eps, m0, n0, acc, smem);

  constexpr int kNCol = BN / TN;
  const int cx = threadIdx.x % kNCol;
  const int ry = threadIdx.x / kNCol;
  const int nbase = n0 + cx * TN;
  const int qkvg = 4 * Hd;
  const bool is_gate = nbase >= 3 * Hd && nbase < qkvg;
#pragma unroll
  for (int m = 0; m < TM; ++m) {
    const int lr = m0 + ry * TM + m;
    if (lr >= S1) continue;
#pragma unroll
    for (int n = 0; n < TN; ++n) {
      const int c = nbase + n;
      if (c < qkvg) {
        proj[(size_t)lr * qkvg + c] = is_gate ? sigmoidf(acc[m][n]) : acc[m][n];
      } else if (c < qkvg + Hn) {
        tbias[(size_t)(c - qkvg) * S1 + lr] = acc[m][n];
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Transition first half: LayerNorm, the two concatenated projections and
// silu(a)*b. The weight interleaves (a, b) per hidden channel so both halves of
// the product land in the same thread.
// ---------------------------------------------------------------------------
template <int BM, int BN, int TM, int TN, int KT>
__global__ void transition_ffn1_kernel(
    const __nv_bfloat16* __restrict__ src, const float* __restrict__ packed,
    float* __restrict__ h_out, int64_t o_w9, int64_t o_b9, int64_t o_lnw,
    int64_t o_lnb, int S1, int C, int F, int NP, int N, float eps) {
  extern __shared__ float smem[];
  const int b = blockIdx.z;
  const int m0 = blockIdx.y * BM;
  const int n0 = blockIdx.x * BN;
  src += (size_t)b * S1 * C;
  h_out += (size_t)b * S1 * F;

  float acc[TM][TN];
  ln_gemm_tile<BM, BN, TM, TN, KT>(src, packed + o_w9, packed + o_b9,
                                   packed + o_lnw, packed + o_lnb, S1, C, NP, N,
                                   /*transpose=*/0, eps, m0, n0, acc, smem);

  constexpr int kNCol = BN / TN;
  static_assert(TN % 2 == 0, "the interleaved (a, b) pair must fit one tile");
  const int cx = threadIdx.x % kNCol;
  const int ry = threadIdx.x / kNCol;
  const int f0 = (n0 + cx * TN) >> 1;
#pragma unroll
  for (int m = 0; m < TM; ++m) {
    const int lr = m0 + ry * TM + m;
    if (lr >= S1) continue;
#pragma unroll
    for (int p = 0; p < TN / 2; ++p) {
      const int f = f0 + p;
      if (f < F) {
        h_out[(size_t)lr * F + f] =
            rnd(rnd(siluf(acc[m][2 * p])) * acc[m][2 * p + 1]);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// The triangle einsum as index arithmetic, then layer_norm_out.
//
//   outgoing: x[i,k,c] = sum_j a[i,j,c] * b[k,j,c]
//   incoming: x[i,k,c] = sum_j a[j,i,c] * b[j,k,c]
//
// Both are index arithmetic on contiguous rows; neither needs a transpose copy.
// linear_z is deliberately *not* fused here -- see the file header.
// ---------------------------------------------------------------------------
template <int RPB, int kThreads>
__global__ void trimul_einsum_ln_kernel(
    const float* __restrict__ a, const float* __restrict__ b,
    const float* __restrict__ packed, float* __restrict__ xhat, int64_t o_lnw,
    int64_t o_lnb, int S1, int Mh, int N, float eps, int outgoing) {
  extern __shared__ float smem[];
  const int bz = blockIdx.z;
  const int tid = threadIdx.x;
  const int r0 = blockIdx.x * RPB;
  const int xstride = Mh + kSmemPad;
  float* xs = smem;
  float* red = xs + RPB * xstride;
  constexpr int kNWarp = kThreads / kWarp;
  const int wid = tid / kWarp, lane = tid % kWarp;

  a += (size_t)bz * S1 * Mh;
  b += (size_t)bz * S1 * Mh;
  xhat += (size_t)bz * S1 * Mh;
  const float* lnw = packed + o_lnw;
  const float* lnb = packed + o_lnb;
  (void)lnw;  // read only when FK_PB_FAITHFUL applies the affine here
  (void)lnb;

  for (int idx = tid; idx < RPB * xstride; idx += kThreads) xs[idx] = 0.0f;
  __syncthreads();

  for (int rr = 0; rr < RPB; ++rr) {
    const int lr = r0 + rr;
    if (lr >= S1) break;
    const int i = lr / N, k = lr - i * N;
    for (int m = tid; m < Mh; m += kThreads) {
      float s = 0.0f;
      if (outgoing) {
        const float* ap = a + (size_t)i * N * Mh + m;
        const float* bp = b + (size_t)k * N * Mh + m;
        for (int j = 0; j < N; ++j) s += ap[(size_t)j * Mh] * bp[(size_t)j * Mh];
      } else {
        const float* ap = a + (size_t)i * Mh + m;
        const float* bp = b + (size_t)k * Mh + m;
        for (int j = 0; j < N; ++j)
          s += ap[(size_t)j * N * Mh] * bp[(size_t)j * N * Mh];
      }
      xs[rr * xstride + m] = rnd(s);
    }
  }
  __syncthreads();

  // Two-pass row statistics, never E[x^2]-E[x]^2.
  for (int rr = 0; rr < RPB; ++rr) {
    if (r0 + rr >= S1) break;
    float s = 0.0f;
    for (int m = tid; m < Mh; m += kThreads) s += xs[rr * xstride + m];
    s = warp_sum(s);
    if (lane == 0) red[wid] = s;
    __syncthreads();
    float tot = 0.0f;
    for (int w = 0; w < kNWarp; ++w) tot += red[w];
    const float mu = tot / (float)Mh;
    float v = 0.0f;
    for (int m = tid; m < Mh; m += kThreads) {
      const float d = xs[rr * xstride + m] - mu;
      v += d * d;
    }
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) red[wid] = v;
    __syncthreads();
    float vt = 0.0f;
    for (int w = 0; w < kNWarp; ++w) vt += red[w];
    const float rs = rsqrtf(vt / (float)Mh + eps);
    for (int m = tid; m < Mh; m += kThreads) {
      float h = (xs[rr * xstride + m] - mu) * rs;
#if FK_PB_FAITHFUL
      h = rnd(h * lnw[m] + lnb[m]);
#endif
      xhat[(size_t)(r0 + rr) * Mh + m] = h;
    }
    __syncthreads();
  }
}

// ---------------------------------------------------------------------------
// TriAttention: scores, the two differently-indexed biases, softmax, PV and the
// gate, leaving the attended values for the shared output-projection kernel.
//
// mask_bias is indexed by (outer, key) and does not depend on the query, so a CTA
// that owns one outer index needs one mask row -- which is why the grid's y axis
// is the outer index. triangle_bias is indexed by (head, query, key) and
// broadcasts over the outer index, so it reads the completed projection at pair
// row (query, key): a genuine global dependency on the projection kernel, not an
// oversight, which is why the two cannot be one kernel.
//
// The bias is the baseline's finite inf*(mask-1), never a true -inf: a fully
// masked row then yields exactly uniform 1/J under max-subtracted softmax and
// stays finite.
//
// One warp per (query, head) pair, so c_hidden_pair_att must be the warp width.
// ---------------------------------------------------------------------------
template <int QPB, int KKT>
__global__ void triatt_attn_kernel(
    const float* __restrict__ proj, const float* __restrict__ tbias,
    const __nv_bfloat16* __restrict__ mask, float* __restrict__ o_out, int S1,
    int Hd, int Hn, int N, float inf, int transpose) {
  extern __shared__ float smem[];
  const int bz = blockIdx.z;
  const int i = blockIdx.y;
  const int q0 = blockIdx.x * QPB;
  const int tid = threadIdx.x;
  const int qq = tid / Hd;
  const int ht = tid - qq * Hd;
  const int h = ht / kWarp, lane = ht - h * kWarp;
  const int qkvg = 4 * Hd;

  proj += (size_t)bz * S1 * qkvg;
  tbias += (size_t)bz * Hn * S1;
  mask += (size_t)bz * S1;
  o_out += (size_t)bz * S1 * Hd;

  float* mbs = smem;                  // [N]            shared by the whole CTA
  float* scr = mbs + N;               // [QPB][Hn][N]
  float* tbs = scr + QPB * Hn * N;    // [QPB][Hn][N]

  for (int kk = tid; kk < N; kk += QPB * Hd) {
    const float mv = __bfloat162float(transpose ? mask[kk * N + i]
                                                : mask[i * N + kk]);
    mbs[kk] = inf * (mv - 1.0f);
  }

  const int q = q0 + qq;
  const bool live = q < N;
  const int lr = i * N + q;
  float* my_scr = scr + (size_t)(qq * Hn + h) * N;
  float* my_tbs = tbs + (size_t)(qq * Hn + h) * N;
  if (live) {
    // Contiguous in the compact bias buffer, so this is one or two sectors
    // instead of N strided single-float reads.
    for (int kk = lane; kk < N; kk += kWarp) {
      my_tbs[kk] = tbias[(size_t)h * S1 + (size_t)q * N + kk];
    }
  }
  __syncthreads();

  if (!live) return;
  const float* qrow = proj + (size_t)lr * qkvg;
  const float qv = qrow[h * kWarp + lane];

  for (int kk0 = 0; kk0 < N; kk0 += KKT) {
    // Load KKT key vectors before reducing any of them: otherwise the loop is N
    // dependent global loads in series.
    float kv[KKT];
#pragma unroll
    for (int t = 0; t < KKT; ++t) {
      const int kk = kk0 + t;
      kv[t] = kk < N
                  ? proj[(size_t)(i * N + kk) * qkvg + Hd + h * kWarp + lane] * qv
                  : 0.0f;
    }
#pragma unroll
    for (int o = kWarp / 2; o > 0; o >>= 1) {
#pragma unroll
      for (int t = 0; t < KKT; ++t) kv[t] += __shfl_xor_sync(kFull, kv[t], o);
    }
    if (lane == 0) {
#pragma unroll
      for (int t = 0; t < KKT; ++t) {
        const int kk = kk0 + t;
        if (kk < N) my_scr[kk] = rnd(rnd(rnd(kv[t]) + mbs[kk]) + my_tbs[kk]);
      }
    }
  }
  __syncwarp();

  float mx = -INFINITY;
  for (int kk = lane; kk < N; kk += kWarp) mx = fmaxf(mx, my_scr[kk]);
  mx = warp_max(mx);
  float sum = 0.0f;
  for (int kk = lane; kk < N; kk += kWarp) {
    const float e = __expf(my_scr[kk] - mx);
    my_scr[kk] = e;
    sum += e;
  }
  sum = warp_sum(sum);
  const float inv = 1.0f / sum;
  for (int kk = lane; kk < N; kk += kWarp) my_scr[kk] = rnd(my_scr[kk] * inv);
  __syncwarp();

  float o = 0.0f;
  for (int kk0 = 0; kk0 < N; kk0 += KKT) {
    float vv[KKT];
#pragma unroll
    for (int t = 0; t < KKT; ++t) {
      const int kk = kk0 + t;
      vv[t] = kk < N
                  ? my_scr[kk] *
                        proj[(size_t)(i * N + kk) * qkvg + 2 * Hd + h * kWarp + lane]
                  : 0.0f;
    }
#pragma unroll
    for (int t = 0; t < KKT; ++t) o += vv[t];
  }
  o_out[(size_t)lr * Hd + ht] = rnd(rnd(o) * qrow[3 * Hd + ht]);
}

// ---------------------------------------------------------------------------
// The wide output projection shared by all three stages that have one: the
// triangle multiplication's linear_z (with its output gate), the attention's
// linear_o (with the ending node's transposed write), and the transition's
// linear_out (with its mask multiply). Every one of them then adds the residual.
//
// This exists as its own kernel rather than as an epilogue because at 148 SMs a
// [rows, 128] x [128, 128] product fused into the kernel that produced its input
// runs 64 CTAs and stalls on a scalar weight walk; tiled here it runs 128 and
// stages both operands.
// ---------------------------------------------------------------------------
template <int BM, int BN, int TM, int TN, int KT>
__global__ void gemm_residual_kernel(
    const float* __restrict__ x, const float* __restrict__ packed,
    const float* __restrict__ gate, const __nv_bfloat16* __restrict__ mask,
    const __nv_bfloat16* __restrict__ zin, __nv_bfloat16* __restrict__ zout,
    int64_t o_w, int64_t o_beta, int S1, int C, int K, int N, int use_beta,
    int use_gate, int use_mask, int transpose_out) {
  extern __shared__ float smem[];
  constexpr int kThreads = (BM / TM) * (BN / TN);
  constexpr int kNCol = BN / TN;
  const int bz = blockIdx.z;
  const int tid = threadIdx.x;
  const int cx = tid % kNCol, ry = tid / kNCol;
  const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
  const int xstride = KT + kSmemPad;
  float* xs = smem;
  float* ws = xs + BM * xstride;

  x += (size_t)bz * S1 * K;
  zin += (size_t)bz * S1 * C;
  zout += (size_t)bz * S1 * C;
  if (use_gate) gate += (size_t)bz * S1 * C;
  if (use_mask) mask += (size_t)bz * S1;
  const float* w = packed + o_w;
  const float* beta = packed + o_beta;

  // Fixed staging slots, so the index arithmetic is one division per thread rather
  // than one per staged element. Both axes are strided, so the tile is fully
  // covered for any block size.
  constexpr int kXCols = KT < kThreads ? KT : kThreads;
  constexpr int kXRows = kThreads / kXCols;
  const int xc = tid % kXCols, xr = tid / kXCols;
  constexpr int kWCols = BN < kThreads ? BN : kThreads;
  constexpr int kWRowStep = kThreads / kWCols;
  const int wc = tid % kWCols, wr = tid / kWCols;

  float acc[TM][TN];
#pragma unroll
  for (int m = 0; m < TM; ++m)
#pragma unroll
    for (int n = 0; n < TN; ++n) acc[m][n] = 0.0f;

  for (int k0 = 0; k0 < K; k0 += KT) {
    for (int kk = xc; kk < KT; kk += kXCols) {
      for (int r = xr; r < BM; r += kXRows) {
        const int lr = m0 + r, k = k0 + kk;
        xs[r * xstride + kk] = (lr < S1 && k < K) ? x[(size_t)lr * K + k] : 0.0f;
      }
    }
    for (int kk = wr; kk < KT; kk += kWRowStep) {
      for (int nn = wc; nn < BN; nn += kWCols) {
        const int k = k0 + kk, c = n0 + nn;
        ws[kk * BN + nn] = (k < K && c < C) ? w[(size_t)k * C + c] : 0.0f;
      }
    }
    __syncthreads();
    const int kmax = min(KT, K - k0);
    if (kmax == KT) {
#pragma unroll 8
      for (int kk = 0; kk < KT; ++kk) {
        mac_step<BM, BN, TM, TN>(xs, ws, acc, xstride, ry, cx, kk, kk);
      }
    } else {
      for (int kk = 0; kk < kmax; ++kk) {
        mac_step<BM, BN, TM, TN>(xs, ws, acc, xstride, ry, cx, kk, kk);
      }
    }
    __syncthreads();
  }

  const int nbase = n0 + cx * TN;
#pragma unroll
  for (int m = 0; m < TM; ++m) {
    const int lr = m0 + ry * TM + m;
    if (lr >= S1) continue;
    // The ending node transposes (i, j) back on write; every other caller does
    // not, and the residual is read from wherever the result is written.
    size_t zr = lr;
    if (transpose_out) {
      const int i = lr / N;
      zr = (size_t)(lr - i * N) * N + i;
    }
    const float mv = use_mask ? __bfloat162float(mask[lr]) : 1.0f;
#pragma unroll
    for (int n = 0; n < TN; ++n) {
      const int c = nbase + n;
      if (c >= C) continue;
      float y = rnd(acc[m][n] + (use_beta ? beta[c] : 0.0f));
      if (use_gate) y = rnd(y * gate[(size_t)lr * C + c]);
      if (use_mask) y = rnd(y * mv);
      zout[zr * C + c] = __float2bfloat16_rn(
          __bfloat162float(zin[zr * C + c]) + y);
    }
  }
}

// ===========================================================================
// Host-side launchers.
//
// Each is a plain C++ function rather than an operator body, so the same launch
// is reachable from a per-sub-block operator (useful while benchmarking one rung
// at a time, or comparing it against its untouched baseline submodule) and from
// the single top-level operator that the scored path uses. The ~5.0 us per
// dispatch is dispatcher and Python overhead, not cudaLaunchKernel, so collapsing
// dispatches is a separate and cheaper lever than collapsing kernels -- and
// keeping the launchers as functions means changing the dispatch granularity
// never means rewriting a kernel.
// ===========================================================================
constexpr int kProjBM = FK_PB_PROJ_BM;
constexpr int kProjBN = FK_PB_PROJ_BN;
constexpr int kProjTM = FK_PB_PROJ_TM;
constexpr int kProjTN = FK_PB_PROJ_TN;
constexpr int kProjKT = FK_PB_PROJ_KT;
constexpr int kProjThreads = (kProjBM / kProjTM) * (kProjBN / kProjTN);

constexpr int kEinsumRPB = FK_PB_EINSUM_RPB;
constexpr int kEinsumThreads = 128;

constexpr int kAttnQPB = FK_PB_ATTN_QPB;
constexpr int kAttnKKT = FK_PB_ATTN_KKT;

constexpr int kGemmBM = FK_PB_GEMM_BM;
constexpr int kGemmBN = FK_PB_GEMM_BN;
constexpr int kGemmTM = FK_PB_GEMM_TM;
constexpr int kGemmTN = FK_PB_GEMM_TN;
constexpr int kGemmKT = FK_PB_GEMM_KT;
constexpr int kGemmThreads = (kGemmBM / kGemmTM) * (kGemmBN / kGemmTN);

// The largest c_hidden_pair_att*no_heads_pair the attention block shape allows.
constexpr int kMaxHd = 1024 / kAttnQPB;

static_assert(kColPad % kProjBN == 0, "column padding must cover a projection tile");
static_assert(kProjThreads % kWarp == 0, "projection tile needs whole warps");
static_assert(kGemmThreads % kWarp == 0, "output tile needs whole warps");
static_assert(kProjBN % 4 == 0, "the staged weight row is read as float4");
// A tile whose thread count exceeds the block limit fails at launch, which a metadata
// gate cannot predict; catch it at compile time instead.
static_assert(kProjThreads <= 1024, "projection tile exceeds the block size limit");
static_assert(kGemmThreads <= 1024, "output tile exceeds the block size limit");
static_assert(kAttnQPB * kWarp <= 1024, "attention block exceeds the size limit");

int ceil_div(int a, int b) { return (a + b - 1) / b; }

// Dynamic shared memory above 48 KB needs an explicit opt-in, and without it the
// launch fails outright -- which a metadata gate cannot predict, so it would reach
// the harness as a RUNTIME_ERROR rather than as a fallback. The static is per
// kernel instantiation and the common case is a single compare.
template <typename K>
void opt_in_smem(K kernel, size_t bytes) {
  constexpr size_t kDefaultLimit = 48u * 1024u;
  if (bytes <= kDefaultLimit) return;
  static size_t granted = 0;
  if (bytes > granted) {
    cudaFuncSetAttribute(reinterpret_cast<const void*>(kernel),
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)bytes);
    granted = bytes;
  }
}

void launch_trimul_proj(const __nv_bfloat16* src, const float* packed,
                        const __nv_bfloat16* mask, float* a, float* b, float* g,
                        const Layout& L, int which, int B, int S1, int N,
                        float eps, cudaStream_t stream) {
  const int base = which * kSecPerTriMul;
  const size_t smem =
      sizeof(float) * proj_smem_floats(kProjBM, kProjBN, kProjKT, L.C);
  const dim3 grid(L.p1p / kProjBN, ceil_div(S1, kProjBM), B);
  opt_in_smem(trimul_proj_kernel<kProjBM, kProjBN, kProjTM, kProjTN, kProjKT>, smem);
  trimul_proj_kernel<kProjBM, kProjBN, kProjTM, kProjTN, kProjKT>
      <<<grid, kProjThreads, smem, stream>>>(
          src, packed, mask, a, b, g, L.off[base + kW1], L.off[base + kB1],
          L.off[base + kLnwIn], L.off[base + kLnbIn], S1, L.C, L.M, L.p1p, N,
          eps);
}

void launch_trimul_einsum_ln(const float* a, const float* b, const float* packed,
                             float* xhat, const Layout& L, int which, int B,
                             int S1, int N, float eps, cudaStream_t stream) {
  const int base = which * kSecPerTriMul;
  const size_t smem = sizeof(float) * (kEinsumRPB * (L.M + kSmemPad) +
                                       kEinsumThreads / kWarp);
  const dim3 grid(ceil_div(S1, kEinsumRPB), 1, B);
  opt_in_smem(trimul_einsum_ln_kernel<kEinsumRPB, kEinsumThreads>, smem);
  trimul_einsum_ln_kernel<kEinsumRPB, kEinsumThreads>
      <<<grid, kEinsumThreads, smem, stream>>>(
          a, b, packed, xhat, L.off[base + kLnwOut], L.off[base + kLnbOut], S1,
          L.M, N, eps, /*outgoing=*/which == 0);
}

void launch_triatt_proj(const __nv_bfloat16* src, const float* packed,
                        float* proj, float* tbias, const Layout& L, int which,
                        int B, int S1, int N, float eps, cudaStream_t stream) {
  const int base = kAttW5 + which * kSecPerAtt;
  const size_t smem =
      sizeof(float) * proj_smem_floats(kProjBM, kProjBN, kProjKT, L.C);
  const dim3 grid(L.p5p / kProjBN, ceil_div(S1, kProjBM), B);
  opt_in_smem(triatt_proj_kernel<kProjBM, kProjBN, kProjTM, kProjTN, kProjKT>, smem);
  triatt_proj_kernel<kProjBM, kProjBN, kProjTM, kProjTN, kProjKT>
      <<<grid, kProjThreads, smem, stream>>>(
          src, packed, proj, tbias, L.off[base + 0], L.off[base + 1],
          L.off[base + 2], L.off[base + 3], S1, L.C, L.Hd, L.Hn, L.p5p, N, eps,
          /*transpose=*/which == 1);
}

void launch_triatt_attn(const float* proj, const float* tbias,
                        const __nv_bfloat16* mask, float* o, const Layout& L,
                        int which, int B, int S1, int N, float inf,
                        cudaStream_t stream) {
  const size_t smem = sizeof(float) * (N + 2 * (size_t)kAttnQPB * L.Hn * N);
  const dim3 grid(ceil_div(N, kAttnQPB), N, B);
  opt_in_smem(triatt_attn_kernel<kAttnQPB, kAttnKKT>, smem);
  triatt_attn_kernel<kAttnQPB, kAttnKKT>
      <<<grid, kAttnQPB * L.Hd, smem, stream>>>(
          proj, tbias, mask, o, S1, L.Hd, L.Hn, N, inf,
          /*transpose=*/which == 1);
}

void launch_ffn1(const __nv_bfloat16* src, const float* packed, float* h,
                 const Layout& L, int B, int S1, int N, float eps,
                 cudaStream_t stream) {
  const size_t smem =
      sizeof(float) * proj_smem_floats(kProjBM, kProjBN, kProjKT, L.C);
  const dim3 grid(L.p9p / kProjBN, ceil_div(S1, kProjBM), B);
  opt_in_smem(transition_ffn1_kernel<kProjBM, kProjBN, kProjTM, kProjTN, kProjKT>, smem);
  transition_ffn1_kernel<kProjBM, kProjBN, kProjTM, kProjTN, kProjKT>
      <<<grid, kProjThreads, smem, stream>>>(
          src, packed, h, L.off[kTrW9], L.off[kTrB9], L.off[kTrLnw],
          L.off[kTrLnb], S1, L.C, L.F, L.p9p, N, eps);
}

void launch_gemm_residual(const float* x, const float* packed, const float* gate,
                          const __nv_bfloat16* mask, const __nv_bfloat16* zin,
                          __nv_bfloat16* zout, int64_t o_w, int64_t o_beta,
                          int B, int S1, int N, int C, int K, bool use_beta,
                          bool use_gate, bool use_mask, bool transpose_out,
                          cudaStream_t stream) {
  const size_t smem = sizeof(float) * (kGemmBM * (kGemmKT + kSmemPad) +
                                       kGemmKT * kGemmBN);
  const dim3 grid(ceil_div(C, kGemmBN), ceil_div(S1, kGemmBM), B);
  opt_in_smem(gemm_residual_kernel<kGemmBM, kGemmBN, kGemmTM, kGemmTN, kGemmKT>, smem);
  gemm_residual_kernel<kGemmBM, kGemmBN, kGemmTM, kGemmTN, kGemmKT>
      <<<grid, kGemmThreads, smem, stream>>>(
          x, packed, gate, mask, zin, zout, o_w, o_beta, S1, C, K, N,
          use_beta ? 1 : 0, use_gate ? 1 : 0, use_mask ? 1 : 0,
          transpose_out ? 1 : 0);
}

// ===========================================================================
// Operator bodies.
// ===========================================================================
struct Shapes {
  int B, N, S1;
};

Shapes check_inputs(const at::Tensor& z, const at::Tensor& pair_mask,
                    const at::Tensor& packed, const Layout& L) {
  TORCH_CHECK(z.is_cuda() && pair_mask.is_cuda() && packed.is_cuda(),
              "pair_block: all tensors must be on CUDA");
  TORCH_CHECK(z.scalar_type() == at::kBFloat16, "pair_block: z must be bf16");
  TORCH_CHECK(pair_mask.scalar_type() == at::kBFloat16,
              "pair_block: pair_mask must be bf16");
  TORCH_CHECK(packed.scalar_type() == at::kFloat,
              "pair_block: packed weights must be fp32");
  TORCH_CHECK(z.dim() == 4 && pair_mask.dim() == 3,
              "pair_block: expected z [B,N,N,C] and pair_mask [B,N,N]");
  TORCH_CHECK(z.is_contiguous() && pair_mask.is_contiguous() &&
                  packed.is_contiguous(),
              "pair_block: inputs must be contiguous");
  const int B = (int)z.size(0), N = (int)z.size(1);
  TORCH_CHECK(z.size(2) == N && z.size(3) == L.C,
              "pair_block: z does not match the configured shape");
  TORCH_CHECK(pair_mask.size(0) == B && pair_mask.size(1) == N &&
                  pair_mask.size(2) == N,
              "pair_block: pair_mask does not match z");
  TORCH_CHECK(packed.numel() >= L.total,
              "pair_block: packed weight buffer is too small");
  TORCH_CHECK(L.Dh == kWarp,
              "pair_block: one warp per head requires c_hidden_pair_att == 32");
  TORCH_CHECK(L.Hd % kWarp == 0 && L.Hd <= kMaxHd,
              "pair_block: c_hidden_pair_att*no_heads_pair must be a whole "
              "number of warps, at most ", kMaxHd);
  TORCH_CHECK(L.C % kBf16Vec == 0 && L.M % 4 == 0 && L.F % 2 == 0,
              "pair_block: channel counts must match the vector widths");
  return Shapes{B, N, N * N};
}

// Scratch for one call: the stage buffers plus one bf16 buffer for the residual
// chain's ping-pong. One allocation rather than seven, because at this size the
// allocator shows up next to a ~5 us dispatch budget.
struct Scratch {
  at::Tensor buf;
  __nv_bfloat16* alt;
  float* stage;
};

Scratch make_scratch(const at::Tensor& z, const Layout& L, int B, int S1) {
  const int64_t S = (int64_t)B * S1;
  const int64_t bf16_floats = (S * L.C + 1) / 2;
  const int64_t stage_floats = std::max({S * (3 * L.M + L.C),
                                         S * 5 * L.Hd + (int64_t)L.Hn * S,
                                         S * L.F});
  auto buf = at::empty({bf16_floats + stage_floats},
                       z.options().dtype(at::kFloat));
  auto* base = buf.data_ptr<float>();
  return Scratch{buf, reinterpret_cast<__nv_bfloat16*>(base),
                 base + bf16_floats};
}

// The three stage chains, as plain functions so the top-level operator and the
// per-sub-block operators run the same code.
void run_trimul(const __nv_bfloat16* src, const float* pk,
                const __nv_bfloat16* mask, const __nv_bfloat16* zin,
                __nv_bfloat16* zout, const Scratch& sc, const Layout& L,
                int which, int B, int S1, int N, float eps,
                cudaStream_t stream) {
  const int64_t rows = (int64_t)B * S1;
  float* a = sc.stage;
  float* b = a + rows * L.M;
  float* g = b + rows * L.M;
  float* xhat = g + rows * L.C;
  const int base = which * kSecPerTriMul;
  launch_trimul_proj(src, pk, mask, a, b, g, L, which, B, S1, N, eps, stream);
  launch_trimul_einsum_ln(a, b, pk, xhat, L, which, B, S1, N, eps, stream);
  launch_gemm_residual(xhat, pk, g, nullptr, zin, zout, L.off[base + kWz],
                       L.off[base + kBz], B, S1, N, L.C, L.M, /*use_beta=*/true,
                       /*use_gate=*/true, /*use_mask=*/false,
                       /*transpose_out=*/false, stream);
}

void run_triatt(const __nv_bfloat16* src, const float* pk,
                const __nv_bfloat16* mask, const __nv_bfloat16* zin,
                __nv_bfloat16* zout, const Scratch& sc, const Layout& L,
                int which, int B, int S1, int N, float eps, float inf,
                cudaStream_t stream) {
  const int64_t rows = (int64_t)B * S1;
  float* proj = sc.stage;
  float* tbias = proj + rows * L.np4;
  float* o = tbias + (int64_t)L.Hn * rows;
  const int base = kAttW5 + which * kSecPerAtt;
  launch_triatt_proj(src, pk, proj, tbias, L, which, B, S1, N, eps, stream);
  launch_triatt_attn(proj, tbias, mask, o, L, which, B, S1, N, inf, stream);
  launch_gemm_residual(o, pk, nullptr, nullptr, zin, zout, L.off[base + 4], 0, B,
                       S1, N, L.C, L.Hd, /*use_beta=*/false, /*use_gate=*/false,
                       /*use_mask=*/false, /*transpose_out=*/which == 1, stream);
}

void run_transition(const __nv_bfloat16* src, const float* pk,
                    const __nv_bfloat16* mask, const __nv_bfloat16* zin,
                    __nv_bfloat16* zout, const Scratch& sc, const Layout& L,
                    int B, int S1, int N, float eps, bool use_mask,
                    cudaStream_t stream) {
  launch_ffn1(src, pk, sc.stage, L, B, S1, N, eps, stream);
  launch_gemm_residual(sc.stage, pk, nullptr, mask, zin, zout, L.off[kTrW10], 0,
                       B, S1, N, L.C, L.F, /*use_beta=*/false,
                       /*use_gate=*/false, use_mask, /*transpose_out=*/false,
                       stream);
}

at::Tensor pair_block(const at::Tensor& z, const at::Tensor& pair_mask,
                      const at::Tensor& packed, int64_t c_z,
                      int64_t c_hidden_mul, int64_t c_hidden_pair_att,
                      int64_t no_heads_pair, int64_t transition_n, double inf,
                      double eps, bool mask_trans) {
  const Layout L = make_layout((int)c_z, (int)c_hidden_mul,
                               (int)c_hidden_pair_att, (int)no_heads_pair,
                               (int)transition_n);
  const Shapes sh = check_inputs(z, pair_mask, packed, L);
  const c10::cuda::CUDAGuard guard(z.device());
  // Read the stream per call: capturing one at build time would pin every later
  // call to whatever stream happened to be current then.
  const auto stream = c10::cuda::getCurrentCUDAStream();

  auto out = at::empty(z.sizes(), z.options());
  const Scratch sc = make_scratch(z, L, sh.B, sh.S1);

  const auto* zin = reinterpret_cast<const __nv_bfloat16*>(z.const_data_ptr());
  const auto* mask =
      reinterpret_cast<const __nv_bfloat16*>(pair_mask.const_data_ptr());
  const float* pk = packed.const_data_ptr<float>();
  auto* zout = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  __nv_bfloat16* alt = sc.alt;
  const float e = (float)eps, f = (float)inf;

  // Five residual stages ping-pong between the returned tensor and one scratch
  // buffer, ending on the returned tensor. No stage reads the buffer it writes,
  // and the first reads the caller's z without touching it.
  run_trimul(zin, pk, mask, zin, zout, sc, L, 0, sh.B, sh.S1, sh.N, e, stream);
  run_trimul(zout, pk, mask, zout, alt, sc, L, 1, sh.B, sh.S1, sh.N, e, stream);
  run_triatt(alt, pk, mask, alt, zout, sc, L, 0, sh.B, sh.S1, sh.N, e, f, stream);
  run_triatt(zout, pk, mask, zout, alt, sc, L, 1, sh.B, sh.S1, sh.N, e, f, stream);
  run_transition(alt, pk, mask, alt, zout, sc, L, sh.B, sh.S1, sh.N, e,
                 mask_trans, stream);
  return out;
}

// Per-sub-block operators. The scored path never calls these; they exist so one
// rung of the ladder can be benchmarked or compared against its untouched
// baseline submodule without the other four in the way.
at::Tensor trimul_stage(const at::Tensor& z, const at::Tensor& pair_mask,
                        const at::Tensor& packed, int64_t c_z,
                        int64_t c_hidden_mul, int64_t c_hidden_pair_att,
                        int64_t no_heads_pair, int64_t transition_n, double eps,
                        bool outgoing) {
  const Layout L = make_layout((int)c_z, (int)c_hidden_mul,
                               (int)c_hidden_pair_att, (int)no_heads_pair,
                               (int)transition_n);
  const Shapes sh = check_inputs(z, pair_mask, packed, L);
  const c10::cuda::CUDAGuard guard(z.device());
  const auto stream = c10::cuda::getCurrentCUDAStream();
  auto out = at::empty(z.sizes(), z.options());
  const Scratch sc = make_scratch(z, L, sh.B, sh.S1);
  const auto* zp = reinterpret_cast<const __nv_bfloat16*>(z.const_data_ptr());
  run_trimul(zp, packed.const_data_ptr<float>(),
             reinterpret_cast<const __nv_bfloat16*>(pair_mask.const_data_ptr()),
             zp, reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), sc, L,
             outgoing ? 0 : 1, sh.B, sh.S1, sh.N, (float)eps, stream);
  return out;
}

at::Tensor triatt_stage(const at::Tensor& z, const at::Tensor& pair_mask,
                        const at::Tensor& packed, int64_t c_z,
                        int64_t c_hidden_mul, int64_t c_hidden_pair_att,
                        int64_t no_heads_pair, int64_t transition_n, double inf,
                        double eps, bool starting) {
  const Layout L = make_layout((int)c_z, (int)c_hidden_mul,
                               (int)c_hidden_pair_att, (int)no_heads_pair,
                               (int)transition_n);
  const Shapes sh = check_inputs(z, pair_mask, packed, L);
  const c10::cuda::CUDAGuard guard(z.device());
  const auto stream = c10::cuda::getCurrentCUDAStream();
  auto out = at::empty(z.sizes(), z.options());
  const Scratch sc = make_scratch(z, L, sh.B, sh.S1);
  const auto* zp = reinterpret_cast<const __nv_bfloat16*>(z.const_data_ptr());
  run_triatt(zp, packed.const_data_ptr<float>(),
             reinterpret_cast<const __nv_bfloat16*>(pair_mask.const_data_ptr()),
             zp, reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), sc, L,
             starting ? 0 : 1, sh.B, sh.S1, sh.N, (float)eps, (float)inf,
             stream);
  return out;
}

at::Tensor transition_stage(const at::Tensor& z, const at::Tensor& pair_mask,
                            const at::Tensor& packed, int64_t c_z,
                            int64_t c_hidden_mul, int64_t c_hidden_pair_att,
                            int64_t no_heads_pair, int64_t transition_n,
                            double eps, bool use_mask) {
  const Layout L = make_layout((int)c_z, (int)c_hidden_mul,
                               (int)c_hidden_pair_att, (int)no_heads_pair,
                               (int)transition_n);
  const Shapes sh = check_inputs(z, pair_mask, packed, L);
  const c10::cuda::CUDAGuard guard(z.device());
  const auto stream = c10::cuda::getCurrentCUDAStream();
  auto out = at::empty(z.sizes(), z.options());
  const Scratch sc = make_scratch(z, L, sh.B, sh.S1);
  const auto* zp = reinterpret_cast<const __nv_bfloat16*>(z.const_data_ptr());
  run_transition(zp, packed.const_data_ptr<float>(),
                 reinterpret_cast<const __nv_bfloat16*>(pair_mask.const_data_ptr()),
                 zp, reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), sc, L,
                 sh.B, sh.S1, sh.N, (float)eps, use_mask, stream);
  return out;
}

// The packed-weight layout, so the host builds the buffer from the same
// definition the kernels index it with rather than from a mirrored copy. The two
// trailing rows carry (total, whether this build reproduces the baseline's bf16
// rounding points) and (largest supported c_hidden_pair_att*no_heads_pair, the
// attention kernel's queries per CTA).
at::Tensor pair_block_layout(int64_t c_z, int64_t c_hidden_mul,
                             int64_t c_hidden_pair_att, int64_t no_heads_pair,
                             int64_t transition_n) {
  const Layout L = make_layout((int)c_z, (int)c_hidden_mul,
                               (int)c_hidden_pair_att, (int)no_heads_pair,
                               (int)transition_n);
  auto t = at::empty({L.nsec + 2, 2}, at::TensorOptions().dtype(at::kLong));
  auto* p = t.data_ptr<int64_t>();
  for (int s = 0; s < L.nsec; ++s) {
    p[2 * s] = L.off[s];
    p[2 * s + 1] = L.num[s];
  }
  p[2 * L.nsec] = L.total;
  p[2 * L.nsec + 1] = FK_PB_FAITHFUL;
  p[2 * L.nsec + 2] = kMaxHd;
  p[2 * L.nsec + 3] = kAttnQPB;
  return t;
}

}  // namespace

TORCH_LIBRARY(fk_af3_pair_block, m) {
  m.def(
      "pair_block(Tensor z, Tensor pair_mask, Tensor packed, int c_z, "
      "int c_hidden_mul, int c_hidden_pair_att, int no_heads_pair, "
      "int transition_n, float inf, float eps, bool mask_trans) -> Tensor",
      &pair_block);
  m.def(
      "trimul_stage(Tensor z, Tensor pair_mask, Tensor packed, int c_z, "
      "int c_hidden_mul, int c_hidden_pair_att, int no_heads_pair, "
      "int transition_n, float eps, bool outgoing) -> Tensor",
      &trimul_stage);
  m.def(
      "triatt_stage(Tensor z, Tensor pair_mask, Tensor packed, int c_z, "
      "int c_hidden_mul, int c_hidden_pair_att, int no_heads_pair, "
      "int transition_n, float inf, float eps, bool starting) -> Tensor",
      &triatt_stage);
  m.def(
      "transition_stage(Tensor z, Tensor pair_mask, Tensor packed, int c_z, "
      "int c_hidden_mul, int c_hidden_pair_att, int no_heads_pair, "
      "int transition_n, float eps, bool use_mask) -> Tensor",
      &transition_stage);
  m.def(
      "layout(int c_z, int c_hidden_mul, int c_hidden_pair_att, "
      "int no_heads_pair, int transition_n) -> Tensor",
      &pair_block_layout);
}
