"""Conv2d — hand-written NCHW implicit-GEMM CUDA kernels (Blackwell sm_100).

Drop-in replacement for the ``F.conv2d`` baseline.  The captured workloads are
all small/latency-bound convolutions where cuDNN either picks a poor
``implicit_convolve_sgemm`` variant or pays NCHW<->NHWC transposes around a
CUTLASS kernel; a single NCHW-native implicit-GEMM launch avoids both.
"""

from __future__ import annotations

import hashlib
import os
import subprocess

import torch
import torch.nn as nn
import torch.nn.functional as F

_CUDA_SRC = r"""
// Hand-written NCHW implicit-GEMM conv2d for Blackwell (sm_100).
//
//   GEMM view:  C[Cout, P*Q] = A[Cout, R*S*Cin] @ B[R*S*Cin, P*Q]   (per image)
//     A = weight pre-permuted to [Cout][R][S][Cin]  -> coalesced/vector loads
//     B = im2col of x, gathered on the fly          -> coalesced along q
//     C = out, per image [Cout][P*Q] contiguous     -> coalesced stores
//
// fp16/bf16 accumulate on tensor cores (wmma 16x16x16 -> fp32); fp32 uses a
// SIMT register-tiled inner product.  One kernel launch on the native NCHW
// layout: no NCHW<->NHWC transposes, no im2col workspace.
//
// Three things matter at these (small, latency-bound) sizes and drive the
// shape of the code:
//   * The k loop walks (r, s) outside and input channels inside, so a k-tile
//     has one (r, s) and the gather's row/column bounds plus the h/w address
//     terms are uniform over the tile -- they hoist out, leaving a pointer
//     bump + load + store as the innermost statement.
//   * Every per-thread loop has a compile-time trip count, so it unrolls and a
//     tile's global loads are all in flight together; the tiles themselves are
//     double buffered through registers so a tile's loads overlap the previous
//     tile's math instead of serialising on it.
//   * A 1x1 stride-1 conv needs no gather at all: both mma operands are plain
//     row-major matrices in global memory, so that case runs a separate
//     barrier-free kernel.

#include <torch/extension.h>

#include <array>
#include <map>
#include <mutex>
#include <vector>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>

#define DIVUP(a, b) (((a) + (b) - 1) / (b))

struct CP {
  int C, H, W, Cout, P, Q, R, S;
  int sh, sw, ph, pw, dh, dw;
  int HW, PQ, RS, Kc;
  int Cg, Coutg, NCT;   // per-group channels, per-group out channels, ci tiles
};

// ---------------------------------------------------------------------------
// general implicit-GEMM tensor-core kernel (fp16 / bf16)
// ---------------------------------------------------------------------------
template <typename T, int BM, int BN, int BK, int WM, int WN>
__global__ __launch_bounds__(32 * (BM / WM) * (BN / WN))
void conv_mma_kernel(const T* __restrict__ X, const T* __restrict__ Wt,
                     const T* __restrict__ Bias, T* __restrict__ Out, CP p) {
  using namespace nvcuda::wmma;
  constexpr int NWM = BM / WM, NWN = BN / WN, NW = NWM * NWN, NT = 32 * NW;
  constexpr int LDA = BK + 8;
  constexpr int LDB = BN + 8;
  constexpr int SA = BM * LDA, SB = BK * LDB;
  constexpr int EPN = NW * 16 * WN;
  constexpr int STAGE = (int)(2 * (SA + SB)) * (int)sizeof(T);
  constexpr int RAW = STAGE > EPN * 4 ? STAGE : EPN * 4;
  constexpr int BNT = BN < NT ? BN : NT;
  constexpr int NJ = BN / BNT;
  constexpr int KSTEP = NT / BNT;
  constexpr int NK = BK / KSTEP;
  constexpr int NA = BM * BK / NT;
  static_assert(BM * BK % NT == 0, "A tile must divide across threads");
  static_assert(BK % KSTEP == 0, "bad B tile split");
  static_assert(RAW <= 200 * 1024, "shared memory limit");

  __shared__ __align__(32) char raw[RAW];
  T* As0 = reinterpret_cast<T*>(raw);
  T* Bs0 = As0 + 2 * SA;
  float* Ep = reinterpret_cast<float*>(raw);

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int wm = warp / NWN, wn = warp % NWN;
  const int m0 = blockIdx.y * BM;
  const int n0 = blockIdx.x * BN;
  const int nb = blockIdx.z;
  const int grp = p.Coutg == p.Cout ? 0 : (m0 / p.Coutg);

  fragment<accumulator, 16, 16, 16, float> acc[WM / 16][WN / 16];
#pragma unroll
  for (int i = 0; i < WM / 16; i++)
#pragma unroll
    for (int j = 0; j < WN / 16; j++) fill_fragment(acc[i][j], 0.0f);

  // ---- addressing, hoisted out of the k loop ------------------------------
  // Out-of-range rows/columns are clamped rather than masked: they compute
  // garbage into accumulator slots the epilogue never stores.
  const T* Xb = X + ((long)nb * p.C + (long)grp * p.Cg) * p.HW;
  const int jj = tid % BNT;
  const int kbase = tid / BNT;
  int hb[NJ], wb[NJ];
#pragma unroll
  for (int u = 0; u < NJ; u++) {
    const int nnr = n0 + u * BNT + jj;
    const int nn = nnr < p.PQ ? nnr : p.PQ - 1;
    const int pp = nn / p.Q;
    hb[u] = pp * p.sh - p.ph;
    wb[u] = (nn - pp * p.Q) * p.sw - p.pw;
  }
  // A tile: thread tid owns column (tid % BK) of rows (tid / BK) + t * MSTEP, so
  // its NA elements are one base pointer plus a fixed row stride -- no arrays.
  constexpr int MSTEP = NT / BK;
  static_assert(NT % BK == 0, "A tile rows must divide across threads");
  const int akk = tid % BK;
  const int ammb = tid / BK;
  const long astride = (long)MSTEP * p.Kc;
  const T* ap0 = Wt + (long)(m0 + ammb < p.Cout ? m0 + ammb : p.Cout - 1) * p.Kc
                 + akk;
  const bool kall = p.Cg % BK == 0;
  const bool afast = (m0 + BM <= p.Cout) && kall;
  const long hstride = (long)KSTEP * p.HW;

  const int ntile = p.RS * p.NCT;
  T ra[NA], rb[NJ * NK];

  // Tile coordinates walk incrementally (cit within a (r, s), then s, then r):
  // the obvious it/NCT and rs/S divisions cost ~58 SASS instructions a tile.
  int cit = 0, trs = 0, tr = 0, ts = 0;
#define FK_ADVANCE_TILE                                                       \
  {                                                                           \
    if (++cit == p.NCT) {                                                     \
      cit = 0;                                                                \
      trs++;                                                                  \
      if (++ts == p.S) { ts = 0; tr++; }                                      \
    }                                                                         \
  }

  // one staged tile (the one cit/trs/tr/ts point at): global -> registers
#define FK_LOAD_TILE                                                          \
  {                                                                           \
    const int ci0_ = cit * BK;                                                \
    const int r_ = tr, s_ = ts;                                               \
    const int koff_ = trs * p.Cg + ci0_;                                      \
    if (afast) {                                                          \
      const T* aq_ = ap0 + koff_;                                         \
      _Pragma("unroll")                                                   \
      for (int t = 0; t < NA; t++) { ra[t] = aq_[0]; aq_ += astride; }\
    } else {                                                              \
      const bool cok_ = kall || ci0_ + akk < p.Cg;                        \
      _Pragma("unroll")                                                   \
      for (int t = 0; t < NA; t++) {                                      \
        const int gm_ = m0 + ammb + t * MSTEP;                            \
        ra[t] = (gm_ < p.Cout && cok_)                                    \
                    ? Wt[(long)gm_ * p.Kc + koff_ + akk] : (T)0.f;        \
      }                                                                   \
    }                                                                     \
    const int dr_ = r_ * p.dh, ds_ = s_ * p.dw;                               \
    _Pragma("unroll")                                                         \
    for (int u = 0; u < NJ; u++) {                                            \
      const int h_ = hb[u] + dr_, w_ = wb[u] + ds_;                           \
      const bool oku_ = (unsigned)h_ < (unsigned)p.H &&                       \
                        (unsigned)w_ < (unsigned)p.W;                         \
      const T* pu_ = Xb + (ci0_ * p.HW + h_ * p.W + w_);                      \
      if (!oku_) {                                                        \
        _Pragma("unroll")                                                 \
        for (int t = 0; t < NK; t++) rb[u * NK + t] = (T)0.f;             \
      } else if (kall) {                                                  \
        const T* pq_ = pu_ + kbase * p.HW;                                \
        _Pragma("unroll")                                                 \
        for (int t = 0; t < NK; t++) {                                    \
          rb[u * NK + t] = pq_[0];                                        \
          pq_ += hstride;                                                 \
        }                                                                 \
      } else {                                                            \
        _Pragma("unroll")                                                 \
        for (int t = 0; t < NK; t++) {                                    \
          const int kk_ = kbase + t * KSTEP;                              \
          rb[u * NK + t] = ci0_ + kk_ < p.Cg ? pu_[kk_ * p.HW] : (T)0.f;  \
        }                                                                 \
      }                                                                   \
    }                                                                         \
  }

  // registers -> shared buffer BUF
#define FK_STORE_TILE(BUF)                                                    \
  {                                                                           \
    T* as_ = As0 + (BUF)*SA;                                                  \
    T* bs_ = Bs0 + (BUF)*SB + jj;                                             \
    _Pragma("unroll")                                                         \
    for (int t = 0; t < NA; t++)                                              \
      as_[(ammb + t * MSTEP) * LDA + akk] = ra[t];                   \
    _Pragma("unroll")                                                         \
    for (int u = 0; u < NJ; u++)                                              \
      _Pragma("unroll")                                                       \
      for (int t = 0; t < NK; t++)                                            \
        bs_[(kbase + t * KSTEP) * LDB + u * BNT] = rb[u * NK + t];       \
  }

  FK_LOAD_TILE
  FK_ADVANCE_TILE
  FK_STORE_TILE(0)
  __syncthreads();

#pragma unroll 2
  for (int it = 0; it < ntile; it++) {
    const int cur = it & 1;
    if (it + 1 < ntile) { FK_LOAD_TILE FK_ADVANCE_TILE }
    const T* as = As0 + cur * SA;
    const T* bs = Bs0 + cur * SB;
#pragma unroll
    for (int kt = 0; kt < BK / 16; kt++) {
      fragment<matrix_a, 16, 16, 16, T, row_major> af[WM / 16];
      fragment<matrix_b, 16, 16, 16, T, row_major> bf[WN / 16];
#pragma unroll
      for (int i = 0; i < WM / 16; i++)
        load_matrix_sync(af[i], as + (wm * WM + i * 16) * LDA + kt * 16, LDA);
#pragma unroll
      for (int j = 0; j < WN / 16; j++)
        load_matrix_sync(bf[j], bs + (kt * 16) * LDB + wn * WN + j * 16, LDB);
#pragma unroll
      for (int i = 0; i < WM / 16; i++)
#pragma unroll
        for (int j = 0; j < WN / 16; j++)
          mma_sync(acc[i][j], af[i], bf[j], acc[i][j]);
    }
    if (it + 1 < ntile) FK_STORE_TILE(1 - cur)
    __syncthreads();
  }
#undef FK_LOAD_TILE
#undef FK_ADVANCE_TILE
#undef FK_STORE_TILE

  // ---- epilogue -----------------------------------------------------------
  float* ep = Ep + warp * 16 * WN;
  T* Ob = Out + (long)nb * p.Cout * p.PQ;
  const bool vec = (p.PQ & 7) == 0;
#pragma unroll
  for (int i = 0; i < WM / 16; i++) {
#pragma unroll
    for (int j = 0; j < WN / 16; j++)
      store_matrix_sync(ep + j * 16, acc[i][j], WN, mem_row_major);
    __syncwarp();
    const int co0 = m0 + wm * WM + i * 16;
    if (vec) {
      constexpr int RUNS = 16 * WN / 8;
#pragma unroll
      for (int t = 0; t < DIVUP(RUNS, 32); t++) {
        const int run = lane + t * 32;
        if (RUNS % 32 != 0 && run >= RUNS) break;
        const int rr = run / (WN / 8), cc = (run - rr * (WN / 8)) * 8;
        const int co = co0 + rr;
        const int nn = n0 + wn * WN + cc;
        if (co < p.Cout && nn + 8 <= p.PQ) {
          const float* sp = ep + rr * WN + cc;
          T v[8];
#pragma unroll
          for (int e = 0; e < 8; e++) {
            float f = sp[e];
            if (Bias) f += (float)Bias[co];
            v[e] = (T)f;
          }
          *reinterpret_cast<uint4*>(Ob + (long)co * p.PQ + nn) =
              *reinterpret_cast<const uint4*>(v);
        } else if (co < p.Cout) {
#pragma unroll
          for (int e = 0; e < 8; e++)
            if (nn + e < p.PQ) {
              float f = ep[rr * WN + cc + e];
              if (Bias) f += (float)Bias[co];
              Ob[(long)co * p.PQ + nn + e] = (T)f;
            }
        }
      }
    } else {
#pragma unroll
      for (int t = 0; t < 16 * WN / 32; t++) {
        const int idx = lane + t * 32;
        const int rr = idx / WN, cc = idx - rr * WN;
        const int co = co0 + rr;
        const int nn = n0 + wn * WN + cc;
        if (co < p.Cout && nn < p.PQ) {
          float f = ep[idx];
          if (Bias) f += (float)Bias[co];
          Ob[(long)co * p.PQ + nn] = (T)f;
        }
      }
    }
    __syncwarp();
  }
}

// ---------------------------------------------------------------------------
// 1x1 stride-1 tensor-core kernel.  Here B *is* x: per image x is a row-major
// [Cin, H*W] matrix and the weight a row-major [Cout, Cin] one, so both mma
// operands load straight from global memory -- no im2col, no shared staging,
// no barriers.  One warp owns one MWxNWt output tile and the whole k reduction
// is a flat stream of independent loads.
// ---------------------------------------------------------------------------
template <typename T, int MW, int NWt, int NWARP>
__global__ __launch_bounds__(32 * NWARP)
void conv1x1_direct_kernel(const T* __restrict__ X, const T* __restrict__ Wt,
                           const T* __restrict__ Bias, T* __restrict__ Out,
                           CP p) {
  using namespace nvcuda::wmma;
  __shared__ __align__(32) float ep_all[NWARP * 16 * NWt];

  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int m0 = (blockIdx.y * NWARP + warp) * MW;
  const int n0 = blockIdx.x * NWt;
  const int nb = blockIdx.z;
  if (m0 >= p.Cout) return;

  fragment<accumulator, 16, 16, 16, float> acc[MW / 16][NWt / 16];
#pragma unroll
  for (int i = 0; i < MW / 16; i++)
#pragma unroll
    for (int j = 0; j < NWt / 16; j++) fill_fragment(acc[i][j], 0.0f);

  const T* aq = Wt + (long)m0 * p.Cg;
  const T* bq = X + (long)nb * p.C * p.HW + n0;
  const int astep = 16 * p.Cg, bstep = 16 * p.HW;
#pragma unroll 8
  for (int k0 = 0; k0 < p.Cg; k0 += 16) {
    fragment<matrix_a, 16, 16, 16, T, row_major> af[MW / 16];
    fragment<matrix_b, 16, 16, 16, T, row_major> bf[NWt / 16];
#pragma unroll
    for (int i = 0; i < MW / 16; i++)
      load_matrix_sync(af[i], aq + i * astep, p.Cg);
#pragma unroll
    for (int j = 0; j < NWt / 16; j++)
      load_matrix_sync(bf[j], bq + j * 16, p.HW);
#pragma unroll
    for (int i = 0; i < MW / 16; i++)
#pragma unroll
      for (int j = 0; j < NWt / 16; j++)
        mma_sync(acc[i][j], af[i], bf[j], acc[i][j]);
    aq += 16;
    bq += bstep;
  }

  float* ep = ep_all + warp * 16 * NWt;
  T* Ob = Out + (long)nb * p.Cout * p.PQ + n0;
#pragma unroll
  for (int i = 0; i < MW / 16; i++) {
#pragma unroll
    for (int j = 0; j < NWt / 16; j++)
      store_matrix_sync(ep + j * 16, acc[i][j], NWt, mem_row_major);
    __syncwarp();
    constexpr int RUNS = 16 * NWt / 8;
#pragma unroll
    for (int t = 0; t < DIVUP(RUNS, 32); t++) {
      const int run = lane + t * 32;
      if (RUNS % 32 != 0 && run >= RUNS) break;
      const int rr = run / (NWt / 8), cc = (run - rr * (NWt / 8)) * 8;
      const int co = m0 + i * 16 + rr;
      const float* sp = ep + rr * NWt + cc;
      T v[8];
#pragma unroll
      for (int e = 0; e < 8; e++) {
        float f = sp[e];
        if (Bias) f += (float)Bias[co];
        v[e] = (T)f;
      }
      *reinterpret_cast<uint4*>(Ob + (long)co * p.PQ + cc) =
          *reinterpret_cast<const uint4*>(v);
    }
    __syncwarp();
  }
}

// ---------------------------------------------------------------------------
// 1x1 stride-1 conv == batched GEMM: per image x is a row-major [Cin, H*W]
// matrix and the weight a row-major [Cout, Cin] one.  With no gather in the
// way both operands stage into shared memory with 16-byte vector loads, which
// is what the im2col path cannot do (the padding offset s - pw misaligns it),
// so this case gets its own kernel.
// ---------------------------------------------------------------------------
template <typename T, int BM, int BN, int BK, int WM, int WN>
__global__ __launch_bounds__(32 * (BM / WM) * (BN / WN))
void gemm1x1_kernel(const T* __restrict__ X, const T* __restrict__ Wt,
                    const T* __restrict__ Bias, T* __restrict__ Out, CP p) {
  using namespace nvcuda::wmma;
  constexpr int NWM = BM / WM, NWN = BN / WN, NW = NWM * NWN, NT = 32 * NW;
  constexpr int LDA = BK + 8;
  constexpr int LDB = BN + 8;
  constexpr int SA = BM * LDA, SB = BK * LDB;
  constexpr int EPN = NW * 16 * WN;
  constexpr int STAGE = (int)(2 * (SA + SB)) * (int)sizeof(T);
  constexpr int RAW = STAGE > EPN * 4 ? STAGE : EPN * 4;
  constexpr int AVG = BK / 8, NAV = BM * AVG / NT, ARSTEP = NT / AVG;
  constexpr int BVG = BN / 8, NBV = BK * BVG / NT, BRSTEP = NT / BVG;
  static_assert(BM * AVG % NT == 0 && NT % AVG == 0, "bad A vector split");
  static_assert(BK * BVG % NT == 0 && NT % BVG == 0, "bad B vector split");
  static_assert(RAW <= 200 * 1024, "shared memory limit");

  __shared__ __align__(32) char raw[RAW];
  T* As0 = reinterpret_cast<T*>(raw);
  T* Bs0 = As0 + 2 * SA;
  float* Ep = reinterpret_cast<float*>(raw);

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int wm = warp / NWN, wn = warp % NWN;
  const int m0 = blockIdx.y * BM;
  const int n0 = blockIdx.x * BN;
  const int nb = blockIdx.z;

  fragment<accumulator, 16, 16, 16, float> acc[WM / 16][WN / 16];
#pragma unroll
  for (int i = 0; i < WM / 16; i++)
#pragma unroll
    for (int j = 0; j < WN / 16; j++) fill_fragment(acc[i][j], 0.0f);

  const int a_c8 = (tid % AVG) * 8, a_r = tid / AVG;
  const int b_c8 = (tid % BVG) * 8, b_r = tid / BVG;
  const int bcol = n0 + b_c8 + 8 <= p.PQ ? n0 + b_c8 : p.PQ - 8;
  const T* Xb = X + (long)nb * p.C * p.HW + bcol;

  // Source pointers for this thread's staging slots, advanced by one k-tile per
  // iteration: every address term that does not depend on k is computed once.
  const T* apt[NAV];
#pragma unroll
  for (int t = 0; t < NAV; t++) {
    const int gm = m0 + a_r + t * ARSTEP;
    apt[t] = Wt + (long)(gm < p.Cout ? gm : p.Cout - 1) * p.Cg + a_c8;
  }
  const T* bpt[NBV];
#pragma unroll
  for (int t = 0; t < NBV; t++) bpt[t] = Xb + (long)(b_r + t * BRSTEP) * p.HW;
  const long bkhw = (long)BK * p.HW;

  uint4 ra[NAV], rb[NBV];
  const uint4 zero4 = make_uint4(0u, 0u, 0u, 0u);

#define FK1_LOAD(K0)                                                          \
  {                                                                           \
    if ((K0) + BK <= p.Cg) {                                                  \
      _Pragma("unroll")                                                       \
      for (int t = 0; t < NAV; t++) {                                         \
        ra[t] = *reinterpret_cast<const uint4*>(apt[t]);                  \
        apt[t] += BK;                                                         \
      }                                                                       \
      _Pragma("unroll")                                                       \
      for (int t = 0; t < NBV; t++) {                                         \
        rb[t] = *reinterpret_cast<const uint4*>(bpt[t]);                  \
        bpt[t] += bkhw;                                                       \
      }                                                                       \
    } else {                                                                  \
      const bool akok_ = (K0) + a_c8 < p.Cg;                                  \
      _Pragma("unroll")                                                       \
      for (int t = 0; t < NAV; t++) {                                         \
        ra[t] = akok_ ? *reinterpret_cast<const uint4*>(apt[t]) : zero4;  \
        apt[t] += BK;                                                         \
      }                                                                       \
      _Pragma("unroll")                                                       \
      for (int t = 0; t < NBV; t++) {                                         \
        rb[t] = ((K0) + b_r + t * BRSTEP < p.Cg)                         \
                         ? *reinterpret_cast<const uint4*>(bpt[t]) : zero4;    \
        bpt[t] += bkhw;                                                       \
      }                                                                       \
    }                                                                         \
  }

#define FK1_STORE(BUF)                                                        \
  {                                                                           \
    T* as_ = As0 + (BUF)*SA;                                                  \
    T* bs_ = Bs0 + (BUF)*SB;                                                  \
    _Pragma("unroll")                                                         \
    for (int t = 0; t < NAV; t++)                                             \
      *reinterpret_cast<uint4*>(as_ + (a_r + t * ARSTEP) * LDA + a_c8) =       \
          ra[t];                                                         \
    _Pragma("unroll")                                                         \
    for (int t = 0; t < NBV; t++)                                             \
      *reinterpret_cast<uint4*>(bs_ + (b_r + t * BRSTEP) * LDB + b_c8) =       \
          rb[t];                                                         \
  }

  FK1_LOAD(0)
  FK1_STORE(0)
  __syncthreads();

#pragma unroll 2
  for (int k0 = 0; k0 < p.Cg; k0 += BK) {
    const int cur = (k0 / BK) & 1;
    if (k0 + BK < p.Cg) FK1_LOAD(k0 + BK)
    const T* as = As0 + cur * SA;
    const T* bs = Bs0 + cur * SB;
#pragma unroll
    for (int kt = 0; kt < BK / 16; kt++) {
      fragment<matrix_a, 16, 16, 16, T, row_major> af[WM / 16];
      fragment<matrix_b, 16, 16, 16, T, row_major> bf[WN / 16];
#pragma unroll
      for (int i = 0; i < WM / 16; i++)
        load_matrix_sync(af[i], as + (wm * WM + i * 16) * LDA + kt * 16, LDA);
#pragma unroll
      for (int j = 0; j < WN / 16; j++)
        load_matrix_sync(bf[j], bs + (kt * 16) * LDB + wn * WN + j * 16, LDB);
#pragma unroll
      for (int i = 0; i < WM / 16; i++)
#pragma unroll
        for (int j = 0; j < WN / 16; j++)
          mma_sync(acc[i][j], af[i], bf[j], acc[i][j]);
    }
    if (k0 + BK < p.Cg) FK1_STORE(1 - cur)
    __syncthreads();
  }
#undef FK1_LOAD
#undef FK1_STORE

  float* ep = Ep + warp * 16 * WN;
  T* Ob = Out + (long)nb * p.Cout * p.PQ;
#pragma unroll
  for (int i = 0; i < WM / 16; i++) {
#pragma unroll
    for (int j = 0; j < WN / 16; j++)
      store_matrix_sync(ep + j * 16, acc[i][j], WN, mem_row_major);
    __syncwarp();
    const int co0 = m0 + wm * WM + i * 16;
    constexpr int RUNS = 16 * WN / 8;
#pragma unroll
    for (int t = 0; t < DIVUP(RUNS, 32); t++) {
      const int run = lane + t * 32;
      if (RUNS % 32 != 0 && run >= RUNS) break;
      const int rr = run / (WN / 8), cc = (run - rr * (WN / 8)) * 8;
      const int co = co0 + rr;
      const int nn = n0 + wn * WN + cc;
      if (co < p.Cout && nn + 8 <= p.PQ) {
        const float* sp = ep + rr * WN + cc;
        T v[8];
#pragma unroll
        for (int e = 0; e < 8; e++) {
          float f = sp[e];
          if (Bias) f += (float)Bias[co];
          v[e] = (T)f;
        }
        *reinterpret_cast<uint4*>(Ob + (long)co * p.PQ + nn) =
            *reinterpret_cast<const uint4*>(v);
      }
    }
    __syncwarp();
  }
}

// ---------------------------------------------------------------------------
// SIMT path (fp32)
// ---------------------------------------------------------------------------
template <int BM, int BN, int BK, int TM, int TN>
__global__ __launch_bounds__((BM / TM) * (BN / TN))
void conv_simt_kernel(const float* __restrict__ X, const float* __restrict__ Wt,
                      const float* __restrict__ Bias, float* __restrict__ Out,
                      CP p) {
  constexpr int TX = BN / TN, TY = BM / TM, NT = TX * TY;
  constexpr int BNT = BN < NT ? BN : NT;
  constexpr int NJ = BN / BNT;
  constexpr int KSTEP = NT / BNT;
  constexpr int NK = BK / KSTEP;
  constexpr int NA = BM * BK / NT;
  static_assert(BM * BK % NT == 0, "A tile must divide across threads");

  __shared__ float As[2][BK][BM + 1];
  __shared__ float Bsh[2][BK][BN + 1];

  const int tid = threadIdx.x;
  const int tx = tid % TX, ty = tid / TX;
  const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN, nb = blockIdx.z;
  const int grp = p.Coutg == p.Cout ? 0 : (m0 / p.Coutg);

  float acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; i++)
#pragma unroll
    for (int j = 0; j < TN; j++) acc[i][j] = 0.f;

  const float* Xb = X + ((long)nb * p.C + (long)grp * p.Cg) * p.HW;
  const int jj = tid % BNT, kbase = tid / BNT;
  int hb[NJ], wb[NJ];
#pragma unroll
  for (int u = 0; u < NJ; u++) {
    const int nnr = n0 + u * BNT + jj;
    const int nn = nnr < p.PQ ? nnr : p.PQ - 1;
    const int pp = nn / p.Q;
    hb[u] = pp * p.sh - p.ph;
    wb[u] = (nn - pp * p.Q) * p.sw - p.pw;
  }
  constexpr int MSTEP = NT / BK;
  static_assert(NT % BK == 0, "A tile rows must divide across threads");
  const int akk = tid % BK;
  const int ammb = tid / BK;
  const long astride = (long)MSTEP * p.Kc;
  const float* ap0 =
      Wt + (long)(m0 + ammb < p.Cout ? m0 + ammb : p.Cout - 1) * p.Kc + akk;
  const bool kall = p.Cg % BK == 0;
  const bool afast = (m0 + BM <= p.Cout) && kall;
  const long hstride = (long)KSTEP * p.HW;

  const int ntile = p.RS * p.NCT;
  float ra[NA], rb[NJ * NK];

  // Tile coordinates walk incrementally (cit within a (r, s), then s, then r):
  // the obvious it/NCT and rs/S divisions cost ~58 SASS instructions a tile.
  int cit = 0, trs = 0, tr = 0, ts = 0;
#define FK_ADVANCE_TILE                                                       \
  {                                                                           \
    if (++cit == p.NCT) {                                                     \
      cit = 0;                                                                \
      trs++;                                                                  \
      if (++ts == p.S) { ts = 0; tr++; }                                      \
    }                                                                         \
  }

#define FK_LOAD_TILE                                                          \
  {                                                                           \
    const int ci0_ = cit * BK;                                                \
    const int r_ = tr, s_ = ts;                                               \
    const int koff_ = trs * p.Cg + ci0_;                                      \
    if (afast) {                                                          \
      const float* aq_ = ap0 + koff_;                                     \
      _Pragma("unroll")                                                   \
      for (int t = 0; t < NA; t++) { ra[t] = aq_[0]; aq_ += astride; }\
    } else {                                                              \
      const bool cok_ = kall || ci0_ + akk < p.Cg;                        \
      _Pragma("unroll")                                                   \
      for (int t = 0; t < NA; t++) {                                      \
        const int gm_ = m0 + ammb + t * MSTEP;                            \
        ra[t] = (gm_ < p.Cout && cok_)                                    \
                    ? Wt[(long)gm_ * p.Kc + koff_ + akk] : 0.f;           \
      }                                                                   \
    }                                                                     \
    const int dr_ = r_ * p.dh, ds_ = s_ * p.dw;                               \
    _Pragma("unroll")                                                         \
    for (int u = 0; u < NJ; u++) {                                            \
      const int h_ = hb[u] + dr_, w_ = wb[u] + ds_;                           \
      const bool oku_ = (unsigned)h_ < (unsigned)p.H &&                       \
                        (unsigned)w_ < (unsigned)p.W;                         \
      const float* pu_ = Xb + (ci0_ * p.HW + h_ * p.W + w_);                  \
      if (!oku_) {                                                        \
        _Pragma("unroll")                                                 \
        for (int t = 0; t < NK; t++) rb[u * NK + t] = 0.f;                \
      } else if (kall) {                                                  \
        const float* pq_ = pu_ + kbase * p.HW;                            \
        _Pragma("unroll")                                                 \
        for (int t = 0; t < NK; t++) {                                    \
          rb[u * NK + t] = pq_[0];                                        \
          pq_ += hstride;                                                 \
        }                                                                 \
      } else {                                                            \
        _Pragma("unroll")                                                 \
        for (int t = 0; t < NK; t++) {                                    \
          const int kk_ = kbase + t * KSTEP;                              \
          rb[u * NK + t] = ci0_ + kk_ < p.Cg ? pu_[kk_ * p.HW] : 0.f;     \
        }                                                                 \
      }                                                                   \
    }                                                                         \
  }

#define FK_STORE_TILE(BUF)                                                    \
  {                                                                           \
    _Pragma("unroll")                                                         \
    for (int t = 0; t < NA; t++)                                              \
      As[BUF][akk][ammb + t * MSTEP] = ra[t];                            \
    _Pragma("unroll")                                                         \
    for (int u = 0; u < NJ; u++)                                              \
      _Pragma("unroll")                                                       \
      for (int t = 0; t < NK; t++)                                            \
        Bsh[BUF][kbase + t * KSTEP][u * BNT + jj] = rb[u * NK + t];      \
  }

  FK_LOAD_TILE
  FK_ADVANCE_TILE
  FK_STORE_TILE(0)
  __syncthreads();

#pragma unroll 2
  for (int it = 0; it < ntile; it++) {
    const int cur = it & 1;
    if (it + 1 < ntile) { FK_LOAD_TILE FK_ADVANCE_TILE }
#pragma unroll
    for (int kk = 0; kk < BK; kk++) {
      float a[TM], b[TN];
#pragma unroll
      for (int i = 0; i < TM; i++) a[i] = As[cur][kk][ty * TM + i];
#pragma unroll
      for (int j = 0; j < TN; j++) b[j] = Bsh[cur][kk][tx * TN + j];
#pragma unroll
      for (int i = 0; i < TM; i++)
#pragma unroll
        for (int j = 0; j < TN; j++) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
    }
    if (it + 1 < ntile) FK_STORE_TILE(1 - cur)
    __syncthreads();
  }
#undef FK_LOAD_TILE
#undef FK_ADVANCE_TILE
#undef FK_STORE_TILE

  float* Ob = Out + (long)nb * p.Cout * p.PQ;
#pragma unroll
  for (int i = 0; i < TM; i++) {
    const int co = m0 + ty * TM + i;
    if (co >= p.Cout) continue;
    const float bv = Bias ? Bias[co] : 0.f;
    float* dst = Ob + (long)co * p.PQ + n0 + tx * TN;
    // float4 store needs the *whole* offset 16B-aligned, so PQ (the row
    // stride) must be a multiple of 4 as well as the tile origin.
    if (TN == 4 && (p.PQ & 3) == 0 && n0 + tx * TN + TN <= p.PQ) {
      float4 v;
      v.x = acc[i][0] + bv;
      v.y = acc[i][1] + bv;
      v.z = acc[i][2] + bv;
      v.w = acc[i][3] + bv;
      *reinterpret_cast<float4*>(dst) = v;
    } else {
#pragma unroll
      for (int j = 0; j < TN; j++)
        if (n0 + tx * TN + j < p.PQ) dst[j] = acc[i][j] + bv;
    }
  }
}

// ---------------------------------------------------------------------------
// launch helpers
// ---------------------------------------------------------------------------
#define D1_CFG_LIST \
  D(0, 16, 16, 1)                         \
  D(1, 16, 32, 1)                         \
  D(2, 16, 16, 2)                         \
  D(3, 32, 16, 1)

struct D1Desc { int mw, nw, nwarp; };
static const D1Desc kD1Cfg[] = {
#define D(id, mw, nw, nwarp) {mw, nw, nwarp},
    D1_CFG_LIST
#undef D
};
static const int kND1 = (int)(sizeof(kD1Cfg) / sizeof(D1Desc));

template <typename T>
static bool launch_d1(int cfg, const T* X, const T* Wt, const T* Bias, T* Out,
                      const CP& p, int batch, cudaStream_t st) {
  switch (cfg) {
#define D(id, mw, nw, nwarp)                                                 \
  case id: {                                                                 \
    dim3 g(p.PQ / nw, DIVUP(p.Cout, mw * nwarp), batch);                     \
    conv1x1_direct_kernel<T, mw, nw, nwarp>                                  \
        <<<g, 32 * nwarp, 0, st>>>(X, Wt, Bias, Out, p);                     \
    return true;                                                             \
  }
    D1_CFG_LIST
#undef D
    default:
      return false;
  }
}

// Eligible only when both mma operands are directly loadable: 1x1 stride-1,
// every tile origin / row stride 32B-aligned for load_matrix_sync.
static bool d1_ok(const CP& p, int cfg) {
  if (!(p.R == 1 && p.S == 1 && p.sh == 1 && p.sw == 1 && p.ph == 0 &&
        p.pw == 0 && p.dh == 1 && p.dw == 1 && p.Cg == p.C &&
        p.Coutg == p.Cout))
    return false;
  if (p.Cg % 16 || p.HW % 16 || p.Cout % 16) return false;
  const int nw = kD1Cfg[cfg].nw;
  return p.PQ % nw == 0 && nw % 16 == 0;
}


#define G1_CFG_LIST \
  G(0, 32, 64, 64, 16, 32)                \
  G(1, 16, 64, 64, 16, 32)                \
  G(2, 64, 64, 64, 32, 32)                \
  G(3, 32, 64, 32, 16, 32)                \
  G(4, 64, 128, 32, 32, 32)               \
  G(5, 16, 128, 64, 16, 32)               \
  G(6, 128, 64, 32, 32, 32)               \
  G(7, 32, 32, 32, 16, 32)                \
  G(8, 32, 64, 128, 16, 32)               \
  G(9, 32, 32, 128, 16, 32)               \
  G(10, 16, 32, 128, 16, 16)

struct G1Desc { int bm, bn, bk, nt; };
static const G1Desc kG1Cfg[] = {
#define G(id, bm, bn, bk, wm, wn) {bm, bn, bk, 32 * (bm / wm) * (bn / wn)},
    G1_CFG_LIST
#undef G
};
static const int kNG1 = (int)(sizeof(kG1Cfg) / sizeof(G1Desc));

template <typename T>
static bool launch_g1(int cfg, const T* X, const T* Wt, const T* Bias, T* Out,
                      const CP& p, int batch, cudaStream_t st) {
  switch (cfg) {
#define G(id, bm, bn, bk, wm, wn)                                            \
  case id: {                                                                 \
    dim3 g(DIVUP(p.PQ, bn), DIVUP(p.Cout, bm), batch);                       \
    gemm1x1_kernel<T, bm, bn, bk, wm, wn>                                    \
        <<<g, 32 * (bm / wm) * (bn / wn), 0, st>>>(X, Wt, Bias, Out, p);     \
    return true;                                                             \
  }
    G1_CFG_LIST
#undef G
    default:
      return false;
  }
}

static bool g1_ok(const CP& p) {
  return p.R == 1 && p.S == 1 && p.sh == 1 && p.sw == 1 && p.ph == 0 &&
         p.pw == 0 && p.dh == 1 && p.dw == 1 && p.Cg == p.C &&
         p.Coutg == p.Cout && p.Cg % 8 == 0 && p.HW % 8 == 0 && p.PQ >= 8;
}


#define MMA_CFG_LIST \
  X(0, 32, 256, 16, 16, 64)               \
  X(1, 32, 128, 16, 16, 32)               \
  X(2, 32, 64, 64, 16, 16)                \
  X(3, 16, 32, 64, 16, 16)                \
  X(4, 16, 128, 64, 16, 16)               \
  X(5, 64, 64, 64, 32, 16)                \
  X(6, 16, 128, 16, 16, 32)               \
  X(7, 64, 128, 32, 32, 64)               \
  X(8, 128, 64, 32, 64, 32)               \
  X(9, 16, 64, 16, 16, 16)                \
  X(10, 64, 64, 32, 32, 32)               \
  X(11, 32, 64, 32, 32, 32)               \
  X(12, 16, 256, 16, 16, 64)              \
  X(13, 128, 128, 32, 64, 32)             \
  X(14, 32, 128, 32, 16, 32)              \
  X(15, 64, 256, 32, 32, 64)

#define SIMT_CFG_LIST \
  Y(0, 64, 32, 16, 4, 2)                  \
  Y(1, 64, 64, 16, 4, 4)                  \
  Y(2, 32, 32, 16, 2, 2)                  \
  Y(3, 16, 64, 16, 2, 4)                  \
  Y(4, 128, 32, 16, 8, 2)                 \
  Y(5, 64, 32, 32, 4, 4)                  \
  Y(6, 64, 64, 32, 4, 4)                  \
  Y(7, 32, 64, 32, 4, 4)                  \
  Y(8, 128, 64, 32, 8, 4)                 \
  Y(9, 32, 128, 16, 2, 8)                 \
  Y(10, 64, 128, 32, 4, 8)                \
  Y(11, 16, 32, 32, 2, 4)

struct CfgDesc { int bm, bn, bk, nt; };

static const CfgDesc kMmaCfg[] = {
#define X(id, bm, bn, bk, wm, wn) {bm, bn, bk, 32 * (bm / wm) * (bn / wn)},
    MMA_CFG_LIST
#undef X
};
static const CfgDesc kSimtCfg[] = {
#define Y(id, bm, bn, bk, tm, tn) {bm, bn, bk, (bm / tm) * (bn / tn)},
    SIMT_CFG_LIST
#undef Y
};
static const int kNMma = (int)(sizeof(kMmaCfg) / sizeof(CfgDesc));
static const int kNSimt = (int)(sizeof(kSimtCfg) / sizeof(CfgDesc));


template <typename T>
static bool launch_mma(int cfg, const T* X, const T* Wt, const T* Bias, T* Out,
                       const CP& p, int batch, cudaStream_t st) {
  const int M = p.Cout, Nn = p.PQ;
  switch (cfg) {
#define X(id, bm, bn, bk, wm, wn)                                            \
  case id: {                                                                 \
    dim3 g(DIVUP(Nn, bn), DIVUP(M, bm), batch);                              \
    conv_mma_kernel<T, bm, bn, bk, wm, wn>                                   \
        <<<g, 32 * (bm / wm) * (bn / wn), 0, st>>>(X, Wt, Bias, Out, p);     \
    return true;                                                             \
  }
    MMA_CFG_LIST
#undef X
    default:
      return false;
  }
}

static bool launch_simt(int cfg, const float* X, const float* Wt,
                        const float* Bias, float* Out, const CP& p, int batch,
                        cudaStream_t st) {
  const int M = p.Cout, Nn = p.PQ;
  switch (cfg) {
#define Y(id, bm, bn, bk, tm, tn)                                            \
  case id: {                                                                 \
    dim3 g(DIVUP(Nn, bn), DIVUP(M, bm), batch);                              \
    conv_simt_kernel<bm, bn, bk, tm, tn>                                     \
        <<<g, (bm / tm) * (bn / tn), 0, st>>>(X, Wt, Bias, Out, p);          \
    return true;                                                             \
  }
    SIMT_CFG_LIST
#undef Y
    default:
      return false;
  }
}

// ---------------------------------------------------------------------------
// Config selection.  The five kernels above span tile shapes whose relative
// cost at these sizes is dominated by SM coverage and staging-instruction
// count, which no closed-form model predicted well -- so the first call for a
// given problem shape measures every eligible config (with an L2 flush, to
// match how the op is actually timed) and the winner is cached.
// ---------------------------------------------------------------------------
using CfgKey = std::array<int, 16>;
static std::map<CfgKey, int> g_cfg_cache;
static std::mutex g_cfg_mu;

// code = family * 1000 + id;  families: 0 simt(fp32), 1 mma, 2 direct, 3 gemm
template <typename T>
static bool launch_code(int code, const T* X, const T* Wt, const T* Bias, T* Out,
                        CP p, int batch, cudaStream_t st) {
  const int fam = code / 1000, id = code % 1000;
  if (fam == 1) {
    p.NCT = DIVUP(p.Cg, kMmaCfg[id].bk);
    return launch_mma<T>(id, X, Wt, Bias, Out, p, batch, st);
  }
  if (fam == 2) return launch_d1<T>(id, X, Wt, Bias, Out, p, batch, st);
  if (fam == 3) return launch_g1<T>(id, X, Wt, Bias, Out, p, batch, st);
  return false;
}

static bool launch_code_f32(int code, const float* X, const float* Wt,
                            const float* Bias, float* Out, CP p, int batch,
                            cudaStream_t st) {
  if (code / 1000 != 0) return false;
  const int id = code % 1000;
  p.NCT = DIVUP(p.Cg, kSimtCfg[id].bk);
  return launch_simt(id, X, Wt, Bias, Out, p, batch, st);
}

static std::vector<int> cfg_candidates(const CP& p, bool f32) {
  std::vector<int> v;
  if (f32) {
    for (int i = 0; i < kNSimt; i++) v.push_back(i);
    return v;
  }
  for (int i = 0; i < kNMma; i++) v.push_back(1000 + i);
  for (int i = 0; i < kND1; i++) if (d1_ok(p, i)) v.push_back(2000 + i);
  if (g1_ok(p)) for (int i = 0; i < kNG1; i++) v.push_back(3000 + i);
  return v;
}

template <typename F>
static int autotune(const std::vector<int>& cands, F&& run, const at::Tensor& x,
                    cudaStream_t st) {
  if (cands.empty()) return -1;
  if (cands.size() == 1) return cands[0];
  // Kept alive across tunes: repeatedly allocating a few hundred MB would
  // fragment the caching allocator for whatever runs next.
  static at::Tensor flush;
  static size_t nflush = 0;
  if (!flush.defined() || flush.device() != x.device()) {
    try {
      nflush = 2 * (size_t)at::cuda::getCurrentDeviceProperties()->l2CacheSize;
      flush = at::empty({(long)nflush}, x.options().dtype(at::kByte));
    } catch (...) {
      nflush = 0;
      flush = at::Tensor();
    }
  }
  cudaEvent_t e0, e1;
  if (cudaEventCreate(&e0) != cudaSuccess) return cands[0];
  if (cudaEventCreate(&e1) != cudaSuccess) { cudaEventDestroy(e0); return cands[0]; }
  int best = cands[0];
  float bestms = 3.4e38f;
  for (int code : cands) {
    if (!run(code)) continue;
    if (cudaStreamSynchronize(st) != cudaSuccess || cudaGetLastError() != cudaSuccess)
      continue;
    // Trimmed mean of several flushed reps.  The event timer quantises these
    // durations coarsely, so a plain median puts near-equal configs in a tie;
    // averaging the middle samples resolves them without letting one outlier
    // (fast or slow) decide.
    constexpr int REPS = 15;
    float t[REPS];
    bool bad = false;
    for (int r = 0; r < REPS && !bad; r++) {
      if (nflush) cudaMemsetAsync(flush.data_ptr(), 0, nflush, st);
      cudaEventRecord(e0, st);
      run(code);
      cudaEventRecord(e1, st);
      if (cudaStreamSynchronize(st) != cudaSuccess) { bad = true; break; }
      if (cudaEventElapsedTime(&t[r], e0, e1) != cudaSuccess) { bad = true; break; }
    }
    if (bad) { cudaGetLastError(); continue; }
    for (int i = 1; i < REPS; i++)
      for (int j = i; j > 0 && t[j] < t[j - 1]; j--) {
        const float tmp = t[j]; t[j] = t[j - 1]; t[j - 1] = tmp;
      }
    float acc = 0.f;
    for (int i = REPS / 4; i < REPS - REPS / 4; i++) acc += t[i];
    acc /= (float)(REPS - 2 * (REPS / 4));
    if (acc < bestms) { bestms = acc; best = code; }
    cudaGetLastError();   // never leave a sticky error for the next op
  }
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  cudaGetLastError();
  return best;
}

// ---------------------------------------------------------------------------
// entry point.  ``wp`` is the weight permuted to [Cout, R, S, Cin/groups].
// ---------------------------------------------------------------------------
at::Tensor conv2d_forward(const at::Tensor& x_, const at::Tensor& wp,
                          const c10::optional<at::Tensor>& bias_,
                          int64_t sh, int64_t sw, int64_t ph, int64_t pw,
                          int64_t dh, int64_t dw, int64_t groups,
                          int64_t cfg_override) {
  at::Tensor x = x_.is_contiguous() ? x_ : x_.contiguous();
  const int batch = (int)x.size(0);

  CP p;
  p.C = (int)x.size(1);
  p.H = (int)x.size(2);
  p.W = (int)x.size(3);
  p.Cout = (int)wp.size(0);
  p.R = (int)wp.size(1);
  p.S = (int)wp.size(2);
  p.Cg = (int)wp.size(3);
  p.sh = (int)sh; p.sw = (int)sw;
  p.ph = (int)ph; p.pw = (int)pw;
  p.dh = (int)dh; p.dw = (int)dw;
  p.P = (p.H + 2 * p.ph - p.dh * (p.R - 1) - 1) / p.sh + 1;
  p.Q = (p.W + 2 * p.pw - p.dw * (p.S - 1) - 1) / p.sw + 1;
  p.HW = p.H * p.W;
  p.PQ = p.P * p.Q;
  p.RS = p.R * p.S;
  p.Kc = p.RS * p.Cg;
  p.Coutg = p.Cout / (int)groups;
  p.NCT = 1;

  at::Tensor out = at::empty({batch, p.Cout, p.P, p.Q}, x.options());
  if (p.PQ == 0 || p.Cout == 0 || batch == 0) return out;

  auto st = at::cuda::getCurrentCUDAStream();
  const bool f32 = x.scalar_type() == at::kFloat;
  TORCH_CHECK(f32 || x.scalar_type() == at::kHalf ||
                  x.scalar_type() == at::kBFloat16,
              "conv2d_forward: unsupported dtype");

  auto run = [&](int code) -> bool {
    if (f32) {
      const float* bp = bias_.has_value() ? bias_->data_ptr<float>() : nullptr;
      return launch_code_f32(code, x.data_ptr<float>(), wp.data_ptr<float>(), bp,
                             out.data_ptr<float>(), p, batch, st);
    }
    if (x.scalar_type() == at::kHalf) {
      using T = __half;
      const T* bp = bias_.has_value() ? (const T*)bias_->data_ptr() : nullptr;
      return launch_code<T>(code, (const T*)x.data_ptr(), (const T*)wp.data_ptr(),
                            bp, (T*)out.data_ptr(), p, batch, st);
    }
    using T = __nv_bfloat16;
    const T* bp = bias_.has_value() ? (const T*)bias_->data_ptr() : nullptr;
    return launch_code<T>(code, (const T*)x.data_ptr(), (const T*)wp.data_ptr(),
                          bp, (T*)out.data_ptr(), p, batch, st);
  };

  int code;
  if (cfg_override >= 0) {
    code = (int)cfg_override;
  } else {
    const CfgKey key = {p.C, p.H, p.W, p.Cout, p.R, p.S, p.sh, p.sw, p.ph, p.pw,
                        p.dh, p.dw, (int)groups, batch, (int)x.scalar_type(), 0};
    bool have = false;
    {
      std::lock_guard<std::mutex> lk(g_cfg_mu);
      auto it = g_cfg_cache.find(key);
      if (it != g_cfg_cache.end()) { code = it->second; have = true; }
    }
    if (!have) {
      code = autotune(cfg_candidates(p, f32), run, x, st);
      TORCH_CHECK(code >= 0, "conv2d_forward: no usable config");
      std::lock_guard<std::mutex> lk(g_cfg_mu);
      g_cfg_cache[key] = code;
    }
  }
  const bool ok = run(code);
  TORCH_CHECK(ok, "conv2d_forward: unsupported dtype/config");
  return out;
}

"""

_CPP_SRC = r"""
at::Tensor conv2d_forward(const at::Tensor& x_, const at::Tensor& wp,
                          const c10::optional<at::Tensor>& bias_,
                          int64_t sh, int64_t sw, int64_t ph, int64_t pw,
                          int64_t dh, int64_t dw, int64_t groups,
                          int64_t cfg_override);
"""

_EXT = None
_EXT_FAILED = False
# -1 lets the extension pick (and cache) the fastest tile config for each
# problem shape; set FK_CONV_CFG to pin one when debugging.
_CFG_OVERRIDE = int(os.environ.get("FK_CONV_CFG", "-1"))


def _arch_list() -> str:
    try:
        major, minor = torch.cuda.get_device_capability()
    except Exception:
        return ""
    suffix = "a" if major >= 9 else ""
    return f"{major}.{minor}{suffix}"


def _build_ext():
    from torch.utils.cpp_extension import load_inline

    arch = _arch_list()
    if arch:
        os.environ["TORCH_CUDA_ARCH_LIST"] = arch
    tag = hashlib.sha1(_CUDA_SRC.encode()).hexdigest()[:10]
    return load_inline(
        name=f"fk_conv2d_nchw_{tag}",
        cpp_sources=_CPP_SRC,
        cuda_sources=_CUDA_SRC,
        functions=["conv2d_forward"],
        extra_cflags=["-O3"],
        extra_cuda_cflags=[
            "-O3",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "--expt-relaxed-constexpr",
        ],
        verbose=False,
    )


def _ext():
    """The compiled extension, or None if it cannot be built on this machine.

    A build failure degrades to ``F.conv2d`` instead of breaking the operator.
    """
    global _EXT, _EXT_FAILED
    if _EXT is None and not _EXT_FAILED:
        try:
            _EXT = _build_ext()
        except Exception:
            _EXT_FAILED = True
    return _EXT


_OK_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


class Conv2d(nn.Module):
    """Parametric 2D convolution: stores weight and bias internally."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        groups: int = 1,
        dilation: int | tuple[int, int] = 1,
        bias: bool = True,
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.dilation = dilation

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, *kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        self._fast = None
        self._wp = None
        self._wkey = None

    # -- dispatch ---------------------------------------------------------
    def _eligible(self, x: torch.Tensor) -> bool:
        """Whether the CUDA kernels cover this configuration.

        Anything they do not cover (grouped/depthwise convolutions, padding
        wider than the kernel, exotic dtypes, CPU tensors) falls through to
        ``F.conv2d``; none of the captured shapes for this operator land there.
        """
        w = self.weight
        if not (x.is_cuda and x.dim() == 4 and w.dim() == 4):
            return False
        if x.dtype not in _OK_DTYPES or w.dtype != x.dtype:
            return False
        if self.bias is not None and self.bias.dtype != x.dtype:
            return False
        if self.groups != 1:
            return False
        ph, pw = self.padding
        if ph >= w.shape[2] or pw >= w.shape[3]:
            return False  # heavy padding: rare, leave to torch
        return _ext() is not None

    def _perm_weight(self) -> torch.Tensor:
        """Weight as [Cout, R, S, Cin/groups]; cached until the weight changes."""
        w = self.weight
        key = (w.data_ptr(), w.dtype, w.shape, w._version)
        if self._wkey != key:
            self._wp = w.permute(0, 2, 3, 1).contiguous()
            self._wkey = key
        return self._wp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._fast is None:
            self._fast = self._eligible(x)
        if self._fast:
            sh, sw = self.stride
            ph, pw = self.padding
            dh, dw = self.dilation
            return _ext().conv2d_forward(
                x, self._perm_weight(), self.bias, sh, sw, ph, pw, dh, dw,
                self.groups, _CFG_OVERRIDE)
        return F.conv2d(
            x,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
