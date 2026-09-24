"""YOLOv10 Conv-BN-Act building block -- one fused CUDA kernel.

The baseline runs ``silu(bn(conv(x)))`` as three separate ops.  The captured
shapes are tiny (26 MFLOP .. 1 GFLOP; 3 of the 5 benched cases move under 1 MB),
and the bench flushes the L2 before every timed iteration, so each *additional*
launch in the timed region costs ~6us of measured time before it does any work
of its own -- measured here, ``conv`` alone lands at 12us, ``+bn`` at 27us and
``+silu`` at 33us against a 9us empty-call floor.  The op count, not the
arithmetic, is the whole cost of this block.

So the candidate collapses it into a single kernel launch:

* the convolution is an NCHW-native implicit GEMM on tensor cores (the same
  formulation as the frozen ``L1.conv2d`` winner, which avoids both cuDNN's
  NCHW<->NHWC transposes and its plan lookup),
* the BatchNorm is applied *in the epilogue* straight from the module's
  ``weight/bias/running_mean/running_var`` -- folded per output channel into
  one ``scale``/``shift`` pair in shared memory at block start, so nothing has
  to be precomputed on the host and no cached fold can ever go stale,
* SiLU rides along in the same epilogue through the half-angle identity
  ``silu(x) = h * (1 + tanh(h)), h = x/2`` (one MUFU, as in ``L1.silu``).

Four kernels share that epilogue: a general implicit-GEMM one, and -- for 1x1
stride-1, where the im2col gather disappears -- a barrier-free one that feeds the
MMA straight from global memory, a double-buffered one, and one that stages the
whole reduction in a single shot (the benched 1x1 cases run at 0.1 waves per SM,
where a shorter dependency chain beats everything else: ncu puts a third of
their issue stalls on global latency and a fifth on barriers).  The first call
for a problem shape measures every eligible tile config with a flushed L2 -- the
way the op is actually timed -- and caches the winner.

What the kernels do not cover degrades in three steps: the same kernel without
the BN/act epilogue when torch still has to run the BatchNorm (training mode) or
an activation the epilogue lacks; the frozen ``L1.conv2d`` plus a fused BN+act
elementwise kernel for grouped/depthwise and fp32 (two launches, not three);
and finally the baseline ops themselves on CPU tensors and exotic dtypes.
"""

from __future__ import annotations

import hashlib
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..L1.batch_norm2d import BatchNorm2d
from ..L1.conv2d import Conv2d
from ..L1.silu import SiLU

_CUDA_SRC = r"""
// Fused conv2d + BatchNorm2d(eval) + SiLU for Blackwell (sm_100), NCHW fp16/bf16.
//
//   GEMM view:  C[Cout, P*Q] = A[Cout, R*S*Cin] @ B[R*S*Cin, P*Q]   (per image)
//     A = the conv weight, read in its native [Cout][Cin][R][S] layout or the
//         [Cout][R][S][Cin] permutation (WKC/WKRS carry the two k-strides)
//     B = im2col of x, gathered on the fly
//     C = out, per image [Cout][P*Q] contiguous
//
// The epilogue turns an accumulator into the block's final answer:
//     y = silu(scale[co] * acc + shift[co])
//     scale[co] = bn.w[co] * rsqrt(bn.var[co] + eps)
//     shift[co] = bn.b[co] - bn.mean[co] * scale[co] + conv.bias[co] * scale[co]
// ``scale``/``shift`` are built once per block into shared memory (BM values,
// one thread each) while the mma pipeline is still filling, so folding the BN
// costs nothing measurable and stays exact even if the BN buffers are mutated
// between calls.

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
  int Cg, Coutg, NCT;    // per-group channels, per-group out channels, ci tiles
  int wkc, wkrs;         // weight k-strides: ci and (r,s)
  int act;               // 0 = none, 1 = silu
  int bkind[4];          // dtype of bn weight/bias/mean/var: 0 half 1 float 2 bf16
  float eps;
};

// bn tensors may be fp16/bf16 (params, cast with the module) or fp32 (buffers).
__device__ __forceinline__ float ldf(const void* p, int i, int kind) {
  if (kind == 1) return ((const float*)p)[i];
  if (kind == 0) return __half2float(((const __half*)p)[i]);
  return __bfloat162float(((const __nv_bfloat16*)p)[i]);
}

// silu(x) = h * (1 + tanh(h)), h = x/2 -- one MUFU instead of exp + divide.
__device__ __forceinline__ float silu_f(float x) {
  float h = x * 0.5f, t;
  asm("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(h));
  return fmaf(h, t, h);
}

// Fold BN (+ conv bias) into one scale/shift pair per output channel of the
// block's M tile.  ``BnW == nullptr`` means "no BN" (a fused module): then the
// epilogue is just the conv bias.
template <typename T, int BM>
__device__ __forceinline__ void fold_bn(float* sc, float* sb, int m0,
                                        const void* BnW, const void* BnB,
                                        const void* Rm, const void* Rv,
                                        const T* Bias, const CP& p, int tid,
                                        int nthreads) {
  for (int i = tid; i < BM; i += nthreads) {
    const int co = m0 + i;
    float s = 1.f, b = 0.f;
    if (co < p.Cout) {
      if (BnW != nullptr) {
        s = ldf(BnW, co, p.bkind[0]) * rsqrtf(ldf(Rv, co, p.bkind[3]) + p.eps);
        b = ldf(BnB, co, p.bkind[1]) - ldf(Rm, co, p.bkind[2]) * s;
      }
      if (Bias != nullptr) b += (float)Bias[co] * s;
    }
    sc[i] = s;
    sb[i] = b;
  }
}

__device__ __forceinline__ float epi_apply(float f, float s, float b, int act) {
  f = fmaf(f, s, b);
  return act ? silu_f(f) : f;
}

#define FK_EPI(f, i_) epi_apply((f), sc[i_], sb[i_], p.act)

// ---------------------------------------------------------------------------
// general implicit-GEMM tensor-core kernel (fp16 / bf16)
//
// Three things matter at these (small, latency-bound) sizes and drive the shape
// of the code:
//   * The k loop walks (r, s) outside and input channels inside, so a k-tile
//     has one (r, s) and the gather's row/column bounds plus the h/w address
//     terms are uniform over the tile -- they hoist out, leaving a pointer
//     bump + load + store as the innermost statement.
//   * Every per-thread loop has a compile-time trip count, so it unrolls and a
//     tile's global loads are all in flight together; the tiles themselves are
//     double buffered through registers so a tile's loads overlap the previous
//     tile's math instead of serialising on it.
//   * Out-of-range rows/columns are clamped rather than masked: they compute
//     garbage into accumulator slots the epilogue never stores.
// ---------------------------------------------------------------------------
template <typename T, int BM, int BN, int BK, int WM, int WN>
__global__ __launch_bounds__(32 * (BM / WM) * (BN / WN))
void conv_mma_kernel(const T* __restrict__ X, const T* __restrict__ Wt,
                     const T* __restrict__ Bias, const void* __restrict__ BnW,
                     const void* __restrict__ BnB, const void* __restrict__ Rm,
                     const void* __restrict__ Rv, T* __restrict__ Out, CP p) {
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
  __shared__ float sc[BM], sb[BM];
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

  fold_bn<T, BM>(sc, sb, m0, BnW, BnB, Rm, Rv, Bias, p, tid, NT);

  fragment<accumulator, 16, 16, 16, float> acc[WM / 16][WN / 16];
#pragma unroll
  for (int i = 0; i < WM / 16; i++)
#pragma unroll
    for (int j = 0; j < WN / 16; j++) fill_fragment(acc[i][j], 0.0f);

  // ---- addressing, hoisted out of the k loop ------------------------------
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
                 + (long)akk * p.wkc;
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
    const int koff_ = trs * p.wkrs + ci0_ * p.wkc;                            \
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
                    ? Wt[(long)gm_ * p.Kc + koff_ + (long)akk * p.wkc]    \
                    : (T)0.f;                                             \
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

  // ---- epilogue: BN fold + activation, then coalesced (vector) stores -----
  float* ep = Ep + warp * 16 * WN;
  T* Ob = Out + (long)nb * p.Cout * p.PQ;
  const bool vec = (p.PQ & 7) == 0;
#pragma unroll
  for (int i = 0; i < WM / 16; i++) {
#pragma unroll
    for (int j = 0; j < WN / 16; j++)
      store_matrix_sync(ep + j * 16, acc[i][j], WN, mem_row_major);
    __syncwarp();
    const int mi0 = wm * WM + i * 16;
    const int co0 = m0 + mi0;
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
          for (int e = 0; e < 8; e++) v[e] = (T)FK_EPI(sp[e], mi0 + rr);
          *reinterpret_cast<uint4*>(Ob + (long)co * p.PQ + nn) =
              *reinterpret_cast<const uint4*>(v);
        } else if (co < p.Cout) {
#pragma unroll
          for (int e = 0; e < 8; e++)
            if (nn + e < p.PQ)
              Ob[(long)co * p.PQ + nn + e] =
                  (T)FK_EPI(ep[rr * WN + cc + e], mi0 + rr);
        }
      }
    } else {
#pragma unroll
      for (int t = 0; t < 16 * WN / 32; t++) {
        const int idx = lane + t * 32;
        const int rr = idx / WN, cc = idx - rr * WN;
        const int co = co0 + rr;
        const int nn = n0 + wn * WN + cc;
        if (co < p.Cout && nn < p.PQ)
          Ob[(long)co * p.PQ + nn] = (T)FK_EPI(ep[idx], mi0 + rr);
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
                           const T* __restrict__ Bias,
                           const void* __restrict__ BnW,
                           const void* __restrict__ BnB,
                           const void* __restrict__ Rm,
                           const void* __restrict__ Rv, T* __restrict__ Out,
                           CP p) {
  using namespace nvcuda::wmma;
  constexpr int MB = MW * NWARP;
  __shared__ __align__(32) float ep_all[NWARP * 16 * NWt];
  __shared__ float sc[MB], sb[MB];

  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int mblk = blockIdx.y * MB;
  const int m0 = mblk + warp * MW;
  const int n0 = blockIdx.x * NWt;
  const int nb = blockIdx.z;

  fold_bn<T, MB>(sc, sb, mblk, BnW, BnB, Rm, Rv, Bias, p, threadIdx.x,
                 32 * NWARP);

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
  __syncthreads();   // sc/sb visible
#pragma unroll
  for (int i = 0; i < MW / 16; i++) {
#pragma unroll
    for (int j = 0; j < NWt / 16; j++)
      store_matrix_sync(ep + j * 16, acc[i][j], NWt, mem_row_major);
    __syncwarp();
    const int mi0 = warp * MW + i * 16;
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
      for (int e = 0; e < 8; e++) v[e] = (T)FK_EPI(sp[e], mi0 + rr);
      *reinterpret_cast<uint4*>(Ob + (long)co * p.PQ + cc) =
          *reinterpret_cast<const uint4*>(v);
    }
    __syncwarp();
  }
}

// ---------------------------------------------------------------------------
// 1x1 stride-1 conv == batched GEMM: with no gather in the way both operands
// stage into shared memory with 16-byte vector loads, which is what the im2col
// path cannot do (the padding offset s - pw misaligns it), so this case gets
// its own kernel.
// ---------------------------------------------------------------------------
template <typename T, int BM, int BN, int BK, int WM, int WN>
__global__ __launch_bounds__(32 * (BM / WM) * (BN / WN))
void gemm1x1_kernel(const T* __restrict__ X, const T* __restrict__ Wt,
                    const T* __restrict__ Bias, const void* __restrict__ BnW,
                    const void* __restrict__ BnB, const void* __restrict__ Rm,
                    const void* __restrict__ Rv, T* __restrict__ Out, CP p) {
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
  __shared__ float sc[BM], sb[BM];
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

  fold_bn<T, BM>(sc, sb, m0, BnW, BnB, Rm, Rv, Bias, p, tid, NT);

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
    const int mi0 = wm * WM + i * 16;
    const int co0 = m0 + mi0;
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
        for (int e = 0; e < 8; e++) v[e] = (T)FK_EPI(sp[e], mi0 + rr);
        *reinterpret_cast<uint4*>(Ob + (long)co * p.PQ + nn) =
            *reinterpret_cast<const uint4*>(v);
      }
    }
    __syncwarp();
  }
}

// ---------------------------------------------------------------------------
// 1x1 stride-1, whole-K-in-one-stage tensor-core kernel.
//
// The captured 1x1 shapes are so small (0.1 waves per SM) that nothing can be
// hidden behind anything else: ncu puts ~33% of the issue stalls on global
// latency, ~20% on barriers and ~16% on shared latency.  With BK >= Cin the
// block's whole A and B tile fits in one register-staged batch, so the kernel
// degenerates to exactly one global round trip, one barrier and one shared
// round trip -- half the chain of the double-buffered loop above, and a small
// enough body that its instructions survive the cold i-cache.
// ---------------------------------------------------------------------------
template <typename T, int BM, int BN, int BK, int WM, int WN>
__global__ __launch_bounds__(32 * (BM / WM) * (BN / WN))
void gemm1x1_s1_kernel(const T* __restrict__ X, const T* __restrict__ Wt,
                       const T* __restrict__ Bias, const void* __restrict__ BnW,
                       const void* __restrict__ BnB, const void* __restrict__ Rm,
                       const void* __restrict__ Rv, T* __restrict__ Out, CP p) {
  using namespace nvcuda::wmma;
  constexpr int NWM = BM / WM, NWN = BN / WN, NW = NWM * NWN, NT = 32 * NW;
  constexpr int LDA = BK + 8;
  constexpr int LDB = BN + 8;
  constexpr int SA = BM * LDA, SB = BK * LDB;
  constexpr int EPN = NW * 16 * WN;
  constexpr int STAGE = (int)(SA + SB) * (int)sizeof(T);
  constexpr int RAW = STAGE > EPN * 4 ? STAGE : EPN * 4;
  constexpr int AVG = BK / 8, NAV = BM * AVG / NT, ARSTEP = NT / AVG;
  constexpr int BVG = BN / 8, NBV = BK * BVG / NT, BRSTEP = NT / BVG;
  static_assert(BM * AVG % NT == 0 && NT % AVG == 0, "bad A vector split");
  static_assert(BK * BVG % NT == 0 && NT % BVG == 0, "bad B vector split");
  static_assert(RAW <= 47 * 1024, "static shared memory limit");

  __shared__ __align__(32) char raw[RAW];
  __shared__ float sc[BM], sb[BM];
  T* As = reinterpret_cast<T*>(raw);
  T* Bs = As + SA;
  float* Ep = reinterpret_cast<float*>(raw);

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int wm = warp / NWN, wn = warp % NWN;
  const int m0 = blockIdx.y * BM;
  const int n0 = blockIdx.x * BN;
  const int nb = blockIdx.z;

  fold_bn<T, BM>(sc, sb, m0, BnW, BnB, Rm, Rv, Bias, p, tid, NT);

  const int a_c8 = (tid % AVG) * 8, a_r = tid / AVG;
  const int b_c8 = (tid % BVG) * 8, b_r = tid / BVG;
  const int bcol = n0 + b_c8 + 8 <= p.PQ ? n0 + b_c8 : p.PQ - 8;
  const T* Xb = X + (long)nb * p.C * p.HW + bcol;

  // One batch of 16B loads for the whole tile: every address is independent, so
  // they all sit in flight together and cost a single memory latency.
  uint4 ra[NAV], rb[NBV];
  const uint4 zero4 = make_uint4(0u, 0u, 0u, 0u);
  const bool akok = a_c8 < p.Cg;
#pragma unroll
  for (int t = 0; t < NAV; t++) {
    const int gm = m0 + a_r + t * ARSTEP;
    ra[t] = (akok && gm < p.Cout)
                ? *reinterpret_cast<const uint4*>(Wt + (long)gm * p.Cg + a_c8)
                : zero4;
  }
#pragma unroll
  for (int t = 0; t < NBV; t++) {
    const int k = b_r + t * BRSTEP;
    rb[t] = k < p.Cg ? *reinterpret_cast<const uint4*>(Xb + (long)k * p.HW)
                     : zero4;
  }
#pragma unroll
  for (int t = 0; t < NAV; t++)
    *reinterpret_cast<uint4*>(As + (a_r + t * ARSTEP) * LDA + a_c8) = ra[t];
#pragma unroll
  for (int t = 0; t < NBV; t++)
    *reinterpret_cast<uint4*>(Bs + (b_r + t * BRSTEP) * LDB + b_c8) = rb[t];
  __syncthreads();

  fragment<accumulator, 16, 16, 16, float> acc[WM / 16][WN / 16];
#pragma unroll
  for (int i = 0; i < WM / 16; i++)
#pragma unroll
    for (int j = 0; j < WN / 16; j++) fill_fragment(acc[i][j], 0.0f);

#pragma unroll
  for (int kt = 0; kt < BK / 16; kt++) {
    if (kt * 16 >= p.Cg) break;
    fragment<matrix_a, 16, 16, 16, T, row_major> af[WM / 16];
    fragment<matrix_b, 16, 16, 16, T, row_major> bf[WN / 16];
#pragma unroll
    for (int i = 0; i < WM / 16; i++)
      load_matrix_sync(af[i], As + (wm * WM + i * 16) * LDA + kt * 16, LDA);
#pragma unroll
    for (int j = 0; j < WN / 16; j++)
      load_matrix_sync(bf[j], Bs + (kt * 16) * LDB + wn * WN + j * 16, LDB);
#pragma unroll
    for (int i = 0; i < WM / 16; i++)
#pragma unroll
      for (int j = 0; j < WN / 16; j++)
        mma_sync(acc[i][j], af[i], bf[j], acc[i][j]);
  }
  __syncthreads();   // the epilogue reuses the staging buffer

  float* ep = Ep + warp * 16 * WN;
  T* Ob = Out + (long)nb * p.Cout * p.PQ;
#pragma unroll
  for (int i = 0; i < WM / 16; i++) {
#pragma unroll
    for (int j = 0; j < WN / 16; j++)
      store_matrix_sync(ep + j * 16, acc[i][j], WN, mem_row_major);
    __syncwarp();
    const int mi0 = wm * WM + i * 16;
    const int co0 = m0 + mi0;
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
        for (int e = 0; e < 8; e++) v[e] = (T)FK_EPI(sp[e], mi0 + rr);
        *reinterpret_cast<uint4*>(Ob + (long)co * p.PQ + nn) =
            *reinterpret_cast<const uint4*>(v);
      }
    }
    __syncwarp();
  }
}

// ---------------------------------------------------------------------------
// BN + activation alone, in place -- the epilogue of the fallback path (grouped
// convolutions, fp32, a non-SiLU activation), so even there the block costs two
// launches instead of the baseline's three.
// ---------------------------------------------------------------------------
template <typename T, int U>
__global__ __launch_bounds__(256)
void bn_act_kernel(T* __restrict__ Y, const void* __restrict__ BnW,
                   const void* __restrict__ BnB, const void* __restrict__ Rm,
                   const void* __restrict__ Rv, int hwv, int Cout, int nvec,
                   CP p) {
  // One 16B vector (8 halves / 4 floats) per thread per unroll step; HW is a
  // multiple of the vector width, so a vector never straddles two channels.
  constexpr int VE = 16 / (int)sizeof(T);
  int i = blockIdx.x * (256 * U) + threadIdx.x;
#pragma unroll
  for (int u = 0; u < U; u++, i += 256) {
    if (i >= nvec) return;
    const int co = (i / hwv) % Cout;
    float s = 1.f, b = 0.f;
    if (BnW != nullptr) {
      s = ldf(BnW, co, p.bkind[0]) * rsqrtf(ldf(Rv, co, p.bkind[3]) + p.eps);
      b = ldf(BnB, co, p.bkind[1]) - ldf(Rm, co, p.bkind[2]) * s;
    }
    T* q = Y + (long)i * VE;
    uint4 v = *reinterpret_cast<const uint4*>(q);
    T* e = reinterpret_cast<T*>(&v);
#pragma unroll
    for (int t = 0; t < VE; t++) e[t] = (T)epi_apply((float)e[t], s, b, p.act);
    *reinterpret_cast<uint4*>(q) = v;
  }
}

// ---------------------------------------------------------------------------
// launch helpers
// ---------------------------------------------------------------------------
#define FK_KARGS X, Wt, Bias, BnW, BnB, Rm, Rv, Out

#define D1_CFG_LIST \
  D(0, 16, 16, 1)                         \
  D(1, 16, 32, 1)                         \
  D(2, 16, 16, 2)                         \
  D(3, 32, 16, 1)                         \
  D(4, 16, 32, 2)

struct D1Desc { int mw, nw, nwarp; };
static const D1Desc kD1Cfg[] = {
#define D(id, mw, nw, nwarp) {mw, nw, nwarp},
    D1_CFG_LIST
#undef D
};
static const int kND1 = (int)(sizeof(kD1Cfg) / sizeof(D1Desc));

template <typename T>
static bool launch_d1(int cfg, const T* X, const T* Wt, const T* Bias,
                      const void* BnW, const void* BnB, const void* Rm,
                      const void* Rv, T* Out, const CP& p, int batch,
                      cudaStream_t st) {
  switch (cfg) {
#define D(id, mw, nw, nwarp)                                                 \
  case id: {                                                                 \
    dim3 g(p.PQ / nw, DIVUP(p.Cout, mw * nwarp), batch);                     \
    conv1x1_direct_kernel<T, mw, nw, nwarp>                                  \
        <<<g, 32 * nwarp, 0, st>>>(FK_KARGS, p);                             \
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
  // The epilogue stores a whole MW x NWt tile unguarded, so both grid
  // dimensions must divide exactly -- a partial M tile would write past this
  // image's channels into the next image.
  const D1Desc& d = kD1Cfg[cfg];
  return p.PQ % d.nw == 0 && d.nw % 16 == 0 && p.Cout % (d.mw * d.nwarp) == 0;
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
  G(10, 16, 32, 128, 16, 16)              \
  G(11, 16, 64, 32, 16, 32)               \
  G(12, 32, 128, 64, 16, 32)

struct G1Desc { int bm, bn, bk, nt; };
static const G1Desc kG1Cfg[] = {
#define G(id, bm, bn, bk, wm, wn) {bm, bn, bk, 32 * (bm / wm) * (bn / wn)},
    G1_CFG_LIST
#undef G
};
static const int kNG1 = (int)(sizeof(kG1Cfg) / sizeof(G1Desc));

template <typename T>
static bool launch_g1(int cfg, const T* X, const T* Wt, const T* Bias,
                      const void* BnW, const void* BnB, const void* Rm,
                      const void* Rv, T* Out, const CP& p, int batch,
                      cudaStream_t st) {
  switch (cfg) {
#define G(id, bm, bn, bk, wm, wn)                                            \
  case id: {                                                                 \
    dim3 g(DIVUP(p.PQ, bn), DIVUP(p.Cout, bm), batch);                       \
    gemm1x1_kernel<T, bm, bn, bk, wm, wn>                                    \
        <<<g, 32 * (bm / wm) * (bn / wn), 0, st>>>(FK_KARGS, p);              \
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

#define S1_CFG_LIST \
  S(0, 32, 32, 256, 16, 16)               \
  S(1, 32, 32, 128, 16, 16)               \
  S(2, 32, 64, 128, 16, 32)               \
  S(3, 64, 64, 128, 16, 32)               \
  S(4, 32, 128, 128, 16, 32)              \
  S(5, 16, 64, 128, 16, 32)               \
  S(6, 64, 32, 128, 16, 16)               \
  S(7, 16, 32, 256, 16, 16)               \
  S(8, 16, 128, 128, 16, 32)              \
  S(9, 32, 16, 256, 16, 16)               \
  S(10, 64, 32, 64, 16, 16)               \
  S(11, 32, 32, 64, 16, 16)               \
  S(12, 32, 64, 64, 16, 32)               \
  S(13, 64, 128, 64, 16, 32)              \
  S(14, 32, 128, 64, 16, 32)

struct S1Desc { int bm, bn, bk, nt; };
static const S1Desc kS1Cfg[] = {
#define S(id, bm, bn, bk, wm, wn) {bm, bn, bk, 32 * (bm / wm) * (bn / wn)},
    S1_CFG_LIST
#undef S
};
static const int kNS1 = (int)(sizeof(kS1Cfg) / sizeof(S1Desc));

template <typename T>
static bool launch_s1(int cfg, const T* X, const T* Wt, const T* Bias,
                      const void* BnW, const void* BnB, const void* Rm,
                      const void* Rv, T* Out, const CP& p, int batch,
                      cudaStream_t st) {
  switch (cfg) {
#define S(id, bm, bn, bk, wm, wn)                                            \
  case id: {                                                                 \
    dim3 g(DIVUP(p.PQ, bn), DIVUP(p.Cout, bm), batch);                       \
    gemm1x1_s1_kernel<T, bm, bn, bk, wm, wn>                                 \
        <<<g, 32 * (bm / wm) * (bn / wn), 0, st>>>(FK_KARGS, p);              \
    return true;                                                             \
  }
    S1_CFG_LIST
#undef S
    default:
      return false;
  }
}

// Same eligibility as the looped 1x1 GEMM, plus "the whole k fits in one tile".
static bool s1_ok(const CP& p, int cfg) {
  return g1_ok(p) && p.Cg <= kS1Cfg[cfg].bk;
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
  X(15, 64, 256, 32, 32, 64)              \
  X(16, 16, 64, 32, 16, 32)               \
  X(17, 32, 32, 32, 16, 32)               \
  X(18, 16, 128, 32, 16, 64)              \
  X(19, 16, 16, 16, 16, 16)               \
  X(20, 16, 32, 16, 16, 32)               \
  X(21, 16, 64, 16, 16, 64)               \
  X(22, 32, 32, 16, 16, 16)               \
  X(23, 16, 32, 32, 16, 32)               \
  X(24, 16, 16, 32, 16, 16)               \
  X(25, 32, 32, 64, 16, 16)               \
  X(26, 16, 64, 64, 16, 32)               \
  X(27, 32, 64, 16, 16, 16)               \
  X(28, 64, 64, 16, 16, 32)

struct CfgDesc { int bm, bn, bk, nt; };

static const CfgDesc kMmaCfg[] = {
#define X(id, bm, bn, bk, wm, wn) {bm, bn, bk, 32 * (bm / wm) * (bn / wn)},
    MMA_CFG_LIST
#undef X
};
static const int kNMma = (int)(sizeof(kMmaCfg) / sizeof(CfgDesc));

template <typename T>
static bool launch_mma(int cfg, const T* X, const T* Wt, const T* Bias,
                       const void* BnW, const void* BnB, const void* Rm,
                       const void* Rv, T* Out, const CP& p, int batch,
                       cudaStream_t st) {
  const int M = p.Cout, Nn = p.PQ;
  switch (cfg) {
#define X(id, bm, bn, bk, wm, wn)                                            \
  case id: {                                                                 \
    dim3 g(DIVUP(Nn, bn), DIVUP(M, bm), batch);                              \
    conv_mma_kernel<T, bm, bn, bk, wm, wn>                                   \
        <<<g, 32 * (bm / wm) * (bn / wn), 0, st>>>(FK_KARGS, p);              \
    return true;                                                             \
  }
    MMA_CFG_LIST
#undef X
    default:
      return false;
  }
}

// code = family * 1000 + id;  families: 1 mma, 2 direct 1x1, 3 gemm 1x1
template <typename T>
static bool launch_code(int code, const T* X, const T* Wt, const T* Bias,
                        const void* BnW, const void* BnB, const void* Rm,
                        const void* Rv, T* Out, CP p, int batch,
                        cudaStream_t st) {
  const int fam = code / 1000, id = code % 1000;
  if (fam == 1) {
    p.NCT = DIVUP(p.Cg, kMmaCfg[id].bk);
    return launch_mma<T>(id, FK_KARGS, p, batch, st);
  }
  if (fam == 2) return launch_d1<T>(id, FK_KARGS, p, batch, st);
  if (fam == 3) return launch_g1<T>(id, FK_KARGS, p, batch, st);
  if (fam == 4) return launch_s1<T>(id, FK_KARGS, p, batch, st);
  return false;
}

static std::vector<int> cfg_candidates(const CP& p) {
  std::vector<int> v;
  for (int i = 0; i < kNMma; i++) v.push_back(1000 + i);
  for (int i = 0; i < kND1; i++) if (d1_ok(p, i)) v.push_back(2000 + i);
  if (g1_ok(p)) for (int i = 0; i < kNG1; i++) v.push_back(3000 + i);
  for (int i = 0; i < kNS1; i++) if (s1_ok(p, i)) v.push_back(4000 + i);
  return v;
}

// ---------------------------------------------------------------------------
// Config selection.  The kernels above span tile shapes whose relative cost at
// these sizes is dominated by SM coverage and staging-instruction count, which
// no closed-form model predicted well -- so the first call for a given problem
// shape measures every eligible config (with an L2 flush, to match how the op
// is actually timed) and the winner is cached.
// ---------------------------------------------------------------------------
using CfgKey = std::array<int, 16>;
static std::map<CfgKey, int> g_cfg_cache;
static std::mutex g_cfg_mu;

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
    constexpr int REPS = 21;
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
// host entry points
// ---------------------------------------------------------------------------
static int bn_kind(const at::Tensor& t) {
  switch (t.scalar_type()) {
    case at::kHalf: return 0;
    case at::kFloat: return 1;
    case at::kBFloat16: return 2;
    default: TORCH_CHECK(false, "conv_bn_act: bad batchnorm dtype");
  }
}

static void fill_bn(CP& p, const c10::optional<at::Tensor>& bn_w,
                    const c10::optional<at::Tensor>& bn_b,
                    const c10::optional<at::Tensor>& rm,
                    const c10::optional<at::Tensor>& rv, const void** ptrs) {
  if (!bn_w.has_value()) {
    ptrs[0] = ptrs[1] = ptrs[2] = ptrs[3] = nullptr;
    return;
  }
  const at::Tensor* ts[4] = {&*bn_w, &*bn_b, &*rm, &*rv};
  for (int i = 0; i < 4; i++) {
    TORCH_CHECK(ts[i]->is_contiguous() && ts[i]->numel() == p.Cout,
                "conv_bn_act: bad batchnorm tensor");
    p.bkind[i] = bn_kind(*ts[i]);
    ptrs[i] = ts[i]->const_data_ptr();
  }
}

at::Tensor conv_bn_act_forward(const at::Tensor& x_, const at::Tensor& w,
                               const c10::optional<at::Tensor>& bias_,
                               const c10::optional<at::Tensor>& bn_w,
                               const c10::optional<at::Tensor>& bn_b,
                               const c10::optional<at::Tensor>& bn_rm,
                               const c10::optional<at::Tensor>& bn_rv,
                               double eps, int64_t sh, int64_t sw, int64_t ph,
                               int64_t pw, int64_t dh, int64_t dw,
                               int64_t groups, int64_t act, int64_t wperm,
                               int64_t cfg_override) {
  at::Tensor x = x_.is_contiguous() ? x_ : x_.contiguous();
  const int batch = (int)x.size(0);

  CP p;
  p.C = (int)x.size(1);
  p.H = (int)x.size(2);
  p.W = (int)x.size(3);
  p.Cout = (int)w.size(0);
  if (wperm) {           // weight given as [Cout, R, S, Cin/g]
    p.R = (int)w.size(1); p.S = (int)w.size(2); p.Cg = (int)w.size(3);
    p.wkc = 1; p.wkrs = p.Cg;
  } else {               // native [Cout, Cin/g, R, S]
    p.Cg = (int)w.size(1); p.R = (int)w.size(2); p.S = (int)w.size(3);
    p.wkc = p.R * p.S; p.wkrs = 1;
  }
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
  p.act = (int)act;
  p.eps = (float)eps;
  for (int i = 0; i < 4; i++) p.bkind[i] = 1;

  at::Tensor out = at::empty({batch, p.Cout, p.P, p.Q}, x.options());
  if (p.PQ == 0 || p.Cout == 0 || batch == 0) return out;

  const void* bnp[4];
  fill_bn(p, bn_w, bn_b, bn_rm, bn_rv, bnp);

  auto st = at::cuda::getCurrentCUDAStream();
  TORCH_CHECK(x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16,
              "conv_bn_act_forward: unsupported dtype");

  auto run = [&](int code) -> bool {
    if (x.scalar_type() == at::kHalf) {
      using T = __half;
      const T* bp = bias_.has_value() ? (const T*)bias_->const_data_ptr() : nullptr;
      return launch_code<T>(code, (const T*)x.const_data_ptr(),
                            (const T*)w.const_data_ptr(), bp, bnp[0], bnp[1],
                            bnp[2], bnp[3], (T*)out.data_ptr(), p, batch, st);
    }
    using T = __nv_bfloat16;
    const T* bp = bias_.has_value() ? (const T*)bias_->const_data_ptr() : nullptr;
    return launch_code<T>(code, (const T*)x.const_data_ptr(),
                          (const T*)w.const_data_ptr(), bp, bnp[0], bnp[1],
                          bnp[2], bnp[3], (T*)out.data_ptr(), p, batch, st);
  };

  int code;
  if (cfg_override >= 0) {
    code = (int)cfg_override;
  } else {
    const CfgKey key = {p.C, p.H, p.W, p.Cout, p.R, p.S, p.sh, p.sw, p.ph, p.pw,
                        p.dh, p.dw, (int)groups, batch, (int)x.scalar_type(),
                        (int)wperm};
    bool have = false;
    {
      std::lock_guard<std::mutex> lk(g_cfg_mu);
      auto it = g_cfg_cache.find(key);
      if (it != g_cfg_cache.end()) { code = it->second; have = true; }
    }
    if (!have) {
      code = autotune(cfg_candidates(p), run, x, st);
      TORCH_CHECK(code >= 0, "conv_bn_act_forward: no usable config");
      std::lock_guard<std::mutex> lk(g_cfg_mu);
      g_cfg_cache[key] = code;
    }
  }
  TORCH_CHECK(run(code), "conv_bn_act_forward: unsupported dtype/config");
  return out;
}

// In-place BN + activation on a conv output (fallback path).
at::Tensor bn_act_(at::Tensor y, const c10::optional<at::Tensor>& bn_w,
                   const c10::optional<at::Tensor>& bn_b,
                   const c10::optional<at::Tensor>& bn_rm,
                   const c10::optional<at::Tensor>& bn_rv, double eps,
                   int64_t act) {
  TORCH_CHECK(y.is_contiguous() && y.dim() == 4, "bn_act_: need contiguous NCHW");
  CP p;
  p.Cout = (int)y.size(1);
  p.act = (int)act;
  p.eps = (float)eps;
  for (int i = 0; i < 4; i++) p.bkind[i] = 1;
  const void* bnp[4];
  fill_bn(p, bn_w, bn_b, bn_rm, bn_rv, bnp);

  const int HW = (int)(y.size(2) * y.size(3));
  const int esz = (int)y.element_size();
  const int VE = 16 / esz;
  auto st = at::cuda::getCurrentCUDAStream();
  if (HW % VE || y.numel() % VE) {   // rare (odd spatial size): fall back to ATen
    at::Tensor r = y.to(at::kFloat);
    if (bn_w.has_value()) {
      at::Tensor s = at::rsqrt(bn_rv->to(at::kFloat) + eps) * bn_w->to(at::kFloat);
      at::Tensor b = bn_b->to(at::kFloat) - bn_rm->to(at::kFloat) * s;
      r = r * s.view({1, -1, 1, 1}) + b.view({1, -1, 1, 1});
    }
    if (act) r = at::silu(r);
    y.copy_(r);
    return y;
  }
  const int nvec = (int)(y.numel() / VE);
  const int hwv = HW / VE;
  const int U = nvec > 256 * 2048 ? 2 : 1;
  const int blocks = DIVUP(nvec, 256 * U);
#define FK_BNACT(T)                                                          \
  if (U == 2)                                                                \
    bn_act_kernel<T, 2><<<blocks, 256, 0, st>>>((T*)y.data_ptr(), bnp[0],    \
        bnp[1], bnp[2], bnp[3], hwv, p.Cout, nvec, p);                       \
  else                                                                       \
    bn_act_kernel<T, 1><<<blocks, 256, 0, st>>>((T*)y.data_ptr(), bnp[0],    \
        bnp[1], bnp[2], bnp[3], hwv, p.Cout, nvec, p);
  if (y.scalar_type() == at::kHalf) { FK_BNACT(__half) }
  else if (y.scalar_type() == at::kBFloat16) { FK_BNACT(__nv_bfloat16) }
  else if (y.scalar_type() == at::kFloat) { FK_BNACT(float) }
  else TORCH_CHECK(false, "bn_act_: unsupported dtype");
#undef FK_BNACT
  return y;
}
"""

_CPP_SRC = r"""
at::Tensor conv_bn_act_forward(const at::Tensor& x_, const at::Tensor& w,
                               const c10::optional<at::Tensor>& bias_,
                               const c10::optional<at::Tensor>& bn_w,
                               const c10::optional<at::Tensor>& bn_b,
                               const c10::optional<at::Tensor>& bn_rm,
                               const c10::optional<at::Tensor>& bn_rv,
                               double eps, int64_t sh, int64_t sw, int64_t ph,
                               int64_t pw, int64_t dh, int64_t dw,
                               int64_t groups, int64_t act, int64_t wperm,
                               int64_t cfg_override);
at::Tensor bn_act_(at::Tensor y, const c10::optional<at::Tensor>& bn_w,
                   const c10::optional<at::Tensor>& bn_b,
                   const c10::optional<at::Tensor>& bn_rm,
                   const c10::optional<at::Tensor>& bn_rv, double eps,
                   int64_t act);
"""

_EXT = None
_EXT_FAILED = False
# -1 lets the extension pick (and cache) the fastest tile config for each
# problem shape; set FK_YOLOCONV_CFG to pin one when debugging.
_CFG_OVERRIDE = int(os.environ.get("FK_YOLOCONV_CFG", "-1"))
# Weight layout handed to the kernel: 0 = the parameter's own
# [Cout, Cin/g, R, S] (the kernel's two k-strides absorb it), 1 = a cached
# [Cout, R, S, Cin/g] permutation, contiguous along the GEMM's k.  The
# permutation is the textbook-friendlier layout but measured slower on the
# captured 3x3 shapes -- their weight is a few tens of KB and stays in L2, so
# the strided A-tile loads cost nothing and the repack is pure overhead.
_WPERM = int(os.environ.get("FK_YOLOCONV_WPERM", "0"))


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
        name=f"fk_yolo_conv_bn_act_{tag}",
        cpp_sources=_CPP_SRC,
        cuda_sources=_CUDA_SRC,
        functions=["conv_bn_act_forward", "bn_act_"],
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

    A build failure degrades to the baseline ops instead of breaking the op.
    """
    global _EXT, _EXT_FAILED
    if _EXT is None and not _EXT_FAILED:
        try:
            _EXT = _build_ext()
        except Exception:
            _EXT_FAILED = True
    return _EXT


_FUSED_DTYPES = (torch.float16, torch.bfloat16)
_BN_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

# plan codes
_FALLBACK = -1     # baseline ops (act(bn(conv(x))))
_SPLIT = 0         # L1 conv + fused BN/act epilogue (grouped convs, fp32)
_FUSED = 1         # one kernel, BN folded in the epilogue
_FUSED_NOBN = 2    # one kernel, no BN (after fuse())
_FUSED_CONV = 3    # one kernel for the convolution only; torch does BN/act
                   # (training mode -- the BN has to update its running stats --
                   #  or an activation the epilogue does not implement)


def autopad(k: int | tuple[int, int], p=None, d: int = 1):
    if isinstance(k, tuple):
        if d > 1:
            k = tuple(d * (x - 1) + 1 for x in k)
        if p is None:
            return tuple(x // 2 for x in k)
        return p
    if d > 1:
        k = d * (k - 1) + 1
    return k // 2 if p is None else p


def _fuse_conv_bn(conv: Conv2d, bn: BatchNorm2d) -> tuple[torch.Tensor, torch.Tensor]:
    w_conv = conv.weight.clone().view(conv.weight.shape[0], -1)
    w_bn = torch.diag(
        bn.weight.to(dtype=conv.weight.dtype).div(
            torch.sqrt(bn.eps + bn.running_var.to(dtype=conv.weight.dtype))
        )
    )
    fused_weight = torch.mm(w_bn, w_conv).view_as(conv.weight)

    conv_bias = conv.bias
    if conv_bias is None:
        conv_bias = torch.zeros(conv.weight.shape[0], device=conv.weight.device, dtype=conv.weight.dtype)
    b_bn = (
        bn.bias.to(dtype=conv.weight.dtype)
        - bn.weight.to(dtype=conv.weight.dtype)
        .mul(bn.running_mean.to(dtype=conv.weight.dtype))
        .div(torch.sqrt(bn.running_var.to(dtype=conv.weight.dtype) + bn.eps))
    )
    fused_bias = torch.mm(w_bn, conv_bias.reshape(-1, 1)).reshape(-1) + b_bn
    return fused_weight, fused_bias


def _act_kind(act) -> int:
    """0 = identity, 1 = SiLU, -1 = anything the epilogue cannot do."""
    if isinstance(act, nn.Identity):
        return 0
    if isinstance(act, (nn.SiLU, SiLU)) or type(act).__name__ == "SiLU":
        return 1
    return -1


class YOLOConv(nn.Module):
    default_act = SiLU()

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 1,
        s: int = 1,
        p=None,
        g: int = 1,
        d: int = 1,
        act=True,
    ):
        super().__init__()
        self.conv = Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = BatchNorm2d(c2, eps=1e-3, momentum=0.03)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self._is_fused = False
        self._plan = None
        self._geom = None
        self._wp = None
        self._wkey = None
        self._akind = -1

    # -- dispatch ---------------------------------------------------------
    def _invalidate(self) -> None:
        self._plan = None
        self._wp = None
        self._wkey = None

    def _fused_ok(self, x: torch.Tensor) -> bool:
        w = self.conv.weight
        if not (x.is_cuda and x.dim() == 4 and w.dim() == 4):
            return False
        if x.dtype not in _FUSED_DTYPES or w.dtype != x.dtype:
            return False
        b = self.conv.bias
        if b is not None and b.dtype != x.dtype:
            return False
        if self.conv.groups != 1:
            return False
        ph, pw = self.conv.padding
        if ph >= w.shape[2] or pw >= w.shape[3]:
            return False  # heavy padding: rare, leave to torch
        return True

    def _bn_ok(self) -> bool:
        """Whether the BN can be folded into an epilogue: eval-mode affine
        BatchNorm with running stats (training mode has to update them)."""
        bn = self.bn
        if self.training or not bn.track_running_stats:
            return False
        ts = (bn.weight, bn.bias, bn.running_mean, bn.running_var)
        if any(t is None for t in ts):
            return False
        c2 = self.conv.weight.shape[0]
        return all(
            t.dim() == 1 and t.shape[0] == c2 and t.is_contiguous()
            and t.dtype in _BN_DTYPES for t in ts
        )

    def _make_plan(self, x: torch.Tensor) -> int:
        """Pick the launch path for this module + input, once; cached in
        ``self._plan`` and dropped by ``_invalidate`` on any mutation
        (``train()``, ``.to()``, ``fuse()``, a state-dict load)."""
        ak = _act_kind(self.act)
        # Nothing here is worth compiling an extension for on CPU / a stray dtype.
        cuda = x.is_cuda and x.dim() == 4 and x.dtype in _BN_DTYPES
        ext = _ext() if cuda else None
        plan = _FALLBACK
        if ext is not None:
            fast_conv = self._fused_ok(x)
            epi_ok = ak >= 0 and (self._is_fused or self._bn_ok())
            if fast_conv and epi_ok:
                plan = _FUSED_NOBN if self._is_fused else _FUSED
            elif fast_conv:
                plan = _FUSED_CONV
            elif epi_ok and not self._is_fused:
                plan = _SPLIT
        c = self.conv
        wperm = 1 if (_WPERM and c.weight.shape[2] * c.weight.shape[3] > 1) else 0
        # The activation the *kernel* applies: none when torch still has to run
        # the BatchNorm (``_FUSED_CONV``) or the activation itself.
        kact = ak if plan in (_FUSED, _FUSED_NOBN) else 0
        self._geom = (
            c.stride[0], c.stride[1], c.padding[0], c.padding[1],
            c.dilation[0], c.dilation[1], c.groups, kact, wperm, _CFG_OVERRIDE,
        )
        self._akind = ak
        self._plan = plan
        return plan

    def _pweight(self) -> torch.Tensor:
        """Weight as [Cout, R, S, Cin/groups]; cached until the weight changes."""
        w = self.conv.weight
        key = (w.data_ptr(), w._version)
        if self._wkey != key:
            self._wp = w.permute(0, 2, 3, 1).contiguous()
            self._wkey = key
        return self._wp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        plan = self._plan
        if plan is None:
            plan = self._make_plan(x)
        if plan == _FUSED:
            g = self._geom
            bn = self.bn
            c = self.conv
            return _EXT.conv_bn_act_forward(
                x, self._pweight() if g[8] else c.weight, c.bias,
                bn.weight, bn.bias, bn.running_mean, bn.running_var, bn.eps, *g)
        if plan == _FUSED_NOBN:
            g = self._geom
            c = self.conv
            return _EXT.conv_bn_act_forward(
                x, self._pweight() if g[8] else c.weight, c.bias,
                None, None, None, None, 0.0, *g)
        if plan == _FUSED_CONV:
            g = self._geom
            c = self.conv
            y = _EXT.conv_bn_act_forward(
                x, self._pweight() if g[8] else c.weight, c.bias,
                None, None, None, None, 0.0, *g)
            return self.act(y) if self._is_fused else self.act(self.bn(y))
        if plan == _SPLIT:
            bn = self.bn
            return _EXT.bn_act_(self.conv(x), bn.weight, bn.bias,
                                bn.running_mean, bn.running_var, bn.eps,
                                self._akind)
        if self._is_fused:
            return self.act(self.conv(x))
        return self.act(self.bn(self.conv(x)))

    # -- cache invalidation on every mutation torch routes through ---------
    def train(self, mode: bool = True):
        self._invalidate()
        return super().train(mode)

    def _apply(self, *args, **kwargs):
        self._invalidate()
        return super()._apply(*args, **kwargs)

    def _load_from_state_dict(self, *args, **kwargs):
        self._invalidate()
        return super()._load_from_state_dict(*args, **kwargs)

    @torch.no_grad()
    def fuse(self):
        if self._is_fused:
            return self
        fused_weight, fused_bias = _fuse_conv_bn(self.conv, self.bn)
        # ``copy_``, not ``weight.data.copy_``: ``.data`` carries its own version
        # counter, so an in-place write through it leaves every ``_version``-keyed
        # weight cache downstream (this module's, ``L1.Conv2d``'s permuted copy)
        # holding the pre-fusion weight.
        self.conv.weight.copy_(fused_weight)
        self.conv.bias = nn.Parameter(fused_bias)
        delattr(self, "bn")
        self._is_fused = True
        self._invalidate()
        return self


def fuse_module(module: nn.Module) -> nn.Module:
    from .yolov10_repvggdw import YOLORepVGGDW

    for child in module.children():
        fuse_module(child)
    if isinstance(module, YOLOConv):
        module.fuse()
    elif isinstance(module, YOLORepVGGDW):
        module.fuse()
    return module
