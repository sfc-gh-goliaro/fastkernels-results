"""YOLOv10 Spatial Pyramid Pooling - Fast.

The block is ten launches of almost no work: a 1x1 conv (256->128) + BN + SiLU,
three 5x5/stride-1 max pools, a concat, then a second 1x1 conv (512->256) + BN
+ SiLU.  Every tensor is 800 KB or less, so on a B200 each of those kernels sits
at its dispatch floor and the operator spends ~110 us moving 1.6 MB.  Two
hand-written kernels replace all ten:

* ``k1`` owns one (image, 16 z-channel) slab.  It runs the first 1x1 conv as a
  GEMM on tensor cores -- for a 1x1 stride-1 conv both operands already *are*
  row-major matrices, so ``C[16][400] = W1[16][256] @ x[256][400]`` needs no
  im2col -- folds BN into the fp32 epilogue and applies SiLU there, then takes
  the three pools in shared memory.  Each pool is separable: one thread owns one
  of the 320 rows, the 5-wide horizontal max runs in registers and the vertical
  max reads the four neighbouring rows (which are the threads next door, so the
  shared reads stay conflict-free).  ``y2``/``y3`` re-pool ``y1``/``y2`` in
  place, so the concat is never a separate pass -- each of the four 16x400
  planes is dumped into its slice of the concatenated tensor with coalesced
  16-byte stores.
* ``k2`` runs the second conv the same way (``C[64][80] = W2[64][512] @
  cat[512][80]`` per block), again folding BN + SiLU into the epilogue.

Both GEMMs stage their operands with ``cp.async``.  That is what makes the shape
work at these sizes: the grid is only ``Cq/16 * N`` blocks, far too few warps to
cover a global load's latency, so the k loop has to be one round trip rather
than one per tile -- and cp.async needs no registers for the payload, so the
whole K dimension can be in flight at once (~217 KB of shared memory for k1).
Feeding the mma fragments from global memory directly instead costs 8 sectors
per load request at a row stride of 400, and staging through registers caps the
bytes in flight at what the register file holds; both measured ~2x slower.

BN folding happens *inside* the kernels from ``weight/bias/running_mean/
running_var``, which keeps the Python side to one extension call with a fixed
argument tuple and leaves nothing to invalidate when the weights change.  The
accumulator stays fp32 all the way to the activation, so the fused path is
slightly *more* accurate than the baseline's round-to-fp16 between conv and BN.

Anything the kernels do not cover (non-20x20 planes, a non-SiLU activation, an
already-fused conv, a failed build) falls back to the baseline module graph.
"""

from __future__ import annotations

import hashlib
import os

import torch
import torch.nn as nn

from ..L1.max_pool2d import MaxPool2d
from .yolov10_conv import YOLOConv

_CUDA_SRC = r"""
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <mma.h>

namespace {

// Plane geometry of the captured workload.  The pooling phase keeps a whole
// 20-element row in registers, so W is a compile-time constant.
#define FK_H 20
#define FK_W 20
#define FK_HW (FK_H * FK_W)

template <typename T> struct TT;
template <> struct TT<__half> {
  using v2 = __half2;
  static __device__ __forceinline__ __half cvt(float f) { return __float2half(f); }
  static __device__ __forceinline__ v2 mx2(v2 a, v2 b) { return __hmax2(a, b); }
  static __device__ __forceinline__ v2 pack(__half a, __half b) { return __halves2half2(a, b); }
  static __device__ __forceinline__ __half lo(v2 a) { return __low2half(a); }
  static __device__ __forceinline__ __half hi(v2 a) { return __high2half(a); }
};
template <> struct TT<__nv_bfloat16> {
  using v2 = __nv_bfloat162;
  static __device__ __forceinline__ __nv_bfloat16 cvt(float f) { return __float2bfloat16(f); }
  static __device__ __forceinline__ v2 mx2(v2 a, v2 b) { return __hmax2(a, b); }
  static __device__ __forceinline__ v2 pack(__nv_bfloat16 a, __nv_bfloat16 b) { return __halves2bfloat162(a, b); }
  static __device__ __forceinline__ __nv_bfloat16 lo(v2 a) { return __low2bfloat16(a); }
  static __device__ __forceinline__ __nv_bfloat16 hi(v2 a) { return __high2bfloat16(a); }
};

// 8-byte shared-memory access unit: two packed pairs, so a 20-element row moves
// in five LDS.64/STS.64 instead of ten 32-bit ones.
template <typename T> struct __align__(8) Pair2 { typename TT<T>::v2 a, b; };

using AccFrag = nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float>;

__device__ __forceinline__ float fk_silu(float f) {
  return f / (1.0f + __expf(-f));
}

// BN in eval mode is a per-channel affine, folded here so the activation sees
// the fp32 accumulator instead of a round-tripped fp16 conv output.
template <typename T, typename S>
__device__ __forceinline__ void fk_bn_fold(int c, const T* __restrict__ G,
                                           const T* __restrict__ BE,
                                           const S* __restrict__ RM,
                                           const S* __restrict__ RV, float eps,
                                           float* s_out, float* b_out) {
  const float s = (float)G[c] * rsqrtf((float)RV[c] + eps);
  *s_out = s;
  *b_out = (float)BE[c] - (float)RM[c] * s;
}

// ---------------------------------------------------------------------------
// C[BM][BN] = Wg[BM][K] @ Xg[K][BN], each warp owning a 16 x (16*WN) tile.
//
// Both operands are staged into shared memory with ``cp.async``, in NCH chunks
// of BK k-rows that are all issued before any of them is waited on.  Three
// things made this the shape that works at these sizes:
//
//  * A 16x16 fp16 tile of a matrix whose row stride is 400 costs 8 sectors per
//    load *request* (measured), so pulling the mma fragments straight out of
//    global memory -- which a 1x1 conv invites, both operands already being
//    row-major matrices there -- buries the kernel in L1 latency.
//  * Staging through registers instead fixes the coalescing but caps the bytes
//    in flight at what the register file can hold, and with only Cq/16 * N
//    blocks there are nowhere near enough warps to cover a global load's
//    latency.  Each k-tile then costs a full round trip: k1 measured 12.2us for
//    3.3 MFLOP of work per block.
//  * cp.async needs no registers for the payload, so the whole K dimension can
//    be in flight at once (~217 KB of shared memory for k1) and the k loop
//    costs one round trip plus NCH barriers instead of one per tile.
// ---------------------------------------------------------------------------
template <typename T, int BM, int BN, int WN, int TPB, int BK, int NCH>
__device__ __forceinline__ void fk_block_gemm(const T* __restrict__ Wg, int ldw,
                                              const T* __restrict__ Xg, int K,
                                              T* As, T* Bs, AccFrag (&acc)[WN]) {
  using namespace nvcuda::wmma;
  constexpr int KP = BK * NCH;                       // k rows held in shared
  constexpr int LDA = KP + 8, LDB = BN + 8;
  constexpr int AV = BK / 8, BV = BN / 8;            // uint4 per staged row
  constexpr int NA = BM * AV, NB = BK * BV;
  constexpr int NWN = BN / (16 * WN);
  static_assert(BM % 16 == 0 && BN % (16 * WN) == 0 && BK % 16 == 0, "tile shape");
  static_assert(32 * (BM / 16) * NWN == TPB, "warps must cover the block tile");

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const T* as0 = As + (warp / NWN) * 16 * LDA;
  const T* bs0 = Bs + (warp % NWN) * (16 * WN);

#pragma unroll
  for (int j = 0; j < WN; ++j) fill_fragment(acc[j], 0.0f);

  for (int kb = 0; kb < K; kb += KP) {
#pragma unroll
    for (int s = 0; s < NCH; ++s) {
#pragma unroll
      for (int i = tid; i < NA; i += TPB) {
        const int r = i / AV, c = (i - r * AV) * 8;
        const int kk = kb + s * BK + c;
        T* d = As + r * LDA + s * BK + c;
        if (kk + 8 <= K)
          __pipeline_memcpy_async(d, Wg + (size_t)r * ldw + kk, 16);
        else
          __pipeline_memcpy_async(d, Wg, 16, 16);
      }
#pragma unroll
      for (int i = tid; i < NB; i += TPB) {
        const int r = i / BV, c = (i - r * BV) * 8;
        const int kk = kb + s * BK + r;
        T* d = Bs + (s * BK + r) * LDB + c;
        if (kk < K)
          __pipeline_memcpy_async(d, Xg + (size_t)kk * FK_HW + c, 16);
        else
          __pipeline_memcpy_async(d, Xg, 16, 16);
      }
      __pipeline_commit();
    }
#pragma unroll
    for (int s = 0; s < NCH; ++s) {
      __pipeline_wait_prior(NCH - 1 - s);
      __syncthreads();
#pragma unroll
      for (int kk = s * BK; kk < (s + 1) * BK; kk += 16) {
        fragment<matrix_a, 16, 16, 16, T, row_major> af;
        fragment<matrix_b, 16, 16, 16, T, row_major> bf[WN];
        load_matrix_sync(af, as0 + kk, LDA);
#pragma unroll
        for (int j = 0; j < WN; ++j)
          load_matrix_sync(bf[j], bs0 + kk * LDB + j * 16, LDB);
#pragma unroll
        for (int j = 0; j < WN; ++j) mma_sync(acc[j], af, bf[j], acc[j]);
      }
    }
    __syncthreads();
  }
}

// Scatter one warp's 16x16 fp32 tile through `ep`, apply the folded BN + SiLU
// and store it as eight-wide fp16 vectors.  Lane l owns row l/2, columns
// (l&1)*8 .. +8 of the tile.
template <typename T>
__device__ __forceinline__ void fk_epilogue(AccFrag& acc, float* ep,
                                            const float* scs, const float* bcs,
                                            int srow, T* dst, int ldd) {
  using namespace nvcuda::wmma;
  const int lane = threadIdx.x & 31;
  const int er = lane >> 1, ec = (lane & 1) * 8;
  store_matrix_sync(ep, acc, 16, mem_row_major);
  __syncwarp();
  const float s = scs[srow + er], b = bcs[srow + er];
  T v[8];
#pragma unroll
  for (int e = 0; e < 8; ++e)
    v[e] = TT<T>::cvt(fk_silu(s * ep[er * 16 + ec + e] + b));
  *reinterpret_cast<uint4*>(dst + (size_t)er * ldd + ec) =
      *reinterpret_cast<const uint4*>(v);
}

// 5-wide sliding max of one 20-element row, held as ten packed pairs.
// Splitting the row into even/odd lanes turns the +-2 window into whole-pair
// shifts, so the whole thing is 31 packed max ops instead of 80 scalar ones:
//   m[i] = max(P[i-1], P[i])                       (per lane)
//   h[2i]   = max(m[i].e, m[i+1].e, m[i].o)
//   h[2i+1] = max(m[i].o, m[i+1].o, m[i+1].e)
template <typename T>
__device__ __forceinline__ void fk_hmax5(const typename TT<T>::v2* P,
                                         typename TT<T>::v2* H) {
  using v2 = typename TT<T>::v2;
  v2 m[FK_W / 2 + 1];
  m[0] = P[0];
#pragma unroll
  for (int i = 1; i < FK_W / 2; ++i) m[i] = TT<T>::mx2(P[i - 1], P[i]);
  m[FK_W / 2] = P[FK_W / 2 - 1];
#pragma unroll
  for (int i = 0; i < FK_W / 2; ++i) {
    const v2 both = TT<T>::mx2(m[i], m[i + 1]);
    const v2 cross = TT<T>::pack(TT<T>::hi(m[i]), TT<T>::lo(m[i + 1]));
    H[i] = TT<T>::mx2(both, cross);
  }
}

// ---------------------------------------------------------------------------
// k1: cv1 (1x1 conv + BN + SiLU) followed by the three max pools, writing the
// four concatenated planes for one (image, CB-channel) slab.  The pools need
// the whole 20x20 plane of a channel in one block, which fixes BN = 400.
// ---------------------------------------------------------------------------
template <typename T, typename S, int CB, int WN, int TPB, int BK, int NCH>
__global__ __launch_bounds__(TPB, 1)
void k1_kernel(const T* __restrict__ X, const T* __restrict__ W1,
               const T* __restrict__ G, const T* __restrict__ BE,
               const S* __restrict__ RM, const S* __restrict__ RV,
               const float eps, T* __restrict__ CAT, const int C1,
               const int Cq) {
  using v2 = typename TT<T>::v2;
  using P2 = Pair2<T>;
  constexpr int NW = TPB / 32;
  constexpr int NWN = FK_HW / (16 * WN);
  constexpr int ROWS = CB * FK_H;
  constexpr int RPT = (ROWS + TPB - 1) / TPB;
  constexpr int NV = CB * FK_HW * (int)sizeof(T) / 16;   // 16B plane stores
  constexpr int NVT = (NV + TPB - 1) / TPB;
  static_assert(TPB % FK_H == 0, "row/channel index must stay derivable");

  extern __shared__ __align__(16) char smem[];
  // GEMM staging, then (once the k loop retires) the plane being pooled, the
  // fp32 epilogue staging and the horizontal-pass scratch.
  T* As = reinterpret_cast<T*>(smem);
  T* Bs = As + CB * (BK * NCH + 8);
  T* Zp = reinterpret_cast<T*>(smem);
  float* EP = reinterpret_cast<float*>(smem + CB * FK_HW * sizeof(T));
  T* Hs = reinterpret_cast<T*>(EP + NW * 256);
  __shared__ float scs[CB], bcs[CB];

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int cb0 = blockIdx.x * CB;
  const int nb = blockIdx.y;

  if (tid < CB)
    fk_bn_fold<T, S>(cb0 + tid, G, BE, RM, RV, eps, &scs[tid], &bcs[tid]);

  AccFrag acc[WN];
  fk_block_gemm<T, CB, FK_HW, WN, TPB, BK, NCH>(
      W1 + (size_t)cb0 * C1, C1, X + (size_t)nb * C1 * FK_HW, C1, As, Bs, acc);

  {
    const int zr = (warp / NWN) * 16, zc = (warp % NWN) * (16 * WN);
#pragma unroll
    for (int j = 0; j < WN; ++j)
      fk_epilogue<T>(acc[j], EP + warp * 256, scs, bcs, zr,
                     Zp + zr * FK_HW + zc + j * 16, FK_HW);
  }
  __syncthreads();

  // Coalesced dump of one CBx400 plane into its slice of the concat.
  const uint4* zsrc = reinterpret_cast<const uint4*>(Zp);
  uint4* cdst = reinterpret_cast<uint4*>(
      CAT + ((size_t)nb * 4 * Cq + cb0) * FK_HW);
  const size_t segstride = (size_t)Cq * FK_HW * sizeof(T) / 16;
#pragma unroll
  for (int t = 0; t < NVT; ++t)
    if (NV % TPB == 0 || tid + t * TPB < NV)
      cdst[tid + t * TPB] = zsrc[tid + t * TPB];

  // y1 = pool5(z), y2 = pool5(y1), y3 = pool5(y2).  Each pool is separable: a
  // 5-wide horizontal max in registers, then a 5-row vertical max.  One thread
  // owns one row, so the vertical neighbours are the threads next door and the
  // shared reads stay bank-conflict free.
#pragma unroll
  for (int seg = 1; seg <= 3; ++seg) {
#pragma unroll
    for (int u = 0; u < RPT; ++u) {
      const int row = tid + u * TPB;
      if (RPT * TPB == ROWS || row < ROWS) {
        const P2* zrow = reinterpret_cast<const P2*>(Zp + row * FK_W);
        P2 t5[FK_W / 4];
#pragma unroll
        for (int i = 0; i < FK_W / 4; ++i) t5[i] = zrow[i];
        v2 P[FK_W / 2], H[FK_W / 2];
#pragma unroll
        for (int i = 0; i < FK_W / 4; ++i) { P[2 * i] = t5[i].a; P[2 * i + 1] = t5[i].b; }
        fk_hmax5<T>(P, H);
#pragma unroll
        for (int i = 0; i < FK_W / 4; ++i) { t5[i].a = H[2 * i]; t5[i].b = H[2 * i + 1]; }
        P2* hrow = reinterpret_cast<P2*>(Hs + row * FK_W);
#pragma unroll
        for (int i = 0; i < FK_W / 4; ++i) hrow[i] = t5[i];
      }
    }
    __syncthreads();
#pragma unroll
    for (int u = 0; u < RPT; ++u) {
      const int row = tid + u * TPB;
      if (RPT * TPB == ROWS || row < ROWS) {
        const int ti = row % FK_H;
        const P2* hrow = reinterpret_cast<const P2*>(Hs + row * FK_W);
        P2 r[FK_W / 4];
#pragma unroll
        for (int i = 0; i < FK_W / 4; ++i) r[i] = hrow[i];
#pragma unroll
        for (int d = -2; d <= 2; ++d) {
          if (d == 0) continue;
          if ((unsigned)(ti + d) < (unsigned)FK_H) {
            const P2* q = hrow + d * (FK_W / 4);
#pragma unroll
            for (int i = 0; i < FK_W / 4; ++i) {
              const P2 o = q[i];
              r[i].a = TT<T>::mx2(r[i].a, o.a);
              r[i].b = TT<T>::mx2(r[i].b, o.b);
            }
          }
        }
        P2* orow = reinterpret_cast<P2*>(Zp + row * FK_W);
#pragma unroll
        for (int i = 0; i < FK_W / 4; ++i) orow[i] = r[i];
      }
    }
    __syncthreads();
    uint4* dst = cdst + (size_t)seg * segstride;
#pragma unroll
    for (int t = 0; t < NVT; ++t)
      if (NV % TPB == 0 || tid + t * TPB < NV)
        dst[tid + t * TPB] = zsrc[tid + t * TPB];
  }
}

// ---------------------------------------------------------------------------
// k2: cv2 (1x1 conv + BN + SiLU) over the concatenated tensor.
// ---------------------------------------------------------------------------
template <typename T, typename S, int BM, int BN, int WN, int TPB, int BK, int NCH>
__global__ __launch_bounds__(TPB, 1)
void k2_kernel(const T* __restrict__ CAT, const T* __restrict__ W2,
               const T* __restrict__ G, const T* __restrict__ BE,
               const S* __restrict__ RM, const S* __restrict__ RV,
               const float eps, T* __restrict__ OUT, const int K,
               const int Cout) {
  constexpr int NW = TPB / 32;
  constexpr int NWN = BN / (16 * WN);
  extern __shared__ __align__(16) char smem[];
  T* As = reinterpret_cast<T*>(smem);
  T* Bs = As + BM * (BK * NCH + 8);
  float* EP = reinterpret_cast<float*>(smem);
  __shared__ float scs[BM], bcs[BM];

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int n0 = blockIdx.x * BN;
  const int m0 = blockIdx.y * BM;
  const int nb = blockIdx.z;

  if (tid < BM)
    fk_bn_fold<T, S>(m0 + tid, G, BE, RM, RV, eps, &scs[tid], &bcs[tid]);

  AccFrag acc[WN];
  fk_block_gemm<T, BM, BN, WN, TPB, BK, NCH>(
      W2 + (size_t)m0 * K, K, CAT + (size_t)nb * K * FK_HW + n0, K, As, Bs, acc);
  __syncthreads();   // EP aliases the staging buffers

  const int wm = (warp / NWN) * 16, wn = (warp % NWN) * (16 * WN);
  T* ob = OUT + ((size_t)nb * Cout + m0 + wm) * FK_HW + n0 + wn;
#pragma unroll
  for (int j = 0; j < WN; ++j)
    fk_epilogue<T>(acc[j], EP + warp * 256, scs, bcs, wm, ob + j * 16, FK_HW);
}

// --- launch -----------------------------------------------------------------
// k1 is pinned to 16 z-channels per block (one mma m-tile: the pools need a
// channel's whole 20x20 plane in one block, so the only free knob is how many
// channels it takes, and 16 maximises the blocks in flight).  k2 picks its
// output-channel tile from the block count: 64 keeps the concat re-reads down
// for the batched shape, 32 doubles the blocks when there are too few.
//
// Shapes swept against the bench (geomean over both captured shapes, one
// process so the clock is comparable): k1 BK*NCH 64*4 beat 128*2 / 64*2 / 32*4
// and beat WN=5 (five tiles per warp, 160 threads); k2 BK*NCH 128*4 tied 256*2
// and beat 128*2 / 64*4.
#define FK_CB1 16
#define FK_WN1 1
#define FK_BK1 64
#define FK_NCH1 4
#define FK_TPB1 (32 * (FK_CB1 / 16) * (FK_HW / (16 * FK_WN1)))
#define FK_BN2 80
#define FK_WN2 1
#define FK_BK2 128
#define FK_NCH2 4
#define FK_TPB2(BM) (32 * ((BM) / 16) * (FK_BN2 / (16 * FK_WN2)))

constexpr int k1_shared() {
  constexpr int stage = FK_CB1 * (FK_BK1 * FK_NCH1 + 8) +
                        FK_BK1 * FK_NCH1 * (FK_HW + 8);
  constexpr int post = 2 * FK_CB1 * FK_HW + (FK_TPB1 / 32) * 512;
  return stage > post ? stage : post;
}
template <int BM>
constexpr int k2_shared() {
  constexpr int stage = BM * (FK_BK2 * FK_NCH2 + 8) +
                        FK_BK2 * FK_NCH2 * (FK_BN2 + 8);
  constexpr int post = (FK_TPB2(BM) / 32) * 512;
  return stage > post ? stage : post;
}

template <typename T, typename S, int BM2>
void launch(const at::Tensor& x, const at::Tensor& w1, const at::Tensor& g1,
            const at::Tensor& be1, const at::Tensor& rm1, const at::Tensor& rv1,
            float eps1, const at::Tensor& w2, const at::Tensor& g2,
            const at::Tensor& be2, const at::Tensor& rm2, const at::Tensor& rv2,
            float eps2, at::Tensor& cat, at::Tensor& out, int N, int C1, int Cq,
            int Cout, cudaStream_t st) {
  const int K = 4 * Cq;
  constexpr int SH1 = k1_shared() * (int)sizeof(T);
  constexpr int SH2 = k2_shared<BM2>() * (int)sizeof(T);
  constexpr int TPB2 = FK_TPB2(BM2);
  auto f1 = k1_kernel<T, S, FK_CB1, FK_WN1, FK_TPB1, FK_BK1, FK_NCH1>;
  auto f2 = k2_kernel<T, S, BM2, FK_BN2, FK_WN2, TPB2, FK_BK2, FK_NCH2>;
  static bool primed = false;
  if (!primed) {
    cudaFuncSetAttribute(f1, cudaFuncAttributeMaxDynamicSharedMemorySize, SH1);
    cudaFuncSetAttribute(f2, cudaFuncAttributeMaxDynamicSharedMemorySize, SH2);
    primed = true;
  }

  const T* xp = (const T*)x.const_data_ptr();
  const T* w1p = (const T*)w1.const_data_ptr();
  const T* g1p = (const T*)g1.const_data_ptr();
  const T* b1p = (const T*)be1.const_data_ptr();
  const S* m1p = (const S*)rm1.const_data_ptr();
  const S* v1p = (const S*)rv1.const_data_ptr();
  T* catp = (T*)cat.data_ptr();
  const T* ccp = catp;
  const T* w2p = (const T*)w2.const_data_ptr();
  const T* g2p = (const T*)g2.const_data_ptr();
  const T* b2p = (const T*)be2.const_data_ptr();
  const S* m2p = (const S*)rm2.const_data_ptr();
  const S* v2p = (const S*)rv2.const_data_ptr();
  T* outp = (T*)out.data_ptr();

  f1<<<dim3(Cq / FK_CB1, N), FK_TPB1, SH1, st>>>(xp, w1p, g1p, b1p, m1p, v1p,
                                                 eps1, catp, C1, Cq);
  f2<<<dim3(FK_HW / FK_BN2, Cout / BM2, N), TPB2, SH2, st>>>(
      ccp, w2p, g2p, b2p, m2p, v2p, eps2, outp, K, Cout);
}

template <typename T, typename S>
void launch_bm(const at::Tensor& x, const at::Tensor& w1, const at::Tensor& g1,
               const at::Tensor& be1, const at::Tensor& rm1,
               const at::Tensor& rv1, float eps1, const at::Tensor& w2,
               const at::Tensor& g2, const at::Tensor& be2,
               const at::Tensor& rm2, const at::Tensor& rv2, float eps2,
               at::Tensor& cat, at::Tensor& out, int N, int C1, int Cq, int Cout,
               cudaStream_t st) {
  // 64 out channels per block unless that leaves too few blocks to fill the
  // machine, in which case halve the tile and double the grid.
  const int blocks64 = (FK_HW / FK_BN2) * (Cout / 64) * N;
  if (Cout % 64 == 0 && blocks64 >= 64)
    launch<T, S, 64>(x, w1, g1, be1, rm1, rv1, eps1, w2, g2, be2, rm2, rv2,
                     eps2, cat, out, N, C1, Cq, Cout, st);
  else
    launch<T, S, 32>(x, w1, g1, be1, rm1, rv1, eps1, w2, g2, be2, rm2, rv2,
                     eps2, cat, out, N, C1, Cq, Cout, st);
}

template <typename T>
void launch_s(const at::Tensor& x, const at::Tensor& w1, const at::Tensor& g1,
              const at::Tensor& be1, const at::Tensor& rm1,
              const at::Tensor& rv1, float eps1, const at::Tensor& w2,
              const at::Tensor& g2, const at::Tensor& be2,
              const at::Tensor& rm2, const at::Tensor& rv2, float eps2,
              at::Tensor& cat, at::Tensor& out, int N, int C1, int Cq, int Cout,
              cudaStream_t st) {
  // Running stats are buffers, so they stay fp32 when the module is cast by
  // parameter (the bench) but follow the dtype under a plain ``.half()``.
  if (rv1.scalar_type() == at::kFloat) {
    launch_bm<T, float>(x, w1, g1, be1, rm1, rv1, eps1, w2, g2, be2, rm2, rv2,
                        eps2, cat, out, N, C1, Cq, Cout, st);
  } else {
    launch_bm<T, T>(x, w1, g1, be1, rm1, rv1, eps1, w2, g2, be2, rm2, rv2, eps2,
                    cat, out, N, C1, Cq, Cout, st);
  }
}

}  // namespace

at::Tensor sppf_forward(const at::Tensor& x, const at::Tensor& w1,
                        const at::Tensor& g1, const at::Tensor& be1,
                        const at::Tensor& rm1, const at::Tensor& rv1,
                        double eps1, const at::Tensor& w2, const at::Tensor& g2,
                        const at::Tensor& be2, const at::Tensor& rm2,
                        const at::Tensor& rv2, double eps2) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 4 && x.is_contiguous(),
              "sppf: x must be a contiguous 4D CUDA tensor");
  const int N = (int)x.size(0), C1 = (int)x.size(1);
  TORCH_CHECK(x.size(2) == FK_H && x.size(3) == FK_W, "sppf: plane must be 20x20");
  const int Cq = (int)w1.size(0);
  const int Cout = (int)w2.size(0);
  TORCH_CHECK(C1 % 16 == 0 && Cq % FK_CB1 == 0 && Cout % 32 == 0,
              "sppf: channel counts must be tile aligned");
  TORCH_CHECK(w1.numel() == (int64_t)Cq * C1, "sppf: bad cv1 weight");
  TORCH_CHECK(w2.numel() == (int64_t)Cout * 4 * Cq, "sppf: bad cv2 weight");
  TORCH_CHECK(rv1.scalar_type() == rv2.scalar_type() &&
                  rm1.scalar_type() == rv1.scalar_type() &&
                  rm2.scalar_type() == rv1.scalar_type(),
              "sppf: running stats must share a dtype");

  const at::cuda::OptionalCUDAGuard guard(device_of(x));
  auto st = at::cuda::getCurrentCUDAStream();
  at::Tensor cat = at::empty({N, 4 * Cq, FK_H, FK_W}, x.options());
  at::Tensor out = at::empty({N, Cout, FK_H, FK_W}, x.options());
  if (N == 0) return out;

  const bool sfp32 = rv1.scalar_type() == at::kFloat;
  if (x.scalar_type() == at::kHalf) {
    TORCH_CHECK(sfp32 || rv1.scalar_type() == at::kHalf, "sppf: bad stats dtype");
    launch_s<__half>(x, w1, g1, be1, rm1, rv1, (float)eps1, w2, g2, be2, rm2,
                     rv2, (float)eps2, cat, out, N, C1, Cq, Cout, st);
  } else {
    TORCH_CHECK(x.scalar_type() == at::kBFloat16, "sppf: unsupported dtype");
    TORCH_CHECK(sfp32 || rv1.scalar_type() == at::kBFloat16, "sppf: bad stats dtype");
    launch_s<__nv_bfloat16>(x, w1, g1, be1, rm1, rv1, (float)eps1, w2, g2, be2,
                            rm2, rv2, (float)eps2, cat, out, N, C1, Cq, Cout, st);
  }
  return out;
}
"""

_CPP_SRC = r"""
at::Tensor sppf_forward(const at::Tensor& x, const at::Tensor& w1,
                        const at::Tensor& g1, const at::Tensor& be1,
                        const at::Tensor& rm1, const at::Tensor& rv1,
                        double eps1, const at::Tensor& w2, const at::Tensor& g2,
                        const at::Tensor& be2, const at::Tensor& rm2,
                        const at::Tensor& rv2, double eps2);
"""

_EXT = None
_EXT_FAILED = False


def _arch_list() -> str:
    try:
        major, minor = torch.cuda.get_device_capability()
    except Exception:
        return ""
    return f"{major}.{minor}{'a' if major >= 9 else ''}"


def _ext():
    """The compiled extension, or None when it cannot be built here."""
    global _EXT, _EXT_FAILED
    if _EXT is None and not _EXT_FAILED:
        try:
            from torch.utils.cpp_extension import load_inline

            arch = _arch_list()
            if arch:
                os.environ["TORCH_CUDA_ARCH_LIST"] = arch
            tag = hashlib.sha1(_CUDA_SRC.encode()).hexdigest()[:10]
            _EXT = load_inline(
                name=f"fk_yolo_sppf_{tag}",
                cpp_sources=_CPP_SRC,
                cuda_sources=_CUDA_SRC,
                functions=["sppf_forward"],
                extra_cflags=["-O3"],
                extra_cuda_cflags=[
                    "-O3",
                    "--use_fast_math",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_HALF2_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "--expt-relaxed-constexpr",
                ],
                verbose=False,
            )
        except Exception:
            _EXT_FAILED = True
    return _EXT


class YOLOSPPF(nn.Module):
    def __init__(self, c1: int, c2: int, k: int = 5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = YOLOConv(c1, c_, 1, 1)
        self.cv2 = YOLOConv(c_ * 4, c2, 1, 1)
        self.m = MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self._k = k
        self._args = None
        self._guard = None

    # -- fast path setup --------------------------------------------------
    def _plain_1x1(self, cv: YOLOConv) -> bool:
        conv = cv.conv
        w = conv.weight
        return (
            not getattr(cv, "_is_fused", False)
            and hasattr(cv, "bn")
            and type(cv.act).__name__ == "SiLU"
            and conv.bias is None
            and conv.groups == 1
            and tuple(conv.stride) == (1, 1)
            and tuple(conv.padding) == (0, 0)
            and w.dim() == 4
            and w.shape[2] == 1
            and w.shape[3] == 1
        )

    def _setup(self, x: torch.Tensor):
        """Build the extension argument tuple, or ``False`` if unsupported."""
        if not (x.is_cuda and x.dim() == 4 and x.is_contiguous()):
            return False
        if x.dtype not in (torch.float16, torch.bfloat16):
            return False
        if self._k != 5 or x.shape[2] != 20 or x.shape[3] != 20:
            return False
        if not (self._plain_1x1(self.cv1) and self._plain_1x1(self.cv2)):
            return False
        c1 = int(x.shape[1])
        cq = int(self.cv1.conv.weight.shape[0])
        if c1 % 16 or cq % 16 or int(self.cv2.conv.weight.shape[0]) % 32:
            return False
        if int(self.cv1.conv.weight.shape[1]) != c1:
            return False
        if int(self.cv2.conv.weight.shape[1]) != 4 * cq:
            return False
        ts = []
        for cv in (self.cv1, self.cv2):
            bn = cv.bn
            if not bn.affine or bn.running_mean is None or bn.running_var is None:
                return False
            if cv.conv.weight.dtype != x.dtype or bn.weight.dtype != x.dtype:
                return False
            if bn.running_mean.dtype != bn.running_var.dtype:
                return False
            if bn.running_var.dtype not in (torch.float32, x.dtype):
                return False
            ts += [cv.conv.weight, bn.weight, bn.bias, bn.running_mean,
                   bn.running_var, float(bn.eps)]
        if _ext() is None:
            return False
        return tuple(ts)

    def _baseline(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ``_guard`` is (dtype, C1, cv1, cv2): everything the cached argument
        # tuple assumes and that a caller could still change under us -- N is
        # free, and ``fuse()`` on either conv (which rewrites its weight and
        # drops its BN) has to invalidate the tuple.
        g = self._guard
        if g is not None:
            sz = x.shape
            if (x.dtype is g[0] and sz[1] == g[1] and sz[2] == 20
                    and sz[3] == 20 and not (g[2]._is_fused or g[3]._is_fused)
                    and x.is_contiguous()):
                return _ext().sppf_forward(x, *self._args)
        a = self._setup(x)
        if a is False:
            return self._baseline(x)
        self._args = a
        self._guard = (x.dtype, int(x.shape[1]), self.cv1, self.cv2)
        return _ext().sppf_forward(x, *a)
