"""YOLOv10 PSA (Partial Self-Attention) block -- one fused CUDA kernel.

The baseline runs the block as ~34 separate CUDA kernels (7 conv+BN+act pairs,
two attention matmuls, a softmax, a split and a cat).  At the captured sizes the
arithmetic is tiny -- about 1 GFLOP for the 4x256x20x20 case -- so essentially
all of the time is per-launch overhead, which on this machine is ~4us per kernel.

So this implementation runs the entire block in a **single cooperative kernel**:
BatchNorm is folded into the preceding convolution once on the host, every 1x1
conv becomes a GEMM over the native NCHW layout with the residual add and SiLU
fused into its epilogue, attention keeps K/V of one (image, head) in shared
memory so the softmax needs no online rescaling and the depthwise 3x3 "pe" term
reads its neighbourhood out of the same tile, and the stage boundaries are
grid-wide barriers (~2us) instead of kernel launches.

Consecutive convolutions with no nonlinearity between them are also composed on
the host -- proj folds into the ffn and ffn2 into cv2, residuals included -- so
the block runs as five stages (cv1, qkv, attention, ffn, cv2) instead of seven:
two fewer barriers for a few percent more arithmetic.

Anything the kernel does not cover (other channel counts, non-fp16, odd spatial
sizes, a build failure) falls back to the baseline module path.
"""

from __future__ import annotations

import hashlib
import os

import torch
import torch.nn as nn

from .yolov10_attention import YOLOAttention
from .yolov10_conv import YOLOConv

_CPP_SRC = r"""
void psa_forward(at::Tensor x, at::Tensor out, at::Tensor w, at::Tensor bias,
                 at::Tensor ws);
"""

_CUDA_SRC = r"""#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
// Fused YOLOv10 PSA block -- device code (Blackwell sm_100).
//
// The whole block -- cv1, attention (qkv / softmax / pe), ffn, cv2 -- runs in ONE
// kernel launch.  At the captured sizes the arithmetic is tiny (~1 GFLOP for the
// 4x256x20x20 case) and the baseline's ~34 kernels are pure launch overhead
// (~4us each here), so the stage boundaries become cooperative_groups grid
// barriers (~1.7us) and the activations stay in a device-side workspace.
//
// BatchNorm is folded into the preceding convolution on the host, so every 1x1
// conv is a GEMM  C[M,N] = W[M,K] @ A[K,N] + bias  over the native NCHW layout
// (N = the 400 spatial positions, contiguous), with the residual add and SiLU
// fused into its epilogue.  Weight and activation tiles are staged through
// shared memory with cp.async -- every chunk is issued before the first wait so
// the loads of one tile overlap each other and the mma pipeline.

#include <cuda_fp16.h>
#include <mma.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;
using namespace nvcuda;

#define NPOS  400          // H*W
#define IMH   20
#define IMW   20
#define CTPI  (NPOS / 16)  // 16-position column tiles per image
#define CFULL 256          // c1 == c2
#define CHALF 128          // c = c1/2
#define KDIM  32           // attention key dim per head
#define HDIM  64           // attention head dim
#define NWARP 8
#define NTHR  (NWARP * 32)
#define KCH   64           // GEMM k-chunk (one cp.async group)
#define BQA   32           // queries per attention tile

// Packed weights (halves).  Between two nonlinearities the convolutions compose,
// so proj is folded into the ffn and ffn2 into cv2 on the host:
//   T   = SiLU(Wf1 @ b0 + (Wf1 Wp) @ Z + bf1')                  [K = 256]
//   out = SiLU(W6a @ a + W6b @ b0 + (W6b Wp) @ Z
//                       + (W6b Wf2) @ T + b6')                  [K = 640]
// That trades a few extra MACs for two fewer stages -- and two fewer grid
// barriers, which at these sizes cost more than the arithmetic.
#define OFF_W1  0        // cv1        (256 x 256)
#define OFF_WQ  65536    // qkv        (256 x 128), q rows pre-scaled by 1/sqrt(kd)
#define OFF_WF1 98304    // ffn1'      (256 x 256)  [Wf1 | Wf1 Wp]
#define OFF_W6  163840   // cv2'       (256 x 640)  [W6a | W6b | W6b Wp | W6b Wf2]
#define OFF_WPE 327680   // pe         (128 x 9)
#define WBUF_N  328832
// packed bias offsets (floats)
#define OFF_B1  0
#define OFF_BQ  256
#define OFF_BF1 512
#define OFF_B6  768
#define OFF_BPE 1024
#define BBUF_N  1152

__device__ __forceinline__ void cpa16(void *dst, const void *src) {
  unsigned a = (unsigned)__cvta_generic_to_shared(dst);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(a), "l"(src) : "memory");
}
__device__ __forceinline__ void cpa_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N> __device__ __forceinline__ void cpa_wait() {
  asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory");
}
// n is uniform across the block; the K=640 stage has ten k-chunks in flight
__device__ __forceinline__ void cpa_wait_n(int n) {
  switch (n) {
    case 0: cpa_wait<0>(); break;
    case 1: cpa_wait<1>(); break;
    case 2: cpa_wait<2>(); break;
    case 3: cpa_wait<3>(); break;
    case 4: cpa_wait<4>(); break;
    case 5: cpa_wait<5>(); break;
    case 6: cpa_wait<6>(); break;
    case 7: cpa_wait<7>(); break;
    case 8: cpa_wait<8>(); break;
    default: cpa_wait<9>(); break;
  }
}
// x * sigmoid(x) with the fast-math reciprocal: a full-precision float divide is
// ~20 instructions and this epilogue runs on every output element.
__device__ __forceinline__ float siluf(float v) { return __fdividef(v, 1.f + __expf(-v)); }
__device__ __forceinline__ void prefetch_l2(const void *p) {
  asm volatile("prefetch.global.L2 [%0];" ::"l"(p) : "memory");
}
__device__ __forceinline__ void zero16(void *p) { *(uint4 *)p = make_uint4(0, 0, 0, 0); }


// ---------------------------------------------------------------------------
// One GEMM tile:  D[BM, BN] = W[BM, K] @ A[K, BN] (+ residual) (+ SiLU)
//
// The K rows of the A operand come from up to two activation arrays (cv2 reads
// the concatenation of cv1's first half and the ffn output); segment 0 supplies
// rows [0, K0), segment 1 the rest.  BN is BNT column tiles of 16 positions,
// each from one image, so a tile may straddle images.
// ---------------------------------------------------------------------------
// Issue the weight (A) tile of a future GEMM stage.  Called just before a grid
// barrier: the weights do not depend on the stage being waited for, so their
// ~1-2us of load latency disappears into the barrier.  The copies land in one
// cp.async group committed ahead of the B groups, so the in-order group waits
// inside gemm_tile cover them unchanged.
template <int BM, int BNT, int K>
__device__ __forceinline__ void gemm_preload_A(int t, int ntok, const half *__restrict__ W,
                                               char *smem) {
  constexpr int LDA = K + 8;
  constexpr int AUPC = BM * (KCH / 8);
  constexpr int ANI = AUPC / NTHR;
  constexpr int AMSTEP = NTHR / (KCH / 8);
  constexpr int NCH = K / KCH;
  const int tid = threadIdx.x;
  const int m0 = (t / ntok) * BM;
  const int am = tid / (KCH / 8), akk = tid - am * (KCH / 8);
  const half *aw = W + (size_t)(m0 + am) * K + akk * 8;
  half *ad = (half *)smem + am * LDA + akk * 8;
#pragma unroll
  for (int c = 0; c < NCH; ++c)
#pragma unroll
    for (int n = 0; n < ANI; ++n)
      cpa16(ad + (n * AMSTEP) * LDA + c * KCH, aw + (size_t)(n * AMSTEP) * K + c * KCH);
  cpa_commit();
}

template <int BM, int BNT, int K, int SILU, int RESID>
__device__ __forceinline__ void gemm_tile(
    int t, int ntok, int nimg, bool a_ready,
    const half *__restrict__ W, const float *__restrict__ bias,
    const half *__restrict__ S0, int S0ch, int S0off,
    const half *__restrict__ S1, int S1ch, int S1off, int K0,
    half *__restrict__ D, int Dch, int Doff,
    const half *__restrict__ R, int Rch, int Roff,
    char *smem) {
  constexpr int BN = BNT * 16;
  constexpr int LDA = K + 8;
  constexpr int LDB = BN + 8;
  constexpr int LDC = BN + 4;        // fp32 staging pad: keeps the 16-row store off one bank
  constexpr int MT = BM / 16;
  constexpr int WM = MT < NWARP ? MT : NWARP;
  constexpr int WN = NWARP / WM;
  constexpr int TM = MT / WM;
  constexpr int TN = BNT / WN;
  constexpr int NCH = K / KCH;
  constexpr int AUPC = BM * (KCH / 8);      // A 16B units per k-chunk
  constexpr int BUPC = KCH * BNT * 2;       // B 16B units per k-chunk
  constexpr int ANI = AUPC / NTHR, BNI = BUPC / NTHR, ENI = (BM * BN / 8) / NTHR;
  constexpr int AMSTEP = NTHR / (KCH / 8);  // rows covered per A iteration
  constexpr int BKSTEP = NTHR / (BNT * 2);  // k rows covered per B iteration
  constexpr int EMSTEP = NTHR / (BN / 8);   // rows covered per epilogue iteration
  static_assert(MT % WM == 0 && BNT % WN == 0 && K % KCH == 0, "tile config");
  static_assert(AUPC % NTHR == 0 && BUPC % NTHR == 0 && (BM * BN / 8) % NTHR == 0, "tile config");

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int wm = warp / WN, wn = warp % WN;
  const int mi = t / ntok, ti = t - mi * ntok;
  const int m0 = mi * BM;
  const int CT = nimg * CTPI;

  int cimg[BNT], cpos[BNT];
#pragma unroll
  for (int j = 0; j < BNT; ++j) {
    int ct = ti * BNT + j;
    cimg[j] = ct < CT ? ct / CTPI : -1;
    cpos[j] = ct < CT ? (ct % CTPI) * 16 : 0;
  }

  half *Ash = (half *)smem;             // BM x LDA
  half *Bsh = Ash + BM * LDA;           // K  x LDB
  float *Cst = (float *)(Bsh + K * LDB);

  // Every thread's (row, column) assignment is fixed, so all of the address
  // arithmetic -- including the 64-bit image/channel offsets -- hoists out of
  // the copy loops; only the k-chunk offset is added per instruction.
  const int am = tid / (KCH / 8), akk = tid - am * (KCH / 8);
  const half *aw = W + (size_t)(m0 + am) * K + akk * 8;
  half *ad = Ash + am * LDA + akk * 8;

  const int bu = tid & (BNT * 2 - 1), bk = tid / (BNT * 2);
  const int bj = bu >> 1, bhh = bu & 1;
  const bool bvalid = cimg[bj] >= 0;
  const int bimg = bvalid ? cimg[bj] : 0;
  const half *bp0 = S0 + (size_t)bimg * S0ch * NPOS + (size_t)S0off * NPOS + cpos[bj] + bhh * 8;
  const half *bp1 = S1 + (size_t)bimg * S1ch * NPOS + (ptrdiff_t)(S1off - K0) * NPOS + cpos[bj] + bhh * 8;
  half *bd = Bsh + bk * LDB + bj * 16 + bhh * 8;

  // ---- issue every k-chunk up front: one cp.async group per chunk ----
#pragma unroll
  for (int c = 0; c < NCH; ++c) {
    if (!a_ready) {   // set only for the tile whose weights were preloaded across the barrier
#pragma unroll
      for (int n = 0; n < ANI; ++n)
        cpa16(ad + (n * AMSTEP) * LDA + c * KCH, aw + (size_t)(n * AMSTEP) * K + c * KCH);
    }
    const half *bsrc = (c * KCH < K0) ? bp0 : bp1;
#pragma unroll
    for (int n = 0; n < BNI; ++n) {
      int kg = c * KCH + bk + n * BKSTEP;
      half *d = bd + (size_t)(kg - bk) * LDB;
      if (bvalid) cpa16(d, bsrc + (size_t)kg * NPOS);
      else zero16(d);
    }
    cpa_commit();
  }

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) wmma::fill_fragment(acc[i][j], 0.f);

#pragma unroll
  for (int c = 0; c < NCH; ++c) {
    cpa_wait_n(NCH - 1 - c);
    __syncthreads();
#pragma unroll
    for (int ks = c * KCH; ks < (c + 1) * KCH; ks += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af[TM];
      wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bf[TN];
#pragma unroll
      for (int i = 0; i < TM; ++i)
        wmma::load_matrix_sync(af[i], Ash + (wm * TM + i) * 16 * LDA + ks, LDA);
#pragma unroll
      for (int j = 0; j < TN; ++j)
        wmma::load_matrix_sync(bf[j], Bsh + ks * LDB + (wn * TN + j) * 16, LDB);
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) wmma::mma_sync(acc[i][j], af[i], bf[j], acc[i][j]);
    }
  }

#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j)
      wmma::store_matrix_sync(Cst + (wm * TM + i) * 16 * LDC + (wn * TN + j) * 16, acc[i][j], LDC,
                              wmma::mem_row_major);
  __syncthreads();

  // ---- epilogue: + bias (+ residual) (+ SiLU) -> fp16, 16B per store ----
  {
    const int em = tid / (BN / 8), eu = tid - em * (BN / 8);
    const int ej = (eu * 8) >> 4, eoff = (eu * 8) & 15;
    if (cimg[ej] >= 0) {
      const float *cs = Cst + em * LDC + eu * 8;
      const float *bp = bias + m0 + em;
      half *dp = D + (size_t)cimg[ej] * Dch * NPOS + (size_t)(Doff + m0 + em) * NPOS + cpos[ej] + eoff;
      const half *rp =
          R + (size_t)cimg[ej] * Rch * NPOS + (size_t)(Roff + m0 + em) * NPOS + cpos[ej] + eoff;
#pragma unroll
      for (int n = 0; n < ENI; ++n) {
        const int ro = n * EMSTEP;
        const float bv = bp[ro];
        const float *cp = cs + (size_t)ro * LDC;
        float v[8];
#pragma unroll
        for (int q = 0; q < 8; ++q) v[q] = cp[q] + bv;
        if (RESID) {
          uint4 rr = *(const uint4 *)(rp + (size_t)ro * NPOS);
          const half *rh = (const half *)&rr;
#pragma unroll
          for (int q = 0; q < 8; ++q) v[q] += __half2float(rh[q]);
        }
        half o[8];
#pragma unroll
        for (int q = 0; q < 8; ++q) o[q] = __float2half(SILU ? siluf(v[q]) : v[q]);
        *(uint4 *)(dp + (size_t)ro * NPOS) = *(const uint4 *)o;
      }
    }
  }
  __syncthreads();
}

// ---------------------------------------------------------------------------
// Attention tile: BQA queries of one (image, head).
//   S = (Q^T K) * scale   ->  softmax over all 400 keys  ->  O = V P^T
// K and V of the whole (image, head) live in shared memory -- each is read
// exactly once per tile -- so the softmax is a plain two-pass over a full row
// and needs no online rescaling, and the depthwise 3x3 "pe" term reads its
// neighbourhood out of the same V tile.
//
// Layout choices that matter:
//   * Q is transposed while it is loaded and P is written out transposed, so
//     every wmma operand is row-major: a col-major fragment load from shared
//     degenerates into 8 scalar LDS.
//   * S stays fp32 in shared, written straight from the accumulators, so the
//     mma loop has no per-tile staging round trip.  P then overwrites it (every
//     element is in registers before the first write, with a barrier between).
//   * pe reads one 64-element window of V per thread (8 vector loads) and takes
//     all 72 taps from registers instead of 72 conflicting scalar loads.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void attn_tile(int t, int nimg, const half *__restrict__ qkv,
                                          const half *__restrict__ Wpe,
                                          const float *__restrict__ bpe, half *__restrict__ Z,
                                          int Zch, char *smem) {
  constexpr int NQT = (NPOS + BQA - 1) / BQA;
  constexpr int LDK = NPOS + 8, LDV = NPOS + 8, LDQ = KDIM + 8, LDSF = NPOS + 4, LDP = NPOS + 8;

  const int qt = t % NQT;
  const int h = (t / NQT) & 1;
  const int b = t / (NQT * 2);
  const int p0 = qt * BQA;
  const int nq = min(BQA, NPOS - p0);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;

  half *Ksh = (half *)smem;                 // KDIM x LDK
  half *Vsh = Ksh + KDIM * LDK;             // HDIM x LDV
  half *Qsh = Vsh + HDIM * LDV;             // BQA x LDQ (transposed Q; later the pe weights)
  float *Sf = (float *)(Qsh + BQA * LDQ);   // BQA x LDSF, then overwritten by P (NPOS x LDP fp16)
  half *Psh = (half *)Sf;
  float *Ost = Sf + BQA * LDSF;             // HDIM x BQA

  const half *qb = qkv + (size_t)b * CFULL * NPOS + (size_t)h * 128 * NPOS;
  // K (32 rows) and V (64 rows) are contiguous in both qkv and shared memory, so
  // one walk covers both; the row/unit indices step instead of dividing.
  {
    constexpr int UPR = NPOS / 8;
    const half *src = qb + (size_t)KDIM * NPOS;
    int r = tid / UPR, u = tid - r * UPR;
#pragma unroll
    for (int n = 0; n < (KDIM * UPR + NTHR - 1) / NTHR; ++n) {   // K: needed by the S GEMM
      if (r < KDIM) cpa16(Ksh + r * LDK + u * 8, src + (size_t)r * NPOS + u * 8);
      u += NTHR - (NTHR / UPR) * UPR;
      r += NTHR / UPR;
      if (u >= UPR) { u -= UPR; r += 1; }
    }
    cpa_commit();
    r = KDIM + tid / UPR;
    u = tid - (tid / UPR) * UPR;
#pragma unroll
    for (int n = 0; n < (HDIM * UPR + NTHR - 1) / NTHR; ++n) {   // V: not needed until O = V P^T
      if (r < KDIM + HDIM) cpa16(Ksh + r * LDK + u * 8, src + (size_t)r * NPOS + u * 8);
      u += NTHR - (NTHR / UPR) * UPR;
      r += NTHR / UPR;
      if (u >= UPR) { u -= UPR; r += 1; }
    }
  }
  cpa_commit();
  // Q is transposed on the way in (plain loads + scalar stores; 1024 elements)
  for (int i = tid; i < KDIM * (BQA / 8); i += NTHR) {
    int d = i / (BQA / 8), u = i - d * (BQA / 8);
    half v[8];
    if (u * 8 < nq) *(uint4 *)v = *(const uint4 *)(qb + (size_t)d * NPOS + p0 + u * 8);
    else *(uint4 *)v = make_uint4(0, 0, 0, 0);
#pragma unroll
    for (int q = 0; q < 8; ++q) Qsh[(u * 8 + q) * LDQ + d] = v[q];
  }
  cpa_wait<1>();          // K and Q have landed; V is still in flight
  __syncthreads();

  // ---- S = Q^T K (the 1/sqrt(d) scale is folded into the qkv weights) ----
  constexpr int SNT = NPOS / 16;
  constexpr int SMT = BQA / 16;
  for (int idx = warp; idx < SMT * SNT; idx += NWARP) {
    int mt = idx / SNT, nt = idx - mt * SNT;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.f);
#pragma unroll
    for (int ks = 0; ks < KDIM; ks += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bf;
      wmma::load_matrix_sync(af, Qsh + mt * 16 * LDQ + ks, LDQ);
      wmma::load_matrix_sync(bf, Ksh + ks * LDK + nt * 16, LDK);
      wmma::mma_sync(acc, af, bf, acc);
    }
    wmma::store_matrix_sync(Sf + mt * 16 * LDSF + nt * 16, acc, LDSF, wmma::mem_row_major);
  }
  __syncthreads();

  // ---- row softmax over all 400 keys ----
  // Each lane owns two adjacent keys per step so the fp32 reads are 8B and the
  // fp16 writes 4B (half the LSU traffic of scalar accesses), and the four rows
  // a warp owns are interleaved to give the reduction chains some ILP.
  {
    constexpr int RPW = BQA / NWARP;
    constexpr int JIT = (NPOS + 63) / 64;
    float2 v[RPW][JIT];
    float mx[RPW], sum[RPW];
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
      const float *row = Sf + (warp * RPW + r) * LDSF;
#pragma unroll
      for (int it = 0; it < JIT; ++it) {
        int j = 2 * lane + it * 64;
        v[r][it] = (it + 1) * 64 <= NPOS || j < NPOS ? *(const float2 *)(row + j)
                                                    : make_float2(-1e30f, -1e30f);
      }
      mx[r] = -1e30f;
      sum[r] = 0.f;
    }
#pragma unroll
    for (int it = 0; it < JIT; ++it)
#pragma unroll
      for (int r = 0; r < RPW; ++r) mx[r] = fmaxf(mx[r], fmaxf(v[r][it].x, v[r][it].y));
#pragma unroll
    for (int o = 16; o; o >>= 1)
#pragma unroll
      for (int r = 0; r < RPW; ++r) mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffff, mx[r], o));
#pragma unroll
    for (int it = 0; it < JIT; ++it)
#pragma unroll
      for (int r = 0; r < RPW; ++r) {
        bool ok = (it + 1) * 64 <= NPOS || (2 * lane + it * 64) < NPOS;
        v[r][it].x = ok ? __expf(v[r][it].x - mx[r]) : 0.f;
        v[r][it].y = ok ? __expf(v[r][it].y - mx[r]) : 0.f;
        sum[r] += v[r][it].x + v[r][it].y;
      }
#pragma unroll
    for (int o = 16; o; o >>= 1)
#pragma unroll
      for (int r = 0; r < RPW; ++r) sum[r] += __shfl_xor_sync(0xffffffff, sum[r], o);
#pragma unroll
    for (int r = 0; r < RPW; ++r) sum[r] = __fdividef(1.f, sum[r]);
    // every S element is in registers now, so P may overwrite it
    __syncthreads();
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
      half *prow = Psh + (warp * RPW + r) * LDP;
#pragma unroll
      for (int it = 0; it < JIT; ++it) {
        int j = 2 * lane + it * 64;
        if ((it + 1) * 64 <= NPOS || j < NPOS)
          *(half2 *)(prow + j) = __floats2half2_rn(v[r][it].x * sum[r], v[r][it].y * sum[r]);
      }
    }
  }
  // pe weights for this head go into the (now dead) Q tile
  __syncthreads();
  for (int i = tid; i < HDIM * 9; i += NTHR) Qsh[i] = Wpe[(size_t)(h * HDIM) * 9 + i];
  __syncthreads();

  // ---- O = V P^T  (HDIM x BQA) ----
  {
    cpa_wait<0>();
    __syncthreads();
    constexpr int ONT = BQA / 16;
    int mt = warp / ONT, nt = warp - mt * ONT;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2];
    wmma::fill_fragment(acc[0], 0.f);
    wmma::fill_fragment(acc[1], 0.f);
    // two accumulators so the mma chain has ILP; NPOS/16 is odd, hence the tail
#pragma unroll 4
    for (int ks = 0; ks + 32 <= NPOS; ks += 32) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af[2];
      wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> bf[2];
#pragma unroll
      for (int z = 0; z < 2; ++z) {
        wmma::load_matrix_sync(af[z], Vsh + mt * 16 * LDV + ks + z * 16, LDV);
        wmma::load_matrix_sync(bf[z], Psh + ks + z * 16 + nt * 16 * LDP, LDP);
      }
#pragma unroll
      for (int z = 0; z < 2; ++z) wmma::mma_sync(acc[z], af[z], bf[z], acc[z]);
    }
    if (NPOS % 32) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> bf;
      wmma::load_matrix_sync(af, Vsh + mt * 16 * LDV + (NPOS - 16), LDV);
      wmma::load_matrix_sync(bf, Psh + (NPOS - 16) + nt * 16 * LDP, LDP);
      wmma::mma_sync(acc[0], af, bf, acc[0]);
    }
#pragma unroll
    for (int z = 0; z < acc[0].num_elements; ++z) acc[0].x[z] += acc[1].x[z];
    wmma::store_matrix_sync(Ost + mt * 16 * BQA + nt * 16, acc[0], BQA, wmma::mem_row_major);
  }
  __syncthreads();

  // ---- epilogue: + depthwise 3x3 of V, store Z ----
  {
    const int e = tid >> 2, ig = (tid & 3) * 8;
    if (ig < nq) {
      const int c = h * HDIM + e;
      const half *pw = Qsh + e * 9;
      const float *op = Ost + e * BQA + ig;
      const half *vr = Vsh + e * LDV + (p0 + ig) - 24;  // window base (16B aligned)
      float w[9];
#pragma unroll
      for (int q = 0; q < 9; ++q) w[q] = __half2float(pw[q]);
      half win[64];
#pragma unroll
      for (int u = 0; u < 8; ++u) *(uint4 *)(win + u * 8) = *(const uint4 *)(vr + u * 8);
      const float pb = bpe[c];
      half o[8];
#pragma unroll
      for (int q = 0; q < 8; ++q) {
        int p = p0 + ig + q;
        int y = p / IMW, x = p - y * IMW;
        float acc = op[q] + pb;
        {
#pragma unroll
          for (int r = 0; r < 3; ++r) {
            bool oky = (unsigned)(y + r - 1) < (unsigned)IMH;
#pragma unroll
            for (int sx = 0; sx < 3; ++sx) {
              bool ok = oky && (unsigned)(x + sx - 1) < (unsigned)IMW;
              float vv = __half2float(win[24 + q + (r - 1) * IMW + (sx - 1)]);
              acc += ok ? w[r * 3 + sx] * vv : 0.f;
            }
          }
        }
        o[q] = __float2half(acc);
      }
      *(uint4 *)(Z + (size_t)b * Zch * NPOS + (size_t)c * NPOS + p0 + ig) = *(const uint4 *)o;
    }
  }
  __syncthreads();
}

template <int BNT>
__global__ void psa_kernel(const half *__restrict__ X, half *__restrict__ OUT,
                           const half *__restrict__ W, const float *__restrict__ BIAS,
                           half *__restrict__ ws, int nimg) {
  extern __shared__ __align__(16) char smem[];
  cg::grid_group grid = cg::this_grid();
  constexpr int NQT = (NPOS + BQA - 1) / BQA;
  const int CT = nimg * CTPI;
  const int ntok = (CT + BNT - 1) / BNT;
  const int nt256 = 4 * ntok;
  const int ntatt = nimg * 2 * NQT;
  half *Y = ws;
  half *QKV = Y + (size_t)nimg * CFULL * NPOS;
  half *ZT = QKV + (size_t)nimg * CFULL * NPOS;
  constexpr int ZTCH = CHALF + CFULL;
  const int bid = blockIdx.x;
  {
    const char *wp = (const char *)(W + OFF_WQ);
    const int bytes = (WBUF_N - OFF_WQ) * 2;
    for (int o = (bid * NTHR + threadIdx.x) * 128; o < bytes; o += gridDim.x * NTHR * 128)
      prefetch_l2(wp + o);
  }
  for (int t = bid; t < nt256; t += gridDim.x)
    gemm_tile<64, BNT, 256, 1, 0>(t, ntok, nimg, false, W + OFF_W1, BIAS + OFF_B1, X, CFULL, 0, X, CFULL, 0,
                                  256, Y, CFULL, 0, (const half *)nullptr, 0, 0, smem);
  if (bid < nt256) gemm_preload_A<64, BNT, 128>(bid, ntok, W + OFF_WQ, smem);
  grid.sync();
  for (int t = bid; t < nt256; t += gridDim.x)
    gemm_tile<64, BNT, 128, 0, 0>(t, ntok, nimg, t == bid, W + OFF_WQ, BIAS + OFF_BQ, Y, CFULL, CHALF, Y,
                                     CFULL, CHALF, 128, QKV, CFULL, 0, (const half *)nullptr, 0, 0,
                                     smem);
  grid.sync();
  for (int t = bid; t < ntatt; t += gridDim.x)
    attn_tile(t, nimg, QKV, W + OFF_WPE, BIAS + OFF_BPE, ZT, ZTCH, smem);
  if (bid < nt256) gemm_preload_A<64, BNT, 256>(bid, ntok, W + OFF_WF1, smem);
  grid.sync();
  for (int t = bid; t < nt256; t += gridDim.x)
    gemm_tile<64, BNT, 256, 1, 0>(t, ntok, nimg, t == bid, W + OFF_WF1, BIAS + OFF_BF1, Y, CFULL, CHALF, ZT,
                                     ZTCH, 0, 128, ZT, ZTCH, CHALF, (const half *)nullptr, 0, 0,
                                     smem);
  if (bid < nt256) gemm_preload_A<64, BNT, 640>(bid, ntok, W + OFF_W6, smem);
  grid.sync();
  for (int t = bid; t < nt256; t += gridDim.x)
    gemm_tile<64, BNT, 640, 1, 0>(t, ntok, nimg, t == bid, W + OFF_W6, BIAS + OFF_B6, Y, CFULL, 0, ZT, ZTCH,
                                     0, 256, OUT, CFULL, 0, (const half *)nullptr, 0, 0, smem);
}

// shared-memory footprint (bytes) for a given column-tile count
static inline int psa_smem_bytes(int bnt) {
  int bn = bnt * 16;
  int gemm = 64 * (640 + 8) * 2 + 640 * (bn + 8) * 2 + 64 * (bn + 4) * 4;
  int sp = BQA * (NPOS + 4) * 4;  // fp32 S, later overwritten by fp16 P
  if (BQA * (NPOS + 8) * 2 > sp) sp = BQA * (NPOS + 8) * 2;
  int attn = (KDIM * (NPOS + 8) + HDIM * (NPOS + 8) + BQA * (KDIM + 8)) * 2 + sp +
             HDIM * BQA * 4;
  return gemm > attn ? gemm : attn;
}

// ---------------------------------------------------------------------------


void psa_forward(at::Tensor x, at::Tensor out, at::Tensor w, at::Tensor bias, at::Tensor ws) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf && x.is_contiguous(), "psa: bad input");
  TORCH_CHECK(x.dim() == 4 && x.size(1) == CFULL && x.size(2) * x.size(3) == NPOS, "psa: bad shape");
  TORCH_CHECK(w.numel() >= WBUF_N && bias.numel() >= BBUF_N, "psa: weights not packed");
  TORCH_CHECK(ws.numel() >= (int64_t)x.size(0) * (2 * CFULL + CHALF + CFULL) * NPOS,
              "psa: workspace too small");
  const int nimg = (int)x.size(0);
  const int bnt = nimg >= 2 ? 4 : 2;
  const int sh = psa_smem_bytes(bnt);
  auto stream = at::cuda::getCurrentCUDAStream();

  void *kern = bnt == 4 ? (void *)psa_kernel<4> : (void *)psa_kernel<2>;
  // The opt-in shared memory size and the resident-block count are per (kernel,
  // device); cache them so the steady-state call is just the launch.
  static int grid_sz[2][16] = {};
  const int ki = bnt == 4 ? 0 : 1;
  const int dev = (int)x.device().index();
  TORCH_CHECK(dev >= 0 && dev < 16, "psa: unexpected device index");
  if (!grid_sz[ki][dev]) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, sh));
    int nblk = 0, nsm = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nblk, kern, NTHR, sh));
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev));
    TORCH_CHECK(nblk > 0 && nsm > 0, "psa: kernel does not fit for a cooperative launch");
    grid_sz[ki][dev] = nblk * nsm;
  }
  const int ct = nimg * CTPI;
  const int ntok = (ct + bnt - 1) / bnt;
  int need = 4 * ntok;
  int natt = nimg * 2 * ((NPOS + BQA - 1) / BQA);
  if (natt > need) need = natt;
  const int gmax = grid_sz[ki][dev];
  const int grid = gmax < need ? gmax : need;

  const half *xp = (const half *)x.data_ptr<at::Half>();
  half *op = (half *)out.data_ptr<at::Half>();
  const half *wp = (const half *)w.data_ptr<at::Half>();
  const float *bp = bias.data_ptr<float>();
  half *wsp = (half *)ws.data_ptr<at::Half>();
  void *args[] = {(void *)&xp, (void *)&op, (void *)&wp, (void *)&bp, (void *)&wsp, (void *)&nimg};
  C10_CUDA_CHECK(cudaLaunchCooperativeKernel(kern, dim3(grid), dim3(NTHR), args, sh, stream));
}
"""

_EXT = None
_EXT_FAILED = False


def _arch_list() -> str:
    try:
        major, minor = torch.cuda.get_device_capability()
    except Exception:
        return ""
    return f"{major}.{minor}{'a' if major >= 9 else ''}"


def _build_ext():
    from torch.utils.cpp_extension import load_inline

    arch = _arch_list()
    if arch:
        os.environ["TORCH_CUDA_ARCH_LIST"] = arch
    tag = hashlib.sha1(_CUDA_SRC.encode()).hexdigest()[:10]
    return load_inline(
        name=f"fk_yolov10_psa_{tag}",
        cpp_sources=_CPP_SRC,
        cuda_sources=_CUDA_SRC,
        functions=["psa_forward"],
        extra_cflags=["-O3"],
        extra_cuda_cflags=[
            "-O3",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
            "--expt-relaxed-constexpr",
        ],
        verbose=False,
    )


def _ext():
    """The compiled extension, or None if it cannot be built here."""
    global _EXT, _EXT_FAILED
    if _EXT is None and not _EXT_FAILED:
        try:
            _EXT = _build_ext()
        except Exception:
            _EXT_FAILED = True
    return _EXT


def _fuse(conv_bn: YOLOConv):
    """conv(+BN) -> (weight, bias) of the equivalent bias-ed convolution.

    Always returns fresh fp32 tensors: the caller scales rows of them in place.
    """
    w = conv_bn.conv.weight.detach().to(torch.float32, copy=True)
    bn = getattr(conv_bn, "bn", None)
    b = conv_bn.conv.bias
    b = (torch.zeros(w.shape[0], device=w.device, dtype=torch.float32) if b is None
         else b.detach().to(torch.float32, copy=True))
    if bn is None:
        return w, b
    s = bn.weight.float() / torch.sqrt(bn.running_var.float() + bn.eps)
    return w * s.view(-1, 1, 1, 1), b * s + bn.bias.float() - bn.running_mean.float() * s


class YOLOPSA(nn.Module):
    def __init__(self, c1: int, c2: int, e: float = 0.5):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = YOLOConv(c1, 2 * self.c, 1, 1)
        self.cv2 = YOLOConv(2 * self.c, c1, 1, 1)
        self.attn = YOLOAttention(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1))
        self.ffn = nn.Sequential(
            YOLOConv(self.c, self.c * 2, 1, 1),
            YOLOConv(self.c * 2, self.c, 1, 1, act=False),
        )
        self._c1 = c1
        self._packed = None
        self._ws = None

    # -- weight packing (once, lazily: weights are loaded after __init__) --
    @torch.no_grad()
    def _pack(self):
        a = self.attn
        if not (self._c1 == 256 and self.c == 128 and a.num_heads == 2
                and a.key_dim == 32 and a.head_dim == 64):
            self._packed = False
            return
        sc = a.scale
        w1, b1 = _fuse(self.cv1)
        wq, bq = _fuse(a.qkv)
        wp, bp = _fuse(a.proj)
        wf1, bf1 = _fuse(self.ffn[0])
        wf2, bf2 = _fuse(self.ffn[1])
        w6, b6 = _fuse(self.cv2)
        wpe, bpe = _fuse(a.pe)
        # the 1/sqrt(key_dim) attention scale rides on the q rows of qkv
        per = 2 * a.key_dim + a.head_dim
        for h in range(a.num_heads):
            q = slice(h * per, h * per + a.key_dim)
            wq[q] = wq[q] * sc
            bq[q] = bq[q] * sc
        wp2, wf1_2, wf2_2 = wp[:, :, 0, 0], wf1[:, :, 0, 0], wf2[:, :, 0, 0]
        w6a, w6b = w6[:, : self.c, 0, 0], w6[:, self.c :, 0, 0]
        # ffn1 absorbs proj;  cv2 absorbs ffn2 (and proj through the residual)
        wf1m = torch.cat((wf1_2, wf1_2 @ wp2), dim=1)
        bf1m = bf1 + wf1_2 @ bp
        w6m = torch.cat((w6a, w6b, w6b @ wp2, w6b @ wf2_2), dim=1)
        b6m = b6 + w6b @ (bp + bf2)
        ws = [w1.reshape(-1), wq.reshape(-1), wf1m.reshape(-1), w6m.reshape(-1), wpe.reshape(-1)]
        bs = [b1, bq, bf1m, b6m, bpe]
        self._packed = (torch.cat(ws).half().contiguous(), torch.cat(bs).float().contiguous())

    def _load_from_state_dict(self, *args, **kwargs):
        self._packed = None          # weights changed -> repack on the next forward
        return super()._load_from_state_dict(*args, **kwargs)

    def _apply(self, *args, **kwargs):
        self._packed = None          # .to()/.half()/.cuda() move or retype the weights
        self._ws = None
        return super()._apply(*args, **kwargs)

    def _baseline(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._packed is None:
            self._pack()
        ext = _ext()
        if (self._packed is False or ext is None or not x.is_cuda
                or x.dtype != torch.float16 or x.dim() != 4 or x.shape[1] != 256
                or x.shape[2] != 20 or x.shape[3] != 20
                or self._packed[0].device != x.device):
            return self._baseline(x)
        x = x.contiguous()
        n = x.shape[0]
        w, bias = self._packed
        # scratch for the three intermediate activation buffers (Y, QKV, Z|T)
        if self._ws is None or self._ws.shape[0] < n * 896 * 400:
            self._ws = torch.empty(n * 896 * 400, device=x.device, dtype=torch.float16)
        out = torch.empty_like(x)
        ext.psa_forward(x, out, w, bias, self._ws)
        return out
