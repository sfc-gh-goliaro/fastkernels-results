"""Outer product mean for AlphaFold3 (L2).

Implements AF3 Algorithm 9. Computes an outer product of MSA
representations and averages over the MSA dimension to produce
a pair representation update.

Reference: openfold3/core/model/layers/outer_product_mean.py OuterProductMean

The whole algorithm -- layer norm, both hidden projections, the masked outer
product, the ``c_hidden**2 -> c_z`` projection and the pair-count normalisation
-- runs as a single CUDA kernel: a block owns one output residue row and a
slice of the pair channels, and keeps every intermediate in shared memory. So
the op costs one launch instead of the ~20 eager ones (two of which are
``torch.einsum`` calls, whose host-side cost alone dominates at this size).
Anything the kernel is not specialised for falls back to the reference sequence
of torch ops below.
"""

from __future__ import annotations

import os
import threading

import torch
import torch.nn as nn

from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear

_CUDA_SRC = r"""// Fused AF3 OuterProductMean (Algorithm 9), specialised for
// c_m = 64, c_hidden = 32, c_z = 128 (the AF3 / OpenFold3 shape).
//
// One block owns one output residue row i (and, when ZS > 1, a slice of the
// c_z channels). It runs the whole chain in shared memory:
//
//   ab   = layer_norm(m) @ [W1;W2]^T * mask          (in place over ln)
//   out  = sum_s a[s,i,:] (x) b[s,j,:]               ("outer", 32x32 per j)
//   res  = (out . Wout[z] + bias[z]) / (norm[i,j] + eps)
//
// so the operator costs one launch instead of ~20 eager ones. Fragments are
// moved with ldmatrix (including the transposing form, which is what lets the
// outer product read a^T / b^T straight out of the row-major ab buffer).

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

#define CM 64                  // c_m
#define CH 32                  // c_hidden
#define CZ 128                 // c_z
#define K2 (CH * CH)           // flattened outer-product width
#define LNS (CM + 8)           // ab_s row stride, halves (36 words: 36%32 == 4)
#define W12S (CM + 8)          // w12_s row stride, halves
#ifndef OSC
#define OSC 40                 // out_s c-stride, halves: 20 words, so the 8
#endif                         // c values of an mma store hit 8 bank quads
#define OS (CH * OSC + 8)      // out_s row stride, halves (%32 == 4 in words)
#ifndef KC
#define KC 128                 // k per staged Wout chunk
#endif
#define KCS (KC + 8)           // staged chunk row stride, halves (68 words, %32 == 4)
#define NC (K2 / KC)           // staged chunks
#define NWARP 8
#define NTHREAD (NWARP * 32)
#ifndef ZS
#define ZS 8                   // blocks per output row; each owns CZ/ZS channels
#endif
#define CZB (CZ / ZS)
#ifndef PK
#define PK 4                   // warp groups splitting k in the output GEMM
#endif
#define GN (NWARP / PK)        // warp groups splitting the c_z channels
#define ZT (CZB / 8 / GN)      // output n-tiles per warp
#define KTC (KC / 16)          // k-tiles per staged chunk
#define KTW (KTC / PK)         // ... of which one warp group takes

typedef __nv_bfloat16 bf16;

// Guard the tile split: every one of these has to divide out exactly or warps
// would silently cover the wrong slice of the output.
static_assert(ZT >= 1 && CZB % (8 * GN) == 0, "c_z does not split over warps");
static_assert(KTC % PK == 0, "k-tiles per chunk do not split over warp groups");
static_assert(KC % 8 == 0 && NTHREAD % (KC / 8) == 0 &&
                  CZB % (NTHREAD / (KC / 8)) == 0,
              "Wout staging does not tile over threads");
static_assert(NC >= 2 && NC % 2 == 0, "staging is double buffered");

// ---------------------------------------------------------------------------
// fragment helpers
// ---------------------------------------------------------------------------
__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4],
                                         const uint32_t (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ void mma1688(float (&d)[4], const uint32_t (&a)[2],
                                        uint32_t b) {
  asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(b));
}

__device__ __forceinline__ uint32_t sm_addr(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
// 16x16 row-major tile -> mma A fragment (lane supplies row (l&15), col (l>>4)*8).
__device__ __forceinline__ void ldm(uint32_t (&r)[4], const bf16* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(sm_addr(p)));
}
// same, transposing each 8x8: turns a row-major [k][n] tile into (n,k) fragments.
__device__ __forceinline__ void ldm_t(uint32_t (&r)[4], const bf16* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(sm_addr(p)));
}

__device__ __forceinline__ uint32_t ld32(const bf16* p) {
  return *reinterpret_cast<const uint32_t*>(p);
}
__device__ __forceinline__ void st32(bf16* p, uint32_t v) {
  *reinterpret_cast<uint32_t*>(p) = v;
}
__device__ __forceinline__ uint32_t pack2(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}
__device__ __forceinline__ float rbf(float x) {  // round to bf16, keep in fp32
  return __bfloat162float(__float2bfloat16(x));
}

// smem: [ ab_s | w12_s | out_s | nrm_s ]
//   ab_s  : RT*16 rows of LNS halves, row = n*SP + s (ln, then a|b in place)
//   w12_s : 2*CH rows of W12S halves  (linear_1 stacked on linear_2)
//   out_s : PN rows of OS halves      (the flattened outer product per j)
template <int MT>
__global__ __launch_bounds__(NTHREAD) void opm_kernel(
    const bf16* __restrict__ mm,    // [B, S, N, CM]
    const bf16* __restrict__ msk,   // [B, S, N]
    const bf16* __restrict__ lnw,   // [CM]
    const bf16* __restrict__ lnb,   // [CM]
    const bf16* __restrict__ w1,    // [CH, CM]
    const bf16* __restrict__ w2,    // [CH, CM]
    const bf16* __restrict__ wo,    // [CZ, K2]
    const bf16* __restrict__ bo,    // [CZ]
    bf16* __restrict__ out,         // [B, N, N, CZ]
    int S, int N, int R, int SP, int SPSH, float eps) {
  extern __shared__ __align__(16) char smem[];
  const int NSP = N * SP;                 // live rows of ab_s
  const int RT = (NSP + 15) >> 4;         // m-tiles covering them
  const int PN = (N + 15) & ~15;          // output rows padded to a m-tile
  bf16* ab_s = reinterpret_cast<bf16*>(smem);
  bf16* wq_s = ab_s;                            // [2][CZB][KCS]; ab_s is dead
  const int head = RT * 16 * LNS > 2 * CZB * KCS ? RT * 16 * LNS : 2 * CZB * KCS;
  bf16* w12_s = ab_s + head;
  bf16* out_s = w12_s + 2 * CH * W12S;
  bf16* msk_s = out_s + PN * OS;                // [N*SP], indexed like ab_s rows
  float* nrm_s = reinterpret_cast<float*>(msk_s + ((NSP + 7) & ~7));

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int gid = lane >> 2;
  const int t4 = lane & 3;
#if ZS == 1
  const int zs = 0;
  const int rest = blockIdx.x;
#else
  const int zs = blockIdx.x % ZS;
  const int rest = blockIdx.x / ZS;
#endif
  const int bi = rest / N;
  const int ii = rest - bi * N;
  const bf16* mb = mm + (size_t)bi * R * CM;
  const bf16* maskb = msk + (size_t)bi * R;

  // ---- issue the prologue's global reads together -------------------------
  // [W1;W2] (staged once, read by every warp in phase 2), this thread's slice
  // of the mask, and its m rows are independent: getting all three in flight
  // costs one L2 round trip instead of three.
  const int wrow = tid >> 2, wseg = (tid & 3) * 16;
  const bf16* wsrc = (wrow < CH ? w1 + wrow * CM : w2 + (wrow - CH) * CM) + wseg;
  const uint4 wv0 = *reinterpret_cast<const uint4*>(wsrc);
  const uint4 wv1 = *reinterpret_cast<const uint4*>(wsrc + 8);
  bf16 mkv[2];
#pragma unroll
  for (int u = 0; u < 2; ++u) {
    const int r = tid + u * NTHREAD;
    const int s = r & (SP - 1), n = r >> SPSH;
    mkv[u] = (r < NSP && s < S) ? maskb[s * N + n] : __float2bfloat16(0.f);
  }
  {
    bf16* wd = w12_s + wrow * W12S + wseg;
    *reinterpret_cast<uint4*>(wd) = wv0;
    *reinterpret_cast<uint4*>(wd + 8) = wv1;
#pragma unroll
    for (int u = 0; u < 2; ++u) {
      const int r = tid + u * NTHREAD;
      if (r < NSP) msk_s[r] = mkv[u];
    }
  }
  // ---- phase 1: layer_norm(m) -> ab_s ------------------------------------
  {
    const int h = (tid & 1) << 5;
    uint4 gw[4], gb[4];
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      gw[q] = *reinterpret_cast<const uint4*>(lnw + h + q * 8);
      gb[q] = *reinterpret_cast<const uint4*>(lnb + h + q * 8);
    }
    for (int r = tid >> 1; r < RT * 16; r += NTHREAD / 2) {
      const int s = r & (SP - 1), n = r >> SPSH;
      const bool live = (r < NSP && s < S);
      uint4 v[4] = {make_uint4(0, 0, 0, 0), make_uint4(0, 0, 0, 0),
                    make_uint4(0, 0, 0, 0), make_uint4(0, 0, 0, 0)};
      if (live) {
        const uint4* p =
            reinterpret_cast<const uint4*>(mb + (size_t)(s * N + n) * CM + h);
#pragma unroll
        for (int q = 0; q < 4; ++q) v[q] = p[q];
      }
      float sum = 0.f, sq = 0.f;
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const uint32_t* u = reinterpret_cast<const uint32_t*>(&v[q]);
#pragma unroll
        for (int t = 0; t < 4; ++t) {
          float2 f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u[t]));
          sum += f.x + f.y;
          sq += f.x * f.x + f.y * f.y;
        }
      }
      // the two lanes of a row are adjacent, and the row's validity is uniform
      // across them, so a full-warp mask is safe here
      sum += __shfl_xor_sync(0xffffffffu, sum, 1);
      sq += __shfl_xor_sync(0xffffffffu, sq, 1);
      const float mean = sum * (1.f / CM);
      const float rstd = rsqrtf(sq * (1.f / CM) - mean * mean + 1e-5f);
      uint4 o[4];
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const uint32_t* u = reinterpret_cast<const uint32_t*>(&v[q]);
        const uint32_t* uw = reinterpret_cast<const uint32_t*>(&gw[q]);
        const uint32_t* ub = reinterpret_cast<const uint32_t*>(&gb[q]);
        uint32_t* uo = reinterpret_cast<uint32_t*>(&o[q]);
#pragma unroll
        for (int t = 0; t < 4; ++t) {
          float2 f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u[t]));
          float2 w = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&uw[t]));
          float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&ub[t]));
          uo[t] = live ? pack2((f.x - mean) * rstd * w.x + b.x,
                               (f.y - mean) * rstd * w.y + b.y)
                       : 0u;
        }
      }
      uint4* d = reinterpret_cast<uint4*>(ab_s + r * LNS + h);
#pragma unroll
      for (int q = 0; q < 4; ++q) d[q] = o[q];
    }
  }
  __syncthreads();

  // ---- norm[i][j] = sum_s mask[s,i] * mask[s,j] --------------------------
  if (tid < N) {
    float acc = 0.f;
    for (int s = 0; s < S; ++s)
      acc += __bfloat162float(msk_s[ii * SP + s]) *
             __bfloat162float(msk_s[tid * SP + s]);
    nrm_s[tid] = rbf(rbf(acc) + eps);
  }

  // ---- phase 2: a|b = ln @ [W1;W2]^T * mask, in place over ab_s ----------
  // One warp owns a whole m-tile (all 64 output channels), so it is the only
  // thread group reading those ln rows and may overwrite them with a|b.
  {
    for (int mt = warp; mt < RT; mt += NWARP) {
      float acc[8][4];
#pragma unroll
      for (int v = 0; v < 8; ++v)
#pragma unroll
        for (int t = 0; t < 4; ++t) acc[v][t] = 0.f;
#pragma unroll
      for (int kt = 0; kt < CM / 16; ++kt) {
        uint32_t af[4], bf[4][4];
        ldm(af, ab_s + (mt * 16 + (lane & 15)) * LNS + kt * 16 + ((lane >> 4) << 3));
        // one ldmatrix per 2 n-tiles: {r0,r2} = lower tile, {r1,r3} = upper
#pragma unroll
        for (int v = 0; v < 4; ++v)
          ldm(bf[v], w12_s + (v * 16 + (lane & 15)) * W12S + kt * 16 +
                         ((lane >> 4) << 3));
#pragma unroll
        for (int v = 0; v < 8; ++v) {
          const uint32_t bb[2] = {bf[v >> 1][v & 1], bf[v >> 1][(v & 1) + 2]};
          mma16816(acc[v], af, bb);
        }
      }
      __syncwarp();
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int r = mt * 16 + gid + half * 8;
        const int s = r & (SP - 1), n = r >> SPSH;
        const float mv =
            (r < NSP && s < S) ? __bfloat162float(maskb[s * N + n]) : 0.f;
#pragma unroll
        for (int v = 0; v < 8; ++v)
          st32(ab_s + r * LNS + v * 8 + t4 * 2,
               pack2(rbf(acc[v][half * 2]) * mv, rbf(acc[v][half * 2 + 1]) * mv));
      }
    }
  }
  __syncthreads();

  // ---- phase 3: outer[j][c*CH+e] = sum_s a[s,i,c] * b[s,j,e] -------------
  if (N & 15) {  // pad rows the output m-tile covers but phase 3 never writes
    uint4* p = reinterpret_cast<uint4*>(out_s + N * OS);
    const int n4 = ((PN - N) * OS) >> 3;
    for (int i = tid; i < n4; i += NTHREAD) p[i] = make_uint4(0, 0, 0, 0);
  }
  {
    const int KS = SP >> 3;                     // k-steps of 8 sequences
    const int ls = lane & 7, lo = (lane >> 3) << 3;
    uint32_t ar[2][4];
#pragma unroll
    for (int ks = 0; ks < 2; ++ks)
      if (ks < KS) ldm_t(ar[ks], ab_s + (size_t)(ii * SP + ks * 8 + ls) * LNS + lo);
    for (int j = warp; j < N; j += NWARP) {
      float acc[2][4][4];
#pragma unroll
      for (int u = 0; u < 2; ++u)
#pragma unroll
        for (int v = 0; v < 4; ++v)
#pragma unroll
          for (int t = 0; t < 4; ++t) acc[u][v][t] = 0.f;
#pragma unroll
      for (int ks = 0; ks < 2; ++ks) {
        if (ks >= KS) break;
        uint32_t br[4];
        ldm_t(br, ab_s + (size_t)(j * SP + ks * 8 + ls) * LNS + CH + lo);
#pragma unroll
        for (int u = 0; u < 2; ++u) {
          const uint32_t aa[2] = {ar[ks][u * 2], ar[ks][u * 2 + 1]};
#pragma unroll
          for (int v = 0; v < 4; ++v) mma1688(acc[u][v], aa, br[v]);
        }
      }
      bf16* d = out_s + j * OS + gid * OSC + t4 * 2;
#pragma unroll
      for (int u = 0; u < 2; ++u)
#pragma unroll
        for (int v = 0; v < 4; ++v) {
          bf16* q = d + u * 16 * OSC + v * 8;
          st32(q, pack2(acc[u][v][0], acc[u][v][1]));
          st32(q + 8 * OSC, pack2(acc[u][v][2], acc[u][v][3]));
        }
    }
  }
  __syncthreads();

  // ---- phase 4: res = outer @ Wout^T ------------------------------------
  // Wout is staged through shared memory in k-chunks. Read as mma B fragments
  // straight from global a warp's eight rows are 2 KB apart, so every load
  // costs eight L1 passes; staging turns that into fully coalesced 512 B loads
  // and the fragments then come out of shared conflict-free. Double buffered,
  // so one barrier per chunk and the global latency sits under the previous
  // chunk's mma. Warps split as GN groups over c_z x PK groups over k, the
  // latter folded back together in phase 5.
  float acc[MT][ZT][4];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int q = 0; q < ZT; ++q)
#pragma unroll
      for (int t = 0; t < 4; ++t) acc[mt][q][t] = 0.f;
  {
    constexpr int TPR = KC / 8;                    // threads per staged row
    constexpr int RPP = NTHREAD / TPR;             // rows staged per pass
    constexpr int ROUNDS = CZB / RPP;              // staging passes
    const int gz = warp % GN, pk = warp / GN;     // c_z group / k-split group
    const int srow = tid / TPR, scol = (tid % TPR) * 8;
    const bf16* gsrc = wo + (size_t)(zs * CZB + srow) * K2 + scol;
    bf16* sdst = wq_s + srow * KCS + scol;
    const bf16* ap0 = out_s + (size_t)(lane & 15) * OS + ((lane >> 4) << 3);
    uint4 pre[ROUNDS];
#define OPM_FETCH(c)                                                         \
  _Pragma("unroll")                                                          \
  for (int rr = 0; rr < ROUNDS; ++rr)                                        \
    pre[rr] = *reinterpret_cast<const uint4*>(                               \
        gsrc + (size_t)rr * RPP * K2 + (c) * KC);
#define OPM_PUT(c)                                                           \
  _Pragma("unroll")                                                          \
  for (int rr = 0; rr < ROUNDS; ++rr)                                        \
    *reinterpret_cast<uint4*>(sdst + ((c) & 1) * CZB * KCS +                 \
                              (size_t)rr * RPP * KCS) = pre[rr];
    OPM_FETCH(0)
#pragma unroll
    for (int c = 0; c < NC; ++c) {
      OPM_PUT(c)
      __syncthreads();
      if (c + 1 < NC) { OPM_FETCH(c + 1) }
      const bf16* bp0 = wq_s + (c & 1) * CZB * KCS +
                        (size_t)(gz * ZT * 8 + gid) * KCS + t4 * 2;
#pragma unroll
      for (int uu = 0; uu < KTW; ++uu) {
        const int u = pk * KTW + uu;
        const int kt = c * KTC + u;
        const bf16* ap = ap0 + (kt >> 1) * OSC + (kt & 1) * 16;
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
          uint32_t af[4];
          ldm(af, ap + mt * 16 * OS);
#pragma unroll
          for (int q = 0; q < ZT; ++q) {
            const bf16* bp = bp0 + (size_t)q * 8 * KCS + u * 16;
            const uint32_t bf[2] = {ld32(bp), ld32(bp + 8)};
            mma16816(acc[mt][q], af, bf);
          }
        }
      }
    }
#undef OPM_FETCH
#undef OPM_PUT
  }
  // ---- phase 5: fold the k-split partials, bias, normalise, store --------
#if PK > 1
  {
    const int gz = warp % GN, pk = warp / GN;
    float* red = reinterpret_cast<float*>(wq_s);
    const int slot = ((pk - 1) * GN + gz) * 32 + lane;
    if (pk > 0) {
      __syncthreads();  // wq_s was the staging buffer
#pragma unroll
      for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int q = 0; q < ZT; ++q)
          *reinterpret_cast<float4*>(red + (slot * MT * ZT + mt * ZT + q) * 4) =
              make_float4(acc[mt][q][0], acc[mt][q][1], acc[mt][q][2], acc[mt][q][3]);
      __syncthreads();
    } else {
      __syncthreads();
      __syncthreads();
#pragma unroll
      for (int pp = 1; pp < PK; ++pp) {
        const int sl = ((pp - 1) * GN + gz) * 32 + lane;
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
          for (int q = 0; q < ZT; ++q) {
            float4 v = *reinterpret_cast<const float4*>(
                red + (sl * MT * ZT + mt * ZT + q) * 4);
            acc[mt][q][0] += v.x; acc[mt][q][1] += v.y;
            acc[mt][q][2] += v.z; acc[mt][q][3] += v.w;
          }
      }
    }
    if (pk > 0) return;
  }
#endif
  bf16* ob = out + ((size_t)bi * N + ii) * N * CZ;
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
    for (int q = 0; q < ZT; ++q) {
      const int z = zs * CZB + ((warp % GN) * ZT + q) * 8 + t4 * 2;
      const float b0 = __bfloat162float(bo[z]), b1 = __bfloat162float(bo[z + 1]);
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int j = mt * 16 + gid + half * 8;
        if (j >= N) continue;
        const float inv = 1.f / nrm_s[j];
        st32(ob + j * CZ + z,
             pack2(rbf(acc[mt][q][half * 2] + b0) * inv,
                   rbf(acc[mt][q][half * 2 + 1] + b1) * inv));
      }
    }
  }
}

// ---------------------------------------------------------------------------
// host launcher
// ---------------------------------------------------------------------------
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

namespace {
// A registered weight set: the six parameter tensors of one module resolved to
// raw device pointers once. Handing six at::Tensor arguments across pybind per
// call costs more host time than the kernel itself at these sizes, so the
// module registers its weights and then calls with (m, mask, slot).
struct WeightSet {
  at::Tensor keep[6];
  const bf16* lnw; const bf16* lnb;
  const bf16* w1;  const bf16* w2;
  const bf16* wo;  const bf16* bo;
  float eps;
};
std::vector<WeightSet>& registry() {
  static std::vector<WeightSet> r;
  return r;
}
void check_weights(const at::Tensor& lnw, const at::Tensor& lnb, const at::Tensor& w1,
                   const at::Tensor& w2, const at::Tensor& wo, const at::Tensor& bo) {
  TORCH_CHECK(lnw.numel() == CM && lnb.numel() == CM && bo.numel() == CZ, "opm: norm/bias");
  TORCH_CHECK(w1.size(0) == CH && w1.size(1) == CM && w1.is_contiguous(), "opm: w1");
  TORCH_CHECK(w2.size(0) == CH && w2.size(1) == CM && w2.is_contiguous(), "opm: w2");
  TORCH_CHECK(wo.size(0) == CZ && wo.size(1) == K2 && wo.is_contiguous(), "opm: wout");
  TORCH_CHECK(lnw.is_contiguous() && lnb.is_contiguous() && bo.is_contiguous(), "opm: affine");
  for (const at::Tensor& t : {lnw, lnb, w1, w2, wo, bo})
    TORCH_CHECK(t.is_cuda() && t.scalar_type() == at::kBFloat16, "opm: weight dtype");
}
int max_dyn_smem() {
  static int v = [] {
    int dev = 0, n = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&n, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    if (n > 1024) {
      cudaFuncSetAttribute(opm_kernel<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, n);
      cudaFuncSetAttribute(opm_kernel<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, n);
    }
    return n;
  }();
  return v;
}
at::Tensor launch(const at::Tensor& m, const at::Tensor& mask, const WeightSet& w) {
  const int nd = m.dim();
  TORCH_CHECK(nd >= 3 && nd <= 10 && mask.dim() == nd - 1, "opm: rank");
  TORCH_CHECK(m.is_cuda() && m.scalar_type() == at::kBFloat16, "opm: m dtype/device");
  TORCH_CHECK(mask.scalar_type() == at::kBFloat16, "opm: mask dtype");
  TORCH_CHECK(m.is_contiguous() && mask.is_contiguous(), "opm: contiguous");
  const int S = m.size(nd - 3), N = m.size(nd - 2), R = S * N;
  TORCH_CHECK(m.size(nd - 1) == CM && S >= 1 && S <= 16 && N <= 32, "opm: shape");
  TORCH_CHECK(mask.size(nd - 3) == S && mask.size(nd - 2) == N, "opm: mask shape");
  TORCH_CHECK(mask.numel() == m.numel() / CM, "opm: mask batch");
  const int B = (int)(m.numel() / ((int64_t)R * CM));
  const int SP = (S <= 8) ? 8 : 16, SPSH = (S <= 8) ? 3 : 4;
  const int RT = (N * SP + 15) >> 4, PN = (N + 15) & ~15;

  int64_t osz[10];
  int nz = 0;
  for (int d = 0; d < nd - 3; ++d) osz[nz++] = m.size(d);
  osz[nz++] = N;
  osz[nz++] = N;
  osz[nz++] = CZ;
  at::Tensor out = at::empty(at::IntArrayRef(osz, nz), m.options());

  const int red = 2 * (PK - 1) * (NWARP / PK) * 32 * ((N > 16) ? 2 : 1) * ZT * 4;
  const int head = std::max(RT * 16 * LNS, std::max(2 * CZB * KCS, red));
  const size_t shb = (size_t)(head + 2 * CH * W12S + PN * OS +
                              ((N * SP + 7) & ~7)) * 2 + (size_t)N * 4;
  TORCH_CHECK((int)shb <= max_dyn_smem(), "opm: shared memory");
  auto kern = (N > 16) ? opm_kernel<2> : opm_kernel<1>;
  kern<<<B * N * ZS, NTHREAD, shb, c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const bf16*>(m.data_ptr()),
      reinterpret_cast<const bf16*>(mask.data_ptr()),
      w.lnw, w.lnb, w.w1, w.w2, w.wo, w.bo,
      reinterpret_cast<bf16*>(out.data_ptr()), S, N, R, SP, SPSH, w.eps);
  return out;
}
}  // namespace

int64_t opm_register(const at::Tensor& lnw, const at::Tensor& lnb, const at::Tensor& w1,
                     const at::Tensor& w2, const at::Tensor& wo, const at::Tensor& bo,
                     double eps) {
  check_weights(lnw, lnb, w1, w2, wo, bo);
  WeightSet w;
  w.keep[0] = lnw; w.keep[1] = lnb; w.keep[2] = w1;
  w.keep[3] = w2;  w.keep[4] = wo;  w.keep[5] = bo;
  w.lnw = reinterpret_cast<const bf16*>(lnw.data_ptr());
  w.lnb = reinterpret_cast<const bf16*>(lnb.data_ptr());
  w.w1 = reinterpret_cast<const bf16*>(w1.data_ptr());
  w.w2 = reinterpret_cast<const bf16*>(w2.data_ptr());
  w.wo = reinterpret_cast<const bf16*>(wo.data_ptr());
  w.bo = reinterpret_cast<const bf16*>(bo.data_ptr());
  w.eps = (float)eps;
  registry().push_back(std::move(w));
  return (int64_t)registry().size() - 1;
}

at::Tensor opm_run(const at::Tensor& m, const at::Tensor& mask, int64_t slot) {
  return launch(m, mask, registry()[slot]);
}
"""  # noqa: E501 - verbatim kernel source

_CPP_SRC = r"""
#include <torch/extension.h>
int64_t opm_register(const at::Tensor& lnw, const at::Tensor& lnb, const at::Tensor& w1,
                     const at::Tensor& w2, const at::Tensor& wo, const at::Tensor& bo,
                     double eps);
at::Tensor opm_run(const at::Tensor& m, const at::Tensor& mask, int64_t slot);
"""

_EXT = None
_EXT_FAILED = False
_EXT_ERROR = None
_EXT_LOCK = threading.Lock()


def _load_ext():
    """JIT-build (once per process) the fused kernel; None if unavailable.

    The build is pinned to the local GPU's arch: the ambient
    ``TORCH_CUDA_ARCH_LIST`` on these machines spans sm_75..sm_120 and the
    bf16 ``mma`` this kernel is built on needs sm_80+, so a multi-arch build
    fails in ptxas. Restored afterwards so nothing else sees the change.
    """
    global _EXT, _EXT_FAILED, _EXT_ERROR
    with _EXT_LOCK:
        if _EXT is not None or _EXT_FAILED:
            return _EXT
        prev = os.environ.get("TORCH_CUDA_ARCH_LIST")
        try:
            from torch.utils.cpp_extension import load_inline

            major, minor = torch.cuda.get_device_capability()
            if major < 8:
                raise RuntimeError("bf16 mma needs sm_80+")
            os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
            _EXT = load_inline(
                name="fk_af3_outer_product_mean",
                cpp_sources=[_CPP_SRC],
                cuda_sources=[_CUDA_SRC],
                functions=["opm_register", "opm_run"],
                extra_cflags=["-O3"],
                extra_cuda_cflags=["-O3", "--use_fast_math"],
                verbose=False,
            )
        except Exception:  # noqa: BLE001 - no nvcc / old arch: stay eager
            import traceback

            _EXT_ERROR = traceback.format_exc()
            _EXT_FAILED = True
            _EXT = None
        finally:
            if prev is None:
                os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
            else:
                os.environ["TORCH_CUDA_ARCH_LIST"] = prev
        return _EXT


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

        self.layer_norm = LayerNorm(c_m)
        self.linear_1 = Linear(c_m, c_hidden, bias=False)
        self.linear_2 = Linear(c_m, c_hidden, bias=False)
        self.linear_out = Linear(c_hidden ** 2, c_z, bias=True)

        # The fused kernel is specialised for the AF3 channel counts.
        self._fusable = (c_m == 64 and c_hidden == 32 and c_z == 128)
        self._run = None
        self._slot = -1

    def _apply(self, *args, **kwargs):  # noqa: D102 - .to()/.cuda() move params
        self._run = None
        return super()._apply(*args, **kwargs)

    def _setup(self):
        """Register the weights with the extension once; returns the call handle.

        Six ``at::Tensor`` arguments per call cost more host time than the kernel
        itself at these sizes, so the weights are resolved to device pointers
        once and every call passes only ``(m, mask, slot)``.
        """
        ext = _load_ext() if self._fusable else None
        lnw = self.layer_norm.weight
        lnb = self.layer_norm.bias
        if ext is None or lnw is None or lnb is None:
            self._fusable = False
            return None
        try:
            self._slot = ext.opm_register(
                lnw, lnb, self.linear_1.weight, self.linear_2.weight,
                self.linear_out.weight, self.linear_out.bias, float(self.eps),
            )
        except RuntimeError:
            self._fusable = False
            return None
        self._run = ext.opm_run
        return self._run

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
        if mask is not None:
            run = self._run or self._setup()
            if run is not None:
                try:
                    return run(m, mask, self._slot)
                except RuntimeError:
                    pass  # shape the kernel is not specialised for: stay eager

        if mask is None:
            mask = m.new_ones(m.shape[:-1])

        ln = self.layer_norm(m)

        mask = mask.unsqueeze(-1)
        a = self.linear_1(ln) * mask
        b = self.linear_2(ln) * mask

        del ln

        # [*, N_res, N_seq, C]
        a = a.transpose(-2, -3)
        b = b.transpose(-2, -3)

        # [*, N_res, N_res, C, C]
        outer = torch.einsum("...bac,...dae->...bdce", a, b)

        # [*, N_res, N_res, C * C]
        outer = outer.reshape(outer.shape[:-2] + (-1,))

        # [*, N_res, N_res, C_z]
        outer = self.linear_out(outer)

        # Normalization: count valid sequence pairs per residue pair
        norm = torch.einsum("...abc,...adc->...bdc", mask, mask)
        norm = norm + self.eps

        outer = outer / norm

        return outer
