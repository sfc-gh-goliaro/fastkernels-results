"""PairBlock for AlphaFold3 (fused CUDA implementation).

Shared pair-representation update block used by PairFormer, MSA module,
and template embedder. Sequence: TriMulOut -> TriMulIn -> TriAttStart ->
TriAttEnd -> SwiGLUTransition.

The captured workload is one tiny shape (z[1,16,16,128], bf16): ~0.3 GFLOP of
math spread over ~130 eager kernel launches, so the reference is entirely
launch-bound. This implementation runs the block as ten hand-written kernels
wired into a single CUDA graph (one graph launch per forward, with programmatic
dependent launch between nodes so each kernel's weight loads overlap its
producer's tail). All GEMMs use warp-level bf16 ``mma.m16n8k16`` against weights
pre-swizzled into mma fragment order on the first call, and every LayerNorm,
sigmoid gate, softmax, mask and residual add is folded into the epilogue of the
kernel that produces the value, so nothing round-trips through memory just to be
scaled or added.

Reference: openfold3/core/model/latent/base_blocks.py PairBlock
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn

from .alphafold3_triangle_multiplication import (
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)
from .alphafold3_triangle_attention import TriangleAttention
from .alphafold3_swiglu_transition import SwiGLUTransition

_CUDA_SRC = r"""
// Fused AlphaFold3 PairBlock for N=16, C=128 (bf16) on sm_100.
//
// The block is tiny (256 rows x 128 channels, ~0.3 GFLOP total) so every kernel
// here is latency-bound, not throughput-bound: the wins come from (a) one CUDA
// graph launch instead of ~130 eager launches, (b) issuing every operand load of
// a tile before the first mma, and (c) 8 warps per block so those latencies
// overlap.
#include <cuda_bf16.h>
#include <cuda_runtime.h>

typedef __nv_bfloat16 bf16;

#define SA_PAD 136           // padded row stride (halves) for 128-wide tiles

__device__ __forceinline__ float rb(float x) {
  return __bfloat162float(__float2bfloat16(x));
}
__device__ __forceinline__ unsigned pk(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<unsigned*>(&v);
}
// exp + fast reciprocal: two SFU ops, ~1e-7 relative error (bf16 has ~4e-3),
// and no Newton refinement, which these epilogues are sensitive to.
__device__ __forceinline__ float sigf(float x) { return __fdividef(1.f, 1.f + __expf(-x)); }
__device__ __forceinline__ float lo_(unsigned v) { return __bfloat162float(*(const bf16*)&v); }
__device__ __forceinline__ float hi_(unsigned v) { return __bfloat162float(((const bf16*)&v)[1]); }

// Programmatic dependent launch: a kernel is allowed to start while its
// producer is still draining, run its (producer-independent) weight loads, and
// only then wait for the producer's writes to be visible.
__device__ __forceinline__ void pdl_wait() {
#if __CUDA_ARCH__ >= 900
  cudaGridDependencySynchronize();
#endif
}
__device__ __forceinline__ void pdl_done() {
#if __CUDA_ARCH__ >= 900
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}

__device__ __forceinline__ void mma_k16(float* c, const unsigned* a, const unsigned* b) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// A-fragment of a 16x16 tile from a row-major smem tile (stride SA halves).
// ldmatrix pulls the whole fragment with one instruction; the row stride must
// keep every lane address 16B aligned (SA_PAD = 136 halves = 17 x 16B).
__device__ __forceinline__ void ldA(unsigned* a, const bf16* sA, int SA, int ks, int lane) {
  const bf16* p = sA + (lane & 15) * SA + ((lane >> 4) << 3) + (ks << 4);
  unsigned addr = (unsigned)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(addr));
}
// B-fragments from pre-swizzled weights: [n/8][k/32][32 lanes][8 halves], i.e.
// two consecutive k-steps per lane so one 16B load feeds two mmas.
__device__ __forceinline__ uint4 ldB(const bf16* W, int KP, int ns, int kp, int lane) {
  return *(const uint4*)(W + (((size_t)ns * KP + kp) * 32 + lane) * 8);
}

// Weight fragments for a 16 x (8*NSUB) x (16*KS) tile. Loading them is split
// from the mma so a kernel can issue the (dependency-free) weight loads before
// its activation prologue and overlap the two memory round trips.
template <int KS, int NSUB>
struct BFrag { uint4 b[KS / 2][NSUB]; };

template <int KS, int NSUB>
__device__ __forceinline__ void loadB(BFrag<KS, NSUB>& f, const bf16* W, int ns0, int lane) {
#pragma unroll
  for (int kp = 0; kp < KS / 2; kp++)
#pragma unroll
    for (int u = 0; u < NSUB; u++) f.b[kp][u] = ldB(W, KS / 2, ns0 + u, kp, lane);
  // keep ptxas from sinking these back down into the mma loop
  asm volatile("" ::: "memory");
}

// Two independent accumulator chains (even / odd k-steps) so the mma latency is
// not fully serialized, summed at the end.
template <int KS, int NSUB>
__device__ __forceinline__ void mmaTile(float acc[NSUB][4], const bf16* sA, int SA,
                                        const BFrag<KS, NSUB>& f, int lane) {
  unsigned af[KS][4];
  float odd[NSUB][4] = {};
#pragma unroll
  for (int ks = 0; ks < KS; ks++) ldA(af[ks], sA, SA, ks, lane);
#pragma unroll
  for (int kp = 0; kp < KS / 2; kp++)
#pragma unroll
    for (int u = 0; u < NSUB; u++) {
      mma_k16(acc[u], af[2 * kp], (const unsigned*)&f.b[kp][u].x);
      mma_k16(odd[u], af[2 * kp + 1], (const unsigned*)&f.b[kp][u].z);
    }
#pragma unroll
  for (int u = 0; u < NSUB; u++)
#pragma unroll
    for (int i = 0; i < 4; i++) acc[u][i] += odd[u][i];
}

template <int KS, int NSUB>
__device__ __forceinline__ void mmaTile2(float ap[NSUB][4], float ag[NSUB][4],
                                         const bf16* sA, int SA, const BFrag<KS, NSUB>& fp,
                                         const BFrag<KS, NSUB>& fg, int lane) {
  unsigned af[KS][4];
  float op[NSUB][4] = {}, og[NSUB][4] = {};
#pragma unroll
  for (int ks = 0; ks < KS; ks++) ldA(af[ks], sA, SA, ks, lane);
#pragma unroll
  for (int kp = 0; kp < KS / 2; kp++)
#pragma unroll
    for (int u = 0; u < NSUB; u++) {
      mma_k16(ap[u], af[2 * kp], (const unsigned*)&fp.b[kp][u].x);
      mma_k16(op[u], af[2 * kp + 1], (const unsigned*)&fp.b[kp][u].z);
      mma_k16(ag[u], af[2 * kp], (const unsigned*)&fg.b[kp][u].x);
      mma_k16(og[u], af[2 * kp + 1], (const unsigned*)&fg.b[kp][u].z);
    }
#pragma unroll
  for (int u = 0; u < NSUB; u++)
#pragma unroll
    for (int i = 0; i < 4; i++) { ap[u][i] += op[u][i]; ag[u][i] += og[u][i]; }
}

// LayerNorm of 16 rows x 128 ch into a padded smem tile.
// NTH threads: NTH/16 per row, 128/(NTH/16) channels each.
template <int NTH>
__device__ __forceinline__ void ln16(bf16* sA, int SA, const bf16* src, int row_global,
                                     const bf16* lnwb, bf16* xout) {
  constexpr int TPR = NTH / 16, CPT = 128 / TPR;
  const int tid = threadIdx.x;
  const int r = tid / TPR, part = tid % TPR;
  const bf16* p = src + (size_t)row_global * 128 + part * CPT;
  bf16 x[CPT];
  if (CPT == 4) {
    *(uint2*)(x) = *(const uint2*)(p);
  } else {
#pragma unroll
    for (int i = 0; i < CPT; i += 8) *(uint4*)(x + i) = *(const uint4*)(p + i);
  }
  bf16 wb[2 * CPT];
#pragma unroll
  for (int i = 0; i < 2 * CPT; i += 8) *(uint4*)(wb + i) = *(const uint4*)(lnwb + part * 2 * CPT + i);
  float v[CPT];
#pragma unroll
  for (int i = 0; i < CPT; i++) v[i] = __bfloat162float(x[i]);
  // Two passes (mean, then sum of squared deviations): E[x^2]-E[x]^2 cancels
  // badly once a row's mean dwarfs its spread, which happens as soon as the
  // upstream activations grow.
  float s = 0.f;
#pragma unroll
  for (int i = 0; i < CPT; i++) s += v[i];
#pragma unroll
  for (int m = 1; m < TPR; m <<= 1) s += __shfl_xor_sync(0xffffffffu, s, m);
  const float mean = s * (1.f / 128.f);
  float ss = 0.f;
#pragma unroll
  for (int i = 0; i < CPT; i++) { const float d = v[i] - mean; ss += d * d; }
#pragma unroll
  for (int m = 1; m < TPR; m <<= 1) ss += __shfl_xor_sync(0xffffffffu, ss, m);
  const float rstd = rsqrtf(ss * (1.f / 128.f) + 1e-5f);
  bf16* d = sA + r * SA + part * CPT;
#pragma unroll
  for (int i = 0; i < CPT / 2; i++)
    *(unsigned*)(d + 2 * i) =
        pk((v[2 * i] - mean) * rstd * __bfloat162float(wb[4 * i]) + __bfloat162float(wb[4 * i + 1]),
           (v[2 * i + 1] - mean) * rstd * __bfloat162float(wb[4 * i + 2]) + __bfloat162float(wb[4 * i + 3]));
  if (xout) {
#pragma unroll
    for (int i = 0; i < CPT; i++) xout[i] = x[i];
  }
}

// ===================================================================== K1 / K3
// LN + {a_p,a_g} -> A, {b_p,b_g} -> B, {g} -> G  (triangle multiplication in)
struct ProjMulP {
  const bf16* src;
  const bf16* mask;
  bf16* zs;
  bf16* ms;
  const bf16* lnwb;
  const bf16* wp[3];
  const bf16* wg[2];
  bf16* out[3];
};

template <int STAGE, int NTH>
__global__ __launch_bounds__(NTH, 1) void k_projmul(ProjMulP p) {
  constexpr int NWP = NTH / 32;
  __shared__ bf16 sA[16 * SA_PAD];
  const int rt = blockIdx.x;           // 16-row tile
  const int gz = blockIdx.y;           // 0:a 1:b 2:g
  const int cs = blockIdx.z;           // 64-channel slice
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
  const int g = lane >> 2, t = (lane & 3) << 1;
  const int row0 = rt * 16 + g, row1 = row0 + 8;
  float m0 = 1.f, m1 = 1.f;
  if (gz < 2) {
    m0 = __bfloat162float(p.mask[row0]);
    m1 = __bfloat162float(p.mask[row1]);
  }
  const int ns0 = cs * NWP + w;
  // Select by branch, not by indexing the pointer array: a runtime index into a
  // param-struct array lands the whole array in local memory.
  const bf16* Wp = (gz == 0) ? p.wp[0] : ((gz == 1) ? p.wp[1] : p.wp[2]);
  const bf16* Wg = (gz == 0) ? p.wg[0] : p.wg[1];
  bf16* dst = (gz == 0) ? p.out[0] : ((gz == 1) ? p.out[1] : p.out[2]);
  BFrag<8, 1> fp, fg;
  loadB(fp, Wp, ns0, lane);
  if (gz < 2) loadB(fg, Wg, ns0, lane);
  constexpr int TPR = NTH / 16, CPT = 128 / TPR;
  bf16 xr[CPT];
  ln16<NTH>(sA, SA_PAD, p.src, rt * 16 + tid / TPR, p.lnwb, xr);
  if (STAGE == 0 && gz == 0 && cs == 0) {
    bf16* zd = p.zs + (size_t)(rt * 16 + tid / TPR) * 128 + (tid % TPR) * CPT;
    if (CPT == 4) *(uint2*)(zd) = *(const uint2*)(xr);
    else {
#pragma unroll
      for (int i = 0; i < CPT; i += 8) *(uint4*)(zd + i) = *(const uint4*)(xr + i);
    }
    if (tid < 16) p.ms[rt * 16 + tid] = p.mask[rt * 16 + tid];
  }
  __syncthreads();

  float cp[1][4] = {}, cg[1][4] = {};
  if (gz < 2) mmaTile2(cp, cg, sA, SA_PAD, fp, fg, lane);
  else        mmaTile(cp, sA, SA_PAD, fp, lane);

#pragma unroll
  for (int u = 0; u < 1; u++) {
    const int col = (ns0 + u) * 8 + t;
#pragma unroll
    for (int h = 0; h < 2; h++) {
      const int row = h ? row1 : row0;
      float v0, v1;
      if (gz < 2) {
        const float mv = h ? m1 : m0;
        v0 = rb(mv * rb(sigf(rb(cg[u][2 * h])))) * rb(cp[u][2 * h]);
        v1 = rb(mv * rb(sigf(rb(cg[u][2 * h + 1])))) * rb(cp[u][2 * h + 1]);
      } else {
        v0 = sigf(rb(cp[u][2 * h]));
        v1 = sigf(rb(cp[u][2 * h + 1]));
      }
      *(unsigned*)(dst + (size_t)row * 128 + col) = pk(v0, v1);
    }
  }
  pdl_done();
}

// ===================================================================== K2 / K4
// triangle multiplication combine: bmm over j, layer_norm_out, linear_z, gate,
// residual add.  Each block owns 4 i x 4 k output rows.
struct CombP {
  const bf16* A; const bf16* B; const bf16* G; const bf16* Zres;
  const bf16* lnwb;
  const bf16* Wz;
  bf16* out;
};

template <int N>
__device__ __forceinline__ void ldv(bf16* d, const bf16* src) {
  if (N == 2) {
    *(unsigned*)d = *(const unsigned*)src;
  } else if (N == 4) {
    *(uint2*)d = *(const uint2*)src;
  } else {
#pragma unroll
    for (int i = 0; i < N; i += 8) *(uint4*)(d + i) = *(const uint4*)(src + i);
  }
}

// TT x TT output rows per block: smaller tiles stage less of A/B per block and
// spread the j-reduction over more SMs, at the cost of leaving part of the
// 16-row mma tile of linear_z idle.
template <int INCOMING, int NTH, int TT>
__global__ __launch_bounds__(NTH, 1) void k_comb(CombP p) {
  constexpr int NWP = NTH / 32;
  constexpr int NSZ = 16 / NWP;                  // linear_z subtiles per warp
  constexpr int NROW = TT * TT;                  // valid output rows
  constexpr int CB = 128 * NROW / NTH;           // bmm channels per thread
  constexpr int NLD = TT * 256 / NTH;            // uint4 stage loads per thread
  __shared__ bf16 sA[TT * 16 * 128];
  __shared__ bf16 sB[TT * 16 * 128];
  __shared__ float sX[NROW * 128];
  __shared__ bf16 sL[16 * SA_PAD];
  const int k0 = blockIdx.x * TT, i0 = blockIdx.y * TT;
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
  const int g = lane >> 2, t = (lane & 3) << 1;

  BFrag<8, NSZ> fz;
  loadB(fz, p.Wz, w * NSZ, lane);
  pdl_wait();

  int erow[2];
  bool ev[2];
#pragma unroll
  for (int h = 0; h < 2; h++) {
    const int m = g + 8 * h;
    ev[h] = (m < NROW);
    const int mm = ev[h] ? m : 0;
    erow[h] = (i0 + mm / TT) * 16 + k0 + mm % TT;
  }
  unsigned gv[NSZ][2], zv[NSZ][2];
#pragma unroll
  for (int u = 0; u < NSZ; u++) {
    const int col = (w * NSZ + u) * 8 + t;
#pragma unroll
    for (int h = 0; h < 2; h++) {
      gv[u][h] = *(const unsigned*)(p.G + (size_t)erow[h] * 128 + col);
      zv[u][h] = *(const unsigned*)(p.Zres + (size_t)erow[h] * 128 + col);
    }
  }
  uint4 la[NLD], lb[NLD];
#pragma unroll
  for (int n = 0; n < NLD; n++) {
    const int u = tid + n * NTH;
    const int r = u >> 4, part = u & 15;         // r: row within the TT*16 block
    const int ii = r >> 4, j = r & 15;
    const int ra = INCOMING ? (j * 16 + i0 + ii) : ((i0 + ii) * 16 + j);
    const int rb2 = INCOMING ? (j * 16 + k0 + ii) : ((k0 + ii) * 16 + j);
    la[n] = *(const uint4*)(p.A + (size_t)ra * 128 + part * 8);
    lb[n] = *(const uint4*)(p.B + (size_t)rb2 * 128 + part * 8);
  }
#pragma unroll
  for (int n = 0; n < NLD; n++) {
    const int u = tid + n * NTH;
    *(uint4*)(sA + (u >> 4) * 128 + (u & 15) * 8) = la[n];
    *(uint4*)(sB + (u >> 4) * 128 + (u & 15) * 8) = lb[n];
  }
  if (NROW < 16) {                               // zero the idle mma rows
    for (int i = tid; i < (16 - NROW) * SA_PAD / 8; i += NTH)
      *(uint4*)(sL + NROW * SA_PAD + i * 8) = make_uint4(0, 0, 0, 0);
  }
  __syncthreads();

  // bmm over j
  {
    const int cg = tid % (128 / CB), rg = tid / (128 / CB);
    const int cb = cg * CB, ii = rg / TT, kk = rg % TT;
    if (rg < NROW) {
      bf16 av[16][CB], bvv[CB];
      float acc[CB];
#pragma unroll
      for (int e = 0; e < CB; e++) acc[e] = 0.f;
#pragma unroll
      for (int j = 0; j < 16; j++) ldv<CB>(av[j], sA + (ii * 16 + j) * 128 + cb);
#pragma unroll
      for (int j = 0; j < 16; j++) {
        ldv<CB>(bvv, sB + (kk * 16 + j) * 128 + cb);
#pragma unroll
        for (int e = 0; e < CB; e++)
          acc[e] += __bfloat162float(av[j][e]) * __bfloat162float(bvv[e]);
      }
#pragma unroll
      for (int e = 0; e < CB; e += 2)
        *(float2*)(sX + rg * 128 + cb + e) = make_float2(acc[e], acc[e + 1]);
    }
  }
  __syncthreads();

  // layer_norm_out over the 128 channels (at most one warp per row)
  {
    constexpr int TPR = (NTH / NROW > 32) ? 32 : NTH / NROW;
    constexpr int CPT = 128 / TPR;
    if (tid < TPR * NROW) {
      const int r = tid / TPR, part = tid % TPR;
      float x[CPT];
#pragma unroll
      for (int i = 0; i < CPT; i++) x[i] = rb(sX[r * 128 + part * CPT + i]);
      float s = 0.f;
#pragma unroll
      for (int i = 0; i < CPT; i++) s += x[i];
#pragma unroll
      for (int m = 1; m < TPR; m <<= 1) s += __shfl_xor_sync(0xffffffffu, s, m);
      const float mean = s * (1.f / 128.f);
      float ss = 0.f;
#pragma unroll
      for (int i = 0; i < CPT; i++) { const float d = x[i] - mean; ss += d * d; }
#pragma unroll
      for (int m = 1; m < TPR; m <<= 1) ss += __shfl_xor_sync(0xffffffffu, ss, m);
      const float rstd = rsqrtf(ss * (1.f / 128.f) + 1e-5f);
      bf16 wb[2 * CPT];
#pragma unroll
      for (int i = 0; i < 2 * CPT; i += 8) *(uint4*)(wb + i) = *(const uint4*)(p.lnwb + part * 2 * CPT + i);
      bf16* d = sL + r * SA_PAD + part * CPT;
#pragma unroll
      for (int i = 0; i < CPT / 2; i++)
        *(unsigned*)(d + 2 * i) =
            pk((x[2 * i] - mean) * rstd * __bfloat162float(wb[4 * i]) + __bfloat162float(wb[4 * i + 1]),
               (x[2 * i + 1] - mean) * rstd * __bfloat162float(wb[4 * i + 2]) + __bfloat162float(wb[4 * i + 3]));
    }
  }
  __syncthreads();

  float acc[NSZ][4] = {};
  mmaTile(acc, sL, SA_PAD, fz, lane);

#pragma unroll
  for (int u = 0; u < NSZ; u++) {
    const int col = (w * NSZ + u) * 8 + t;
#pragma unroll
    for (int h = 0; h < 2; h++) {
      if (!ev[h]) continue;
      *(unsigned*)(p.out + (size_t)erow[h] * 128 + col) =
          pk(lo_(zv[u][h]) + rb(rb(acc[u][2 * h]) * lo_(gv[u][h])),
             hi_(zv[u][h]) + rb(rb(acc[u][2 * h + 1]) * hi_(gv[u][h])));
    }
  }
  pdl_done();
}

// ===================================================================== K5 / K7
// triangle attention projections: LN -> q (scaled), k, v, g (sigmoid), z (bias)
struct AttProjP {
  const bf16* src;
  const bf16* lnwb;
  const bf16* w[4];
  const bf16* wbz;
  bf16* o[4];
  bf16* bz;
};

template <int TRANSPOSED, int NTH>
__global__ __launch_bounds__(NTH, 1) void k_attproj(AttProjP p) {
  constexpr int NWP = NTH / 32, TPR = NTH / 16;
  __shared__ bf16 sA[16 * SA_PAD];
  const int rt = blockIdx.x;
  const int gz = blockIdx.y;          // 0:q 1:k 2:v 3:g
  const int cs = blockIdx.z;          // 64-channel slice
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
  const int rr = rt * 16 + tid / TPR;
  const int srow = TRANSPOSED ? ((rr & 15) * 16 + (rr >> 4)) : rr;
  const int ns0 = cs * NWP + w;
  const bool dobz = (gz == 0 && w == 0 && cs == 0);
  const bf16* Wg2 = (gz == 0) ? p.w[0] : ((gz == 1) ? p.w[1] : ((gz == 2) ? p.w[2] : p.w[3]));
  bf16* dst = (gz == 0) ? p.o[0] : ((gz == 1) ? p.o[1] : ((gz == 2) ? p.o[2] : p.o[3]));
  BFrag<8, 1> fw, fbz;
  loadB(fw, Wg2, ns0, lane);
  if (dobz) loadB(fbz, p.wbz, 0, lane);
  pdl_wait();
  ln16<NTH>(sA, SA_PAD, p.src, srow, p.lnwb, (bf16*)nullptr);
  __syncthreads();

  float acc[1][4] = {};
  mmaTile(acc, sA, SA_PAD, fw, lane);
  float bacc[1][4] = {};
  if (dobz) mmaTile(bacc, sA, SA_PAD, fbz, lane);

  const int g = lane >> 2, t = (lane & 3) << 1;
  const float qscale = 0.1767766952966369f;   // 1/sqrt(32)
#pragma unroll
  for (int u = 0; u < 1; u++) {
    const int col = (ns0 + u) * 8 + t;
#pragma unroll
    for (int h = 0; h < 2; h++) {
      const int row = rt * 16 + g + 8 * h;
      float v0 = rb(acc[u][2 * h]), v1 = rb(acc[u][2 * h + 1]);
      if (gz == 0) { v0 *= qscale; v1 *= qscale; }
      else if (gz == 3) { v0 = sigf(v0); v1 = sigf(v1); }
      *(unsigned*)(dst + (size_t)row * 128 + col) = pk(v0, v1);
    }
  }
  if (dobz) {
#pragma unroll
    for (int h = 0; h < 2; h++) {
      const int row = rt * 16 + g + 8 * h;
      *(unsigned*)(p.bz + (size_t)row * 8 + t) = pk(rb(bacc[0][2 * h]), rb(bacc[0][2 * h + 1]));
    }
  }
  pdl_done();
}

// ===================================================================== K6 / K8
// attention core: scores + biases + softmax, o = P V, gate, linear_o, residual.
// Warps 0..3 own one head each for the attention; all 8 warps share linear_o.
struct AttCoreP {
  const bf16* Q; const bf16* K; const bf16* V; const bf16* G; const bf16* bz;
  const bf16* ms;
  const bf16* Wo;
  const bf16* Zres;
  bf16* out;
  float inf;
};

template <int TRANSPOSED, int NTH>
__global__ __launch_bounds__(NTH, 1) void k_attcore(AttCoreP p) {
  constexpr int NWP = NTH / 32, TPR = NTH / 16, CPT = 128 / TPR;
  __shared__ bf16 sQ[16 * SA_PAD];
  __shared__ bf16 sK[16 * SA_PAD];
  __shared__ bf16 sVT[128 * 18];
  __shared__ bf16 sO[16 * SA_PAD];
  const int ch = blockIdx.x;
  const int ib = blockIdx.y;
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
  const int g = lane >> 2, t = (lane & 3) << 1;

  BFrag<8, 1> fo;
  loadB(fo, p.Wo, ch * NWP + w, lane);
  pdl_wait();
  unsigned zres[1][2];
  {
    const int col = (ch * NWP + w) * 8 + t;
#pragma unroll
    for (int hh = 0; hh < 2; hh++) {
      const int q = g + 8 * hh;
      const int row = TRANSPOSED ? (q * 16 + ib) : (ib * 16 + q);
      zres[0][hh] = *(const unsigned*)(p.Zres + (size_t)row * 128 + col);
    }
  }
  {
    const int r = tid / TPR, part = tid % TPR;
    const size_t off = (size_t)(ib * 16 + r) * 128 + part * CPT;
    bf16 qv[CPT], kv[CPT], v[CPT];
    if (CPT == 4) {
      *(uint2*)(qv) = *(const uint2*)(p.Q + off);
      *(uint2*)(kv) = *(const uint2*)(p.K + off);
      *(uint2*)(v) = *(const uint2*)(p.V + off);
      *(uint2*)(sQ + r * SA_PAD + part * CPT) = *(const uint2*)(qv);
      *(uint2*)(sK + r * SA_PAD + part * CPT) = *(const uint2*)(kv);
    } else {
#pragma unroll
      for (int i = 0; i < CPT; i += 8) {
        *(uint4*)(qv + i) = *(const uint4*)(p.Q + off + i);
        *(uint4*)(kv + i) = *(const uint4*)(p.K + off + i);
        *(uint4*)(v + i) = *(const uint4*)(p.V + off + i);
        *(uint4*)(sQ + r * SA_PAD + part * CPT + i) = *(const uint4*)(qv + i);
        *(uint4*)(sK + r * SA_PAD + part * CPT + i) = *(const uint4*)(kv + i);
      }
    }
#pragma unroll
    for (int i = 0; i < CPT; i++) sVT[(part * CPT + i) * 18 + r] = v[i];
  }
  if (w < 4) {
    const int h = w;
    float bzv[2][4], mbv[2][4];
#pragma unroll
    for (int ns = 0; ns < 2; ns++)
#pragma unroll
      for (int r4 = 0; r4 < 4; r4++) {
        const int q = g + 8 * (r4 >> 1);
        const int kk = ns * 8 + t + (r4 & 1);
        const int mrow = TRANSPOSED ? (kk * 16 + ib) : (ib * 16 + kk);
        mbv[ns][r4] = rb(p.inf * (__bfloat162float(p.ms[mrow]) - 1.f));
        bzv[ns][r4] = __bfloat162float(p.bz[(size_t)(q * 16 + kk) * 8 + h]);
      }
    unsigned gat[4][2];
#pragma unroll
    for (int ns = 0; ns < 4; ns++) {
      const int col = h * 32 + ns * 8 + t;
#pragma unroll
      for (int hh = 0; hh < 2; hh++)
        gat[ns][hh] = *(const unsigned*)(p.G + (size_t)(ib * 16 + g + 8 * hh) * 128 + col);
    }
    __syncthreads();

    float sc[2][4] = {};
    {
      unsigned af[2][4];
      unsigned bf[2][2][2];
#pragma unroll
      for (int ks = 0; ks < 2; ks++) {
#pragma unroll
        for (int ns = 0; ns < 2; ns++) {
          const bf16* bp = sK + (size_t)(ns * 8 + g) * SA_PAD + h * 32 + ks * 16 + t;
          bf[ks][ns][0] = *(const unsigned*)bp;
          bf[ks][ns][1] = *(const unsigned*)(bp + 8);
        }
        ldA(af[ks], sQ, SA_PAD, h * 2 + ks, lane);
      }
#pragma unroll
      for (int ks = 0; ks < 2; ks++)
#pragma unroll
        for (int ns = 0; ns < 2; ns++) mma_k16(sc[ns], af[ks], bf[ks][ns]);
    }
    float e[2][4];
#pragma unroll
    for (int ns = 0; ns < 2; ns++)
#pragma unroll
      for (int r4 = 0; r4 < 4; r4++)
        e[ns][r4] = rb(rb(rb(sc[ns][r4]) + mbv[ns][r4]) + bzv[ns][r4]);
#pragma unroll
    for (int half = 0; half < 2; half++) {
      float m = fmaxf(fmaxf(e[0][2 * half], e[0][2 * half + 1]),
                      fmaxf(e[1][2 * half], e[1][2 * half + 1]));
      m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
      m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
      float sum = 0.f;
#pragma unroll
      for (int ns = 0; ns < 2; ns++)
#pragma unroll
        for (int u = 0; u < 2; u++) {
          const float v = __expf(e[ns][2 * half + u] - m);
          e[ns][2 * half + u] = v; sum += v;
        }
      sum += __shfl_xor_sync(0xffffffffu, sum, 1);
      sum += __shfl_xor_sync(0xffffffffu, sum, 2);
      const float inv = __frcp_rn(sum);
#pragma unroll
      for (int ns = 0; ns < 2; ns++)
#pragma unroll
        for (int u = 0; u < 2; u++) e[ns][2 * half + u] = rb(e[ns][2 * half + u] * inv);
    }
    unsigned pa[4];
    pa[0] = pk(e[0][0], e[0][1]);
    pa[1] = pk(e[0][2], e[0][3]);
    pa[2] = pk(e[1][0], e[1][1]);
    pa[3] = pk(e[1][2], e[1][3]);
    float ov[4][4] = {};
    {
      unsigned bv[4][2];
#pragma unroll
      for (int ns = 0; ns < 4; ns++) {
        const bf16* bp = sVT + (size_t)(h * 32 + ns * 8 + g) * 18 + t;
        bv[ns][0] = *(const unsigned*)bp;
        bv[ns][1] = *(const unsigned*)(bp + 8);
      }
#pragma unroll
      for (int ns = 0; ns < 4; ns++) mma_k16(ov[ns], pa, bv[ns]);
    }
#pragma unroll
    for (int ns = 0; ns < 4; ns++) {
      const int col = h * 32 + ns * 8 + t;
#pragma unroll
      for (int hh = 0; hh < 2; hh++)
        *(unsigned*)(sO + (g + 8 * hh) * SA_PAD + col) =
            pk(rb(ov[ns][2 * hh]) * lo_(gat[ns][hh]),
               rb(ov[ns][2 * hh + 1]) * hi_(gat[ns][hh]));
    }
  } else {
    __syncthreads();
  }
  __syncthreads();

  float acc[1][4] = {};
  mmaTile(acc, sO, SA_PAD, fo, lane);

  {
    const int col = (ch * NWP + w) * 8 + t;
#pragma unroll
    for (int hh = 0; hh < 2; hh++) {
      const int q = g + 8 * hh;
      const int row = TRANSPOSED ? (q * 16 + ib) : (ib * 16 + q);
      *(unsigned*)(p.out + (size_t)row * 128 + col) =
          pk(lo_(zres[0][hh]) + rb(acc[0][2 * hh]),
             hi_(zres[0][hh]) + rb(acc[0][2 * hh + 1]));
    }
  }
  pdl_done();
}

// ======================================================================== K9
// transition: LN -> swiglu hidden [256,512]
struct SwigP {
  const bf16* src;
  const bf16* lnwb;
  const bf16* wa; const bf16* wbb;
  bf16* h;
};
template <int NTH>
__global__ __launch_bounds__(NTH, 1) void k_swiglu(SwigP p) {
  constexpr int NWP = NTH / 32, TPR = NTH / 16;
  __shared__ bf16 sA[16 * SA_PAD];
  const int cs = blockIdx.x;
  const int rt = blockIdx.y;
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
  const int ns0 = cs * NWP + w;
  BFrag<8, 1> fa, fb;
  loadB(fa, p.wa, ns0, lane);
  loadB(fb, p.wbb, ns0, lane);
  pdl_wait();
  ln16<NTH>(sA, SA_PAD, p.src, rt * 16 + tid / TPR, p.lnwb, (bf16*)nullptr);
  __syncthreads();
  float ca[1][4] = {}, cb[1][4] = {};
  mmaTile2(ca, cb, sA, SA_PAD, fa, fb, lane);
  const int g = lane >> 2, t = (lane & 3) << 1;
#pragma unroll
  for (int u = 0; u < 1; u++) {
    const int col = (ns0 + u) * 8 + t;
#pragma unroll
    for (int hh = 0; hh < 2; hh++) {
      const int row = rt * 16 + g + 8 * hh;
      const float a0 = rb(ca[u][2 * hh]), a1 = rb(ca[u][2 * hh + 1]);
      *(unsigned*)(p.h + (size_t)row * 512 + col) =
          pk(rb(a0 * sigf(a0)) * rb(cb[u][2 * hh]),
             rb(a1 * sigf(a1)) * rb(cb[u][2 * hh + 1]));
    }
  }
  pdl_done();
}

// ======================================================================= K10
// transition output: linear_out (K=512) * mask + residual
struct OutP {
  const bf16* h; const bf16* Wout; const bf16* Zres; const bf16* ms;
  bf16* out;
};
template <int NTH>
__global__ __launch_bounds__(NTH, 1) void k_outproj(OutP p) {
  constexpr int NWP = NTH / 32;
  __shared__ bf16 sH[16 * 520];
  const int cs = blockIdx.x;
  const int rt = blockIdx.y;
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
  const int g = lane >> 2, t = (lane & 3) << 1;
  const int col = (cs * NWP + w) * 8 + t;
  uint4 bfr[16];
#pragma unroll
  for (int kp = 0; kp < 16; kp++) bfr[kp] = ldB(p.Wout, 16, cs * NWP + w, kp, lane);
  pdl_wait();
  float mv[2]; unsigned zv[2];
#pragma unroll
  for (int hh = 0; hh < 2; hh++) {
    const int row = rt * 16 + g + 8 * hh;
    mv[hh] = __bfloat162float(p.ms[row]);
    zv[hh] = *(const unsigned*)(p.Zres + (size_t)row * 128 + col);
  }
  constexpr int NLD = 1024 / NTH;
  uint4 lh[NLD];
#pragma unroll
  for (int n = 0; n < NLD; n++) {
    const int u = tid + n * NTH;
    lh[n] = *(const uint4*)(p.h + (size_t)(rt * 16 + (u >> 6)) * 512 + (u & 63) * 8);
  }
#pragma unroll
  for (int n = 0; n < NLD; n++) {
    const int u = tid + n * NTH;
    *(uint4*)(sH + (u >> 6) * 520 + (u & 63) * 8) = lh[n];
  }
  __syncthreads();
  float acc[4][4] = {};
#pragma unroll
  for (int s2 = 0; s2 < 4; s2++) {
    unsigned af[8][4];
#pragma unroll
    for (int u = 0; u < 4; u++) {
      ldA(af[2 * u], sH, 520, u * 8 + 2 * s2, lane);
      ldA(af[2 * u + 1], sH, 520, u * 8 + 2 * s2 + 1, lane);
    }
#pragma unroll
    for (int u = 0; u < 4; u++) {
      mma_k16(acc[u], af[2 * u], (const unsigned*)&bfr[u * 4 + s2].x);
      mma_k16(acc[u], af[2 * u + 1], (const unsigned*)&bfr[u * 4 + s2].z);
    }
  }
#pragma unroll
  for (int hh = 0; hh < 2; hh++) {
    const int row = rt * 16 + g + 8 * hh;
    const float o0 = acc[0][2 * hh] + acc[1][2 * hh] + acc[2][2 * hh] + acc[3][2 * hh];
    const float o1 = acc[0][2 * hh + 1] + acc[1][2 * hh + 1] + acc[2][2 * hh + 1] + acc[3][2 * hh + 1];
    *(unsigned*)(p.out + (size_t)row * 128 + col) =
        pk(lo_(zv[hh]) + rb(rb(o0) * mv[hh]), hi_(zv[hh]) + rb(rb(o1) * mv[hh]));
  }
}
// ====================================================================== host
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>
#include <cstdio>

#ifndef NT_PROJ
#define NT_PROJ 256
#endif
#ifndef TT_COMB
#define TT_COMB 2
#endif
#ifndef NT_COMB
#define NT_COMB 256
#endif
#ifndef NT_APROJ
#define NT_APROJ 256
#endif
#ifndef NT_ACORE
#define NT_ACORE 128
#endif
#ifndef NT_SWIG
#define NT_SWIG 256
#endif
#ifndef NT_OUT
#define NT_OUT 128
#endif
#define NWARP(n) ((n) / 32)

namespace {

// Weight slots, in the order ``PairBlock._prepare`` appends them:
//   0..7    tri_mul_out : ln_in, a_p, a_g, b_p, b_g, g, ln_out, z
//   8..15   tri_mul_in  : same
//   16..22  tri_att_start: ln, q, k, v, g, z(bias), o
//   23..29  tri_att_end : same
//   30..33  pair_transition: ln, swiglu_a, swiglu_b, out
struct Ctx {
  bool ready = false;
  // workspace
  bf16 *Zs, *Ms, *A1, *B1, *G1, *Z1, *A2, *B2, *G2, *Z2, *Q, *K, *V, *G, *BZ, *Z3, *Z4, *H;
  // weights, indexed as in python
  const void* w[64];
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
  cudaGraphNode_t n_first = nullptr, n_last = nullptr;
  ProjMulP pm0, pm1;  CombP cb0, cb1;  AttProjP ap0, ap1;  AttCoreP ac0, ac1;
  SwigP sw; OutP op;
};
// One context per module instance: a PairFormer stack instantiates many of these
// blocks, and each needs its own weights, workspace and graph.
std::vector<Ctx*> g_ctx;

template <class T> const T* WP(Ctx& C, int i) { return (const T*)C.w[i]; }

void fill_params(Ctx& C, const bf16* zin, const bf16* mask, bf16* zout) {
  // ---- tri_mul_out
  C.pm0.src = zin; C.pm0.mask = mask; C.pm0.zs = C.Zs; C.pm0.ms = C.Ms;
  C.pm0.lnwb = WP<bf16>(C, 0);
  C.pm0.wp[0] = WP<bf16>(C, 1); C.pm0.wg[0] = WP<bf16>(C, 2);
  C.pm0.wp[1] = WP<bf16>(C, 3); C.pm0.wg[1] = WP<bf16>(C, 4);
  C.pm0.wp[2] = WP<bf16>(C, 5);
  C.pm0.out[0] = C.A1; C.pm0.out[1] = C.B1; C.pm0.out[2] = C.G1;
  C.cb0.A = C.A1; C.cb0.B = C.B1; C.cb0.G = C.G1; C.cb0.Zres = C.Zs;
  C.cb0.lnwb = WP<bf16>(C, 6); C.cb0.Wz = WP<bf16>(C, 7); C.cb0.out = C.Z1;
  // ---- tri_mul_in
  C.pm1.src = C.Z1; C.pm1.mask = C.Ms; C.pm1.zs = nullptr; C.pm1.ms = nullptr;
  C.pm1.lnwb = WP<bf16>(C, 8);
  C.pm1.wp[0] = WP<bf16>(C, 9); C.pm1.wg[0] = WP<bf16>(C, 10);
  C.pm1.wp[1] = WP<bf16>(C, 11); C.pm1.wg[1] = WP<bf16>(C, 12);
  C.pm1.wp[2] = WP<bf16>(C, 13);
  C.pm1.out[0] = C.A2; C.pm1.out[1] = C.B2; C.pm1.out[2] = C.G2;
  C.cb1.A = C.A2; C.cb1.B = C.B2; C.cb1.G = C.G2; C.cb1.Zres = C.Z1;
  C.cb1.lnwb = WP<bf16>(C, 14); C.cb1.Wz = WP<bf16>(C, 15); C.cb1.out = C.Z2;
  // ---- att start
  C.ap0.src = C.Z2; C.ap0.lnwb = WP<bf16>(C, 16);
  C.ap0.w[0] = WP<bf16>(C, 17); C.ap0.w[1] = WP<bf16>(C, 18);
  C.ap0.w[2] = WP<bf16>(C, 19); C.ap0.w[3] = WP<bf16>(C, 20); C.ap0.wbz = WP<bf16>(C, 21);
  C.ap0.o[0] = C.Q; C.ap0.o[1] = C.K; C.ap0.o[2] = C.V; C.ap0.o[3] = C.G; C.ap0.bz = C.BZ;
  C.ac0.Q = C.Q; C.ac0.K = C.K; C.ac0.V = C.V; C.ac0.G = C.G; C.ac0.bz = C.BZ;
  C.ac0.ms = C.Ms; C.ac0.Wo = WP<bf16>(C, 22); C.ac0.Zres = C.Z2; C.ac0.out = C.Z3; C.ac0.inf = 1e9f;
  // ---- att end
  C.ap1.src = C.Z3; C.ap1.lnwb = WP<bf16>(C, 23);
  C.ap1.w[0] = WP<bf16>(C, 24); C.ap1.w[1] = WP<bf16>(C, 25);
  C.ap1.w[2] = WP<bf16>(C, 26); C.ap1.w[3] = WP<bf16>(C, 27); C.ap1.wbz = WP<bf16>(C, 28);
  C.ap1.o[0] = C.Q; C.ap1.o[1] = C.K; C.ap1.o[2] = C.V; C.ap1.o[3] = C.G; C.ap1.bz = C.BZ;
  C.ac1.Q = C.Q; C.ac1.K = C.K; C.ac1.V = C.V; C.ac1.G = C.G; C.ac1.bz = C.BZ;
  C.ac1.ms = C.Ms; C.ac1.Wo = WP<bf16>(C, 29); C.ac1.Zres = C.Z3; C.ac1.out = C.Z4; C.ac1.inf = 1e9f;
  // ---- transition
  C.sw.src = C.Z4; C.sw.lnwb = WP<bf16>(C, 30);
  C.sw.wa = WP<bf16>(C, 31); C.sw.wbb = WP<bf16>(C, 32); C.sw.h = C.H;
  C.op.h = C.H; C.op.Wout = WP<bf16>(C, 33); C.op.Zres = C.Z4; C.op.ms = C.Ms; C.op.out = zout;
}

void launch_all(Ctx& C, cudaStream_t st) {
  k_projmul<0, NT_PROJ><<<dim3(16, 3, 16 / NWARP(NT_PROJ)), NT_PROJ, 0, st>>>(C.pm0);
  k_comb<0, NT_COMB, TT_COMB><<<dim3(16 / TT_COMB, 16 / TT_COMB), NT_COMB, 0, st>>>(C.cb0);
  k_projmul<1, NT_PROJ><<<dim3(16, 3, 16 / NWARP(NT_PROJ)), NT_PROJ, 0, st>>>(C.pm1);
  k_comb<1, NT_COMB, TT_COMB><<<dim3(16 / TT_COMB, 16 / TT_COMB), NT_COMB, 0, st>>>(C.cb1);
  k_attproj<0, NT_APROJ><<<dim3(16, 4, 16 / NWARP(NT_APROJ)), NT_APROJ, 0, st>>>(C.ap0);
  k_attcore<0, NT_ACORE><<<dim3(16 / NWARP(NT_ACORE), 16), NT_ACORE, 0, st>>>(C.ac0);
  k_attproj<1, NT_APROJ><<<dim3(16, 4, 16 / NWARP(NT_APROJ)), NT_APROJ, 0, st>>>(C.ap1);
  k_attcore<1, NT_ACORE><<<dim3(16 / NWARP(NT_ACORE), 16), NT_ACORE, 0, st>>>(C.ac1);
  k_swiglu<NT_SWIG><<<dim3(64 / NWARP(NT_SWIG), 16), NT_SWIG, 0, st>>>(C.sw);
  k_outproj<NT_OUT><<<dim3(16 / NWARP(NT_OUT), 16), NT_OUT, 0, st>>>(C.op);
}

struct NodeSpec { void* func; dim3 grid; void* arg; int nt; };

void build_graph(Ctx& C) {
  if (C.exec) { cudaGraphExecDestroy(C.exec); C.exec = nullptr; }
  if (C.graph) { cudaGraphDestroy(C.graph); C.graph = nullptr; }
  NodeSpec sp[10] = {
      {(void*)k_projmul<0, NT_PROJ>, dim3(16, 3, 16 / NWARP(NT_PROJ)), &C.pm0, NT_PROJ},
      {(void*)k_comb<0, NT_COMB, TT_COMB>, dim3(16 / TT_COMB, 16 / TT_COMB), &C.cb0, NT_COMB},
      {(void*)k_projmul<1, NT_PROJ>, dim3(16, 3, 16 / NWARP(NT_PROJ)), &C.pm1, NT_PROJ},
      {(void*)k_comb<1, NT_COMB, TT_COMB>, dim3(16 / TT_COMB, 16 / TT_COMB), &C.cb1, NT_COMB},
      {(void*)k_attproj<0, NT_APROJ>, dim3(16, 4, 16 / NWARP(NT_APROJ)), &C.ap0, NT_APROJ},
      {(void*)k_attcore<0, NT_ACORE>, dim3(16 / NWARP(NT_ACORE), 16),  &C.ac0, NT_ACORE},
      {(void*)k_attproj<1, NT_APROJ>, dim3(16, 4, 16 / NWARP(NT_APROJ)), &C.ap1, NT_APROJ},
      {(void*)k_attcore<1, NT_ACORE>, dim3(16 / NWARP(NT_ACORE), 16),  &C.ac1, NT_ACORE},
      {(void*)k_swiglu<NT_SWIG>,     dim3(64 / NWARP(NT_SWIG), 16),    &C.sw,  NT_SWIG},
      {(void*)k_outproj<NT_OUT>,     dim3(16 / NWARP(NT_OUT), 16),     &C.op,  NT_OUT},
  };
  cudaGraphCreate(&C.graph, 0);
  cudaGraphNode_t prev = nullptr, node = nullptr;
  bool pdl = true;
  for (int i = 0; i < 10; i++) {
    cudaKernelNodeParams np = {};
    np.func = sp[i].func;
    np.gridDim = sp[i].grid;
    np.blockDim = dim3(sp[i].nt, 1, 1);
    np.sharedMemBytes = 0;
    void* args[1] = {sp[i].arg};
    np.kernelParams = args;
    np.extra = nullptr;
    auto e = cudaGraphAddKernelNode(&node, C.graph, nullptr, 0, &np);
    if (e != cudaSuccess) { printf("[pairblock] addnode %d: %s\n", i, cudaGetErrorString(e)); return; }
    if (prev) {
      // Programmatic edge: the consumer may start (and run its weight loads)
      // while the producer is still draining; cudaGridDependencySynchronize()
      // inside the kernel is what actually waits for the producer's stores.
      cudaGraphEdgeData ed = {};
      ed.from_port = cudaGraphKernelNodePortProgrammatic;
      ed.type = cudaGraphDependencyTypeProgrammatic;
      auto ee = cudaGraphAddDependencies(C.graph, &prev, &node, &ed, 1);
      if (ee != cudaSuccess) {
        if (pdl) printf("[pairblock] pdl edge: %s (falling back)\n", cudaGetErrorString(ee));
        pdl = false;
        cudaGraphAddDependencies(C.graph, &prev, &node, nullptr, 1);
      }
    }
    if (i == 0) C.n_first = node;
    if (i == 9) C.n_last = node;
    prev = node;
  }
  auto e = cudaGraphInstantiate(&C.exec, C.graph, 0);
  if (e != cudaSuccess) { printf("[pairblock] instantiate: %s\n", cudaGetErrorString(e)); C.exec = nullptr; }
}

void set_dyn(Ctx& C, const bf16* zin, const bf16* mask, bf16* zout) {
  C.pm0.src = zin; C.pm0.mask = mask; C.op.out = zout;
  cudaKernelNodeParams np = {};
  np.sharedMemBytes = 0; np.extra = nullptr;
  np.blockDim = dim3(NT_PROJ, 1, 1);
  np.func = (void*)k_projmul<0, NT_PROJ>; np.gridDim = dim3(16, 3, 16 / NWARP(NT_PROJ));
  void* a0[1] = {&C.pm0}; np.kernelParams = a0;
  cudaGraphExecKernelNodeSetParams(C.exec, C.n_first, &np);
  np.blockDim = dim3(NT_OUT, 1, 1);
  np.func = (void*)k_outproj<NT_OUT>; np.gridDim = dim3(16 / NWARP(NT_OUT), 16);
  void* a1[1] = {&C.op}; np.kernelParams = a1;
  cudaGraphExecKernelNodeSetParams(C.exec, C.n_last, &np);
}

}  // namespace

int64_t pb_init(std::vector<torch::Tensor> w, std::vector<torch::Tensor> ws, int64_t handle) {
  TORCH_CHECK(w.size() == 34, "expected 34 weight tensors");
  TORCH_CHECK(ws.size() == 18, "expected 18 workspace tensors");
  if (handle < 0) { g_ctx.push_back(new Ctx()); handle = (int64_t)g_ctx.size() - 1; }
  TORCH_CHECK(handle < (int64_t)g_ctx.size(), "bad handle");
  Ctx& C = *g_ctx[handle];
  for (size_t i = 0; i < w.size(); i++) C.w[i] = w[i].data_ptr();
  bf16** dst[18] = {&C.Zs, &C.Ms, &C.A1, &C.B1, &C.G1, &C.Z1, &C.A2, &C.B2, &C.G2,
                    &C.Z2, &C.Q, &C.K, &C.V, &C.G, &C.BZ, &C.Z3, &C.Z4, &C.H};
  for (int i = 0; i < 18; i++) *dst[i] = (bf16*)ws[i].data_ptr();
  fill_params(C, nullptr, nullptr, nullptr);
  // A re-init means new weight/workspace pointers: drop any graph built against
  // the old ones so the next call rebuilds it.
  if (C.exec) { cudaGraphExecDestroy(C.exec); C.exec = nullptr; }
  if (C.graph) { cudaGraphDestroy(C.graph); C.graph = nullptr; }
  C.ready = true;
  return handle;
}

void pb_run(int64_t handle, torch::Tensor z, torch::Tensor mask, torch::Tensor out,
            bool use_graph) {
  TORCH_CHECK(handle >= 0 && handle < (int64_t)g_ctx.size(), "bad handle");
  Ctx& C = *g_ctx[handle];
  const bf16* zp = (const bf16*)z.data_ptr();
  const bf16* mp = (const bf16*)mask.data_ptr();
  bf16* op = (bf16*)out.data_ptr();
  auto st = at::cuda::getCurrentCUDAStream();
  if (!use_graph) {
    fill_params(C, zp, mp, op);
    launch_all(C, st);
    return;
  }
  if (!C.exec) { fill_params(C, zp, mp, op); build_graph(C); }
  if (!C.exec) {                       // graph unavailable: plain stream launches
    fill_params(C, zp, mp, op);
    launch_all(C, st);
    return;
  }
  set_dyn(C, zp, mp, op);
  auto e = cudaGraphLaunch(C.exec, st);
  if (e != cudaSuccess) {              // fall back for the rest of the run
    printf("[pairblock] graph launch failed (%s); using stream launches\n",
           cudaGetErrorString(e));
    cudaGraphExecDestroy(C.exec);
    C.exec = nullptr;
    fill_params(C, zp, mp, op);
    launch_all(C, st);
  }
}


"""

_CPP_SRC = r"""
#include <torch/extension.h>
#include <vector>
int64_t pb_init(std::vector<torch::Tensor> w, std::vector<torch::Tensor> ws, int64_t handle);
void pb_run(int64_t handle, torch::Tensor z, torch::Tensor mask, torch::Tensor out,
            bool use_graph);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("init", &pb_init);
  m.def("run", &pb_run);
}
"""

_EXT = None
_EXT_FAILED = False


def _ext():
    """JIT-build (once) the fused PairBlock extension, or None if unavailable."""
    global _EXT, _EXT_FAILED
    if _EXT is None and not _EXT_FAILED:
        from torch.utils.cpp_extension import load_inline
        major, minor = torch.cuda.get_device_capability()
        arch = f"{major}.{minor}" + ("a" if major in (9, 10, 12) else "")
        # Pin the build to the local architecture: the ambient list usually spans
        # older archs, and bf16 ``mma`` / ldmatrix do not exist below sm_80.
        os.environ["TORCH_CUDA_ARCH_LIST"] = arch
        try:
            _EXT = load_inline(
                name="fk_af3_pairblock",
                cpp_sources=_CPP_SRC,
                cuda_sources=_CUDA_SRC,
                extra_cflags=["-O3"],
                extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr"],
                verbose=False,
            )
        except Exception as exc:   # no nvcc / unsupported arch -> reference path
            print(f"[PairBlock] fused kernel unavailable ({type(exc).__name__}: "
                  f"{str(exc)[:200]}); falling back to the reference path")
            _EXT_FAILED = True
    return _EXT


def _frag(w: torch.Tensor) -> torch.Tensor:
    """Re-order a [N_out, K] bf16 weight into ``mma.m16n8k16`` B-fragment order.

    Layout: ``[N_out/8][K/32][32 lanes][8 halves]`` -- the B operands of two
    consecutive k-steps, so a warp picks both up with one 16-byte load per lane.
    """
    n_out, k = w.shape
    assert n_out % 8 == 0 and k % 32 == 0
    dev = w.device
    lane = torch.arange(32, device=dev)
    g = (lane // 4).view(1, 1, 32)
    t = ((lane % 4) * 2).view(1, 1, 32)
    n = torch.arange(n_out // 8, device=dev).view(-1, 1, 1) * 8 + g
    kb = torch.arange(k // 32, device=dev).view(1, -1, 1) * 32 + t
    idx = n.unsqueeze(-1) * k + torch.stack(
        [kb, kb + 1, kb + 8, kb + 9, kb + 16, kb + 17, kb + 24, kb + 25], -1)
    return w.reshape(-1)[idx.reshape(-1)].contiguous()


def _lnwb(mod: nn.Module) -> torch.Tensor:
    """LayerNorm weight/bias interleaved as bf16 pairs (one 16B load per 8 ch)."""
    w = mod.weight.detach().to(torch.bfloat16)
    b = mod.bias.detach().to(torch.bfloat16)
    return torch.stack([w, b], -1).reshape(-1).contiguous()


class PairBlock(nn.Module):
    """Shared pair stack block for AF3 PairFormer / MSA module / template.

    Args:
        c_z: Pair embedding channel dimension
        c_hidden_mul: Hidden dim for triangle multiplication
        c_hidden_pair_att: Per-head hidden dim for triangle attention
        no_heads_pair: Number of heads in triangle attention
        transition_n: Scale of pair transition hidden dimension
        pair_dropout: Dropout rate (unused in inference baseline)
        inf: Large constant for masking
    """

    def __init__(
        self,
        c_z: int,
        c_hidden_mul: int,
        c_hidden_pair_att: int,
        no_heads_pair: int,
        transition_n: int,
        pair_dropout: float = 0.0,
        fuse_projection_weights: bool = False,
        inf: float = 1e9,
    ):
        super().__init__()

        self.tri_mul_out = TriangleMultiplicationOutgoing(c_z, c_hidden_mul)
        self.tri_mul_in = TriangleMultiplicationIncoming(c_z, c_hidden_mul)

        self.tri_att_start = TriangleAttention(
            c_z, c_hidden_pair_att, no_heads_pair, starting=True, inf=inf,
        )
        self.tri_att_end = TriangleAttention(
            c_z, c_hidden_pair_att, no_heads_pair, starting=False, inf=inf,
        )

        self.pair_transition = SwiGLUTransition(c_in=c_z, n=transition_n)

        # The fused path is specialized to the captured configuration; anything
        # else falls back to the reference module composition.
        self._fusable = (
            c_z == 128 and c_hidden_mul == 128 and c_hidden_pair_att == 32
            and no_heads_pair == 4 and transition_n == 4 and inf == 1e9
        )
        self._prepared = False
        self._keep = None
        self._handle = -1
        self._use_graph = os.environ.get("FK_PB_NOGRAPH", "0") != "1"

    # -- fused path setup ---------------------------------------------------
    def _ln(self, mod, out):
        out.append(_lnwb(mod))

    def _prepare(self, device):
        """Pre-swizzle the weights into mma fragment order (once, on first call)."""
        w = []
        for tm in (self.tri_mul_out, self.tri_mul_in):
            self._ln(tm.layer_norm_in, w)
            for lin in (tm.linear_a_p, tm.linear_a_g, tm.linear_b_p, tm.linear_b_g,
                        tm.linear_g):
                w.append(_frag(lin.weight.detach()))
            self._ln(tm.layer_norm_out, w)
            w.append(_frag(tm.linear_z.weight.detach()))
        for ta in (self.tri_att_start, self.tri_att_end):
            self._ln(ta.layer_norm, w)
            mha = ta.mha
            for lin in (mha.linear_q, mha.linear_k, mha.linear_v, mha.linear_g):
                w.append(_frag(lin.weight.detach()))
            wz = ta.linear_z.weight.detach()
            pad = torch.zeros(8 - wz.shape[0], wz.shape[1], device=wz.device, dtype=wz.dtype)
            w.append(_frag(torch.cat([wz, pad], 0).contiguous()))
            w.append(_frag(mha.linear_o.weight.detach()))
        pt = self.pair_transition
        self._ln(pt.layer_norm, w)
        w.append(_frag(pt.swiglu.linear_a.weight.detach()))
        w.append(_frag(pt.swiglu.linear_b.weight.detach()))
        w.append(_frag(pt.linear_out.weight.detach()))

        bf = dict(device=device, dtype=torch.bfloat16)
        ws = [torch.empty(256, 128, **bf)]                     # Zs
        ws.append(torch.empty(256, **bf))                      # Ms
        for _ in range(12):                                    # A1 B1 G1 Z1 A2 B2 G2 Z2 Q K V G
            ws.append(torch.empty(256, 128, **bf))
        ws.append(torch.empty(256, 8, **bf))                   # BZ
        ws.append(torch.empty(256, 128, **bf))                 # Z3
        ws.append(torch.empty(256, 128, **bf))                 # Z4
        ws.append(torch.empty(256, 512, **bf))                 # H
        self._keep = (w, ws)
        self._handle = _ext().init(w, ws, self._handle)
        self._prepared = True

    # -- reference path -----------------------------------------------------
    def _reference(self, z, pair_mask, _mask_trans):
        pair_trans_mask = pair_mask if _mask_trans else None
        z = z + self.tri_mul_out(z, mask=pair_mask)
        z = z + self.tri_mul_in(z, mask=pair_mask)
        z = z + self.tri_att_start(z, mask=pair_mask)
        z = z + self.tri_att_end(z, mask=pair_mask)
        z = z + self.pair_transition(z, mask=pair_trans_mask)
        return z

    def forward(
        self,
        z: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size: int | None = None,
        use_deepspeed_evo_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
        _mask_trans: bool = True,
        _attn_chunk_size: int | None = None,
    ) -> torch.Tensor:
        """
        Args:
            z:         [*, N, N, C_z] pair embedding
            pair_mask: [*, N, N] pair mask

        Returns:
            [*, N, N, C_z] updated pair embedding
        """
        if (self._fusable and _mask_trans and z.is_cuda
                and z.dtype == torch.bfloat16 and pair_mask.dtype == torch.bfloat16
                and z.shape == (1, 16, 16, 128) and pair_mask.shape == (1, 16, 16)
                and z.is_contiguous() and pair_mask.is_contiguous()):
            ext = _ext()
            if ext is not None:
                if not self._prepared:
                    self._prepare(z.device)
                out = torch.empty_like(z)
                ext.run(self._handle, z, pair_mask, out, self._use_graph)
                return out
        return self._reference(z, pair_mask, _mask_trans)
