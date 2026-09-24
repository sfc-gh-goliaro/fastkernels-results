"""Diffusion conditioning for AlphaFold3 (fused single-kernel candidate).

Same module tree / state_dict as the baseline, so bench weight sharing works.
One CUDA kernel does the whole op -- pair stage-1, single stage-1 (including
the Fourier noise embedding) and both SwiGLU transitions of each stream --
across 112 co-resident blocks that hand off through global flag counters.
The baseline's ~90 small kernels are launch- and latency-bound at this size
(16 tokens), so collapsing them into one launch is the whole win.  Any input
the kernel was not specialized for falls back to the reference path.

Reference: openfold3/core/model/layers/diffusion_conditioning.py
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn as nn

from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear
from .alphafold3_input_embedder import relpos_complex
from .alphafold3_swiglu_transition import SwiGLUTransition


__targets__ = ["DiffusionConditioning"]

_EXT = None
_EXT_TRIED = False

_CU_SOURCE = r"""// Fused AlphaFold3 DiffusionConditioning: one kernel, three flag-synced phases.
//
// 112 co-resident blocks of 512 threads cover the whole op:
//   phase 1  stage-1 projections
//            pair:   LN(cat(zij, relpos)) @ Wz            -> z1buf
//            single: LN(cat(si_trunk, si_input)) @ Ws
//                    + linear_n(LN(fourier(t)))           -> sibuf
//   phase 2  first SwiGLU transition of each stream
//   phase 3  second SwiGLU transition of each stream, and the outputs
//
// Blocks are owned by the phase-2 and phase-3 transition slices; the phase-3
// blocks additionally carry all of phase 1 (they would otherwise idle until
// phase 2 retires) and reuse the same shared memory for their own weights
// afterwards.  Weight slices land in shared memory via cp.async, the GEMMs run
// on bf16 m16n8k16 mma with fp32 accumulation and the reference's bf16 rounding
// points reproduced exactly, hidden-dim splits are recombined with v4 fp32
// reductions, and the last arriving block of each group writes the residual.
// Phase hand-off is a release/acquire counter in global memory, so the whole
// op is a single launch.
#include <torch/extension.h>
#include <cuda_bf16.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>

typedef __nv_bfloat16 bf16;

#define NT 16
#define NPAIR 256
#define CZ 128
#define CS 384
#define CIN 449
#define KZ1 267
#define KZ1P 272
#define KS1 833
#define KS1P 848
#define HZ 256
#define HS 768
#define CF 256
#define NRP 139

// Phase 1 (both stage-1 projections) is carried by the phase-3 blocks, which
// would otherwise idle until phase 2 finishes; they reuse the same shared
// memory for their transition weights afterwards.
#define NB_PT  32          // pair transition:   8 rowgroups(32) x 4 hslices(64)
#define NB_ST  24          // single transition: 24 hslices(32)
// phase-1 task split on those same blocks:
//   pair   stage1: 16 rowgroups(16 rows) x 2 nslices(64 ch)  -> NB_PT tasks
//   single stage1: 24 nslices(16 ch)                          -> NB_ST tasks
#define B_PT1 0
#define B_ST1 (B_PT1 + NB_PT)
#define B_PT2 (B_ST1 + NB_ST)
#define B_ST2 (B_PT2 + NB_PT)
#define NBLK  (B_ST2 + NB_ST)
#define NPH1  (NB_PT + NB_ST)   // stage-1 arrivals (= the phase-3 blocks)
#define NPH2  (NB_PT + NB_ST)   // phase-2 arrivals
#define NTHR 512
#define NW   (NTHR / 32)
#define NACC (2 * NPAIR * CZ + 2 * NT * CS)

struct Args {
  const bf16 *zij, *si_trunk, *si_input, *tsc;
  const bf16 *res, *tok, *asym, *ent, *sym, *tmask;
  const bf16 *Wz, *Ws, *Wn, *fw, *fb;
  const float *lnz, *lns, *lnn;
  const bf16 *tWa[4], *tWb[4], *tWo[4];
  const float *tlnw[4], *tlnb[4];
  bf16 *z1buf, *sibuf, *zout, *siout;
  float *acc;
  unsigned long long *ctr;
  unsigned long long gen;
};

// ---- primitives -----------------------------------------------------------
__device__ __forceinline__ float rbf(float x) {
  return __bfloat162float(__float2bfloat16(x));
}
__device__ __forceinline__ void cpa16(void *dst, const void *src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
               :: "r"((unsigned)__cvta_generic_to_shared(dst)), "l"(src) : "memory");
}
__device__ __forceinline__ void cpa_commit() { asm volatile("cp.async.commit_group;\n"); }
template <int N> __device__ __forceinline__ void cpa_wait() {
  asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}
template <int ROWS, int CHUNKS>
__device__ __forceinline__ void cpa2d(char *dst, int dstride, const char *src, int sstride) {
  for (int i = threadIdx.x; i < ROWS * CHUNKS; i += NTHR) {
    int r = i / CHUNKS, c = i - r * CHUNKS;
    cpa16(dst + r * dstride + c * 16, src + (size_t)r * sstride + c * 16);
  }
}
template <int BYTES>
__device__ __forceinline__ void cpa_flat(char *dst, const char *src) {
  for (int i = threadIdx.x; i < BYTES / 16; i += NTHR) cpa16(dst + i * 16, src + i * 16);
}
__device__ __forceinline__ void mma16816(float *d, const unsigned *a, const unsigned *b) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void ldA(unsigned *a, const bf16 *base, int S, int lane) {
  int gid = lane >> 2, tig = lane & 3;
  a[0] = *(const unsigned *)(base + gid * S + tig * 2);
  a[1] = *(const unsigned *)(base + (gid + 8) * S + tig * 2);
  a[2] = *(const unsigned *)(base + gid * S + tig * 2 + 8);
  a[3] = *(const unsigned *)(base + (gid + 8) * S + tig * 2 + 8);
}
__device__ __forceinline__ void ldB(unsigned *b, const bf16 *base, int S, int lane) {
  int gid = lane >> 2, tig = lane & 3;
  b[0] = *(const unsigned *)(base + gid * S + tig * 2);
  b[1] = *(const unsigned *)(base + gid * S + tig * 2 + 8);
}
__device__ __forceinline__ float wsum(float v) {
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
__device__ __forceinline__ void st2bf(bf16 *p, float a, float b) {
  __nv_bfloat162 v = __floats2bfloat162_rn(a, b);
  *(unsigned *)p = *(const unsigned *)&v;
}
__device__ __forceinline__ unsigned long long ldacq(const unsigned long long *p) {
  unsigned long long v;
  asm volatile("ld.acquire.gpu.u64 %0, [%1];\n" : "=l"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ unsigned long long redrel(unsigned long long *p) {
  unsigned long long old;
  asm volatile("atom.add.release.gpu.u64 %0, [%1], 1;\n" : "=l"(old) : "l"(p) : "memory");
  return old;
}
__device__ __forceinline__ void arrive(unsigned long long *p) {
  __syncthreads();
  if (threadIdx.x == 0) redrel(p);
}
__device__ __forceinline__ void wait_for(const unsigned long long *p, unsigned long long tgt) {
  if (threadIdx.x == 0) { while (ldacq(p) < tgt) __nanosleep(64); }
  __syncthreads();
}
__device__ __forceinline__ void red4(float *p, const float *v) {
  asm volatile("red.global.add.v4.f32 [%0], {%1,%2,%3,%4};\n"
               :: "l"(p), "f"(v[0]), "f"(v[1]), "f"(v[2]), "f"(v[3]) : "memory");
}
__device__ __forceinline__ float relfeat(int k, float f1, float f2, float f3, float se) {
  if (k < 66) return f1 > (float)k ? 1.f : 0.f;
  if (k < 132) return f2 > (float)(k - 66) ? 1.f : 0.f;
  if (k == 132) return se;
  return f3 > (float)(k - 133) ? 1.f : 0.f;
}
__device__ __forceinline__ float clampbf(float x, float hi) {
  return fminf(fmaxf(rbf(x), 0.f), hi);
}
__device__ __forceinline__ void zero_acc(const Args &A, int p) {
  const int per = (NACC / 4 + NPH1 - 1) / NPH1;
  float4 *q = (float4 *)A.acc;
  int beg = p * per, end = min(beg + per, NACC / 4);
  for (int i = beg + (int)threadIdx.x; i < end; i += NTHR)
    q[i] = make_float4(0.f, 0.f, 0.f, 0.f);
}
// silu(a)*b with the reference's bf16 rounding points
__device__ __forceinline__ float swig(float a, float b) {
  float x = rbf(a);
  return rbf(rbf(x / (1.f + __expf(-x))) * rbf(b));
}

// ---- pair stage1 (runs on the 32 pair phase-3 blocks) --------------------
__device__ void stage1_pair(const Args &A, int blk, char *sm) {
  const int rg = blk >> 1, ns = blk & 1;
  const int row0 = rg * 16, n0 = ns * 64;
  const int SW = KZ1P + 8;
  bf16 *Wzs = (bf16 *)sm;                        // [64][280]
  bf16 *As = Wzs + 64 * SW;                      // [16][280]
  bf16 *zraw = As + NT * SW;                     // [16][128]
  float *lnzs = (float *)(zraw + NT * CZ);       // [272]
  float *idx = lnzs + KZ1P;                      // [5][16]
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;

  cpa_flat<NT * CZ * 2>((char *)zraw, (const char *)(A.zij + (size_t)row0 * CZ));
  cpa_commit();
  cpa2d<64, KZ1P / 8>((char *)Wzs, SW * 2, (const char *)(A.Wz + (size_t)n0 * KZ1P), KZ1P * 2);
  cpa_commit();
  cpa_flat<KZ1P * 4>((char *)lnzs, (const char *)A.lnz);
  if (tid < 16) {
    idx[tid] = __bfloat162float(A.res[tid]);
    idx[16 + tid] = __bfloat162float(A.tok[tid]);
    idx[32 + tid] = __bfloat162float(A.asym[tid]);
    idx[48 + tid] = __bfloat162float(A.ent[tid]);
    idx[64 + tid] = __bfloat162float(A.sym[tid]);
  }
  cpa_wait<0>();
  arrive(A.ctr + 11);

  if (warp < NT) {
    const int r = warp;
    int gi = (row0 + r) >> 4, gj = (row0 + r) & 15;
    float ri = idx[gi], rj = idx[gj];
    bool sc = idx[32 + gi] == idx[32 + gj];
    bool se = idx[48 + gi] == idx[48 + gj];
    float f1 = sc ? clampbf(rbf(ri - rj) + 32.f, 64.f) : 65.f;
    float f2 = (sc && ri == rj) ? clampbf(rbf(idx[16 + gi] - idx[16 + gj]) + 32.f, 64.f) : 65.f;
    float f3 = se ? clampbf(rbf(idx[64 + gi] - idx[64 + gj]) + 2.f, 4.f) : 5.f;
    float sef = se ? 1.f : 0.f;
    float zv[4], s = 0.f, ss = 0.f;
#pragma unroll
    for (int u = 0; u < 4; u++) {
      float v = __bfloat162float(zraw[r * CZ + lane + u * 32]);
      zv[u] = v; s += v; ss += v * v;
    }
    s = wsum(s); ss = wsum(ss);
    // relpos features are 0/1 prefix indicators: their sum (== sum of squares)
    // is just how many bins each threshold covers.
    float nb1 = fminf(ceilf(f1), 66.f), nb2 = fminf(ceilf(f2), 66.f);
    float nb3 = fminf(ceilf(f3), 6.f);
    s += nb1 + nb2 + nb3 + sef; ss += nb1 + nb2 + nb3 + sef;
    float mean = s * (1.f / (float)KZ1);
    float rstd = rsqrtf(ss * (1.f / (float)KZ1) - mean * mean + 1e-5f);
#pragma unroll
    for (int u = 0; u < 4; u++) {
      int k = lane + u * 32;
      As[r * SW + k] = __float2bfloat16((zv[u] - mean) * rstd * lnzs[k]);
    }
#pragma unroll
    for (int u = 0; u < 5; u++) {
      int kk = lane + u * 32;
      if (kk < NRP)
        As[r * SW + CZ + kk] =
            __float2bfloat16((relfeat(kk, f1, f2, f3, sef) - mean) * rstd * lnzs[CZ + kk]);
    }
    if (lane < KZ1P - KZ1) As[r * SW + KZ1 + lane] = __float2bfloat16(0.f);
  }
  __syncthreads();

  if (warp < 8) {                                // 8 n-tiles, one per warp
    const int nt = warp;
    float acc[4] = {};
    unsigned a[4], b[2];
#pragma unroll
    for (int kt = 0; kt < KZ1P / 16; kt++) {
      ldA(a, As + kt * 16, SW, lane);
      ldB(b, Wzs + nt * 8 * SW + kt * 16, SW, lane);
      mma16816(acc, a, b);
    }
    int gid = lane >> 2, tig = lane & 3, c = n0 + nt * 8 + 2 * tig, r = row0 + gid;
    st2bf(A.z1buf + r * CZ + c, acc[0], acc[1]);
    st2bf(A.z1buf + (r + 8) * CZ + c, acc[2], acc[3]);
  }
  arrive(A.ctr);
}

// ---- role: single stage1 -------------------------------------------------
__device__ void stage1_single(const Args &A, int blk, char *sm) {
  const int n0 = blk * 16;
  const int SW = KS1P + 8;
  bf16 *Wss = (bf16 *)sm;                         // [16][856]
  bf16 *Wns = Wss + NT * SW;                      // [16][256]
  bf16 *As = Wns + NT * CF;                       // [16][856]
  bf16 *straw = As + NT * SW;                     // [16][384]
  bf16 *sinraw = straw + NT * CS;                 // [16][449]
  float *lnss = (float *)(sinraw + NT * CIN);     // [848]
  float *lnns = lnss + KS1P;                      // [256]
  float *nemb = lnns + CF;                        // [256]
  float *rd = nemb + CF;                          // [4][4][32][4]
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;

  cpa_flat<NT * CS * 2>((char *)straw, (const char *)A.si_trunk);
  cpa_flat<NT * CIN * 2>((char *)sinraw, (const char *)A.si_input);
  cpa_commit();
  cpa2d<NT, KS1P / 8>((char *)Wss, SW * 2, (const char *)(A.Ws + (size_t)n0 * KS1P), KS1P * 2);
  cpa2d<NT, CF / 8>((char *)Wns, CF * 2, (const char *)(A.Wn + (size_t)n0 * CF), CF * 2);
  cpa_commit();
  cpa_flat<KS1P * 4>((char *)lnss, (const char *)A.lns);
  cpa_flat<CF * 4>((char *)lnns, (const char *)A.lnn);
  if (tid < CF) {                                  // fourier time embedding
    float tv = __bfloat162float(A.tsc[0]);
    float n = rbf(0.25f * rbf(logf(rbf(tv * (1.f / 16.f)))));
    float xf = rbf(rbf(n * __bfloat162float(A.fw[tid])) + __bfloat162float(A.fb[tid]));
    nemb[tid] = rbf(cosf(rbf(6.2831855f * xf)));
  }
  cpa_wait<0>();
  arrive(A.ctr + 11);
  if (warp == 0) {                                 // LN over the 256 fourier feats
    float v[8], s = 0.f, ss = 0.f;
#pragma unroll
    for (int u = 0; u < 8; u++) { float e = nemb[lane + u * 32]; v[u] = e; s += e; ss += e * e; }
    s = wsum(s); ss = wsum(ss);
    float mean = s * (1.f / (float)CF);
    float rstd = rsqrtf(ss * (1.f / (float)CF) - mean * mean + 1e-5f);
#pragma unroll
    for (int u = 0; u < 8; u++) {
      int k = lane + u * 32;
      nemb[k] = rbf((v[u] - mean) * rstd * lnns[k]);
    }
  }
  if (warp < NT) {                                 // one warp per token row
    const int r = warp;
    float xv[27], s = 0.f, ss = 0.f;
#pragma unroll
    for (int u = 0; u < 27; u++) {
      int k = lane + u * 32;
      float v = 0.f;
      if (k < KS1) v = __bfloat162float(k < CS ? straw[r * CS + k] : sinraw[r * CIN + k - CS]);
      xv[u] = v; s += v; ss += v * v;
    }
    s = wsum(s); ss = wsum(ss);
    float mean = s * (1.f / (float)KS1);
    float rstd = rsqrtf(ss * (1.f / (float)KS1) - mean * mean + 1e-5f);
#pragma unroll
    for (int u = 0; u < 27; u++) {
      int k = lane + u * 32;
      if (k < KS1P)
        As[r * SW + k] = __float2bfloat16(k < KS1 ? (xv[u] - mean) * rstd * lnss[k] : 0.f);
    }
  }
  __syncthreads();

  // 2 n-tiles x 8 k-parts over 16 warps, then reduce the k-parts.
  const int nt = warp & 1, kp = warp >> 1;
  float acc[4] = {};
  unsigned a[4], b[2];
#pragma unroll 4
  for (int kt = kp; kt < KS1P / 16; kt += 8) {
    ldA(a, As + kt * 16, SW, lane);
    ldB(b, Wss + nt * 8 * SW + kt * 16, SW, lane);
    mma16816(acc, a, b);
  }
#pragma unroll
  for (int q = 0; q < 4; q++) rd[((kp * 2 + nt) * 32 + lane) * 4 + q] = acc[q];
  __syncthreads();
  if (warp < 2) {
    const int nn = warp;
    float g[4] = {};
#pragma unroll
    for (int p = 0; p < 8; p++)
#pragma unroll
      for (int q = 0; q < 4; q++) g[q] += rd[((p * 2 + nn) * 32 + lane) * 4 + q];
    float fv[8];
#pragma unroll
    for (int cc = 0; cc < 8; cc++) {
      float p = 0.f;
#pragma unroll
      for (int u = 0; u < 8; u++) {
        int k = lane + u * 32;
        p += nemb[k] * __bfloat162float(Wns[(nn * 8 + cc) * CF + k]);
      }
      fv[cc] = rbf(wsum(p));
    }
    int gid = lane >> 2, tig = lane & 3, c = n0 + nn * 8 + 2 * tig;
    float g0 = fv[2 * tig], g1 = fv[2 * tig + 1];
    st2bf(A.sibuf + gid * CS + c, rbf(rbf(g[0]) + g0), rbf(rbf(g[1]) + g1));
    st2bf(A.sibuf + (gid + 8) * CS + c, rbf(rbf(g[2]) + g0), rbf(rbf(g[3]) + g1));
  }
  arrive(A.ctr);
}

// ---- role: pair transition (TI = 0 | 1) ----------------------------------
template <int TI>
__device__ void role_pt(const Args &A, int idx, char *sm) {
  if (TI == 1) {                 // phase 1: this block's slice of pair stage-1
    zero_acc(A, idx);
    stage1_pair(A, idx, sm);
    __syncthreads();             // stage-1 shared state is dead from here
  }
  const int rg = idx >> 2, hs = idx & 3;
  const int row0 = rg * 32, h0 = hs * 64, HB = 64;
  const int SA = CZ + 8, SO = HB + 8;
  bf16 *Was = (bf16 *)sm;                         // [64][136]
  bf16 *Wbs = Was + HB * SA;
  bf16 *Wos = Wbs + HB * SA;                      // [128][72]
  bf16 *A0 = Wos + CZ * SO;                       // [32][136]
  bf16 *A1 = A0 + 32 * SA;                        // [32][72]
  bf16 *zraw = A1 + 32 * SO;                      // [32][128]
  float *tmp = (float *)(zraw + 32 * CZ);         // [32][128]
  float *lnw = tmp + 32 * CZ;
  float *lnb = lnw + CZ;
  float *msk = lnb + CZ;                          // [16]
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  float *accg = A.acc + TI * NPAIR * CZ;

  wait_for(A.ctr + 11, (A.gen + 1) * NPH1);
  cpa2d<64, CZ / 8>((char *)Was, SA * 2, (const char *)(A.tWa[TI] + (size_t)h0 * CZ), CZ * 2);
  cpa2d<64, CZ / 8>((char *)Wbs, SA * 2, (const char *)(A.tWb[TI] + (size_t)h0 * CZ), CZ * 2);
  cpa2d<CZ, 64 / 8>((char *)Wos, SO * 2, (const char *)(A.tWo[TI] + h0), HZ * 2);
  cpa_commit();
  cpa_flat<CZ * 4>((char *)lnw, (const char *)A.tlnw[TI]);
  cpa_flat<CZ * 4>((char *)lnb, (const char *)A.tlnb[TI]);
  if (tid < NT) msk[tid] = __bfloat162float(A.tmask[tid]);
  wait_for(A.ctr + TI, (A.gen + 1) * (TI == 0 ? NPH1 : NPH2));
  cpa_flat<32 * CZ * 2>((char *)zraw, (const char *)(A.z1buf + (size_t)row0 * CZ));
  cpa_commit();
  if (TI == 1) {
    const float4 *src = (const float4 *)(A.acc + (size_t)row0 * CZ);
    float4 *dst = (float4 *)tmp;
    for (int i = tid; i < 32 * CZ / 4; i += NTHR) dst[i] = src[i];
  }
  cpa_wait<0>();
  __syncthreads();

  if (TI == 1) {          // z2 = rb(z1 + rb(rb(acc1) * mask))
    for (int i = tid; i < 32 * CZ; i += NTHR) {
      int r = row0 + (i >> 7);
      float m = rbf(msk[r >> 4] * msk[r & 15]);
      zraw[i] = __float2bfloat16(rbf(__bfloat162float(zraw[i]) + rbf(rbf(tmp[i]) * m)));
    }
    __syncthreads();
  }
#pragma unroll 2
  for (int r = warp; r < 32; r += NW) {
    float xv[4], s = 0.f, ss = 0.f;
#pragma unroll
    for (int u = 0; u < 4; u++) {
      float v = __bfloat162float(zraw[r * CZ + lane + u * 32]);
      xv[u] = v; s += v; ss += v * v;
    }
    s = wsum(s); ss = wsum(ss);
    float mean = s * (1.f / (float)CZ);
    float rstd = rsqrtf(ss * (1.f / (float)CZ) - mean * mean + 1e-5f);
#pragma unroll
    for (int u = 0; u < 4; u++) {
      int k = lane + u * 32;
      A0[r * SA + k] = __float2bfloat16((xv[u] - mean) * rstd * lnw[k] + lnb[k]);
    }
  }
  __syncthreads();

  {   // 8 n-tiles x 2 m-tiles over 16 warps
    const int nt = warp & 7, mt = warp >> 3;
    float aca[4] = {}, acb[4] = {};
    unsigned a[4], ba[2], bb[2];
#pragma unroll
    for (int kt = 0; kt < CZ / 16; kt++) {
      ldA(a, A0 + mt * 16 * SA + kt * 16, SA, lane);
      ldB(ba, Was + nt * 8 * SA + kt * 16, SA, lane);
      mma16816(aca, a, ba);
      ldB(bb, Wbs + nt * 8 * SA + kt * 16, SA, lane);
      mma16816(acb, a, bb);
    }
    int gid = lane >> 2, tig = lane & 3, hc = nt * 8 + 2 * tig;
    st2bf(A1 + (mt * 16 + gid) * SO + hc, swig(aca[0], acb[0]), swig(aca[1], acb[1]));
    st2bf(A1 + (mt * 16 + gid + 8) * SO + hc, swig(aca[2], acb[2]), swig(aca[3], acb[3]));
  }
  __syncthreads();

  {   // 16 n-tiles x 2 m-tiles over 16 warps -> 2 n-tiles each
    const int mt = warp >> 3, nt0 = (warp & 7) * 2;
    float aco[2][4] = {};
    unsigned a[4], b[2];
#pragma unroll
    for (int kt = 0; kt < HB / 16; kt++) {
      ldA(a, A1 + mt * 16 * SO + kt * 16, SO, lane);
#pragma unroll
      for (int j = 0; j < 2; j++) {
        ldB(b, Wos + (nt0 + j) * 8 * SO + kt * 16, SO, lane);
        mma16816(aco[j], a, b);
      }
    }
    __syncthreads();
    int gid = lane >> 2, tig = lane & 3;
#pragma unroll
    for (int j = 0; j < 2; j++) {
      int c = (nt0 + j) * 8 + 2 * tig;
      tmp[(mt * 16 + gid) * CZ + c] = aco[j][0];
      tmp[(mt * 16 + gid) * CZ + c + 1] = aco[j][1];
      tmp[(mt * 16 + gid + 8) * CZ + c] = aco[j][2];
      tmp[(mt * 16 + gid + 8) * CZ + c + 1] = aco[j][3];
    }
  }
  __syncthreads();
  for (int i = tid; i < 32 * CZ / 4; i += NTHR)
    red4(accg + (size_t)row0 * CZ + i * 4, tmp + i * 4);

  if (TI == 0) { arrive(A.ctr + 1); return; }
  __syncthreads();
  unsigned long long old = 0;
  if (tid == 0) old = redrel(A.ctr + 2 + rg);
  if (__syncthreads_or(tid == 0 && old + 1 == (A.gen + 1) * 4)) {
    const float *ac2 = A.acc + NPAIR * CZ + (size_t)row0 * CZ;
    for (int i = tid; i < 32 * CZ; i += NTHR) {
      int r = row0 + (i >> 7);
      float m = rbf(msk[r >> 4] * msk[r & 15]);
      A.zout[(size_t)row0 * CZ + i] = __float2bfloat16(
          rbf(__bfloat162float(zraw[i]) + rbf(rbf(ac2[i]) * m)));
    }
  }
}

// ---- role: single transition (TI = 2 | 3) --------------------------------
template <int TI>
__device__ void role_st(const Args &A, int idx, char *sm) {
  if (TI == 3) {                 // phase 1: this block's slice of single stage-1
    zero_acc(A, NB_PT + idx);
    stage1_single(A, idx, sm);
    __syncthreads();
  }
  const int h0 = idx * 32, HB = 32;
  const int SA = CS + 8, SO = HB + 8;
  bf16 *Was = (bf16 *)sm;                         // [32][392]
  bf16 *Wbs = Was + HB * SA;
  bf16 *Wos = Wbs + HB * SA;                      // [384][40]
  bf16 *A0 = Wos + CS * SO;                       // [16][392]
  bf16 *A1 = A0 + NT * SA;                        // [16][40]
  bf16 *sraw = A1 + NT * SO;                      // [16][384]
  float *tmp = (float *)(sraw + NT * CS);         // [16][384], also k-part reduce
  float *lnw = tmp + NT * CS;
  float *lnb = lnw + CS;
  float *msk = lnb + CS;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  float *accg = A.acc + 2 * NPAIR * CZ + (TI - 2) * NT * CS;

  wait_for(A.ctr + 11, (A.gen + 1) * NPH1);
  cpa2d<32, CS / 8>((char *)Was, SA * 2, (const char *)(A.tWa[TI] + (size_t)h0 * CS), CS * 2);
  cpa2d<32, CS / 8>((char *)Wbs, SA * 2, (const char *)(A.tWb[TI] + (size_t)h0 * CS), CS * 2);
  cpa2d<CS, 32 / 8>((char *)Wos, SO * 2, (const char *)(A.tWo[TI] + h0), HS * 2);
  cpa_commit();
  cpa_flat<CS * 4>((char *)lnw, (const char *)A.tlnw[TI]);
  cpa_flat<CS * 4>((char *)lnb, (const char *)A.tlnb[TI]);
  if (tid < NT) msk[tid] = __bfloat162float(A.tmask[tid]);
  wait_for(A.ctr + (TI - 2), (A.gen + 1) * (TI == 2 ? NPH1 : NPH2));
  cpa_flat<NT * CS * 2>((char *)sraw, (const char *)A.sibuf);
  cpa_commit();
  if (TI == 3) {
    const float4 *src = (const float4 *)(A.acc + 2 * NPAIR * CZ);
    float4 *dst = (float4 *)tmp;
    for (int i = tid; i < NT * CS / 4; i += NTHR) dst[i] = src[i];
  }
  cpa_wait<0>();
  __syncthreads();
  if (TI == 3) {
    for (int i = tid; i < NT * CS; i += NTHR)
      sraw[i] = __float2bfloat16(
          rbf(__bfloat162float(sraw[i]) + rbf(rbf(tmp[i]) * msk[i / CS])));
    __syncthreads();
  }
  if (warp < NT) {
    const int r = warp;
    float xv[12], s = 0.f, ss = 0.f;
#pragma unroll
    for (int u = 0; u < 12; u++) {
      float v = __bfloat162float(sraw[r * CS + lane + u * 32]);
      xv[u] = v; s += v; ss += v * v;
    }
    s = wsum(s); ss = wsum(ss);
    float mean = s * (1.f / (float)CS);
    float rstd = rsqrtf(ss * (1.f / (float)CS) - mean * mean + 1e-5f);
#pragma unroll
    for (int u = 0; u < 12; u++) {
      int k = lane + u * 32;
      A0[r * SA + k] = __float2bfloat16((xv[u] - mean) * rstd * lnw[k] + lnb[k]);
    }
  }
  __syncthreads();

  {   // 4 n-tiles x 4 k-parts over 16 warps
    const int nt = warp & 3, kp = warp >> 2;
    float aca[4] = {}, acb[4] = {};
    unsigned a[4], ba[2], bb[2];
#pragma unroll 6
    for (int kt = kp; kt < CS / 16; kt += 4) {
      ldA(a, A0 + kt * 16, SA, lane);
      ldB(ba, Was + nt * 8 * SA + kt * 16, SA, lane);
      mma16816(aca, a, ba);
      ldB(bb, Wbs + nt * 8 * SA + kt * 16, SA, lane);
      mma16816(acb, a, bb);
    }
#pragma unroll
    for (int q = 0; q < 4; q++) {
      tmp[((kp * 4 + nt) * 32 + lane) * 8 + q] = aca[q];
      tmp[((kp * 4 + nt) * 32 + lane) * 8 + 4 + q] = acb[q];
    }
  }
  __syncthreads();
  if (warp < 4) {
    const int nt = warp;
    float ga[4] = {}, gb[4] = {};
#pragma unroll
    for (int p = 0; p < 4; p++)
#pragma unroll
      for (int q = 0; q < 4; q++) {
        ga[q] += tmp[((p * 4 + nt) * 32 + lane) * 8 + q];
        gb[q] += tmp[((p * 4 + nt) * 32 + lane) * 8 + 4 + q];
      }
    int gid = lane >> 2, tig = lane & 3, hc = nt * 8 + 2 * tig;
    st2bf(A1 + gid * SO + hc, swig(ga[0], gb[0]), swig(ga[1], gb[1]));
    st2bf(A1 + (gid + 8) * SO + hc, swig(ga[2], gb[2]), swig(ga[3], gb[3]));
  }
  __syncthreads();

  {   // 48 n-tiles over 16 warps -> 3 each
    const int nt0 = warp * 3;
    float aco[3][4] = {};
    unsigned a[4], b[2];
#pragma unroll
    for (int kt = 0; kt < HB / 16; kt++) {
      ldA(a, A1 + kt * 16, SO, lane);
#pragma unroll
      for (int j = 0; j < 3; j++) {
        ldB(b, Wos + (nt0 + j) * 8 * SO + kt * 16, SO, lane);
        mma16816(aco[j], a, b);
      }
    }
    __syncthreads();
    int gid = lane >> 2, tig = lane & 3;
#pragma unroll
    for (int j = 0; j < 3; j++) {
      int c = (nt0 + j) * 8 + 2 * tig;
      tmp[gid * CS + c] = aco[j][0];
      tmp[gid * CS + c + 1] = aco[j][1];
      tmp[(gid + 8) * CS + c] = aco[j][2];
      tmp[(gid + 8) * CS + c + 1] = aco[j][3];
    }
  }
  __syncthreads();
  for (int i = tid; i < NT * CS / 4; i += NTHR) red4(accg + i * 4, tmp + i * 4);

  if (TI == 2) { arrive(A.ctr + 1); return; }
  __syncthreads();
  unsigned long long old = 0;
  if (tid == 0) old = redrel(A.ctr + 10);
  if (__syncthreads_or(tid == 0 && old + 1 == (A.gen + 1) * NB_ST)) {
    const float *ac2 = A.acc + 2 * NPAIR * CZ + NT * CS;
    for (int i = tid; i < NT * CS; i += NTHR)
      A.siout[i] = __float2bfloat16(
          rbf(__bfloat162float(sraw[i]) + rbf(rbf(ac2[i]) * msk[i / CS])));
  }
}

__global__ __launch_bounds__(NTHR, 1) void k_main(const Args A) {
  extern __shared__ char sm[];
  const int blk = blockIdx.x;
  if (blk < B_ST1) role_pt<0>(A, blk - B_PT1, sm);
  else if (blk < B_PT2) role_st<2>(A, blk - B_ST1, sm);
  else if (blk < B_ST2) role_pt<1>(A, blk - B_PT2, sm);
  else role_st<3>(A, blk - B_ST2, sm);
}

// ---- host ---------------------------------------------------------------
struct Reg { Args a; int shmem; unsigned long long gen; };
static std::vector<Reg> g_regs;

int64_t dc_init(std::vector<torch::Tensor> w, std::vector<torch::Tensor> scratch) {
  Reg R;
  memset(&R.a, 0, sizeof(Args));
  int i = 0;
  auto B = [&](void) { return (const bf16 *)w[i++].data_ptr(); };
  auto F = [&](void) { return (const float *)w[i++].data_ptr(); };
  R.a.Wz = B(); R.a.lnz = F();
  R.a.Ws = B(); R.a.lns = F();
  R.a.Wn = B(); R.a.lnn = F();
  R.a.fw = B(); R.a.fb = B();
  for (int t = 0; t < 4; t++) {
    R.a.tWa[t] = B(); R.a.tWb[t] = B(); R.a.tWo[t] = B();
    R.a.tlnw[t] = F(); R.a.tlnb[t] = F();
  }
  TORCH_CHECK(i == (int)w.size(), "weight list size mismatch");
  R.a.z1buf = (bf16 *)scratch[0].data_ptr();
  R.a.sibuf = (bf16 *)scratch[1].data_ptr();
  R.a.acc = (float *)scratch[2].data_ptr();
  R.a.ctr = (unsigned long long *)scratch[3].data_ptr();
  int sh[4];
  sh[0] = (64 * 280 + NT * 280 + NT * CZ) * 2 + (KZ1P + 80) * 4;
  sh[1] = (NT * 856 + NT * CF + NT * 856 + NT * CS + NT * CIN) * 2
          + (KS1P + 2 * CF + 2048) * 4;
  sh[2] = (64 * 136 * 2 + CZ * 72 + 32 * 136 + 32 * 72 + 32 * CZ) * 2
          + (32 * CZ + 2 * CZ + NT) * 4;
  sh[3] = (32 * 392 * 2 + CS * 40 + NT * 392 + NT * 40 + NT * CS) * 2
          + (NT * CS + 2 * CS + NT) * 4;
  R.shmem = 0;
  for (int k = 0; k < 4; k++) R.shmem = std::max(R.shmem, sh[k] + 256);
  int optin = 0;
  cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
  TORCH_CHECK(R.shmem <= optin, "need ", R.shmem, " smem, device allows ", optin);
  TORCH_CHECK(cudaFuncSetAttribute(k_main, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   R.shmem) == cudaSuccess, "smem opt-in failed");
  int nsm = 0;
  cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0);
  TORCH_CHECK(nsm >= NBLK, "need ", NBLK, " resident blocks, device has ", nsm, " SMs");
  int maxblk = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxblk, k_main, NTHR, R.shmem);
  TORCH_CHECK(maxblk >= 1, "kernel not resident-schedulable");
  R.gen = 0;
  g_regs.push_back(R);
  return (int64_t)g_regs.size() - 1;
}

std::vector<torch::Tensor> dc_run(int64_t h, torch::Tensor zij, torch::Tensor si_trunk,
                                  torch::Tensor si_input, torch::Tensor t,
                                  torch::Tensor res, torch::Tensor tok, torch::Tensor asym,
                                  torch::Tensor ent, torch::Tensor sym, torch::Tensor tmask) {
  Reg &R = g_regs[h];
  auto opt = zij.options();
  auto siout = torch::empty({1, NT, CS}, opt);
  auto zout = torch::empty({1, NT, NT, CZ}, opt);
  Args a = R.a;
  a.zij = (const bf16 *)zij.data_ptr();
  a.si_trunk = (const bf16 *)si_trunk.data_ptr();
  a.si_input = (const bf16 *)si_input.data_ptr();
  a.tsc = (const bf16 *)t.data_ptr();
  a.res = (const bf16 *)res.data_ptr();
  a.tok = (const bf16 *)tok.data_ptr();
  a.asym = (const bf16 *)asym.data_ptr();
  a.ent = (const bf16 *)ent.data_ptr();
  a.sym = (const bf16 *)sym.data_ptr();
  a.tmask = (const bf16 *)tmask.data_ptr();
  a.zout = (bf16 *)zout.data_ptr();
  a.siout = (bf16 *)siout.data_ptr();
  a.gen = R.gen++;
  k_main<<<NBLK, NTHR, R.shmem, c10::cuda::getCurrentCUDAStream().stream()>>>(a);
  return {siout, zout};
}
"""

_CPP_SOURCE = r"""
#include <torch/extension.h>
#include <vector>
int64_t dc_init(std::vector<torch::Tensor> w, std::vector<torch::Tensor> scratch);
std::vector<torch::Tensor> dc_run(int64_t h, torch::Tensor zij, torch::Tensor si_trunk,
                                  torch::Tensor si_input, torch::Tensor t,
                                  torch::Tensor res, torch::Tensor tok, torch::Tensor asym,
                                  torch::Tensor ent, torch::Tensor sym, torch::Tensor tmask);
"""


def _ext():
    """JIT-build the fused kernel once; None if unavailable."""
    global _EXT, _EXT_TRIED
    if _EXT_TRIED:
        return _EXT
    _EXT_TRIED = True
    try:
        from torch.utils.cpp_extension import load_inline
        major, minor = torch.cuda.get_device_capability()
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{major}.{minor}a")
        _EXT = load_inline(
            name="af3_diffcond_fused",
            cpp_sources=[_CPP_SOURCE],
            cuda_sources=[_CU_SOURCE],
            functions=["dc_init", "dc_run"],
            extra_cuda_cflags=["-O3", f"-arch=sm_{major}{minor}a", "--use_fast_math"],
            verbose=False,
        )
    except Exception:
        _EXT = None
    return _EXT


class FourierEmbedding(nn.Module):
    """Fourier time embedding for diffusion conditioning."""

    def __init__(self, c: int = 256, seed: int = 42):
        super().__init__()
        self.c = c
        generator = torch.Generator()
        generator.manual_seed(seed)
        self.register_buffer("w", torch.randn(c, generator=generator))
        self.register_buffer("b", torch.randn(c, generator=generator))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        x = t * self.w + self.b
        return torch.cos(2 * math.pi * x)


class DiffusionConditioning(nn.Module):
    """Conditioning for diffusion module (fused)."""

    def __init__(
        self,
        c_s: int = 384,
        c_z: int = 128,
        c_s_input: int = 449,
        sigma_data: float = 16.0,
        relpos_k: int = 32,
        max_relative_chain: int = 2,
        c_fourier_emb: int = 256,
        seed_fourier_emb: int = 42,
    ):
        super().__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.c_s_input = c_s_input
        self.c_fourier_emb = c_fourier_emb
        self.sigma_data = sigma_data
        self.relpos_k = relpos_k
        self.max_relative_chain = max_relative_chain

        num_rel_pos_bins = 2 * relpos_k + 2
        num_rel_token_bins = 2 * relpos_k + 2
        num_rel_chain_bins = 2 * max_relative_chain + 2
        num_same_entity_features = 1
        num_relpos_dims = (
            num_rel_pos_bins + num_rel_token_bins
            + num_rel_chain_bins + num_same_entity_features
        )
        self.num_relpos_dims = num_relpos_dims

        self.layer_norm_z = LayerNorm(num_relpos_dims + c_z, create_offset=False)
        self.linear_z = Linear(num_relpos_dims + c_z, c_z, bias=False)

        self.transition_z = nn.ModuleList([
            SwiGLUTransition(c_in=c_z, n=2)
            for _ in range(2)
        ])

        self.layer_norm_s = LayerNorm(c_s + c_s_input, create_offset=False)
        self.linear_s = Linear(c_s + c_s_input, c_s, bias=False)

        self.fourier_emb = FourierEmbedding(c=c_fourier_emb, seed=seed_fourier_emb)
        self.layer_norm_n = LayerNorm(c_fourier_emb, create_offset=False)
        self.linear_n = Linear(c_fourier_emb, c_s, bias=False)

        self.transition_s = nn.ModuleList([
            SwiGLUTransition(c_in=c_s, n=2)
            for _ in range(2)
        ])

        self._handle = None
        self._ready = False
        self._tried = False
        self._sig = None
        self._keep = None

    # ---- fused path setup -------------------------------------------------
    # The kernel is specialized for the captured OpenFold3 geometry (16 tokens,
    # c_s=384, c_z=128, c_s_input=449, bf16).  Weights are laid out once, after
    # the bench has loaded them, and the layout plus the scratch buffers are
    # registered with the extension so the per-call Python work is one call.
    _DIMS = dict(c_s=384, c_z=128, c_s_input=449, c_fourier_emb=256,
                 relpos_k=32, max_relative_chain=2, n_token=16)

    def _supported(self, batch, t, si_input, si_trunk, zij_trunk) -> bool:
        d = self._DIMS
        if (self.c_s, self.c_z, self.c_s_input, self.c_fourier_emb) != (
                d["c_s"], d["c_z"], d["c_s_input"], d["c_fourier_emb"]):
            return False
        if (self.relpos_k, self.max_relative_chain) != (
                d["relpos_k"], d["max_relative_chain"]):
            return False
        if self.sigma_data != 16.0:
            return False
        n = d["n_token"]
        want = {
            "t": (), "si_input": (1, n, d["c_s_input"]), "si_trunk": (1, n, d["c_s"]),
            "zij_trunk": (1, n, n, d["c_z"]),
        }
        for name, tensor in (("t", t), ("si_input", si_input),
                             ("si_trunk", si_trunk), ("zij_trunk", zij_trunk)):
            if not isinstance(tensor, torch.Tensor):
                return False
            if tuple(tensor.shape) != want[name] or tensor.dtype != torch.bfloat16:
                return False
            if not (tensor.is_cuda and tensor.is_contiguous()):
                return False
        if not isinstance(batch, dict):
            return False
        for key in ("residue_index", "token_index", "asym_id", "entity_id",
                    "sym_id", "token_mask"):
            v = batch.get(key)
            if not isinstance(v, torch.Tensor) or tuple(v.shape) != (1, n):
                return False
            if v.dtype != torch.bfloat16 or not (v.is_cuda and v.is_contiguous()):
                return False
        params = list(self.parameters())
        if any(p.dtype != torch.bfloat16 for p in params):
            return False
        if any(p.device != zij_trunk.device for p in params):
            return False
        if self.fourier_emb.w.dtype != torch.bfloat16:
            return False
        return True

    def _build(self, device):
        ext = _ext()
        if ext is None:
            return False
        kz, kzp = self.num_relpos_dims + self.c_z, 272
        ks, ksp = self.c_s + self.c_s_input, 848

        def pad(w, k):
            out = torch.zeros(w.shape[0], k, device=device, dtype=torch.bfloat16)
            out[:, :w.shape[1]] = w
            return out.contiguous()

        wz = pad(self.linear_z.weight.detach(), kzp)
        lnz = torch.zeros(kzp, device=device, dtype=torch.float32)
        lnz[:kz] = self.layer_norm_z.weight.detach().float()
        ws = pad(self.linear_s.weight.detach(), ksp)
        lns = torch.zeros(ksp, device=device, dtype=torch.float32)
        lns[:ks] = self.layer_norm_s.weight.detach().float()
        weights = [
            wz, lnz,
            ws, lns,
            self.linear_n.weight.detach().contiguous(),
            self.layer_norm_n.weight.detach().float().contiguous(),
            self.fourier_emb.w.detach().contiguous(),
            self.fourier_emb.b.detach().contiguous(),
        ]
        for layer in list(self.transition_z) + list(self.transition_s):
            weights += [
                layer.swiglu.linear_a.weight.detach().contiguous(),
                layer.swiglu.linear_b.weight.detach().contiguous(),
                layer.linear_out.weight.detach().contiguous(),
                layer.layer_norm.weight.detach().float().contiguous(),
                layer.layer_norm.bias.detach().float().contiguous(),
            ]
        n = self._DIMS["n_token"]
        scratch = [
            torch.zeros(n * n, self.c_z, device=device, dtype=torch.bfloat16),
            torch.zeros(n, self.c_s, device=device, dtype=torch.bfloat16),
            torch.zeros(2 * n * n * self.c_z + 2 * n * self.c_s,
                        device=device, dtype=torch.float32),
            torch.zeros(32, device=device, dtype=torch.int64),
        ]
        try:
            self._handle = ext.dc_init(weights, scratch)
        except Exception:
            return False
        self._keep = (weights, scratch, ext)
        return True

    def forward(
        self,
        batch: dict,
        t: torch.Tensor,
        si_input: torch.Tensor,
        si_trunk: torch.Tensor,
        zij_trunk: torch.Tensor,
        use_conditioning: bool,
        chunk_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._ready and use_conditioning and zij_trunk.shape == self._sig:
            ext, get = self._keep[2], batch.get
            return ext.dc_run(
                self._handle, zij_trunk, si_trunk, si_input, t,
                get("residue_index"), get("token_index"), get("asym_id"),
                get("entity_id"), get("sym_id"), get("token_mask"),
            )
        if not self._tried and use_conditioning and chunk_size is None:
            # One attempt only: a failed build (no nvcc, unexpected geometry)
            # must not be retried on every call.
            self._tried = True
            if (self._supported(batch, t, si_input, si_trunk, zij_trunk)
                    and self._build(zij_trunk.device)):
                self._ready = True
                self._sig = zij_trunk.shape
                return self.forward(batch, t, si_input, si_trunk, zij_trunk,
                                    use_conditioning, chunk_size)
        return self._reference(batch, t, si_input, si_trunk, zij_trunk,
                               use_conditioning, chunk_size)

    def _reference(self, batch, t, si_input, si_trunk, zij_trunk,
                   use_conditioning, chunk_size=None):
        if use_conditioning:
            if "asym_id" in batch:
                relpos_zij = relpos_complex(
                    batch=batch,
                    max_relative_idx=self.relpos_k,
                    max_relative_chain=self.max_relative_chain,
                ).to(dtype=zij_trunk.dtype)
            else:
                relpos_dim = self.linear_z.weight.shape[-1] - self.c_z
                relpos_zij = zij_trunk.new_zeros(
                    zij_trunk.shape[:-1] + (relpos_dim,),
                )

            zij = torch.cat([zij_trunk, relpos_zij], dim=-1)
            zij = self.linear_z(self.layer_norm_z(zij))

            si = torch.cat([si_trunk, si_input], dim=-1)
            si = self.linear_s(self.layer_norm_s(si))
        else:
            zij = zij_trunk.new_zeros(zij_trunk.shape)
            si = si_trunk.new_zeros(si_trunk.shape[:-1] + (self.c_s,))

        n = 0.25 * torch.log(t / self.sigma_data)
        n_emb = self.fourier_emb(n.unsqueeze(-1) if n.dim() == 0 else n)
        si = si + self.linear_n(self.layer_norm_n(n_emb)).unsqueeze(-2)

        token_mask = batch.get("token_mask")
        if token_mask is not None:
            pair_mask = token_mask[..., :, None] * token_mask[..., None, :]
        else:
            pair_mask = None

        for layer in self.transition_z:
            zij = zij + layer(zij, mask=pair_mask)

        for layer in self.transition_s:
            si = si + layer(si, mask=token_mask)

        return si, zij
