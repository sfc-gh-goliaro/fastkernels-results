"""SwiGLU transition composites for AlphaFold3 (L2), each fused into one kernel.

At the captured shapes both operators are launch-bound, not compute-bound.  The
baseline issues ~16 (ConditionedTransitionBlock) / ~8 (SwiGLUTransition) kernels
to move at most a few hundred rows through three small matmuls, and on this
machine a tiny back-to-back kernel costs ~4 us of wall clock inside the bench's
timing loop -- measured with an empty kernel: 1 launch 7.2 us, 26 launches 108
us.  The baseline's 120-220 us is almost entirely launch overhead; its arithmetic
(<= 141 MFLOP) is noise.

So each class here runs as a *single* kernel launch.  The dependency chain is cut
into stages separated by a device-wide barrier (~1.5 us, against ~4 us for
another launch):

  ConditionedTransitionBlock -- 3 stages, 2 barriers
    1. AdaLN (row statistics of ``a``/``s``, gate)             -> a1
    2. SwiGLU hidden                                           -> bh
    3. ``linear_out`` x sigmoid(``linear_g(s)``) x mask         -> out
  SwiGLUTransition -- 2 stages, 1 barrier
    1. LayerNorm + SwiGLU hidden                               -> bh
    2. ``linear_out`` x mask                                    -> out

The matmuls use ``mma.m16n8k16`` (bf16 in, fp32 accumulate), which fits these
shapes exactly: every captured row count is a multiple of 16 and every reduction
length a multiple of 64.  One warp owns one (8 output columns x 64 reduction)
block and the reduction is split across the warps of a block (summed through
shared memory at the end) -- that split is what keeps enough loads in flight,
which is the binding constraint here: the whole 8.85 MB weight set of the largest
shape is requested within ~2 rounds of memory latency.  The weight operand goes
straight from global memory into registers, a whole 8x64 block at a time; for an
operand read once by 8 rows at a time that is eight fully-used 32 B sectors per
instruction, so staging it through shared memory would only add a barrier.

The LayerNorm needs more care.  Its output *is* the matmul's A operand, so the
prologue of stage 1 loads each row once, takes the mean/rstd in fp32, and writes
the normalised row into shared memory out of the same registers -- one memory
latency for the whole tile, and it reuses the barrier that publishes the
statistics.  Folding the affine into the weights instead (``W' = w*W`` plus a
``mu * rowsum(W')`` correction) removes that pass entirely and measured slightly
*slower*, but it also skips the reference's rounding of the normalised
activations to bf16; the last matmul sums 1536 mixed-sign terms, so it amplifies
a bf16-level difference by ~30x, and agreement with the reference fell from
1.000 to 0.54 once the weights were large enough for rtol rather than atol to
bind.  Every intermediate here is therefore rounded exactly where the reference
composition rounds it (see ``rb`` in the CUDA source).

Anything the fused path does not cover (non-bf16, a row count that is not a
multiple of 16, a mask that does not line up) falls back to ``_ref``, the
baseline composition of the frozen L1 kernels.
"""

from __future__ import annotations

import os
import subprocess

import torch
import torch.nn as nn

from ..L1.sigmoid import Sigmoid
from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear
from .alphafold3_swiglu import AdaLN, SwiGLU

_CPP_SRC = r"""#include <torch/extension.h>
#include <c10/util/Optional.h>

at::Tensor fk_ctb(const at::Tensor& a, const at::Tensor& s,
                  const c10::optional<at::Tensor>& mask, const at::Tensor& wp,
                  const at::Tensor& scr, int64_t M, int64_t S, int64_t A, int64_t H,
                  int64_t grid, int64_t smem, int64_t plan, double eps);
at::Tensor fk_swiglu(const at::Tensor& x, const c10::optional<at::Tensor>& mask,
                     const at::Tensor& wp, const at::Tensor& scr, int64_t M, int64_t C,
                     int64_t H, int64_t grid, int64_t smem, int64_t plan, double eps);
int64_t fk_max_blocks(int64_t which, int64_t smem);
"""

_CUDA_SRC = r"""// Fused AF3 SwiGLU-transition kernels (rationale in the Python module docstring).
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#define NWARP   8
#define THREADS (NWARP * 32)
#define SMPAD   8            /* bf16 elements of row padding in the staged tile */
#define MAXV    4            /* max uint4 chunks per lane in a row reduction    */
#define SPINCAP 200000       /* safety valve so a lost barrier cannot wedge a GPU */

using bf16 = __nv_bfloat16;

// ------------------------------------------------------------------ helpers
__device__ __forceinline__ uint32_t lds32(const bf16* p) {
  return *reinterpret_cast<const uint32_t*>(p);
}
__device__ __forceinline__ uint32_t ldg32(const bf16* p) {
  return __ldg(reinterpret_cast<const unsigned int*>(p));
}
__device__ __forceinline__ void st32(bf16* p, uint32_t v) {
  *reinterpret_cast<uint32_t*>(p) = v;
}
// The folded LayerNorm correction terms are kept in fp32 (bit-cast into the bf16
// blob): they are sums over the whole reduction length, so a bf16 copy would put
// ~0.4% of ``mu * R`` into every output element.
__device__ __forceinline__ float2 f2(uint32_t u) {
  return __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u));
}
__device__ __forceinline__ uint32_t pk(float x, float y) {
  __nv_bfloat162 h = __floats2bfloat162_rn(x, y);
  return *reinterpret_cast<const uint32_t*>(&h);
}
// Round through bf16.  The reference composition stores every intermediate as
// bf16, and the last matmul sums 1536 terms with mixed signs, so a value that
// rounds differently here can move the result by far more than its own error.
// Matching the reference's rounding points keeps the two bit-close instead.
__device__ __forceinline__ float rb(float x) {
  return __bfloat162float(__float2bfloat16(x));
}

__device__ __forceinline__ float sigm(float x) { return __frcp_rn(1.f + __expf(-x)); }
__device__ __forceinline__ float silu(float x) { return x * sigm(x); }

__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4],
                                         uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// B operand (8 output columns x 64 reduction) for one warp, in registers: all
// eight loads are issued before the first mma consumes one, which is the only
// way a problem this small keeps enough requests in flight to use the bus.
struct BF { uint32_t r[8]; };

__device__ __forceinline__ void load_b(BF& b, const bf16* W, int ldw, int n0, int kg) {
  const int lane = threadIdx.x & 31;
  const bf16* wp = W + (size_t)(n0 + (lane >> 2)) * ldw + kg + ((lane & 3) << 1);
#pragma unroll
  for (int kk = 0; kk < 4; ++kk) {
    b.r[kk * 2 + 0] = ldg32(wp + kk * 16);
    b.r[kk * 2 + 1] = ldg32(wp + kk * 16 + 8);
  }
}

// A operand (16 rows x 64 reduction), read straight from global.  The fragment
// layout wants 4 B per lane from 8 distinct rows, i.e. eight 32 B sectors per
// instruction with every byte used -- no staging pass and no block barrier.
__device__ __forceinline__ void load_a_g(uint32_t (&af)[4][4], const bf16* A, int lda,
                                         int kg) {
  const int lane = threadIdx.x & 31;
  const bf16* p0 = A + (size_t)(lane >> 2) * lda + kg + ((lane & 3) << 1);
  const bf16* p1 = p0 + (size_t)8 * lda;
#pragma unroll
  for (int kk = 0; kk < 4; ++kk) {
    af[kk][0] = ldg32(p0 + kk * 16);
    af[kk][1] = ldg32(p1 + kk * 16);
    af[kk][2] = ldg32(p0 + kk * 16 + 8);
    af[kk][3] = ldg32(p1 + kk * 16 + 8);
  }
}

// A operand from the staged (normalised) shared-memory tile.
__device__ __forceinline__ void load_a_s(uint32_t (&af)[4][4], const bf16* A, int lda,
                                         int kg) {
  const int lane = threadIdx.x & 31;
  const bf16* p0 = A + (lane >> 2) * lda + kg + ((lane & 3) << 1);
  const bf16* p1 = p0 + 8 * lda;
#pragma unroll
  for (int kk = 0; kk < 4; ++kk) {
    af[kk][0] = lds32(p0 + kk * 16);
    af[kk][1] = lds32(p1 + kk * 16);
    af[kk][2] = lds32(p0 + kk * 16 + 8);
    af[kk][3] = lds32(p1 + kk * 16 + 8);
  }
}

__device__ __forceinline__ void mma4(const uint32_t (&af)[4][4], const BF& b,
                                     float (&acc)[4]) {
#pragma unroll
  for (int kk = 0; kk < 4; ++kk) mma16816(acc, af[kk], b.r[kk * 2], b.r[kk * 2 + 1]);
}

__device__ __forceinline__ void wred2(float& x, float& y) {
#pragma unroll
  for (int o = 16; o; o >>= 1) {
    x += __shfl_xor_sync(0xffffffffu, x, o);
    y += __shfl_xor_sync(0xffffffffu, y, o);
  }
}

__device__ __forceinline__ void acc2(uint32_t u, float& s1, float& s2) {
  const float2 f = f2(u);
  s1 += f.x + f.y;
  s2 += f.x * f.x + f.y * f.y;
}

// One warp, one row: mean / rstd in fp32, matching the reference LayerNorm.
__device__ __forceinline__ void row_stats(const bf16* x, int nv, float inv, float eps,
                                          float& mu, float& rstd) {
  const int lane = threadIdx.x & 31;
  float s1 = 0.f, s2 = 0.f;
#pragma unroll
  for (int i = 0; i < MAXV; ++i) {
    const int v = lane + (i << 5);
    if (v < nv) {
      const uint4 q = *reinterpret_cast<const uint4*>(x + (v << 3));
      const uint32_t* u = reinterpret_cast<const uint32_t*>(&q);
      acc2(u[0], s1, s2); acc2(u[1], s1, s2); acc2(u[2], s1, s2); acc2(u[3], s1, s2);
    }
  }
  wred2(s1, s2);
  mu = s1 * inv;
  rstd = rsqrtf(fmaxf(s2 * inv - mu * mu, 0.f) + eps);
}

// LayerNorm one 16-row tile into shared memory and, in the same pass, take the
// row statistics of a second tile (AdaLN needs them for ``a``).  Two rows per
// warp; each row is read once and normalised out of registers, so the whole
// prologue costs one memory latency and the mma operand is the same bf16 tile the
// reference would have produced.
__device__ __forceinline__ void ln_tile2(const bf16* x, int ldx, int cx,
                                         const bf16* y, int ldy, int cy,
                                         const bf16* w, const bf16* b, float eps,
                                         float* xmu, float* xrs, float* ymu,
                                         float* yrs, bf16* sm, int lds) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int nvx = cx >> 3, nvy = cy >> 3;
  const float ix = 1.f / (float)cx, iy = 1.f / (float)cy;
#pragma unroll
  for (int rr = 0; rr < 16 / NWARP; ++rr) {
    const int r = warp + rr * NWARP;
    uint4 qx[MAXV], qy[MAXV];
#pragma unroll
    for (int i = 0; i < MAXV; ++i) {
      const int v = lane + (i << 5);
      if (v < nvx) qx[i] = *reinterpret_cast<const uint4*>(x + (size_t)r * ldx + (v << 3));
      if (y && v < nvy) qy[i] = *reinterpret_cast<const uint4*>(y + (size_t)r * ldy + (v << 3));
    }
    float a1 = 0.f, a2 = 0.f, b1 = 0.f, b2 = 0.f;
#pragma unroll
    for (int i = 0; i < MAXV; ++i) {
      const int v = lane + (i << 5);
      if (v < nvx) {
        const uint32_t* u = reinterpret_cast<const uint32_t*>(&qx[i]);
        acc2(u[0], a1, a2); acc2(u[1], a1, a2); acc2(u[2], a1, a2); acc2(u[3], a1, a2);
      }
      if (y && v < nvy) {
        const uint32_t* u = reinterpret_cast<const uint32_t*>(&qy[i]);
        acc2(u[0], b1, b2); acc2(u[1], b1, b2); acc2(u[2], b1, b2); acc2(u[3], b1, b2);
      }
    }
    wred2(a1, a2);
    if (y) wred2(b1, b2);
    const float mu = a1 * ix;
    const float rs = rsqrtf(fmaxf(a2 * ix - mu * mu, 0.f) + eps);
    if (lane == 0) {
      xmu[r] = mu; xrs[r] = rs;
      if (y) {
        const float n = b1 * iy;
        ymu[r] = n; yrs[r] = rsqrtf(fmaxf(b2 * iy - n * n, 0.f) + eps);
      }
    }
#pragma unroll
    for (int i = 0; i < MAXV; ++i) {
      const int v = lane + (i << 5);
      if (v < nvx) {
        const uint32_t* qu = reinterpret_cast<const uint32_t*>(&qx[i]);
        const uint4 wq = *reinterpret_cast<const uint4*>(w + (v << 3));
        const uint4 bq = *reinterpret_cast<const uint4*>(b + (v << 3));
        const uint32_t* wu = reinterpret_cast<const uint32_t*>(&wq);
        const uint32_t* bu = reinterpret_cast<const uint32_t*>(&bq);
        uint4 o;
        uint32_t* ou = reinterpret_cast<uint32_t*>(&o);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const float2 xv = f2(qu[j]), wv = f2(wu[j]), bv = f2(bu[j]);
          ou[j] = pk((xv.x - mu) * rs * wv.x + bv.x, (xv.y - mu) * rs * wv.y + bv.y);
        }
        *reinterpret_cast<uint4*>(sm + r * lds + (v << 3)) = o;
      }
    }
  }
}

// Sum the NA accumulator sets across the nwk k-groups into the kgrp==0 warps.
template <int NA>
__device__ __forceinline__ void kreduce(float* rbuf, int nwn, int nwk,
                                        float (&acc)[NA][4]) {
  if (nwk == 1) return;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int kgrp = warp / nwn, nsub = warp - kgrp * nwn;
  if (kgrp != 0) {
    float* mine = rbuf + (size_t)warp * (128 * NA) + lane * (4 * NA);
#pragma unroll
    for (int i = 0; i < NA; ++i)
      *reinterpret_cast<float4*>(mine + i * 4) =
          make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
  }
  __syncthreads();
  if (kgrp == 0) {
    for (int kg = 1; kg < nwk; ++kg) {
      const float* o = rbuf + (size_t)(kg * nwn + nsub) * (128 * NA) + lane * (4 * NA);
#pragma unroll
      for (int i = 0; i < NA; ++i) {
        const float4 v = *reinterpret_cast<const float4*>(o + i * 4);
        acc[i][0] += v.x; acc[i][1] += v.y; acc[i][2] += v.z; acc[i][3] += v.w;
      }
    }
  }
}

// Device-wide barrier.  The grid is clamped to the occupancy limit, so every
// block is resident and a monotonic counter with a host-side generation base
// needs no reset between launches.
__device__ __forceinline__ void gbar(unsigned long long* ctr, unsigned long long target) {
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    atomicAdd(ctr, 1ull);
    for (int i = 0; i < SPINCAP; ++i) {
      unsigned long long v;
      asm volatile("ld.acquire.gpu.u64 %0, [%1];" : "=l"(v) : "l"(ctr) : "memory");
      if (v >= target) break;
    }
  }
  __syncthreads();
}

// ------------------------------------------------------------------ params
struct CtbP {
  const bf16* a; const bf16* s; const bf16* mask; const bf16* w;
  bf16* out; bf16* a1; bf16* bh;
  unsigned long long* ctr; unsigned long long base;
  int M, S, A, H;
  int nwn1, nwk1, nwn2, nwk2, nwn3, nwk3;
  float eps;
};

struct SwP {
  const bf16* x; const bf16* mask; const bf16* w;
  bf16* out; bf16* bh;
  unsigned long long* ctr; unsigned long long base;
  int M, C, H;
  int nwn1, nwk1, nwn2, nwk2;
  float eps;
};

// ------------------------------------------------------------------ CTB kernel
__global__ void __launch_bounds__(THREADS) ctb_kernel(CtbP p) {
  extern __shared__ __align__(16) char shm[];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int g = lane >> 2, t2 = (lane & 3) << 1;
  const int mt = p.M >> 4;

  float* smu = reinterpret_cast<float*>(shm);   // [16] mean of s's rows
  float* srs = smu + 16;                        // [16] rstd of s's rows
  float* amu = srs + 16;                        // [16] mean of a's rows
  float* ars = amu + 16;                        // [16] rstd of a's rows
  float* rbuf = ars + 16;                       // k-group reduction
  bf16* sn = reinterpret_cast<bf16*>(rbuf + NWARP * 128 * 2);   // [16][S+SMPAD]

  const bf16* Wln = p.w;                             // [S] AdaLN layer_norm_s
  const bf16* Bln = Wln + p.S;                       // [S]
  const bf16* Wg1 = Bln + p.S;                       // [A,S]
  const bf16* Bg1 = Wg1 + (size_t)p.A * p.S;         // [A]
  const bf16* Ws  = Bg1 + p.A;                       // [A,S]
  const bf16* Wg2 = Ws + (size_t)p.A * p.S;          // [A,S] (raw s, no LN)
  const bf16* Bg2 = Wg2 + (size_t)p.A * p.S;         // [A]
  const bf16* Wa  = Bg2 + p.A;                       // [H,A]
  const bf16* Wb  = Wa + (size_t)p.H * p.A;          // [H,A]
  const bf16* Wo  = Wb + (size_t)p.H * p.A;          // [A,H]

  // ---- stage 1: AdaLN -> a1 ----
  {
    const int nwn = p.nwn1, nwk = p.nwk1;
    const int lds = p.S + SMPAD, nt = p.A / (8 * nwn), nch = p.S >> 6;
    const int kgrp = warp / nwn, nsub = warp - kgrp * nwn;
    for (int tile = blockIdx.x; tile < mt * nt; tile += gridDim.x) {
      const int mi = tile / nt, m0 = mi << 4;
      const int n0 = (tile - mi * nt) * (8 * nwn) + nsub * 8;
      const int c = n0 + t2;
      ln_tile2(p.s + (size_t)m0 * p.S, p.S, p.S, p.a + (size_t)m0 * p.A, p.A, p.A,
               Wln, Bln, p.eps, smu, srs, amu, ars, sn, lds);
      const float2 bg = f2(ldg32(Bg1 + c));
      __syncthreads();
      float acc[2][4] = {};
#pragma unroll 2
      for (int kc = kgrp; kc < nch; kc += nwk) {
        const int kg = kc << 6;
        uint32_t af[4][4];
        BF bb0, bb1;
        load_a_s(af, sn, lds, kg);
        load_b(bb0, Wg1, p.S, n0, kg);
        load_b(bb1, Ws, p.S, n0, kg);
        mma4(af, bb0, acc[0]);
        mma4(af, bb1, acc[1]);
      }
      kreduce<2>(rbuf, nwn, nwk, acc);
      if (kgrp == 0) {
#pragma unroll
        for (int rh = 0; rh < 2; ++rh) {
          const int r = g + (rh << 3);
          const size_t off = (size_t)(m0 + r) * p.A + c;
          const float2 av = f2(ldg32(p.a + off));
          const float mu = amu[r], rs = ars[r];
          const float g0 = rb(sigm(rb(acc[0][rh * 2] + bg.x)));
          const float g1v = rb(sigm(rb(acc[0][rh * 2 + 1] + bg.y)));
          const float s0 = rb(rb((av.x - mu) * rs) + rb(acc[1][rh * 2]));
          const float s1v = rb(rb((av.y - mu) * rs) + rb(acc[1][rh * 2 + 1]));
          st32(p.a1 + off, pk(g0 * s0, g1v * s1v));
        }
      }
      __syncthreads();
    }
  }
  gbar(p.ctr, p.base + gridDim.x);

  // ---- stage 2: SwiGLU hidden -> bh ----
  {
    const int nwn = p.nwn2, nwk = p.nwk2;
    const int nt = p.H / (8 * nwn), nch = p.A >> 6;
    const int kgrp = warp / nwn, nsub = warp - kgrp * nwn;
    for (int tile = blockIdx.x; tile < mt * nt; tile += gridDim.x) {
      const int mi = tile / nt, m0 = mi << 4;
      const int n0 = (tile - mi * nt) * (8 * nwn) + nsub * 8;
      const bf16* Ab = p.a1 + (size_t)m0 * p.A;
      float acc[2][4] = {};
#pragma unroll 2
      for (int kc = kgrp; kc < nch; kc += nwk) {
        const int kg = kc << 6;
        uint32_t af[4][4];
        BF bb0, bb1;
        load_a_g(af, Ab, p.A, kg);
        load_b(bb0, Wa, p.A, n0, kg);
        load_b(bb1, Wb, p.A, n0, kg);
        mma4(af, bb0, acc[0]);
        mma4(af, bb1, acc[1]);
      }
      kreduce<2>(rbuf, nwn, nwk, acc);
      if (kgrp == 0) {
        const int c = n0 + t2;
#pragma unroll
        for (int rh = 0; rh < 2; ++rh) {
          const int r = g + (rh << 3);
          st32(p.bh + (size_t)(m0 + r) * p.H + c,
               pk(rb(silu(rb(acc[0][rh * 2]))) * rb(acc[1][rh * 2]),
                  rb(silu(rb(acc[0][rh * 2 + 1]))) * rb(acc[1][rh * 2 + 1])));
        }
      }
      if (nwk > 1) __syncthreads();
    }
  }
  gbar(p.ctr, p.base + 2 * gridDim.x);

  // ---- stage 3: linear_out, output gate and mask ----
  {
    const int nwn = p.nwn3, nwk = p.nwk3;
    const int nt = p.A / (8 * nwn), nch = p.H >> 6, nchs = p.S >> 6;
    const int kgrp = warp / nwn, nsub = warp - kgrp * nwn;
    const int nmax = nch > nchs ? nch : nchs;
    for (int tile = blockIdx.x; tile < mt * nt; tile += gridDim.x) {
      const int mi = tile / nt, m0 = mi << 4;
      const int n0 = (tile - mi * nt) * (8 * nwn) + nsub * 8;
      const int c = n0 + t2;
      const bf16* Ab = p.bh + (size_t)m0 * p.H;
      const bf16* Sb = p.s + (size_t)m0 * p.S;
      const float2 bg2 = f2(ldg32(Bg2 + c));
      float mv0 = 1.f, mv1 = 1.f;
      if (p.mask) {
        mv0 = __bfloat162float(p.mask[m0 + g]);
        mv1 = __bfloat162float(p.mask[m0 + g + 8]);
      }
      float acc[2][4] = {};
      for (int kc = kgrp; kc < nmax; kc += nwk) {
        const int kg = kc << 6;
        if (kc < nch) {
          uint32_t af[4][4];
          BF bb0;
          load_a_g(af, Ab, p.H, kg);
          load_b(bb0, Wo, p.H, n0, kg);
          mma4(af, bb0, acc[0]);
        }
        if (kc < nchs) {
          uint32_t af[4][4];
          BF bb1;
          load_a_g(af, Sb, p.S, kg);
          load_b(bb1, Wg2, p.S, n0, kg);
          mma4(af, bb1, acc[1]);
        }
      }
      kreduce<2>(rbuf, nwn, nwk, acc);
      if (kgrp == 0) {
        st32(p.out + (size_t)(m0 + g) * p.A + c,
             pk(rb(rb(sigm(rb(acc[1][0] + bg2.x))) * rb(acc[0][0])) * mv0,
                rb(rb(sigm(rb(acc[1][1] + bg2.y))) * rb(acc[0][1])) * mv0));
        st32(p.out + (size_t)(m0 + g + 8) * p.A + c,
             pk(rb(rb(sigm(rb(acc[1][2] + bg2.x))) * rb(acc[0][2])) * mv1,
                rb(rb(sigm(rb(acc[1][3] + bg2.y))) * rb(acc[0][3])) * mv1));
      }
      if (nwk > 1) __syncthreads();
    }
  }
}

// ------------------------------------------------------------------ SwiGLU kernel
__global__ void __launch_bounds__(THREADS) swiglu_kernel(SwP p) {
  extern __shared__ __align__(16) char shm[];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int g = lane >> 2, t2 = (lane & 3) << 1;
  const int mt = p.M >> 4;

  float* xmu = reinterpret_cast<float*>(shm);   // [16] mean of x's rows
  float* xrs = xmu + 16;                        // [16] rstd of x's rows
  float* rbuf = xrs + 16;                       // k-group reduction
  bf16* xn = reinterpret_cast<bf16*>(rbuf + NWARP * 128 * 2);   // [16][C+SMPAD]

  const bf16* Wln = p.w;                        // [C] LayerNorm scale
  const bf16* Bln = Wln + p.C;                  // [C] LayerNorm offset
  const bf16* Wa  = Bln + p.C;                  // [H,C]
  const bf16* Wb  = Wa + (size_t)p.H * p.C;     // [H,C]
  const bf16* Wo  = Wb + (size_t)p.H * p.C;     // [C,H]

  // ---- stage 1: LayerNorm + SwiGLU hidden -> bh ----
  {
    const int nwn = p.nwn1, nwk = p.nwk1;
    const int lds = p.C + SMPAD, nt = p.H / (8 * nwn), nch = p.C >> 6;
    const int kgrp = warp / nwn, nsub = warp - kgrp * nwn;
    for (int tile = blockIdx.x; tile < mt * nt; tile += gridDim.x) {
      const int mi = tile / nt, m0 = mi << 4;
      const int n0 = (tile - mi * nt) * (8 * nwn) + nsub * 8;
      ln_tile2(p.x + (size_t)m0 * p.C, p.C, p.C, nullptr, 0, 0, Wln, Bln, p.eps,
               xmu, xrs, nullptr, nullptr, xn, lds);
      __syncthreads();
      float acc[2][4] = {};
#pragma unroll 2
      for (int kc = kgrp; kc < nch; kc += nwk) {
        const int kg = kc << 6;
        uint32_t af[4][4];
        BF bb0, bb1;
        load_a_s(af, xn, lds, kg);
        load_b(bb0, Wa, p.C, n0, kg);
        load_b(bb1, Wb, p.C, n0, kg);
        mma4(af, bb0, acc[0]);
        mma4(af, bb1, acc[1]);
      }
      kreduce<2>(rbuf, nwn, nwk, acc);
      if (kgrp == 0) {
        const int c = n0 + t2;
#pragma unroll
        for (int rh = 0; rh < 2; ++rh) {
          const int r = g + (rh << 3);
          st32(p.bh + (size_t)(m0 + r) * p.H + c,
               pk(rb(silu(rb(acc[0][rh * 2]))) * rb(acc[1][rh * 2]),
                  rb(silu(rb(acc[0][rh * 2 + 1]))) * rb(acc[1][rh * 2 + 1])));
        }
      }
      __syncthreads();
    }
  }

  gbar(p.ctr, p.base + gridDim.x);

  // ---- stage 2: linear_out and mask ----
  {
    const int nwn = p.nwn2, nwk = p.nwk2;
    const int nt = p.C / (8 * nwn), nch = p.H >> 6;
    const int kgrp = warp / nwn, nsub = warp - kgrp * nwn;
    for (int tile = blockIdx.x; tile < mt * nt; tile += gridDim.x) {
      const int mi = tile / nt, m0 = mi << 4;
      const int n0 = (tile - mi * nt) * (8 * nwn) + nsub * 8;
      const bf16* Ab = p.bh + (size_t)m0 * p.H;
      float mv0 = 1.f, mv1 = 1.f;
      if (p.mask) {
        mv0 = __bfloat162float(p.mask[m0 + g]);
        mv1 = __bfloat162float(p.mask[m0 + g + 8]);
      }
      float acc[1][4] = {};
#pragma unroll 2
      for (int kc = kgrp; kc < nch; kc += nwk) {
        uint32_t af[4][4];
        BF bb0;
        load_a_g(af, Ab, p.H, kc << 6);
        load_b(bb0, Wo, p.H, n0, kc << 6);
        mma4(af, bb0, acc[0]);
      }
      kreduce<1>(rbuf, nwn, nwk, acc);
      if (kgrp == 0) {
        const int c = n0 + t2;
        st32(p.out + (size_t)(m0 + g) * p.C + c,
             pk(rb(acc[0][0]) * mv0, rb(acc[0][1]) * mv0));
        st32(p.out + (size_t)(m0 + g + 8) * p.C + c,
             pk(rb(acc[0][2]) * mv1, rb(acc[0][3]) * mv1));
      }
      if (nwk > 1) __syncthreads();
    }
  }
}

// ------------------------------------------------------------------ host side
// Barrier counter, one per device: monotonic, so the kernel's target is simply
// the host-side running total.  Kernels on a device are serialised by the stream
// they are launched on, so a single counter per device is enough.
static unsigned long long* bar_state(unsigned long long nadd,
                                     unsigned long long& base) {
  constexpr int kMaxDev = 16;
  static unsigned long long* ptr[kMaxDev] = {};
  static unsigned long long acc[kMaxDev] = {};
  const int dev = c10::cuda::current_device();
  TORCH_CHECK(dev >= 0 && dev < kMaxDev, "fk_af3: device index out of range");
  if (ptr[dev] == nullptr) {
    C10_CUDA_CHECK(cudaMalloc(&ptr[dev], 256));
    C10_CUDA_CHECK(
        cudaMemsetAsync(ptr[dev], 0, 256, c10::cuda::getCurrentCUDAStream()));
  }
  base = acc[dev];
  acc[dev] += nadd;
  return ptr[dev];
}

int64_t fk_max_blocks(int64_t which, int64_t smem) {
  int n = 0;
  const void* f = which == 0 ? (const void*)ctb_kernel : (const void*)swiglu_kernel;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, f, THREADS,
                                                               (size_t)smem));
  const int sm = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  return (int64_t)n * sm;
}

at::Tensor fk_ctb(const at::Tensor& a, const at::Tensor& s,
                  const c10::optional<at::Tensor>& mask, const at::Tensor& wp,
                  const at::Tensor& scr, int64_t M, int64_t S, int64_t A, int64_t H,
                  int64_t grid, int64_t smem, int64_t plan, double eps) {
  TORCH_CHECK(a.scalar_type() == at::kBFloat16 && s.scalar_type() == at::kBFloat16,
              "fk_ctb: bf16 inputs required");
  const at::Tensor ac = a.contiguous(), sc = s.contiguous();
  TORCH_CHECK(ac.numel() == M * A && sc.numel() == M * S, "fk_ctb: shape mismatch");
  TORCH_CHECK(ac.device() == wp.device() && sc.device() == wp.device()
                  && scr.device() == wp.device(),
              "fk_ctb: inputs, weights and scratch must share one device");
  at::Tensor out = at::empty_like(ac);
  CtbP p;
  p.a = (const bf16*)ac.const_data_ptr();
  p.s = (const bf16*)sc.const_data_ptr();
  p.mask = nullptr;
  if (mask.has_value() && mask->defined()) {
    TORCH_CHECK(mask->numel() == M && mask->scalar_type() == at::kBFloat16 &&
                mask->is_contiguous(), "fk_ctb: bad mask");
    p.mask = (const bf16*)mask->const_data_ptr();
  }
  p.w = (const bf16*)wp.const_data_ptr();
  p.out = (bf16*)out.data_ptr();
  p.a1 = (bf16*)scr.data_ptr();
  p.bh = p.a1 + M * A;
  p.M = (int)M; p.S = (int)S; p.A = (int)A; p.H = (int)H;
  p.nwn1 = (int)(plan & 15);         p.nwk1 = (int)((plan >> 4) & 15);
  p.nwn2 = (int)((plan >> 8) & 15);  p.nwk2 = (int)((plan >> 12) & 15);
  p.nwn3 = (int)((plan >> 16) & 15); p.nwk3 = (int)((plan >> 20) & 15);
  p.eps = (float)eps;
  p.ctr = bar_state(2ull * (unsigned long long)grid, p.base);
  ctb_kernel<<<(int)grid, THREADS, (size_t)smem, c10::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor fk_swiglu(const at::Tensor& x, const c10::optional<at::Tensor>& mask,
                     const at::Tensor& wp, const at::Tensor& scr, int64_t M, int64_t C,
                     int64_t H, int64_t grid, int64_t smem, int64_t plan, double eps) {
  TORCH_CHECK(x.scalar_type() == at::kBFloat16, "fk_swiglu: bf16 input required");
  const at::Tensor xc = x.contiguous();
  TORCH_CHECK(xc.numel() == M * C, "fk_swiglu: shape mismatch");
  TORCH_CHECK(xc.device() == wp.device() && scr.device() == wp.device(),
              "fk_swiglu: inputs, weights and scratch must share one device");
  at::Tensor out = at::empty_like(xc);
  SwP p;
  p.x = (const bf16*)xc.const_data_ptr();
  p.mask = nullptr;
  if (mask.has_value() && mask->defined()) {
    TORCH_CHECK(mask->numel() == M && mask->scalar_type() == at::kBFloat16 &&
                mask->is_contiguous(), "fk_swiglu: bad mask");
    p.mask = (const bf16*)mask->const_data_ptr();
  }
  p.w = (const bf16*)wp.const_data_ptr();
  p.out = (bf16*)out.data_ptr();
  p.bh = (bf16*)scr.data_ptr();
  p.M = (int)M; p.C = (int)C; p.H = (int)H;
  p.nwn1 = (int)(plan & 15);        p.nwk1 = (int)((plan >> 4) & 15);
  p.nwn2 = (int)((plan >> 8) & 15); p.nwk2 = (int)((plan >> 12) & 15);
  p.eps = (float)eps;
  p.ctr = bar_state((unsigned long long)grid, p.base);
  swiglu_kernel<<<(int)grid, THREADS, (size_t)smem,
                  c10::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
"""

# --------------------------------------------------------------------------- JIT

_EXT = None
_TRIED = False
_NWARP = 8          # must match NWARP in the CUDA source
_SMEM_LIMIT = 48 * 1024


def _arch() -> str | None:
    """Local compute capability, without creating a CUDA context."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL, timeout=10)
    except Exception:  # noqa: BLE001 - no nvidia-smi: leave the ambient list alone
        return None
    caps = sorted({c.strip() for c in out.splitlines() if c.strip()})
    return " ".join(f"{c}a" if c.split(".")[0] in ("9", "10", "12") else c for c in caps)


def _ext():
    """JIT-compile the fused kernels once; None if that is not possible."""
    global _EXT, _TRIED
    if _TRIED:
        return _EXT
    _TRIED = True
    try:
        from torch.utils.cpp_extension import load_inline
        prev = os.environ.get("TORCH_CUDA_ARCH_LIST")
        arch = os.environ.get("FASTKERNELS_CUDA_ARCH_LIST") or _arch()
        if arch:
            os.environ["TORCH_CUDA_ARCH_LIST"] = arch
        try:
            _EXT = load_inline(
                name="fk_l2_af3_swiglu_transition",
                cpp_sources=_CPP_SRC,
                cuda_sources=_CUDA_SRC,
                functions=["fk_ctb", "fk_swiglu", "fk_max_blocks"],
                extra_cuda_cflags=[
                    "-O3",
                    "--expt-relaxed-constexpr",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                ],
                verbose=False,
            )
        finally:
            if arch:
                if prev is None:
                    os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
                else:
                    os.environ["TORCH_CUDA_ARCH_LIST"] = prev
    except Exception:  # noqa: BLE001 - no nvcc / no GPU: run the reference path
        _EXT = None
    return _EXT


def _split(nchunk: int, n: int):
    """Warp layout for one GEMM stage: (warps over output columns, warps over K).

    Each warp owns 8 output columns and a 64-wide slice of the reduction, so the
    two factors trade block count against warp utilisation.  Prefer the largest
    K split that does not leave warps idle -- more K parallelism means more
    concurrent weight loads, which is what these shapes are bound by.
    """
    fallback = None
    for nwk in (1, 2, 4, _NWARP):
        nwn = _NWARP // nwk
        if n % (8 * nwn):
            continue
        if nwk >= nchunk or nwk == _NWARP:
            return nwn, nwk
        fallback = (nwn, nwk)
    return fallback


def _pack(parts) -> torch.Tensor:
    """One contiguous bf16 blob of every weight, in the order the kernel expects."""
    return torch.cat([p.detach().reshape(-1) for p in parts]).contiguous()


def _rows(t: torch.Tensor, width: int) -> int:
    n = t.numel()
    return n // width if width and n % width == 0 else -1


class SwiGLUTransition(nn.Module):
    """AF3 Algorithm 11: SwiGLU-based transition.

    Args:
        c_in: Input channel dimension
        n: Factor multiplied to c_in for hidden dimension
    """

    def __init__(self, c_in: int, n: int):
        super().__init__()
        self.c_in = c_in
        self.n = n

        self.layer_norm = LayerNorm(c_in)
        self.swiglu = SwiGLU(c_in, n * c_in)
        self.linear_out = Linear(n * c_in, c_in, bias=False)

        self._key = None      # x.shape the current plan was built for
        self._plan = None     # () extension args, or None for the reference path
        self._m = 0

    # -- fused path ---------------------------------------------------------
    def _setup(self, x: torch.Tensor) -> None:
        self._key = x.shape
        self._plan = None
        ext = _ext()
        if ext is None or not x.is_cuda or x.dtype is not torch.bfloat16:
            return
        c, h = self.c_in, self.n * self.c_in
        w = self.linear_out.weight
        if w.dtype is not torch.bfloat16 or self.layer_norm.weight is None:
            return
        m = _rows(x, c)
        if m <= 0 or m % 16 or c % 64 or h % 64 or c > 1024:
            return
        s1 = _split(c // 64, h)
        s2 = _split(h // 64, c)
        if s1 is None or s2 is None:
            return
        smem = 128 + _NWARP * 128 * 2 * 4 + 32 * (c + 8)
        if smem > _SMEM_LIMIT:
            return
        mt = m // 16
        tiles = max(mt * (h // (8 * s1[0])), mt * (c // (8 * s2[0])))
        grid = min(tiles, int(ext.fk_max_blocks(1, smem)))
        bias = self.layer_norm.bias
        if bias is None:
            bias = torch.zeros_like(self.layer_norm.weight)
        wp = _pack([self.layer_norm.weight, bias,
                    self.swiglu.linear_a.weight, self.swiglu.linear_b.weight,
                    self.linear_out.weight])
        scr = torch.empty(m * h, dtype=torch.bfloat16, device=x.device)
        plan = s1[0] | s1[1] << 4 | s2[0] << 8 | s2[1] << 12
        self._m = m
        self._plan = (wp, scr, m, c, h, grid, smem, plan, float(self.layer_norm.eps))

    # -- reference path (baseline composition) ------------------------------
    def _ref(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if mask is None:
            mask = x.new_ones(x.shape[:-1])
        mask = mask.unsqueeze(-1)
        x = self.layer_norm(x)
        x = self.swiglu(x)
        x = self.linear_out(x)
        return x * mask

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        chunk_size: int | None = None,
        ckpt_chunk_size: int | None = None,
    ) -> torch.Tensor:
        if x.shape != self._key:
            self._setup(x)
        plan = self._plan
        if plan is not None and (mask is None or (
                mask.numel() == self._m and mask.dtype is torch.bfloat16
                and mask.is_contiguous())):
            return _EXT.fk_swiglu(x, mask, *plan)
        return self._ref(x, mask)


class ConditionedTransitionBlock(nn.Module):
    """AF3 Algorithm 25: SwiGLU transition with AdaLN-Zero conditioning.

    Submodule names match the reference checkpoint:
    - layer_norm: AdaLN
    - swiglu: SwiGLU
    - linear_g: output gate Linear(c_s, c_a, bias=True)
    - linear_out: Linear(n*c_a, c_a, bias=False)

    Args:
        c_a: Activation channel dimension
        c_s: Conditioning channel dimension
        n: Factor for hidden dimension
    """

    def __init__(self, c_a: int, c_s: int, n: int):
        super().__init__()
        self.layer_norm = AdaLN(c_a=c_a, c_s=c_s)
        self.swiglu = SwiGLU(c_a, n * c_a)
        self.sigmoid = Sigmoid()
        self.linear_g = Linear(c_s, c_a, bias=True)
        self.linear_out = Linear(n * c_a, c_a, bias=False)

        self._c_a = c_a
        self._c_s = c_s
        self._n = n
        self._key = None      # (a.shape, s.shape) the current plan was built for
        self._plan = None
        self._m = 0

    def _setup(self, a: torch.Tensor, s: torch.Tensor) -> None:
        self._key = (a.shape, s.shape)
        self._plan = None
        ext = _ext()
        if ext is None or not a.is_cuda:
            return
        if a.dtype is not torch.bfloat16 or s.dtype is not torch.bfloat16:
            return
        ca, cs, h = self._c_a, self._c_s, self._n * self._c_a
        ln = self.layer_norm
        if (self.linear_out.weight.dtype is not torch.bfloat16
                or ln.layer_norm_s.weight is None
                or ln.layer_norm_a.eps != ln.layer_norm_s.eps
                or ln.layer_norm_a.weight is not None
                or ln.layer_norm_a.bias is not None):
            return
        m = _rows(a, ca)
        if m <= 0 or m != _rows(s, cs) or m % 16:
            return
        if cs % 64 or ca % 64 or h % 64 or cs > 1024 or ca > 1024:
            return
        s1 = _split(cs // 64, ca)
        s2 = _split(ca // 64, h)
        s3 = _split(h // 64, ca)
        if s1 is None or s2 is None or s3 is None:
            return
        smem = 256 + _NWARP * 128 * 2 * 4 + 32 * (cs + 8)
        if smem > _SMEM_LIMIT:
            return
        mt = m // 16
        tiles = max(mt * (ca // (8 * s1[0])), mt * (h // (8 * s2[0])),
                    mt * (ca // (8 * s3[0])))
        grid = min(tiles, int(ext.fk_max_blocks(0, smem)))
        lnb = ln.layer_norm_s.bias
        if lnb is None:
            lnb = torch.zeros_like(ln.layer_norm_s.weight)
        wp = _pack([ln.layer_norm_s.weight, lnb,
                    ln.linear_g.weight, ln.linear_g.bias, ln.linear_s.weight,
                    self.linear_g.weight, self.linear_g.bias,
                    self.swiglu.linear_a.weight, self.swiglu.linear_b.weight,
                    self.linear_out.weight])
        scr = torch.empty(m * (ca + h), dtype=torch.bfloat16, device=a.device)
        plan = (s1[0] | s1[1] << 4 | s2[0] << 8 | s2[1] << 12
                | s3[0] << 16 | s3[1] << 20)
        self._m = m
        self._plan = (wp, scr, m, cs, ca, h, grid, smem, plan,
                      float(ln.layer_norm_a.eps))

    def _ref(self, a: torch.Tensor, s: torch.Tensor,
             mask: torch.Tensor | None) -> torch.Tensor:
        if mask is None:
            mask = a.new_ones(a.shape[:-1])
        mask = mask.unsqueeze(-1)
        a = self.layer_norm(a, s)
        b = self.swiglu(a)
        a = self.sigmoid(self.linear_g(s)) * self.linear_out(b)
        return a * mask

    def forward(
        self,
        a: torch.Tensor,
        s: torch.Tensor,
        mask: torch.Tensor | None = None,
        chunk_size: int | None = None,
    ) -> torch.Tensor:
        if (a.shape, s.shape) != self._key:
            self._setup(a, s)
        plan = self._plan
        if plan is not None and (mask is None or (
                mask.numel() == self._m and mask.dtype is torch.bfloat16
                and mask.is_contiguous())):
            return _EXT.fk_ctb(a, s, mask, *plan)
        return self._ref(a, s, mask)
