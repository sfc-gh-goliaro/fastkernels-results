"""SwiGLU activation and AdaLN for AlphaFold3 (L2 composites) -- fused CUDA.

SwiGLU: SiLU(linear_a(x)) * linear_b(x)
AdaLN: Adaptive Layer Normalization

Both composites are launch-bound at the captured AlphaFold3 shapes (16..1536
rows, 64..768 channels): the reference builds each one from 4-12 separate eager
ops whose launches cost far more than the arithmetic -- AdaLN spends ~70 us of a
~70 us call on them. Each class here runs as a single fused kernel (layer norms,
both projections on bf16 tensor cores with fp32 accumulate, and the elementwise
tail in one pass), and its forward body is one extension call, which drops the
Python-side cost from ~100 us to ~8 us.

Every place the eager path rounds to bfloat16 -- each F.linear, layer_norm and
sigmoid result -- is rounded here too, so outputs track it to within a bfloat16
ulp instead of merely landing inside the tolerance. Shapes the kernels do not
cover (non-bf16, non-contiguous, K or N not a multiple of 16) fall back to the
same op sequence inside the extension.

Reference: openfold3/core/model/primitives/activations.py SwiGLU
           openfold3/core/model/primitives/normalization.py AdaLN
"""

from __future__ import annotations

import hashlib
import os

import torch
import torch.nn as nn

from ..L1.sigmoid import Sigmoid
from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear
from ..L1.silu import SiLU

_CUDA_SRC = r"""
// Fused AlphaFold3 SwiGLU + AdaLN kernels (bfloat16, mma.m16n8k16 tensor cores).
//
// These problems are small enough that both kernels are bound by dependent
// memory round trips, not by bandwidth or FLOPs, which drives the shape of the
// code: every global access is a 16-byte vector or a 4-byte mma fragment slice,
// four K steps are issued before the first mma so the loads overlap, tile
// staging is a division-free contiguous copy, reductions stay inside a warp
// wherever possible, and the epilogue reads its inputs from registers.
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>

using bf16_t = __nv_bfloat16;
using bf162_t = __nv_bfloat162;

#define FULLM 0xffffffffu
#define BF2F(x) __bfloat162float(x)
#define F2BF(x) __float2bfloat16(x)
// Round a fp32 value through bfloat16, as every eager op boundary does.
#define RND(x) BF2F(F2BF(x))

// 16 bytes == 8 bfloat16 == 4 bfloat16x2. Alignment matters: without it nvcc
// emits eight 2-byte loads per element group instead of one LDG.128.
struct alignas(16) Vec8 {
  unsigned w[4];
};

__device__ __forceinline__ bf162_t as_b2(unsigned u) {
  bf162_t r;
  __builtin_memcpy(&r, &u, 4);
  return r;
}

__device__ __forceinline__ unsigned as_u(bf162_t h) {
  unsigned u;
  __builtin_memcpy(&u, &h, 4);
  return u;
}

// ------------------------------------------------------- mma primitives ----
// One warp computes a 16x16 fp32 tile as two m16n8k16 bf16 tensor-core ops.
// Fragments are loaded by hand (ldmatrix for a shared-memory A tile, plain
// 32-bit global loads otherwise) rather than through nvcuda::wmma, whose
// load_matrix_sync compiles to *generic* 4-byte loads here -- one un-pipelined
// memory round trip per K step, which dominated everything else.
#define MMA_M16N8K16(d, a, b)                                                  \
  asm volatile(                                                                \
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "                   \
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"                \
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])                         \
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]))

// A operand (16x16, row-major) from shared memory. The caller passes this
// lane's row address -- row l%16, column block (l/16)*8 -- and ldmatrix.x4 lands
// the four 8x8 quadrants in a0..a3 exactly as mma wants them.
__device__ __forceinline__ void ld_a_at_shared(unsigned (&a)[4],
                                               const bf16_t *p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
               : "r"((unsigned)__cvta_generic_to_shared(p)));
}

// A operand straight from a row-major [M, K] matrix in global memory.
// Lane l holds rows {l/4, l/4+8} x cols {(l%4)*2, +1} and the same at col+8.
// The pointer is this lane's k=0 element; the caller bumps it per K step, so
// the inner loop costs one 64-bit add instead of re-deriving the address.
__device__ __forceinline__ void ld_a_at(unsigned (&a)[4], const bf16_t *p,
                                        int ldm, bool ok0, bool ok1) {
  a[0] = ok0 ? *reinterpret_cast<const unsigned *>(p) : 0u;
  a[2] = ok0 ? *reinterpret_cast<const unsigned *>(p + 8) : 0u;
  a[1] = ok1 ? *reinterpret_cast<const unsigned *>(p + 8 * ldm) : 0u;
  a[3] = ok1 ? *reinterpret_cast<const unsigned *>(p + 8 * ldm + 8) : 0u;
}

// B operand (16x8, column-major == 8 rows of a row-major [N, K] weight).
// Lane l holds w[n0 + l/4][k + (l%4)*2 .. +1] and the same at k+8.
__device__ __forceinline__ void ld_b_at(unsigned (&b)[2], const bf16_t *p) {
  b[0] = *reinterpret_cast<const unsigned *>(p);
  b[1] = *reinterpret_cast<const unsigned *>(p + 8);
}

// This lane's element of a 16x8 B tile at k = 0.
__device__ __forceinline__ const bf16_t *b_base(const bf16_t *W, int ldm,
                                                int n0, int lane) {
  return W + (size_t)(n0 + (lane >> 2)) * ldm + ((lane & 3) << 1);
}

// Accumulator element (i) of a 16x8 tile lives at row (lane/4 + 8*(i/2)),
// col (lane%4)*2 + i%2 -- so each lane owns two horizontally adjacent pairs.
#define ACC_ROW(lane, i) (((lane) >> 2) + (((i) >> 1) << 3))
#define ACC_COL(lane) (((lane) & 3) << 1)

// ---------------------------------------------------------------- SwiGLU ----
// out[m,n] = silu(x[m,:] . wa[n,:]) * (x[m,:] . wb[n,:])
//
// A warp owns a 16x16 output tile and walks K in steps of 16*WK; x and both
// weights stream directly from global memory into mma fragments, so there is no
// staging, no barrier before the math, and the K steps pipeline. WK warps split
// K and reduce their fp32 partials through shared memory (skipped when WK==1).
template <int BM, int BN, int WM, int WN, int WK>
__global__ __launch_bounds__(32 * WM * WN * WK) void swiglu_kernel(
    const bf16_t *__restrict__ X, const bf16_t *__restrict__ WA,
    const bf16_t *__restrict__ WB, bf16_t *__restrict__ O,
    int M, int K, int N) {
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int wm = warp % WM;
  const int wn = (warp / WM) % WN;
  const int wk = warp / (WM * WN);
  const int m0 = blockIdx.y * BM + wm * 16;
  const int n0 = blockIdx.x * BN + wn * 16;

  extern __shared__ float red[];

  float da[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
  float db[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
  if (n0 < N && m0 < M) {
    // KU K steps per iteration: 48 loads are issued back to back before the
    // first mma, which is what keeps enough bytes in flight to cover DRAM
    // latency at these tiny occupancies. Offsets are compile-time constants, so
    // the addresses cost nothing beyond the three pointer bumps.
    constexpr int KS = WK * 16;  // K advanced per warp per step
    constexpr int KU = 4;
    const int arow = m0 + (lane >> 2);
    const bool ok0 = arow < M, ok1 = arow + 8 < M;
    const bf16_t *px = X + (size_t)arow * K + wk * 16 + ((lane & 3) << 1);
    const bf16_t *pa0 = b_base(WA, K, n0, lane) + wk * 16;
    const bf16_t *pb0 = b_base(WB, K, n0, lane) + wk * 16;
    const int nstep = (K - wk * 16 + KS - 1) / KS;
#pragma unroll 1
    for (int st = 0; st < nstep; st += KU) {
      unsigned a[KU][4], ba[KU][2][2], bb[KU][2][2];
#pragma unroll
      for (int u = 0; u < KU; ++u) {
        if (st + u < nstep) {
          ld_a_at(a[u], px + u * KS, K, ok0, ok1);
          ld_b_at(ba[u][0], pa0 + u * KS);
          ld_b_at(ba[u][1], pa0 + 8 * K + u * KS);
          ld_b_at(bb[u][0], pb0 + u * KS);
          ld_b_at(bb[u][1], pb0 + 8 * K + u * KS);
        } else {  // zero fragments: the mma is a no-op, no divergence
#pragma unroll
          for (int j = 0; j < 4; ++j) a[u][j] = 0u;
#pragma unroll
          for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int j = 0; j < 2; ++j) ba[u][h][j] = bb[u][h][j] = 0u;
        }
      }
      px += KU * KS;
      pa0 += KU * KS;
      pb0 += KU * KS;
#pragma unroll
      for (int u = 0; u < KU; ++u) {
        MMA_M16N8K16(da[0], a[u], ba[u][0]);
        MMA_M16N8K16(da[1], a[u], ba[u][1]);
        MMA_M16N8K16(db[0], a[u], bb[u][0]);
        MMA_M16N8K16(db[1], a[u], bb[u][1]);
      }
    }
  }

  if (WK > 1) {  // fold the K-split partials into the wk==0 warp's registers
    // One slot per (tile, wk>0, lane): 16 floats == four 128-bit accesses.
    // (A shared atomicAdd would be a CAS spin loop on this architecture.)
    float *slot = red + ((size_t)((wm * WN + wn) * (WK - 1)) * 32 + lane) * 16;
    if (wk != 0) {
      float *mine = slot + (size_t)(wk - 1) * 32 * 16;
#pragma unroll
      for (int h = 0; h < 2; ++h)
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          mine[h * 4 + i] = da[h][i];
          mine[8 + h * 4 + i] = db[h][i];
        }
    }
    __syncthreads();
    if (wk != 0) return;
#pragma unroll
    for (int q = 0; q < WK - 1; ++q) {
      const float *o = slot + (size_t)q * 32 * 16;
#pragma unroll
      for (int h = 0; h < 2; ++h)
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          da[h][i] += o[h * 4 + i];
          db[h][i] += o[8 + h * 4 + i];
        }
    }
  }
  if (n0 >= N || m0 >= M) return;

  // Epilogue from registers: lane owns cols {c, c+1} of rows {r, r+8}.
  const int col = ACC_COL(lane);
#pragma unroll
  for (int h = 0; h < 2; ++h)
#pragma unroll
    for (int i = 0; i < 4; i += 2) {
      const int row = m0 + ACC_ROW(lane, i);
      if (row >= M) continue;
      const float ax = RND(da[h][i]), ay = RND(da[h][i + 1]);
      const float bx = RND(db[h][i]), by = RND(db[h][i + 1]);
      *reinterpret_cast<unsigned *>(O + (size_t)row * N + n0 + h * 8 + col) =
          as_u(__float22bfloat162_rn(make_float2(
              RND(ax / (1.f + __expf(-ax))) * bx,
              RND(ay / (1.f + __expf(-ay))) * by)));
    }
}

// ----------------------------------------------------------------- AdaLN ----
// sn = LN(s) * lnw ; g = sigmoid(sn @ wg^T + bg) ; out = g * (LN(a) + sn @ ws^T)
//
// LN(s) is shared by every column tile of the block, so it is computed once and
// kept in shared memory as bfloat16 -- exactly the value the eager path feeds to
// both projections. LN(a) needs the whole row for its statistics but only the
// block's own columns for the output, so only mean/rstd are kept.
//
// Both norms run in one pass with LPR lanes per row (every lane active, all BM
// rows in flight at once) because at these sizes the kernel is bound by the
// number of dependent memory round trips, not by arithmetic. The epilogue's own
// inputs are prefetched before the norms for the same reason.
template <int BM, int BN, int WM, int WN, int WK>
__global__ __launch_bounds__(32 * WM * WN * WK) void adaln_kernel(
    const bf16_t *__restrict__ A, const bf16_t *__restrict__ S,
    const bf16_t *__restrict__ LNW, const bf16_t *__restrict__ WG,
    const bf16_t *__restrict__ BG, const bf16_t *__restrict__ WS,
    bf16_t *__restrict__ O, int R, int K, int N, float epsS, float epsA) {
  constexpr int NW = WM * WN * WK;
  constexpr int NT = 32 * NW;
  constexpr int LPR = (NT / BM) > 32 ? 32 : (NT / BM);  // lanes per norm row
  constexpr int RPP = NT / LPR;                         // rows per norm pass
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int wm = warp % WM;
  const int wn = (warp / WM) % WN;
  const int wk = warp / (WM * WN);
  const int mb = blockIdx.y * BM;
  const int n0 = blockIdx.x * BN + wn * 16;

  extern __shared__ bf16_t smem[];
  bf16_t *xs = smem;                                      // BM x K, bf16
  float *red = reinterpret_cast<float *>(smem + BM * K);  // K-split partials
  __shared__ float amean[BM], arstd[BM];

  // Prefetch what the epilogue needs: a[row][col..col+1] and the gate bias.
  const int col = ACC_COL(lane);
  const int erow = mb + wm * 16 + ACC_ROW(lane, 0);
  unsigned av[2][2], bias[2];
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const int c = n0 + h * 8 + col;
    const bool ok = c < N;
    bias[h] = ok ? *reinterpret_cast<const unsigned *>(BG + c) : 0u;
    av[h][0] = (ok && erow < R)
                   ? *reinterpret_cast<const unsigned *>(A + (size_t)erow * N + c)
                   : 0u;
    av[h][1] = (ok && erow + 8 < R)
                   ? *reinterpret_cast<const unsigned *>(A + (size_t)(erow + 8) * N + c)
                   : 0u;
  }

  // --- both layer norms, one row group per LPR lanes
  const int vK = K >> 3, vN = N >> 3;
  const int rlane = tid & (LPR - 1);
  for (int r = tid / LPR; r < BM; r += RPP) {
    const int row = mb + r;
    const Vec8 *sp = reinterpret_cast<const Vec8 *>(S + (size_t)row * K);
    const Vec8 *ap = reinterpret_cast<const Vec8 *>(A + (size_t)row * N);
    float s1 = 0.f, s2 = 0.f, a1 = 0.f, a2 = 0.f;
    if (row < R) {
      for (int c = rlane; c < vK; c += LPR) {
        const Vec8 v = sp[c];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const float2 f = __bfloat1622float2(as_b2(v.w[j]));
          s1 += f.x + f.y;
          s2 = fmaf(f.x, f.x, fmaf(f.y, f.y, s2));
        }
      }
      for (int c = rlane; c < vN; c += LPR) {
        const Vec8 v = ap[c];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const float2 f = __bfloat1622float2(as_b2(v.w[j]));
          a1 += f.x + f.y;
          a2 = fmaf(f.x, f.x, fmaf(f.y, f.y, a2));
        }
      }
    }
#pragma unroll
    for (int o = LPR >> 1; o > 0; o >>= 1) {
      s1 += __shfl_xor_sync(FULLM, s1, o);
      s2 += __shfl_xor_sync(FULLM, s2, o);
      a1 += __shfl_xor_sync(FULLM, a1, o);
      a2 += __shfl_xor_sync(FULLM, a2, o);
    }
    const float smean = s1 / K;
    const float srstd = rsqrtf(s2 / K - smean * smean + epsS);
    if (rlane == 0) {
      const float m = a1 / N;
      amean[r] = m;
      arstd[r] = rsqrtf(a2 / N - m * m + epsA);
    }
    Vec8 *xp = reinterpret_cast<Vec8 *>(xs + (size_t)r * K);
    const Vec8 *wp = reinterpret_cast<const Vec8 *>(LNW);
    for (int c = rlane; c < vK; c += LPR) {
      Vec8 o;
      if (row < R) {
        const Vec8 v = sp[c], w = wp[c];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const float2 f = __bfloat1622float2(as_b2(v.w[j]));
          const float2 g = __bfloat1622float2(as_b2(w.w[j]));
          o.w[j] = as_u(__float22bfloat162_rn(make_float2(
              (f.x - smean) * srstd * g.x, (f.y - smean) * srstd * g.y)));
        }
      } else {
        o.w[0] = o.w[1] = o.w[2] = o.w[3] = 0u;
      }
      xp[c] = o;
    }
  }
  __syncthreads();

  float dg[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
  float dt[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
  if (n0 < N) {
    constexpr int KS = WK * 16;
    constexpr int KU = 4;  // see SwiGLU: K steps issued per iteration
    const bf16_t *px = xs + (size_t)(wm * 16 + (lane & 15)) * K + wk * 16 +
                       ((lane >> 4) << 3);
    const bf16_t *pg0 = b_base(WG, K, n0, lane) + wk * 16;
    const bf16_t *ps0 = b_base(WS, K, n0, lane) + wk * 16;
    const int nstep = (K - wk * 16 + KS - 1) / KS;
#pragma unroll 1
    for (int st = 0; st < nstep; st += KU) {
      unsigned a[KU][4], bg[KU][2][2], bs[KU][2][2];
#pragma unroll
      for (int u = 0; u < KU; ++u) {
        if (st + u < nstep) {
          ld_a_at_shared(a[u], px + u * KS);
          ld_b_at(bg[u][0], pg0 + u * KS);
          ld_b_at(bg[u][1], pg0 + 8 * K + u * KS);
          ld_b_at(bs[u][0], ps0 + u * KS);
          ld_b_at(bs[u][1], ps0 + 8 * K + u * KS);
        } else {
#pragma unroll
          for (int j = 0; j < 4; ++j) a[u][j] = 0u;
#pragma unroll
          for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int j = 0; j < 2; ++j) bg[u][h][j] = bs[u][h][j] = 0u;
        }
      }
      px += KU * KS;
      pg0 += KU * KS;
      ps0 += KU * KS;
#pragma unroll
      for (int u = 0; u < KU; ++u) {
        MMA_M16N8K16(dg[0], a[u], bg[u][0]);
        MMA_M16N8K16(dg[1], a[u], bg[u][1]);
        MMA_M16N8K16(dt[0], a[u], bs[u][0]);
        MMA_M16N8K16(dt[1], a[u], bs[u][1]);
      }
    }
  }

  if (WK > 1) {  // fold the K-split partials into the wk==0 warp's registers
    // One slot per (tile, wk>0, lane): 16 floats == four 128-bit accesses.
    // (A shared atomicAdd would be a CAS spin loop on this architecture.)
    float *slot = red + ((size_t)((wm * WN + wn) * (WK - 1)) * 32 + lane) * 16;
    if (wk != 0) {
      float *mine = slot + (size_t)(wk - 1) * 32 * 16;
#pragma unroll
      for (int h = 0; h < 2; ++h)
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          mine[h * 4 + i] = dg[h][i];
          mine[8 + h * 4 + i] = dt[h][i];
        }
    }
    __syncthreads();
    if (wk != 0) return;
#pragma unroll
    for (int q = 0; q < WK - 1; ++q) {
      const float *o = slot + (size_t)q * 32 * 16;
#pragma unroll
      for (int h = 0; h < 2; ++h)
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          dg[h][i] += o[h * 4 + i];
          dt[h][i] += o[8 + h * 4 + i];
        }
    }
  }
  if (n0 >= N) return;

  // Epilogue from registers: lane owns cols {c, c+1} of rows {r, r+8}.
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const float2 bs2 = __bfloat1622float2(as_b2(bias[h]));
#pragma unroll
    for (int i = 0; i < 4; i += 2) {
      const int rr = wm * 16 + ACC_ROW(lane, i), row = mb + rr;
      if (row >= R) continue;
      const float2 a2 = __bfloat1622float2(as_b2(av[h][i >> 1]));
      const float mu = amean[rr], rs = arstd[rr];
      const float gx = RND(1.f / (1.f + __expf(-RND(dg[h][i] + bs2.x))));
      const float gy = RND(1.f / (1.f + __expf(-RND(dg[h][i + 1] + bs2.y))));
      const float nx = RND((a2.x - mu) * rs), ny = RND((a2.y - mu) * rs);
      *reinterpret_cast<unsigned *>(O + (size_t)row * N + n0 + h * 8 + col) =
          as_u(__float22bfloat162_rn(make_float2(gx * RND(nx + RND(dt[h][i])),
                                                 gy * RND(ny + RND(dt[h][i + 1])))));
    }
  }
}

// ------------------------------------------------------------------ host ----
// Reference paths, used for anything the fused kernels do not cover (non-bf16,
// non-contiguous, K or N not a multiple of 16, a staging tile that will not fit
// in shared memory). They mirror the eager composition exactly.
static at::Tensor swiglu_ref(const at::Tensor &x, const at::Tensor &wa,
                             const at::Tensor &wb) {
  return at::silu(at::linear(x, wa)).mul_(at::linear(x, wb));
}

static at::Tensor adaln_ref(const at::Tensor &a, const at::Tensor &s,
                            const at::Tensor &lnw, const at::Tensor &wg,
                            const at::Tensor &bg, const at::Tensor &ws,
                            double epsS, double epsA) {
  const auto dt = s.scalar_type();
  const int64_t K = s.size(-1), N = a.size(-1);
  auto sn = at::layer_norm(s.to(at::kFloat), {K}, lnw.to(at::kFloat), {}, epsS)
                .to(dt);
  auto g = at::sigmoid(at::linear(sn, wg, bg));
  auto an = at::layer_norm(a.to(at::kFloat), {N}, {}, {}, epsA).to(dt);
  return g * (an + at::linear(sn, ws));
}

static const int SHM_MAX = 47 * 1024;  // stay inside the default per-block cap

// Shared memory a config needs: the AdaLN LN(s) tile plus, when K is split
// across warps, one 16-float-per-lane slot per (output tile, contributing warp).
#define RED_BYTES(BM, BN, WK) \
  (((WK) > 1) ? ((BM) / 16) * ((BN) / 16) * ((WK) - 1) * 32 * 16 * 4 : 0)

// Tile shapes were chosen by sweeping ~15 candidates per op over the captured
// shapes. The trade-off at these sizes: enough blocks to cover 148 SMs, enough
// warps per block to hide memory latency (every one of these kernels is
// latency-bound, not FLOP-bound), and few enough column blocks that work each of
// them repeats -- the weight slab, and AdaLN's two layer norms -- stays cheap.
at::Tensor swiglu(const at::Tensor &x, const at::Tensor &wa,
                  const at::Tensor &wb) {
  if (!(x.is_cuda() && x.scalar_type() == at::kBFloat16 &&
        wa.scalar_type() == at::kBFloat16 && wb.scalar_type() == at::kBFloat16 &&
        x.is_contiguous() && wa.is_contiguous() && wb.is_contiguous() &&
        wa.dim() == 2 && wb.dim() == 2 && x.dim() >= 1))
    return swiglu_ref(x, wa, wb);
  const int64_t K = x.size(-1), N = wa.size(0), M = x.numel() / (K ? K : 1);
  if (!(K > 0 && M > 0 && wa.size(1) == K && wb.size(1) == K &&
        wb.size(0) == N && K % 16 == 0 && N % 16 == 0))
    return swiglu_ref(x, wa, wb);

  const at::cuda::OptionalCUDAGuard device_guard(at::device_of(x));
  auto sizes = x.sizes().vec();
  sizes.back() = N;
  at::Tensor out = at::empty(sizes, x.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  const bf16_t *xp = reinterpret_cast<const bf16_t *>(x.data_ptr());
  const bf16_t *wap = reinterpret_cast<const bf16_t *>(wa.data_ptr());
  const bf16_t *wbp = reinterpret_cast<const bf16_t *>(wb.data_ptr());
  bf16_t *op = reinterpret_cast<bf16_t *>(out.data_ptr());
  // One warp per 16x16 output tile; K is split over enough warps that no warp
  // walks more than four K steps, which is where the loop stops being latency
  // bound (see the unrolled-by-KU loop in the kernel).
  dim3 grid((unsigned)((N + 15) / 16), (unsigned)((M + 15) / 16));
#define SW_LAUNCH(WK)                                                          \
  swiglu_kernel<16, 16, 1, 1, WK><<<grid, 32 * (WK), RED_BYTES(16, 16, WK),    \
                                    stream>>>(xp, wap, wbp, op, (int)M, (int)K, \
                                              (int)N)
  if (K >= 512)
    SW_LAUNCH(8);
  else if (K >= 256)
    SW_LAUNCH(4);
  else
    SW_LAUNCH(2);
#undef SW_LAUNCH
  return out;
}

at::Tensor adaln(const at::Tensor &a, const at::Tensor &s,
                 const at::Tensor &lnw, const at::Tensor &wg,
                 const at::Tensor &bg, const at::Tensor &ws, double epsS,
                 double epsA) {
  const int64_t K = s.size(-1), N = a.size(-1);
  // g broadcasts over a's leading dims, so the fast path needs a and s to agree
  // row for row: a's trailing dims must match s's, and a must have no extra
  // (non-unit) leading dims.
  const bool shapes_ok =
      a.dim() >= s.dim() && s.dim() >= 2 && a.numel() == s.numel() / K * N &&
      [&] {
        const int d = (int)s.dim();
        for (int i = 0; i < d - 1; ++i)
          if (a.size(a.dim() - d + i) != s.size(i)) return false;
        return true;
      }();
  if (!(a.is_cuda() && a.scalar_type() == at::kBFloat16 &&
        s.scalar_type() == at::kBFloat16 && lnw.scalar_type() == at::kBFloat16 &&
        wg.scalar_type() == at::kBFloat16 && bg.scalar_type() == at::kBFloat16 &&
        ws.scalar_type() == at::kBFloat16 && a.is_contiguous() &&
        s.is_contiguous() && lnw.is_contiguous() && wg.is_contiguous() &&
        bg.is_contiguous() && ws.is_contiguous() && shapes_ok &&
        K % 16 == 0 && N % 16 == 0 && wg.size(0) == N && wg.size(1) == K &&
        ws.size(0) == N && ws.size(1) == K && bg.numel() == N))
    return adaln_ref(a, s, lnw, wg, bg, ws, epsS, epsA);

  const int64_t R = s.numel() / K;
  const int xs_bytes = (int)(16 * K * 2);  // every config stages 16 rows
  const at::cuda::OptionalCUDAGuard device_guard(at::device_of(a));
  at::Tensor out = at::empty_like(a);
  auto stream = at::cuda::getCurrentCUDAStream();
  const bf16_t *ap = reinterpret_cast<const bf16_t *>(a.data_ptr());
  const bf16_t *sp = reinterpret_cast<const bf16_t *>(s.data_ptr());
  const bf16_t *lp = reinterpret_cast<const bf16_t *>(lnw.data_ptr());
  const bf16_t *gp = reinterpret_cast<const bf16_t *>(wg.data_ptr());
  const bf16_t *bp = reinterpret_cast<const bf16_t *>(bg.data_ptr());
  const bf16_t *wsp = reinterpret_cast<const bf16_t *>(ws.data_ptr());
  bf16_t *op = reinterpret_cast<bf16_t *>(out.data_ptr());

#define AD_LAUNCH(BN, WN, WK)                                                  \
  do {                                                                         \
    const int shm = xs_bytes + RED_BYTES(16, BN, WK);                          \
    if (shm <= SHM_MAX) {                                                      \
      dim3 grid((unsigned)((N + (BN) - 1) / (BN)), (unsigned)((R + 15) / 16));  \
      adaln_kernel<16, BN, 1, WN, WK><<<grid, 32 * (WN) * (WK), shm, stream>>>( \
          ap, sp, lp, gp, bp, wsp, op, (int)R, (int)K, (int)N, (float)epsS,     \
          (float)epsA);                                                        \
      return out;                                                              \
    }                                                                          \
  } while (0)

  if (R <= 64) {
    // Only a handful of output tiles exist, so split K eight ways for warps.
    AD_LAUNCH(16, 1, 8);
  } else if (R > 512 && N <= 128) {
    // Plenty of row blocks: one column block per row keeps both norms, and the
    // weight slab, read exactly once per block.
    AD_LAUNCH(128, 8, 1);
  } else {
    AD_LAUNCH(32, 2, 4);
  }
  AD_LAUNCH(16, 1, 4);  // narrowest staging tile, for long K
#undef AD_LAUNCH
  return adaln_ref(a, s, lnw, wg, bg, ws, epsS, epsA);
}
"""


_CPP_SRC = r"""
at::Tensor swiglu(const at::Tensor &x, const at::Tensor &wa, const at::Tensor &wb);
at::Tensor adaln(const at::Tensor &a, const at::Tensor &s, const at::Tensor &lnw,
                 const at::Tensor &wg, const at::Tensor &bg, const at::Tensor &ws,
                 double epsS, double epsA);
"""


def _local_arch() -> str | None:
    """The one arch to build for. The inherited TORCH_CUDA_ARCH_LIST spans
    sm_75..sm_120 here; building every one of them is slow, and the bfloat16
    mma this kernel issues does not exist below sm_80 anyway."""
    if os.environ.get("FASTKERNELS_CUDA_ARCH_LIST", "").strip():
        return os.environ["FASTKERNELS_CUDA_ARCH_LIST"]
    try:
        from ....infra.cuda_ext import _local_cuda_arch
        arch = _local_cuda_arch()
        if arch:
            return arch.split()[0]
    except Exception:
        pass
    try:
        major, minor = torch.cuda.get_device_capability()
        return f"{major}.{minor}" + ("a" if major in (9, 10, 12) else "")
    except Exception:
        return None


def _build():
    """JIT-compile the fused kernels; None if that is not possible here."""
    try:
        from torch.utils.cpp_extension import load_inline
        arch, prev = _local_arch(), os.environ.get("TORCH_CUDA_ARCH_LIST")
        if arch:
            os.environ["TORCH_CUDA_ARCH_LIST"] = arch
        # Source hash in the name: the build cache is shared across workspaces,
        # so a fixed name could collide with a different source of the same op.
        tag = hashlib.md5(_CUDA_SRC.encode()).hexdigest()[:10]
        try:
            return load_inline(
                name=f"fk_af3_swiglu_adaln_{tag}",
                cpp_sources=_CPP_SRC,
                cuda_sources=_CUDA_SRC,
                functions=["swiglu", "adaln"],
                with_pytorch_error_handling=False,
                extra_cuda_cflags=[
                    "-O3",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT162_OPERATORS__",
                    "--expt-relaxed-constexpr",
                    "--use_fast_math",
                ],
                verbose=False,
            )
        finally:
            if arch:
                if prev is None:
                    os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
                else:
                    os.environ["TORCH_CUDA_ARCH_LIST"] = prev
    except Exception as exc:  # noqa: BLE001 -- no CUDA / no nvcc: use eager
        print(f"[alphafold3_swiglu] fused kernel unavailable ({exc}); "
              f"falling back to eager", flush=True)
        return None


_C = _build()
_swiglu = getattr(_C, "swiglu", None)
_adaln = getattr(_C, "adaln", None)


class SwiGLU(nn.Module):
    """SwiGLU activation: SiLU(Wa x) * Wb x.

    Args:
        c_in: Number of input channels
        c_out: Number of output channels
    """

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.silu = SiLU()
        self.linear_a = Linear(c_in, c_out, bias=False)
        self.linear_b = Linear(c_in, c_out, bias=False)
        # Parameter handles hoisted out of the nn.Module attribute protocol:
        # ``self.linear_a.weight`` is two __getattr__ walks through _modules /
        # _parameters per call, which is a measurable share of the launch-bound
        # budget here. Held in a plain list so they are not re-registered as
        # parameters of this module (that would add state_dict keys). Storing
        # the Parameter objects -- not their .data -- keeps the handles valid
        # across ``.to(dtype/device)`` and ``load_state_dict``.
        self._w = [self.linear_a.weight, self.linear_b.weight]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self._w
        return _swiglu(x, w[0], w[1])


class AdaLN(nn.Module):
    """Adaptive Layer Normalization matching the reference AdaLN.

    Submodule structure matches checkpoint keys:
    - layer_norm_s: LayerNorm(c_s), weight-only
    - linear_g: Linear(c_s, c_a, bias=True) — gating
    - linear_s: Linear(c_s, c_a, bias=False) — additive conditioning

    Reference: openfold3/core/model/primitives/normalization.py AdaLN

    Args:
        c_a: Activation channel dimension
        c_s: Conditioning channel dimension
    """

    def __init__(self, c_a: int, c_s: int):
        super().__init__()
        self.c_a = c_a
        self.c_s = c_s

        self.layer_norm_a = LayerNorm(c_a, create_scale=False, create_offset=False)
        self.layer_norm_s = LayerNorm(c_s, create_offset=False)
        self.sigmoid = Sigmoid()
        self.linear_g = Linear(c_s, c_a, bias=True)
        self.linear_s = Linear(c_s, c_a, bias=False)
        # See SwiGLU._w: pre-resolved parameter handles + both eps values.
        self._w = [self.layer_norm_s.weight, self.linear_g.weight,
                   self.linear_g.bias, self.linear_s.weight,
                   self.layer_norm_s.eps, self.layer_norm_a.eps]

    def forward(self, a: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        w = self._w
        return _adaln(a, s, w[0], w[1], w[2], w[3], w[4], w[5])


if _swiglu is None:  # eager reference path (no CUDA extension available)
    def _swiglu_eager(self, x):
        return self.silu(self.linear_a(x)) * self.linear_b(x)

    def _adaln_eager(self, a, s):
        s_norm = self.layer_norm_s(s)
        g = self.sigmoid(self.linear_g(s_norm))
        a_norm = self.layer_norm_a(a)
        return g * (a_norm + self.linear_s(s_norm))

    SwiGLU.forward = _swiglu_eager
    AdaLN.forward = _adaln_eager
