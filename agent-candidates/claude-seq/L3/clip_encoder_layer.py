"""CLIP encoder layer (L3): one cooperative kernel, matched almost bit for bit.

The layer is pre-norm attention + QuickGELU MLP with two residuals, at
B=1, S=77, D=768, fp32 -- 1.1 GFLOP, which is a couple of microseconds of B200
tensor-core work against ~200 us measured for the baseline's ~25 eager kernels.
Almost all of the baseline is per-op overhead (an empty kernel costs ~3 us here,
back to back), so the obvious move is to fuse.  The obstacle is numerical, and it
is severe enough to be the whole story:

``torch.backends.cuda.matmul.fp32_precision`` is 'tf32' here, so every reference
matmul rounds its operands to TF32 and accumulates in fp32.  That rounding is a
quantizer, which makes the layer chaotically sensitive upstream -- a 6e-8
relative change in ``layer_norm1``'s output arrives at the layer output as 5e-5,
an 800x amplification, already 2.5x the comparison's budget.  Composing the
frozen L1/L2 winners (each of which passes standalone) therefore scores
matched=0.974 against the required 0.99.  Being *more* accurate than the
reference is worse still: an exact-fp32 layer differs from the TF32 reference by
~1.5e-4.

The way out is that the same quantizer absorbs small differences: perturb a TF32
GEMM's input by 3e-9 relative and its output is bit-identical.  So this
implementation reproduces the reference bit for bit rather than approximating it.
layer_norm1, q/k/v and the logits come out bit-identical; the one step that does
not is P@V, whose k residue cuBLAS handles in some order this file does not
reproduce, and the 8e-8 relative difference there is what leaves the layer at
matched ~0.999 instead of 1.0 -- a 60x margin on the 0.99 the comparison wants,
against 0.974 for the frozen-winner composition.  What the bit-exact route costs,
and why it is possible at all, is documented in the CUDA source: cuBLAS's TF32
GEMM at these shapes is ``mma.sync.m16n8k8`` with k accumulated in order into one
accumulator, ``F.layer_norm`` is its vectorized Welford kernel at (32,4) threads,
and ``F.softmax`` over 77 elements is
``softmax_warp_forward<...,7,false,false>``; all three are transcribed.

The eight stages run as one ``cudaLaunchCooperativeKernel`` with grid-wide
barriers between them; a per-stage launch path is kept for machines where a
cooperative launch of that size is not resident.

The module keeps the baseline's submodules -- and so its ``state_dict`` keys --
and falls back to the eager composition for any input the kernel does not cover
(non-fp32, batch > 1, a sequence length outside 65..80, a head dim other than 64,
a hidden or intermediate size the tiling does not divide, an oddly shaped mask,
or no working nvcc).
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
from transformers import CLIPTextConfig

from ..L1.layer_norm import LayerNorm
from ..L2.clip_attention import CLIPAttention
from ..L2.clip_mlp import CLIPMLP

_MROWS = 80   # padded row count (5 m16 mma tiles); must match the kernel
_HD = 64      # head dim the attention stage is specialized for


_CPP = r"""
torch::Tensor fk_pack_b(torch::Tensor W);
int64_t fk_ws_floats(int64_t D, int64_t I, int64_t H);
torch::Tensor fk_clip_layer(
    torch::Tensor x, torch::Tensor mask,
    torch::Tensor g1, torch::Tensor b1, torch::Tensor g2, torch::Tensor b2,
    torch::Tensor wqkv, torch::Tensor bqkv, torch::Tensor wo, torch::Tensor bo,
    torch::Tensor w1, torch::Tensor bf1, torch::Tensor w2, torch::Tensor bf2,
    torch::Tensor ws, double eps, int64_t num_heads);
"""

_CUDA = r"""
// Fused CLIP text-encoder layer (L3): fp32 in, TF32 on the tensor cores.
//
// Why this is a numerics problem before it is a throughput problem
// ---------------------------------------------------------------
// The captured shape (B=1, S=77, D=768, fp32) is 1.1 GFLOP -- a couple of
// microseconds of B200 tensor-core work against 187 us measured for the
// baseline's ~25 eager kernels.  So the speed half of the job is "stop
// launching kernels".  The hard half is matching the reference closely enough,
// and at this shape "closely" means bit-exactly.
//
// ``torch.backends.cuda.matmul.fp32_precision`` is 'tf32' here, so every
// reference matmul rounds its operands to TF32 (10 mantissa bits) and
// accumulates in fp32.  That rounding is a quantizer, and it makes the layer
// *chaotically* sensitive to upstream perturbation: feeding F.linear an input
// perturbed by one fp32 ulp (1e-7 relative) moves its output by 1e-5 relative,
// because ~1 operand in 4000 lands on the other side of a rounding boundary and
// jumps a whole 5e-4 quantum.  Measured end to end on this layer, a 6e-8
// relative change in layer_norm1's output arrives as 5e-5 at the layer output --
// an 800x amplification, and 2.5x the comparison's budget on its own.
// Composing the frozen L1/L2 winners (each of which passes standalone) lands at
// matched=0.974 against the required 0.99 for exactly this reason; it is not a
// defect in them, the error simply has nowhere to go but up.
//
// The one escape is that the same quantizer *absorbs*: perturb F.linear's input
// by 3e-9 relative and its output is bit-identical, because no operand changes
// after rounding.  So if every intermediate is reproduced bit for bit the whole
// chain is, and the comparison becomes free.  That is what this file does, and
// it is possible only because the reference's kernels turn out to be
// reproducible:
//
// * cuBLAS's TF32 GEMM at these shapes is ``mma.sync.m16n8k8`` with k
//   accumulated strictly in order into one accumulator per output tile, operands
//   rounded to nearest with ties to even.  A hand-written GEMM doing the same is
//   bit-identical on all four shapes this layer uses (M=77 x K/N in
//   {768, 2304, 3072}).  The price is that K may not be split and the
//   accumulator may not be partitioned, which is what fixes the tiling below.
// * ``F.layer_norm`` is ``vectorized_layer_norm_kernel``: 128 threads as (32,4),
//   vec_size 4, Welford in fp32.  ``ln_frag`` is a transcription of it down to
//   the shuffle offsets and the ``1.f/count`` reciprocals, because mean and rstd
//   have to come out bit-identical -- one ulp of rstd is 1e-7 relative on the
//   output, three orders of magnitude more than the next GEMM's rounding would
//   swallow.
// * ``F.softmax`` over 77 elements is ``softmax_warp_forward<...,7,false,false>``:
//   one row per warp, lane l holding elements l, l+32, l+64, l+96, max and sum
//   reduced by xor-shuffle butterflies.  Transcribed in ``attention``.
//
// The single exception is P@V (K=77).  cuBLAS's k residue handling there is not
// reproduced by any ordering tried (plain, and 2/3/4/5-way split-k); plain
// sequential k is the closest at 8e-8 relative.  That is affordable because of
// where it sits -- four orders of magnitude below out_proj's TF32 quantum, so it
// reaches the layer output as ~3e-6 against a ~1e-3 budget.
//
// Layout
// ------
// Every GEMM operand is staged in *mma fragment order* and pre-rounded to TF32
// by whichever stage produces it, so an operand fetch is one LDG.128 / LDG.64 of
// a fully coalesced run and the GEMM issues no conversions of its own (20 cvt
// per k-step would otherwise outnumber the 10 mma they feed).  Rounding at the
// producer is exact, not an approximation: these buffers are read only by the
// tensor cores, which would drop the low 13 mantissa bits anyway.
//
// The padded row count is 80 (five m16 tiles).  Rows 77..79 of every staged
// buffer are zeroed once, when the workspace is allocated, and never written
// again -- so the mma's row padding costs nothing per call.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include <algorithm>

#include <cooperative_groups.h>
#include <cuda_runtime.h>

namespace {

constexpr int MT = 5;           // m-tiles of 16 rows: 77 rows -> 80
constexpr int MROWS = MT * 16;  // 80
constexpr int HD = 64;          // head dim

// fc2's split-k factor.  fc2 is the last GEMM, so an accumulation order cuBLAS
// does not share costs ~7e-7 on the output instead of being amplified by a
// downstream TF32 rounding; see the file header.
constexpr int kKS3 = 4;


// Shared-memory row strides for the attention stage, chosen so the fragment each
// consumer reads lands in 32 distinct banks.  72 works for both of the k/v tile's
// access patterns -- row-major by (g, t) for QK^T and transposed, by (t, g), for
// P@V -- which is what lets them share one buffer.
constexpr int LDKV = 72;
constexpr int LDP = 100;

// fp32 -> TF32, round to nearest with ties to even -- the rounding cuBLAS
// applies on the way into the tensor cores.  mma.sync's own conversion
// *truncates*, which is biased and drifts from the reference by several times
// the allowed tolerance, so operands are rounded here instead, once, when they
// are staged.
__device__ __forceinline__ unsigned tf32_rn(float x) {
  unsigned r;
  asm("cvt.rn.satfinite.tf32.f32 %0, %1;" : "=r"(r) : "f"(x));
  return r;
}

__device__ __forceinline__ float tf32_rn_f(float x) {
  return __int_as_float((int)tf32_rn(x));
}

// D[16x8] += A[16x8] * B[8x8], one warp, TF32 in / fp32 accumulate.
// Per-lane fragments (g = lane/4, t = lane%4):
//   a0=A[g][t]  a1=A[g+8][t]  a2=A[g][t+4]  a3=A[g+8][t+4]
//   b0=B[t][g]  b1=B[t+4][g]        (B indexed [k][n])
//   d0=D[g][2t] d1=D[g][2t+1] d2=D[g+8][2t] d3=D[g+8][2t+1]
__device__ __forceinline__ void mma_m16n8k8(float *d, const unsigned *a, const unsigned *b) {
  asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// ---------------------------------------------------------------------------
// Staging layouts.
//
// A operand (activations), K columns over MROWS rows:
//   AF[((k/8)*MT + m/16)*128 + lane*4 + j]
// so a warp's entire A fragment for one (m-tile, k-step) is one LDG.128.
//
// B operand (weights) from an [N, K] row-major source:
//   BP[((n/8)*(K/8) + k/8)*64 + lane*2 + j]
// one LDG.64 per fragment, and a whole column strip's weights are one linear
// run -- which matters because the weights are the only DRAM-resident operand
// (28 MB per call) and are read exactly once.
// ---------------------------------------------------------------------------
__device__ __forceinline__ int af_index(int m, int k) {
  int mt = m >> 4, r = m & 15;
  int g = r & 7, half = r >> 3;
  int ks = k >> 3, kk = k & 7;
  return ((ks * MT + mt) * 32 + g * 4 + (kk & 3)) * 4 + half + ((kk >> 2) << 1);
}

enum Epilogue { EPI_QKV = 0, EPI_ADD_ROW = 1, EPI_GELU_FRAG = 2 };

__global__ void pack_b(const float *__restrict__ W, float *__restrict__ BP, int K, long total) {
  long o = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (o >= total) return;
  int j = (int)(o & 1);
  int lane = (int)((o >> 1) & 31);
  long nk = K >> 3;
  int ks = (int)((o >> 6) % nk);
  long nt = (o >> 6) / nk;
  long n = nt * 8 + (lane >> 2);
  int k = ks * 8 + (lane & 3) + 4 * j;
  BP[o] = tf32_rn_f(W[n * K + k]);
}

// ---------------------------------------------------------------------------
// LayerNorm -- a transcription of ATen's vectorized_layer_norm_kernel.  The
// launch geometry (128 threads as (32,4), vec_size 4) and every reduction step
// have to match so that mean and rstd are bit-identical; see the file header.
// The normalized row is written straight into A-fragment order, TF32-rounded.
// ---------------------------------------------------------------------------
struct WD {
  float mean, sigma2, count;
};

__device__ __forceinline__ WD welford_online(float val, WD c) {
  float delta = val - c.mean;
  float new_count = c.count + 1.f;
  float new_mean = c.mean + delta * (1.f / new_count);
  return {new_mean, c.sigma2 + delta * (val - new_mean), new_count};
}

// Argument order mirrors ATen's cuWelfordCombine(dataB, dataA).
__device__ __forceinline__ WD welford_combine(WD B, WD A) {
  float delta = B.mean - A.mean;
  float count = A.count + B.count;
  float mean, sigma2;
  if (count > 0.f) {
    float coef = 1.f / count;
    float nA = A.count * coef;
    float nB = B.count * coef;
    mean = nA * A.mean + nB * B.mean;
    sigma2 = A.sigma2 + B.sigma2 + delta * delta * A.count * nB;
  } else {
    mean = 0.f;
    sigma2 = 0.f;
  }
  return {mean, sigma2, count};
}

// One normalized row, by the whole 128-thread block.  ``buf`` (6 floats) is the
// caller's scratch, so the fused kernel can carve it out of its own allocation.
__device__ __forceinline__ void ln_row(
    const float *__restrict__ X, const float *__restrict__ gamma,
    const float *__restrict__ beta, float eps, float *__restrict__ AF, int N, int row,
    float *buf) {
  const float4 *Xv = reinterpret_cast<const float4 *>(X + (long)row * N);
  constexpr int numx = 128;        // ATen launches (32, 4); the geometry and the
  constexpr int ydim = numx / 32;   // reduction order below both have to match it.
  const int lane = threadIdx.x & 31, ywarp = (threadIdx.x >> 5) & (ydim - 1);
  const int thrx = lane + ywarp * 32;
  const int nvec = N / 4;

  // All three cold streams are issued before any of them is consumed: the row,
  // gamma and beta are each ~3 KB and, read one after another, their L2 latency
  // was most of this stage's time.
  constexpr int MAXIT = 4;
  float4 xv[MAXIT], gv[MAXIT], bv[MAXIT];
  const float4 *gp = reinterpret_cast<const float4 *>(gamma);
  const float4 *bp = reinterpret_cast<const float4 *>(beta);
  const int nit = (nvec - thrx + numx - 1) / numx;
#pragma unroll
  for (int it = 0; it < MAXIT; ++it) {
    if (it < nit) {
      const int i = thrx + it * numx;
      xv[it] = Xv[i];
      gv[it] = gp[i];
      bv[it] = bp[i];
    }
  }

  WD wd{0.f, 0.f, 0.f};
#pragma unroll
  for (int it = 0; it < MAXIT; ++it) {
    if (it < nit) {
      wd = welford_online(xv[it].x, wd);
      wd = welford_online(xv[it].y, wd);
      wd = welford_online(xv[it].z, wd);
      wd = welford_online(xv[it].w, wd);
    }
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    WD b{__shfl_down_sync(0xffffffffu, wd.mean, off),
         __shfl_down_sync(0xffffffffu, wd.sigma2, off),
         __shfl_down_sync(0xffffffffu, wd.count, off)};
    wd = welford_combine(wd, b);
  }
  float *msb = buf;
  float *cb = buf + ydim;
#pragma unroll
  for (int off = ydim / 2; off > 0; off /= 2) {
    if (lane == 0 && ywarp >= off && ywarp < 2 * off) {
      int wy = ywarp - off;
      msb[2 * wy] = wd.mean;
      msb[2 * wy + 1] = wd.sigma2;
      cb[wy] = wd.count;
    }
    __syncthreads();
    if (lane == 0 && ywarp < off) {
      WD b{msb[2 * ywarp], msb[2 * ywarp + 1], cb[ywarp]};
      wd = welford_combine(wd, b);
    }
    __syncthreads();
  }
  if (thrx == 0) {
    msb[0] = wd.mean;
    msb[1] = wd.sigma2 / (float)N;
  }
  __syncthreads();
  const float mean = msb[0];
  const float rstd = rsqrtf(msb[1] + eps);
  __syncthreads();  // every warp has read buf; the next row may overwrite it

#pragma unroll
  for (int it = 0; it < MAXIT; ++it) {
    if (it >= nit) continue;
    const int i = thrx + it * numx;
    const float4 d = xv[it], g4 = gv[it], b4 = bv[it];
    // The four columns of one float4 are k = 4i .. 4i+3, i.e. one t-quad of a
    // single fragment slot, so they land 4 floats apart in fragment order.
    const int base = af_index(row, 4 * i);
    AF[base + 0] = tf32_rn_f(g4.x * (rstd * (d.x - mean)) + b4.x);
    AF[base + 4] = tf32_rn_f(g4.y * (rstd * (d.y - mean)) + b4.y);
    AF[base + 8] = tf32_rn_f(g4.z * (rstd * (d.z - mean)) + b4.z);
    AF[base + 12] = tf32_rn_f(g4.w * (rstd * (d.w - mean)) + b4.w);
  }
}

__global__ __launch_bounds__(128) void ln_frag(
    const float *__restrict__ X, const float *__restrict__ gamma,
    const float *__restrict__ beta, float eps, float *__restrict__ AF, int N) {
  __shared__ float buf[6];  // ydim * 3/2 floats, as ATen sizes it
  ln_row(X, gamma, beta, eps, AF, N, blockIdx.x, buf);
}

// Shared epilogue for both GEMM variants.  With KS > 1 the accumulators are raw
// partial sums and ``reduce_epi`` adds the slices in order and applies bias /
// residual; only fc2 uses that path, because it is the last GEMM -- an
// accumulation order cuBLAS does not share costs ~7e-7 on the output there
// instead of being amplified by a downstream TF32 rounding.
template <int MTW, int NT>
__device__ __forceinline__ void gemm_epilogue(
    float acc[MTW][NT][4], const float *__restrict__ bias, float *__restrict__ Out,
    const float *__restrict__ resid, float *__restrict__ qkv, int nt_first, int mt0, int N,
    int M, int epi, int KS, int slice, int g, int t) {
  const int PD = N / 3;  // per-projection width, only meaningful for EPI_QKV
#pragma unroll
  for (int i = 0; i < NT; ++i) {
    const int n0 = (nt_first + i) * 8;
    const float bA = (KS > 1) ? 0.f : bias[n0 + 2 * t];
    const float bB = (KS > 1) ? 0.f : bias[n0 + 2 * t + 1];
#pragma unroll
    for (int m = 0; m < MTW; ++m) {
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int row = (mt0 + m) * 16 + g + 8 * h;
        if (row >= M) continue;
#pragma unroll
        for (int jj = 0; jj < 2; ++jj) {
          const int col = n0 + 2 * t + jj;
          const float v = acc[m][i][h * 2 + jj] + (jj ? bB : bA);
          if (KS > 1) {
            Out[((long)slice * MROWS + row) * N + col] = v;
          } else if (epi == EPI_QKV) {
            // n -> (projection, head, head-dim); 8 divides 64, so an n-tile
            // never straddles a head boundary.
            int proj = col / PD, rem = col - proj * PD;
            qkv[(((long)proj * (PD / HD) + (rem >> 6)) * MROWS + row) * HD + (rem & 63)] =
                tf32_rn_f(v);
          } else if (epi == EPI_ADD_ROW) {
            Out[(long)row * N + col] = v + resid[(long)row * N + col];
          } else {  // EPI_GELU_FRAG: x * sigmoid(1.702 x), staged for fc2
            Out[af_index(row, col)] = tf32_rn_f(v * (1.f / (1.f + expf(-(1.702f * v)))));
          }
        }
      }
    }
  }
}

template <int MTW, int NT, int P>
__device__ __forceinline__ void gemm_warp(
    const float *__restrict__ AF, const float *__restrict__ BP,
    const float *__restrict__ bias, float *__restrict__ Out,
    const float *__restrict__ resid, float *__restrict__ qkv,
    int K, int N, int M, int epi, int KS, int ntb, int mt0, int slice) {
  const int lane = threadIdx.x & 31;
  const int g = lane >> 2, t = lane & 3;
  const int nk = K >> 3;
  const int nks = nk / KS;              // k-steps in this slice

  float acc[MTW][NT][4];
#pragma unroll
  for (int m = 0; m < MTW; ++m)
#pragma unroll
    for (int i = 0; i < NT; ++i)
#pragma unroll
      for (int j = 0; j < 4; ++j) acc[m][i][j] = 0.f;

  const float *ap = AF + (long)(nks * slice) * (MT * 128) + mt0 * 128 + lane * 4;
  const float *bp = BP + ((long)ntb * nk + nks * slice) * 64 + lane * 2;

  unsigned a[P][MTW][4], b[P][NT][2];
  // The loop variables are named _fm/_fn rather than m/i: the macro is also
  // invoked with the prologue's own induction variable as both arguments, and a
  // plainly-named inner loop variable would shadow it.
#define FK_PF(slot, kstep)                                                             \
  {                                                                                    \
    const long _fk = (long)(kstep);                                                    \
    const int _fs = (slot);                                                            \
    _Pragma("unroll") for (int _fm = 0; _fm < MTW; ++_fm) {                            \
      float4 f = *reinterpret_cast<const float4 *>(ap + _fk * (MT * 128) + _fm * 128);  \
      a[_fs][_fm][0] = __float_as_uint(f.x);                                           \
      a[_fs][_fm][1] = __float_as_uint(f.y);                                           \
      a[_fs][_fm][2] = __float_as_uint(f.z);                                           \
      a[_fs][_fm][3] = __float_as_uint(f.w);                                           \
    }                                                                                  \
    _Pragma("unroll") for (int _fn = 0; _fn < NT; ++_fn) {                             \
      float2 fb = *reinterpret_cast<const float2 *>(bp + ((long)_fn * nk + _fk) * 64);  \
      b[_fs][_fn][0] = __float_as_uint(fb.x);                                          \
      b[_fs][_fn][1] = __float_as_uint(fb.y);                                          \
    }                                                                                  \
  }
#pragma unroll
  for (int pf = 0; pf < P; ++pf)
    if (pf < nks) FK_PF(pf, pf)

  // The slot index has to be a compile-time constant or the buffer lands in local
  // memory and the prefetch does nothing, so the k loop steps by P with the slot
  // as the unrolled inner index rather than being ``ks % P``.  ``nks`` is a
  // multiple of P by construction (the caller checks it).
  for (int kb = 0; kb < nks; kb += P) {
#pragma unroll
    for (int j = 0; j < P; ++j) {
#pragma unroll
      for (int i = 0; i < NT; ++i)
#pragma unroll
        for (int m = 0; m < MTW; ++m) mma_m16n8k8(acc[m][i], a[j][m], b[j][i]);
      if (kb + P + j < nks) FK_PF(j, kb + P + j)
    }
  }
#undef FK_PF

  gemm_epilogue<MTW, NT>(acc, bias, Out, resid, qkv, ntb, mt0, N, M, epi, KS, slice, g, t);
}

template <int MTW, int NT, int P>
__global__ __launch_bounds__(32) void gemm_direct(
    const float *__restrict__ AF, const float *__restrict__ BP,
    const float *__restrict__ bias, float *__restrict__ Out,
    const float *__restrict__ resid, float *__restrict__ qkv,
    int K, int N, int M, int epi, int KS) {
  gemm_warp<MTW, NT, P>(AF, BP, bias, Out, resid, qkv, K, N, M, epi, KS,
                        blockIdx.x * NT, blockIdx.y * MTW, blockIdx.z);
}

// Sum a split-k GEMM's slices in slice order, then bias + residual.
__global__ void reduce_epi(const float *__restrict__ part, const float *__restrict__ bias,
                           const float *__restrict__ resid, float *__restrict__ Out,
                           int N, int M, int KS) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M * N) return;
  int row = i / N, col = i - row * N;
  float v = part[(long)row * N + col];
  for (int s = 1; s < KS; ++s) v += part[((long)s * MROWS + row) * N + col];
  Out[i] = v + bias[col] + resid[i];
}

// ---------------------------------------------------------------------------
// Attention: one block per (head, 16-row band).  QK^T, the scale, the mask add,
// the softmax and P@V all happen here, so q/k/v never leave the chip between
// them, and the result is written in out_proj's A-fragment order -- which is
// also the layout the baseline builds with transpose(1,2).contiguous().
// ---------------------------------------------------------------------------
template <int NWARP>
__device__ __forceinline__ void attn_task(
    const float *__restrict__ qkv, const float *__restrict__ mask, int mask_head_stride,
    float *__restrict__ AFout, float scale, int S, int H, int task, float *sm) {
  float *qs = sm;                      // [16][LDKV]
  float *kv = qs + 16 * LDKV;          // [MROWS][LDKV]: k, then v over the same space
  float *ps = kv + MROWS * LDKV;       // [16][LDP]

  const int head = task / MT;
  const int band = task - head * MT;
  const int tid = threadIdx.x;
  __syncthreads();  // the shared tiles may still be in use by a previous task
  const long hoff = (long)head * MROWS * HD;
  const long plane = (long)H * MROWS * HD;
  const float *qh = qkv + hoff + (long)band * 16 * HD;
  const float *kh = qkv + plane + hoff;
  const float *vh = qkv + 2 * plane + hoff;

  for (int i = tid; i < 16 * HD / 4; i += NWARP * 32) {
    int r = i >> 4, c = (i & 15) << 2;
    *reinterpret_cast<float4 *>(qs + r * LDKV + c) =
        *reinterpret_cast<const float4 *>(qh + r * HD + c);
  }
  for (int i = tid; i < MROWS * HD / 4; i += NWARP * 32) {
    int r = i >> 4, c = (i & 15) << 2;
    *reinterpret_cast<float4 *>(kv + r * LDKV + c) =
        *reinterpret_cast<const float4 *>(kh + r * HD + c);
  }
  // Key padding: columns S..MROWS-1 of P are read by P@V's k loop.
  for (int i = tid; i < 16 * (MROWS - S); i += NWARP * 32) ps[(i / (MROWS - S)) * LDP + S + i % (MROWS - S)] = 0.f;
  __syncthreads();

  const int warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, t = lane & 3;
  const int nkt = (S + 7) / 8;  // key tiles

  // QK^T -> scale -> mask.
  //
  // A warp keeps all of its key tiles live at once and shares one q fragment
  // between them.  Walking them one at a time instead leaves a single
  // accumulator chain, and at 21.5 cycles of mma dependent latency that is three
  // quarters idle -- the same reason the GEMM owns several tiles per warp.
  constexpr int QT = (MROWS / 8 + NWARP - 1) / NWARP;
  float qacc[QT][4];
#pragma unroll
  for (int j = 0; j < QT; ++j)
#pragma unroll
    for (int i = 0; i < 4; ++i) qacc[j][i] = 0.f;
#pragma unroll
  for (int ks = 0; ks < HD / 8; ++ks) {
    const int k0 = ks * 8;
    const unsigned a[4] = {__float_as_uint(qs[g * LDKV + k0 + t]),
                           __float_as_uint(qs[(g + 8) * LDKV + k0 + t]),
                           __float_as_uint(qs[g * LDKV + k0 + t + 4]),
                           __float_as_uint(qs[(g + 8) * LDKV + k0 + t + 4])};
#pragma unroll
    for (int j = 0; j < QT; ++j) {
      const int nti = warp + j * NWARP;
      if (nti >= nkt) continue;
      const unsigned b[2] = {__float_as_uint(kv[(nti * 8 + g) * LDKV + k0 + t]),
                             __float_as_uint(kv[(nti * 8 + g) * LDKV + k0 + t + 4])};
      mma_m16n8k8(qacc[j], a, b);
    }
  }
#pragma unroll
  for (int j = 0; j < QT; ++j) {
    const int nti = warp + j * NWARP;
    if (nti >= nkt) continue;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int r = g + 8 * h;
#pragma unroll
      for (int jj = 0; jj < 2; ++jj) {
        const int c = nti * 8 + 2 * t + jj;
        if (c >= S) continue;
        float v = qacc[j][h * 2 + jj] * scale;
        if (mask) v += mask[(long)head * mask_head_stride + (long)(band * 16 + r) * S + c];
        ps[r * LDP + c] = v;
      }
    }
  }
  __syncthreads();  // every warp is done reading k

  for (int i = tid; i < MROWS * HD / 4; i += NWARP * 32) {
    int r = i >> 4, c = (i & 15) << 2;
    *reinterpret_cast<float4 *>(kv + r * LDKV + c) =
        *reinterpret_cast<const float4 *>(vh + r * HD + c);
  }

  // softmax, transcribing softmax_warp_forward<float,float,float,7,false,false>.
  // All of a warp's rows are reduced in one butterfly (ATen's WARP_BATCH does the
  // same), so the shuffle latency is paid once instead of once per row.
  constexpr int RB = 16 / NWARP;
  float e[RB][4], mx[RB], sum[RB];
#pragma unroll
  for (int r = 0; r < RB; ++r) {
    const int row = warp * RB + r;
#pragma unroll
    for (int it = 0; it < 4; ++it) {
      const int idx = lane + 32 * it;
      e[r][it] = (idx < S) ? ps[row * LDP + idx] : -INFINITY;
    }
    mx[r] = e[r][0];
#pragma unroll
    for (int it = 0; it < 4; ++it) mx[r] = mx[r] > e[r][it] ? mx[r] : e[r][it];
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
#pragma unroll
    for (int r = 0; r < RB; ++r) {
      const float b = __shfl_xor_sync(0xffffffffu, mx[r], off);
      mx[r] = mx[r] < b ? b : mx[r];
    }
#pragma unroll
  for (int r = 0; r < RB; ++r) {
    sum[r] = 0.f;
#pragma unroll
    for (int it = 0; it < 4; ++it) {
      e[r][it] = expf(e[r][it] - mx[r]);
      sum[r] += e[r][it];
    }
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
#pragma unroll
    for (int r = 0; r < RB; ++r) sum[r] += __shfl_xor_sync(0xffffffffu, sum[r], off);
#pragma unroll
  for (int r = 0; r < RB; ++r) {
    const int row = warp * RB + r;
#pragma unroll
    for (int it = 0; it < 4; ++it) {
      const int idx = lane + 32 * it;
      if (idx < S) ps[row * LDP + idx] = tf32_rn_f(e[r][it] / sum[r]);
    }
  }
  __syncthreads();

  // P@V -> out_proj's A-fragment order
  constexpr int PT = (HD / 8 + NWARP - 1) / NWARP;
  float pacc[PT][4];
#pragma unroll
  for (int j = 0; j < PT; ++j)
#pragma unroll
    for (int i = 0; i < 4; ++i) pacc[j][i] = 0.f;
#pragma unroll
  for (int ks = 0; ks < MROWS / 8; ++ks) {
    const int k0 = ks * 8;
    const unsigned a[4] = {__float_as_uint(ps[g * LDP + k0 + t]),
                           __float_as_uint(ps[(g + 8) * LDP + k0 + t]),
                           __float_as_uint(ps[g * LDP + k0 + t + 4]),
                           __float_as_uint(ps[(g + 8) * LDP + k0 + t + 4])};
#pragma unroll
    for (int j = 0; j < PT; ++j) {
      const int nti = warp + j * NWARP;
      const unsigned b[2] = {__float_as_uint(kv[(k0 + t) * LDKV + nti * 8 + g]),
                             __float_as_uint(kv[(k0 + t + 4) * LDKV + nti * 8 + g])};
      mma_m16n8k8(pacc[j], a, b);
    }
  }
#pragma unroll
  for (int j = 0; j < PT; ++j) {
    const int nti = warp + j * NWARP;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int sq = band * 16 + g + 8 * h;
      if (sq >= S) continue;
#pragma unroll
      for (int jj = 0; jj < 2; ++jj)
        AFout[af_index(sq, head * HD + nti * 8 + 2 * t + jj)] = tf32_rn_f(pacc[j][h * 2 + jj]);
    }
  }
}

template <int NWARP>
__global__ __launch_bounds__(NWARP * 32) void attention(
    const float *__restrict__ qkv, const float *__restrict__ mask, int mask_head_stride,
    float *__restrict__ AFout, float scale, int S, int H) {
  extern __shared__ float sm[];
  attn_task<NWARP>(qkv, mask, mask_head_stride, AFout, scale, S, H, blockIdx.x, sm);
}


// ---------------------------------------------------------------------------
// The whole layer in one cooperative kernel.
//
// On this machine a kernel launch costs ~2.8 us of GPU-side gap -- an empty
// kernel measures 3.4 us back to back -- which is why the eager baseline spends
// ~70 of its 200 us doing nothing.  Eight kernels would hand ~22 us of that back;
// one kernel with grid-wide barriers between the stages costs a fraction of it.
//
// Every stage is the same device function the standalone kernels use, mapped
// onto (blockIdx, warp) block-major so a stage with few tasks (out_proj has 480,
// against 592 warps) still spreads one warp per SM rather than packing four onto
// one.
// ---------------------------------------------------------------------------
template <int MTW, int NT, int P, int NWARP>
__device__ __forceinline__ void gemm_stage(
    const float *__restrict__ AF, const float *__restrict__ BP,
    const float *__restrict__ bias, float *__restrict__ Out,
    const float *__restrict__ resid, float *__restrict__ qkv, int K, int N, int M, int epi,
    int KS, int warp) {
  constexpr int nmg = MT / MTW;
  const int ntt = N / (8 * NT);
  const int ntasks = ntt * nmg * KS;
  const int G = gridDim.x;
  for (int task = blockIdx.x + warp * G; task < ntasks; task += G * NWARP) {
    const int nt = task % ntt;
    const int r = task / ntt;
    gemm_warp<MTW, NT, P>(AF, BP, bias, Out, resid, qkv, K, N, M, epi, KS, nt * NT,
                          (r % nmg) * MTW, r / nmg);
  }
}

// The second launch bound matters: a cooperative grid can only be as large as the
// resident block count, so the register budget is pinned to three blocks per SM
// rather than left to whatever the allocator happens to pick -- it drifted between
// two and four blocks/SM across edits, and the grid size with it.  Asking for four
// is not free: it caps registers at 128 and the k-prefetch buffers start spilling
// (248 B of local per thread, and 2x the runtime).
template <int MTW, int NT, int P, int NWARP>
__global__ __launch_bounds__(NWARP * 32, 3) void fused_layer(
    const float *__restrict__ x, const float *__restrict__ mask, int mask_hs,
    const float *__restrict__ g1, const float *__restrict__ b1,
    const float *__restrict__ g2, const float *__restrict__ b2,
    const float *__restrict__ wqkv, const float *__restrict__ bqkv,
    const float *__restrict__ wo, const float *__restrict__ bo,
    const float *__restrict__ w1, const float *__restrict__ bf1,
    const float *__restrict__ w2, const float *__restrict__ bf2,
    float *__restrict__ ws, float *__restrict__ out, float eps, int S, int D, int I, int H,
    int KS3, float scale) {
  extern __shared__ float sm[];
  auto grid = cooperative_groups::this_grid();
  const int warp = threadIdx.x >> 5;
  const int G = gridDim.x;

  float *af_h1 = ws;
  float *qkvbuf = af_h1 + (long)D * MROWS;
  float *af_attn = qkvbuf + 3L * H * MROWS * HD;
  float *resid1 = af_attn + (long)D * MROWS;
  float *af_h2 = resid1 + (long)D * MROWS;
  float *af_act = af_h2 + (long)D * MROWS;
  float *part = af_act + (long)I * MROWS;

  for (int row = blockIdx.x; row < S; row += G) ln_row(x, g1, b1, eps, af_h1, D, row, sm);
  grid.sync();
  gemm_stage<MTW, NT, P, NWARP>(af_h1, wqkv, bqkv, nullptr, nullptr, qkvbuf, D, 3 * D, S,
                                EPI_QKV, 1, warp);
  grid.sync();
  for (int t = blockIdx.x; t < H * MT; t += G)
    attn_task<NWARP>(qkvbuf, mask, mask_hs, af_attn, scale, S, H, t, sm);
  grid.sync();
  gemm_stage<MTW, NT, P, NWARP>(af_attn, wo, bo, resid1, x, nullptr, D, D, S, EPI_ADD_ROW,
                                1, warp);
  grid.sync();
  for (int row = blockIdx.x; row < S; row += G)
    ln_row(resid1, g2, b2, eps, af_h2, D, row, sm);
  grid.sync();
  gemm_stage<MTW, NT, P, NWARP>(af_h2, w1, bf1, af_act, nullptr, nullptr, D, I, S,
                                EPI_GELU_FRAG, 1, warp);
  grid.sync();
  // fc2 is the one stage where two n-tiles per warp pays: it is K = 3072, so even
  // split six ways its k walk is long enough that halving the number of warps
  // costs less than the extra accumulator chain buys.
  gemm_stage<MTW, 2, P, NWARP>(af_act, w2, bf2, kKS3 > 1 ? part : out, resid1, nullptr, I, D,
                               S, EPI_ADD_ROW, KS3, warp);
  if (KS3 > 1) {
    grid.sync();
    const int nthr = NWARP * 32;
    for (int i = blockIdx.x * nthr + threadIdx.x; i < S * D; i += G * nthr) {
      const int row = i / D, col = i - row * D;
      float v = part[(long)row * D + col];
      for (int sl = 1; sl < KS3; ++sl) v += part[((long)sl * MROWS + row) * D + col];
      out[i] = v + bf2[col] + resid1[i];
    }
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// Host entry points
// ---------------------------------------------------------------------------

namespace {

// Attention's shared tiles are the largest per-block allocation in the fused
// kernel, and they set its occupancy (and so the cooperative grid).
constexpr int kFusedSmem = (16 * LDKV + MROWS * LDKV + 16 * LDP) * (int)sizeof(float);

template <int MTW, int NT, int P, int NWARP>
int fused_grid(int maxtasks) {
  int per_sm = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &per_sm, (void *)fused_layer<MTW, NT, P, NWARP>, NWARP * 32, kFusedSmem);
  int nsm = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  int cap = per_sm * nsm;
  int want = (maxtasks + NWARP - 1) / NWARP;
  return cap <= 0 ? 0 : (want < cap ? want : cap);
}

}  // namespace

// Floats the persistent workspace needs: five staged activations, q/k/v, and
// fc2's split-k partials.
int64_t fk_ws_floats(int64_t D, int64_t I, int64_t H) {
  return (4 * D + I) * MROWS + 3 * H * MROWS * HD + (int64_t)kKS3 * MROWS * D;
}

torch::Tensor fk_pack_b(torch::Tensor W) {
  TORCH_CHECK(W.is_cuda() && W.scalar_type() == torch::kFloat && W.dim() == 2);
  auto Wc = W.contiguous();
  const int N = (int)Wc.size(0), K = (int)Wc.size(1);
  TORCH_CHECK(N % 8 == 0 && K % 8 == 0, "pack_b needs N,K multiples of 8");
  auto out = torch::empty({(long)N * K}, Wc.options());
  const long total = (long)N * K;
  const int thr = 256;
  pack_b<<<(total + thr - 1) / thr, thr, 0, at::cuda::getCurrentCUDAStream()>>>(
      Wc.data_ptr<float>(), out.data_ptr<float>(), K, total);
  return out;
}

torch::Tensor fk_clip_layer(

    torch::Tensor x, torch::Tensor mask,
    torch::Tensor g1, torch::Tensor b1, torch::Tensor g2, torch::Tensor b2,
    torch::Tensor wqkv, torch::Tensor bqkv, torch::Tensor wo, torch::Tensor bo,
    torch::Tensor w1, torch::Tensor bf1, torch::Tensor w2, torch::Tensor bf2,
    torch::Tensor ws, double eps, int64_t num_heads) {
  const at::cuda::CUDAGuard guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  const int S = (int)x.size(1), D = (int)x.size(2);
  const int I = (int)bf1.size(0);
  const int H = (int)num_heads;

  float *w = ws.data_ptr<float>();
  float *af_h1 = w;                            // [K=D]  fragment order
  float *qkvbuf = af_h1 + (long)D * MROWS;     // [3][H][MROWS][HD]
  float *af_attn = qkvbuf + 3L * H * MROWS * HD;
  float *resid1 = af_attn + (long)D * MROWS;   // [MROWS][D] row-major
  float *af_h2 = resid1 + (long)D * MROWS;
  float *af_act = af_h2 + (long)D * MROWS;     // [K=I] fragment order
  float *part = af_act + (long)I * MROWS;      // split-k partials for fc2

  auto out = torch::empty({1, S, D}, x.options());
  // "no mask" arrives as an empty tensor (Python cannot hand over an undefined
  // one), so it is numel, not defined(), that distinguishes it.
  const bool has_mask = mask.defined() && mask.numel() > 0;
  const float *maskp = has_mask ? mask.data_ptr<float>() : nullptr;
  const int mask_hs = (has_mask && mask.size(1) == H) ? S * S : 0;

  {
    // One cooperative launch for the whole layer.  MTW/NT/P match the winning
    // standalone config (a 1x1 warp tile with an 8-deep prefetch).
    constexpr int MTW = 1, NT = 1, P = 16, NWARP = 4;
    const int nmg = MT / MTW;
    int maxtasks = (3 * D / (8 * NT)) * nmg;
    maxtasks = std::max(maxtasks, (I / (8 * NT)) * nmg);
    maxtasks = std::max(maxtasks, (D / (8 * NT)) * nmg * kKS3);
    // The shared-memory opt-in has to be in place before the occupancy query, or
    // it reports zero resident blocks for a >48 KB request and the cooperative
    // launch looks impossible.  Both kernels are raised here, so the per-stage
    // fallback is launchable too.
    static bool attrs = false;
    if (!attrs) {
      cudaFuncSetAttribute((void *)fused_layer<MTW, NT, P, NWARP>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize, kFusedSmem);
      cudaFuncSetAttribute((void *)attention<4>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize, kFusedSmem);
      cudaGetLastError();
      attrs = true;
    }
    const int G = fused_grid<MTW, NT, P, NWARP>(maxtasks);
    const bool kdiv = ((D / 8) % P == 0) && (((I / 8) / kKS3) % P == 0) &&
                      ((I / 8) % kKS3 == 0);
    if (G > 0 && kdiv) {
      float epsf = (float)eps, scale = 1.f / sqrtf((float)(D / H));
      int Sv = S, Dv = D, Iv = I, Hv = H, KSv = kKS3, mhs = mask_hs;
      const float *xp = x.data_ptr<float>();
      const float *g1p = g1.data_ptr<float>(), *b1p = b1.data_ptr<float>();
      const float *g2p = g2.data_ptr<float>(), *b2p = b2.data_ptr<float>();
      const float *wqp = wqkv.data_ptr<float>(), *bqp = bqkv.data_ptr<float>();
      const float *wop = wo.data_ptr<float>(), *bop = bo.data_ptr<float>();
      const float *w1p = w1.data_ptr<float>(), *bf1p = bf1.data_ptr<float>();
      const float *w2p = w2.data_ptr<float>(), *bf2p = bf2.data_ptr<float>();
      float *outp = out.data_ptr<float>();
      void *args[] = {(void *)&xp,  (void *)&maskp, (void *)&mhs,  (void *)&g1p,
                      (void *)&b1p, (void *)&g2p,   (void *)&b2p,  (void *)&wqp,
                      (void *)&bqp, (void *)&wop,   (void *)&bop,  (void *)&w1p,
                      (void *)&bf1p, (void *)&w2p,  (void *)&bf2p, (void *)&w,
                      (void *)&outp, (void *)&epsf, (void *)&Sv,   (void *)&Dv,
                      (void *)&Iv,  (void *)&Hv,    (void *)&KSv,  (void *)&scale};
      // The occupancy query says how many blocks are resident, but a cooperative
      // launch at exactly that number is refused if anything else holds
      // resources on the device, and the stage loops are written to work at any
      // grid size -- so shrink and retry rather than giving up the single launch.
      for (int g = G; g > 0; g /= 2) {
        if (cudaLaunchCooperativeKernel((void *)fused_layer<MTW, NT, P, NWARP>, dim3(g),
                                        dim3(NWARP * 32), args, kFusedSmem,
                                        stream) == cudaSuccess)
          return out;
        cudaGetLastError();
      }
      // fall through to the per-stage path
    }
  }

  ln_frag<<<S, 128, 0, stream>>>(
      x.data_ptr<float>(), g1.data_ptr<float>(), b1.data_ptr<float>(), (float)eps, af_h1, D);

  // Per-stage fallback, taken only if a cooperative launch is unavailable: the
  // same device code, one kernel per stage, paying ~2.8 us of launch gap each.
  gemm_direct<1, 1, 16><<<dim3(3 * D / 8, MT), 32, 0, stream>>>(
      af_h1, wqkv.data_ptr<float>(), bqkv.data_ptr<float>(), nullptr, nullptr, qkvbuf, D,
      3 * D, S, EPI_QKV, 1);
  attention<4><<<H * MT, 128, kFusedSmem, stream>>>(
      qkvbuf, maskp, mask_hs, af_attn, 1.f / sqrtf((float)(D / H)), S, H);
  gemm_direct<1, 1, 16><<<dim3(D / 8, MT), 32, 0, stream>>>(
      af_attn, wo.data_ptr<float>(), bo.data_ptr<float>(), resid1, x.data_ptr<float>(),
      nullptr, D, D, S, EPI_ADD_ROW, 1);
  ln_frag<<<S, 128, 0, stream>>>(resid1, g2.data_ptr<float>(), b2.data_ptr<float>(),
                                 (float)eps, af_h2, D);
  gemm_direct<1, 1, 16><<<dim3(I / 8, MT), 32, 0, stream>>>(
      af_h2, w1.data_ptr<float>(), bf1.data_ptr<float>(), af_act, nullptr, nullptr, D, I, S,
      EPI_GELU_FRAG, 1);
  gemm_direct<1, 1, 16><<<dim3(D / 8, MT, kKS3), 32, 0, stream>>>(
      af_act, w2.data_ptr<float>(), bf2.data_ptr<float>(), part, resid1, nullptr, I, D, S,
      EPI_ADD_ROW, kKS3);
  {
    const int nthr = 256, nel = S * D;
    reduce_epi<<<(nel + nthr - 1) / nthr, nthr, 0, stream>>>(
        part, bf2.data_ptr<float>(), resid1, out.data_ptr<float>(), D, S, kKS3);
  }
  return out;
}
"""


def _build():
    from torch.utils.cpp_extension import load_inline
    try:
        from fastkernels.infra.cuda_ext import _pin_build_arch
        _pin_build_arch()
    except Exception:
        try:
            major, minor = torch.cuda.get_device_capability()
            os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}a"
        except Exception:
            pass
    # No --use_fast_math: expf / rsqrtf / fp32 division all have to be the
    # IEEE-accurate forms the reference kernels were compiled with, or the
    # softmax and the QuickGELU stop matching bit for bit.
    return load_inline(
        name="fk_cand_clip_encoder_layer",
        cpp_sources=_CPP,
        cuda_sources=_CUDA,
        functions=["fk_pack_b", "fk_ws_floats", "fk_clip_layer"],
        with_cuda=True,
        verbose=False,
        extra_cuda_cflags=["-O3"],
    )


try:
    _EXT = _build()
except Exception:      # no nvcc / unsupported toolchain -> eager fallback
    _EXT = None


class CLIPEncoderLayer(nn.Module):
    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        self.self_attn = CLIPAttention(config)
        self.layer_norm1 = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = CLIPMLP(config)
        self.layer_norm2 = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self._plan = None

    # -- one-time staging ---------------------------------------------------
    def _build_plan(self):
        """Pack the four weights into mma-fragment order and size the workspace.

        Done once, on the first forward: weights are frozen by then, and the
        packing also applies the TF32 rounding so the GEMMs convert nothing.
        ``()`` marks "do not try the kernel".
        """
        self._plan = ()
        if _EXT is None:
            return ()
        a, m = self.self_attn, self.mlp
        ps = [self.layer_norm1.weight, self.layer_norm1.bias,
              self.layer_norm2.weight, self.layer_norm2.bias,
              a.q_proj.weight, a.k_proj.weight, a.v_proj.weight,
              a.q_proj.bias, a.k_proj.bias, a.v_proj.bias,
              a.out_proj.weight, a.out_proj.bias,
              m.fc1.weight, m.fc1.bias, m.fc2.weight, m.fc2.bias]
        if any(p is None for p in ps):
            return ()
        if not all(p.is_cuda and p.dtype is torch.float32 for p in ps):
            return ()
        D, I, H = a.embed_dim, m.fc1.weight.shape[0], a.num_heads
        # 128 | D and 512 | I keep every GEMM's k extent a whole number of the
        # 16-k-step prefetch blocks the inner loop steps by (fc2's is also split
        # four ways), and D <= 2048 keeps a LayerNorm row inside its register
        # staging.
        if a.head_dim != _HD or H * _HD != D or D % 128 or I % 512 or D > 2048:
            return ()
        with torch.no_grad():
            wqkv = _EXT.fk_pack_b(torch.cat((a.q_proj.weight, a.k_proj.weight,
                                             a.v_proj.weight), 0))
            bqkv = torch.cat((a.q_proj.bias, a.k_proj.bias, a.v_proj.bias)).contiguous()
            wo = _EXT.fk_pack_b(a.out_proj.weight)
            w1 = _EXT.fk_pack_b(m.fc1.weight)
            w2 = _EXT.fk_pack_b(m.fc2.weight)
            ws = torch.zeros(_EXT.fk_ws_floats(D, I, H), device=wqkv.device,
                             dtype=torch.float32)
        plan = (self.layer_norm1.weight, self.layer_norm1.bias,
                self.layer_norm2.weight, self.layer_norm2.bias,
                wqkv, bqkv, wo, a.out_proj.bias.contiguous(),
                w1, m.fc1.bias.contiguous(), w2, m.fc2.bias.contiguous(),
                ws, float(self.layer_norm1.eps), int(H))
        self._plan = plan
        return plan

    def _apply(self, *args, **kwargs):          # .to() / .cuda() / dtype casts
        self._plan = None
        return super()._apply(*args, **kwargs)

    def _load_from_state_dict(self, *args, **kwargs):
        self._plan = None
        return super()._load_from_state_dict(*args, **kwargs)

    # -- forward -----------------------------------------------------------
    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        plan = self._plan
        if plan is None:
            plan = self._build_plan()
        if plan and self._kernel_takes(hidden_states, attention_mask, plan[14]):
            mask = attention_mask if attention_mask is not None else torch.Tensor()
            return _EXT.fk_clip_layer(hidden_states, mask, *plan)
        return self._forward_eager(hidden_states, attention_mask)

    @staticmethod
    def _kernel_takes(x: torch.Tensor, mask: torch.Tensor | None, heads: int) -> bool:
        if (x.dtype is not torch.float32 or not x.is_cuda or x.dim() != 3
                or x.shape[0] != 1 or not x.is_contiguous()):
            return False
        S = x.shape[1]
        # 65..80: five m16 tiles of padding room, and the sequence length range
        # over which F.softmax picks the 128-wide (4 iterations per lane) warp
        # kernel that ``attention`` transcribes.
        if S < 65 or S > _MROWS:
            return False
        if mask is None:
            return True
        return (mask.dtype is torch.float32 and mask.dim() == 4 and mask.shape[0] == 1
                and mask.shape[1] in (1, heads) and mask.shape[2] == S
                and mask.shape[3] == S and mask.is_contiguous())

    def _forward_eager(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask=attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states
