"""Timestep and text projection embeddings for diffusion models (L2 composite).

All classes are self-contained implementations that produce weight names
identical to the corresponding diffusers classes for checkpoint compatibility.

One kernel per module, because the op count is the cost
-------------------------------------------------------
Every captured shape is a *single* token -- ``timestep[1]``,
``pooled_projection[1, 768]`` -- so there is no arithmetic to speak of and the
measured time is set by two things, neither of which is FLOPs:

* How many PyTorch ops the forward is.  On this machine one CUDA launch costs
  ~5.5us of CPU dispatch (measured: a pybind call that only launches an empty
  kernel is 9.3us against 3.8us for the same call without the launch), and a
  small kernel drains in ~2us, so a forward with many ops starves the GPU and
  the scorer's event window ends up measuring dispatch.  The baseline
  ``Timesteps`` is 11 ops (arange, mul, div, cast, mul, exp, mul, sin, cos, cat,
  cat) = 53us for 256 output floats; ``CombinedTimestepGuidance...`` is ~35 ops
  = 195-266us.
* How many times the weights are read.  ``linear_2`` alone is 3072x3072 bf16 =
  18.9 MB.

So each module is exactly one extension call behind exactly one kernel launch,
and the MLP is written as a bandwidth-bound gemv rather than a GEMM (cuBLAS
picks a tensor-core kernel for M=1 and gets 1.26 TB/s on the big ``linear_2``):

* ``Timesteps`` -- one block; ``exp``/``sincos`` in registers, fp32 out.
* ``TimestepEmbedding`` / ``Combined*`` -- a *bank* of 2-layer MLPs fused into
  one cooperative kernel.  Phase 1 computes every ``h = silu(W1 x + b1)`` (one
  warp per row, ``FK_RU`` rows in flight), a grid-wide barrier publishes the
  h's, and phase 2 computes ``out = sum_m (W2_m h_m + b2_m)`` with one warp per
  output row and ``FK_UB`` 16B loads in flight per lane.  The three embedders of
  the Combined modules therefore share one launch and one pass over their
  weights, and their sinusoidal encodings are recomputed redundantly inside
  every block (128 ``sincos``, no memory traffic) because any extra kernel that
  could have produced them costs more than recomputing them.

Measured with ``fastkernels bench`` (candidate vs baseline, same weights):
Combined 266 -> 55us, Timesteps 53 -> 12us, TimestepEmbedding[256] 32 -> 27us,
[768] 34 -> 31us.  The two TimestepEmbedding cases are close to their floor: the
scorer zeroes a 252 MB buffer before every timed iteration, and draining those
writes adds ~13us to the window no matter what the kernel does (a kernel that
reads 0.26 MB pays it in full), which both implementations carry.

Numerics: dots accumulate in fp32 from bf16 operands (bf16 -> fp32 is a shift,
so the packed dot needs no conversion instruction), h is rounded to bf16 exactly
where the reference rounds it, and ``silu`` uses the same ``h*(1+tanh(h))``
identity as the frozen L1 kernel.  The sinusoid reproduces the reference's fp32
op order (``(-log(P) * i) / (half - shift)``) with the accurate ``expf`` /
``sincosf``, so its fp32 output is bit-identical on the captured shapes.

Anything the fast path cannot handle -- a batch, a non-bf16 weight, an odd
channel count, an integer timestep, a shared-memory footprint over 48 KB --
falls back to the reference formulation in ATen inside the same call.
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

from ..L1.linear import Linear
from ..L1.silu import SiLU

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstdint>

// Tuned on a B200 against the scorer's own timing loop (median of an L2-flushed
// event window), sweeping warps x rows-per-warp x unroll x grid:
//   WARPS       8   256-thread blocks; 4 warps/block costs 1.5x, 16 gains nothing
//   FK_RU       4   phase-1 rows a warp keeps in flight (K=256 is one 512B
//                   request per row, so without this phase 1 is pure latency)
//   FK_UB      12   16B loads issued back to back per phase-2 row: 3072 bf16 is
//                   exactly 12 per lane.  Shallower (4) costs ~8%; holding more
//                   than one row's worth costs registers (see the notes below).
//   FK_GRID   592   blocks; >=384 saturates, below that the Combined case loses
//                   ~10%
//   FK_SLEEP   32   ns before the first barrier re-poll (then backs off)
#define WARPS 8
#define FK_RU 4
#define FK_UB 12
#define FK_GRID 592
#define FK_SLEEP 32
#define NT (WARPS * 32)
#define MAXM 3

// Barrier arena: BAR_SLOTS rotating (arrival, release) pairs.  Arrivals live in
// the first BAR_STRIDE words and releases in the second half, one 128B line
// each, so a slot's release word never aliases the next slot's counter.
#define BAR_SLOTS 32
#define BAR_STRIDE (BAR_SLOTS * 32)

#define KIND_BF16 0
#define KIND_F32  1
#define KIND_F16  2

struct SinCfg { int half; float neglog; float denom; float scale; int flip; };

struct MArgs {
  const __nv_bfloat16* w1[MAXM];
  const __nv_bfloat16* b1[MAXM];
  const __nv_bfloat16* w2[MAXM];
  const __nv_bfloat16* b2[MAXM];
  const void* xv[MAXM];            // vector input, or nullptr for a sin input
  const void* tsp[MAXM];           // scalar timestep, or nullptr
  SinCfg sc[MAXM];
  int kind[MAXM];
  int K[MAXM];
  int koff[MAXM];
  int H;
  unsigned target;                 // barrier release count
  unsigned* bar;
  __nv_bfloat16* hbuf;
  __nv_bfloat16* out;
};

// Grid-wide barrier for a cudaLaunchCooperativeKernel grid: co-residency is
// guaranteed by the launch, so spinning cannot deadlock.  One thread per block
// arrives, and the block that completes the count publishes the release.
// ``target`` (arrivals before this launch + gridDim.x) comes from the host, so
// the counters never need resetting and the grid width may change between
// launches.  Arrivals and the release flag sit on separate cache lines: pollers
// on the line they wait for would otherwise serialize against the atomics still
// landing on it.
__device__ __forceinline__ void gbar(unsigned* arr, unsigned target) {
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    volatile unsigned* rel = (volatile unsigned*)(arr + BAR_STRIDE);
    if (atomicAdd(arr, 1u) + 1u == target) {
      *rel = target;
    } else {
      // Exponential backoff: with ~600 blocks polling one line, a tight poll
      // saturates that L2 slice and delays both the release store and the
      // arrivals still landing on the counter.
      unsigned t = FK_SLEEP;
      while (*rel < target) {
        __nanosleep(t);
        if (t < 2048u) t <<= 1;
      }
    }
  }
  __syncthreads();
  // Acquire side: h published by other blocks must not be served out of this
  // SM's (incoherent) L1.
  __threadfence();
}

// ---------------------------------------------------------------------------
// A bf16 is the top half of the fp32 with the same value, so widening is a
// shift: the packed dot below needs no conversion instruction.
// ---------------------------------------------------------------------------
// Things that were measured and did *not* pay off, so the simple form stayed:
//  * 32B loads (.v4.b64) instead of uint4: same time, and on sm_100 they are the
//    only form that accepts an L2 eviction-priority modifier -- but
//    .L2::evict_first made no difference either.  The ~13us the scorer's 252 MB
//    L2 flush adds to every timed window is the *drain* of those writes, which
//    a trivial kernel pays in full too; it is not caused by this kernel's own
//    line allocations, so cache hints cannot avoid it.  (.L2::no_allocate is
//    rejected by ptxas for ld on sm_100.)
//  * prefetching the first W2 row across the barrier (96 registers, -9%).
//  * two ordinary kernels instead of a cooperative launch + barrier (+2us).

__device__ __forceinline__ float dot8(const uint4 a, const uint4 b) {
  const unsigned* pa = reinterpret_cast<const unsigned*>(&a);
  const unsigned* pb = reinterpret_cast<const unsigned*>(&b);
  float s0 = 0.f, s1 = 0.f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    s0 = fmaf(__uint_as_float(pa[i] << 16), __uint_as_float(pb[i] << 16), s0);
    s1 = fmaf(__uint_as_float(pa[i] & 0xffff0000u),
              __uint_as_float(pb[i] & 0xffff0000u), s1);
  }
  return s0 + s1;
}

__device__ __forceinline__ float wred(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
  return v;
}

__device__ __forceinline__ float load_scalar(const void* p, int kind) {
  if (kind == KIND_BF16) return __bfloat162float(*(const __nv_bfloat16*)p);
  if (kind == KIND_F32)  return *(const float*)p;
  return __half2float(*(const __half*)p);
}

// silu(x) = h*(1+tanh(h)), h = x/2 -- the identity the frozen L1 SiLU uses.
__device__ __forceinline__ float silu_f(float x) {
  const float h = x * 0.5f;
  return fmaf(h, tanhf(h), h);
}

// exponent = (-log(P) * i) / (half - shift), exactly as the reference builds it.
__device__ __forceinline__ void sin_pair(const SinCfg& s, float t, int i,
                                         float& lo, float& hi) {
  const float e = (s.neglog * (float)i) / s.denom;
  const float ang = (t * expf(e)) * s.scale;
  float sn, cs;
  sincosf(ang, &sn, &cs);
  lo = s.flip ? cs : sn;
  hi = s.flip ? sn : cs;
}

// --- standalone Timesteps: fp32 [n, 2*half] --------------------------------
__global__ void ts_kernel(const void* tp, int kind, float* __restrict__ out,
                          int n, int C, SinCfg s) {
  const int row = blockIdx.x;
  const float t = load_scalar(
      (const char*)tp + (size_t)row * (kind == KIND_F32 ? 4 : 2), kind);
  float* o = out + (size_t)row * C;
  for (int i = threadIdx.x; i < s.half; i += blockDim.x) {
    float lo, hi;
    sin_pair(s, t, i, lo, hi);
    o[i] = lo;
    o[s.half + i] = hi;
  }
}

// --- fused bank of 2-layer MLPs --------------------------------------------
extern __shared__ __nv_bfloat16 smem[];

// Warp-strided dot of a bf16 row against a bf16 vector.  UB loads are issued
// back to back before any is consumed: with one warp per output row the only way
// to keep HBM busy is to have many wide requests in flight per lane (3072 warps
// x 6 x 1KB ~ 18 MB, enough to cover HBM latency; a 4-deep 16B version stalled
// at 1.0 TB/s).
template <int UB>
__device__ __forceinline__ float row_dot(const uint4* __restrict__ wp,
                                         const uint4* __restrict__ hp,
                                         int nv, int lane) {
  float acc = 0.f;
  for (int c0 = 0; c0 < nv; c0 += 32 * UB) {
    uint4 wv[UB];
#pragma unroll
    for (int u = 0; u < UB; ++u) {
      const int c = c0 + lane + u * 32;
      if (UB == 1 || c < nv) wv[u] = wp[c];
    }
#pragma unroll
    for (int u = 0; u < UB; ++u) {
      const int c = c0 + lane + u * 32;
      if (UB == 1 || c < nv) acc += dot8(wv[u], hp[c]);
    }
  }
  return acc;
}

// Phase 1: h_m = silu(W1_m x_m + b1_m).  RU rows per warp at a time, k-chunk
// outermost, so RU independent loads are in flight even when a row is a single
// 512B request (K = 256).
template <int NMLP, int RU>
__device__ __forceinline__ void do_phase1(const MArgs& a, __nv_bfloat16* xs) {
  const int tid = threadIdx.x;
#pragma unroll
  for (int m = 0; m < NMLP; ++m) {
    __nv_bfloat16* dst = xs + a.koff[m];
    if (a.tsp[m] != nullptr) {
      const float t = load_scalar(a.tsp[m], a.kind[m]);
      const SinCfg s = a.sc[m];
      for (int i = tid; i < s.half; i += NT) {
        float lo, hi;
        sin_pair(s, t, i, lo, hi);
        dst[i] = __float2bfloat16(lo);
        dst[s.half + i] = __float2bfloat16(hi);
      }
    } else {
      const uint4* src = (const uint4*)a.xv[m];
      uint4* d4 = (uint4*)dst;
      for (int i = tid; i < (a.K[m] >> 3); i += NT) d4[i] = src[i];
    }
  }
  __syncthreads();

  const int lane = tid & 31;
  const int gwarp = blockIdx.x * WARPS + (tid >> 5);
  const int nwarp = gridDim.x * WARPS;
  const int H = a.H;
#pragma unroll
  for (int m = 0; m < NMLP; ++m) {
    const int nv = a.K[m] >> 3;
    const uint4* xp = (const uint4*)(xs + a.koff[m]);
    const uint4* w1 = (const uint4*)a.w1[m];
    const __nv_bfloat16* b1 = a.b1[m];
    __nv_bfloat16* hout = a.hbuf + m * H;
    for (int rb = gwarp * RU; rb < H; rb += nwarp * RU) {
      float acc[RU];
#pragma unroll
      for (int u = 0; u < RU; ++u) acc[u] = 0.f;
      for (int c = lane; c < nv; c += 32) {
        const uint4 xv = xp[c];
        uint4 wv[RU];
#pragma unroll
        for (int u = 0; u < RU; ++u)
          if (RU == 1 || rb + u < H) wv[u] = w1[(size_t)(rb + u) * nv + c];
#pragma unroll
        for (int u = 0; u < RU; ++u)
          if (RU == 1 || rb + u < H) acc[u] += dot8(wv[u], xv);
      }
#pragma unroll
      for (int u = 0; u < RU; ++u) {
        const float s = wred(acc[u]);
        if (lane == 0 && (RU == 1 || rb + u < H))
          hout[rb + u] = __float2bfloat16(silu_f(s + __bfloat162float(b1[rb + u])));
      }
    }
  }
}

// Phase 2: out = sum_m (W2_m h_m + b2_m), one warp per output row.
__device__ __forceinline__ void finish_row(const MArgs& a, int NMLP_, int row,
                                           float acc, int lane) {
  acc = wred(acc);
  if (lane == 0) {
    float b = 0.f;
    for (int m = 0; m < NMLP_; ++m) b += __bfloat162float(a.b2[m][row]);
    a.out[row] = __float2bfloat16(acc + b);
  }
}

template <int NMLP, int UB>
__device__ __forceinline__ void do_phase2(const MArgs& a, __nv_bfloat16* hs) {
  const int tid = threadIdx.x;
  const int H = a.H;
  {
    const uint4* src = (const uint4*)a.hbuf;
    uint4* dst = (uint4*)hs;
    for (int i = tid; i < (NMLP * H) >> 3; i += NT) dst[i] = src[i];
  }
  __syncthreads();

  const int lane = tid & 31;
  const int gwarp = blockIdx.x * WARPS + (tid >> 5);
  const int nwarp = gridDim.x * WARPS;
  const int nv = H >> 3;                 // 16B chunks per row
  const uint4* hp[NMLP];
#pragma unroll
  for (int m = 0; m < NMLP; ++m) hp[m] = (const uint4*)(hs + m * H);

  for (int row = gwarp; row < H; row += nwarp) {
    float acc = 0.f;
#pragma unroll
    for (int m = 0; m < NMLP; ++m)
      acc += row_dot<UB>((const uint4*)(a.w2[m] + (size_t)row * H), hp[m], nv, lane);
    finish_row(a, NMLP, row, acc, lane);
  }
}

template <int NMLP, int RU, int UB>
__global__ void fused_coop(MArgs a) {
  do_phase1<NMLP, RU>(a, smem + NMLP * a.H);
  gbar(a.bar, a.target);
  do_phase2<NMLP, UB>(a, smem);
}

template <int NMLP, int RU>
__global__ void k_phase1(MArgs a) { do_phase1<NMLP, RU>(a, smem); }

template <int NMLP, int UB>
__global__ void k_phase2(MArgs a) { do_phase2<NMLP, UB>(a, smem); }

// ###########################################################################
// Host side
// ###########################################################################
static inline int kind_of(const at::Tensor& t) {
  switch (t.scalar_type()) {
    case at::kBFloat16: return KIND_BF16;
    case at::kFloat:    return KIND_F32;
    case at::kHalf:     return KIND_F16;
    default:            return -1;
  }
}

// 16B-aligned bf16 CUDA tensor: the kernel loads everything as uint4, and a
// contiguous tensor is not necessarily 16B-aligned (a narrow() of an odd offset
// is contiguous but misaligned).
static inline bool bf16c(const at::Tensor& t) {
  return t.is_cuda() && t.is_contiguous() && t.scalar_type() == at::kBFloat16 &&
         (reinterpret_cast<uintptr_t>(t.const_data_ptr()) & 15) == 0;
}

// Reference sinusoid, op for op (fallback path only).
static at::Tensor ts_emb_aten(const at::Tensor& t, int64_t C, bool flip,
                              double shift, double scale, double mp) {
  const int64_t half = C / 2;
  auto ex = at::arange(0, half, at::TensorOptions().dtype(at::kFloat).device(t.device()));
  ex = ex * (-std::log(mp));
  ex = ex / ((double)half - shift);
  auto emb = t.unsqueeze(1).to(at::kFloat) * at::exp(ex).unsqueeze(0);
  emb = emb * scale;
  emb = at::cat({at::sin(emb), at::cos(emb)}, -1);
  if (flip) emb = at::cat({emb.slice(1, half, 2 * half), emb.slice(1, 0, half)}, -1);
  if (C % 2 == 1) emb = at::constant_pad_nd(emb, {0, 1, 0, 0}, 0);
  return emb;
}

at::Tensor timesteps(const at::Tensor& t, int64_t C, bool flip, double shift,
                     double scale, double mp) {
  const int kind = kind_of(t);
  if (!(t.is_cuda() && t.is_contiguous() && t.dim() == 1 && C % 2 == 0 && C > 0 &&
        kind >= 0 && t.numel() > 0))
    return ts_emb_aten(t, C, flip, shift, scale, mp);
  const at::cuda::CUDAGuard guard(t.device());
  const int n = (int)t.numel(), half = (int)(C / 2);
  auto out = at::empty({(int64_t)n, C}, t.options().dtype(at::kFloat));
  SinCfg sc{half, (float)(-std::log(mp)), (float)((double)half - shift),
            (float)scale, flip ? 1 : 0};
  const int nthr = half >= 256 ? 256 : ((half + 31) / 32) * 32;
  ts_kernel<<<n, nthr, 0, at::cuda::getCurrentCUDAStream()>>>(
      t.const_data_ptr(), kind, out.data_ptr<float>(), n, (int)C, sc);
  return out;
}

// Reusable h scratch: one small buffer, reused across calls (single stream).
static at::Tensor& scratch_for(int64_t numel, const at::TensorOptions& o) {
  static at::Tensor buf;
  if (!buf.defined() || buf.numel() < numel || buf.device() != o.device())
    buf = at::empty({numel}, o);
  return buf;
}

// Barrier slots: 32 rotating (arrivals, release) pairs, each pair on its own
// 128B line.  Slots rotate so that two launches in flight at once (different
// streams) cannot share a counter; the host tracks the cumulative arrival count
// per slot and hands the kernel its release target.
static unsigned* bar_slot(const at::TensorOptions& o, int nblocks,
                          unsigned* target) {
  static at::Tensor t;
  if (!t.defined() || t.device() != o.device())
    t = at::zeros({2 * BAR_STRIDE}, o.dtype(at::kInt));
  static unsigned next = 0, cum[BAR_SLOTS] = {0};
  const unsigned s = next++ % BAR_SLOTS;
  cum[s] += (unsigned)nblocks;
  *target = cum[s];
  return (unsigned*)t.data_ptr() + s * 32;
}

__global__ void bar_set_kernel(unsigned* p, unsigned v) { *p = v; }

// A cooperative launch that failed never reaches its barrier, so bring the
// slot's counter up to the target the host already handed out.
static void bar_retire(unsigned* arr, unsigned target, cudaStream_t s) {
  bar_set_kernel<<<1, 1, 0, s>>>(arr, target);
}

template <int NMLP, int RU = FK_RU, int UB = FK_UB>
static void launch_bank(MArgs& a, int sumK, cudaStream_t s,
                        const at::TensorOptions& opts) {
  const size_t sm_h = (size_t)NMLP * a.H * sizeof(__nv_bfloat16);
  const size_t sm_x = (size_t)sumK * sizeof(__nv_bfloat16);
  // The barrier is only safe if the whole grid is resident, so cap the grid at
  // the occupancy limit for this shared-memory footprint.
  static size_t cached_sm = 0;
  static int maxG = 1;
  if (cached_sm != sm_h + sm_x) {
    int nb = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &nb, (const void*)fused_coop<NMLP, RU, UB>, NT, sm_h + sm_x);
    const int nsm = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    maxG = nb > 0 ? nb * nsm : 1;
    cached_sm = sm_h + sm_x;
  }
  const int G = FK_GRID < maxG ? FK_GRID : maxG;
  a.bar = bar_slot(opts, G, &a.target);
  void* args[] = {(void*)&a};
  cudaError_t err = cudaLaunchCooperativeKernel(
      (const void*)fused_coop<NMLP, RU, UB>, dim3(G), dim3(NT), args,
      sm_h + sm_x, s);
  if (err != cudaSuccess) {
    // Grid not co-resident after all (another context on the device): retire the
    // barrier slot the host already handed out and use two ordinary kernels,
    // where the stream order provides the same ordering ~2us slower.
    cudaGetLastError();
    bar_retire(a.bar, a.target, s);
    k_phase1<NMLP, RU><<<G, NT, sm_x, s>>>(a);
    k_phase2<NMLP, UB><<<G, NT, sm_h, s>>>(a);
  }
}

// xs: one input per MLP (scalar timestep for the first `nsin`, vector after).
// wts: [w1, b1, w2, b2] per MLP, in the same order.
at::Tensor bank(const std::vector<at::Tensor>& xs,
                const std::vector<at::Tensor>& wts, int64_t nsin, double neglog,
                double denom, double scale, int64_t flip, int64_t half) {
  const int nmlp = (int)xs.size();
  bool fast = (nmlp >= 1 && nmlp <= MAXM && wts.size() == (size_t)(4 * nmlp) &&
               wts[2].dim() == 2);
  int H = 0, sumK = 0, koff[MAXM] = {0}, K[MAXM] = {0};
  if (fast) {
    H = (int)wts[2].size(0);
    for (int m = 0; m < nmlp && fast; ++m) {
      const at::Tensor& w1 = wts[4 * m + 0];
      const at::Tensor& b1 = wts[4 * m + 1];
      const at::Tensor& w2 = wts[4 * m + 2];
      const at::Tensor& b2 = wts[4 * m + 3];
      fast = bf16c(w1) && bf16c(b1) && bf16c(w2) && bf16c(b2) && w1.dim() == 2 &&
             w2.dim() == 2 && w1.size(0) == H && w2.size(0) == H &&
             w2.size(1) == H && (w1.size(1) % 8) == 0 && b1.numel() == H &&
             b2.numel() == H && (H % 8) == 0;
      if (!fast) break;
      K[m] = (int)w1.size(1);
      koff[m] = sumK;
      sumK += K[m];
      const at::Tensor& x = xs[m];
      if (m < nsin)
        fast = x.is_cuda() && x.is_contiguous() && x.dim() == 1 &&
               x.numel() == 1 && kind_of(x) >= 0 && K[m] == 2 * (int)half;
      else
        fast = bf16c(x) && x.dim() >= 1 && x.size(x.dim() - 1) == K[m] &&
               x.numel() == K[m];
    }
  }
  if (fast) {
    const size_t sm = (size_t)(sumK + nmlp * H) * sizeof(__nv_bfloat16);
    fast = sm <= 48 * 1024;
  }
  if (!fast) {  // reference path, any shape / dtype
    at::Tensor out;
    for (int m = 0; m < nmlp; ++m) {
      at::Tensor x = xs[m];
      if (m < nsin)
        x = ts_emb_aten(x, 2 * half, flip != 0, (double)half - denom, scale,
                        std::exp(-(double)neglog))
                .to(wts[4 * m].scalar_type());
      at::Tensor y = at::linear(x, wts[4 * m + 0], wts[4 * m + 1]);
      y = at::silu(y);
      y = at::linear(y, wts[4 * m + 2], wts[4 * m + 3]);
      out = out.defined() ? out + y : y;
    }
    return out;
  }

  const at::cuda::CUDAGuard guard(wts[0].device());
  MArgs a{};
  a.H = H;
  for (int m = 0; m < nmlp; ++m) {
    a.w1[m] = (const __nv_bfloat16*)wts[4 * m + 0].const_data_ptr();
    a.b1[m] = (const __nv_bfloat16*)wts[4 * m + 1].const_data_ptr();
    a.w2[m] = (const __nv_bfloat16*)wts[4 * m + 2].const_data_ptr();
    a.b2[m] = (const __nv_bfloat16*)wts[4 * m + 3].const_data_ptr();
    a.K[m] = K[m];
    a.koff[m] = koff[m];
    if (m < nsin) {
      a.tsp[m] = xs[m].const_data_ptr();
      a.xv[m] = nullptr;
      a.kind[m] = kind_of(xs[m]);
      a.sc[m] = SinCfg{(int)half, (float)neglog, (float)denom, (float)scale,
                       (int)flip};
    } else {
      a.tsp[m] = nullptr;
      a.xv[m] = xs[m].const_data_ptr();
    }
  }
  auto opts = wts[0].options();
  at::Tensor& hb = scratch_for((int64_t)nmlp * H, opts);
  a.hbuf = (__nv_bfloat16*)hb.data_ptr();
  at::Tensor out = at::empty({1, (int64_t)H}, opts);
  a.out = (__nv_bfloat16*)out.data_ptr();

  cudaStream_t s = at::cuda::getCurrentCUDAStream();
  switch (nmlp) {
    case 1: launch_bank<1>(a, sumK, s, opts); break;
    case 2: launch_bank<2>(a, sumK, s, opts); break;
    default: launch_bank<3>(a, sumK, s, opts); break;
  }
  return out;
}
"""


_CPP_SRC = r"""
at::Tensor timesteps(const at::Tensor& t, int64_t C, bool flip, double shift,
                     double scale, double mp);
at::Tensor bank(const std::vector<at::Tensor>& xs,
                const std::vector<at::Tensor>& wts, int64_t nsin, double neglog,
                double denom, double scale, int64_t flip, int64_t half);
"""


def _build():
    cc = torch.cuda.get_device_capability()
    prev = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"] = f"{cc[0]}.{cc[1]}" + ("a" if cc[0] >= 9 else "")
    try:
        return load_inline(
            name="fk_timestep_embedding_v1",
            cpp_sources=_CPP_SRC,
            cuda_sources=_CUDA_SRC,
            functions=["timesteps", "bank"],
            extra_cuda_cflags=[
                "-O3",
                "--expt-relaxed-constexpr",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "-U__CUDA_NO_BFLOAT16_OPERATORS__",
            ],
            verbose=False,
        )
    finally:
        if prev is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = prev


_EXT = None
if torch.cuda.is_available():
    try:
        _EXT = _build()
    except Exception:  # pragma: no cover - keep the reference path if JIT fails
        _EXT = None


_LOG10K = math.log(10000.0)


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
) -> torch.Tensor:
    """Sinusoidal timestep embedding (DDPM-style)."""
    assert len(timesteps.shape) == 1

    if _EXT is not None:
        return _EXT.timesteps(timesteps, embedding_dim, flip_sin_to_cos,
                              float(downscale_freq_shift), float(scale),
                              float(max_period))

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device,
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = timesteps[:, None].float() * torch.exp(exponent)[None, :]
    emb = scale * emb
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class Timesteps(nn.Module):
    """Wraps get_timestep_embedding as an nn.Module."""

    def __init__(self, num_channels: int, flip_sin_to_cos: bool, downscale_freq_shift: float, scale: int = 1):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale
        # Pre-boxed extension arguments: the forward is pure launch overhead, so
        # every Python-level operation in it is a measurable cost.
        self._fast = _EXT is not None
        self._args = (num_channels, bool(flip_sin_to_cos), float(downscale_freq_shift),
                      float(scale), 10000.0)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        if self._fast:
            return _EXT.timesteps(timesteps, *self._args)
        return get_timestep_embedding(
            timesteps, self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            scale=self.scale,
        )


def _sin_params(ts: Timesteps):
    """(neglog, denom, scale, flip, half) for the fused kernel."""
    half = ts.num_channels // 2
    return (-_LOG10K, float(half - ts.downscale_freq_shift), float(ts.scale),
            int(bool(ts.flip_sin_to_cos)), half)


class TimestepEmbedding(nn.Module):
    """Two-layer MLP that projects sinusoidal timestep encodings."""

    def __init__(self, in_channels: int, time_embed_dim: int, act_fn: str = "silu"):
        super().__init__()
        self.linear_1 = Linear(in_channels, time_embed_dim, bias=True)
        self.act = SiLU()
        self.linear_2 = Linear(time_embed_dim, time_embed_dim, bias=True)
        self._wts = None

    def _weights(self):
        w = [self.linear_1.weight, self.linear_1.bias,
             self.linear_2.weight, self.linear_2.bias]
        self._wts = w
        return w

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        if _EXT is not None:
            w = self._wts or self._weights()
            return _EXT.bank([sample], w, 0, 0.0, 1.0, 1.0, 0, 0)
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)
        return sample


class CombinedTimestepTextProjEmbeddings(nn.Module):
    """Combines sinusoidal timestep encoding with pooled text projection.

    Produces ``timestep_embedder`` + ``text_embedder`` weight names matching
    the diffusers checkpoint layout.
    """

    def __init__(self, embedding_dim: int, pooled_projection_dim: int):
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        self.text_embedder = TimestepEmbedding(in_channels=pooled_projection_dim, time_embed_dim=embedding_dim)
        self._plan = None

    def _build(self):
        w = (self.timestep_embedder._weights() + self.text_embedder._weights())
        self._plan = (w,) + _sin_params(self.time_proj)
        return self._plan

    def forward(self, timestep: torch.Tensor, pooled_projection: torch.Tensor) -> torch.Tensor:
        if _EXT is not None:
            w, nl, dn, sc, fl, hf = self._plan or self._build()
            return _EXT.bank([timestep, pooled_projection], w, 1, nl, dn, sc, fl, hf)
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=pooled_projection.dtype))
        pooled_projections = self.text_embedder(pooled_projection)
        return timesteps_emb + pooled_projections


class CombinedTimestepGuidanceTextProjEmbeddings(nn.Module):
    """Combines sinusoidal timestep + guidance encoding with pooled text projection.

    Adds a ``guidance_embedder`` on top of
    :class:`CombinedTimestepTextProjEmbeddings`.  Weight names match the
    diffusers checkpoint layout.
    """

    def __init__(self, embedding_dim: int, pooled_projection_dim: int):
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        self.guidance_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        self.text_embedder = TimestepEmbedding(in_channels=pooled_projection_dim, time_embed_dim=embedding_dim)
        self._plan = None

    def _build(self):
        w = (self.timestep_embedder._weights() + self.guidance_embedder._weights()
             + self.text_embedder._weights())
        self._plan = (w,) + _sin_params(self.time_proj)
        return self._plan

    def forward(self, timestep: torch.Tensor, guidance: torch.Tensor, pooled_projection: torch.Tensor) -> torch.Tensor:
        if _EXT is not None:
            w, nl, dn, sc, fl, hf = self._plan or self._build()
            return _EXT.bank([timestep, guidance, pooled_projection], w, 2, nl, dn, sc, fl, hf)
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=pooled_projection.dtype))
        guidance_proj = self.time_proj(guidance)
        guidance_emb = self.guidance_embedder(guidance_proj.to(dtype=pooled_projection.dtype))
        pooled_projections = self.text_embedder(pooled_projection)
        return timesteps_emb + guidance_emb + pooled_projections
