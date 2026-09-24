"""Timestep and text projection embeddings for diffusion models (L2 composite).

All classes are self-contained implementations that produce weight names
identical to the corresponding diffusers classes for checkpoint compatibility.

Optimized path
--------------
Every captured call is batch-1, so the reference formulation is dominated by
per-op launch/dispatch cost (~40 CUDA launches for the combined embedder) plus
one bandwidth-bound ``3072x3072`` mat-vec per branch.  Two hand-written CUDA
kernels collapse all of it:

* stage 1 -- for every MLP branch: build the sinusoidal encoding straight into
  shared memory when the input is a raw timestep, then ``silu(W1 @ x + b1)``.
  All branches share one grid.
* stage 2 -- ``sum_branches(W2 @ h + b2)`` as a single pass over the branches, so
  the three residual additions of the combined embedder disappear.

The two stages normally run as *one* cooperative launch with a grid-wide barrier
between them, because at batch 1 a kernel launch costs as much as several
microseconds of real work.  Both use one warp per output row with several 16-byte
loads per warp in flight; that memory-level parallelism is what brings the
mat-vec up to the achievable streaming bandwidth for this small working set.

Every entry point falls back to the reference implementation for the
shapes/dtypes/devices the kernels do not cover, and the Python side caches a
closure per module so the hot path is one guard plus one extension call.
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn as nn

from ..L1.linear import Linear
from ..L1.silu import SiLU

# ---------------------------------------------------------------------------
# CUDA kernels
# ---------------------------------------------------------------------------

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <vector>

namespace {

namespace cg = cooperative_groups;

typedef __nv_bfloat16 bf16;
typedef __nv_bfloat162 bf162;

// log2(10000)/128.  The reference computes exp(-log(10000)*j/denom); folding the
// base change in lets the per-frequency scale be a single MUFU.EX2.
#define FK_LOG2P (0.1038102529652301f)
#define FK_MAX_SEG 4

__device__ __forceinline__ uint4 fk_ld(const void* p) {
  uint4 v;
  asm("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"
      : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}

__device__ __forceinline__ float fk_warp_sum(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
  return v;
}

// acc += dot(8 bf16 of wv, 8 bf16 of hv); two accumulators shorten the chain.
__device__ __forceinline__ void fk_fma8(const uint4& wv, const uint4& hv,
                                        float& a0, float& a1) {
  const bf162* wq = reinterpret_cast<const bf162*>(&wv);
  const bf162* hq = reinterpret_cast<const bf162*>(&hv);
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    const float2 wf = __bfloat1622float2(wq[q]);
    const float2 hf = __bfloat1622float2(hq[q]);
    a0 = fmaf(wf.x, hf.x, a0);
    a1 = fmaf(wf.y, hf.y, a1);
  }
}

__device__ __forceinline__ float fk_rbf(float v) {  // round-trip through bf16
  return __bfloat162float(__float2bfloat16(v));
}

template <typename OT> __device__ __forceinline__ OT fk_cvt(float v);
template <> __device__ __forceinline__ float fk_cvt<float>(float v) { return v; }
template <> __device__ __forceinline__ bf16 fk_cvt<bf16>(float v) { return __float2bfloat16(v); }

// ---------------------------------------------------------------- sinusoid
// ``exp(-log(10000)*j/denom)`` is evaluated as a single MUFU.EX2; that is within
// one ulp of torch's ``exp`` and the comparison tolerance for fp32 is 1e-5/1e-3.
// ``sincosf`` (not ``__sinf``) keeps the angle accurate for any timestep
// magnitude.
__device__ __forceinline__ float fk_freq_step(float denom) {
  return FK_LOG2P * (128.0f / denom);
}

template <typename OT>
__global__ void sinusoid_kernel(const bf16* __restrict__ t, OT* __restrict__ out,
                                int half, int channels, float denom, float scale,
                                int flip) {
  const int b = blockIdx.x;
  const float tv = __bfloat162float(t[b]);
  const float step = fk_freq_step(denom);
  OT* o = out + (size_t)b * (size_t)channels;
  for (int j = threadIdx.x; j < half; j += blockDim.x) {
    const float a = (tv * exp2f(-(float)j * step)) * scale;
    float sv, cv;
    sincosf(a, &sv, &cv);
    // emb = cat([sin, cos]); flip_sin_to_cos swaps the two halves.
    o[flip ? j : (half + j)] = fk_cvt<OT>(cv);
    o[flip ? (half + j) : j] = fk_cvt<OT>(sv);
  }
}

// ---------------------------------------------------------------- stage 1
struct Seg1 {
  const bf16* w;    // [M, K]
  const bf16* b;    // [M] or nullptr
  const bf16* x;    // [K] dense input; nullptr => sinusoid of *tsc*
  const bf16* tsc;  // scalar timestep
  bf16* out;        // [M]
  int K;
  int M;
};
// ``bstart`` splits the fused grid between branches in proportion to their
// stage-1 work, so a wide branch does not become the critical path.
struct Args1 { Seg1 s[FK_MAX_SEG]; int bstart[FK_MAX_SEG + 1]; };

// silu(W1 @ x + b1) for one branch. *gidx*/*gcnt* are this block's index and
// count within the set of blocks assigned to the branch. One warp owns R
// consecutive rows so its R weight loads are independent.
template <int R>
__device__ void stage1_body(const Seg1 S, int gidx, int gcnt, void* smem, int half,
                            float denom, float scale, int flip) {
  bf16* xs = reinterpret_cast<bf16*>(smem);
  const int tid = threadIdx.x;
  const int nw = blockDim.x >> 5;

  if (S.x == nullptr) {
    const float tv = __bfloat162float(*S.tsc);
    const float step = fk_freq_step(denom);
    for (int j = tid; j < half; j += blockDim.x) {
      const float ang = (tv * exp2f(-(float)j * step)) * scale;
      float sv, cv;
      sincosf(ang, &sv, &cv);
      xs[flip ? j : (half + j)] = __float2bfloat16(cv);
      xs[flip ? (half + j) : j] = __float2bfloat16(sv);
    }
  } else {
    const uint4* src = reinterpret_cast<const uint4*>(S.x);
    uint4* dst = reinterpret_cast<uint4*>(xs);
    for (int i = tid; i < (S.K >> 3); i += blockDim.x) dst[i] = src[i];
  }
  __syncthreads();

  const int lane = tid & 31;
  const int nchunk = S.K >> 3;
  const uint4* hp = reinterpret_cast<const uint4*>(xs);
  const int stride = gcnt * nw * R;

  for (int base = (gidx * nw + (tid >> 5)) * R; base < S.M; base += stride) {
    const uint4* wp[R];
    float a0[R], a1[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const int rr = (base + r < S.M) ? (base + r) : (S.M - 1);
      wp[r] = reinterpret_cast<const uint4*>(S.w + (size_t)rr * (size_t)S.K);
      a0[r] = 0.f;
      a1[r] = 0.f;
    }
    for (int c = lane; c < nchunk; c += 32) {
      const uint4 hv = hp[c];
      uint4 wv[R];
#pragma unroll
      for (int r = 0; r < R; ++r) wv[r] = fk_ld(wp[r] + c);
#pragma unroll
      for (int r = 0; r < R; ++r) fk_fma8(wv[r], hv, a0[r], a1[r]);
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const float acc = fk_warp_sum(a0[r] + a1[r]);
      const int row = base + r;
      if (lane == 0 && row < S.M) {
        // The reference rounds the linear output to bf16 before the activation.
        const float v = fk_rbf(acc + (S.b ? __bfloat162float(S.b[row]) : 0.f));
        S.out[row] = __float2bfloat16(v * (1.f / (1.f + expf(-v))));
      }
    }
  }
}

template <int R>
__global__ void stage1_kernel(const Args1 a, int half, float denom, float scale,
                              int flip) {
  extern __shared__ __align__(16) char fk_sm[];
  stage1_body<R>(a.s[blockIdx.y], blockIdx.x, gridDim.x, fk_sm, half, denom, scale,
                 flip);
}

// ---------------------------------------------------------------- stage 2
struct Seg2 {
  const bf16* w;  // [N, K]
  const bf16* b;  // [N] or nullptr
  int K;
  int off;        // start of this branch inside the packed activation vector
};
struct Args2 { Seg2 s[FK_MAX_SEG]; int nseg; };

// out = sum_s bf16(W2_s @ h_s + b2_s), the sum itself carried in bf16 so the
// rounding matches the reference's ``emb_a + emb_b + emb_c``.
//
// One warp owns R consecutive rows and walks K with U independent 16-byte loads
// per row in flight.  U is chosen on the host so that ``(K/8) % (32*U) == 0``: a
// bounds test here would have to guard an inline-asm load, which the compiler can
// only do with a branch -- and that branch serializes the loads, costing ~2x of
// the achievable streaming bandwidth.
//
// Branches are processed one at a time so the shared activation buffer only has
// to hold max(K_s) instead of their concatenation, which keeps occupancy up for
// the 3-branch combined embedder.
template <int R, int U>
__device__ void stage2_body(const Args2 a, const bf16* __restrict__ h,
                            bf16* __restrict__ out, int N, void* smem) {
  uint4* hs = reinterpret_cast<uint4*>(smem);
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int nw = blockDim.x >> 5;
  const int nseg = a.nseg;
  const int stride = gridDim.x * nw * R;
  const int base0 = (blockIdx.x * nw + (tid >> 5)) * R;
  // Uniform trip count: __syncthreads() below must be reached by every warp of
  // the block the same number of times.
  const int iters = (N + stride - 1) / stride;

  for (int it = 0; it < iters; ++it) {
    const int base = base0 + it * stride;
    const bool live = base < N;
    int row0 = live ? base : 0;
    if (row0 > N - R) row0 = N - R;
    float run[R];
    for (int sg = 0; sg < nseg; ++sg) {
      const Seg2 SS = a.s[sg];
      const int nchunk = SS.K >> 3;
      __syncthreads();
      {
        const uint4* src = reinterpret_cast<const uint4*>(h + SS.off);
        for (int i = tid; i < nchunk; i += blockDim.x) hs[i] = src[i];
      }
      __syncthreads();
      const uint4* wp[R];
      float a0[R], a1[R];
#pragma unroll
      for (int r = 0; r < R; ++r) {
        wp[r] = reinterpret_cast<const uint4*>(SS.w + (size_t)(row0 + r) * (size_t)SS.K);
        a0[r] = 0.f;
        a1[r] = 0.f;
      }
      for (int c = lane; c < nchunk; c += 32 * U) {
        uint4 wv[R][U], hv[U];
#pragma unroll
        for (int u = 0; u < U; ++u) {
          const int j = c + 32 * u;
          hv[u] = hs[j];
#pragma unroll
          for (int r = 0; r < R; ++r) wv[r][u] = fk_ld(wp[r] + j);
        }
#pragma unroll
        for (int u = 0; u < U; ++u)
#pragma unroll
          for (int r = 0; r < R; ++r) fk_fma8(wv[r][u], hv[u], a0[r], a1[r]);
      }
#pragma unroll
      for (int r = 0; r < R; ++r) {
        const float acc = fk_warp_sum(a0[r] + a1[r]);
        if (lane == 0) {
          const float o = fk_rbf(acc + (SS.b ? __bfloat162float(SS.b[row0 + r]) : 0.f));
          run[r] = (sg == 0) ? o : fk_rbf(run[r] + o);
        }
      }
    }
    if (lane == 0 && live) {
#pragma unroll
      for (int r = 0; r < R; ++r)
        if (row0 + r < N) out[row0 + r] = __float2bfloat16(run[r]);
    }
  }
}

template <int R, int U>
__global__ void stage2_kernel(const Args2 a, const bf16* __restrict__ h,
                              bf16* __restrict__ out, int N) {
  extern __shared__ __align__(16) char fk_sm[];
  stage2_body<R, U>(a, h, out, N, fk_sm);
}

// Both stages in one launch: at batch 1 a kernel launch costs as much as several
// microseconds of real work, so removing the second one is worth a grid-wide
// barrier.  Launched cooperatively, which also makes the driver reject (rather
// than deadlock on) a grid that cannot be co-resident.
template <int R, int U>
__global__ void fused_kernel(const Args1 a1, const Args2 a2,
                             const bf16* __restrict__ h, bf16* __restrict__ out,
                             int N, int half, float denom, float scale, int flip) {
  extern __shared__ __align__(16) char fk_sm[];
  const int nseg = a2.nseg;
  int sg = 0;
  while (sg + 1 < nseg && (int)blockIdx.x >= a1.bstart[sg + 1]) ++sg;
  stage1_body<R>(a1.s[sg], (int)blockIdx.x - a1.bstart[sg],
                 a1.bstart[sg + 1] - a1.bstart[sg], fk_sm, half, denom, scale, flip);
  cg::this_grid().sync();
  stage2_body<R, U>(a2, h, out, N, fk_sm);
}

// ---------------------------------------------------------------- host side
int fk_env(const char* name, int dflt) {
  const char* v = std::getenv(name);
  if (!v || !*v) return dflt;
  const int x = std::atoi(v);
  return x > 0 ? x : dflt;
}

// Launch-shape tuning (rows per warp, load unroll, block size), picked by a sweep
// on the target GPU; overridable by env var for re-tuning elsewhere.
int g_s1r = fk_env("FK_TE_S1R", 1);
int g_s1b = fk_env("FK_TE_S1B", 256);
int g_s2r = fk_env("FK_TE_S2R", 1);
int g_s2u = fk_env("FK_TE_S2U", 3);
int g_s2b = fk_env("FK_TE_S2B", 256);
int g_fuse = fk_env("FK_TE_FUSE", 1);

int fk_sms() {
  static int n = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  return n;
}

}  // namespace

at::Tensor timestep_sinusoid(const at::Tensor& t, int64_t channels, double denom,
                             double scale, bool flip, bool out_float) {
  TORCH_CHECK(t.is_cuda() && t.dim() == 1 && t.scalar_type() == at::kBFloat16);
  TORCH_CHECK(channels > 0 && channels % 2 == 0);
  const int B = (int)t.size(0);
  const int half = (int)(channels / 2);
  auto out = at::empty({B, channels},
                       t.options().dtype(out_float ? at::kFloat : at::kBFloat16));
  int threads = 32;
  while (threads < half && threads < 256) threads <<= 1;
  auto stream = at::cuda::getCurrentCUDAStream();
  const bf16* T = (const bf16*)t.data_ptr();
  const float dn = (float)denom, sc = (float)scale;
  const int fl = flip ? 1 : 0;
  if (out_float)
    sinusoid_kernel<float><<<B, threads, 0, stream>>>(
        T, out.data_ptr<float>(), half, (int)channels, dn, sc, fl);
  else
    sinusoid_kernel<bf16><<<B, threads, 0, stream>>>(
        T, (bf16*)out.data_ptr(), half, (int)channels, dn, sc, fl);
  return out;
}

// ``t`` is [x_0..x_{n-1}, w1_0.., b1_0.., w2_0.., b2_0..] (5n tensors) and
// ``meta`` is [nseg, sin_channels, is_sin_0..is_sin_{n-1}].  One flat vector
// keeps the per-call pybind marshalling cheap: at batch 1 the whole op is only a
// few microseconds, so argument conversion is a real term.
at::Tensor fused_embed(const std::vector<at::Tensor>& t,
                       const std::vector<int64_t>& meta,
                       double denom, double scale, bool flip) {
  const int nseg = (int)meta[0];
  const int64_t sin_channels = meta[1];
  TORCH_CHECK(nseg >= 1 && nseg <= FK_MAX_SEG);
  TORCH_CHECK((int)t.size() == 5 * nseg && (int)meta.size() == 2 + nseg);
  const at::Tensor* xs = t.data();
  const at::Tensor* w1 = xs + nseg;
  const at::Tensor* b1 = w1 + nseg;
  const at::Tensor* w2 = b1 + nseg;
  const at::Tensor* b2 = w2 + nseg;
  const int64_t* is_sin = meta.data() + 2;
  for (int i = 0; i < 5 * nseg; ++i)
    TORCH_CHECK(t[i].is_cuda() && t[i].is_contiguous()
                    && t[i].scalar_type() == at::kBFloat16,
                "fused_embed: every tensor must be contiguous CUDA bfloat16");
  const int N = (int)w2[0].size(0);

  int Ktot = 0, maxK = 0, maxM = 0;
  for (int s = 0; s < nseg; ++s) {
    const int M = (int)w1[s].size(0);
    const int K = (int)w1[s].size(1);
    Ktot += M;
    maxK = K > maxK ? K : maxK;
    maxM = M > maxM ? M : maxM;
  }
  // One allocation holds the packed stage-1 activations and the output.
  auto buf = at::empty({Ktot + N}, w1[0].options());
  bf16* hp = (bf16*)buf.data_ptr();
  bf16* op = hp + Ktot;

  Args1 a1;
  Args2 a2;
  a2.nseg = nseg;
  int off = 0;
  for (int s = 0; s < nseg; ++s) {
    const int M = (int)w1[s].size(0);
    Seg1& p = a1.s[s];
    p.w = (const bf16*)w1[s].data_ptr();
    p.b = b1[s].defined() ? (const bf16*)b1[s].data_ptr() : nullptr;
    p.x = is_sin[s] ? nullptr : (const bf16*)xs[s].data_ptr();
    p.tsc = is_sin[s] ? (const bf16*)xs[s].data_ptr() : nullptr;
    p.out = hp + off;
    p.K = (int)w1[s].size(1);
    p.M = M;
    Seg2& q = a2.s[s];
    q.w = (const bf16*)w2[s].data_ptr();
    q.b = b2[s].defined() ? (const bf16*)b2[s].data_ptr() : nullptr;
    q.K = M;
    q.off = off;
    off += M;
    TORCH_CHECK(M % 8 == 0 && p.K % 8 == 0 && (int)w2[s].size(1) == M
                    && (int)w2[s].size(0) == N,
                "fused_embed: unsupported weight shapes");
    TORCH_CHECK(is_sin[s] ? (xs[s].numel() == 1 && sin_channels == p.K)
                          : (xs[s].numel() == p.K),
                "fused_embed: input does not match linear_1");
  }
  for (int s = nseg; s < FK_MAX_SEG; ++s) { a1.s[s] = a1.s[0]; a2.s[s] = a2.s[0]; }

  const int half = (int)(sin_channels / 2);
  auto stream = at::cuda::getCurrentCUDAStream();
  const float dn = (float)denom, sc = (float)scale;
  const int fl = flip ? 1 : 0;

  // Rows per warp must divide N; the load unroll must divide the chunk count of
  // every branch (no tail -> no branch around an asm load).
  int r2 = g_s2r;
  while (r2 > 1 && (N % r2)) r2 >>= 1;
  int u2 = g_s2u;
  while (u2 > 1) {
    bool ok = true;
    for (int s = 0; s < nseg; ++s)
      if ((a2.s[s].K >> 3) % (32 * u2)) { ok = false; break; }
    if (ok) break;
    --u2;
  }
  const size_t smem = (size_t)(maxK > maxM ? maxK : maxM) * sizeof(bf16);

  // --- one cooperative launch for both stages when the grid can be co-resident
  if (g_fuse) {
    const int bl = g_s2b;
    int nblk = (N + r2 * (bl >> 5) - 1) / (r2 * (bl >> 5));
    void* fn = nullptr;
#define FK_PICK(UU) fn = (void*)(r2 == 1 ? (void*)fused_kernel<1, UU>               \
                                : r2 == 4 ? (void*)fused_kernel<4, UU>              \
                                          : (void*)fused_kernel<2, UU>)
    if (u2 == 1) FK_PICK(1);
    else if (u2 == 2) FK_PICK(2);
    else if (u2 == 3) FK_PICK(3);
    else if (u2 == 6) FK_PICK(6);
    else FK_PICK(4);
#undef FK_PICK
    // A cooperative grid must be co-resident; clamp to that (both phases walk
    // their rows grid-strided, so fewer blocks is still correct) and fall back to
    // two plain launches if the occupancy query or the launch itself fails.
    int per_sm = 0;
    const int cap = (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, fn, bl,
                                                                   smem) == cudaSuccess)
                        ? per_sm * fk_sms() : 0;
    if (cap >= nseg && nblk >= nseg) {
      if (nblk > cap) nblk = cap;
      // Split the grid between branches in proportion to stage-1 work (M_s*K_s).
      long long total = 0;
      for (int s = 0; s < nseg; ++s)
        total += (long long)a1.s[s].M * a1.s[s].K;
      long long acc = 0;
      a1.bstart[0] = 0;
      for (int s = 0; s < nseg; ++s) {
        acc += (long long)a1.s[s].M * a1.s[s].K;
        int e = (int)((acc * nblk) / total);
        if (e < a1.bstart[s] + 1) e = a1.bstart[s] + 1;
        if (e > nblk - (nseg - 1 - s)) e = nblk - (nseg - 1 - s);
        a1.bstart[s + 1] = e;
      }
      a1.bstart[nseg] = nblk;
      Args1 la1 = a1;
      Args2 la2 = a2;
      const bf16* ch = hp;
      int lN = N, lhalf = half, lfl = fl;
      float ldn = dn, lsc = sc;
      void* args[] = {&la1, &la2, &ch, &op, &lN, &lhalf, &ldn, &lsc, &lfl};
      if (cudaLaunchCooperativeKernel(fn, dim3(nblk), dim3(bl), args, smem, stream)
          == cudaSuccess) {
        return buf.narrow(0, Ktot, N).view({1, N});
      }
      cudaGetLastError();
    }
  }

  // --- fallback: two ordinary launches
  const int b1t = g_s1b;
  int r1 = g_s1r;
  while (r1 > 1 && (maxM % r1)) r1 >>= 1;
  {
    const int nblk1 = (maxM + r1 * (b1t >> 5) - 1) / (r1 * (b1t >> 5));
#define FK_S1(RR)                                                                \
    stage1_kernel<RR><<<dim3(nblk1, nseg), b1t, smem, stream>>>(a1, half, dn, sc, fl)
    if (r1 == 1)      FK_S1(1);
    else if (r1 == 2) FK_S1(2);
    else if (r1 == 8) FK_S1(8);
    else              FK_S1(4);
#undef FK_S1
  }
  {
    const int b2t = g_s2b;
    const int nblk2 = (N + r2 * (b2t >> 5) - 1) / (r2 * (b2t >> 5));
#define FK_S2(RR, UU) \
    stage2_kernel<RR, UU><<<nblk2, b2t, smem, stream>>>(a2, hp, op, N)
#define FK_S2_U(RR)                                                              \
    do {                                                                         \
      if (u2 == 1) FK_S2(RR, 1);                                                  \
      else if (u2 == 2) FK_S2(RR, 2);                                             \
      else if (u2 == 3) FK_S2(RR, 3);                                             \
      else if (u2 == 6) FK_S2(RR, 6);                                             \
      else FK_S2(RR, 4);                                                          \
    } while (0)
    if (r2 == 1)      FK_S2_U(1);
    else if (r2 == 4) FK_S2_U(4);
    else              FK_S2_U(2);
#undef FK_S2_U
#undef FK_S2
  }
  return buf.narrow(0, Ktot, N).view({1, N});
}
"""

_CPP_SRC = r"""
#include <vector>
at::Tensor timestep_sinusoid(const at::Tensor& t, int64_t channels, double denom,
                             double scale, bool flip, bool out_float);
at::Tensor fused_embed(const std::vector<at::Tensor>& t,
                       const std::vector<int64_t>& meta,
                       double denom, double scale, bool flip);
"""

_EXT = None
_EXT_TRIED = False


def _ext():
    """JIT-build (once) and return the CUDA extension, or ``None`` if unavailable."""
    global _EXT, _EXT_TRIED
    if _EXT_TRIED:
        return _EXT
    _EXT_TRIED = True
    prev = os.environ.get("TORCH_CUDA_ARCH_LIST")
    try:
        if not torch.cuda.is_available():
            return None
        major, minor = torch.cuda.get_device_capability()
        # Pin to the local arch: the ambient multi-arch list both slows the build
        # down ~7x and rejects the sm_100 vector/cache-hint instructions.
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
        from torch.utils.cpp_extension import load_inline

        _EXT = load_inline(
            name="fk_l2_timestep_embedding",
            cpp_sources=_CPP_SRC,
            cuda_sources=_CUDA_SRC,
            functions=["timestep_sinusoid", "fused_embed"],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
    except Exception:  # noqa: BLE001 - any build/runtime problem -> reference path
        if os.environ.get("FK_TE_DEBUG"):
            raise
        _EXT = None
    finally:
        if prev is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = prev
    return _EXT


_MAX_SMEM = 48 * 1024


def _mlp_fast_ok(mods) -> bool:
    """Every ``linear_1``/``linear_2`` pair must be a plain contiguous bf16 GEMV."""
    n_out = None
    for m in mods:
        w1, b1 = m.linear_1.weight, m.linear_1.bias
        w2, b2 = m.linear_2.weight, m.linear_2.bias
        for t in (w1, w2):
            if t.dtype != torch.bfloat16 or not t.is_cuda or not t.is_contiguous():
                return False
        for t in (b1, b2):
            if t is None or t.dtype != torch.bfloat16 or not t.is_contiguous():
                return False
        if w1.dim() != 2 or w2.dim() != 2:
            return False
        if w1.size(1) % 8 or w1.size(0) % 8 or w2.size(1) != w1.size(0):
            return False
        if n_out is None:
            n_out = w2.size(0)
        elif w2.size(0) != n_out:
            return False
    need = max(max(m.linear_1.weight.size(0) for m in mods),
               max(m.linear_1.weight.size(1) for m in mods))
    if need * 2 > _MAX_SMEM:
        return False
    return True


def _build_call(mods, n_sin, sin_channels, denom, scale, in_shapes):
    """Freeze the (constant) weight arguments into a closure.

    The captured module state never changes across calls, so the per-call work
    is one shape/dtype guard plus the extension call -- checking the weights
    again on every forward costs more Python time than the kernels themselves.
    """
    ext = _ext()
    if ext is None:
        return None
    nseg = len(mods)
    weights = ([m.linear_1.weight for m in mods] + [m.linear_1.bias for m in mods]
               + [m.linear_2.weight for m in mods] + [m.linear_2.bias for m in mods])
    meta = [nseg, sin_channels] + [1] * n_sin + [0] * (nseg - n_sin)
    fn = ext.fused_embed
    bf16 = torch.bfloat16

    def call(*inputs, _w=weights, _m=meta, _f=fn, _n=nseg, _sh=in_shapes, _bf=bf16,
             _dn=float(denom), _sc=float(scale)):
        for i in range(_n):
            x = inputs[i]
            if x.shape != _sh[i] or x.dtype is not _bf or not x.is_contiguous():
                return None
        return _f(list(inputs) + _w, _m, _dn, _sc, True)

    return call


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

    if (
        max_period == 10000
        and embedding_dim % 2 == 0
        and timesteps.is_cuda
        and timesteps.dtype == torch.bfloat16
        and timesteps.is_contiguous()
    ):
        ext = _ext()
        if ext is not None:
            return ext.timestep_sinusoid(
                timesteps, embedding_dim,
                float(embedding_dim // 2 - downscale_freq_shift),
                float(scale), bool(flip_sin_to_cos), True,
            )

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

    _fk_call = None
    _fk_planned = False

    def _fk_plan(self, timesteps: torch.Tensor) -> None:
        self._fk_planned = True
        if (self.num_channels % 2 or timesteps.dim() != 1 or not timesteps.is_cuda
                or timesteps.dtype != torch.bfloat16 or not timesteps.is_contiguous()):
            return
        ext = _ext()
        if ext is None:
            return
        fn = ext.timestep_sinusoid
        ch = int(self.num_channels)
        denom = float(ch // 2 - self.downscale_freq_shift)
        scale = float(self.scale)
        flip = bool(self.flip_sin_to_cos)
        shape = timesteps.shape

        def call(t, _f=fn, _c=ch, _d=denom, _s=scale, _fl=flip, _sh=shape,
                 _bf=torch.bfloat16):
            if t.shape != _sh or t.dtype is not _bf or not t.is_contiguous():
                return None
            return _f(t, _c, _d, _s, _fl, True)

        self._fk_call = call

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        if not self._fk_planned:
            self._fk_plan(timesteps)
        call = self._fk_call
        if call is not None:
            out = call(timesteps)
            if out is not None:
                return out
        return get_timestep_embedding(
            timesteps, self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            scale=self.scale,
        )


class TimestepEmbedding(nn.Module):
    """Two-layer MLP that projects sinusoidal timestep encodings."""

    def __init__(self, in_channels: int, time_embed_dim: int, act_fn: str = "silu"):
        super().__init__()
        self.linear_1 = Linear(in_channels, time_embed_dim, bias=True)
        self.act = SiLU()
        self.linear_2 = Linear(time_embed_dim, time_embed_dim, bias=True)

    _fk_call = None
    _fk_planned = False

    def _fk_plan(self, sample: torch.Tensor) -> None:
        self._fk_planned = True
        if (sample.dim() == 2 and sample.size(0) == 1 and sample.is_cuda
                and sample.dtype == torch.bfloat16 and sample.is_contiguous()
                and _mlp_fast_ok((self,))
                and self.linear_1.weight.size(1) == sample.size(1)):
            self._fk_call = _build_call((self,), 0, 0, 1.0, 1.0, (sample.shape,))

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        if not self._fk_planned:
            self._fk_plan(sample)
        call = self._fk_call
        if call is not None:
            out = call(sample)
            if out is not None:
                return out
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)
        return sample


def _combined_plan(scalars, scalar_mlps, pooled, text_mlp, time_proj):
    """Build the fused call for ``sum_i mlp_i(sinusoid(t_i)) + text_mlp(pooled)``.

    Returns ``None`` when the fast path does not apply.
    """
    tp = time_proj
    if tp.num_channels % 2 or not bool(tp.flip_sin_to_cos):
        return None
    if pooled.dim() != 2 or pooled.size(0) != 1 or pooled.dtype != torch.bfloat16:
        return None
    if not pooled.is_cuda or not pooled.is_contiguous():
        return None
    for t in scalars:
        if (t.dim() != 1 or t.size(0) != 1 or t.dtype != torch.bfloat16
                or not t.is_cuda or not t.is_contiguous()):
            return None
    mods = tuple(scalar_mlps) + (text_mlp,)
    if not _mlp_fast_ok(mods):
        return None
    for m in scalar_mlps:
        if m.linear_1.weight.size(1) != tp.num_channels:
            return None
    if text_mlp.linear_1.weight.size(1) != pooled.size(1):
        return None
    shapes = tuple(t.shape for t in scalars) + (pooled.shape,)
    return _build_call(mods, len(scalars), tp.num_channels,
                       tp.num_channels // 2 - tp.downscale_freq_shift, tp.scale,
                       shapes)


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

    _fk_call = None
    _fk_planned = False

    def forward(self, timestep: torch.Tensor, pooled_projection: torch.Tensor) -> torch.Tensor:
        if not self._fk_planned:
            self._fk_planned = True
            self._fk_call = _combined_plan(
                (timestep,), (self.timestep_embedder,),
                pooled_projection, self.text_embedder, self.time_proj)
        call = self._fk_call
        if call is not None:
            out = call(timestep, pooled_projection)
            if out is not None:
                return out
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

    _fk_call = None
    _fk_planned = False

    def forward(self, timestep: torch.Tensor, guidance: torch.Tensor, pooled_projection: torch.Tensor) -> torch.Tensor:
        if not self._fk_planned:
            self._fk_planned = True
            self._fk_call = _combined_plan(
                (timestep, guidance), (self.timestep_embedder, self.guidance_embedder),
                pooled_projection, self.text_embedder, self.time_proj)
        call = self._fk_call
        if call is not None:
            out = call(timestep, guidance, pooled_projection)
            if out is not None:
                return out
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=pooled_projection.dtype))
        guidance_proj = self.time_proj(guidance)
        guidance_emb = self.guidance_embedder(guidance_proj.to(dtype=pooled_projection.dtype))
        pooled_projections = self.text_embedder(pooled_projection)
        return timesteps_emb + guidance_emb + pooled_projections
