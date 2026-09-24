"""SwiGLU transition composites for AlphaFold3 (L2) -- fused CUDA kernels.

The reference composition (LayerNorm/AdaLN -> SwiGLU -> Linear -> gate -> mask)
issues ~20 eager kernels for tensors of a few tens of KB, so it is entirely
launch-latency bound.  This implementation keeps the module structure -- and
therefore the ``state_dict`` keys -- but replaces the forward with three
hand-written bf16 wmma kernels:

  ``SwiGLUTransition``            k_ln  -> k_gate -> k_out
  ``ConditionedTransitionBlock``  k_pre -> k_gate -> k_out

They are submitted as one CUDA graph whose nodes are re-parameterized in place
each call (only the input/output pointers ever change), which removes most of
the per-launch cost of the chain.

The shapes are tiny and very skinny (M = 16..368, K = 64..1536), so the kernels
are organized for parallelism rather than reuse: one 16x16 output tile per
*block*, with the K reduction split across the block's warps and folded through
shared memory.  Weights -- and the ``g`` intermediate -- are stored pre-packed in
wmma fragment order so a fragment load is one coalesced 16B-per-lane fetch
instead of four strided generic loads.

Anything the kernels do not cover (non-bf16, channel counts that are not
multiples of 16, a mask that does not match the row count, no CUDA/nvcc) falls
back to the reference composition.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn

from ..L1.sigmoid import Sigmoid
from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear
from .alphafold3_swiglu import AdaLN, SwiGLU

_CPP_SRC = r"""
#include <torch/extension.h>

int64_t create_swiglu(at::Tensor, at::Tensor, at::Tensor, at::Tensor, int64_t,
                      int64_t, bool);
int64_t create_cond(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                    at::Tensor, at::Tensor, int64_t, int64_t, int64_t, bool);
at::Tensor swiglu_forward(at::Tensor, c10::optional<at::Tensor>, int64_t);
at::Tensor cond_forward(at::Tensor, at::Tensor, c10::optional<at::Tensor>, int64_t);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("create_swiglu", &create_swiglu);
  m.def("create_cond", &create_cond);
  m.def("swiglu_forward", &swiglu_forward);
  m.def("cond_forward", &cond_forward);
}
"""

_CUDA_SRC = r"""// Fused AlphaFold3 SwiGLU-transition kernels (bf16, wmma m16n16k16).
//
// Shapes here are tiny and extremely skinny (M = 16..368, K = 64..1536), so the
// only real lever is parallelism: every kernel computes one 16x16 output tile
// per *block* and splits the K reduction across the block's warps, which turns
// ~100 latency-exposed warps into ~1000 and lets the fragment loads overlap.
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include <map>
#include <memory>

using namespace nvcuda;
using bf16 = __nv_bfloat16;

// wmma bf16 fragments need sm_80+; keep the kernels compilable (as no-ops) for
// the older architectures torch's default TORCH_CUDA_ARCH_LIST still includes.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800)
#define FK_NO_WMMA 1
#endif

#define CUDA_OK(x)                                                    \
  do {                                                                \
    cudaError_t e_ = (x);                                             \
    TORCH_CHECK(e_ == cudaSuccess, "cuda: ", cudaGetErrorString(e_)); \
  } while (0)

// ---------------------------------------------------------------------------
// kernel argument PODs.  These live inside the Plan so graph nodes can be
// re-parameterized in place (pointers change per call, dims never do).
// ---------------------------------------------------------------------------

struct LnArgs {
  const bf16* x;
  const float* w;
  const float* b;
  bf16* xn;
  int N, C;
};

struct GArgs {
  const bf16* xn;   // [Npad, C] (LayerNorm / AdaLN output)
  const bf16* Wab;  // [2H, C] repacked
  bf16* g;          // [Npad, H]
  int N, C, H;
};

struct YArgs {
  const bf16* g;     // [Npad, H]
  const bf16* Wo;    // [C, H] repacked
  const bf16* gate;  // [Npad, C] or null
  const bf16* mask;  // [N] or null
  bf16* out;         // [N, C]
  int N, C, H;
};

struct PreArgs {
  const bf16* a;    // [N, Ca]
  const bf16* s;    // [N, Cs]
  const float* lnsw;
  const bf16* Wgs;  // [2Ca, Cs]
  const float* bg;
  const bf16* Wog;  // [Ca, Cs]
  const float* bog;
  bf16* xn;         // [Npad, Ca]
  bf16* og;         // [Npad, Ca]
  int N, Ca, Cs;
};

// ---------------------------------------------------------------------------
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
__device__ __forceinline__ float sigmoidf_(float x) {
  return __frcp_rn(1.0f + __expf(-x));
}
__device__ __forceinline__ float rbf(float x) {  // round through bf16
  return __bfloat162float(__float2bfloat16(x));
}

// write one 16x16 fp32 tile (row-major, ld 16, in smem) out as bf16 rows
template <typename EPI>
__device__ __forceinline__ void write_tile(const float* tile, bf16* dst, int ld,
                                          int rows, int lane, EPI epi) {
  const int r = lane >> 1;
  const int c0 = (lane & 1) * 8;
  if (r < rows) {
    __align__(16) bf16 v[8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
      v[i] = __float2bfloat16(epi(tile[r * 16 + c0 + i], r, c0 + i));
    *reinterpret_cast<uint4*>(dst + r * ld + c0) = *reinterpret_cast<const uint4*>(v);
  }
}


// --- vectorized (8 bf16 = 16B per lane) row reduction / normalization -------
struct Stats { float mean, rstd; };

__device__ __forceinline__ Stats row_stats(const bf16* src, int L, int lane) {
  const uint4* v = reinterpret_cast<const uint4*>(src);
  const int nv = L >> 3;
  float s1 = 0.f, s2 = 0.f;
  for (int i = lane; i < nv; i += 32) {
    uint4 r = v[i];
    const bf16* h = reinterpret_cast<const bf16*>(&r);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      float x = __bfloat162float(h[j]);
      s1 += x;
      s2 += x * x;
    }
  }
  s1 = warp_sum(s1);
  s2 = warp_sum(s2);
  const float mean = s1 / L;
  Stats st;
  st.mean = mean;
  st.rstd = rsqrtf(fmaxf(s2 / L - mean * mean, 0.f) + 1e-5f);
  return st;
}

using FragA = wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major>;
using FragB = wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::col_major>;
using FragC = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;

// A ``load_matrix_sync`` of a 16x16 bf16 tile out of a row-major matrix costs
// four generic 32-bit loads whose lanes touch eight separate 32B sectors.  The
// weights never change, so they are repacked once (below, *by* load_matrix_sync
// so the register order is exact) into per-tile blocks of 256 bf16 laid out in
// lane/element order -- then a fragment is one fully coalesced 16B-per-lane load.
#ifndef FK_NO_WMMA
__device__ __forceinline__ void load_packed(FragB& b, const bf16* p) {
  const uint4 raw = *reinterpret_cast<const uint4*>(p);
  const bf16* h = reinterpret_cast<const bf16*>(&raw);
#pragma unroll
  for (int i = 0; i < 8; ++i) b.x[i] = h[i];
}
__device__ __forceinline__ void load_packed(FragA& a, const bf16* p) {
  const uint4 raw = *reinterpret_cast<const uint4*>(p);
  const bf16* h = reinterpret_cast<const bf16*>(&raw);
#pragma unroll
  for (int i = 0; i < 8; ++i) a.x[i] = h[i];
}
#endif

// emit a 16x16 bf16 tile (row-major, ld 16, in smem) as a packed FragA block
#ifndef FK_NO_WMMA
__device__ __forceinline__ void store_packed_a(const bf16* tile, bf16* dst,
                                              int lane) {
  FragA a;
  wmma::load_matrix_sync(a, tile, 16);
  __align__(16) bf16 v[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) v[i] = a.x[i];
  *reinterpret_cast<uint4*>(dst + lane * 8) = *reinterpret_cast<const uint4*>(v);
}
#endif

// repack src[N, K] (row-major) -> dst[(N/16)*(K/16)][256], k-tile contiguous
__global__ void k_repack(const bf16* src, bf16* dst, int N, int K) {
#ifndef FK_NO_WMMA
  const int kt = K >> 4;
  const int ntile = (N >> 4) * kt;
  const int lane = threadIdx.x & 31;
  for (int t = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5); t < ntile;
       t += gridDim.x * (blockDim.x >> 5)) {
    const int n0 = (t / kt) << 4;
    const int k0 = (t % kt) << 4;
    FragB b;
    wmma::load_matrix_sync(b, src + (size_t)n0 * K + k0, K);
    __align__(16) bf16 v[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) v[i] = b.x[i];
    *reinterpret_cast<uint4*>(dst + ((size_t)t << 8) + lane * 8) =
        *reinterpret_cast<const uint4*>(v);
  }
#endif
}

// ---------------------------------------------------------------------------
// LayerNorm (fp32 reduction, bf16 out) -> xn[N, C].  One warp per row.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(128) void k_ln(LnArgs p) {
  const int C = p.C;
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * 4 + (threadIdx.x >> 5);
  if (row >= p.N) return;
  const bf16* xr = p.x + (size_t)row * C;
  const Stats st = row_stats(xr, C, lane);
  const uint4* xv = reinterpret_cast<const uint4*>(xr);
  uint4* ov = reinterpret_cast<uint4*>(p.xn + (size_t)row * C);
  const int nv = C >> 3;
  for (int i = lane; i < nv; i += 32) {
    uint4 r = xv[i];
    const bf16* h = reinterpret_cast<const bf16*>(&r);
    const float* wp = p.w + i * 8;
    const float* bp = p.b + i * 8;
    __align__(16) bf16 o[8];
#pragma unroll
    for (int j = 0; j < 8; ++j)
      o[j] = __float2bfloat16((__bfloat162float(h[j]) - st.mean) * st.rstd * wp[j] +
                              bp[j]);
    ov[i] = *reinterpret_cast<const uint4*>(o);
  }
}

// ---------------------------------------------------------------------------
// SwiGLU: g = silu(xn @ Wa^T) * (xn @ Wb^T)   ->  g[N, H]
//   grid (H/16, ceil(N/16)), block 32*W, W warps split the C reduction
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(512) void k_gate(GArgs p) {
#ifndef FK_NO_WMMA
  extern __shared__ __align__(16) float sRed[];  // [2][W][256]
  const int C = p.C, H = p.H;
  const int nw = blockDim.x >> 5;
  const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int m0 = blockIdx.y * 16, h0 = blockIdx.x * 16;
  const int rows = min(16, p.N - m0);
  const int per = ((C >> 4) + nw - 1) / nw;
  const int k0 = w * per * 16;
  const int k1 = min(C, k0 + per * 16);

  FragC au, av;
  wmma::fill_fragment(au, 0.f);
  wmma::fill_fragment(av, 0.f);
  const int kt = C >> 4;
  const bf16* A = p.xn + (size_t)m0 * C;
  const bf16* Wa = p.Wab + (((size_t)(h0 >> 4) * kt) << 8) + lane * 8;
  const bf16* Wb = p.Wab + (((size_t)(((H + h0) >> 4)) * kt) << 8) + lane * 8;
  int k = k0;
  // Distinct fragment variables (not an indexed array) so ptxas batches a whole
  // chunk of loads ahead of the mma group instead of serializing load->mma.
  for (; k + 64 <= k1; k += 64) {
    FragA a0, a1, a2, a3;
    FragB u0, u1, u2, u3, v0, v1, v2, v3;
    const size_t t = (size_t)(k >> 4) << 8;
    wmma::load_matrix_sync(a0, A + k, C);
    wmma::load_matrix_sync(a1, A + k + 16, C);
    wmma::load_matrix_sync(a2, A + k + 32, C);
    wmma::load_matrix_sync(a3, A + k + 48, C);
    load_packed(u0, Wa + t);
    load_packed(u1, Wa + t + 256);
    load_packed(u2, Wa + t + 512);
    load_packed(u3, Wa + t + 768);
    load_packed(v0, Wb + t);
    load_packed(v1, Wb + t + 256);
    load_packed(v2, Wb + t + 512);
    load_packed(v3, Wb + t + 768);
    wmma::mma_sync(au, a0, u0, au);
    wmma::mma_sync(av, a0, v0, av);
    wmma::mma_sync(au, a1, u1, au);
    wmma::mma_sync(av, a1, v1, av);
    wmma::mma_sync(au, a2, u2, au);
    wmma::mma_sync(av, a2, v2, av);
    wmma::mma_sync(au, a3, u3, au);
    wmma::mma_sync(av, a3, v3, av);
  }
  for (; k + 32 <= k1; k += 32) {
    FragA a0, a1;
    FragB u0, u1, v0, v1;
    const size_t t = (size_t)(k >> 4) << 8;
    wmma::load_matrix_sync(a0, A + k, C);
    wmma::load_matrix_sync(a1, A + k + 16, C);
    load_packed(u0, Wa + t);
    load_packed(u1, Wa + t + 256);
    load_packed(v0, Wb + t);
    load_packed(v1, Wb + t + 256);
    wmma::mma_sync(au, a0, u0, au);
    wmma::mma_sync(av, a0, v0, av);
    wmma::mma_sync(au, a1, u1, au);
    wmma::mma_sync(av, a1, v1, av);
  }
  for (; k < k1; k += 16) {
    FragA a0;
    FragB u0, v0;
    const size_t t = (size_t)(k >> 4) << 8;
    wmma::load_matrix_sync(a0, A + k, C);
    load_packed(u0, Wa + t);
    load_packed(v0, Wb + t);
    wmma::mma_sync(au, a0, u0, au);
    wmma::mma_sync(av, a0, v0, av);
  }
  float* sU = sRed;
  float* sV = sRed + nw * 256;
  bf16* sBf = reinterpret_cast<bf16*>(sRed + 2 * nw * 256);  // 16x16 bf16 staging
  wmma::store_matrix_sync(sU + w * 256, au, 16, wmma::mem_row_major);
  wmma::store_matrix_sync(sV + w * 256, av, 16, wmma::mem_row_major);
  __syncthreads();
  for (int i = threadIdx.x; i < 256; i += blockDim.x) {
    float u = 0.f, v = 0.f;
    for (int j = 0; j < nw; ++j) {
      u += sU[j * 256 + i];
      v += sV[j * 256 + i];
    }
    u = rbf(u);
    v = rbf(v);
    sBf[i] = __float2bfloat16(rbf(u * sigmoidf_(u)) * v);
  }
  __syncthreads();
  // g is only ever consumed as wmma A-fragments by k_out, so store it already
  // packed in fragment order (one coalesced 16B/lane load there).
  if (w == 0)
    store_packed_a(sBf, p.g + ((size_t)(blockIdx.y * gridDim.x + blockIdx.x) << 8), lane);
#endif
}

// ---------------------------------------------------------------------------
// out = (g @ Wo^T) * gate * mask   ->  out[N, C]
//   grid (C/16, ceil(N/16)), block 32*W, W warps split the H reduction
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(512) void k_out(YArgs p) {
#ifndef FK_NO_WMMA
  extern __shared__ __align__(16) float sRed[];  // [W][256]
  __shared__ float sMask[16];
  const int C = p.C, H = p.H;
  const int nw = blockDim.x >> 5;
  const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int m0 = blockIdx.y * 16, c0 = blockIdx.x * 16;
  const int rows = min(16, p.N - m0);
  if (threadIdx.x < 16)
    sMask[threadIdx.x] =
        (p.mask && threadIdx.x < rows) ? __bfloat162float(p.mask[m0 + threadIdx.x]) : 1.0f;

  const int per = ((H >> 4) + nw - 1) / nw;
  const int k0 = w * per * 16;
  const int k1 = min(H, k0 + per * 16);

  FragC acc0, acc1;
  wmma::fill_fragment(acc0, 0.f);
  wmma::fill_fragment(acc1, 0.f);
  const int kt = H >> 4;
  const bf16* ga = p.g + (((size_t)blockIdx.y * kt) << 8) + lane * 8;
  const bf16* wo = p.Wo + (((size_t)(c0 >> 4) * kt) << 8) + lane * 8;
  int k = k0;
  for (; k + 64 <= k1; k += 64) {
    FragA a0, a1, a2, a3;
    FragB b0, b1, b2, b3;
    const size_t t = (size_t)(k >> 4) << 8;
    load_packed(a0, ga + t);
    load_packed(a1, ga + t + 256);
    load_packed(a2, ga + t + 512);
    load_packed(a3, ga + t + 768);
    load_packed(b0, wo + t);
    load_packed(b1, wo + t + 256);
    load_packed(b2, wo + t + 512);
    load_packed(b3, wo + t + 768);
    wmma::mma_sync(acc0, a0, b0, acc0);
    wmma::mma_sync(acc1, a1, b1, acc1);
    wmma::mma_sync(acc0, a2, b2, acc0);
    wmma::mma_sync(acc1, a3, b3, acc1);
  }
  for (; k + 32 <= k1; k += 32) {
    FragA a0, a1;
    FragB b0, b1;
    const size_t t = (size_t)(k >> 4) << 8;
    load_packed(a0, ga + t);
    load_packed(a1, ga + t + 256);
    load_packed(b0, wo + t);
    load_packed(b1, wo + t + 256);
    wmma::mma_sync(acc0, a0, b0, acc0);
    wmma::mma_sync(acc1, a1, b1, acc1);
  }
  for (; k < k1; k += 16) {
    FragA a0;
    FragB b0;
    const size_t t = (size_t)(k >> 4) << 8;
    load_packed(a0, ga + t);
    load_packed(b0, wo + t);
    wmma::mma_sync(acc0, a0, b0, acc0);
  }
#pragma unroll
  for (int i = 0; i < FragC::num_elements; ++i) acc0.x[i] += acc1.x[i];
  wmma::store_matrix_sync(sRed + w * 256, acc0, 16, wmma::mem_row_major);
  __syncthreads();
  for (int i = threadIdx.x; i < 256; i += blockDim.x) {
    float v = 0.f;
    for (int j = 0; j < nw; ++j) v += sRed[j * 256 + i];
    sRed[i] = rbf(v);
  }
  __syncthreads();
  if (w == 0) {
    const int base = m0 * C + c0;
    if (p.gate) {
      const bf16* gp = p.gate + base;
      write_tile(sRed, p.out + base, C, rows, lane,
                 [=] __device__(float v, int r, int c) {
                   return rbf(v * __bfloat162float(gp[r * C + c])) * sMask[r];
                 });
    } else {
      write_tile(sRed, p.out + base, C, rows, lane,
                 [=] __device__(float v, int r, int) { return v * sMask[r]; });
    }
  }
#endif
}

// ---------------------------------------------------------------------------
// AdaLN pre-stage:
//   s_norm = LN(s) * w ; gate = sigmoid(Wg s_norm + bg)
//   xn = gate * (LN(a) + Ws s_norm) ; og = sigmoid(Wog s + bog)
//   grid (Ca/16, ceil(N/16)), block 32*W, W warps split the Cs reduction
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(512) void k_pre(PreArgs p) {
#ifndef FK_NO_WMMA
  extern __shared__ __align__(16) char smem_raw[];
  const int Ca = p.Ca, Cs = p.Cs;
  const int lds = Cs + 8;
  bf16* sS = reinterpret_cast<bf16*>(smem_raw);  // LN(s)*w
  bf16* sR = sS + 16 * lds;                      // raw s
  float* sRed = reinterpret_cast<float*>(sR + 16 * lds);  // [2][W][256]
  __shared__ float sMean[16], sRstd[16];

  const int nw = blockDim.x >> 5;
  const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int m0 = blockIdx.y * 16, j0 = blockIdx.x * 16;
  const int rows = min(16, p.N - m0);

  for (int r = w; r < 16; r += nw) {
    const int nvs = Cs >> 3;
    if (r < rows) {
      const bf16* ar = p.a + (size_t)(m0 + r) * Ca;
      const Stats sa = row_stats(ar, Ca, lane);
      if (lane == 0) {
        sMean[r] = sa.mean;
        sRstd[r] = sa.rstd;
      }
      const bf16* sr = p.s + (size_t)(m0 + r) * Cs;
      const Stats ss = row_stats(sr, Cs, lane);
      const uint4* sv = reinterpret_cast<const uint4*>(sr);
      uint4* dn = reinterpret_cast<uint4*>(sS + r * lds);
      uint4* dr = reinterpret_cast<uint4*>(sR + r * lds);
      for (int i = lane; i < nvs; i += 32) {
        uint4 raw = sv[i];
        const bf16* h = reinterpret_cast<const bf16*>(&raw);
        const float* wf = reinterpret_cast<const float*>(p.lnsw + i * 8);
        __align__(16) bf16 o[8];
#pragma unroll
        for (int j = 0; j < 8; ++j)
          o[j] = __float2bfloat16((__bfloat162float(h[j]) - ss.mean) * ss.rstd * wf[j]);
        dn[i] = *reinterpret_cast<const uint4*>(o);
        dr[i] = raw;
      }
    } else {
      uint4* dn = reinterpret_cast<uint4*>(sS + r * lds);
      uint4* dr = reinterpret_cast<uint4*>(sR + r * lds);
      const uint4 z = make_uint4(0, 0, 0, 0);
      for (int i = lane; i < nvs; i += 32) {
        dn[i] = z;
        dr[i] = z;
      }
    }
  }
  __syncthreads();

  const int per = ((Cs >> 4) + nw - 1) / nw;
  const int k0 = w * per * 16;
  const int k1 = min(Cs, k0 + per * 16);
  FragC ag, as_, ao;
  wmma::fill_fragment(ag, 0.f);
  wmma::fill_fragment(as_, 0.f);
  wmma::fill_fragment(ao, 0.f);
  const int kt = Cs >> 4;
  const bf16* Wg = p.Wgs + (((size_t)(j0 >> 4) * kt) << 8) + lane * 8;
  const bf16* Ws = p.Wgs + (((size_t)(((Ca + j0) >> 4)) * kt) << 8) + lane * 8;
  const bf16* Wo = p.Wog + (((size_t)(j0 >> 4) * kt) << 8) + lane * 8;
  int k = k0;
  for (; k + 32 <= k1; k += 32) {
    FragA n0, n1, r0, r1;
    FragB g0, g1, s0, s1, o0, o1;
    const size_t t = (size_t)(k >> 4) << 8;
    wmma::load_matrix_sync(n0, sS + k, lds);
    wmma::load_matrix_sync(n1, sS + k + 16, lds);
    wmma::load_matrix_sync(r0, sR + k, lds);
    wmma::load_matrix_sync(r1, sR + k + 16, lds);
    load_packed(g0, Wg + t);
    load_packed(g1, Wg + t + 256);
    load_packed(s0, Ws + t);
    load_packed(s1, Ws + t + 256);
    load_packed(o0, Wo + t);
    load_packed(o1, Wo + t + 256);
    wmma::mma_sync(ag, n0, g0, ag);
    wmma::mma_sync(as_, n0, s0, as_);
    wmma::mma_sync(ao, r0, o0, ao);
    wmma::mma_sync(ag, n1, g1, ag);
    wmma::mma_sync(as_, n1, s1, as_);
    wmma::mma_sync(ao, r1, o1, ao);
  }
  for (; k < k1; k += 16) {
    FragA n0, r0;
    FragB g0, s0, o0;
    const size_t t = (size_t)(k >> 4) << 8;
    wmma::load_matrix_sync(n0, sS + k, lds);
    wmma::load_matrix_sync(r0, sR + k, lds);
    load_packed(g0, Wg + t);
    load_packed(s0, Ws + t);
    load_packed(o0, Wo + t);
    wmma::mma_sync(ag, n0, g0, ag);
    wmma::mma_sync(as_, n0, s0, as_);
    wmma::mma_sync(ao, r0, o0, ao);
  }

  float* sG = sRed;
  float* sX = sRed + nw * 256;
  wmma::store_matrix_sync(sG + w * 256, ag, 16, wmma::mem_row_major);
  wmma::store_matrix_sync(sX + w * 256, as_, 16, wmma::mem_row_major);
  __syncthreads();
  for (int i = threadIdx.x; i < 256; i += blockDim.x) {
    float gv = 0.f, sv = 0.f;
    for (int j = 0; j < nw; ++j) {
      gv += sG[j * 256 + i];
      sv += sX[j * 256 + i];
    }
    const int r = i >> 4, c = i & 15;
    const float gate = rbf(sigmoidf_(rbf(gv + p.bg[j0 + c])));
    const float anorm =
        (r < rows) ? rbf((__bfloat162float(p.a[(size_t)(m0 + r) * Ca + j0 + c]) -
                          sMean[r]) * sRstd[r])
                   : 0.f;
    sG[i] = gate * rbf(anorm + rbf(sv));
  }
  __syncthreads();
  if (w == 0)
    write_tile(sG, p.xn + (size_t)m0 * Ca + j0, Ca, rows, lane,
               [] __device__(float v, int, int) { return v; });
  __syncthreads();
  wmma::store_matrix_sync(sG + w * 256, ao, 16, wmma::mem_row_major);
  __syncthreads();
  for (int i = threadIdx.x; i < 256; i += blockDim.x) {
    float ov = 0.f;
    for (int j = 0; j < nw; ++j) ov += sG[j * 256 + i];
    sX[i] = rbf(sigmoidf_(rbf(ov + p.bog[j0 + (i & 15)])));
  }
  __syncthreads();
  if (w == 0)
    write_tile(sX, p.og + (size_t)m0 * Ca + j0, Ca, rows, lane,
               [] __device__(float v, int, int) { return v; });
#endif
}

// ---------------------------------------------------------------------------
// host side
// ---------------------------------------------------------------------------
namespace {

int env_int(const char* name) {
  const char* e = getenv(name);
  return e ? atoi(e) : 0;
}

// Warps per block.  Measured: a wide block (256/512 threads) wins even when the
// K-split leaves some warps idle, because the cross-warp reduction and the
// epilogue are spread over more threads.  Env knobs exist for re-tuning.
int warps_for(int fallback, const char* knob) {
  const int forced = env_int(knob);
  const int all = env_int("FK_AF3_WARPS");
  if (forced > 0) return forced;
  if (all > 0) return all;
  return fallback;
}

struct Node {
  cudaGraphNode_t node{};
  cudaKernelNodeParams params{};
};

struct Plan {
  int N = 0, Npad = 0;
  at::Tensor xn, gbuf, ogbuf;
  LnArgs ln{};
  PreArgs pre{};
  GArgs gate{};
  YArgs out{};
  void* ln_kp[1]{};
  void* pre_kp[1]{};
  void* gate_kp[1]{};
  void* out_kp[1]{};
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
  std::vector<Node> nodes;
  ~Plan() {
    if (exec) cudaGraphExecDestroy(exec);
    if (graph) cudaGraphDestroy(graph);
  }
};

struct Ctx {
  int C = 0, H = 0, Cs = 0;
  bool cond = false;
  at::Tensor lnw, lnb, Wab, Wo, lnsw, Wgs, bg, Wog, bog;
  bool use_graph = true;
  std::map<int, std::unique_ptr<Plan>> plans;
};

std::vector<std::unique_ptr<Ctx>> g_ctxs;

void fill_plan(Ctx& c, Plan& pl, int N, void* out_ptr) {
  auto opt = c.Wab.options();
  pl.N = N;
  pl.Npad = ((N + 15) / 16) * 16;
  pl.gbuf = at::zeros({(int64_t)pl.Npad, (int64_t)c.H}, opt);
  pl.xn = at::zeros({(int64_t)pl.Npad, (int64_t)c.C}, opt);
  if (c.cond) {
    pl.ogbuf = at::zeros({(int64_t)pl.Npad, (int64_t)c.C}, opt);
    pl.pre.lnsw = c.lnsw.data_ptr<float>();
    pl.pre.Wgs = (const bf16*)c.Wgs.data_ptr();
    pl.pre.bg = c.bg.data_ptr<float>();
    pl.pre.Wog = (const bf16*)c.Wog.data_ptr();
    pl.pre.bog = c.bog.data_ptr<float>();
    pl.pre.xn = (bf16*)pl.xn.data_ptr();
    pl.pre.og = (bf16*)pl.ogbuf.data_ptr();
    pl.pre.N = N;
    pl.pre.Ca = c.C;
    pl.pre.Cs = c.Cs;
    pl.out.gate = (const bf16*)pl.ogbuf.data_ptr();
  } else {
    pl.ln.w = c.lnw.data_ptr<float>();
    pl.ln.b = c.lnb.data_ptr<float>();
    pl.ln.xn = (bf16*)pl.xn.data_ptr();
    pl.ln.N = N;
    pl.ln.C = c.C;
    pl.out.gate = nullptr;
  }
  pl.gate.xn = (const bf16*)pl.xn.data_ptr();
  pl.gate.Wab = (const bf16*)c.Wab.data_ptr();
  pl.gate.g = (bf16*)pl.gbuf.data_ptr();
  pl.gate.N = N;
  pl.gate.C = c.C;
  pl.gate.H = c.H;
  pl.out.g = (const bf16*)pl.gbuf.data_ptr();
  pl.out.Wo = (const bf16*)c.Wo.data_ptr();
  pl.out.out = (bf16*)out_ptr;
  pl.out.N = N;
  pl.out.C = c.C;
  pl.out.H = c.H;
  pl.ln_kp[0] = &pl.ln;
  pl.pre_kp[0] = &pl.pre;
  pl.gate_kp[0] = &pl.gate;
  pl.out_kp[0] = &pl.out;
}

size_t smem_pre(int Cs, int w) {
  return 32 * (size_t)(Cs + 8) * 2 + 2 * (size_t)w * 256 * 4;
}

void kparams(cudaKernelNodeParams& p, void* func, dim3 grid, int threads,
             size_t smem, void** kp) {
  p.func = func;
  p.gridDim = grid;
  p.blockDim = dim3(threads, 1, 1);
  p.sharedMemBytes = (unsigned)smem;
  p.kernelParams = kp;
  p.extra = nullptr;
}

// the three/four stages of one forward, in dependency order
void stage_params(Ctx& c, Plan& pl, std::vector<cudaKernelNodeParams>& out) {
  const int mt = (pl.N + 15) / 16;
  if (c.cond) {
    int w = warps_for(8, "FK_AF3_W_PRE");
    while (w > 1 && smem_pre(c.Cs, w) > 47 * 1024) w >>= 1;
    cudaKernelNodeParams p{};
    kparams(p, (void*)k_pre, dim3(c.C / 16, mt, 1), 32 * w,
            smem_pre(c.Cs, w), pl.pre_kp);
    out.push_back(p);
  } else {
    cudaKernelNodeParams p{};
    kparams(p, (void*)k_ln, dim3((pl.N + 3) / 4, 1, 1), 128, 0, pl.ln_kp);
    out.push_back(p);
  }
  {
    const int w = warps_for(8, "FK_AF3_W_GATE");
    cudaKernelNodeParams p{};
    kparams(p, (void*)k_gate, dim3(c.H / 16, mt, 1), 32 * w,
            2 * (size_t)w * 256 * 4 + 256 * 2, pl.gate_kp);
    out.push_back(p);
  }
  {
    const int w = warps_for(c.H >= 512 ? 16 : 8, "FK_AF3_W_OUT");
    cudaKernelNodeParams p{};
    kparams(p, (void*)k_out, dim3(c.C / 16, mt, 1), 32 * w,
            (size_t)w * 256 * 4, pl.out_kp);
    out.push_back(p);
  }
}

void build_graph(Ctx& c, Plan& pl) {
  std::vector<cudaKernelNodeParams> ps;
  stage_params(c, pl, ps);
  CUDA_OK(cudaGraphCreate(&pl.graph, 0));
  cudaGraphNode_t prev = nullptr;
  for (auto& p : ps) {
    Node n{};
    n.params = p;
    CUDA_OK(cudaGraphAddKernelNode(&n.node, pl.graph, prev ? &prev : nullptr,
                                   prev ? 1 : 0, &n.params));
    prev = n.node;
    pl.nodes.push_back(n);
  }
  CUDA_OK(cudaGraphInstantiateWithFlags(&pl.exec, pl.graph, 0));
}

void launch_stream(Ctx& c, Plan& pl, cudaStream_t st) {
  std::vector<cudaKernelNodeParams> ps;
  stage_params(c, pl, ps);
  for (auto& p : ps)
    CUDA_OK(cudaLaunchKernel(p.func, p.gridDim, p.blockDim, p.kernelParams,
                             p.sharedMemBytes, st));
}

Plan& get_plan(Ctx& c, int N, void* out_ptr) {
  auto it = c.plans.find(N);
  if (it == c.plans.end()) {
    auto pl = std::make_unique<Plan>();
    fill_plan(c, *pl, N, out_ptr);
    if (c.use_graph) build_graph(c, *pl);
    it = c.plans.emplace(N, std::move(pl)).first;
  }
  return *it->second;
}

void run(Ctx& c, Plan& pl) {
  cudaStream_t st = c10::cuda::getCurrentCUDAStream();
  if (!c.use_graph) {
    launch_stream(c, pl, st);
    return;
  }
  // only the first (input ptrs) and last (mask + out ptrs) nodes change
  Node& first = pl.nodes.front();
  Node& last = pl.nodes.back();
  CUDA_OK(cudaGraphExecKernelNodeSetParams(pl.exec, first.node, &first.params));
  CUDA_OK(cudaGraphExecKernelNodeSetParams(pl.exec, last.node, &last.params));
  CUDA_OK(cudaGraphLaunch(pl.exec, st));
}

inline at::Tensor cont(const at::Tensor& t) {
  return t.is_contiguous() ? t : t.contiguous();
}

// one-time: [N, K] row-major bf16 -> per-16x16-tile fragment-order blocks
at::Tensor repack(const at::Tensor& w) {
  const int N = (int)w.size(0), K = (int)w.size(1);
  at::Tensor out = at::empty({(int64_t)N * K}, w.options());
  const int tiles = (N / 16) * (K / 16);
  const int blocks = std::min(1024, (tiles + 7) / 8);
  k_repack<<<blocks, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
      (const bf16*)w.data_ptr(), (bf16*)out.data_ptr(), N, K);
  CUDA_OK(cudaGetLastError());
  return out;
}

}  // namespace

int64_t create_swiglu(at::Tensor lnw, at::Tensor lnb, at::Tensor Wab, at::Tensor Wo,
                      int64_t C, int64_t H, bool use_graph) {
  auto c = std::make_unique<Ctx>();
  c->cond = false;
  c->lnw = lnw;
  c->lnb = lnb;
  c->Wab = repack(Wab);
  c->Wo = repack(Wo);
  c->C = (int)C;
  c->H = (int)H;
  c->use_graph = use_graph;
  g_ctxs.push_back(std::move(c));
  return (int64_t)(g_ctxs.size() - 1);
}

int64_t create_cond(at::Tensor lnsw, at::Tensor Wgs, at::Tensor bg, at::Tensor Wog,
                    at::Tensor bog, at::Tensor Wab, at::Tensor Wo, int64_t Ca,
                    int64_t Cs, int64_t H, bool use_graph) {
  auto c = std::make_unique<Ctx>();
  c->cond = true;
  c->lnsw = lnsw;
  c->Wgs = repack(Wgs);
  c->bg = bg;
  c->Wog = repack(Wog);
  c->bog = bog;
  c->Wab = repack(Wab);
  c->Wo = repack(Wo);
  c->C = (int)Ca;
  c->Cs = (int)Cs;
  c->H = (int)H;
  c->use_graph = use_graph;
  g_ctxs.push_back(std::move(c));
  return (int64_t)(g_ctxs.size() - 1);
}

at::Tensor swiglu_forward(at::Tensor x, c10::optional<at::Tensor> mask, int64_t h) {
  Ctx& c = *g_ctxs[h];
  auto xc = cont(x);
  const int N = (int)(xc.numel() / c.C);
  at::Tensor out = at::empty(x.sizes(), x.options());
  Plan& pl = get_plan(c, N, out.data_ptr());
  pl.ln.x = (const bf16*)xc.data_ptr();
  pl.out.out = (bf16*)out.data_ptr();
  at::Tensor mc;
  if (mask.has_value()) {
    mc = cont(*mask);
    pl.out.mask = (const bf16*)mc.data_ptr();
  } else {
    pl.out.mask = nullptr;
  }
  run(c, pl);
  return out;
}

at::Tensor cond_forward(at::Tensor a, at::Tensor s, c10::optional<at::Tensor> mask,
                        int64_t h) {
  Ctx& c = *g_ctxs[h];
  auto ac = cont(a), sc = cont(s);
  const int N = (int)(ac.numel() / c.C);
  at::Tensor out = at::empty(a.sizes(), a.options());
  Plan& pl = get_plan(c, N, out.data_ptr());
  pl.pre.a = (const bf16*)ac.data_ptr();
  pl.pre.s = (const bf16*)sc.data_ptr();
  pl.out.out = (bf16*)out.data_ptr();
  at::Tensor mc;
  if (mask.has_value()) {
    mc = cont(*mask);
    pl.out.mask = (const bf16*)mc.data_ptr();
  } else {
    pl.out.mask = nullptr;
  }
  run(c, pl);
  return out;
}
"""

_MOD = None
_LOAD_FAILED = False


def _use_graph():
    """CUDA-graph submission; set FK_AF3_GRAPH=0 to launch into the stream."""
    return os.environ.get("FK_AF3_GRAPH", "1") == "1"


def _load():
    """JIT-build (once) and return the fused extension, or None if unavailable."""
    global _MOD, _LOAD_FAILED
    if _MOD is not None or _LOAD_FAILED:
        return _MOD
    try:
        from torch.utils.cpp_extension import load_inline

        # Pin the build to the local architecture (same convention as
        # ``fastkernels.infra.cuda_ext``): torch's default list spans sm_75..sm_120,
        # which both slows the one-time JIT ~6x and drags in pre-Ampere passes
        # where the bf16 wmma fragments do not exist.
        override = os.environ.get("FASTKERNELS_CUDA_ARCH_LIST")
        if override is None:
            major, minor = torch.cuda.get_device_capability()
            os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
        elif override.strip():
            os.environ["TORCH_CUDA_ARCH_LIST"] = override
        _MOD = load_inline(
            name="fk_af3_swiglu_transition",
            cpp_sources=[_CPP_SRC],
            cuda_sources=[_CUDA_SRC],
            extra_cflags=["-O3"],
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                "--expt-extended-lambda",
                "--expt-relaxed-constexpr",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "-U__CUDA_NO_BFLOAT16_OPERATORS__",
            ],
            verbose=False,
        )
    except Exception:  # noqa: BLE001 -- no nvcc / unsupported arch: use reference
        _LOAD_FAILED = True
        _MOD = None
    return _MOD


def _f32(t, n, device):
    return (torch.ones(n, dtype=torch.float32, device=device) if t is None
            else t.detach().float().contiguous())


def _zeros32(t, n, device):
    return (torch.zeros(n, dtype=torch.float32, device=device) if t is None
            else t.detach().float().contiguous())


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
        self._h = None
        self._keep = None

    # -- fused path ---------------------------------------------------------
    def _setup(self, x: torch.Tensor) -> int:
        wo = self.linear_out.weight
        c_in, hidden = wo.shape[0], wo.shape[1]
        if (c_in % 16 or hidden % 16 or not wo.is_cuda
                or wo.dtype is not torch.bfloat16):
            return -1
        ext = _load()
        if ext is None:
            return -1
        dev = wo.device
        lnw = _f32(self.layer_norm.weight, c_in, dev)
        lnb = _zeros32(self.layer_norm.bias, c_in, dev)
        wab = torch.cat([self.swiglu.linear_a.weight.detach(),
                         self.swiglu.linear_b.weight.detach()], 0).contiguous()
        wout = wo.detach().contiguous()
        if wab.dtype is not torch.bfloat16:
            return -1
        self._keep = (lnw, lnb, wab, wout)
        try:
            return ext.create_swiglu(lnw, lnb, wab, wout, c_in, hidden, _use_graph())
        except Exception:  # noqa: BLE001
            return -1

    def _reference(self, x, mask):
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
        h = self._h
        if h is None:
            h = self._h = self._setup(x)
        if (h >= 0 and x.dtype is torch.bfloat16 and x.is_cuda
                and x.shape[-1] == self.c_in
                and (mask is None or mask.numel() * self.c_in == x.numel())):
            return _MOD.swiglu_forward(x, mask, h)
        return self._reference(x, mask)


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
        self.c_a = c_a
        self.c_s = c_s
        self._h = None
        self._keep = None

    def _setup(self) -> int:
        wo = self.linear_out.weight
        c_a, hidden = wo.shape[0], wo.shape[1]
        c_s = self.linear_g.weight.shape[1]
        if (c_a % 16 or hidden % 16 or c_s % 16 or not wo.is_cuda
                or wo.dtype is not torch.bfloat16):
            return -1
        ext = _load()
        if ext is None:
            return -1
        dev = wo.device
        ada = self.layer_norm
        lnsw = _f32(ada.layer_norm_s.weight, c_s, dev)
        wgs = torch.cat([ada.linear_g.weight.detach(),
                         ada.linear_s.weight.detach()], 0).contiguous()
        bg = _zeros32(ada.linear_g.bias, c_a, dev)
        wog = self.linear_g.weight.detach().contiguous()
        bog = _zeros32(self.linear_g.bias, c_a, dev)
        wab = torch.cat([self.swiglu.linear_a.weight.detach(),
                         self.swiglu.linear_b.weight.detach()], 0).contiguous()
        wout = wo.detach().contiguous()
        if wab.dtype is not torch.bfloat16 or wgs.dtype is not torch.bfloat16:
            return -1
        self._keep = (lnsw, wgs, bg, wog, bog, wab, wout)
        try:
            return ext.create_cond(lnsw, wgs, bg, wog, bog, wab, wout,
                                   c_a, c_s, hidden, _use_graph())
        except Exception:  # noqa: BLE001
            return -1

    def _reference(self, a, s, mask):
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
        h = self._h
        if h is None:
            h = self._h = self._setup()
        if (h >= 0 and a.dtype is torch.bfloat16 and s.dtype is torch.bfloat16
                and a.is_cuda and a.shape[-1] == self.c_a
                and s.shape[-1] == self.c_s
                and a.numel() * self.c_s == s.numel() * self.c_a
                and (mask is None or mask.numel() * self.c_a == a.numel())):
            return _MOD.cond_forward(a, s, mask, h)
        return self._reference(a, s, mask)
