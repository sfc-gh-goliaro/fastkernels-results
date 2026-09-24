"""GemmaRMSNorm: RMSNorm where the stored weight is an offset from 1.0.

Semantics are identical to vLLM's ``GemmaRMSNorm`` (which the baseline ports):

  1. the runtime scale is ``(1 + weight)`` rather than ``weight`` -- the
     checkpoint stores values near zero;
  2. the cast back to the input dtype happens *after* the weight multiply,
     ``(x * w).to(orig_dtype)`` instead of ``x.to(orig_dtype) * w``.

Where the time actually goes
----------------------------
Every captured call is ``x:bf16[m, 2048]`` with ``residual=None``, and the
benched row counts are 1 / 26 / 60 / 445 / 16384.  For the four small ones the
whole operator costs less than a single kernel launch, so the baseline loses on
``torch.compile``'s wrapper plus its multi-pass epilogue rather than on
bandwidth; for m=16384 it is purely a streaming-bandwidth problem (64 MiB in,
64 MiB out).  Both ends are served by the same shape: *one* custom kernel,
reached through a direct pybind11 call (no dispatcher, no guard machinery), that
reads ``x`` from HBM exactly once.

Kernel
------
One block per row, 16-byte (8-element) vector accesses, the row held in
registers across the reduction so there is no second read, ``__ldcs``/``__stcs``
cache hints because neither ``x`` nor the output is reused, and fp32 math
throughout (matching the baseline's f32 promotion).  For hidden_size 2048 that
is 128 threads x 2 vectors, which measured best (or tied best) at every benched
row count.  Other hidden sizes get the same kernel at a different width, or a
vectorized / scalar fallback.

The pure-PyTorch path from the baseline is retained for the residual variant,
for inputs the kernel does not cover, and for hosts where the extension cannot
be built.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn

_CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/EmptyTensor.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace {

template <typename T> struct Pk;

template <> struct Pk<__nv_bfloat16> {
  using p2 = __nv_bfloat162;
  static __device__ __forceinline__ float2 tof(p2 v) { return __bfloat1622float2(v); }
  static __device__ __forceinline__ p2 fromf(float2 v) { return __float22bfloat162_rn(v); }
  static __device__ __forceinline__ float to1(__nv_bfloat16 v) { return __bfloat162float(v); }
  static __device__ __forceinline__ __nv_bfloat16 from1(float v) { return __float2bfloat16_rn(v); }
};

template <> struct Pk<__half> {
  using p2 = __half2;
  static __device__ __forceinline__ float2 tof(p2 v) { return __half22float2(v); }
  static __device__ __forceinline__ p2 fromf(float2 v) { return __float22half2_rn(v); }
  static __device__ __forceinline__ float to1(__half v) { return __half2float(v); }
  static __device__ __forceinline__ __half from1(float v) { return __float2half_rn(v); }
};

// One 16B word = 8 half/bfloat16 values.
template <typename T>
union V16 {
  uint4 u;
  typename Pk<T>::p2 p[4];
};

template <typename T>
__device__ __forceinline__ float sq8(const V16<T>& v, float acc) {
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    float2 f = Pk<T>::tof(v.p[j]);
    acc = fmaf(f.x, f.x, acc);
    acc = fmaf(f.y, f.y, acc);
  }
  return acc;
}

// out = x * scale * (1 + w), rounded back to T only at the end (Gemma order).
template <typename T>
__device__ __forceinline__ uint4 scale8(const V16<T>& xv, const V16<T>& wv,
                                        float scale) {
  V16<T> ov;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    float2 f = Pk<T>::tof(xv.p[j]);
    float2 g = Pk<T>::tof(wv.p[j]);
    float2 r;
    r.x = (f.x * scale) * (1.f + g.x);
    r.y = (f.y * scale) * (1.f + g.y);
    ov.p[j] = Pk<T>::fromf(r);
  }
  return ov.u;
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// Block-wide sum broadcast to every thread.  NW == 1 needs no shared memory.
template <int NW>
__device__ __forceinline__ float blk_sum(float v, float* smem) {
  v = warp_sum(v);
  if (NW == 1) return v;
  if ((threadIdx.x & 31) == 0) smem[threadIdx.x >> 5] = v;
  __syncthreads();
  float s = 0.f;
#pragma unroll
  for (int i = 0; i < NW; ++i) s += smem[i];
  return s;
}

// ---------------------------------------------------------------------------
// Main kernel: one block per row, vecs == BLOCK * NVEC.  x is loaded once and
// kept in registers across the reduction.
// ---------------------------------------------------------------------------
template <typename T, int BLOCK, int NVEC>
__global__ __launch_bounds__(BLOCK) void rms_block(
    const uint4* __restrict__ xg, const uint4* __restrict__ wg,
    uint4* __restrict__ og, float inv_n, float eps) {
  constexpr int NW = BLOCK / 32;
  __shared__ float smem[NW > 1 ? NW : 1];

  const long base = (long)blockIdx.x * (BLOCK * NVEC) + threadIdx.x;
  V16<T> xv[NVEC], wv[NVEC];
  // Issue the x and w loads together: w is only needed after the reduction, so
  // fetching it there would expose its (L2-hit) latency between the reduction
  // and the store.
#pragma unroll
  for (int k = 0; k < NVEC; ++k) {
    xv[k].u = __ldcs(&xg[base + k * BLOCK]);
    wv[k].u = wg[threadIdx.x + k * BLOCK];
  }
  float ss = 0.f;
#pragma unroll
  for (int k = 0; k < NVEC; ++k) ss = sq8<T>(xv[k], ss);
  const float scale = rsqrtf(blk_sum<NW>(ss, smem) * inv_n + eps);
#pragma unroll
  for (int k = 0; k < NVEC; ++k)
    __stcs(&og[base + k * BLOCK], scale8<T>(xv[k], wv[k], scale));
}

// Fallback for a vectorizable hidden size with no matching (BLOCK, NVEC) pair:
// same math, but x is re-read for the scaling pass.
template <typename T, int BLOCK>
__global__ __launch_bounds__(BLOCK) void rms_gen(
    const uint4* __restrict__ xg, const uint4* __restrict__ wg,
    uint4* __restrict__ og, float inv_n, float eps, int vecs) {
  constexpr int NW = BLOCK / 32;
  __shared__ float smem[NW > 1 ? NW : 1];
  __shared__ float sscale;
  const long base = (long)blockIdx.x * vecs;
  float ss = 0.f;
  for (int i = threadIdx.x; i < vecs; i += BLOCK) {
    V16<T> xv;
    xv.u = xg[base + i];
    ss = sq8<T>(xv, ss);
  }
  ss = blk_sum<NW>(ss, smem);
  if (threadIdx.x == 0) sscale = rsqrtf(ss * inv_n + eps);
  __syncthreads();
  const float scale = sscale;
  for (int i = threadIdx.x; i < vecs; i += BLOCK) {
    V16<T> xv, wv;
    xv.u = xg[base + i];
    wv.u = wg[i];
    og[base + i] = scale8<T>(xv, wv, scale);
  }
}

// Fallback for a hidden size that is not a multiple of 8.
template <typename T, int BLOCK>
__global__ __launch_bounds__(BLOCK) void rms_scalar(
    const T* __restrict__ xg, const T* __restrict__ wg, T* __restrict__ og,
    float inv_n, float eps, int n) {
  constexpr int NW = BLOCK / 32;
  __shared__ float smem[NW > 1 ? NW : 1];
  __shared__ float sscale;
  const long base = (long)blockIdx.x * n;
  float ss = 0.f;
  for (int i = threadIdx.x; i < n; i += BLOCK) {
    float f = Pk<T>::to1(xg[base + i]);
    ss = fmaf(f, f, ss);
  }
  ss = blk_sum<NW>(ss, smem);
  if (threadIdx.x == 0) sscale = rsqrtf(ss * inv_n + eps);
  __syncthreads();
  const float scale = sscale;
  for (int i = threadIdx.x; i < n; i += BLOCK) {
    float f = Pk<T>::to1(xg[base + i]) * scale;
    og[base + i] = Pk<T>::from1(f * (1.f + Pk<T>::to1(wg[i])));
  }
}

}  // namespace

#define BLK(T, B, NV)                                                       \
  rms_block<T, B, NV><<<(unsigned)m, B, 0, stream>>>(                       \
      (const uint4*)xp, (const uint4*)wp, (uint4*)op, inv_n, epsf)

// Two vectors per thread is the shape that measured best (or tied best) at
// every benched row count on B200: at m=1 it keeps the whole row in one 128-wide
// block, and at m=16384 the two loads in flight per thread are already enough to
// saturate HBM.  Wider-per-thread shapes lose ~1.5 us at m=1; narrower ones lose
// ~8% at m=16384.
template <typename T>
static void launch(const void* xp, const void* wp, void* op, long m, int n,
                   float inv_n, float epsf, cudaStream_t stream) {
  if ((n & 7) == 0) {
    switch (n >> 3) {  // 16B vectors per row
      case 32:   BLK(T, 32, 1);  return;
      case 64:   BLK(T, 32, 2);  return;
      case 128:  BLK(T, 64, 2);  return;
      case 256:  BLK(T, 128, 2); return;
      case 512:  BLK(T, 256, 2); return;
      case 1024: BLK(T, 256, 4); return;
      case 2048: BLK(T, 512, 4); return;
      default: break;
    }
    rms_gen<T, 256><<<(unsigned)m, 256, 0, stream>>>(
        (const uint4*)xp, (const uint4*)wp, (uint4*)op, inv_n, epsf, n >> 3);
    return;
  }
  rms_scalar<T, 256><<<(unsigned)m, 256, 0, stream>>>(
      (const T*)xp, (const T*)wp, (T*)op, inv_n, epsf, n);
}

// Returns an undefined tensor (``None`` on the Python side) for anything the
// kernel does not handle, so the caller can fall back to PyTorch.  Doing the
// checks here rather than in Python keeps the hot path down to one call.
at::Tensor gemma_rms_norm(const at::Tensor& x, const at::Tensor& w, double eps) {
  const auto st = x.scalar_type();
  if ((st != at::kBFloat16 && st != at::kHalf) || w.scalar_type() != st ||
      !x.is_cuda() || !x.is_contiguous() || !w.is_contiguous() ||
      x.dim() < 1 || w.dim() != 1 || x.size(-1) != w.numel()) {
    return at::Tensor();
  }
  const int n = (int)w.numel();
  const long m = x.numel() / n;
  at::Tensor out =
      at::detail::empty_cuda(x.sizes(), st, x.device(), std::nullopt);
  if (m == 0 || n == 0) return out;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const float inv_n = 1.f / (float)n;
  const float epsf = (float)eps;
  if (st == at::kBFloat16) {
    launch<__nv_bfloat16>(x.data_ptr(), w.data_ptr(), out.data_ptr(), m, n,
                          inv_n, epsf, stream);
  } else {
    launch<__half>(x.data_ptr(), w.data_ptr(), out.data_ptr(), m, n, inv_n,
                   epsf, stream);
  }
  return out;
}
'''

_CPP_SRC = r'''
#include <torch/extension.h>
at::Tensor gemma_rms_norm(const at::Tensor& x, const at::Tensor& w, double eps);
'''


def _build_ext():
    if not torch.cuda.is_available():
        return None
    try:
        from torch.utils.cpp_extension import load_inline

        if not os.environ.get("TORCH_CUDA_ARCH_LIST"):
            major, minor = torch.cuda.get_device_capability()
            os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
        return load_inline(
            name="fk_gemma_rms_norm_v3",
            cpp_sources=_CPP_SRC,
            cuda_sources=_CUDA_SRC,
            functions=["gemma_rms_norm"],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False,
        )
    except Exception:  # noqa: BLE001 - any build problem -> PyTorch fallback
        return None


_rms_norm = getattr(_build_ext(), "gemma_rms_norm", None)


class GemmaRMSNorm(nn.Module):
    """RMSNorm with weight stored as offset from 1.0 (Gemma convention)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.variance_epsilon = eps
        self.weight = nn.Parameter(torch.zeros(hidden_size))

    @staticmethod
    def _forward_static_no_residual(
        weight: torch.Tensor,
        variance_epsilon: float,
        x: torch.Tensor,
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + variance_epsilon)
        x = x * (1.0 + weight.float())
        return x.to(orig_dtype)

    @staticmethod
    def _forward_static_with_residual(
        weight: torch.Tensor,
        variance_epsilon: float,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        # Match vLLM: promote to f32 only when the residual add would lose
        # precision (i.e. fp16 inputs); otherwise add in the input dtype.
        x = (
            x.float() + residual.float()
            if orig_dtype == torch.float16
            else x + residual
        )
        residual = x.to(orig_dtype) if x.dtype != orig_dtype else x

        x = x.float()
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + variance_epsilon)
        x = x * (1.0 + weight.float())
        return x.to(orig_dtype), residual

    def forward_native(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self._forward_static_no_residual(
                self.weight.data, self.variance_epsilon, x,
            )
        return self._forward_static_with_residual(
            self.weight.data, self.variance_epsilon, x, residual,
        )

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None and _rms_norm is not None:
            out = _rms_norm(x, self.weight, self.variance_epsilon)
            if out is not None:
                return out
        return self.forward_native(x, residual)
