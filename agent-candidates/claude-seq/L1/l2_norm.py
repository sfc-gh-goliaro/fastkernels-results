"""L2 normalization along a single dimension: x / ||x||_2.

Fused CUDA kernel for the common case (float32, contiguous, last dim): one
kernel reads each row once into registers, reduces the sum of squares, and
writes the scaled row back. ``F.normalize`` needs four dispatches (norm,
clamp_min, expand_as, div) and three passes over the data; this needs one of
each. Anything the kernel does not cover falls back to ``F.normalize``.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#define FULL_MASK 0xffffffffu

__device__ __forceinline__ float warp_all_sum(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(FULL_MASK, v, off);
  return v;
}

template <int NW>
__device__ __forceinline__ float block_all_sum(float v, float* sm) {
  v = warp_all_sum(v);
  if (NW == 1) return v;
  if ((threadIdx.x & 31) == 0) sm[threadIdx.x >> 5] = v;
  __syncthreads();
  float s = 0.f;
#pragma unroll
  for (int i = 0; i < NW; ++i) s += sm[i];
  return s;
}

__device__ __forceinline__ float sq_acc(const float4& v, float ss) {
  ss = fmaf(v.x, v.x, ss);
  ss = fmaf(v.y, v.y, ss);
  ss = fmaf(v.z, v.z, ss);
  ss = fmaf(v.w, v.w, ss);
  return ss;
}

// One block per row; the whole row lives in registers (VPT float4 per thread),
// so the data is read from memory exactly once.
template <int VPT, int THREADS>
__global__ __launch_bounds__(THREADS) void l2norm_rows(
    const float4* __restrict__ x, float4* __restrict__ y, float eps) {
  constexpr int NW = THREADS / 32;
  __shared__ float sm[NW > 1 ? NW : 1];
  const size_t base = (size_t)blockIdx.x * (THREADS * VPT) + threadIdx.x;
  const float4* __restrict__ xp = x + base;
  float4 v[VPT];
#pragma unroll
  for (int i = 0; i < VPT; ++i) v[i] = xp[i * THREADS];
  float ss = 0.f;
#pragma unroll
  for (int i = 0; i < VPT; ++i) ss = sq_acc(v[i], ss);
  ss = block_all_sum<NW>(ss, sm);
  const float s = 1.0f / fmaxf(sqrtf(ss), eps);
  float4* __restrict__ yp = y + base;
#pragma unroll
  for (int i = 0; i < VPT; ++i) {
    float4 o;
    o.x = v[i].x * s;
    o.y = v[i].y * s;
    o.z = v[i].z * s;
    o.w = v[i].w * s;
    yp[i * THREADS] = o;
  }
}

// Any row length: two passes (the second hits L2), scalar access.
template <int THREADS>
__global__ __launch_bounds__(THREADS) void l2norm_generic(
    const float* __restrict__ x, float* __restrict__ y, int n, float eps) {
  constexpr int NW = THREADS / 32;
  __shared__ float sm[NW];
  const size_t off = (size_t)blockIdx.x * n;
  const float* __restrict__ xp = x + off;
  float ss = 0.f;
  for (int i = threadIdx.x; i < n; i += THREADS) {
    const float v = xp[i];
    ss = fmaf(v, v, ss);
  }
  ss = block_all_sum<NW>(ss, sm);
  const float s = 1.0f / fmaxf(sqrtf(ss), eps);
  float* __restrict__ yp = y + off;
  for (int i = threadIdx.x; i < n; i += THREADS) yp[i] = xp[i] * s;
}

#define LAUNCH(VPT, T)                                                       \
  l2norm_rows<VPT, T><<<rows, T, 0, stream>>>(xp, yp, e);                    \
  return y;

torch::Tensor l2norm_last(const torch::Tensor& x, double eps) {
  auto y = torch::empty_like(x);
  const int n = (int)x.size(-1);
  const int rows = (int)(x.numel() / n);
  const float e = (float)eps;
  auto stream = at::cuda::getCurrentCUDAStream();

  if ((n & 3) == 0) {
    const int n4 = n >> 2;
    const float4* __restrict__ xp =
        reinterpret_cast<const float4*>(x.data_ptr<float>());
    float4* __restrict__ yp = reinterpret_cast<float4*>(y.data_ptr<float>());
    // Prefer 2 float4/thread, then 4, 1, 8 (measured fastest in that order).
    const int vpts[4] = {2, 4, 1, 8};
    for (int k = 0; k < 4; ++k) {
      const int vpt = vpts[k];
      if (n4 % vpt) continue;
      const int t = n4 / vpt;
      if (t < 32 || t > 512 || (t & 31)) continue;
      switch (vpt * 1024 + t) {
        case 2 * 1024 + 32: LAUNCH(2, 32)
        case 2 * 1024 + 64: LAUNCH(2, 64)
        case 2 * 1024 + 96: LAUNCH(2, 96)
        case 2 * 1024 + 128: LAUNCH(2, 128)
        case 2 * 1024 + 160: LAUNCH(2, 160)
        case 2 * 1024 + 192: LAUNCH(2, 192)
        case 2 * 1024 + 256: LAUNCH(2, 256)
        case 2 * 1024 + 320: LAUNCH(2, 320)
        case 2 * 1024 + 384: LAUNCH(2, 384)
        case 2 * 1024 + 512: LAUNCH(2, 512)
        case 4 * 1024 + 32: LAUNCH(4, 32)
        case 4 * 1024 + 64: LAUNCH(4, 64)
        case 4 * 1024 + 96: LAUNCH(4, 96)
        case 4 * 1024 + 128: LAUNCH(4, 128)
        case 4 * 1024 + 192: LAUNCH(4, 192)
        case 4 * 1024 + 256: LAUNCH(4, 256)
        case 4 * 1024 + 384: LAUNCH(4, 384)
        case 4 * 1024 + 512: LAUNCH(4, 512)
        case 1 * 1024 + 32: LAUNCH(1, 32)
        case 1 * 1024 + 64: LAUNCH(1, 64)
        case 1 * 1024 + 96: LAUNCH(1, 96)
        case 1 * 1024 + 128: LAUNCH(1, 128)
        case 1 * 1024 + 160: LAUNCH(1, 160)
        case 1 * 1024 + 192: LAUNCH(1, 192)
        case 1 * 1024 + 224: LAUNCH(1, 224)
        case 1 * 1024 + 256: LAUNCH(1, 256)
        case 1 * 1024 + 288: LAUNCH(1, 288)
        case 1 * 1024 + 320: LAUNCH(1, 320)
        case 1 * 1024 + 352: LAUNCH(1, 352)
        case 1 * 1024 + 384: LAUNCH(1, 384)
        case 1 * 1024 + 416: LAUNCH(1, 416)
        case 1 * 1024 + 448: LAUNCH(1, 448)
        case 1 * 1024 + 480: LAUNCH(1, 480)
        case 1 * 1024 + 512: LAUNCH(1, 512)
        case 8 * 1024 + 32: LAUNCH(8, 32)
        case 8 * 1024 + 64: LAUNCH(8, 64)
        case 8 * 1024 + 128: LAUNCH(8, 128)
        case 8 * 1024 + 256: LAUNCH(8, 256)
        case 8 * 1024 + 512: LAUNCH(8, 512)
        default: break;
      }
    }
  }
  l2norm_generic<256><<<rows, 256, 0, stream>>>(
      x.data_ptr<float>(), y.data_ptr<float>(), n, e);
  return y;
}
"""

_CPP_DECL = "torch::Tensor l2norm_last(const torch::Tensor& x, double eps);"


def _build():
    """JIT-compile the extension for the local arch; None if unavailable."""
    if not torch.cuda.is_available():
        return None
    from torch.utils.cpp_extension import load_inline

    prev = os.environ.get("TORCH_CUDA_ARCH_LIST")
    major, minor = torch.cuda.get_device_capability()
    os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
    try:
        return load_inline(
            name="fk_cand_l2_norm_v1",
            cpp_sources=[_CPP_DECL],
            cuda_sources=[_CUDA_SRC],
            functions=["l2norm_last"],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
    finally:
        if prev is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = prev


try:
    _C = _build()
except Exception:  # pragma: no cover - fall back to the reference path
    _C = None


class L2Norm(nn.Module):
    def __init__(self, dim: int = -1, eps: float = 1e-12):
        super().__init__()
        self.dim = dim
        self.eps = eps
        # Resolved once: the kernel only handles normalization over the last dim.
        self._last_dim = dim in (-1,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            _C is not None
            and x.dtype is torch.float32
            and x.is_cuda
            and x.is_contiguous()
            and x.dim() >= 1
            and 0 < x.numel() < 2**31
            and (self._last_dim or self.dim == x.dim() - 1)
        ):
            return _C.l2norm_last(x, self.eps)
        return F.normalize(x, p=2.0, dim=self.dim, eps=self.eps)
