"""MaxPool2d with a fused CUDA kernel for the 5x5 / stride-1 / pad-2 case.

``F.max_pool2d`` on CUDA routes through ``max_pool2d_with_indices``, so it also
materializes and writes an int64 argmax tensor nobody asked for, and every
output re-reads all 25 window elements from global memory.  For the shapes this
operator sees (fp16 ``[N, 128, 20, 20]``, k=5/s=1/p=2 -- the YOLOv10 SPPF pool)
that kernel measures ~10.4us on a B200 while a plain 400KB copy takes ~2.5us.

The fast path here takes the separable 5x5 max in the order that vectorizes:
one thread owns one output *row*, loads the five input rows it needs as aligned
8-byte chunks, maxes them elementwise with ``__hmax2`` (the y pass needs no
shifts at all), then slides the 5-wide x window entirely in registers.  No
shared memory, no barriers and no index output, which puts it at the empty-
kernel launch floor (~1.9us).

``W != 20`` planes fall back to a shared-memory separable kernel, and anything
that is not 5/1/2 pooling to ``at::max_pool2d`` inside the extension, so the
Python side is a single call either way.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

_CUDA_SRC = r"""
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace {

template <typename T> struct Traits;
template <> struct Traits<__half> {
  using v2 = __half2;
  static __device__ __forceinline__ v2 splat(float f) { return __half2half2(__float2half(f)); }
  static __device__ __forceinline__ v2 max2(v2 a, v2 b) { return __hmax2(a, b); }
  static __device__ __forceinline__ __half mx(__half a, __half b) { return __hmax(a, b); }
  static __device__ __forceinline__ float to(__half v) { return __half2float(v); }
  static __device__ __forceinline__ __half from(float v) { return __float2half(v); }
  static __device__ __forceinline__ __half neg() { return __float2half(-65504.0f); }
};
template <> struct Traits<__nv_bfloat16> {
  using v2 = __nv_bfloat162;
  static __device__ __forceinline__ v2 splat(float f) { return __bfloat162bfloat162(__float2bfloat16(f)); }
  static __device__ __forceinline__ v2 max2(v2 a, v2 b) { return __hmax2(a, b); }
  static __device__ __forceinline__ __nv_bfloat16 mx(__nv_bfloat16 a, __nv_bfloat16 b) { return __hmax(a, b); }
  static __device__ __forceinline__ float to(__nv_bfloat16 v) { return __bfloat162float(v); }
  static __device__ __forceinline__ __nv_bfloat16 from(float v) { return __float2bfloat16(v); }
  static __device__ __forceinline__ __nv_bfloat16 neg() { return __float2bfloat16(-3.0e38f); }
};
template <> struct Traits<float> {
  static __device__ __forceinline__ float to(float v) { return v; }
  static __device__ __forceinline__ float from(float v) { return v; }
  static __device__ __forceinline__ float neg() { return -3.0e38f; }
};

// ---------------------------------------------------------------------------
// Fast path: one output row per thread, all in registers.  W is a compile-time
// multiple of 4 so the row loads/stores are 8-byte vectors; H is free.
// ---------------------------------------------------------------------------
// __launch_bounds__ carries a min-blocks-per-SM of 1 on purpose: with the
// default target ptxas trims the kernel to ~36 registers, which forces it to
// fold the five row loads into a serial load/max chain instead of issuing them
// all up front.  That costs ~2us of exposed memory latency at these sizes.
template <typename T, int W, int TPB>
__global__ __launch_bounds__(TPB, 1) void mp5_row_kernel(const T* __restrict__ in,
                                                         T* __restrict__ out,
                                                         int H, int nplanes) {
  using Tr = Traits<T>;
  using v2 = typename Tr::v2;
  const int VEC = W / 4;   // uint2 chunks per row
  const int HALF = W / 2;  // v2 elements per row

  const int g = blockIdx.x * TPB + threadIdx.x;
  const int plane = g / H;
  if (plane >= nplanes) return;
  const int y = g - plane * H;
  const T* p = in + (long)plane * H * W;

  const v2 NEG = Tr::splat(-65504.0f);
  v2 rows[5][HALF];
#pragma unroll
  for (int k = 0; k < 5; ++k) {
    const int yy = y - 2 + k;
    if ((unsigned)yy < (unsigned)H) {
      const uint2* rp = reinterpret_cast<const uint2*>(p + yy * W);
      uint2 b[VEC];
#pragma unroll
      for (int i = 0; i < VEC; ++i) b[i] = rp[i];
#pragma unroll
      for (int i = 0; i < HALF; ++i) rows[k][i] = reinterpret_cast<v2*>(b)[i];
    } else {
#pragma unroll
      for (int i = 0; i < HALF; ++i) rows[k][i] = NEG;
    }
  }

  // y pass: elementwise max of the five rows (no shifts -> pure SIMD).
  v2 c[HALF];
#pragma unroll
  for (int i = 0; i < HALF; ++i) c[i] = rows[0][i];
#pragma unroll
  for (int k = 1; k < 5; ++k)
#pragma unroll
    for (int i = 0; i < HALF; ++i) c[i] = Tr::max2(c[i], rows[k][i]);

  // x pass: 5-wide sliding max over the W register values.
  const T* cv = reinterpret_cast<const T*>(c);
  T o[W];
#pragma unroll
  for (int x = 0; x < W; ++x) {
    T m = cv[x];
#pragma unroll
    for (int dx = -2; dx <= 2; ++dx) {
      const int xx = x + dx;
      if (dx != 0 && (unsigned)xx < (unsigned)W) m = Tr::mx(m, cv[xx]);
    }
    o[x] = m;
  }
  uint2* op = reinterpret_cast<uint2*>(out + (long)plane * H * W + y * W);
#pragma unroll
  for (int i = 0; i < VEC; ++i) op[i] = reinterpret_cast<uint2*>(o)[i];
}

// ---------------------------------------------------------------------------
// General k=5/s=1/p=2 path: one thread per output, one plane per block, the
// separable max taken through shared memory.  Handles any H*W <= 1024.
// ---------------------------------------------------------------------------
template <typename T>
__global__ void mp5_smem_kernel(const T* __restrict__ in, T* __restrict__ out,
                                int H, int W, int nplanes) {
  extern __shared__ __align__(16) char smem_raw[];
  T* sa = reinterpret_cast<T*>(smem_raw);
  T* sb = sa + H * W;

  const int t = threadIdx.x;
  const int plane = blockIdx.x;
  const int y = t / W;
  const int x = t - y * W;
  const long base = (long)plane * H * W;

  sa[t] = in[base + t];
  __syncthreads();

  float m = -1.0e30f;
#pragma unroll
  for (int dx = -2; dx <= 2; ++dx) {
    const int xx = x + dx;
    if ((unsigned)xx < (unsigned)W) m = fmaxf(m, Traits<T>::to(sa[y * W + xx]));
  }
  sb[t] = Traits<T>::from(m);
  __syncthreads();

  float o = -1.0e30f;
#pragma unroll
  for (int dy = -2; dy <= 2; ++dy) {
    const int yy = y + dy;
    if ((unsigned)yy < (unsigned)H) o = fmaxf(o, Traits<T>::to(sb[yy * W + x]));
  }
  out[base + t] = Traits<T>::from(o);
}

template <typename T>
void launch_row(const void* ip, void* op, int H, int W, int64_t nplanes,
                cudaStream_t stream) {
  const T* i = reinterpret_cast<const T*>(ip);
  T* o = reinterpret_cast<T*>(op);
  const int64_t nrows = nplanes * H;
  // One warp per block is the lowest-latency shape at these sizes; widen the
  // block once there is enough work for occupancy to matter instead.
  if (nrows <= 16384) {
    const int TPB = 32;
    mp5_row_kernel<T, 20, TPB><<<(int)((nrows + TPB - 1) / TPB), TPB, 0, stream>>>(
        i, o, H, (int)nplanes);
  } else {
    const int TPB = 128;
    mp5_row_kernel<T, 20, TPB><<<(int)((nrows + TPB - 1) / TPB), TPB, 0, stream>>>(
        i, o, H, (int)nplanes);
  }
}

template <typename T>
void launch_smem(const void* ip, void* op, int H, int W, int64_t nplanes,
                 cudaStream_t stream) {
  mp5_smem_kernel<T><<<(int)nplanes, H * W, 2 * H * W * sizeof(T), stream>>>(
      reinterpret_cast<const T*>(ip), reinterpret_cast<T*>(op), H, W, (int)nplanes);
}

}  // namespace

at::Tensor max_pool2d_k5s1p2(const at::Tensor& x) {
  const auto st = x.scalar_type();
  const bool typed = st == at::kHalf || st == at::kBFloat16 || st == at::kFloat;
  if (x.dim() != 4 || !x.is_cuda() || !x.is_contiguous() || !typed) {
    return at::max_pool2d(x, {5, 5}, {1, 1}, {2, 2});
  }
  const int H = (int)x.size(2), W = (int)x.size(3);
  const bool row_path = (W == 20) && (st == at::kHalf || st == at::kBFloat16);
  if (!row_path && H * W > 1024) {
    return at::max_pool2d(x, {5, 5}, {1, 1}, {2, 2});
  }

  const at::cuda::OptionalCUDAGuard guard(device_of(x));
  at::Tensor out = at::empty(x.sizes(), x.options());
  const int64_t nplanes = x.size(0) * x.size(1);
  if (nplanes == 0) return out;
  auto stream = at::cuda::getCurrentCUDAStream();
  const void* ip = x.const_data_ptr();
  void* op = out.data_ptr();

  if (row_path) {
    if (st == at::kHalf) launch_row<__half>(ip, op, H, W, nplanes, stream);
    else launch_row<__nv_bfloat16>(ip, op, H, W, nplanes, stream);
  } else if (st == at::kHalf) {
    launch_smem<__half>(ip, op, H, W, nplanes, stream);
  } else if (st == at::kBFloat16) {
    launch_smem<__nv_bfloat16>(ip, op, H, W, nplanes, stream);
  } else {
    launch_smem<float>(ip, op, H, W, nplanes, stream);
  }
  return out;
}
"""

_CPP_SRC = r"""
at::Tensor max_pool2d_k5s1p2(const at::Tensor& x);
"""

_EXT = None


def _ext():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load_inline

        _EXT = load_inline(
            name="fk_max_pool2d_k5s1p2",
            cpp_sources=_CPP_SRC,
            cuda_sources=_CUDA_SRC,
            functions=["max_pool2d_k5s1p2"],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False,
        )
    return _EXT


def _as_pair(v) -> tuple[int, int]:
    if isinstance(v, (tuple, list)):
        return int(v[0]), int(v[1])
    return int(v), int(v)


class MaxPool2d(nn.Module):
    def __init__(
        self,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] | None = None,
        padding: int | tuple[int, int] = 0,
        ceil_mode: bool = False,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.ceil_mode = ceil_mode

        try:
            fast = (
                _as_pair(kernel_size) == (5, 5)
                and _as_pair(self.stride) == (1, 1)
                and _as_pair(padding) == (2, 2)
                and not ceil_mode
                and torch.cuda.is_available()
            )
        except Exception:
            fast = False
        # A build failure must not break the op, only its speed.
        try:
            self._fast_fn = _ext().max_pool2d_k5s1p2 if fast else None
        except Exception:
            self._fast_fn = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fn = self._fast_fn
        if fn is not None:
            return fn(x)
        return F.max_pool2d(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
            ceil_mode=self.ceil_mode,
        )
