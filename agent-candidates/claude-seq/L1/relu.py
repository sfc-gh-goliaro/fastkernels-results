"""ReLU activation: max(0, x).

The captured shapes are small (96 KB .. 1.5 MB of bfloat16), so on a B200 this
operator is entirely launch-bound.  Measured with the bench's own timing loop
(L2 flush + shifting input pool), the window around one call decomposes as:

    output allocation only ............  9.2 us   (the harness' input copy)
    + *any* kernel launch ............ +3.1 us   (empty, write-only, or full relu
                                                  all land on the same number)
    ``F.relu`` on [1,12,32,128,16] ... 14.3 us

So there is nothing to win in the arithmetic and nothing to win on the host
path -- the only thing that shows up is a relu kernel whose execution does not
poke above the fixed 3.1 us launch cost.  ``F.relu`` does poke above it on the
largest shape; this one does not:

* one 128-bit-vectorized kernel, 8 bfloat16 lanes per thread, one element per
  thread (no grid-stride loop, no tail branch on the captured shapes);
* relu on IEEE floats is "zero the lane whose sign bit is set", so the body is
  three integer ops per 32-bit word and no float pipeline at all;
* reached by a single pybind call that allocates with ``at::detail::empty_cuda``
  and launches directly, skipping ``F.relu``'s Python wrapper, the ATen
  dispatcher and TensorIterator setup.

A CUDA-graph launch path (patch the kernel node's params, replay) was also
measured: it halves the host-side launch cost but adds GPU-side dispatch cost,
which is what this regime actually pays, so it came out ~0.2 us slower and is
not used.

Anything the vector path does not cover (unsupported dtype, non-contiguous,
unaligned) falls back to a scalar kernel or, for non-CUDA tensors, ``at::relu``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/EmptyTensor.h>

#define BLK 256

// relu for two packed 16-bit IEEE floats: clear every lane whose sign bit is
// set.  `s - (s >> 15)` turns a 0x8000 lane into 0x7fff, so `s | that` is the
// 0xffff mask of the negative lanes (and 0 for the non-negative ones).
__device__ __forceinline__ uint32_t relu_h2(uint32_t v) {
  uint32_t s = v & 0x80008000u;
  return v & ~(s | (s - (s >> 15)));
}

// relu for one 32-bit IEEE float.
__device__ __forceinline__ uint32_t relu_f1(uint32_t v) {
  return (v & 0x80000000u) ? 0u : v;
}

__device__ __forceinline__ uint4 relu_v16(uint4 v) {
  v.x = relu_h2(v.x); v.y = relu_h2(v.y);
  v.z = relu_h2(v.z); v.w = relu_h2(v.w);
  return v;
}

__device__ __forceinline__ uint4 relu_v32(uint4 v) {
  v.x = relu_f1(v.x); v.y = relu_f1(v.y);
  v.z = relu_f1(v.z); v.w = relu_f1(v.w);
  return v;
}

// 16 bytes per thread; `n` counts 16-byte vectors.
__global__ __launch_bounds__(BLK) void k_relu16(const uint4* __restrict__ in,
                                                uint4* __restrict__ out, int n) {
  int i = blockIdx.x * BLK + threadIdx.x;
  if (i < n) out[i] = relu_v16(in[i]);
}

__global__ __launch_bounds__(BLK) void k_relu32(const uint4* __restrict__ in,
                                                uint4* __restrict__ out, int n) {
  int i = blockIdx.x * BLK + threadIdx.x;
  if (i < n) out[i] = relu_v32(in[i]);
}

// Scalar fallbacks for sizes/alignments the vector path cannot take.
__global__ __launch_bounds__(BLK) void k_relu16_s(const uint16_t* __restrict__ in,
                                                  uint16_t* __restrict__ out, int64_t n) {
  int64_t i = (int64_t)blockIdx.x * BLK + threadIdx.x;
  if (i < n) {
    uint16_t v = in[i];
    out[i] = (v & 0x8000u) ? 0u : v;
  }
}

__global__ __launch_bounds__(BLK) void k_relu32_s(const uint32_t* __restrict__ in,
                                                  uint32_t* __restrict__ out, int64_t n) {
  int64_t i = (int64_t)blockIdx.x * BLK + threadIdx.x;
  if (i < n) out[i] = relu_f1(in[i]);
}

at::Tensor fk_relu(const at::Tensor& x) {
  const auto st = x.scalar_type();
  const bool h = (st == at::kBFloat16 || st == at::kHalf);
  if (!x.is_cuda() || !x.is_contiguous() || !(h || st == at::kFloat))
    return at::relu(x);

  const int64_t numel = x.numel();
  at::Tensor out = at::detail::empty_cuda(x.sizes(), st, x.device(), c10::nullopt);
  if (numel == 0) return out;

  const void* ip = x.const_data_ptr();
  void* op = out.mutable_data_ptr();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int64_t bytes = numel * (h ? 2 : 4);

  if ((bytes & 15) == 0 && (((uintptr_t)ip | (uintptr_t)op) & 15) == 0) {
    const int n = (int)(bytes >> 4);
    const int grid = (n + BLK - 1) / BLK;
    if (h)
      k_relu16<<<grid, BLK, 0, stream>>>((const uint4*)ip, (uint4*)op, n);
    else
      k_relu32<<<grid, BLK, 0, stream>>>((const uint4*)ip, (uint4*)op, n);
  } else {
    const int64_t grid = (numel + BLK - 1) / BLK;
    if (h)
      k_relu16_s<<<grid, BLK, 0, stream>>>((const uint16_t*)ip, (uint16_t*)op, numel);
    else
      k_relu32_s<<<grid, BLK, 0, stream>>>((const uint32_t*)ip, (uint32_t*)op, numel);
  }
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("relu", &fk_relu, py::arg("x"));
}
"""


def _build():
    from torch.utils.cpp_extension import load_inline

    return load_inline(
        name="fk_l1_relu_vec",
        cpp_sources="",
        cuda_sources=_CUDA_SRC,
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        with_cuda=True,
        verbose=False,
    )


try:
    _RELU = _build().relu
except Exception:  # pragma: no cover - stay usable if nvcc is unavailable
    _RELU = None


class ReLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if _RELU is None:
            return F.relu(x)
        return _RELU(x)
