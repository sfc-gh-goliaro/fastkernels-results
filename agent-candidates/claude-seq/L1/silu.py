"""SiLU (Swish) activation: x * sigmoid(x).

Hand-written elementwise CUDA kernel.  Two observations drive the design:

* The captured shapes are either tiny decode activations
  ([1..256, 1, 2560/6912]) -- where the op is pure launch latency -- or one
  very large prefill activation ([181, 1081, 6912], 2.7 GB) -- where it is
  pure HBM bandwidth.  Neither regime tolerates extra arithmetic: ATen's
  generic path spends an ``exp`` plus a full-precision divide per element and
  ends up ~1.7x off a device-to-device copy on the large case.
* SiLU can be evaluated through the half-angle tanh identity

      silu(x) = x * sigmoid(x) = h * (1 + tanh(h)),   h = x / 2

  which on sm_90+ is *one* ``tanh.approx.bf16x2`` MUFU instruction per **two**
  elements (``tanh.approx.f16x2`` for fp16, ``tanh.approx.f32`` for fp32).

The kernel moves 16 bytes per thread per access (one ``uint4``); it processes
one such vector per thread for small inputs (maximum block parallelism, lowest
latency) and two for large ones (more memory-level parallelism per thread,
which is worth ~12% on the 2.7 GB case).  Both regimes land at
device-copy bandwidth (~6.2 TB/s measured on B200).
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#define NT 256
// Above this many 16B vectors, two vectors per thread beats one.
#define BIG_NVEC (NT * 2048)

// ---------------------------------------------------------------------------
// silu(x) = h * (1 + tanh(h)),  h = x/2   -- one MUFU per packed pair.
// ---------------------------------------------------------------------------
// KIND: 0 = bfloat16x2, 1 = float16x2, 2 = float32 (scalar in a uint slot)
template <int KIND>
__device__ __forceinline__ unsigned silu_pack(unsigned v) {
  unsigned h, t, r;
  if (KIND == 0) {
    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(h) : "r"(v), "r"(0x3f003f00u));
    asm("tanh.approx.bf16x2 %0, %1;" : "=r"(t) : "r"(h));
    asm("fma.rn.bf16x2 %0, %1, %2, %1;" : "=r"(r) : "r"(h), "r"(t));
  } else if (KIND == 1) {
    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(h) : "r"(v), "r"(0x38003800u));
    asm("tanh.approx.f16x2 %0, %1;" : "=r"(t) : "r"(h));
    asm("fma.rn.f16x2 %0, %1, %2, %1;" : "=r"(r) : "r"(h), "r"(t));
  } else {
    float hf = __uint_as_float(v) * 0.5f, tf;
    asm("tanh.approx.f32 %0, %1;" : "=f"(tf) : "f"(hf));
    r = __float_as_uint(fmaf(hf, tf, hf));
  }
  return r;
}

struct alignas(16) V16 { unsigned w[4]; };

template <int KIND, int U, bool EXACT>
__global__ __launch_bounds__(NT) void silu_vec_kernel(
    const V16* __restrict__ in, V16* __restrict__ out, int nvec) {
  const int base = blockIdx.x * (NT * U) + threadIdx.x;
  V16 v[U];
#pragma unroll
  for (int u = 0; u < U; ++u) {
    const int i = base + u * NT;
    if (EXACT || i < nvec) v[u] = in[i];
  }
#pragma unroll
  for (int u = 0; u < U; ++u)
#pragma unroll
    for (int j = 0; j < 4; ++j) v[u].w[j] = silu_pack<KIND>(v[u].w[j]);
#pragma unroll
  for (int u = 0; u < U; ++u) {
    const int i = base + u * NT;
    if (EXACT || i < nvec) out[i] = v[u];
  }
}

// Any element count / any alignment (never hit by the captured shapes).
template <typename T, int KIND>
__global__ void silu_scalar_kernel(const T* __restrict__ in, T* __restrict__ out,
                                   long long n) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    const float x = static_cast<float>(in[i]);
    float h = x * 0.5f, t;
    asm("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(h));
    out[i] = static_cast<T>(fmaf(h, t, h));
  }
}

template <int KIND>
static void launch_vec(const void* ip, void* op, int nvec, cudaStream_t s) {
  if (nvec >= BIG_NVEC) {
    const int per = NT * 2;
    const int blocks = (nvec + per - 1) / per;
    if (nvec % per == 0)
      silu_vec_kernel<KIND, 2, true><<<blocks, NT, 0, s>>>(
          (const V16*)ip, (V16*)op, nvec);
    else
      silu_vec_kernel<KIND, 2, false><<<blocks, NT, 0, s>>>(
          (const V16*)ip, (V16*)op, nvec);
  } else {
    const int blocks = (nvec + NT - 1) / NT;
    if (nvec % NT == 0)
      silu_vec_kernel<KIND, 1, true><<<blocks, NT, 0, s>>>(
          (const V16*)ip, (V16*)op, nvec);
    else
      silu_vec_kernel<KIND, 1, false><<<blocks, NT, 0, s>>>(
          (const V16*)ip, (V16*)op, nvec);
  }
}

at::Tensor silu(const at::Tensor& x) {
  const auto st = x.scalar_type();
  const bool supported = (st == at::kBFloat16 || st == at::kHalf || st == at::kFloat);
  if (!supported || !x.is_cuda() || !x.is_contiguous() || x.requires_grad())
    return at::silu(x);

  at::Tensor out = at::empty_like(x);
  const int64_t n = x.numel();
  if (n == 0) return out;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const void* ip = x.const_data_ptr();
  void* op = out.data_ptr();

  const int64_t lanes = 16 / x.element_size();  // elements per 16B vector
  const bool aligned =
      ((reinterpret_cast<uintptr_t>(ip) | reinterpret_cast<uintptr_t>(op)) & 15) == 0;
  if (aligned && (n % lanes) == 0 && (n / lanes) <= 0x7fffffffLL) {
    const int nvec = static_cast<int>(n / lanes);
    if (st == at::kBFloat16)    launch_vec<0>(ip, op, nvec, stream);
    else if (st == at::kHalf)   launch_vec<1>(ip, op, nvec, stream);
    else                        launch_vec<2>(ip, op, nvec, stream);
    return out;
  }

  const int64_t blocks = (n + NT - 1) / NT;
  if (st == at::kBFloat16)
    silu_scalar_kernel<__nv_bfloat16, 0><<<blocks, NT, 0, stream>>>(
        (const __nv_bfloat16*)ip, (__nv_bfloat16*)op, n);
  else if (st == at::kHalf)
    silu_scalar_kernel<__half, 1><<<blocks, NT, 0, stream>>>(
        (const __half*)ip, (__half*)op, n);
  else
    silu_scalar_kernel<float, 2><<<blocks, NT, 0, stream>>>(
        (const float*)ip, (float*)op, n);
  return out;
}
"""

_CPP_SRC = "at::Tensor silu(const at::Tensor& x);"


def _build():
    cc = torch.cuda.get_device_capability()
    # tanh.approx.bf16x2 needs sm_90+; pin the build to the local arch only for
    # the duration of this build (the env var is global to the process).
    prev = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"] = f"{cc[0]}.{cc[1]}" + ("a" if cc[0] >= 9 else "")
    try:
        return _load()
    finally:
        if prev is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = prev


def _load():
    return load_inline(
        name="fk_silu_tanh_v2",
        cpp_sources=_CPP_SRC,
        cuda_sources=_CUDA_SRC,
        functions=["silu"],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        ],
        verbose=False,
    ).silu


_silu = F.silu
if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9:
    try:
        _silu = _build()
    except Exception:  # pragma: no cover - keep ATen if the JIT build fails
        _silu = F.silu


class SiLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _silu(x)
