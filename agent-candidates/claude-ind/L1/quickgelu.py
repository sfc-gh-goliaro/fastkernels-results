"""QuickGELU activation: x * sigmoid(1.702 * x).

Approximation of GELU used in Qwen2-VL vision encoder.

The eager form is three elementwise passes over the tensor (``1.702 * x``,
``sigmoid``, ``x * ...``). At the captured size -- 236544 fp32 elements, under
1 MiB -- each pass costs far more in launch/round-trip than in bandwidth, so the
whole op is launch-bound: collapsing the three kernels into one fused pass is
the entire win, and the arithmetic inside it is free by comparison.

The fused kernel rewrites the activation as a division so no separate sigmoid
pass is needed::

    x * sigmoid(1.702 * x) == x / (1 + exp(-1.702 * x))

and evaluates ``exp`` as ``ex2.approx.f32`` on ``-(1.702*log2 e) * x`` (one
MUFU op), with ``rcp.approx.f32`` for the divide. Both approximations are
accurate to ~1 ulp of fp32, comfortably inside the fp32 bench tolerance
(atol 1e-5, rtol 1e-3).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_CPP_SRC = r"""
#include <ATen/ATen.h>
at::Tensor fk_quickgelu(const at::Tensor &x);
"""

_CUDA_SRC = r"""
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/EmptyTensor.h>
#include <cuda_runtime.h>

// 1.702 * log2(e), so exp(-1.702*x) == exp2(-QG_C * x).
#define QG_C 2.45561981201171875f
#define QG_BLOCK 256

__device__ __forceinline__ float qg_ex2(float x) {
  float r;
  asm("ex2.approx.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}

__device__ __forceinline__ float qg_rcp(float x) {
  float r;
  asm("rcp.approx.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}

__device__ __forceinline__ float qg_one(float x) {
  return x * qg_rcp(1.0f + qg_ex2(-QG_C * x));
}

// One float4 per thread; EXACT drops the bounds check when the grid covers the
// tensor exactly (it does for the captured shape: 59136 float4 == 231 * 256).
template <bool EXACT>
__global__ __launch_bounds__(QG_BLOCK) void qg_vec4(const float4 *__restrict__ in,
                                                    float4 *__restrict__ out,
                                                    int n4) {
  const int i = blockIdx.x * QG_BLOCK + threadIdx.x;
  if (!EXACT && i >= n4) return;
  float4 v = in[i];
  v.x = qg_one(v.x);
  v.y = qg_one(v.y);
  v.z = qg_one(v.z);
  v.w = qg_one(v.w);
  out[i] = v;
}

__global__ __launch_bounds__(QG_BLOCK) void qg_scalar(const float *__restrict__ in,
                                                      float *__restrict__ out, int n) {
  const int i = blockIdx.x * QG_BLOCK + threadIdx.x;
  if (i < n) out[i] = qg_one(in[i]);
}

at::Tensor fk_quickgelu(const at::Tensor &x) {
  // Anything that is not a contiguous fp32 CUDA tensor takes the eager path.
  if (!x.is_cuda() || x.scalar_type() != at::kFloat || !x.is_contiguous()) {
    return x * at::sigmoid(x * 1.702);
  }
  // ``at::detail::empty_cuda`` allocates without the dispatcher hop that
  // ``at::empty`` takes -- cheap, and this op has no cycles to spare.
  at::Tensor out =
      at::detail::empty_cuda(x.sizes(), at::kFloat, x.device(), c10::nullopt);
  const int n = static_cast<int>(x.numel());
  if (n == 0) return out;

  const float *ip = static_cast<const float *>(x.const_data_ptr());
  float *op = static_cast<float *>(out.mutable_data_ptr());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const bool aligned =
      ((reinterpret_cast<uintptr_t>(ip) | reinterpret_cast<uintptr_t>(op)) & 15u) == 0;
  if (aligned && (n & 3) == 0) {
    const int n4 = n >> 2;
    const int grid = (n4 + QG_BLOCK - 1) / QG_BLOCK;
    if (n4 == grid * QG_BLOCK) {
      qg_vec4<true><<<grid, QG_BLOCK, 0, stream>>>(
          reinterpret_cast<const float4 *>(ip), reinterpret_cast<float4 *>(op), n4);
    } else {
      qg_vec4<false><<<grid, QG_BLOCK, 0, stream>>>(
          reinterpret_cast<const float4 *>(ip), reinterpret_cast<float4 *>(op), n4);
    }
  } else {
    qg_scalar<<<(n + QG_BLOCK - 1) / QG_BLOCK, QG_BLOCK, 0, stream>>>(ip, op, n);
  }
  return out;
}
"""

_ext = load_inline(
    name="fk_quickgelu_fused",
    cpp_sources=_CPP_SRC,
    cuda_sources=_CUDA_SRC,
    functions=["fk_quickgelu"],
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)

# Bound once at import: ``forward`` is one global lookup plus one call.
_fk_quickgelu = _ext.fk_quickgelu


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _fk_quickgelu(x)
