"""T5-style RMSNorm with fp32 variance computation.

Matches HuggingFace's T5LayerNorm exactly: upcasts to fp32 for variance
and rsqrt, then casts back before multiplying by the weight.  This is
numerically distinct from the fused _C.rmsnorm kernel used by other
models (which stays in bf16), but required for bit-exact parity with
the HuggingFace / vllm-omni T5 encoder.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import triton
import triton.language as tl


_CPP_SOURCE = """
#include <torch/extension.h>

torch::Tensor t5_layer_norm_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    double eps);
"""

_CUDA_SOURCE = """
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>

__global__ void t5_layer_norm_4096_kernel(
    const __nv_bfloat162* __restrict__ input,
    const float2* __restrict__ weight,
    float2* __restrict__ output,
    float eps) {
  constexpr int kPairsPerRow = 2048;
  constexpr int kThreads = 512;
  constexpr int kIterations = kPairsPerRow / kThreads;

  const int row = blockIdx.x;
  const int row_offset = row * kPairsPerRow;
  float values[kIterations * 2];
  float sum = 0.0f;

#pragma unroll
  for (int i = 0; i < kIterations; ++i) {
    const int col = threadIdx.x + i * kThreads;
    const float2 value = __bfloat1622float2(input[row_offset + col]);
    values[2 * i] = value.x;
    values[2 * i + 1] = value.y;
    sum = fmaf(value.x, value.x, sum);
    sum = fmaf(value.y, value.y, sum);
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(0xffffffff, sum, offset);
  }

  __shared__ float warp_sums[kThreads / 32];
  if ((threadIdx.x & 31) == 0) {
    warp_sums[threadIdx.x >> 5] = sum;
  }
  __syncthreads();

  if (threadIdx.x < 32) {
    sum = threadIdx.x < kThreads / 32 ? warp_sums[threadIdx.x] : 0.0f;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    if (threadIdx.x == 0) {
      warp_sums[0] = rsqrtf(sum * (1.0f / 4096.0f) + eps);
    }
  }
  __syncthreads();

  const float scale = warp_sums[0];
#pragma unroll
  for (int i = 0; i < kIterations; ++i) {
    const int col = threadIdx.x + i * kThreads;
    const float2 w = weight[col];
    output[row_offset + col] =
        make_float2(values[2 * i] * scale * w.x,
                    values[2 * i + 1] * scale * w.y);
  }
}

torch::Tensor t5_layer_norm_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    double eps) {
  auto output = torch::empty(
      input.sizes(), input.options().dtype(torch::kFloat32));
  const int rows = input.numel() / 4096;
  const auto* input_ptr = reinterpret_cast<const __nv_bfloat162*>(
      input.data_ptr<at::BFloat16>());
  const auto* weight_ptr =
      reinterpret_cast<const float2*>(weight.data_ptr<float>());
  auto* output_ptr = reinterpret_cast<float2*>(output.data_ptr<float>());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  t5_layer_norm_4096_kernel<<<rows, 512, 0, stream>>>(
      input_ptr, weight_ptr, output_ptr, static_cast<float>(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
"""

_previous_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST")
os.environ["TORCH_CUDA_ARCH_LIST"] = "10.0"
try:
    _cuda_ext = load_inline(
        name="_fastkernels_t5_layer_norm_cuda",
        cpp_sources=_CPP_SOURCE,
        cuda_sources=_CUDA_SOURCE,
        functions=["t5_layer_norm_cuda"],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        with_cuda=True,
        verbose=False,
    )
finally:
    if _previous_arch_list is None:
        os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
    else:
        os.environ["TORCH_CUDA_ARCH_LIST"] = _previous_arch_list


@triton.jit
def _t5_layer_norm_kernel(
    x_ptr,
    weight_ptr,
    output_ptr,
    eps: tl.constexpr,
    hidden_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_size

    x = tl.load(x_ptr + row * hidden_size + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    variance = tl.sum(x * x, axis=0) / hidden_size
    scale = tl.rsqrt(variance + eps)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
    tl.store(
        output_ptr + row * hidden_size + offsets,
        x * scale * weight,
        mask=mask,
    )


class T5LayerNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_size = hidden_states.shape[-1]
        rows = hidden_states.numel() // hidden_size
        if (
            hidden_size == 4096
            and hidden_states.dtype == torch.bfloat16
            and self.weight.dtype == torch.float32
        ):
            return _cuda_ext.t5_layer_norm_cuda(
                hidden_states, self.weight, self.variance_epsilon
            )

        output_dtype = (
            self.weight.dtype
            if self.weight.dtype in (torch.float16, torch.bfloat16)
            else torch.float32
        )
        output = torch.empty_like(hidden_states, dtype=output_dtype)
        block_size = triton.next_power_of_2(hidden_size)
        _t5_layer_norm_kernel[(rows,)](
            hidden_states,
            self.weight,
            output,
            self.variance_epsilon,
            hidden_size,
            BLOCK_SIZE=block_size,
            num_warps=8,
        )
        return output
