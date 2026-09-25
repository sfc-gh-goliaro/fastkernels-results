"""Fast rotary position embedding for the diffusion-model capture shapes."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline


_CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>

template <int HEADS_PER_BLOCK, int ROWS_PER_BLOCK, bool INTERLEAVED>
__global__ void rope_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ cos,
    const __nv_bfloat16* __restrict__ sin,
    __nv_bfloat16* __restrict__ out,
    int seqlen,
    int nheads,
    int headdim,
    int rotary_half) {
  const int row = blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / 64;
  const int dim = threadIdx.x % 64;
  if (row >= seqlen || dim >= rotary_half) {
    return;
  }

  const float c = __bfloat162float(cos[row * rotary_half + dim]);
  const float s = __bfloat162float(sin[row * rotary_half + dim]);
  const int first_head = blockIdx.y * HEADS_PER_BLOCK;
  const int row_offset = row * nheads * headdim;

  #pragma unroll
  for (int local_head = 0; local_head < HEADS_PER_BLOCK; ++local_head) {
    const int head = first_head + local_head;
    if (head >= nheads) {
      continue;
    }
    const int head_offset = row_offset + head * headdim;
    if constexpr (INTERLEAVED) {
      const int offset = head_offset + 2 * dim;
      const __nv_bfloat162 packed =
          *reinterpret_cast<const __nv_bfloat162*>(x + offset);
      const float2 values = __bfloat1622float2(packed);
      const float y0 = values.x * c - values.y * s;
      const float y1 = values.x * s + values.y * c;
      *reinterpret_cast<__nv_bfloat162*>(out + offset) =
          __floats2bfloat162_rn(y0, y1);
    } else {
      const int offset = head_offset + dim;
      const float x0 = __bfloat162float(x[offset]);
      const float x1 = __bfloat162float(x[offset + rotary_half]);
      out[offset] = __float2bfloat16_rn(x0 * c - x1 * s);
      out[offset + rotary_half] = __float2bfloat16_rn(x0 * s + x1 * c);
    }
  }
}

template <int HEADS_PER_BLOCK, int ROWS_PER_BLOCK>
void launch_rope(
    const torch::Tensor& x,
    const torch::Tensor& cos,
    const torch::Tensor& sin,
    torch::Tensor& out,
    bool interleaved) {
  const int seqlen = x.size(1);
  const int nheads = x.size(2);
  const int headdim = x.size(3);
  const int rotary_half = cos.size(1);
  const dim3 grid(
      (seqlen + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK,
      (nheads + HEADS_PER_BLOCK - 1) / HEADS_PER_BLOCK);
  const dim3 block(64 * ROWS_PER_BLOCK);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const auto* x_ptr =
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>());
  const auto* cos_ptr =
      reinterpret_cast<const __nv_bfloat16*>(cos.data_ptr<at::BFloat16>());
  const auto* sin_ptr =
      reinterpret_cast<const __nv_bfloat16*>(sin.data_ptr<at::BFloat16>());
  auto* out_ptr =
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>());
  if (interleaved) {
    rope_kernel<HEADS_PER_BLOCK, ROWS_PER_BLOCK, true><<<grid, block, 0, stream>>>(
        x_ptr, cos_ptr, sin_ptr, out_ptr, seqlen, nheads, headdim, rotary_half);
  } else {
    rope_kernel<HEADS_PER_BLOCK, ROWS_PER_BLOCK, false><<<grid, block, 0, stream>>>(
        x_ptr, cos_ptr, sin_ptr, out_ptr, seqlen, nheads, headdim, rotary_half);
  }
}

torch::Tensor rope(
    const torch::Tensor& x,
    const torch::Tensor& cos,
    const torch::Tensor& sin,
    bool interleaved) {
  auto out = torch::empty_like(x);
  launch_rope<8, 4>(x, cos, sin, out, interleaved);
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rope", &rope, "Diffusion RoPE");
}
"""


_extension = load_inline(
    name="diffusion_rope_cuda_ext_v5",
    cpp_sources="",
    cuda_sources=_CUDA_SOURCE,
    extra_cuda_cflags=["-O3"],
    with_cuda=True,
    verbose=False,
)


class DiffusionRoPE(nn.Module):
    def __init__(self, is_neox_style: bool = False) -> None:
        super().__init__()
        self.interleaved = not is_neox_style

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        if cos.dim() == 3:
            cos = cos[0]
            sin = sin[0]

        return _extension.rope(x, cos, sin, self.interleaved)
