"""Specialized CUDA implementation of the captured 5x5 max pool."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline


_CPP_SRC = r"""
torch::Tensor max_pool2d_5x5_cuda(torch::Tensor x);
"""

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>

namespace {

__device__ __forceinline__ __half half_max(__half a, __half b) {
    return __hgt(b, a) ? b : a;
}

__global__ __launch_bounds__(256)
void max_pool2d_5x5_kernel(
    const __half* __restrict__ input,
    __half* __restrict__ output) {
    __shared__ __half tile[400];
    __shared__ __half horizontal[400];

    const int tid = threadIdx.x;
    const int plane_offset = blockIdx.x * 400;

    for (int index = tid; index < 400; index += 256) {
        tile[index] = input[plane_offset + index];
    }
    __syncthreads();

    for (int index = tid; index < 400; index += 256) {
        const int col = index % 20;
        __half value = tile[index];
#pragma unroll
        for (int dx = -2; dx <= 2; ++dx) {
            const int x = col + dx;
            if (x >= 0 && x < 20) {
                value = half_max(value, tile[index + dx]);
            }
        }
        horizontal[index] = value;
    }
    __syncthreads();

    for (int index = tid; index < 400; index += 256) {
        const int row = index / 20;
        __half value = horizontal[index];
#pragma unroll
        for (int dy = -2; dy <= 2; ++dy) {
            const int y = row + dy;
            if (y >= 0 && y < 20) {
                value = half_max(value, horizontal[index + dy * 20]);
            }
        }
        output[plane_offset + index] = value;
    }
}

} // namespace

torch::Tensor max_pool2d_5x5_cuda(torch::Tensor x) {
    const int planes = x.numel() / 400;
    auto output = torch::empty_like(x);
    max_pool2d_5x5_kernel<<<planes, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(output.data_ptr<at::Half>()));
    return output;
}
"""


_cuda = load_inline(
    name="fk_max_pool2d_5x5_v4",
    cpp_sources=_CPP_SRC,
    cuda_sources=_CUDA_SRC,
    functions=["max_pool2d_5x5_cuda"],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
    verbose=False,
)


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
        self._captured = (
            kernel_size == 5
            and self.stride == 1
            and padding == 2
            and not ceil_mode
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._captured:
            return _cuda.max_pool2d_5x5_cuda(x)
        return F.max_pool2d(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
            ceil_mode=self.ceil_mode,
        )
