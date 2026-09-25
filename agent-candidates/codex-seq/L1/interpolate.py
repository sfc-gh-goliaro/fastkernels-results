"""Interpolate wrapping F.interpolate."""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline


os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "10.0a")

_CPP_SRC = """
torch::Tensor interpolate_nearest2x_cuda(torch::Tensor x);
"""

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>

template <int WIDTH>
__global__ void nearest2x_kernel(const half* __restrict__ x,
                                  half* __restrict__ out,
                                  int elements) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;

    int row = index / WIDTH;
    int column = index - row * WIDTH;
    int output_pair = row * (2 * WIDTH) + column;
    half value = x[index];
    half2 pair = __halves2half2(value, value);
    reinterpret_cast<half2*>(out)[output_pair] = pair;
    reinterpret_cast<half2*>(out)[output_pair + WIDTH] = pair;
}

torch::Tensor interpolate_nearest2x_cuda(torch::Tensor x) {
    c10::cuda::CUDAGuard device_guard(x.device());
    auto out = torch::empty(
        {x.size(0), x.size(1), x.size(2) * 2, x.size(3) * 2},
        x.options());
    int threads = x.size(0) == 1 ? 256 : 1024;
    int width = static_cast<int>(x.size(3));
    int elements = static_cast<int>(x.numel());
    int blocks = (elements + threads - 1) / threads;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (width == 20) {
        nearest2x_kernel<20><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
            reinterpret_cast<half*>(out.data_ptr<at::Half>()), elements);
    } else {
        nearest2x_kernel<40><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
            reinterpret_cast<half*>(out.data_ptr<at::Half>()), elements);
    }
    return out;
}
"""

_cuda = load_inline(
    name="fk_interpolate_nearest2x_v3",
    cpp_sources=_CPP_SRC,
    cuda_sources=_CUDA_SRC,
    functions=["interpolate_nearest2x_cuda"],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
    verbose=False,
)


class Interpolate(nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        size: int | tuple[int, ...] | None = None,
        scale_factor: float | tuple[float, ...] | None = None,
        mode: str = "nearest",
        align_corners: bool | None = None,
    ) -> torch.Tensor:
        if (
            size is None
            and scale_factor == 2.0
            and mode == "nearest"
            and align_corners is None
            and x.is_cuda
            and x.dtype == torch.float16
            and x.is_contiguous()
            and x.ndim == 4
            and x.shape[3] in (20, 40)
        ):
            return _cuda.interpolate_nearest2x_cuda(x)
        return F.interpolate(
            x,
            size=size,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=align_corners,
        )
