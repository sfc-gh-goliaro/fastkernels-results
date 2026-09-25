"""YOLOv10 Spatial Pyramid Pooling - Fast."""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch.utils.cpp_extension import load_inline

from ..L1.max_pool2d import MaxPool2d
from .yolov10_conv import YOLOConv


_CPP_SRC = r"""
torch::Tensor sppf_pool_cat_cuda(torch::Tensor x);
"""

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>

namespace {

__device__ __forceinline__ __half half_max(__half a, __half b) {
    return __hgt(b, a) ? b : a;
}

template<int START, int END>
__device__ __forceinline__ __half extend_horizontal(
    const __half* tile, int index, int col, __half value) {
#pragma unroll
    for (int distance = START; distance <= END; ++distance) {
        if (col >= distance) {
            value = half_max(value, tile[index - distance]);
        }
        if (col + distance < 20) {
            value = half_max(value, tile[index + distance]);
        }
    }
    return value;
}

template<int RADIUS>
__device__ __forceinline__ __half vertical_max(
    const __half* tile, int index, int row) {
    __half value = tile[index];
#pragma unroll
    for (int distance = 1; distance <= RADIUS; ++distance) {
        if (row >= distance) {
            value = half_max(value, tile[index - distance * 20]);
        }
        if (row + distance < 20) {
            value = half_max(value, tile[index + distance * 20]);
        }
    }
    return value;
}

__global__ __launch_bounds__(256)
void sppf_pool_cat_kernel(
    const __half* __restrict__ input,
    __half* __restrict__ output,
    int channels) {
    __shared__ __half source[400];
    __shared__ __half horizontal2[400];
    __shared__ __half horizontal4[400];
    __shared__ __half horizontal6[400];

    const int tid = threadIdx.x;
    const int plane = blockIdx.x;
    const int channel = plane % channels;
    const int batch = plane / channels;
    const int input_offset = plane * 400;
    const int output_offset = (batch * (4 * channels) + channel) * 400;

    for (int index = tid; index < 400; index += 256) {
        const __half value = input[input_offset + index];
        const int col = index % 20;
        source[index] = value;
        output[output_offset + index] = value;
    }
    __syncthreads();

    for (int index = tid; index < 400; index += 256) {
        const int col = index % 20;
        const __half value2 =
            extend_horizontal<1, 2>(source, index, col, source[index]);
        const __half value4 =
            extend_horizontal<3, 4>(source, index, col, value2);
        horizontal2[index] = value2;
        horizontal4[index] = value4;
        horizontal6[index] =
            extend_horizontal<5, 6>(source, index, col, value4);
    }
    __syncthreads();

    for (int index = tid; index < 400; index += 256) {
        const int row = index / 20;
        output[output_offset + channels * 400 + index] =
            vertical_max<2>(horizontal2, index, row);
        output[output_offset + 2 * channels * 400 + index] =
            vertical_max<4>(horizontal4, index, row);
        output[output_offset + 3 * channels * 400 + index] =
            vertical_max<6>(horizontal6, index, row);
    }
}

} // namespace

torch::Tensor sppf_pool_cat_cuda(torch::Tensor x) {
    const int batch = x.size(0);
    const int channels = x.size(1);
    auto output = torch::empty(
        {batch, 4 * channels, 20, 20}, x.options());
    sppf_pool_cat_kernel<<<batch * channels, 256, 0,
        at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
        channels);
    return output;
}
"""


_cuda = load_inline(
    name="fk_yolov10_sppf_pool_cat_v2",
    cpp_sources=_CPP_SRC,
    cuda_sources=_CUDA_SRC,
    functions=["sppf_pool_cat_cuda"],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
    verbose=False,
)


@triton.jit
def _pointwise_silu_kernel(
    x,
    weight,
    bias,
    out,
    C: tl.constexpr,
    N: tl.constexpr,
    S: tl.constexpr,
    B: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    image = offs_m // S
    position = offs_m % S
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, C, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        a = tl.load(
            x + image[:, None] * (C * S)
            + offs_k[None, :] * S + position[:, None],
            mask=(offs_m[:, None] < B * S) & (offs_k[None, :] < C),
            other=0.0,
        )
        w = tl.load(
            weight + offs_n[None, :] * C + offs_k[:, None],
            mask=(offs_n[None, :] < N) & (offs_k[:, None] < C),
            other=0.0,
        )
        acc = tl.dot(a, w, acc)

    acc += tl.load(bias + offs_n, mask=offs_n < N, other=0.0)[None, :]
    acc = acc * tl.sigmoid(acc)
    tl.store(
        out + image[:, None] * (N * S)
        + offs_n[None, :] * S + position[:, None],
        acc,
        mask=(offs_m[:, None] < B * S) & (offs_n[None, :] < N),
    )


def _pointwise_silu(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    batch, _, height, width = x.shape
    out_channels, channels = weight.shape[:2]
    spatial = height * width
    out = torch.empty(
        (batch, out_channels, height, width), device=x.device, dtype=x.dtype
    )
    block_m = 16
    block_n = 64
    block_k = 128
    _pointwise_silu_kernel[
        (triton.cdiv(batch * spatial, block_m), triton.cdiv(out_channels, block_n))
    ](
        x,
        weight,
        bias,
        out,
        C=channels,
        N=out_channels,
        S=spatial,
        B=batch,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
        num_stages=4,
    )
    return out


class YOLOSPPF(nn.Module):
    def __init__(self, c1: int, c2: int, k: int = 5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = YOLOConv(c1, c_, 1, 1)
        self.cv2 = YOLOConv(c_ * 4, c2, 1, 1)
        self.m = MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.cv1._is_fused:
            self.cv1.fuse()
            self.cv2.fuse()
        captured = (
            x.is_cuda
            and x.dtype == torch.float16
            and x.shape[1:] == (256, 20, 20)
        )
        if captured:
            x = _pointwise_silu(x, self.cv1.conv.weight, self.cv1.conv.bias)
            x = _cuda.sppf_pool_cat_cuda(x)
            return _pointwise_silu(x, self.cv2.conv.weight, self.cv2.conv.bias)
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))
