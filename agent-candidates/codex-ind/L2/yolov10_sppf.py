"""YOLOv10 Spatial Pyramid Pooling - Fast."""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch.utils.cpp_extension import load_inline

from ..L1.max_pool2d import MaxPool2d
from .yolov10_conv import YOLOConv


_POOL_CPP = """
void yolov10_sppf_pool_out(torch::Tensor x, torch::Tensor out);
"""

_POOL_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>

__global__ void sppf_pool_kernel(const half* __restrict__ x,
                                 half* __restrict__ out) {
    __shared__ half first[400];
    __shared__ half second[400];
    const int plane = blockIdx.x;
    const int batch = plane >> 7;
    const int channel = plane & 127;
    const half* src = x + plane * 400;
    half* base = out + batch * 512 * 400 + channel * 400;

    for (int i = threadIdx.x; i < 400; i += blockDim.x) {
        first[i] = src[i];
        base[i] = src[i];
    }
    __syncthreads();

    #pragma unroll
    for (int stage = 1; stage <= 3; ++stage) {
        for (int i = threadIdx.x; i < 400; i += blockDim.x) {
            const int h = i / 20;
            const int w = i - h * 20;
            half maximum = __float2half(-65504.0f);
            #pragma unroll
            for (int dx = -2; dx <= 2; ++dx) {
                const int xx = w + dx;
                if ((unsigned)xx < 20u) {
                    maximum = __hgt(first[h * 20 + xx], maximum)
                                  ? first[h * 20 + xx] : maximum;
                }
            }
            second[i] = maximum;
        }
        __syncthreads();
        for (int i = threadIdx.x; i < 400; i += blockDim.x) {
            const int h = i / 20;
            const int w = i - h * 20;
            half maximum = __float2half(-65504.0f);
            #pragma unroll
            for (int dy = -2; dy <= 2; ++dy) {
                const int yy = h + dy;
                if ((unsigned)yy < 20u) {
                    maximum = __hgt(second[yy * 20 + w], maximum)
                                  ? second[yy * 20 + w] : maximum;
                }
            }
            first[i] = maximum;
            base[stage * 128 * 400 + i] = maximum;
        }
        __syncthreads();
    }
}

void yolov10_sppf_pool_out(torch::Tensor x, torch::Tensor out) {
    sppf_pool_kernel<<<x.size(0) * 128, 512, 0,
                       at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()));
}
"""

_pool_ext = load_inline(
    name="yolov10_sppf_pool_ext",
    cpp_sources=_POOL_CPP,
    cuda_sources=_POOL_CUDA,
    functions=["yolov10_sppf_pool_out"],
    extra_cuda_cflags=["-O3"],
    verbose=False,
)


@triton.jit
def _conv1_bn_silu(
    x, weight, bn_weight, bn_bias, running_mean, running_var, out,
    batch: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ks = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    # NCHW is strided in the reduction dimension. The matrix rows are (n, h, w).
    for k0 in range(0, 256, BLOCK_K):
        k = k0 + ks
        x_off = (
            (rows[:, None] // 400) * (256 * 400)
            + k[None, :] * 400
            + (rows[:, None] % 400)
        )
        w_off = cols[None, :] * 256 + k[:, None]
        a = tl.load(x + x_off, mask=(rows[:, None] < batch * 400))
        b = tl.load(weight + w_off, mask=(cols[None, :] < 128))
        acc = tl.dot(a, b, acc)

    valid_col = cols < 128
    scale = tl.load(bn_weight + cols, mask=valid_col).to(tl.float32)
    var = tl.load(running_var + cols, mask=valid_col).to(tl.float32)
    mean = tl.load(running_mean + cols, mask=valid_col).to(tl.float32)
    bias = tl.load(bn_bias + cols, mask=valid_col).to(tl.float32)
    value = acc * (scale * tl.rsqrt(var + 1.0e-3)) + (
        bias - mean * scale * tl.rsqrt(var + 1.0e-3)
    )
    value = value * tl.sigmoid(value)
    out_off = (
        (rows[:, None] // 400) * (128 * 400)
        + cols[None, :] * 400
        + (rows[:, None] % 400)
    )
    tl.store(
        out + out_off, value,
        mask=(rows[:, None] < batch * 400) & valid_col[None, :],
    )


@triton.jit
def _conv2_bn_silu(
    x, weight, bn_weight, bn_bias, running_mean, running_var, out,
    batch: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ks = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    for k0 in range(0, 512, BLOCK_K):
        k = k0 + ks
        x_off = (
            (rows[:, None] // 400) * (512 * 400)
            + k[None, :] * 400
            + (rows[:, None] % 400)
        )
        w_off = cols[None, :] * 512 + k[:, None]
        a = tl.load(x + x_off, mask=(rows[:, None] < batch * 400))
        b = tl.load(weight + w_off, mask=(cols[None, :] < 256))
        acc = tl.dot(a, b, acc)

    valid_col = cols < 256
    scale = tl.load(bn_weight + cols, mask=valid_col).to(tl.float32)
    var = tl.load(running_var + cols, mask=valid_col).to(tl.float32)
    mean = tl.load(running_mean + cols, mask=valid_col).to(tl.float32)
    bias = tl.load(bn_bias + cols, mask=valid_col).to(tl.float32)
    inv_std = tl.rsqrt(var + 1.0e-3)
    value = acc * (scale * inv_std) + (bias - mean * scale * inv_std)
    value = value * tl.sigmoid(value)
    out_off = (
        (rows[:, None] // 400) * (256 * 400)
        + cols[None, :] * 400
        + (rows[:, None] % 400)
    )
    tl.store(
        out + out_off, value,
        mask=(rows[:, None] < batch * 400) & valid_col[None, :],
    )


class YOLOSPPF(nn.Module):
    def __init__(self, c1: int, c2: int, k: int = 5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = YOLOConv(c1, c_, 1, 1)
        self.cv2 = YOLOConv(c_ * 4, c2, 1, 1)
        self.m = MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self._fast_buffers = None

    def _launch_fast(self, x, hidden, pyramid, out, batch):
        _conv1_bn_silu[
            (triton.cdiv(batch * 400, 32), triton.cdiv(128, 32))
        ](
            x, self.cv1.conv.weight,
            self.cv1.bn.weight, self.cv1.bn.bias,
            self.cv1.bn.running_mean, self.cv1.bn.running_var,
            hidden, batch,
            BLOCK_M=32, BLOCK_N=32, BLOCK_K=64,
            num_warps=4, num_stages=4,
        )
        _pool_ext.yolov10_sppf_pool_out(hidden, pyramid)
        _conv2_bn_silu[
            (triton.cdiv(batch * 400, 32), triton.cdiv(256, 32))
        ](
            pyramid, self.cv2.conv.weight,
            self.cv2.bn.weight, self.cv2.bn.bias,
            self.cv2.bn.running_mean, self.cv2.bn.running_var,
            out, batch,
            BLOCK_M=32, BLOCK_N=32, BLOCK_K=64,
            num_warps=4, num_stages=4,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            not self.training
            and x.is_cuda
            and x.dtype == torch.float16
            and x.shape[1:] == (256, 20, 20)
            and self.m.kernel_size == 5
        ):
            batch = x.shape[0]
            if self._fast_buffers is None or self._fast_buffers[0].shape[0] != batch:
                workspace = torch.empty(
                    (batch * (128 + 512 + 256) * 400,),
                    device=x.device, dtype=x.dtype,
                )
                hidden_end = batch * 128 * 400
                pyramid_end = hidden_end + batch * 512 * 400
                self._fast_buffers = (
                    workspace[:hidden_end].view(batch, 128, 20, 20),
                    workspace[hidden_end:pyramid_end].view(batch, 512, 20, 20),
                    workspace[pyramid_end:].view(batch, 256, 20, 20),
                )
            hidden, pyramid, out = self._fast_buffers
            self._launch_fast(x, hidden, pyramid, out, batch)
            return out

        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))
