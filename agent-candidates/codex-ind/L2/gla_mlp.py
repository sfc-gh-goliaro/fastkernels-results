"""Fused SwiGLU MLP kernels for the fixed-width GLA decoder."""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch.utils.cpp_extension import load_inline

from ..L1.linear import Linear


_CPP_SRC = r"""
void gla_mlp_m1(
    torch::Tensor x, torch::Tensor gate, torch::Tensor up,
    torch::Tensor down, torch::Tensor mid, torch::Tensor out);
"""

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {

constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;
constexpr int ROWS = 4;

__device__ __forceinline__ float pair_dot(__nv_bfloat162 a, __nv_bfloat162 b) {
    const float2 product = __bfloat1622float2(__hmul2(a, b));
    return product.x + product.y;
}

__global__ void gate_up_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    __nv_bfloat16* __restrict__ out) {
    constexpr int K = 2560;
    constexpr int N = 6912;
    __shared__ float sums[WARPS][2 * ROWS];
    float acc[2 * ROWS] = {};
    const int first_row = blockIdx.x * ROWS;
    const auto* x2 = reinterpret_cast<const __nv_bfloat162*>(x);

    for (int k2 = threadIdx.x; k2 < K / 2; k2 += THREADS) {
        const __nv_bfloat162 xv = x2[k2];
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            const int row = first_row + r;
            if (row < N) {
                const auto* g2 = reinterpret_cast<const __nv_bfloat162*>(gate + row * K);
                const auto* u2 = reinterpret_cast<const __nv_bfloat162*>(up + row * K);
                acc[2 * r] += pair_dot(xv, g2[k2]);
                acc[2 * r + 1] += pair_dot(xv, u2[k2]);
            }
        }
    }

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
#pragma unroll
    for (int c = 0; c < 2 * ROWS; ++c) {
        float v = acc[c];
#pragma unroll
        for (int delta = 16; delta > 0; delta >>= 1)
            v += __shfl_down_sync(0xffffffff, v, delta);
        if (lane == 0) sums[warp][c] = v;
    }
    __syncthreads();
    if (warp == 0 && lane < 2 * ROWS) {
        float v = 0.0f;
#pragma unroll
        for (int w = 0; w < WARPS; ++w) v += sums[w][lane];
        sums[0][lane] = v;
    }
    __syncthreads();

    if (threadIdx.x < ROWS) {
        const int row = first_row + threadIdx.x;
        if (row < N) {
            const float g = __bfloat162float(__float2bfloat16_rn(sums[0][2 * threadIdx.x]));
            const float u = __bfloat162float(__float2bfloat16_rn(sums[0][2 * threadIdx.x + 1]));
            const float silu = __bfloat162float(
                __float2bfloat16_rn(g / (1.0f + __expf(-g))));
            out[row] = __float2bfloat16_rn(silu * u);
        }
    }
}

__global__ void down_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ out) {
    constexpr int K = 6912;
    constexpr int N = 2560;
    __shared__ float sums[WARPS][ROWS];
    float acc[ROWS] = {};
    const int first_row = blockIdx.x * ROWS;
    const auto* x2 = reinterpret_cast<const __nv_bfloat162*>(x);

    for (int k2 = threadIdx.x; k2 < K / 2; k2 += THREADS) {
        const __nv_bfloat162 xv = x2[k2];
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            const int row = first_row + r;
            if (row < N) {
                const auto* w2 = reinterpret_cast<const __nv_bfloat162*>(weight + row * K);
                acc[r] += pair_dot(xv, w2[k2]);
            }
        }
    }

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        float v = acc[r];
#pragma unroll
        for (int delta = 16; delta > 0; delta >>= 1)
            v += __shfl_down_sync(0xffffffff, v, delta);
        if (lane == 0) sums[warp][r] = v;
    }
    __syncthreads();
    if (warp == 0 && lane < ROWS) {
        float v = 0.0f;
#pragma unroll
        for (int w = 0; w < WARPS; ++w) v += sums[w][lane];
        sums[0][lane] = v;
    }
    __syncthreads();
    if (threadIdx.x < ROWS) {
        const int row = first_row + threadIdx.x;
        if (row < N) out[row] = __float2bfloat16_rn(sums[0][threadIdx.x]);
    }
}

}  // namespace

void gla_mlp_m1(
    torch::Tensor x, torch::Tensor gate, torch::Tensor up,
    torch::Tensor down, torch::Tensor mid, torch::Tensor out) {
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    gate_up_kernel<<<(6912 + ROWS - 1) / ROWS, THREADS, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(gate.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(up.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(mid.data_ptr<at::BFloat16>()));
    down_kernel<<<(2560 + ROWS - 1) / ROWS, THREADS, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(mid.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(down.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()));
}

"""

_GEMV = load_inline(
    name="gla_mlp_gemv_ext",
    cpp_sources=_CPP_SRC,
    cuda_sources=_CUDA_SRC,
    functions=["gla_mlp_m1"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    with_cuda=True,
    verbose=False,
)


_SCRATCH = {}


def _alloc_scratch(size, alignment, stream):
    key = (torch.cuda.current_device(), stream, size)
    buf = _SCRATCH.get(key)
    if buf is None:
        buf = torch.empty(size, device="cuda", dtype=torch.uint8)
        _SCRATCH[key] = buf
    return buf


triton.set_allocator(_alloc_scratch)


@triton.jit
def _tile_pid(tile_id, num_m, num_n, BM: tl.constexpr):
    group = 8
    width = group * num_n
    group_id = tile_id // width
    first_m = group_id * group
    group_m = tl.minimum(num_m - first_m, group)
    pid_m = first_m + tile_id % group_m
    pid_n = tile_id % width // group_m
    return pid_m, pid_n


@triton.jit
def _swiglu_ws_kernel(
    x,
    packed_weight,
    out,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    NUM_SMS: tl.constexpr,
    WS: tl.constexpr,
):
    x_desc = tl.make_tensor_descriptor(
        x, shape=[M, K], strides=[K, 1], block_shape=[BM, BK]
    )
    weight_desc = tl.make_tensor_descriptor(
        packed_weight,
        shape=[2 * N, K],
        strides=[K, 1],
        block_shape=[2 * BN, BK],
    )
    num_m = tl.cdiv(M, BM)
    num_n = tl.cdiv(N, BN)
    num_tiles = num_m * num_n

    for tile_id in tl.range(
        tl.program_id(0), num_tiles, NUM_SMS, flatten=True, warp_specialize=WS
    ):
        pid_m, pid_n = _tile_pid(tile_id, num_m, num_n, BM)
        off_m = pid_m * BM
        off_n = pid_n * BN
        acc = tl.zeros((BM, 2 * BN), tl.float32)
        for off_k in range(0, K, BK):
            a = tl.load_tensor_descriptor(x_desc, [off_m, off_k])
            w = tl.load_tensor_descriptor(weight_desc, [2 * off_n, off_k])
            acc += tl.dot(a, tl.trans(w))

        paired = tl.reshape(acc, (BM, BN, 2))
        gate, up = tl.split(paired)
        gate = gate.to(tl.bfloat16)
        up = up.to(tl.bfloat16)
        silu = (gate.to(tl.float32) * tl.sigmoid(gate.to(tl.float32))).to(tl.bfloat16)
        value = (silu.to(tl.float32) * up.to(tl.float32)).to(tl.bfloat16)
        offs_m = off_m + tl.arange(0, BM)
        offs_n = off_n + tl.arange(0, BN)
        tl.store(
            out + offs_m[:, None] * N + offs_n[None, :],
            value,
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        )
    tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _down_ws_kernel(
    x,
    weight,
    out,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    NUM_SMS: tl.constexpr,
    WS: tl.constexpr,
):
    tl.extra.cuda.gdc_wait()
    x_desc = tl.make_tensor_descriptor(
        x, shape=[M, K], strides=[K, 1], block_shape=[BM, BK]
    )
    weight_desc = tl.make_tensor_descriptor(
        weight, shape=[N, K], strides=[K, 1], block_shape=[BN, BK]
    )
    num_m = tl.cdiv(M, BM)
    num_n = tl.cdiv(N, BN)
    num_tiles = num_m * num_n

    for tile_id in tl.range(
        tl.program_id(0), num_tiles, NUM_SMS, flatten=True, warp_specialize=WS
    ):
        pid_m, pid_n = _tile_pid(tile_id, num_m, num_n, BM)
        off_m = pid_m * BM
        off_n = pid_n * BN
        acc = tl.zeros((BM, BN), tl.float32)
        for off_k in range(0, K, BK):
            a = tl.load_tensor_descriptor(x_desc, [off_m, off_k])
            w = tl.load_tensor_descriptor(weight_desc, [off_n, off_k])
            acc += tl.dot(a, tl.trans(w))
        offs_m = off_m + tl.arange(0, BM)
        offs_n = off_n + tl.arange(0, BN)
        tl.store(
            out + offs_m[:, None] * N + offs_n[None, :],
            acc,
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        )


@triton.jit
def _down_splitk_kernel(
    x,
    weight,
    partials,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SPLIT_K: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    tl.extra.cuda.gdc_wait()
    x_desc = tl.make_tensor_descriptor(
        x, shape=[M, K], strides=[K, 1], block_shape=[BM, BK]
    )
    weight_desc = tl.make_tensor_descriptor(
        weight, shape=[N, K], strides=[K, 1], block_shape=[BN, BK]
    )
    num_m = tl.cdiv(M, BM)
    num_n = tl.cdiv(N, BN)
    num_tiles = num_m * num_n * SPLIT_K
    split_size = K // SPLIT_K

    for work_id in tl.range(
        tl.program_id(0), num_tiles, NUM_SMS, flatten=True, warp_specialize=True
    ):
        split_id = work_id % SPLIT_K
        tile_id = work_id // SPLIT_K
        pid_m, pid_n = _tile_pid(tile_id, num_m, num_n, BM)
        off_m = pid_m * BM
        off_n = pid_n * BN
        acc = tl.zeros((BM, BN), tl.float32)
        split_start = split_id * split_size
        for k_delta in range(0, split_size, BK):
            off_k = split_start + k_delta
            a = tl.load_tensor_descriptor(x_desc, [off_m, off_k])
            w = tl.load_tensor_descriptor(weight_desc, [off_n, off_k])
            acc += tl.dot(a, tl.trans(w))
        offs_m = off_m + tl.arange(0, BM)
        offs_n = off_n + tl.arange(0, BN)
        mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        offsets = split_id * M * N + offs_m[:, None] * N + offs_n[None, :]
        tl.store(partials + offsets, acc, mask=mask)
    tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _reduce_splitk_kernel(
    partials,
    out,
    n_elements: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tl.extra.cuda.gdc_wait()
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    acc = tl.zeros((BLOCK,), tl.float32)
    for split_id in tl.static_range(SPLIT_K):
        acc += tl.load(
            partials + split_id * n_elements + offsets, mask=mask, other=0.0
        )
    tl.store(out + offsets, acc, mask=mask)


class GLAMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = Linear(intermediate_size, hidden_size, bias=False)
        self.register_buffer("_packed_weight", torch.empty(0), persistent=False)

    def load_state_dict(self, state_dict, *args, **kwargs):
        result = super().load_state_dict(state_dict, *args, **kwargs)
        # Interleaved rows let one Blackwell MMA produce gate/up pairs.
        self._packed_weight = torch.stack(
            (self.gate_proj.weight, self.up_proj.weight), dim=1
        ).reshape(-1, self.gate_proj.weight.shape[1]).contiguous()
        return result

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        m = x.numel() // shape[-1]
        hidden = shape[-1]
        intermediate = self.gate_proj.weight.shape[0]
        x2 = x.reshape(m, hidden)
        if m == 1:
            fused = torch.empty((1, intermediate), device=x.device, dtype=x.dtype)
            out = torch.empty((1, hidden), device=x.device, dtype=x.dtype)
            _GEMV.gla_mlp_m1(
                x2,
                self.gate_proj.weight,
                self.up_proj.weight,
                self.down_proj.weight,
                fused,
                out,
            )
            return out.reshape(*shape[:-1], hidden)

        fused = torch.empty((m, intermediate), device=x.device, dtype=x.dtype)
        if m <= 64:
            _swiglu_ws_kernel[(min(148, triton.cdiv(m, 64) * triton.cdiv(intermediate, 64)),)](
                x2,
                self._packed_weight,
                fused,
                M=m,
                N=intermediate,
                K=hidden,
                BM=64,
                BN=64,
                BK=64,
                NUM_SMS=148,
                WS=True,
                num_warps=4,
                num_stages=4,
            )
        elif m <= 128:
            _swiglu_ws_kernel[(min(148, triton.cdiv(m, 128) * triton.cdiv(intermediate, 64)),)](
                x2,
                self._packed_weight,
                fused,
                M=m,
                N=intermediate,
                K=hidden,
                BM=128,
                BN=64,
                BK=64,
                NUM_SMS=148,
                WS=True,
                num_warps=4,
                num_stages=4,
            )
        else:
            _swiglu_ws_kernel[(min(148, triton.cdiv(m, 128) * triton.cdiv(intermediate, 128)),)](
                x2,
                self._packed_weight,
                fused,
                M=m,
                N=intermediate,
                K=hidden,
                BM=128,
                BN=128,
                BK=64,
                NUM_SMS=148,
                WS=True,
                num_warps=8,
                num_stages=4,
            )
        out = torch.empty((m, hidden), device=x.device, dtype=x.dtype)
        if m <= 64:
            partials = torch.empty(
                (6, m, hidden), device=x.device, dtype=x.dtype
            )
            grid = min(
                148,
                triton.cdiv(m, 64) * triton.cdiv(hidden, 128) * 6,
            )
            _down_splitk_kernel[(grid,)](
                fused,
                self.down_proj.weight,
                partials,
                M=m,
                N=hidden,
                K=intermediate,
                BM=64,
                BN=128,
                BK=128,
                SPLIT_K=6,
                NUM_SMS=148,
                num_warps=8,
                num_stages=3,
                launch_pdl=True,
            )
            _reduce_splitk_kernel[(triton.cdiv(m * hidden, 256),)](
                partials,
                out,
                n_elements=m * hidden,
                SPLIT_K=6,
                BLOCK=256,
                num_warps=8,
                launch_pdl=True,
            )
        elif m <= 128:
            partials = torch.empty(
                (3, m, hidden), device=x.device, dtype=x.dtype
            )
            grid = min(
                148,
                triton.cdiv(m, 128) * triton.cdiv(hidden, 64) * 3,
            )
            _down_splitk_kernel[(grid,)](
                fused,
                self.down_proj.weight,
                partials,
                M=m,
                N=hidden,
                K=intermediate,
                BM=128,
                BN=64,
                BK=128,
                SPLIT_K=3,
                NUM_SMS=148,
                num_warps=8,
                num_stages=4,
                launch_pdl=True,
            )
            _reduce_splitk_kernel[(triton.cdiv(m * hidden, 1024),)](
                partials,
                out,
                n_elements=m * hidden,
                SPLIT_K=3,
                BLOCK=1024,
                num_warps=8,
                launch_pdl=True,
            )
        elif m == 256:
            partials = torch.empty(
                (3, m, hidden), device=x.device, dtype=x.dtype
            )
            grid = min(
                148,
                triton.cdiv(m, 128) * triton.cdiv(hidden, 128) * 3,
            )
            _down_splitk_kernel[(grid,)](
                fused,
                self.down_proj.weight,
                partials,
                M=m,
                N=hidden,
                K=intermediate,
                BM=128,
                BN=128,
                BK=128,
                SPLIT_K=3,
                NUM_SMS=148,
                num_warps=8,
                num_stages=3,
                launch_pdl=True,
            )
            _reduce_splitk_kernel[(triton.cdiv(m * hidden, 1024),)](
                partials,
                out,
                n_elements=m * hidden,
                SPLIT_K=3,
                BLOCK=1024,
                num_warps=8,
                launch_pdl=True,
            )
        elif 64 <= m <= 256:
            partials = torch.empty(
                (6, m, hidden), device=x.device, dtype=x.dtype
            )
            grid = min(
                148,
                triton.cdiv(m, 64) * triton.cdiv(hidden, 128) * 6,
            )
            _down_splitk_kernel[(grid,)](
                fused,
                self.down_proj.weight,
                partials,
                M=m,
                N=hidden,
                K=intermediate,
                BM=64,
                BN=128,
                BK=128,
                SPLIT_K=6,
                NUM_SMS=148,
                num_warps=8,
                num_stages=3,
                launch_pdl=True,
            )
            _reduce_splitk_kernel[(triton.cdiv(m * hidden, 256),)](
                partials,
                out,
                n_elements=m * hidden,
                SPLIT_K=6,
                BLOCK=256,
                num_warps=8,
                launch_pdl=True,
            )
        elif m > 256:
            _down_ws_kernel[(min(148, triton.cdiv(m, 128) * triton.cdiv(hidden, 256)),)](
                fused,
                self.down_proj.weight,
                out,
                M=m,
                N=hidden,
                K=intermediate,
                BM=128,
                BN=256,
                BK=64,
                NUM_SMS=148,
                WS=True,
                num_warps=8,
                num_stages=4,
                launch_pdl=True,
            )
        return out.reshape(*shape[:-1], hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)
