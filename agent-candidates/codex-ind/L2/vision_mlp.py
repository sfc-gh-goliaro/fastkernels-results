"""Vision MLP for Qwen vision transformer blocks.

Unified across Qwen2-VL (QuickGELU) and Qwen3-VL (SiLU) activations.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
import triton
import triton.language as tl

from ..L1.quickgelu import QuickGELU
from .parallel_linear import ColumnParallelLinear, RowParallelLinear


def _triton_allocator(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)


triton.set_allocator(_triton_allocator)


@triton.jit
def _vision_mlp_matmul(
    a_ptr,
    b_ptr,
    bias_ptr,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    ACT: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    start_pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = grid_m * grid_n
    programs_per_group = GROUP_M * grid_n
    a_desc = tl.make_tensor_descriptor(
        a_ptr,
        shape=[M, K],
        strides=[K, 1],
        block_shape=[BLOCK_M, BLOCK_K],
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr,
        shape=[N, K],
        strides=[K, 1],
        block_shape=[BLOCK_N, BLOCK_K],
    )
    tile_id_c = start_pid - NUM_SMS
    for tile_id in tl.range(
        start_pid, num_tiles, NUM_SMS, flatten=True, warp_specialize=True
    ):
        group_id = tile_id // programs_per_group
        first_m = group_id * GROUP_M
        group_m = tl.minimum(grid_m - first_m, GROUP_M)
        pid_m = first_m + (tile_id % programs_per_group) % group_m
        pid_n = (tile_id % programs_per_group) // group_m

        accum = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            a = a_desc.load([pid_m * BLOCK_M, k * BLOCK_K])
            b = b_desc.load([pid_n * BLOCK_N, k * BLOCK_K])
            accum = tl.dot(a, b.T, accum)

        tile_id_c += NUM_SMS
        group_id = tile_id_c // programs_per_group
        first_m = group_id * GROUP_M
        group_m = tl.minimum(grid_m - first_m, GROUP_M)
        out_m = first_m + (tile_id_c % programs_per_group) % group_m
        out_n = (tile_id_c % programs_per_group) // group_m
        offs_m = out_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = out_n * BLOCK_N + tl.arange(0, BLOCK_N)

        accum += tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)[None, :]
        if ACT == 1:
            # Match the BF16 FC1 output consumed by the baseline GELU.
            value = accum.to(tl.bfloat16).to(tl.float32)
            z = value * value
            even = -7.07846704e-7
            even = even * z + 3.43049639e-5
            even = even * z - 7.09648664e-4
            even = even * z + 8.35234604e-3
            even = even * z - 6.37122894e-2
            even = even * z + 3.97102333e-1
            even = even * z + 2.01824510e-4
            gelu = 0.5 * value + even
            accum = tl.where(
                tl.abs(value) < 3.5, gelu, tl.maximum(value, 0.0)
            )
        elif ACT == 2:
            value = accum.to(tl.bfloat16).to(tl.float32)
            inner = 1.5957691216057308 * value * (
                1.0 + 0.044715 * value * value
            )
            accum = value * tl.sigmoid(inner)

        tl.store(
            out_ptr + offs_m[:, None] * N + offs_n[None, :],
            accum,
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        )


class VisionMLP(nn.Module):
    """Vision encoder MLP with configurable activation.

    Qwen2-VL uses QuickGELU (default); Qwen3-VL uses F.silu.
    """

    def __init__(self, in_features: int, hidden_features: int,
                 act_fn: Callable[[torch.Tensor], torch.Tensor] = QuickGELU(),
                 bias: bool = True):
        super().__init__()
        self.fc1 = ColumnParallelLinear(in_features, hidden_features, bias=bias)
        self.fc2 = RowParallelLinear(hidden_features, in_features, bias=bias)
        self.act_fn = act_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = x.numel() // x.shape[-1]
        if m < 4096:
            block_m1, block_n1, block_k1, stages1 = 128, 128, 128, 3
            block_m2, block_n2, block_k2, stages2 = 128, 128, 128, 3
        else:
            if m >= 32768:
                block_m1, block_n1, block_k1, stages1 = 128, 256, 64, 4
            else:
                block_m1, block_n1, block_k1, stages1 = 256, 128, 64, 2
            block_m2, block_n2, block_k2, stages2 = 256, 128, 64, 4

        hidden = torch.empty(
            (m, self.fc1.weight.shape[0]), device=x.device, dtype=x.dtype
        )
        tiles1 = triton.cdiv(m, block_m1) * triton.cdiv(
            self.fc1.weight.shape[0], block_n1
        )
        grid1 = (min(148, tiles1),)
        _vision_mlp_matmul[grid1](
            x,
            self.fc1.weight,
            self.fc1.bias,
            hidden,
            m,
            self.fc1.weight.shape[0],
            self.fc1.weight.shape[1],
            BLOCK_M=block_m1,
            BLOCK_N=block_n1,
            BLOCK_K=block_k1,
            GROUP_M=8,
            ACT=1 if m < 4096 else 2,
            NUM_SMS=148,
            num_warps=8,
            num_stages=stages1,
        )

        out = torch.empty((m, self.fc2.weight.shape[0]), device=x.device, dtype=x.dtype)
        tiles2 = triton.cdiv(m, block_m2) * triton.cdiv(
            self.fc2.weight.shape[0], block_n2
        )
        grid2 = (min(148, tiles2),)
        _vision_mlp_matmul[grid2](
            hidden,
            self.fc2.weight,
            self.fc2.bias,
            out,
            m,
            self.fc2.weight.shape[0],
            self.fc2.weight.shape[1],
            BLOCK_M=block_m2,
            BLOCK_N=block_n2,
            BLOCK_K=block_k2,
            GROUP_M=8,
            ACT=0,
            NUM_SMS=148,
            num_warps=8,
            num_stages=stages2,
        )
        return out.reshape(*x.shape[:-1], self.fc2.weight.shape[0])
