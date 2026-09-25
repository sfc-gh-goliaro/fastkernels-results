"""T5 feed-forward dense layers with TP sharding (L2).

T5DenseActDense: standard FFN (ColumnParallel -> act -> RowParallel).
T5DenseGatedActDense: gated FFN (MergedColumnParallel -> gate*up -> RowParallel).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import triton
import triton.language as tl
from transformers import T5Config

from ..L1.gelu import GELU
from ..L1.silu import SiLU
from .parallel_linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)


__targets__ = ["T5DenseGatedActDense", "T5DenseActDense"]


@triton.jit
def _gated_gemm_kernel(
    x_ptr,
    wi_ptr,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    x_ptrs = x_ptr + offs_m[:, None] * K + offs_k[None, :]
    pair_cols = tl.arange(0, 2 * BLOCK_N)
    wi_ptrs = wi_ptr + offs_k[:, None] * (2 * N) + 2 * pid_n * BLOCK_N + pair_cols[None, :]

    gate_up = tl.zeros((BLOCK_M, 2 * BLOCK_N), tl.float32)
    for _ in range(0, K, BLOCK_K):
        x = tl.load(x_ptrs)
        wi = tl.load(wi_ptrs)
        gate_up = tl.dot(x, wi, acc=gate_up)
        x_ptrs += BLOCK_K
        wi_ptrs += BLOCK_K * (2 * N)

    # Round the projections as F.linear does before applying NewGELU.
    gate_up = tl.reshape(gate_up, (BLOCK_M, BLOCK_N, 2))
    gate, up = tl.split(gate_up)
    g = gate.to(tl.bfloat16)
    cube = (g * g * g).to(tl.bfloat16)
    inner = (g + (0.044715 * cube).to(tl.bfloat16)).to(tl.bfloat16)
    tanh_arg = (0.7978845608028654 * inner).to(tl.bfloat16)
    tanh_out = (2.0 * tl.sigmoid(2.0 * tanh_arg.to(tl.float32)) - 1.0).to(tl.bfloat16)
    gelu = ((0.5 * g).to(tl.bfloat16) * (1.0 + tanh_out).to(tl.bfloat16)).to(tl.bfloat16)
    out = (gelu * up.to(tl.bfloat16)).to(tl.bfloat16)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], out, mask=mask)


@triton.jit
def _gemm_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    x_ptrs = x_ptr + offs_m[:, None] * K + offs_k[None, :]
    w_ptrs = w_ptr + offs_k[:, None] * N + offs_n[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask = k * BLOCK_K + offs_k < K
        x = tl.load(x_ptrs, mask=k_mask[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=k_mask[:, None], other=0.0)
        acc = tl.dot(x, w, acc=acc)
        x_ptrs += BLOCK_K
        w_ptrs += BLOCK_K * N

    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], acc, mask=mask)


class NewGELUActivation(nn.Module):
    """GELU approximation matching HuggingFace's NewGELUActivation exactly."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))


def _get_act_fn(name: str) -> nn.Module:
    act_fns = {
        "relu": nn.ReLU(),
        "gelu": GELU(),
        "gelu_new": NewGELUActivation(),
        "silu": SiLU(),
    }
    if name in act_fns:
        return act_fns[name]
    raise ValueError(f"Unknown activation function: {name}")


class T5DenseGatedActDense(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.wi = MergedColumnParallelLinear(
            config.d_model, [config.d_ff, config.d_ff], bias=False,
        )
        self.wo = RowParallelLinear(config.d_ff, config.d_model, bias=False)
        self.act = _get_act_fn(config.dense_act_fn)
        self._wi_packed = None
        self._wo_packed = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        shape = hidden_states.shape
        x = hidden_states.view(-1, shape[-1])
        m, k = x.shape
        n = self.wo.weight.shape[1]
        if self._wi_packed is None:
            self._wi_packed = (
                self.wi.weight.view(2, n, k).permute(2, 1, 0).contiguous().view(k, 2 * n)
            )
            self._wo_packed = self.wo.weight.t().contiguous()

        gated = torch.empty((m, n), device=x.device, dtype=x.dtype)
        grid = lambda meta: (
            triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),
        )
        _gated_gemm_kernel[grid](
            x, self._wi_packed, gated,
            M=m, N=n, K=k,
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=64,
            num_warps=8, num_stages=3,
        )

        out_features = self.wo.weight.shape[0]
        out = torch.empty((m, out_features), device=x.device, dtype=x.dtype)
        grid = lambda meta: (
            triton.cdiv(m, meta["BLOCK_M"])
            * triton.cdiv(out_features, meta["BLOCK_N"]),
        )
        _gemm_kernel[grid](
            gated, self._wo_packed, out,
            M=m, N=out_features, K=n,
            BLOCK_M=64, BLOCK_N=128, BLOCK_K=64,
            num_warps=8, num_stages=4,
        )
        return out.view(*shape[:-1], out_features)


class T5DenseActDense(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.wi = ColumnParallelLinear(config.d_model, config.d_ff, bias=False)
        self.wo = RowParallelLinear(config.d_ff, config.d_model, bias=False)
        self.act = _get_act_fn(config.dense_act_fn)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.wi(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states
