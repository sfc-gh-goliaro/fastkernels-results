"""T5 feed-forward dense layers with TP sharding (L2).

T5DenseActDense: standard FFN (ColumnParallel -> act -> RowParallel).
T5DenseGatedActDense: gated FFN (MergedColumnParallel -> gate*up -> RowParallel).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
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
def _gated_gelu_new_kernel(
    gate_up,
    width: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = cols < width
    input_offsets = row * 2 * width + cols
    gate = tl.load(gate_up + input_offsets, mask=mask).to(tl.float32)
    up = tl.load(gate_up + input_offsets + width, mask=mask).to(tl.float32)
    inner = 0.7978845608028654 * (gate + 0.044715 * gate * gate * gate)
    tanh_inner = tl.inline_asm_elementwise(
        "tanh.approx.f32 $0, $1;",
        "=f,f",
        [inner],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    activated = (0.5 * gate * (1.0 + tanh_inner)).to(tl.bfloat16).to(tl.float32)
    tl.store(gate_up + input_offsets, activated * up, mask=mask)


def _gated_gelu_new(gate_up: torch.Tensor) -> torch.Tensor:
    width = gate_up.shape[-1] // 2
    rows = gate_up.numel() // (2 * width)
    _gated_gelu_new_kernel[(rows, triton.cdiv(width, 1024))](
        gate_up,
        width,
        BLOCK_SIZE=1024,
        num_warps=4,
    )
    return gate_up[..., :width]


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
        self._fused_gelu_new = config.dense_act_fn == "gelu_new"

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up = F.linear(hidden_states, self.wi.weight)
        if self._fused_gelu_new:
            hidden_states = _gated_gelu_new(gate_up)
        else:
            gate, up = gate_up.chunk(2, dim=-1)
            hidden_states = self.act(gate) * up
        return F.linear(hidden_states, self.wo.weight)


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
