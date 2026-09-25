"""Optimized T5 encoder block for the captured FLUX T5 workload."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra import libdevice
from transformers import T5Config

from ..L2.t5_attention import T5SelfAttention
from ..L2.t5_dense import T5DenseActDense, T5DenseGatedActDense


__targets__ = ["T5Block"]


@triton.jit
def _rms_norm_kernel(x, weight, out, n_cols: tl.constexpr,
                     eps: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    values = tl.load(x + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(values * values, axis=0) / n_cols
    values *= tl.rsqrt(variance + eps)
    values *= tl.load(weight + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(out + row * n_cols + cols, values, mask=mask)


@triton.jit
def _add_rms_norm_kernel(x, delta, weight, residual, out,
                         n_cols: tl.constexpr, eps: tl.constexpr,
                         BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    values = (
        tl.load(x + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
        + tl.load(delta + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    )
    rounded = values.to(tl.bfloat16)
    rounded_f32 = rounded.to(tl.float32)
    variance = tl.sum(rounded_f32 * rounded_f32, axis=0) / n_cols
    normed = rounded_f32 * tl.rsqrt(variance + eps)
    normed *= tl.load(weight + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(residual + row * n_cols + cols, rounded, mask=mask)
    tl.store(out + row * n_cols + cols, normed, mask=mask)


@triton.jit
def _add_kernel(x, delta, out, n_elements: tl.constexpr,
                BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(x + offsets, mask=mask) + tl.load(delta + offsets, mask=mask)
    tl.store(out + offsets, values, mask=mask)


@triton.jit
def _transpose_heads_kernel(x, out, n_elements: tl.constexpr,
                            seq_len: tl.constexpr, n_heads: tl.constexpr,
                            head_dim: tl.constexpr, BLOCK: tl.constexpr):
    output_offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = output_offset < n_elements
    d = output_offset % head_dim
    major = output_offset // head_dim
    head = major % n_heads
    token_batch = major // n_heads
    token = token_batch % seq_len
    batch = token_batch // seq_len
    input_offset = (
        ((batch * n_heads + head) * seq_len + token) * head_dim + d
    )
    values = tl.load(x + input_offset, mask=mask)
    tl.store(out + output_offset, values, mask=mask)


@triton.jit
def _gated_gelu_kernel(gate_up, out, n_elements: tl.constexpr,
                       width: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    row = offsets // width
    col = offsets - row * width
    base = row * (2 * width) + col
    gate = tl.load(gate_up + base, mask=mask).to(tl.float32)
    up = tl.load(gate_up + base + width, mask=mask).to(tl.float32)

    # Match the BF16 rounding boundaries in NewGELUActivation.
    cube = (gate * gate * gate).to(tl.bfloat16).to(tl.float32)
    cubic_term = (0.044715 * cube).to(tl.bfloat16).to(tl.float32)
    inner_sum = (gate + cubic_term).to(tl.bfloat16).to(tl.float32)
    inner = (0.7978845608028654 * inner_sum).to(tl.bfloat16).to(tl.float32)
    tanh_inner = libdevice.tanh(inner).to(tl.bfloat16).to(tl.float32)
    left = (0.5 * gate).to(tl.bfloat16).to(tl.float32)
    right = (1.0 + tanh_inner).to(tl.bfloat16).to(tl.float32)
    gelu = (left * right).to(tl.bfloat16).to(tl.float32)
    tl.store(out + offsets, gelu * up, mask=mask)


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    out = torch.empty_like(x)
    rows, cols = x.numel() // x.shape[-1], x.shape[-1]
    _rms_norm_kernel[(rows,)](
        x, weight, out, cols, eps, BLOCK=triton.next_power_of_2(cols),
        num_warps=8,
    )
    return out


def _add_rms_norm(
    x: torch.Tensor, delta: torch.Tensor, weight: torch.Tensor, eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    residual = torch.empty_like(x)
    out = torch.empty_like(x)
    rows, cols = x.numel() // x.shape[-1], x.shape[-1]
    _add_rms_norm_kernel[(rows,)](
        x, delta, weight, residual, out, cols, eps,
        BLOCK=triton.next_power_of_2(cols), num_warps=8,
    )
    return residual, out


def _add(x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    block = 256
    _add_kernel[(triton.cdiv(x.numel(), block),)](
        x, delta, out, x.numel(), BLOCK=block, num_warps=4,
    )
    return out


class T5LayerSelfAttention(nn.Module):
    def __init__(self, config: T5Config, has_relative_attention_bias: bool = False):
        super().__init__()
        self.SelfAttention = T5SelfAttention(config, has_relative_attention_bias)
        self.layer_norm = nn.Module()
        self.layer_norm.weight = nn.Parameter(torch.ones(config.d_model))
        self.layer_norm.variance_epsilon = config.layer_norm_epsilon
        self._cached_position_bias = None

    def _position_bias(self, hidden_states: torch.Tensor) -> torch.Tensor:
        seq_len = hidden_states.shape[1]
        cached = self._cached_position_bias
        if (
            cached is not None
            and cached.shape[-1] == seq_len
            and cached.device == hidden_states.device
            and cached.dtype == hidden_states.dtype
        ):
            return cached
        attention = self.SelfAttention
        if attention.has_relative_attention_bias:
            bias = attention.compute_bias(
                seq_len, seq_len, device=hidden_states.device,
            )
        else:
            bias = torch.zeros(
                (1, attention.n_heads_per_partition, seq_len, seq_len),
                dtype=hidden_states.dtype, device=hidden_states.device,
            )
        self._cached_position_bias = bias
        return bias

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor | None = None,
        position_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attention = self.SelfAttention
        normed = _rms_norm(
            hidden_states, self.layer_norm.weight,
            self.layer_norm.variance_epsilon,
        )
        qkv = F.linear(normed, attention.qkv_proj.weight)
        if position_bias is None:
            position_bias = self._position_bias(hidden_states)
            if mask is not None:
                position_bias = position_bias + mask

        batch, seq_len = hidden_states.shape[:2]
        query, key, value = qkv.chunk(3, dim=-1)
        shape = (
            batch, seq_len, attention.n_heads_per_partition, attention.d_kv,
        )
        query = query.view(shape).transpose(1, 2)
        key = key.view(shape).transpose(1, 2)
        value = value.view(shape).transpose(1, 2)
        scores = torch.matmul(query, key.transpose(3, 2))
        scores += position_bias
        probabilities = torch.softmax(scores, dim=-1)
        attn_values = torch.matmul(probabilities, value)

        transposed = torch.empty(
            (batch, seq_len, attention.inner_dim),
            dtype=hidden_states.dtype, device=hidden_states.device,
        )
        block = 512
        _transpose_heads_kernel[
            (triton.cdiv(transposed.numel(), block),)
        ](
            attn_values, transposed, transposed.numel(), seq_len,
            attention.n_heads_per_partition, attention.d_kv,
            BLOCK=block, num_warps=4,
        )
        attn_output = F.linear(transposed, attention.o.weight)
        return attn_output, position_bias


class T5LayerFF(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        if config.is_gated_act:
            self.DenseReluDense = T5DenseGatedActDense(config)
        else:
            self.DenseReluDense = T5DenseActDense(config)
        self.layer_norm = nn.Module()
        self.layer_norm.weight = nn.Parameter(torch.ones(config.d_model))
        self.layer_norm.variance_epsilon = config.layer_norm_epsilon


class T5Block(nn.Module):
    def __init__(self, config: T5Config, has_relative_attention_bias: bool = False):
        super().__init__()
        self.layer = nn.ModuleList([
            T5LayerSelfAttention(config, has_relative_attention_bias),
            T5LayerFF(config),
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor | None = None,
        position_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attn_output, position_bias = self.layer[0](
            hidden_states, mask=mask, position_bias=position_bias,
        )
        ff_layer = self.layer[1]
        hidden_states, normed = _add_rms_norm(
            hidden_states, attn_output, ff_layer.layer_norm.weight,
            ff_layer.layer_norm.variance_epsilon,
        )
        dense = ff_layer.DenseReluDense
        if isinstance(dense, T5DenseGatedActDense):
            gate_up = F.linear(normed, dense.wi.weight)
            activated = torch.empty(
                (*normed.shape[:-1], dense.wo.weight.shape[1]),
                dtype=normed.dtype, device=normed.device,
            )
            block = 512
            _gated_gelu_kernel[
                (triton.cdiv(activated.numel(), block),)
            ](
                gate_up, activated, activated.numel(), activated.shape[-1],
                BLOCK=block, num_warps=4,
            )
            hidden_states = torch.addmm(
                hidden_states.view(-1, hidden_states.shape[-1]),
                activated.view(-1, activated.shape[-1]),
                dense.wo.weight.t(),
            ).view_as(hidden_states)
        else:
            ff_output = dense(normed)
            hidden_states = _add(hidden_states, ff_output)
        return hidden_states, position_bias
