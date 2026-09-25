"""T5 encoder block: self-attention + FFN with pre-norm residuals (L3).

T5LayerSelfAttention: T5LayerNorm -> T5SelfAttention -> residual add.
T5LayerFF: T5LayerNorm -> T5Dense{Gated}ActDense -> residual add.
T5Block: T5LayerSelfAttention + T5LayerFF.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from transformers import T5Config

from ..L1.t5_layer_norm import T5LayerNorm
from ..L2.t5_attention import T5SelfAttention, _t5_attention_kernel
from ..L2.t5_dense import T5DenseActDense, T5DenseGatedActDense


__targets__ = ["T5Block"]


@triton.jit
def _gated_gelu_new_exact_kernel(
    gate_up,
    width: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = cols < width
    offsets = row * 2 * width + cols
    gate = tl.load(gate_up + offsets, mask=mask).to(tl.float32)
    up = tl.load(gate_up + offsets + width, mask=mask).to(tl.float32)

    # Match the bf16 rounding boundaries in NewGELUActivation's eager ops.
    cube = (gate * gate * gate).to(tl.bfloat16).to(tl.float32)
    cubic_term = (0.044715 * cube).to(tl.bfloat16).to(tl.float32)
    inner = (gate + cubic_term).to(tl.bfloat16).to(tl.float32)
    inner = (0.7978845608028654 * inner).to(tl.bfloat16).to(tl.float32)
    tanh_inner = tl.inline_asm_elementwise(
        "tanh.approx.f32 $0, $1;",
        "=f,f",
        [inner],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    tanh_inner = tanh_inner.to(tl.bfloat16).to(tl.float32)
    one_plus_tanh = (1.0 + tanh_inner).to(tl.bfloat16).to(tl.float32)
    half_gate = (0.5 * gate).to(tl.bfloat16).to(tl.float32)
    activated = (half_gate * one_plus_tanh).to(tl.bfloat16).to(tl.float32)
    tl.store(gate_up + offsets, activated * up, mask=mask)


def _gated_gelu_new_exact(gate_up: torch.Tensor) -> torch.Tensor:
    width = gate_up.shape[-1] // 2
    rows = gate_up.numel() // (2 * width)
    _gated_gelu_new_exact_kernel[(rows, triton.cdiv(width, 2048))](
        gate_up,
        width,
        BLOCK_SIZE=2048,
        num_warps=8,
    )
    return gate_up[..., :width]


@triton.jit
def _residual_t5_norm_kernel(
    hidden_ptr,
    update_ptr,
    weight_ptr,
    residual_ptr,
    normed_ptr,
    eps: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    offsets = row * N + cols
    residual = (
        tl.load(hidden_ptr + offsets, mask=mask, other=0.0)
        + tl.load(update_ptr + offsets, mask=mask, other=0.0)
    ).to(tl.bfloat16)
    tl.store(residual_ptr + offsets, residual, mask=mask)
    residual = residual.to(tl.float32)
    variance = tl.sum(residual * residual, axis=0) / N
    scale = tl.rsqrt(variance + eps)
    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0)
    tl.store(
        normed_ptr + offsets,
        residual * scale * weight,
        mask=mask,
    )


def _residual_t5_norm(
    hidden: torch.Tensor,
    update: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    residual = torch.empty_like(hidden)
    normed = torch.empty_like(hidden)
    rows = hidden.numel() // hidden.shape[-1]
    _residual_t5_norm_kernel[(rows,)](
        hidden,
        update,
        weight,
        residual,
        normed,
        eps=eps,
        N=4096,
        BLOCK_SIZE=4096,
        num_warps=8,
    )
    return residual, normed


def _dense_forward(dense: nn.Module, normed: torch.Tensor) -> torch.Tensor:
    if (
        isinstance(dense, T5DenseGatedActDense)
        and dense._fused_gelu_new
        and normed.dtype == torch.bfloat16
    ):
        gate_up = F.linear(normed, dense.wi.weight)
        return F.linear(_gated_gelu_new_exact(gate_up), dense.wo.weight)
    return dense(normed)


@triton.jit
def _add_transposed_bias_kernel(
    scores_ptr,
    bias_ptr,
    stride_bh: tl.constexpr,
    stride_bm: tl.constexpr,
    stride_bn: tl.constexpr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    query = tl.program_id(0)
    key_block = tl.program_id(1)
    keys = key_block * BLOCK + tl.arange(0, BLOCK)
    heads = tl.arange(0, 64)
    bias = tl.load(
        bias_ptr
        + query * stride_bm
        + keys[:, None] * stride_bn
        + heads[None, :] * stride_bh,
    )
    bias = tl.trans(bias)
    score_offsets = heads[:, None] * N * N + query * N + keys[None, :]
    scores = tl.load(scores_ptr + score_offsets)
    tl.store(scores_ptr + score_offsets, scores + bias)


@triton.jit
def _relative_bias_tiled_kernel(
    weight_ptr,
    output_ptr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    query = tl.program_id(0)
    key_block = tl.program_id(1)
    keys = key_block * BLOCK + tl.arange(0, BLOCK)
    heads = tl.arange(0, 64)
    relative = keys - query
    distance = tl.abs(relative)
    large_bucket = 8 + (
        tl.log(tl.maximum(distance, 8).to(tl.float32) / 8.0)
        * 2.8853900817779268
    ).to(tl.int32)
    large_bucket = tl.minimum(large_bucket, 15)
    bucket = tl.where(distance < 8, distance, large_bucket)
    bucket += tl.where(relative > 0, 16, 0)
    bias = tl.load(weight_ptr + bucket[:, None] * 64 + heads[None, :])
    bias = tl.trans(bias)
    output_offsets = heads[:, None] * N * N + query * N + keys[None, :]
    tl.store(output_ptr + output_offsets, bias)


def _attention_specialized(
    attention: T5SelfAttention,
    hidden_states: torch.Tensor,
    position_bias: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    qkv = attention.qkv_proj(hidden_states)
    query, key, value = qkv.split([4096, 4096, 4096], dim=-1)
    query = query.view(1, 512, 64, 64).transpose(1, 2)
    key = key.view(1, 512, 64, 64).transpose(1, 2)
    value = value.view(1, 512, 64, 64).transpose(1, 2)
    scores = attention.bmm(query, key.transpose(3, 2))
    computed_position_bias = position_bias is None
    if computed_position_bias:
        position_bias = torch.empty(
            (1, 64, 512, 512),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        _relative_bias_tiled_kernel[(512, 8)](
            attention.relative_attention_bias.emb.weight,
            position_bias,
            N=512,
            BLOCK=64,
            num_warps=8,
        )
    else:
        _add_transposed_bias_kernel[(512, 8)](
            scores,
            position_bias,
            stride_bh=position_bias.stride(1),
            stride_bm=position_bias.stride(2),
            stride_bn=position_bias.stride(3),
            N=512,
            BLOCK=64,
            num_warps=8,
        )
    attn_output = torch.empty_like(hidden_states)
    _t5_attention_kernel[(8, 64)](
        scores,
        value,
        position_bias,
        attn_output,
        stride_vm=value.stride(2),
        stride_vh=value.stride(1),
        stride_bh=position_bias.stride(1),
        stride_bm=position_bias.stride(2),
        stride_bn=position_bias.stride(3),
        N_CTX=512,
        HEAD_DIM=64,
        BLOCK_M=64,
        BLOCK_N=64,
        HAS_BIAS=computed_position_bias,
        num_warps=8,
        num_stages=3,
    )
    return attention.o(attn_output), position_bias


class T5LayerSelfAttention(nn.Module):
    def __init__(self, config: T5Config, has_relative_attention_bias: bool = False):
        super().__init__()
        self.SelfAttention = T5SelfAttention(config, has_relative_attention_bias)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor | None = None,
        position_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normed = self.layer_norm(hidden_states)
        attn_output, position_bias = self.SelfAttention(
            normed, mask=mask, position_bias=position_bias,
        )
        hidden_states = hidden_states + attn_output
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        return hidden_states, position_bias


class T5LayerFF(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        if config.is_gated_act:
            self.DenseReluDense = T5DenseGatedActDense(config)
        else:
            self.DenseReluDense = T5DenseActDense(config)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normed = self.layer_norm(hidden_states)
        dense = self.DenseReluDense
        ff_output = _dense_forward(dense, normed)
        hidden_states = hidden_states + ff_output
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        return hidden_states


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
        if (
            hidden_states.shape == (1, 512, 4096)
            and hidden_states.dtype == torch.bfloat16
            and mask is None
            and self.layer[0].SelfAttention.n_heads_per_partition == 64
            and self.layer[0].SelfAttention.d_kv == 64
            and (
                position_bias is not None
                or self.layer[0].SelfAttention.has_relative_attention_bias
            )
            and (
                position_bias is None
                or position_bias.shape == (1, 64, 512, 512)
            )
        ):
            self_attention = self.layer[0]
            feed_forward = self.layer[1]
            normed = self_attention.layer_norm(hidden_states)
            attn_output, position_bias = _attention_specialized(
                self_attention.SelfAttention, normed, position_bias,
            )
            hidden_states, normed = _residual_t5_norm(
                hidden_states,
                attn_output,
                feed_forward.layer_norm.weight,
                feed_forward.layer_norm.variance_epsilon,
            )
            ff_output = _dense_forward(feed_forward.DenseReluDense, normed)
            return hidden_states + ff_output, position_bias

        hidden_states, position_bias = self.layer[0](
            hidden_states, mask=mask, position_bias=position_bias,
        )
        hidden_states = self.layer[1](hidden_states)
        return hidden_states, position_bias
