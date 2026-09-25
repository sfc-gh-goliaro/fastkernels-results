"""FLUX transformer blocks (L3 composites).

FluxTransformerBlock: Dual-stream block with AdaLayerNormZero conditioning.
  Separate attention/FFN for image and text (encoder) streams.

FluxSingleTransformerBlock: Single-stream block with AdaLayerNormZeroSingle.
  Concatenates text and image, applies self-attention and MLP in parallel.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from ..L1.gelu import GELU
from ..L1.layer_norm import LayerNorm
from ..L2.ada_layer_norm import AdaLayerNormZero, AdaLayerNormZeroSingle
from ..L2.flux_attention import FluxAttention
from ..L2.flux_feedforward import FeedForward
from ..L2.parallel_linear import ReplicatedLinear


@triton.jit
def _gated_residual_kernel(
    residual_ptr,
    value_ptr,
    gate_ptr,
    out_ptr,
    n_elements,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    cols = offsets % n_cols
    residual = tl.load(residual_ptr + offsets, mask=mask).to(tl.float32)
    value = tl.load(value_ptr + offsets, mask=mask).to(tl.float32)
    gate = tl.load(gate_ptr + cols, mask=mask).to(tl.float32)
    value = (value * gate).to(tl.bfloat16).to(tl.float32)
    tl.store(out_ptr + offsets, residual + value, mask=mask)


def _modulate(
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    return x * (1 + scale[:, None]) + shift[:, None]


def _gated_residual(
    residual: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    if not (
        residual.is_cuda
        and residual.dtype == torch.bfloat16
        and residual.is_contiguous()
        and value.is_contiguous()
        and residual.shape[0] == 1
    ):
        return residual + gate.unsqueeze(1) * value
    out = torch.empty_like(residual)
    _gated_residual_kernel[(triton.cdiv(residual.numel(), 1024),)](
        residual, value, gate, out, residual.numel(),
        n_cols=residual.shape[-1], BLOCK_SIZE=1024, num_warps=4,
    )
    return out


@triton.jit
def _gelu_tanh_inplace_kernel(
    x_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    inner = 0.7978845608028654 * (x + 0.044715 * x * x * x)
    tanh_inner = tl.inline_asm_elementwise(
        "tanh.approx.f32 $0, $1;",
        "=f,f",
        [inner],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    tl.store(x_ptr + offsets, 0.5 * x * (1.0 + tanh_inner), mask=mask)


def _gelu_tanh_inplace(x: torch.Tensor) -> torch.Tensor:
    if not (x.is_cuda and x.dtype == torch.bfloat16 and x.is_contiguous()):
        return F.gelu(x, approximate="tanh")
    _gelu_tanh_inplace_kernel[(triton.cdiv(x.numel(), 4096),)](
        x, x.numel(), BLOCK_SIZE=4096, num_warps=8,
    )
    return x


def _exact_single_ada_norm(
    module: AdaLayerNormZeroSingle,
    x: torch.Tensor,
    emb: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    params = module.linear(F.silu(emb)).chunk(3, dim=1)
    x = F.layer_norm(x, (x.shape[-1],), eps=module.norm.eps)
    x = _modulate(x, params[0], params[1])
    return x, params[2]


def _exact_norm_from_params(
    module: AdaLayerNormZero,
    x: torch.Tensor,
    emb: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    params = emb.chunk(6, dim=1)
    x = F.layer_norm(x, (x.shape[-1],), eps=module.norm.eps)
    x = _modulate(x, params[0], params[1])
    return x, params[2], params[3], params[4], params[5]


class FluxTransformerBlock(nn.Module):
    """Dual-stream DiT block: joint attention over text+image, then separate FFNs."""

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        qk_norm: str = "rms_norm",
        eps: float = 1e-6,
        quant_config: dict | None = None,
    ):
        super().__init__()
        self.norm1 = AdaLayerNormZero(dim, promote_fp32=False)
        self.norm1_context = AdaLayerNormZero(dim, promote_fp32=False)

        self.attn = FluxAttention(
            query_dim=dim,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=False,
            bias=True,
            eps=eps,
            quant_config=quant_config,
        )

        self.norm2 = LayerNorm(dim, elementwise_affine=False, eps=1e-6, promote_fp32=False)
        self.ff = FeedForward(dim=dim, dim_out=dim, quant_config=quant_config)

        self.norm2_context = LayerNorm(dim, elementwise_affine=False, eps=1e-6, promote_fp32=False)
        self.ff_context = FeedForward(dim=dim, dim_out=dim, quant_config=quant_config)
        self._context_stream = None
        self._parallel_min_tokens = 4096
        self._ada_weight = None
        self._ada_bias = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        temb_silu = F.silu(temb)
        if self._ada_weight is None:
            self._ada_weight = torch.cat(
                (self.norm1.linear.weight, self.norm1_context.linear.weight), dim=0
            )
            self._ada_bias = torch.cat(
                (self.norm1.linear.bias, self.norm1_context.linear.bias), dim=0
            )
        ada_params = F.linear(temb_silu, self._ada_weight, self._ada_bias)
        image_params, context_params = ada_params.chunk(2, dim=1)

        current_stream = None
        context_stream = None
        parallel_branches = (
            hidden_states.is_cuda
            and hidden_states.shape[1] >= self._parallel_min_tokens
        )
        if parallel_branches:
            current_stream = torch.cuda.current_stream(hidden_states.device)
            if self._context_stream is None:
                self._context_stream = torch.cuda.Stream(device=hidden_states.device)
            context_stream = self._context_stream
            context_stream.wait_stream(current_stream)
            with torch.cuda.stream(context_stream):
                context_norm = _exact_norm_from_params(
                    self.norm1_context, encoder_hidden_states, context_params
                )

        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            _exact_norm_from_params(self.norm1, hidden_states, image_params)
        )
        if not parallel_branches:
            context_norm = _exact_norm_from_params(
                self.norm1_context, encoder_hidden_states, context_params
            )
        else:
            current_stream.wait_stream(context_stream)
        (
            norm_encoder_hidden_states,
            c_gate_msa,
            c_shift_mlp,
            c_scale_mlp,
            c_gate_mlp,
        ) = context_norm
        joint_attention_kwargs = joint_attention_kwargs or {}

        attention_outputs = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        if len(attention_outputs) == 2:
            attn_output, context_attn_output = attention_outputs
        elif len(attention_outputs) == 3:
            attn_output, context_attn_output, ip_attn_output = attention_outputs

        if parallel_branches:
            context_stream.wait_stream(current_stream)
            with torch.cuda.stream(context_stream):
                encoder_hidden_states = (
                    encoder_hidden_states + c_gate_msa.unsqueeze(1) * context_attn_output
                )
                norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
                norm_encoder_hidden_states = _modulate(
                    norm_encoder_hidden_states, c_shift_mlp, c_scale_mlp
                )
                context_ff_output = self.ff_context(norm_encoder_hidden_states)
                encoder_hidden_states = (
                    encoder_hidden_states
                    + c_gate_mlp.unsqueeze(1) * context_ff_output
                )

        hidden_states = hidden_states + gate_msa.unsqueeze(1) * attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = _modulate(norm_hidden_states, shift_mlp, scale_mlp)

        ff_output = self.ff(norm_hidden_states)
        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * ff_output

        if len(attention_outputs) == 3:
            hidden_states = hidden_states + ip_attn_output

        if parallel_branches:
            current_stream.wait_stream(context_stream)
            encoder_hidden_states.record_stream(current_stream)
        else:
            encoder_hidden_states = (
                encoder_hidden_states + c_gate_msa.unsqueeze(1) * context_attn_output
            )
            norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
            norm_encoder_hidden_states = _modulate(
                norm_encoder_hidden_states, c_shift_mlp, c_scale_mlp
            )
            context_ff_output = self.ff_context(norm_encoder_hidden_states)
            encoder_hidden_states = (
                encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
            )

        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states


class FluxSingleTransformerBlock(nn.Module):
    """Single-stream DiT block: text+image concatenated, self-attention + MLP in parallel."""

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float = 4.0,
        quant_config: dict | None = None,
    ):
        super().__init__()
        self.mlp_hidden_dim = int(dim * mlp_ratio)

        self.norm = AdaLayerNormZeroSingle(dim, promote_fp32=False)
        self.proj_mlp = ReplicatedLinear(dim, self.mlp_hidden_dim, bias=True,
                                         quant_config=quant_config)
        self.act_mlp = GELU(approximate="tanh")
        self.proj_out = ReplicatedLinear(dim + self.mlp_hidden_dim, dim, bias=True,
                                         quant_config=quant_config)

        self.attn = FluxAttention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=True,
            eps=1e-6,
            pre_only=True,
            quant_config=quant_config,
        )
        self._mlp_stream = None
        self._parallel_min_tokens = 4096
        self._proj_attn_weight = None
        self._proj_mlp_weight = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        text_seq_len = encoder_hidden_states.shape[1]
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        residual = hidden_states
        norm_hidden_states, gate = _exact_single_ada_norm(
            self.norm, hidden_states, temb
        )

        current_stream = None
        mlp_stream = None
        if (
            norm_hidden_states.is_cuda
            and norm_hidden_states.shape[1] >= self._parallel_min_tokens
        ):
            current_stream = torch.cuda.current_stream(norm_hidden_states.device)
            if self._mlp_stream is None:
                self._mlp_stream = torch.cuda.Stream(device=norm_hidden_states.device)
            mlp_stream = self._mlp_stream
            mlp_stream.wait_stream(current_stream)
            with torch.cuda.stream(mlp_stream):
                mlp_hidden_states = _gelu_tanh_inplace(
                    self.proj_mlp(norm_hidden_states)
                )
        else:
            mlp_hidden_states = _gelu_tanh_inplace(
                self.proj_mlp(norm_hidden_states)
            )

        joint_attention_kwargs = joint_attention_kwargs or {}
        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )
        if mlp_stream is not None:
            current_stream.wait_stream(mlp_stream)
            mlp_hidden_states.record_stream(current_stream)

        if mlp_stream is not None:
            if self._proj_attn_weight is None:
                self._proj_attn_weight = (
                    self.proj_out.weight[:, :attn_output.shape[-1]].contiguous()
                )
                self._proj_mlp_weight = (
                    self.proj_out.weight[:, attn_output.shape[-1]:].contiguous()
                )
            mlp_stream.wait_stream(current_stream)
            with torch.cuda.stream(mlp_stream):
                mlp_output = F.linear(
                    mlp_hidden_states, self._proj_mlp_weight, None
                )
            attn_output = F.linear(
                attn_output, self._proj_attn_weight, self.proj_out.bias
            )
            current_stream.wait_stream(mlp_stream)
            mlp_output.record_stream(current_stream)
            hidden_states = _gated_residual(
                residual, attn_output + mlp_output, gate
            )
        else:
            hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
            hidden_states = _gated_residual(
                residual, self.proj_out(hidden_states), gate
            )

        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        encoder_hidden_states, hidden_states = (
            hidden_states[:, :text_seq_len],
            hidden_states[:, text_seq_len:],
        )
        return encoder_hidden_states, hidden_states
