"""Decoder layer: attention + MLP with RMSNorm residual connections.

Unified across Llama, Qwen2, and Qwen3 architectures:
  - bias:    Qwen2 uses bias=True on QKV projection.
  - qk_norm: Qwen3 applies per-head RMSNorm to Q and K before RoPE.
"""

from __future__ import annotations

import torch.nn as nn
from fastkernels.infra.cuda_ext import lazy_op
from fastkernels.infra.context import get_context

from ..L1.rms_norm import RMSNorm
from ..L2.attention import LlamaAttention
from ..L2.llama_mlp import LlamaMLP, _C as _MLP_C

_C = lazy_op("llama_decoder_candidate", "llama_decoder.cu")


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config, rotary_emb: nn.Module | None = None,
                 bias: bool = False, qk_norm: bool = False,
                 quant_config: dict | None = None):
        super().__init__()
        self.self_attn = LlamaAttention(
            config.hidden_size, config.num_attention_heads,
            config.num_key_value_heads, config.head_dim,
            rotary_emb=rotary_emb,
            bias=bias, qk_norm=qk_norm,
            rms_norm_eps=config.rms_norm_eps,
            quant_config=quant_config,
        )
        self.mlp = LlamaMLP(config, quant_config=quant_config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self._direct_path = (
            config.hidden_size == 4096
            and quant_config is None
            and self.self_attn.q_norm is None
            and self.self_attn.q_wl_norm is None
            and not self.self_attn.attn_temperature_tuning
            and self.self_attn.o_proj.tp_size == 1
            and self.mlp.down_proj.tp_size == 1
        )

    def forward(self, positions, hidden_states, residual):
        if residual is None:
            residual = hidden_states
            hidden_states = _C.rmsnorm(
                hidden_states,
                self.input_layernorm.weight,
                self.input_layernorm.eps,
            )
        else:
            _C.fused_add_rmsnorm(
                hidden_states,
                residual,
                self.input_layernorm.weight,
                self.input_layernorm.eps,
            )
        if self._direct_path:
            attention = self.self_attn
            qkv = attention.qkv_proj.forward(hidden_states)
            q_size = attention.num_heads * attention.head_dim
            kv_size = attention.num_kv_heads * attention.head_dim
            attn_impl = attention.attn
            single_token_prefill = False
            if (
                hidden_states.shape[0] == 1
                and attn_impl.sinks is None
                and not attn_impl.k_cache.numel()
                and not attn_impl.v_cache.numel()
            ):
                ctx = get_context()
                single_token_prefill = (
                    ctx.is_prefill
                    and not ctx.is_mixed
                    and not getattr(ctx, "is_tree_verify", False)
                )
            if single_token_prefill:
                v = qkv.narrow(-1, q_size + kv_size, kv_size)
                attn_output = _C.single_token_value(
                    v,
                    attention.num_heads,
                    attention.num_kv_heads,
                    attention.head_dim,
                )
            else:
                q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
                if attention.rotary_emb is not None:
                    q, k = attention.rotary_emb.forward(positions, q, k)
                attn_output = attn_impl.forward(q, k, v)
            hidden_states = attention.o_proj.forward(attn_output)
        else:
            hidden_states = self.self_attn(positions, hidden_states)
        _C.fused_add_rmsnorm(
            hidden_states,
            residual,
            self.post_attention_layernorm.weight,
            self.post_attention_layernorm.eps,
        )
        if self._direct_path:
            hidden_states = self.mlp.gate_up_proj.forward(hidden_states)
            hidden_states = _MLP_C.silu_and_mul(hidden_states)
            hidden_states = self.mlp.down_proj.forward(hidden_states)
        else:
            hidden_states = self.mlp(hidden_states)
        return hidden_states, residual
