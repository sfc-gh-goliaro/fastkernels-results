"""GPT-OSS decoder layer: attention + MoE with RMSNorm residual connections.

Uses the shared ``LlamaAttention`` with ``use_sinks=True`` and
``sliding_window`` to implement GPT-OSS attention sinks and per-layer
sliding window. Rotary embedding is passed through forward (created
once at the model level and shared across layers).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

from ..L1.rms_norm import RMSNorm
from ..L2.attention import LlamaAttention
from ..L2.gpt_oss_moe import GptOssMoE
from ....infra.context import get_context


@triton.jit
def _single_token_sink_attention(
    query,
    key,
    value,
    sinks,
    output,
    NUM_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    SCALE: tl.constexpr,
):
    head = tl.program_id(0)
    dims = tl.arange(0, HEAD_SIZE)
    kv_head = head // (NUM_HEADS // NUM_KV_HEADS)
    q = tl.load(query + head * HEAD_SIZE + dims).to(tl.float32)
    k = tl.load(key + kv_head * HEAD_SIZE + dims).to(tl.float32)
    v = tl.load(value + kv_head * HEAD_SIZE + dims)
    score = tl.sum(q * k, axis=0) * SCALE
    sink = tl.load(sinks + head).to(tl.float32)
    row_max = tl.maximum(score, sink)
    probability = tl.exp(score - row_max)
    probability /= probability + tl.exp(sink - row_max)
    tl.store(output + head * HEAD_SIZE + dims, probability * v)


class GptOssDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.self_attn = LlamaAttention(
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
            bias=True,
            o_proj_bias=True,
            use_sinks=True,
            sliding_window=config.sliding_window,
            layer_idx=layer_idx,
        )
        self.mlp = GptOssMoE(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self._graph_cache = {}

    def _forward_eager(self, positions, hidden_states, residual, rotary_emb):
        if residual is None:
            hidden_states, residual = (
                self.input_layernorm.forward(hidden_states),
                hidden_states,
            )
        else:
            hidden_states, residual = self.input_layernorm.forward(
                hidden_states, residual
            )

        attn = self.self_attn
        qkv = attn.qkv_proj.forward(hidden_states)
        q_size = attn.num_heads * attn.head_dim
        kv_size = attn.num_kv_heads * attn.head_dim
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
        if rotary_emb is not None:
            q, k = rotary_emb(positions, q, k)
        attention = attn.attn
        use_single_token = False
        if (
            q.shape[0] == 1
            and attention.sinks is not None
            and not attention.k_cache.numel()
            and not attention.v_cache.numel()
        ):
            ctx = get_context()
            use_single_token = (
                ctx.is_prefill
                and not ctx.is_mixed
                and not getattr(ctx, "is_tree_verify", False)
                and attention.attention_chunk_size is None
                and ctx.cu_seqlens_q.numel() == 2
                and ctx.cu_seqlens_k.numel() == 2
                and ctx.max_seqlen_q == 1
                and ctx.max_seqlen_k == 1
                and (
                    attention.sliding_window is None
                    or attention.sliding_window >= 1
                )
            )
        if use_single_token:
            hidden_states = torch.empty_like(q)
            _single_token_sink_attention[(attention.num_heads,)](
                q,
                k,
                v,
                attention.sinks,
                hidden_states,
                NUM_HEADS=attention.num_heads,
                NUM_KV_HEADS=attention.num_kv_heads,
                HEAD_SIZE=attention.head_size,
                SCALE=attention.scale,
                num_warps=1,
            )
        else:
            hidden_states = attention.forward(q, k, v)
        hidden_states = attn.o_proj.forward(hidden_states)

        hidden_states, residual = self.post_attention_layernorm.forward(
            hidden_states, residual
        )
        hidden_states = self.mlp.forward_impl(hidden_states)
        return hidden_states, residual

    def _capture_graph(self, positions, hidden_states, residual):
        saved_positions = positions.clone()
        saved_hidden = hidden_states.clone()
        saved_residual = residual.clone() if residual is not None else None
        static_positions = torch.empty_like(positions)
        static_hidden = torch.empty_like(hidden_states)
        static_residual = (
            torch.empty_like(residual) if residual is not None else None
        )

        stream = torch.cuda.Stream(device=hidden_states.device)
        stream.wait_stream(torch.cuda.current_stream(hidden_states.device))
        with torch.cuda.stream(stream):
            for _ in range(3):
                static_positions.copy_(saved_positions)
                static_hidden.copy_(saved_hidden)
                if static_residual is not None:
                    static_residual.copy_(saved_residual)
                self._forward_eager(
                    static_positions, static_hidden, static_residual, None
                )
        torch.cuda.current_stream(hidden_states.device).wait_stream(stream)
        torch.cuda.synchronize(hidden_states.device)

        static_positions.copy_(saved_positions)
        static_hidden.copy_(saved_hidden)
        if static_residual is not None:
            static_residual.copy_(saved_residual)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs = self._forward_eager(
                static_positions, static_hidden, static_residual, None
            )
        return (
            graph,
            static_positions,
            static_hidden,
            static_residual,
            outputs,
        )

    def forward(self, positions, hidden_states, residual, rotary_emb):
        attention = self.self_attn.attn
        can_graph = (
            rotary_emb is None
            and hidden_states.is_cuda
            and hidden_states.shape[0] == 1
            and not attention.k_cache.numel()
            and not attention.v_cache.numel()
        )
        if can_graph:
            ctx = get_context()
            can_graph = (
                ctx.is_prefill
                and not ctx.is_mixed
                and not getattr(ctx, "is_tree_verify", False)
                and ctx.cu_seqlens_q.numel() == 2
                and ctx.cu_seqlens_k.numel() == 2
                and ctx.max_seqlen_q == 1
                and ctx.max_seqlen_k == 1
            )
        if not can_graph:
            return self._forward_eager(
                positions, hidden_states, residual, rotary_emb
            )

        key = (hidden_states.shape[0], residual is not None)
        entry = self._graph_cache.get(key)
        if entry is None:
            entry = self._capture_graph(positions, hidden_states, residual)
            self._graph_cache[key] = entry
        graph, static_positions, static_hidden, static_residual, outputs = entry
        static_positions.copy_(positions)
        static_hidden.copy_(hidden_states)
        if static_residual is not None:
            static_residual.copy_(residual)
        graph.replay()
        return outputs
