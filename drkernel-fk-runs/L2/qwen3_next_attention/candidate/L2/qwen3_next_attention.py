from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

from fastkernels.infra.context import get_attn_backend_config, get_context
from fastkernels.infra.tp import _tp_size
from fastkernels.infra.triton import (
    _gate_mul_inplace_kernel,
    _fused_qk_rmsnorm_rope_gate_kernel,
    fused_qk_rmsnorm_rope_gate,
)

# Helper: always launch a real Triton kernel to apply sigmoid gate in-place.
def _gate_mul_inplace(out: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    n_elements = out.numel()
    if n_elements == 0:
        return out
    block = 1024
    _gate_mul_inplace_kernel[(triton.cdiv(n_elements, block),)](
        out,
        gate,
        n_elements,
        BLOCK=block,
    )
    return out

# Fused: QK RMSNorm + partial RoPE + gate copy.
def _vllm_fused_qk_rmsnorm_rope_gate(
    q_gate: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    eps: float,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_out, k_out, gate_out = fused_qk_rmsnorm_rope_gate(
        q_gate,
        k,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        eps,
        num_q_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
    )
    q_out = q_out.view(-1, num_q_heads, head_dim)
    k_out = k_out.view(-1, num_kv_heads, head_dim)
    gate_out = gate_out.view(-1, num_q_heads, head_dim)
    return q_out, k_out, gate_out

class Model(nn.Module):
    """Triton-optimized Qwen3-Next attention:
    - Fuses QK RMSNorm + partial RoPE + gate copy into a single Triton kernel when positions is provided.
    - Always launches a Triton kernel for the final sigmoid gating.
    - Preserves the original attribute structure and forward contract.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        layer_idx: int,
        rms_norm_eps: float = 1e-6,
        reduce_output: bool = True,
    ):
        super().__init__()
        tp = _tp_size()
        self.layer_idx = layer_idx
        self.num_heads = num_attention_heads // tp
        # Mirror vLLM GQA behavior: if not divisible, num_kv_heads unchanged across ranks
        self.num_kv_heads = num_key_value_heads // tp if num_key_value_heads % tp == 0 else num_key_value_heads
        self.head_dim = head_dim
        self.scaling = head_dim ** -0.5

        # Per-head norms (GemmaRMSNorm), top-level
        self.q_norm = GemmaRMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = GemmaRMSNorm(head_dim, eps=rms_norm_eps)

        # QKV projection: Q outputs 2x heads (Q + gate)
        self.qkv_proj = QKVParallelLinear(
            hidden_size, head_dim,
            num_attention_heads * 2,  # doubled for output gate
            num_key_value_heads,
        )

        # Row-parallel output projection
        self.o_proj = RowParallelLinear(
            num_attention_heads * head_dim, hidden_size,
            reduce_results=reduce_output,
        )

        # Choose attention backend and cache helpers like original
        attn_cfg = get_attn_backend_config()
        self.kv_layout = attn_cfg.kv_layout
        self._use_trtllm = attn_cfg.use_trtllm

        if self._use_trtllm:
            self.store_kvcache = StoreKVCacheHND(page_size=attn_cfg.block_size)
            self.flash_attn_prefill = TRTLLMPrefill(
                self.num_heads, self.num_kv_heads, self.head_dim,
            )
            self.flash_attn_decode = TRTLLMDecode(
                self.num_heads, self.num_kv_heads, self.head_dim,
            )
        else:
            self.store_kvcache = StoreKVCache()
            self.flash_attn_prefill = FlashAttnPrefill(
                self.num_heads, self.num_kv_heads, self.head_dim,
            )
            self.flash_attn_decode = FlashAttnDecode(
                self.num_heads, self.num_kv_heads, self.head_dim,
            )

        # We will launch a real Triton kernel for gate-mul; fused pre-path when possible
        self._fused_qk_rope_gate = True

    def forward(self, hidden_states, rotary_emb=None, positions=None, state_manager=None):
        # Ensure we launch at least one Triton kernel (_gate_mul_inplace).
        return self._forward_impl(hidden_states, rotary_emb, positions, state_manager)

    def _forward_impl(self, hidden_states, rotary_emb=None, positions=None, state_manager=None):
        md = get_context().kda_metadata
        if state_manager is None:
            state_manager = get_context().kda_state
        self.rotary_emb = rotary_emb

        if md is None or state_manager is None:
            raise RuntimeError(
                "Qwen3NextAttention requires engine-managed KV state and metadata",
            )

        # Flatten tokens
        x = hidden_states.reshape(-1, hidden_states.shape[-1])
        N = x.shape[0]

        # Projections
        qkv = self.qkv_proj(x)
        q_gate_size = self.num_heads * 2 * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        q_gate, k, v = qkv.split([q_gate_size, kv_size, kv_size], dim=-1)

        # Use fused pre-path when positions is provided
        use_fused = self._fused_qk_rope_gate and (positions is not None)

        if use_fused:
            # q_norm.weight + 1 and k_norm.weight + 1 as fp32
            q_gain = self.q_norm.weight.float() + 1.0
            k_gain = self.k_norm.weight.float() + 1.0
            # Launch fused kernel: QK RMSNorm + partial RoPE + gate copy
            cos_sin_cache = getattr(rotary_emb, "cos_sin_cache", None) if rotary_emb is not None else None
            q, k, gate = _vllm_fused_qk_rmsnorm_rope_gate(
                q_gate,
                k,
                q_gain,
                k_gain,
                cos_sin_cache,
                positions.reshape(-1),
                self.q_norm.variance_epsilon,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                (rotary_emb.head_dim if rotary_emb is not None else 0),
            )
            q = q.view(N, self.num_heads, self.head_dim)
            k = k.view(N, self.num_kv_heads, self.head_dim)
            gate = gate.view(N, self.num_heads, self.head_dim)
        else:
            # Fallback: pure PyTorch ops for pre-attention
            q_gate = q_gate.view(N, self.num_heads, 2 * self.head_dim)
            q = q_gate[:, :, :self.head_dim].contiguous()
            gate = q_gate[:, :, self.head_dim:].contiguous()
            k = k.view(N, self.num_kv_heads, self.head_dim)

            # RMSNorm (Gemma)
            q = self.q_norm(q.reshape(-1, self.head_dim)).view(N, self.num_heads, self.head_dim)
            k = self.k_norm(k.reshape(-1, self.head_dim)).view(N, self.num_kv_heads, self.head_dim)

            # Partial RoPE if positions provided and rotary_emb available
            if (positions is not None) and (rotary_emb is not None):
                pos_flat = positions.reshape(-1) if positions.dim() > 1 else positions
                rotary_dim = rotary_emb.head_dim
                q_rot, q_pass = q[..., :rotary_dim].contiguous(), q[..., rotary_dim:]
                k_rot, k_pass = k[..., :rotary_dim].contiguous(), k[..., rotary_dim:]
                q_rot, k_rot = rotary_emb(pos_flat, q_rot, k_rot)
                q = torch.cat([q_rot, q_pass], dim=-1)
                k = torch.cat([k_rot, k_pass], dim=-1)

        v = v.view(N, self.num_kv_heads, self.head_dim)

        # KV cache
        layer_idx = self.layer_idx
        k_cache = state_manager.k_cache[layer_idx]
        v_cache = state_manager.v_cache[layer_idx]
        # Store to cache (HND or NHD)
        self.store_kvcache(k, v, k_cache, v_cache, md.slot_mapping)

        # Allocate output
        out = torch.empty(
            N,
            self.num_heads,
            self.head_dim,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        # Mixed prefill/decode using attention backends
        nd = md.num_decodes
        ndt = md.num_decode_tokens
        np_ = md.num_prefills

        if nd > 0:
            out[:ndt] = self.flash_attn_decode(
                q[:ndt],
                k_cache,
                v_cache,
                cache_seqlens=md.seq_lens[:nd].to(torch.int32),
                block_table=md.block_tables[:nd],
                softmax_scale=self.scaling,
                causal=True,
                max_seq_len=md.max_seq_len,
            )

        if np_ > 0:
            cu_pf = (md.query_start_loc[nd:] - md.query_start_loc[nd]).to(torch.int32)
            seqs_k = md.seq_lens[nd:]
            cu_k_pf = torch.zeros(np_ + 1, dtype=torch.int32, device=q.device)
            cu_k_pf[1:] = torch.cumsum(seqs_k.to(torch.int32), dim=0)
            out[ndt:] = self.flash_attn_prefill(
                q[ndt:],
                k_cache,
                v_cache,
                cu_seqlens_q=cu_pf,
                cu_seqlens_k=cu_k_pf,
                max_seqlen_q=md.max_query_len,
                max_seqlen_k=md.max_seq_len,
                softmax_scale=self.scaling,
                causal=True,
                block_table=md.block_tables[nd:],
            )

        # Always launch a real Triton kernel: apply sigmoid gate in-place
        o = _gate_mul_inplace(out, gate)

        # Output projection
        o = o.reshape(N, self.num_heads * self.head_dim)
        return self.o_proj(o)

Qwen3NextAttention = ModelNew
