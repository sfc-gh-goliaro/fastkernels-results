"""Qwen3-Next decoder layer: hybrid GDN/full attention + MoE.

Dispatches to GDN linear attention or full attention based on layer type.
All layers use MoE (every layer is sparse in Qwen3-Next).
Uses GemmaRMSNorm (weight + 1 convention).

Under tensor parallelism the two parallel regions per layer (attention output,
MoE output) hand back un-reduced partials and the following norm does
all-reduce + residual-add + RMSNorm in one FlashInfer kernel -- the same fusion
vLLM's ``fuse_allreduce_rms`` pass applies
(``AllReduceFusedAddGemmaRMSNormPattern``). The MoE's partial is consumed by the
*next* layer's ``input_layernorm``, or by ``Qwen3NextModel.norm`` for the last
layer, so the model owns that half of the contract.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ....infra.context import get_context
from ....infra.tp import _tp_size
from ..L2.flashinfer_allreduce_fusion import fused_allreduce_add_gemma_rmsnorm
from ..L1.gemma_rms_norm import GemmaRMSNorm
from ..L2.qwen3_next_gdn_attention import Qwen3NextGDNAttention
from ..L2.qwen3_next_attention import Qwen3NextAttention
from ..L2.shared_expert_moe import SharedExpertMoE


@torch.compile
def _residual_gemma_rms_norm(weight, variance_epsilon: float, x, residual):
    orig_dtype = x.dtype
    x = (
        x.float() + residual.float()
        if orig_dtype == torch.float16
        else x + residual
    )
    residual = x.to(orig_dtype) if x.dtype != orig_dtype else x
    x = x.float()
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + variance_epsilon)
    x = x * (1.0 + weight.float())
    return x.to(orig_dtype), residual


def fused_ar_norm(norm: GemmaRMSNorm, hidden_states, residual, fuse: bool):
    """``norm(all_reduce(hidden_states), residual)``, fused when ``fuse``."""
    if fuse:
        # Opaque under torch.compile: tracing into FlashInfer's fused
        # collective hits Python logging / datetime and aborts Dynamo.
        if torch.compiler.is_compiling():
            return torch.ops.fastkernels.fused_allreduce_add_gemma_rmsnorm(
                hidden_states, residual, norm.weight, float(norm.variance_epsilon),
            )
        return fused_allreduce_add_gemma_rmsnorm(hidden_states, residual, norm)
    if hidden_states.is_cuda and hidden_states.shape[-1] == 2048:
        return _residual_gemma_rms_norm(
            norm.weight, float(norm.variance_epsilon), hidden_states, residual,
        )
    return norm(hidden_states, residual)


class Qwen3NextDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        # Only worth deferring when there is a collective to defer.
        self.fuse_ar_norm = _tp_size() > 1

        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3NextGDNAttention(
                hidden_size=config.hidden_size,
                num_k_heads=config.linear_num_key_heads,
                num_v_heads=config.linear_num_value_heads,
                head_k_dim=config.linear_key_head_dim,
                head_v_dim=config.linear_value_head_dim,
                layer_idx=layer_idx,
                conv_kernel_size=config.linear_conv_kernel_dim,
                rms_norm_eps=config.rms_norm_eps,
                reduce_output=not self.fuse_ar_norm,
            )
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3NextAttention(
                hidden_size=config.hidden_size,
                num_attention_heads=config.num_attention_heads,
                num_key_value_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
                layer_idx=layer_idx,
                rms_norm_eps=config.rms_norm_eps,
                reduce_output=not self.fuse_ar_norm,
            )
        else:
            raise ValueError(f"Invalid layer_type: {self.layer_type}")

        # MoE for all Qwen3-Next layers (every layer is sparse).
        self.mlp = SharedExpertMoE(
            hidden_size=config.hidden_size,
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            moe_intermediate_size=config.moe_intermediate_size,
            routing="softmax",
            correction_bias=False,
            renormalize=config.norm_topk_prob,
            routed_scaling_factor=1.0,
            shared_expert_intermediate_size=config.shared_expert_intermediate_size,
            shared_expert_attr_name="shared_expert",
            shared_expert_gate=True,
            reduce_results=not self.fuse_ar_norm,
        )

        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(self, hidden_states, residual, positions=None,
                rotary_emb=None, state_manager=None):
        if residual is None:
            # Layer 0: the input is the vocab-parallel embedding's output, which
            # is already reduced, and there is no residual stream yet.
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = fused_ar_norm(
                self.input_layernorm, hidden_states, residual, self.fuse_ar_norm,
            )

        # Attention
        if self.layer_type == "linear_attention":
            md = get_context().kda_metadata
            force_regular_prefill = (
                1 < hidden_states.shape[0] <= 128
                and md is not None
                and md.num_prefills == 1
                and md.num_decodes == 0
                and not md.any_have_initial_state
            )
            if force_regular_prefill:
                # The short recurrence's small error can change MoE routes.
                previous_any_have_initial_state = md.any_have_initial_state
                md.any_have_initial_state = True
                try:
                    hidden_states = self.linear_attn(
                        hidden_states, state_manager=state_manager,
                    )
                finally:
                    md.any_have_initial_state = previous_any_have_initial_state
            else:
                hidden_states = self.linear_attn(
                    hidden_states, state_manager=state_manager,
                )
        else:
            hidden_states = self.self_attn(
                hidden_states, rotary_emb=rotary_emb, positions=positions,
                state_manager=state_manager,
            )

        # Post-attention norm + MLP
        hidden_states, residual = fused_ar_norm(
            self.post_attention_layernorm, hidden_states, residual,
            self.fuse_ar_norm,
        )
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual
