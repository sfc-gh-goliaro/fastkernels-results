from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ....infra.cuda_ext import lazy_op
from ....infra.context import get_context
from ....infra.tp import _tp_size
from ..L1.rms_norm import RMSNorm
from .mla_attention_impl import MLAAttention
from .parallel_linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)

_C = lazy_op("kimi_mla_attention_opt", "kimi_mla_attention.cu")


def _pack_k_nope_k_pe(k_nope, k_pe):
    return _C.pack_k(k_nope, k_pe)


class KimiMLAAttention(nn.Module):
    """Kimi MLA path matching vLLM's latent-attention formulation."""

    def __init__(self, config, quant_config: dict | None = None):
        super().__init__()
        tp = _tp_size()
        self.hidden_size = config.hidden_size
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.num_heads = config.num_attention_heads
        self.num_local_heads = self.num_heads // tp
        self.scaling = self.qk_head_dim ** -0.5

        assert self.q_lora_rank is None
        assert getattr(config, "mla_use_nope", True)

        self.kv_a_proj_with_mqa = ReplicatedLinear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
            quant_config=quant_config,
        )
        self.q_proj = ColumnParallelLinear(
            self.hidden_size,
            self.num_heads * self.qk_head_dim,
            bias=False,
            quant_config=quant_config,
        )
        self.kv_a_layernorm = RMSNorm(
            self.kv_lora_rank,
            eps=config.rms_norm_eps,
        )
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
        )
        self.o_proj = RowParallelLinear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
        )

        self.attn = MLAAttention(
            num_heads=self.num_local_heads,
            scale=self.scaling,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
            kv_lora_rank=self.kv_lora_rank,
            is_sparse=False,
        )
        object.__setattr__(self.attn, "_kv_b_proj", self.kv_b_proj)
        object.__setattr__(self.attn, "_concat_k_nope_k_pe", _pack_k_nope_k_pe)
        object.__setattr__(self, "_qkv_a_weight", None)
        object.__setattr__(self, "_v_proj_weight", None)

    def compute_absorbed_weights(self):
        """Compute absorbed MLA decode weights from ``kv_b_proj``."""
        weight = self.kv_b_proj.weight.data
        if hasattr(self.kv_b_proj, "use_fp8") and self.kv_b_proj.use_fp8:
            scale = self.kv_b_proj.weight_scale_inv.data
            weight = self._dequant_fp8_block(weight, scale)
        else:
            weight = weight.to(torch.bfloat16)

        weight = weight.T
        latent = self.kv_lora_rank
        heads = self.num_local_heads
        nope = self.qk_nope_head_dim
        value = self.v_head_dim
        weight = weight.view(latent, heads, nope + value)
        w_uk = weight[:, :, :nope]
        w_uv = weight[:, :, nope:]
        self.attn.W_UV = w_uv.permute(1, 0, 2).contiguous()
        self.attn.W_UK_T = w_uk.permute(1, 2, 0).contiguous()

    @staticmethod
    def _dequant_fp8_block(
        w_fp8: torch.Tensor,
        scale_inv: torch.Tensor,
        block_size: int = 128,
    ) -> torch.Tensor:
        import math

        n, k = w_fp8.shape
        sn = math.ceil(n / block_size)
        sk = math.ceil(k / block_size)
        scale = scale_inv[:sn, :sk]
        scale_expanded = scale.repeat_interleave(block_size, dim=0)[:n]
        scale_expanded = scale_expanded.repeat_interleave(block_size, dim=1)[:, :k]
        return (w_fp8.float() * scale_expanded).to(torch.bfloat16)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        positions: torch.Tensor | None = None,
        state_manager=None,
    ) -> torch.Tensor:
        del positions, state_manager
        num_tokens = hidden_states.shape[0]

        ctx = get_context()
        if (
            num_tokens == 1
            and ctx.is_prefill
            and not ctx.is_mixed
            and ctx.chunked_context is None
        ):
            kv = self.kv_a_proj_with_mqa(hidden_states)
            kv_c = _C.strided_rms_norm(
                kv[:, : self.kv_lora_rank],
                self.kv_a_layernorm.weight,
                self.kv_a_layernorm.eps,
            )
            v_proj_weight = self._v_proj_weight
            if v_proj_weight is None:
                v_proj_weight = (
                    self.kv_b_proj.weight.view(
                        self.num_local_heads,
                        self.qk_nope_head_dim + self.v_head_dim,
                        self.kv_lora_rank,
                    )[:, self.qk_nope_head_dim :]
                    .reshape(-1, self.kv_lora_rank)
                    .contiguous()
                )
                object.__setattr__(self, "_v_proj_weight", v_proj_weight)
            return self.o_proj(F.linear(kv_c, v_proj_weight))

        qkv_a_weight = self._qkv_a_weight
        if qkv_a_weight is None:
            qkv_a_weight = torch.cat(
                (self.q_proj.weight, self.kv_a_proj_with_mqa.weight), dim=0
            )
            object.__setattr__(self, "_qkv_a_weight", qkv_a_weight)
        q_size = self.num_local_heads * self.qk_head_dim
        q, kv = F.linear(hidden_states, qkv_a_weight).split(
            [q_size, self.kv_lora_rank + self.qk_rope_head_dim], dim=-1
        )
        q = q.view(num_tokens, self.num_local_heads, self.qk_head_dim)

        kv_c, k_pe = kv.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_c = _C.strided_rms_norm(
            kv_c, self.kv_a_layernorm.weight, self.kv_a_layernorm.eps
        )
        k_pe = k_pe.unsqueeze(1)

        attn_output = self.attn(
            q,
            kv_c,
            k_pe,
            output_shape=(num_tokens, self.num_local_heads * self.v_head_dim),
        )
        return self.o_proj(attn_output)
