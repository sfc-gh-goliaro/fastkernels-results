from __future__ import annotations

import torch
import torch.nn as nn

from ....infra.context import get_context
from ....infra.cuda_ext import lazy_op
from ....infra.tp import _tp_size
from ..L1.rms_norm import RMSNorm
from .mla_attention_impl import MLAAttention
from .parallel_linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)

_C = lazy_op("kimi_mla_fused", "mla_fused.cu")

# Head geometry the fused kernels are specialised for.
_NOPE, _ROPE, _VDIM, _LATENT = 128, 64, 128, 512

# Above this many tokens FlashAttention-4's tcgen05 pipeline beats the
# mma.sync fused kernel, so the wider (identity-padded) KV weight is used and
# attention goes back to FA4 -- still without the reference's concat copies.
_FUSED_ATTN_TOKENS = 64


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

        self._tp = tp
        self._geometry_ok = (
            self.qk_nope_head_dim == _NOPE
            and self.qk_rope_head_dim == _ROPE
            and self.v_head_dim == _VDIM
            and self.kv_lora_rank == _LATENT
            and quant_config is None
            and self.kv_a_layernorm.elementwise_affine
        )
        self._packed = False

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
        # Same post-weight-load hook the engine already calls, so rebuild the
        # fused projection weights next time the fast path runs.
        self._packed = False

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

    # -- fused fast path ---------------------------------------------------

    def _pack(self):
        """Build the fused projection weights.

        Called once, the first time the fast path runs -- the engine loads
        weights (and calls :meth:`compute_absorbed_weights`) before the first
        forward, same as the reference's absorbed-weight precompute.

        ``w1t``   q_proj and kv_a_proj_with_mqa concatenated, so both come out
                  of one GEMM and the rope columns land next to the latent.
        ``w2t``   kv_b_proj as-is; the fused attention kernel reads k_nope and v
                  straight out of its [tokens, heads, 256] output.
        ``wkvt``  k and v projections merged, with an identity block that copies
                  k_pe into the rope half of every head.  Hands FlashAttention a
                  ready [tokens, heads, 192] K without the reference's two
                  strided concat copies.
        """
        wq = self.q_proj.weight.data
        wa = self.kv_a_proj_with_mqa.weight.data
        wb = self.kv_b_proj.weight.data
        h = self.num_local_heads

        w1 = torch.cat([wq, wa], dim=0).contiguous()
        wb_v = wb.view(h, _NOPE + _VDIM, _LATENT)
        wkv = wb.new_zeros(h * (_NOPE + _ROPE) + h * _VDIM, _LATENT + _ROPE)
        k_part = wkv[: h * (_NOPE + _ROPE)].view(h, _NOPE + _ROPE, _LATENT + _ROPE)
        k_part[:, :_NOPE, :_LATENT] = wb_v[:, :_NOPE, :]
        k_part[:, _NOPE:, _LATENT:] = torch.eye(
            _ROPE, dtype=wb.dtype, device=wb.device)
        wkv[h * (_NOPE + _ROPE):, :_LATENT] = wb_v[:, _NOPE:, :].reshape(
            h * _VDIM, _LATENT)

        self._w1t = w1.t()
        self._w2t = wb.t()
        self._wkvt = wkv.t()
        self._wot = self.o_proj.weight.data.t()
        self._gamma = self.kv_a_layernorm.weight.data.to(
            device=wq.device, dtype=torch.bfloat16).contiguous()
        self._eps = self.kv_a_layernorm.eps
        self._qw = h * self.qk_head_dim
        self._vw = h * _VDIM
        self._norm = _C.rmsnorm_latent
        self._attn = _C.mla_attention
        self._packed = True

    def _forward_fast(self, hidden_states, ctx):
        m = hidden_states.shape[0]
        h = self.num_local_heads
        qw = self._qw

        y = torch.mm(hidden_states, self._w1t)
        a576, latent, rope = self._norm(
            y, self._gamma, self._eps, qw, _LATENT, _ROPE)

        if m <= _FUSED_ATTN_TOKENS:
            kv = torch.mm(latent, self._w2t)
            o = self._attn(y, kv, rope, self.scaling, h)
        else:
            kvc = torch.mm(a576, self._wkvt)
            dqk = self.qk_head_dim
            o = self.attn.varlen_attn(
                y.as_strided((m, h, dqk), (y.stride(0), dqk, 1)),
                kvc.as_strided((m, h, dqk), (kvc.stride(0), dqk, 1)),
                kvc.as_strided((m, h, _VDIM), (kvc.stride(0), _VDIM, 1), qw),
                cu_seqlens_q=ctx.cu_seqlens_q,
                cu_seqlens_k=ctx.cu_seqlens_k,
                max_seqlen_q=ctx.max_seqlen_q,
                max_seqlen_k=ctx.max_seqlen_k,
                softmax_scale=self.scaling,
                causal=True,
            ).view(m, self._vw)
        # tp == 1 skips RowParallelLinear's all-reduce, so the bare GEMM is
        # equivalent; above that, go through o_proj so the reduce still happens.
        if self._tp == 1:
            return torch.mm(o, self._wot)
        return self.o_proj(o)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        positions: torch.Tensor | None = None,
        state_manager=None,
    ) -> torch.Tensor:
        del positions, state_manager
        ctx = get_context()
        # The fused path covers the dense-prefill case the reference reaches
        # through MLAAttention._forward_mha: one contiguous bf16 sequence, no
        # paged cache, no chunked context.  Anything else runs the reference.
        if (
            self._geometry_ok
            and ctx.is_prefill
            and not ctx.is_mixed
            and ctx.chunked_context is None
            and ctx.slot_mapping is None
            and hidden_states.dtype is torch.bfloat16
            and hidden_states.dim() == 2
            and hidden_states.shape[0] > 0
            and hidden_states.stride(1) == 1
            and not self.attn.k_cache.numel()
            and ctx.cu_seqlens_q is not None
            and ctx.cu_seqlens_q.numel() == 2
        ):
            if not self._packed:
                self._pack()
            return self._forward_fast(hidden_states, ctx)

        num_tokens = hidden_states.shape[0]

        q = self.q_proj(hidden_states)
        q = q.view(num_tokens, self.num_local_heads, self.qk_head_dim)

        kv = self.kv_a_proj_with_mqa(hidden_states)
        kv_c, k_pe = kv.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_c = self.kv_a_layernorm(kv_c)
        k_pe = k_pe.unsqueeze(1)

        attn_output = self.attn(
            q,
            kv_c,
            k_pe,
            output_shape=(num_tokens, self.num_local_heads * self.v_head_dim),
        )
        return self.o_proj(attn_output)
