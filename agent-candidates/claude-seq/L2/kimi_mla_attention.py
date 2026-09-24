"""Kimi MLA attention -- fastkernels candidate.

Same module tree, ``__init__``/``forward`` signatures and state dict as the
baseline; ``forward`` grows a fast path for the shape the captures actually
exercise (dense prefill, empty paged cache, no chunked context).  Anything else
-- sparse decode, chunked context, a populated paged cache, a mixed batch, an
fp8 projection, tp > 1 -- falls through to the baseline sequence.

What the fast path changes, measured on a B200 against the captured shapes:

* **One projection GEMM instead of two.**  ``q_proj`` and ``kv_a_proj_with_mqa``
  read the same activation, so their weights are concatenated once (lazily, from
  the loaded parameters) and the two ``F.linear`` calls collapse into one.

* **Nothing is copied to build K/V.**  The baseline's ``k = empty();
  k[..., :128] = k_nope; k[..., 128:] = k_pe`` plus FlashAttention's
  ``maybe_contiguous`` clone move ~600 MB at 16k tokens and measure 348 us --
  more than the ``kv_b_proj`` GEMM that produced the data.  On the long shapes
  ``mla_fast.build_kv`` writes K once with 16-byte vector traffic and V is handed
  to FlashAttention as a strided view (only its last dim has to be dense).  On
  the short ones even that goes away: ``mla_fast.attn_small`` gathers its K/V
  tiles straight out of the ``kv_b_proj`` output.

* **No FlashAttention launcher on the short shapes.**  Up to ~160 tokens the
  forward is host-bound -- ~20-45 us of GPU work behind ~110 us of enqueue -- and
  FA4's CuTeDSL launcher is ~20-40 us of that by itself.  ``attn_small`` is a
  single extension call whose ``mma``-based kernel also holds its own on GPU time
  at these lengths (see ``mla_fast.cu``); past the crossover FA4's deeper tile
  pipeline wins and the fast path switches back to it.

* **Fewer Python-side ops.**  Everything the fast path touches -- the fused
  weight, the norm module, the attention op, the head dims -- is cached in the
  instance ``__dict__`` on first use, because at these sizes every
  ``nn.Module.__getattr__`` walk through ``_parameters``/``_modules`` is
  measurable.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ....infra.context import get_context
from ....infra.cuda_ext import load_op
from ....infra.tp import _tp_size
from ..L1.rms_norm import RMSNorm
from .mla_attention_impl import MLAAttention
from .parallel_linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)

_C = load_op("mla_fast", "mla_fast.cu")
_build_kv = _C.build_kv
_attn_small = _C.attn_small

# Crossover between ``attn_small`` and FlashAttention-4, measured as whole
# forwards under the harness' own timing loop: at 128 tokens they are 67 us vs
# 71 us, at 256 they are 93 us vs 74 us.  Below the line ``attn_small`` wins on
# host time (no CuTeDSL launcher, no K build); above it FA4's deeper tile
# pipeline wins on GPU time.  Every captured shape but the 16k prefill and the
# 443-token median sits below it.
_SMALL_TOKENS = 160

# 0 lets the extension pick the block width (it measured 8 warps everywhere).
_ATTN_WARPS = 0


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

        # Fast-path scratch, built on the first forward that can use it (the
        # weights are only meaningful once loaded) and dropped again whenever a
        # state dict is loaded on top.
        self.register_load_state_dict_post_hook(_drop_fast_weights)

    # -- absorbed-decode weights (unchanged) --------------------------------

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

    # -- forward ------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        positions: torch.Tensor | None = None,
        state_manager=None,
    ) -> torch.Tensor:
        del positions, state_manager
        ctx = get_context()
        fast = self.__dict__.get("_fast")
        if fast is None:
            if not self._fast_ok(ctx):
                return self._forward_ref(hidden_states)
            fast = self._build_fast()
        elif not self._fast_ok(ctx):
            return self._forward_ref(hidden_states)
        return self._forward_fast(hidden_states, ctx, fast)

    def _fast_ok(self, ctx) -> bool:
        """True for the dense-prefill shape the fast path reproduces."""
        attn = self.attn
        if attn.is_sparse or self.q_proj.use_fp8 or self.kv_b_proj.use_fp8:
            return False
        if not (getattr(ctx, "is_prefill", False)
                and not getattr(ctx, "is_mixed", False)):
            return False
        if getattr(ctx, "chunked_context", None) is not None:
            return False
        if attn.k_cache.numel() and getattr(ctx, "slot_mapping", None) is not None:
            return False
        if ctx.cu_seqlens_q is None or ctx.cu_seqlens_k is None:
            return False
        return self.o_proj.tp_size == 1

    def _build_fast(self):
        """Cache the fused weight and every hot lookup in the instance dict.

        ``nn.Module.__getattr__`` walks ``_parameters``/``_buffers``/``_modules``
        for anything that is not a plain attribute, and at 1-128 tokens the
        forward is host-bound, so each of those walks is measurable.
        """
        qkv_w = torch.cat(
            (self.q_proj.weight.data, self.kv_a_proj_with_mqa.weight.data),
            dim=0,
        ).contiguous()
        fast = (
            qkv_w,
            self.kv_b_proj.weight.data,
            self.o_proj.weight.data,
            self.kv_a_layernorm,
            self.attn.varlen_attn,
            self.num_local_heads,
            self.qk_head_dim,
            self.kv_lora_rank,
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            self.v_head_dim,
            self.scaling,
        )
        object.__setattr__(self, "_fast", fast)
        return fast

    def _forward_fast(self, hidden_states: torch.Tensor, ctx, fast) -> torch.Tensor:
        (qkv_w, kvb_w, o_w, rms, varlen, heads, qk, lora, nope, rope, dv,
         scale) = fast
        n = hidden_states.shape[0]
        qw = heads * qk

        # q_proj + kv_a_proj_with_mqa share their input, so they share one GEMM.
        qkv = F.linear(hidden_states, qkv_w)
        kv_b = F.linear(rms(qkv[:, qw:qw + lora]), kvb_w)
        k_pe = qkv[:, qw + lora:]
        q = qkv[:, :qw]

        if n <= _SMALL_TOKENS:
            o = _attn_small(q, kv_b, k_pe, ctx.cu_seqlens_q, heads, nope, rope,
                            dv, ctx.max_seqlen_q, scale, _ATTN_WARPS)
        else:
            k = _build_kv(kv_b, k_pe, heads, nope, rope, dv)
            o = varlen(
                q.view(n, heads, qk),
                k.view(n, heads, qk),
                kv_b.view(n, heads, nope + dv)[:, :, nope:],
                cu_seqlens_q=ctx.cu_seqlens_q,
                cu_seqlens_k=ctx.cu_seqlens_q,
                max_seqlen_q=ctx.max_seqlen_q,
                max_seqlen_k=ctx.max_seqlen_q,
                softmax_scale=scale,
                causal=True,
                return_softmax_lse=False,
            )
            if isinstance(o, tuple):
                o = o[0]
            o = o.reshape(n, heads * dv)
        return F.linear(o, o_w)

    def _forward_ref(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Baseline sequence, for every shape the fast path does not cover."""
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


def _drop_fast_weights(module, incompatible_keys):
    module.__dict__.pop("_fast", None)
