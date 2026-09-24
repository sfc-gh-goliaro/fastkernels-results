"""Qwen3 MoE decoder layer: QK-norm attention + MoE with RMSNorm residual connections.

Where the time goes
-------------------
On the captured shapes the MoE block is 68-89% of the layer, the attention front
end (projections, QK-norm, RoPE, the varlen kernel) most of the rest, and the
two RMSNorms about 1%.  So the work here is in the MoE and in the glue around
the attention, not in the attention kernel itself.

* **The MoE's activation and its FP8 requantization are one pass.**  The
  reference materializes ``silu(gate) * up`` as a full bfloat16 tensor
  (``rows x 1536``, 805 MB at 16384 tokens), reads it back to find each 128-wide
  group's absmax, then writes the FP8 copy: three traversals and an extra
  allocation.  ``silu_mul_quant`` does it in one, and the two expert GEMMs are
  the Blackwell block-scaled kernels from the frozen ``moe_grouped_gemm`` /
  ``fp8_linear`` candidates rather than the reference's generic
  ``_fused_moe_kernel`` and its DeepGEMM permute/unpermute pair.

* **QK-norm and M-RoPE are one pass over the packed QKV buffer.**  The reference
  needs five kernels: ``.contiguous()`` on the Q and K slices (a strided view of
  QKV cannot be fed to its norm kernel), a norm for each, then the rotary kernel
  re-reading and rewriting both.  At 16384 tokens that is 268 MB of Q copied
  twice before the first useful FLOP.

* **The rotary cos/sin cache is cast once, not once per call.**
  ``MRotaryEmbedding.forward`` does ``cache.to(query.dtype)`` on entry; the
  cache is float32 and ``4 * max_position_embeddings`` rows, so for this config
  that is a 537 MB read plus a 268 MB write *per call*.  The cast is idempotent,
  so it is hoisted into the module's buffer on first use.

Bit-exactness
-------------
Every kernel here reproduces the reference op's arithmetic exactly -- same
reduction order, same intermediate roundings, same multiply-by-reciprocal.  That
is not perfectionism.  The MoE quantizes to FP8 twice at 128-element group
granularity, and a perturbation ``p`` (relative) upstream of a quantization
comes out the far side of it as roughly ``sqrt(step_fp8 * p)``: one bfloat16 ULP
(0.4%) in becomes ~2% out, which is double the scorer's per-element band.  Two
quantizations in series make that worse.  So either the front end is
bit-identical to the reference or the layer's output is wrong -- there is no
"close enough".  Concretely, this ruled out the *more accurate* variant of four
kernels: an fp32 RoPE rotation, an fp32 SiLU, a single-rounding normalize, and
the fused quantize+GEMV the frozen attention candidate uses for one-token
projections.  All four are nearer the true value; all four fail.

What is free is the *order* of an fp32 accumulation: a grouped GEMM that sums
its 128-wide scale groups in a different order lands within ~1e-7, which the
downstream quantization only turns into ~0.06%.  That is what lets the expert
GEMMs, the routing layout and the alignment be replaced at all.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from fastkernels.infra.cuda_ext import load_op

from ..L1.rms_norm import RMSNorm
from ..L1.moe_align import MoeAlign
from ..L1.moe_grouped_gemm import (
    MoeGroupedGemm,
    _valid_deep_gemm,
    get_triton_config,
)
from ..L1.moe_sum import MoeSum
from ..L1.fp8_linear import PerTokenGroupQuantFp8
from ..L2.attention import LlamaAttention
from ..L2.qwen3_moe import Qwen3MoE as _Qwen3MoE

_C = load_op("qwen3_moe_decoder_fk", "qwen3_moe_decoder_fk.cu")

_FP8_GROUP = 128
# ``add_rmsnorm`` reproduces the reference launch geometry only for these row
# widths; anything else falls back to the reference norm module.
_NORM_WIDTHS = (4096,)

# Rotary classes whose math ``qk_norm_rope`` reproduces: NeoX-style RoPE over
# the full head, and M-RoPE (sectioned or interleaved).  Subclasses (YaRN,
# Gemma's proportional RoPE, ...) rotate differently and keep the reference path.
_ROPE_1D = "RotaryEmbedding"
_ROPE_MULTI = "MRotaryEmbedding"


class _Scratch:
    """Layer-independent scratch pool; decoder layers run one at a time."""

    def __init__(self):
        self._bufs: dict[str, torch.Tensor] = {}

    def get(self, key, shape, dtype, device):
        n = 1
        for s in shape:
            n *= s
        t = self._bufs.get(key)
        if t is None or t.numel() < n or t.dtype != dtype or t.device != device:
            t = torch.empty(n, dtype=dtype, device=device)
            self._bufs[key] = t
        return t[:n].view(shape)


_SCRATCH = _Scratch()


class _MoE(_Qwen3MoE):
    """Qwen3 MoE with the activation and its requantization fused.

    Parameters, loaders and routing are the reference's (``gate`` and
    ``topk_softmax`` decide which experts run, and a different tie break there
    would pick different experts, not merely round differently); only the expert
    computation is replaced.
    """

    def __init__(self, config, quant_config: dict | None = None):
        super().__init__(config, quant_config=quant_config)
        self.moe_align = MoeAlign()
        self.moe_grouped_gemm = MoeGroupedGemm()
        self.moe_sum = MoeSum()
        self.quant_fp8 = PerTokenGroupQuantFp8()
        self._fast = None
        self._plans: dict[int, tuple] = {}

    def _can_fuse(self) -> bool:
        if self._fast is None:
            self._fast = bool(
                self.use_fp8
                and self.block_shape is not None
                and len(self.block_shape) == 2
                and self.block_shape[0] == _FP8_GROUP
                and self.block_shape[1] == _FP8_GROUP
                and self.w13.dtype == torch.float8_e4m3fn
                and self.w2.dtype == torch.float8_e4m3fn
                and self.w13_scale is not None
                and self.w2_scale is not None
                and self.hidden_size % _FP8_GROUP == 0
                and self.intermediate_per_tp % _FP8_GROUP == 0
                and self.tp_size == 1
            )
        return self._fast

    def _build_plan(self, x: torch.Tensor, M: int) -> tuple:
        """Per-token-count launch plan: GEMM config, routing alignment, and
        whether the routing weight is applied late.

        ``get_triton_config`` walks a config table and builds a filename on
        every call; at one token the MoE is launch-bound, so it is resolved once
        per shape.  The alignment and the naive-routing flag only move rows
        between GEMM tiles, so they are free to tune -- a row's dot product is
        the same wherever it lands.  Routing-weight placement is *not* free:
        see ``forward_impl``.
        """
        config = get_triton_config(
            M, self.w13.shape, self.w2.shape, self.top_k,
            use_fp8=True, block_shape=self.block_shape, default_style="legacy",
        )
        late_weight = bool(_valid_deep_gemm(x, self.w13, self.w2)
                           and not torch.cuda.is_current_stream_capturing())
        plan = (config, config["BLOCK_SIZE_M"], False, late_weight)
        self._plans[M] = plan
        return plan

    def forward_impl(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self._can_fuse():
            return super().forward_impl(hidden_states)

        orig_shape = hidden_states.shape
        x = hidden_states.view(-1, self.hidden_size)
        if not x.is_contiguous():
            x = x.contiguous()
        M, K = x.shape
        top_k = self.top_k
        E = self.num_experts
        N = self.intermediate_per_tp
        rows = M * top_k
        dev = x.device

        router_logits = self.gate(x)
        topk_weights, topk_ids = self.topk_softmax(
            router_logits, top_k, renormalize=self.renormalize,
        )

        plan = self._plans.get(M)
        if plan is None:
            plan = self._build_plan(x, M)
        config, align, naive, late_weight = plan

        a1 = _SCRATCH.get("a1", (M, K), torch.float8_e4m3fn, dev)
        a1s = _SCRATCH.get("a1s", (M, K // _FP8_GROUP), torch.float32, dev)
        self.quant_fp8(x, a1, a1s)

        sorted_ids, expert_ids, npp = self.moe_align(
            topk_ids, align, E, naive=naive,
        )

        # One span backs both GEMM outputs: the intermediate is fully consumed
        # by silu_mul_quant before the second GEMM starts writing.
        span = _SCRATCH.get("cache", (rows * max(2 * N, K),), x.dtype, dev)
        inter = span[:rows * 2 * N].view(rows, 2 * N)
        self.moe_grouped_gemm(
            a1, self.w13, inter, topk_weights, sorted_ids, expert_ids, npp,
            mul_routed_weight=False, top_k=top_k, config=config,
            a_scale=a1s, b_scale=self.w13_scale,
            use_fp8_w8a8=True, block_shape=self.block_shape,
        )

        a2 = _SCRATCH.get("a2", (rows, N), torch.float8_e4m3fn, dev)
        a2s = _SCRATCH.get("a2s", (rows, N // _FP8_GROUP), torch.float32, dev)
        _C.silu_mul_quant(inter, a2, a2s)

        # Where the routing weight is applied is observable: above the grouped
        # GEMM's shape threshold the reference switches to a path that scales
        # the *rounded* second-GEMM output during the reduction, below it the
        # weight rides the GEMM epilogue and the product is what gets rounded.
        y = span[:rows * K].view(rows, K)
        self.moe_grouped_gemm(
            a2, self.w2, y, topk_weights, sorted_ids, expert_ids, npp,
            mul_routed_weight=not late_weight, top_k=1, config=config,
            a_scale=a2s, b_scale=self.w2_scale,
            use_fp8_w8a8=True, block_shape=self.block_shape,
        )
        if late_weight:
            return _C.weighted_moe_sum(y, topk_weights, top_k).view(orig_shape)
        return self.moe_sum(y, top_k).view(orig_shape)


class Qwen3MoEDecoderLayer(nn.Module):
    def __init__(self, config, rotary_emb: nn.Module | None = None,
                 quant_config: dict | None = None):
        super().__init__()
        self.self_attn = LlamaAttention(
            config.hidden_size, config.num_attention_heads,
            config.num_key_value_heads, config.head_dim,
            rotary_emb=rotary_emb,
            qk_norm=True,
            rms_norm_eps=config.rms_norm_eps,
            quant_config=quant_config,
        )
        self.mlp = _MoE(config, quant_config=quant_config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self._eps = float(config.rms_norm_eps)
        # Cached per-layer state, resolved on the first forward (the benchmark
        # harness replaces FP8 weights and casts the bf16 ones *after*
        # construction, so nothing layout-dependent can be decided in __init__).
        self._w_src = None
        self._w_in = None
        self._w_post = None
        self._plan_key = None
        self._plan = None
        self._norm_ok = config.hidden_size in _NORM_WIDTHS

    # -- norm -------------------------------------------------------------
    def _weights(self, x: torch.Tensor):
        src = self.input_layernorm.weight
        if self._w_src is not src or self._w_in.dtype != x.dtype:
            self._w_src = src
            self._w_in = self._cast(src, x)
            self._w_post = self._cast(self.post_attention_layernorm.weight, x)
        return self._w_in, self._w_post

    @staticmethod
    def _cast(w: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        w = w.detach()
        if w.dtype != x.dtype or w.device != x.device:
            w = w.to(device=x.device, dtype=x.dtype)
        return w.contiguous()

    # -- attention --------------------------------------------------------
    def _build_plan(self, rope, positions, qkv):
        """Trailing arguments for ``qk_norm_rope``, or None for the reference path."""
        attn = self.self_attn
        if attn.head_dim != 128 or attn.q_norm is None or rope is None:
            return None
        if qkv.dim() != 2 or qkv.stride(1) != 1 or qkv.dtype != torch.bfloat16:
            return None
        qn, kn = attn.q_norm, attn.k_norm
        if not (qn.elementwise_affine and kn.elementwise_affine) or qn.eps != kn.eps:
            return None
        qw, kw = self._cast(qn.weight, qkv), self._cast(kn.weight, qkv)
        if qw.numel() != 128 or kw.numel() != 128:
            return None
        name = type(rope).__name__
        cache = getattr(rope, "cos_sin_cache", None)
        if (name not in (_ROPE_1D, _ROPE_MULTI)
                or not isinstance(cache, torch.Tensor)
                or cache.dtype != torch.bfloat16 or not cache.is_contiguous()
                or cache.dim() != 2 or cache.size(1) != 128
                or getattr(rope, "head_dim", None) != 128
                or getattr(rope, "rotary_dim", 128) != 128
                or positions.dtype != torch.int64):
            return None
        if name == _ROPE_MULTI and positions.dim() == 2:
            sec = rope.mrope_section
            if len(sec) != 3 or sum(sec) != 64:
                return None
            if rope.mrope_interleaved:
                mode, p0, p1 = 3, 3 * int(sec[1]), 3 * int(sec[2])
            else:
                mode, p0, p1 = 2, int(sec[0]), int(sec[0]) + int(sec[1])
        else:
            if name == _ROPE_1D and not rope.is_neox_style:
                return None
            if positions.dim() != 1:
                return None
            mode, p0, p1 = 1, 0, 0
        return (qw, kw, float(qn.eps), cache, p0, p1, mode)

    def _attention(self, positions, hidden_states):
        attn = self.self_attn
        if attn._qkv is None:
            attn._setup()
            # The one- and two-token projections have a fused quantize+GEMV
            # variant, which is faster but not bit-identical to the reference's
            # quantize + block-scaled GEMM.  At a single token the layer's whole
            # output hangs off eight routing decisions, and a one-ULP difference
            # in the QKV projection is enough to swap the eighth expert: the
            # decode shapes then miss by 40% of the output.  Pin both
            # projections to the GEMM path.
            attn._qkv.fp8_gemv = False
            attn._o.fp8_gemv = False
        qkv = attn._qkv(hidden_states)

        rope = attn.rotary_emb
        if rope is not None and attn._rope_cache_ready != id(rope):
            attn._prime_rope(rope, hidden_states.dtype)

        key = (id(rope), positions.dim(), qkv.dtype)
        if self._plan_key != key:
            self._plan_key = key
            self._plan = self._build_plan(rope, positions, qkv)
        plan = self._plan
        if plan is None:
            return attn(positions, hidden_states)

        qw, kw, eps, cache, p0, p1, mode = plan
        q, k = _C.qk_norm_rope(qkv, qw, kw, eps, cache, positions,
                               attn.num_heads, attn.num_kv_heads, p0, p1, mode)
        v = qkv.narrow(1, attn._q_size + attn._kv_size, attn._kv_size)
        return attn._o(attn._attn(q, k, v, hidden_states.shape[0]))

    # -- layer ------------------------------------------------------------
    def forward(self, positions, hidden_states, residual):
        if not (self._norm_ok and hidden_states.dtype == torch.bfloat16
                and hidden_states.dim() == 2 and hidden_states.stride(1) == 1
                and (residual is None or residual.is_contiguous())):
            return self._forward_reference(positions, hidden_states, residual)
        w_in, w_post = self._weights(hidden_states)
        if residual is None:
            h = torch.empty_like(hidden_states)
            _C.add_rmsnorm(h, hidden_states, None, w_in, self._eps)
            residual = hidden_states
        else:
            _C.add_rmsnorm(hidden_states, hidden_states, residual, w_in,
                           self._eps)
            h = hidden_states
        h = self._attention(positions, h)
        _C.add_rmsnorm(h, h, residual, w_post, self._eps)
        return self.mlp(h), residual

    def _forward_reference(self, positions, hidden_states, residual):
        """Reference op sequence, for the layouts ``add_rmsnorm`` does not cover
        (a hidden size other than the captured one, a non-bfloat16 activation, a
        column-strided input, or a strided residual)."""
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self._attention(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        return self.mlp(hidden_states), residual
