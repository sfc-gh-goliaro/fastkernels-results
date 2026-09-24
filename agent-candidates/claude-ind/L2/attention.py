"""Model-level multi-head attention (thin wrapper).

Consolidates vLLM's ``LlamaAttention``, ``Llama4Attention``,
``Qwen3Attention``, and GPT-OSS attention:
QKV projection, optional QK-norm, optional RoPE, then delegates to
``Attention`` for KV cache storage and kernel dispatch.

Unified across Llama, Llama 4, Qwen2, Qwen3, Mixtral, and GPT-OSS:
  - bias:                    Qwen2/GPT-OSS use bias=True on QKV/O projections.
  - qk_norm:                 Qwen3 applies per-head RMSNorm to Q and K before RoPE.
  - nope:                    Llama 4 NoPE layers skip RoPE entirely.
  - use_weightless_qk_norm:  Llama 4 RoPE layers apply weight-less QK RMSNorm after RoPE.
  - attn_temperature_tuning: Llama 4 NoPE layers apply position-dependent temperature.
  - use_sinks:               GPT-OSS learnable attention sinks (per-head biases).
  - sliding_window:          GPT-OSS sliding window attention (even layers only).

Optimisation notes (candidate)
------------------------------
Three costs dominated this layer in a kernel profile, none of them the GEMMs:

1. Both rotary modules keep ``cos_sin_cache`` in fp32 (a non-persistent buffer)
   and re-cast *all* of it to the activation dtype inside every forward -- 0.5 GB
   read + 0.27 GB written per layer per step for Qwen3-VL's 1M-row cache, ~130 us.
   It is cast once here and kept, which is what vLLM stores by construction.

2. Between the QKV GEMM and the attention kernel the eager path walks q five
   times: ``contiguous()`` inside ``RMSNorm.forward_cuda``, the norm, the RoPE
   cos/sin gather, another ``contiguous()``, and the rotation.
   ``attn_fused.cu:qkv_post`` does the norm and the rotation in one
   read-modify-write over the packed QKV buffer and leaves q/k/v as strided
   views the attention kernel reads directly -- the layout the configs without
   QK-norm already used.

3. At decode sizes the layer is *host* bound, and FlashAttention's CuTeDSL
   launcher alone costs ~35 us of host time to attend one query row to one key.
   ``attn_short`` covers short key spans for ~5 us of host time; everything
   longer stays on FlashAttention, which wins outright once its tensor cores
   have work to do.

The FP8 projections also get their activation quantizer replaced
(``quant_fp8_groups``, bit-identical to the reference, ~1.35x its throughput);
the block-scaled GEMMs themselves are left alone -- measured against DeepGEMM
and cuBLAS, hand-written decode GEMVs were not an improvement on this machine.

Everything the fused path cannot express (NoPE, weight-less QK norm, temperature
tuning, non-neox or partial rotary, unusual head dims, a paged KV cache, mixed
or tree-verify batches) falls back to the reference implementation below, so
behaviour there is unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ....infra.tp import _tp_size
from ....infra.context import get_context
from .parallel_linear import QKVParallelLinear, RowParallelLinear
from .attention_impl import Attention
from ..L1.rms_norm import RMSNorm

_ROPE_NONE, _ROPE_1D, _ROPE_MROPE_INTERLEAVED, _ROPE_MROPE_SECTIONED = 0, 1, 2, 3

# Longest key span still handled by the short-span attention kernel.  Measured
# crossover is around 128 keys per query (where its one-key-per-iteration
# reduction stops being cheaper than FlashAttention's host launch); staying well
# inside that keeps the choice safe.
_SHORT_ATTN_MAX_K = 32
# Batch width below which the FP8 activation scratch is worth keeping around:
# decode-shaped calls care about the two saved allocations, prefill-shaped ones
# would pin tens of MB per layer for nothing.
_QBUF_MAX_M = 8

_EXT = None
_UE8M0 = None


def _ext():
    """JIT-load the fused kernels on first use."""
    global _EXT
    if _EXT is None:
        from ....infra.cuda_ext import load_op
        _EXT = load_op("fk_cand_attn_fused", "attn_fused.cu")
    return _EXT


def _ue8m0() -> bool:
    """Scale format the block-scaled weights were prepared with."""
    global _UE8M0
    if _UE8M0 is None:
        from ..L1.fp8_grouped_gemm_contiguous import _is_deep_gemm_e8m0_used
        _UE8M0 = bool(_is_deep_gemm_e8m0_used())
    return _UE8M0


class LlamaAttention(nn.Module):
    """Model-level attention: qkv_proj -> [qk_norm] -> [rope] -> Attention -> o_proj."""

    def __init__(self, hidden_size: int, num_attention_heads: int,
                 num_key_value_heads: int, head_dim: int,
                 rotary_emb: nn.Module | None = None,
                 bias: bool = False,              # Qwen2 / GPT-OSS
                 qk_norm: bool = False,           # Qwen3
                 rms_norm_eps: float = 1e-6,
                 nope: bool = False,              # Llama 4
                 use_weightless_qk_norm: bool = False,   # Llama 4
                 attn_temperature_tuning: bool = False,  # Llama 4
                 floor_scale: float = 8192.0,            # Llama 4
                 attn_scale: float = 0.1,                # Llama 4
                 quant_config: dict | None = None,
                 attention_chunk_size: int | None = None,
                 o_proj_bias: bool = False,              # GPT-OSS
                 use_sinks: bool = False,                # GPT-OSS
                 sliding_window: int | None = None,      # GPT-OSS
                 layer_idx: int = 0):                     # GPT-OSS
        super().__init__()
        tp = _tp_size()
        self.num_heads = num_attention_heads // tp
        if num_key_value_heads >= tp:
            self.num_kv_heads = num_key_value_heads // tp
        else:
            self.num_kv_heads = 1
        self.head_dim = head_dim
        self.rotary_emb = rotary_emb
        self.nope = nope
        self.attn_temperature_tuning = attn_temperature_tuning and nope
        self.floor_scale = floor_scale
        self.attn_scale = attn_scale

        self.qkv_proj = QKVParallelLinear(
            hidden_size, head_dim,
            num_attention_heads, num_key_value_heads,
            bias=bias,
            quant_config=quant_config,
        )
        self.o_proj = RowParallelLinear(
            num_attention_heads * head_dim, hidden_size,
            bias=o_proj_bias,
            quant_config=quant_config,
        )

        self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps) if qk_norm else None  # Qwen3
        self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps) if qk_norm else None  # Qwen3

        wl_qk = use_weightless_qk_norm and not nope  # Llama 4 RoPE layers only
        self.q_wl_norm = RMSNorm(head_dim, eps=rms_norm_eps, elementwise_affine=False) if wl_qk else None
        self.k_wl_norm = RMSNorm(head_dim, eps=rms_norm_eps, elementwise_affine=False) if wl_qk else None

        # GPT-OSS: per-layer sliding window (even layers only) and attention sinks
        per_layer_sw = sliding_window if layer_idx % 2 == 0 else None

        if use_sinks:
            self.sinks = nn.Parameter(torch.zeros(self.num_heads))
            self.sinks.weight_loader = self._sinks_weight_loader
        else:
            self.sinks = None

        self.attn = Attention(
            self.num_heads, head_dim, head_dim ** -0.5,
            num_kv_heads=self.num_kv_heads,
            sliding_window=per_layer_sw,
            sinks=self.sinks,
            attention_chunk_size=attention_chunk_size,
        )

        self.q_size = self.num_heads * head_dim
        self.kv_size = self.num_kv_heads * head_dim
        # Fused-path eligibility that depends only on init args.
        self._fusable = (
            not nope
            and not wl_qk
            and not self.attn_temperature_tuning
            and head_dim in (64, 128, 256)
        )
        self._plans: dict = {}
        self._qbuf: dict = {}
        # Init-time halves of the attention dispatch conditions.
        self._dense_ok = (not self.attn._triton_only
                          and attention_chunk_size is None)
        self._short_ok = (self._dense_ok
                          and self.num_heads % self.num_kv_heads == 0
                          and head_dim in (64, 128, 256)
                          and self.attn._fa3_window_size[1] <= 0)
        # RowParallelLinear also all-reduces; only bypass it at tp=1.
        self._o_fast = (tp == 1)

    def _sinks_weight_loader(self, param, loaded_weight):
        """TP-shard attention sinks across heads."""
        from ....infra.tp import _tp_rank
        rank = _tp_rank()
        heads_per_rank = param.data.size(0)
        start = rank * heads_per_rank
        param.data.copy_(loaded_weight.narrow(0, start, heads_per_rank))

    def _get_attn_scale(self, positions):  # Llama 4 NoPE only
        """Position-dependent attention temperature scaling."""
        floor = torch.floor((positions.float() + 1.0) / self.floor_scale)
        scale = torch.log(floor + 1.0) * self.attn_scale + 1.0
        return scale.unsqueeze(-1)

    # ------------------------------------------------------------------
    # Fused path helpers
    # ------------------------------------------------------------------
    def _make_plan(self, rope, positions):
        """Classify *rope* for the fused kernel: ``(mode, sections)`` or None.

        The caller caches this per positions rank for its own rotary: the
        classification costs two module imports and a few isinstance checks,
        which is real time at decode sizes where the layer is tens of
        microseconds end to end.
        """
        if rope is None:
            return (_ROPE_NONE, (0, 0, 0))
        from ..L1.rotary_emb import RotaryEmbedding
        from ..L1.mrope import MRotaryEmbedding
        cache = getattr(rope, "cos_sin_cache", None)
        if not isinstance(cache, torch.Tensor) or cache.dim() != 2:
            return None
        if int(cache.shape[1]) != self.head_dim:  # partial / non-full rotary
            return None
        if isinstance(rope, MRotaryEmbedding):
            if positions.dim() == 1:
                return (_ROPE_1D, (0, 0, 0))
            if positions.dim() != 2 or int(positions.shape[0]) != 3:
                return None
            return ((_ROPE_MROPE_INTERLEAVED if rope.mrope_interleaved
                     else _ROPE_MROPE_SECTIONED),
                    tuple(int(x) for x in rope.mrope_section))
        if isinstance(rope, RotaryEmbedding):
            if not rope.is_neox_style or positions.dim() != 1:
                return None
            return (_ROPE_1D, (0, 0, 0))
        return None

    def _project(self, lin, x):
        """``lin(x)``, but with our own activation quantizer on the FP8 path.

        The reference re-runs vLLM's ``per_token_group_quant_8bit`` kernel (16
        threads per 128-wide group, staged through shared memory, scale stores
        scattered a row-stride apart) before every DeepGEMM call;
        ``quant_fp8_groups`` produces bit-identical fp8 and UE8M0 scales with one
        warp per group and coalesced scale stores, at ~1.35x the throughput.
        The GEMM itself is left to the projection's own kernel -- measured
        against it, a hand-written decode GEMV was not an improvement here.
        """
        if not lin.use_fp8 or x.dim() != 2:
            return lin(x)
        W, Ws = lin.weight, lin.weight_scale_inv
        M, K = x.shape
        if K % 128 or not x.is_contiguous():
            return lin(x)
        # Scratch for the quantized activation and its scales.  Stream order
        # already serialises the GEMM that reads them against the next call's
        # quant, so one buffer per (M, K) is enough and saves two allocations
        # per projection -- host time the decode shapes feel.
        key = (M, K)
        buf = self._qbuf.get(key) if M <= _QBUF_MAX_M else None
        if buf is None:
            buf = (
                torch.empty(M, K, dtype=torch.float8_e4m3fn, device=x.device),
                # Column-major, as DeepGEMM's TMA loads require.
                torch.empty((K // 128, M), dtype=torch.float32,
                            device=x.device).permute(1, 0),
            )
            if M <= _QBUF_MAX_M:
                if len(self._qbuf) > 7:
                    self._qbuf.clear()
                self._qbuf[key] = buf
        q, s = buf
        _ext().quant_fp8_groups(x, q, s, _ue8m0())
        out = torch.empty(M, W.shape[0], dtype=x.dtype, device=x.device)
        torch.ops.fastkernels_fp8.fp8_gemm_nt(q, s, W, Ws, out)
        if lin.bias is not None:
            out = out + lin.bias
        return out

    def _attention(self, q, k, v, N):
        """Dense (cacheless) prefill straight into the attention kernel.

        Mirrors ``Attention._forward_pure``'s unpaged branch but skips the
        dispatch bookkeeping, and swaps FlashAttention for ``attn_short`` when
        the key span is short enough that FA4's host-side launch (~35 us) and
        tile choice cost more than the whole attention does.  Anything else --
        paged prefill/decode, mixed batches, tree verify, chunked local
        attention, the Triton/SDPA fallbacks -- defers to the layer itself.
        """
        a = self.attn
        ctx = get_context()
        if (self._dense_ok and not a._use_custom_op and not a.k_cache.numel()
                and not a.v_cache.numel() and ctx.is_prefill and not ctx.is_mixed
                and not getattr(ctx, "is_tree_verify", False)
                and a._group_block_tables(ctx) is None):
            cu_q, cu_k = ctx.cu_seqlens_q, ctx.cu_seqlens_k
            if (self._short_ok and int(ctx.max_seqlen_k) <= _SHORT_ATTN_MAX_K
                    and cu_q.dtype == torch.int32 and cu_k.dtype == torch.int32
                    and (a._fa3_sinks is None or a._fa3_sinks.dtype == q.dtype)):
                return _ext().attn_short(q, k, v, cu_q, cu_k, a._fa3_sinks,
                                         a.scale, a._fa3_window_size[0])
            fa_extra = {}
            if a._fa3_sinks is not None:
                fa_extra["s_aux"] = a._fa3_sinks
            if a._fa3_window_size != (-1, -1):
                fa_extra["window_size"] = a._fa3_window_size
            o = a.prefill_op(
                q, k, v,
                cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
                max_seqlen_q=ctx.max_seqlen_q, max_seqlen_k=ctx.max_seqlen_k,
                softmax_scale=a.scale, causal=True, **fa_extra,
            )
            return o.reshape(N, self.num_heads * self.head_dim)
        return a(q.reshape(N, -1), k.reshape(N, -1), v.reshape(N, -1))

    # ------------------------------------------------------------------
    def forward(self, positions, hidden_states, rotary_emb=None):
        N = hidden_states.shape[0]
        qkv = self._project(self.qkv_proj, hidden_states)
        rope = rotary_emb if rotary_emb is not None else self.rotary_emb

        if self._fusable and qkv.dim() == 2 and qkv.is_contiguous():
            # Own rotary: classify once per positions rank and keep it (the
            # module outlives us, so the cache can't go stale).  A rotary handed
            # in per call is classified per call.
            if rope is self.rotary_emb:
                ndim = positions.dim()
                plan = self._plans.get(ndim, False)
                if plan is False:
                    plan = self._plans[ndim] = self._make_plan(rope, positions)
            else:
                plan = self._make_plan(rope, positions)
            if plan is not None:
                mode, sec = plan
                nh, nkv, hd = self.num_heads, self.num_kv_heads, self.head_dim
                if mode != _ROPE_NONE or self.q_norm is not None:
                    cache = None
                    if mode != _ROPE_NONE:
                        # Cast the cos/sin table once, not once per call: the
                        # reference casts all of it (0.5 GB for Qwen3-VL) every
                        # time.  Same values, non-persistent buffer.
                        cache = rope.cos_sin_cache
                        if cache.dtype != qkv.dtype:
                            cache = cache.to(qkv.dtype)
                            rope.cos_sin_cache = cache
                        if not positions.is_contiguous():
                            positions = positions.contiguous()
                    qn = self.q_norm
                    _ext().qkv_post(
                        qkv, positions,
                        None if qn is None else qn.weight,
                        None if qn is None else self.k_norm.weight,
                        cache, 0.0 if qn is None else qn.eps,
                        nh, nkv, hd, mode, sec[0], sec[1], sec[2],
                    )
                qkv3 = qkv.view(N, nh + 2 * nkv, hd)
                o = self._attention(qkv3[:, :nh], qkv3[:, nh:nh + nkv],
                                    qkv3[:, nh + nkv:], N)
                if self._o_fast:
                    return self._project(self.o_proj, o)
                return self.o_proj(o)

        return self._forward_ref(positions, hidden_states, qkv, rope, N)

    # ------------------------------------------------------------------
    def _forward_ref(self, positions, hidden_states, qkv, rope, N):
        """Reference (unfused) path, kept for configs the kernel can't express."""
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)

        # Learnable QK norm (Qwen3: before RoPE)
        if self.q_norm is not None:
            q_shape, k_shape = q.shape, k.shape
            q = self.q_norm(
                q.view(N, self.num_heads, self.head_dim)).view(q_shape)
            k = self.k_norm(
                k.view(N, self.num_kv_heads, self.head_dim)).view(k_shape)

        if not self.nope and rope is not None:
            q, k = rope(positions, q, k)

        # Weight-less QK norm (Llama 4: after RoPE, only on RoPE layers)
        if self.q_wl_norm is not None:
            q = self.q_wl_norm(q.view(-1, self.head_dim)).view(N, -1)
            k = self.k_wl_norm(k.view(-1, self.head_dim)).view(N, -1)

        # Temperature tuning (Llama 4: only on NoPE layers)
        if self.attn_temperature_tuning:
            q = (q * self._get_attn_scale(positions)).to(q.dtype)

        attn_output = self.attn(q, k, v)
        return self.o_proj(attn_output)
