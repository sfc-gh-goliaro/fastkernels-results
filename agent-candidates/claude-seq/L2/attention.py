"""Model-level multi-head attention (thin wrapper).

Consolidates vLLM's ``LlamaAttention``, ``Llama4Attention``,
``Qwen3Attention``, and GPT-OSS attention:
QKV projection, optional QK-norm, optional RoPE, then delegates to
``Attention`` for KV cache storage and kernel dispatch.

Optimizations over the baseline wrapper, all of them in ``forward``'s glue
rather than in the attention kernel itself:

* **The rotary cos/sin cache is cast once, not once per call.**  Both
  ``RotaryEmbedding.forward_cuda`` and ``MRotaryEmbedding.forward`` do
  ``cache.to(query.dtype)`` on entry.  The cache is float32 and sized
  ``max_position_embeddings * {1,4}`` rows, so for Qwen3-VL that is a 537 MB
  read plus a 268 MB write *per layer per step* -- 170 us, which is 70% of a
  decode step for this operator.  The cast is idempotent, so we hoist it into
  the module's own buffer on first use and every later call sees
  ``cache.dtype == query.dtype`` and skips it.

* **QK-norm and RoPE run as one in-place pass over the fused QKV buffer.**
  The baseline materializes a fresh contiguous Q (67 MB at 1000 tokens,
  268 MB at 16384) out of the norm, then RoPE reads and rewrites it.  Fusing
  them removes one full read+write of Q and K and the allocation, and leaves
  Q/K as strided views of the QKV buffer, which is what the attention kernel
  wants anyway.

* **The FP8 activation quantizer is rewritten.**  The reference one stages
  every 128-element group through shared memory behind a barrier and runs at
  ~0.94 TB/s, which costs more than the block-scaled GEMM it feeds (DeepGEMM
  finishes the captured projections at ~3.5 PFLOP/s).  A register-resident
  version is ~2x faster and bit-identical, so the GEMM result is unchanged.

* **Small-M QKV / output projections run as a fused FP8 GEMV** instead of
  quantize + block-scaled GEMM: at one or two tokens the projection is nothing
  but a read of the FP8 weight, and a GEMM tile wastes most of its shape.

* **The attention call is reduced to one kernel launch.**  At a few tokens per
  step the dense-prefill dispatch (five Python frames of per-call backend
  re-derivation, then a CuTe-DSL launcher that rebuilds its argument wrapper)
  costs more than the kernel; the per-layer parts of that decision are hoisted
  into ``_setup``, and for short sequences a warp-per-query kernel replaces the
  reference kernel outright.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ....infra.context import get_context
from ....infra.cuda_ext import lazy_op
from ....infra.fa_utils import FA_VERSION as _FA_VERSION
from ....infra.fa_utils import flash_attn_varlen_func as _flash_attn_varlen
from ....infra.tp import _tp_size
from .parallel_linear import QKVParallelLinear, RowParallelLinear
from .attention_impl import Attention
from ..L1.rms_norm import RMSNorm

_C = lazy_op("attention_l2", "attention_l2.cu")

_EMPTY = torch.empty(0)

# ``_build_plan`` sentinel: the layer has neither QK-norm nor RoPE, so there is
# no kernel to launch but the short path through ``forward`` still applies.
_NOOP = ()

# Rotary classes whose math the fused kernel reproduces: NeoX-style RoPE over
# the full head, and non-interleaved M-RoPE.  Subclasses (Gemma4's proportional
# RoPE, YaRN, ...) rotate differently and keep the baseline path.
_ROPE_1D = "RotaryEmbedding"
_ROPE_MULTI = "MRotaryEmbedding"

# ``fp8_gemv`` stages the quantized activation in shared memory; keep it inside
# the default (no opt-in) dynamic limit.
_FP8_GEMV_SMEM = 48 * 1024

# The block-scaled FP8 GEMM itself is left exactly as the baseline's (its
# registered ``fastkernels_fp8::fp8_gemm_nt``, which finishes the captured
# projections at ~3.5 PFLOP/s); only the activation-quantization pass in front
# of it is replaced, see ``quant_fp8_group128`` in attention_l2.cu.
_fp8_gemm_nt = None


def _fp8_linear(x: torch.Tensor, w: torch.Tensor, ws: torch.Tensor,
                bias: torch.Tensor | None) -> torch.Tensor:
    """Block-scaled FP8 linear with the fast activation quantizer."""
    n, k = w.shape
    if not x.is_contiguous():
        x = x.contiguous()
    m = x.shape[0]
    q = torch.empty((m, k), dtype=torch.float8_e4m3fn, device=x.device)
    # Column-major scales (stride (1, m)), the layout the GEMM reads.
    scale = torch.empty((k // 128, m), dtype=torch.float32,
                        device=x.device).permute(1, 0)
    _C.quant_fp8_group128(x, q, scale)
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    _fp8_gemm_nt(q, scale, w, ws, out)
    if bias is not None:
        out = out + bias
    return out


# Above this many rows the block-scaled GEMM wins; at or below it the FP8
# projection is nothing but a read of the weight.  (The bf16 projections are not
# worth replacing at any M -- cuBLAS already runs the M=1 case at roughly a
# device read of the weight.)
_GEMV_MAX_M = 2

# Above this many keys per query the O(keys) warp reductions in ``attn_small``
# cost more than the reference kernel's launcher, so the reference wins.
_SMALL_ATTN_MAX_SEQ = 64


class _Proj:
    """Pre-resolved projection: which GEMM path a given linear should take.

    The choice only depends on the layer's weights, so it is made once instead
    of re-deriving ``use_fp8`` / shape / layout predicates on every token.
    """

    __slots__ = ("proj", "weight", "scale", "bias", "fp8_gemv", "fp8_quant")

    def __init__(self, proj):
        self.proj = proj
        self.weight = proj.weight
        self.bias = proj.bias
        self.scale = getattr(proj, "weight_scale_inv", None)
        k = self.weight.shape[1]
        self.fp8_gemv = bool(
            proj.use_fp8 and k % 512 == 0 and self.scale is not None
            and self.scale.dtype == torch.int32 and self.scale.stride(0) == 1
            and _GEMV_MAX_M * k + _GEMV_MAX_M * (k // 128) * 4 <= _FP8_GEMV_SMEM
        )
        self.fp8_quant = bool(proj.use_fp8 and k % 128 == 0
                              and self.scale is not None)
        if self.fp8_quant:
            global _fp8_gemm_nt
            if _fp8_gemm_nt is None:
                _fp8_gemm_nt = torch.ops.fastkernels_fp8.fp8_gemm_nt

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            if x.shape[0] <= _GEMV_MAX_M and self.fp8_gemv:
                return _C.fp8_gemv(x, self.weight, self.scale, self.bias)
            if self.fp8_quant:
                return _fp8_linear(x, self.weight, self.scale, self.bias)
        return self.proj(x)


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

        self._q_size = self.num_heads * head_dim
        self._kv_size = self.num_kv_heads * head_dim
        self._split = [self._q_size, self._kv_size, self._kv_size]
        # id() of the rotary module whose cos/sin cache has already been cast to
        # the activation dtype (see the module docstring).
        self._rope_cache_ready: int = 0
        # Cached fused-kernel plan: (key, args tuple) or (key, None) when this
        # configuration falls back to the baseline op sequence.
        self._plan_key = None
        self._plan = None
        self._qkv = None
        self._o = None
        # Attention call stripped to the one kernel launch (see _attn).
        self._fa_kwargs = None
        self._fa_ok = False
        self._small_ok = False
        self._window_left = -1
        self._sinks_f32 = _EMPTY

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

    def _setup(self) -> None:
        """Resolve the per-layer GEMM paths and the attention fast path once.

        Deferred out of ``__init__`` because the benchmark harness replaces the
        FP8 ``weight``/``weight_scale_inv`` Parameters (and casts the bf16 ones)
        *after* construction, so the layouts this branches on are not final yet.
        """
        self._qkv = _Proj(self.qkv_proj)
        self._o = _Proj(self.o_proj)

        attn = self.attn
        # The dense (unpaged) prefill branch of ``Attention.forward_impl``:
        # nothing to store into a KV cache, no chunked local attention, no
        # Triton-only head size, no sliding-window KV group.  Everything below
        # that branch is constant per layer, so it is hoisted here and the call
        # becomes one FlashAttention launch instead of five Python frames of
        # backend re-derivation per layer per step.
        self._fa_ok = (
            attn.attention_chunk_size is None
            and attn._use_trtllm          # the dense fallback inside TRTLLMPrefill
            and getattr(attn, "_sliding_group_id", None) is None
            and not attn._use_custom_op
        )
        if not self._fa_ok:
            return
        kwargs = {
            "softmax_scale": attn.scale,
            "causal": True,
            "fa_version": _FA_VERSION,
            # Dense prefill is compute bound; the FA4 auto heuristic otherwise
            # picks a split-KV kernel that fails to compile in this build.
            "num_splits": 1,
        }
        if attn._fa3_sinks is not None:
            kwargs["s_aux"] = attn._fa3_sinks
        if attn._fa3_window_size != (-1, -1):
            kwargs["window_size"] = attn._fa3_window_size
        self._fa_kwargs = kwargs

        self._small_ok = self.head_dim in (64, 128)
        self._window_left = attn._fa3_window_size[0]
        sinks = attn._fa3_sinks
        self._sinks_f32 = (
            sinks.detach().float().contiguous() if sinks is not None else _EMPTY)

    def _attn(self, q, k, v, N):
        """Dense varlen prefill, bypassing the per-call backend re-derivation."""
        attn = self.attn
        if self._fa_ok and not attn.k_cache.numel():
            ctx = get_context()
            if (ctx.is_prefill and not ctx.is_mixed
                    and ctx.block_tables is None
                    and not getattr(ctx, "is_tree_verify", False)):
                nh, hd = self.num_heads, self.head_dim
                kvh = self.num_kv_heads
                cu_q = ctx.cu_seqlens_q
                if (self._small_ok
                        and ctx.max_seqlen_q <= _SMALL_ATTN_MAX_SEQ
                        and cu_q.data_ptr() == ctx.cu_seqlens_k.data_ptr()
                        and cu_q.dtype == torch.int32):
                    return _C.attn_small(
                        q.view(N, nh, hd), k.view(N, kvh, hd),
                        v.view(N, kvh, hd), cu_q, self._sinks_f32,
                        self._window_left, attn.scale, ctx.max_seqlen_q,
                    ).view(N, nh * hd)
                o = _flash_attn_varlen(
                    q.view(N, nh, hd), k.view(N, kvh, hd), v.view(N, kvh, hd),
                    cu_seqlens_q=ctx.cu_seqlens_q, cu_seqlens_k=ctx.cu_seqlens_k,
                    max_seqlen_q=ctx.max_seqlen_q, max_seqlen_k=ctx.max_seqlen_k,
                    **self._fa_kwargs,
                )
                return o.view(N, nh * hd)
        return attn(q, k, v)

    def _prime_rope(self, rope: nn.Module, dtype: torch.dtype) -> None:
        """Cast the rotary cos/sin cache to *dtype* once, in place.

        ``RotaryEmbedding.forward_cuda`` / ``MRotaryEmbedding.forward`` both do
        this cast per call and then throw the result away.  Writing it back into
        the (non-persistent, so state-dict-invisible) buffer makes every later
        call's ``cache.dtype != query.dtype`` test false.
        """
        self._rope_cache_ready = id(rope)
        cache = getattr(rope, "cos_sin_cache", None)
        if isinstance(cache, torch.Tensor) and cache.dtype != dtype:
            rope.cos_sin_cache = cache.to(dtype)

    def _build_plan(self, rope, positions, qkv):
        """Decide whether one fused pass can replace QK-norm + RoPE.

        Returns the trailing argument tuple for ``_C.qk_norm_rope`` -- weights,
        eps, cos/sin cache, rope mode and M-RoPE section bounds -- or ``None``
        when the configuration is outside what the kernel reproduces, in which
        case ``forward`` runs the baseline op sequence.
        """
        hd = self.head_dim
        if hd not in (64, 128):
            return None
        # Llama 4 weight-less QK norm / temperature tuning run after RoPE and
        # are not part of the fused pass.
        if self.q_wl_norm is not None or self.attn_temperature_tuning:
            return None
        if qkv.dim() != 2 or qkv.stride(1) != 1 or qkv.dtype != torch.bfloat16:
            return None

        if self.q_norm is not None:
            qn, kn = self.q_norm, self.k_norm
            if not (qn.elementwise_affine and kn.elementwise_affine):
                return None
            if qn.eps != kn.eps:
                return None
            qw, kw = qn.weight.detach(), kn.weight.detach()
            if (qw.dtype != torch.bfloat16 or qw.numel() != hd
                    or not qw.is_contiguous() or kw.numel() != hd
                    or not kw.is_contiguous()):
                return None
            eps = float(qn.eps)
        else:
            qw = kw = _EMPTY
            eps = 0.0

        mode, s0, s1 = 0, 0, 0
        cache = _EMPTY
        if not self.nope and rope is not None:
            name = type(rope).__name__
            cache = getattr(rope, "cos_sin_cache", None)
            if (name not in (_ROPE_1D, _ROPE_MULTI)
                    or not isinstance(cache, torch.Tensor)
                    or cache.dtype != torch.bfloat16
                    or not cache.is_contiguous()
                    or cache.dim() != 2 or cache.size(1) != hd
                    or getattr(rope, "head_dim", None) != hd):
                return None
            if name == _ROPE_MULTI and positions.dim() == 2:
                sec = rope.mrope_section
                if len(sec) != 3 or sum(sec) != hd // 2:
                    return None
                if rope.mrope_interleaved:
                    mode, s0, s1 = 3, 3 * int(sec[1]), 3 * int(sec[2])
                else:
                    mode, s0, s1 = 2, int(sec[0]), int(sec[0]) + int(sec[1])
            else:
                if name == _ROPE_1D and not rope.is_neox_style:
                    return None
                if positions.dim() != 1:
                    return None
                mode = 1
            if positions.dtype != torch.int64:
                return None
        elif self.q_norm is None:
            return _NOOP
        return (qw, kw, eps, cache, mode, s0, s1)

    def forward(self, positions, hidden_states, rotary_emb=None):
        N = hidden_states.shape[0]
        if self._qkv is None:
            self._setup()
        qkv = self._qkv(hidden_states)

        rope = rotary_emb if rotary_emb is not None else self.rotary_emb
        if rope is not None and self._rope_cache_ready != id(rope):
            self._prime_rope(rope, hidden_states.dtype)

        key = (id(rope), positions.dim())
        if self._plan_key != key:
            self._plan_key = key
            self._plan = self._build_plan(rope, positions, qkv)
        plan = self._plan

        if plan is not None:
            if plan is not _NOOP:
                qw, kw, eps, cache, mode, s0, s1 = plan
                _C.qk_norm_rope(qkv, qw, kw, eps, cache,
                                positions if mode else _EMPTY,
                                self.num_heads, self.num_kv_heads,
                                self.head_dim, s0, s1, mode)
            q, k, v = qkv.split(self._split, dim=-1)
            return self._o(self._attn(q, k, v, N))

        q, k, v = qkv.split(self._split, dim=-1)

        # Learnable QK norm (Qwen3: before RoPE)
        if self.q_norm is not None:
            # Normalise per head through a *view*, matching vLLM's
            # Qwen3Attention.forward.  Reducing over the last dim of the 3-D
            # view is bit-identical to reshaping to (N*heads, head_dim) and
            # avoids materialising a copy of the whole Q slice.
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

        return self._o(self._attn(q, k, v, N))
