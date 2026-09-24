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

Nothing in this module is a new kernel.  Every hot spot the profile attributes to
this layer lives one level down, and the faster kernels for all of them are
already on the candidate path -- so the work here is *reaching* them from a
module that stays interchangeable with the baseline's contract, module tree,
``state_dict`` and numerics.  ``RMSNorm`` arrives by import alone: the relative
import below resolves to ``candidate/L1/rms_norm.py``, whose kernel reads each
row once and normalizes a strided view where it lies instead of forcing a copy.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from torch.nn.modules.module import (
    _global_backward_hooks,
    _global_backward_pre_hooks,
    _global_forward_hooks,
    _global_forward_pre_hooks,
)

from ....infra.context import get_context
from ....infra.tp import _tp_size
from .parallel_linear import QKVParallelLinear, RowParallelLinear
from .attention_impl import Attention
from ..L1.rms_norm import RMSNorm
from ..L1.fp8_linear import Fp8Linear
from ..L1.mrope import MRotaryEmbedding
from ..L1.rotary_emb import RotaryEmbedding
from ...baseline.L1.mrope import MRotaryEmbedding as BaselineMRotaryEmbedding
from ...baseline.L1.rotary_emb import RotaryEmbedding as BaselineRotaryEmbedding


# Shapes the throwaway table a stand-in is constructed with and never reads.
_THROWAWAY_ROPE_THETA = 10000.0


def _rope_bridge_for(rotary: nn.Module | None) -> nn.Module | None:
    """A frozen rotary module able to stand in for *rotary*, or ``None``.

    The engine hands this layer a rotary module it built itself, and the module
    it builds re-materializes ``cos_sin_cache.to(query.dtype)`` on every call --
    for Qwen3-VL that is a 512 MiB float32 table converted and discarded per
    forward, ~130 us regardless of token count.  The faster modules are already
    on the candidate path but the layer does not get to choose which one it is
    given, so it builds a stand-in of the same rotary shape and drives that
    instead, leaving ``self.rotary_emb`` exactly as it was handed over.

    The allowlist is keyed on the *exact* type, never ``isinstance``.  A subclass
    is free to override ``forward`` and rotate by some other law entirely, and
    nothing about the class tells the layer whether it did -- so admitting
    subclasses would mean silently applying the base rotation to an unknown
    number of them.  (Some are in fact safe: the Gemma4 proportional variant only
    builds a differently-shaped table and inherits ``forward``, so transplanting
    that table would preserve its frequency law.  It stays out anyway, because the
    rule has to hold for subclasses that do not exist yet.)  A module that already
    is one of the frozen classes is left alone for the same reason rather than
    wrapped twice.

    Only ``head_dim``, ``is_neox_style``, ``mrope_section`` and
    ``mrope_interleaved`` are read off the incoming module: those are what change
    the stand-in's runtime behaviour, and they are the only ones retained as
    attributes.  ``rope_theta`` and the Llama rope-scaling parameters are *not*
    retained anywhere, and are not needed -- they only ever shaped the table, and
    the live table is transplanted rather than rebuilt.  Hence
    ``max_position_embeddings=1``: the constructor's own table is a throwaway that
    is replaced on the first call.

    Those fields are *copied* here, and the module they came from reads its own
    live on every call, so ``_bridge_still_matches`` re-checks them before the
    stand-in is used.
    """
    if rotary is None:
        return None
    kind = type(rotary)
    if kind is BaselineMRotaryEmbedding and kind is not MRotaryEmbedding:
        return MRotaryEmbedding(
            head_dim=rotary.head_dim,
            max_position_embeddings=1,
            rope_theta=_THROWAWAY_ROPE_THETA,
            mrope_section=rotary.mrope_section,
            mrope_interleaved=rotary.mrope_interleaved,
        )
    if kind is BaselineRotaryEmbedding and kind is not RotaryEmbedding:
        return RotaryEmbedding(
            head_dim=rotary.head_dim,
            max_position_embeddings=1,
            rope_theta=_THROWAWAY_ROPE_THETA,
            is_neox_style=rotary.is_neox_style,
        )
    return None


def _bridge_still_matches(bridge: nn.Module, rotary: nn.Module) -> bool:
    """Whether *bridge* still describes the same rotation as *rotary*.

    ``_rope_bridge_for`` copies the fields that shape the rotation; the module it
    stands in for reads its own live on every call.  A caller that rewires
    ``mrope_section`` or flips ``is_neox_style`` after construction would
    otherwise keep getting the rotation those fields described at ``__init__``.
    Compared rather than re-derived so nothing is allocated on the hot path.

    ``mrope_section`` is compared by identity because the stand-in was handed the
    very same list: an in-place edit is already shared with it -- the kernel reads
    the three section sizes per launch -- so only a rebind needs catching.
    """
    if bridge.head_dim != rotary.head_dim:
        return False
    if type(bridge) is MRotaryEmbedding:
        return (bridge.mrope_section is rotary.mrope_section
                and bridge.mrope_interleaved == rotary.mrope_interleaved)
    return bridge.is_neox_style == rotary.is_neox_style


def _would_only_call_forward(module: nn.Module) -> bool:
    """Whether calling *module* runs nothing besides its class's ``forward``.

    Mirrors the predicate ``nn.Module._call_impl`` uses to take its own hook-free
    fast path, process-wide registries included, and adds the two things that sit
    in *front* of it: ``Module.compile()`` installs a ``_compiled_call_impl`` that
    ``_wrapped_call_impl`` checks before ``_call_impl`` at all, and an
    instance-level ``forward`` in ``__dict__`` shadows the class's.

    Every caller here uses this per call rather than once at construction.  Hooks
    and compilation both arrive after ``__init__`` in normal use -- an engine
    compiles the model, a profiler installs a global hook -- so a decision made at
    construction time would be exactly the wrong one.
    """
    return (
        module._compiled_call_impl is None
        and "forward" not in module.__dict__
        and not (
            module._backward_hooks or module._backward_pre_hooks
            or module._forward_hooks or module._forward_pre_hooks
            or _global_backward_pre_hooks or _global_backward_hooks
            or _global_forward_hooks or _global_forward_pre_hooks
        )
    )


# One instance, off every layer's module tree.  It owns no parameters or buffers,
# so registering it would add no state_dict keys -- but it would change what
# modules() walks, and the harness compares the candidate's module tree with the
# baseline's.  Sharing a single instance between the two projections is safe
# because the scratch it draws on is keyed by (K, device): the QKV projection's
# K and the output projection's K get separate arenas, and the two calls are
# ordered on one stream.
_FP8_GEMM = Fp8Linear()


def _frozen_fp8_applies(projection: nn.Module) -> bool:
    """Whether a *quantized* projection can be served by the frozen GEMM.

    Callers test ``use_fp8`` first, so this covers only the reasons a quantized
    projection still has to go through its own ``forward``: anything that would
    run besides that ``forward`` -- a hook, a compiled call, an instance-level
    override -- and a ``linear_op`` that an engine has handed pre-sized activation
    buffers for graph capture, since those are the buffers it expects written.
    """
    if not _would_only_call_forward(projection):
        return False
    op = projection.linear_op
    return op._a_buf is None and op._pf is None


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

        # Held off the module tree on purpose.  Registered as a submodule it
        # would be visited by ``Module._apply``, which replaces every module's
        # buffers independently -- so ``.to(device)`` would give it its own
        # full-size copy of the rotary table instead of a shared view, and a
        # 512 MiB table would be allocated twice.  Kept off the tree it is never
        # visited at all, which is why the table is transplanted per call rather
        # than once here: at construction time the real module's buffer is still
        # on the CPU, and a reference taken now would stay there.
        object.__setattr__(self, "_rope_bridge", _rope_bridge_for(rotary_emb))

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

    def _project_qkv(self, hidden_states, projection):
        """``qkv_proj``'s own contract, over the frozen GEMM where it applies."""
        if _frozen_fp8_applies(projection):
            # The bias goes in unconditionally.  QKVParallelLinear, unlike the
            # row-parallel class, does not gate its bias on tp_rank -- each rank
            # owns whole heads of the output, so each rank's bias slice is its
            # own.  Applying the row-parallel rule here would drop the bias on
            # every rank but 0 under TP > 1.
            return _FP8_GEMM(hidden_states, projection.weight,
                             projection.weight_scale_inv, projection.bias)
        return projection(hidden_states)

    def _project_out(self, attn_output, projection):
        """``o_proj``'s own contract, over the frozen GEMM where it applies."""
        if not _frozen_fp8_applies(projection):
            return projection(attn_output)
        # Row-parallel: every rank holds a shard of the input dim and produces a
        # partial sum, so the bias belongs to rank 0 alone and is added once,
        # before the all-reduce that completes the sum.
        bias = projection.bias if projection.tp_rank == 0 else None
        out = _FP8_GEMM(attn_output, projection.weight,
                        projection.weight_scale_inv, bias)
        if projection.reduce_results and projection.tp_size > 1:
            out = projection.allreduce(out)
        return out

    def _attend(self, q, k, v, N):
        """``Attention.forward``, or its dense-prefill body called directly.

        ``Attention`` is a dispatcher: it picks between a custom op and eager, a
        paged kernel and a dense one, FlashAttention and Triton and SDPA, prefill
        and decode and tree-verify, and rewrites its metadata for chunked local
        attention.  Every one of those decisions is reachable, and every one of
        them is settled by state this layer can read before the call.  When they
        all land on "dense unpaged prefill through ``prefill_op``" -- which is
        every shape this operator is scored on -- the dispatch itself is the only
        thing left to remove, so the body is called directly and the two N=1
        shapes stop paying for a chain of Python frames the GPU is waiting on.

        The allowlist is the *whole* set of decisions, not the interesting ones.
        Anything outside it delegates to ``self.attn(q, k, v)`` unchanged, which
        is also what makes this droppable: it removes no kernel.
        """
        attn = self.attn
        # Exact type: a subclass could override forward_impl, _forward_pure or
        # _group_block_tables, none of which the body below reads.
        if (type(attn) is Attention
                and attn.k_cache.numel() == 0 and attn.v_cache.numel() == 0
                and not attn._use_custom_op
                # ``forward_impl`` routes to the Triton or SDPA bodies rather
                # than ``_forward_pure`` whenever this is set.
                and not attn._triton_only
                # Chunked local attention rewrites cu_seqlens and the page table.
                and attn.attention_chunk_size is None
                and _would_only_call_forward(attn)):
            ctx = get_context()
            if (ctx.is_prefill and not ctx.is_mixed
                    and not ctx.is_tree_verify
                    # Not ``ctx.block_tables``: a sliding-window layer reads its
                    # own group's table out of ``ctx.sliding_block_tables``, so
                    # the raw field can be None while the layer is still paged.
                    and attn._group_block_tables(ctx) is None):
                # Read live rather than snapshotted at construction.  Not
                # because ``process_weights_after_loading`` rewrites them -- it
                # does not; it primes an fp32 sink copy on each trtllm *op*, and
                # must, since a trtllm layer still falls back to FlashAttention
                # for unpaged prefill and that build requires the model dtype.
                # Rather because ``_fa3_sinks`` is a plain attribute installed
                # with ``object.__setattr__``, so anything may rebind it, and the
                # live read costs one lookup.
                extra = {}
                if attn._fa3_sinks is not None:
                    extra["s_aux"] = attn._fa3_sinks
                if attn._fa3_window_size != (-1, -1):
                    extra["window_size"] = attn._fa3_window_size
                head_size = attn.head_size
                out = attn.prefill_op(
                    q.view(N, attn.num_heads, head_size),
                    k.view(N, attn.num_kv_heads, head_size),
                    v.view(N, attn.num_kv_heads, head_size),
                    cu_seqlens_q=ctx.cu_seqlens_q,
                    cu_seqlens_k=ctx.cu_seqlens_k,
                    max_seqlen_q=ctx.max_seqlen_q,
                    max_seqlen_k=ctx.max_seqlen_k,
                    softmax_scale=attn.scale, causal=True, **extra,
                )
                return out.reshape(N, attn.num_heads * head_size)
        return attn(q, k, v)

    def forward(self, positions, hidden_states, rotary_emb=None):
        N = hidden_states.shape[0]
        # A bf16 projection goes straight to its own submodule.  The routing only
        # ever applies to a quantized one, and on the launch-bound shapes -- where
        # the GPU finishes before the host has enqueued the next call -- one extra
        # Python frame per projection is a measurable fraction of the latency.
        qkv_proj = self.qkv_proj
        qkv = (self._project_qkv(hidden_states, qkv_proj) if qkv_proj.use_fp8
               else qkv_proj(hidden_states))
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)

        # Learnable QK norm (Qwen3: before RoPE)
        if self.q_norm is not None:
            # Normalise per head through a *view*, matching vLLM's
            # Qwen3Attention.forward:
            #     q_by_head = q.view(*q.shape[:-1], -1, head_dim)
            #     q = self.q_norm(q_by_head).view(q.shape)
            # Reshaping to (N*heads, head_dim) first cannot be a view of a qkv
            # slice -- the slice's row stride is the packed qkv width
            # (q_size + 2*kv_size), not num_heads*head_dim -- so ``reshape``
            # would materialise a full copy of q and k on every layer.  The
            # 3-D view is also what lets the norm kernel claim this layout:
            # it collapses the leading dims to two (size, stride) pairs and
            # normalizes in place of a copy, so the call form is load-bearing
            # here and not just a transcription detail.
            q_shape, k_shape = q.shape, k.shape
            q = self.q_norm(
                q.view(N, self.num_heads, self.head_dim)).view(q_shape)
            k = self.k_norm(
                k.view(N, self.num_kv_heads, self.head_dim)).view(k_shape)

        rope = rotary_emb if rotary_emb is not None else self.rotary_emb
        if not self.nope and rope is not None:
            bridge = self._rope_bridge
            if (bridge is not None and rope is self.rotary_emb
                    and _bridge_still_matches(bridge, rope)
                    and _would_only_call_forward(rope)):
                # Point the stand-in at whichever table the real module holds
                # right now.  Identity, not equality: a rebind or an in-place
                # write to ``cos_sin_cache`` has to be picked up, and the frozen
                # modules key their derived state on the source buffer, so
                # handing back the same tensor is a memo hit and the dtype
                # conversion is paid once during warmup rather than per call.
                live = self.rotary_emb.cos_sin_cache
                if bridge.cos_sin_cache is not live:
                    bridge.cos_sin_cache = live
                q, k = bridge(positions, q, k)
            else:
                # Used as given.  Either a rotary was supplied per call -- the
                # stand-in was built for the one from ``__init__`` and says
                # nothing about this one -- or the module has since been rewired,
                # compiled, or had hooks installed, and only calling it does what
                # it now means.
                q, k = rope(positions, q, k)

        # Weight-less QK norm (Llama 4: after RoPE, only on RoPE layers)
        if self.q_wl_norm is not None:
            q = self.q_wl_norm(q.view(-1, self.head_dim)).view(N, -1)
            k = self.k_wl_norm(k.view(-1, self.head_dim)).view(N, -1)

        # Temperature tuning (Llama 4: only on NoPE layers)
        if self.attn_temperature_tuning:
            q = (q * self._get_attn_scale(positions)).to(q.dtype)

        attn_output = self._attend(q, k, v, N)
        o_proj = self.o_proj
        return (self._project_out(attn_output, o_proj) if o_proj.use_fp8
                else o_proj(attn_output))
