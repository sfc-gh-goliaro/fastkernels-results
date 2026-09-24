"""Kimi MLA attention — a short dense-prefill pipeline, delegating everything else.

The baseline routes every configuration through ``MLAAttention``, whose dispatch
covers paged decode, sparse (DSA) attention, chunked context, mixed batches and
three KV-cache dtypes. Kimi's full-attention layers reach almost none of that in
the configuration this operator is exercised in: one causal segment, no paged KV
cache, no chunked context, no fp8, ``tp_size == 1``. For that configuration the
graph reduces to two input projections, a normalization of the latent, an
up-projection, a concatenation, attention and an output projection.

This module keeps the baseline's class surface exactly — same submodule names, so
weight sharing by ``state_dict`` key works; same ``compute_absorbed_weights`` so
the decode-time absorbed weights and any decoder-level fusion keep working — and
adds one guarded route for the dense-prefill case. Every other configuration is
handed to the same ``MLAAttention`` the baseline uses, through a body that is a
literal transcription of the baseline's ``forward``. Generality is not traded for
the fast path; it is kept in the branch that owns it.

What the dense-prefill route does differently:

* **One input projection instead of two.** ``q_proj`` and ``kv_a_proj_with_mqa``
  read the same ``hidden_states`` and differ only in their output rows, so their
  weights concatenate into one ``[6720, 2304]`` matrix. ``hidden_states`` is read
  once and one launch disappears. ``q`` is then a strided view of the result and
  is never copied.

* **The latent is normalized where it lies.** Columns ``[6144, 6656)`` of the
  fused output are the latent; ``[6656, 6720)`` are the rope part. Normalizing
  the latent *in place* leaves ``[normed latent | k_pe]`` already adjacent, which
  is exactly the input the packed up-projection wants — no second buffer, no copy
  of the rope part, one launch. That is what ``mla_inplace_norm.cu`` is for: the
  frozen ``L1`` normalization reads a strided view happily but allocates its
  output, which cannot produce that adjacency.

* **One GEMM emits the final ``[k_nope | k_pe | v]`` layout.** ``kv_b_proj``
  produces ``[k_nope | v]`` per head and the baseline then builds
  ``k = [k_nope | k_pe]`` into a fresh buffer with two slice copies. Widening the
  weight to ``[10240, 576]`` — the original rows, plus a 64x64 identity block
  that carries ``k_pe`` through the multiply — makes the up-projection emit the
  packed layout directly. ``k`` and ``v`` come back as last-dim-contiguous views
  with head stride 320, which is all FlashAttention-4 asks of them. The
  concatenation and its buffer are gone.

  The identity block is not free: it is 64 zero rows of ``K``, so the GEMM does
  about 41% more multiply-accumulates than the original ``kv_b`` GEMM and writes
  25% more output. Whether that pays where arithmetic rather than launches decides
  the latency was too close to predict, so it was measured rather than argued:
  ``tests/ab_packed.py`` times this route against the ``[k_nope | v]`` plus
  concatenation route in one process, paired and order-balanced, and the packed
  route won every one of six rounds at every token count from 26 to 16384. Per
  kernel the packing costs about 6 us of GEMM time at N = 16384 and removes about
  349 us of copying. There is therefore no token-count threshold here: one route
  serves every shape, and the alternative it replaced lives in that test rather
  than as a branch nothing reaches.

* **A single prefill token needs no attention kernel, and no separate value
  projection either.** One query token, one key token and no cached context give
  the causal softmax a single logit, so
  ``exp(s - s) / 1 == 1`` and the attention output *is* ``v``. Under that
  condition the query projection, the key half of the up-projection and the
  attention call are all dead. What is left is the latent projection, its
  normalization, and one GEMM: the value half of the up-projection composes with
  ``o_proj`` into a single ``[2304, 512]`` matrix, so the route ends in three
  device operations against the general route's five. This is the first token of a
  fresh sequence — a real serving case, and the guards are written as the semantic
  conditions that make the identity hold rather than as a token count.

* **Attention reaches FlashAttention-4's dense entry.** The frozen ``L1``
  ``FlashAttnVarlen`` re-routes a single-segment call to FA4's dense entry point,
  where FA4 enables two-SM cooperative MMA and a better tile scheduler. The
  baseline's ``MLAAttention`` imports its ``FlashAttnVarlen`` from the baseline
  tree, so it does not get that routing; issuing the call from here does.

Derived weights are built on first use, never in ``__init__``: the harness casts
parameters and shares weights *after* construction, so anything derived in the
constructor would capture pre-load garbage. They are plain attributes rather than
parameters or buffers, so they stay out of ``state_dict`` and cannot perturb
weight sharing, and a ``load_state_dict`` post-hook drops them so a later load
rebuilds. There is deliberately no per-call validity check: ``load_state_dict``
copies into existing storage, so the source ``data_ptr`` does *not* change and a
pointer comparison would be both unsound as an invalidation signal and a per-call
cost on exactly the shapes that cannot afford one.

No guard on the fast path reads device memory. Reading a ``cu_seqlens`` element
would settle several questions cheaply and is not done: it would put a
device-to-host synchronization in a path whose latency is otherwise dominated by
dispatch, and would make that latency depend on input values.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ....infra.context import get_context
from ....infra.cuda_ext import load_op
from ....infra.tp import _tp_size
from ..L1.flash_attn_varlen import FlashAttnVarlen
from ..L1.rms_norm import RMSNorm
from .mla_attention_impl import MLAAttention
from .parallel_linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)

# Built eagerly at import, not on first use: the import happens outside the
# harness's per-case error handling and outside its timing window, so a compile
# failure surfaces here as a traceback rather than as a runtime error on the
# first forward, and no build can land inside a timed call. The extension name
# has to be unique across the process -- torch keys both the build directory and
# the pybind module on the name alone.
_NORM_EXT = load_op("kimi_mla_inplace_norm", "mla_inplace_norm.cu")

# Bound at import. On four of the five benchmarked shapes the device is starved
# and the cost is host-side, so each attribute lookup avoided on the way to the
# kernel is a real fraction of it. Calling the pybind symbol directly also keeps
# the PyTorch dispatcher out of the path, which is the same reason FlashInfer
# moved its norm entry points off ``torch.library``.
_rms_norm_inplace = _NORM_EXT.rms_norm_inplace


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

        # Attention is issued from here rather than through ``self.attn`` so the
        # call reaches the frozen L1 winner's dense re-entry. Parameterless, so it
        # adds nothing to ``state_dict``.
        self.varlen_attn = FlashAttnVarlen()

        # Column offsets into the fused projection output, and the widths the
        # fast route reshapes by. Computed here so the route itself does no
        # arithmetic: on four of the five benchmarked shapes the device is
        # starved and host-side work inside the call is the latency.
        self._q_width = self.num_local_heads * self.qk_head_dim
        self._latent_end = self._q_width + self.kv_lora_rank
        self._fused_width = self._latent_end + self.qk_rope_head_dim
        self._o_width = self.num_local_heads * self.v_head_dim
        self._kv_width = self.qk_nope_head_dim + self.v_head_dim
        self._packed_head_width = self.qk_head_dim + self.v_head_dim
        self._norm_eps = self.kv_a_layernorm.eps

        # Everything about the fast route that a constructed module already
        # settles: dtypes, biases, ranks and the TP degree cannot change later.
        linears = (self.q_proj, self.kv_a_proj_with_mqa, self.kv_b_proj,
                   self.o_proj)
        self._dense_route = (
            self.q_lora_rank is None
            and not any(getattr(lin, "use_fp8", False) for lin in linears)
            and all(getattr(lin, "bias", None) is None for lin in linears)
            and self.o_proj.tp_size == 1
            and self.num_local_heads == self.num_heads
            and not self.attn.is_sparse
            and not self.attn.use_flashinfer_sparse
        )

        # Derived weights, built on first use. A plain attribute: ``nn.Module``
        # only intercepts parameters, buffers and submodules, so this stays out of
        # ``state_dict``.
        self._derived = None
        self.register_load_state_dict_post_hook(_drop_derived)

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

    # -- Derived weights -----------------------------------------------------

    def _apply(self, *args, **kwargs):
        """Drop the derived weights when the module is moved or recast.

        ``.to()``, ``.cuda()`` and ``.half()`` all funnel through here. Without
        this, a module moved or recast after its first forward would keep derived
        tensors on the old device or in the old dtype -- which the benchmark never
        does, but a drop-in replacement has to survive.

        A caller who mutates a parameter's storage in place, without
        ``load_state_dict`` and without ``_apply``, still gets stale derived
        weights. That is not detectable without a per-call check, which would cost
        Python time on every call for a case no correct caller reaches; it is the
        same limitation every weight-packing scheme in this codebase carries.
        """
        self._derived = None
        return super()._apply(*args, **kwargs)

    def _build_derived(self) -> dict:
        """Weights the fast route needs, in the module's own dtype and device.

        Called once, on the first fast-route forward. Not in ``__init__``: the
        harness casts parameters to the benchmark dtype and copies the reference
        weights in only after construction, so a constructor-time build would
        capture uninitialized values.
        """
        heads = self.num_local_heads
        nope, value = self.qk_nope_head_dim, self.v_head_dim
        latent = self.kv_lora_rank

        # One projection for ``q`` and the latent+rope block: same input, disjoint
        # output rows. ``q`` takes the leading rows so it stays a leading column
        # slice of the output and the latent block stays contiguous per row.
        qkv_a = torch.cat(
            (self.q_proj.weight.data, self.kv_a_proj_with_mqa.weight.data),
            dim=0,
        )

        kvb = self.kv_b_proj.weight.data.view(heads, nope + value, latent)
        rope = self.qk_rope_head_dim
        qk = self.qk_head_dim
        dtype, device = qkv_a.dtype, qkv_a.device

        # An up-projection that emits ``[k_nope | k_pe | v]`` per head directly.
        # Its input is ``[normed latent | k_pe]``, so the rope columns of the
        # input have to reach the rope columns of the output untouched: a 64x64
        # identity block does that inside the multiply.
        #
        # For finite inputs the passthrough is exact, not approximate: the sum
        # over the 576 terms of a rope row is one exact ``1.0 * k_pe`` product
        # plus 512 exact fp32 zeros, which no reassociation can perturb. It is
        # *not* exact for a non-finite latent, because a dense GEMM evaluates the
        # zero-weight products and ``0 * inf`` is NaN -- so an infinity in the
        # latent would contaminate the rope columns here where the baseline's
        # concatenation would have copied ``k_pe`` through untouched, and an
        # infinity in ``k_pe`` would contaminate ``k_nope`` and ``v``
        # symmetrically. Reaching that needs a non-finite latent, which needs
        # non-finite weights or activations; it is a real behavioural difference
        # from the baseline under those inputs and not a claim this makes.
        packed = torch.zeros(heads, qk + value, latent + rope,
                             dtype=dtype, device=device)
        packed[:, :nope, :latent] = kvb[:, :nope, :]
        packed[:, nope:qk, latent:] = torch.eye(rope, dtype=dtype, device=device)
        packed[:, qk:, :latent] = kvb[:, nope:, :]

        # The single-token route needs only ``v``, and then only to feed
        # ``o_proj`` -- so the two compose into one ``[hidden, latent]`` matrix and
        # the route ends in a single GEMM. Composed in fp32 and rounded once,
        # because the product of two bf16 matrices is not representable in bf16 and
        # the accumulation that builds it should not throw away more than that
        # final rounding.
        #
        # This re-associates a 4096-term sum into a 512-term one and skips a bf16
        # rounding of ``v``, so it is a numerics change, not a pure refactor. It
        # ships because it was measured, not because it looks safe: 100% of
        # elements inside the harness bound over two independent weight draws and
        # the harness's three correctness seeds, and 1.39x faster at one token over
        # six paired rounds. See ``tests/experiments.py``.
        v_only = kvb[:, nope:, :].reshape(heads * value, latent)
        o_v_fused = (self.o_proj.weight.data.float() @ v_only.float()).to(dtype)

        return {
            "qkv_a": qkv_a,
            "kvb_packed": packed.reshape(heads * (qk + value), latent + rope),
            "o_v_fused": o_v_fused,
        }

    # -- Forward -------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        positions: torch.Tensor | None = None,
        state_manager=None,
    ) -> torch.Tensor:
        del positions, state_manager

        if self._dense_route and hidden_states.dim() == 2:
            ctx = get_context()
            # Every term reads a Python attribute, a shape, a ``dim()`` or a
            # ``numel()``. None reads device memory, so none of this synchronizes.
            #
            # Two cumulative-length entries on both sides is what "one causal
            # segment" means in the packed varlen layout, and it is the
            # configuration this route is scoped to. A multi-segment context is
            # delegated even though the route computes it correctly -- every step
            # before attention is per-token and independent of segmentation, and
            # the frozen L1 attention op routes a multi-segment call to its own
            # varlen fallback. Correctness is not the reason for the boundary;
            # keeping the optimized route to the configuration it was designed,
            # measured and profiled against is.
            if (ctx.is_prefill
                    and not ctx.is_mixed
                    and ctx.chunked_context is None
                    and ctx.cu_seqlens_q is not None
                    and ctx.cu_seqlens_k is not None
                    and ctx.cu_seqlens_q.dim() == 1
                    and ctx.cu_seqlens_k.dim() == 1
                    and ctx.cu_seqlens_q.numel() == 2
                    and ctx.cu_seqlens_k.numel() == 2
                    and self.attn.k_cache.numel() == 0):
                return self._forward_dense_prefill(hidden_states, ctx)

        return self._forward_general(hidden_states)

    def _forward_general(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """The baseline's own sequence, for every configuration the fast route
        does not claim: paged decode, sparse attention, chunked context, mixed
        batches, multi-segment prefill, fp8 linears, a TP degree above one.

        ``self.attn`` expects an already-normalized latent, so the normalization
        happens here and not inside it.
        """
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

    def _forward_dense_prefill(self, hidden_states, ctx) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]
        heads = self.num_local_heads
        nope = self.qk_nope_head_dim
        derived = self._derived
        if derived is None:
            derived = self._derived = self._build_derived()

        # One query token attending to one key token with nothing cached leaves
        # the causal softmax a single logit, so ``exp(s - s) / 1`` is exactly one
        # and the attention output is ``v``. That makes the query projection, the
        # key half of the up-projection and the attention kernel all dead. Every
        # term of the test is a shape, a ``numel()`` or a host-side integer that
        # the context carries, so establishing it costs no synchronization.
        #
        # Two qualifications, both of which the guards cannot state because
        # stating them would need a device read:
        #
        # * The identity needs a finite score. ``s - s`` is zero for finite ``s``
        #   and NaN for an infinite one, so a query-key product that overflows
        #   bf16's range would make the attention kernel produce NaN where this
        #   returns ``v``. That needs non-finite activations.
        # * It needs well-formed packed cumulative lengths, i.e.
        #   ``cu_seqlens[0] == 0`` and ``cu_seqlens[-1] == num_tokens``. Reading
        #   either value would put a device-to-host synchronization in the timed
        #   path, so this relies on the same packed-layout invariant the frozen
        #   ``L1`` attention op documents as a precondition -- a caller passing
        #   ``[0, 0]`` with one token already gets an unwritten output row from
        #   the varlen path, so neither implementation is defined there.
        # The single-segment condition the identity also needs is already
        # established by the route's own guard, so only the length terms are left.
        if (num_tokens == 1
                and ctx.max_seqlen_q <= 1
                and ctx.max_seqlen_k <= 1):
            return self._forward_single_token(hidden_states, derived)

        # ``q`` and the latent+rope block come out of one GEMM: same input rows,
        # disjoint output columns. ``q`` stays a strided view of that output --
        # row stride 6720 rather than 6144 -- which FlashAttention-4 accepts
        # because its last dimension is contiguous.
        fused = F.linear(hidden_states, derived["qkv_a"])
        q = fused[:, :self._q_width].view(num_tokens, heads, self.qk_head_dim)

        # Normalized where it lies, so ``[normed latent | k_pe]`` is already
        # adjacent and the up-projection's input is a plain slice of ``fused``.
        # A layout the kernel declines is reported, not guessed at; the
        # out-of-place normalization then writes back through the same view.
        latent = fused[:, self._q_width:self._latent_end]
        norm_weight = self.kv_a_layernorm.weight
        if not _rms_norm_inplace(latent, norm_weight, self._norm_eps):
            latent.copy_(self.kv_a_layernorm(latent))

        # One GEMM for the whole of ``[k_nope | k_pe | v]``. ``k`` and ``v`` come
        # back as views with head stride 320 and a contiguous last dimension,
        # which is all FlashAttention-4 requires of them, so neither is copied.
        kv = F.linear(fused[:, self._q_width:self._fused_width],
                      derived["kvb_packed"])
        kv = kv.view(num_tokens, heads, self._packed_head_width)

        out = self.varlen_attn(
            q, kv[..., :self.qk_head_dim], kv[..., self.qk_head_dim:],
            cu_seqlens_q=ctx.cu_seqlens_q,
            cu_seqlens_k=ctx.cu_seqlens_q,
            max_seqlen_q=ctx.max_seqlen_q,
            max_seqlen_k=ctx.max_seqlen_q,
            softmax_scale=self.scaling,
            causal=True,
        )
        return F.linear(out.reshape(num_tokens, self._o_width),
                        self.o_proj.weight)

    def _forward_single_token(self, hidden_states, derived) -> torch.Tensor:
        """The first token of a fresh sequence: attention is the identity on ``v``.

        Three operations. The query projection and the key half of the
        up-projection are never touched, and the value half is already folded into
        ``o_proj``, so this reads 22 MB of weight where the general route reads 58
        and issues three device operations where it issues five.

        The attention identity itself is exact. The route as a whole is not
        bit-identical to the general one: the folded projection re-associates the
        sum and rounds once where the two-GEMM form rounded twice.
        """
        kv = F.linear(hidden_states, self.kv_a_proj_with_mqa.weight)
        latent = kv[:, :self.kv_lora_rank]
        if not _rms_norm_inplace(latent, self.kv_a_layernorm.weight,
                                 self._norm_eps):
            latent.copy_(self.kv_a_layernorm(latent))
        return F.linear(latent, derived["o_v_fused"])


def _drop_derived(module: KimiMLAAttention, incompatible_keys) -> None:
    """Invalidate the derived weights after a ``load_state_dict``.

    This is the only invalidation signal there is. ``load_state_dict`` copies into
    the existing parameter storage, so the source tensors' ``data_ptr`` values are
    unchanged by it and a pointer comparison would never fire.
    """
    del incompatible_keys
    module._derived = None
