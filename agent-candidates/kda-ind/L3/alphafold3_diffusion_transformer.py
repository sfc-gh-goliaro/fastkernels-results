"""Diffusion transformer for AlphaFold3 (AF3 Algorithm 23), latency-optimized.

The operator is a stack of ``no_blocks`` blocks, each

    a = a + AttentionPairBias(a, z, s, mask)
    a = a + ConditionedTransitionBlock(a, s, mask)

Two instantiations occur in practice and take different paths: a token-level
self-attention stack (``n_query is None``) and an atom-level, sequence-local
cross-attention stack.  Neither is arithmetic-bound; both are dominated by CPU
dispatch and by sheer kernel count.  So this module

  * keeps the reference parameter tree verbatim, which makes ``state_dict()``
    keys, shapes and dtypes byte-identical to the reference implementation;
  * coalesces the per-block weights into batched tensors that the parameters
    themselves alias, so the batched form can never go stale and costs no extra
    memory;
  * hoists every block-invariant part of the ``s`` / ``z`` conditioning out of
    the block loop and evaluates it for all blocks in a handful of GEMMs;
  * replays the whole stack as a CUDA graph, which removes the dispatch cost
    that dominates the wall clock.

The unrestructured reference math is retained as ``_reference_stack`` and owns
every configuration, layout, flag, mask and rank the restructured executors do
not claim.  Set ``FK_AF3_DIFFUSION_NO_CUDA_GRAPH=1`` to force the eager path.
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..L1.layer_norm import LayerNorm
from ..L2.alphafold3_atom_attention import _get_block_key_indices
from ..L2.alphafold3_attention_pair_bias import (
    AttentionPairBias,
    CrossAttentionPairBias,
    _permute_final_dims,
)
from ..L2.alphafold3_swiglu_transition import ConditionedTransitionBlock


__targets__ = ["DiffusionTransformer"]


# Truthy to force the eager path and skip CUDA-graph capture.  A debugging aid:
# the eager path alone is correct, so a numerical question can be answered
# without graph capture in the way.
_NO_GRAPH_ENV = "FK_AF3_DIFFUSION_NO_CUDA_GRAPH"

# Upper bound on simultaneously retained graphs, so a caller sweeping shapes
# cannot grow device memory without limit.
_MAX_GRAPHS = 8

# The reference LayerNorm's eps.
_LN_EPS = 1e-5

# Which restructurings the executors apply.  All are on; a measurement script
# turns one off at a time to price it individually against the same shapes.
# Every "off" branch routes that group's work back through the reference
# submodules, so it is the unrestructured form by construction rather than a
# second implementation that could drift.  See tools/attribute.py.
RESTRUCTURINGS = {
    # Route through the restructured executors at all.  Off gives the
    # unrestructured stack, still graph-replayed: the measurement floor every
    # restructuring below is priced against.
    "use_restructured_executors": True,
    # Recompute the sequence-local key windows, block mask and mask bias once
    # per call rather than once per block.
    "hoist_window_invariants": True,
    # One batched GEMM over all AdaLNs for the normalized-s gate and shift.
    "batch_conditioning": True,
    # Issue that conditioning projection as a single batched GEMM instead of one
    # per AdaLN into the same output buffer.  The batched form picks a different
    # cuBLAS path than F.linear does and differs from it by about one bf16 ulp,
    # which the deep configuration's residual chain then amplifies; the per-AdaLN
    # form is bit-identical to the reference.  Priced in tools/attribute.py.
    "fuse_conditioning_gemm": False,
    # One GEMM for every block's two raw-s output gates.
    "batch_output_gates": True,
    # One batched GEMM for every block's pair bias.
    "batch_pair_bias": True,
    # q/k/v/gate (and the SwiGLU pair) merged where they share an input.
    "merge_column_parallel_gemms": True,
    # Sequence-local key/value projections over the real rows, window read after.
    "dedup_key_value_projections": True,
}

# Warmup iterations before capture: enough to settle any lazily allocated
# scratch and any lazily built cache in the code under capture, so that
# allocation does not happen inside the graph.
_CAPTURE_WARMUP = 3


def _graphs_disabled() -> bool:
    return bool(os.environ.get(_NO_GRAPH_ENV, ""))


def _layer_norm_plain(x: torch.Tensor, c: int) -> torch.Tensor:
    """LayerNorm with neither scale nor offset, reduced in fp32.

    The reference promotes the activation to fp32 before calling
    ``F.layer_norm``; for a low-precision input the ATen kernel already
    accumulates in fp32 and rounds once on store, so the explicit promotion only
    adds two elementwise passes.

    This is an empirical equality, not one the API promises: the two forms may
    select different kernels, and nothing guarantees the same reduction order.
    It is verified bit-exact for the shapes and backend in use, and
    ``tools/parity.py --blocks`` re-checks it whenever either changes.
    """
    return F.layer_norm(x, (c,), None, None, _LN_EPS)


def _normalize_fp32(x: torch.Tensor, c: int) -> torch.Tensor:
    """The fp32 half of a weight-only LayerNorm: normalize, do not scale.

    Scaling by the fp32 weight and rounding to the activation dtype is left to
    the caller, so one normalization can feed several differently-scaled
    consumers while preserving the reference's single rounding step.  This
    materializes the unscaled fp32 normalization that a fused affine LayerNorm
    would keep internal; the bf16 boundary is unmoved, and the equality is
    verified rather than promised -- see ``_layer_norm_plain``.
    """
    return F.layer_norm(x.float(), (c,), None, None, _LN_EPS)


def _split_heads(x: torch.Tensor, no_heads: int, c_hidden: int) -> torch.Tensor:
    """``[..., R, H * C] -> [..., H, R, C]``."""
    return x.view(*x.shape[:-1], no_heads, c_hidden).transpose(-2, -3)


def _alias_parameter(param: nn.Parameter, slot: torch.Tensor) -> None:
    """Copy *param* into *slot* and re-point the parameter's storage at it."""
    if tuple(param.shape) != tuple(slot.shape):
        raise RuntimeError(
            f"cannot alias a parameter of shape {tuple(param.shape)} onto a "
            f"slot of shape {tuple(slot.shape)}")
    with torch.no_grad():
        slot.copy_(param.detach())
    param.data = slot


class DiffusionTransformerBlock(nn.Module):
    """One AF3 Algorithm 23 block: attention with pair bias, then a transition.

    A parameter container plus the unrestructured reference math.  The submodule
    names are load-bearing -- they are what ``state_dict()`` keys are built from.

    Args:
        c_a: Token activation channel dimension
        c_s: Single activation channel dimension
        c_z: Pair activation channel dimension
        c_hidden: Per-head hidden dimension
        no_heads: Number of attention heads
        n_transition: Transition layer scale
        use_ada_layer_norm: Whether to use AdaLN-Zero
        n_query: Query block height for the sequence-local path
        n_key: Key block width for the sequence-local path
        inf: Large masking constant
    """

    def __init__(
        self,
        c_a: int,
        c_s: int,
        c_z: int,
        c_hidden: int,
        no_heads: int,
        n_transition: int,
        use_ada_layer_norm: bool = True,
        n_query: int | None = None,
        n_key: int | None = None,
        inf: float = 1e9,
    ):
        super().__init__()
        self.use_cross_attention = n_query is not None

        if not self.use_cross_attention:
            self.attention_pair_bias = AttentionPairBias(
                c_q=c_a, c_k=c_a, c_v=c_a,
                c_s=c_s, c_z=c_z,
                c_hidden=c_hidden,
                no_heads=no_heads,
                use_ada_layer_norm=use_ada_layer_norm,
                gating=True,
                inf=inf,
            )
        else:
            self.attention_pair_bias = CrossAttentionPairBias(
                c_q=c_a, c_k=c_a, c_v=c_a,
                c_s=c_s, c_z=c_z,
                c_hidden=c_hidden,
                no_heads=no_heads,
                use_ada_layer_norm=use_ada_layer_norm,
                n_query=n_query,
                n_key=n_key,
                gating=True,
                inf=inf,
            )

        self.conditioned_transition = ConditionedTransitionBlock(
            c_a=c_a, c_s=c_s, n=n_transition,
        )

    def forward(
        self,
        a: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor | None = None,
        use_deepspeed_evo_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_lma: bool = False,
        use_high_precision_attention: bool = False,
        _mask_trans: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            a:    [*, N, C_token] token-level embedding
            s:    [*, N, C_s] single embedding
            z:    [*, N, N, C_z] pair embedding
            mask: [*, N] mask

        Returns:
            [*, N, C_token] updated token embedding
        """
        a = a + self.attention_pair_bias(a=a, z=z, s=s, mask=mask)

        trans_mask = mask if _mask_trans else None
        a = a + self.conditioned_transition(a=a, s=s, mask=trans_mask)

        return a


class _StackedWeights:
    """Per-block weights coalesced into batched tensors that the parameters alias.

    Each batched tensor is allocated once, every parameter is copied into its
    slice, and then the ``Parameter``'s storage is re-pointed at that slice.
    Parameters and batched tensors therefore *share memory*, which means the
    batched form cannot go stale behind an in-place weight update,
    ``state_dict()`` still yields tensors of the reference's shape and dtype, and
    total parameter memory does not grow.

    Nothing here is a ``Parameter`` or a registered buffer, so nothing here adds
    a ``state_dict()`` key.  Construction happens on the first forward and never
    in ``__init__``, because callers cast and load weights in between.
    """

    def __init__(self, root: "DiffusionTransformer"):
        blocks = root.blocks
        probe = next(root.parameters())
        dtype, device = probe.dtype, probe.device
        nb = len(blocks)
        c_a, c_s, c_z = root.c_a, root.c_s, root.c_z
        c_qkv, n_t = root.c_qkv, root.n_transition
        cross = root.use_cross_attention

        def stack(shape, params):
            buf = torch.empty(shape, dtype=dtype, device=device)
            for i, p in enumerate(params):
                _alias_parameter(p, buf[i])
            return buf

        def stack_pairs(rows, pairs):
            """Two row groups of *rows* rows per item, the lower one optional."""
            buf = torch.empty(len(pairs), 2 * rows, pairs[0][0].shape[-1],
                              dtype=dtype, device=device)
            for i, (upper, lower) in enumerate(pairs):
                _alias_parameter(upper, buf[i, :rows])
                if lower is None:
                    buf[i, rows:].zero_()
                else:
                    _alias_parameter(lower, buf[i, rows:])
            return buf

        def stack_bias(width, pairs):
            buf = torch.zeros(len(pairs), 2 * width, dtype=dtype, device=device)
            for i, (upper, lower) in enumerate(pairs):
                if upper is not None:
                    _alias_parameter(upper, buf[i, :width])
                if lower is not None:
                    _alias_parameter(lower, buf[i, width:])
            return buf

        attn = [b.attention_pair_bias for b in blocks]
        trans = [b.conditioned_transition for b in blocks]

        # Every AdaLN instance, in the order the block loop consumes them.  The
        # sequence-local path has separate query and key AdaLNs; the
        # self-attention path shares one between query and key.
        adalns = []
        for at, tr in zip(attn, trans):
            if cross:
                adalns += [at.layer_norm_a_q, at.layer_norm_a_k, tr.layer_norm]
            else:
                adalns += [at.layer_norm_a, tr.layer_norm]
        self.adaln_per_block = 3 if cross else 2

        # One conditioning-LayerNorm scale per AdaLN.  These are distinct
        # weights, which is why the batched conditioning GEMM has one item per
        # AdaLN rather than one per block.
        self.ln_s_w = stack((len(adalns), c_s),
                            [m.layer_norm_s.weight for m in adalns])
        # ``linear_g`` (gate, biased) stacked above ``linear_s`` (shift, unbiased).
        self.s_proj_w = stack_pairs(
            c_a, [(m.linear_g.weight, m.linear_s.weight) for m in adalns])
        self.s_proj_b = stack_bias(c_a, [(m.linear_g.bias, None) for m in adalns])

        # The two projections that read *raw* s -- the attention output gate and
        # the transition output gate.  Both are followed by a sigmoid.
        self.raw_w = stack_pairs(
            c_a, [(at.linear_ada_out.weight, tr.linear_g.weight)
                  for at, tr in zip(attn, trans)])
        self.raw_b = stack_bias(
            c_a, [(at.linear_ada_out.bias, tr.linear_g.bias)
                  for at, tr in zip(attn, trans)])

        self.z_w = stack((nb, root.no_heads, c_z),
                        [at.linear_z.weight for at in attn])
        # The sequence-local path normalizes the pair rep once at the top level;
        # the self-attention path carries a per-block scale.
        self.ln_z_w = (None if cross else
                       stack((nb, c_z), [at.layer_norm_z.weight for at in attn]))

        if cross:
            # Query-side and key-side projections read different AdaLN outputs
            # (and a different number of rows), so they merge in two pairs
            # rather than in one quadruple.
            self.qg_w = stack_pairs(
                c_qkv, [(at.mha.linear_q.weight, at.mha.linear_g.weight)
                        for at in attn])
            self.qg_b = stack_bias(c_qkv, [(at.mha.linear_q.bias, None)
                                           for at in attn])
            self.kv_w = stack_pairs(
                c_qkv, [(at.mha.linear_k.weight, at.mha.linear_v.weight)
                        for at in attn])
            self.qkvg_w = self.qkvg_b = None
        else:
            # One column-parallel GEMM: q, k, v and the head gate all read the
            # same AdaLN output.  ``c_qkv`` is ``no_heads * c_hidden``, which is
            # not in general ``c_a``.
            buf = torch.empty(nb, 4 * c_qkv, c_a, dtype=dtype, device=device)
            for i, at in enumerate(attn):
                for j, weight in enumerate((at.mha.linear_q.weight,
                                            at.mha.linear_k.weight,
                                            at.mha.linear_v.weight,
                                            at.mha.linear_g.weight)):
                    _alias_parameter(weight, buf[i, j * c_qkv:(j + 1) * c_qkv])
            self.qkvg_w = buf
            # Only the query projection is biased, so the merged bias vector is
            # the query bias followed by zeros.
            bias = torch.zeros(nb, 4 * c_qkv, dtype=dtype, device=device)
            for i, at in enumerate(attn):
                _alias_parameter(at.mha.linear_q.bias, bias[i, :c_qkv])
            self.qkvg_b = bias
            self.qg_w = self.qg_b = self.kv_w = None

        self.o_w = stack((nb, c_a, c_qkv), [at.mha.linear_o.weight for at in attn])
        self.swiglu_w = stack_pairs(
            n_t * c_a, [(tr.swiglu.linear_a.weight, tr.swiglu.linear_b.weight)
                        for tr in trans])
        self.out_w = stack((nb, c_a, n_t * c_a),
                           [tr.linear_out.weight for tr in trans])

        # Everything above aliases a parameter.  Deliberately absent: cached fp32
        # copies of the LayerNorm scales.  A copy does not alias, so an in-place
        # change to a scale would not reach it, and the fp32 cast is two tiny
        # kernels per call over a few thousand elements -- far cheaper than a
        # staleness window.
        #
        # Views the batched GEMMs consume.  Views inherit the aliasing above.
        self.s_proj_wt = self.s_proj_w.transpose(1, 2)
        self.s_proj_b3 = self.s_proj_b.unsqueeze(1)
        self.raw_w_flat = self.raw_w.reshape(-1, c_s)
        self.raw_b_flat = self.raw_b.reshape(-1)
        self.z_w_flat = self.z_w.reshape(-1, c_z)
        self.z_wt = self.z_w.transpose(1, 2)


class _GraphRunner:
    """CUDA-graph capture and replay for a whole-stack evaluation.

    Every call copies the caller's *current* inputs into per-signature static
    buffers before replaying, so a caller that hands a different address and
    different contents on each call still gets a per-call-correct result.  The
    result is cloned out of the graph's memory, so it neither aliases that
    memory nor is overwritten by the next call.  Both the copies and the clone
    are inside the region a caller would time.

    Capture is guarded: a signature whose capture fails is remembered and runs
    eagerly from then on.

    One set of static buffers per signature means one consumer at a time, as for
    any graph-replay wrapper: two calls with the same signature running
    concurrently on different streams would overwrite each other's buffers.
    """

    def __init__(self, max_graphs: int = _MAX_GRAPHS):
        self._graphs: dict = {}
        self._uncapturable: set = set()
        self._max_graphs = max_graphs

    def run(self, fn, key, inputs):
        entry = self._graphs.get(key)
        if entry is None:
            if key in self._uncapturable:
                return fn(*inputs)
            entry = self._capture(fn, inputs)
            if entry is None:
                self._uncapturable.add(key)
                return fn(*inputs)
            if len(self._graphs) >= self._max_graphs:
                self._graphs.pop(next(iter(self._graphs)))
            self._graphs[key] = entry
        static_in, graph, static_out = entry
        for static, live in zip(static_in, inputs):
            if static is not None:
                static.copy_(live)
        graph.replay()
        return static_out.clone()

    def _capture(self, fn, inputs):
        static_in = None
        try:
            static_in = tuple(None if t is None else t.clone() for t in inputs)
            # Warm up off the capturing stream so any lazily allocated scratch
            # comes from the ordinary allocator rather than the graph pool.
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(_CAPTURE_WARMUP):
                    fn(*static_in)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                static_out = fn(*static_in)
        except Exception:  # noqa: BLE001 - any capture failure falls back to eager
            torch.cuda.synchronize()
            return None
        if not isinstance(static_out, torch.Tensor):
            return None
        return static_in, graph, static_out

    def reset(self) -> None:
        self._graphs.clear()
        self._uncapturable.clear()


class DiffusionTransformer(nn.Module):
    """AF3 Algorithm 23: diffusion transformer stack.

    Args:
        c_a: Token activation channel dimension
        c_s: Single activation channel dimension
        c_z: Pair activation channel dimension
        c_hidden: Per-head hidden dimension
        no_heads: Number of attention heads
        no_blocks: Number of transformer blocks
        n_transition: Transition layer scale
        use_ada_layer_norm: Whether to use AdaLN-Zero
        n_query: Query block height for the sequence-local path
        n_key: Key block width for the sequence-local path
        inf: Large masking constant
    """

    def __init__(
        self,
        c_a: int,
        c_s: int,
        c_z: int,
        c_hidden: int,
        no_heads: int,
        no_blocks: int,
        n_transition: int,
        use_ada_layer_norm: bool = True,
        n_query: int | None = None,
        n_key: int | None = None,
        inf: float = 1e9,
        blocks_per_ckpt: int | None = None,
        **kwargs,
    ):
        super().__init__()

        self.use_cross_attention = n_query is not None
        if self.use_cross_attention:
            self.layer_norm_z = LayerNorm(c_z, create_offset=False)

        self.blocks = nn.ModuleList([
            DiffusionTransformerBlock(
                c_a=c_a, c_s=c_s, c_z=c_z,
                c_hidden=c_hidden, no_heads=no_heads,
                n_transition=n_transition,
                use_ada_layer_norm=use_ada_layer_norm,
                n_query=n_query,
                n_key=n_key,
                inf=inf,
            )
            for _ in range(no_blocks)
        ])

        self.c_a = c_a
        self.c_s = c_s
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.no_blocks = no_blocks
        self.n_transition = n_transition
        self.use_ada_layer_norm = use_ada_layer_norm
        self.n_query = n_query
        self.n_key = n_key
        self.inf = inf
        # ``no_heads * c_hidden`` need not equal ``c_a``; every slice of a
        # q/k/v/gate projection uses this rather than ``c_a``.
        self.c_qkv = c_hidden * no_heads

        # Built on the first forward, not here: callers cast parameters to a
        # different dtype and load weights between construction and use.
        self._stacked: _StackedWeights | None = None
        self._weight_generation = 0
        self._parameter_list: list = []
        self._addresses: tuple = ()
        self._graph_runner = _GraphRunner()

        self._executor = (self._run_cross_attention if self.use_cross_attention
                          else self._run_self_attention)

        self.register_load_state_dict_post_hook(
            lambda module, incompatible_keys: module._invalidate_stacked_weights())

    # -- weight-generation bookkeeping --------------------------------------

    def _apply(self, *args, **kwargs):
        # ``.to()``, ``.cuda()``, ``.float()`` all route here and replace
        # parameter storage, which breaks both the aliasing and any captured
        # graph.
        self._invalidate_stacked_weights()
        return super()._apply(*args, **kwargs)

    def _invalidate_stacked_weights(self) -> None:
        self._stacked = None
        self._addresses = ()
        self._weight_generation += 1
        self._graph_runner.reset()

    def _note_weight_addresses(self) -> None:
        """Notice replaced parameter storage, on whichever path a call takes.

        Replacing a parameter's storage (``p.data = ...``) leaves the batched
        tensors holding the old values and leaves any captured graph reading the
        old addresses.  It is the one weight change that aliasing does not cover,
        so it is checked on every call rather than sampled: measured at 34.5 us
        for the 552-parameter configuration, about 1.6% of its latency, against a
        stale-weight result if skipped.  The check runs even when the call is
        heading for the unrestructured path, because that path can be captured
        into a graph too.

        A change to a parameter's *values* needs no detection: the batched
        tensors are views of the same memory, and a replayed graph reads that
        memory rather than a snapshot of it.

        Not covered: replacing a whole ``Parameter`` *object* on a submodule.
        The scan walks the parameter objects it cached, so a substituted object
        is never consulted; detecting it needs a full module traversal, which
        measures 784 us -- more than a third of this operator's entire latency.
        Callers doing that should call ``refresh_weights()``.
        """
        if not self._parameter_list:
            self._parameter_list = list(self.parameters())
            self._addresses = self._address_state()
        elif self._address_state() != self._addresses:
            self.refresh_weights()

    def refresh_weights(self) -> None:
        """Drop the batched weights and any captured graph, and re-read them."""
        self._invalidate_stacked_weights()
        self._parameter_list = list(self.parameters())
        self._addresses = self._address_state()

    def _stacked_weights(self):
        if self._stacked is None:
            self._stacked = _StackedWeights(self)
            # Building re-points every aliased parameter, so the addresses the
            # scan compares against have to be re-read afterwards.
            self._addresses = self._address_state()
        return self._stacked

    def _address_state(self) -> tuple:
        return tuple(p.data_ptr() for p in self._parameter_list)

    # -- reference math -----------------------------------------------------

    def _reference_stack(
        self,
        a: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor | None,
        _mask_trans: bool = True,
        per_block: list | None = None,
    ) -> torch.Tensor:
        """The unrestructured stack.

        Owns every case the restructured executors do not claim, and is the
        correctness reference they are validated against.
        """
        if self.use_cross_attention:
            z = self.layer_norm_z(z)

        for block in self.blocks:
            a = block(a=a, s=s, z=z, mask=mask, _mask_trans=_mask_trans)
            if per_block is not None:
                per_block.append(a)

        return a

    # -- restructured executors --------------------------------------------

    def _fast_path_applicable(self, a, s, z, mask) -> bool:
        """Whether the restructured executors cover this call.

        Anything not covered is routed to ``_reference_stack``, which is exact by
        construction, rather than being approximated here.
        """
        if not self.use_ada_layer_norm:
            return False
        try:
            dtype = next(self.parameters()).dtype
        except StopIteration:
            return False
        tensors = [a, s, z] + ([] if mask is None else [mask])
        for t in tensors:
            if t.dtype is not dtype or not t.is_contiguous():
                return False
        if a.dim() < 3 or s.dim() != 3:
            return False
        if s.shape[-1] != self.c_s or a.shape[-1] != self.c_a:
            return False
        if z.shape[-1] != self.c_z:
            return False
        batch_dims = a.shape[:-2]
        batch = s.shape[0]
        # The conditioning tensors are shaped from ``s``; they must broadcast
        # against ``a`` without reordering.
        if not (tuple(batch_dims) == (batch,)
                or (batch == 1 and math.prod(batch_dims) == 1)):
            return False
        if z.shape[0] != batch:
            return False
        if mask is not None and (mask.dim() != 2 or mask.shape[0] != batch):
            return False

        n_row = a.shape[-2]
        if mask is not None and mask.shape[-1] != n_row:
            return False
        if self.use_cross_attention:
            n_q, n_k = self.n_query, self.n_key
            num_win = math.ceil(n_row / n_q)
            return (z.dim() == 5 and z.shape[-4] == num_win
                    and z.shape[-3] == n_q and z.shape[-2] == n_k)
        return (z.dim() == 4 and s.shape[-2] == n_row
                and z.shape[-3] == n_row and z.shape[-2] == n_row)

    def _conditioning(self, s, weights, n_ada, dtype):
        """Per-AdaLN gate and shift for every block, from one normalization.

        ``normalize(s)`` is block-invariant, so it is computed once in fp32 and
        each AdaLN's own conditioning scale is applied to it -- one batched GEMM
        with one item per AdaLN, in place of two small GEMMs and a LayerNorm per
        AdaLN.  Sharing one *scaled* input across AdaLNs would be wrong: their
        conditioning scales differ.
        """
        c_s, c_a = self.c_s, self.c_a
        s32 = _normalize_fp32(s, c_s)
        scaled = (s32.unsqueeze(0)
                  * weights.ln_s_w.float()[:, None, None, :]).to(dtype)
        rows = scaled.reshape(n_ada, -1, c_s)
        projected = torch.empty(n_ada, rows.shape[1], 2 * c_a,
                                dtype=dtype, device=rows.device)
        if RESTRUCTURINGS["fuse_conditioning_gemm"]:
            torch.baddbmm(weights.s_proj_b3, rows, weights.s_proj_wt,
                          out=projected)
        else:
            # One GEMM per AdaLN, written straight into the batched output, so
            # the gate and the shift can still be sigmoided and consumed in
            # batched form while each projection stays bit-identical to the
            # reference's F.linear.
            bias, weight = weights.s_proj_b, weights.s_proj_wt
            for i in range(n_ada):
                torch.addmm(bias[i], rows[i], weight[i], out=projected[i])
        projected = projected.view(n_ada, *s.shape[:-1], 2, c_a)
        return torch.sigmoid(projected[..., 0, :]), projected[..., 1, :]

    def _raw_s_gates(self, s, weights, dtype):
        """The two output gates that read raw ``s``, for every block at once.

        Both consumers apply a sigmoid to the projection, so one GEMM plus one
        sigmoid covers ``2 * no_blocks`` of them.  These cannot join the
        conditioning GEMM above: that one reads normalized ``s``.
        """
        gates = F.linear(s, weights.raw_w_flat, weights.raw_b_flat)
        return torch.sigmoid(gates).view(*s.shape[:-1], self.no_blocks, 2, self.c_a)

    @staticmethod
    def _block_pair_bias(attention, z):
        """One block's pair bias the reference's way: normalize, project, permute."""
        return _permute_final_dims(attention.linear_z(attention.layer_norm_z(z)),
                                   [2, 0, 1])

    @staticmethod
    def _block_output_gates(batched, attention, transition, s, b):
        """One block's two raw-s output gates, batched or one GEMM each."""
        if batched is not None:
            return batched[:, :, b, 0, :], batched[:, :, b, 1, :]
        return (torch.sigmoid(attention.linear_ada_out(s)),
                torch.sigmoid(transition.linear_g(s)))

    def _run_self_attention(self, a, s, z, mask, _mask_trans=True, per_block=None):
        weights = self._stacked_weights()
        c_a, c_z, c_qkv = self.c_a, self.c_z, self.c_qkv
        c_hidden, heads, n_t = self.c_hidden, self.no_heads, self.n_transition
        nb = self.no_blocks
        dtype = a.dtype
        batch_dims = a.shape[:-2]
        n_row = a.shape[-2]
        batch = s.shape[0]
        # Divide, as the reference does, rather than multiply by a reciprocal.
        # The two agree in bf16 for the captured head sizes but not in general:
        # x * fl(1/y) and x / y differ for some y (c_hidden = 175, for one), and
        # the cost is the same either way.
        scale = math.sqrt(c_hidden)

        mask_given = mask is not None
        if not mask_given:
            mask = a.new_ones(a.shape[:-1])
        mask_bias = (self.inf * (mask.expand((*batch_dims, -1)) - 1))[
            ..., None, None, :]

        # Pair bias for every block: the pair normalization is block-invariant,
        # so one normalization feeds a batched GEMM over the per-block scales
        # and projections.  The head permute is a stride choice, not a copy.
        batched_pair_bias = RESTRUCTURINGS["batch_pair_bias"]
        if batched_pair_bias:
            z32 = _normalize_fp32(z, c_z)
            z_scaled = (z32.unsqueeze(0)
                        * weights.ln_z_w.float().view(nb, 1, 1, 1, c_z)).to(dtype)
            pair_bias = torch.bmm(z_scaled.reshape(nb, -1, c_z), weights.z_wt)
            pair_bias = pair_bias.view(nb, batch, n_row, n_row, heads).permute(
                0, 1, 4, 2, 3)

        batched_conditioning = RESTRUCTURINGS["batch_conditioning"]
        if batched_conditioning:
            gate_s, shift_s = self._conditioning(s, weights, nb * 2, dtype)
        merged_gemms = RESTRUCTURINGS["merge_column_parallel_gemms"]
        out_gates = (self._raw_s_gates(s, weights, dtype)
                     if RESTRUCTURINGS["batch_output_gates"] else None)
        trans_mask = mask.unsqueeze(-1) if (_mask_trans and mask_given) else None

        for b in range(nb):
            i = 2 * b
            block = self.blocks[b]
            attention, transition = (block.attention_pair_bias,
                                     block.conditioned_transition)
            if batched_conditioning:
                conditioned = gate_s[i] * (_layer_norm_plain(a, c_a) + shift_s[i])
            else:
                conditioned = attention.layer_norm_a(a, s)

            if merged_gemms:
                qkvg = F.linear(conditioned, weights.qkvg_w[b], weights.qkvg_b[b])
                q, k, v, head_gate = qkvg.split(c_qkv, dim=-1)
            else:
                mha = attention.mha
                q, k, v, head_gate = (mha.linear_q(conditioned),
                                      mha.linear_k(conditioned),
                                      mha.linear_v(conditioned),
                                      mha.linear_g(conditioned))
            q = _split_heads(q, heads, c_hidden) / scale
            k = _split_heads(k, heads, c_hidden)
            v = _split_heads(v, heads, c_hidden)

            scores = torch.matmul(q, k.transpose(-1, -2))
            scores = scores + mask_bias
            scores = scores + (pair_bias[b] if batched_pair_bias
                               else self._block_pair_bias(attention, z))
            attended = torch.matmul(torch.softmax(scores, dim=-1), v)
            attended = attended.transpose(-2, -3)

            gated = torch.sigmoid(head_gate)
            gated = gated.view(*gated.shape[:-1], heads, c_hidden)
            attended = (attended * gated).reshape(*batch_dims, n_row, c_qkv)
            attention_gate, transition_gate = self._block_output_gates(
                out_gates, attention, transition, s, b)
            a = a + attention_gate * F.linear(attended, weights.o_w[b])

            if batched_conditioning:
                conditioned = gate_s[i + 1] * (_layer_norm_plain(a, c_a)
                                               + shift_s[i + 1])
            else:
                conditioned = transition.layer_norm(a, s)
            if merged_gemms:
                hidden = F.linear(conditioned, weights.swiglu_w[b])
                up, gate = hidden.split(n_t * c_a, dim=-1)
            else:
                up = transition.swiglu.linear_a(conditioned)
                gate = transition.swiglu.linear_b(conditioned)
            update = F.linear(F.silu(up) * gate, weights.out_w[b])
            update = transition_gate * update
            if trans_mask is not None:
                update = update * trans_mask
            a = a + update
            if per_block is not None:
                per_block.append(a)

        return a

    def _run_cross_attention(self, a, s, z, mask, _mask_trans=True, per_block=None):
        weights = self._stacked_weights()
        c_a, c_z, c_qkv = self.c_a, self.c_z, self.c_qkv
        c_hidden, heads, n_t = self.c_hidden, self.no_heads, self.n_transition
        nb, n_q, n_k = self.no_blocks, self.n_query, self.n_key
        dtype = a.dtype
        batch_dims = a.shape[:-2]
        n_atom = a.shape[-2]
        batch = s.shape[0]
        num_win = math.ceil(n_atom / n_q)
        pad = (-n_atom) % n_q
        n_padded = n_atom + pad
        flat = math.prod(batch_dims) if batch_dims else 1
        # Divide, as the reference does, rather than multiply by a reciprocal.
        # The two agree in bf16 for the captured head sizes but not in general:
        # x * fl(1/y) and x / y differ for some y (c_hidden = 175, for one), and
        # the cost is the same either way.
        scale = math.sqrt(c_hidden)

        mask_given = mask is not None
        if not mask_given:
            mask = a.new_ones(a.shape[:-1])

        # Key windows, block mask and pair bias are all block-invariant, so they
        # are computed once per call rather than once per block.  The window
        # arithmetic is the reference's own, reproduced rather than re-derived:
        # it runs in the mask's dtype, and in bf16 the windows are neither
        # contiguous nor duplicate-free.
        def window_invariants():
            mask_padded = F.pad(mask, (0, pad)) if pad else mask
            mask_padded = mask_padded.expand((*batch_dims, -1))
            key_index, invalid = _get_block_key_indices(mask_padded, n_q, n_k)

            index_flat = key_index.reshape(flat, num_win * n_k)
            at_keys = torch.gather(mask_padded.reshape(flat, -1), 1, index_flat)
            at_keys = at_keys.reshape(*batch_dims, num_win, n_k)
            in_range = (~invalid).to(dtype)
            block_mask = (mask_padded.reshape(*batch_dims, num_win, n_q).unsqueeze(-1)
                          * (in_range * at_keys).unsqueeze(-2))
            bias = (self.inf * (block_mask - 1))[..., None, :, :]
            # Out-of-range key positions must contribute exactly zero, which is
            # what the reference achieves by zeroing the gathered activations.
            # Filled rather than multiplied by a 0/1 flag: a projection that
            # overflowed to inf, or an input carrying a NaN, would survive
            # `x * 0` as a NaN where the reference's masked_fill_ gives zero.
            return index_flat, invalid.unsqueeze(-1), bias

        hoisted = RESTRUCTURINGS["hoist_window_invariants"]
        if hoisted:
            index_flat, out_of_range, mask_bias = window_invariants()
            gather_index = index_flat.unsqueeze(-1).expand(-1, -1, 2 * c_qkv)

        # Normalized once for the whole stack, inline rather than through the
        # LayerNorm module: that module caches an fp32 copy of its scale and only
        # notices a change of Parameter *identity*, so an in-place change to the
        # scale would not reach it.  Computing it here also matches how the
        # self-attention executor treats its per-block pair scales.
        normalized_z = (_normalize_fp32(z, c_z)
                        * self.layer_norm_z.weight.float()).to(dtype)
        batched_pair_bias = RESTRUCTURINGS["batch_pair_bias"]
        if batched_pair_bias:
            pair_bias = F.linear(normalized_z, weights.z_w_flat)
            pair_bias = pair_bias.view(*pair_bias.shape[:-1], nb, heads)
            pair_bias = pair_bias.permute(4, 0, 1, 5, 2, 3)

        # Rows are zero-padded to a whole number of query blocks.  This is exact
        # because a zero row normalizes to zero and the shift projection is
        # unbiased, so a padded row contributes nothing; padded query rows are
        # discarded before the residual add.
        s_padded = F.pad(s, (0, 0, 0, pad)) if pad else s
        batched_conditioning = RESTRUCTURINGS["batch_conditioning"]
        if batched_conditioning:
            gate_s, shift_s = self._conditioning(s_padded, weights, nb * 3, dtype)
        merged_gemms = RESTRUCTURINGS["merge_column_parallel_gemms"]
        deduplicated = RESTRUCTURINGS["dedup_key_value_projections"]
        out_gates = (self._raw_s_gates(s, weights, dtype)
                     if RESTRUCTURINGS["batch_output_gates"] else None)
        trans_mask = mask.unsqueeze(-1) if (_mask_trans and mask_given) else None

        for b in range(nb):
            i = 3 * b
            block = self.blocks[b]
            attention, transition = (block.attention_pair_bias,
                                     block.conditioned_transition)
            mha = attention.mha
            if not hoisted:
                index_flat, out_of_range, mask_bias = window_invariants()
                gather_index = index_flat.unsqueeze(-1).expand(-1, -1, 2 * c_qkv)

            padded = F.pad(a, (0, 0, 0, pad)) if pad else a
            if batched_conditioning:
                # Both AdaLNs normalize the activation with no affine
                # parameters, so they share the normalization but not the
                # conditioning.
                normed = _layer_norm_plain(padded, c_a)
                query_rows = gate_s[i] * (normed + shift_s[i])
                key_rows = gate_s[i + 1] * (normed + shift_s[i + 1])
            else:
                query_rows = attention.layer_norm_a_q(padded, s_padded)
                key_rows = attention.layer_norm_a_k(padded, s_padded)

            if deduplicated:
                # The key and value projections run once over the real rows,
                # not once per gathered key position; the window is read after.
                if merged_gemms:
                    kv = F.linear(key_rows, weights.kv_w[b])
                else:
                    kv = torch.cat((mha.linear_k(key_rows),
                                    mha.linear_v(key_rows)), dim=-1)
                kv = torch.gather(kv.reshape(flat, n_padded, 2 * c_qkv), 1,
                                  gather_index)
                kv = kv.view(*batch_dims, num_win, n_k, 2 * c_qkv)
                kv.masked_fill_(out_of_range, 0.0)
            else:
                rows = torch.gather(
                    key_rows.reshape(flat, n_padded, c_a), 1,
                    index_flat.unsqueeze(-1).expand(-1, -1, c_a))
                rows = rows.view(*batch_dims, num_win, n_k, c_a)
                rows.masked_fill_(out_of_range, 0.0)
                kv = (F.linear(rows, weights.kv_w[b]) if merged_gemms else
                      torch.cat((mha.linear_k(rows), mha.linear_v(rows)), dim=-1))

            if merged_gemms:
                qg = F.linear(query_rows, weights.qg_w[b], weights.qg_b[b])
            else:
                qg = torch.cat((mha.linear_q(query_rows),
                                mha.linear_g(query_rows)), dim=-1)

            q, head_gate = qg.view(*batch_dims, num_win, n_q,
                                   2 * c_qkv).split(c_qkv, dim=-1)
            k, v = kv.split(c_qkv, dim=-1)
            q = _split_heads(q, heads, c_hidden) / scale
            k = _split_heads(k, heads, c_hidden)
            v = _split_heads(v, heads, c_hidden)

            scores = torch.matmul(q, k.transpose(-1, -2))
            scores = scores + mask_bias
            if batched_pair_bias:
                scores = scores + pair_bias[b]
            else:
                scores = scores + _permute_final_dims(
                    attention.linear_z(normalized_z), [2, 0, 1])
            attended = torch.matmul(torch.softmax(scores, dim=-1), v)
            attended = attended.transpose(-2, -3)

            gated = torch.sigmoid(head_gate)
            gated = gated.view(*gated.shape[:-1], heads, c_hidden)
            attended = (attended * gated).reshape(*batch_dims, n_padded, c_qkv)
            update = F.linear(attended, weights.o_w[b])[..., :n_atom, :]
            attention_gate, transition_gate = self._block_output_gates(
                out_gates, attention, transition, s, b)
            a = a + attention_gate * update

            if batched_conditioning:
                conditioned = (gate_s[i + 2][..., :n_atom, :]
                               * (_layer_norm_plain(a, c_a)
                                  + shift_s[i + 2][..., :n_atom, :]))
            else:
                conditioned = transition.layer_norm(a, s)
            if merged_gemms:
                hidden = F.linear(conditioned, weights.swiglu_w[b])
                up, gate = hidden.split(n_t * c_a, dim=-1)
            else:
                up = transition.swiglu.linear_a(conditioned)
                gate = transition.swiglu.linear_b(conditioned)
            update = F.linear(F.silu(up) * gate, weights.out_w[b])
            update = transition_gate * update
            if trans_mask is not None:
                update = update * trans_mask
            a = a + update
            if per_block is not None:
                per_block.append(a)

        return a

    # -- dev-time hook ------------------------------------------------------

    def _per_block_probe(self, a, s, z, mask, _mask_trans=True, out=None):
        """Per-block activations of whichever path this call would take.

        Used by ``tools/parity.py`` to locate a divergence in a deep residual
        chain rather than merely detect one at the output.
        """
        if out is None:
            out = []
        self._note_weight_addresses()
        if (RESTRUCTURINGS["use_restructured_executors"]
                and self._fast_path_applicable(a, s, z, mask)):
            self._stacked_weights()
            self._executor(a, s, z, mask, _mask_trans, out)
        else:
            self._reference_stack(a, s, z, mask, _mask_trans, out)
        return out

    # -- entry point --------------------------------------------------------

    def forward(
        self,
        a: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor | None = None,
        use_deepspeed_evo_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_lma: bool = False,
        use_high_precision_attention: bool = False,
        _mask_trans: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            a:    [*, N, C_token] token-level embedding
            s:    [*, N, C_s] single embedding
            z:    [*, N, N, C_z] pair embedding
            mask: [*, N] mask

        Returns:
            [*, N, C_token] updated token embedding
        """
        self._note_weight_addresses()

        # The restructured executors read the batched weights rather than the
        # Parameters, so they produce no parameter gradients.  Under grad the
        # unrestructured path -- which goes through the Parameters -- is the only
        # correct one.  Building the batched store also re-points parameter
        # storage and allocates, neither of which belongs inside an enclosing
        # capture, so a first call made during someone else's capture stays on
        # the unrestructured path too.
        restructured = (RESTRUCTURINGS["use_restructured_executors"]
                        and not torch.is_grad_enabled()
                        and self._fast_path_applicable(a, s, z, mask)
                        and not (self._stacked is None and a.is_cuda
                                 and torch.cuda.is_current_stream_capturing()))
        if restructured:
            run = self._executor
            self._stacked_weights()
        else:
            run = self._reference_stack

        if not self._graph_eligible(a, s, z, mask):
            return run(a, s, z, mask, _mask_trans)

        key = (self._weight_generation, _mask_trans, mask is None, restructured,
               self.inf, a.device, torch.is_autocast_enabled(),
               torch.get_autocast_dtype("cuda"),
               tuple(a.shape), tuple(a.stride()), a.dtype,
               tuple(s.shape), s.dtype, tuple(z.shape), z.dtype,
               None if mask is None else (tuple(mask.shape), mask.dtype))

        def evaluate(a_, s_, z_, mask_):
            return run(a_, s_, z_, mask_, _mask_trans)

        return self._graph_runner.run(evaluate, key, (a, s, z, mask))

    def _graph_eligible(self, a, s, z, mask) -> bool:
        if _graphs_disabled() or torch.is_grad_enabled():
            return False
        if not a.is_cuda or torch.cuda.is_current_stream_capturing():
            return False
        for t in (a, s, z, mask):
            if t is not None and not t.is_contiguous():
                return False
        return True
