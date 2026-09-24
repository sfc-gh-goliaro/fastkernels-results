"""Qwen3-Next decoder layer: the whole layer as one CUDA-graph replay.

At the captured decode-band token counts this layer is *glue*, not math. Seven of
the eight captured shapes are T <= 69, the harness benchmarks five of them (four
below T = 302), and at [1, 2048] the layer as shipped spent 207 us of wall clock
on ~60 us of GPU work: two norms, the attention's launches, the MoE's own
static-in copy and static-out clone, and the dispatch branching in between.

Four changes. The first two are *correctness* fixes, they are the reason the third
one has to exist, and the fourth is what is left once the third has removed all
the host-side dispatch: the GPU work itself, in the one component that was nowhere
near its roofline.

0. The trap both fixes are about. The MoE's first act is
   ``topk(softmax(gate(x)), 10)`` over 512 experts, and the 10th/11th logit gap
   on the harness's random weights is ~0.006. So a **one-ulp bf16** difference
   anywhere upstream of the post-attention norm changes ~1-2% of tokens' expert
   *sets*, and a flipped token is a whole wrong row: at T=60 one flip is 1.7% of
   the output against the harness's 1%-of-elements budget. Three of the five
   benched shapes failed INCORRECT_NUMERICAL as shipped (matched 0.968 / 0.989 /
   0.987), and the pass/fail was close to a coin toss. Everything feeding that
   router therefore has to be bit-identical to the reference, not merely correct
   to bf16.

1. ``torch.compile`` for the residual norm.
   The baseline's ``GemmaRMSNorm.forward_cuda`` wraps its two pure-PyTorch
   helpers in ``torch.compile``; the L1 winner replaced only the *no*-residual
   path with a Triton kernel and left the residual path running the same helper
   *eagerly*. Eager and Inductor disagree on the last bf16 bit of ~0.8% of
   elements (measured: 969 of 122880 at [60, 2048], up to 1.6e-2 apart). Which
   config Inductor autotunes into the baseline's kernel also varies run to run,
   so the disagreement is not even stable.
   Compiling the L1 winner's *own* helper -- byte-identical source to the
   baseline's -- lands on the same Inductor cache entry and is therefore
   bit-identical, not close: 0 mismatches over T in [1 .. 16384] x 3 draws. It is
   also strictly faster than the eager path it replaces, one fused kernel instead
   of nine elementwise passes -- 492 us -> 104 us per norm at [16384, 2048],
   where eager moves ~1.5 GB against 268 MB fused.
   The compile is per *norm instance*, mirroring the baseline, so both sides see
   the same shape history and neither can slip into a dynamic-shape variant the
   other does not have.

2. The reference GDN math, not the L2 winner's fused kernel (``_attend``).
   That kernel is a different algorithm -- correct to bf16 in isolation, which is
   all its own L2 compare can see, but ~1 ulp from the reference on the [T, 2048]
   it hands back, straight into the router above. Measured through this layer
   with it on: 1 bad row of 60, 5 of 301. There is no L3-side repair; the flip
   needs *bit* equality and no reimplementation of the recurrence gives that.

3. One graph per (token count, entry shape, state identity) for the whole layer.
   This is what pays for (2). The reference GDN's cost was never its arithmetic
   -- a couple of microseconds of real work at T=60 -- but the ~390 us of host
   dispatch its eighteen launches spend, and a replay does not spend host
   dispatch. Measured at [1, 2048]: the reference attention is 393 us eager
   (425 us of host enqueue) and 63 us of GPU time inside the graph. Norm -> GDN
   -> norm -> MoE becomes two ``_foreach_copy_`` launches around one
   ``cudaGraphLaunch``, and the whole layer goes 635 us -> 138 us. Across the
   five shapes the graph is worth 1.99x of geomean (2.26x -> 4.55x, A/B'd with
   ``FK_L3_GRAPH=0``).
   The guards that matter:
     * ``is_current_stream_capturing()`` -- the MoE child replays its *own* graph
       at 8 < M <= 4096, which is illegal inside a capture. It already checks the
       flag and degrades to its (capturable) eager body, so capturing the layer
       collapses the nested graph into ours instead of failing; we only have to
       decline to capture while someone else is capturing us.
     * ``_tactic_for`` -- the MoE resolves its trtllm-gen tile by *timing* it,
       which synchronizes and captures throwaway graphs. The warmup runs the
       eager body off the capture stream first so that is settled, and the same
       warmup result is the reference the capture is validated against.
     * address stability -- the recurrent and conv states live at fixed
       addresses, but the harness hands out a *fresh* metadata object (and a
       fresh ``state_indices``) per correctness round, and the children re-cut
       their scratch by shape. The cache key carries the metadata identity and
       the state addresses, the entry holds a strong reference to every scratch
       slot the capture baked in, and a per-element bf16-tolerance check against
       the eager body gates the graph before it is ever used.
     * an aborted capture is contained rather than local -- see ``_capture``.
   T > ``_GRAPH_MAX_T`` stays eager: at [16384, 2048] the layer is device-bound
   (3.7 ms of GPU against 1.4 ms of host), the static buffers would be 134 MB,
   and a replay saves nothing it does not cost.

4. An L3-owned routed-expert MoE below ``SmallTokenMoE._SMALL_MAX`` tokens.
   With (3) in place four of the five benched shapes are device-bound inside a
   single replay, and at [1, 2048] the MoE is 43 of the ~140 us. It was not
   bandwidth-bound: the 69 MB of expert weights that top-10-of-512 selects reads
   at 3.3 TB/s, but so does a *read-only* kernel over the same footprint -- 69 MB
   is simply too small a transfer for this B200 to reach the 7.0 TB/s it gets on
   4 GB (measured: 4 GB 7.06, 1 GB 6.22, 138 MB 4.08, 69 MB 3.55 TB/s). So the
   floor was ~21 us, not the ~10 us a peak-bandwidth figure suggests, and the win
   was in the 19 us the child spent *around* the stream plus the reduction inside
   it. See ``moe_small.cu``; measured 43.0 -> 30.7 us at M=1, and the whole layer
   142.3 -> 123.9 us. Routing is untouched in substance -- the expert ids the
   kernel selects are identical to the reference's for every token at every token
   count tested, which is the part that has no error budget at all (a flipped
   route is a whole wrong output row, and at M=1 one row is 100% of the elements
   against a 99% bar).

Under tensor parallelism the two parallel regions per layer (attention output,
MoE output) hand back un-reduced partials and the following norm does
all-reduce + residual-add + RMSNorm in one FlashInfer kernel -- the same fusion
vLLM's ``fuse_allreduce_rms`` pass applies
(``AllReduceFusedAddGemmaRMSNormPattern``). The MoE's partial is consumed by the
*next* layer's ``input_layernorm``, or by ``Qwen3NextModel.norm`` for the last
layer, so the model owns that half of the contract. That path is untouched:
``fuse_ar_norm`` short-circuits both the compiled norm (the collective is opaque
to Dynamo) and the capture (a layer that all-reduces is not one we capture), so
tp > 1 runs the reference sequence exactly as before.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn

from ....infra.context import get_context
from ....infra.cuda_ext import lazy_op
from ....infra.tp import _tp_size
from ..L2.flashinfer_allreduce_fusion import fused_allreduce_add_gemma_rmsnorm
from ..L1.gemma_rms_norm import GemmaRMSNorm
from ..L2.qwen3_next_gdn_attention import Qwen3NextGDNAttention
from ...baseline.L2.qwen3_next_gdn_attention import (
    Qwen3NextGDNAttention as _RefGDNAttention,
)
from ..L2.qwen3_next_attention import Qwen3NextAttention
from ..L2.shared_expert_moe import SharedExpertMoE


# The routed-expert MoE for the small-token regime, ours rather than the L2
# child's. See ``moe_small.cu`` for what it does and why the roofline it is
# aiming at is ~21 us of weight stream, not the ~10 us a 6.9 TB/s figure implies.
_MOE = lazy_op("l3_qwen3_next_moe_small", "moe_small.cu")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


# Token counts above this stay eager: the layer is device-bound there, so a
# replay only adds a 134 MB pair of static buffers and two copies over it.
_GRAPH_MAX_T = _env_int("FK_L3_GRAPH_MAX_T", 2048)
# One entry per (token count, entry shape, state identity). The harness rebuilds
# its metadata per correctness round, so a handful of keys is normal; past this
# the layer simply stays eager rather than growing graphs without bound.
_GRAPH_CACHE_MAX = _env_int("FK_L3_GRAPH_CACHE", 12)
_GRAPH_ENABLED = _env_int("FK_L3_GRAPH", 1)
# The GDN child's fused kernel is a *different algorithm*: correct to bf16 in
# isolation (that is what its own L2 compare measures) but not bit-identical to
# the reference, ~1 bf16 ulp apart on the [T, hidden] it hands back. This layer
# feeds that straight into a 512-expert top-10 router, and one ulp there flips
# the 10th/11th expert for ~1-2% of tokens; a flipped token is a whole wrong row,
# so at T=60 a single flip is 1.7% of the output against the harness's 1% budget.
# Measured with the fast kernel on: 1 bad row of 60 / 5 of 301, i.e. INCORRECT
# roughly every other draw. The reference implementation the class inherits is
# bit-identical by construction, and the graph below is what pays for it -- its
# cost was never the arithmetic (a couple of microseconds at T=60) but the 386 us
# of host dispatch its eighteen launches spend, which a replay does not spend.
# ``FK_L3_GDN_FAST=1`` puts the fused kernel back for A/B measurement.
_GDN_FAST = _env_int("FK_L3_GDN_FAST", 0)
_DEBUG = _env_int("FK_L3_DEBUG", 0)
# ``FK_L3_MOE_SMALL=0`` puts the L2 child's own fused small-token path back for
# A/B measurement; the others are sweep handles for ``moe_small.cu``. Bit 0 of
# ``FK_L3_MOE_FLAGS`` folds the output cast into the expert kernel (measured
# slower), bit 1 runs the routing in the router's last CTA instead of its own
# kernel (measured a wash, and it makes the workspace's arrival counter
# load-bearing, so it is off), bits 4-6 pick the router's block shape and bits
# 8-9 the L2 eviction policy for the weight stream.
_MOE_SMALL = _env_int("FK_L3_MOE_SMALL", 1)
_MOE_GRID = _env_int("FK_L3_MOE_GRID", 0)
_MOE_CFG = _env_int("FK_L3_MOE_CFG", -1)
_MOE_FLAGS = _env_int("FK_L3_MOE_FLAGS", 0)
# Per-dtype bound the harness itself compares with (bfloat16 row of
# ``bench._TOLERANCES``), used here against the eager body rather than the
# baseline, and required of every element.
_VALIDATE_ATOL = 1e-2
_VALIDATE_RTOL = 1e-2


def _repair_cuda_rng() -> None:
    """Best-effort undo of a capture that aborted before ``capture_epilogue()``.

    The generator is left flagged as capturing, which poisons every later RNG
    call in the process. A *clean* capture runs the epilogue, so one throwaway
    graph is the cheapest thing that can put it back. Probes first so the normal
    path costs nothing, and swallows everything: this runs on a path that has
    already failed.
    """
    try:
        torch.empty(1, device="cuda").normal_()
        return
    except Exception:
        pass
    try:
        g = torch.cuda.CUDAGraph()
        t = torch.zeros(1, device="cuda")
        with torch.no_grad(), torch.cuda.graph(g):
            t.add_(1)
        del g
    except Exception:
        pass


# ---------------------------------------------------------------------------
# GemmaRMSNorm, through the same Inductor kernel the baseline uses.
# ---------------------------------------------------------------------------
# The L1 winner renamed the reference helpers; fall back to the baseline names so
# a standalone run (candidate L1 resolved to baseline) still works.
_NATIVE_NO_RES = getattr(
    GemmaRMSNorm, "_native_no_residual", None,
) or GemmaRMSNorm._forward_static_no_residual
_NATIVE_RES = getattr(
    GemmaRMSNorm, "_native_with_residual", None,
) or GemmaRMSNorm._forward_static_with_residual


def _compiled(norm: GemmaRMSNorm):
    """``(no_residual, with_residual)`` compiled once per *norm instance*.

    Per instance, not per module: the baseline compiles into ``self.__dict__``
    too, so both sides see the same (shape, dtype) history and Dynamo makes the
    same static-vs-dynamic decision for both. A module-level wrapper would see
    every shape the process ever benchmarks while the baseline's saw one.
    """
    fns = norm.__dict__.get("_ako_compiled")
    if fns is None:
        fns = (torch.compile(_NATIVE_NO_RES), torch.compile(_NATIVE_RES))
        norm.__dict__["_ako_compiled"] = fns
    return fns


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
    if torch.compiler.is_compiling():
        return _NATIVE_RES(
            norm.weight.data, norm.variance_epsilon, hidden_states, residual,
        )
    return _compiled(norm)[1](
        norm.weight.data, norm.variance_epsilon, hidden_states, residual,
    )


def _entry_norm(norm: GemmaRMSNorm, hidden_states):
    """``norm(hidden_states)`` with no residual stream yet (layer 0)."""
    if torch.compiler.is_compiling():
        return _NATIVE_NO_RES(norm.weight.data, norm.variance_epsilon, hidden_states)
    return _compiled(norm)[0](
        norm.weight.data, norm.variance_epsilon, hidden_states,
    )


class SmallTokenMoE(SharedExpertMoE):
    """The L2 MoE child with the routed-expert path replaced below a token count.

    Above ``_SMALL_MAX`` this is the child, unchanged: trtllm-gen reads the expert
    weights at 6.2-6.3 TB/s at M=60/301 against a measured ~6.85 TB/s ceiling for
    that footprint, so there is nothing there to win, and above M~130 its tensor
    cores are the only way to stay off the fp32 compute roof. Below it the layer is
    a latency problem, and 19 of the child's 43 us at M=1 were outside the weight
    stream (a 512-wide top-k serialized in one CTA, a 2 MB router read on 64 CTAs,
    a separate cast launch) with another 4 us inside it in scalar fp32 atomics.
    Measured 43.0 -> 30.7 us at M=1; see ``moe_small.cu``.

    Weight layout: none of this needs a new one. ``process_weights_after_loading``
    already builds the plain ``[E, 2N, H]`` w13 and the transposed ``[E, N, H]``
    w2 the child's own fused path wants, and this kernel reads exactly those, so
    the 3.2 GB of expert weights is not copied again.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._small_ws = None
        self._small_ws_m = -1
        self._small_broken = False

    # Token count below which the L3 kernel takes the routed experts. Re-derived
    # by interleaved A/B against both alternatives rather than inherited
    # (``tools/expdev.py``, us: at M=8 L3 104 / trtllm 115, at M=16 174 / 176, at
    # M=26 227 / 227, at M=60 395 / 367). It lands on the child's 8 again, but for
    # a sharper reason: 8 is the last count with a real margin, and 8-26 is a wash.
    _SMALL_MAX = _env_int("FK_L3_MOE_SMALL_MAX", 8)

    def _small_ready(self, m: int, x) -> bool:
        return (
            _MOE_SMALL
            and not self._small_broken
            and self._fused_ready          # same configuration check, same tensors
            and 0 < m <= self._SMALL_MAX
            and self.num_experts == 512    # the kernel's routing is specialized on it
            # The kernel streams activations with 16-byte vector loads; a caller
            # could hand in a slice that is not aligned, and that would fault
            # rather than degrade.
            and x.data_ptr() % 16 == 0
        )

    def _small_ws_bytes(self, m: int) -> int:
        """Size of the L3 path's scratch. Kept in sync with the ``take()``
        sequence in ``moe_small.cu``."""
        e, h = self.num_experts, self.hidden_size
        need = 0
        for nbytes in (m * e * 2, e * 4, 4, e * 4, e * m * 4, e * m * 4, m * 4,
                       8 * m * 4, 4, 4, m * h * 4):
            need += (nbytes + 255) & ~255
        return need

    def _small_workspace(self, m: int):
        """One cached allocation, carved up inside the kernel.

        Grown, never shrunk, and the *previous* buffer stays alive through the
        decoder layer's ``_scratch_refs`` -- a captured graph has its address baked
        into kernel arguments, and this is pure scratch written and read inside the
        same captured region, so replaying against a stale slot stays correct.
        """
        if m == self._small_ws_m:
            return self._small_ws
        need = self._small_ws_bytes(m)
        ws = self._small_ws
        if ws is None or ws.numel() < need:
            # Zeroed, not empty: the optional merged-routing epilogue
            # (``FK_L3_MOE_FLAGS`` bit 1) needs its arrival counter to start at
            # zero, and a fresh workspace is the only place that is guaranteed.
            ws = torch.zeros(need, dtype=torch.uint8, device=self.w13.device)
            self._small_ws = ws
        self._small_ws_m = m
        return ws

    def forward_impl(self, hidden_states: torch.Tensor) -> torch.Tensor:
        m = hidden_states.shape[0]
        if not self._small_ready(m, hidden_states):
            return super().forward_impl(hidden_states)
        out = torch.empty_like(hidden_states)
        try:
            _MOE.moe_small(hidden_states, *self._fargs, out,
                           self._small_workspace(m), self.top_k, _MOE_GRID,
                           _MOE_CFG, _MOE_FLAGS)
        except Exception:
            # The extension is JIT-compiled on first use, which happens in the
            # decoder's pre-capture warmup. A build or launch failure must
            # degrade to the child's path rather than propagate: an exception
            # raised *inside* a capture region is what poisons the process's
            # CUDA RNG (see ``Qwen3NextDecoderLayer._capture``), and this one
            # would raise on every later call as well.
            self._small_broken = True
            if _DEBUG:
                import traceback
                traceback.print_exc()
            return super().forward_impl(hidden_states)
        return out


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
        self.mlp = SmallTokenMoE(
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

        # key -> (graph, static_in, static_res, outputs, keepalive) | ()
        self._graphs: dict[tuple, tuple] = {}
        self._graph_pool = None
        self._graph_broken = False

    # ---- the reference sequence, unchanged in structure -------------------
    def _attend(self, hidden_states, state_manager):
        """GDN linear attention, through the bit-identical reference math.

        The child is still the module that was built (its ``__init__``, weight
        loader and ``isinstance`` identity are what the harness's recurrent-state
        prep and this layer's contract are written against); only its fused
        dispatch is declined, the same way the child declines it itself for a
        layout its kernel does not handle. ``_use_custom_op`` deployments keep
        their registered op.
        """
        if _GDN_FAST or getattr(self.linear_attn, "_use_custom_op", False):
            return self.linear_attn(hidden_states, state_manager=state_manager)
        return _RefGDNAttention.forward_impl(
            self.linear_attn, hidden_states, state_manager,
        )

    def _body(self, hidden_states, residual, positions, rotary_emb, state_manager):
        if residual is None:
            # Layer 0: the input is the vocab-parallel embedding's output, which
            # is already reduced, and there is no residual stream yet.
            residual = hidden_states
            hidden_states = _entry_norm(self.input_layernorm, hidden_states)
        else:
            hidden_states, residual = fused_ar_norm(
                self.input_layernorm, hidden_states, residual, self.fuse_ar_norm,
            )

        # Attention
        if self.layer_type == "linear_attention":
            hidden_states = self._attend(hidden_states, state_manager)
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

    # ---- capture ----------------------------------------------------------
    def _scratch_refs(self):
        """Strong refs to every child buffer a capture bakes an address into.

        The children recycle scratch by slot name and re-cut it when the shape
        changes, which would *free* storage a captured graph still points at.
        Holding it alive keeps a stale slot merely stale: it is pure scratch,
        written and read inside the same captured region, so a replay using the
        buffer it was captured with stays correct.
        """
        attn = getattr(self, "linear_attn", None)
        if attn is None:
            attn = getattr(self, "self_attn", None)
        refs = [getattr(self.mlp, "_fused_ws", None),
                getattr(self.mlp, "_small_ws", None)]
        for slot in ("_fbuf_p", "_fbuf_o", "_lb_x", "_lb_g", "_lb_s", "_lb_d"):
            refs.append(getattr(attn, slot, None))
        return tuple(r for r in refs if r is not None)

    def _capture(self, hidden_states, residual, positions, rotary_emb,
                 state_manager, keep):
        """Capture the whole layer for this key. Raises if it cannot be done.

        Returns ``(graph, static_in, static_res, outputs, keepalive)``.
        """
        static_in = torch.empty_like(hidden_states)
        static_in.copy_(hidden_states)
        static_res = None
        if residual is not None:
            static_res = torch.empty_like(residual)
            static_res.copy_(residual)

        # Warm up off the capture stream: this is what resolves the MoE's
        # trtllm-gen tactic (which synchronizes and captures throwaway graphs),
        # compiles both norms, settles cuBLAS's heuristic picks and cuts every
        # child scratch slot at this shape. All of that is illegal once capture
        # has begun. Repeating the call is safe: the GDN fast path is only taken
        # when no sequence carries an initial recurrent state, so a call is a
        # pure function of its input that happens to *overwrite* that state.
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side), torch.no_grad():
            for _ in range(3):
                ref = self._body(static_in, static_res, positions, rotary_emb,
                                 state_manager)
            ref = tuple(t.clone() for t in ref)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()

        if self._graph_pool is None:
            self._graph_pool = torch.cuda.graph_pool_handle()
        graph = torch.cuda.CUDAGraph()
        # An exception raised *inside* the capture region is not a local failure:
        # ``torch.cuda.graph.__exit__`` still calls ``capture_end()``, that call
        # errors on the invalidated capture, and the generator's
        # ``capture_epilogue()`` is skipped -- after which every CUDA RNG call in
        # the process dies with "Offset increment outside graph capture
        # encountered unexpectedly". (Measured: one rejected capture turned a
        # 5-shape run into 5 x RUNTIME_ERROR, the failures landing in the
        # harness's own ``torch.randn``.) So the region is bracketed on its own:
        # if it ever aborts, this module stops capturing entirely and tries to
        # put the generator back.
        try:
            with torch.no_grad(), torch.cuda.graph(graph, pool=self._graph_pool):
                outs = self._body(static_in, static_res, positions, rotary_emb,
                                  state_manager)
        except Exception:
            self._graph_broken = True
            _repair_cuda_rng()
            raise
        graph.replay()
        torch.cuda.synchronize()
        if len(outs) != len(ref):
            raise RuntimeError("captured output arity differs from the eager one")
        for got, want in zip(outs, ref):
            if got.shape != want.shape or got.dtype != want.dtype:
                raise RuntimeError("captured output does not match the eager one")
            # Not bit-identical, by design: the MoE child accumulates the routed
            # experts with fp32 atomics, so *any* two runs of the eager body
            # differ in the last bit or two as well. What the replay must not do
            # is change a routing decision or drop a term -- either shows up as a
            # whole wrong row, orders of magnitude outside this bound, which is
            # the harness's own bf16 tolerance applied to every element instead
            # of 99% of them.
            bad = ((got.float() - want.float()).abs()
                   > _VALIDATE_ATOL + _VALIDATE_RTOL * want.float().abs())
            if bool(bad.any()):
                if _DEBUG:
                    d = (got.float() - want.float()).abs()
                    print(f"[L3] replay diverges: {int(bad.sum())} of {bad.numel()} "
                          f"elements, max_abs={d.max().item():.3e}", flush=True)
                raise RuntimeError("replay does not reproduce the eager body")
        return (graph, static_in, static_res, outs, keep + self._scratch_refs())

    def _entry(self, key, hidden_states, residual, positions, rotary_emb,
               state_manager, keep):
        ent = self._graphs.get(key)
        if ent is not None:
            return ent or None
        if self._graph_broken or len(self._graphs) >= _GRAPH_CACHE_MAX:
            return None
        try:
            ent = self._capture(hidden_states, residual, positions, rotary_emb,
                                state_manager, keep)
        except Exception:
            # Anything at all -- an unsupported launch, OOM, a capture-time
            # sync, a replay that does not reproduce the eager result -- and this
            # key stays eager.
            if _DEBUG:
                import traceback
                traceback.print_exc()
            self._graphs[key] = ()
            return None
        self._graphs[key] = ent
        return ent

    # ---- entry point ------------------------------------------------------
    def _graph_key(self, hidden_states, residual, rotary_emb, state_manager):
        """``(key, keepalive)``, or ``None`` when this call is not one we capture.

        The key is everything a replay's correctness depends on and that is not
        copied in per call: the token count and entry shape, the metadata regime
        the GDN child dispatches on, and the *addresses* of the recurrent state,
        the conv state and ``state_indices``. ``id(metadata)`` stands in for the
        rest of the metadata a fallback GDN path would read; the entry holds the
        object alive, so the id cannot be recycled under us.

        A serving loop that builds a fresh metadata object per step therefore
        gets one entry per step until the cache is full and then runs eagerly --
        the same cost as not capturing at all. The harness rebuilds metadata once
        per correctness round, which is four keys.
        """
        if not (_GRAPH_ENABLED and self.layer_type == "linear_attention"
                and not self.fuse_ar_norm and state_manager is not None
                and rotary_emb is None):
            return None
        if torch.compiler.is_compiling() or torch.cuda.is_current_stream_capturing():
            return None
        if (hidden_states.dim() != 2 or not hidden_states.is_cuda
                or not hidden_states.is_contiguous()):
            return None
        n = hidden_states.shape[0]
        if not 0 < n <= _GRAPH_MAX_T:
            return None
        if residual is not None and (
                residual.shape != hidden_states.shape
                or residual.dtype != hidden_states.dtype
                or not residual.is_contiguous()):
            return None
        md = get_context().kda_metadata
        if md is None:
            return None
        li = self.layer_idx
        try:
            sidx = md.non_spec_state_indices_tensor
            conv = state_manager.gdn_conv[li]
            rec = state_manager.recurrent[li]
        except (AttributeError, IndexError, TypeError):
            return None
        if sidx is None or conv is None or rec is None:
            return None
        key = (n, hidden_states.dtype, residual is None, id(md),
               conv.data_ptr(), rec.data_ptr(), sidx.data_ptr(),
               md.num_prefills, md.num_decodes, md.any_have_initial_state)
        return key, (md, state_manager, sidx, conv, rec)

    def forward(self, hidden_states, residual, positions=None,
                rotary_emb=None, state_manager=None):
        keyed = self._graph_key(hidden_states, residual, rotary_emb, state_manager)
        if keyed is not None:
            ent = self._entry(keyed[0], hidden_states, residual, positions,
                              rotary_emb, state_manager, keyed[1])
            if ent is not None:
                graph, static_in, static_res, outs, _keep = ent
                # One multi-tensor-apply launch per side instead of four separate
                # copies. At T=1 a 4 KB copy is ~3.4 us of pure GPU-side launch
                # latency against a 125 us replay, and there are four of them
                # around it; ``_foreach_copy_`` makes that two.
                if static_res is None:
                    static_in.copy_(hidden_states)
                else:
                    torch._foreach_copy_((static_in, static_res),
                                         (hidden_states, residual))
                graph.replay()
                # The graph writes into pool memory the next replay overwrites,
                # so the caller gets tensors it owns.
                out_h = torch.empty_like(outs[0])
                out_r = torch.empty_like(outs[1])
                torch._foreach_copy_((out_h, out_r), (outs[0], outs[1]))
                return out_h, out_r
        return self._body(hidden_states, residual, positions, rotary_emb,
                          state_manager)
