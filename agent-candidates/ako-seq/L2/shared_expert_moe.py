from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from ....infra.cuda_ext import lazy_op
from ....infra.tp import _tp_rank, _tp_size
from ..L1.allreduce import AllReduce
from ..L1.gate_linear import GateLinear
from ..L1.grouped_topk import GroupedTopK
from ..L1.moe_shared_gate_add import moe_shared_gate_add
from ..L1.silu_and_mul import SiluAndMul
from .trtllm_bf16_moe import (
    TrtLlmBf16MoE,
    prepare_trtllm_bf16_moe_weights,
    trtllm_bf16_moe_supported,
)
from .fused_experts import FusedExperts
from .parallel_linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)

# One extension, one pybind entry point, four kernel launches inside it. The
# baseline's wall clock at small token counts is almost entirely host-side
# dispatch (flashinfer's trtllm wrapper alone is ~740 us of Python per call at
# M=1 against 42 us of GPU work), so the fused path's first job is to be one
# call, not to be clever.
_C = lazy_op("l2_shared_expert_moe_fused", "moe_fused.cu")


# ---------------------------------------------------------------------------
# trtllm-gen, called directly and with a tactic we picked ourselves.
#
# Two separate facts about the reference path, both measured (see ITERATIONS.md):
#
# 1. Its per-call *host* cost is ~620 us at M=445 -- and only ~90 us of that is
#    flashinfer's Python wrapper. The rest is CPU work inside trtllm-gen's C++
#    launcher, which allocates ~15 tensors and re-resolves its tile candidates on
#    every call. It grows as M *shrinks* (more tile candidates are valid), so at
#    M=445 the layer spends 0.959 ms of wall clock on 0.515 ms of GPU work. That
#    is what ``_replay`` below removes with a CUDA graph.
#
# 2. Its *tactic* is the launcher's fallback, which ``selectDefaultTileN``
#    defines as the smallest valid token tile -- nothing in this benchmark ever
#    enters flashinfer's ``autotune()``, so the tuned path is never taken. A
#    measured pick over the tile ladder is 1.13-1.27x on the mid-range shapes and
#    1.18x at M=16384, bit-for-bit identical output. That is what ``_tactic_for``
#    does.
# ---------------------------------------------------------------------------
_TRT_BF16 = 1052672        # DtypeTrtllmGen.Bfloat16
_TRT_BLOCK_MAJOR_K = 2     # WeightLayout.BlockMajorK
_TRT_SWIGLU = 3            # ActivationType.Swiglu
_TRT_FALLBACK_TACTIC = [-1, -1]


def _trtllm_raw_op():
    """flashinfer's raw ``trtllm_bf16_moe`` binding, or ``None``.

    Importing the wrapper module first is not optional: it installs the cubin
    loader callback, without which the raw op raises
    ``FlashInferSetCubinCallback not set``.
    """
    try:
        from flashinfer.fused_moe.core import get_trtllm_moe_sm100_module
        get_trtllm_moe_sm100_module()
        from flashinfer.jit.fused_moe import gen_trtllm_gen_fused_moe_sm100_module
        return gen_trtllm_gen_fused_moe_sm100_module().build_and_load()
    except Exception:
        return None


def trtllm_routing_method_type(
    routing: str,
    renormalize: bool,
    has_e_score_bias: bool,
    num_expert_group: int | None,
) -> int | None:
    """Map a routing config to a trtllm-gen ``RoutingMethodType``, else None.

    Mirrors vLLM's ``get_routing_method_type``
    (``vllm/model_executor/layers/fused_moe/config.py``). ``None`` stands for
    vLLM's ``Unspecified``, which ``TrtLlmBf16ExpertsMonolithic`` does not
    accept -- callers fall back to the Triton path in that case.
    """
    if has_e_score_bias:
        if routing != "sigmoid" or not renormalize:
            return None
        if (num_expert_group or 0) > 0:
            return 2  # DeepSeekV3
        return None
    if routing == "sigmoid":
        return 6 if renormalize else 8  # SigmoidRenorm / Sigmoid
    if routing == "softmax":
        return 4 if renormalize else 0  # RenormalizeNaive / Default
    return None



class _TPSwiGLUMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int,
                 reduce_results: bool = True):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size, intermediate_size],
        )
        # ``reduce_results=False`` lets the caller add this partial to the
        # routed-expert partial and all-reduce the sum once, instead of
        # all-reducing both separately. vLLM does the same.
        self.down_proj = RowParallelLinear(
            intermediate_size, hidden_size, reduce_results=reduce_results,
        )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x = x.reshape(-1, orig_shape[-1])
        out = self.down_proj(self.act_fn(self.gate_up_proj(x)))
        return out.view(*orig_shape[:-1], out.shape[-1])


class SharedExpertMoE(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        moe_intermediate_size: int,
        routing: Literal["sigmoid", "softmax"] = "softmax",
        correction_bias: bool = False,
        renormalize: bool = True,
        routed_scaling_factor: float = 1.0,
        use_grouped_topk: bool = False,
        num_expert_group: int = 1,
        topk_group: int = 1,
        force_grouped_topk_sorted: bool = False,
        keep_router_weights_fp32: bool = False,
        shared_expert_intermediate_size: int = 0,
        shared_expert_attr_name: str = "shared_expert",
        shared_expert_gate: bool = False,
        reduce_results: bool = True,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.routing = routing
        self.correction_bias = correction_bias
        self.renormalize = renormalize
        self.routed_scaling_factor = routed_scaling_factor
        self.use_grouped_topk = use_grouped_topk
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.keep_router_weights_fp32 = keep_router_weights_fp32

        tp = _tp_size()
        self.tp_size = tp
        self.intermediate_per_tp = moe_intermediate_size // tp

        self.gate = ReplicatedLinear(hidden_size, num_experts, bias=False)
        if correction_bias:
            self.gate.e_score_correction_bias = nn.Parameter(torch.zeros(num_experts))
            self.gate.e_score_correction_bias.weight_loader = (
                lambda p, w: p.data.copy_(w)
            )
        if use_grouped_topk:
            self.gate_linear = GateLinear()
            self.grouped_topk = GroupedTopK(
                scoring_func=routing,
                renormalize=renormalize,
                routed_scaling_factor=1.0,
                force_sorted=force_grouped_topk_sorted,
            )
        else:
            self.gate_linear = None
            self.grouped_topk = None

        n = self.intermediate_per_tp
        self.w13 = nn.Parameter(torch.empty(num_experts, 2 * n, hidden_size))
        self.w13.weight_loader = self._w13_weight_loader
        self.w2 = nn.Parameter(torch.empty(num_experts, hidden_size, n))
        self.w2.weight_loader = self._w2_weight_loader

        self.fused_experts = FusedExperts()
        self.allreduce = AllReduce()
        # ``reduce_results=False`` hands the un-reduced partial sum back to the
        # caller, which folds the collective into its next norm.
        self.reduce_results = reduce_results
        self._use_custom_op = False
        self._layer_name = ""

        # trtllm-gen BF16 MoE: what vLLM 0.26 runs for this MoE on Blackwell
        # (``FLASHINFER_TRTLLM`` unquantized backend ->
        # ``TrtLlmBf16ExpertsMonolithic``). It fuses routing, both GEMMs and the
        # weighted reduction into one kernel, replacing gate + top-k +
        # ``_fused_moe_kernel``.
        self._trtllm_routing = trtllm_routing_method_type(
            routing, renormalize, correction_bias, num_expert_group,
        )
        self.use_trtllm = (
            trtllm_bf16_moe_supported() and self._trtllm_routing is not None
        )
        self.trtllm_moe = (
            TrtLlmBf16MoE(
                num_experts=num_experts,
                top_k=top_k,
                intermediate_size_per_partition=self.intermediate_per_tp,
                routing_method_type=self._trtllm_routing,
                num_expert_group=num_expert_group if use_grouped_topk else None,
                topk_group=topk_group if use_grouped_topk else None,
                routed_scaling_factor=(
                    routed_scaling_factor if routed_scaling_factor != 1.0 else None
                ),
            )
            if self.use_trtllm
            else None
        )
        self._trtllm_weights_ready = False
        self._fused_ready = False
        self._fused_s = 0        # 0 = kernel picks the chunk split
        self._fused_cfg = -1     # <0 = kernel picks the expert-kernel shape
        self._fused_ws = None
        self._fused_ws_m = -1
        self._fw13 = None
        self._fw2t = None
        self._fsh_gu = None
        self._fsh_dnt = None
        self._fsh_gate = None
        self._fgate = None
        self._fargs = ()
        # Direct trtllm-gen dispatch + measured tactic + CUDA graphs. All three
        # are best-effort: every one of them falls back to the reference path.
        self._trt_op = None
        self._trt_op_tried = False
        self._trt_tactics = {}       # token count -> [tile_N, config]
        self._trt_scratch = None     # (topk_ids, expert_weights) empties
        self._graphs = {}            # token count -> (graph, static_in, static_out)
        self._graph_pool = None

        self.has_shared_expert = shared_expert_intermediate_size > 0
        self.shared_expert_attr_name = shared_expert_attr_name
        if self.has_shared_expert:
            setattr(
                self,
                shared_expert_attr_name,
                _TPSwiGLUMLP(
                    hidden_size, shared_expert_intermediate_size,
                    # Defer the shared expert's reduce so it can be folded into
                    # the routed output's -- one all-reduce per layer instead of
                    # two. Decode profile: cross_device_reduce_1stage was 18.8%
                    # of decode time at ~3 all-reduces per layer per step where
                    # 2 suffice.
                    reduce_results=(_tp_size() == 1),
                ),
            )
        if shared_expert_gate:
            self.shared_expert_gate = ReplicatedLinear(hidden_size, 1, bias=False)
        else:
            self.shared_expert_gate = None

    def _w13_weight_loader(
        self,
        param,
        loaded_weight,
        expert_id: int,
        is_w1: bool | None = None,
        is_gate: bool | None = None,
    ):
        if is_w1 is None and is_gate is None:
            raise TypeError("must pass is_w1 or is_gate to w13 loader")
        is_first = bool(is_w1 if is_w1 is not None else is_gate)
        rank = _tp_rank()
        n = self.intermediate_per_tp
        shard = loaded_weight.narrow(0, rank * n, n)
        offset = 0 if is_first else n
        param.data[expert_id, offset:offset + n, :].copy_(shard)

    def _w2_weight_loader(self, param, loaded_weight, expert_id: int):
        rank = _tp_rank()
        n = self.intermediate_per_tp
        param.data[expert_id].copy_(loaded_weight.narrow(1, rank * n, n))

    # Token count above which the fused small-token path hands back to
    # trtllm-gen. Below it every active expert sees only a couple of tokens, the
    # layer is pure weight streaming, and the fused kernel's ~19 us of host cost
    # beats anything that has to allocate.
    #
    # This used to be 256, chosen against a reference whose *measured* cost was
    # ~0.95 ms at every token count -- but that 0.95 ms was host-side dispatch,
    # not GPU work. Once the reference is replayed from a CUDA graph its real
    # cost shows: 0.24 ms at M=26, 0.37 ms at M=60, 0.51 ms at M=256, versus this
    # kernel's 0.24 / 0.41 / 0.98 ms. It does both routed GEMMs on CUDA cores at
    # ~15 TMAC/s where trtllm-gen uses tensor cores, so it only wins where the
    # layer is latency-bound rather than throughput-bound. Interleaved A/B under
    # bench conditions (``dev/ab_small.py``), fused vs graphed reference, in us:
    # M=1 47.9/58.4, M=8 123.9/126.0, M=16 189.4/177.1, M=26 246.8/236.6 -- so
    # the crossover is just above 8.
    _FUSED_MAX_TOKENS = 8

    # Token counts the reference path is replayed from a CUDA graph for. The
    # lower bound is where a graph starts beating the fused kernel; the upper
    # bound is where the layer becomes GPU-bound, past which the graph's static
    # in-copy and out-clone are pure additions (at M=16384 they cost 0.14 ms and
    # buy nothing, because the 0.31 ms of host work is already hidden behind
    # 1.74 ms of GPU work).
    _GRAPH_MAX_TOKENS = 4096
    # Graphs are cached per exact token count; a serving stack sees a handful of
    # hot shapes, and this bounds the pinned memory when it does not.
    _GRAPH_CACHE_MAX = 8

    # ---- trtllm-gen: direct call with a measured tactic --------------------

    def _trt_direct(self):
        """The raw trtllm-gen binding plus its two dummy inputs, or ``None``.

        ``topk_ids`` / ``expert_weights`` must be present but empty when routing
        logits are supplied; flashinfer allocates them per call, we allocate them
        once.
        """
        if not self._trt_op_tried:
            self._trt_op_tried = True
            if self.use_trtllm and self._trtllm_weights_ready:
                op = _trtllm_raw_op()
                if op is not None and hasattr(op, "trtllm_bf16_moe"):
                    dev = self.w13.device
                    self._trt_op = op
                    self._trt_scratch = (
                        torch.empty(0, dtype=torch.int32, device=dev),
                        torch.empty(0, dtype=torch.bfloat16, device=dev),
                    )
        return self._trt_op

    def _trt_call(self, x, logits, out, tactic):
        """One trtllm-gen MoE launch, bit-identical to what the wrapper issues."""
        topk_ids, expert_weights = self._trt_scratch
        self._trt_op.trtllm_bf16_moe(
            logits, None, topk_ids, expert_weights, x, self.w13, self.w2,
            None, None, None, None, out,
            self.num_experts, self.top_k, None, None, self.intermediate_per_tp,
            0, self.num_experts, None, self._trtllm_routing, True,
            _TRT_BLOCK_MAJOR_K, True, True, tactic, _TRT_SWIGLU, True, None,
        )
        return out

    def _trt_call_nofinalize(self, x, logits, tactic):
        """Same MoE launch, stopping before trtllm-gen's finalize.

        Returns ``(gemm2_output[P, H], expert_weights[M, K], idx[M*K])`` -- the
        weighted top-k reduction is left to :func:`moe_finalize_gated`, which
        folds the shared expert's gate projection and gated add into the same
        pass.
        """
        topk_ids, expert_weights = self._trt_scratch
        res = self._trt_op.trtllm_bf16_moe(
            logits, None, topk_ids, expert_weights, x, self.w13, self.w2,
            None, None, None, None, torch.empty(0, dtype=x.dtype, device=x.device),
            self.num_experts, self.top_k, None, None, self.intermediate_per_tp,
            0, self.num_experts, None, self._trtllm_routing, True,
            _TRT_BLOCK_MAJOR_K, False, True, tactic, _TRT_SWIGLU, True, None,
        )
        out = []
        for t in res[:3]:
            out.append(t if isinstance(t, torch.Tensor) else torch.from_dlpack(t))
        return out[0], out[1], out[2]

    def _epilogue_fusable(self) -> bool:
        """Whether the fused epilogue implements this layer's configuration.

        It needs the sigmoid-gated shared expert (that is the ``sigmoid(gate) *
        shared`` term it folds in), hidden=2048 (its thread mapping covers H
        exactly), and no routed scaling (trtllm-gen applies that inside its own
        finalize, which this replaces).
        """
        return (
            self._fsh_gate is not None
            and self.has_shared_expert
            and self.shared_expert_gate is not None
            and self.hidden_size == 2048
            and self.routed_scaling_factor == 1.0
            and 1 <= self.top_k <= 32
            and not self.correction_bias
            and self.gate_linear is None
        )

    def _trt_candidates(self, m: int):
        """A short, deterministic tile ladder to time: up to two configs per
        distinct token tile. The full valid set is 356 tactics at M=445, and the
        spread *within* one tile is a few percent while the spread *between*
        tiles is 20-30%, so the tile is what has to be measured."""
        key = (_TRT_BF16, _TRT_BF16, 0, self.top_k, self.hidden_size,
               self.intermediate_per_tp, self.num_experts, _TRT_SWIGLU, True,
               _TRT_BLOCK_MAJOR_K, False, m, False)
        try:
            valid = self._trt_op.trtllm_get_valid_moe_configs(*key)
        except Exception:
            return []
        per_tile = {}
        for t in valid:
            t = list(t)
            if len(t) != 2:
                continue
            per_tile.setdefault(t[0], []).append(t)
        out = []
        for tile in sorted(per_tile):
            out.extend(per_tile[tile][:2])
        return out[:8]

    @staticmethod
    def _graph_time_ms(fn, iters: int = 10) -> float:
        """GPU time of ``fn`` measured through a throwaway CUDA graph.

        Timing these tactics directly does not work: trtllm-gen's launcher costs
        280-400 us of *host* time per call against 250-500 us of GPU work, so
        back-to-back events measure the host and every tactic looks identical,
        while syncing per call adds a host cost that itself varies by tactic.
        Replaying a captured graph has ~10 us of host cost, so the GPU is
        unambiguously the thing being measured. ``float('inf')`` if the call
        cannot be captured.
        """
        try:
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side), torch.no_grad():
                for _ in range(3):
                    fn()
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.no_grad(), torch.cuda.graph(g):
                fn()
            g.replay()
            torch.cuda.synchronize()
            ev0 = torch.cuda.Event(enable_timing=True)
            ev1 = torch.cuda.Event(enable_timing=True)
            ev0.record()
            for _ in range(iters):
                g.replay()
            ev1.record()
            torch.cuda.synchronize()
            ms = ev0.elapsed_time(ev1) / iters
            del g
            return ms
        except Exception:
            return float("inf")

    def _tactic_for(self, m: int, x):
        """Pick trtllm-gen's tactic for this token count by timing it once.

        The launcher's own fallback is ``*selected.begin()`` -- the smallest
        valid token tile -- which measures 1.13-1.27x slower than the best tile
        across the mid-range and 1.18x slower at M=16384. What is timed is the
        *whole* forward, not the MoE call alone: the epilogue that follows it
        differs between paths, and picking on the MoE call alone chose a tactic
        1.6% off the end-to-end best at M=16384.

        Candidates are checked against the fallback's output first and dropped
        unless they agree, so a bad tile can only cost time, never correctness.
        """
        cached = self._trt_tactics.get(m)
        if cached is not None:
            return cached
        # Seeded before anything else so the recursive ``_forward_eager`` calls
        # below read a resolved value instead of re-entering this function.
        self._trt_tactics[m] = _TRT_FALLBACK_TACTIC
        best = _TRT_FALLBACK_TACTIC
        try:
            with torch.no_grad():
                ref = self._forward_eager(x).float()
                torch.cuda.synchronize()
                tol = 1e-2 + 1e-2 * ref.abs().max().item()
                best_ms = self._graph_time_ms(lambda: self._forward_eager(x))
                for t in self._trt_candidates(m):
                    if t == _TRT_FALLBACK_TACTIC:
                        continue
                    self._trt_tactics[m] = t
                    try:
                        got = self._forward_eager(x)
                        torch.cuda.synchronize()
                    except Exception:
                        continue
                    if (got.float() - ref).abs().max().item() > tol:
                        continue
                    ms = self._graph_time_ms(lambda: self._forward_eager(x))
                    if ms < best_ms:
                        best_ms, best = ms, t
        except Exception:
            best = _TRT_FALLBACK_TACTIC
        self._trt_tactics[m] = best
        return best

    # ---- CUDA graph replay of the whole reference path ---------------------

    def _graph_entry(self, m: int, x, body):
        """``(graph, static_in, static_out, keepalive)`` for this token count,
        or ``None``.

        ``body`` is the path to capture. Only ``_forward_eager`` is passed today
        -- the reference path, whose C++ launcher costs ~0.62 ms of host work per
        call; capturing the fused path was measured and rejected (see
        ``forward_impl``). ``body`` stays a parameter because the fused path needs
        a private workspace when captured and that plumbing is the interesting
        part to keep.

        Capture is one-shot per token count and every failure mode is sticky, so a
        shape that cannot be captured simply runs eagerly forever after.
        """
        ent = self._graphs.get(m)
        if ent is not None:
            return ent or None
        if (len(self._graphs) >= self._GRAPH_CACHE_MAX
                or torch.cuda.is_current_stream_capturing()):
            return None
        try:
            static_in = torch.empty(m, self.hidden_size, dtype=x.dtype,
                                    device=x.device)
            static_in.copy_(x)
            keep = None
            # ``==`` not ``is``: a bound method is a fresh object per attribute
            # access, so ``self.f is self.f`` is False.
            if body == self._forward_fused:
                # Its own scratch, referenced by the entry we return, so nothing
                # can resize it out from under the captured kernel arguments.
                keep = torch.empty(self._fused_ws_bytes(m), dtype=torch.uint8,
                                   device=x.device)
                body = lambda t, _w=keep: self._forward_fused(t, _w)
            # Warm up off the capture stream, so cuBLAS's heuristic picks and
            # the caching allocator's block sizes are settled before capture
            # freezes them. (The tactic is already resolved -- ``forward_impl``
            # does that before it ever asks for a graph.)
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side), torch.no_grad():
                for _ in range(3):
                    body(static_in)
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            if self._graph_pool is None:
                self._graph_pool = torch.cuda.graph_pool_handle()
            graph = torch.cuda.CUDAGraph()
            with torch.no_grad(), torch.cuda.graph(graph, pool=self._graph_pool):
                static_out = body(static_in)
            # ``keep`` is in the tuple purely to hold a reference.
            ent = (graph, static_in, static_out, keep)
        except Exception:
            # Anything at all -- unsupported launch, OOM, capture-time sync --
            # and this shape stays eager.
            self._graphs[m] = ()
            return None
        self._graphs[m] = ent
        return ent

    def _fused_supported(self) -> bool:
        """Whether the fused kernel implements this layer's exact configuration."""
        return (
            self.hidden_size == 2048
            and self.intermediate_per_tp == 512
            and self.routing == "softmax"
            and self.renormalize
            and not self.correction_bias
            and not self.use_grouped_topk
            and self.routed_scaling_factor == 1.0
            and self.top_k <= 32
            and self.top_k <= self.num_experts
            and self.has_shared_expert
            and self.shared_expert_gate is not None
            and self.tp_size == 1
            # The kernel is specialized on hidden=2048 and an intermediate of
            # 512 shared by the routed and the shared expert.
            and getattr(self, self.shared_expert_attr_name)
            .gate_up_proj.weight.shape[0] == 2 * 512
        )

    def _fused_ws_bytes(self, m: int) -> int:
        """Size of the fused path's scratch: router logits, the per-expert work
        lists, the compacted active-expert list, the shared-expert gate and the
        fp32 output accumulator. Kept in sync with the ``take()`` sequence in
        ``moe_fused.cu``."""
        e = self.num_experts
        need = 0
        for nbytes in (m * e * 2, e * 4, 4, e * 4, e * m * 4, e * m * 4, m * 4,
                       m * self.hidden_size * 4):
            need += (nbytes + 255) & ~255
        return need

    def _fused_workspace(self, m: int) -> torch.Tensor:
        """The shared scratch buffer for *eager* fused calls.

        One cached allocation carved up inside the kernel -- a ``torch.empty``
        costs ~1.9 us of host time, which is not free against a ~40 us total.

        A captured graph must *not* use this: the buffer is resized when a larger
        token count comes through, which frees the storage a previously captured
        graph still has baked into its kernel arguments. ``_graph_entry`` hands
        the fused path a private workspace for that reason -- it no longer
        captures the fused path at all, but the hazard is real and the guard is
        kept.
        """
        if m == self._fused_ws_m:
            return self._fused_ws
        need = self._fused_ws_bytes(m)
        ws = self._fused_ws
        if ws is None or ws.numel() < need:
            ws = torch.empty(need, dtype=torch.uint8, device=self.w13.device)
            self._fused_ws = ws
        self._fused_ws_m = m
        return ws

    def process_weights_after_loading(self) -> None:
        """Build both weight layouts this layer needs: the fused path's and
        trtllm-gen's.

        Shuffle expert weights into trtllm-gen's 4D BlockMajorK layout.

        Mirrors vLLM's ``convert_to_unquantized_kernel_format`` for the
        ``FLASHINFER_TRTLLM`` backend, which runs once after loading. The
        original ``[E, 2*I, H]`` / ``[E, H, I]`` tensors are replaced, so the
        Triton path is unavailable afterwards -- guarded by ``use_trtllm``.
        """
        if self._trtllm_weights_ready or self._fused_ready:
            return

        # The shared-expert gate weight, flattened, is what the fused epilogue
        # needs to do the gate projection itself. Kept independent of the
        # small-token path's own requirements, which are stricter.
        if self.shared_expert_gate is not None:
            self._fsh_gate = self.shared_expert_gate.weight.data.reshape(-1)

        # Layouts for the fused path. Both GEMMs want their reduction dimension
        # contiguous: w13 already is ([E, 2I, H], reduce over H), but w2 is
        # [E, H, I] and reduces over I, so a chunk of I is strided. Transposing
        # it to [E, I, H] makes an intermediate-column chunk a contiguous slab
        # and turns phase 2 into fully coalesced row streaming.
        if self._fused_supported():
            self._fw13 = self.w13.data
            self._fw2t = self.w2.data.transpose(1, 2).contiguous()
            shared = getattr(self, self.shared_expert_attr_name)
            self._fsh_gu = shared.gate_up_proj.weight.data
            self._fsh_dnt = shared.down_proj.weight.data.t().contiguous()
            self._fsh_gate = self.shared_expert_gate.weight.data.reshape(-1)
            self._fgate = self.gate.weight.data
            # Pre-bound argument tuple: unpacking one tuple is cheaper than six
            # attribute lookups, and at M=1 Python is the critical path.
            self._fargs = (self._fgate, self._fw13, self._fw2t, self._fsh_gu,
                           self._fsh_dnt, self._fsh_gate)
            self._fused_ready = True

        if not self.use_trtllm:
            return
        w13, w2 = prepare_trtllm_bf16_moe_weights(self.w13.data, self.w2.data)
        self.w13 = nn.Parameter(w13, requires_grad=False)
        self.w2 = nn.Parameter(w2, requires_grad=False)
        self._trtllm_weights_ready = True

    def _route(self, router_logits: torch.Tensor):
        if self.grouped_topk is not None:
            e_score_correction_bias = (
                self.gate.e_score_correction_bias if self.correction_bias else None
            )
            return self.grouped_topk(
                router_logits,
                e_score_correction_bias,
                num_expert_group=self.num_expert_group,
                topk_group=self.topk_group,
                topk=self.top_k,
            )
        if self.routing == "sigmoid":
            scores = torch.sigmoid(router_logits.float())
            if self.correction_bias:
                scores_for_choice = scores + self.gate.e_score_correction_bias
                _, topk_ids = scores_for_choice.topk(self.top_k, dim=-1)
                topk_weights = scores.gather(-1, topk_ids)
            else:
                topk_weights, topk_ids = scores.topk(self.top_k, dim=-1)
        else:
            scores = torch.softmax(router_logits.float(), dim=-1)
            topk_weights, topk_ids = scores.topk(self.top_k, dim=-1)
        if self.renormalize:
            topk_weights = topk_weights / (
                topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            )
        return topk_weights.to(torch.float32), topk_ids.to(torch.int32)

    # Token count below which the shared-expert gate projection is folded into
    # the epilogue kernel instead of run as its own gemv. The gemv is a
    # ``[hidden] -> [1]`` dot whose 5.33 us is all launch latency, so folding
    # wins while that dominates; past this the projection is a real GEMM and the
    # kernel's per-tile recomputation of the dot would start to cost more than
    # the launch it saves.
    _FUSE_GATE_MAX_TOKENS = 256

    def _shared_expert_output(
        self, hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Return ``(shared_output, raw_gate)``, both ``None`` without a shared expert.

        The gate's sigmoid and the scaling are deliberately *not* applied here:
        the caller folds them into the routed-output add with one kernel (see
        :func:`moe_shared_gate_add`). ``raw_gate`` is ``None`` when there is
        nothing for the caller to apply -- either the shared expert has no gate,
        or the gate projection itself is being folded into that same kernel.
        """
        if not self.has_shared_expert:
            return None, None
        shared_mlp = getattr(self, self.shared_expert_attr_name)
        out = shared_mlp(hidden_states)
        if self.shared_expert_gate is None:
            return out, None
        if hidden_states.shape[0] <= self._FUSE_GATE_MAX_TOKENS:
            return out, None
        return out, self.shared_expert_gate(hidden_states)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, self.hidden_size)
        if self._use_custom_op:
            # All-reduce stays outside the opaque op so the decoder's fused
            # AR+norm can still match it. vLLM / KimiMoE do the same.
            output = torch.ops.fastkernels.moe_forward(
                hidden_states, self._layer_name,
            )
            if self.tp_size > 1 and self.reduce_results:
                output = self.allreduce(output)
            return output.view(orig_shape)
        return self.forward_impl(hidden_states).view(orig_shape)

    def forward_impl(self, hidden_states: torch.Tensor) -> torch.Tensor:
        m = hidden_states.shape[0]
        # The hand-written kernel owns the small token counts, where the layer is
        # latency-bound and its four launches beat anything that allocates. Above
        # that the layer runs the reference's own kernels: trtllm-gen is at the
        # memory roofline for the routed GEMMs (0.460 ms against a 0.463 ms floor
        # at M=445), so there is nothing to win inside them -- what is left is the
        # ~0.62 ms of per-call host work its launcher does *around* them, which is
        # what the graph removes.
        fused = (self._fused_ready and 0 < m <= self._FUSED_MAX_TOKENS
                 # The kernel streams activations with 16-byte vector loads.
                 # Every tensor it allocates itself is over-aligned, but a caller
                 # could hand in a slice that is not, and that would fault rather
                 # than degrade.
                 and hidden_states.data_ptr() % 16 == 0)
        capturing = torch.cuda.is_current_stream_capturing()
        if (not fused and self._trt_direct() is not None
                and not self.correction_bias and not capturing):
            # Resolved once per token count, outside the eager path so tuning can
            # freely call it, synchronize and capture throwaway graphs -- none of
            # which is legal once the caller has started capturing.
            self._tactic_for(m, hidden_states)
        # ``capturing`` guards the case where *our caller* is capturing a graph --
        # a serving stack capturing its decode step is the normal case, and
        # replaying a graph inside a capture is an error. Both eager paths below
        # are fully capturable, so this degrades rather than fails.
        # Graphs are for the reference path only. Capturing the *fused* path was
        # measured and rejected: at M=1 it goes 47.9 -> 50.2 us, because its four
        # kernels are not actually launch-gap-bound. (An earlier reading said they
        # were, from replaying in a tight loop with no L2 flush -- but 69 MB of
        # expert weights fits in this B200's 126 MB L2, so those replays were
        # reading cache. Under the bench's per-call flush the gap disappears.)
        if (not fused and 0 < m <= self._GRAPH_MAX_TOKENS and not capturing
                and self._trt_direct() is not None):
            ent = self._graph_entry(m, hidden_states, self._forward_eager)
            if ent is not None:
                graph, static_in, static_out = ent[:3]
                static_in.copy_(hidden_states)
                graph.replay()
                # The graph writes into pool-owned memory that the next replay
                # overwrites, so the caller gets a copy it owns.
                return static_out.clone()
        if fused:
            return self._forward_fused(hidden_states)
        return self._forward_eager(hidden_states)

    def _forward_eager(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if (self._trt_direct() is not None and self.use_trtllm
                and self._epilogue_fusable()):
            return self._forward_eager_fused_epilogue(hidden_states)
        shared_output, shared_gate = self._shared_expert_output(hidden_states)
        if self.gate_linear is not None:
            router_logits = self.gate_linear(
                hidden_states,
                self.gate.weight,
                out_dtype=torch.float32,
            )
        else:
            router_logits = self.gate(hidden_states)

        if self.use_trtllm:
            # Routing, both GEMMs and the weighted reduction happen inside the
            # kernel, including ``routed_scaling_factor``.
            if self._trt_direct() is not None and not self.correction_bias:
                # Same kernel, same arguments, one less Python layer -- and the
                # tactic ``forward_impl`` measured for this token count instead
                # of the launcher's smallest-tile fallback.
                routed_output = self._trt_call(
                    hidden_states, router_logits,
                    torch.empty_like(hidden_states),
                    self._trt_tactics.get(hidden_states.shape[0],
                                          _TRT_FALLBACK_TACTIC),
                )
            else:
                routed_output = self.trtllm_moe(
                    hidden_states,
                    self.w13,
                    self.w2,
                    router_logits,
                    routing_bias=(
                        self.gate.e_score_correction_bias
                        if self.correction_bias
                        else None
                    ),
                )
        else:
            topk_weights, topk_ids = self._route(router_logits)
            if not self.keep_router_weights_fp32:
                topk_weights = topk_weights.to(hidden_states.dtype)

            routed_output = self.fused_experts(
                hidden_states, self.w13, self.w2,
                topk_weights, topk_ids, self.num_experts,
            )
            if self.routed_scaling_factor != 1.0:
                routed_output = routed_output * self.routed_scaling_factor

        # Add the shared expert's *unreduced* partial first, then all-reduce the
        # sum once. Both terms are per-rank partial sums over the same output
        # space, so summing before the reduce is exact, and it halves the
        # all-reduce count for layers that have a shared expert. When the shared
        # expert is gated, its sigmoid, the scaling and the add are a single
        # kernel -- three separate elementwise launches per layer is what
        # Inductor fuses away for vLLM, and at batch 1 that is pure overhead.
        if shared_output is None:
            output = routed_output
        elif self.shared_expert_gate is None:
            output = routed_output + shared_output
        elif shared_gate is None:
            # Small batch: the epilogue kernel projects the gate itself.
            output = moe_shared_gate_add(
                routed_output, shared_output,
                hidden_states=hidden_states,
                gate_weight=self.shared_expert_gate.weight,
            )
        else:
            output = moe_shared_gate_add(
                routed_output, shared_output, shared_gate,
            )
        if self.tp_size > 1 and self.reduce_results and not self._use_custom_op:
            output = self.allreduce(output)
        return output

    def _forward_eager_fused_epilogue(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """The reference's kernels for the two routed GEMMs, our own epilogue.

        The routed GEMMs are at the memory roofline and stay trtllm-gen's. What
        the reference spends *around* them is three separate passes over the
        [M, H] activation -- its finalize (a weighted top-k row reduction), the
        shared expert's gate gemv, and ``routed + shared * sigmoid(gate)``. At
        M=16384 those are 0.167 + 0.019 + 0.031 = 0.217 ms of the 1.50 ms total,
        and trtllm's finalize alone moves 738 MB at 4.4 TB/s where this machine
        reaches ~6.9 TB/s. One kernel does all three in a single pass.
        """
        shared_mlp = getattr(self, self.shared_expert_attr_name)
        shared_output = shared_mlp(hidden_states)
        router_logits = self.gate(hidden_states)
        g2, expert_w, idx = self._trt_call_nofinalize(
            hidden_states, router_logits,
            self._trt_tactics.get(hidden_states.shape[0], _TRT_FALLBACK_TACTIC),
        )
        output = torch.empty_like(hidden_states)
        _C.moe_finalize_gated(g2, idx, expert_w, shared_output, hidden_states,
                              self._fsh_gate, output)
        if self.tp_size > 1 and self.reduce_results and not self._use_custom_op:
            output = self.allreduce(output)
        return output

    def _forward_fused(self, hidden_states: torch.Tensor, ws=None) -> torch.Tensor:
        """The whole layer -- router, top-k, both routed GEMMs, the shared
        expert and the gated add -- in one call.

        Every kernel launch is issued from inside the extension, so the Python
        side is deliberately just a workspace lookup, an output allocation and
        one call.
        """
        m = hidden_states.shape[0]
        out = torch.empty_like(hidden_states)
        _C.shared_expert_moe_fused(
            hidden_states, *self._fargs, out,
            self._fused_workspace(m) if ws is None else ws, self.top_k,
            self._fused_s, self._fused_cfg,
        )
        return out
