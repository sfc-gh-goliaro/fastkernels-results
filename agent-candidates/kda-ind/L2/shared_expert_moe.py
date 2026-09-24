"""Shared-expert MoE for Qwen3-Next on Blackwell: launch-bound at decode sizes.

Same math as the baseline, same building blocks, two changes where the
measurements say the time actually goes.

**The small-token forward is host-bound, not device-bound.** One
``trtllm_bf16_moe`` invocation costs 650-750 us of *host* time independent of
token count (``profile/probe_hostoverhead.py``: 744.5 us at T=1, 751.4 us at
T=60, all of it enqueue -- the sync adds 1.5 us), against 61 us of device work
at T=1 and 380 us at T=60 (``profile/probe_kernels.py``). ~200 us of that host
cost is the FlashInfer Python wrapper (MoERunner construction,
``_make_tuning_config``, ``AutoTuner.choose_one``, three ``torch.empty``); the
rest is the C++ launch path. ``fastkernels/infra/engine.py`` measures the same
thing from the other side -- "the trtllm-gen fused MoE call costs 412-676 us of
CPU dispatch (autotuner lookup + routing config + cooperative launch) ... that a
graph replay pays once" -- and its own remedy is a CUDA graph.

So at T <= 2048 this module replays a captured graph of ``forward_impl`` instead
of re-enqueueing seven launches. Whole-forward replay measures 42.1 us at T=1
against 1051 us eager, and 513.2 us at T=445 against 1066 us
(``profile/probe_graph_overlap.py``, max error vs eager 0.0). The crossover is
at ~2-4 k tokens (2.37x at 445, 1.96x at 1024, 1.48x at 2048, 0.98x at 4096 --
``profile/probe_threshold.py``), hence the threshold. This is the same
shape-keyed, in-``forward`` capture the ``L4/yolov10`` baseline uses for the same
stated reason, except that this one never hands its static output back to the
caller.

**The routed kernel has no headroom worth contesting.** It moves 60 MiB of
expert weights in 13.5 us at T=1 (4.6 TB/s) and 2.07 GiB in 326 us at T=60
(6.4 TB/s) against a B200 speed-of-light of ~8 TB/s, so it is left exactly as
the baseline runs it.

**The shared expert's tail looks recoverable and is not, on this stack.** The
down-projection plus the gated-add epilogue move 274 MiB in 59-64 us at T=16384,
and 128 MiB of that is a round trip through ``shared_output`` that exists only
because the two ops are separate launches. :func:`shared_down_gate_add` fuses
them into ``routed + sigmoid(gate) * (h @ W2^T)``, touching 146 MiB whose
speed-of-light is 18.3 us. It is correct at every token count and in both gate
modes, and it is **off by default**, because it measures slower than the pair it
replaces: 72.5 us at T=16384 against 59.3 us, the best of 216 swept tile
configurations. Nsight Compute says why -- 12.2% achieved occupancy, one block
per SM by both the register and the shared-memory limit, 250 registers per thread
for a 128x256 fp32 accumulator living in the register file, DRAM at 16.5% and the
SM pipes at 26%, i.e. latency-bound for want of resident warps. On Blackwell that
accumulator belongs in TMEM, which is what lets cuBLAS reach ~58% of bf16 peak on
the same GEMM; getting there from Triton needs the warp-specialized TMA path, a
different kernel. Shrinking tiles to buy occupancy is worse (85-142 us, because
each output-column tile re-reads ``h``), and pre-transposing ``W2`` to K-major
changes nothing, so the transpose was never the problem. Full write-up in
``profile/shared_down_gate_add_v1_t16384/REPORT.md``.

Its apparent win below ~4 k tokens -- 22-26 us against the pair's ~31 us -- is
host launch cost, one launch instead of two, at token counts where neither op
does meaningful device work. The graph path already removes all of that, so the
two layers overlap rather than compose, and enabling the kernel underneath the
graph traded ``max_abs_error = 0`` for a 4-10 us regression. Set
``FASTKERNELS_SEMOE_FUSED_EPILOGUE=1`` to enable it anyway.

**Scope of each win.** The graph path is a standalone-benchmark win. Under
``fastkernels eval`` on B200 the engine records the entire model forward -- this
operator included -- inside one outer ``torch.cuda.graph`` and skips
``_compile_model()``, so operator-level launch overhead is already zero
end-to-end and the ``is_current_stream_capturing()`` guard correctly makes this
path inert there. Nothing in this module improves the end-to-end path.

Every number quoted here is reproducible from a script under ``profile/``; see
``profile/phase1_measurements.md``.

Conditions the graph fast path relies on
----------------------------------------

Replay reuses one fixed set of device addresses, so the path is guarded down to
the cases below and falls through to the eager path -- which is always correct
and is the only path the numbers above are measured against -- for everything
else.

* Inference only. Autograd would record through the captured region, so
  grad-enabled calls run eager.
* Fixed token count, dtype and device per cached entry, and ``tp_size == 1``
  (capturing a collective needs a capture-safe communicator).
* One in-flight ``forward`` per instance. The static input and output are shared
  by every replay of an entry, so two overlapping calls -- concurrently, or on
  two streams without an explicit event dependency -- would race. Callers that
  need concurrency must serialize on one stream or add their own ordering.
* Parameters and buffers must stay alive at stable addresses after capture: the
  graph records their pointers. Moving the module or replacing a parameter's
  storage invalidates every captured entry. ``process_weights_after_loading``
  replaces ``w13``/``w2``, so it must run *before* the first capture, which it
  does -- the harness calls it during setup.
* No other CUDA capture and no unrelated CUDA work anywhere in the process while
  a capture is underway; PyTorch permits only one capture at a time.
* Captured entries are never evicted, only retired past a bounded count. Freeing
  a graph's private pool has no synchronization guarantee behind it, so an
  eviction racing an in-flight replay would be unordered.
* The FlashInfer tactic is chosen once, during warm-up, and baked in. Anything
  that would change the tactic, the workspace size or the routing method needs a
  fresh capture.
* A capture that fails part-way is not generally recoverable, so the first such
  failure disables graphing for the instance rather than retrying on a later
  shape.

Environment switches
--------------------

The four performance switches are read once, in ``__init__``, so an A/B is a
separate process rather than a mutated module.

===================================================  =======  ====================
Variable                                             Default  Effect
===================================================  =======  ====================
``FASTKERNELS_SEMOE_CUDA_GRAPH``                     ``1``    Graph fast path.
``FASTKERNELS_SEMOE_GRAPH_MAX_TOKENS``               ``2048`` Graph eligibility bound.
``FASTKERNELS_SEMOE_FUSED_EPILOGUE``                 ``0``    Fused shared-expert tail.
``FASTKERNELS_SEMOE_FUSED_MAX_TOKENS``               ``4096`` Bound for the above.
===================================================  =======  ====================

One further switch is **not** a performance switch and is **not** read in
``__init__``: ``FASTKERNELS_SEMOE_FAULT_INJECT_CAPTURE`` (default off) is read at
the capture site, once per capture attempt, and makes every capture fail. It exists
because the capture-failure recovery path is the one branch here that no input can
provoke, so without it that path cannot be exercised through the real harness. It is
read late precisely so a test can toggle it per attempt rather than per instance.
"""

from __future__ import annotations

import os
from typing import Literal

import torch
import torch.nn as nn

import triton
import triton.language as tl

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


def _env_flag(name: str, default: bool = True) -> bool:
    return os.environ.get(name, "1" if default else "0") != "0"


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


@triton.jit
def _shared_down_gate_add_kernel(
    h_ptr,          # [T, I]  post-SiLU shared-expert intermediate
    w2_ptr,         # [H, I]  shared down_proj weight (out_features, in_features)
    routed_ptr,     # [T, H]  routed-expert output
    gate_ptr,       # [T]     raw pre-sigmoid gate, when already projected
    x_ptr,          # [T, H]  hidden states, when the gate is projected in-kernel
    gw_ptr,         # [H]     gate weight, when the gate is projected in-kernel
    out_ptr,        # [T, H]
    n_tokens,
    hidden,
    inter,
    stride_h_t, stride_h_i,
    stride_w_o, stride_w_i,
    stride_r_t, stride_r_h,
    stride_gate,
    stride_x_t, stride_x_h,
    stride_o_t, stride_o_h,
    W2_IS_KN: tl.constexpr,
    FUSE_GATE: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_I: tl.constexpr,
    BLOCK_G: tl.constexpr,
):
    """``out = routed + sigmoid(gate) * (h @ w2^T)`` in one pass over the output.

    One program owns a [BLOCK_T, BLOCK_H] tile of the output and never writes the
    down-projection to memory: the accumulator goes straight from the MMA into
    the gated add. That is the 128 MiB of ``shared_output`` round trip the
    two-kernel composition pays at T=16384.

    ``W2_IS_KN`` selects how the weight is fed to the MMA. ``down_proj.weight`` is
    [H, I], i.e. N-major, so the B operand has to be transposed per K-step; with a
    [I, H] copy it is already K-major and loads straight into the MMA layout.
    """
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_t = offs_t < n_tokens
    mask_h = offs_h < hidden

    # Down projection for this tile, fp32 accumulator throughout. The baseline
    # rounds this to bf16 on the way out of ``down_proj`` and reads it back as
    # fp32 in the epilogue; keeping it in the accumulator drops that one
    # rounding.
    acc = tl.zeros((BLOCK_T, BLOCK_H), dtype=tl.float32)
    for i0 in range(0, inter, BLOCK_I):
        offs_i = i0 + tl.arange(0, BLOCK_I)
        mask_i = offs_i < inter
        a = tl.load(
            h_ptr + offs_t[:, None] * stride_h_t + offs_i[None, :] * stride_h_i,
            mask=mask_t[:, None] & mask_i[None, :], other=0.0,
        )
        if W2_IS_KN:
            b = tl.load(
                w2_ptr + offs_i[:, None] * stride_w_i + offs_h[None, :] * stride_w_o,
                mask=mask_i[:, None] & mask_h[None, :], other=0.0,
            )
        else:
            b = tl.trans(tl.load(
                w2_ptr + offs_h[:, None] * stride_w_o + offs_i[None, :] * stride_w_i,
                mask=mask_h[:, None] & mask_i[None, :], other=0.0,
            ))
        acc = tl.dot(a, b, acc)

    if FUSE_GATE:
        # gate[t] = x[t] . gate_weight, in fp32, recomputed per output tile
        # rather than staged through memory -- the same trade
        # ``moe_shared_gate_add`` makes, for the same reason: at these token
        # counts the separate gemv is 5.33 us of pure launch latency. Chunked
        # over hidden because a [BLOCK_T, hidden] fp32 tile would not fit.
        gate = tl.zeros((BLOCK_T,), dtype=tl.float32)
        for g0 in range(0, hidden, BLOCK_G):
            offs_g = g0 + tl.arange(0, BLOCK_G)
            mask_g = offs_g < hidden
            xr = tl.load(
                x_ptr + offs_t[:, None] * stride_x_t + offs_g[None, :] * stride_x_h,
                mask=mask_t[:, None] & mask_g[None, :], other=0.0,
            ).to(tl.float32)
            wg = tl.load(gw_ptr + offs_g, mask=mask_g, other=0.0).to(tl.float32)
            gate += tl.sum(xr * wg[None, :], axis=1)
    else:
        gate = tl.load(gate_ptr + offs_t * stride_gate, mask=mask_t,
                       other=0.0).to(tl.float32)
    scale = tl.sigmoid(gate)

    routed = tl.load(
        routed_ptr + offs_t[:, None] * stride_r_t + offs_h[None, :] * stride_r_h,
        mask=mask_t[:, None] & mask_h[None, :], other=0.0,
    ).to(tl.float32)
    out = routed + acc * scale[:, None]
    tl.store(
        out_ptr + offs_t[:, None] * stride_o_t + offs_h[None, :] * stride_o_h,
        out.to(out_ptr.dtype.element_ty),
        mask=mask_t[:, None] & mask_h[None, :],
    )


# Tile configurations, from the 216-point sweep in
# ``profile/probe_fused_epilogue.py --sweep`` (table in
# ``profile/phase1_measurements.md``). Keyed by the smallest token count each
# config is used from. Below ~1 k tokens the kernel is launch-bound and every
# config in the sweep landed within 1% of 18.4 us, so the small-T entry is chosen
# for a grid that does not waste programs on padding.
#
# The large-T entry is deliberately the *smaller* of the two front-runners.
# 128x256 with 4 pipeline stages needs 4*(128+256)*64*2 = 193 KiB of shared
# memory out of the 227 KiB an SM has, which caps the kernel at one block per SM
# -- 8 warps of a possible 64, a measured 12.2% achieved occupancy, and 250
# registers per thread (``profile/shared_down_gate_add_v1_t16384``). 128x128 with
# 3 stages measures the same at T=2048 (19.8 us vs 18.8 us) and slightly better
# at T=16384 (72.5 us vs 73.1 us) with half the shared-memory footprint.
_FUSED_TILES: tuple[tuple[int, dict], ...] = (
    (0, dict(BLOCK_T=32, BLOCK_H=128, BLOCK_I=64, BLOCK_G=512,
             num_warps=4, num_stages=4)),
    (1024, dict(BLOCK_T=128, BLOCK_H=128, BLOCK_I=64, BLOCK_G=512,
                num_warps=4, num_stages=3)),
)


def _fused_tile_config(n_tokens: int) -> dict:
    cfg = _FUSED_TILES[0][1]
    for lo, candidate in _FUSED_TILES:
        if n_tokens >= lo:
            cfg = candidate
    return cfg


def shared_down_gate_add(
    routed: torch.Tensor,
    h: torch.Tensor,
    w2: torch.Tensor,
    gate: torch.Tensor | None = None,
    hidden_states: torch.Tensor | None = None,
    gate_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """``routed + sigmoid(gate) * (h @ w2.T)`` -- the shared expert's whole tail.

    Fuses what the baseline runs as ``down_proj`` followed by
    :func:`moe_shared_gate_add`. Not bit-exact against that composition and not
    claimed to be: the tiling changes the accumulation tree of the projection,
    and the bf16 rounding of ``shared_output`` disappears.

    Args:
        routed: [T, H] routed-expert output.
        h: [T, I] post-SiLU shared-expert intermediate.
        w2: [H, I] shared ``down_proj`` weight, used in place -- no transposed
            copy is materialized.
        gate: [T] or [T, 1] raw (pre-sigmoid) gate, already projected. May be a
            strided column.
        hidden_states, gate_weight: [T, H] and [H] -- supply these *instead of*
            ``gate`` to have the kernel project the gate itself.

    Returns a fresh [T, H] tensor in ``routed``'s dtype.
    """
    n_tokens, hidden = routed.shape
    if h.shape[0] != n_tokens:
        raise ValueError(
            f"h has {h.shape[0]} rows but routed has {n_tokens}",
        )
    inter = h.shape[1]
    # Accept either the native [H, I] ``down_proj.weight`` or a [I, H] copy of it.
    w2_is_kn = tuple(w2.shape) == (inter, hidden) and hidden != inter
    if not w2_is_kn and w2.shape != (hidden, inter):
        raise ValueError(
            f"w2 {tuple(w2.shape)} must be (hidden, inter) = {(hidden, inter)} "
            f"or its transpose",
        )
    out = torch.empty_like(routed)
    if n_tokens == 0:
        return out

    fuse_gate = gate is None
    # Triton needs a real tensor for every pointer argument even on the branch
    # that never dereferences it, so the unused side aliases ``routed``.
    if fuse_gate:
        if hidden_states is None or gate_weight is None:
            raise ValueError("pass gate, or both hidden_states and gate_weight")
        if hidden_states.shape != (n_tokens, hidden):
            raise ValueError(
                f"hidden_states {tuple(hidden_states.shape)} must match routed "
                f"{(n_tokens, hidden)} for the fused gate projection",
            )
        gate_flat, stride_gate = routed, 0
        x, gw = hidden_states, gate_weight.reshape(-1)
        stride_x_t, stride_x_h = x.stride(0), x.stride(1)
    else:
        gate_flat = gate.reshape(-1) if gate.dim() > 1 else gate
        stride_gate = gate_flat.stride(0) if gate_flat.dim() else 0
        x, gw = routed, routed
        stride_x_t, stride_x_h = 0, 0

    cfg = dict(_fused_tile_config(n_tokens))
    num_warps = cfg.pop("num_warps")
    num_stages = cfg.pop("num_stages")
    grid = (triton.cdiv(n_tokens, cfg["BLOCK_T"]),
            triton.cdiv(hidden, cfg["BLOCK_H"]))
    # ``stride_w_o`` always indexes the hidden (N) axis and ``stride_w_i`` the
    # intermediate (K) axis, whichever way round the tensor is stored.
    stride_w_o, stride_w_i = (
        (w2.stride(1), w2.stride(0)) if w2_is_kn
        else (w2.stride(0), w2.stride(1))
    )
    _shared_down_gate_add_kernel[grid](
        h, w2, routed, gate_flat, x, gw, out,
        n_tokens, hidden, inter,
        h.stride(0), h.stride(1),
        stride_w_o, stride_w_i,
        routed.stride(0), routed.stride(1),
        stride_gate,
        stride_x_t, stride_x_h,
        out.stride(0), out.stride(1),
        W2_IS_KN=w2_is_kn,
        FUSE_GATE=fuse_gate,
        num_warps=num_warps,
        num_stages=num_stages,
        **cfg,
    )
    return out


# What makes one captured graph reusable: a graph records absolute device
# pointers and a fixed launch geometry, so an entry is valid only for the token
# count, dtype and device it was captured for.
_GraphKey = tuple[int, torch.dtype, int]


class _GraphEntry:
    """One captured ``forward_impl`` plus every tensor its replay touches.

    The entry owns the strong references that keep the replay valid:
    ``graph`` owns its private memory pool, ``static_input`` is the only buffer
    the caller writes, and ``static_output`` is pool-resident -- it was allocated
    by the caching allocator *during* capture, so it stays alive exactly as long
    as ``graph`` does and every replay writes the same addresses. Dropping the
    entry while a replay is still queued would free the pool underneath in-flight
    work, so entries are never evicted (see ``_GRAPH_CACHE_MAX_ENTRIES``).
    """

    __slots__ = ("graph", "static_input", "static_output")

    def __init__(self, graph, static_input, static_output):
        self.graph = graph
        self.static_input = static_input
        self.static_output = static_output


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

        # Graph fast path state. Read from the environment once, here, so an A/B
        # run is a separate process with a flag rather than a mutated module.
        self._graph_enabled = _env_flag("FASTKERNELS_SEMOE_CUDA_GRAPH")
        self._graph_max_tokens = _env_int(
            "FASTKERNELS_SEMOE_GRAPH_MAX_TOKENS", self._GRAPH_MAX_TOKENS,
        )
        self._graphs: dict[_GraphKey, _GraphEntry] = {}
        # Token counts that must never be captured again: capture raised, or the
        # cache is full. Both fall through to the eager path, which is always
        # correct, so a failure here costs latency and nothing else.
        self._graph_ineligible: set[_GraphKey] = set()

        # Off by default: measured slower than the two-op composition it replaces
        # at every token count once whole-operator latency is what is measured
        # rather than isolated launch cost. See the module docstring and
        # ``profile/probe_layer_ab.py``.
        self._fused_epilogue_enabled = _env_flag(
            "FASTKERNELS_SEMOE_FUSED_EPILOGUE", default=False,
        )
        self._fused_max_tokens = _env_int(
            "FASTKERNELS_SEMOE_FUSED_MAX_TOKENS", self._FUSED_MAX_TOKENS,
        )

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

    def process_weights_after_loading(self) -> None:
        """Shuffle expert weights into trtllm-gen's 4D BlockMajorK layout.

        Mirrors vLLM's ``convert_to_unquantized_kernel_format`` for the
        ``FLASHINFER_TRTLLM`` backend, which runs once after loading. The
        original ``[E, 2*I, H]`` / ``[E, H, I]`` tensors are replaced, so the
        Triton path is unavailable afterwards -- guarded by ``use_trtllm``.
        """
        if not self.use_trtllm or self._trtllm_weights_ready:
            return
        w13, w2 = prepare_trtllm_bf16_moe_weights(self.w13.data, self.w2.data)
        self.w13 = nn.Parameter(w13, requires_grad=False)
        self.w2 = nn.Parameter(w2, requires_grad=False)
        self._trtllm_weights_ready = True
        # This rebinds w13/w2 to fresh Parameters at new addresses, and a captured
        # graph holds the old pointers. In the normal order nothing is captured yet
        # (the harness runs this during setup, before the first forward), but
        # dropping the cache makes the ordering a non-issue rather than a
        # convention. Entries are released here, not while a replay could be in
        # flight: this runs outside ``forward``.
        self._graphs.clear()

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

    # Token count above which graph replay stops paying. Copy-in plus replay
    # against eager measures 2.37x at 445, 1.96x at 1024, 1.48x at 2048, 0.98x at
    # 4096 and 0.87x at 8192 (``profile/probe_threshold.py``): past a few thousand
    # tokens the forward is device-bound, replay saves nothing, and the static
    # input copy is pure added traffic. None of the five timed shapes sits near
    # this boundary, so it governs generality rather than the score.
    _GRAPH_MAX_TOKENS = 2048
    # Warm-up iterations before capture. Three is what the ``L4/yolov10`` baseline
    # and ``infra/engine.py``'s own capture site use, and it is what the
    # FlashInfer autotuner needs to have chosen and cached a tactic *before* the
    # capture -- ``AutoTuner.choose_one`` cannot profile inside a capture.
    _GRAPH_WARMUP_ITERS = 3
    # Cache bound. Under ``fastkernels bench`` exactly one entry is ever created,
    # because ``_bench_one_case`` builds a fresh module pair per shape. The bound
    # exists so a serving caller that cycles token counts cannot grow reserved
    # memory without limit; past it, new token counts run eager rather than
    # evicting a live entry, since freeing a graph's pool while its replay is
    # still queued would pull memory out from under in-flight work.
    _GRAPH_CACHE_MAX_ENTRIES = 4

    @staticmethod
    def _graph_key(hidden_states: torch.Tensor) -> _GraphKey:
        # ``get_device()`` returns an int; ``.device`` would allocate a Python
        # object on every call, and at T=1 the whole forward budget is ~50 us.
        return (hidden_states.shape[0], hidden_states.dtype,
                hidden_states.get_device())

    def _graph_eligible(self, hidden_states: torch.Tensor) -> bool:
        """Whether this call may use a captured replay.

        Ordered cheapest-predicate-first: the two CUDA/compiler queries at the
        end are the only ones that are not a plain attribute read, and at T=1 the
        whole forward budget is ~50 us.
        """
        tokens = hidden_states.shape[0]
        return (
            self._graph_enabled
            # Only the trtllm-gen path is worth capturing; it is the launch cost
            # the graph exists to remove.
            and self.use_trtllm
            # ``forward_impl`` all-reduces when tp_size > 1. Capturing a
            # collective needs the same communicator and a capture-safe NCCL
            # path; the benched config is tp=1, so restrict rather than guess.
            and self.tp_size == 1
            and 0 < tokens <= self._graph_max_tokens
            and self._graph_key(hidden_states) not in self._graph_ineligible
            # A strided input would be laundered into a contiguous static buffer
            # by the copy below, and trtllm-gen's MoE reads ``hidden_states`` as
            # if it were contiguous -- so the graph would return the *correct*
            # answer where the eager path reproduces the baseline's. Keeping
            # non-contiguous inputs on the eager path makes this module
            # bit-identical to the baseline for every input rather than only for
            # the contiguous ones the harness generates. ``is_contiguous()`` is a
            # flag read, unlike ``.contiguous()``, which would cost host time in
            # the one regime where host time is the whole budget.
            and hidden_states.is_contiguous()
            # Autograd would try to record through the captured region. The
            # harness always calls under ``no_grad``; anything else runs eager.
            and not torch.is_grad_enabled()
            and not torch.compiler.is_compiling()
            # Under ``fastkernels eval`` on B200 the engine records the whole
            # model forward -- this operator included -- in one outer graph, so
            # the launch cost is already gone and a nested capture is illegal.
            # ``L2/fused_experts.py`` disables its DeepGEMM path the same way.
            and not torch.cuda.is_current_stream_capturing()
        )

    def _warmup_for_capture(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Fill a static input with real data and prime the kernels off-capture.

        Returns the static input buffer, allocated *before* any capture begins
        and therefore never a member of a graph's private pool -- the allocator
        only starts routing into that pool at ``capture_begin``.

        The ordering is load-bearing twice over. ``static_input`` is filled
        *before* the warm-up because the warm-up runs the router: on an
        uninitialized buffer the top-k would be chosen from garbage (or NaN)
        logits, and the trtllm-gen routing kernel would be primed for a
        distribution the real inputs never produce. And the warm-up runs
        *outside* the capture because that is where FlashInfer is allowed to
        construct its ``MoERunner``, let ``AutoTuner.choose_one`` profile and
        cache a tactic, allocate its workspaces and JIT its cubins -- none of
        which can happen during a capture. This is the same
        warm-up-on-a-side-stream idiom as ``infra/engine.py``'s capture site
        ("Warmup eager so kernels autotune outside the graph") and the
        ``L4/yolov10`` baseline's.

        Kept separate from the capture itself because a failure *here* is
        cleanly recoverable -- no capture was ever begun -- whereas a failure
        inside the capture is not (see ``_graph_forward``).
        """
        static_input = torch.empty_like(hidden_states)
        static_input.copy_(hidden_states)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(self._GRAPH_WARMUP_ITERS):
                self.forward_impl(static_input)
        torch.cuda.current_stream().wait_stream(side)
        return static_input

    def _record_graph(self, static_input: torch.Tensor) -> _GraphEntry:
        """Record ``forward_impl`` into a graph with its own private pool.

        ``static_input`` must already hold real data and must already have been
        warmed up through :meth:`_warmup_for_capture`.
        """
        graph = torch.cuda.CUDAGraph()
        # ``pool=None`` gives this graph its own private memory pool, so two
        # entries for two token counts can never alias each other's
        # intermediates. A shared pool would additionally require that the
        # graphs never overlap and are replayed in a disciplined order, which a
        # token-count-keyed cache cannot promise.
        #
        # No explicit synchronize here: ``torch.cuda.graph.__enter__`` already
        # calls ``torch.cuda.synchronize()`` and ``empty_cache()``, and it swaps
        # in its own internal side stream for the duration -- the capture does
        # *not* happen on the caller's stream.
        with torch.cuda.graph(graph):
            static_output = self.forward_impl(static_input)
        return _GraphEntry(graph, static_input, static_output)

    def _graph_forward(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        """Replay a captured ``forward_impl``, or None to fall back to eager."""
        key = self._graph_key(hidden_states)
        entry = self._graphs.get(key)
        if entry is None:
            if len(self._graphs) >= self._GRAPH_CACHE_MAX_ENTRIES:
                # Retire the shape rather than evict a live entry. Dropping an
                # entry frees its private pool, and neither ``CUDAGraph.reset()``
                # nor its destructor synchronizes the stream, so an eviction
                # while a replay is still in flight has no ordering guarantee
                # behind it. Never evicting removes the question.
                self._graph_ineligible.add(key)
                return None
            try:
                static_input = self._warmup_for_capture(hidden_states)
            except Exception:
                # No capture was begun, so the process is in a clean state and
                # this shape simply runs eager from now on.
                self._graph_ineligible.add(key)
                return None
            try:
                if _env_flag("FASTKERNELS_SEMOE_FAULT_INJECT_CAPTURE", default=False):
                    # Fault injection, default off. The recovery path below is the
                    # one thing here that cannot be exercised by any input, so it
                    # gets a switch: it lets a test drive a capture failure through
                    # the real harness end to end and confirm the operator still
                    # answers correctly. Never read on the default path.
                    raise RuntimeError("injected capture failure")
                entry = self._record_graph(static_input)
            except Exception:
                # A capture that fails part-way can leave the graph, the capture
                # stream, the RNG state or the allocator in a state PyTorch does
                # not promise is recoverable, so give up on graphing entirely for
                # this instance rather than try another shape later. Correctness
                # never depended on the graph: the eager path takes over.
                self._graph_enabled = False
                self._graph_ineligible.add(key)
                torch.cuda.synchronize()
                return None
            self._graphs[key] = entry
        # The copy is inside ``forward`` and therefore inside the timed region:
        # the harness hands a different ``data_ptr`` every call
        # (``bench._ShiftingPool``), so the graph's fixed input address has to be
        # refilled, and that cost belongs to the candidate. Measured 4.83 us at
        # T=1 (``profile/probe_details.py``).
        entry.static_input.copy_(hidden_states)
        entry.graph.replay()
        # ``static_output`` lives in the graph's private pool and the next replay
        # overwrites it, so the caller must never see it. The clone is the whole
        # reason this differs from the ``L4/yolov10`` baseline, which hands its
        # static output back. Measured 5.53 us at T=1.
        return entry.static_output.clone()

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
        if self._graph_eligible(hidden_states):
            output = self._graph_forward(hidden_states)
            if output is not None:
                return output.view(orig_shape)
        return self.forward_impl(hidden_states).view(orig_shape)

    # Token count above which the fused shared-expert tail is never used, for
    # callers who enable it. Past a few thousand tokens the pair it replaces stops
    # being launch-dominated and becomes real device work that cuBLAS does better
    # than a Triton ``tl.dot`` on this K=512 shape: at T=16384 the pair is 59.3 us
    # against 72.5 us for the best of 216 swept tile configurations. Below the
    # threshold the two are within noise of each other at the whole-operator
    # level. See ``profile/probe_fused_epilogue.py --bench`` / ``--sweep``,
    # ``profile/probe_layer_ab.py``, and the rejected variants in
    # ``benchmark.csv``.
    _FUSED_MAX_TOKENS = 4096

    def _use_fused_epilogue(self, n_tokens: int) -> bool:
        """Whether to run the shared expert's tail as one fused kernel."""
        if not (self._fused_epilogue_enabled and self.has_shared_expert):
            return False
        if self.shared_expert_gate is None or n_tokens > self._fused_max_tokens:
            return False
        # The fused kernel consumes ``down_proj.weight`` directly as a dense bf16
        # [hidden, inter] matrix and does the row-parallel reduce nowhere, so it
        # only stands in for the plain ``F.linear`` case.
        down = getattr(self, self.shared_expert_attr_name).down_proj
        return (self.tp_size == 1 and not down.use_fp8 and down.bias is None)

    def forward_impl(self, hidden_states: torch.Tensor) -> torch.Tensor:
        n_tokens = hidden_states.shape[0]
        fused_tail = self._use_fused_epilogue(n_tokens)
        if fused_tail:
            # Stop the shared expert one op early and hand the post-SiLU
            # intermediate to the epilogue kernel, which does the
            # down-projection and the gated add in a single pass -- so
            # ``shared_output`` is never written to memory. The gate follows the
            # baseline's own rule: projected separately above
            # ``_FUSE_GATE_MAX_TOKENS``, folded into the kernel below it.
            shared_mlp = getattr(self, self.shared_expert_attr_name)
            shared_h = shared_mlp.act_fn(shared_mlp.gate_up_proj(hidden_states))
            shared_output = None
            shared_gate = (
                None if n_tokens <= self._FUSE_GATE_MAX_TOKENS
                else self.shared_expert_gate(hidden_states)
            )
        else:
            shared_h = None
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
        if fused_tail:
            # One kernel for ``routed + sigmoid(gate) * (h @ W2^T)``: the 128 MiB
            # round trip through ``shared_output`` that the two-launch version
            # pays at T=16384 never happens.
            output = shared_down_gate_add(
                routed_output, shared_h,
                getattr(self, self.shared_expert_attr_name).down_proj.weight,
                shared_gate,
                hidden_states=hidden_states if shared_gate is None else None,
                gate_weight=(self.shared_expert_gate.weight
                             if shared_gate is None else None),
            )
        elif shared_output is None:
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
