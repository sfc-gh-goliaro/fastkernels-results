"""TRTLLM-gen BF16 fused MoE (via FlashInfer, Blackwell only).

This is the kernel vLLM 0.26 actually runs for the unquantized BF16 MoE of
Qwen3-Next and Kimi-Linear on SM100. Its oracle picks the
``FLASHINFER_TRTLLM`` unquantized backend and ``TrtLlmBf16ExpertsMonolithic``,
then autotunes ``flashinfer::trtllm_bf16_moe`` -- visible in a reference run's
log as ``[AutoTuner]: Tuning flashinfer::trtllm_bf16_moe``.

The Triton ``_fused_moe_kernel`` path in :mod:`fused_experts` computes the same
math and is much slower here: it took 22.5% of a Qwen3-Next 32768-token prefill
profile, with a further ~7% in PyTorch's ``mbtopk`` where the reference has the
routing fused into this same kernel.

Beyond the kernel swap, two things differ from the Triton path:

* **Routing is fused in.** ``trtllm_bf16_moe`` takes the raw router logits plus
  the routing description (method, groups, bias, scaling) and does gating, top-k
  and the weighted reduction itself, so the separate gate/top-k/scale steps go
  away. ``routed_scaling_factor`` is applied inside the kernel.
* **A shuffled 4D BlockMajorK weight layout** for the transposed MMA epilogue,
  plus a gate/up row rotation because trtllm-gen defines SwiGLU with the two
  halves in the opposite order. :func:`prepare_trtllm_bf16_moe_weights` is a
  port of vLLM's ``convert_moe_weights_to_flashinfer_trtllm_block_layout``.

Mirrors ``TrtLlmBf16ExpertsMonolithic.apply``
(``vllm/model_executor/layers/fused_moe/experts/trtllm_bf16_moe.py``).

Dispatch
--------
Which resource binds moves with the token count, and the vendor entry point
leaves latency on the table at both ends.

At small batch the operator is *host*-bound, not GEMM-bound: at one token the
whole call is ~0.71 ms of CPU time against ~0.02 ms of device time, so the
Python and launcher bookkeeping -- not the MMA -- sets the latency. Two things
pay for that, and :meth:`TrtLlmBf16MoE.forward` hoists both out of the
steady-state path:

* ``flashinfer.fused_moe.trtllm_bf16_moe`` re-derives its whole dispatch on
  every call: argument validation, an output plus two placeholder tensors, a
  fresh ``MoERunner``, a ``TuningConfig`` rebuilt from
  ``get_hybrid_num_tokens_buckets(16384)``, and an ``AutoTuner.choose_one``
  cache probe. None of it depends on the tensor *data*, and outside an
  ``autotune()`` context ``choose_one`` can only ever return the fallback
  tactic. ~0.12 ms per call.
* The C++ launcher builds an ``MoE::Runner`` for the chosen ``tile_N`` and then
  scans every one of its candidate configs to validate the tactic (~1.1 us per
  config). The vendor's default ``tile_N`` at one token is 8, which carries 144
  candidates; the largest tile it offers there carries 64. ~0.29 ms per call.

So the module resolves the loaded trtllm-gen handle, the placeholder tensors and
the ``[tile_N, config]`` tactic once per (shape, dtype) key and then calls the
binding directly.

At large batch device time binds instead, and there the vendor's *tactic* is the
cost: at 16384 tokens it lands on a tile-64 kernel, and the tile-128 kernel of
the same ladder is 18% faster on E256/H2304 (1.193 -> 0.977 ms) and 7% faster on
E512/H2048 (0.901 -> 0.839 ms). Both ends are served by the same mechanism --
pin a measured ``[tile_N, config]`` per shape -- described at
:data:`_TACTIC_TABLE`.

Numerics are untouched at every token count: the tactic selects a tile/config
pair, and the trtllm-gen kernels accumulate over K in a fixed order, so every
valid tactic returns bit-identical output. Checked against the vendor path over
the whole ladder of both captured expert configs -- the 352 small-M tactics
(tiles 8/16/32) and the 8 large-M ones (tiles 64/128) -- by ``prof/biteq.py``.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn

from flashinfer.fused_moe import trtllm_bf16_moe as _trtllm_bf16_moe


# ``RoutingMethodType`` values (vllm/model_executor/layers/fused_moe/config.py).
# The kernel implements each scoring/normalization scheme internally, so the
# caller only names the one its config asks for.
ROUTING_RENORMALIZE = 1
ROUTING_DEEPSEEK_V3 = 2
ROUTING_RENORMALIZE_NAIVE = 4

# ``ActivationType.Swiglu`` == 3 -- what
# ``activation_to_flashinfer_int(MoEActivation.SILU)`` resolves to (vLLM maps SILU
# to its *gated* form). The value matters structurally, not just numerically: the
# launcher derives ``intermediate_size_factor`` from it, so a non-gated id makes
# ``check_weights_shape`` reject w13's 2*I rows.
ACTIVATION_SWIGLU = 3

# vLLM's ``fi_moe_largest_bucket``: ``max(max_num_tokens * dp_size, 8192)``. Both
# engines run ``max_num_batched_tokens=16384`` at dp=1, so vLLM tunes to 16384 and
# we were tuning to the 8192 floor -- a different bucket set, and so potentially a
# different tactic selected for *every* shape including the batch-1 decode ones.
# TODO: plumb ``max_num_batched_tokens`` through from the engine instead of
# restating its value here; L1 cannot import the engine without a cycle.
DEFAULT_TUNE_MAX_NUM_TOKENS = 16384

# ``epilogue_tile_m`` / ``block_k`` from
# ``convert_moe_weights_to_flashinfer_trtllm_block_layout``.
_EPILOGUE_TILE_M = 128
_BLOCK_K = 128


def trtllm_bf16_moe_supported() -> bool:
    """True when the trtllm-gen BF16 MoE kernel can run on this device.

    vLLM gates ``TrtLlmBf16ExpertsBase`` on ``is_device_capability_family(100)``
    plus ``has_flashinfer_trtllm_fused_moe()``, i.e. Blackwell only.
    ``FASTKERNELS_TRTLLM_BF16_MOE=0`` forces the Triton ``fused_experts`` path
    instead, for A/B against the reference.
    """
    if os.environ.get("FASTKERNELS_TRTLLM_BF16_MOE", "1") == "0":
        return False
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability()[0] == 10


def _copy_permuted_expert_to_block_layout(
    out: torch.Tensor,
    expert_uint8: torch.Tensor,
    source_indices: torch.Tensor,
) -> None:
    expert_blocks = expert_uint8.view(
        expert_uint8.shape[0], out.shape[0], _BLOCK_K,
    ).permute(1, 0, 2)
    torch.index_select(
        expert_blocks,
        1,
        source_indices.to(expert_uint8.device),
        out=out,
    )


def prepare_trtllm_bf16_moe_weights(
    w13: torch.Tensor,
    w2: torch.Tensor,
    is_gated_act_gemm: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shuffle BF16 expert weights into FlashInfer's 4D BlockMajorK layout.

    ``w13`` is ``[E, 2*I, H]`` and ``w2`` is ``[E, H, I]`` (the layout the
    checkpoint loaders already produce). Returns
    ``[E, H // 128, 2*I, 128]`` and ``[E, I // 128, H, 128]``.

    Port of vLLM's ``convert_moe_weights_to_flashinfer_trtllm_block_layout``.
    """
    if w13.dtype != torch.bfloat16 or w2.dtype != torch.bfloat16:
        raise ValueError("trtllm-gen BF16 MoE requires bfloat16 weights")

    from flashinfer.fused_moe.core import (
        _maybe_get_cached_w3_w1_permute_indices,
        get_w2_permute_indices_with_cache,
    )

    cache: dict[torch.Size, torch.Tensor] = {}
    num_experts = w13.shape[0]
    w13_rows, w13_cols = w13[0].view(torch.uint8).shape
    w2_rows, w2_cols = w2[0].view(torch.uint8).shape

    w13_shuffled = torch.empty(
        (num_experts, w13_cols // _BLOCK_K, w13_rows, _BLOCK_K),
        dtype=torch.uint8,
        device=w13.device,
    )
    w2_shuffled = torch.empty(
        (num_experts, w2_cols // _BLOCK_K, w2_rows, _BLOCK_K),
        dtype=torch.uint8,
        device=w2.device,
    )

    for i in range(num_experts):
        w13_expert = w13[i].view(torch.uint8)
        permute = _maybe_get_cached_w3_w1_permute_indices(
            cache, w13_expert, _EPILOGUE_TILE_M,
            is_gated_act_gemm=is_gated_act_gemm,
        )
        if is_gated_act_gemm:
            # trtllm-gen's SwiGLU expects [w3; w1] where the checkpoint gives
            # [w1; w3], so rotate the row permutation by half.
            rows = w13_expert.shape[0]
            permute = (permute + rows // 2) % rows
        _copy_permuted_expert_to_block_layout(w13_shuffled[i], w13_expert, permute)

        w2_expert = w2[i].view(torch.uint8)
        _copy_permuted_expert_to_block_layout(
            w2_shuffled[i],
            w2_expert,
            get_w2_permute_indices_with_cache(cache, w2_expert, _EPILOGUE_TILE_M),
        )

    return w13_shuffled.view(torch.bfloat16), w2_shuffled.view(torch.bfloat16)


# ``WeightLayout.BlockMajorK`` -- the only layout this BF16 entry point accepts.
_WEIGHT_LAYOUT_BLOCK_MAJOR_K = 2

# ---------------------------------------------------------------------------
# Tactic selection
# ---------------------------------------------------------------------------
# ``trtllm_get_valid_moe_configs`` offers a ladder of ``[tile_N, config]``
# candidates whose top end grows with the token count (``prof/ladder_map.py``,
# both captured expert configs):
#
#     M <  269 / 413    {8: 144, 16: 144, 32: 64}
#     M >= 269 / 413    + {64: 4}
#     M >= 715 / 879    {16: 144, 32: 64, 64: 4, 128: 4}
#     M >= ~4096        {64: 4, 128: 4}
#
# (thresholds: E256/H2304 / E512/H2048.) Two costs pick out of that ladder, and
# which one binds moves with M:
#
# * **Device time**, measured with a saturated queue so per-call host dispatch is
#   amortized out of the comparison (``prof/rawop.py``: park the GPU on a spin
#   kernel, enqueue N calls behind it, one event pair around the N; cross-checked
#   against CUDA-graph replay, which agrees to ~1%). This is the whole story at
#   M=16384, where the tile choice is worth 18%: on E256/H2304 tile 128 measures
#   0.977 ms against the vendor tactic's 1.193 ms, and on E512/H2048 0.839 vs
#   0.901 ms.
# * **Host time.** ``prepare_moe_common`` builds an ``MoE::Runner`` for the
#   pinned tile and scans *every* candidate config of that tile to validate the
#   tactic, ~1.1 us each: a 144-config tile costs ~0.58 ms per call, the
#   64-config tile ~0.42 ms, the 4-config tiles ~0.30 ms. Below a few hundred
#   tokens that dwarfs the 0.02-0.17 ms of device work.
#
# So the tactic that minimizes the operator's per-call latency minimizes
# ``max(host, device)``, and that is what ``prof/sweep.py`` searches: device time
# for every candidate, host time once per tile (it does not depend on the config
# index). The winners for the hottest captured shapes are tabulated below;
# everything else takes :func:`_heuristic_tactic`.
#
# Within a tile the config index selects the ``(gemm1, gemm2)`` kernel pair, and
# the 64-config tile-32 list is strongly structured (``prof/sweep_top8.json``):
# indices repeat with period 32, the second half of each 32-block is the fast
# group, and inside a block of 8 the fast positions are ``{1, 3, 6, 7}``. At 64
# tokens on E256/H2304 that is 0.037 ms against 0.069 ms for the slow group --
# nearly 2x -- so the group matters far more than the pick within it (<=4%).
# The 4-config tiles have no such structure and put their fastest kernel at
# index 0.

# Measured ``[tile_N, config]`` per shape, keyed on the tactic-relevant scalars
# only: (num_tokens, hidden_size, intermediate_size, top_k, local_num_experts).
# Never on a data pointer or on tensor contents. From ``prof/sweep.py``, which
# times every one of the 352 (small M) or 8 (large M) valid tactics.
_TACTIC_TABLE: dict[tuple[int, int, int, int, int], list[int]] = {
    # E512/H2048 (Qwen3-Next), top_k=10
    (1, 2048, 256, 10, 512): [32, 25],
    (26, 2048, 256, 10, 512): [32, 63],
    (31, 2048, 256, 10, 512): [32, 31],
    (60, 2048, 256, 10, 512): [32, 31],
    (16384, 2048, 256, 10, 512): [128, 0],
    # E256/H2304 (Kimi-Linear), top_k=8
    (1, 2304, 512, 8, 256): [32, 26],
    (64, 2304, 512, 8, 256): [32, 17],
    (16384, 2304, 512, 8, 256): [128, 0],
}

# A pinned tactic is only valid for the ``num_tokens`` it was resolved at
# (``isValidConfigIndex`` folds ``maxNumCtasInBatchDim`` in), so the tactic cache
# is keyed on the exact token count and bounded rather than bucketed.
_MAX_CACHED_TACTICS = 256

_raw_moe = None  # tvm-ffi handle for flashinfer::trtllm_bf16_moe
_raw_valid_configs = None  # ... and for trtllm_get_valid_moe_configs
_raw_dtype_bf16 = None
_raw_fp8_none = None
_fast_dispatch_state = 0  # 0 unresolved, 1 available, -1 unavailable

# (device index, routing-logits dtype) -> the two placeholder tensors that tell
# the kernel "routing is not pre-computed". Zero-element, never read or written.
_placeholders: dict[tuple[int, torch.dtype], tuple[torch.Tensor, torch.Tensor]] = {}

# Resolved [tile_N, config] tactics, keyed on shape/dtype scalars only.
_tactics: dict[tuple, list[int]] = {}


def _resolve_fast_dispatch() -> bool:
    """Resolve the loaded trtllm-gen binding, once per process.

    ``get_trtllm_moe_sm100_module`` is ``functools.cache``d and also installs the
    cubin loader, so it is the supported way in; it just does not hand the raw
    module back. ``MoERunner.forward`` closes over it.
    """
    global _raw_moe, _raw_valid_configs, _raw_dtype_bf16, _raw_fp8_none
    global _fast_dispatch_state
    if _fast_dispatch_state:
        return _fast_dispatch_state > 0
    try:
        from flashinfer.fused_moe.core import (
            DtypeTrtllmGen,
            Fp8QuantizationType,
            get_trtllm_moe_sm100_module,
        )

        namespace = get_trtllm_moe_sm100_module()
        runner_forward = namespace.MoERunner.forward
        runner_forward = getattr(runner_forward, "__wrapped__", runner_forward)
        cells = runner_forward.__code__.co_freevars
        module = runner_forward.__closure__[cells.index("moe_op")].cell_contents
        _raw_moe = module.trtllm_bf16_moe
        _raw_valid_configs = module.trtllm_get_valid_moe_configs
        _raw_dtype_bf16 = int(DtypeTrtllmGen.Bfloat16)
        _raw_fp8_none = int(Fp8QuantizationType.NoneFp8)
    except Exception:  # noqa: BLE001 -- any change here just means the vendor path
        _fast_dispatch_state = -1
        return False
    _fast_dispatch_state = 1
    return True


def _get_placeholders(
    device: torch.device, logits_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The ``topk_ids`` / ``expert_weights`` empties the launcher wants when it is
    the one doing the routing. ``expert_weights`` must carry the logits dtype."""
    key = (device.index, logits_dtype)
    got = _placeholders.get(key)
    if got is None:
        got = (
            torch.empty(0, dtype=torch.int32, device=device),
            torch.empty(0, dtype=logits_dtype, device=device),
        )
        _placeholders[key] = got
    return got


def _heuristic_tactic(valid: list[tuple[int, int]]) -> list[int]:
    """The tactic to pin for a shape that is not in :data:`_TACTIC_TABLE`.

    Take the **largest offered tile**, then the fastest config index of it.

    The largest tile is the right choice for two independent reasons that happen
    to agree everywhere measured. It always carries the fewest candidate configs,
    so it is the cheapest for ``prepare_moe_common`` to validate -- which is what
    binds below a few hundred tokens. And where device time binds instead, at
    M=16384, it is also the fastest: tile 128 beats tile 64 by 4-17% on both
    expert configs. In between the two agree by accident rather than by luck --
    ``prof/sweep_mid.py`` finds device time nearly flat across tiles at 302-961
    tokens (E512/H2048 M=432: 0.253-0.256 ms over tiles 8/16/32/64), so host cost
    decides and the largest tile wins on it.

    Checked against a bounded search over five mid-M captured shapes the table
    does not cover (302, 432, 715 x2, 949, 961): this rule lands on the measured
    optimum at four of them and within 1.2% at the other two.

    The config index then follows the structure of the candidate list. The
    64-config tile-32 list is bimodal with period 32 -- fast group in the upper
    half of each 32-block, fast positions ``{1, 3, 6, 7}`` inside each 8 -- and
    picking wrongly costs up to 2x (0.037 ms vs 0.069 ms at 64 tokens on
    E256/H2304), whereas any member of the fast group is within 4% of the best.
    The sparse 4-config tiles have no such structure and are fastest at index 0.
    """
    tile_n = max(tile for tile, _ in valid)
    cfgs = sorted(cfg for tile, cfg in valid if tile == tile_n)
    fast = [c for c in cfgs if c % 32 >= 16 and c % 8 in (1, 3, 6, 7)]
    return [tile_n, min(fast) if fast else min(cfgs)]


def _resolve_tactic(
    num_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    top_k: int,
    local_num_experts: int,
    activation_type: int,
) -> list[int]:
    """The ``[tile_N, config]`` tactic to pin, or ``[-1, -1]`` for the vendor's.

    A measured entry from :data:`_TACTIC_TABLE` when the shape has one, otherwise
    :func:`_heuristic_tactic`. Either way the choice is validated against the
    ladder ``trtllm_get_valid_moe_configs`` reports for *this exact token count*:
    ``isValidConfigIndex`` folds ``maxNumCtasInBatchDim`` in, so a tactic that is
    valid at one token count can be rejected at another, and pinning a rejected
    one throws "Invalid MoE tactic". A tabulated tactic that is not in the ladder
    (a different FlashInfer build, say) degrades to the heuristic, and anything
    unexpected degrades to the vendor's own choice.

    Resolution happens once per shape key and is cached, so no search cost can
    reach the steady-state path -- which is the reason the table is static rather
    than timed at import.
    """
    key = (num_tokens, hidden_size, intermediate_size, top_k, local_num_experts,
           activation_type)
    got = _tactics.get(key)
    if got is not None:
        return got
    tactic = [-1, -1]
    try:
        valid = [
            (int(pair[0]), int(pair[1]))
            for pair in _raw_valid_configs(
                _raw_dtype_bf16,  # dtype_act
                _raw_dtype_bf16,  # dtype_weights
                _raw_fp8_none,
                top_k,
                hidden_size,
                intermediate_size,
                local_num_experts,
                activation_type,
                True,  # use_shuffled_weight
                _WEIGHT_LAYOUT_BLOCK_MAJOR_K,
                False,  # use_per_token_scaling
                num_tokens,
                False,  # has_gemm1_lora_delta
            )
        ]
        if valid:
            measured = _TACTIC_TABLE.get(
                (num_tokens, hidden_size, intermediate_size, top_k,
                 local_num_experts)
            )
            if measured is not None and tuple(measured) in valid:
                tactic = list(measured)
            else:
                tactic = _heuristic_tactic(valid)
    except Exception:  # noqa: BLE001 -- fall back to the vendor tactic
        tactic = [-1, -1]
    if len(_tactics) < _MAX_CACHED_TACTICS:
        _tactics[key] = tactic
    return tactic


class TrtLlmBf16MoE(nn.Module):
    """Monolithic trtllm-gen BF16 MoE: routing, both GEMMs and the reduction.

    ``w13``/``w2`` must already be in the shuffled BlockMajorK layout produced
    by :func:`prepare_trtllm_bf16_moe_weights`.
    """

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        intermediate_size_per_partition: int,
        routing_method_type: int = ROUTING_RENORMALIZE,
        local_expert_offset: int = 0,
        local_num_experts: int | None = None,
        num_expert_group: int | None = None,
        topk_group: int | None = None,
        routed_scaling_factor: float | None = None,
        tune_max_num_tokens: int = DEFAULT_TUNE_MAX_NUM_TOKENS,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.intermediate_size_per_partition = intermediate_size_per_partition
        self.routing_method_type = routing_method_type
        self.local_expert_offset = local_expert_offset
        self.local_num_experts = (
            num_experts if local_num_experts is None else local_num_experts
        )
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.routed_scaling_factor = routed_scaling_factor
        self.tune_max_num_tokens = tune_max_num_tokens
        # (num_tokens, hidden_size, logits dtype, device) -> plan.
        self._plans: dict[tuple, tuple] = {}

    def forward(
        self,
        hidden_states: torch.Tensor,
        w13: torch.Tensor,
        w2: torch.Tensor,
        router_logits: torch.Tensor,
        routing_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_tokens, hidden_size = hidden_states.shape
        key = (num_tokens, hidden_size, router_logits.dtype, hidden_states.device)
        plan = self._plans.get(key)
        if plan is None:
            plan = self._plan(num_tokens, hidden_size, router_logits, hidden_states)
            if plan is None:
                return self._vendor_forward(
                    hidden_states, w13, w2, router_logits, routing_bias)
        tactic, topk_ids, expert_weights = plan
        out = torch.empty(
            num_tokens, hidden_size, dtype=torch.bfloat16,
            device=hidden_states.device,
        )
        # Positionally identical to what ``trtllm_bf16_moe_op`` passes the
        # binding, minus the per-call bookkeeping that produced these values.
        # Wrapped because a pinned tactic is the one argument here the launcher
        # can reject outright ("Invalid MoE tactic"): drop the plan and let the
        # vendor wrapper serve this call and re-resolve the next one. Costs
        # nothing when nothing throws.
        try:
            _raw_moe(
                router_logits,
                routing_bias,
                topk_ids,
                expert_weights,
                hidden_states,
                w13,
                w2,
                None,  # gemm1_lora_delta
                None,  # gemm1_alpha
                None,  # gemm1_beta
                None,  # gemm1_clamp_limit
                out,
                self.num_experts,
                self.top_k,
                self.num_expert_group,
                self.topk_group,
                self.intermediate_size_per_partition,
                self.local_expert_offset,
                self.local_num_experts,
                self.routed_scaling_factor,
                self.routing_method_type,
                True,  # use_shuffled_weight
                _WEIGHT_LAYOUT_BLOCK_MAJOR_K,
                True,  # do_finalize
                True,  # enable_pdl -- the public wrapper's default
                tactic,
                ACTIVATION_SWIGLU,
                True,  # norm_topk_prob -- the public wrapper's default
                None,  # routing_replay_out
            )
        except Exception:  # noqa: BLE001 -- rejected tactic; use the vendor path
            self._plans.pop(key, None)
            return self._vendor_forward(
                hidden_states, w13, w2, router_logits, routing_bias)
        return out

    def _plan(
        self,
        num_tokens: int,
        hidden_size: int,
        router_logits: torch.Tensor,
        hidden_states: torch.Tensor,
    ):
        """Everything the steady-state call needs that depends only on shapes.

        ``None`` means the fast path is unavailable and the caller should use the
        vendor wrapper. The two validations the wrapper does --
        ``_validate_routing_replay_out`` and
        ``_validate_bf16_gemm1_activation_params`` -- are both no-ops for this
        module (it passes no replay buffer and no SwiGLU alpha/beta/clamp), so
        nothing is skipped by going direct.
        """
        if not _resolve_fast_dispatch():
            return None
        device = hidden_states.device
        topk_ids, expert_weights = _get_placeholders(device, router_logits.dtype)
        tactic = _resolve_tactic(
            num_tokens,
            hidden_size,
            self.intermediate_size_per_partition,
            self.top_k,
            self.local_num_experts,
            ACTIVATION_SWIGLU,
        )
        plan = (tactic, topk_ids, expert_weights)
        self._plans[(num_tokens, hidden_size, router_logits.dtype, device)] = plan
        return plan

    def _vendor_forward(
        self,
        hidden_states: torch.Tensor,
        w13: torch.Tensor,
        w2: torch.Tensor,
        router_logits: torch.Tensor,
        routing_bias: torch.Tensor | None,
    ) -> torch.Tensor:
        out = _trtllm_bf16_moe(
            routing_logits=router_logits,
            routing_bias=routing_bias,
            hidden_states=hidden_states,
            gemm1_weights=w13,
            gemm2_weights=w2,
            num_experts=self.num_experts,
            top_k=self.top_k,
            n_group=self.num_expert_group,
            topk_group=self.topk_group,
            intermediate_size=self.intermediate_size_per_partition,
            local_expert_offset=self.local_expert_offset,
            local_num_experts=self.local_num_experts,
            routed_scaling_factor=self.routed_scaling_factor,
            routing_method_type=self.routing_method_type,
            activation_type=ACTIVATION_SWIGLU,
            tune_max_num_tokens=self.tune_max_num_tokens,
        )
        return out[0] if isinstance(out, (list, tuple)) else out
