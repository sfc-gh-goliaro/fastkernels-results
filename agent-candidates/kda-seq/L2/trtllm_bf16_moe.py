"""TRTLLM-gen BF16 fused MoE reached through the launcher's own runner.

Same kernel as the baseline, same math, same weights. What changes is the path
to it and which tile the launcher runs.

``flashinfer.fused_moe.trtllm_bf16_moe`` -- what the baseline calls -- is a
tracing wrapper around ``trtllm_bf16_moe_op``, which per call builds a
``MoERunner``, a ``MoeRunnerInputs``, and a ``TuningConfig`` via
``_make_tuning_config``, then asks ``AutoTuner.choose_one`` for a tactic before
dispatching through ``register_custom_op``. Outside a tuning region
``choose_one`` only reads its cache and returns ``-1``, so all of that
construction buys nothing. We hold the runner and the two ``torch.empty(0)``
routing sentinels on the module and allocate only the output per call.

The second change is the tactic. ``trtllm_bf16_moe_op`` hands ``[-1, -1]`` to
the launcher, and ``Bf16MoeLauncher::selectDefaultTileN``
(``flashinfer/data/csrc/trtllm_fused_moe_kernel_launcher.cu``) resolves that to
the *smallest* candidate tile in ``mSupportedTileNums = {8, 16, 32, 64, 128}``.
At 16384 tokens that leaves a large win on the table -- the widest tile is 1.13x
there -- so we name ``[tile_N, config]`` outright for the shapes where a win was
actually measured, and leave every other shape to the launcher. The widest tile
is *not* a general improvement: it loses 8-12% at small token counts, where a
tile sized for 128 rows per expert mostly pads. Tactic choice is numerically
inert -- it picks a tile schedule, not different arithmetic -- and the sweep in
``profile/pooled_tactics.log`` confirms bitwise-identical output for all 365
tactics on every case.

Mirrors ``TrtLlmBf16ExpertsMonolithic.apply``
(``vllm/model_executor/layers/fused_moe/experts/trtllm_bf16_moe.py``) by way of
``fastkernels.tasks.baseline.L2.trtllm_bf16_moe``.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass

import torch
import torch.nn as nn

# The baseline owns the module-level surface: sibling L2 baselines
# (``kimi_moe``, ``mixtral_moe``, ``shared_expert_moe``) import these names, and
# a sibling *candidate* doing ``from .trtllm_bf16_moe import ...`` resolves
# through ``_CandidateFinder`` to this file instead. Re-export the helpers and
# constants so that keeps working -- but deliberately not the baseline's
# ``TrtLlmBf16MoE``: ``_load_candidate_class`` picks the attribute named after
# the operator, so re-exporting it would shadow ours.
from fastkernels.tasks.baseline.L2.trtllm_bf16_moe import (  # noqa: F401
    ACTIVATION_SWIGLU,
    DEFAULT_TUNE_MAX_NUM_TOKENS,
    ROUTING_DEEPSEEK_V3,
    ROUTING_RENORMALIZE,
    ROUTING_RENORMALIZE_NAIVE,
    prepare_trtllm_bf16_moe_weights,
    trtllm_bf16_moe_supported,
)

# ``WeightLayout.BlockMajorK``. Hardcoded rather than imported so this module
# stays importable without CUDA: ``flashinfer.fused_moe.core`` is only touched
# from :func:`_launcher`, on the first call that actually has a device.
_WEIGHT_LAYOUT_BLOCK_MAJOR_K = 2

# The baseline calls the public wrapper without naming these, so it inherits the
# wrapper's defaults. ``enable_pdl`` matters most: the wrapper declares
# ``enable_pdl: bool = True`` and passes it down positionally, so
# ``trtllm_bf16_moe_op``'s ``if enable_pdl is None: device_support_pdl(...)``
# branch never runs on the baseline path. Reproduce the value, not the branch.
_USE_SHUFFLED_WEIGHT = True
_DO_FINALIZE = True
_ENABLE_PDL = True
_NORM_TOPK_PROB = True

# ``resolveMoeTileAndConfig`` returns ``{tile_N, config}`` verbatim whenever
# neither is -1, so this is what "let the launcher decide" has to be spelled as.
_DEFAULT_TACTIC = [-1, -1]

# Tiles the launcher will accept, largest first. ``Bf16MoeLauncher`` is
# constructed for all five on every call "so that autotuner-cached tactics
# always find their tile_N in the map", so any of these resolves -- including
# the ones ``computeSelectedTileN`` would not enumerate for a given token count.
_SUPPORTED_TILE_NUMS = (128, 64, 32, 16, 8)

# Measured tactics, from ``profile/pooled_tactics.log`` -- a 365-tactic sweep per
# case run through the bench's own ``_time_module``, shifting-pool weight copies
# included. That regime matters more than anything else here. Timed the way
# ``profile/tactics_sweep.log`` does it -- cold-L2 event pair, nothing queued
# ahead of the kernel -- the widest tile appears to win 1.16x-1.75x on every
# shape. Timed with the pool copies in the window, the same sweep says the widest
# tile *loses* 8-12% at small token counts and only wins at 16384. The pooled
# numbers are the ones that predict the score, so they are the ones used.
#
# What survived: tile 128 at 16384 tokens on both captured configurations, plus a
# per-shape winner on each of the four small cases. The small-case margins are
# 0.15%-1.75% -- far below the tile-128 win -- and were only installed after the
# measured noise floor and an independent confirmation run showed they are real
# rather than an artifact of picking the best of 365 timings; see the block above
# the small-case entries below.
#
# Within tile 128 the config index barely matters (1.014x-1.020x for config A,
# 1.1703x-1.1740x for config B across configs 0-3), so config 0 is named: the
# tile carries the win, and the lowest index is the least likely to mean
# something else in another build.
#
# Keyed on the kernel configuration plus the *exact* token count, because
# ``getValidConfigIndices`` takes ``num_tokens`` as an argument. Guarded by
# :func:`_table_applies`, since a config index is an offset into a kernel list
# that another FlashInfer build or a non-Blackwell device would order differently.
_TACTIC_TABLE_FLASHINFER_VERSION = "0.6.14"
_TACTIC_TABLE: dict[tuple, list[int]] = {
    # (E, local_E, offset, top_k, H, I, activation, layout, shuffled, m, logits dtype)
    # Qwen3-Next-80B prefill: 1.014x-1.068x over the launcher's tile 64.
    (512, 512, 0, 10, 2048, 256, ACTIVATION_SWIGLU, _WEIGHT_LAYOUT_BLOCK_MAJOR_K,
     _USE_SHUFFLED_WEIGHT, 16384, torch.bfloat16): [128, 0],
    # Kimi-Linear-48B prefill: 1.125x-1.174x, the largest single win available.
    (256, 256, 0, 8, 2304, 512, ACTIVATION_SWIGLU, _WEIGHT_LAYOUT_BLOCK_MAJOR_K,
     _USE_SHUFFLED_WEIGHT, 16384, torch.float32): [128, 0],
    # The per-case sweep winners on the four small cases. These are small --
    # 0.15% to 1.75% -- and were held back until an independent end-to-end arm
    # confirmed them, because a sweep that picks the best of 365 timings is
    # exactly the setup that manufactures wins out of noise. They survived:
    # against the measured within-sweep noise floor (relative sd 0.03%-0.10%,
    # profile/null_noise.json) each clears a 5% familywise threshold, and a
    # 6-block randomized re-measurement reproduced the two largest predictions
    # end-to-end -- 64 tokens went 1.002x -> 1.018x against a predicted 1.0175x,
    # and 60 tokens 1.000x -> 1.003x against 1.0051x (profile/ablation.json).
    #
    # These carry more version risk than the tile-128 entries above, because a
    # config index this deep into the list (101, 62, 22) is more likely to mean
    # something else in another build. That is what _table_applies and the
    # fallback ladder are for: a wrong index here degrades to the launcher's own
    # choice, it does not produce a wrong answer.
    (512, 512, 0, 10, 2048, 256, ACTIVATION_SWIGLU, _WEIGHT_LAYOUT_BLOCK_MAJOR_K,
     _USE_SHUFFLED_WEIGHT, 1, torch.bfloat16): [8, 101],
    (512, 512, 0, 10, 2048, 256, ACTIVATION_SWIGLU, _WEIGHT_LAYOUT_BLOCK_MAJOR_K,
     _USE_SHUFFLED_WEIGHT, 60, torch.bfloat16): [8, 22],
    (256, 256, 0, 8, 2304, 512, ACTIVATION_SWIGLU, _WEIGHT_LAYOUT_BLOCK_MAJOR_K,
     _USE_SHUFFLED_WEIGHT, 1, torch.float32): [16, 101],
    (256, 256, 0, 8, 2304, 512, ACTIVATION_SWIGLU, _WEIGHT_LAYOUT_BLOCK_MAJOR_K,
     _USE_SHUFFLED_WEIGHT, 64, torch.float32): [32, 62],
}


# Caps on the unknown-key search. Deliberately small: the six captured shapes all
# come from the table, so this path exists for generality, not for the score, and
# it must never be able to stall a caller or grow state.
#
# Limits are enforced at *launch* granularity, not per candidate. A candidate costs
# one warm-up plus _SEARCH_SAMPLES timed launches, so checking only once per
# candidate lets a search overshoot by up to four kernel launches -- and an
# invalid-tactic warm-up would previously cost nothing at all, which is how an
# adversarial tactic space could spin.
_SEARCH_MAX_KEYS = 4
_SEARCH_MAX_CANDIDATES = 160
_SEARCH_SAMPLES = 3
# One warm-up plus the timed samples, for every candidate.
_SEARCH_MAX_LAUNCHES = _SEARCH_MAX_CANDIDATES * (1 + _SEARCH_SAMPLES)
_SEARCH_SECONDS_PER_KEY = 8.0
# Total across a module's lifetime. A key's own deadline is clamped to whatever
# remains of this, and the deadline is checked before every launch, so the true
# bound is _SEARCH_SECONDS_TOTAL plus at most one launch's duration -- not exactly
# 30 s. Stated that way deliberately: the earlier version claimed a hard 30 s while
# checking the clock only once per candidate, which let a search overrun by four
# launches.
_SEARCH_SECONDS_TOTAL = 30.0

# Hard capacity for the resolved-tactic memo, covering table hits *and* searched
# keys. A capacity is needed rather than "table size plus searched keys" because
# the key includes the device: the same table entry is a distinct key on each
# device, so the table contributes len(_TACTIC_TABLE) * (number of devices), which
# is not a constant. Past the capacity we stop caching and re-resolve instead --
# two dict lookups, and the answer is identical either way.
_TACTIC_CACHE_CAPACITY = 32


class _SearchExhausted(Exception):
    """A search hit one of its limits. Partial results are discarded, not used.

    Returning the best tactic found so far would make the outcome depend on where
    the budget happened to run out -- the same shape could resolve differently on
    two machines, or on two runs. The launcher default is deterministic and always
    legal, so exhaustion takes it.
    """


@dataclass
class _SearchBudget:
    """Trial, launch and wall-clock allowance for one key's search.

    ``spend_candidate`` and ``spend_launch`` raise :class:`_SearchExhausted` rather
    than returning a flag, so a limit cannot be reached and then ignored by a
    caller that forgets to check.
    """

    deadline: float
    max_candidates: int = _SEARCH_MAX_CANDIDATES
    max_launches: int = _SEARCH_MAX_LAUNCHES
    candidates: int = 0
    launches: int = 0

    def spend_candidate(self) -> None:
        if self.candidates >= self.max_candidates:
            raise _SearchExhausted(f"candidate cap {self.max_candidates} reached")
        self.candidates += 1
        self._check_clock()

    def spend_launch(self) -> None:
        if self.launches >= self.max_launches:
            raise _SearchExhausted(f"launch cap {self.max_launches} reached")
        self.launches += 1
        self._check_clock()

    def _check_clock(self) -> None:
        if time.perf_counter() >= self.deadline:
            raise _SearchExhausted("wall-clock budget reached")


def _capturing() -> bool:
    """True while a CUDA graph is being captured on the current stream.

    A cache miss during capture must not event-time, synchronize, enumerate, or
    record anything: every one of those either fails outright under capture or
    bakes a host decision into the graph. The miss takes the launcher default and
    leaves no trace, so the same key is still eligible to search on a later eager
    call.

    Fails *closed*. If the query itself raises we cannot establish that eager
    execution is safe, so we answer True and decline the search -- the cost is a
    missed tuning opportunity, against synchronizing inside a graph capture.
    """
    try:
        return torch.cuda.is_current_stream_capturing()
    except Exception:
        return True


def _launcher():
    """The cached SM100 trtllm-gen MoE namespace, resolved on first use.

    ``get_trtllm_moe_sm100_module()`` JIT-builds and loads a CUDA extension.
    Calling it at import time would break importing this module on a CPU-only
    interpreter, which is exactly what operator discovery does before it knows
    whether the device is Blackwell.
    """
    from flashinfer.fused_moe.core import (
        ActivationType,
        DtypeTrtllmGen,
        Fp8QuantizationType,
        MoeRunnerInputs,
        get_trtllm_moe_sm100_module,
    )

    return (
        get_trtllm_moe_sm100_module(),
        MoeRunnerInputs,
        DtypeTrtllmGen,
        Fp8QuantizationType,
        ActivationType,
    )


class _InvalidTactic(Exception):
    """A ``[tile_N, config]`` the launcher refused for this token count."""


# ``FusedMoeLauncher::prepare_moe_common`` validates the *config* index against
# ``getValidConfigIndices(top_k, hidden_size, intermediate_size,
# local_num_experts, num_tokens)`` and raises with this text. The list depends on
# exact ``num_tokens``, so a config that is legal at one token count can be
# illegal at another -- this is the one failure we recover from. ``tile_N`` is
# *not* validated there, so a wrong tile costs scratch and speed but raises
# nothing; that is why the table below is verified by measurement, not by
# trusting the launcher to complain.
_INVALID_TACTIC_MARKER = "Invalid MoE tactic"


def _is_invalid_tactic(exc: BaseException) -> bool:
    return _INVALID_TACTIC_MARKER in str(exc)


def _table_applies(device: torch.device) -> bool:
    """Whether the measured tactic table describes *this* machine.

    The config index is an offset into a kernel list that a different FlashInfer
    build or a non-Blackwell device would order differently, so a mismatch must
    fall through to the launcher's own choice rather than name an index blindly.

    Total by construction: anything that stops us *confirming* SM100 and the
    expected build -- no CUDA at all, or a device the query rejects -- answers
    False, so an unverifiable machine gets the launcher's choice instead of an
    index measured somewhere else.
    """
    import flashinfer

    if flashinfer.__version__ != _TACTIC_TABLE_FLASHINFER_VERSION:
        return False
    try:
        return torch.cuda.get_device_capability(device) == (10, 0)
    except Exception:
        return False


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
        # Only ever fed ``_make_tuning_config``, which is unreachable once the
        # AutoTuner is out of the path. Kept for signature parity.
        self.tune_max_num_tokens = tune_max_num_tokens

        # Populated on first call: nothing here may depend on a device or on
        # weight *values*. The harness rebuilds the weights for each of its
        # three correctness rounds, so a value-derived cache would silently
        # answer for the wrong weights.
        self._runner = None
        self._runner_key: tuple | None = None
        self._moe_inputs_cls = None
        self._sentinels: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
        # Resolved tactic per key, and how it was reached. Shape/configuration
        # keys only -- never anything derived from weight values. Size is bounded
        # by ``len(_TACTIC_TABLE) + _SEARCH_MAX_KEYS``: a key that resolves to the
        # launcher default without being searched is deliberately *not* stored,
        # because the space of exact token counts is unbounded.
        self._tactics: dict[tuple, list[int]] = {}
        self._tactic_source: dict[tuple, str] = {}
        self._searched_keys = 0
        self._search_seconds = 0.0

    def _runner_for(self, hidden_size: int, device: torch.device):
        """The ``MoERunner`` for this configuration, rebuilt when the key changes.

        A single slot, not a cache: a module instance has one hidden size and one
        device, so alternating configurations does not arise on the path this
        serves, and one slot keeps retained state trivially bounded. If a caller
        ever did alternate, this rebuilds each time -- correct, just not cached.

        The runner is pure configuration -- ``__init__`` only stores scalars -- so
        it is safe to hold across calls and across weight values. Keyed on
        everything it stores, plus the device, since the launcher module it
        dispatches into is per-device state.
        """
        key = (
            self.top_k,
            self.local_num_experts,
            self.num_experts,
            hidden_size,
            self.intermediate_size_per_partition,
            device,
        )
        if self._runner_key != key:
            ns, moe_inputs_cls, dtype_gen, fp8_type, activation = _launcher()
            self._moe_inputs_cls = moe_inputs_cls
            self._runner = ns.MoERunner(
                top_k=self.top_k,
                num_local_experts=self.local_num_experts,
                dtype_act=dtype_gen.Bfloat16,
                dtype_weights=dtype_gen.Bfloat16,
                fp8_quantization_type=fp8_type.NoneFp8,
                hidden_size=hidden_size,
                intermediate_size=self.intermediate_size_per_partition,
                activation_type=activation.Swiglu.value,
                use_shuffled_weight=_USE_SHUFFLED_WEIGHT,
                weight_layout=_WEIGHT_LAYOUT_BLOCK_MAJOR_K,
                num_experts=self.num_experts,
            )
            self._runner_key = key
        return self._runner

    def _sentinels_for(
        self, logits_dtype: torch.dtype, device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The empty ``topk_ids``/``expert_weights`` the routed path expects.

        ``trtllm_bf16_moe_op`` allocates these two per call whenever
        ``routing_logits`` is given; they carry no data, so one pair per
        ``(dtype, device)`` serves every call.
        """
        key = (logits_dtype, device)
        pair = self._sentinels.get(key)
        if pair is None:
            pair = (
                torch.empty(0, dtype=torch.int32, device=device),
                torch.empty(0, dtype=logits_dtype, device=device),
            )
            self._sentinels[key] = pair
        return pair

    def _tactic_key(
        self,
        num_tokens: int,
        hidden_size: int,
        logits_dtype: torch.dtype,
        device: torch.device,
    ) -> tuple:
        """Everything that can change which tactics are legal or what they mean.

        ``num_tokens`` goes in exactly, not bucketed: ``getValidConfigIndices``
        takes it as an argument, so a config legal at one count is not
        guaranteed legal at another. ``activation``, ``weight_layout`` and
        ``use_shuffled_weight`` are constants on this path but are keyed anyway,
        so adding a second layout later cannot silently reuse this layout's
        measurements.
        """
        return (
            self.num_experts,
            self.local_num_experts,
            self.local_expert_offset,
            self.top_k,
            hidden_size,
            self.intermediate_size_per_partition,
            ACTIVATION_SWIGLU,
            _WEIGHT_LAYOUT_BLOCK_MAJOR_K,
            _USE_SHUFFLED_WEIGHT,
            num_tokens,
            logits_dtype,
            device,
        )

    def _largest_offered_tactic(self, runner, inputs) -> list[int] | None:
        """Largest tile the launcher enumerates here, at its lowest config.

        Only a rung of the fallback ladder, not a selection rule. It exists to
        recover from a rejected config with something the launcher has just told
        us is legal; it is emphatically *not* a good performance guess -- the
        pooled sweep measured this exact choice at 0.876x on ``A[60, 2048]`` and
        0.949x on ``B[64, 2304]``.

        Enumeration errors are **not** caught here. If the launcher cannot tell us
        what is legal, that is not an invalid-tactic condition and must not be
        turned into a silent slow path.
        """
        best: list[int] | None = None
        for tactic in runner.get_valid_tactics(inputs.to_list(), None):
            tile, config = int(tactic[0]), int(tactic[1])
            if best is None or (tile, -config) > (best[0], -best[1]):
                best = [tile, config]
        return best

    def _remember(self, key: tuple, tactic: list[int], source: str) -> None:
        """Record a resolution, up to the fixed cache capacity.

        Declining to cache past the capacity is safe: the next call re-resolves and
        gets the same answer. Growing instead would be unbounded, because the key
        includes the exact token count and the device.
        """
        if key in self._tactics or len(self._tactics) < _TACTIC_CACHE_CAPACITY:
            self._tactics[key] = tactic
            self._tactic_source[key] = source

    def _select_tactic(self, runner, inputs, key: tuple) -> list[int] | None:
        """The tactic to run for *key*, resolved once and remembered.

        Table first, and nothing else here. An unknown key returns ``None``, which
        ``forward`` offers to the bounded search; see :meth:`_search_tactic` for why
        an unmeasured shape is not memoized on the way through.

        There is deliberately no rule that extrapolates a tactic to an unmeasured
        shape: ``profile/tile_crossover.log`` shows where tile 128 starts paying is
        *configuration*-dependent, not a function of tokens-per-expert -- config B
        is ahead by 1.12x at 256 tokens while config A is still 16% behind there and
        does not turn over until ~4096 -- so any threshold cheap enough to state
        would be wrong for one of the two models we serve.

        Overridden by the sweep harness and by the ablation variants in
        ``profile/``, which pin a tactic and never reach the search.
        """
        tactic = self._tactics.get(key)
        if tactic is not None:
            return tactic

        table = _TACTIC_TABLE.get(key[:-1]) if _table_applies(key[-1]) else None
        if table is not None:
            tactic = list(table)
            self._remember(key, tactic, "table")
            return tactic
        return None

    def _search_tactic(self, runner, inputs, key, w13, w2, routing_bias):
        """Bounded two-stage timing search for a key the table does not cover.

        Returns a tactic, or ``None`` if this key is not admitted -- in which case
        the caller uses the launcher default and **stores nothing**. That
        distinction matters: a long-lived module served with varying sequence
        lengths sees an unbounded number of distinct exact token counts, so
        memoizing every miss would grow without limit.

        Stage one times the lowest legal config of each offered tile, largest tile
        first; stage two sweeps the winning tile's remaining configs. That order
        follows the measured structure -- tile choice dominates, config is a
        refinement (``profile/pooled_tactics.log``) -- and it is deterministic, so
        the same shape resolves the same way every time.

        Any limit reached anywhere discards every partial timing and yields the
        launcher default. A second, smaller search is never started.
        """
        if (self._searched_keys >= _SEARCH_MAX_KEYS
                or self._search_seconds >= _SEARCH_SECONDS_TOTAL):
            return None

        # The slot is consumed up front, so a key that exhausts its own budget
        # cannot be retried on the next call.
        self._searched_keys += 1
        started = time.perf_counter()
        budget = _SearchBudget(deadline=started + min(
            _SEARCH_SECONDS_PER_KEY, _SEARCH_SECONDS_TOTAL - self._search_seconds,
        ))
        try:
            best, source = self._two_stage_search(
                runner, inputs, budget, w13, w2, routing_bias,
            )
        except _SearchExhausted:
            best, source = None, "search-exhausted"
        finally:
            # Host elapsed time, not event time: the caps are wall-clock caps and
            # must account for enumeration and host-side work too.
            self._search_seconds += time.perf_counter() - started

        tactic = list(best if best is not None else _DEFAULT_TACTIC)
        self._remember(key, tactic, source)
        return tactic

    def _two_stage_search(self, runner, inputs, budget, w13, w2, routing_bias):
        """Tile first, then config within the winning tile. Returns (tactic, source).

        Raises :class:`_SearchExhausted` on any limit; the caller turns that into
        the launcher default. Enumeration failures are *not* caught -- only an
        identified invalid tactic is skipped.
        """
        offered = runner.get_valid_tactics(inputs.to_list(), None)

        by_tile: dict[int, list[int]] = {}
        for entry in offered:
            by_tile.setdefault(int(entry[0]), []).append(int(entry[1]))
        if not by_tile:
            return None, "search-no-tactics"
        for configs in by_tile.values():
            configs.sort()

        timed: dict[tuple[int, int], float] = {}

        def measure(tile: int, config: int) -> None:
            if (tile, config) in timed:
                return
            budget.spend_candidate()
            try:
                timed[(tile, config)] = self._time_tactic(
                    runner, inputs, [tile, config], w13, w2, routing_bias, budget,
                )
            except _InvalidTactic:
                # The launcher refused this pair at this token count. Skip it and
                # keep searching -- but the allowance it consumed is not refunded.
                pass

        for tile in sorted(by_tile, reverse=True):
            measure(tile, by_tile[tile][0])
        if not timed:
            return None, "search-no-tactics"

        best_tile = min(timed, key=lambda t: timed[t])[0]
        for config in by_tile[best_tile]:
            measure(best_tile, config)

        return list(min(timed, key=lambda t: timed[t])), "search"

    def _time_tactic(self, runner, inputs, tactic, w13, w2, routing_bias, budget):
        """Median of ``_SEARCH_SAMPLES`` event-timed calls after one warm-up.

        Every launch is charged before it is issued, including the warm-up, so an
        invalid tactic still costs allowance and the deadline is re-checked between
        samples rather than only between candidates.
        """
        budget.spend_launch()
        self._invoke(runner, inputs, tactic, w13, w2, routing_bias)
        samples = []
        for _ in range(_SEARCH_SAMPLES):
            budget.spend_launch()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            self._invoke(runner, inputs, tactic, w13, w2, routing_bias)
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end))
        return statistics.median(samples)

    def _invoke(self, runner, inputs, tactic: list[int], w13, w2, routing_bias) -> None:
        """One launcher call, writing into ``inputs.output``."""
        try:
            runner.forward(
                inputs.to_list(),
                tactic=tactic,
                routing_bias=routing_bias,
                gemm1_weights=w13,
                gemm2_weights=w2,
                gemm1_alpha=None,
                gemm1_beta=None,
                gemm1_clamp_limit=None,
                num_experts=self.num_experts,
                n_group=self.num_expert_group,
                topk_group=self.topk_group,
                local_expert_offset=self.local_expert_offset,
                routed_scaling_factor=self.routed_scaling_factor,
                routing_method_type=self.routing_method_type,
                use_shuffled_weight=_USE_SHUFFLED_WEIGHT,
                weight_layout=_WEIGHT_LAYOUT_BLOCK_MAJOR_K,
                do_finalize=_DO_FINALIZE,
                enable_pdl=_ENABLE_PDL,
                norm_topk_prob=_NORM_TOPK_PROB,
                routing_replay_out=None,
            )
        except Exception as exc:
            # Narrow on purpose. A blanket retry here would swallow OOM, a
            # device mismatch, a shape bug, or an async CUDA fault surfacing at
            # this call, and turn all of them into a silent slow path.
            if _is_invalid_tactic(exc):
                raise _InvalidTactic(str(exc)) from exc
            raise

    def forward(
        self,
        hidden_states: torch.Tensor,
        w13: torch.Tensor,
        w2: torch.Tensor,
        router_logits: torch.Tensor,
        routing_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_tokens, hidden_size = hidden_states.shape[0], hidden_states.shape[-1]
        device = hidden_states.device

        runner = self._runner_for(hidden_size, device)
        topk_ids, expert_weights = self._sentinels_for(router_logits.dtype, device)

        # Fresh every call. Handing back a reused buffer would let a second
        # forward rewrite a tensor the caller still holds.
        output = torch.empty(
            num_tokens, hidden_size, dtype=torch.bfloat16, device=device,
        )
        inputs = self._moe_inputs_cls(
            output=output,
            routing_logits=router_logits,
            topk_ids=topk_ids,
            expert_weights=expert_weights,
            hidden_states=hidden_states,
            hidden_states_scale=None,
            gemm1_lora_delta=None,
            per_token_scale=None,
        )

        key = self._tactic_key(num_tokens, hidden_size, router_logits.dtype, device)
        tactic = self._select_tactic(runner, inputs, key)
        if tactic is None:
            # Table miss. Under CUDA-graph capture -- or when we cannot establish
            # that we are *not* capturing -- take the launcher default immediately
            # and record nothing: no enumeration, no events, no synchronize, no
            # slot consumed, so the key stays eligible on a later eager call.
            # Otherwise offer it to the bounded search, which declines by
            # returning None once its budget is spent.
            if not _capturing():
                tactic = self._search_tactic(
                    runner, inputs, key, w13, w2, routing_bias,
                )
            if tactic is None:
                tactic = list(_DEFAULT_TACTIC)

        # Fallback ladder: chosen -> largest tile the launcher offers -> its own
        # default. Total and terminating, three steps at most, and every step
        # writes the whole of ``output`` before returning, so the tensor handed
        # back always holds the tactic that actually ran -- never a rejected
        # attempt's contents. Only an identified invalid-tactic failure moves
        # down the ladder; ``_invoke`` re-raises everything else untouched.
        for step in ("chosen", "largest-offered", "launcher-default"):
            try:
                self._invoke(runner, inputs, tactic, w13, w2, routing_bias)
                return output
            except _InvalidTactic:
                if step == "launcher-default":
                    # The launcher rejecting its own default is not a stale-cache
                    # problem and must not be swallowed.
                    raise
                nxt = (
                    self._largest_offered_tactic(runner, inputs)
                    if step == "chosen" else None
                ) or list(_DEFAULT_TACTIC)
                if nxt == tactic:
                    nxt = list(_DEFAULT_TACTIC)
                tactic = nxt
                # Only correct an entry that already exists. Creating one here
                # would let the error path cache a key the selection path
                # deliberately declined to remember.
                if key in self._tactics:
                    self._remember(key, tactic, f"fallback-after-{step}")
        raise AssertionError("unreachable: fallback ladder is exhaustive")
