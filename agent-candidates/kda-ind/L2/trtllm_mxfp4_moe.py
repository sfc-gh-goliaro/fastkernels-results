"""TRTLLM-gen MXFP4 fused MoE, launched directly and with a chosen kernel config.

Same math, same kernel, same arguments as :mod:`baseline`. Two things change, and
neither of them touches the arithmetic:

* **The kernel configuration is chosen instead of guessed.** ``trtllm-gen`` ships
  128-208 valid configurations for this problem, and
  ``flashinfer.trtllm_fp4_block_scale_moe`` only profiles them inside a
  ``flashinfer.autotune()`` context. Nothing in this benchmark ever enters one, so
  every call runs the built-in heuristic. Measured on this B200 with CUDA-graph
  replay (so host cost cannot mask the kernel), the heuristic leaves 1.70x on the
  table at 398 tokens and 1.81x at 16384 -- its N tile is simply too
  small once several hundred tokens land on each expert. Below ~64 tokens the gain
  is 1-2%, i.e. inside noise, and the heuristic is kept.
* **The Python wrapper is bypassed.** Per call it builds a ``MoERunner``, a
  ``TuningConfig`` including its bucket list, takes the autotuner lock, hashes a
  cache key, deduces two tensor dtypes and allocates ``topk_ids`` / ``topk_weights``
  -- about 0.125 ms of host time. Calling the JIT binding underneath it with
  hoisted scratch is bitwise identical (``torch.equal``) and drops host cost from
  0.459 ms to 0.334 ms per call. At one token, where the kernel itself is 17 us,
  that is the only lever there is.

Every entry point the fast path needs is private FlashInfer API, so it is resolved
once behind a guard that records *why* it was unavailable, and there are three
tiers to degrade through:

===========  ==================================================================
``tuned``    raw JIT binding, per-token-count selected configuration
``raw``      raw JIT binding, heuristic configuration (``[-1, -1]``)
``wrapper``  ``flashinfer.trtllm_fp4_block_scale_moe`` -- exactly the baseline
===========  ==================================================================

``FASTKERNELS_MXFP4_MOE_PATH`` pins the tier (``auto`` / ``raw`` / ``wrapper``);
``auto`` is the default. Pinning ``raw`` and ``wrapper`` is how the two levers are
measured apart, and how the fallback tiers get tested on their own.

Configuration selection runs on the first forward for a token count it has not
seen, which in this benchmark is a correctness round -- never a timed iteration.
It measures under CUDA-graph replay because plain event timing is host-bound here
and picks the wrong configuration (``[8, 13]`` at 0.367 ms instead of the real best
``[32, 29]`` at 0.212 ms at 398 tokens). Graphs are a measurement instrument only:
the timed forward path never captures or replays one, because the harness shifts
every argument's base address by 256 B per iteration and a captured graph would
read stale pointers.
"""

from __future__ import annotations

import os
import statistics
import time

import torch
import torch.nn as nn

from flashinfer import trtllm_fp4_block_scale_moe


# ``get_routing_method_type("softmax", renormalize=True, has_e_score_bias=False)``
# for gpt-oss -> RenormalizeNaive.
ROUTING_RENORMALIZE_NAIVE = 4

# gpt-oss SwiGLU-OAI constants, passed as gemm1_alpha / gemm1_beta /
# gemm1_clamp_limit.
SWIGLU_ALPHA = 1.702
SWIGLU_BETA = 1.0
SWIGLU_LIMIT = 7.0

# vLLM passes ``max(moe_config.max_capture_size, 1)``, 1024 for gpt-oss.
DEFAULT_TUNE_MAX_NUM_TOKENS = 1024

# The configuration sentinel that means "use the built-in heuristic". A real
# configuration is a two-element ``[tile_N, config_index]`` -- the launcher rejects
# any other length.
HEURISTIC_CONFIG = (-1, -1)

# ``fastkernels.bench`` compares bf16 outputs at atol=rtol=1e-2 and requires 99%
# of elements inside that bound. A selected configuration is held to exactly this
# bar against the heuristic's own output -- no tighter, or configurations the
# benchmark would have accepted get thrown away.
_BF16_ATOL = 1e-2
_BF16_RTOL = 1e-2
_REQUIRED_MATCHED_RATIO = 0.99

# A configuration has to be clearly, not marginally, faster to be worth adopting.
# The real wins are 1.70-1.81x; the shapes where the sweep reports 1-2% are
# reporting launch jitter, and those keep the heuristic.
_ADOPTION_MARGIN = 1.05

# Absolute times taken minutes apart under changing load are not comparable, so no
# adoption decision rests on them. Discovery only nominates one finalist per tile;
# the decision comes from re-timing each finalist against the heuristic back to back,
# twice, and comparing those *adjacent* ratios. One contended run had adopted an
# ordinary configuration at 26 and 60 tokens claiming 1.5x (real ceiling 1.02x)
# because its baseline sample was stale; another picked an N tile of 16 at 398 tokens
# where 32 is right, because the sweep mis-ranked and nothing downstream could notice.
_RERANK_REPS = 5
_RERANK_PASSES = 2

# Two finalists this close are not distinguishable by measurement -- at 398 tokens the
# best tile-16 and tile-32 configurations land within 0.5% of each other, and which one
# wins is a coin flip. Among finalists inside this band the larger N tile is taken: it
# runs fewer, bigger MMA tiles for the same work (52 128 blocks against 27 600 at 16384
# tokens), so it is the choice the profiling evidence points at, and picking it by rule
# rather than by whichever sample came out lower makes the adopted configuration
# reproducible instead of load-dependent.
_TIE_TOLERANCE = 0.02

# Share of the per-count budget spent discovering finalists. The rest is reserved for
# the rerank, so a slow discovery phase cannot leave the decision unmade -- it either
# covers every tile in time or the count keeps the heuristic.
_DISCOVERY_BUDGET_FRACTION = 0.7

# Selection budget. The benchmark kills a worker after 1200 s of wall clock and
# after 600 s with no output; the whole five-shape sweep measures in the tens of
# seconds, so these are guard rails rather than limits. Once the total budget is
# spent, later token counts silently keep the heuristic instead of tuning.
_BUDGET_PER_COUNT_S = float(os.environ.get("FASTKERNELS_MXFP4_MOE_BUDGET", "120"))
_BUDGET_TOTAL_S = float(os.environ.get("FASTKERNELS_MXFP4_MOE_BUDGET_TOTAL", "420"))

# Progress cadence, in seconds. Anything well under the 600 s stall watchdog does
# the job; 5 s also makes a slow sweep legible in the worker log.
_PROGRESS_INTERVAL_S = 5.0

# Measurement shape. ``_INNER_TARGET_MS`` is how much GPU work one graph replay
# should contain, so that event overhead stays a rounding error at every token
# count; ``_MAX_INNER`` caps how many calls get folded into one capture.
_INNER_TARGET_MS = 1.5
_MAX_INNER = 16
_GRAPH_WARMUP = 3
_SWEEP_REPS = 3
_BASELINE_REPS = 5

# A configuration this much worse than the best so far is not worth re-measuring.
_ABORT_RATIO = 1.25

# Every valid configuration ran cleanly when this device was swept, so a timing
# exception means the capture machinery is unhappy rather than the configuration being
# unusable -- and a failed capture can leave CUDA state that poisons unrelated work.
# One is therefore enough to forfeit tuning for the count in hand and to stop
# selecting for the rest of the module's life.

# Distinct token counts to remember configurations for. The capture directory for
# this operator holds ~600 of them, so an unbounded dict is a slow leak in a real
# serving loop. The diagnostic log is keyed the same way and is capped with it.
_CONFIG_CACHE_CAPACITY = 64

class _DirectLauncher:
    """The private trtllm-gen entry points, resolved once per process and device.

    ``moe_op`` is the raw JIT binding -- the thing the public wrapper calls on its
    last line. It is deliberately *not*
    ``flashinfer.fused_moe.core.get_trtllm_moe_sm100_module()``, which hands back
    FlashInfer's wrapper namespace: those members re-enter the autotuner and take
    their arguments in a different order.
    """

    __slots__ = (
        "moe_op",
        "routing_from_logits",
        "activation_swiglu",
        "activation_swiglu_value",
        "weight_layout_major_k",
        "fp8_none",
        "deduce_dtype",
        "enable_pdl",
    )

    def __init__(self, device: torch.device):
        # Diagnostic switch, read only: proves the fallback chain end to end without
        # having to break the install. Named so it cannot be mistaken for a tuning knob.
        if os.environ.get("FASTKERNELS_MXFP4_MOE_FORCE_RESOLUTION_FAILURE", "0") == "1":
            raise RuntimeError(
                "resolution failure forced by "
                "FASTKERNELS_MXFP4_MOE_FORCE_RESOLUTION_FAILURE=1")

        from flashinfer.fused_moe.core import (
            ActivationType,
            Fp8QuantizationType,
            RoutingInputMode,
            gen_trtllm_gen_fused_moe_sm100_module,
            get_trtllm_moe_sm100_module,
        )
        from flashinfer.jit.cubin_loader import setup_cubin_loader
        from flashinfer.tllm_enums import WeightLayout, deduce_trtllm_gen_tensor_dtype
        from flashinfer.utils import device_support_pdl

        # Warms the JIT module and its cubin loader through the public path first,
        # so a build failure surfaces here rather than half way through a capture.
        get_trtllm_moe_sm100_module()
        jit_module = gen_trtllm_gen_fused_moe_sm100_module()
        self.moe_op = jit_module.build_and_load()
        setup_cubin_loader(str(jit_module.get_library_path()))

        self.routing_from_logits = RoutingInputMode.FromLogits
        self.activation_swiglu = ActivationType(ActivationType.Swiglu.value)
        self.activation_swiglu_value = ActivationType.Swiglu.value
        self.weight_layout_major_k = WeightLayout(WeightLayout.MajorK)
        self.fp8_none = Fp8QuantizationType.NoneFp8
        self.deduce_dtype = deduce_trtllm_gen_tensor_dtype

        # Takes a ``torch.device``, not a device index: it reads ``device.type``,
        # so an int raises and would silently cost the whole fast path.
        self.enable_pdl = bool(device_support_pdl(device))

        for name in ("trtllm_fp4_block_scale_moe", "trtllm_get_valid_moe_configs"):
            if not hasattr(self.moe_op, name):
                raise AttributeError(f"JIT binding has no {name}")


class _SelectionFailure(Exception):
    """A timing or enumeration call failed, tagged with the stage it failed in.

    One exception type for every stage means the caller has exactly one place to
    forfeit tuning, record which stage broke, and account for the time spent -- rather
    than a per-stage patchwork where a stage added later quietly has no handler.
    """

    def __init__(self, stage: str, cause: BaseException):
        super().__init__(f"{stage} failed ({type(cause).__name__}: {cause})")
        self.stage = stage
        self.cause = cause
        self.reason = f"{type(cause).__name__}: {cause}"


class _VerificationMismatch(Exception):
    """The winner ran, but its output missed the benchmark's own bf16 bar."""


_LAUNCHER_CACHE: dict[int, tuple[_DirectLauncher | None, str | None]] = {}


def _direct_launcher(device: torch.device) -> tuple[_DirectLauncher | None, str | None]:
    """Resolve (and memoize) the raw binding, or the reason there isn't one."""
    key = torch.cuda.current_device() if device.index is None else device.index
    hit = _LAUNCHER_CACHE.get(key)
    if hit is not None:
        return hit
    try:
        resolved: tuple[_DirectLauncher | None, str | None] = (
            _DirectLauncher(torch.device("cuda", key)),
            None,
        )
    except Exception as exc:  # noqa: BLE001 - every name used above is private API
        resolved = (None, f"{type(exc).__name__}: {exc}")
    _LAUNCHER_CACHE[key] = resolved
    return resolved


def _within_bench_tolerance(out: torch.Tensor, ref: torch.Tensor) -> bool:
    """The benchmark's own bf16 verdict: 99% of elements inside atol+rtol*|ref|."""
    x = out.detach().to(torch.float32)
    y = ref.detach().to(torch.float32)
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        return False
    if torch.linalg.vector_norm(y).item() > 0 and torch.linalg.vector_norm(x).item() == 0:
        return False
    if x.numel() == 0:
        return True
    exceed = (x - y).abs() > (_BF16_ATOL + _BF16_RTOL * y.abs())
    return (1.0 - exceed.sum().item() / exceed.numel()) >= _REQUIRED_MATCHED_RATIO


class TrtLlmMxfp4MoE(nn.Module):
    """Router + experts in one trtllm-gen launch.

    ``hidden_states`` arrives at the *padded* hidden width; the returned tensor is
    ``hidden_size_unpadded`` wide, matching vLLM's ``has_unpadded_output``.

    Diagnostics, all public attributes: :attr:`launch_path` is the tier actually in
    use, :attr:`launch_path_reason` says why it is not a faster one,
    :attr:`enable_pdl` is the resolved PDL flag, :attr:`config_log` records what was
    chosen at each token count and what it cost to find, and
    :attr:`selection_seconds` is the total spent selecting.
    """

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        intermediate_size: int,
        hidden_size_unpadded: int,
        max_capture_size: int = DEFAULT_TUNE_MAX_NUM_TOKENS,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.intermediate_size = intermediate_size
        self.hidden_size_unpadded = hidden_size_unpadded
        self.max_capture_size = max(int(max_capture_size), 1)
        dev = torch.cuda.current_device()
        # Per-expert scalars, exactly as TrtLlmMxfp4ExpertsBase builds them.
        self.register_buffer(
            "gemm1_alpha",
            torch.full((num_experts,), SWIGLU_ALPHA, dtype=torch.float32, device=dev),
            persistent=False,
        )
        self.register_buffer(
            "gemm1_beta",
            torch.full((num_experts,), SWIGLU_BETA, dtype=torch.float32, device=dev),
            persistent=False,
        )
        self.register_buffer(
            "gemm1_clamp_limit",
            torch.full((num_experts,), SWIGLU_LIMIT, dtype=torch.float32, device=dev),
            persistent=False,
        )

        requested = os.environ.get("FASTKERNELS_MXFP4_MOE_PATH", "auto").strip().lower()
        if requested not in ("auto", "raw", "wrapper"):
            requested = "auto"
        self._requested_path = requested
        self._verbose = os.environ.get("FASTKERNELS_MXFP4_MOE_QUIET", "0") != "1"

        self._launcher: _DirectLauncher | None = None
        self.launch_path = "wrapper"
        self.launch_path_reason: str | None = None
        self.enable_pdl: bool | None = None
        self.config_log: dict[int, dict] = {}
        self.selection_seconds = 0.0
        self.selection_state = "active"
        # Distinct reasons the fast path stepped down, from a fixed vocabulary.
        self.degradations: list[str] = []

        if requested == "wrapper":
            self.launch_path_reason = "pinned to the public wrapper by environment"
        else:
            launcher, reason = _direct_launcher(torch.device("cuda", dev))
            if launcher is None:
                self.launch_path_reason = f"raw binding unavailable ({reason})"
            else:
                self._launcher = launcher
                self.enable_pdl = launcher.enable_pdl
                self.launch_path = "raw" if requested == "raw" else "tuned"
                if requested == "raw":
                    self.launch_path_reason = (
                        "pinned to the heuristic configuration by environment"
                    )

        # Hoisted routing scratch, grown on demand. The public wrapper allocates
        # both of these on every single call.
        self._topk_ids: torch.Tensor | None = None
        self._topk_weights: torch.Tensor | None = None
        self._config_cache: dict[int, tuple[int, int]] = {}
        # Set once the total selection budget is gone: later token counts keep the
        # heuristic rather than re-tuning forever.
        self._selection_closed = False
        # Verified equivalent to the public wrapper before the raw tier is trusted.
        self._binding_checked = False

        if self.launch_path != "tuned":
            self._selection_closed = True
            self.selection_state = f"not applicable on the {self.launch_path} path"
        self._log(
            f"path={self.launch_path} enable_pdl={self.enable_pdl}"
            + (f" reason={self.launch_path_reason}" if self.launch_path_reason else "")
        )

    # -- diagnostics ------------------------------------------------------------

    def _log(self, message: str) -> None:
        # The benchmark worker points fd 1 at stderr and writes its JSONL results
        # to a saved descriptor, so an ordinary print cannot corrupt the result
        # stream -- and it does refresh the log mtime the stall watchdog reads.
        if self._verbose:
            print(f"[mxfp4_moe] {message}", flush=True)

    def _degraded(self, reason: str) -> None:
        if reason not in self.degradations:
            self.degradations.append(reason)
            self._log(f"degraded: {reason}")

    def _close_selection(self, reason: str) -> None:
        self._selection_closed = True
        self.selection_state = f"closed: {reason}"
        self._degraded(f"configuration selection closed ({reason})")

    def _record(self, num_tokens: int, entry: dict) -> None:
        # Evict only for a genuinely new count; overwriting an existing one must not
        # cost an unrelated entry.
        if (num_tokens not in self.config_log
                and len(self.config_log) >= _CONFIG_CACHE_CAPACITY):
            self.config_log.pop(next(iter(self.config_log)))
        self.config_log[num_tokens] = entry

    def describe(self) -> dict:
        """Everything a reviewer needs to see which tier and config actually ran."""
        return {
            "launch_path": self.launch_path,
            "launch_path_reason": self.launch_path_reason,
            "enable_pdl": self.enable_pdl,
            "selection_state": self.selection_state,
            "degradations": list(self.degradations),
            "selection_seconds": round(self.selection_seconds, 3),
            "configs": {
                tokens: dict(entry) for tokens, entry in sorted(self.config_log.items())
            },
        }

    # -- launching --------------------------------------------------------------

    def _wrapper_call(
        self,
        output: torch.Tensor,
        routing_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        w13_weight: torch.Tensor,
        w13_weight_scale: torch.Tensor,
        w13_bias: torch.Tensor,
        w2_weight: torch.Tensor,
        w2_weight_scale: torch.Tensor,
        w2_bias: torch.Tensor,
    ) -> None:
        trtllm_fp4_block_scale_moe(
            routing_logits=routing_logits,
            routing_bias=None,
            hidden_states=hidden_states,
            hidden_states_scale=None,
            gemm1_weights=w13_weight,
            gemm1_weights_scale=w13_weight_scale,
            gemm1_bias=w13_bias,
            gemm1_alpha=self.gemm1_alpha,
            gemm1_beta=self.gemm1_beta,
            gemm1_clamp_limit=self.gemm1_clamp_limit,
            gemm2_weights=w2_weight,
            gemm2_weights_scale=w2_weight_scale,
            gemm2_bias=w2_bias,
            output1_scale_scalar=None,
            output1_scale_gate_scalar=None,
            output2_scale_scalar=None,
            num_experts=self.num_experts,
            top_k=self.top_k,
            n_group=None,
            topk_group=None,
            intermediate_size=self.intermediate_size,
            local_expert_offset=0,
            local_num_experts=self.num_experts,
            routed_scaling_factor=None,
            routing_method_type=ROUTING_RENORMALIZE_NAIVE,
            do_finalize=True,
            tune_max_num_tokens=self.max_capture_size,
            output=output,
        )

    def _routing_scratch(
        self, num_tokens: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ids, weights = self._topk_ids, self._topk_weights
        if ids is None or ids.shape[0] < num_tokens or ids.device != device:
            ids = torch.empty(num_tokens, self.top_k, dtype=torch.int32, device=device)
            self._topk_ids = ids
        if (
            weights is None
            or weights.shape[0] < num_tokens
            or weights.device != device
            or weights.dtype != dtype
        ):
            weights = torch.empty(num_tokens, self.top_k, dtype=dtype, device=device)
            self._topk_weights = weights
        return ids[:num_tokens], weights[:num_tokens]

    def _direct_call(
        self,
        launcher: _DirectLauncher,
        config,
        output: torch.Tensor,
        routing_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        w13_weight: torch.Tensor,
        w13_weight_scale: torch.Tensor,
        w13_bias: torch.Tensor,
        w2_weight: torch.Tensor,
        w2_weight_scale: torch.Tensor,
        w2_bias: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> None:
        """The raw binding, in its positional order, with the baseline's values."""
        launcher.moe_op.trtllm_fp4_block_scale_moe(
            launcher.routing_from_logits,
            routing_logits,
            topk_ids,
            topk_weights,
            None,                       # routing_bias
            hidden_states,
            None,                       # hidden_states_scale
            w13_weight,
            w13_weight_scale,
            w13_bias,
            self.gemm1_alpha,
            self.gemm1_beta,
            self.gemm1_clamp_limit,
            w2_weight,
            w2_weight_scale,
            w2_bias,
            None,                       # output1_scale_scalar
            None,                       # output1_scale_gate_scalar
            None,                       # output2_scale_scalar
            None,                       # per_token_scale
            self.num_experts,
            self.top_k,
            None,                       # n_group
            None,                       # topk_group
            self.intermediate_size,
            0,                          # local_expert_offset
            self.num_experts,           # local_num_experts
            None,                       # routed_scaling_factor
            ROUTING_RENORMALIZE_NAIVE,
            True,                       # do_finalize
            launcher.enable_pdl,
            launcher.activation_swiglu_value,
            output,
            list(config),
            True,                       # norm_topk_prob
            None,                       # routing_replay_out
        )

    # -- configuration selection ------------------------------------------------

    def _valid_configs(
        self,
        launcher: _DirectLauncher,
        hidden_states: torch.Tensor,
        w13_weight: torch.Tensor,
        w13_weight_scale: torch.Tensor,
        num_tokens: int,
    ) -> list:
        """Configurations trtllm-gen reports as valid for exactly this problem.

        The key mirrors ``MoERunner.get_valid_tactics``: it is token-count
        dependent, which is why a configuration chosen at one count may not even be
        listed at another.
        """
        return launcher.moe_op.trtllm_get_valid_moe_configs(
            launcher.deduce_dtype(hidden_states, None),
            launcher.deduce_dtype(w13_weight, w13_weight_scale),
            launcher.fp8_none,
            self.top_k,
            hidden_states.shape[-1],
            self.intermediate_size,
            self.num_experts,
            launcher.activation_swiglu,
            True,                       # use_shuffled_weight
            launcher.weight_layout_major_k,
            False,                      # use_per_token_scaling
            num_tokens,
            False,                      # has_gemm1_lora_delta
        )

    def _graph_ms(self, stage: str, call, config, inner: int, reps: int,
                  abort_above=None) -> float:
        """Median per-call GPU time, measured by replaying a captured graph.

        Replay is the point: the launcher costs ~0.33 ms of host time per call, so
        event-timing live calls measures the host and ranks configurations wrongly at
        every token count where the kernel is faster than that.

        Any failure is re-raised as a :class:`_SelectionFailure` tagged with *stage*, so
        the caller never has to guess which measurement broke.
        """
        try:
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(_GRAPH_WARMUP):
                    call(config)
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()

            graph = torch.cuda.CUDAGraph()
            try:
                self._capture(graph, call, config, inner)
                graph.replay()
                torch.cuda.synchronize()
                samples = []
                for _ in range(reps):
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    graph.replay()
                    end.record()
                    torch.cuda.synchronize()
                    samples.append(start.elapsed_time(end) / inner)
                    if abort_above is not None and samples[0] > abort_above:
                        break
                return statistics.median(samples)
            finally:
                del graph
        except Exception as exc:  # noqa: BLE001 - tagged and handled by the caller
            self._quiesce()
            raise _SelectionFailure(stage, exc) from exc

    def _capture(self, graph, call, config, inner: int) -> None:
        """Capture ``inner`` back-to-back calls, unwinding cleanly if it fails.

        Each capture gets its own private pool: sharing one ``graph_pool_handle()``
        across captures whose lifetimes do not overlap double-releases the mempool.

        ``torch.cuda.graph.__enter__`` enters its stream context *before* calling
        ``capture_begin``, and ``capture_begin`` registers the default CUDA generator
        for capture before it touches the allocator. So a failure inside
        ``capture_begin`` skips ``__exit__`` entirely and leaves two things broken
        process-wide: the side stream stays current, and the generator stays flagged
        as capturing, after which every ``torch.randn`` on this device raises. Both
        are unwound by hand here so a failed capture stays a local failure.
        """
        entry_stream = torch.cuda.current_stream()
        context = torch.cuda.graph(graph)
        try:
            context.__enter__()
        except BaseException:
            self._suppress(graph.capture_end)
            self._suppress(torch.cuda.set_stream, entry_stream)
            raise
        try:
            for _ in range(inner):
                call(config)
        finally:
            context.__exit__(None, None, None)

    @staticmethod
    def _suppress(fn, *args) -> None:
        """Best-effort cleanup step; already handling a failure, so never raise."""
        try:
            fn(*args)
        except Exception:  # noqa: BLE001
            pass

    # -- configuration selection ------------------------------------------------

    def _select_config(self, launcher, num_tokens, *call_args) -> tuple[int, int]:
        """Choose a configuration for this token count, or keep the heuristic.

        Owns the two things that must happen exactly once no matter how selection
        ends: the elapsed time is added to :attr:`selection_seconds`, and one
        diagnostic record is written for the count.
        """
        started = time.perf_counter()
        entry = {"config": list(HEURISTIC_CONFIG), "adopted": False}
        chosen = HEURISTIC_CONFIG
        try:
            chosen = self._discover_and_rerank(launcher, num_tokens, entry, *call_args)
        except _SelectionFailure as failure:
            entry["failed_stage"] = failure.stage
            entry["failure"] = failure.reason
            entry["verdict"] = (
                f"{failure.stage} failed ({failure.reason}); keeping the heuristic"
            )
            entry["cuda_rng_healthy"] = self._cuda_rng_healthy(
                self.gemm1_alpha.device)
            self._close_selection(
                f"{failure.stage} failed at {num_tokens} tokens ({failure.reason})")
        except _VerificationMismatch as mismatch:
            # Not a broken measurement: this configuration really does compute a
            # different answer, so drop it and carry on selecting for other counts.
            entry["failed_stage"] = "verification"
            entry["verdict"] = f"rejected at verification: {mismatch}"
        finally:
            elapsed = time.perf_counter() - started
            self.selection_seconds += elapsed
            entry["select_seconds"] = round(elapsed, 3)
            self._record(num_tokens, entry)

        if not self._selection_closed and self.selection_seconds >= _BUDGET_TOTAL_S:
            self._close_selection(
                f"total budget of {_BUDGET_TOTAL_S:g}s spent after "
                f"{len(self.config_log)} token counts")
        self._log(f"{num_tokens} tokens: chose {list(chosen)} -- {entry['verdict']} "
                  f"({entry['select_seconds']}s)")
        return chosen

    def _discover_and_rerank(self, launcher, num_tokens, entry, routing_logits,
                             hidden_states, w13_weight, w13_weight_scale, w13_bias,
                             w2_weight, w2_weight_scale, w2_bias, topk_ids,
                             topk_weights) -> tuple[int, int]:
        """Nominate one finalist per N tile, then decide between them head to head.

        Writes only into its own scratch output, so the caller's output tensor is
        untouched and the real call that follows is the one that fills it.
        """
        started = time.perf_counter()
        scratch = torch.empty(
            num_tokens, self.hidden_size_unpadded,
            dtype=torch.bfloat16, device=hidden_states.device,
        )

        def call(config):
            self._direct_call(
                launcher, config, scratch, routing_logits, hidden_states,
                w13_weight, w13_weight_scale, w13_bias,
                w2_weight, w2_weight_scale, w2_bias, topk_ids, topk_weights,
            )

        try:
            configs = [tuple(int(v) for v in c) for c in self._valid_configs(
                launcher, hidden_states, w13_weight, w13_weight_scale, num_tokens)]
        except Exception as exc:  # noqa: BLE001
            raise _SelectionFailure("enumeration", exc) from exc

        # Group by N tile. The tile is what actually decides performance here -- 32
        # against 64 at 16384 tokens is a 1.89x difference at identical DRAM traffic --
        # so every tile gets a representative rather than every configuration getting
        # an equal chance of being lost to a truncated sweep.
        by_tile: dict[int, list[tuple[int, int]]] = {}
        for config in configs:
            by_tile.setdefault(config[0], []).append(config)
        entry["candidates"] = len(configs)
        entry["tiles"] = sorted(by_tile)

        probe = self._graph_ms("probe", call, HEURISTIC_CONFIG, inner=4, reps=3)
        inner = int(min(_MAX_INNER, max(1, round(_INNER_TARGET_MS / max(probe, 1e-4)))))
        heuristic_ms = self._graph_ms(
            "baseline", call, HEURISTIC_CONFIG, inner=inner, reps=_BASELINE_REPS)
        entry["heuristic_ms"] = round(heuristic_ms, 5)
        entry["inner"] = inner

        budget = min(_BUDGET_PER_COUNT_S,
                     max(0.0, _BUDGET_TOTAL_S - self.selection_seconds))
        deadline = started + budget
        discovery_deadline = started + budget * _DISCOVERY_BUDGET_FRACTION

        finalists = self._discover(call, by_tile, inner, heuristic_ms,
                                   discovery_deadline, entry, num_tokens)
        if finalists is None:
            return HEURISTIC_CONFIG
        return self._rerank(call, scratch, finalists, inner, deadline, entry,
                            num_tokens)

    def _discover(self, call, by_tile, inner, heuristic_ms, deadline, entry,
                  num_tokens):
        """Time configurations tile by tile in round-robin order.

        Round-robin matters: a sequential sweep that runs out of budget has covered a
        prefix of one tile and knows nothing about the others, and then adopts whatever
        it happened to see. Interleaving means a truncated pass is missing the *tail* of
        every tile rather than all of most of them -- and if any tile ends up with no
        measurement at all, nothing is adopted.

        Returns the per-tile finalists, or ``None`` if a tile went uncovered.
        """
        rounds = max(len(v) for v in by_tile.values())
        order = [by_tile[tile][i]
                 for i in range(rounds)
                 for tile in sorted(by_tile) if i < len(by_tile[tile])]

        best_per_tile: dict[int, tuple[float, tuple[int, int]]] = {}
        best_overall = heuristic_ms
        last_progress = time.perf_counter()
        measured = 0
        for index, config in enumerate(order):
            if time.perf_counter() > deadline:
                entry["discovery_truncated_after"] = measured
                break
            elapsed = self._graph_ms("sweep", call, config, inner=inner,
                                     reps=_SWEEP_REPS,
                                     abort_above=_ABORT_RATIO * best_overall)
            measured += 1
            best_overall = min(best_overall, elapsed)
            tile = config[0]
            if tile not in best_per_tile or elapsed < best_per_tile[tile][0]:
                best_per_tile[tile] = (elapsed, config)
            now = time.perf_counter()
            if now - last_progress >= _PROGRESS_INTERVAL_S:
                last_progress = now
                self._log(f"{num_tokens} tokens: {index + 1}/{len(order)} "
                          f"configurations, {len(best_per_tile)}/{len(by_tile)} tiles, "
                          f"best {best_overall:.4f} ms")

        entry["measured"] = measured
        entry["tiles_covered"] = sorted(best_per_tile)
        missing = sorted(set(by_tile) - set(best_per_tile))
        if missing:
            entry["verdict"] = (
                f"discovery covered {len(best_per_tile)}/{len(by_tile)} tiles "
                f"(missing {missing}); keeping the heuristic"
            )
            return None
        return [best_per_tile[tile][1] for tile in sorted(by_tile)]

    def _rerank(self, call, scratch, finalists, inner, deadline, entry,
                num_tokens):
        """Decide between the tile finalists on adjacent, normalized measurements.

        Each finalist is timed immediately after the heuristic, twice, and scored on
        the ratio of those adjacent readings. Comparing ratios rather than absolute
        times is what makes the decision independent of how loaded the machine was when
        a given configuration happened to come up during discovery.
        """
        scores = []
        for config in finalists:
            if time.perf_counter() > deadline:
                entry["rerank_truncated_after"] = len(scores)
                entry["verdict"] = (
                    f"rerank truncated after {len(scores)}/{len(finalists)} finalists; "
                    f"keeping the heuristic"
                )
                return HEURISTIC_CONFIG
            ratios = []
            for _ in range(_RERANK_PASSES):
                incumbent = self._graph_ms("rerank", call, HEURISTIC_CONFIG,
                                           inner=inner, reps=_RERANK_REPS)
                challenger = self._graph_ms("rerank", call, config, inner=inner,
                                            reps=_RERANK_REPS)
                ratios.append(incumbent / challenger)
            # The worst of the passes: a win has to survive both.
            scores.append((min(ratios), config))

        scores.sort(key=lambda s: -s[0])
        entry["rerank"] = [{"config": list(c), "margin": round(m, 4)} for m, c in scores]
        # Break a measured tie toward the larger N tile rather than toward whichever
        # sample happened to come out lower.
        tied = [(m, c) for m, c in scores if m >= scores[0][0] * (1.0 - _TIE_TOLERANCE)]
        margin, winner = max(tied, key=lambda s: s[1][0])
        entry["tied_within_tolerance"] = [list(c) for _, c in tied]
        entry["margin"] = round(margin, 4)
        if margin < _ADOPTION_MARGIN:
            entry["verdict"] = f"best confirmed margin {margin:.3f}x under {_ADOPTION_MARGIN}x"
            return HEURISTIC_CONFIG

        self._verify(winner, call, scratch)
        entry["config"] = list(winner)
        entry["adopted"] = True
        entry["verdict"] = f"{margin:.3f}x faster, output verified"
        return winner

    def _verify(self, config, call, scratch) -> None:
        """Hold the winner to the benchmark's own bf16 bar before trusting it.

        Raises :class:`_SelectionFailure` if the configuration could not be run at all,
        and :class:`_VerificationMismatch` if it ran and disagreed. Those are different
        findings: the first means the measuring machinery is suspect, the second means
        this one configuration is wrong.
        """
        try:
            call(HEURISTIC_CONFIG)
            torch.cuda.synchronize()
            reference = scratch.clone()
            call(config)
            torch.cuda.synchronize()
        except Exception as exc:  # noqa: BLE001 - must not fail the caller's forward
            self._quiesce()
            raise _SelectionFailure("verification", exc) from exc
        if not _within_bench_tolerance(scratch, reference):
            raise _VerificationMismatch("output disagreed with the heuristic")

    @staticmethod
    def _quiesce() -> None:
        """Drain the device after a failure, without raising a second time."""
        try:
            torch.cuda.synchronize()
        except Exception:  # noqa: BLE001 - already handling a failure
            pass

    @staticmethod
    def _cuda_rng_healthy(device: torch.device) -> bool:
        """Whether the default CUDA generator still works outside graph capture.

        An aborted capture can leave it flagged as capturing, after which every
        ``torch.randn`` on the device raises -- including the benchmark's own input
        materialization, several scenarios later, with nothing pointing back here.
        Recording the answer is what makes that failure mode legible.
        """
        try:
            state = torch.cuda.get_rng_state(device)
        except Exception:  # noqa: BLE001
            return False
        try:
            torch.empty(1, device=device).normal_()
            return True
        except Exception:  # noqa: BLE001
            return False
        finally:
            try:
                torch.cuda.set_rng_state(state, device)
            except Exception:  # noqa: BLE001
                pass

    def _config_for(self, launcher, num_tokens, *call_args) -> tuple[int, int]:
        """The configuration to use at this token count, selecting once if needed.

        Cached strictly per token count and never reused across counts: the valid
        set is itself token-count dependent (144 entries at 26 tokens, 208 at 398,
        128 at 16384), so a configuration carried over may not even be listed, let
        alone correct.
        """
        cached = self._config_cache.get(num_tokens)
        if cached is not None:
            return cached
        if self.launch_path != "tuned":
            return HEURISTIC_CONFIG
        if self._selection_closed:
            # Record per count, so "this count never got tuned" is visible rather
            # than indistinguishable from "this count was tuned and lost".
            self._record(num_tokens, {
                "config": list(HEURISTIC_CONFIG),
                "adopted": False,
                "verdict": self.selection_state,
            })
            return HEURISTIC_CONFIG
        try:
            chosen = self._select_config(launcher, num_tokens, *call_args)
        except Exception as exc:  # noqa: BLE001 - a bug here must not fail the forward
            self._quiesce()
            chosen = HEURISTIC_CONFIG
            self._close_selection(f"selection raised ({type(exc).__name__})")
            self._record(num_tokens, {
                "config": list(HEURISTIC_CONFIG),
                "adopted": False,
                "failed_stage": "unexpected",
                "failure": f"{type(exc).__name__}: {exc}",
                "verdict": f"unexpected failure ({type(exc).__name__}: {exc})",
            })
            self._log(f"{num_tokens} tokens: selection raised ({exc!r}); heuristic")
        if len(self._config_cache) >= _CONFIG_CACHE_CAPACITY:
            self._config_cache.pop(next(iter(self._config_cache)))
        self._config_cache[num_tokens] = chosen
        return chosen

    def _check_binding(self, output, routing_logits, hidden_states, weights, scratch_args):
        """Confirm the raw binding reproduces the wrapper bitwise, once per module.

        The two entry points take their arguments in a different order and the fast
        one is private, so this is the thing standing between a rename upstream and
        silently wrong numbers.
        """
        launcher = self._launcher
        reference = torch.empty_like(output)
        self._wrapper_call(reference, routing_logits, hidden_states, *weights)
        try:
            self._direct_call(
                launcher, HEURISTIC_CONFIG, output, routing_logits, hidden_states,
                *weights, *scratch_args,
            )
            torch.cuda.synchronize()
            matched = torch.equal(output, reference)
            reason = None if matched else "did not match the public wrapper bitwise"
        except Exception as exc:  # noqa: BLE001 - a private binding may change shape
            self._quiesce()
            reason = f"raised on first call ({type(exc).__name__}: {exc})"
        if reason is None:
            self._binding_checked = True
            self._log("raw binding is bitwise equal to the public wrapper")
            return True
        self._fall_back_to_wrapper(f"raw binding {reason}")
        output.copy_(reference)
        return False

    def _fall_back_to_wrapper(self, reason: str) -> None:
        self._launcher = None
        self.launch_path = "wrapper"
        self.launch_path_reason = reason
        self.selection_state = "closed: fell back to the public wrapper"
        self._selection_closed = True
        self._degraded(reason)

    # -- forward ----------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        w13_weight: torch.Tensor,
        w13_weight_scale: torch.Tensor,
        w13_bias: torch.Tensor,
        w2_weight: torch.Tensor,
        w2_weight_scale: torch.Tensor,
        w2_bias: torch.Tensor,
    ) -> torch.Tensor:
        assert hidden_states.dtype == torch.bfloat16
        output = torch.empty(
            *hidden_states.shape[:-1],
            self.hidden_size_unpadded,
            dtype=torch.bfloat16,
            device=hidden_states.device,
        )
        routing_logits = router_logits.to(torch.bfloat16)
        weights = (
            w13_weight, w13_weight_scale, w13_bias,
            w2_weight, w2_weight_scale, w2_bias,
        )

        launcher = self._launcher
        # The raw binding indexes tokens as dim 0; anything but a 2-D activation
        # goes through the wrapper, which is what the baseline does anyway.
        if launcher is None:
            self._wrapper_call(output, routing_logits, hidden_states, *weights)
            return output
        if hidden_states.dim() != 2:
            self._degraded(
                f"activation is {hidden_states.dim()}-D, not 2-D; using the wrapper")
            self._wrapper_call(output, routing_logits, hidden_states, *weights)
            return output

        num_tokens = hidden_states.shape[0]
        topk_ids, topk_weights = self._routing_scratch(
            num_tokens, hidden_states.device, routing_logits.dtype)

        if not self._binding_checked:
            if not self._check_binding(
                output, routing_logits, hidden_states, weights,
                (topk_ids, topk_weights),
            ):
                return output
            # The check already left the heuristic result in ``output``; fall
            # through so a selected configuration still gets a chance to run.

        config = self._config_for(
            launcher, num_tokens, routing_logits, hidden_states,
            *weights, topk_ids, topk_weights,
        )
        try:
            self._direct_call(
                launcher, config, output, routing_logits, hidden_states, *weights,
                topk_ids, topk_weights,
            )
        except Exception as exc:  # noqa: BLE001 - degrade rather than fail the call
            self._quiesce()
            self._fall_back_to_wrapper(
                f"raw binding raised with configuration {list(config)} "
                f"({type(exc).__name__}: {exc})")
            self._wrapper_call(output, routing_logits, hidden_states, *weights)
        return output
