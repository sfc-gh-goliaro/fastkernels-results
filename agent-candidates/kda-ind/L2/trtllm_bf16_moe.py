"""TRTLLM-gen BF16 fused MoE reached through the launcher's tactic-aware runner.

Same kernels as the baseline, same cubins, a shorter host path. On the four
decode shapes ``fastkernels bench`` scores, the baseline is host-bound rather
than device-bound: the CPU needs ~0.70 ms to enqueue one forward while the
device work is ~0.55 ms, so the measured latency sits at ~0.78 ms whether the
MoE kernels take 27 us or 195 us. Two costs make up that host time and both are
avoidable without touching a kernel.

*FlashInfer's Python wrapper spends ~0.11 ms per call on an autotuner that never
fires here.* ``trtllm_bf16_moe_op`` builds a ``MoERunner``, a
``MoeRunnerInputs`` and a fresh ``TuningConfig`` on every call and hands them to
``AutoTuner.choose_one``. Under ``fastkernels bench`` nothing populates the
autotuner -- ``is_tuning_mode`` is False, ``profiling_cache`` and
``_file_configs`` are empty and ``FLASHINFER_AUTOTUNER_LOAD_FROM_FILE`` defaults
to ``0`` -- so ``choose_one`` misses every time and returns tactic ``-1``.

*The C++ launcher spends ~0.2 ms selecting a config it was never told to
choose.* With ``tactic == -1`` it runs ``getValidConfigIndices`` twice, once
inside ``getDefaultValidConfigIndex`` and once to validate the answer, sorting
the passing configs and re-running ``isValidConfig`` against a freshly built
vector of per-expert token counts for both GEMMs. Passing an explicit tactic
skips the duplicate scan.

The seam used here is ``get_trtllm_moe_sm100_module().MoERunner``, which the
library exposes as its "canonical tactic-aware TunableRunner ... so the unified
MoE API's TrtllmFp4RoutedRunner can delegate to it instead of re-deriving the raw
op's positional call". Measured under the harness's own ``_time_module`` on all
five scored shapes it reaches 1.172x against the baseline wrapper, versus 1.182x
for a hand-written 29-argument positional call into the raw TVM-FFI op -- 0.8%
apart, which does not buy the raw call's failure mode, where a type-compatible
argument reordering in a future FlashInfer release computes silently wrong
results instead of raising. ``tools/seam_bench.py`` records the numbers.

Tactics come from ``docs/tactics.json``, measured by ``tools/tune_tactics.py``
under CUDA-graph replay across three seeds. Every valid tactic on this launcher
reproduces the ``[-1, -1]`` result **bitwise**: a ``[tile_N, config]`` pair
selects the output tiling and the schedule, not the order of the K-dimension
reduction. So the table costs nothing numerically while buying both the skipped
config scan and, where the tiling genuinely suits the shape, real device time
(Kimi n=16384 1.138 -> 0.992 ms, Kimi n=64 0.059 -> 0.049 ms).

Anything the fast path cannot vouch for is decided *before* a kernel is
enqueued and routed to :meth:`TrtLlmBf16MoE._fallback`, which is the baseline's
``flashinfer.fused_moe.trtllm_bf16_moe`` call argument for argument.

Mirrors ``TrtLlmBf16ExpertsMonolithic.apply``
(``vllm/model_executor/layers/fused_moe/experts/trtllm_bf16_moe.py``).
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import torch
import torch.nn as nn

try:
    from flashinfer.fused_moe import trtllm_bf16_moe as _trtllm_bf16_moe
except Exception:  # noqa: BLE001 - degrade at call time, not at import time
    _trtllm_bf16_moe = None


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

# ``blockK`` is 128 *bytes*, so a bf16 BlockMajorK tile is 64 elements wide. Every
# 4D expert weight this kernel accepts ends in that dimension.
_BLOCK_ELEMS_BF16 = 64

# Enum values the launcher's own call site passes, restated rather than imported
# so this module still imports with no flashinfer present. Restating them makes
# them an upgrade-sensitive surface, so ``_seam_intact`` checks them against the
# live enums once per process before the fast path is ever used.
_WEIGHT_LAYOUT_BLOCK_MAJOR_K = 2
_DTYPE_TRTLLM_GEN_BF16 = 1052672      # DtypeTrtllmGen.Bfloat16
_FP8_QUANT_NONE = 0                   # Fp8QuantizationType.NoneFp8

# ``MoeRunnerInputs._FIELDS``: the field order that defines the flat input list
# this module builds by hand. A reordering upstream would be type-compatible and
# silent, so it is checked rather than trusted.
_INPUT_FIELDS = (
    "output",
    "routing_logits",
    "topk_ids",
    "expert_weights",
    "hidden_states",
    "hidden_states_scale",
    "gemm1_lora_delta",
    "per_token_scale",
)

# The launcher rejects any routing method whose scoring this module has not been
# measured against. 1 Renormalize, 2 DeepSeekV3, 4 RenormalizeNaive.
_SUPPORTED_ROUTING = frozenset((ROUTING_RENORMALIZE, ROUTING_DEEPSEEK_V3,
                                ROUTING_RENORMALIZE_NAIVE))
_GROUPED_ROUTING = frozenset((ROUTING_DEEPSEEK_V3,))

_DEFAULT_TACTIC = [-1, -1]

# ``get_trtllm_moe_sm100_module`` is ``functools.cache``d upstream but
# ``JitSpec.build_and_load`` underneath it takes a ``FileLock`` and re-runs
# ``build()``, so the first resolution can be slow and must not be raced. Cache
# the outcome -- including a failure -- behind one lock, and never start a thread
# to warm it: the harness fails a candidate whose thread count rises during the
# timed region.
_MODULE_LOCK = threading.Lock()
_MODULE_CACHE: list = []

# The uncached JIT handle, held only for the read-only validity query and the
# library hash -- never for launching. ``gen_trtllm_gen_fused_moe_sm100_module``
# is not ``functools.cache``d upstream and ``build_and_load`` re-takes a
# ``FileLock`` and re-runs ``build()``, so resolving it costs ~55 ms and must
# happen once. Its own lock, not ``_MODULE_LOCK``: ``_raw_module`` calls
# ``_moe_module`` first, and ``threading.Lock`` is not reentrant.
_RAW_LOCK = threading.Lock()
_RAW_CACHE: list = []

# Whether the library still has the shape this module was measured against.
# Resolved once, before the first fast-path call.
_SEAM_CACHE: list = []

# Tactic resolution is keyed on shapes and scalars only. Nothing here may key on
# a ``data_ptr``: the harness re-copies every contiguous input into a fresh pool
# slot on every iteration, so pointers change by design and a pointer-keyed cache
# would be both useless and a correctness hazard.
# Both caches are scoped by CUDA device index. A process holding two different
# SM100 products would otherwise load one device's table and reuse one device's
# prevalidated tactic on the other, since the launcher's validity query takes only
# scalars and consults the current device rather than a tensor's.
_TACTIC_CACHE: dict[tuple, list[int]] = {}
_TABLE_CACHE: dict[int, dict[tuple, list[int]]] = {}
_TABLE_SCHEMA_VERSION = 1

# The launcher's validity answer per (device index, applicability key). Owned
# here rather than read from ``MoERunner.valid_tactics_dict``, which is a *class*
# attribute keyed on the 13 scalars alone: it has no device component, and
# ``trtllm_get_valid_moe_configs`` takes no device argument and answers for
# whichever device is current. Sharing that cache across devices would let a set
# measured on one GPU authorize a tactic on another.
_VALID_CACHE: dict[tuple, frozenset] = {}

# Setting this to any nonempty value makes FlashInfer skip cubin checksum
# verification (``jit/cubin_loader.py`` reads it with ``os.getenv`` truthiness, so
# even "0" disables it). The committed table's fingerprint hashes the launcher
# ``.so``, not the individual tactic cubins, so under this variable modified
# kernel bytes can load while every fingerprint field still matches -- exactly the
# "applied to an environment it was not measured on" case the fingerprint exists
# to prevent. The table is refused outright rather than trusted.
_CHECKSUM_DISABLED_ENV = "FLASHINFER_CUBIN_CHECKSUM_DISABLED"

# ``torch.cuda.get_device_capability`` is cheap but not free, and the admission
# gate asks for it on every call. Keyed on the device index, which is all the
# answer depends on.
_SM_MAJOR: dict[int, int] = {}


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


def _moe_module():
    """The launcher module the baseline's wrapper also reaches, or ``None``.

    Resolved once per process. Returning ``None`` rather than raising is what
    makes a cold cache, a missing artifact or a non-Blackwell device degrade to
    the wrapper instead of failing a benchmark round.
    """
    if _MODULE_CACHE:
        return _MODULE_CACHE[0]
    with _MODULE_LOCK:
        if _MODULE_CACHE:
            return _MODULE_CACHE[0]
        try:
            from flashinfer.fused_moe.core import get_trtllm_moe_sm100_module
            module = get_trtllm_moe_sm100_module()
        except Exception:  # noqa: BLE001 - no artifact, no network, no device
            module = None
        # A failure is cached too. A loader that failed once in this process
        # fails again for the same reason -- a missing artifact, no network, a
        # build error -- and re-attempting it costs ~55 ms per forward because
        # gen_trtllm_gen_fused_moe_sm100_module is not cached upstream. The cost
        # of caching it is that a genuinely transient failure retires the fast
        # path for the process; the fallback still answers every call.
        _MODULE_CACHE.append(module)
        return module


def _raw_module():
    """``(jit_spec, loaded_module)`` for the launcher, or ``None``.

    Held for two read-only purposes: hashing the library for the fingerprint, and
    calling ``trtllm_get_valid_moe_configs``, which the ``SimpleNamespace`` from
    ``get_trtllm_moe_sm100_module`` does not expose. Launching still goes through
    ``MoERunner``, so the argument-order hazard that ruled out a hand-written
    positional call is not reintroduced -- a query whose signature changed raises
    or returns something that fails the membership test, and either way the
    tactic degrades to ``[-1, -1]``.

    ``_moe_module`` is called before the lock is taken, both because it runs
    ``setup_cubin_loader`` and because taking ``_MODULE_LOCK`` underneath this one
    would deadlock on a non-reentrant lock.
    """
    if _RAW_CACHE:
        return _RAW_CACHE[0]
    module = _moe_module()
    with _RAW_LOCK:
        if _RAW_CACHE:
            return _RAW_CACHE[0]
        raw = None
        if module is not None:
            try:
                from flashinfer.jit.fused_moe import (
                    gen_trtllm_gen_fused_moe_sm100_module,
                )
                spec = gen_trtllm_gen_fused_moe_sm100_module()
                raw = (spec, spec.build_and_load())
            except Exception:  # noqa: BLE001 - no artifact, no build, no device
                raw = None
        _RAW_CACHE.append(raw)
        return _RAW_CACHE[0]


def _valid_tactics(device: torch.device, key: tuple) -> frozenset:
    """The tactics the launcher itself calls valid for *key*, asked of *device*.

    The query is run under ``torch.cuda.device(device)`` because it takes no
    device argument and answers for whichever device is current -- so asking it
    about a tensor on a device the process has not made current would return
    another GPU's answer. Cached per (device index, key), which is once per
    distinct token count per device per process.
    """
    index = device.index if device.index is not None else torch.cuda.current_device()
    cache_key = (index, key)
    cached = _VALID_CACHE.get(cache_key)
    if cached is not None:
        return cached
    valid: frozenset = frozenset()
    raw = _raw_module()
    if raw is not None:
        try:
            with torch.cuda.device(device):
                answer = raw[1].trtllm_get_valid_moe_configs(*key)
            valid = frozenset(
                (int(t), -1) if isinstance(t, int) else tuple(int(x) for x in t)
                for t in answer)
        except Exception:  # noqa: BLE001 - host-side query, nothing enqueued
            valid = frozenset()
    _VALID_CACHE[cache_key] = valid
    return valid


def _seam_intact() -> bool:
    """Does the installed library still match the assumptions compiled in here?

    Two of this module's assumptions would break silently rather than loudly if
    FlashInfer changed them: the flat ``MoeRunnerInputs`` field order, which the
    fast path builds positionally, and the four enum values restated above. Both
    are checked once, and a mismatch retires the fast path for the process rather
    than producing a plausible wrong answer.
    """
    if _SEAM_CACHE:
        return _SEAM_CACHE[0]
    with _MODULE_LOCK:
        if _SEAM_CACHE:
            return _SEAM_CACHE[0]
        try:
            from flashinfer.fused_moe.core import MoeRunnerInputs
            from flashinfer.tllm_enums import (
                ActivationType,
                DtypeTrtllmGen,
                Fp8QuantizationType,
                WeightLayout,
            )
            ok = (
                tuple(MoeRunnerInputs._FIELDS) == _INPUT_FIELDS
                and int(DtypeTrtllmGen.Bfloat16) == _DTYPE_TRTLLM_GEN_BF16
                and int(Fp8QuantizationType.NoneFp8) == _FP8_QUANT_NONE
                and int(ActivationType.Swiglu) == ACTIVATION_SWIGLU
                and int(WeightLayout.BlockMajorK) == _WEIGHT_LAYOUT_BLOCK_MAJOR_K
            )
        except Exception:  # noqa: BLE001 - an unreadable seam is a mismatch
            ok = False
        _SEAM_CACHE.append(bool(ok))
        return _SEAM_CACHE[0]


def _live_fingerprint(device: torch.device) -> dict:
    """What the running process is, in every field a committed table pins.

    Read off *device* rather than the current device: the two differ whenever a
    caller passes tensors on a device it has not made current, and a table
    measured on one GPU must not be honoured on another.

    The library hash is the field that actually pins kernel identity: a rebuilt
    artifact under the same version and filename still changes which kernel a
    ``[tile_N, config]`` pair names. Resolving it costs ~70 ms once per process
    -- ``gen_trtllm_gen_fused_moe_sm100_module`` is not cached upstream and
    re-derives its JitSpec on each call, then 14 MB has to be hashed -- which is
    why this runs behind ``_TABLE_CACHE`` and lands in a correctness round rather
    than the timed region.

    Not covered, and worth knowing: the CUDA driver version, and the individual
    tactic cubins, which ``get_artifact`` checksums on download but which a
    ``FLASHINFER_CUBIN_CHECKSUM_DISABLED`` environment could let diverge while
    the launcher hash still matches.
    """
    import flashinfer

    props = torch.cuda.get_device_properties(device)
    fingerprint = {
        "schema_version": _TABLE_SCHEMA_VERSION,
        "flashinfer_version": flashinfer.__version__,
        "torch_version": torch.__version__,
        "gpu_name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "moe_library": None,
        "moe_library_sha256_16": None,
    }
    try:
        import hashlib

        raw = _raw_module()
        lib = Path(str(raw[0].get_library_path()))
        fingerprint["moe_library"] = lib.name
        fingerprint["moe_library_sha256_16"] = hashlib.sha256(
            lib.read_bytes()).hexdigest()[:16]
    except Exception:  # noqa: BLE001 - absence is a mismatch, not an error
        pass
    return fingerprint


def _tactic_table(device: torch.device) -> dict[tuple, list[int]]:
    """``docs/tactics.json`` as an applicability-key -> tactic map, or empty.

    The table is only honoured when its recorded environment still matches the
    one *device* is in. A FlashInfer bump, a different GPU or a rebuilt cubin all
    change which kernel a ``[tile_N, config]`` pair names, so a mismatch in any
    fingerprint field drops every entry and the launcher's own default is used --
    silently, because a stale table is a performance question, not a correctness
    one.

    Resolved once per device for the life of the process, so a table file or an
    environment variable that changes afterwards is not reconsidered.
    """
    index = device.index if device.index is not None else torch.cuda.current_device()
    cached = _TABLE_CACHE.get(index)
    if cached is not None:
        return cached
    table: dict[tuple, list[int]] = {}
    path = os.environ.get("FASTKERNELS_MOE_TACTICS")
    path = Path(path) if path else Path(__file__).resolve().parents[2] / "docs" / "tactics.json"
    try:
        # Cubin checksum verification turned off means the fingerprint's library
        # hash no longer implies the kernel bytes are the measured ones, so no
        # entry can be honoured.
        if os.environ.get(_CHECKSUM_DISABLED_ENV):
            raise RuntimeError(f"{_CHECKSUM_DISABLED_ENV} is set")
        raw = json.loads(path.read_text())
        recorded = raw["fingerprint"]
        live = _live_fingerprint(device)
        if all(recorded.get(k) == v for k, v in live.items()):
            for entry in raw["entries"]:
                table[tuple(entry["applicability_key"])] = list(entry["tactic"])
    except Exception:  # noqa: BLE001 - no table, bad table, no device, no checksums
        table = {}
    _TABLE_CACHE[index] = table
    return table


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
        # Runners are built lazily and keyed on every scalar they are
        # constructed from: __init__ must touch no CUDA state, since the harness
        # constructs both modules before any device work.
        self._runners: dict[tuple, object] = {}

    # -- fast-path admission ------------------------------------------------
    def _fast_ok(self, hidden_states, w13, w2, router_logits, routing_bias) -> bool:
        """Whether the tactic-aware runner can be trusted with these inputs.

        Runs to completion before anything is enqueued. The alternative -- launch
        and catch -- discovers a rejected argument only after routing kernels and
        thirteen scratch allocations are already on the stream.
        """
        if self.routing_method_type not in _SUPPORTED_ROUTING:
            return False
        e, k = self.num_experts, self.top_k
        h_size = self.intermediate_size_per_partition
        if not 1 <= k <= e:
            return False
        if not 0 <= self.local_expert_offset:
            return False
        if self.local_num_experts <= 0 or self.local_expert_offset + self.local_num_experts > e:
            return False
        if h_size <= 0 or h_size % _BLOCK_ELEMS_BF16:
            return False

        # Grouped routing needs a consistent group description; ungrouped routing
        # must not carry one, or the launcher reads group counts it never set.
        grouped = self.routing_method_type in _GROUPED_ROUTING
        n_group, topk_group = self.num_expert_group, self.topk_group
        if grouped:
            if n_group is None or topk_group is None:
                return False
            if n_group <= 0 or not 1 <= topk_group <= n_group or e % n_group:
                return False
        elif n_group is not None or topk_group is not None:
            return False

        if hidden_states.dim() != 2 or router_logits.dim() != 2:
            return False
        if hidden_states.dtype is not torch.bfloat16:
            return False
        if w13.dtype is not torch.bfloat16 or w2.dtype is not torch.bfloat16:
            return False
        if router_logits.dtype not in (torch.bfloat16, torch.float32):
            return False

        num_tokens, hidden = hidden_states.shape
        if num_tokens < 1 or hidden <= 0 or hidden % _BLOCK_ELEMS_BF16:
            return False
        if router_logits.shape != (num_tokens, e):
            return False
        # The 4D BlockMajorK shapes the launcher indexes, spelled out: a weight
        # that merely has four dimensions is not enough, and a mismatch here is
        # the difference between reading the right expert and reading noise.
        blocks = _BLOCK_ELEMS_BF16
        if tuple(w13.shape) != (e, hidden // blocks, 2 * h_size, blocks):
            return False
        if tuple(w2.shape) != (e, h_size // blocks, hidden, blocks):
            return False

        device = hidden_states.device
        if device.type != "cuda":
            return False
        tensors = [hidden_states, w13, w2, router_logits]
        if routing_bias is not None:
            # Kimi routes with a bfloat16[E] bias; the launcher accepts no other
            # dtype for it.
            if (routing_bias.dim() != 1 or routing_bias.shape[0] != e
                    or routing_bias.dtype is not torch.bfloat16):
                return False
            tensors.append(routing_bias)
        for t in tensors:
            if t.device != device or not t.is_contiguous():
                return False

        index = device.index if device.index is not None else torch.cuda.current_device()
        major = _SM_MAJOR.get(index)
        if major is None:
            major = torch.cuda.get_device_capability(device)[0]
            _SM_MAJOR[index] = major
        if major != 10:
            return False
        return _seam_intact() and _moe_module() is not None

    # -- tactic resolution -------------------------------------------------
    def _instance_key(self, hidden: int, num_tokens: int) -> tuple:
        """The key ``MoERunner.get_valid_tactics`` builds, field for field.

        ``local_num_experts``, not global ``num_experts``: the launcher's own
        validity query uses the local count, and a table keyed on the global one
        would silently mis-apply under any expert-parallel split.
        """
        return (
            _DTYPE_TRTLLM_GEN_BF16,           # dtype_act
            _DTYPE_TRTLLM_GEN_BF16,           # dtype_weights
            _FP8_QUANT_NONE,                  # fp8_quantization_type
            self.top_k,
            hidden,
            self.intermediate_size_per_partition,
            self.local_num_experts,
            ACTIVATION_SWIGLU,
            True,                             # use_shuffled_weight
            _WEIGHT_LAYOUT_BLOCK_MAJOR_K,
            False,                            # use_per_token_scaling
            num_tokens,
            False,                            # has_gemm1_lora_delta
        )

    def _tactic(self, hidden: int, num_tokens: int,
                device: torch.device) -> list[int]:
        """The committed tactic for this exact shape, proven valid before use.

        Membership is checked against the launcher's own validity answer for this
        token count *on this device* rather than discovered from an exception,
        because by the time the launcher raises it has already enqueued routing.
        Cached per (device index, key), so it costs one query per distinct token
        count per device and the benchmark's sixty-plus calls per shape amortize
        it immediately.
        """
        index = device.index if device.index is not None else torch.cuda.current_device()
        key = self._instance_key(hidden, num_tokens)
        cache_key = (index, key)
        cached = _TACTIC_CACHE.get(cache_key)
        if cached is not None:
            return cached
        tactic = _tactic_table(device).get(key, _DEFAULT_TACTIC)
        if tactic != _DEFAULT_TACTIC and tuple(tactic) not in _valid_tactics(device, key):
            tactic = _DEFAULT_TACTIC
        _TACTIC_CACHE[cache_key] = tactic
        return tactic

    def _runner(self, hidden: int):
        """The cached runner for this configuration, or ``None`` to fall back.

        Keyed on every scalar the runner is constructed from, not just the hidden
        size. The runner keeps ``top_k``, ``intermediate_size`` and
        ``num_local_experts`` inside itself and reads them in its own ``forward``,
        so a caller that rebinds ``self.top_k`` after the first call would
        otherwise get a plausible answer computed at the old ``top_k`` -- the gate
        and the instance key would both see the new value while the launcher saw
        the old one.

        Construction is guarded and happens before anything is enqueued, so a
        constructor whose keywords have changed upstream sends the call to the
        wrapper instead of escaping as an exception from the middle of a forward.
        """
        key = (hidden, self.top_k, self.local_num_experts, self.num_experts,
               self.intermediate_size_per_partition)
        runner = self._runners.get(key)
        if runner is None:
            try:
                runner = _moe_module().MoERunner(
                    top_k=self.top_k,
                    num_local_experts=self.local_num_experts,
                    dtype_act=_DTYPE_TRTLLM_GEN_BF16,
                    dtype_weights=_DTYPE_TRTLLM_GEN_BF16,
                    fp8_quantization_type=_FP8_QUANT_NONE,
                    hidden_size=hidden,
                    intermediate_size=self.intermediate_size_per_partition,
                    activation_type=ACTIVATION_SWIGLU,
                    use_shuffled_weight=True,
                    weight_layout=_WEIGHT_LAYOUT_BLOCK_MAJOR_K,
                    num_experts=self.num_experts,
                )
            except Exception:  # noqa: BLE001 - nothing enqueued yet
                return None
            self._runners[key] = runner
        return runner

    # -- the baseline call, verbatim ---------------------------------------
    def _fallback(self, hidden_states, w13, w2, router_logits, routing_bias):
        """``flashinfer.fused_moe.trtllm_bf16_moe`` as the baseline calls it.

        Argument for argument, so an input this module declines is accepted or
        rejected exactly as it would have been without this module -- including
        raising the same exception at the same point.
        """
        if _trtllm_bf16_moe is None:
            raise ImportError(
                "flashinfer.fused_moe.trtllm_bf16_moe is unavailable; the "
                "trtllm-gen BF16 MoE path needs FlashInfer on a Blackwell device")
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        w13: torch.Tensor,
        w2: torch.Tensor,
        router_logits: torch.Tensor,
        routing_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self._fast_ok(hidden_states, w13, w2, router_logits, routing_bias):
            return self._fallback(hidden_states, w13, w2, router_logits, routing_bias)

        num_tokens, hidden = hidden_states.shape
        device = hidden_states.device
        runner = self._runner(hidden)
        if runner is None:
            return self._fallback(hidden_states, w13, w2, router_logits, routing_bias)
        tactic = self._tactic(hidden, num_tokens, device)
        # A fresh output every call. Reusing one buffer would save a couple of
        # microseconds and alias the tensor the harness holds for comparison
        # across its three correctness rounds.
        out = torch.empty(num_tokens, hidden, dtype=torch.bfloat16, device=device)
        # ``MoeRunnerInputs`` field order, flattened. The empty ``expert_weights``
        # must carry ``router_logits``' dtype -- float32 for Kimi, bfloat16 for
        # Qwen -- because the launcher writes the routing weights back through
        # it; one shared placeholder would be wrong for one of the two configs.
        inputs = [
            out,                                                    # output
            router_logits,                                          # routing_logits
            torch.empty(0, dtype=torch.int32, device=device),        # topk_ids
            torch.empty(0, dtype=router_logits.dtype, device=device),  # expert_weights
            hidden_states,
            None,                                                   # hidden_states_scale
            None,                                                   # gemm1_lora_delta
            None,                                                   # per_token_scale
        ]
        try:
            runner.forward(
                inputs,
                tactic=tactic,
                routing_bias=routing_bias,
                gemm1_weights=w13,
                gemm2_weights=w2,
                num_experts=self.num_experts,
                n_group=self.num_expert_group,
                topk_group=self.topk_group,
                local_expert_offset=self.local_expert_offset,
                local_num_experts=self.local_num_experts,
                routed_scaling_factor=self.routed_scaling_factor,
                routing_method_type=self.routing_method_type,
                use_shuffled_weight=True,
                weight_layout=_WEIGHT_LAYOUT_BLOCK_MAJOR_K,
                do_finalize=True,
                # The baseline reaches the op through the public wrapper, whose
                # default is True rather than None, so the launcher never
                # consults device_support_pdl on that path. True is what the
                # baseline actually runs.
                enable_pdl=True,
                norm_topk_prob=True,
            )
        except Exception:  # noqa: BLE001 - last resort, never the first line
            # Prevalidation above is the mechanism; this catches a synchronous
            # host-side rejection no query anticipated. Degrading to the wrapper
            # re-runs the whole call, which is safe because the launcher
            # allocates its scratch per call and no input has been mutated -- but
            # it never retries the private path, which would re-enqueue routing.
            index = (device.index if device.index is not None
                     else torch.cuda.current_device())
            _TACTIC_CACHE[(index, self._instance_key(hidden, num_tokens))] = \
                _DEFAULT_TACTIC
            return self._fallback(hidden_states, w13, w2, router_logits, routing_bias)
        # ``do_finalize=True`` with no LoRA delta means the launcher wrote the
        # result into the buffer above, which is why nothing is unpacked here.
        return out
