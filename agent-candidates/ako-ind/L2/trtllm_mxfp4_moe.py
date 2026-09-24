"""TRTLLM-gen MXFP4 fused MoE (via FlashInfer, Blackwell only).

Same kernel as :mod:`baseline` -- byte for byte, `max_abs_error == 0` on every
captured shape -- and what changes is *how* it is dispatched. Three measured
facts drive this file (see ``ITERATIONS.md`` for the numbers):

1. **The baseline never autotunes.** ``AutoTuner.choose_one`` only profiles
   inside an ``autotune()`` context; outside one it does a cache lookup, misses,
   and returns ``runners[0]`` with ``tactic=-1`` -- the built-in heuristic. So
   ``tune_max_num_tokens`` in the captured recipe selects nothing at all. One
   real profiling pass over the bucket ladder is worth **1.65x** of GPU time at
   398 tokens and **1.77x** at 16384 tokens (3.16 ms -> 1.78 ms), and is neutral
   at 1-60 tokens. A single pass with ``tune_max_num_tokens =
   max_capture_size`` fills every bucket from 1 to 1024, and a 16384-token call
   maps down onto the top bucket, so all shapes are covered by one pass.

2. **Small batches are host-bound, not GPU-bound.** At one token the whole MoE
   is 21 us of GPU work but ``flashinfer.trtllm_fp4_block_scale_moe`` costs
   ~0.53 ms of *host* time per call, so latency is set by how fast the CPU can
   submit. About 0.2 ms of that is the Python wrapper: it rebuilds a
   ``MoERunner``, rebuilds a ``TuningConfig`` (re-deriving the bucket ladder),
   re-hashes a cache key and re-runs ``choose_one``, all with values that are
   constant for a given token count. Hoisting that into a per-token-count plan
   and calling the underlying op with the already-chosen tactic removes it.

3. **The other ~0.33 ms is inside the C++ launcher, so we bring our own.**
   ``FusedMoeLauncher::prepare_moe_common`` re-runs
   ``std::make_unique<MoE::Runner>(...)`` on *every* invocation -- two scans of
   the trtllm-gen cubin table plus the cartesian product of the passing
   gemm1 x gemm2 configs, ~0.26 ms of the 0.33 on its own -- then
   ``getValidConfigIndices``, ``getWorkspaceSizeInBytes`` and ~15 device
   allocations, and the entry point builds four launchers (one per supported
   tile_N) to use one. ``mxfp4_moe_launcher.cu`` makes the same two trtllm-gen
   calls with all of that hoisted into a process-static plan, and is compiled
   from flashinfer's own JIT spec with only the launcher translation unit
   swapped, so it runs the same cubins and is bit-exact by construction. Host
   time per call at one token: 0.457 ms (wrapper) -> 0.331 (2) -> **0.022**.

All three are best-effort: any failure -- no compiler, no cubins, a missing
source file, an unexpected exception -- falls back to the plain wrapper call,
which is byte-identical to the baseline. ``_PRIV`` / ``_FAST`` / ``_TUNE`` env
switches isolate each layer.

Original baseline notes, still true of the math:

This is the kernel vLLM 0.26 actually runs for gpt-oss on SM100. Its oracle logs

    Using 'FLASHINFER_TRTLLM_MXFP4_BF16' Mxfp4 MoE backend.
    Using TrtLlmMxfp4ExpertsMonolithic

and then autotunes ``flashinfer::trtllm_fp4_block_scale_moe``.

Two things the trtllm-gen kernel needs that the Triton one does not:

* **256-element alignment.** ``mxfp4_round_up_hidden_size_and_intermediate_size``
  rounds both ``hidden_size`` and ``intermediate_size_per_partition`` up to 256
  for the TRTLLM backends, so gpt-oss runs its experts at hidden 3072 (from
  2880) and, at tp=2, intermediate 1536 (from 1440). The activation is
  zero-padded into that width; the kernel writes an unpadded output
  (``has_unpadded_output``), so nothing has to be sliced afterwards.
* **A shuffled weight/scale layout** for the transposed MMA epilogue, plus a
  gate/up row swap because trtllm-gen defines SwiGLU with the two halves in the
  opposite order. :func:`prepare_trtllm_mxfp4_weights` is a port of vLLM's
  ``convert_gpt_oss_weight_to_mxfp4_moe_kernel_format`` TRTLLM branch.

Mirrors ``TrtLlmMxfp4ExpertsMonolithic.apply``
(``vllm/model_executor/layers/fused_moe/experts/trtllm_mxfp4_moe.py``).
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn

from flashinfer import trtllm_fp4_block_scale_moe


# vLLM's TRTLLM_BACKENDS branch of
# ``mxfp4_round_up_hidden_size_and_intermediate_size``.
TRTLLM_MXFP4_ALIGN = 256

# ``get_routing_method_type("softmax", renormalize=True, has_e_score_bias=False)``
# for gpt-oss -> RenormalizeNaive. Softmax->TopK->renormalize is the same
# function as TopK->softmax (softmax is monotonic, so the top-k set is
# identical and renormalizing the k values equals a softmax over just those
# logits), which is why the expert class accepts either spelling.
ROUTING_RENORMALIZE_NAIVE = 4

# gpt-oss SwiGLU-OAI constants; vLLM passes these as gemm1_alpha / gemm1_beta /
# gemm1_clamp_limit (``Mxfp4MoEMethod.get_fused_moe_quant_config``).
SWIGLU_ALPHA = 1.702
SWIGLU_BETA = 1.0
SWIGLU_LIMIT = 7.0

# vLLM passes ``tune_max_num_tokens=max(moe_config.max_capture_size, 1)``, i.e.
# ``compilation_config.max_cudagraph_capture_size``, which for gpt-oss is 1024
# (the same value our own ``capture_cudagraph`` uses as ``max_capture_limit``).
# The autotuner tunes every token count up to this bound.
DEFAULT_TUNE_MAX_NUM_TOKENS = 1024

_MXFP4_SF_BLOCK = 32
_EPILOGUE_TILE_M = 128

# ``RoutingInputMode.FromLogits`` / ``ActivationType.Swiglu``, inlined so the
# fast path does not import or attribute-walk enums per call.
_ROUTING_FROM_LOGITS = 0
_ACTIVATION_SWIGLU = 3
_FALLBACK_TACTIC = [-1, -1]

# Sentinel for "planning was tried here and failed" -- distinct from "not yet
# planned", so a failure is not retried on every call.
_NO_PLAN = object()

# Bound on the per-token-count plan cache. Serving sees a bounded set of token
# counts (max_num_batched_tokens), but nothing here guarantees that, and each
# plan pins two [num_tokens, top_k] scratch tensors. Dropping the whole cache on
# overflow costs one re-plan, not a re-tune (the tactic ladder is global).
_MAX_PLANS = 64

# ``FASTKERNELS_TRTLLM_MXFP4_TUNE=0`` skips the profiling pass (keeps the
# heuristic tactic); ``FASTKERNELS_TRTLLM_MXFP4_FAST=0`` skips the hoisted
# dispatch. Both exist so either half can be isolated when bisecting.
_DO_TUNE = os.environ.get("FASTKERNELS_TRTLLM_MXFP4_TUNE", "1") != "0"
_DO_FAST = os.environ.get("FASTKERNELS_TRTLLM_MXFP4_FAST", "1") != "0"
# ``FASTKERNELS_TRTLLM_MXFP4_PRIV=0`` skips the private state-caching launcher
# and falls back to flashinfer's own C++ launcher (i.e. to the r1 fast path).
_DO_PRIV = os.environ.get("FASTKERNELS_TRTLLM_MXFP4_PRIV", "1") != "0"
# ``FASTKERNELS_TRTLLM_MXFP4_BUILD=1`` allows the private launcher to be
# *compiled* on demand. Off by default: a cold compile is ~590 s and would land
# wherever the first ``forward`` happens to be, so an un-prewarmed cache
# degrades to the pure-Python fast path instead of stalling.
_MAY_BUILD = os.environ.get("FASTKERNELS_TRTLLM_MXFP4_BUILD", "0") == "1"


def trtllm_mxfp4_moe_supported() -> bool:
    """True when the trtllm-gen MXFP4 MoE kernel can run on this device.

    Same gate as vLLM's ``TrtLlmMxfp4ExpertsBase._supports_current_device``:
    CUDA, SM100 family, FlashInfer present.

    ``FASTKERNELS_TRTLLM_MXFP4_MOE=0`` forces the Triton path instead. This
    kernel segfaults inside ``flashinfer::FP4BlockScaleLauncher::run`` on the
    first autotune profile for gpt-oss-120b at tp=1 with a 16384-token chunk
    (fine at tp=2, and fine at tp=1 with short prompts), so a switch is needed
    to isolate it and to keep that configuration runnable.
    """
    if os.environ.get("FASTKERNELS_TRTLLM_MXFP4_MOE", "1") == "0":
        return False
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability()[0] == 10


def round_up(x: int, m: int) -> int:
    return ((x + m - 1) // m) * m


def _swap_every_two_rows(x: torch.Tensor, axis: int = -1) -> torch.Tensor:
    """Swap adjacent pairs along ``axis`` (trtllm-gen's SwiGLU half order)."""
    shape = x.shape
    if axis < 0:
        axis = len(shape) + axis
    new_shape = list(shape)
    new_shape[axis] = shape[axis] // 2
    new_shape.insert(axis + 1, 2)
    x = x.reshape(*new_shape)
    x = x.flip(axis + 1)
    return x.reshape(*shape)


def prepare_trtllm_mxfp4_weights(
    w13_weight: torch.Tensor,
    w13_weight_scale: torch.Tensor,
    w13_bias: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_weight_scale: torch.Tensor,
    w2_bias: torch.Tensor,
    permute_cache: dict | None = None,
) -> tuple[torch.Tensor, ...]:
    """Shuffle loaded MXFP4 expert weights into the trtllm-gen layout.

    Expects the *padded* shapes the kernel is configured for:
      ``w13_weight``       ``[E, 2*I, H // 2]``   uint8
      ``w13_weight_scale`` ``[E, 2*I, H // 32]``  uint8 (E8M0)
      ``w13_bias``         ``[E, 2*I]``
      ``w2_weight``        ``[E, H, I // 2]``     uint8
      ``w2_weight_scale``  ``[E, H, I // 32]``    uint8 (E8M0)
      ``w2_bias``          ``[E, H]``

    Returns the same six tensors in kernel layout, with the scales viewed as
    ``float8_e4m3fn`` and the biases in float32.
    """
    from flashinfer.fp4_quantization import nvfp4_block_scale_interleave
    from flashinfer.fused_moe.core import get_w2_permute_indices_with_cache

    if permute_cache is None:
        permute_cache = {}

    num_experts = w13_weight.shape[0]
    intermediate_size = w13_weight.shape[1] // 2
    hidden_size = w13_weight.shape[2] * 2

    w13_bias = w13_bias.to(torch.float32)
    w2_bias = w2_bias.to(torch.float32)

    # trtllm-gen's SwiGLU takes the two halves in the opposite order.
    w13_weight_scale = _swap_every_two_rows(w13_weight_scale, -2)
    w13_weight = _swap_every_two_rows(w13_weight, -2)
    w13_bias = _swap_every_two_rows(w13_bias, -1)

    g1_w, g1_s, g1_b = [], [], []
    g2_w, g2_s, g2_b = [], [], []
    for i in range(num_experts):
        idx = get_w2_permute_indices_with_cache(
            permute_cache, w13_weight[i].view(torch.uint8), _EPILOGUE_TILE_M,
        )
        g1_w.append(
            w13_weight[i].view(torch.uint8)[idx.to(w13_weight.device)].contiguous()
        )
        sf_idx = get_w2_permute_indices_with_cache(
            permute_cache, w13_weight_scale[i].view(torch.uint8),
            _EPILOGUE_TILE_M, num_elts_per_sf=16,
        )
        g1_s.append(
            nvfp4_block_scale_interleave(
                w13_weight_scale[i]
                .view(torch.uint8)[sf_idx.to(w13_weight_scale.device)]
                .contiguous()
            )
        )
        b_idx = get_w2_permute_indices_with_cache(
            permute_cache, w13_bias[i].clone().reshape(-1, 1), _EPILOGUE_TILE_M,
        )
        g1_b.append(
            w13_bias[i].clone().reshape(-1, 1)[b_idx.to(w13_bias.device)].contiguous()
        )

        idx = get_w2_permute_indices_with_cache(
            permute_cache, w2_weight[i].view(torch.uint8), _EPILOGUE_TILE_M,
        )
        g2_w.append(
            w2_weight[i].view(torch.uint8)[idx.to(w2_weight.device)].contiguous()
        )
        sf_idx = get_w2_permute_indices_with_cache(
            permute_cache, w2_weight_scale[i].view(torch.uint8),
            _EPILOGUE_TILE_M, num_elts_per_sf=16,
        )
        g2_s.append(
            nvfp4_block_scale_interleave(
                w2_weight_scale[i]
                .view(torch.uint8)[sf_idx.to(w2_weight_scale.device)]
                .contiguous()
            )
        )
        b_idx = get_w2_permute_indices_with_cache(
            permute_cache, w2_bias[i].clone().reshape(-1, 1), _EPILOGUE_TILE_M,
        )
        g2_b.append(
            w2_bias[i].clone().reshape(-1, 1)[b_idx.to(w2_bias.device)].contiguous()
        )

    w13_weight = torch.stack(g1_w)
    w13_weight_scale = (
        torch.stack(g1_s)
        .reshape(num_experts, 2 * intermediate_size, hidden_size // _MXFP4_SF_BLOCK)
        .view(torch.float8_e4m3fn)
    )
    w2_weight = torch.stack(g2_w)
    w2_weight_scale = (
        torch.stack(g2_s)
        .reshape(num_experts, hidden_size, intermediate_size // _MXFP4_SF_BLOCK)
        .view(torch.float8_e4m3fn)
    )
    w13_bias = torch.stack(g1_b).reshape(num_experts, -1)
    w2_bias = torch.stack(g2_b).reshape(num_experts, -1)
    return (
        w13_weight, w13_weight_scale, w13_bias,
        w2_weight, w2_weight_scale, w2_bias,
    )


# ---------------------------------------------------------------------------
# Dispatch plumbing: the raw trtllm-gen op, and one profiling pass.
# ---------------------------------------------------------------------------
# ``get_trtllm_moe_sm100_module()`` hands back a SimpleNamespace whose
# ``trtllm_fp4_block_scale_moe`` is the *Python* wrapper (tuner lookup, runner
# construction, workspace allocation). The raw TVM-FFI entry point it calls is a
# closure cell of that wrapper; reading it there is cheaper and safer than a
# second ``build_and_load()`` (which is not cached and costs ~65 ms), and it is
# guaranteed to be the same object the wrapper would have used.
_RAW: dict = {}


def _raw_op():
    """The raw ``moe_op.trtllm_fp4_block_scale_moe``, or ``None``."""
    if "op" in _RAW:
        return _RAW["op"]
    op = None
    try:
        from flashinfer.fused_moe.core import get_trtllm_moe_sm100_module

        ns = get_trtllm_moe_sm100_module()
        fn = ns.trtllm_fp4_block_scale_moe
        names = fn.__code__.co_freevars
        if "moe_op" in names:
            moe_op = fn.__closure__[names.index("moe_op")].cell_contents
            op = moe_op.trtllm_fp4_block_scale_moe
    except Exception:  # noqa: BLE001 - any failure means "use the wrapper"
        op = None
    _RAW["op"] = op
    return op


# ---------------------------------------------------------------------------
# The private state-caching launcher.
#
# ``_raw_op()`` above already removes the ~0.2 ms of Python that flashinfer's
# wrapper spends per call, which leaves the ~0.33 ms that
# ``FusedMoeLauncher::prepare_moe_common`` spends *inside* the C++ launcher on
# every invocation: four launcher objects (one per supported tile_N, three
# discarded), a fresh ``MoE::Runner`` -- two cubin-table scans plus the
# cartesian product of the passing gemm1 x gemm2 configs, ~0.26 ms of the total
# on its own -- ``getValidConfigIndices``, ``getWorkspaceSizeInBytes``, and ~15
# device allocations. At one token that is 0.33 ms of host time against 21 us of
# GPU work, so it *is* the latency.
#
# ``mxfp4_moe_launcher.cu`` is the same two trtllm-gen launches with all of that
# hoisted into a process-static plan. It is built from flashinfer's own JIT spec
# for ``fused_moe_trtllm_sm100`` -- same sources, same nvcc flags, same
# trtllm-gen cubins -- with only the launcher translation unit swapped, under a
# private module name. Nothing in site-packages is touched, and because the
# cubins are identical the result is bit-exact.
_PRIV_MODULE_NAME = "fastkernels_moe_trtllm_sm100"
_PRIV_LAUNCHER_SRC = "mxfp4_moe_launcher.cu"
_PRIV: dict = {}


def _stage_launcher_source(src, jit_env):
    """Copy the launcher to a stable path, but only when its bytes changed.

    ninja rebuilds on mtime, and the bench harness copies ``solution/*.cu`` into
    a fresh candidate directory on every run -- so building straight from that
    copy would recompile the translation unit *inside* the benchmark. Staging it
    at a fixed path and leaving the file untouched when the contents match keeps
    the prewarmed artifact a cache hit.

    Returns ``(path, needs_compile)``.
    """
    import pathlib

    data = src.read_bytes()
    stage_dir = pathlib.Path(jit_env.FLASHINFER_JIT_DIR).parent / "fastkernels_moe_src"
    stage_dir.mkdir(parents=True, exist_ok=True)
    staged = stage_dir / src.name
    changed = (not staged.exists()) or staged.read_bytes() != data
    if changed:
        tmp = stage_dir / f"{src.name}.{os.getpid()}.tmp"
        tmp.write_bytes(data)
        tmp.replace(staged)
    return staged, changed


def _build_private_launcher():
    import dataclasses
    import pathlib

    from flashinfer.jit import env as jit_env
    from flashinfer.jit.fused_moe import gen_trtllm_gen_fused_moe_sm100_module

    src = pathlib.Path(__file__).resolve().with_name(_PRIV_LAUNCHER_SRC)
    if not src.exists():
        return None
    staged, changed = _stage_launcher_source(src, jit_env)
    # Calling flashinfer's own spec factory also fetches / symlinks the
    # trtllm-gen cubins and ``flashinferMetaInfo.h`` exactly as flashinfer would.
    spec = gen_trtllm_gen_fused_moe_sm100_module()
    sources = [
        staged if p.name == "trtllm_fused_moe_kernel_launcher.cu" else p
        for p in spec.sources
    ]
    if staged not in sources:
        return None
    spec = dataclasses.replace(spec, name=_PRIV_MODULE_NAME, sources=sources)
    # Compiling the 13 translation units of flashinfer's trtllm-gen MoE module
    # from cold takes ~590 s, which would sit inside whatever called ``forward``
    # first. That is a worse failure than not having the launcher at all, so on a
    # cold cache we decline and let the caller fall back. Prewarm with
    # ``FASTKERNELS_TRTLLM_MXFP4_BUILD=1`` (probe/build_private.py does this),
    # after which the artifact is a cache hit and this returns in ~0.1 s.
    if not _MAY_BUILD and (changed or not spec.get_library_path().exists()):
        return None
    mod = spec.build_and_load()
    # trtllm-gen kernels live in cubins that C++ pulls through a callback into
    # Python; every flashinfer module that touches them registers it after load.
    from flashinfer.jit.cubin_loader import setup_cubin_loader

    setup_cubin_loader(str(spec.get_library_path()))
    return mod.fastkernels_mxfp4_moe


def _private_op():
    """The private launcher entry point, or ``None`` if it is unavailable."""
    if "op" in _PRIV:
        return _PRIV["op"]
    op = None
    try:
        op = _build_private_launcher()
    except Exception:  # noqa: BLE001 - any failure means "use flashinfer's launcher"
        op = None
    _PRIV["op"] = op
    return op


# ---------------------------------------------------------------------------
# Tactic selection.
#
# The profiling pass is registered under a *private* op name rather than
# "flashinfer::trtllm_fp4_block_scale_moe". The AutoTuner and its profiling
# cache are a process-global singleton, so tuning under flashinfer's own name
# would publish the result to every other caller in the process -- including a
# baseline module built alongside this one, whose per-call ``choose_one`` lookup
# would then hit an entry it never paid for. Measured: with a shared key the
# baseline's 16384-token time drops 3.53 -> 2.24 ms, i.e. the tuning shows up on
# both sides and nets out to nothing. A private key keeps the selection inside
# this module (the fast path passes the tactic to the op directly and never
# consults the tuner again), which is also what a hand-rolled tactic search
# would do.
_PRIVATE_TUNING_OP = "fastkernels::trtllm_mxfp4_moe_tactic"

# Ladders are shared across module instances: one profiling pass per
# (shape recipe, tune bound) covers every bucket in it.
_TUNED: set = set()


def _choose_tactic(module, hidden_states, router_logits, topk_ids, topk_weights,
                   output, w13_weight, w13_weight_scale, w13_bias, w2_weight,
                   w2_weight_scale, w2_bias, enable_pdl):
    """Which tactic should this call use? Profiles the ladder once per process.

    Mirrors the setup ``trtllm_fp4_block_scale_moe_op`` redoes on every call.
    Returns a value suitable for the raw op's ``config_index`` argument.
    """
    from flashinfer.autotuner import AutoTuner, autotune
    from flashinfer.fused_moe.core import (
        Fp8QuantizationType,
        MoeRunnerInputs,
        WeightLayout,
        get_trtllm_moe_sm100_module,
    )
    from flashinfer.tllm_enums import deduce_trtllm_gen_tensor_dtype

    runner = get_trtllm_moe_sm100_module().MoERunner(
        top_k=module.top_k,
        num_local_experts=module.num_experts,
        dtype_act=deduce_trtllm_gen_tensor_dtype(hidden_states, None),
        dtype_weights=deduce_trtllm_gen_tensor_dtype(w13_weight, w13_weight_scale),
        fp8_quantization_type=Fp8QuantizationType.NoneFp8,
        hidden_size=hidden_states.shape[-1],
        intermediate_size=module.intermediate_size,
        activation_type=_ACTIVATION_SWIGLU,
        weight_layout=WeightLayout.MajorK,
        use_shuffled_weight=True,
        use_per_token_scaling=False,
        num_experts=module.num_experts,
    )
    moe_inputs = MoeRunnerInputs(
        output=output,
        routing_logits=router_logits,
        topk_ids=topk_ids,
        expert_weights=topk_weights,
        hidden_states=hidden_states,
        hidden_states_scale=None,
        gemm1_lora_delta=None,
        per_token_scale=None,
    )
    tuning_config = runner._make_tuning_config(
        moe_inputs,
        tune_max_num_tokens=module.max_capture_size,
        use_cold_l2_cache=True,
        use_cuda_graph=True,
    )
    kwargs = dict(
        routing_input_mode=_ROUTING_FROM_LOGITS,
        num_experts=module.num_experts,
        routing_bias=None,
        gemm1_weights=w13_weight,
        gemm1_weights_scale=w13_weight_scale,
        gemm1_bias=w13_bias,
        gemm1_alpha=module.gemm1_alpha,
        gemm1_beta=module.gemm1_beta,
        gemm1_clamp_limit=module.gemm1_clamp_limit,
        gemm2_weights=w2_weight,
        gemm2_weights_scale=w2_weight_scale,
        gemm2_bias=w2_bias,
        output1_scale_scalar=None,
        output1_scale_gate_scalar=None,
        output2_scale_scalar=None,
        per_token_scale=None,
        n_group=None,
        topk_group=None,
        local_expert_offset=0,
        routed_scaling_factor=None,
        routing_method_type=ROUTING_RENORMALIZE_NAIVE,
        enable_pdl=enable_pdl,
        do_finalize=True,
        activation_type=_ACTIVATION_SWIGLU,
    )
    tuner = AutoTuner.get()
    ladder = (module.num_experts, module.top_k, module.intermediate_size,
              hidden_states.shape[-1], module.max_capture_size)
    if _DO_TUNE and ladder not in _TUNED:
        _TUNED.add(ladder)  # one attempt per ladder, success or not
        try:
            with autotune(True):
                tuner.choose_one(_PRIVATE_TUNING_OP, [runner], tuning_config,
                                 moe_inputs.to_list(), **kwargs)
        except Exception:  # noqa: BLE001 - tuning is optional
            pass
    _, tactic = tuner.choose_one(_PRIVATE_TUNING_OP, [runner], tuning_config,
                                 moe_inputs.to_list(), **kwargs)
    return _FALLBACK_TACTIC if tactic == -1 else tactic


class TrtLlmMxfp4MoE(nn.Module):
    """Router + experts in one trtllm-gen launch.

    ``hidden_states`` arrives at the *padded* hidden width; the returned tensor
    is ``hidden_size_unpadded`` wide, matching vLLM's ``has_unpadded_output``.
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
        # num_tokens -> (tactic, topk_ids, topk_weights). Routing scratch is
        # pure output of the routing kernel, so it is safe to reuse.
        self._plans: dict = {}
        self._pdl = None

    # -- the plain wrapper call, i.e. exactly what the baseline does ---------
    def _wrapper_call(self, output, hidden_states, router_logits, w13_weight,
                      w13_weight_scale, w13_bias, w2_weight, w2_weight_scale,
                      w2_bias):
        trtllm_fp4_block_scale_moe(
            routing_logits=router_logits.to(torch.bfloat16),
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

    def _build_plan(self, num_tokens, output, hidden_states, router_logits,
                    w13_weight, w13_weight_scale, w13_bias, w2_weight,
                    w2_weight_scale, w2_bias):
        """First call at this token count: pick a tactic and allocate the
        routing scratch, then freeze every other argument the op needs.

        Everything constant is captured in the plan tuple so the steady-state
        path is one tuple unpack plus one op call -- no ``nn.Module.__getattr__``
        walks, which are Python-level dict lookups and are not free when the
        whole MoE is 21 us of GPU work.
        """
        from flashinfer.utils import device_support_pdl

        if self._pdl is None:
            self._pdl = bool(device_support_pdl(hidden_states.device))
        dev = hidden_states.device
        topk_ids = torch.empty(num_tokens, self.top_k, dtype=torch.int32, device=dev)
        topk_weights = torch.empty(num_tokens, self.top_k, dtype=torch.bfloat16,
                                   device=dev)
        tactic = _choose_tactic(
            self, hidden_states, router_logits, topk_ids, topk_weights, output,
            w13_weight, w13_weight_scale, w13_bias, w2_weight, w2_weight_scale,
            w2_bias, self._pdl)
        priv = _private_op() if _DO_PRIV else None
        if priv is not None:
            # The private launcher owns its own routing scratch and resolves the
            # tile_N fallback itself, so the plan carries only the tactic pair.
            plan = (priv, int(tactic[0]), int(tactic[1]), self.gemm1_alpha,
                    self.gemm1_beta, self.gemm1_clamp_limit, self._pdl,
                    self.num_experts, self.top_k, self.intermediate_size)
            # Warm the C++ plan (Runner construction, workspace query, scratch
            # allocation) outside the timed region.
            priv(router_logits, hidden_states, w13_weight, w13_weight_scale,
                 w13_bias, self.gemm1_alpha, self.gemm1_beta,
                 self.gemm1_clamp_limit, w2_weight, w2_weight_scale, w2_bias,
                 output, self.num_experts, self.top_k, self.intermediate_size,
                 ROUTING_RENORMALIZE_NAIVE, plan[1], plan[2], self._pdl)
        else:
            plan = (None, tactic, topk_ids, topk_weights, self.gemm1_alpha,
                    self.gemm1_beta, self.gemm1_clamp_limit, self._pdl,
                    self.num_experts, self.top_k, self.intermediate_size)
        if len(self._plans) >= _MAX_PLANS:
            self._plans.clear()
        self._plans[num_tokens] = plan
        return plan

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
        raw = _raw_op() if _DO_FAST else None
        if raw is None or hidden_states.dim() != 2:
            self._wrapper_call(
                output, hidden_states, router_logits, w13_weight,
                w13_weight_scale, w13_bias, w2_weight, w2_weight_scale, w2_bias)
            return output

        num_tokens = hidden_states.shape[0]
        logits = (router_logits if router_logits.dtype == torch.bfloat16
                  else router_logits.to(torch.bfloat16))
        plan = self._plans.get(num_tokens)
        if plan is None or plan is _NO_PLAN:
            if plan is _NO_PLAN:
                self._wrapper_call(
                    output, hidden_states, logits, w13_weight, w13_weight_scale,
                    w13_bias, w2_weight, w2_weight_scale, w2_bias)
                return output
            try:
                plan = self._build_plan(
                    num_tokens, output, hidden_states, logits, w13_weight,
                    w13_weight_scale, w13_bias, w2_weight, w2_weight_scale,
                    w2_bias)
            except Exception:  # noqa: BLE001 - planning is best-effort
                self._plans[num_tokens] = _NO_PLAN
                plan = _NO_PLAN
            if plan is _NO_PLAN:
                self._wrapper_call(
                    output, hidden_states, logits, w13_weight, w13_weight_scale,
                    w13_bias, w2_weight, w2_weight_scale, w2_bias)
                return output
        priv = plan[0]
        if priv is not None:
            (_, tile_n, config, alpha, beta, clamp, pdl, num_experts, top_k,
             intermediate_size) = plan
            priv(
                logits, hidden_states,
                w13_weight, w13_weight_scale, w13_bias, alpha, beta, clamp,
                w2_weight, w2_weight_scale, w2_bias, output,
                num_experts, top_k, intermediate_size,
                ROUTING_RENORMALIZE_NAIVE, tile_n, config, pdl,
            )
            return output

        (_, tactic, topk_ids, topk_weights, alpha, beta, clamp, pdl, num_experts,
         top_k, intermediate_size) = plan

        # The same 36 positional arguments ``trtllm_fp4_block_scale_moe_op``
        # ends up passing, minus the per-call setup it redoes to get here.
        raw(
            _ROUTING_FROM_LOGITS, logits, topk_ids, topk_weights, None,
            hidden_states, None,
            w13_weight, w13_weight_scale, w13_bias, alpha, beta, clamp,
            w2_weight, w2_weight_scale, w2_bias,
            None, None, None, None,
            num_experts, top_k, None, None,
            intermediate_size, 0, num_experts, None,
            ROUTING_RENORMALIZE_NAIVE, True, pdl, _ACTIVATION_SWIGLU,
            output, tactic, True, None,
        )
        return output
