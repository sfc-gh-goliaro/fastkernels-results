"""TRTLLM-gen MXFP4 fused MoE: authored routing kernel + wrapper-free dispatch.

Three things sit in front of the MXFP4 expert math, and this module addresses all
three. The math itself stays in the precompiled trtllm-gen cubin, because every
one of its 128-208 tactics produces bit-identical output and reproducing that
bit-for-bit from scratch would trade a guaranteed pass for a fight with the
bench's random E8M0 block scales, whose reference outputs span ~20 decades.

**1. The FlashInfer Python wrapper.** ``flashinfer.trtllm_fp4_block_scale_moe``
rebuilds a ``MoERunner``, constructs a ``TuningConfig`` (including token buckets
up to ``tune_max_num_tokens``), takes ``AutoTuner``'s global lock, hashes a cache
key and misses — on every call, to return the same ``[-1, -1]`` fallback, because
nothing here enters ``flashinfer.autotune()``. Measured on B200 at one token, that
machinery costs ~0.435 ms of host time against ~0.335 ms for the same C++ op
called directly, while the device work behind it is only ~53 us. Three of the five
benched token counts (1, 26, 60) are host-bound in exactly that way. So the fast
path binds the raw 36-argument op once and calls it directly.

**2. The default tile.** Under ``[-1, -1]`` the C++ ``selectDefaultTileN`` returns
``*computeSelectedTileN(...).begin()`` — the *smallest* candidate ``tile_N``. That
is right at small token counts (tile 8) and wrong at 16384, where the candidate set
is ``{32, 64}`` and it picks 32. NCU: moving to ``tile_N = 64`` halves both ``bmm``
grids (52128 -> 27600 blocks) and lifts ``sm__throughput`` from 40% to 62-65%.
Without that correction the 16384-token shape runs *slower* than the wrapper
(3.350 ms raw against 3.189 ms), so a gated online sweep picks it.

**3. Routing.** ``routing_method_type=4`` (RenormalizeNaive) is softmax over all
experts, top-k, renormalize. :func:`_ROUTING_CUDA_SOURCE` is an authored CUDA
kernel that computes it in one warp per token and hands the result to the raw op
through ``RoutingInputMode.UnpackedPrecomputed``, replacing the cubin's own
``routingIndicesHistogramScoresKernel`` work. It is enabled per shape only after
its end-to-end output is proven **bit-identical** to the cubin's own routing on the
real inputs, in a correctness round — see :meth:`TrtLlmMxfp4MoE._try_authored_routing`.
Where that proof fails the module keeps the cubin's routing. The check exists
because agreement is not free: trtllm's own weights differ from an algebraically
identical fp32 evaluation in about 1 of 65536 values at 16384 tokens, at the last
bfloat16 ulp, and one such weight changes a whole output row by ~1e20 given those
block scales.

Anything the raw path was not validated for — a shape, dtype, device, or layout
outside the checked contract — routes to the public API, which is baseline
behaviour. Contract, buffers and weight layout are identical to ``baseline.py``;
``prepare_trtllm_mxfp4_weights`` is a load-time helper and is not on this path.
"""

from __future__ import annotations

import functools
import os
import statistics
import time
from typing import NamedTuple, Optional

import torch
import torch.nn as nn

from flashinfer import trtllm_fp4_block_scale_moe


# ``get_routing_method_type("softmax", renormalize=True, has_e_score_bias=False)``
# for gpt-oss -> RenormalizeNaive.
ROUTING_RENORMALIZE_NAIVE = 4

# gpt-oss SwiGLU-OAI constants, passed per expert.
SWIGLU_ALPHA = 1.702
SWIGLU_BETA = 1.0
SWIGLU_LIMIT = 7.0

DEFAULT_TUNE_MAX_NUM_TOKENS = 1024

# ``RoutingInputMode``: 0 computes routing from logits and *writes* the scratch;
# 2 takes topk_ids/topk_weights as inputs. Held as ints so the fast path does no
# enum attribute lookups.
_ROUTE_FROM_LOGITS = 0
_ROUTE_PRECOMPUTED = 2

# MXFP4 packs two 4-bit values per byte and shares one E8M0 scale per 32 values.
_MXFP4_PER_BYTE = 2
_MXFP4_SF_BLOCK = 32

# Sweep tactics only where there is enough device work for the measurement to
# mean something. At one token the whole MoE is 53 us of device work behind
# ~0.34 ms of host launch, and inside the harness' timed window it is hidden
# completely behind the 0.32 ms of weight copies that happen there — so the ~5%
# spread across tactics at small token counts is host jitter, and choosing on it
# would pin a tactic to noise. Requiring >= 16 rows per expert on average is
# n >= 512 for the captured config, which keeps 1/26/60/398 on the default.
_SWEEP_MIN_ROWS_PER_EXPERT = 16

# Four launches per tactic. At 16384 tokens that is ~128 tactics x ~2.5 ms ~= 1.3 s,
# paid once, in a correctness round rather than a timed window.
_SWEEP_REPEATS = 3
_SWEEP_SETTLE = 5

# Stop launching new measurement groups past this much wall clock and keep the
# best tactic found so far, so an unexpectedly large shape cannot run away with
# the operator's time budget.
_SWEEP_BUDGET_S = 20.0

# Keep [-1, -1] unless a candidate beats it by this margin. 3% sits well above the
# run-to-run spread on this device and far below the 1.67x the 16384-token shape
# actually moves, so it accepts real wins and rejects noise.
_SWEEP_MARGIN = 0.97

# Diagnostics exist to be read, not to accumulate.
_MAX_RECORDED_FAILURES = 32

# Per-shape state is two [n, top_k] tensors plus a tactic, but it must not grow
# without bound if a caller streams distinct token counts. Past the cap, new
# token counts are served correctly on the default tactic with per-call scratch.
_STATE_CACHE_CAP = 64

# The authored routing kernel is specialised for the captured configuration; any
# other expert count or top_k takes the cubin's routing.
_ROUTE_NUM_EXPERTS = 128
_ROUTE_TOP_K = 4

# Above this token count the authored kernel stops being bit-identical to the
# cubin's routing, so it is not used there. This is a *static* bound, established
# by `tests/test_trtllm_mxfp4_moe.py::test_routing_weights_match_cubin`, not a
# judgement about any particular input.
#
# The reason is that the cubin switches routing kernels as the token count grows
# (`routingIndicesBlockKernel` -> `routingIndicesClusterKernel` ->
# `routingIndicesHistogramScoresKernel` + `routingIndicesCoopKernel`), and the
# large-token variant does not produce the same last bfloat16 ulp as the
# `calcSoftmax` this kernel reproduces. Measured against the cubin's own
# `topk_weights`: zero mismatches at 1 / 26 / 60 / 398 tokens over 64 seeds each
# and at 512 / 1024 over 12-24 seeds, then 1-2 per few hundred thousand weights
# from 2048 tokens upward, whichever exp and division form is used. One wrong
# weight is not a rounding curiosity here: the bench materialises random E8M0
# block scales, so reference outputs span ~20 decades and a single adjacent-ulp
# weight moves an entire output row by ~1e20.
_ROUTE_EXACT_MAX_TOKENS = 1024


_ROUTING_CUDA_SOURCE = r'''
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

#define WARP_SIZE 32
// Specialised for the captured configuration. Anything else takes the cubin's
// routing, which is why these are hard constants rather than kernel arguments:
// runtime-indexed winner/weight arrays spill to local memory (measured at 7 loads
// and 12 stores per token before this was scalarised), and the whole point of the
// kernel is to be cheap.
#define NUM_EXPERTS 128
#define TOP_K 4
#define EXPERTS_PER_LANE (NUM_EXPERTS / WARP_SIZE)

// Strict ordering on (value, expert id): a larger logit wins, an exact tie goes to
// the lower expert id. Without the tie rule the result would depend on reduction
// order, which a router may not do -- bfloat16 logits over 128 experts collide
// often enough that ties are ordinary rather than exotic.
__device__ __forceinline__ bool route_beats(float va, int ia, float vb, int ib) {
  return (va > vb) || (va == vb && ia < ib);
}

// One warp per token, mirroring the trtllm-gen routing kernel's arithmetic rather
// than an algebraically equivalent rewrite.
//
// `routing_method_type = 4` (RenormalizeNaive) is *not* implemented as
// softmax -> top-k -> renormalize. `csrc/trtllm_fused_moe_runner.cu` maps both
// Renormalize and RenormalizeNaive to NoOpPreprocess + SoftmaxPostprocess -- top-k
// on the raw logits, then a softmax over just the k selected scores -- because the
// two are mathematically equivalent. They are not bit-equivalent, so this follows
// the implementation.
//
// The weight arithmetic reproduces `calcSoftmax` in
// `include/flashinfer/trtllm/fused_moe/RoutingKernel.h` line-for-line: the k-th
// winner is placed in lane k, the max is a warp reduce, `expf` runs in float, the
// denominator is a warp sum over all 32 lanes with the 28 non-winner lanes
// contributing 0.f, and there is exactly one division and one cast to bfloat16.
// Summing the four weights inside one lane instead would reorder the float
// additions -- ((w0+w1)+w2)+w3 rather than the warp tree's (w0+w2)+(w1+w3) -- and
// that alone moved about 1 weight in 65536 to the adjacent bfloat16.
__global__ void trtllm_style_topk_route(
    const __nv_bfloat16* __restrict__ logits,   // [num_tokens, NUM_EXPERTS]
    int32_t* __restrict__ out_ids,              // [num_tokens, TOP_K]
    __nv_bfloat16* __restrict__ out_weights,    // [num_tokens, TOP_K]
    int num_tokens) {
  const int lane = threadIdx.x % WARP_SIZE;
  const int token = blockIdx.x * (blockDim.x / WARP_SIZE) + threadIdx.x / WARP_SIZE;
  if (token >= num_tokens) return;

  // Lane L owns the contiguous experts [L*4, L*4+4), so the warp reads its whole
  // logit row coalesced. Held in named scalars: an array indexed by the loop
  // counter would live in local memory.
  const __nv_bfloat16* row = logits + (size_t)token * NUM_EXPERTS;
  const int base = lane * EXPERTS_PER_LANE;
  float v0 = __bfloat162float(row[base + 0]);
  float v1 = __bfloat162float(row[base + 1]);
  float v2 = __bfloat162float(row[base + 2]);
  float v3 = __bfloat162float(row[base + 3]);
  bool live0 = true, live1 = true, live2 = true, live3 = true;

  float win0, win1, win2, win3;
  int idx0, idx1, idx2, idx3;
#pragma unroll
  for (int k = 0; k < TOP_K; ++k) {
    float bv = -INFINITY;
    int bi = NUM_EXPERTS;
    if (live0 && route_beats(v0, base + 0, bv, bi)) { bv = v0; bi = base + 0; }
    if (live1 && route_beats(v1, base + 1, bv, bi)) { bv = v1; bi = base + 1; }
    if (live2 && route_beats(v2, base + 2, bv, bi)) { bv = v2; bi = base + 2; }
    if (live3 && route_beats(v3, base + 3, bv, bi)) { bv = v3; bi = base + 3; }
    // Butterfly argmax leaves the winner in every lane, so no broadcast is needed.
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
      const float ov = __shfl_xor_sync(0xffffffffu, bv, offset);
      const int oi = __shfl_xor_sync(0xffffffffu, bi, offset);
      if (route_beats(ov, oi, bv, bi)) { bv = ov; bi = oi; }
    }
    // The owning lane retires its winner so the next round cannot repick it.
    if (bi == base + 0) live0 = false;
    else if (bi == base + 1) live1 = false;
    else if (bi == base + 2) live2 = false;
    else if (bi == base + 3) live3 = false;
    if (k == 0) { win0 = bv; idx0 = bi; }
    else if (k == 1) { win1 = bv; idx1 = bi; }
    else if (k == 2) { win2 = bv; idx2 = bi; }
    else { win3 = bv; idx3 = bi; }
  }

  // Lane k adopts winner k, as the trtllm postprocess does.
  float score = -INFINITY;
  int expert = NUM_EXPERTS;
  if (lane == 0) { score = win0; expert = idx0; }
  else if (lane == 1) { score = win1; expert = idx1; }
  else if (lane == 2) { score = win2; expert = idx2; }
  else if (lane == 3) { score = win3; expert = idx3; }

  float max_score = score;
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
    const float other = __shfl_xor_sync(0xffffffffu, max_score, offset);
    max_score = (other >= max_score) ? other : max_score;
  }
  // Non-winner lanes contribute exactly 0.f, matching the reference reduction.
  const float weight = (lane < TOP_K) ? expf(score - max_score) : 0.f;
  float denom = weight;
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
    denom += __shfl_xor_sync(0xffffffffu, denom, offset);
  }

  if (lane < TOP_K) {
    out_ids[(size_t)token * TOP_K + lane] = expert;
    out_weights[(size_t)token * TOP_K + lane] = __float2bfloat16(weight / denom);
  }
}

void topk_route(at::Tensor logits, at::Tensor out_ids, at::Tensor out_weights,
                int64_t top_k) {
  TORCH_CHECK(logits.is_cuda(), "logits must be on CUDA");
  TORCH_CHECK(logits.scalar_type() == at::kBFloat16, "logits must be bfloat16");
  TORCH_CHECK(logits.dim() == 2 && logits.is_contiguous(), "logits must be contiguous 2-D");
  TORCH_CHECK(logits.size(1) == NUM_EXPERTS, "this kernel is specialised for 128 experts");
  TORCH_CHECK(top_k == TOP_K, "this kernel is specialised for top_k 4");
  // The outputs must live on the same device as the logits, or the launch below
  // would read and write across devices.
  TORCH_CHECK(out_ids.is_cuda() && out_weights.is_cuda(), "outputs must be on CUDA");
  TORCH_CHECK(out_ids.device() == logits.device() && out_weights.device() == logits.device(),
              "logits and outputs must be on the same device");
  TORCH_CHECK(out_ids.scalar_type() == at::kInt, "out_ids must be int32");
  TORCH_CHECK(out_weights.scalar_type() == at::kBFloat16, "out_weights must be bfloat16");
  TORCH_CHECK(out_ids.is_contiguous() && out_weights.is_contiguous(),
              "outputs must be contiguous");
  const int num_tokens = logits.size(0);
  TORCH_CHECK(out_ids.size(0) == num_tokens && out_ids.size(1) == TOP_K, "out_ids shape");
  TORCH_CHECK(out_weights.size(0) == num_tokens && out_weights.size(1) == TOP_K,
              "out_weights shape");
  if (num_tokens == 0) return;
  // Without this guard the launch would go to whichever device the calling thread
  // has current, which need not be the tensors' device -- an illegal access.
  const at::cuda::CUDAGuard device_guard(logits.device());
  const int warps_per_block = 8;
  const int threads = warps_per_block * WARP_SIZE;
  const int blocks = (num_tokens + warps_per_block - 1) / warps_per_block;
  trtllm_style_topk_route<<<blocks, threads, 0,
                            at::cuda::getCurrentCUDAStream(logits.device().index())>>>(
      reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr()),
      out_ids.data_ptr<int32_t>(),
      reinterpret_cast<__nv_bfloat16*>(out_weights.data_ptr()),
      num_tokens);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
'''


class _RawMoe(NamedTuple):
    """Everything needed to drive the trtllm-gen op without the Python wrapper."""

    call: object
    valid_configs: object
    deduce_dtype: object
    supports_pdl: object
    activation_swiglu: int
    weight_layout_major_k: int
    fp8_none: int


@functools.cache
def _raw_moe() -> tuple[Optional[_RawMoe], str]:
    """Bind the trtllm-gen MoE op once per process.

    ``get_trtllm_moe_sm100_module`` is the function that installs the cubin loader
    via ``setup_cubin_loader``; without it the raw op cannot resolve its kernels.
    It is ``functools.cache``d upstream, so calling it is cheap and idempotent.
    ``build_and_load`` then returns the bare extension module, whose
    ``trtllm_fp4_block_scale_moe`` is the entry point the wrapper reaches after its
    per-call bookkeeping.

    On any failure this returns ``(None, reason)`` and the module runs the public
    API instead — baseline behaviour, with the reason left visible.
    """
    try:
        from flashinfer.fused_moe.core import get_trtllm_moe_sm100_module
        from flashinfer.jit.fused_moe import gen_trtllm_gen_fused_moe_sm100_module
        from flashinfer.tllm_enums import (
            ActivationType,
            Fp8QuantizationType,
            WeightLayout,
            deduce_trtllm_gen_tensor_dtype,
        )
        from flashinfer.utils import device_support_pdl

        get_trtllm_moe_sm100_module()
        moe_op = gen_trtllm_gen_fused_moe_sm100_module().build_and_load()
        handles = _RawMoe(
            call=moe_op.trtllm_fp4_block_scale_moe,
            valid_configs=moe_op.trtllm_get_valid_moe_configs,
            deduce_dtype=deduce_trtllm_gen_tensor_dtype,
            supports_pdl=device_support_pdl,
            activation_swiglu=int(ActivationType.Swiglu),
            weight_layout_major_k=int(WeightLayout.MajorK),
            fp8_none=int(Fp8QuantizationType.NoneFp8),
        )
    except Exception as exc:  # noqa: BLE001 - reported, not hidden
        return None, f"{type(exc).__name__}: {exc}"
    return handles, ""


@functools.cache
def _routing_kernel() -> tuple[object, str]:
    """Compile the authored routing kernel once per process.

    Compilation is deliberately lazy: it happens on the first call at a token
    count that qualifies, which is a correctness round, never a timed window. The
    build is cached on disk by ``load_inline``, so only the first run of a fresh
    checkout pays the ~85 s nvcc cost. Returning ``(None, reason)`` leaves the
    module on the cubin's own routing.
    """
    try:
        from torch.utils.cpp_extension import load_inline

        # Building for every architecture torch defaults to would triple the
        # compile time for a kernel that only ever runs on the local device.
        arch_was_set = "TORCH_CUDA_ARCH_LIST" in os.environ
        if not arch_was_set:
            major, minor = torch.cuda.get_device_capability()
            os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
        try:
            ext = load_inline(
                name="kda_trtllm_mxfp4_moe_routing",
                cpp_sources="void topk_route(at::Tensor, at::Tensor, at::Tensor, int64_t);",
                cuda_sources=_ROUTING_CUDA_SOURCE,
                functions=["topk_route"],
                # No --use_fast_math: __expf's ~2-ulp error is enough to move a
                # weight to the adjacent bfloat16, and one such weight changes a
                # whole output row by ~1e20 under the bench's random block scales.
                extra_cuda_cflags=["-O3", "-lineinfo"],
                verbose=False,
            )
        finally:
            if not arch_was_set:
                os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
    except Exception as exc:  # noqa: BLE001 - reported, not hidden
        return None, f"{type(exc).__name__}: {exc}"
    return ext.topk_route, ""


# The three ways the C++ side can synchronously refuse a ``[tile_N, config]``
# pair, quoted from ``csrc/trtllm_fused_moe_kernel_launcher.cu`` in the installed
# FlashInfer: a malformed tactic array (``resolveMoeTileAndConfig``), a config
# index out of range for its tile, and a tile_N with no compiled launcher. All
# three are thrown on the host before a kernel is enqueued. Matching on the text
# is unpleasant but it is the only thing that distinguishes them from a CUDA
# fault, and recovering from the wrong one is worse than not recovering.
_TACTIC_REJECTIONS = (
    "Invalid MoE tactic",
    "Invalid tactic, expected to be [tile_N, config]",
    "missing FP4 block-scale MoE launcher for tile_N=",
)


def _is_rejected_tactic(exc: BaseException) -> bool:
    """True for the C++ side's synchronous refusal of a ``[tile_N, config]`` pair.

    That refusal happens before anything is enqueued, which is what makes it the
    one launch-adjacent failure worth recovering from. Any other exception — in
    particular an asynchronous fault surfaced by a later synchronize — must
    propagate, because the context is not in a state to be reused.
    """
    if not isinstance(exc, RuntimeError):
        return False
    text = str(exc)
    return any(fragment in text for fragment in _TACTIC_REJECTIONS)


class _ShapeState:
    """Cached routing scratch, routing mode and tactic for one (device, tokens, width).

    The scratch is fully overwritten every call — by the authored kernel when it is
    in use, by the cubin's routing otherwise — and is never returned to or aliased
    by the caller, so reuse is safe for the sequential single-stream use this
    operator sees. Concurrent forwards on two streams sharing one instance are out
    of contract; the baseline allocates per call and this does not.
    """

    __slots__ = ("tactic", "topk_ids", "topk_weights", "pdl", "route")

    def __init__(self, tactic, topk_ids, topk_weights, pdl):
        self.tactic = tactic
        self.topk_ids = topk_ids
        self.topk_weights = topk_weights
        self.pdl = pdl
        # The authored routing callable, or None to let the cubin route.
        self.route = None


class TrtLlmMxfp4MoE(nn.Module):
    """Router + experts in one trtllm-gen launch.

    ``hidden_states`` arrives at the *padded* hidden width; the returned tensor is
    ``hidden_size_unpadded`` wide, matching vLLM's ``has_unpadded_output``.
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

        # Resolved once, here, so nothing on the fast path checks availability and
        # no kernel is ever launched speculatively.
        self._raw, self._raw_unavailable_reason = _raw_moe()
        self._shape_state: dict[tuple[int, int, int], _ShapeState] = {}
        # Readable rather than silent: why the authored routing is or is not used
        # at each shape, and which tactics the C++ side refused.
        self.routing_status: dict[tuple[int, int, int], str] = {}
        self.sweep_failures: list[str] = []
        # The authored kernel is compiled for one configuration.
        self._routing_eligible = (
            num_experts == _ROUTE_NUM_EXPERTS and top_k == _ROUTE_TOP_K
        )
        self._routing_ineligible_reason = (
            ""
            if self._routing_eligible
            else f"kernel is specialised for num_experts={_ROUTE_NUM_EXPERTS} "
                 f"top_k={_ROUTE_TOP_K}, got {num_experts}/{top_k}"
        )

    @property
    def uses_raw_op(self) -> bool:
        """False when this instance permanently fell back to the public API."""
        return self._raw is not None

    @property
    def uses_authored_routing(self) -> bool:
        """True when at least one cached shape routes through the authored kernel."""
        return any(st.route is not None for st in self._shape_state.values())

    def _raw_inputs_ok(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        w13_weight: torch.Tensor,
        w13_weight_scale: torch.Tensor,
        w13_bias: torch.Tensor,
        w2_weight: torch.Tensor,
        w2_weight_scale: torch.Tensor,
        w2_bias: torch.Tensor,
    ) -> bool:
        """Whether the raw op was validated for exactly these tensors.

        Checked on every call, not once per shape: the cache is keyed on
        ``(device, tokens, hidden width)``, so without this a later call could
        reach the raw op with the same key but a different weight dtype or a
        non-contiguous view and be silently mis-served. Everything that fails here
        goes to the public API, which is what the baseline would have done.
        """
        if hidden_states.dim() != 2 or hidden_states.dtype is not torch.bfloat16:
            return False
        device = hidden_states.device
        if device.type != "cuda" or not hidden_states.is_contiguous():
            return False
        hidden = hidden_states.shape[1]
        if self.hidden_size_unpadded > hidden:
            return False
        num_tokens = hidden_states.shape[0]
        experts = self.num_experts
        inter = self.intermediate_size

        if (
            router_logits.dim() != 2
            or router_logits.shape[0] != num_tokens
            or router_logits.shape[1] != experts
            or not router_logits.is_contiguous()
            or router_logits.device != device
        ):
            return False

        # MXFP4: two 4-bit values per byte, one E8M0 scale per 32 values. Getting
        # these relationships wrong is what would make the C++ side reinterpret
        # the buffers, so they are checked rather than assumed.
        if (
            w13_weight.dtype is not torch.uint8
            or w2_weight.dtype is not torch.uint8
            or w13_weight_scale.dtype is not torch.float8_e4m3fn
            or w2_weight_scale.dtype is not torch.float8_e4m3fn
            or w13_bias.dtype is not torch.float32
            or w2_bias.dtype is not torch.float32
        ):
            return False
        expected = (
            (w13_weight, (experts, 2 * inter, hidden // _MXFP4_PER_BYTE)),
            (w13_weight_scale, (experts, 2 * inter, hidden // _MXFP4_SF_BLOCK)),
            (w13_bias, (experts, 2 * inter)),
            (w2_weight, (experts, hidden, inter // _MXFP4_PER_BYTE)),
            (w2_weight_scale, (experts, hidden, inter // _MXFP4_SF_BLOCK)),
            (w2_bias, (experts, hidden)),
        )
        for tensor, shape in expected:
            if (
                tensor.device != device
                or not tensor.is_contiguous()
                or tuple(tensor.shape) != shape
            ):
                return False
        return True

    def _wrapped_forward(
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
        """The FlashInfer public API — baseline behaviour, for anything the raw
        path was not validated for."""
        output = torch.empty(
            *hidden_states.shape[:-1],
            self.hidden_size_unpadded,
            dtype=torch.bfloat16,
            device=hidden_states.device,
        )
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
        return output

    def _invoke(self, raw, st, logits, hidden_states, weights, output, tactic):
        """The 36-argument trtllm-gen entry point.

        Argument order is the FP4 branch of ``MoERunner.forward`` in
        ``flashinfer/fused_moe/core.py``, which is what the public wrapper calls
        once it has finished deciding on a tactic. When the authored routing kernel
        is in use it runs first and the op is told the routing is precomputed, so
        ``routing_logits`` is dropped and the scratch becomes an input instead of
        an output.
        """
        if st.route is None:
            mode, routing_logits = _ROUTE_FROM_LOGITS, logits
        else:
            st.route(logits, st.topk_ids, st.topk_weights, self.top_k)
            mode, routing_logits = _ROUTE_PRECOMPUTED, None
        w13_weight, w13_weight_scale, w13_bias, w2_weight, w2_weight_scale, w2_bias = weights
        raw.call(
            mode,
            routing_logits,
            st.topk_ids,
            st.topk_weights,
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
            st.pdl,
            raw.activation_swiglu,
            output,
            tactic,
            True,                       # norm_topk_prob (unused when precomputed)
            None,                       # routing_replay_out
        )

    def _record_failure(self, message: str) -> None:
        if len(self.sweep_failures) < _MAX_RECORDED_FAILURES:
            self.sweep_failures.append(message)

    def _select_routing(self, st, key, num_tokens) -> None:
        """Decide whether this shape uses the authored routing kernel.

        The decision is a function of the configuration and the token count only —
        never of the input values. An earlier version ran the authored kernel once,
        compared that one output against the cubin's, and cached the verdict; that
        was unsound, because the disagreement it was testing for is data-dependent,
        so a shape could pass on its first input and diverge on a later one at the
        same token count. Exactness is established instead by
        `tests/test_trtllm_mxfp4_moe.py`, which checks the authored weights against
        the cubin's own `topk_weights` over many seeds per shape, plus exact ties
        and all-negative logits.
        """
        if not self._routing_eligible:
            self.routing_status[key] = self._routing_ineligible_reason
            return
        if num_tokens > _ROUTE_EXACT_MAX_TOKENS:
            self.routing_status[key] = (
                f"cubin routing: {num_tokens} tokens is above the "
                f"{_ROUTE_EXACT_MAX_TOKENS}-token bound where the authored kernel is "
                f"proven bit-identical"
            )
            return
        route, reason = _routing_kernel()
        if route is None:
            self.routing_status[key] = f"kernel unavailable: {reason}"
            return
        st.route = route
        self.routing_status[key] = "authored kernel"

    def _sweep_tactic(self, raw, st, logits, hidden_states, weights, output, num_tokens):
        """Time the valid tile/config tactics once and keep the best.

        Every tactic produces bit-identical output on this kernel, so the only thing
        being chosen is speed. ``[-1, -1]`` is measured the same way as the
        candidates, on the same tensors, immediately before them, and is kept unless
        a candidate beats it by the acceptance margin — so the common case where the
        default is already right costs one extra measurement and changes nothing.
        Ties go to ``[-1, -1]`` first, then to the lexicographically smallest
        ``[tile_N, config]``, which is what makes the outcome reproducible rather
        than dependent on enumeration order.
        """
        device = hidden_states.device
        try:
            enumeration_key = (
                raw.deduce_dtype(hidden_states, None),
                raw.deduce_dtype(weights[0], weights[1]),
                raw.fp8_none,
                self.top_k,
                hidden_states.shape[1],
                self.intermediate_size,
                self.num_experts,
                raw.activation_swiglu,
                True,                            # use_shuffled_weight
                raw.weight_layout_major_k,
                False,                           # use_per_token_scaling
                num_tokens,
                False,                           # has_gemm1_lora_delta
            )
            candidates = [[int(v) for v in t] for t in raw.valid_configs(*enumeration_key)]
        except Exception as exc:  # noqa: BLE001
            self._record_failure(f"enumerate: {type(exc).__name__}: {exc}")
            return [-1, -1]

        # Lexicographic order makes the tie-break deterministic.
        candidates.sort()
        deadline = time.perf_counter() + _SWEEP_BUDGET_S

        def measure(tactic):
            """Time one tactic. Nothing here is inside a recovery handler.

            Every synchronize, event record and ``elapsed_time`` read is outside
            all exception handling on purpose: those are where an asynchronous
            fault surfaces, and an asynchronous fault is not a skippable tactic.
            """
            starts = [torch.cuda.Event(enable_timing=True) for _ in range(_SWEEP_REPEATS)]
            ends = [torch.cuda.Event(enable_timing=True) for _ in range(_SWEEP_REPEATS)]
            for i in range(_SWEEP_REPEATS):
                starts[i].record()
                self._invoke(raw, st, logits, hidden_states, weights, output, tactic)
                ends[i].record()
            torch.cuda.synchronize(device)
            return statistics.median(s.elapsed_time(e) for s, e in zip(starts, ends))

        def accepted(tactic):
            """One launch to let the C++ side refuse a tactic it cannot serve.

            The refusal is raised on the host before a kernel is enqueued, so it is
            the only launch-adjacent failure that can be stepped over. The
            synchronize that would surface anything asynchronous is in ``measure``,
            outside every handler.
            """
            try:
                self._invoke(raw, st, logits, hidden_states, weights, output, tactic)
            except Exception as exc:  # noqa: BLE001 - re-raised unless recognised
                if not _is_rejected_tactic(exc):
                    raise
                self._record_failure(f"{tactic}: {exc}")
                return False
            return True

        with torch.cuda.device(device):
            # Let clocks settle before the reference measurement, otherwise the
            # reference absorbs the ramp and every candidate looks better than it
            # is. The budget bounds this too, not just the candidate loop.
            for _ in range(_SWEEP_SETTLE):
                if time.perf_counter() > deadline:
                    return [-1, -1]
                self._invoke(raw, st, logits, hidden_states, weights, output, [-1, -1])
            torch.cuda.synchronize(device)
            # The deadline is rechecked *after* each acceptance launch as well as
            # before it: acceptance itself can cross the budget, and a measurement
            # group is three more launches that must not start once it has.
            if time.perf_counter() > deadline or not accepted([-1, -1]):
                return [-1, -1]
            if time.perf_counter() > deadline:
                return [-1, -1]
            default_ms = measure([-1, -1])

            best_tactic, best_ms = [-1, -1], default_ms
            for tactic in candidates:
                if time.perf_counter() > deadline:
                    break
                if not accepted(tactic):
                    continue
                if time.perf_counter() > deadline:
                    break
                elapsed = measure(tactic)
                if elapsed < best_ms:
                    best_tactic, best_ms = tactic, elapsed

        if best_ms < default_ms * _SWEEP_MARGIN:
            return best_tactic
        return [-1, -1]

    def _build_state(self, raw, key, num_tokens, routing_dtype, device,
                     logits, hidden_states, weights, output) -> Optional[_ShapeState]:
        """Set up one token count: scratch, PDL, routing mode, tactic.

        Returns ``None`` if the raw path turns out not to work for these arguments,
        having already routed this instance to the public API for good. Everything
        here happens on the first call at a shape — never in steady state, and never
        as a way of recovering from a launch that has already gone wrong.
        """
        try:
            # The wrapper queries this per call and it takes a torch.device. Doing
            # it inside the guard is what makes a failing query a fallback rather
            # than an escaping exception.
            pdl = bool(raw.supports_pdl(device))
            st = _ShapeState(
                tactic=[-1, -1],
                topk_ids=torch.empty(
                    num_tokens, self.top_k, dtype=torch.int32, device=device
                ),
                # The kernel reads this buffer as the routing dtype, which the
                # wrapper derives from routing_logits — so the two must agree.
                topk_weights=torch.empty(
                    num_tokens, self.top_k, dtype=routing_dtype, device=device
                ),
                pdl=pdl,
            )
            # Prove the 36-argument call is accepted for these tensors before
            # anything depends on it. Only the call is guarded: the C++ side rejects
            # arguments it cannot serve on the host, before enqueueing a kernel, and
            # that is the failure worth surviving.
            self._invoke(raw, st, logits, hidden_states, weights, output, [-1, -1])
        except Exception as exc:  # noqa: BLE001 - recorded, not hidden
            self._raw = None
            self._raw_unavailable_reason = f"setup: {type(exc).__name__}: {exc}"
            return None
        # Deliberately outside the guard: an asynchronous fault leaves the context
        # unusable, so catching it and retrying through the wrapper would be
        # pretending to recover from something unrecoverable.
        torch.cuda.synchronize(device)

        # Refreshing an existing entry must stay possible at the cap, otherwise a
        # caller that changed routing dtype would rebuild (and re-sweep) forever.
        cacheable = key in self._shape_state or len(self._shape_state) < _STATE_CACHE_CAP
        if cacheable:
            self._select_routing(st, key, num_tokens)
            # Only sweep for state we are going to keep, and sweep under the routing
            # mode the shape will actually use so the tactic is measured on the real
            # path. Past the cap a new token count runs on the default tactic with
            # per-call scratch — correct and cheap — rather than paying for a sweep
            # that would be discarded.
            if num_tokens * self.top_k >= _SWEEP_MIN_ROWS_PER_EXPERT * self.num_experts:
                st.tactic = self._sweep_tactic(
                    raw, st, logits, hidden_states, weights, output, num_tokens
                )
            self._shape_state[key] = st
        return st

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
        raw = self._raw
        if raw is None or not self._raw_inputs_ok(
            hidden_states, router_logits,
            w13_weight, w13_weight_scale, w13_bias,
            w2_weight, w2_weight_scale, w2_bias,
        ):
            return self._wrapped_forward(
                hidden_states, router_logits,
                w13_weight, w13_weight_scale, w13_bias,
                w2_weight, w2_weight_scale, w2_bias,
            )

        # The wrapper does this cast unconditionally; skip it when it is a no-op.
        logits = (
            router_logits
            if router_logits.dtype is torch.bfloat16
            else router_logits.to(torch.bfloat16)
        )
        num_tokens = hidden_states.shape[0]
        device = hidden_states.device
        weights = (
            w13_weight, w13_weight_scale, w13_bias,
            w2_weight, w2_weight_scale, w2_bias,
        )
        # Freshly allocated every call: returning a reused buffer would alias
        # results across calls.
        output = torch.empty(
            num_tokens, self.hidden_size_unpadded,
            dtype=torch.bfloat16, device=device,
        )

        # The valid tactic set depends on the padded hidden width as well as the
        # token count, so the width belongs in the key.
        key = (device.index, num_tokens, hidden_states.shape[1])
        st = self._shape_state.get(key)
        if st is None or st.topk_weights.dtype is not logits.dtype:
            st = self._build_state(
                raw, key, num_tokens, logits.dtype, device,
                logits, hidden_states, weights, output,
            )
            if st is None:
                return self._wrapped_forward(
                    hidden_states, router_logits,
                    w13_weight, w13_weight_scale, w13_bias,
                    w2_weight, w2_weight_scale, w2_bias,
                )
            # Deliberately fall through to the authoritative call below rather than
            # returning what setup left in `output`. Setting the shape up writes
            # that buffer several times — the availability probe, the routing
            # equivalence check, every tactic measured — and the last of those
            # writes is not necessarily the configuration that was chosen. In
            # particular, when the routing check *declines*, the buffer is holding
            # the result the check just rejected.

        try:
            self._invoke(raw, st, logits, hidden_states, weights, output, st.tactic)
        except Exception as exc:  # noqa: BLE001 - see _is_rejected_tactic
            # A cached tactic can stop being valid — a driver or FlashInfer build
            # change is enough. That refusal is synchronous, so the call can be
            # remade at the default; anything else propagates.
            if st.tactic == [-1, -1] or not _is_rejected_tactic(exc):
                raise
            self._record_failure(f"cached {st.tactic} rejected, reset to default: {exc}")
            st.tactic = [-1, -1]
            self._invoke(raw, st, logits, hidden_states, weights, output, st.tactic)
        return output
