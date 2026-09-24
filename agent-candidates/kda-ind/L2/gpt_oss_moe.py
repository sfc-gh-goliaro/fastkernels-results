"""GPT-OSS MoE with the trtllm-gen launch path rebuilt for SM100.

The expert math is the baseline's, byte for byte: same parameters, same
``prepare_trtllm_mxfp4_weights`` layout, same trtllm-gen MXFP4 kernel, same
SwiGLU constants, same renormalize-naive routing. What changes is the *path that
reaches* the kernel.

Why: on B200 the baseline is host-bound for every benched token count up to ~400.
``flashinfer.trtllm_fp4_block_scale_moe`` costs ~0.55 ms of host time per call and
is nearly flat in token count, while the GPU timeline for those shapes is
0.037-0.418 ms -- so 6-15x of the GPU timeline is idle waiting for Python. The
wrapper rebuilds a ``MoERunner``, a ``MoeRunnerInputs`` and a ``TuningConfig``,
takes the global ``AutoTuner`` lock, and allocates ``topk_ids`` / ``topk_weights``
on every call; ``forward_impl`` adds three ``nn.Module.__call__`` hops and an
``F.pad`` that costs a fill kernel plus a copy. None of that is expert math.

Four changes, each independently removable (see the ``_USE_*`` switches):

1. **Persistent pre-zeroed workspaces.** ``xpad`` is allocated with
   ``torch.zeros``; the pad columns ``xpad[:, hidden_size:]`` are written once at
   allocation and never touched again, which deletes ``F.pad``'s fill kernel. The
   per-call row copy that remains *is* the pad.
2. **An ``addmm`` router** into a persistent logits buffer over a strided view of
   ``xpad``, replacing an ``nn.Linear`` hop and its output allocation. Measured
   caveat: this was expected to also drop cuBLAS's separate
   ``splitKreduce_kernel`` at small token counts, and it does **not** --
   ``addmm`` with an ``out=`` buffer reaches the same cuBLAS heuristic as
   ``F.linear``, so the router still costs two kernels (11.5 + 7.6 us at one
   token; see profile/phase1_candidate_timeline/REPORT.md). The win here is host
   time only, not GPU kernels.
3. **A cached ``MoERunner`` called directly** with a prebuilt kwargs dict and a
   prebuilt inputs list. This is the internal API ``flashinfer``'s own in-tree
   ``TrtllmFp4RoutedRunner`` delegates to, mirrored here the same way (cache the
   runner, prebuild the static kwargs, resolve ``enable_pdl`` once, read the
   result out of the inputs list). It is *not* a public API: ``MoERunner`` is in
   no ``__all__`` and its ``forward`` bracket-indexes about twenty mandatory
   keyword arguments, so a ``flashinfer`` bump can break it loudly (``KeyError``)
   or quietly (reordered inputs list). Every failure path falls back to the
   baseline's public-wrapper call.
4. **Self-managed tactic selection and CUDA-graph replay** per exact token count. A
   padding ladder is implemented and reachable (``_USE_BUCKETS``) but off by default,
   because measured against exact-shape graphs it does not pay.

``forward`` returns a view of a persistent buffer rather than a fresh allocation;
see ``_CLONE_OUTPUT`` for the reasoning and the opt-out.

Facts worth not rediscovering:

* The activation copy is issued **outside** the graph. This is mandatory, not
  stylistic: the bench's shifting input pool hands every timed iteration a tensor
  at a different ``data_ptr``, so a captured copy would bake in a stale source
  pointer and silently read old memory. Anyone who "optimizes" it back inside the
  capture reintroduces that bug, and the bench's 1e-2 tolerance may not catch it.
* A ``tactic`` is not an integer. It is a two-element ``[tile_N, config]`` sequence
  as returned by ``get_valid_tactics``; the scalar ``-1`` is the "use the built-in
  heuristic" sentinel, and the launcher's own size check rejects any other bare
  integer.
* The tactic cache is keyed by the *exact* token count and the ``-1`` fallback wraps
  replay as well as search. Measured caveat on the reason: ``get_valid_tactics`` does
  return different sets per token count (144 entries at 1-256 tokens, 192 at 1024, 128
  at 4096), but on this ``flashinfer`` build those sets are **advisory** -- a tactic
  listed at one token is accepted at 4096 and computes a bit-exact result. Only a
  tactic outside the kernel's range is refused. So the keying is not load-bearing for
  correctness *today*; it is kept because it costs nothing, because tactic choice at
  the wrong token count is a performance mistake even when it is accepted, and because
  a future version may enforce the sets. See docs/notes.md.
* ``do_finalize=True`` is required on the direct path: the non-finalized
  intermediate results are discarded there.
* ``topk_ids`` is int32 and ``expert_weights`` matches the routing-logits dtype.
  The FP4 launcher does not dtype-check caller-supplied buffers, so a mismatch
  would be written through silently.
* ``F.pad``'s fill kernel really is gone: ``at::FillFunctor`` appears in every
  baseline timeline and none of the candidate's, worth 5.3 us at 60 tokens and
  46.9 us plus 36.8 MB at 16384. The row copy that remains *is* the pad.
* The tactic search, not the launch path, is what wins the prefill case: at 16384
  tokens the expert kernels move from a ``t128x32x256`` tile to ``t128x64x256``,
  9606 us -> 4770 us at unchanged DRAM traffic.
* ``flashinfer``'s autotuner cache is *not* missed because of bucketed keys --
  ``map_to_hybrid_bucket`` is applied symmetrically on store and lookup and is
  idempotent on ladder values, so bucketing alone cannot cause a miss. The
  observed "No tuned config covers ..." fallback is better explained by nothing
  having been stored for that profile (the tuner early-returns without storing on
  OOM, and its cold-L2 profiling clones the whole input list several times) or by
  a differing ``tune_max_num_tokens`` between the tuning call and the measured
  one, or by ``_find_nearest_profile`` mishandling non-power-of-2 token counts
  (flashinfer PR 2821, which also added launchers for supported ``tileN`` values
  that previously had none). Recorded here so the non-bug is not "fixed" later;
  see docs/notes.md. Selecting the tactic
  ourselves is still right on cost and control grounds: one tuning call costs
  1.6-29 s per shape, tunes every bucket in the ladder, and leaves the per-call
  host overhead in place.
"""

from __future__ import annotations

import collections
import functools
import os
import sys
import time

import torch

# The baseline class carries the contract this module must not perturb: the six
# expert parameters and their weight loaders, the lazy MXFP4 shuffle, the
# non-SM100 delegation and the _use_custom_op / AllReduce placement. Subclassing
# it means there is no copied code to drift out of sync. This import must be
# absolute: a relative ``.gpt_oss_moe`` resolves back to *this* module under the
# bench's candidate finder.
from fastkernels.tasks.baseline.L2.gpt_oss_moe import GptOssMoE as _BaselineGptOssMoE

from .trtllm_mxfp4_moe import ROUTING_RENORMALIZE_NAIVE


def _switch(name: str, default: bool = True) -> bool:
    raw = os.environ.get("FASTKERNELS_GPT_OSS_MOE_" + name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


# Each stage is switchable so a regression can be bisected by dropping the most
# recent one without touching the others.
_USE_DIRECT_RUNNER = _switch("DIRECT", True)
_USE_TACTIC_SEARCH = _switch("TACTIC", True)
_USE_CUDA_GRAPH = _switch("GRAPH", True)
# Off by default: the fused router is not bit-exact against ``addmm`` and the
# measured win does not justify the drift. See tools/probe_router.py.
_USE_FUSED_ROUTER = _switch("FUSED_ROUTER", False)
# ``forward`` returns a *view* of a persistent output buffer, which the next call
# for the same bucket overwrites. That is safe under this bench (each correctness
# round synchronizes and compares before the next call, and the timed loop
# discards the result) and it is what keeps the one-token case at 48 us -- a clone
# there would add an allocation and a copy to a 48 us budget. It is nonetheless a
# weaker contract than the baseline's freshly allocated output, so a caller that
# needs to hold on to results across calls can set this and get owning tensors
# back at the cost of one allocation and one copy per forward.
_CLONE_OUTPUT = _switch("CLONE_OUTPUT", False)
# Off by default: measured per ladder rung against the alternative that actually
# competes -- a graph captured at the exact token count -- padding does not pay.
# Kept reachable so that configuration is real code rather than a claim, so the
# comparison behind the default can be reproduced at this commit, and so the
# padded path in `_launch` is exercised by tools/probe_equiv.py.
_USE_BUCKETS = _switch("BUCKETS", False)
# The shipped default, kept separately so a probe that toggles _USE_BUCKETS to
# measure both configurations can still assert what the module actually ships.
_USE_BUCKETS_DEFAULT = _USE_BUCKETS

# Token counts get a graph at their *exact* count by default. The padding ladder
# below is only consulted when _USE_BUCKETS is set.
#
# The plan called for this ladder with round-up padding, on the theory that padded
# rows are nearly free. Measured per rung at the token count that pads the most,
# against the alternative that actually competes -- a graph captured at the exact
# count, not an eager call -- padding does not pay. The numbers live in one place,
# docs/notes.md, and are produced by tools/probe_runtime.py::probe_padding_ladder;
# they are deliberately not restated here, because duplicating them is how the
# Round 0 records drifted apart.
#
# What a ladder *would* buy is fewer distinct workspaces, which matters for a
# workload that sweeps batch size continuously. Eviction (below) is what makes that
# unnecessary for correctness; the ladder remains a latency/memory trade a caller
# can opt into.
_BUCKETS = (1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024)

# Above this the GPU already dominates and a graph buys nothing (measured at
# M=16384: 3.263 ms replay vs 3.141 ms direct), so the largest shapes keep the
# eager direct call and no graph pool at all.
_GRAPH_MAX_TOKENS = 1024

# Live workspaces are bounded by least-recently-used eviction rather than by a hard
# cap. A hard cap would send the (N+1)-th distinct token count to the baseline path
# for the lifetime of the module, which is not shape-general; eviction keeps every
# token count on the optimized path at a bounded memory cost.
_MAX_WORKSPACES = 16

# Tactic search is bounded so a pathological shape cannot eat the harness's
# wall-clock allowance (parent-side caps: 1200 s per operator, 600 s of log
# silence). The cache is process-global and keyed by (runner config, exact token
# count), so the five benched shapes each pay for their own search once even
# though the harness rebuilds the module per case.
_TACTIC_BUDGET_TOTAL_S = 150.0
_TACTIC_BUDGET_PER_SHAPE_S = 30.0
_TACTIC_FINALISTS = 5
# Keep the searched tactic only if it beats the built-in heuristic by more than
# timing noise; ties and losses keep ``-1``.
_TACTIC_MIN_GAIN = 0.98
# The budget can only be checked *between* timing batches, so the batches are
# sized by wall clock rather than by a fixed iteration count: at a large token
# count a fixed 15 iterations would block for seconds and overshoot the budget by
# more than the budget itself. Iteration counts are derived from a rough first
# measurement so one batch stays around these bounds at any shape.
_TACTIC_PROBE_BATCH_MS = 60.0
_TACTIC_FINAL_BATCH_MS = 300.0

_TACTIC_CACHE: dict = {}
_tactic_seconds_spent = 0.0


@functools.lru_cache(maxsize=1)
def _fused_router_class():
    """Compile the deterministic Triton router GEMM, on first use only.

    At one token the router is the largest remaining item on the GPU timeline:
    cuBLAS services ``[1, 2880] x [2880, 128]`` with a split-K GEMM plus a separate
    ``splitKreduce_kernel``, 12.2 + 7.3 = 19.5 us of a 37 us graph-replay timeline,
    for a problem whose whole working set is 737 kB of router weight. That is
    launch and tiling overhead, not bandwidth.

    Two deliberate departures from the shape this was first sketched in:

    * It does **not** fuse the activation copy. The copy has to stay outside the
      graph (the bench rotates the input pointer), while the router wants to be
      *inside* it, over the persistent padded buffer. Fusing them would force the
      router back out of the graph and hand the per-call host cost back. So this
      replaces the GEMM only.
    * The K reduction is a fixed split into private per-split fp32 partials
      followed by an ordered sum -- never ``atomic_add`` into a shared
      accumulator. Unordered atomics make the logits vary run to run, and a
      near-tied top-4 can then flip and move a token's output by roughly
      ``0.25 * |delta w2_bias|`` (~7e-3): inside the bench's 1e-2 band yet not
      reproducible, which is the one failure mode that must not ship silently.

    The kernels are defined here rather than at module scope so that importing
    this module never requires Triton, and so the default ``addmm`` path carries
    no compile cost.
    """
    import triton
    import triton.language as tl

    @triton.jit
    def router_partials(x_ptr, w_ptr, partial_ptr, M, K,
                        x_stride_m, w_stride_e, partial_stride_s, partial_stride_m,
                        BLOCK_M: tl.constexpr, BLOCK_E: tl.constexpr,
                        BLOCK_K: tl.constexpr, CHUNK_K: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_s = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_e = tl.arange(0, BLOCK_E)
        row_mask = offs_m < M
        k_start = pid_s * CHUNK_K
        acc = tl.zeros((BLOCK_M, BLOCK_E), dtype=tl.float32)
        for k_offset in range(0, CHUNK_K, BLOCK_K):
            offs_k = k_start + k_offset + tl.arange(0, BLOCK_K)
            k_mask = offs_k < K
            a = tl.load(x_ptr + offs_m[:, None] * x_stride_m + offs_k[None, :],
                        mask=row_mask[:, None] & k_mask[None, :], other=0.0)
            b = tl.load(w_ptr + offs_e[:, None] * w_stride_e + offs_k[None, :],
                        mask=k_mask[None, :], other=0.0)
            acc += tl.dot(a, tl.trans(b))
        tl.store(partial_ptr + pid_s * partial_stride_s
                 + offs_m[:, None] * partial_stride_m + offs_e[None, :],
                 acc, mask=row_mask[:, None])

    @triton.jit
    def router_reduce(partial_ptr, bias_ptr, out_ptr, M,
                      partial_stride_s, partial_stride_m, out_stride_m,
                      BLOCK_M: tl.constexpr, BLOCK_E: tl.constexpr,
                      SPLITS: tl.constexpr):
        pid_m = tl.program_id(0)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_e = tl.arange(0, BLOCK_E)
        row_mask = offs_m < M
        acc = tl.zeros((BLOCK_M, BLOCK_E), dtype=tl.float32)
        for split in range(SPLITS):
            acc += tl.load(partial_ptr + split * partial_stride_s
                           + offs_m[:, None] * partial_stride_m + offs_e[None, :],
                           mask=row_mask[:, None], other=0.0)
        acc += tl.load(bias_ptr + offs_e).to(tl.float32)[None, :]
        tl.store(out_ptr + offs_m[:, None] * out_stride_m + offs_e[None, :],
                 acc.to(tl.bfloat16), mask=row_mask[:, None])

    @triton.jit
    def router_single(x_ptr, w_ptr, bias_ptr, out_ptr, M, K,
                      x_stride_m, w_stride_e, out_stride_m,
                      BLOCK_M: tl.constexpr, BLOCK_E: tl.constexpr,
                      BLOCK_K: tl.constexpr):
        pid_m = tl.program_id(0)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_e = tl.arange(0, BLOCK_E)
        row_mask = offs_m < M
        acc = tl.zeros((BLOCK_M, BLOCK_E), dtype=tl.float32)
        for k_offset in range(0, K, BLOCK_K):
            offs_k = k_offset + tl.arange(0, BLOCK_K)
            k_mask = offs_k < K
            a = tl.load(x_ptr + offs_m[:, None] * x_stride_m + offs_k[None, :],
                        mask=row_mask[:, None] & k_mask[None, :], other=0.0)
            b = tl.load(w_ptr + offs_e[:, None] * w_stride_e + offs_k[None, :],
                        mask=k_mask[None, :], other=0.0)
            acc += tl.dot(a, tl.trans(b))
        acc += tl.load(bias_ptr + offs_e).to(tl.float32)[None, :]
        tl.store(out_ptr + offs_m[:, None] * out_stride_m + offs_e[None, :],
                 acc.to(tl.bfloat16), mask=row_mask[:, None])

    class FusedRouter:
        """Drop-in for ``torch.addmm(bias, xview, W.t(), out=logits)``.

        Partial buffers are allocated per token count and reused, so a captured
        graph sees a fixed address; a capture is always preceded by warmup calls at
        the same token count, so the buffer exists before capture.
        """

        BLOCK_M = 16
        BLOCK_E = 128
        BLOCK_K = 64
        # Below this, row parallelism alone leaves the machine idle and the K split
        # earns its second kernel; above it the unsplit single-kernel path wins.
        SPLIT_BELOW_TOKENS = 128
        SPLITS = 8

        def __init__(self, hidden_size, num_experts):
            if num_experts != self.BLOCK_E:
                raise ValueError(
                    f"this router covers all experts in one tile of "
                    f"{self.BLOCK_E}; got {num_experts}")
            self.hidden_size = hidden_size
            self.num_experts = num_experts
            self._partials = {}

        def _partial_buffer(self, tokens, device):
            buffer = self._partials.get(tokens)
            if buffer is None:
                rows = -(-tokens // self.BLOCK_M) * self.BLOCK_M
                buffer = torch.empty(self.SPLITS, rows, self.num_experts,
                                     dtype=torch.float32, device=device)
                self._partials[tokens] = buffer
            return buffer

        def __call__(self, x, weight, bias, out):
            tokens, hidden = x.shape
            grid_m = -(-tokens // self.BLOCK_M)
            if tokens >= self.SPLIT_BELOW_TOKENS:
                router_single[(grid_m,)](
                    x, weight, bias, out, tokens, hidden,
                    x.stride(0), weight.stride(0), out.stride(0),
                    BLOCK_M=self.BLOCK_M, BLOCK_E=self.BLOCK_E,
                    BLOCK_K=self.BLOCK_K)
                return out
            partials = self._partial_buffer(tokens, x.device)
            # Round the split's K span up to BLOCK_K so the inner trip count is a
            # constexpr and every split covers a whole number of tiles.
            chunk = -(-hidden // self.SPLITS)
            chunk = -(-chunk // self.BLOCK_K) * self.BLOCK_K
            router_partials[(grid_m, self.SPLITS)](
                x, weight, partials, tokens, hidden,
                x.stride(0), weight.stride(0), partials.stride(0),
                partials.stride(1), BLOCK_M=self.BLOCK_M, BLOCK_E=self.BLOCK_E,
                BLOCK_K=self.BLOCK_K, CHUNK_K=chunk)
            router_reduce[(grid_m,)](
                partials, bias, out, tokens,
                partials.stride(0), partials.stride(1), out.stride(0),
                BLOCK_M=self.BLOCK_M, BLOCK_E=self.BLOCK_E, SPLITS=self.SPLITS)
            return out

    return FusedRouter


class _Workspace:
    """Persistent per-token-count buffers plus the prebuilt runner inputs list.

    ``tokens`` is the padded token count this workspace is valid at. Every buffer
    is sliced to it, because the runner asserts that ``output``,
    ``routing_logits``, ``topk_ids``, ``expert_weights`` and ``hidden_states`` all
    share the same leading dimension. A captured graph is valid only at this token
    count: the launcher sizes its internal scratch from the token count, so
    replaying at any other count is wrong by construction and never done.
    """

    __slots__ = ("tokens", "xpad", "xview", "logits", "topk_ids",
                 "expert_weights", "out", "inputs", "rows_written", "graph",
                 "tactic")

    def __init__(self, tokens, hidden_size, hidden_pad, num_experts, top_k,
                 device):
        self.tokens = tokens
        # Zeroed at allocation: this is where the pad columns get written, once.
        self.xpad = torch.zeros(tokens, hidden_pad, dtype=torch.bfloat16,
                                device=device)
        self.xview = self.xpad[:, :hidden_size]
        self.logits = torch.empty(tokens, num_experts, dtype=torch.bfloat16,
                                  device=device)
        self.topk_ids = torch.empty(tokens, top_k, dtype=torch.int32,
                                    device=device)
        self.expert_weights = torch.empty(tokens, top_k, dtype=torch.bfloat16,
                                          device=device)
        # The kernel writes an unpadded output (vLLM's ``has_unpadded_output``),
        # so this is hidden_size wide, not hidden_pad.
        self.out = torch.empty(tokens, hidden_size, dtype=torch.bfloat16,
                               device=device)
        self.inputs = None
        # Rows [rows_written, tokens) of ``xview`` are known to be zero.
        self.rows_written = 0
        self.graph = None
        self.tactic = -1

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in
                   (self.xpad, self.logits, self.topk_ids, self.expert_weights,
                    self.out))


class GptOssMoE(_BaselineGptOssMoE):
    """MXFP4-native MoE with the host path rebuilt; expert math unchanged."""

    def __init__(self, config):
        super().__init__(config)
        # Everything below is built lazily on the first forward, after
        # load_state_dict. __init__ must not shuffle weights, build a runner,
        # search a tactic or capture a graph: it runs before the checkpoint (or
        # the bench's weight sharing) has touched the expert payloads.
        self._runner = None
        self._runner_key = None
        self._static_kwargs = None
        self._router_weight_t = None
        self._ws_by_size: collections.OrderedDict = collections.OrderedDict()
        self._ws_for_tokens: dict = {}
        self._fused_router = None
        # Configuration is captured per instance at construction, not read from the
        # module globals on every call. Workspaces and graphs are built lazily on the
        # first forward, so a probe that flipped a global between constructing a module
        # and calling it would silently get the other configuration -- which has caused
        # two void measurements in this workspace already. A probe that wants a
        # non-default configuration sets these attributes on the instance.
        self._use_buckets = _USE_BUCKETS
        self._use_graph = _USE_CUDA_GRAPH
        # Inspectable, for the local gates: how many times the MXFP4 shuffle ran,
        # why the fast path was abandoned if it was, and which setup actions
        # fired on which forward.
        self.prep_calls = 0
        self.fallback_reason = None
        self.setup_log: list = []
        self.forward_calls = 0

    # -- lazy setup ---------------------------------------------------------

    def process_weights_after_loading(self):
        if self._processed:
            return
        super().process_weights_after_loading()
        self.prep_calls += 1
        if not self.use_trtllm or not _USE_DIRECT_RUNNER:
            return
        try:
            self._build_launch_state()
        except Exception as exc:  # noqa: BLE001 - any failure keeps the baseline
            self._abandon_fast_path(exc)

    def _build_launch_state(self):
        """Cache the runner and every argument that does not vary per call.

        Mirrors ``flashinfer.fused_moe.runners.TrtllmFp4RoutedRunner``: build the
        inner ``MoERunner`` once, prebuild the static kwargs its ``forward``
        bracket-indexes, and resolve ``enable_pdl`` here rather than per call so
        the value is also stable across graph capture and replay.
        """
        from flashinfer.fused_moe.core import (
            ActivationType, DtypeTrtllmGen, Fp8QuantizationType,
            RoutingInputMode, WeightLayout, get_trtllm_moe_sm100_module,
        )
        from flashinfer.tllm_enums import deduce_trtllm_gen_tensor_dtype
        from flashinfer.utils import device_support_pdl

        device = self._w13_shuffled.device
        experts = self.num_experts
        # Deduce the two dtypes from the actual tensors, exactly as the public
        # wrapper does, so a weight-layout change cannot silently select a
        # different kernel here than the baseline gets.
        dtype_act = deduce_trtllm_gen_tensor_dtype(
            torch.empty(0, dtype=torch.bfloat16, device=device), None)
        dtype_weights = deduce_trtllm_gen_tensor_dtype(
            self._w13_shuffled, self._w13_scale)

        module = get_trtllm_moe_sm100_module()
        self._runner = module.MoERunner(
            top_k=self.top_k,
            num_local_experts=experts,
            dtype_act=dtype_act,
            dtype_weights=dtype_weights,
            fp8_quantization_type=Fp8QuantizationType.NoneFp8,
            hidden_size=self._H_pad,
            intermediate_size=self._I_pad,
            activation_type=ActivationType.Swiglu.value,
            weight_layout=WeightLayout.MajorK,
            use_shuffled_weight=True,
            use_per_token_scaling=False,
            num_experts=experts,
        )
        # Identifies the kernel configuration a cached tactic was measured
        # against; the token count is the other half of the cache key.
        self._runner_key = (int(dtype_act), int(dtype_weights), self.top_k,
                            self._H_pad, self._I_pad, experts,
                            int(ActivationType.Swiglu.value),
                            int(WeightLayout.MajorK))

        # The SwiGLU alpha/beta/clamp-limit tensors and the routing-method
        # constant come from the baseline's own submodule, so they cannot drift
        # away from the values the baseline kernel sees.
        scalars = self.trtllm_moe
        self._static_kwargs = dict(
            routing_input_mode=RoutingInputMode.FromLogits,
            routing_bias=None,
            gemm1_weights=self._w13_shuffled,
            gemm1_weights_scale=self._w13_scale,
            gemm1_bias=self._w13_bias_f32,
            gemm1_alpha=scalars.gemm1_alpha,
            gemm1_beta=scalars.gemm1_beta,
            gemm1_clamp_limit=scalars.gemm1_clamp_limit,
            gemm2_weights=self._w2_shuffled,
            gemm2_weights_scale=self._w2_scale,
            gemm2_bias=self._w2_bias_f32,
            output1_scale_scalar=None,
            output1_scale_gate_scalar=None,
            output2_scale_scalar=None,
            per_token_scale=None,
            num_experts=experts,
            n_group=None,
            topk_group=None,
            local_expert_offset=0,
            routed_scaling_factor=None,
            routing_method_type=ROUTING_RENORMALIZE_NAIVE,
            do_finalize=True,
            enable_pdl=device_support_pdl(device),
        )
        # Cached transpose: a view, so it tracks in-place weight updates, and
        # nothing rebinds ``router.weight.data`` after the first forward.
        self._router_weight_t = self.router.weight.t()
        if _USE_FUSED_ROUTER:
            # Off by default; see _fused_router_class for why, and
            # tools/probe_router.py for the measurement behind that default.
            self._fused_router = _fused_router_class()(self.hidden_size,
                                                      self.num_experts)

    def _abandon_fast_path(self, exc: BaseException):
        """Drop to the baseline's public-wrapper path for good, and say why."""
        self._runner = None
        self._static_kwargs = None
        self._ws_by_size = collections.OrderedDict()
        self._ws_for_tokens = {}
        if self.fallback_reason is None:
            self.fallback_reason = repr(exc)

    def _note(self, event: str, tokens: int):
        self.setup_log.append((event, tokens, self.forward_calls))

    # -- forward ------------------------------------------------------------

    def forward_impl(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self._processed:
            self.process_weights_after_loading()
        self.forward_calls += 1
        if self._runner is None:
            return super().forward_impl(hidden_states)

        original_shape = hidden_states.shape
        x = self._token_matrix(hidden_states)
        tokens = x.shape[0]

        try:
            workspace = self._workspace_for(tokens)
        except Exception as exc:  # noqa: BLE001
            self._abandon_fast_path(exc)
            return super().forward_impl(hidden_states)
        if workspace is None:
            return super().forward_impl(hidden_states)

        try:
            output = self._launch(workspace, x, tokens)
        except Exception as exc:  # noqa: BLE001 - never return a half-written buffer
            self._abandon_fast_path(exc)
            return super().forward_impl(hidden_states)

        if _CLONE_OUTPUT:
            output = output.clone()
        if self.tp_size > 1 and not self._use_custom_op:
            output = self.allreduce(output)
        return output.view(original_shape)

    def _token_matrix(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Flatten to [tokens, hidden_size] without copying when possible.

        ``reshape`` rather than ``view`` so a narrowed or otherwise
        non-default-strided activation is accepted instead of raising; for the
        contiguous case it is the same view the baseline takes.
        """
        if hidden_states.dim() == 2 and hidden_states.shape[1] == self.hidden_size:
            return hidden_states
        return hidden_states.reshape(-1, self.hidden_size)

    def _launch(self, workspace, x, tokens):
        # Keeps rows [rows_written, tokens) of xview zero. With one workspace per
        # exact token count this never fires; it is retained because the invariant
        # is what makes padding correct, and padding is a two-line change away
        # (see the note on the bucket ladder above).
        if workspace.rows_written > tokens:
            workspace.xview[tokens:workspace.rows_written].zero_()
        workspace.rows_written = tokens
        # The one and only copy of the activation, always outside the graph.
        workspace.xview[:tokens].copy_(x)

        if workspace.graph is not None:
            workspace.graph.replay()
        else:
            self._router_and_experts_guarded(workspace)
        return workspace.out[:tokens]

    def _router_and_experts(self, workspace):
        """The captured region: fixed operand pointers, nothing else."""
        if self._fused_router is not None:
            self._fused_router(workspace.xview, self.router.weight,
                               self.router.bias, workspace.logits)
        else:
            torch.addmm(self.router.bias, workspace.xview,
                        self._router_weight_t, out=workspace.logits)
        self._runner.forward(workspace.inputs, workspace.tactic,
                             **self._static_kwargs)

    def _router_and_experts_guarded(self, workspace):
        try:
            self._router_and_experts(workspace)
        except Exception:
            if workspace.tactic == -1:
                raise
            # Tactic validity is a kernel-side property of the token count, so a
            # replayed tactic can be rejected. Drop this workspace back to the
            # built-in heuristic rather than failing the forward, and poison the
            # process-global cache entry too: without that, every module the bench
            # rebuilds would read the same rejected tactic back out and retry it.
            self._note("tactic rejected on replay", workspace.tokens)
            if self._runner_key is not None:
                _TACTIC_CACHE[(self._runner_key, workspace.tokens)] = -1
            workspace.tactic = -1
            self._router_and_experts(workspace)

    # -- workspaces ---------------------------------------------------------

    def _padded_tokens(self, tokens: int) -> int:
        """The size the workspace serving this token count is built at."""
        if self._use_buckets and self._use_graph and tokens <= _BUCKETS[-1]:
            for bucket in _BUCKETS:
                if bucket >= tokens:
                    return bucket
        return tokens

    def _workspace_for(self, tokens: int):
        """The workspace this token count runs in.

        The token-count -> workspace map is authoritative, not merely observational:
        it is what lets one workspace serve several token counts (a padding ladder),
        and what lets a test bind a token count to a deliberately oversized
        workspace in order to exercise the padded path.
        """
        workspace = self._ws_for_tokens.get(tokens)
        if workspace is not None and self._ws_by_size.get(workspace.tokens) is workspace:
            self._ws_by_size.move_to_end(workspace.tokens)
            return workspace

        padded = self._padded_tokens(tokens)
        workspace = self._ws_by_size.get(padded)
        if workspace is None:
            workspace = self._build_workspace(padded)
            self._ws_by_size[padded] = workspace
            self._evict_workspaces()
        else:
            self._ws_by_size.move_to_end(padded)
        self._ws_for_tokens[tokens] = workspace
        return workspace

    def _evict_workspaces(self):
        """Drop the least recently used workspaces back to the bound.

        Evicting releases the workspace's buffers and its captured graph, and with
        it the graph's private pool. That also invalidates any output view still
        held from a call that ran in it -- the same lifetime rule that already
        applies to two consecutive calls at one token count, see _CLONE_OUTPUT.
        """
        while len(self._ws_by_size) > _MAX_WORKSPACES:
            size, evicted = self._ws_by_size.popitem(last=False)
            for count, workspace in list(self._ws_for_tokens.items()):
                if workspace is evicted:
                    del self._ws_for_tokens[count]
            evicted.graph = None
            self._note("evicted workspace", size)

    def _build_workspace(self, tokens: int):
        from flashinfer.fused_moe.core import MoeRunnerInputs

        workspace = _Workspace(tokens, self.hidden_size, self._H_pad,
                               self.num_experts, self.top_k,
                               self._w13_shuffled.device)
        # Field order is load-bearing: the autotuner and the runner both index
        # this list positionally.
        workspace.inputs = MoeRunnerInputs(
            output=workspace.out,
            routing_logits=workspace.logits,
            topk_ids=workspace.topk_ids,
            expert_weights=workspace.expert_weights,
            hidden_states=workspace.xpad,
            hidden_states_scale=None,
            gemm1_lora_delta=None,
            per_token_scale=None,
        ).to_list()
        self._note("workspace", tokens)

        workspace.tactic = self._select_tactic(workspace)
        if self._use_graph and tokens <= _GRAPH_MAX_TOKENS:
            self._capture(workspace)
        return workspace

    def persistent_bytes(self) -> int:
        """Total bytes held by the workspaces (graph pools excluded)."""
        return sum(w.nbytes() for w in self._ws_by_size.values())

    # -- tactic selection ---------------------------------------------------

    def _select_tactic(self, workspace):
        global _tactic_seconds_spent
        if not _USE_TACTIC_SEARCH:
            return -1
        key = (self._runner_key, workspace.tokens)
        if key in _TACTIC_CACHE:
            return _TACTIC_CACHE[key]
        if _tactic_seconds_spent >= _TACTIC_BUDGET_TOTAL_S:
            self._note("tactic budget exhausted", workspace.tokens)
            return -1
        # Announce the search on stderr. The parent's stall watchdog kills a
        # worker after 600 s with no write to its *stderr* log, and the worker's
        # own progress prints go to stdout, so a long silent search would be
        # indistinguishable from a hang.
        print(f"[gpt_oss_moe] tactic search: {workspace.tokens} tokens",
              file=sys.stderr, flush=True)
        started = time.perf_counter()
        try:
            tactic = self._search_tactic(workspace, started)
        except Exception as exc:  # noqa: BLE001 - any failure keeps the heuristic
            self._note(f"tactic search failed: {exc!r}", workspace.tokens)
            tactic = -1
        spent = time.perf_counter() - started
        _tactic_seconds_spent += spent
        _TACTIC_CACHE[key] = tactic
        self._note(f"tactic {tactic}", workspace.tokens)
        print(f"[gpt_oss_moe] tactic search: {workspace.tokens} tokens -> "
              f"{tactic} in {spent:.1f} s ({_tactic_seconds_spent:.1f} s of "
              f"{_TACTIC_BUDGET_TOTAL_S:.0f} s spent)", file=sys.stderr, flush=True)
        return tactic

    def _search_tactic(self, workspace, started):
        """Two-stage search over ``get_valid_tactics``, bounded by wall clock.

        Stage one times every valid tactic cheaply, stage two re-times the best
        few, and the winner is kept only if it beats the built-in heuristic by
        more than noise. Runs for whatever token count is first seen at runtime;
        there is no table keyed to particular shapes.
        """
        tactics = self._runner.get_valid_tactics(workspace.inputs, None)
        if not tactics:
            return -1
        # Search against a realistic expert distribution: the logits buffer is
        # uninitialized at this point, and an all-equal-logits buffer would send
        # every token to the same four experts and mis-rank the tactics. A
        # private generator keeps the global RNG stream untouched.
        generator = torch.Generator(device=workspace.logits.device)
        generator.manual_seed(0)
        workspace.logits.normal_(0.0, 1.0, generator=generator)

        budget = min(_TACTIC_BUDGET_PER_SHAPE_S,
                     _TACTIC_BUDGET_TOTAL_S - _tactic_seconds_spent)
        # One cheap measurement first, purely to size the batches below. Without
        # it a fixed iteration count would make a single un-interruptible batch
        # cost seconds at a large token count.
        rough = self._time_tactic(workspace, -1, 1, 3)
        if rough is None:
            return -1
        per_call = max(rough, 1e-3)
        probe_iters = max(1, min(3, int(_TACTIC_PROBE_BATCH_MS / per_call)))
        final_iters = max(3, min(15, int(_TACTIC_FINAL_BATCH_MS / per_call)))

        scored = []
        for tactic in tactics:
            if time.perf_counter() - started > 0.6 * budget:
                break
            elapsed = self._time_tactic(workspace, tactic, 1, probe_iters)
            if elapsed is not None:
                scored.append((elapsed, tactic))
        if not scored:
            return -1
        scored.sort(key=lambda pair: pair[0])

        best, best_time = -1, None
        for _, tactic in scored[:_TACTIC_FINALISTS]:
            if time.perf_counter() - started > budget:
                break
            elapsed = self._time_tactic(workspace, tactic, 2, final_iters)
            if elapsed is not None and (best_time is None or elapsed < best_time):
                best, best_time = tactic, elapsed
        if best == -1 or best_time is None:
            return -1
        # The head-to-head re-time below is what makes "never slower than the
        # heuristic" true, so it is not optional: if there is no budget left to
        # run it, the searched tactic is unproven and the heuristic wins.
        if time.perf_counter() - started > budget:
            return -1
        heuristic = self._time_tactic(workspace, -1, 2, final_iters)
        if heuristic is not None and best_time < heuristic * _TACTIC_MIN_GAIN:
            return best
        return -1

    def _time_tactic(self, workspace, tactic, warmup, iters):
        """Median-free mean latency of one tactic, or None if it is rejected.

        A tactic that ``get_valid_tactics`` returns can still be refused by the
        launcher; such a tactic is dropped from the candidate set rather than
        allowed to break the forward.
        """
        try:
            for _ in range(warmup):
                self._runner.forward(workspace.inputs, tactic,
                                     **self._static_kwargs)
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iters):
                self._runner.forward(workspace.inputs, tactic,
                                     **self._static_kwargs)
            end.record()
            torch.cuda.synchronize()
            return start.elapsed_time(end) / iters
        except Exception:  # noqa: BLE001
            return None

    # -- graph capture ------------------------------------------------------

    def _capture(self, workspace):
        try:
            from flashinfer.autotuner import AutoTuner
            if getattr(AutoTuner.get(), "is_tuning_mode", False):
                raise RuntimeError(
                    "refusing to capture while flashinfer's autotuner is in "
                    "tuning mode: that path synchronizes and captures its own "
                    "graph")
            # Warm up on the calling stream first so the capture is not the first
            # time cuBLAS and the launcher reserve their scratch.
            print(f"[gpt_oss_moe] graph capture: {workspace.tokens} tokens",
                  file=sys.stderr, flush=True)
            for _ in range(3):
                self._router_and_experts(workspace)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self._router_and_experts(workspace)
            torch.cuda.synchronize()
            if getattr(AutoTuner.get(), "is_tuning_mode", False):
                # The check above is a read of shared state with no lock, so it is
                # a time-of-check/time-of-use test: something could have entered
                # tuning mode (and its own capture) while we were capturing.
                # Re-reading it afterwards turns that race into a discarded graph
                # instead of a graph that may have captured someone else's work.
                raise RuntimeError(
                    "flashinfer entered autotuner tuning mode during capture")
            workspace.graph = graph
            self._note("graph", workspace.tokens)
        except Exception as exc:  # noqa: BLE001 - keep the eager direct call
            workspace.graph = None
            self._note(f"graph capture failed: {exc!r}", workspace.tokens)
