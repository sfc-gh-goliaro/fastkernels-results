"""Kimi-Linear MoE: trtllm-gen for prefill, one fused kernel for decode.

The baseline routes everything through flashinfer's ``trtllm_bf16_moe``. On
B200 that kernel's *GPU* time is near the HBM roofline, but its native entry
point costs ~615us of **host** time per call in this build (measured on the
raw TVM-FFI function with a fixed tactic, so it is neither the Python wrapper
nor the autotuner), and the benchmark times per-call latency. Meanwhile the
GPU work for the whole module is 57us at M=1 and 613us at M=443. Every
small-token scenario was therefore host-bound by up to 16x, and the four extra
PyTorch ops around the MoE (shared-expert GEMM, SiluAndMul, down GEMM, add)
each added their own launch on top.

So the decode path here is one ``torch.ops``-free extension call
(:mod:`kimi_moe_ako.cu`) that does the router projection, routing, both expert
GEMMs, SwiGLU, the renormalised weighted reduction *and* the shared expert in
chained kernels, with no Python-level kernel call at all. The shared expert is
folded in as expert index ``E``: its ``gate_up_proj`` / ``down_proj`` weights
already have exactly the per-expert layout of ``w13[e]`` / ``w2[e]``, so it
rides the same GEMMs with token list = identity and routing weight = 1, which
removes the separate MLP and the final add rather than just reordering them.

Below :data:`_C.kimi_moe_tok_max_m` tokens the extension switches to a
token-major SIMT decomposition -- one warp per (token, expert slot) over the
full contiguous K -- which needs no block table, no fp32 accumulator, no atomics
and no finalize pass, and runs the whole forward in three kernels. Above it the
wmma tiles win, because there each expert holds enough tokens that grouping them
per expert saves more weight traffic than the finer decomposition gains.

The router projection is computed inside the extension whenever that is provably
free of numerical consequence: ``_gate_fuse_probe`` walks M upward comparing
``torch.equal`` against the real L1 ``gate_linear`` output and records the
largest M that matches bitwise. Expert selection is a top-8 over near-ties, so a
different fp32 reduction order can flip a token, and one flipped token at M=26
is 3.8% of the output elements -- enough to fail the 0.99 match threshold on its
own.

Prefill (M above :data:`_FAST_MAX_M`) keeps the trtllm-gen kernel: there the
GPU is the bottleneck (2.1ms at M=16384, ~890 TFLOP/s) and a hand-written
grouped GEMM would have to beat a tuned trtllm-gen tactic, so the host cost is
already amortised.

``process_weights_after_loading`` keeps the original ``[E, 2I, H]`` /
``[E, H, I]`` tensors alive next to the shuffled BlockMajorK copies trtllm
needs. Both paths then read the layout they want; memory is not scored and the
BlockMajorK block permutation plus gate/up half-rotation is not worth
reproducing in the small-M kernel.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn

from ....infra.cuda_ext import lazy_op
from ....infra.tp import _tp_rank, _tp_size
from ..L1.allreduce import AllReduce
from ..L1.gate_linear import GateLinear
from ..L1.grouped_topk import GroupedTopK
from .trtllm_bf16_moe import (
    ROUTING_DEEPSEEK_V3,
    TrtLlmBf16MoE,
    prepare_trtllm_bf16_moe_weights,
    trtllm_bf16_moe_supported,
)
from .fused_experts import FusedExperts
from .llama_mlp import LlamaMLP
from .parallel_linear import ReplicatedLinear

_C = lazy_op("kimi_moe_ako", "kimi_moe_ako.cu")

# Above this many tokens the trtllm-gen path wins: its ~615us host cost is
# amortised by GPU work that a hand-written kernel would have to beat, and the
# fused kernel's redundant x / intermediate re-reads grow with the number of
# token tiles per expert.
_FAST_MAX_M = 512

# Tile-config override, for sweeping the table in kimi_moe_ako.cu; -1 uses the
# kernel's own M bucket.
_CFG = int(os.environ.get("FASTKERNELS_KIMI_MOE_CFG", "-1"))

# Kill switch for the in-extension router projection (A/B measurement). With
# "0" the bitwise probe never runs, so ``_gate_fuse_max`` stays 0 and every path
# takes the separate L1 ``gate_linear`` call -- including the token-major one,
# which then goes through ``kimi_moe_fused``'s own tiny-M branch.
_GATE_FUSE = os.environ.get("FASTKERNELS_KIMI_MOE_GATE_FUSE", "1") != "0"

# How the tiny path computes the router logits: 0 (shipped) = projection and
# routing in one kernel with an internal arrival barrier, 1 = two kernels.
# Measured; see ITERATIONS.md.
_GATE_MODE = int(os.environ.get("FASTKERNELS_KIMI_MOE_GATE_MODE", "0"))

# Sentinel for "compute the router logits inside the extension"; allocated once
# so the decode path never builds a tensor for it.
_EMPTY = None

# Whether the wmma (M > kTokMaxM) path also lets the extension compute the
# router logits, saving the L1 Python call. Same bitwise proof gates it.
_GEN_GATE = os.environ.get("FASTKERNELS_KIMI_MOE_GEN_GATE", "1") != "0"


class _Scratch:
    """One set of device scratch buffers shared by every KimiMoE layer.

    Layers run sequentially, so reuse is safe, and it keeps the decode path at
    a single allocation per call (the output tensor). Buffers grow to the
    largest M seen; the kernels take M as the stride, so extra capacity is
    simply unused tail.
    """

    __slots__ = ("key", "m", "cnt", "meta", "tlist", "twt", "blk_e", "blk_t",
                 "hbuf", "accf", "lg")

    def __init__(self):
        self.key = None
        self.m = 0

    def get(self, m, h, i, e, device):
        # Keyed on the layer geometry as well as M: every MoE layer in a model
        # shares one scratch set, and a differently shaped layer must not
        # inherit buffers sized for another's hidden / intermediate size.
        key = (h, i, e, device)
        if m <= self.m and key == self.key:
            return self
        nb = _C.kimi_moe_num_blocks(m)
        hrows = _C.kimi_moe_hbuf_rows(m)
        iopt = dict(dtype=torch.int32, device=device)
        # cnt / meta must start zeroed; finalize_kernel re-zeroes them at the
        # end of every call so the hot path needs no memset launch.
        self.cnt = torch.zeros(e + 1, **iopt)
        self.meta = torch.zeros(4, **iopt)
        self.tlist = torch.empty((e + 1) * m, **iopt)
        self.twt = torch.empty((e + 1) * m, dtype=torch.float32, device=device)
        self.blk_e = torch.empty(nb, **iopt)
        self.blk_t = torch.empty(nb, **iopt)
        # The block table is sized for the smallest token-tile height in the
        # config table and the intermediate for the largest (blocks * bm) product
        # over the table, so both bounds hold for every config at once.
        self.hbuf = torch.empty(hrows * i, dtype=torch.bfloat16, device=device)
        self.accf = torch.empty(m * h, dtype=torch.float32, device=device)
        # Router logits, when the wmma path computes them in-extension. Kept
        # separate from accf: route zeroes accf while other blocks still read
        # logits, so aliasing them would race.
        self.lg = torch.empty(m * e, dtype=torch.float32, device=device)
        self.m = m
        self.key = key
        return self


_SCRATCH = _Scratch()


class KimiMoE(nn.Module):
    def __init__(self, config, quant_config: dict | None = None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_token
        self.num_shared_experts = config.num_shared_experts
        self.num_expert_group = config.num_expert_group
        self.topk_group = config.topk_group
        self.routed_scaling_factor = config.routed_scaling_factor
        self.tp_size = _tp_size()
        self.intermediate_per_tp = config.moe_intermediate_size // self.tp_size

        self.gate = ReplicatedLinear(
            self.hidden_size,
            self.num_experts,
            bias=False,
            quant_config=None,
        )
        # Model dtype, not FP32: vLLM's KimiMoE declares this as
        # ``nn.Parameter(torch.empty(num_experts))`` under the model-dtype
        # default, unlike DeepSeek-V3 whose router bias really is FP32.
        self.gate.e_score_correction_bias = nn.Parameter(
            torch.empty(self.num_experts),
        )
        self.gate.e_score_correction_bias.weight_loader = (
            lambda p, w: p.data.copy_(w.to(p.dtype))
        )

        self.grouped_topk = GroupedTopK(
            scoring_func=config.moe_router_activation_func,
            renormalize=config.moe_renormalize,
            routed_scaling_factor=1.0,
            force_sorted=True,
        )
        self.w13 = nn.Parameter(
            torch.empty(
                config.num_experts,
                2 * self.intermediate_per_tp,
                config.hidden_size,
            ),
        )
        self.w13.weight_loader = self._w13_weight_loader
        self.w2 = nn.Parameter(
            torch.empty(
                config.num_experts,
                config.hidden_size,
                self.intermediate_per_tp,
            ),
        )
        self.w2.weight_loader = self._w2_weight_loader
        self.fused_experts = FusedExperts()
        self.gate_linear = GateLinear()
        self.shared_experts = (
            LlamaMLP(
                config,
                quant_config=quant_config,
                intermediate_size=config.moe_intermediate_size * self.num_shared_experts,
                reduce_results=False,
            )
            if self.num_shared_experts
            else None
        )
        self.allreduce = AllReduce()

        # trtllm-gen BF16 MoE: what vLLM 0.26 runs for Kimi's MoE on Blackwell.
        # Kimi routes with sigmoid scoring + a router bias + expert groups, which
        # vLLM's ``get_routing_method_type`` maps to DeepSeekV3; the kernel does
        # the gating, top-k, both GEMMs, ``routed_scaling_factor`` and the
        # weighted reduction itself.
        self.use_trtllm = trtllm_bf16_moe_supported()
        self.trtllm_moe = (
            TrtLlmBf16MoE(
                num_experts=self.num_experts,
                top_k=self.top_k,
                intermediate_size_per_partition=self.intermediate_per_tp,
                routing_method_type=ROUTING_DEEPSEEK_V3,
                num_expert_group=self.num_expert_group,
                topk_group=self.topk_group,
                routed_scaling_factor=self.routed_scaling_factor,
            )
            if self.use_trtllm
            else None
        )
        self._trtllm_weights_ready = False

        # Custom-op dispatch for torch.compile (flipped by enable_custom_ops
        # once the model is wrapped with torch.compile). ``_layer_name`` is
        # populated by auto_register_no_compile_layers.
        self._use_custom_op = False
        self._layer_name = ""

        # Fused decode path: unshuffled expert weights (filled by
        # ``process_weights_after_loading``) plus the shape gate resolved once.
        self._w13_ref = None
        self._w2_ref = None
        self._fast_ready = False
        self._fast_checked = False
        self._gate_fused = False
        self._tok_max = 0
        self._tok_cfg0 = 1 << 30
        self._gate_fuse_max = 0
        self._fn_tok = None
        self._fn_gen = None

    def _w13_weight_loader(self, param, loaded_weight, expert_id: int, is_w1: bool):
        n = self.intermediate_per_tp
        shard = loaded_weight.narrow(0, _tp_rank() * n, n)
        offset = 0 if is_w1 else n
        param.data[expert_id, offset:offset + n, :].copy_(shard)

    def _w2_weight_loader(self, param, loaded_weight, expert_id: int):
        n = self.intermediate_per_tp
        param.data[expert_id].copy_(loaded_weight.narrow(1, _tp_rank() * n, n))

    def process_weights_after_loading(self) -> None:
        """Shuffle expert weights into trtllm-gen's 4D BlockMajorK layout.

        Mirrors vLLM's ``convert_to_unquantized_kernel_format`` for the
        ``FLASHINFER_TRTLLM`` backend. Replaces the ``[E, 2*I, H]`` / ``[E, H, I]``
        tensors, so the Triton path is unavailable afterwards -- guarded by
        ``use_trtllm``.
        """
        if self._trtllm_weights_ready:
            return
        # Held as plain attributes, not parameters: the shuffle below replaces
        # ``self.w13`` / ``self.w2``, and the fused decode kernel wants the
        # original K-contiguous layout.
        self._w13_ref = self.w13.data
        self._w2_ref = self.w2.data
        if self.use_trtllm:
            w13, w2 = prepare_trtllm_bf16_moe_weights(self.w13.data, self.w2.data)
            self.w13 = nn.Parameter(w13, requires_grad=False)
            self.w2 = nn.Parameter(w2, requires_grad=False)
        self._trtllm_weights_ready = True

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._use_custom_op:
            # The all-reduce stays *outside* the opaque op: inside it Inductor
            # cannot see the collective, so ``AllReduceFusedAddRMSNormPass`` has
            # nothing to match at the MoE end of the layer -- half of every
            # layer's collectives. vLLM keeps its MoE reduction in traced Python
            # for the same reason (``moe_runner._maybe_reduce_final_output``).
            out = torch.ops.fastkernels.moe_forward(hidden_states, self._layer_name)
            if self.tp_size > 1:
                out = self.allreduce(out)
            return out
        return self.forward_impl(hidden_states)

    def _fast_probe(self) -> bool:
        """Resolve the fused path once: shapes it was written for, unshuffled
        weights available, extension buildable."""
        self._fast_checked = True
        try:
            if self._w13_ref is None or self.shared_experts is None:
                return False
            i = self.intermediate_per_tp
            h = self.hidden_size
            if (self.num_experts != 256 or self.top_k != 8
                    or self.num_expert_group != 1 or self.topk_group != 1
                    or h % 64 or i % 256):
                return False
            gu = self.shared_experts.gate_up_proj
            dn = self.shared_experts.down_proj
            if getattr(gu, "use_fp8", False) or getattr(dn, "use_fp8", False):
                return False
            if tuple(gu.weight.shape) != (2 * i, h) or tuple(dn.weight.shape) != (h, i):
                return False
            if tuple(self._w13_ref.shape) != (256, 2 * i, h):
                return False
            if tuple(self._w2_ref.shape) != (256, h, i):
                return False
            if self._w13_ref.dtype is not torch.bfloat16:
                return False
            if not (torch.cuda.is_available()
                    and torch.cuda.get_device_capability()[0] >= 9):
                return False
            _C.kimi_moe_num_blocks(1)  # forces the one-time JIT build
            global _EMPTY
            if _EMPTY is None:
                _EMPTY = torch.empty(0, dtype=torch.float32,
                                     device=self._w13_ref.device)
        except Exception:
            return False
        self._fast_ready = True
        # Bound once: `_C` is a lazy-extension proxy whose __getattr__ runs a
        # method call and a dict lookup on every access, and the decode path is
        # tight enough for that to show.
        self._tok_max = _C.kimi_moe_tok_max_m()
        self._tok_cfg0 = _C.kimi_moe_num_cfg()
        self._fn_tok = _C.kimi_moe_fused_tok
        self._fn_gen = _C.kimi_moe_fused
        self._gate_fused = _GATE_FUSE and self._gate_fuse_probe()
        return True

    def _gate_fuse_probe(self) -> bool:
        """Enable the fused router projection only if it is *bitwise* equal to
        the L1 ``gate_linear`` output.

        Expert selection is a top-8 over near-ties, so a different fp32
        reduction order can flip a token, and one flipped token at M=26 is 3.8%
        of the output elements -- enough to fail the 0.99 match threshold by
        itself. The fused kernel reproduces the L1 kernel's tiling exactly, but
        that is only true while ``gate_linear`` takes its own SIMT path: in a
        build where it falls back to cuBLAS or ``F.linear`` the orders differ.
        So prove it here, once, on the shapes the fused path will actually run.
        """
        self._gate_fuse_max = 0
        try:
            if self.hidden_size != 2304 or self.num_experts != 256:
                return False
            w = self.gate.weight
            if w.dtype is not torch.bfloat16 or not w.is_contiguous():
                return False
            g = torch.Generator(device=w.device).manual_seed(0x51DE)
            # Ascending, stopping at the first mismatch: `gate_linear` hands M
            # above its own kFastMaxM to cuBLAS, whose reduction order this does
            # not reproduce, so the largest bitwise-equal M is a property of the
            # L1 kernel's dispatch and is discovered rather than assumed.
            for m in (1, 2, 3, 4, 5, 8, 13, 16, 26, 33, 64, 65, 100, 443, 512):
                x = torch.randn(m, self.hidden_size, generator=g, device=w.device,
                                dtype=torch.bfloat16)
                ref = self.gate_linear(x, w, out_dtype=torch.float32)
                got = _C.kimi_moe_gate_logits(x, w)
                if not torch.equal(ref, got):
                    break
                self._gate_fuse_max = m
        except Exception:
            self._gate_fuse_max = 0
            return False
        return self._gate_fuse_max >= 1

    def _forward_fast(self, hidden_states: torch.Tensor) -> torch.Tensor:
        m = hidden_states.size(0)
        h = self.hidden_size
        i = self.intermediate_per_tp
        sc = _SCRATCH.get(m, h, i, self.num_experts, hidden_states.device)
        out = torch.empty_like(hidden_states)
        # Token-major tiny path with the router projection inside the same
        # extension call, so the whole forward is one native entry and three
        # kernels -- no Python-level kernel call and no separate launch quantum
        # for a 1.18MB GEMV. ``tlist`` / ``twt`` hold the (expert, weight) pair
        # list, ``accf`` the logits and ``cnt`` the arrival counters.
        #
        # When the projection cannot be proved bitwise-equal, the fall-through
        # below still gets the token-major *decomposition* -- ``kimi_moe_fused``
        # picks it for the same M bucket -- just with the L1 gate call in front.
        use_tok = m <= self._tok_max if _CFG < 0 else _CFG >= self._tok_cfg0
        if use_tok and m <= self._gate_fuse_max:
            self._fn_tok(
                hidden_states,
                self.gate.weight,
                self.gate.e_score_correction_bias,
                self._w13_ref,
                self._w2_ref,
                self.shared_experts.gate_up_proj.weight,
                self.shared_experts.down_proj.weight,
                out,
                sc.tlist, sc.twt, sc.accf, sc.cnt, sc.hbuf,
                self.routed_scaling_factor,
                _CFG, _GATE_MODE,
            )
            return out
        # An empty logits tensor tells the extension to do the router projection
        # itself, in the tiling proved bitwise-equal to the L1 kernel's.
        router_logits = (
            _EMPTY
            if (_GEN_GATE and m <= self._gate_fuse_max)
            else self.gate_linear(hidden_states, self.gate.weight,
                                  out_dtype=torch.float32)
        )
        self._fn_gen(
            hidden_states,
            router_logits,
            self.gate.e_score_correction_bias,
            self._w13_ref,
            self._w2_ref,
            self.shared_experts.gate_up_proj.weight,
            self.shared_experts.down_proj.weight,
            out,
            sc.cnt, sc.tlist, sc.twt, sc.meta, sc.blk_e, sc.blk_t, sc.hbuf,
            sc.accf, self.gate.weight, sc.lg,
            self.routed_scaling_factor,
            _CFG,
        )
        return out

    def forward_impl(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_size)

        if (hidden_states.size(0) <= _FAST_MAX_M
                and hidden_states.is_contiguous()
                and (self._fast_ready or not self._fast_checked)):
            if self._fast_ready or self._fast_probe():
                out = self._forward_fast(hidden_states)
                if self.tp_size > 1 and not self._use_custom_op:
                    out = self.allreduce(out)
                return out.view(orig_shape)

        shared_output = (
            self.shared_experts(hidden_states)
            if self.shared_experts is not None
            else None
        )

        router_logits = self.gate_linear(
            hidden_states,
            self.gate.weight,
            out_dtype=torch.float32,
        )
        if self.use_trtllm:
            # Routing, both GEMMs, ``routed_scaling_factor`` and the weighted
            # reduction all happen inside the kernel.
            out = self.trtllm_moe(
                hidden_states,
                self.w13,
                self.w2,
                router_logits,
                routing_bias=self.gate.e_score_correction_bias,
            )
        else:
            topk_weights, topk_ids = self.grouped_topk(
                router_logits,
                self.gate.e_score_correction_bias,
                num_expert_group=self.num_expert_group,
                topk_group=self.topk_group,
                topk=self.top_k,
            )

            out = self.fused_experts(
                hidden_states,
                self.w13,
                self.w2,
                topk_weights,
                topk_ids,
                self.num_experts,
            )
            out = out * self.routed_scaling_factor
        if shared_output is not None:
            out = out + shared_output
        if self.tp_size > 1 and not self._use_custom_op:
            out = self.allreduce(out)
        return out.view(orig_shape)
