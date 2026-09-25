from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

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


@triton.jit
def _shared_gate_up_silu(
    x_ptr,
    w_ptr,
    y_ptr,
    router_w_ptr,
    router_ptr,
    M: tl.constexpr,
    H: tl.constexpr,
    I: tl.constexpr,
    R: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    gate = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    up = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    router = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    for k in range(0, H, BLOCK_K):
        k_idx = k + offs_k
        x = tl.load(
            x_ptr + offs_m[:, None] * H + k_idx[None, :],
            mask=(offs_m[:, None] < M) & (k_idx[None, :] < H),
            other=0.0,
        )
        w_gate = tl.load(
            w_ptr + offs_n[:, None] * H + k_idx[None, :],
            mask=(offs_n[:, None] < I) & (k_idx[None, :] < H),
            other=0.0,
        )
        w_up = tl.load(
            w_ptr + (I + offs_n[:, None]) * H + k_idx[None, :],
            mask=(offs_n[:, None] < I) & (k_idx[None, :] < H),
            other=0.0,
        )
        gate = tl.dot(x, w_gate.T, gate)
        up = tl.dot(x, w_up.T, up)
        if pid_n < R // BLOCK_N:
            w_router = tl.load(
                router_w_ptr + offs_n[:, None] * H + k_idx[None, :],
            )
            router = tl.dot(x, w_router.T, router)

    # Match the BF16 projection boundary in the unfused implementation.
    gate = gate.to(tl.bfloat16).to(tl.float32)
    up = up.to(tl.bfloat16).to(tl.float32)
    y = (gate * tl.sigmoid(gate) * up).to(tl.bfloat16)
    tl.store(
        y_ptr + offs_m[:, None] * I + offs_n[None, :],
        y,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < I),
    )
    if pid_n < R // BLOCK_N:
        tl.store(
            router_ptr + offs_m[:, None] * R + offs_n[None, :],
            router,
            mask=offs_m[:, None] < M,
        )


@triton.jit
def _shared_down_add(
    x_ptr,
    w_ptr,
    routed_ptr,
    M: tl.constexpr,
    H: tl.constexpr,
    I: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    for k in range(0, I, BLOCK_K):
        k_idx = k + offs_k
        x = tl.load(
            x_ptr + offs_m[:, None] * I + k_idx[None, :],
            mask=(offs_m[:, None] < M) & (k_idx[None, :] < I),
            other=0.0,
        )
        w = tl.load(
            w_ptr + offs_n[:, None] * I + k_idx[None, :],
            mask=(offs_n[:, None] < H) & (k_idx[None, :] < I),
            other=0.0,
        )
        acc = tl.dot(x, w.T, acc)

    offsets = offs_m[:, None] * H + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < H)
    routed = tl.load(routed_ptr + offsets, mask=mask)
    shared = acc.to(tl.bfloat16)
    tl.store(routed_ptr + offsets, routed + shared, mask=mask)


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
        self._shared_act_buffer = None
        self._router_buffer = None

        # Custom-op dispatch for torch.compile (flipped by enable_custom_ops
        # once the model is wrapped with torch.compile). ``_layer_name`` is
        # populated by auto_register_no_compile_layers.
        self._use_custom_op = False
        self._layer_name = ""

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
        if not self.use_trtllm or self._trtllm_weights_ready:
            return
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

    def forward_impl(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_size)

        tokens = hidden_states.shape[0]
        fused_shared = (
            self.shared_experts is not None
            and self.tp_size == 1
            and tokens <= 128
            and self.intermediate_per_tp == 1024
            and self.num_shared_experts == 1
            and self.num_experts == 256
        )
        if fused_shared:
            if (
                self._shared_act_buffer is None
                or self._shared_act_buffer.shape[0] < tokens
                or self._shared_act_buffer.device != hidden_states.device
            ):
                self._shared_act_buffer = torch.empty(
                    tokens,
                    self.intermediate_per_tp,
                    dtype=hidden_states.dtype,
                    device=hidden_states.device,
                )
            shared_act = self._shared_act_buffer[:tokens]
            if (
                self._router_buffer is None
                or self._router_buffer.shape[0] < tokens
                or self._router_buffer.device != hidden_states.device
            ):
                self._router_buffer = torch.empty(
                    tokens,
                    self.num_experts,
                    dtype=torch.float32,
                    device=hidden_states.device,
                )
            router_logits = self._router_buffer[:tokens]
            block_m = 16
            _shared_gate_up_silu[
                (triton.cdiv(tokens, block_m), triton.cdiv(self.intermediate_per_tp, 32))
            ](
                hidden_states,
                self.shared_experts.gate_up_proj.weight,
                shared_act,
                self.gate.weight,
                router_logits,
                M=tokens,
                H=self.hidden_size,
                I=self.intermediate_per_tp,
                R=self.num_experts,
                BLOCK_M=block_m,
                BLOCK_N=32,
                BLOCK_K=128,
                num_warps=4,
                num_stages=3,
            )
            shared_output = None
        else:
            shared_output = (
                self.shared_experts(hidden_states)
                if self.shared_experts is not None
                else None
            )

        if not fused_shared:
            router_logits = self.gate_linear(
                hidden_states,
                self.gate.weight,
                out_dtype=torch.float32,
            )
        if self.use_trtllm:
            # Routing, both GEMMs, ``routed_scaling_factor`` and the weighted
            # reduction all happen inside the kernel.
            self.trtllm_moe.tune_max_num_tokens = (
                1 << (tokens - 1).bit_length()
                if tokens <= 128
                else 16384
            )
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
            out.add_(shared_output)
        elif fused_shared:
            _shared_down_add[
                (triton.cdiv(tokens, block_m), triton.cdiv(self.hidden_size, 32))
            ](
                shared_act,
                self.shared_experts.down_proj.weight,
                out,
                M=tokens,
                H=self.hidden_size,
                I=self.intermediate_per_tp,
                BLOCK_M=block_m,
                BLOCK_N=32,
                BLOCK_K=128,
                num_warps=4,
                num_stages=3,
            )
        if self.tp_size > 1 and not self._use_custom_op:
            out = self.allreduce(out)
        return out.view(orig_shape)
