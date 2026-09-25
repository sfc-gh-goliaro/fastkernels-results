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
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import triton
import triton.language as tl
import triton.language.extra.libdevice as libdevice

from ..L1.moe_align import MoeAlign


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


@triton.jit
def _route_kernel(
    logits,
    bias,
    topk_weights,
    topk_ids,
    num_experts: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_E: tl.constexpr,
    DEEPSEEK: tl.constexpr,
    ROUTE_SCALE: tl.constexpr,
):
    token = tl.program_id(0)
    offs = tl.arange(0, BLOCK_E)
    x = tl.load(logits + token * num_experts + offs, mask=offs < num_experts,
                other=-float("inf")).to(tl.float32)

    if DEEPSEEK:
        raw = 1.0 / (1.0 + libdevice.exp(-x))
        select = raw + tl.load(bias + offs, mask=offs < num_experts,
                               other=-float("inf")).to(tl.float32)
    else:
        # Softmax -> top-k -> renormalize is equivalent to normalizing the
        # exponentials of just the selected logits.
        x_max = tl.max(x, axis=0)
        raw = libdevice.exp(x - x_max)
        select = x

    chosen = tl.zeros((BLOCK_E,), tl.int1)
    denom = 0.0
    for k in range(TOP_K):
        available = tl.where(chosen, -float("inf"), select)
        expert = tl.argmax(available, axis=0)
        weight = tl.sum(tl.where(offs == expert, raw, 0.0), axis=0)
        tl.store(topk_ids + token * TOP_K + k, expert)
        tl.store(topk_weights + token * TOP_K + k, weight)
        denom += weight
        chosen |= offs == expert

    for k in range(TOP_K):
        weight = tl.load(topk_weights + token * TOP_K + k)
        tl.store(topk_weights + token * TOP_K + k,
                 weight * ROUTE_SCALE / denom)


@triton.jit
def _shuffle_row_32(row):
    """Map an unshuffled row to FlashInfer's epilogue row position."""
    inner = row % 32
    return (row // 32) * 32 + (inner % 4) * 8 + inner // 4


@triton.jit
def _gemm1_swiglu_kernel(
    hidden,
    weights,
    sorted_token_ids,
    expert_ids,
    num_tokens_post_padded,
    intermediate,
    num_route_rows: tl.constexpr,
    hidden_size: tl.constexpr,
    intermediate_size: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    block_m = tl.program_id(0)
    block_n = tl.program_id(1)
    if block_m * BLOCK_M >= tl.load(num_tokens_post_padded):
        return

    row_offs = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
    route_rows = tl.load(sorted_token_ids + row_offs)
    valid_rows = route_rows < num_route_rows
    token_rows = route_rows // TOP_K
    expert = tl.load(expert_ids + block_m)
    n = block_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # The conversion interleaves [up_i, gate_i] before applying the 32-row
    # epilogue shuffle. Compute both halves with one wider MMA and split the
    # adjacent accumulator columns afterward.
    pair_col = tl.arange(0, 2 * BLOCK_N)
    physical_row = _shuffle_row_32(
        2 * block_n * BLOCK_N + pair_col
    )
    acc_pair = tl.zeros((BLOCK_M, 2 * BLOCK_N), tl.float32)
    k = tl.arange(0, BLOCK_K)
    for kb in range(0, hidden_size, BLOCK_K):
        kk = kb + k
        a = tl.load(
            hidden + token_rows[:, None] * hidden_size + kk[None, :],
            mask=valid_rows[:, None],
            other=0.0,
        )
        # w13 is [E, H/64, 2I, 64]. The logical K dimension crosses a
        # physical block every 64 BF16 values.
        base = (
            expert * (hidden_size // 64) * (2 * intermediate_size) * 64
            + (kk[None, :] // 64) * (2 * intermediate_size) * 64
            + (kk[None, :] % 64)
        )
        pair = tl.trans(tl.load(
            weights + base + physical_row[:, None] * 64,
            mask=(block_n * BLOCK_N + pair_col[:, None] // 2)
            < intermediate_size,
            other=0.0,
        ))
        acc_pair = tl.dot(a, pair, acc_pair)

    paired = tl.reshape(acc_pair, (BLOCK_M, BLOCK_N, 2))
    acc_up, acc_gate = tl.split(paired)
    activated = acc_up * acc_gate / (1.0 + libdevice.exp(-acc_gate))
    tl.store(
        intermediate + route_rows[:, None] * intermediate_size + n[None, :],
        activated,
        mask=valid_rows[:, None] & (n[None, :] < intermediate_size),
    )


@triton.jit
def _gemm2_kernel(
    intermediate,
    weights,
    sorted_token_ids,
    expert_ids,
    num_tokens_post_padded,
    expert_output,
    num_route_rows: tl.constexpr,
    hidden_size: tl.constexpr,
    intermediate_size: tl.constexpr,
    META_BLOCK_M: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    block_m = tl.program_id(0)
    block_n = tl.program_id(1)
    subblocks = META_BLOCK_M // BLOCK_M
    meta_block_m = block_m // subblocks
    sub_block_m = block_m % subblocks
    row_start = meta_block_m * META_BLOCK_M + sub_block_m * BLOCK_M
    if row_start >= tl.load(num_tokens_post_padded):
        return

    row_offs = row_start + tl.arange(0, BLOCK_M)
    route_rows = tl.load(sorted_token_ids + row_offs)
    valid_rows = route_rows < num_route_rows
    expert = tl.load(expert_ids + meta_block_m)
    n = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
    physical_n = _shuffle_row_32(n)

    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    k = tl.arange(0, BLOCK_K)
    for kb in range(0, intermediate_size, BLOCK_K):
        kk = kb + k
        a = tl.load(
            intermediate + route_rows[:, None] * intermediate_size + kk[None, :],
            mask=valid_rows[:, None],
            other=0.0,
        )
        # w2 is [E, I/64, H, 64].
        base = (
            expert * (intermediate_size // 64) * hidden_size * 64
            + (kk[None, :] // 64) * hidden_size * 64
            + (kk[None, :] % 64)
        )
        b = tl.trans(tl.load(
            weights + base + physical_n[:, None] * 64,
            mask=n[:, None] < hidden_size,
            other=0.0,
        ))
        acc = tl.dot(a, b, acc)

    tl.store(
        expert_output + route_rows[:, None] * hidden_size + n[None, :],
        acc,
        mask=valid_rows[:, None] & (n[None, :] < hidden_size),
    )


@triton.jit
def _reduce_kernel(
    expert_output,
    route_weights,
    output,
    hidden_size: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    token = tl.program_id(0)
    n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_N,), tl.float32)
    for k in range(TOP_K):
        value = tl.load(
            expert_output + (token * TOP_K + k) * hidden_size + n,
            mask=n < hidden_size,
            other=0.0,
        ).to(tl.float32)
        weight = tl.load(
            route_weights + token * TOP_K + k
        ).to(tl.bfloat16).to(tl.float32)
        acc += value * weight
    tl.store(output + token * hidden_size + n, acc, mask=n < hidden_size)


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
        self._align = MoeAlign()
        self._buffers: dict[tuple[torch.device, int, int], tuple[torch.Tensor, ...]] = {}

    def _get_buffers(
        self, hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        m, hidden_size = hidden_states.shape
        key = (hidden_states.device, m, hidden_size)
        buffers = self._buffers.get(key)
        route_rows = m * self.top_k
        if buffers is None:
            buffers = (
                torch.empty((m, self.top_k), dtype=torch.float32,
                            device=hidden_states.device),
                torch.empty((m, self.top_k), dtype=torch.int32,
                            device=hidden_states.device),
                torch.empty((route_rows, self.intermediate_size_per_partition),
                            dtype=torch.bfloat16, device=hidden_states.device),
                torch.empty((route_rows, hidden_size), dtype=torch.bfloat16,
                            device=hidden_states.device),
                torch.empty_like(hidden_states),
            )
            self._buffers[key] = buffers
        return buffers

    def forward(
        self,
        hidden_states: torch.Tensor,
        w13: torch.Tensor,
        w2: torch.Tensor,
        router_logits: torch.Tensor,
        routing_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        topk_weights, topk_ids, intermediate, expert_output, output = (
            self._get_buffers(hidden_states)
        )
        m, hidden_size = hidden_states.shape
        route_scale = (
            1.0 if self.routed_scaling_factor is None
            else self.routed_scaling_factor
        )
        _route_kernel[(m,)](
            router_logits,
            routing_bias if routing_bias is not None else router_logits,
            topk_weights,
            topk_ids,
            num_experts=self.num_experts,
            TOP_K=self.top_k,
            BLOCK_E=triton.next_power_of_2(self.num_experts),
            DEEPSEEK=self.routing_method_type == ROUTING_DEEPSEEK_V3,
            ROUTE_SCALE=route_scale,
            num_warps=8,
        )

        large_batch = m > 512
        block_m = 128 if large_batch else 16
        sorted_ids, expert_ids, padded_count = self._align(
            topk_ids, block_m, self.local_num_experts,
        )
        max_blocks = triton.cdiv(sorted_ids.numel(), block_m)
        route_rows = m * self.top_k
        gemm1_block_n = 64
        _gemm1_swiglu_kernel[(max_blocks, triton.cdiv(
            self.intermediate_size_per_partition, gemm1_block_n
        ))](
            hidden_states,
            w13,
            sorted_ids,
            expert_ids,
            padded_count,
            intermediate,
            num_route_rows=route_rows,
            hidden_size=hidden_size,
            intermediate_size=self.intermediate_size_per_partition,
            TOP_K=self.top_k,
            BLOCK_M=block_m,
            BLOCK_N=gemm1_block_n,
            BLOCK_K=128,
            num_warps=8 if large_batch else 4,
            num_stages=3,
        )
        gemm2_block_m = 64 if large_batch else block_m
        gemm2_block_n = 128 if large_batch else 64
        gemm2_subblocks = block_m // gemm2_block_m
        _gemm2_kernel[(max_blocks * gemm2_subblocks, triton.cdiv(
            hidden_size, gemm2_block_n
        ))](
            intermediate,
            w2,
            sorted_ids,
            expert_ids,
            padded_count,
            expert_output,
            num_route_rows=route_rows,
            hidden_size=hidden_size,
            intermediate_size=self.intermediate_size_per_partition,
            META_BLOCK_M=block_m,
            BLOCK_M=gemm2_block_m,
            BLOCK_N=gemm2_block_n,
            BLOCK_K=128,
            num_warps=4,
            num_stages=2 if large_batch else 3,
        )
        reduce_block_n = 256
        _reduce_kernel[(m, triton.cdiv(hidden_size, reduce_block_n))](
            expert_output,
            topk_weights,
            output,
            hidden_size=hidden_size,
            TOP_K=self.top_k,
            BLOCK_N=reduce_block_n,
            num_warps=8,
        )
        return output
