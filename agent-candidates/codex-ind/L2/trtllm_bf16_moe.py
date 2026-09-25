"""Self-contained BF16 MoE kernels for the captured TRT-LLM layouts."""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import triton
import triton.language as tl


ROUTING_RENORMALIZE = 1
ROUTING_DEEPSEEK_V3 = 2
ROUTING_RENORMALIZE_NAIVE = 4
ACTIVATION_SWIGLU = 3
DEFAULT_TUNE_MAX_NUM_TOKENS = 16384


def trtllm_bf16_moe_supported() -> bool:
    if os.environ.get("FASTKERNELS_TRTLLM_BF16_MOE", "1") == "0":
        return False
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10


def prepare_trtllm_bf16_moe_weights(
    w13: torch.Tensor,
    w2: torch.Tensor,
    is_gated_act_gemm: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Kept for compatibility with parent MoE modules. Weight preparation is not
    # part of the timed operator, and the checkpoint path calls this only once.
    from fastkernels.tasks.baseline.L2.trtllm_bf16_moe import (
        prepare_trtllm_bf16_moe_weights as prepare,
    )

    return prepare(w13, w2, is_gated_act_gemm)


@triton.jit
def _shuffle_row(row):
    """Logical row to the physical 32-row MMA epilogue permutation."""
    return (row // 32) * 32 + (row % 4) * 8 + (row % 32) // 4


@triton.jit
def _route_kernel(
    logits,
    bias,
    ids,
    weights,
    counts,
    ranks,
    M: tl.constexpr,
    E: tl.constexpr,
    TOPK: tl.constexpr,
    METHOD: tl.constexpr,
    SCALE: tl.constexpr,
    DO_BUCKET: tl.constexpr,
):
    token = tl.program_id(0)
    cols = tl.arange(0, E)
    x = tl.load(logits + token * E + cols).to(tl.float32)

    if METHOD == 2:
        raw = tl.sigmoid(x)
        scores = raw + tl.load(bias + cols).to(tl.float32)
    else:
        # RenormalizeNaive is exactly softmax over the selected logits.
        raw = x
        scores = x

    chosen = tl.full((E,), -1, tl.int32)
    selected = tl.zeros((E,), tl.float32)
    work = scores
    for j in tl.static_range(0, TOPK):
        expert = tl.argmax(work, axis=0)
        chosen = tl.where(cols == j, expert, chosen)
        selected = tl.where(
            cols == j, tl.sum(tl.where(cols == expert, raw, 0.0)), selected
        )
        work = tl.where(cols == expert, -float("inf"), work)

    if METHOD == 2:
        denom = tl.sum(selected)
        selected = selected * (SCALE / denom)
    else:
        selected = tl.where(cols < TOPK, selected, -float("inf"))
        vmax = tl.max(selected)
        selected = tl.exp(selected - vmax)
        selected = selected / tl.sum(selected)

    out_cols = cols < TOPK
    flat = token * TOPK + cols
    tl.store(ids + flat, chosen, mask=out_cols)
    # TRT-LLM stores routing weights in BF16.
    tl.store(weights + flat, selected, mask=out_cols)

    if DO_BUCKET:
        for j in tl.static_range(0, TOPK):
            expert = tl.sum(tl.where(cols == j, chosen, 0))
            rank = tl.atomic_add(counts + expert, 1)
            tl.store(ranks + token * TOPK + j, rank)


@triton.jit
def _clear_kernel(ptr, N: tl.constexpr):
    off = tl.program_id(0) * 256 + tl.arange(0, 256)
    tl.store(ptr + off, 0, mask=off < N)


@triton.jit
def _prefix_kernel(counts, offsets, E: tl.constexpr, BM: tl.constexpr):
    off = tl.arange(0, E)
    count = tl.load(counts + off)
    padded = ((count + BM - 1) // BM) * BM
    inclusive = tl.cumsum(padded)
    tl.store(offsets + off, inclusive - padded)


@triton.jit
def _make_blocks_kernel(
    counts,
    offsets,
    block_experts,
    block_valid,
    E: tl.constexpr,
    BM: tl.constexpr,
):
    block = tl.program_id(0)
    target = block * BM
    experts = tl.arange(0, E)
    starts = tl.load(offsets + experts)
    expert = tl.sum(tl.where(target >= starts, 1, 0)) - 1
    count = tl.load(counts + expert)
    local_block = (target - tl.load(offsets + expert)) // BM
    valid = tl.maximum(0, tl.minimum(BM, count - local_block * BM))
    tl.store(block_experts + block, expert)
    tl.store(block_valid + block, valid)


@triton.jit
def _scatter_kernel(ids, ranks, offsets, sorted_assign, TOTAL: tl.constexpr):
    off = tl.program_id(0) * 256 + tl.arange(0, 256)
    mask = off < TOTAL
    expert = tl.load(ids + off, mask=mask)
    rank = tl.load(ranks + off, mask=mask)
    dest = tl.load(offsets + expert, mask=mask) + rank
    tl.store(sorted_assign + dest, off, mask=mask)


@triton.jit
def _gemm1_direct_kernel(
    hidden,
    w13,
    ids,
    activated,
    TOTAL: tl.constexpr,
    H: tl.constexpr,
    I: tl.constexpr,
    TOPK: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    assign = tl.program_id(0)
    block_n = tl.program_id(1)
    n = block_n * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    token = assign // TOPK
    expert = tl.load(ids + assign).to(tl.int64)

    gate = tl.zeros((1, BN), tl.float32)
    up = tl.zeros((1, BN), tl.float32)
    gate_row = _shuffle_row(2 * n + 1)
    up_row = _shuffle_row(2 * n)
    expert_stride = (H // 64) * (2 * I) * 64
    for kb in range(0, tl.cdiv(H, BK)):
        kk = kb * BK + k
        a = tl.load(hidden + token * H + kk, mask=kk < H, other=0.0)[None, :]
        common = (
            expert * expert_stride
            + (kk[:, None] // 64) * (2 * I * 64)
            + kk[:, None] % 64
        )
        bg = tl.load(
            w13 + common + gate_row[None, :] * 64,
            mask=(kk[:, None] < H) & (n[None, :] < I),
        )
        bu = tl.load(
            w13 + common + up_row[None, :] * 64,
            mask=(kk[:, None] < H) & (n[None, :] < I),
        )
        gate = tl.dot(a, bg, acc=gate)
        up = tl.dot(a, bu, acc=up)
    y = tl.reshape(tl.sigmoid(gate) * gate * up, (BN,))
    tl.store(activated + assign * I + n, y, mask=n < I)


@triton.jit
def _gemm2_direct_kernel(
    activated,
    w2,
    ids,
    expanded,
    TOTAL: tl.constexpr,
    H: tl.constexpr,
    I: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    assign = tl.program_id(0)
    block_n = tl.program_id(1)
    n = block_n * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    expert = tl.load(ids + assign).to(tl.int64)
    acc = tl.zeros((1, BN), tl.float32)
    phys_n = _shuffle_row(n)
    expert_stride = (I // 64) * H * 64
    for kb in range(0, tl.cdiv(I, BK)):
        kk = kb * BK + k
        a = tl.load(
            activated + assign * I + kk, mask=kk < I, other=0.0
        )[None, :]
        common = (
            expert * expert_stride
            + (kk[:, None] // 64) * (H * 64)
            + kk[:, None] % 64
        )
        b = tl.load(
            w2 + common + phys_n[None, :] * 64,
            mask=(kk[:, None] < I) & (n[None, :] < H),
        )
        acc = tl.dot(a, b, acc=acc)
    tl.store(
        expanded + assign * H + n, tl.reshape(acc, (BN,)), mask=n < H
    )


@triton.jit
def _gemm1_grouped_kernel(
    hidden,
    w13,
    block_experts,
    block_valid,
    sorted_assign,
    activated,
    H: tl.constexpr,
    I: tl.constexpr,
    TOPK: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    block_m = tl.program_id(0)
    block_n = tl.program_id(1)
    expert = tl.load(block_experts + block_m).to(tl.int64)
    lm = tl.arange(0, BM)
    n = block_n * BN + tl.arange(0, BN)
    valid_m = lm < tl.load(block_valid + block_m)
    pos = block_m * BM + lm
    assign = tl.load(sorted_assign + pos, mask=valid_m, other=0)
    token = assign // TOPK
    k = tl.arange(0, BK)

    gate = tl.zeros((BM, BN), tl.float32)
    up = tl.zeros((BM, BN), tl.float32)
    gate_row = _shuffle_row(2 * n + 1)
    up_row = _shuffle_row(2 * n)
    expert_stride = (H // 64) * (2 * I) * 64
    for kb in range(0, tl.cdiv(H, BK)):
        kk = kb * BK + k
        a = tl.load(
            hidden + token[:, None] * H + kk[None, :],
            mask=valid_m[:, None] & (kk[None, :] < H),
            other=0.0,
        )
        common = (
            expert * expert_stride
            + (kk[:, None] // 64) * (2 * I * 64)
            + kk[:, None] % 64
        )
        bg = tl.load(
            w13 + common + gate_row[None, :] * 64,
            mask=(kk[:, None] < H) & (n[None, :] < I),
        )
        bu = tl.load(
            w13 + common + up_row[None, :] * 64,
            mask=(kk[:, None] < H) & (n[None, :] < I),
        )
        gate = tl.dot(a, bg, acc=gate)
        up = tl.dot(a, bu, acc=up)
    y = tl.sigmoid(gate) * gate * up
    tl.store(
        activated + assign[:, None] * I + n[None, :],
        y,
        mask=valid_m[:, None] & (n[None, :] < I),
    )


@triton.jit
def _gemm2_grouped_kernel(
    activated,
    w2,
    block_experts,
    block_valid,
    sorted_assign,
    expanded,
    H: tl.constexpr,
    I: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    block_m = tl.program_id(0)
    block_n = tl.program_id(1)
    expert = tl.load(block_experts + block_m).to(tl.int64)
    lm = tl.arange(0, BM)
    n = block_n * BN + tl.arange(0, BN)
    valid_m = lm < tl.load(block_valid + block_m)
    pos = block_m * BM + lm
    assign = tl.load(sorted_assign + pos, mask=valid_m, other=0)
    k = tl.arange(0, BK)

    acc = tl.zeros((BM, BN), tl.float32)
    phys_n = _shuffle_row(n)
    expert_stride = (I // 64) * H * 64
    for kb in range(0, tl.cdiv(I, BK)):
        kk = kb * BK + k
        a = tl.load(
            activated + assign[:, None] * I + kk[None, :],
            mask=valid_m[:, None] & (kk[None, :] < I),
            other=0.0,
        )
        common = (
            expert * expert_stride
            + (kk[:, None] // 64) * (H * 64)
            + kk[:, None] % 64
        )
        b = tl.load(
            w2 + common + phys_n[None, :] * 64,
            mask=(kk[:, None] < I) & (n[None, :] < H),
        )
        acc = tl.dot(a, b, acc=acc)
    tl.store(
        expanded + assign[:, None] * H + n[None, :],
        acc,
        mask=valid_m[:, None] & (n[None, :] < H),
    )


@triton.jit
def _reduce_kernel(
    expanded,
    weights,
    output,
    M: tl.constexpr,
    H: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    n = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), tl.float32)
    for j in tl.static_range(0, TOPK):
        w = tl.load(weights + token * TOPK + j).to(tl.float32)
        x = tl.load(
            expanded + (token * TOPK + j) * H + n, mask=n < H, other=0.0
        )
        acc += x.to(tl.float32) * w
    tl.store(output + token * H + n, acc, mask=n < H)


class TrtLlmBf16MoE(nn.Module):
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
        self._buffers_by_shape: dict[
            tuple[int, int, torch.device], tuple[torch.Tensor, ...]
        ] = {}

    def _get_buffers(
        self, m: int, h: int, device: torch.device
    ) -> tuple[torch.Tensor, ...]:
        key = (m, h, device)
        if key not in self._buffers_by_shape:
            total = m * self.top_k
            max_padded = total + self.num_experts * 127
            max_blocks = triton.cdiv(max_padded, 128)
            self._buffers_by_shape[key] = (
                torch.empty((m, self.top_k), dtype=torch.int32, device=device),
                torch.empty((m, self.top_k), dtype=torch.bfloat16, device=device),
                torch.empty(self.num_experts, dtype=torch.int32, device=device),
                torch.empty(total, dtype=torch.int32, device=device),
                torch.empty(self.num_experts, dtype=torch.int32, device=device),
                torch.empty(max_padded, dtype=torch.int32, device=device),
                torch.empty(max_blocks, dtype=torch.int32, device=device),
                torch.empty(max_blocks, dtype=torch.int32, device=device),
                torch.empty(
                    (total, self.intermediate_size_per_partition),
                    dtype=torch.bfloat16,
                    device=device,
                ),
                torch.empty((total, h), dtype=torch.bfloat16, device=device),
                torch.empty((m, h), dtype=torch.bfloat16, device=device),
            )
        return self._buffers_by_shape[key]

    def forward(
        self,
        hidden_states: torch.Tensor,
        w13: torch.Tensor,
        w2: torch.Tensor,
        router_logits: torch.Tensor,
        routing_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        m, h = hidden_states.shape
        i = self.intermediate_size_per_partition
        (
            ids,
            weights,
            counts,
            ranks,
            offsets,
            sorted_assign,
            block_experts,
            block_valid,
            activated,
            expanded,
            output,
        ) = self._get_buffers(m, h, hidden_states.device)
        grouped = m > 64
        if grouped:
            _clear_kernel[(triton.cdiv(self.num_experts, 256),)](
                counts, N=self.num_experts, num_warps=4
            )
        scale = (
            1.0
            if self.routed_scaling_factor is None
            else self.routed_scaling_factor
        )
        _route_kernel[(m,)](
            router_logits,
            routing_bias,
            ids,
            weights,
            counts,
            ranks,
            M=m,
            E=self.num_experts,
            TOPK=self.top_k,
            METHOD=self.routing_method_type,
            SCALE=scale,
            DO_BUCKET=grouped,
            num_warps=4,
        )

        total = m * self.top_k
        if grouped:
            max_blocks = triton.cdiv(total + self.num_experts * 127, 128)
            _prefix_kernel[(1,)](
                counts, offsets, E=self.num_experts, BM=128, num_warps=8
            )
            _scatter_kernel[(triton.cdiv(total, 256),)](
                ids,
                ranks,
                offsets,
                sorted_assign,
                TOTAL=total,
                num_warps=4,
            )
            _make_blocks_kernel[(max_blocks,)](
                counts,
                offsets,
                block_experts,
                block_valid,
                E=self.num_experts,
                BM=128,
                num_warps=8,
            )
            bm1, bn1 = 128, 64
            _gemm1_grouped_kernel[
                (
                    max_blocks,
                    triton.cdiv(i, bn1),
                )
            ](
                hidden_states,
                w13,
                block_experts,
                block_valid,
                sorted_assign,
                activated,
                H=h,
                I=i,
                TOPK=self.top_k,
                BM=bm1,
                BN=bn1,
                BK=64,
                num_warps=4,
                num_stages=3,
            )
            bm2, bn2 = 128, 256
            _gemm2_grouped_kernel[
                (
                    max_blocks,
                    triton.cdiv(h, bn2),
                )
            ](
                activated,
                w2,
                block_experts,
                block_valid,
                sorted_assign,
                expanded,
                H=h,
                I=i,
                BM=bm2,
                BN=bn2,
                BK=64,
                num_warps=4,
                num_stages=2,
            )
        else:
            _gemm1_direct_kernel[(total, triton.cdiv(i, 64))](
                hidden_states,
                w13,
                ids,
                activated,
                TOTAL=total,
                H=h,
                I=i,
                TOPK=self.top_k,
                BN=64,
                BK=64,
                num_warps=4,
                num_stages=3,
            )
            _gemm2_direct_kernel[(total, triton.cdiv(h, 128))](
                activated,
                w2,
                ids,
                expanded,
                TOTAL=total,
                H=h,
                I=i,
                BN=128,
                BK=64,
                num_warps=4,
                num_stages=3,
            )
        _reduce_kernel[(m, triton.cdiv(h, 1024))](
            expanded,
            weights,
            output,
            M=m,
            H=h,
            TOPK=self.top_k,
            BLOCK=1024,
            num_warps=8,
        )
        return output
