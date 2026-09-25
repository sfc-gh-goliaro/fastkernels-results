"""T5 self-attention with TP-aware QKV projection and relative position bias (L2).

Mirrors vllm-omni's T5SelfAttention: QKVParallelLinear -> manual SDPA ->
RowParallelLinear, with T5-style relative position bias computed per-partition.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import triton
import triton.language as tl
from transformers import T5Config

from ....infra.tp import _tp_size, _tp_rank
from ..L1.embedding import Embedding
from ..L1.linear import BMM
from ..L1.softmax import Softmax
from .parallel_linear import QKVParallelLinear, RowParallelLinear


@triton.jit
def _t5_attention_kernel(
    scores_ptr,
    v_ptr,
    bias_ptr,
    out_ptr,
    stride_vm: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_bh: tl.constexpr,
    stride_bm: tl.constexpr,
    stride_bn: tl.constexpr,
    N_CTX: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    start_m = tl.program_id(0) * BLOCK_M
    head = tl.program_id(1)
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    row_max = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    row_sum = tl.zeros((BLOCK_M,), tl.float32)
    for start_n in range(0, N_CTX, BLOCK_N):
        scores = tl.load(
            scores_ptr
            + head * N_CTX * N_CTX
            + offs_m[:, None] * N_CTX
            + (start_n + offs_n)[None, :],
        )
        if HAS_BIAS:
            bias = tl.load(
                bias_ptr
                + head * stride_bh
                + offs_m[:, None] * stride_bm
                + (start_n + offs_n)[None, :] * stride_bn,
            )
            scores = (scores + bias).to(tl.bfloat16)
            tl.store(
                scores_ptr
                + head * N_CTX * N_CTX
                + offs_m[:, None] * N_CTX
                + (start_n + offs_n)[None, :],
                scores,
            )
        scores = scores.to(tl.float32)
        block_max = tl.maximum(row_max, tl.max(scores, axis=1))
        alpha = tl.exp(row_max - block_max)
        probs = tl.exp(scores - block_max[:, None])
        row_sum = row_sum * alpha + tl.sum(probs, axis=1)
        row_max = block_max

    acc = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)
    for start_n in range(0, N_CTX, BLOCK_N):
        scores = tl.load(
            scores_ptr
            + head * N_CTX * N_CTX
            + offs_m[:, None] * N_CTX
            + (start_n + offs_n)[None, :],
        )
        probs = (
            tl.exp(scores.to(tl.float32) - row_max[:, None])
            / row_sum[:, None]
        ).to(tl.bfloat16)
        v = tl.load(
            v_ptr
            + head * stride_vh
            + (start_n + offs_n)[:, None] * stride_vm
            + offs_d[None, :],
        )
        acc += tl.dot(probs, v)

    tl.store(
        out_ptr + offs_m[:, None] * (stride_vh * 64) + head * stride_vh
        + offs_d[None, :],
        acc.to(tl.bfloat16),
    )


@triton.jit
def _relative_bias_kernel(
    weight_ptr,
    out_ptr,
    N_CTX: tl.constexpr,
    N_HEADS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    total = N_HEADS * N_CTX * N_CTX
    mask = offsets < total
    key_pos = offsets % N_CTX
    query_pos = (offsets // N_CTX) % N_CTX
    head = offsets // (N_CTX * N_CTX)
    relative = key_pos - query_pos
    distance = tl.abs(relative)
    large_bucket = 8 + (
        tl.log(tl.maximum(distance, 8).to(tl.float32) / 8.0)
        * 2.8853900817779268
    ).to(tl.int32)
    large_bucket = tl.minimum(large_bucket, 15)
    bucket = tl.where(distance < 8, distance, large_bucket)
    bucket += tl.where(relative > 0, 16, 0)
    values = tl.load(weight_ptr + bucket * N_HEADS + head, mask=mask)
    tl.store(out_ptr + offsets, values, mask=mask)


class T5SelfAttention(nn.Module):
    def __init__(self, config: T5Config, has_relative_attention_bias: bool = False):
        super().__init__()
        self.d_model = config.d_model
        self.d_kv = config.d_kv
        self.n_heads = config.num_heads
        self.inner_dim = self.n_heads * self.d_kv
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance

        tp_size = _tp_size()
        assert self.n_heads % tp_size == 0
        self.n_heads_per_partition = self.n_heads // tp_size

        self.qkv_proj = QKVParallelLinear(
            hidden_size=self.d_model,
            head_size=self.d_kv,
            total_num_heads=self.n_heads,
            total_num_kv_heads=self.n_heads,
            bias=False,
        )

        self.o = RowParallelLinear(self.inner_dim, self.d_model, bias=False)

        self.bmm = BMM()
        self.softmax = Softmax(dim=-1)

        if has_relative_attention_bias:
            self.relative_attention_bias = Embedding(
                self.relative_attention_num_buckets, self.n_heads,
            )

    @staticmethod
    def _relative_position_bucket(
        relative_position: torch.Tensor,
        bidirectional: bool = True,
        num_buckets: int = 32,
        max_distance: int = 128,
    ) -> torch.Tensor:
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(
                relative_position, torch.zeros_like(relative_position),
            )
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large,
            torch.full_like(relative_position_if_large, num_buckets - 1),
        )
        relative_buckets += torch.where(
            is_small, relative_position, relative_position_if_large,
        )
        return relative_buckets

    def compute_bias(self, query_length: int, key_length: int, device: torch.device) -> torch.Tensor:
        if (
            query_length == 512
            and key_length == 512
            and self.n_heads_per_partition == 64
            and self.relative_attention_num_buckets == 32
            and self.relative_attention_max_distance == 128
            and _tp_size() == 1
        ):
            values = torch.empty(
                (1, 64, 512, 512),
                device=device,
                dtype=self.relative_attention_bias.emb.weight.dtype,
            )
            _relative_bias_kernel[(triton.cdiv(values.numel(), 2048),)](
                self.relative_attention_bias.emb.weight,
                values,
                N_CTX=512,
                N_HEADS=64,
                BLOCK=2048,
                num_warps=8,
            )
            return values

        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position
        relative_position_bucket = self._relative_position_bucket(
            relative_position, bidirectional=True,
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(relative_position_bucket)
        tp_rank = _tp_rank()
        head_start = tp_rank * self.n_heads_per_partition
        head_end = head_start + self.n_heads_per_partition
        values = values[:, :, head_start:head_end]
        values = values.permute(2, 0, 1).unsqueeze(0)
        return values

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor | None = None,
        position_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_length = hidden_states.shape[:2]

        qkv = self.qkv_proj(hidden_states)
        q_size = self.n_heads_per_partition * self.d_kv
        kv_size = self.n_heads_per_partition * self.d_kv
        query_states, key_states, value_states = qkv.split(
            [q_size, kv_size, kv_size], dim=-1,
        )

        query_states = query_states.view(
            batch_size, seq_length, self.n_heads_per_partition, self.d_kv,
        ).transpose(1, 2)
        key_states = key_states.view(
            batch_size, seq_length, self.n_heads_per_partition, self.d_kv,
        ).transpose(1, 2)
        value_states = value_states.view(
            batch_size, seq_length, self.n_heads_per_partition, self.d_kv,
        ).transpose(1, 2)

        if (
            batch_size == 1
            and seq_length == 512
            and self.n_heads_per_partition == 64
            and self.d_kv == 64
            and hidden_states.dtype == torch.bfloat16
            and mask is None
        ):
            computed_position_bias = position_bias is None
            if computed_position_bias:
                if self.has_relative_attention_bias:
                    position_bias = self.compute_bias(
                        seq_length, seq_length, device=hidden_states.device,
                    )
                else:
                    position_bias = torch.zeros(
                        (1, 64, 512, 512),
                        device=hidden_states.device,
                        dtype=hidden_states.dtype,
                    )
            scores = self.bmm(query_states, key_states.transpose(3, 2))
            if not computed_position_bias:
                scores += position_bias
            attn_output = torch.empty_like(hidden_states)
            _t5_attention_kernel[(8, 64)](
                scores,
                value_states,
                position_bias,
                attn_output,
                stride_vm=value_states.stride(2),
                stride_vh=value_states.stride(1),
                stride_bh=position_bias.stride(1),
                stride_bm=position_bias.stride(2),
                stride_bn=position_bias.stride(3),
                N_CTX=512,
                HEAD_DIM=64,
                BLOCK_M=64,
                BLOCK_N=64,
                HAS_BIAS=computed_position_bias,
                num_warps=8,
                num_stages=3,
            )
            return self.o(attn_output), position_bias

        scores = self.bmm(query_states, key_states.transpose(3, 2))

        if position_bias is None:
            if self.has_relative_attention_bias:
                position_bias = self.compute_bias(
                    seq_length, seq_length, device=scores.device,
                )
            else:
                position_bias = torch.zeros(
                    (1, self.n_heads_per_partition, seq_length, seq_length),
                    device=scores.device, dtype=scores.dtype,
                )
            if mask is not None:
                position_bias = position_bias + mask

        scores += position_bias
        attn_weights = self.softmax(scores.float()).type_as(scores)
        attn_output = self.bmm(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_length, -1)
        attn_output = self.o(attn_output)

        return attn_output, position_bias
