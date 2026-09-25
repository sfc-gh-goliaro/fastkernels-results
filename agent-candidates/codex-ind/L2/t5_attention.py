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
def _attention_512(
    q_ptr,
    k_ptr,
    v_ptr,
    bias_ptr,
    out_ptr,
    q_stride_h: tl.constexpr,
    q_stride_s: tl.constexpr,
    bias_stride_h: tl.constexpr,
    bias_stride_q: tl.constexpr,
    bias_stride_k: tl.constexpr,
    out_stride_s: tl.constexpr,
    out_stride_h: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    block_m = tl.program_id(0)
    head = tl.program_id(1)
    offs_m = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, 64)

    q = tl.load(
        q_ptr + head * q_stride_h
        + offs_m[:, None] * q_stride_s + offs_d[None, :]
    )
    acc = tl.zeros((BLOCK_M, 64), tl.float32)
    row_max = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    row_sum = tl.zeros((BLOCK_M,), tl.float32)

    for start_n in tl.range(0, 512, BLOCK_N):
        cols = start_n + offs_n
        k = tl.load(
            k_ptr + head * q_stride_h
            + cols[None, :] * q_stride_s + offs_d[:, None]
        )
        scores = tl.dot(q, k).to(tl.bfloat16)
        bias = tl.load(
            bias_ptr + head * bias_stride_h
            + offs_m[:, None] * bias_stride_q
            + cols[None, :] * bias_stride_k
        )
        scores = (scores + bias).to(tl.bfloat16)

        block_max = tl.max(scores, axis=1)
        new_max = tl.maximum(row_max, block_max)
        alpha = tl.exp2((row_max - new_max) * 1.4426950408889634)
        probs = tl.exp2((scores - new_max[:, None]) * 1.4426950408889634)
        row_sum = row_sum * alpha + tl.sum(probs, axis=1)
        acc *= alpha[:, None]

        v = tl.load(
            v_ptr + head * q_stride_h
            + cols[:, None] * q_stride_s + offs_d[None, :]
        )
        acc += tl.dot(probs.to(tl.bfloat16), v)
        row_max = new_max

    out = acc / row_sum[:, None]
    tl.store(
        out_ptr + offs_m[:, None] * out_stride_s
        + head * out_stride_h + offs_d[None, :],
        out,
    )


@triton.jit
def _relative_bias_512(weight_ptr, out_ptr, BLOCK: tl.constexpr):
    position = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    head = tl.program_id(1)
    valid = position < 512 * 512
    key = position % 512
    query = position // 512
    relative = key - query
    distance = tl.abs(relative)

    bucket = tl.minimum(distance, 7)
    large_bucket = (
        8
        + (distance >= 12)
        + (distance >= 16)
        + (distance >= 23)
        + (distance >= 32)
        + (distance >= 46)
        + (distance >= 64)
        + (distance >= 91)
    )
    bucket = tl.where(distance < 8, bucket, large_bucket)
    bucket += tl.where(relative > 0, 16, 0)
    value = tl.load(weight_ptr + bucket * 64 + head, mask=valid)
    tl.store(
        out_ptr + head * (512 * 512) + position,
        value,
        mask=valid,
    )


@triton.jit
def _transpose_bias_512(in_ptr, out_ptr, BLOCK_M: tl.constexpr):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    heads = tl.arange(0, 64)
    tile = tl.load(in_ptr + rows[:, None] * 64 + heads[None, :])
    tl.store(
        out_ptr + heads[:, None] * (512 * 512) + rows[None, :],
        tile.trans(),
    )


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
            and self.n_heads == 64
            and self.n_heads_per_partition == 64
            and self.relative_attention_num_buckets == 32
            and self.relative_attention_max_distance == 128
        ):
            values = torch.empty(
                (1, self.n_heads, query_length, key_length),
                device=device,
                dtype=self.relative_attention_bias.emb.weight.dtype,
            )
            _relative_bias_512[(256, 64)](
                self.relative_attention_bias.emb.weight,
                values,
                BLOCK=1024,
                num_warps=4,
            )
            return values
        else:
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

        bias_stream = None
        if (
            position_bias is None
            and mask is None
            and self.has_relative_attention_bias
            and batch_size == 1
            and seq_length == 512
            and self.n_heads == 64
            and self.n_heads_per_partition == 64
            and hidden_states.is_cuda
        ):
            if not hasattr(self, "_bias_stream"):
                self._bias_stream = torch.cuda.Stream(device=hidden_states.device)
            bias_stream = self._bias_stream
            bias_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(bias_stream):
                position_bias = self.compute_bias(
                    seq_length, seq_length, device=hidden_states.device,
                )

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

        if bias_stream is not None:
            torch.cuda.current_stream().wait_stream(bias_stream)

        if position_bias is None:
            if self.has_relative_attention_bias:
                position_bias = self.compute_bias(
                    seq_length, seq_length, device=hidden_states.device,
                )
            else:
                position_bias = torch.zeros(
                    (1, self.n_heads_per_partition, seq_length, seq_length),
                    device=hidden_states.device, dtype=hidden_states.dtype,
                )
            if mask is not None:
                position_bias = position_bias + mask

        attention_bias = position_bias
        if (
            batch_size == 1
            and seq_length == 512
            and self.n_heads_per_partition == 64
            and self.d_kv == 64
            and hidden_states.dtype == torch.bfloat16
            and position_bias.dtype == torch.bfloat16
            and hidden_states.is_cuda
        ):
            if position_bias.stride() == (64, 1, 32768, 64):
                attention_bias = torch.empty(
                    position_bias.shape,
                    device=position_bias.device,
                    dtype=position_bias.dtype,
                )
                _transpose_bias_512[(1024,)](
                    position_bias,
                    attention_bias,
                    BLOCK_M=256,
                    num_warps=8,
                )
            attn_output = torch.empty(
                (batch_size, seq_length, self.n_heads_per_partition, self.d_kv),
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            _attention_512[(4, 64)](
                query_states,
                key_states,
                value_states,
                attention_bias,
                attn_output,
                query_states.stride(1),
                query_states.stride(2),
                attention_bias.stride(1),
                attention_bias.stride(2),
                attention_bias.stride(3),
                attn_output.stride(1),
                attn_output.stride(2),
                BLOCK_M=128,
                BLOCK_N=64,
                num_warps=4,
                num_stages=1,
            )
            attn_output = attn_output.view(batch_size, seq_length, -1)
        else:
            scores = self.bmm(query_states, key_states.transpose(3, 2))
            scores += position_bias
            attn_weights = self.softmax(scores.float()).type_as(scores)
            attn_output = self.bmm(attn_weights, value_states)
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.view(batch_size, seq_length, -1)
        attn_output = self.o(attn_output)

        return attn_output, position_bias
