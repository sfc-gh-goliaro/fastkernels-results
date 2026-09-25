"""Fused CLIP self-attention for the fixed 77-token text encoder."""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl
from transformers import CLIPTextConfig

from ..L1.linear import BMM, Linear
from ..L1.softmax import Softmax


@triton.jit
def _to_tf32(x):
    bits = x.to(tl.uint32, bitcast=True)
    bits += 0xFFF + ((bits >> 13) & 1)
    bits &= 0xFFFFE000
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def _linear_dot_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    y_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    PRECISION: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        x = tl.load(
            x_ptr + offs_m[:, None] * K + k + offs_k[None, :],
            mask=(offs_m[:, None] < M) & (k + offs_k[None, :] < K),
            other=0.0,
        )
        w = tl.load(
            w_ptr + offs_n[None, :] * K + k + offs_k[:, None],
            mask=(offs_n[None, :] < N) & (k + offs_k[:, None] < K),
            other=0.0,
        )
        acc += tl.dot(_to_tf32(x), _to_tf32(w), input_precision=PRECISION)
    acc += tl.load(bias_ptr + offs_n[None, :], mask=offs_n[None, :] < N)
    tl.store(
        y_ptr + offs_m[:, None] * N + offs_n[None, :],
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.jit
def _attention_kernel(
    qkv_ptr,
    mask_ptr,
    out_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    head = tl.program_id(0)
    query_block = tl.program_id(1)
    offs_m = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

    q = tl.load(
        qkv_ptr + offs_m[:, None] * 2304 + head * HEAD_DIM + offs_d[None, :],
        mask=offs_m[:, None] < 77,
        other=0.0,
    )
    offs_n = tl.arange(0, BLOCK_N)
    k = tl.load(
        qkv_ptr
        + offs_n[None, :] * 2304
        + 768
        + head * HEAD_DIM
        + offs_d[:, None],
        mask=offs_n[None, :] < 77,
        other=0.0,
    )
    scores = tl.dot(_to_tf32(q), _to_tf32(k), input_precision="tf32") * 0.125
    scores += tl.load(
        mask_ptr + offs_m[:, None] * 77 + offs_n[None, :],
        mask=(offs_m[:, None] < 77) & (offs_n[None, :] < 77),
        other=-float("inf"),
    )
    scores -= tl.max(scores, axis=1)[:, None]
    p = tl.exp(scores)
    p /= tl.sum(p, axis=1)[:, None]

    v = tl.load(
        qkv_ptr
        + offs_n[:, None] * 2304
        + 1536
        + head * HEAD_DIM
        + offs_d[None, :],
        mask=offs_n[:, None] < 77,
        other=0.0,
    )
    acc = tl.dot(_to_tf32(p), _to_tf32(v), input_precision="tf32")
    tl.store(
        out_ptr + offs_m[:, None] * 768 + head * HEAD_DIM + offs_d[None, :],
        acc,
        mask=offs_m[:, None] < 77,
    )


class CLIPAttention(nn.Module):
    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = Linear(self.embed_dim, self.embed_dim, bias=True)
        self.k_proj = Linear(self.embed_dim, self.embed_dim, bias=True)
        self.v_proj = Linear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = Linear(self.embed_dim, self.embed_dim, bias=True)

        self.bmm = BMM()
        self.softmax = Softmax(dim=-1)

        self.register_buffer(
            "_qkv_weight", torch.empty(3 * self.embed_dim, self.embed_dim), persistent=False
        )
        self.register_buffer("_qkv_bias", torch.empty(3 * self.embed_dim), persistent=False)
        self.register_buffer("_qkv", torch.empty(77, 3 * self.embed_dim), persistent=False)
        self.register_buffer("_attn", torch.empty(77, self.embed_dim), persistent=False)
        self.register_buffer("_output", torch.empty(77, self.embed_dim), persistent=False)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        result = super().load_state_dict(state_dict, strict=False, assign=assign)
        with torch.no_grad():
            self._qkv_weight[:768].copy_(self.q_proj.weight)
            self._qkv_weight[768:1536].copy_(self.k_proj.weight)
            self._qkv_weight[1536:].copy_(self.v_proj.weight)
            self._qkv_bias[:768].copy_(self.q_proj.bias)
            self._qkv_bias[768:1536].copy_(self.k_proj.bias)
            self._qkv_bias[1536:].copy_(self.v_proj.bias)
        return result

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (
            hidden_states.is_cuda
            and hidden_states.dtype == torch.float32
            and hidden_states.shape == (1, 77, 768)
            and attention_mask is not None
            and attention_mask.shape == (1, 1, 77, 77)
        ):
            _linear_dot_kernel[(triton.cdiv(77, 32), triton.cdiv(2304, 32))](
                hidden_states,
                self._qkv_weight,
                self._qkv_bias,
                self._qkv,
                M=77,
                N=2304,
                K=768,
                BLOCK_M=32,
                BLOCK_N=32,
                BLOCK_K=64,
                PRECISION="tf32",
                num_warps=4,
                num_stages=4,
            )
            _attention_kernel[(12, triton.cdiv(77, 8))](
                self._qkv,
                attention_mask,
                self._attn,
                BLOCK_M=8,
                BLOCK_N=128,
                HEAD_DIM=64,
                num_warps=4,
                num_stages=2,
            )
            _linear_dot_kernel[(triton.cdiv(77, 16), triton.cdiv(768, 64))](
                self._attn,
                self.out_proj.weight,
                self.out_proj.bias,
                self._output,
                M=77,
                N=768,
                K=768,
                BLOCK_M=16,
                BLOCK_N=64,
                BLOCK_K=64,
                PRECISION="tf32",
                num_warps=4,
                num_stages=3,
            )
            return self._output.view(1, 77, 768)

        batch_size, seq_length, _ = hidden_states.shape
        queries = self.q_proj(hidden_states)
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)
        queries = queries.view(
            batch_size, seq_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        keys = keys.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(
            batch_size, seq_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        attn_weights = self.bmm(queries, keys.transpose(-1, -2)) * self.scale
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = self.softmax(attn_weights.float()).to(queries.dtype)
        attn_output = self.bmm(attn_weights, values)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_length, self.embed_dim)
        return self.out_proj(attn_output)
