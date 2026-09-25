"""Oasis spatial axial attention."""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

from ..L1.linear import Linear
from ..L1.oasis_rotary import OasisRotaryEmbedding


@triton.jit
def _matmul_kernel(
    a_ptr,
    b_ptr,
    bias_ptr,
    cos_sin_ptr,
    output_ptr,
    m_size: tl.constexpr,
    n_size: tl.constexpr,
    k_size: tl.constexpr,
    has_bias: tl.constexpr,
    apply_rotary: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(m_size, block_m)
    pid_m = pid % grid_m
    pid_n = pid // grid_m

    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_n = pid_n * block_n + tl.arange(0, block_n)
    offs_k = tl.arange(0, block_k)
    a_ptrs = a_ptr + offs_m[:, None] * k_size + offs_k[None, :]
    b_ptrs = b_ptr + offs_k[:, None] * n_size + offs_n[None, :]

    acc = tl.zeros((block_m, block_n), tl.float32)
    for _ in range(0, k_size, block_k):
        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < m_size) & (offs_k[None, :] < k_size),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_k[:, None] < k_size) & (offs_n[None, :] < n_size),
            other=0.0,
        )
        acc = tl.dot(a, b, acc)
        a_ptrs += block_k
        b_ptrs += block_k * n_size

    if has_bias:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < n_size, other=0.0)
        acc += bias[None, :]
    if apply_rotary and pid_n * block_n < 2048:
        pairs = tl.reshape(acc, (block_m, block_n // 2, 2))
        even, odd = tl.split(pairs)
        offs_pairs = pid_n * block_n + 2 * tl.arange(0, block_n // 2)
        cs = ((offs_m[:, None] % 144) * 64 + (offs_pairs[None, :] % 64)) * 2
        cs_mask = (offs_m[:, None] < m_size) & (offs_pairs[None, :] < 2048)
        cos = tl.load(cos_sin_ptr + cs, mask=cs_mask, other=0.0)
        sin = tl.load(cos_sin_ptr + cs + 1, mask=cs_mask, other=0.0)
        rotated = tl.reshape(
            tl.join(even * cos - odd * sin, odd * cos + even * sin),
            (block_m, block_n),
        )
        acc = rotated
    output_ptrs = output_ptr + offs_m[:, None] * n_size + offs_n[None, :]
    tl.store(
        output_ptrs,
        acc,
        mask=(offs_m[:, None] < m_size) & (offs_n[None, :] < n_size),
    )


def _matmul(
    x: torch.Tensor,
    weight_t: torch.Tensor,
    bias: torch.Tensor | None,
    cos_sin: torch.Tensor | None,
    m_size: int,
    n_size: int,
) -> torch.Tensor:
    output = torch.empty((m_size, n_size), device=x.device, dtype=x.dtype)
    block_m = 128 if cos_sin is not None and m_size == 864 else 64
    block_n = 128
    grid = (triton.cdiv(m_size, block_m) * triton.cdiv(n_size, block_n),)
    _matmul_kernel[grid](
        x,
        weight_t,
        bias,
        cos_sin,
        output,
        m_size,
        n_size,
        1024,
        has_bias=bias is not None,
        apply_rotary=cos_sin is not None,
        block_m=block_m,
        block_n=block_n,
        block_k=64,
        num_warps=8,
        num_stages=4,
    )
    return output


@triton.jit
def _attention_kernel(
    qkv_ptr,
    output_ptr,
    tokens: tl.constexpr,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
):
    query_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch = batch_head // heads
    head = batch_head % heads

    offs_m = query_block * block_m + tl.arange(0, block_m)
    offs_d = tl.arange(0, head_dim)
    q_rows = batch * tokens + offs_m
    q_cols = head * head_dim + offs_d
    q_ptrs = qkv_ptr + q_rows[:, None] * (3 * heads * head_dim) + q_cols[None, :]
    q_mask = offs_m[:, None] < tokens
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    m_i = tl.full((block_m,), -float("inf"), tl.float32)
    l_i = tl.full((block_m,), 1.0, tl.float32)
    acc = tl.zeros((block_m, head_dim), tl.float32)
    for start_n in range(0, tokens, block_n):
        offs_n = start_n + tl.arange(0, block_n)
        k_rows = batch * tokens + offs_n
        k_cols = heads * head_dim + head * head_dim + offs_d
        k_ptrs = qkv_ptr + k_rows[None, :] * (3 * heads * head_dim) + k_cols[:, None]
        k_mask = offs_n[None, :] < tokens
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)

        scores = tl.dot(q, k) * (0.125 * 1.4426950408889634)
        scores = tl.where(offs_n[None, :] < tokens, scores, -float("inf"))
        scores = tl.where(offs_m[:, None] < tokens, scores, -float("inf"))
        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        p = tl.math.exp2(scores - m_ij[:, None])
        alpha = tl.math.exp2(m_i - m_ij)
        l_ij = tl.sum(p, axis=1)

        v_cols = 2 * heads * head_dim + head * head_dim + offs_d
        v_ptrs = qkv_ptr + k_rows[:, None] * (3 * heads * head_dim) + v_cols[None, :]
        v = tl.load(v_ptrs, mask=(offs_n[:, None] < tokens), other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(q.dtype), v)
        l_i = l_i * alpha + l_ij
        m_i = m_ij

    acc = acc / l_i[:, None]
    out_cols = head * head_dim + offs_d
    out_ptrs = output_ptr + q_rows[:, None] * (heads * head_dim) + out_cols[None, :]
    tl.store(out_ptrs, acc, mask=q_mask)


def _attention(qkv: torch.Tensor, batches: int) -> torch.Tensor:
    output = torch.empty(
        (batches * 144, 1024), device=qkv.device, dtype=qkv.dtype
    )
    _attention_kernel[(3, batches * 16)](
        qkv,
        output,
        tokens=144,
        heads=16,
        head_dim=64,
        block_m=64,
        block_n=64,
        num_warps=4,
        num_stages=2,
    )
    return output


class OasisSpatialAxialAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        rotary_emb: OasisRotaryEmbedding,
    ):
        super().__init__()
        self.heads = heads
        self.to_qkv = Linear(dim, dim_head * heads * 3, bias=False)
        self.to_out = Linear(dim_head * heads, dim, bias=True)
        self.rotary_emb = rotary_emb
        self._rotary_cache = None
        self._qkv_weight_t = None
        self._out_weight_t = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, time, height, width, _ = x.shape
        rows = bsz * time * height * width
        if (
            self._rotary_cache is None
            or self._rotary_cache.device != x.device
            or self._rotary_cache.dtype != x.dtype
        ):
            freqs = self.rotary_emb.get_axial_freqs(height, width)
            self._rotary_cache = torch.stack((freqs.cos(), freqs.sin()), dim=-1).to(x.dtype).contiguous()

        if (
            self._qkv_weight_t is None
            or self._qkv_weight_t.device != x.device
            or self._qkv_weight_t.dtype != x.dtype
        ):
            self._qkv_weight_t = self.to_qkv.weight.t().contiguous()
            self._out_weight_t = self.to_out.weight.t().contiguous()

        qkv = _matmul(x, self._qkv_weight_t, None, self._rotary_cache, rows, 3072)
        out = _attention(qkv, bsz * time)
        out = _matmul(out, self._out_weight_t, self.to_out.bias, None, rows, 1024)
        return out.view(bsz, time, height, width, -1)
