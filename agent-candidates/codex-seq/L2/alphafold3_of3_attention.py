"""Multi-head attention with bias list support for AlphaFold3 (L2).

Composes QKV projections + SDPA + gated output.

Reference: openfold3/core/model/primitives/attention.py Attention
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

from ..L1.linear import Linear
from ..L1.softmax import Softmax


@triton.jit
def _qkvg_matmul_kernel(
    qx_ptr,
    kvx_ptr,
    wq_ptr,
    wk_ptr,
    wv_ptr,
    wg_ptr,
    bias_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    g_ptr,
    MQ: tl.constexpr,
    MKV: tl.constexpr,
    N: tl.constexpr,
    REDUCTION: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    Q_BLOCKS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    raw_pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    is_kv = raw_pid_m >= Q_BLOCKS
    pid_m = tl.where(is_kv, raw_pid_m - Q_BLOCKS, raw_pid_m)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    current_m = tl.where(is_kv, MKV, MQ)

    x_base = tl.where(is_kv, kvx_ptr, qx_ptr)
    w1_base = tl.where(is_kv, wk_ptr, wq_ptr)
    w2_base = tl.where(is_kv, wv_ptr, wg_ptr)
    y1_base = tl.where(is_kv, k_ptr, q_ptr)
    y2_base = tl.where(is_kv, v_ptr, g_ptr)
    x_ptrs = x_base + offs_m[:, None] * REDUCTION + offs_k[None, :]
    w1_ptrs = w1_base + offs_n[None, :] * REDUCTION + offs_k[:, None]
    w2_ptrs = w2_base + offs_n[None, :] * REDUCTION + offs_k[:, None]
    acc1 = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    acc2 = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k_start in range(0, REDUCTION, BLOCK_K):
        x = tl.load(
            x_ptrs,
            mask=(offs_m[:, None] < current_m)
            & (k_start + offs_k[None, :] < REDUCTION),
            other=0.0,
        )
        w1 = tl.load(
            w1_ptrs,
            mask=(k_start + offs_k[:, None] < REDUCTION)
            & (offs_n[None, :] < N),
            other=0.0,
        )
        w2 = tl.load(
            w2_ptrs,
            mask=(k_start + offs_k[:, None] < REDUCTION)
            & (offs_n[None, :] < N),
            other=0.0,
        )
        acc1 += tl.dot(x, w1)
        acc2 += tl.dot(x, w2)
        x_ptrs += BLOCK_K
        w1_ptrs += BLOCK_K
        w2_ptrs += BLOCK_K

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc1 += tl.where(is_kv, 0.0, bias[None, :])
    mask = (offs_m[:, None] < current_m) & (offs_n[None, :] < N)
    tl.store(y1_base + offs_m[:, None] * N + offs_n[None, :], acc1, mask=mask)
    tl.store(y2_base + offs_m[:, None] * N + offs_n[None, :], acc2, mask=mask)


def _project_qkvg(
    q_x: torch.Tensor,
    kv_x: torch.Tensor,
    linear_q: Linear,
    linear_k: Linear,
    linear_v: Linear,
    linear_g: Linear,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mq = q_x.numel() // q_x.shape[-1]
    mkv = kv_x.numel() // kv_x.shape[-1]
    reduction = q_x.shape[-1]
    n = linear_q.weight.shape[0]
    q = torch.empty((*q_x.shape[:-1], n), device=q_x.device, dtype=q_x.dtype)
    k = torch.empty((*kv_x.shape[:-1], n), device=kv_x.device, dtype=kv_x.dtype)
    v = torch.empty_like(k)
    g = torch.empty_like(q)
    bias = linear_q.bias if linear_q.bias is not None else linear_q.weight
    block_m = 16 if max(mq, mkv) == 16 else 64
    block_n = 64
    q_blocks = triton.cdiv(mq, block_m)
    kv_blocks = triton.cdiv(mkv, block_m)
    _qkvg_matmul_kernel[(q_blocks + kv_blocks, triton.cdiv(n, block_n))](
        q_x,
        kv_x,
        linear_q.weight,
        linear_k.weight,
        linear_v.weight,
        linear_g.weight,
        bias,
        q,
        k,
        v,
        g,
        MQ=mq,
        MKV=mkv,
        N=n,
        REDUCTION=reduction,
        HAS_BIAS=linear_q.bias is not None,
        Q_BLOCKS=q_blocks,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=32,
        num_warps=4,
        num_stages=3,
    )
    return q, k, v, g


@triton.jit
def _fused_attention_gate_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    g_ptr,
    bias1_ptr,
    bias2_ptr,
    out_ptr,
    b1_stride_b: tl.constexpr,
    b1_stride_h: tl.constexpr,
    b1_stride_q: tl.constexpr,
    b1_stride_k: tl.constexpr,
    b2_stride_b: tl.constexpr,
    b2_stride_h: tl.constexpr,
    b2_stride_q: tl.constexpr,
    b2_stride_k: tl.constexpr,
    Q: tl.constexpr,
    K: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    block = tl.program_id(0)
    batch = block // H
    head = block % H

    rows = tl.arange(0, BLOCK_Q)
    keys = tl.arange(0, BLOCK_K)
    cols = tl.arange(0, BLOCK_D)
    q_base = batch * Q * H * D + head * D
    kv_base = batch * K * H * D + head * D

    q = tl.load(
        q_ptr + q_base + rows[:, None] * H * D + cols[None, :],
        mask=(rows[:, None] < Q) & (cols[None, :] < D),
        other=0.0,
    )
    q = (q * (1.0 / tl.sqrt(float(D)))).to(tl.bfloat16)
    k = tl.load(
        k_ptr + kv_base + keys[:, None] * H * D + cols[None, :],
        mask=(keys[:, None] < K) & (cols[None, :] < D),
        other=0.0,
    )

    # Keep the eager path's BF16 materialization after each score addition.
    scores = tl.dot(q, tl.trans(k)).to(tl.bfloat16)
    b1 = tl.load(
        bias1_ptr
        + batch * b1_stride_b
        + head * b1_stride_h
        + rows[:, None] * b1_stride_q
        + keys[None, :] * b1_stride_k,
        mask=(rows[:, None] < Q) & (keys[None, :] < K),
        other=0.0,
    )
    scores = (scores + b1).to(tl.bfloat16)
    b2 = tl.load(
        bias2_ptr
        + batch * b2_stride_b
        + head * b2_stride_h
        + rows[:, None] * b2_stride_q
        + keys[None, :] * b2_stride_k,
        mask=(rows[:, None] < Q) & (keys[None, :] < K),
        other=0.0,
    )
    scores = (scores + b2).to(tl.bfloat16)
    scores = tl.where(keys[None, :] < K, scores, -float("inf"))
    scores = scores.to(tl.float32)
    scores -= tl.max(scores, axis=1)[:, None]
    probs = tl.exp2(scores * 1.4426950408889634)
    probs /= tl.sum(probs, axis=1)[:, None]

    v = tl.load(
        v_ptr + kv_base + keys[:, None] * H * D + cols[None, :],
        mask=(keys[:, None] < K) & (cols[None, :] < D),
        other=0.0,
    )
    attended = tl.dot(probs.to(tl.bfloat16), v)
    gate = tl.load(
        g_ptr + q_base + rows[:, None] * H * D + cols[None, :],
        mask=(rows[:, None] < Q) & (cols[None, :] < D),
        other=0.0,
    ).to(tl.float32)
    gate = 1.0 / (1.0 + tl.exp2(-gate * 1.4426950408889634))
    out = attended * gate
    tl.store(
        out_ptr + q_base + rows[:, None] * H * D + cols[None, :],
        out,
        mask=(rows[:, None] < Q) & (cols[None, :] < D),
    )


def _broadcast_strides(
    bias: torch.Tensor, target_shape: tuple[int, ...],
) -> tuple[int, int, int, int]:
    padding = len(target_shape) - bias.ndim
    shape = (1,) * padding + tuple(bias.shape)
    strides = (0,) * padding + tuple(bias.stride())
    prefix = target_shape[:-3]

    batch_stride = 0
    for axis, size in enumerate(prefix):
        if size > 1:
            batch_stride = strides[axis] if shape[axis] > 1 else 0
            break

    result = [batch_stride]
    for axis in range(len(target_shape) - 3, len(target_shape)):
        result.append(strides[axis] if shape[axis] > 1 else 0)
    return tuple(result)


def _fused_attention_gate(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    biases: list[torch.Tensor],
    no_heads: int,
    c_hidden: int,
) -> torch.Tensor:
    q_len = q.shape[-2]
    kv_len = k.shape[-2]
    target_shape = (*q.shape[:-2], no_heads, q_len, kv_len)
    b1_strides = _broadcast_strides(biases[0], target_shape)
    b2_strides = _broadcast_strides(biases[1], target_shape)
    batch = q.numel() // (q_len * no_heads * c_hidden)
    out = torch.empty_like(q)

    block_q = triton.next_power_of_2(q_len)
    block_k = triton.next_power_of_2(kv_len)
    block_d = triton.next_power_of_2(c_hidden)
    _fused_attention_gate_kernel[(batch * no_heads,)](
        q,
        k,
        v,
        g,
        biases[0],
        biases[1],
        out,
        *b1_strides,
        *b2_strides,
        Q=q_len,
        K=kv_len,
        H=no_heads,
        D=c_hidden,
        BLOCK_Q=block_q,
        BLOCK_K=block_k,
        BLOCK_D=block_d,
        num_warps=8 if kv_len == 128 else 4,
    )
    return out


@triton.jit
def _fused_attention_output_128_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    g_ptr,
    bias1_ptr,
    bias2_ptr,
    weight_ptr,
    out_ptr,
    b1_stride_b: tl.constexpr,
    b1_stride_h: tl.constexpr,
    b1_stride_q: tl.constexpr,
    b1_stride_k: tl.constexpr,
    b2_stride_b: tl.constexpr,
    b2_stride_h: tl.constexpr,
    b2_stride_q: tl.constexpr,
    b2_stride_k: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    batch = tl.program_id(0)
    out_block = tl.program_id(1)
    rows = tl.arange(0, 16)
    keys = tl.arange(0, 16)
    cols = tl.arange(0, 32)
    outs = out_block * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((16, BLOCK_N), tl.float32)

    for head in range(4):
        base = batch * 16 * 128 + head * 32
        q = tl.load(q_ptr + base + rows[:, None] * 128 + cols[None, :])
        q = (q * 0.1767766952966369).to(tl.bfloat16)
        k = tl.load(k_ptr + base + keys[:, None] * 128 + cols[None, :])
        scores = tl.dot(q, tl.trans(k)).to(tl.bfloat16)
        b1 = tl.load(
            bias1_ptr
            + batch * b1_stride_b
            + head * b1_stride_h
            + rows[:, None] * b1_stride_q
            + keys[None, :] * b1_stride_k
        )
        scores = (scores + b1).to(tl.bfloat16)
        b2 = tl.load(
            bias2_ptr
            + batch * b2_stride_b
            + head * b2_stride_h
            + rows[:, None] * b2_stride_q
            + keys[None, :] * b2_stride_k
        )
        scores = (scores + b2).to(tl.bfloat16).to(tl.float32)
        scores -= tl.max(scores, axis=1)[:, None]
        probs = tl.exp2(scores * 1.4426950408889634)
        probs /= tl.sum(probs, axis=1)[:, None]

        value = tl.load(v_ptr + base + keys[:, None] * 128 + cols[None, :])
        attended = tl.dot(probs.to(tl.bfloat16), value)
        gate = tl.load(
            g_ptr + base + rows[:, None] * 128 + cols[None, :]
        ).to(tl.float32)
        gate = 1.0 / (1.0 + tl.exp2(-gate * 1.4426950408889634))
        hidden = (attended * gate).to(tl.bfloat16)
        weight = tl.load(
            weight_ptr
            + (head * 32 + cols[:, None])
            + outs[None, :] * 128
        )
        acc += tl.dot(hidden, weight)

    tl.store(
        out_ptr + batch * 16 * 128 + rows[:, None] * 128 + outs[None, :],
        acc,
        mask=outs[None, :] < 128,
    )


def _fused_attention_output_128(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    biases: list[torch.Tensor],
    weight: torch.Tensor,
) -> torch.Tensor:
    target_shape = (*q.shape[:-2], 4, 16, 16)
    b1_strides = _broadcast_strides(biases[0], target_shape)
    b2_strides = _broadcast_strides(biases[1], target_shape)
    batch = q.numel() // (16 * 128)
    out = torch.empty_like(q)
    _fused_attention_output_128_kernel[(batch, 1)](
        q,
        k,
        v,
        g,
        biases[0],
        biases[1],
        weight,
        out,
        *b1_strides,
        *b2_strides,
        BLOCK_N=128,
        num_warps=8,
        num_stages=2,
    )
    return out


@triton.jit
def _fused_attention_output_128_long_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    g_ptr,
    bias1_ptr,
    bias2_ptr,
    weight_ptr,
    out_ptr,
    b1_stride_b: tl.constexpr,
    b1_stride_h: tl.constexpr,
    b1_stride_q: tl.constexpr,
    b1_stride_k: tl.constexpr,
    b2_stride_b: tl.constexpr,
    b2_stride_h: tl.constexpr,
    b2_stride_q: tl.constexpr,
    b2_stride_k: tl.constexpr,
):
    batch = tl.program_id(0)
    rows = tl.arange(0, 32)
    keys = tl.arange(0, 128)
    cols = tl.arange(0, 32)
    outs = tl.arange(0, 128)
    acc = tl.zeros((32, 128), tl.float32)

    for head in range(4):
        q_base = batch * 32 * 128 + head * 32
        kv_base = batch * 128 * 128 + head * 32
        q = tl.load(q_ptr + q_base + rows[:, None] * 128 + cols[None, :])
        q = (q * 0.1767766952966369).to(tl.bfloat16)
        k = tl.load(k_ptr + kv_base + keys[:, None] * 128 + cols[None, :])
        scores = tl.dot(q, tl.trans(k)).to(tl.bfloat16)
        b1 = tl.load(
            bias1_ptr
            + batch * b1_stride_b
            + head * b1_stride_h
            + rows[:, None] * b1_stride_q
            + keys[None, :] * b1_stride_k
        )
        scores = (scores + b1).to(tl.bfloat16)
        b2 = tl.load(
            bias2_ptr
            + batch * b2_stride_b
            + head * b2_stride_h
            + rows[:, None] * b2_stride_q
            + keys[None, :] * b2_stride_k
        )
        scores = (scores + b2).to(tl.bfloat16).to(tl.float32)
        scores -= tl.max(scores, axis=1)[:, None]
        probs = tl.exp2(scores * 1.4426950408889634)
        probs /= tl.sum(probs, axis=1)[:, None]
        value = tl.load(
            v_ptr + kv_base + keys[:, None] * 128 + cols[None, :]
        )
        attended = tl.dot(probs.to(tl.bfloat16), value)
        gate = tl.load(
            g_ptr + q_base + rows[:, None] * 128 + cols[None, :]
        ).to(tl.float32)
        gate = 1.0 / (1.0 + tl.exp2(-gate * 1.4426950408889634))
        hidden = (attended * gate).to(tl.bfloat16)
        weight = tl.load(
            weight_ptr + head * 32 + cols[:, None] + outs[None, :] * 128
        )
        acc += tl.dot(hidden, weight)

    tl.store(
        out_ptr + batch * 32 * 128 + rows[:, None] * 128 + outs[None, :],
        acc,
    )


def _fused_attention_output_128_long(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    biases: list[torch.Tensor],
    weight: torch.Tensor,
) -> torch.Tensor:
    target_shape = (*q.shape[:-2], 4, 32, 128)
    b1_strides = _broadcast_strides(biases[0], target_shape)
    b2_strides = _broadcast_strides(biases[1], target_shape)
    batch = q.numel() // (32 * 128)
    out = torch.empty_like(q)
    _fused_attention_output_128_long_kernel[(batch,)](
        q,
        k,
        v,
        g,
        biases[0],
        biases[1],
        weight,
        out,
        *b1_strides,
        *b2_strides,
        num_warps=8,
        num_stages=2,
    )
    return out


@triton.jit
def _fused_attention_output_wide_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    g_ptr,
    bias1_ptr,
    bias2_ptr,
    weight_ptr,
    out_ptr,
    b1_stride_b: tl.constexpr,
    b1_stride_h: tl.constexpr,
    b1_stride_q: tl.constexpr,
    b1_stride_k: tl.constexpr,
    b2_stride_b: tl.constexpr,
    b2_stride_h: tl.constexpr,
    b2_stride_q: tl.constexpr,
    b2_stride_k: tl.constexpr,
    C: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    batch = tl.program_id(0)
    row_block = tl.program_id(1)
    out_block = tl.program_id(2)
    rows = row_block * 8 + tl.arange(0, 8)
    keys = tl.arange(0, 16)
    cols = tl.arange(0, BLOCK_D)
    outs = out_block * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((8, BLOCK_N), tl.float32)

    for head in range(16):
        base = batch * 16 * C + head * D
        q = tl.load(
            q_ptr + base + rows[:, None] * C + cols[None, :],
            mask=cols[None, :] < D,
            other=0.0,
        )
        q = (q * (1.0 / tl.sqrt(float(D)))).to(tl.bfloat16)
        k = tl.load(
            k_ptr + base + keys[:, None] * C + cols[None, :],
            mask=cols[None, :] < D,
            other=0.0,
        )
        scores = tl.dot(q, tl.trans(k)).to(tl.bfloat16)
        b1 = tl.load(
            bias1_ptr
            + batch * b1_stride_b
            + head * b1_stride_h
            + rows[:, None] * b1_stride_q
            + keys[None, :] * b1_stride_k
        )
        scores = (scores + b1).to(tl.bfloat16)
        b2 = tl.load(
            bias2_ptr
            + batch * b2_stride_b
            + head * b2_stride_h
            + rows[:, None] * b2_stride_q
            + keys[None, :] * b2_stride_k
        )
        scores = (scores + b2).to(tl.bfloat16).to(tl.float32)
        scores -= tl.max(scores, axis=1)[:, None]
        probs = tl.exp2(scores * 1.4426950408889634)
        probs /= tl.sum(probs, axis=1)[:, None]
        value = tl.load(
            v_ptr + base + keys[:, None] * C + cols[None, :],
            mask=cols[None, :] < D,
            other=0.0,
        )
        attended = tl.dot(probs.to(tl.bfloat16), value)
        gate = tl.load(
            g_ptr + base + rows[:, None] * C + cols[None, :],
            mask=cols[None, :] < D,
            other=0.0,
        ).to(tl.float32)
        gate = 1.0 / (1.0 + tl.exp2(-gate * 1.4426950408889634))
        hidden = (attended * gate).to(tl.bfloat16)
        weight = tl.load(
            weight_ptr + head * D + cols[:, None] + outs[None, :] * C,
            mask=(cols[:, None] < D) & (outs[None, :] < C),
            other=0.0,
        )
        acc += tl.dot(hidden, weight)

    tl.store(
        out_ptr + batch * 16 * C + rows[:, None] * C + outs[None, :],
        acc,
        mask=outs[None, :] < C,
    )


def _fused_attention_output_wide(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    biases: list[torch.Tensor],
    weight: torch.Tensor,
    c_hidden: int,
) -> torch.Tensor:
    c = q.shape[-1]
    target_shape = (*q.shape[:-2], 16, 16, 16)
    b1_strides = _broadcast_strides(biases[0], target_shape)
    b2_strides = _broadcast_strides(biases[1], target_shape)
    batch = q.numel() // (16 * c)
    out = torch.empty_like(q)
    _fused_attention_output_wide_kernel[
        (batch, 2, triton.cdiv(c, 64))
    ](
        q,
        k,
        v,
        g,
        biases[0],
        biases[1],
        weight,
        out,
        *b1_strides,
        *b2_strides,
        C=c,
        D=c_hidden,
        BLOCK_D=triton.next_power_of_2(c_hidden),
        BLOCK_N=64,
        num_warps=4,
        num_stages=2,
    )
    return out


def _attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    biases: list[torch.Tensor],
) -> torch.Tensor:
    """Core SDPA: scores = softmax(Q K^T + biases) V.

    Args:
        query: [*, H, Q, C_hidden]
        key:   [*, H, K, C_hidden]
        value: [*, H, V, C_hidden]
        biases: list of tensors broadcastable to [*, H, Q, K]

    Returns:
        [*, H, Q, C_hidden]
    """
    scores = torch.einsum("...qc,...kc->...qk", query, key)

    for b in biases:
        scores = scores + b

    scores = F.softmax(scores, dim=-1)

    return torch.einsum("...qk,...kc->...qc", scores.to(dtype=value.dtype), value)


class OF3Attention(nn.Module):
    """Standard multi-head attention with gating and bias list support.

    Reference: openfold3/core/model/primitives/attention.py Attention

    Args:
        c_q: Input dimension of query data
        c_k: Input dimension of key data
        c_v: Input dimension of value data
        c_hidden: Per-head hidden dimension
        no_heads: Number of attention heads
        gating: Whether to gate output using query data
    """

    def __init__(
        self,
        c_q: int,
        c_k: int,
        c_v: int,
        c_hidden: int,
        no_heads: int,
        gating: bool = True,
        q_bias: bool = False,
    ):
        super().__init__()
        self.c_q = c_q
        self.c_k = c_k
        self.c_v = c_v
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.gating = gating

        self.linear_q = Linear(c_q, c_hidden * no_heads, bias=q_bias)
        self.linear_k = Linear(c_k, c_hidden * no_heads, bias=False)
        self.linear_v = Linear(c_v, c_hidden * no_heads, bias=False)
        self.linear_o = Linear(c_hidden * no_heads, c_q, bias=False)

        self.linear_g = None
        if gating:
            self.linear_g = Linear(c_q, c_hidden * no_heads, bias=False)

    def _prep_qkv(
        self, q_x: torch.Tensor, kv_x: torch.Tensor, apply_scale: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self.linear_q(q_x)
        k = self.linear_k(kv_x)
        v = self.linear_v(kv_x)

        q = q.view(q.shape[:-1] + (self.no_heads, -1))
        k = k.view(k.shape[:-1] + (self.no_heads, -1))
        v = v.view(v.shape[:-1] + (self.no_heads, -1))

        q = q.transpose(-2, -3)
        k = k.transpose(-2, -3)
        v = v.transpose(-2, -3)

        if apply_scale:
            q = q / math.sqrt(self.c_hidden)

        return q, k, v

    def _wrap_up(self, o: torch.Tensor, q_x: torch.Tensor) -> torch.Tensor:
        if self.linear_g is not None:
            g = torch.sigmoid(self.linear_g(q_x))
            g = g.view(g.shape[:-1] + (self.no_heads, -1))
            o = o * g

        o = o.reshape(o.shape[:-2] + (-1,))
        return self.linear_o(o)

    def forward(
        self,
        q_x: torch.Tensor,
        kv_x: torch.Tensor,
        biases: list[torch.Tensor] | None = None,
        use_deepspeed_evo_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_lma: bool = False,
        lma_q_chunk_size: int = 1024,
        lma_kv_chunk_size: int = 4096,
        use_high_precision: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            q_x:  [*, Q, C_q] query data
            kv_x: [*, K, C_k] key data
            biases: List of biases that broadcast to [*, H, Q, K]

        Returns:
            [*, Q, C_q] attention update
        """
        if biases is None:
            biases = []

        if (
            q_x.is_cuda
            and q_x.dtype == torch.bfloat16
            and len(biases) == 2
            and self.linear_g is not None
        ):
            q, k, v, g = _project_qkvg(
                q_x,
                kv_x,
                self.linear_q,
                self.linear_k,
                self.linear_v,
                self.linear_g,
            )
            if self.c_q == 128 and q_x.shape[-2] == 16:
                return _fused_attention_output_128(
                    q, k, v, g, biases, self.linear_o.weight,
                )
            if self.c_q == 128 and q_x.shape[-2] == 32:
                return _fused_attention_output_128_long(
                    q, k, v, g, biases, self.linear_o.weight,
                )
            if q_x.shape[-2] == 16 and kv_x.shape[-2] == 16:
                return _fused_attention_output_wide(
                    q,
                    k,
                    v,
                    g,
                    biases,
                    self.linear_o.weight,
                    self.c_hidden,
                )
            o = _fused_attention_gate(
                q, k, v, g, biases, self.no_heads, self.c_hidden,
            )
            return self.linear_o(o)

        q, k, v = self._prep_qkv(q_x, kv_x)
        o = _attention(q, k, v, biases)
        o = o.transpose(-2, -3)

        return self._wrap_up(o, q_x)
