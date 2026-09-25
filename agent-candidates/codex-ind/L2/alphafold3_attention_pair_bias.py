"""Attention with pair bias for AlphaFold3.

AttentionPairBias: Used in PairFormer and diffusion transformer. Uses a single
    layer_norm_a for both Q and K (AdaLN or LayerNorm).
CrossAttentionPairBias: Used in atom attention (sequence-local). Uses separate
    layer_norm_a_q and layer_norm_a_k, no layer_norm_z.

Reference: openfold3/core/model/layers/attention_pair_bias.py
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

from ..L1.sigmoid import Sigmoid
from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear
from .alphafold3_swiglu import AdaLN
from .alphafold3_of3_attention import OF3Attention


@triton.jit
def _layer_norm_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    N: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(x_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    centered = tl.where(mask, x - mean, 0.0)
    variance = tl.sum(centered * centered, axis=0) / N
    out = centered * tl.rsqrt(variance + EPS)
    if HAS_WEIGHT:
        weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        out *= weight
    if HAS_BIAS:
        bias = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        out += bias
    tl.store(out_ptr + row * N + cols, out, mask=mask)


@triton.jit
def _adaln_combine_kernel(
    a_ptr,
    gate_ptr,
    shift_ptr,
    out_ptr,
    n_elements: tl.constexpr,
    N: tl.constexpr,
    gate_row_stride: tl.constexpr,
    shift_row_stride: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    rows = offsets // N
    cols = offsets - rows * N
    a = tl.load(a_ptr + offsets, mask=mask)
    gate = tl.load(
        gate_ptr + rows * gate_row_stride + cols, mask=mask,
    ).to(tl.float32)
    shift = tl.load(
        shift_ptr + rows * shift_row_stride + cols, mask=mask,
    )
    gate = (1.0 / (1.0 + tl.exp(-gate))).to(a_ptr.dtype.element_ty)
    summed = (a + shift).to(a_ptr.dtype.element_ty)
    tl.store(out_ptr + offsets, (gate * summed).to(a_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _adaln_kernel(
    a_ptr,
    s_ptr,
    s_norm_weight_ptr,
    gate_weight_ptr,
    gate_bias_ptr,
    shift_weight_ptr,
    out_ptr,
    M: tl.constexpr,
    A: tl.constexpr,
    S: tl.constexpr,
    N_ATOM: tl.constexpr,
    N_QUERY: tl.constexpr,
    N_KEY: tl.constexpr,
    SOURCE: tl.constexpr,
    EPS_A: tl.constexpr,
    EPS_S: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BA: tl.constexpr,
    BS: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    valid = rows < M
    if SOURCE == 1:
        source_rows = rows
        valid &= source_rows < N_ATOM
    elif SOURCE == 2:
        block = rows // N_KEY
        key_offset = rows - block * N_KEY
        center = N_QUERY // 2 + block * N_QUERY
        start = tl.maximum(0, tl.minimum(center - N_KEY // 2, N_ATOM - N_KEY))
        source_rows = start + key_offset
    else:
        source_rows = rows

    a_channels = tl.arange(0, BA)
    a_full = tl.load(
        a_ptr + source_rows[:, None] * A + a_channels[None, :],
        mask=valid[:, None] & (a_channels[None, :] < A),
        other=0.0,
    ).to(tl.float32)
    a_mean = tl.sum(a_full, axis=1) / A
    a_centered = tl.where(
        a_channels[None, :] < A, a_full - a_mean[:, None], 0.0,
    )
    a_inv = tl.rsqrt(tl.sum(a_centered * a_centered, axis=1) / A + EPS_A)

    s_channels = tl.arange(0, BS)
    s_full = tl.load(
        s_ptr + source_rows[:, None] * S + s_channels[None, :],
        mask=valid[:, None] & (s_channels[None, :] < S),
        other=0.0,
    ).to(tl.float32)
    s_mean = tl.sum(s_full, axis=1) / S
    s_centered = tl.where(
        s_channels[None, :] < S, s_full - s_mean[:, None], 0.0,
    )
    s_inv = tl.rsqrt(tl.sum(s_centered * s_centered, axis=1) / S + EPS_S)
    s_norm_weight = tl.load(
        s_norm_weight_ptr + s_channels,
        mask=s_channels < S,
        other=0.0,
    ).to(tl.float32)
    s_norm = (s_centered * s_inv[:, None] * s_norm_weight[None, :]).to(
        s_ptr.dtype.element_ty
    )

    gate_weight = tl.load(
        gate_weight_ptr + cols[:, None] * S + s_channels[None, :],
        mask=(cols[:, None] < A) & (s_channels[None, :] < S),
        other=0.0,
    )
    shift_weight = tl.load(
        shift_weight_ptr + cols[:, None] * S + s_channels[None, :],
        mask=(cols[:, None] < A) & (s_channels[None, :] < S),
        other=0.0,
    )
    gate = tl.dot(s_norm, tl.trans(gate_weight))
    shift = tl.dot(s_norm, tl.trans(shift_weight)).to(a_ptr.dtype.element_ty)
    gate_bias = tl.load(
        gate_bias_ptr + cols, mask=cols < A, other=0.0,
    ).to(tl.float32)
    gate = (gate + gate_bias[None, :]).to(a_ptr.dtype.element_ty).to(tl.float32)
    gate = (1.0 / (1.0 + tl.exp(-gate))).to(a_ptr.dtype.element_ty)

    a_values = tl.load(
        a_ptr + source_rows[:, None] * A + cols[None, :],
        mask=valid[:, None] & (cols[None, :] < A),
        other=0.0,
    ).to(tl.float32)
    a_norm = ((a_values - a_mean[:, None]) * a_inv[:, None]).to(
        a_ptr.dtype.element_ty
    )
    summed = (a_norm + shift).to(a_ptr.dtype.element_ty)
    out = (gate * summed).to(a_ptr.dtype.element_ty)
    tl.store(
        out_ptr + rows[:, None] * A + cols[None, :],
        out,
        mask=(rows[:, None] < M) & (cols[None, :] < A),
    )


@triton.jit
def _sigmoid_mul_kernel(
    x_ptr,
    gate_ptr,
    out_ptr,
    n_elements: tl.constexpr,
    N: tl.constexpr,
    gate_row_stride: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    rows = offsets // N
    cols = offsets - rows * N
    x = tl.load(x_ptr + offsets, mask=mask)
    gate = tl.load(
        gate_ptr + rows * gate_row_stride + cols, mask=mask,
    ).to(tl.float32)
    gate = (1.0 / (1.0 + tl.exp(-gate))).to(x_ptr.dtype.element_ty)
    tl.store(
        out_ptr + offsets,
        (x * gate).to(x_ptr.dtype.element_ty),
        mask=mask,
    )


@triton.jit
def _pair_bias_kernel(
    z_ptr,
    weight_ptr,
    norm_weight_ptr,
    norm_bias_ptr,
    out_ptr,
    M: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    QK: tl.constexpr,
    EPS: tl.constexpr,
    NORMALIZE: tl.constexpr,
    HAS_NORM_BIAS: tl.constexpr,
    BM: tl.constexpr,
    BC: tl.constexpr,
    BH: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    channels = tl.arange(0, BC)
    heads = tl.arange(0, BH)
    z = tl.load(
        z_ptr + rows[:, None] * C + channels[None, :],
        mask=(rows[:, None] < M) & (channels[None, :] < C),
        other=0.0,
    ).to(tl.float32)
    if NORMALIZE:
        mean = tl.sum(z, axis=1) / C
        centered = tl.where(channels[None, :] < C, z - mean[:, None], 0.0)
        variance = tl.sum(centered * centered, axis=1) / C
        z = centered * tl.rsqrt(variance[:, None] + EPS)
        norm_weight = tl.load(
            norm_weight_ptr + channels,
            mask=channels < C,
            other=0.0,
        ).to(tl.float32)
        z *= norm_weight[None, :]
        if HAS_NORM_BIAS:
            norm_bias = tl.load(
                norm_bias_ptr + channels,
                mask=channels < C,
                other=0.0,
            ).to(tl.float32)
            z += norm_bias[None, :]
    z = z.to(z_ptr.dtype.element_ty)
    weight = tl.load(
        weight_ptr + heads[:, None] * C + channels[None, :],
        mask=(heads[:, None] < H) & (channels[None, :] < C),
        other=0.0,
    )
    projected = tl.dot(z, tl.trans(weight))
    blocks = rows // QK
    pair_offsets = rows - blocks * QK
    out_offsets = (
        blocks[:, None] * H * QK
        + heads[None, :] * QK
        + pair_offsets[:, None]
    )
    tl.store(
        out_ptr + out_offsets,
        projected,
        mask=(rows[:, None] < M) & (heads[None, :] < H),
    )


@triton.jit
def _output_projection_kernel(
    x_ptr,
    gate_ptr,
    out_weight_ptr,
    final_x_ptr,
    final_weight_ptr,
    final_bias_ptr,
    out_ptr,
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    S: tl.constexpr,
    gate_row_stride: tl.constexpr,
    HAS_FINAL_GATE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    accumulator = tl.zeros((BM, BN), dtype=tl.float32)
    for k_start in range(0, K, BK):
        inner = k_start + tl.arange(0, BK)
        x = tl.load(
            x_ptr + rows[:, None] * K + inner[None, :],
            mask=(rows[:, None] < M) & (inner[None, :] < K),
            other=0.0,
        )
        gate = tl.load(
            gate_ptr + rows[:, None] * gate_row_stride + inner[None, :],
            mask=(rows[:, None] < M) & (inner[None, :] < K),
            other=0.0,
        ).to(tl.float32)
        gate = (1.0 / (1.0 + tl.exp(-gate))).to(x_ptr.dtype.element_ty)
        x = (x * gate).to(x_ptr.dtype.element_ty)
        weight = tl.load(
            out_weight_ptr + cols[:, None] * K + inner[None, :],
            mask=(cols[:, None] < N) & (inner[None, :] < K),
            other=0.0,
        )
        accumulator += tl.dot(x, tl.trans(weight))
    projected = accumulator.to(x_ptr.dtype.element_ty)

    if HAS_FINAL_GATE:
        gate_accumulator = tl.zeros((BM, BN), dtype=tl.float32)
        for s_start in range(0, S, BK):
            inner = s_start + tl.arange(0, BK)
            final_x = tl.load(
                final_x_ptr + rows[:, None] * S + inner[None, :],
                mask=(rows[:, None] < M) & (inner[None, :] < S),
                other=0.0,
            )
            final_weight = tl.load(
                final_weight_ptr + cols[:, None] * S + inner[None, :],
                mask=(cols[:, None] < N) & (inner[None, :] < S),
                other=0.0,
            )
            gate_accumulator += tl.dot(final_x, tl.trans(final_weight))
        final_bias = tl.load(
            final_bias_ptr + cols, mask=cols < N, other=0.0,
        ).to(tl.float32)
        final_gate = (gate_accumulator + final_bias[None, :]).to(
            x_ptr.dtype.element_ty
        ).to(tl.float32)
        final_gate = (1.0 / (1.0 + tl.exp(-final_gate))).to(
            x_ptr.dtype.element_ty
        )
        projected = (projected * final_gate).to(x_ptr.dtype.element_ty)

    tl.store(
        out_ptr + rows[:, None] * N + cols[None, :],
        projected,
        mask=(rows[:, None] < M) & (cols[None, :] < N),
    )


@triton.jit
def _projected_attention_kernel(
    qx_ptr,
    kx_ptr,
    q_weight_ptr,
    q_bias_ptr,
    k_weight_ptr,
    v_weight_ptr,
    g_weight_ptr,
    bias_ptr,
    out_ptr,
    gate_out_ptr,
    qx_block_stride: tl.constexpr,
    kx_block_stride: tl.constexpr,
    bias_block_stride: tl.constexpr,
    bias_head_stride: tl.constexpr,
    bias_row_stride: tl.constexpr,
    bias_col_stride: tl.constexpr,
    H: tl.constexpr,
    Q: tl.constexpr,
    K: tl.constexpr,
    C: tl.constexpr,
    D: tl.constexpr,
    BQ: tl.constexpr,
    BK_ROW: tl.constexpr,
    BC: tl.constexpr,
    BD: tl.constexpr,
    SCALE: tl.constexpr,
):
    pid = tl.program_id(0)
    block = pid // H
    head = pid % H
    q_rows = tl.arange(0, BQ)
    k_rows = tl.arange(0, BK_ROW)
    dims = tl.arange(0, BD)

    q_acc = tl.zeros((BQ, BD), dtype=tl.float32)
    k_acc = tl.zeros((BK_ROW, BD), dtype=tl.float32)
    for c_start in range(0, C, 32):
        channels = c_start + tl.arange(0, 32)
        qx = tl.load(
            qx_ptr
            + block * qx_block_stride
            + q_rows[:, None] * C
            + channels[None, :],
            mask=(q_rows[:, None] < Q) & (channels[None, :] < C),
            other=0.0,
        )
        kx = tl.load(
            kx_ptr
            + block * kx_block_stride
            + k_rows[:, None] * C
            + channels[None, :],
            mask=(k_rows[:, None] < K) & (channels[None, :] < C),
            other=0.0,
        )
        q_weight = tl.load(
            q_weight_ptr
            + (head * D + dims[:, None]) * C
            + channels[None, :],
            mask=(dims[:, None] < D) & (channels[None, :] < C),
            other=0.0,
        )
        k_weight = tl.load(
            k_weight_ptr
            + (head * D + dims[:, None]) * C
            + channels[None, :],
            mask=(dims[:, None] < D) & (channels[None, :] < C),
            other=0.0,
        )
        q_acc += tl.dot(qx, tl.trans(q_weight))
        k_acc += tl.dot(kx, tl.trans(k_weight))
    q_bias = tl.load(
        q_bias_ptr + head * D + dims, mask=dims < D, other=0.0,
    ).to(tl.float32)
    q = (q_acc + q_bias[None, :]).to(qx_ptr.dtype.element_ty)
    k = k_acc.to(qx_ptr.dtype.element_ty)
    q = (q.to(tl.float32) * SCALE).to(qx_ptr.dtype.element_ty)
    scores = tl.dot(q, tl.trans(k)).to(qx_ptr.dtype.element_ty)
    pair_bias = tl.load(
        bias_ptr
        + block * bias_block_stride
        + head * bias_head_stride
        + q_rows[:, None] * bias_row_stride
        + k_rows[None, :] * bias_col_stride,
        mask=(q_rows[:, None] < Q) & (k_rows[None, :] < K),
        other=-1.0e9,
    )
    scores = (scores + pair_bias).to(qx_ptr.dtype.element_ty).to(tl.float32)
    scores -= tl.max(scores, axis=1)[:, None]
    probs = tl.exp(scores)
    probs = (probs / tl.sum(probs, axis=1)[:, None]).to(
        qx_ptr.dtype.element_ty
    )

    v_acc = tl.zeros((BK_ROW, BD), dtype=tl.float32)
    g_acc = tl.zeros((BQ, BD), dtype=tl.float32)
    for c_start in range(0, C, 32):
        channels = c_start + tl.arange(0, 32)
        qx = tl.load(
            qx_ptr
            + block * qx_block_stride
            + q_rows[:, None] * C
            + channels[None, :],
            mask=(q_rows[:, None] < Q) & (channels[None, :] < C),
            other=0.0,
        )
        kx = tl.load(
            kx_ptr
            + block * kx_block_stride
            + k_rows[:, None] * C
            + channels[None, :],
            mask=(k_rows[:, None] < K) & (channels[None, :] < C),
            other=0.0,
        )
        v_weight = tl.load(
            v_weight_ptr
            + (head * D + dims[:, None]) * C
            + channels[None, :],
            mask=(dims[:, None] < D) & (channels[None, :] < C),
            other=0.0,
        )
        g_weight = tl.load(
            g_weight_ptr
            + (head * D + dims[:, None]) * C
            + channels[None, :],
            mask=(dims[:, None] < D) & (channels[None, :] < C),
            other=0.0,
        )
        v_acc += tl.dot(kx, tl.trans(v_weight))
        g_acc += tl.dot(qx, tl.trans(g_weight))
    v = v_acc.to(qx_ptr.dtype.element_ty)
    gate = g_acc.to(qx_ptr.dtype.element_ty)
    out = tl.dot(probs, v)
    out_offsets = (
        block * Q * H * D
        + q_rows[:, None] * H * D
        + head * D
        + dims[None, :]
    )
    mask = (q_rows[:, None] < Q) & (dims[None, :] < D)
    tl.store(out_ptr + out_offsets, out, mask=mask)
    tl.store(gate_out_ptr + out_offsets, gate, mask=mask)


@triton.jit
def _attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    bias_ptr,
    out_ptr,
    q_block_stride: tl.constexpr,
    q_head_stride: tl.constexpr,
    q_row_stride: tl.constexpr,
    k_block_stride: tl.constexpr,
    k_head_stride: tl.constexpr,
    k_row_stride: tl.constexpr,
    bias_block_stride: tl.constexpr,
    bias_head_stride: tl.constexpr,
    bias_row_stride: tl.constexpr,
    bias_col_stride: tl.constexpr,
    H: tl.constexpr,
    Q: tl.constexpr,
    K: tl.constexpr,
    D: tl.constexpr,
    BQ: tl.constexpr,
    BK: tl.constexpr,
    BD: tl.constexpr,
    SCALE: tl.constexpr,
):
    pid = tl.program_id(0)
    block = pid // H
    head = pid % H
    q_rows = tl.arange(0, BQ)
    k_rows = tl.arange(0, BK)
    dims = tl.arange(0, BD)

    q = tl.load(
        q_ptr
        + block * q_block_stride
        + head * q_head_stride
        + q_rows[:, None] * q_row_stride
        + dims[None, :],
        mask=(q_rows[:, None] < Q) & (dims[None, :] < D),
        other=0.0,
    )
    k = tl.load(
        k_ptr
        + block * k_block_stride
        + head * k_head_stride
        + k_rows[:, None] * k_row_stride
        + dims[None, :],
        mask=(k_rows[:, None] < K) & (dims[None, :] < D),
        other=0.0,
    )
    q = (q.to(tl.float32) * SCALE).to(q_ptr.dtype.element_ty)
    scores = tl.dot(q, tl.trans(k)).to(q_ptr.dtype.element_ty)
    bias = tl.load(
        bias_ptr
        + block * bias_block_stride
        + head * bias_head_stride
        + q_rows[:, None] * bias_row_stride
        + k_rows[None, :] * bias_col_stride,
        mask=(q_rows[:, None] < Q) & (k_rows[None, :] < K),
        other=-1.0e9,
    )
    scores = (scores + bias).to(q_ptr.dtype.element_ty).to(tl.float32)
    scores = scores - tl.max(scores, axis=1)[:, None]
    probs = tl.exp(scores)
    probs = probs / tl.sum(probs, axis=1)[:, None]
    probs = probs.to(q_ptr.dtype.element_ty)

    v = tl.load(
        v_ptr
        + block * k_block_stride
        + head * k_head_stride
        + k_rows[:, None] * k_row_stride
        + dims[None, :],
        mask=(k_rows[:, None] < K) & (dims[None, :] < D),
        other=0.0,
    )
    out = tl.dot(probs, v)
    out_offsets = (
        block * Q * H * D
        + q_rows[:, None] * H * D
        + head * D
        + dims[None, :]
    )
    tl.store(
        out_ptr + out_offsets,
        out,
        mask=(q_rows[:, None] < Q) & (dims[None, :] < D),
    )


def _fused_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    h, q_len, dim = q.shape[-3:]
    k_len = k.shape[-2]
    blocks = q.numel() // (h * q_len * dim)
    out = torch.empty(
        (*q.shape[:-3], q_len, h, dim),
        dtype=q.dtype,
        device=q.device,
    )
    _attention_kernel[(blocks * h,)](
        q,
        k,
        v,
        bias,
        out,
        q.stride(-4),
        q.stride(-3),
        q.stride(-2),
        k.stride(-4),
        k.stride(-3),
        k.stride(-2),
        bias.stride(-4),
        bias.stride(-3),
        bias.stride(-2),
        bias.stride(-1),
        H=h,
        Q=q_len,
        K=k_len,
        D=dim,
        BQ=triton.next_power_of_2(q_len),
        BK=triton.next_power_of_2(k_len),
        BD=triton.next_power_of_2(dim),
        SCALE=scale,
        num_warps=4 if k_len <= 32 else 8,
    )
    return out


def _projected_attention(
    qx: torch.Tensor,
    kx: torch.Tensor,
    mha: OF3Attention,
    bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_len = qx.shape[-2]
    k_len = kx.shape[-2]
    channels = qx.shape[-1]
    heads = mha.no_heads
    dim = mha.c_hidden
    blocks = qx.numel() // (q_len * channels)
    shape = (*qx.shape[:-2], q_len, heads, dim)
    out = torch.empty(shape, dtype=qx.dtype, device=qx.device)
    gate = torch.empty_like(out)
    _projected_attention_kernel[(blocks * heads,)](
        qx,
        kx,
        mha.linear_q.weight,
        mha.linear_q.bias,
        mha.linear_k.weight,
        mha.linear_v.weight,
        mha.linear_g.weight,
        bias,
        out,
        gate,
        qx.stride(-3),
        kx.stride(-3),
        bias.stride(-4),
        bias.stride(-3),
        bias.stride(-2),
        bias.stride(-1),
        H=heads,
        Q=q_len,
        K=k_len,
        C=channels,
        D=dim,
        BQ=triton.next_power_of_2(q_len),
        BK_ROW=triton.next_power_of_2(k_len),
        BC=triton.next_power_of_2(channels),
        BD=triton.next_power_of_2(dim),
        SCALE=1.0 / math.sqrt(dim),
        num_warps=4 if k_len <= 32 else 8,
    )
    return out, gate


def _layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    eps: float,
) -> torch.Tensor:
    n = x.shape[-1]
    out = torch.empty_like(x, memory_format=torch.contiguous_format)
    rows = x.numel() // n
    dummy = x
    _layer_norm_kernel[(rows,)](
        x,
        weight if weight is not None else dummy,
        bias if bias is not None else dummy,
        out,
        N=n,
        EPS=eps,
        BLOCK=triton.next_power_of_2(n),
        HAS_WEIGHT=weight is not None,
        HAS_BIAS=bias is not None,
        num_warps=4 if n <= 256 else 8,
    )
    return out


def _pair_bias(
    z: torch.Tensor,
    weight: torch.Tensor,
    norm_weight: torch.Tensor | None = None,
    norm_bias: torch.Tensor | None = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    q_len, k_len, channels = z.shape[-3:]
    heads = weight.shape[0]
    rows = z.numel() // channels
    out = torch.empty(
        (*z.shape[:-3], heads, q_len, k_len),
        dtype=z.dtype,
        device=z.device,
    )
    normalize = norm_weight is not None
    bm = 32 if channels > 16 else 128
    _pair_bias_kernel[(triton.cdiv(rows, bm),)](
        z,
        weight,
        norm_weight if norm_weight is not None else z,
        norm_bias if norm_bias is not None else z,
        out,
        M=rows,
        C=channels,
        H=heads,
        QK=q_len * k_len,
        EPS=eps,
        NORMALIZE=normalize,
        HAS_NORM_BIAS=norm_bias is not None,
        BM=bm,
        BC=triton.next_power_of_2(channels),
        BH=max(16, triton.next_power_of_2(heads)),
        num_warps=4,
    )
    return out


def _output_projection(
    x: torch.Tensor,
    gate: torch.Tensor,
    out_linear: Linear,
    final_x: torch.Tensor | None = None,
    final_linear: Linear | None = None,
    rows: int | None = None,
) -> torch.Tensor:
    in_features = x.shape[-1]
    out_features = out_linear.weight.shape[0]
    x_2d = x.reshape(-1, in_features)
    gate_2d = gate.reshape(-1, in_features)
    m = x_2d.shape[0] if rows is None else rows
    x_2d = x_2d[:m]
    gate_2d = gate_2d[:m]
    out = torch.empty((m, out_features), dtype=x.dtype, device=x.device)
    has_final = final_x is not None
    final_2d = (
        final_x.reshape(-1, final_x.shape[-1]) if has_final else x_2d
    )
    s_features = final_2d.shape[-1]
    bm = 16 if m <= 32 else 32
    bn = 64 if out_features >= 768 else 128
    num_warps = 4 if bn == 64 else 8
    _output_projection_kernel[
        (triton.cdiv(m, bm), triton.cdiv(out_features, bn))
    ](
        x_2d,
        gate_2d,
        out_linear.weight,
        final_2d,
        final_linear.weight if has_final else out_linear.weight,
        final_linear.bias if has_final else out_linear.weight,
        out,
        M=m,
        K=in_features,
        N=out_features,
        S=s_features,
        gate_row_stride=gate_2d.stride(0),
        HAS_FINAL_GATE=has_final,
        BM=bm,
        BN=bn,
        BK=32,
        num_warps=num_warps,
    )
    return out


def _adaln(
    a: torch.Tensor,
    s: torch.Tensor,
    module: AdaLN,
    source: int = 0,
    n_atom: int = 0,
    n_query: int = 0,
    n_key: int = 0,
) -> torch.Tensor:
    channels = module.c_a
    if source == 0:
        rows = a.numel() // channels
        out_shape = a.shape
    else:
        blocks = (n_atom + n_query - 1) // n_query
        block_size = n_query if source == 1 else n_key
        rows = blocks * block_size
        out_shape = (*a.shape[:-2], blocks, block_size, channels)
    out = torch.empty(out_shape, dtype=a.dtype, device=a.device)
    if channels > 256:
        bm, bn = 8, 64
    else:
        bm, bn = 32, 32
    _adaln_kernel[
        (triton.cdiv(rows, bm), triton.cdiv(channels, bn))
    ](
        a,
        s,
        module.layer_norm_s.weight,
        module.linear_g.weight,
        module.linear_g.bias,
        module.linear_s.weight,
        out,
        M=rows,
        A=channels,
        S=module.c_s,
        N_ATOM=n_atom,
        N_QUERY=n_query,
        N_KEY=n_key,
        SOURCE=source,
        EPS_A=module.layer_norm_a.eps,
        EPS_S=module.layer_norm_s.eps,
        BM=bm,
        BN=bn,
        BA=triton.next_power_of_2(channels),
        BS=triton.next_power_of_2(module.c_s),
        num_warps=4,
    )
    return out


def _sigmoid_mul(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    n = x.shape[-1]
    gate = gate.reshape(-1, n)
    out = torch.empty_like(x)
    _sigmoid_mul_kernel[(triton.cdiv(out.numel(), 1024),)](
        x,
        gate,
        out,
        n_elements=out.numel(),
        N=n,
        gate_row_stride=gate.stride(0),
        BLOCK=1024,
    )
    return out


def _permute_final_dims(tensor: torch.Tensor, inds: list[int]) -> torch.Tensor:
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


class AttentionPairBias(nn.Module):
    """AF3 Algorithm 24: Attention with pair bias.

    When use_ada_layer_norm is True, uses two separate AdaLN instances
    (layer_norm_a_q, layer_norm_a_k) for query and key normalization,
    plus a linear_ada_out for output gating.

    Reference: openfold3/core/model/layers/attention_pair_bias.py

    Args:
        c_q: Input dimension of query/key/value
        c_s: Single activation channel dimension (for AdaLN)
        c_z: Pair activation channel dimension
        c_hidden: Per-head hidden dimension
        no_heads: Number of attention heads
        use_ada_layer_norm: Whether to use AdaLN-Zero conditioning
        gating: Whether to gate output
        inf: Large constant for masking
    """

    def __init__(
        self,
        c_q: int,
        c_k: int = 0,
        c_v: int = 0,
        c_s: int = 0,
        c_z: int = 128,
        c_hidden: int = 32,
        no_heads: int = 4,
        use_ada_layer_norm: bool = False,
        gating: bool = True,
        inf: float = 1e9,
    ):
        super().__init__()
        c_k = c_k or c_q
        c_v = c_v or c_q

        self.c_q = c_q
        self.c_s = c_s
        self.c_z = c_z
        self.inf = inf
        self.use_ada_layer_norm = use_ada_layer_norm

        if use_ada_layer_norm:
            self.layer_norm_a = AdaLN(c_a=c_q, c_s=c_s)
            self.linear_ada_out = Linear(c_s, c_q, bias=True)
        else:
            self.layer_norm_a = LayerNorm(c_q)

        self.layer_norm_z = LayerNorm(
            c_z, create_offset=not use_ada_layer_norm,
        )
        self.linear_z = Linear(c_z, no_heads, bias=False)

        self.sigmoid = Sigmoid()

        self.mha = OF3Attention(
            c_q=c_q, c_k=c_k, c_v=c_v,
            c_hidden=c_hidden, no_heads=no_heads, gating=gating,
            q_bias=True,
        )
        self._qkvg_weight: torch.Tensor | None = None
        self._qkvg_bias: torch.Tensor | None = None

    def _project_qkvg(
        self, x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._qkvg_weight is None:
            layers = (
                self.mha.linear_q,
                self.mha.linear_k,
                self.mha.linear_v,
                self.mha.linear_g,
            )
            self._qkvg_weight = torch.cat([layer.weight for layer in layers], dim=0)
            zeros = torch.zeros_like(self.mha.linear_q.bias)
            self._qkvg_bias = torch.cat(
                [self.mha.linear_q.bias, zeros, zeros, zeros], dim=0,
            )
        projected = F.linear(x, self._qkvg_weight, self._qkvg_bias)
        width = self.mha.no_heads * self.mha.c_hidden
        q, k, v, g = projected.split(width, dim=-1)
        shape = (*q.shape[:-1], self.mha.no_heads, self.mha.c_hidden)
        q = q.view(shape).transpose(-2, -3)
        k = k.view(shape).transpose(-2, -3)
        v = v.view(shape).transpose(-2, -3)
        g = g.view(shape)
        return q, k, v, g

    def _prep_bias(
        self, a: torch.Tensor, z: torch.Tensor, mask: torch.Tensor | None,
    ) -> list[torch.Tensor]:
        z = _pair_bias(
            z,
            self.linear_z.weight,
            self.layer_norm_z.weight,
            self.layer_norm_z.bias,
            eps=self.layer_norm_z.eps,
        )
        return [z]

    def forward(
        self,
        a: torch.Tensor,
        z: torch.Tensor,
        s: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        use_deepspeed_evo_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_lma: bool = False,
        use_high_precision_attention: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            a:    [*, N, C_q] token/atom-level embedding
            z:    [*, N, N, C_z] pair embedding
            s:    [*, N, C_s] single embedding (for AdaLN)
            mask: [*, N] mask

        Returns:
            [*, N, C_q] attention update
        """
        biases = self._prep_bias(a=a, z=z, mask=mask)

        a = (
            _adaln(a, s, self.layer_norm_a)
            if self.use_ada_layer_norm
            else _layer_norm(
                a,
                self.layer_norm_a.weight,
                self.layer_norm_a.bias,
                self.layer_norm_a.eps,
            )
        )

        a, g = _projected_attention(a, a, self.mha, biases[0])
        shape = a.shape[:-2]
        a = _output_projection(
            a.flatten(-2),
            g.flatten(-2),
            self.mha.linear_o,
            s if self.use_ada_layer_norm else None,
            self.linear_ada_out if self.use_ada_layer_norm else None,
        ).reshape(*shape, self.c_q)

        return a


class CrossAttentionPairBias(nn.Module):
    """AF3 Algorithm 24: Cross-attention with pair bias for atom attention.

    Uses separate layer_norm_a_q and layer_norm_a_k for query/key, and
    does NOT apply layer_norm_z (pair bias goes through linear_z directly).
    Handles sequence-local blocked inputs.

    Reference: openfold3/core/model/layers/attention_pair_bias.py CrossAttentionPairBias

    Args:
        c_q: Input dimension of query/key/value
        c_s: Single activation channel dimension (for AdaLN)
        c_z: Pair activation channel dimension
        c_hidden: Per-head hidden dimension
        no_heads: Number of attention heads
        use_ada_layer_norm: Whether to use AdaLN-Zero conditioning
        n_query: Block size for queries
        n_key: Block size for keys
        gating: Whether to gate output
        inf: Large constant for masking
    """

    def __init__(
        self,
        c_q: int,
        c_k: int = 0,
        c_v: int = 0,
        c_s: int = 0,
        c_z: int = 16,
        c_hidden: int = 16,
        no_heads: int = 4,
        use_ada_layer_norm: bool = True,
        n_query: int | None = None,
        n_key: int | None = None,
        gating: bool = True,
        inf: float = 1e9,
    ):
        super().__init__()
        c_k = c_k or c_q
        c_v = c_v or c_q

        self.c_q = c_q
        self.c_s = c_s
        self.c_z = c_z
        self.inf = inf
        self.use_ada_layer_norm = use_ada_layer_norm
        self.n_query = n_query
        self.n_key = n_key

        if use_ada_layer_norm:
            self.layer_norm_a_q = AdaLN(c_a=c_q, c_s=c_s)
            self.layer_norm_a_k = AdaLN(c_a=c_q, c_s=c_s)
            self.linear_ada_out = Linear(c_s, c_q, bias=True)
        else:
            self.layer_norm_a_q = LayerNorm(c_q)
            self.layer_norm_a_k = LayerNorm(c_q)

        self.linear_z = Linear(c_z, no_heads, bias=False)

        self.sigmoid = Sigmoid()

        self.mha = OF3Attention(
            c_q=c_q, c_k=c_k, c_v=c_v,
            c_hidden=c_hidden, no_heads=no_heads, gating=gating,
            q_bias=True,
        )
        self._key_indices: torch.Tensor | None = None
        self._qg_weight: torch.Tensor | None = None
        self._qg_bias: torch.Tensor | None = None
        self._kv_weight: torch.Tensor | None = None

    def _project_attention(
        self,
        q_x: torch.Tensor,
        kv_x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._qg_weight is None:
            self._qg_weight = torch.cat(
                [self.mha.linear_q.weight, self.mha.linear_g.weight], dim=0,
            )
            self._qg_bias = torch.cat(
                [self.mha.linear_q.bias, torch.zeros_like(self.mha.linear_q.bias)],
                dim=0,
            )
            self._kv_weight = torch.cat(
                [self.mha.linear_k.weight, self.mha.linear_v.weight], dim=0,
            )
        width = self.mha.no_heads * self.mha.c_hidden
        q, g = F.linear(q_x, self._qg_weight, self._qg_bias).split(width, dim=-1)
        k, v = F.linear(kv_x, self._kv_weight).split(width, dim=-1)
        q_shape = (*q.shape[:-1], self.mha.no_heads, self.mha.c_hidden)
        kv_shape = (*k.shape[:-1], self.mha.no_heads, self.mha.c_hidden)
        return (
            q.view(q_shape).transpose(-2, -3),
            k.view(kv_shape).transpose(-2, -3),
            v.view(kv_shape).transpose(-2, -3),
            g.view(q_shape),
        )

    def _blocked_inputs(
        self,
        x: torch.Tensor,
        n_atom: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_blocks = (n_atom + self.n_query - 1) // self.n_query
        padded = F.pad(x, (0, 0, 0, (-n_atom) % self.n_query))
        query = padded.reshape(*x.shape[:-2], num_blocks, self.n_query, x.shape[-1])

        if self._key_indices is None or self._key_indices.device != x.device:
            block = torch.arange(num_blocks, device=x.device)[:, None]
            key = torch.arange(self.n_key, device=x.device)[None, :]
            center = self.n_query // 2 + block * self.n_query
            start = (center - self.n_key // 2).clamp(0, n_atom - self.n_key)
            self._key_indices = (start + key).reshape(-1)
        keys = x.index_select(-2, self._key_indices)
        keys = keys.reshape(*x.shape[:-2], num_blocks, self.n_key, x.shape[-1])
        return query, keys

    def forward(
        self,
        a: torch.Tensor,
        z: torch.Tensor,
        s: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            a:    [*, N_atom, C_q] atom-level embedding
            z:    [*, N_blocks, n_key, n_key, C_z] blocked pair embedding
            s:    [*, N_atom, C_s] single embedding (for AdaLN)
            mask: [*, N_atom] mask

        Returns:
            [*, N_atom, C_q] attention update
        """
        batch_dims = a.shape[:-2]
        n_atom, n_dim = a.shape[-2:]

        z_bias = _pair_bias(z, self.linear_z.weight)
        biases = [z_bias]

        if self.use_ada_layer_norm:
            a_q = _adaln(
                a, s, self.layer_norm_a_q, 1,
                n_atom, self.n_query, self.n_key,
            )
            a_k = _adaln(
                a, s, self.layer_norm_a_k, 2,
                n_atom, self.n_query, self.n_key,
            )
        else:
            a_query, a_key = self._blocked_inputs(a, n_atom)
            a_q = _layer_norm(
                a_query,
                self.layer_norm_a_q.weight,
                self.layer_norm_a_q.bias,
                self.layer_norm_a_q.eps,
            )
            a_k = _layer_norm(
                a_key,
                self.layer_norm_a_k.weight,
                self.layer_norm_a_k.bias,
                self.layer_norm_a_k.eps,
            )

        a_out, g = _projected_attention(a_q, a_k, self.mha, biases[0])
        a_out = _output_projection(
            a_out.flatten(-2),
            g.flatten(-2),
            self.mha.linear_o,
            s if self.use_ada_layer_norm else None,
            self.linear_ada_out if self.use_ada_layer_norm else None,
            rows=n_atom,
        ).reshape(*batch_dims, n_atom, n_dim)

        return a_out
