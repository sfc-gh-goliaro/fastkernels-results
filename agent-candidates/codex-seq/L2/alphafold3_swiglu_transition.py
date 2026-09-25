"""SwiGLU transition composites for AlphaFold3 (L2).

SwiGLUTransition: LayerNorm -> SwiGLU -> Linear (AF3 Algorithm 11)
ConditionedTransitionBlock: AdaLN -> SwiGLU -> gated output (AF3 Algorithm 25)

Reference: openfold3/core/model/layers/transition.py SwiGLUTransition
           openfold3/core/model/layers/transition.py ConditionedTransitionBlock
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

from ..L1.sigmoid import Sigmoid
from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear
from .alphafold3_swiglu import AdaLN, SwiGLU


@triton.jit
def _swiglu_kernel(
    x,
    weight_a,
    weight_b,
    out,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc_a = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    acc_b = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k in range(0, K, BLOCK_K):
        k_idx = k + offs_k
        x_tile = tl.load(
            x + offs_m[:, None] * K + k_idx[None, :],
            mask=(offs_m[:, None] < M) & (k_idx[None, :] < K),
            other=0.0,
        )
        w_mask = (offs_n[:, None] < N) & (k_idx[None, :] < K)
        wa = tl.load(
            weight_a + offs_n[:, None] * K + k_idx[None, :],
            mask=w_mask,
            other=0.0,
        )
        wb = tl.load(
            weight_b + offs_n[:, None] * K + k_idx[None, :],
            mask=w_mask,
            other=0.0,
        )
        acc_a += tl.dot(x_tile, tl.trans(wa))
        acc_b += tl.dot(x_tile, tl.trans(wb))

    # The standalone linears and SiLU each materialize BF16 in the reference.
    a = acc_a.to(tl.bfloat16).to(tl.float32)
    b = acc_b.to(tl.bfloat16).to(tl.float32)
    silu = (a * tl.sigmoid(a)).to(tl.bfloat16).to(tl.float32)
    value = silu * b
    tl.store(
        out + offs_m[:, None] * N + offs_n[None, :],
        value,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.jit
def _norm_swiglu_kernel(
    x,
    norm_weight,
    norm_bias,
    weight_a,
    weight_b,
    out,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    valid_rows = offs_m < M

    x_tile = tl.load(
        x + offs_m[:, None] * K + offs_k[None, :],
        mask=valid_rows[:, None] & (offs_k[None, :] < K),
        other=0.0,
    ).to(tl.float32)
    mean = tl.sum(x_tile, axis=1) / K
    centered = x_tile - mean[:, None]
    variance = tl.sum(centered * centered, axis=1) / K
    x_tile = centered * tl.rsqrt(variance[:, None] + EPS)
    scale = tl.load(norm_weight + offs_k, mask=offs_k < K, other=0.0)
    bias = tl.load(norm_bias + offs_k, mask=offs_k < K, other=0.0)
    x_tile = (x_tile * scale[None, :] + bias[None, :]).to(tl.bfloat16)

    w_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)
    wa = tl.load(
        weight_a + offs_n[:, None] * K + offs_k[None, :],
        mask=w_mask,
        other=0.0,
    )
    wb = tl.load(
        weight_b + offs_n[:, None] * K + offs_k[None, :],
        mask=w_mask,
        other=0.0,
    )
    acc_a = tl.dot(x_tile, tl.trans(wa))
    acc_b = tl.dot(x_tile, tl.trans(wb))

    a = acc_a.to(tl.bfloat16).to(tl.float32)
    b = acc_b.to(tl.bfloat16).to(tl.float32)
    silu = (a * tl.sigmoid(a)).to(tl.bfloat16).to(tl.float32)
    tl.store(
        out + offs_m[:, None] * N + offs_n[None, :],
        silu * b,
        mask=valid_rows[:, None] & (offs_n[None, :] < N),
    )


@triton.jit
def _small_transition_kernel(
    x,
    norm_weight,
    norm_bias,
    weight_a,
    weight_b,
    weight_out,
    mask,
    out,
    M: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    EPS: tl.constexpr,
    HAS_MASK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_h = tl.arange(0, BLOCK_H)
    valid_rows = offs_m < M

    x_tile = tl.load(
        x + offs_m[:, None] * N + offs_k[None, :],
        mask=valid_rows[:, None] & (offs_k[None, :] < N),
        other=0.0,
    ).to(tl.float32)
    mean = tl.sum(x_tile, axis=1) / N
    centered = x_tile - mean[:, None]
    variance = tl.sum(centered * centered, axis=1) / N
    x_tile = centered * tl.rsqrt(variance[:, None] + EPS)
    scale = tl.load(norm_weight + offs_k, mask=offs_k < N, other=0.0)
    bias = tl.load(norm_bias + offs_k, mask=offs_k < N, other=0.0)
    x_tile = (x_tile * scale[None, :] + bias[None, :]).to(tl.bfloat16)

    result = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for h in range(0, H, BLOCK_H):
        h_idx = h + offs_h
        w_up_mask = (h_idx[:, None] < H) & (offs_k[None, :] < N)
        wa = tl.load(
            weight_a + h_idx[:, None] * N + offs_k[None, :],
            mask=w_up_mask,
            other=0.0,
        )
        wb = tl.load(
            weight_b + h_idx[:, None] * N + offs_k[None, :],
            mask=w_up_mask,
            other=0.0,
        )
        a = tl.dot(x_tile, tl.trans(wa)).to(tl.bfloat16).to(tl.float32)
        b = tl.dot(x_tile, tl.trans(wb)).to(tl.bfloat16).to(tl.float32)
        hidden = (
            (a * tl.sigmoid(a)).to(tl.bfloat16).to(tl.float32) * b
        ).to(tl.bfloat16)
        wo = tl.load(
            weight_out + offs_n[:, None] * H + h_idx[None, :],
            mask=(offs_n[:, None] < N) & (h_idx[None, :] < H),
            other=0.0,
        )
        result += tl.dot(hidden, tl.trans(wo))

    value = result.to(tl.bfloat16).to(tl.float32)
    if HAS_MASK:
        row_mask = tl.load(mask + offs_m, mask=valid_rows, other=0.0)
        value *= row_mask[:, None]
    tl.store(
        out + offs_m[:, None] * N + offs_n[None, :],
        value,
        mask=valid_rows[:, None] & (offs_n[None, :] < N),
    )


@triton.jit
def _small_conditioned_transition_kernel(
    x,
    s,
    weight_a,
    weight_b,
    weight_out,
    weight_gate,
    bias_gate,
    mask,
    out,
    M: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    HAS_MASK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_h = tl.arange(0, BLOCK_H)
    valid_rows = offs_m < M

    x_tile = tl.load(
        x + offs_m[:, None] * N + offs_n[None, :],
        mask=valid_rows[:, None] & (offs_n[None, :] < N),
        other=0.0,
    )
    result = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for h in range(0, H, BLOCK_H):
        h_idx = h + offs_h
        up_mask = (h_idx[:, None] < H) & (offs_n[None, :] < N)
        wa = tl.load(
            weight_a + h_idx[:, None] * N + offs_n[None, :],
            mask=up_mask,
            other=0.0,
        )
        wb = tl.load(
            weight_b + h_idx[:, None] * N + offs_n[None, :],
            mask=up_mask,
            other=0.0,
        )
        a = tl.dot(x_tile, tl.trans(wa)).to(tl.bfloat16).to(tl.float32)
        b = tl.dot(x_tile, tl.trans(wb)).to(tl.bfloat16).to(tl.float32)
        hidden = (
            (a * tl.sigmoid(a)).to(tl.bfloat16).to(tl.float32) * b
        ).to(tl.bfloat16)
        wo = tl.load(
            weight_out + offs_n[:, None] * H + h_idx[None, :],
            mask=(offs_n[:, None] < N) & (h_idx[None, :] < H),
            other=0.0,
        )
        result += tl.dot(hidden, tl.trans(wo))

    s_tile = tl.load(
        s + offs_m[:, None] * N + offs_n[None, :],
        mask=valid_rows[:, None] & (offs_n[None, :] < N),
        other=0.0,
    )
    wg = tl.load(
        weight_gate + offs_n[:, None] * N + offs_n[None, :],
        mask=(offs_n[:, None] < N) & (offs_n[None, :] < N),
        other=0.0,
    )
    gate = tl.dot(s_tile, tl.trans(wg))
    gate_bias = tl.load(bias_gate + offs_n, mask=offs_n < N, other=0.0)
    gate = (gate + gate_bias[None, :]).to(tl.bfloat16).to(tl.float32)
    gate = tl.sigmoid(gate).to(tl.bfloat16).to(tl.float32)
    value = (
        result.to(tl.bfloat16).to(tl.float32) * gate
    ).to(tl.bfloat16).to(tl.float32)
    if HAS_MASK:
        row_mask = tl.load(mask + offs_m, mask=valid_rows, other=0.0)
        value *= row_mask[:, None]
    tl.store(
        out + offs_m[:, None] * N + offs_n[None, :],
        value,
        mask=valid_rows[:, None] & (offs_n[None, :] < N),
    )


@triton.jit
def _full_narrow_conditioned_kernel(
    a,
    s,
    adaln_norm_weight,
    adaln_weight_g,
    adaln_bias_g,
    adaln_weight_s,
    up_weight_a,
    up_weight_b,
    out_weight,
    out_gate_weight,
    out_gate_bias,
    mask,
    out,
    M: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    EPS_A: tl.constexpr,
    EPS_S: tl.constexpr,
    HAS_MASK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    offs_h = tl.arange(0, BLOCK_H)
    valid_rows = offs_m < M
    tile_mask = valid_rows[:, None] & (cols[None, :] < N)

    av = tl.load(
        a + offs_m[:, None] * N + cols[None, :],
        mask=tile_mask,
        other=0.0,
    ).to(tl.float32)
    a_mean = tl.sum(av, axis=1) / N
    av -= a_mean[:, None]
    a_var = tl.sum(av * av, axis=1) / N
    av = (av * tl.rsqrt(a_var[:, None] + EPS_A)).to(tl.bfloat16)

    raw_s = tl.load(
        s + offs_m[:, None] * N + cols[None, :],
        mask=tile_mask,
        other=0.0,
    )
    sv = raw_s.to(tl.float32)
    s_mean = tl.sum(sv, axis=1) / N
    sv -= s_mean[:, None]
    s_var = tl.sum(sv * sv, axis=1) / N
    scale = tl.load(adaln_norm_weight + cols, mask=cols < N, other=0.0)
    sv = (
        sv * tl.rsqrt(s_var[:, None] + EPS_S) * scale[None, :]
    ).to(tl.bfloat16)

    square_mask = (cols[:, None] < N) & (cols[None, :] < N)
    adaln_wg = tl.load(
        adaln_weight_g + cols[:, None] * N + cols[None, :],
        mask=square_mask,
        other=0.0,
    )
    adaln_ws = tl.load(
        adaln_weight_s + cols[:, None] * N + cols[None, :],
        mask=square_mask,
        other=0.0,
    )
    adaptive_gate = tl.dot(sv, tl.trans(adaln_wg))
    adaptive_shift = tl.dot(sv, tl.trans(adaln_ws))
    adaptive_bias = tl.load(adaln_bias_g + cols, mask=cols < N, other=0.0)
    adaptive_gate = (
        adaptive_gate + adaptive_bias[None, :]
    ).to(tl.bfloat16).to(tl.float32)
    adaptive_gate = tl.sigmoid(adaptive_gate).to(tl.bfloat16).to(tl.float32)
    adaptive_shift = adaptive_shift.to(tl.bfloat16).to(tl.float32)
    adaptive = (
        (
            av.to(tl.float32) + adaptive_shift
        ).to(tl.bfloat16).to(tl.float32)
        * adaptive_gate
    ).to(tl.bfloat16)

    result = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for h in range(0, H, BLOCK_H):
        h_idx = h + offs_h
        up_mask = (h_idx[:, None] < H) & (cols[None, :] < N)
        wa = tl.load(
            up_weight_a + h_idx[:, None] * N + cols[None, :],
            mask=up_mask,
            other=0.0,
        )
        wb = tl.load(
            up_weight_b + h_idx[:, None] * N + cols[None, :],
            mask=up_mask,
            other=0.0,
        )
        up_a = tl.dot(adaptive, tl.trans(wa)).to(tl.bfloat16).to(tl.float32)
        up_b = tl.dot(adaptive, tl.trans(wb)).to(tl.bfloat16).to(tl.float32)
        hidden = (
            (up_a * tl.sigmoid(up_a)).to(tl.bfloat16).to(tl.float32) * up_b
        ).to(tl.bfloat16)
        wo = tl.load(
            out_weight + cols[:, None] * H + h_idx[None, :],
            mask=(cols[:, None] < N) & (h_idx[None, :] < H),
            other=0.0,
        )
        result += tl.dot(hidden, tl.trans(wo))

    gate_weight = tl.load(
        out_gate_weight + cols[:, None] * N + cols[None, :],
        mask=square_mask,
        other=0.0,
    )
    output_gate = tl.dot(raw_s, tl.trans(gate_weight))
    output_bias = tl.load(out_gate_bias + cols, mask=cols < N, other=0.0)
    output_gate = (
        output_gate + output_bias[None, :]
    ).to(tl.bfloat16).to(tl.float32)
    output_gate = tl.sigmoid(output_gate).to(tl.bfloat16).to(tl.float32)
    value = (
        result.to(tl.bfloat16).to(tl.float32) * output_gate
    ).to(tl.bfloat16).to(tl.float32)
    if HAS_MASK:
        row_mask = tl.load(mask + offs_m, mask=valid_rows, other=0.0)
        value *= row_mask[:, None]
    tl.store(
        out + offs_m[:, None] * N + cols[None, :],
        value,
        mask=tile_mask,
    )


@triton.jit
def _down_kernel(
    x,
    weight,
    mask,
    out,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    HAS_MASK: tl.constexpr,
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
    for k in range(0, K, BLOCK_K):
        k_idx = k + offs_k
        x_tile = tl.load(
            x + offs_m[:, None] * K + k_idx[None, :],
            mask=(offs_m[:, None] < M) & (k_idx[None, :] < K),
            other=0.0,
        )
        w = tl.load(
            weight + offs_n[:, None] * K + k_idx[None, :],
            mask=(offs_n[:, None] < N) & (k_idx[None, :] < K),
            other=0.0,
        )
        acc += tl.dot(x_tile, tl.trans(w))

    value = acc.to(tl.bfloat16).to(tl.float32)
    if HAS_MASK:
        row_mask = tl.load(mask + offs_m, mask=offs_m < M, other=0.0)
        value *= row_mask[:, None]
    tl.store(
        out + offs_m[:, None] * N + offs_n[None, :],
        value,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.jit
def _adaln_kernel(
    a,
    s,
    weight_g,
    bias_g,
    weight_s,
    out,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc_g = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    acc_s = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k in range(0, K, BLOCK_K):
        k_idx = k + offs_k
        s_tile = tl.load(
            s + offs_m[:, None] * K + k_idx[None, :],
            mask=(offs_m[:, None] < M) & (k_idx[None, :] < K),
            other=0.0,
        )
        w_mask = (offs_n[:, None] < N) & (k_idx[None, :] < K)
        wg = tl.load(
            weight_g + offs_n[:, None] * K + k_idx[None, :],
            mask=w_mask,
            other=0.0,
        )
        ws = tl.load(
            weight_s + offs_n[:, None] * K + k_idx[None, :],
            mask=w_mask,
            other=0.0,
        )
        acc_g += tl.dot(s_tile, tl.trans(wg))
        acc_s += tl.dot(s_tile, tl.trans(ws))

    bias = tl.load(bias_g + offs_n, mask=offs_n < N, other=0.0)
    gate_in = (acc_g + bias[None, :]).to(tl.bfloat16).to(tl.float32)
    gate = tl.sigmoid(gate_in).to(tl.bfloat16).to(tl.float32)
    shift = acc_s.to(tl.bfloat16).to(tl.float32)
    a_tile = tl.load(
        a + offs_m[:, None] * N + offs_n[None, :],
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        other=0.0,
    ).to(tl.float32)
    shifted = (a_tile + shift).to(tl.bfloat16).to(tl.float32)
    tl.store(
        out + offs_m[:, None] * N + offs_n[None, :],
        gate * shifted,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.jit
def _conditioned_down_kernel(
    x,
    s,
    weight_out,
    weight_gate,
    bias_gate,
    mask,
    out,
    M: tl.constexpr,
    N: tl.constexpr,
    K_X: tl.constexpr,
    K_S: tl.constexpr,
    HAS_MASK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc_out = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k in range(0, K_X, BLOCK_K):
        k_idx = k + offs_k
        x_tile = tl.load(
            x + offs_m[:, None] * K_X + k_idx[None, :],
            mask=(offs_m[:, None] < M) & (k_idx[None, :] < K_X),
            other=0.0,
        )
        wo = tl.load(
            weight_out + offs_n[:, None] * K_X + k_idx[None, :],
            mask=(offs_n[:, None] < N) & (k_idx[None, :] < K_X),
            other=0.0,
        )
        acc_out += tl.dot(x_tile, tl.trans(wo))

    acc_gate = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k in range(0, K_S, BLOCK_K):
        k_idx = k + offs_k
        s_tile = tl.load(
            s + offs_m[:, None] * K_S + k_idx[None, :],
            mask=(offs_m[:, None] < M) & (k_idx[None, :] < K_S),
            other=0.0,
        )
        wg = tl.load(
            weight_gate + offs_n[:, None] * K_S + k_idx[None, :],
            mask=(offs_n[:, None] < N) & (k_idx[None, :] < K_S),
            other=0.0,
        )
        acc_gate += tl.dot(s_tile, tl.trans(wg))

    bias = tl.load(bias_gate + offs_n, mask=offs_n < N, other=0.0)
    projected = acc_out.to(tl.bfloat16).to(tl.float32)
    gate_in = (acc_gate + bias[None, :]).to(tl.bfloat16).to(tl.float32)
    gate = tl.sigmoid(gate_in).to(tl.bfloat16).to(tl.float32)
    value = (projected * gate).to(tl.bfloat16).to(tl.float32)
    if HAS_MASK:
        row_mask = tl.load(mask + offs_m, mask=offs_m < M, other=0.0)
        value *= row_mask[:, None]
    tl.store(
        out + offs_m[:, None] * N + offs_n[None, :],
        value,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.jit
def _dual_layer_norm_kernel(
    a,
    s,
    weight_s,
    a_out,
    s_out,
    M: tl.constexpr,
    N_A: tl.constexpr,
    N_S: tl.constexpr,
    EPS_A: tl.constexpr,
    EPS_S: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)

    a_mask = cols < N_A
    av = tl.load(a + row * N_A + cols, mask=a_mask, other=0.0).to(tl.float32)
    a_mean = tl.sum(av, axis=0) / N_A
    a_centered = tl.where(a_mask, av - a_mean, 0.0)
    a_var = tl.sum(a_centered * a_centered, axis=0) / N_A
    av = a_centered * tl.rsqrt(a_var + EPS_A)
    tl.store(a_out + row * N_A + cols, av, mask=a_mask)

    s_mask = cols < N_S
    sv = tl.load(s + row * N_S + cols, mask=s_mask, other=0.0).to(tl.float32)
    s_mean = tl.sum(sv, axis=0) / N_S
    s_centered = tl.where(s_mask, sv - s_mean, 0.0)
    s_var = tl.sum(s_centered * s_centered, axis=0) / N_S
    sv = s_centered * tl.rsqrt(s_var + EPS_S)
    scale = tl.load(weight_s + cols, mask=s_mask, other=0.0).to(tl.float32)
    tl.store(s_out + row * N_S + cols, sv * scale, mask=s_mask)


@triton.jit
def _fused_adaln_narrow_kernel(
    a,
    s,
    norm_weight_s,
    weight_g,
    bias_g,
    weight_s,
    out,
    M: tl.constexpr,
    N: tl.constexpr,
    EPS_A: tl.constexpr,
    EPS_S: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (cols[None, :] < N)

    av = tl.load(
        a + offs_m[:, None] * N + cols[None, :], mask=mask, other=0.0
    ).to(tl.float32)
    a_mean = tl.sum(av, axis=1) / N
    av -= a_mean[:, None]
    a_var = tl.sum(av * av, axis=1) / N
    av *= tl.rsqrt(a_var[:, None] + EPS_A)

    sv = tl.load(
        s + offs_m[:, None] * N + cols[None, :], mask=mask, other=0.0
    ).to(tl.float32)
    s_mean = tl.sum(sv, axis=1) / N
    sv -= s_mean[:, None]
    s_var = tl.sum(sv * sv, axis=1) / N
    sv *= tl.rsqrt(s_var[:, None] + EPS_S)
    norm_scale = tl.load(norm_weight_s + cols, mask=cols < N, other=0.0)
    sv = (sv * norm_scale[None, :]).to(tl.bfloat16)

    wg = tl.load(
        weight_g + cols[:, None] * N + cols[None, :],
        mask=(cols[:, None] < N) & (cols[None, :] < N),
        other=0.0,
    )
    ws = tl.load(
        weight_s + cols[:, None] * N + cols[None, :],
        mask=(cols[:, None] < N) & (cols[None, :] < N),
        other=0.0,
    )
    gate_in = tl.dot(sv, tl.trans(wg))
    shift = tl.dot(sv, tl.trans(ws)).to(tl.bfloat16).to(tl.float32)
    gate_bias = tl.load(bias_g + cols, mask=cols < N, other=0.0)
    gate_in = (gate_in + gate_bias[None, :]).to(tl.bfloat16).to(tl.float32)
    gate = tl.sigmoid(gate_in).to(tl.bfloat16).to(tl.float32)
    shifted = (
        av.to(tl.bfloat16).to(tl.float32) + shift
    ).to(tl.bfloat16).to(tl.float32)
    tl.store(
        out + offs_m[:, None] * N + cols[None, :],
        gate * shifted,
        mask=mask,
    )


@triton.jit
def _fused_adaln_wide_kernel(
    a,
    s,
    norm_weight_s,
    weight_g,
    bias_g,
    weight_s,
    out,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    EPS_A: tl.constexpr,
    EPS_S: tl.constexpr,
    BLOCK_NORM_A: tl.constexpr,
    BLOCK_NORM_S: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(1)
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    valid_rows = offs_m < M

    a_cols = tl.arange(0, BLOCK_NORM_A)
    a_mask = valid_rows[:, None] & (a_cols[None, :] < N)
    a_stats = tl.load(
        a + offs_m[:, None] * N + a_cols[None, :],
        mask=a_mask,
        other=0.0,
    ).to(tl.float32)
    a_mean = tl.sum(a_stats, axis=1) / N
    a_centered = tl.where(a_cols[None, :] < N, a_stats - a_mean[:, None], 0.0)
    a_var = tl.sum(a_centered * a_centered, axis=1) / N
    a_rstd = tl.rsqrt(a_var + EPS_A)

    s_cols = tl.arange(0, BLOCK_NORM_S)
    s_mask = valid_rows[:, None] & (s_cols[None, :] < K)
    s_stats = tl.load(
        s + offs_m[:, None] * K + s_cols[None, :],
        mask=s_mask,
        other=0.0,
    ).to(tl.float32)
    s_mean = tl.sum(s_stats, axis=1) / K
    s_centered = tl.where(s_cols[None, :] < K, s_stats - s_mean[:, None], 0.0)
    s_var = tl.sum(s_centered * s_centered, axis=1) / K
    s_rstd = tl.rsqrt(s_var + EPS_S)

    offs_k = tl.arange(0, BLOCK_K)
    acc_g = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    acc_s = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k in range(0, K, BLOCK_K):
        k_idx = k + offs_k
        sv = tl.load(
            s + offs_m[:, None] * K + k_idx[None, :],
            mask=valid_rows[:, None] & (k_idx[None, :] < K),
            other=0.0,
        ).to(tl.float32)
        scale = tl.load(
            norm_weight_s + k_idx, mask=k_idx < K, other=0.0
        ).to(tl.float32)
        sv = (
            (sv - s_mean[:, None]) * s_rstd[:, None] * scale[None, :]
        ).to(tl.bfloat16)
        weight_mask = (offs_n[:, None] < N) & (k_idx[None, :] < K)
        wg = tl.load(
            weight_g + offs_n[:, None] * K + k_idx[None, :],
            mask=weight_mask,
            other=0.0,
        )
        ws = tl.load(
            weight_s + offs_n[:, None] * K + k_idx[None, :],
            mask=weight_mask,
            other=0.0,
        )
        acc_g += tl.dot(sv, tl.trans(wg))
        acc_s += tl.dot(sv, tl.trans(ws))

    av = tl.load(
        a + offs_m[:, None] * N + offs_n[None, :],
        mask=valid_rows[:, None] & (offs_n[None, :] < N),
        other=0.0,
    ).to(tl.float32)
    av = ((av - a_mean[:, None]) * a_rstd[:, None]).to(tl.bfloat16)
    shift = acc_s.to(tl.bfloat16).to(tl.float32)
    shifted = (av.to(tl.float32) + shift).to(tl.bfloat16).to(tl.float32)
    gate_bias = tl.load(bias_g + offs_n, mask=offs_n < N, other=0.0)
    gate_in = (acc_g + gate_bias[None, :]).to(tl.bfloat16).to(tl.float32)
    gate = tl.sigmoid(gate_in).to(tl.bfloat16).to(tl.float32)
    tl.store(
        out + offs_m[:, None] * N + offs_n[None, :],
        gate * shifted,
        mask=valid_rows[:, None] & (offs_n[None, :] < N),
    )


def _up_config(m: int) -> tuple[int, int, int, int, int]:
    if m <= 16:
        return 16, 32, 64, 4, 3
    return 64, 64, 32, 4, 3


def _down_config(m: int) -> tuple[int, int, int, int, int]:
    if m <= 16:
        return 16, 64, 128, 4, 3
    return 32, 64, 64, 4, 3


def _adaln_config(m: int) -> tuple[int, int, int, int, int]:
    if m <= 16:
        return 16, 32, 64, 4, 3
    return 64, 64, 32, 4, 3


def _swiglu(x: torch.Tensor, module: SwiGLU) -> torch.Tensor:
    k = x.shape[-1]
    n = module.linear_a.weight.shape[0]
    m = x.numel() // k
    out = torch.empty((*x.shape[:-1], n), device=x.device, dtype=x.dtype)
    bm, bn, bk, warps, stages = _up_config(m)
    _swiglu_kernel[(triton.cdiv(m, bm), triton.cdiv(n, bn))](
        x,
        module.linear_a.weight,
        module.linear_b.weight,
        out,
        M=m,
        N=n,
        K=k,
        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_K=bk,
        num_warps=warps,
        num_stages=stages,
    )
    return out


def _norm_swiglu(
    x: torch.Tensor, norm: LayerNorm, module: SwiGLU
) -> torch.Tensor:
    k = x.shape[-1]
    n = module.linear_a.weight.shape[0]
    m = x.numel() // k
    out = torch.empty((*x.shape[:-1], n), device=x.device, dtype=x.dtype)
    if m <= 16:
        bm, bn, warps, stages = 16, 32, 4, 3
    else:
        bm, bn, warps, stages = 32, 64, 4, 3
    bk = triton.next_power_of_2(k)
    _norm_swiglu_kernel[(triton.cdiv(m, bm), triton.cdiv(n, bn))](
        x,
        norm.weight,
        norm.bias,
        module.linear_a.weight,
        module.linear_b.weight,
        out,
        M=m,
        N=n,
        K=k,
        EPS=norm.eps,
        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_K=bk,
        num_warps=warps,
        num_stages=stages,
    )
    return out


def _small_transition(
    x: torch.Tensor,
    norm: LayerNorm,
    swiglu: SwiGLU,
    weight_out: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    n = x.shape[-1]
    h = swiglu.linear_a.weight.shape[0]
    m = x.numel() // n
    out = torch.empty_like(x)
    if n == 64:
        bm, bn, bh, warps, stages = 16, 64, 64, 8, 3
    else:
        bm, bn, bh, warps, stages = 16, 128, 128, 8, 3
    mask_ptr = mask if mask is not None else x
    _small_transition_kernel[(triton.cdiv(m, bm), triton.cdiv(n, bn))](
        x,
        norm.weight,
        norm.bias,
        swiglu.linear_a.weight,
        swiglu.linear_b.weight,
        weight_out,
        mask_ptr,
        out,
        M=m,
        N=n,
        H=h,
        EPS=norm.eps,
        HAS_MASK=mask is not None,
        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_H=bh,
        BLOCK_K=triton.next_power_of_2(n),
        num_warps=warps,
        num_stages=stages,
    )
    return out


def _small_conditioned_transition(
    x: torch.Tensor,
    s: torch.Tensor,
    swiglu: SwiGLU,
    weight_out: torch.Tensor,
    gate: Linear,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    n = x.shape[-1]
    h = swiglu.linear_a.weight.shape[0]
    m = x.numel() // n
    out = torch.empty_like(x)
    mask_ptr = mask if mask is not None else x
    _small_conditioned_transition_kernel[(triton.cdiv(m, 16),)](
        x,
        s,
        swiglu.linear_a.weight,
        swiglu.linear_b.weight,
        weight_out,
        gate.weight,
        gate.bias,
        mask_ptr,
        out,
        M=m,
        N=n,
        H=h,
        HAS_MASK=mask is not None,
        BLOCK_M=16,
        BLOCK_N=128,
        BLOCK_H=128,
        num_warps=8,
        num_stages=3,
    )
    return out


def _full_narrow_conditioned(
    a: torch.Tensor,
    s: torch.Tensor,
    adaln: AdaLN,
    swiglu: SwiGLU,
    weight_out: torch.Tensor,
    gate: Linear,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    n = a.shape[-1]
    h = swiglu.linear_a.weight.shape[0]
    m = a.numel() // n
    out = torch.empty_like(a)
    mask_ptr = mask if mask is not None else a
    _full_narrow_conditioned_kernel[(triton.cdiv(m, 16),)](
        a,
        s,
        adaln.layer_norm_s.weight,
        adaln.linear_g.weight,
        adaln.linear_g.bias,
        adaln.linear_s.weight,
        swiglu.linear_a.weight,
        swiglu.linear_b.weight,
        weight_out,
        gate.weight,
        gate.bias,
        mask_ptr,
        out,
        M=m,
        N=n,
        H=h,
        EPS_A=adaln.layer_norm_a.eps,
        EPS_S=adaln.layer_norm_s.eps,
        HAS_MASK=mask is not None,
        BLOCK_M=16,
        BLOCK_N=128,
        BLOCK_H=128,
        num_warps=8,
        num_stages=3,
    )
    return out


def _down(
    x: torch.Tensor, weight: torch.Tensor, mask: torch.Tensor | None
) -> torch.Tensor:
    k = x.shape[-1]
    n = weight.shape[0]
    m = x.numel() // k
    out = torch.empty((*x.shape[:-1], n), device=x.device, dtype=x.dtype)
    bm, bn, bk, warps, stages = _down_config(m)
    mask_ptr = mask if mask is not None else x
    _down_kernel[(triton.cdiv(m, bm), triton.cdiv(n, bn))](
        x,
        weight,
        mask_ptr,
        out,
        M=m,
        N=n,
        K=k,
        HAS_MASK=mask is not None,
        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_K=bk,
        num_warps=warps,
        num_stages=stages,
    )
    return out


def _adaln(a: torch.Tensor, s: torch.Tensor, module: AdaLN) -> torch.Tensor:
    n = a.shape[-1]
    k = s.shape[-1]
    m = a.numel() // n
    if n == 128 and k == 128:
        out = torch.empty_like(a)
        _fused_adaln_narrow_kernel[(triton.cdiv(m, 16),)](
            a,
            s,
            module.layer_norm_s.weight,
            module.linear_g.weight,
            module.linear_g.bias,
            module.linear_s.weight,
            out,
            M=m,
            N=n,
            EPS_A=module.layer_norm_a.eps,
            EPS_S=module.layer_norm_s.eps,
            BLOCK_M=16,
            BLOCK_N=128,
            num_warps=8,
            num_stages=3,
        )
        return out
    if n == 768 and k == 384:
        out = torch.empty_like(a)
        _fused_adaln_wide_kernel[(triton.cdiv(m, 16), triton.cdiv(n, 64))](
            a,
            s,
            module.layer_norm_s.weight,
            module.linear_g.weight,
            module.linear_g.bias,
            module.linear_s.weight,
            out,
            M=m,
            N=n,
            K=k,
            EPS_A=module.layer_norm_a.eps,
            EPS_S=module.layer_norm_s.eps,
            BLOCK_NORM_A=1024,
            BLOCK_NORM_S=512,
            BLOCK_M=16,
            BLOCK_N=64,
            BLOCK_K=64,
            num_warps=4,
            num_stages=3,
        )
        return out

    a_norm = torch.empty_like(a)
    s_norm = torch.empty_like(s)
    norm_block = triton.next_power_of_2(max(n, k))
    norm_warps = 1 if norm_block <= 512 else 4
    _dual_layer_norm_kernel[(m,)](
        a,
        s,
        module.layer_norm_s.weight,
        a_norm,
        s_norm,
        M=m,
        N_A=n,
        N_S=k,
        EPS_A=module.layer_norm_a.eps,
        EPS_S=module.layer_norm_s.eps,
        BLOCK=norm_block,
        num_warps=norm_warps,
    )
    out = torch.empty_like(a)
    bm, bn, bk, warps, stages = _adaln_config(m)
    _adaln_kernel[(triton.cdiv(m, bm), triton.cdiv(n, bn))](
        a_norm,
        s_norm,
        module.linear_g.weight,
        module.linear_g.bias,
        module.linear_s.weight,
        out,
        M=m,
        N=n,
        K=k,
        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_K=bk,
        num_warps=warps,
        num_stages=stages,
    )
    return out


def _conditioned_down(
    x: torch.Tensor,
    s: torch.Tensor,
    weight_out: torch.Tensor,
    gate: Linear,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    k_x = x.shape[-1]
    k_s = s.shape[-1]
    n = weight_out.shape[0]
    m = x.numel() // k_x
    out = torch.empty((*x.shape[:-1], n), device=x.device, dtype=x.dtype)
    if m <= 16:
        bm, bn, bk, warps, stages = 16, 128, 64, 8, 4
    else:
        bm, bn, bk, warps, stages = 64, 64, 32, 4, 3
    mask_ptr = mask if mask is not None else x
    _conditioned_down_kernel[(triton.cdiv(m, bm), triton.cdiv(n, bn))](
        x,
        s,
        weight_out,
        gate.weight,
        gate.bias,
        mask_ptr,
        out,
        M=m,
        N=n,
        K_X=k_x,
        K_S=k_s,
        HAS_MASK=mask is not None,
        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_K=bk,
        num_warps=warps,
        num_stages=stages,
    )
    return out


class SwiGLUTransition(nn.Module):
    """AF3 Algorithm 11: SwiGLU-based transition.

    Args:
        c_in: Input channel dimension
        n: Factor multiplied to c_in for hidden dimension
    """

    def __init__(self, c_in: int, n: int):
        super().__init__()
        self.c_in = c_in
        self.n = n

        self.layer_norm = LayerNorm(c_in)
        self.swiglu = SwiGLU(c_in, n * c_in)
        self.linear_out = Linear(n * c_in, c_in, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        chunk_size: int | None = None,
        ckpt_chunk_size: int | None = None,
    ) -> torch.Tensor:
        if not x.is_cuda or x.dtype != torch.bfloat16:
            if mask is None:
                mask = x.new_ones(x.shape[:-1])
            x = self.layer_norm(x)
            x = self.swiglu(x)
            return self.linear_out(x) * mask.unsqueeze(-1)

        if x.shape[-1] <= 128:
            return _small_transition(
                x, self.layer_norm, self.swiglu, self.linear_out.weight, mask
            )
        if x.shape[-1] <= 384:
            x = _norm_swiglu(x, self.layer_norm, self.swiglu)
        else:
            x = self.layer_norm(x)
            x = _swiglu(x, self.swiglu)
        return _down(x, self.linear_out.weight, mask)


class ConditionedTransitionBlock(nn.Module):
    """AF3 Algorithm 25: SwiGLU transition with AdaLN-Zero conditioning.

    Submodule names match the reference checkpoint:
    - layer_norm: AdaLN
    - swiglu: SwiGLU
    - linear_g: output gate Linear(c_s, c_a, bias=True)
    - linear_out: Linear(n*c_a, c_a, bias=False)

    Reference: openfold3/core/model/layers/transition.py ConditionedTransitionBlock

    Args:
        c_a: Activation channel dimension
        c_s: Conditioning channel dimension
        n: Factor for hidden dimension
    """

    def __init__(self, c_a: int, c_s: int, n: int):
        super().__init__()
        self.layer_norm = AdaLN(c_a=c_a, c_s=c_s)
        self.swiglu = SwiGLU(c_a, n * c_a)
        self.sigmoid = Sigmoid()
        self.linear_g = Linear(c_s, c_a, bias=True)
        self.linear_out = Linear(n * c_a, c_a, bias=False)

    def forward(
        self,
        a: torch.Tensor,
        s: torch.Tensor,
        mask: torch.Tensor | None = None,
        chunk_size: int | None = None,
    ) -> torch.Tensor:
        if not a.is_cuda or a.dtype != torch.bfloat16:
            if mask is None:
                mask = a.new_ones(a.shape[:-1])
            a = self.layer_norm(a, s)
            b = self.swiglu(a)
            a = self.sigmoid(self.linear_g(s)) * self.linear_out(b)
            return a * mask.unsqueeze(-1)

        if a.shape[-1] == 128 and s.shape[-1] == 128:
            return _full_narrow_conditioned(
                a,
                s,
                self.layer_norm,
                self.swiglu,
                self.linear_out.weight,
                self.linear_g,
                mask,
            )
        a = _adaln(a, s, self.layer_norm)
        b = _swiglu(a, self.swiglu)
        return _conditioned_down(
            b, s, self.linear_out.weight, self.linear_g, mask
        )
