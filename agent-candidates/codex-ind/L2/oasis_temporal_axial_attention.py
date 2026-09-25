"""Fused temporal axial attention for the captured Oasis shapes."""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

from ..L1.linear import Linear
from ..L1.oasis_rotary import OasisRotaryEmbedding


@triton.jit
def _linear_kernel(
    x,
    weight,
    bias,
    out,
    m: tl.constexpr,
    n: tl.constexpr,
    k: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    group_m: tl.constexpr = 8
    width = group_m * grid_n
    group_id = pid // width
    first_m = group_id * group_m
    group_size = tl.minimum(grid_m - first_m, group_m)
    pid_m = first_m + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    for start in range(0, k, BLOCK_K):
        kk = start + offs_k
        a = tl.load(
            x + offs_m[:, None] * k + kk[None, :],
            mask=offs_m[:, None] < m,
            other=0.0,
        )
        b = tl.load(weight + offs_n[None, :] * k + kk[:, None])
        acc = tl.dot(a, b, acc)

    if HAS_BIAS:
        acc += tl.load(bias + offs_n)[None, :]
    tl.store(
        out + offs_m[:, None] * n + offs_n[None, :],
        acc,
        mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
    )


@triton.jit
def _temporal_attention_kernel(
    qkv,
    freqs,
    out,
    TIME: tl.constexpr,
    SPATIAL: tl.constexpr,
    HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    QKV_DIM: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid = tl.program_id(0)
    query_pos = pid % TIME
    head = (pid // TIME) % HEADS
    spatial = pid // (TIME * HEADS)
    q_row = query_pos * SPATIAL + spatial
    d = tl.arange(0, HEAD_DIM)
    out_base = q_row * (HEADS * HEAD_DIM) + head * HEAD_DIM

    if query_pos == 0:
        first_v = q_row * QKV_DIM + 2048 + head * HEAD_DIM
        tl.store(out + out_base + d, tl.load(qkv + first_v + d))
    else:
        pairs = tl.arange(0, HEAD_DIM // 2)
        even = pairs * 2
        odd = even + 1
        q_base = q_row * QKV_DIM + head * HEAD_DIM
        q_even = tl.load(qkv + q_base + even).to(tl.float32)
        q_odd = tl.load(qkv + q_base + odd).to(tl.float32)

        key_pos = tl.arange(0, BLOCK_T)
        valid = key_pos <= query_pos
        k_row = key_pos[:, None] * SPATIAL + spatial
        k_base = k_row * QKV_DIM + 1024 + head * HEAD_DIM
        k_even = tl.load(
            qkv + k_base + even[None, :],
            mask=valid[:, None],
            other=0.0,
        ).to(tl.float32)
        k_odd = tl.load(
            qkv + k_base + odd[None, :],
            mask=valid[:, None],
            other=0.0,
        ).to(tl.float32)

        # dot(R(q, tq), R(k, tk)) using only the relative RoPE angle.
        freq = tl.load(freqs + pairs).to(tl.float32)
        cos_1 = tl.cos(freq)
        sin_1 = tl.sin(freq)
        distance = (query_pos - key_pos)[:, None]
        cos_angle = tl.where(distance == 0, 1.0, cos_1[None, :])
        sin_angle = tl.where(distance == 0, 0.0, -sin_1[None, :])
        cos_n = cos_1
        sin_n = sin_1
        for step in tl.static_range(2, TIME):
            next_cos = cos_n * cos_1 - sin_n * sin_1
            next_sin = sin_n * cos_1 + cos_n * sin_1
            cos_angle = tl.where(distance == step, next_cos[None, :], cos_angle)
            sin_angle = tl.where(distance == step, -next_sin[None, :], sin_angle)
            cos_n = next_cos
            sin_n = next_sin
        dot = q_even[None, :] * k_even + q_odd[None, :] * k_odd
        cross = q_odd[None, :] * k_even - q_even[None, :] * k_odd
        scores = tl.sum(dot * cos_angle + cross * sin_angle, axis=1) * 0.125
        scores = tl.where(valid, scores, -float("inf"))
        scores -= tl.max(scores, axis=0)
        probs = tl.exp(scores)
        probs /= tl.sum(probs, axis=0)

        value_rows = k_row * QKV_DIM + 2048 + head * HEAD_DIM
        values = tl.load(
            qkv + value_rows + d[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        result = tl.sum(probs[:, None] * values, axis=0)
        tl.store(out + out_base + d, result)


def _linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    m = x.numel() // x.shape[-1]
    k = x.shape[-1]
    n = weight.shape[0]
    out = torch.empty((m, n), device=x.device, dtype=x.dtype)
    block_m = 64
    block_n = 128
    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
    _linear_kernel[grid](
        x,
        weight,
        bias if bias is not None else x,
        out,
        m,
        n,
        k,
        HAS_BIAS=bias is not None,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=64,
        num_warps=4,
        num_stages=4 if n == 3072 else 6,
    )
    return out


class OasisTemporalAxialAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        rotary_emb: OasisRotaryEmbedding,
        *,
        is_causal: bool = True,
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.to_qkv = Linear(dim, dim_head * heads * 3, bias=False)
        self.to_out = Linear(dim_head * heads, dim, bias=True)
        self.rotary_emb = rotary_emb
        self.is_causal = is_causal

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, time, height, width, dim = x.shape
        spatial = bsz * height * width
        qkv = _linear(x, self.to_qkv.weight, None)
        attended = torch.empty(
            (bsz * time * height * width, dim),
            device=x.device,
            dtype=x.dtype,
        )
        _temporal_attention_kernel[(spatial * self.heads * time,)](
            qkv,
            self.rotary_emb.freqs,
            attended,
            TIME=time,
            SPATIAL=spatial,
            HEADS=self.heads,
            HEAD_DIM=self.dim_head,
            QKV_DIM=dim * 3,
            BLOCK_T=triton.next_power_of_2(time),
            num_warps=1,
            num_stages=1,
        )
        out = _linear(attended, self.to_out.weight, self.to_out.bias)
        return out.reshape(bsz, time, height, width, dim)
