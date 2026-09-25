"""Oasis temporal axial attention."""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

from ..L1.dense_attention import DenseAttention
from ..L1.linear import Linear
from ..L1.oasis_rotary import OasisRotaryEmbedding


@triton.jit
def _fused_temporal_attention_t2(
    qkv_ptr,
    freqs_ptr,
    out_ptr,
    spatial: tl.constexpr,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
):
    block = tl.program_id(0)
    position = block // heads
    head = block % heads
    cols = tl.arange(0, head_dim)
    feature = head * head_dim + cols
    qkv_stride = 3 * heads * head_dim
    row0 = position * qkv_stride
    row1 = (spatial + position) * qkv_stride

    pair_cols = cols ^ 1
    pair_feature = head * head_dim + pair_cols
    freq = tl.load(freqs_ptr + cols // 2)
    angle = freq.to(tl.float16).to(tl.float32)
    cos = tl.cos(angle)
    sin = tl.sin(angle)
    pair_sign = tl.where((cols & 1) == 0, -1.0, 1.0)

    q1 = tl.load(qkv_ptr + row1 + feature)
    q1_pair = tl.load(qkv_ptr + row1 + pair_feature)
    q1 = (q1 * cos + q1_pair * sin * pair_sign).to(tl.float16)
    k0 = tl.load(qkv_ptr + row0 + heads * head_dim + feature)
    k1 = tl.load(qkv_ptr + row1 + heads * head_dim + feature)
    k1_pair = tl.load(
        qkv_ptr + row1 + heads * head_dim + pair_feature
    )
    k1 = (k1 * cos + k1_pair * sin * pair_sign).to(tl.float16)

    score_delta = tl.sum(
        q1.to(tl.float32) * (k1.to(tl.float32) - k0.to(tl.float32)),
        axis=0,
    ) * 0.125
    weight1 = 1.0 / (
        1.0 + tl.exp2(-score_delta * 1.4426950408889634)
    )
    v0 = tl.load(qkv_ptr + row0 + 2 * heads * head_dim + feature)
    v1 = tl.load(qkv_ptr + row1 + 2 * heads * head_dim + feature)
    result1 = v0 + (v1 - v0) * weight1

    tl.store(out_ptr + position * (heads * head_dim) + feature, v0)
    tl.store(
        out_ptr + (spatial + position) * (heads * head_dim) + feature,
        result1,
    )


@triton.jit
def _fused_temporal_attention_t3(
    qkv_ptr,
    freqs_ptr,
    out_ptr,
    time: tl.constexpr,
    spatial: tl.constexpr,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
):
    block = tl.program_id(0)
    position = block // heads
    head = block % heads
    cols = tl.arange(0, head_dim)
    feature = head * head_dim + cols
    pair_feature = head * head_dim + (cols ^ 1)
    qkv_stride = 3 * heads * head_dim
    row0 = position * qkv_stride
    row1 = (spatial + position) * qkv_stride
    row2 = (2 * spatial + position) * qkv_stride

    freq = tl.load(freqs_ptr + cols // 2)
    pair_sign = tl.where((cols & 1) == 0, -1.0, 1.0)
    sin1 = tl.sin(freq.to(tl.float16).to(tl.float32))
    cos1 = tl.cos(freq.to(tl.float16).to(tl.float32))
    angle2 = (2.0 * freq).to(tl.float16).to(tl.float32)
    sin2 = tl.sin(angle2)
    cos2 = tl.cos(angle2)

    k0 = tl.load(qkv_ptr + row0 + heads * head_dim + feature)
    k1 = tl.load(qkv_ptr + row1 + heads * head_dim + feature)
    k1_pair = tl.load(
        qkv_ptr + row1 + heads * head_dim + pair_feature
    )
    k1 = (k1 * cos1 + k1_pair * sin1 * pair_sign).to(tl.float16)
    k2 = tl.load(qkv_ptr + row2 + heads * head_dim + feature)
    k2_pair = tl.load(
        qkv_ptr + row2 + heads * head_dim + pair_feature
    )
    k2 = (k2 * cos2 + k2_pair * sin2 * pair_sign).to(tl.float16)

    q1 = tl.load(qkv_ptr + row1 + feature)
    q1_pair = tl.load(qkv_ptr + row1 + pair_feature)
    q1 = (q1 * cos1 + q1_pair * sin1 * pair_sign).to(tl.float16)
    score10 = tl.sum(q1.to(tl.float32) * k0.to(tl.float32), axis=0)
    score11 = tl.sum(q1.to(tl.float32) * k1.to(tl.float32), axis=0)
    weight11 = 1.0 / (
        1.0 + tl.exp2(-(score11 - score10) * 0.18033688011112042)
    )

    q2 = tl.load(qkv_ptr + row2 + feature)
    q2_pair = tl.load(qkv_ptr + row2 + pair_feature)
    q2 = (q2 * cos2 + q2_pair * sin2 * pair_sign).to(tl.float16)
    score20 = tl.sum(q2.to(tl.float32) * k0.to(tl.float32), axis=0) * 0.125
    score21 = tl.sum(q2.to(tl.float32) * k1.to(tl.float32), axis=0) * 0.125
    score22 = tl.sum(q2.to(tl.float32) * k2.to(tl.float32), axis=0) * 0.125
    score_max = tl.maximum(score20, tl.maximum(score21, score22))
    p0 = tl.exp2((score20 - score_max) * 1.4426950408889634)
    p1 = tl.exp2((score21 - score_max) * 1.4426950408889634)
    p2 = tl.exp2((score22 - score_max) * 1.4426950408889634)
    inv_sum = 1.0 / (p0 + p1 + p2)

    v0 = tl.load(qkv_ptr + row0 + 2 * heads * head_dim + feature)
    v1 = tl.load(qkv_ptr + row1 + 2 * heads * head_dim + feature)
    v2 = tl.load(qkv_ptr + row2 + 2 * heads * head_dim + feature)
    out_stride = heads * head_dim
    tl.store(out_ptr + position * out_stride + feature, v0)
    tl.store(
        out_ptr + (spatial + position) * out_stride + feature,
        v0 + (v1 - v0) * weight11,
    )
    tl.store(
        out_ptr + (2 * spatial + position) * out_stride + feature,
        (v0 * p0 + v1 * p1 + v2 * p2) * inv_sum,
    )

    if time >= 4:
        row3 = (3 * spatial + position) * qkv_stride
        angle3 = (3.0 * freq).to(tl.float16).to(tl.float32)
        sin3 = tl.sin(angle3)
        cos3 = tl.cos(angle3)
        k3 = tl.load(qkv_ptr + row3 + heads * head_dim + feature)
        k3_pair = tl.load(
            qkv_ptr + row3 + heads * head_dim + pair_feature
        )
        k3 = (k3 * cos3 + k3_pair * sin3 * pair_sign).to(tl.float16)
        q3 = tl.load(qkv_ptr + row3 + feature)
        q3_pair = tl.load(qkv_ptr + row3 + pair_feature)
        q3 = (q3 * cos3 + q3_pair * sin3 * pair_sign).to(tl.float16)
        score30 = tl.sum(q3.to(tl.float32) * k0.to(tl.float32), axis=0) * 0.125
        score31 = tl.sum(q3.to(tl.float32) * k1.to(tl.float32), axis=0) * 0.125
        score32 = tl.sum(q3.to(tl.float32) * k2.to(tl.float32), axis=0) * 0.125
        score33 = tl.sum(q3.to(tl.float32) * k3.to(tl.float32), axis=0) * 0.125
        score_max3 = tl.maximum(
            tl.maximum(score30, score31), tl.maximum(score32, score33)
        )
        p30 = tl.exp2((score30 - score_max3) * 1.4426950408889634)
        p31 = tl.exp2((score31 - score_max3) * 1.4426950408889634)
        p32 = tl.exp2((score32 - score_max3) * 1.4426950408889634)
        p33 = tl.exp2((score33 - score_max3) * 1.4426950408889634)
        inv_sum3 = 1.0 / (p30 + p31 + p32 + p33)
        v3 = tl.load(qkv_ptr + row3 + 2 * heads * head_dim + feature)
        tl.store(
            out_ptr + (3 * spatial + position) * out_stride + feature,
            (v0 * p30 + v1 * p31 + v2 * p32 + v3 * p33) * inv_sum3,
        )

    if time >= 5:
        row4 = (4 * spatial + position) * qkv_stride
        angle4 = (4.0 * freq).to(tl.float16).to(tl.float32)
        sin4 = tl.sin(angle4)
        cos4 = tl.cos(angle4)
        k4 = tl.load(qkv_ptr + row4 + heads * head_dim + feature)
        k4_pair = tl.load(
            qkv_ptr + row4 + heads * head_dim + pair_feature
        )
        k4 = (k4 * cos4 + k4_pair * sin4 * pair_sign).to(tl.float16)
        q4 = tl.load(qkv_ptr + row4 + feature)
        q4_pair = tl.load(qkv_ptr + row4 + pair_feature)
        q4 = (q4 * cos4 + q4_pair * sin4 * pair_sign).to(tl.float16)
        score40 = tl.sum(q4.to(tl.float32) * k0.to(tl.float32), axis=0) * 0.125
        score41 = tl.sum(q4.to(tl.float32) * k1.to(tl.float32), axis=0) * 0.125
        score42 = tl.sum(q4.to(tl.float32) * k2.to(tl.float32), axis=0) * 0.125
        score43 = tl.sum(q4.to(tl.float32) * k3.to(tl.float32), axis=0) * 0.125
        score44 = tl.sum(q4.to(tl.float32) * k4.to(tl.float32), axis=0) * 0.125
        score_max4 = tl.maximum(
            tl.maximum(score40, score41),
            tl.maximum(tl.maximum(score42, score43), score44),
        )
        p40 = tl.exp2((score40 - score_max4) * 1.4426950408889634)
        p41 = tl.exp2((score41 - score_max4) * 1.4426950408889634)
        p42 = tl.exp2((score42 - score_max4) * 1.4426950408889634)
        p43 = tl.exp2((score43 - score_max4) * 1.4426950408889634)
        p44 = tl.exp2((score44 - score_max4) * 1.4426950408889634)
        inv_sum4 = 1.0 / (p40 + p41 + p42 + p43 + p44)
        v4 = tl.load(qkv_ptr + row4 + 2 * heads * head_dim + feature)
        tl.store(
            out_ptr + (4 * spatial + position) * out_stride + feature,
            (
                v0 * p40
                + v1 * p41
                + v2 * p42
                + v3 * p43
                + v4 * p44
            )
            * inv_sum4,
        )

    if time == 6:
        row5 = (5 * spatial + position) * qkv_stride
        angle5 = (5.0 * freq).to(tl.float16).to(tl.float32)
        sin5 = tl.sin(angle5)
        cos5 = tl.cos(angle5)
        k5 = tl.load(qkv_ptr + row5 + heads * head_dim + feature)
        k5_pair = tl.load(
            qkv_ptr + row5 + heads * head_dim + pair_feature
        )
        k5 = (k5 * cos5 + k5_pair * sin5 * pair_sign).to(tl.float16)
        q5 = tl.load(qkv_ptr + row5 + feature)
        q5_pair = tl.load(qkv_ptr + row5 + pair_feature)
        q5 = (q5 * cos5 + q5_pair * sin5 * pair_sign).to(tl.float16)
        score50 = tl.sum(q5.to(tl.float32) * k0.to(tl.float32), axis=0) * 0.125
        score51 = tl.sum(q5.to(tl.float32) * k1.to(tl.float32), axis=0) * 0.125
        score52 = tl.sum(q5.to(tl.float32) * k2.to(tl.float32), axis=0) * 0.125
        score53 = tl.sum(q5.to(tl.float32) * k3.to(tl.float32), axis=0) * 0.125
        score54 = tl.sum(q5.to(tl.float32) * k4.to(tl.float32), axis=0) * 0.125
        score55 = tl.sum(q5.to(tl.float32) * k5.to(tl.float32), axis=0) * 0.125
        score_max5 = tl.maximum(
            tl.maximum(tl.maximum(score50, score51), tl.maximum(score52, score53)),
            tl.maximum(score54, score55),
        )
        p50 = tl.exp2((score50 - score_max5) * 1.4426950408889634)
        p51 = tl.exp2((score51 - score_max5) * 1.4426950408889634)
        p52 = tl.exp2((score52 - score_max5) * 1.4426950408889634)
        p53 = tl.exp2((score53 - score_max5) * 1.4426950408889634)
        p54 = tl.exp2((score54 - score_max5) * 1.4426950408889634)
        p55 = tl.exp2((score55 - score_max5) * 1.4426950408889634)
        inv_sum5 = 1.0 / (p50 + p51 + p52 + p53 + p54 + p55)
        v5 = tl.load(qkv_ptr + row5 + 2 * heads * head_dim + feature)
        tl.store(
            out_ptr + (5 * spatial + position) * out_stride + feature,
            (
                v0 * p50
                + v1 * p51
                + v2 * p52
                + v3 * p53
                + v4 * p54
                + v5 * p55
            )
            * inv_sum5,
        )


@triton.jit
def _fused_temporal_attention(
    qkv_ptr,
    freqs_ptr,
    out_ptr,
    time: tl.constexpr,
    spatial: tl.constexpr,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
    causal: tl.constexpr,
):
    block = tl.program_id(0)
    position = block // heads
    head = block % heads

    rows = tl.arange(0, 16)
    cols = tl.arange(0, head_dim)
    token = rows[:, None] * spatial + position
    feature = head * head_dim + cols[None, :]
    mask = rows[:, None] < time
    qkv_stride = 3 * heads * head_dim

    q = tl.load(
        qkv_ptr + token * qkv_stride + feature,
        mask=mask,
        other=0.0,
    )
    k = tl.load(
        qkv_ptr + token * qkv_stride + heads * head_dim + feature,
        mask=mask,
        other=0.0,
    )

    pair_cols = cols ^ 1
    pair_feature = head * head_dim + pair_cols[None, :]
    q_pair = tl.load(
        qkv_ptr + token * qkv_stride + pair_feature,
        mask=mask,
        other=0.0,
    )
    k_pair = tl.load(
        qkv_ptr + token * qkv_stride + heads * head_dim + pair_feature,
        mask=mask,
        other=0.0,
    )
    freq = tl.load(freqs_ptr + cols[None, :] // 2)
    angle = (rows[:, None] * freq).to(tl.float16).to(tl.float32)
    cos = tl.cos(angle)
    sin = tl.sin(angle)
    pair_sign = tl.where((cols[None, :] & 1) == 0, -1.0, 1.0)
    q = q * cos + q_pair * sin * pair_sign
    k = k * cos + k_pair * sin * pair_sign

    scores = tl.dot(q.to(tl.float16), tl.trans(k.to(tl.float16))) * 0.125
    valid = rows[None, :] < time
    if causal:
        valid &= rows[None, :] <= rows[:, None]
    scores = tl.where(valid, scores, -float("inf"))
    scores -= tl.max(scores, axis=1)[:, None]
    probs = tl.exp2(scores * 1.4426950408889634)
    probs /= tl.sum(probs, axis=1)[:, None]

    v = tl.load(
        qkv_ptr + token * qkv_stride + 2 * heads * head_dim + feature,
        mask=mask,
        other=0.0,
    )
    result = tl.dot(probs.to(tl.float16), v)
    output_token = rows[:, None] * spatial + position
    tl.store(
        out_ptr + output_token * (heads * head_dim) + feature,
        result,
        mask=mask,
    )


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
        self.to_qkv = Linear(dim, dim_head * heads * 3, bias=False)
        self.to_out = Linear(dim_head * heads, dim, bias=True)
        self.rotary_emb = rotary_emb
        self.is_causal = is_causal
        self.attn = DenseAttention(backend="sdpa")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, time, height, width, _ = x.shape
        qkv = self.to_qkv(x)
        head_dim = qkv.shape[-1] // (3 * self.heads)
        if (
            x.dtype == torch.float16
            and bsz == 1
            and height * width == 144
            and self.heads == 16
            and head_dim == 64
            and time <= 16
            and self.rotary_emb.freqs.numel() == 32
        ):
            out = torch.empty_like(x)
            if time == 2 and self.is_causal:
                _fused_temporal_attention_t2[(height * width * self.heads,)](
                    qkv,
                    self.rotary_emb.freqs,
                    out,
                    spatial=height * width,
                    heads=self.heads,
                    head_dim=head_dim,
                    num_warps=1,
                )
            elif time in (3, 4, 5, 6) and self.is_causal:
                _fused_temporal_attention_t3[(height * width * self.heads,)](
                    qkv,
                    self.rotary_emb.freqs,
                    out,
                    time=time,
                    spatial=height * width,
                    heads=self.heads,
                    head_dim=head_dim,
                    num_warps=1,
                )
            else:
                _fused_temporal_attention[(height * width * self.heads,)](
                    qkv,
                    self.rotary_emb.freqs,
                    out,
                    time=time,
                    spatial=height * width,
                    heads=self.heads,
                    head_dim=head_dim,
                    causal=self.is_causal,
                    num_warps=4,
                )
            return self.to_out(out)

        q, k, v = qkv.chunk(3, dim=-1)
        q = q.reshape(bsz, time, height, width, self.heads, -1).permute(0, 2, 3, 4, 1, 5)
        k = k.reshape(bsz, time, height, width, self.heads, -1).permute(0, 2, 3, 4, 1, 5)
        v = v.reshape(bsz, time, height, width, self.heads, -1).permute(0, 2, 3, 4, 1, 5)

        q = q.reshape(bsz * height * width, self.heads, time, -1)
        k = k.reshape(bsz * height * width, self.heads, time, -1)
        v = v.reshape(bsz * height * width, self.heads, time, -1)

        q = self.rotary_emb.rotate_queries_or_keys(q, self.rotary_emb.freqs)
        k = self.rotary_emb.rotate_queries_or_keys(k, self.rotary_emb.freqs)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = self.attn(q, k, v, causal=self.is_causal)
        out = out.reshape(bsz, height, width, time, self.heads, -1)
        out = out.permute(0, 3, 1, 2, 4, 5).reshape(bsz, time, height, width, -1)
        return self.to_out(out.to(q.dtype))
