"""YOLOv10 PSA (Partial Self-Attention) block."""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

from .yolov10_attention import YOLOAttention
from .yolov10_conv import YOLOConv


@triton.jit
def _pointwise_bn(
    x,
    weight,
    bn_weight,
    bn_bias,
    running_mean,
    running_var,
    residual,
    out,
    M: tl.constexpr,
    S: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    X_BATCH_STRIDE: tl.constexpr,
    RES_BATCH_STRIDE: tl.constexpr,
    EPS: tl.constexpr,
    ACT: tl.constexpr,
    ADD_RESIDUAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    image = offs_m // S
    pos = offs_m % S
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        x_ptrs = x + image[:, None] * X_BATCH_STRIDE + offs_k[None, :] * S + pos[:, None]
        w_ptrs = weight + offs_n[None, :] * K + offs_k[:, None]
        xv = tl.load(
            x_ptrs,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        )
        wv = tl.load(
            w_ptrs,
            mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        acc = tl.dot(xv, wv, acc)

    mask_n = offs_n < N
    mean = tl.load(running_mean + offs_n, mask=mask_n, other=0.0)
    var = tl.load(running_var + offs_n, mask=mask_n, other=1.0)
    gamma = tl.load(bn_weight + offs_n, mask=mask_n, other=0.0)
    beta = tl.load(bn_bias + offs_n, mask=mask_n, other=0.0)
    value = (acc - mean[None, :]) * (
        gamma * tl.rsqrt(var + EPS)
    )[None, :] + beta[None, :]
    if ADD_RESIDUAL:
        r_ptrs = (
            residual
            + image[:, None] * RES_BATCH_STRIDE
            + offs_n[None, :] * S
            + pos[:, None]
        )
        value += tl.load(
            r_ptrs,
            mask=(offs_m[:, None] < M) & mask_n[None, :],
            other=0.0,
        )
    if ACT:
        value = value * tl.sigmoid(value)

    out_ptrs = out + image[:, None] * (N * S) + offs_n[None, :] * S + pos[:, None]
    tl.store(
        out_ptrs,
        value,
        mask=(offs_m[:, None] < M) & mask_n[None, :],
    )


@triton.jit
def _pointwise_cat_bn_act(
    a,
    b,
    weight,
    bn_weight,
    bn_bias,
    running_mean,
    running_var,
    out,
    M: tl.constexpr,
    S: tl.constexpr,
    C: tl.constexpr,
    N: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    image = offs_m // S
    pos = offs_m % S
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    for k0 in range(0, 2 * C, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        from_a = offs_k < C
        channels = tl.where(from_a, offs_k, offs_k - C)
        a_ptrs = a + image[:, None] * (2 * C * S) + channels[None, :] * S + pos[:, None]
        b_ptrs = b + image[:, None] * (C * S) + channels[None, :] * S + pos[:, None]
        xv = tl.load(
            tl.where(from_a[None, :], a_ptrs, b_ptrs),
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < 2 * C),
            other=0.0,
        )
        wv = tl.load(
            weight + offs_n[None, :] * (2 * C) + offs_k[:, None],
            mask=(offs_n[None, :] < N) & (offs_k[:, None] < 2 * C),
            other=0.0,
        )
        acc = tl.dot(xv, wv, acc)

    mask_n = offs_n < N
    mean = tl.load(running_mean + offs_n, mask=mask_n, other=0.0)
    var = tl.load(running_var + offs_n, mask=mask_n, other=1.0)
    gamma = tl.load(bn_weight + offs_n, mask=mask_n, other=0.0)
    beta = tl.load(bn_bias + offs_n, mask=mask_n, other=0.0)
    value = (acc - mean[None, :]) * (
        gamma * tl.rsqrt(var + EPS)
    )[None, :] + beta[None, :]
    value = value * tl.sigmoid(value)
    out_ptrs = out + image[:, None] * (N * S) + offs_n[None, :] * S + pos[:, None]
    tl.store(
        out_ptrs,
        value,
        mask=(offs_m[:, None] < M) & mask_n[None, :],
    )


@triton.jit
def _attention_pe(
    qkv,
    pe_weight,
    pe_bn_weight,
    pe_bn_bias,
    pe_mean,
    pe_var,
    out,
    S: tl.constexpr,
    WIDTH: tl.constexpr,
    SCALE: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    query = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    bh = tl.program_id(1)
    image = bh // 2
    head = bh % 2
    key_dim = tl.arange(0, 32)
    value_dim = tl.arange(0, 64)
    q_base = image * (256 * S) + head * (128 * S)
    q = tl.load(
        qkv + q_base + key_dim[None, :] * S + query[:, None],
        mask=query[:, None] < S,
        other=0.0,
    )

    row_max = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    row_sum = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, 64), tl.float32)
    for n0 in range(0, S, BLOCK_N):
        keys = n0 + tl.arange(0, BLOCK_N)
        k = tl.load(
            qkv + q_base + (32 + key_dim[:, None]) * S + keys[None, :],
            mask=keys[None, :] < S,
            other=0.0,
        )
        logits = tl.dot(q, k) * SCALE
        logits = tl.where(keys[None, :] < S, logits, -float("inf"))
        block_max = tl.max(logits, axis=1)
        new_max = tl.maximum(row_max, block_max)
        alpha = tl.exp(row_max - new_max)
        probs = tl.exp(logits - new_max[:, None])
        row_sum = row_sum * alpha + tl.sum(probs, axis=1)
        v = tl.load(
            qkv
            + q_base
            + (64 + value_dim[None, :]) * S
            + keys[:, None],
            mask=keys[:, None] < S,
            other=0.0,
        )
        acc = acc * alpha[:, None] + tl.dot(probs.to(tl.float16), v)
        row_max = new_max
    attention = acc / row_sum[:, None]

    channel = head * 64 + value_dim
    y = query // WIDTH
    x = query % WIDTH
    pe = tl.zeros((BLOCK_M, 64), tl.float32)
    for kh in range(0, 3):
        for kw in range(0, 3):
            iy = y + kh - 1
            ix = x + kw - 1
            valid = (query < S) & (iy >= 0) & (iy < 20) & (ix >= 0) & (ix < WIDTH)
            source_pos = iy * WIDTH + ix
            source = tl.load(
                qkv
                + q_base
                + (64 + value_dim[None, :]) * S
                + source_pos[:, None],
                mask=valid[:, None],
                other=0.0,
            )
            pw = tl.load(pe_weight + channel * 9 + kh * 3 + kw)
            pe += source * pw[None, :]
    mean = tl.load(pe_mean + channel)
    var = tl.load(pe_var + channel)
    gamma = tl.load(pe_bn_weight + channel)
    beta = tl.load(pe_bn_bias + channel)
    pe = (pe - mean[None, :]) * (
        gamma * tl.rsqrt(var + EPS)
    )[None, :] + beta[None, :]

    out_ptrs = (
        out
        + image * (128 * S)
        + channel[None, :] * S
        + query[:, None]
    )
    tl.store(out_ptrs, attention + pe, mask=query[:, None] < S)


@triton.jit
def _ffn_bn_silu_bn_residual(
    x,
    w1,
    gamma1,
    beta1,
    mean1,
    var1,
    w2,
    gamma2,
    beta2,
    mean2,
    var2,
    out,
    M: tl.constexpr,
    S: tl.constexpr,
    EPS1: tl.constexpr,
    EPS2: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    image = offs_m // S
    pos = offs_m % S
    input_dim = tl.arange(0, 128)
    output_dim = tl.arange(0, 128)
    xv = tl.load(
        x + image[:, None] * (128 * S) + input_dim[None, :] * S + pos[:, None],
        mask=offs_m[:, None] < M,
        other=0.0,
    )
    result = tl.zeros((BLOCK_M, 128), tl.float32)
    for h0 in range(0, 256, 128):
        hidden_dim = h0 + tl.arange(0, 128)
        first_weight = tl.load(w1 + hidden_dim[None, :] * 128 + input_dim[:, None])
        hidden = tl.dot(xv, first_weight)
        hidden_mean = tl.load(mean1 + hidden_dim)
        hidden_var = tl.load(var1 + hidden_dim)
        hidden_gamma = tl.load(gamma1 + hidden_dim)
        hidden_beta = tl.load(beta1 + hidden_dim)
        hidden = (hidden - hidden_mean[None, :]) * (
            hidden_gamma * tl.rsqrt(hidden_var + EPS1)
        )[None, :] + hidden_beta[None, :]
        hidden = hidden * tl.sigmoid(hidden)
        second_weight = tl.load(
            w2 + output_dim[None, :] * 256 + hidden_dim[:, None]
        )
        result += tl.dot(hidden.to(tl.float16), second_weight)

    out_mean = tl.load(mean2 + output_dim)
    out_var = tl.load(var2 + output_dim)
    out_gamma = tl.load(gamma2 + output_dim)
    out_beta = tl.load(beta2 + output_dim)
    result = (result - out_mean[None, :]) * (
        out_gamma * tl.rsqrt(out_var + EPS2)
    )[None, :] + out_beta[None, :]
    result += xv
    out_ptrs = (
        out
        + image[:, None] * (128 * S)
        + output_dim[None, :] * S
        + pos[:, None]
    )
    tl.store(out_ptrs, result, mask=offs_m[:, None] < M)


def _pw(x, layer, out_channels: int, *, act: bool, residual=None):
    batch, in_channels, height, width = x.shape
    spatial = height * width
    out = torch.empty(
        (batch, out_channels, height, width), device=x.device, dtype=x.dtype
    )
    block_m = 32
    block_n = 64 if in_channels == 256 else 32
    _pointwise_bn[
        (triton.cdiv(batch * spatial, block_m), triton.cdiv(out_channels, block_n))
    ](
        x,
        layer.conv.weight,
        layer.bn.weight,
        layer.bn.bias,
        layer.bn.running_mean,
        layer.bn.running_var,
        residual,
        out,
        M=batch * spatial,
        S=spatial,
        K=in_channels,
        N=out_channels,
        X_BATCH_STRIDE=x.stride(0),
        RES_BATCH_STRIDE=residual.stride(0) if residual is not None else 0,
        EPS=layer.bn.eps,
        ACT=act,
        ADD_RESIDUAL=residual is not None,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=64,
        num_warps=4,
        num_stages=3,
    )
    return out


class YOLOPSA(nn.Module):
    def __init__(self, c1: int, c2: int, e: float = 0.5):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = YOLOConv(c1, 2 * self.c, 1, 1)
        self.cv2 = YOLOConv(2 * self.c, c1, 1, 1)
        self.attn = YOLOAttention(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1))
        self.ffn = nn.Sequential(
            YOLOConv(self.c, self.c * 2, 1, 1),
            YOLOConv(self.c * 2, self.c, 1, 1, act=False),
        )
        self._graph = None
        self._graph_ready = False

    def _forward_fused(self, x: torch.Tensor) -> torch.Tensor:
        first = _pw(x, self.cv1, 256, act=True)
        b = first[:, 128:]
        qkv = _pw(b, self.attn.qkv, 256, act=False)

        attn_pe = torch.empty(
            (x.shape[0], 128, 20, 20), device=x.device, dtype=x.dtype
        )
        pe = self.attn.pe
        _attention_pe[
            (triton.cdiv(400, 32), x.shape[0] * 2)
        ](
            qkv,
            pe.conv.weight,
            pe.bn.weight,
            pe.bn.bias,
            pe.bn.running_mean,
            pe.bn.running_var,
            attn_pe,
            S=400,
            WIDTH=20,
            SCALE=self.attn.scale,
            EPS=pe.bn.eps,
            BLOCK_M=32,
            BLOCK_N=256,
            num_warps=4,
            num_stages=2,
        )
        b = _pw(attn_pe, self.attn.proj, 128, act=False, residual=b)
        ffn1 = self.ffn[0]
        ffn2 = self.ffn[1]
        ffn_out = torch.empty_like(b)
        _ffn_bn_silu_bn_residual[
            (triton.cdiv(x.shape[0] * 400, 16),)
        ](
            b,
            ffn1.conv.weight,
            ffn1.bn.weight,
            ffn1.bn.bias,
            ffn1.bn.running_mean,
            ffn1.bn.running_var,
            ffn2.conv.weight,
            ffn2.bn.weight,
            ffn2.bn.bias,
            ffn2.bn.running_mean,
            ffn2.bn.running_var,
            ffn_out,
            M=x.shape[0] * 400,
            S=400,
            EPS1=ffn1.bn.eps,
            EPS2=ffn2.bn.eps,
            BLOCK_M=16,
            num_warps=4,
            num_stages=2,
        )
        b = ffn_out

        out = torch.empty_like(x)
        cv2 = self.cv2
        _pointwise_cat_bn_act[
            (triton.cdiv(x.shape[0] * 400, 32), 4)
        ](
            first,
            b,
            cv2.conv.weight,
            cv2.bn.weight,
            cv2.bn.bias,
            cv2.bn.running_mean,
            cv2.bn.running_var,
            out,
            M=x.shape[0] * 400,
            S=400,
            C=128,
            N=256,
            EPS=cv2.bn.eps,
            BLOCK_M=32,
            BLOCK_N=64,
            BLOCK_K=64,
            num_warps=4,
            num_stages=3,
        )
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            not x.is_cuda
            or x.dtype != torch.float16
            or x.shape[1:] != (256, 20, 20)
            or self.c != 128
        ):
            a, b = self.cv1(x).split((self.c, self.c), dim=1)
            b = b + self.attn(b)
            b = b + self.ffn(b)
            return self.cv2(torch.cat((a, b), 1))

        if self._graph is not None:
            self._graph_input.copy_(x)
            self._graph.replay()
            return self._graph_output

        if not self._graph_ready:
            self._graph_ready = True
            return self._forward_fused(x)

        self._graph_input = torch.empty_like(x)
        self._graph_input.copy_(x)
        torch.cuda.synchronize()
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self._graph_output = self._forward_fused(self._graph_input)
        self._graph.replay()
        return self._graph_output
