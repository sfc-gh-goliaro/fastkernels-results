"""Fused YOLOv10 partial self-attention for the captured 20x20 workload."""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

from .yolov10_attention import YOLOAttention
from .yolov10_conv import YOLOConv


@triton.jit
def _pointwise(
    x,
    weight,
    bn_weight,
    bn_bias,
    running_mean,
    running_var,
    out,
    residual,
    X_BATCH_STRIDE: tl.constexpr,
    R_BATCH_STRIDE: tl.constexpr,
    TOTAL: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    ACT: tl.constexpr,
    ADD: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    om = pid_m * BM + tl.arange(0, BM)
    on = pid_n * BN + tl.arange(0, BN)
    ok = tl.arange(0, BK)

    batch = on // 400
    spatial = on - batch * 400
    w_ptrs = weight + om[:, None] * C_IN + ok[None, :]
    x_ptrs = x + batch[None, :] * X_BATCH_STRIDE + ok[:, None] * 400 + spatial[None, :]
    acc = tl.dot(
        tl.load(w_ptrs, mask=(om[:, None] < C_OUT) & (ok[None, :] < C_IN), other=0.0),
        tl.load(x_ptrs, mask=(on[None, :] < TOTAL) & (ok[:, None] < C_IN), other=0.0),
    )

    # The reference materializes the convolution in fp16 before batch norm.
    acc = acc.to(tl.float16).to(tl.float32)
    gamma = tl.load(bn_weight + om, mask=om < C_OUT, other=0.0).to(tl.float32)
    beta = tl.load(bn_bias + om, mask=om < C_OUT, other=0.0).to(tl.float32)
    mean = tl.load(running_mean + om, mask=om < C_OUT, other=0.0)
    var = tl.load(running_var + om, mask=om < C_OUT, other=1.0)
    y = (acc - mean[:, None]) * tl.rsqrt(var[:, None] + 1.0e-3)
    y = y * gamma[:, None] + beta[:, None]
    y = y.to(tl.float16).to(tl.float32)
    if ACT:
        y = y * tl.sigmoid(y)
    if ADD:
        r_ptrs = residual + batch[None, :] * R_BATCH_STRIDE + om[:, None] * 400 + spatial[None, :]
        y += tl.load(r_ptrs, mask=(om[:, None] < C_OUT) & (on[None, :] < TOTAL))
    out_ptrs = out + batch[None, :] * C_OUT * 400 + om[:, None] * 400 + spatial[None, :]
    tl.store(out_ptrs, y, mask=(om[:, None] < C_OUT) & (on[None, :] < TOTAL))


@triton.jit
def _pointwise_split(
    a,
    b,
    weight,
    bn_weight,
    bn_bias,
    running_mean,
    running_var,
    out,
    TOTAL: tl.constexpr,
    ACT: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    om = pid_m * BM + tl.arange(0, BM)
    on = pid_n * BN + tl.arange(0, BN)
    ok = tl.arange(0, BK)
    batch = on // 400
    spatial = on - batch * 400

    wa = tl.load(weight + om[:, None] * 256 + ok[None, :], mask=om[:, None] < 256)
    wb = tl.load(weight + om[:, None] * 256 + 128 + ok[None, :], mask=om[:, None] < 256)
    ap = a + batch[None, :] * 256 * 400 + ok[:, None] * 400 + spatial[None, :]
    bp = b + batch[None, :] * 128 * 400 + ok[:, None] * 400 + spatial[None, :]
    mask = on[None, :] < TOTAL
    acc = tl.dot(wa, tl.load(ap, mask=mask)) + tl.dot(wb, tl.load(bp, mask=mask))
    acc = acc.to(tl.float16).to(tl.float32)

    gamma = tl.load(bn_weight + om, mask=om < 256, other=0.0).to(tl.float32)
    beta = tl.load(bn_bias + om, mask=om < 256, other=0.0).to(tl.float32)
    mean = tl.load(running_mean + om, mask=om < 256, other=0.0)
    var = tl.load(running_var + om, mask=om < 256, other=1.0)
    y = (acc - mean[:, None]) * tl.rsqrt(var[:, None] + 1.0e-3)
    y = y * gamma[:, None] + beta[:, None]
    y = y.to(tl.float16).to(tl.float32)
    if ACT:
        y = y * tl.sigmoid(y)
    op = out + batch[None, :] * 256 * 400 + om[:, None] * 400 + spatial[None, :]
    tl.store(op, y, mask=(om[:, None] < 256) & (on[None, :] < TOTAL))


@triton.jit
def _attention_pe(
    qkv,
    pe_weight,
    pe_bn_weight,
    pe_bn_bias,
    pe_mean,
    pe_var,
    out,
    BQ: tl.constexpr,
    BK: tl.constexpr,
    DV: tl.constexpr,
):
    q_block = tl.program_id(0)
    bh = tl.program_id(1)
    batch = bh // 2
    head = bh - batch * 2
    oq = q_block * BQ + tl.arange(0, BQ)
    dk = tl.arange(0, 32)
    dv = tl.arange(0, DV)
    qbase = batch * 256 * 400 + head * 128 * 400

    q = tl.load(
        qkv + qbase + dk[None, :] * 400 + oq[:, None],
        mask=oq[:, None] < 400,
        other=0.0,
    )
    q = (q * 0.1767766952966369).to(tl.float16)
    m_i = tl.full((BQ,), -float("inf"), tl.float32)
    l_i = tl.zeros((BQ,), tl.float32)
    acc = tl.zeros((BQ, DV), tl.float32)

    for start in range(0, 400, BK):
        ok = start + tl.arange(0, BK)
        k = tl.load(
            qkv + qbase + (32 + dk[:, None]) * 400 + ok[None, :],
            mask=ok[None, :] < 400,
            other=0.0,
        )
        scores = tl.dot(q, k)
        scores = tl.where(ok[None, :] < 400, scores, -float("inf"))
        scores *= 1.4426950408889634
        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        p = tl.exp2(scores - m_ij[:, None])
        alpha = tl.exp2(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        v = tl.load(
            qkv + qbase + (64 + dv[None, :]) * 400 + ok[:, None],
            mask=ok[:, None] < 400,
            other=0.0,
        )
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v)
        m_i = m_ij

    attn = (acc / l_i[:, None]).to(tl.float16).to(tl.float32)

    # Add the depthwise 3x3 positional encoding of V.
    channel = head * 64 + dv
    row = oq // 20
    col = oq - row * 20
    pe = tl.zeros((BQ, DV), tl.float32)
    for dy in tl.static_range(-1, 2):
        for dx in tl.static_range(-1, 2):
            nr = row + dy
            nc = col + dx
            ns = nr * 20 + nc
            valid = (oq[:, None] < 400) & (nr[:, None] >= 0) & (nr[:, None] < 20)
            valid &= (nc[:, None] >= 0) & (nc[:, None] < 20)
            vv = tl.load(
                qkv + qbase + (64 + dv[None, :]) * 400 + ns[:, None],
                mask=valid,
                other=0.0,
            )
            ww = tl.load(pe_weight + channel * 9 + (dy + 1) * 3 + dx + 1)
            pe += vv * ww[None, :]
    pe = pe.to(tl.float16).to(tl.float32)
    gamma = tl.load(pe_bn_weight + channel).to(tl.float32)
    beta = tl.load(pe_bn_bias + channel).to(tl.float32)
    mean = tl.load(pe_mean + channel)
    var = tl.load(pe_var + channel)
    pe = ((pe - mean[None, :]) * tl.rsqrt(var[None, :] + 1.0e-3)
          * gamma[None, :] + beta[None, :])
    pe = pe.to(tl.float16).to(tl.float32)

    op = out + batch * 128 * 400 + channel[None, :] * 400 + oq[:, None]
    tl.store(op, attn + pe, mask=oq[:, None] < 400)


def _pw(x, layer, residual=None, act=False, out=None):
    batch = x.shape[0]
    c_out, c_in = layer.conv.weight.shape[:2]
    if out is None:
        out = torch.empty((batch, c_out, 20, 20), device=x.device, dtype=x.dtype)
    total = batch * 400
    block_n = 32 if batch == 1 else 64
    _pointwise[(triton.cdiv(c_out, 16), triton.cdiv(total, block_n))](
        x,
        layer.conv.weight,
        layer.bn.weight,
        layer.bn.bias,
        layer.bn.running_mean,
        layer.bn.running_var,
        out,
        residual if residual is not None else out,
        X_BATCH_STRIDE=x.stride(0),
        R_BATCH_STRIDE=(residual.stride(0) if residual is not None else out.stride(0)),
        TOTAL=total,
        C_IN=c_in,
        C_OUT=c_out,
        ACT=act,
        ADD=residual is not None,
        BM=16,
        BN=block_n,
        BK=triton.next_power_of_2(c_in),
        num_warps=4,
        num_stages=2,
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
        self._graph_ab = None
        self._graph_output = None

    def _forward_rest(self, ab):
        qkv = _pw(ab[:, 128:], self.attn.qkv)
        attended = torch.empty(
            (ab.shape[0], 128, 20, 20), device=ab.device, dtype=ab.dtype
        )
        _attention_pe[(triton.cdiv(400, 16), ab.shape[0] * 2)](
            qkv,
            self.attn.pe.conv.weight,
            self.attn.pe.bn.weight,
            self.attn.pe.bn.bias,
            self.attn.pe.bn.running_mean,
            self.attn.pe.bn.running_var,
            attended,
            BQ=16,
            BK=64,
            DV=64,
            num_warps=4,
            num_stages=2,
        )
        b0 = _pw(attended, self.attn.proj, residual=ab[:, 128:])
        hidden = _pw(b0, self.ffn[0], act=True)
        b1 = _pw(hidden, self.ffn[1], residual=b0)

        out = torch.empty_like(ab)
        total = ab.shape[0] * 400
        block_n = 32 if ab.shape[0] == 1 else 64
        _pointwise_split[(16, triton.cdiv(total, block_n))](
            ab,
            b1,
            self.cv2.conv.weight,
            self.cv2.bn.weight,
            self.cv2.bn.bias,
            self.cv2.bn.running_mean,
            self.cv2.bn.running_var,
            out,
            TOTAL=total,
            ACT=True,
            BM=16,
            BN=block_n,
            BK=128,
            num_warps=4,
            num_stages=2,
        )
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.float16 or x.shape[1:] != (256, 20, 20):
            a, b = self.cv1(x).split((self.c, self.c), dim=1)
            b = b + self.attn(b)
            b = b + self.ffn(b)
            return self.cv2(torch.cat((a, b), 1))

        if self._graph is None:
            self._graph_ab = torch.empty_like(x)
            _pw(x, self.cv1, act=True, out=self._graph_ab)
            self._forward_rest(self._graph_ab)
            torch.cuda.synchronize()
            self._graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._graph):
                self._graph_output = self._forward_rest(self._graph_ab)
            self._graph.replay()
            return self._graph_output

        _pw(x, self.cv1, act=True, out=self._graph_ab)
        self._graph.replay()
        return self._graph_output
