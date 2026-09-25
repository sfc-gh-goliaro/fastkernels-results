"""Specialized YOLOv10 Compact Inverted Block."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

from .yolov10_conv import YOLOConv
from .yolov10_repvggdw import YOLORepVGGDW


@triton.jit
def _depthwise3(
    x,
    residual,
    weight,
    bias,
    out,
    C: tl.constexpr,
    S: tl.constexpr,
    ADD_INPUT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tiles: tl.constexpr = tl.cdiv(S, BLOCK)
    pid = tl.program_id(0)
    tile = pid % tiles
    bc = pid // tiles
    channel = bc % C
    batch = bc // C
    pos = tile * BLOCK + tl.arange(0, BLOCK)
    h = pos // 20
    w = pos - h * 20
    mask = pos < S
    acc = tl.zeros((BLOCK,), tl.float32)

    for kh in range(3):
        ih = h + kh - 1
        for kw in range(3):
            iw = w + kw - 1
            valid = mask & (ih >= 0) & (ih < 20) & (iw >= 0) & (iw < 20)
            offsets = (batch * C + channel) * S + ih * 20 + iw
            xv = tl.load(x + offsets, mask=valid, other=0.0)
            wv = tl.load(weight + channel * 9 + kh * 3 + kw)
            acc += xv.to(tl.float32) * wv.to(tl.float32)

    value = acc + tl.load(bias + channel).to(tl.float32)
    value = value.to(tl.float16).to(tl.float32)
    value = value * tl.sigmoid(value)
    offsets = (batch * C + channel) * S + pos
    if ADD_INPUT:
        value = value.to(tl.float16).to(tl.float32)
        value += tl.load(residual + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(out + offsets, value, mask=mask)


@triton.jit
def _pointwise(
    x,
    weight,
    bias,
    out,
    C: tl.constexpr,
    N: tl.constexpr,
    S: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pos = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cout = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    batch = tl.program_id(2)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    for k0 in range(0, C, BLOCK_K):
        cin = k0 + tl.arange(0, BLOCK_K)
        a = tl.load(
            x + batch * C * S + cin[None, :] * S + pos[:, None],
            mask=(pos[:, None] < S) & (cin[None, :] < C),
            other=0.0,
        )
        b = tl.load(
            weight + cout[None, :] * C + cin[:, None],
            mask=(cout[None, :] < N) & (cin[:, None] < C),
            other=0.0,
        )
        acc = tl.dot(a, b, acc)

    value = acc + tl.load(bias + cout, mask=cout < N, other=0.0)[None, :]
    value = value.to(tl.float16).to(tl.float32)
    value = value * tl.sigmoid(value)
    offsets = batch * N * S + cout[None, :] * S + pos[:, None]
    tl.store(
        out + offsets,
        value,
        mask=(pos[:, None] < S) & (cout[None, :] < N),
    )


@triton.jit
def _repvgg_depthwise(
    x,
    weight,
    bias,
    out,
    C: tl.constexpr,
    S: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tiles: tl.constexpr = tl.cdiv(S, BLOCK)
    pid = tl.program_id(0)
    tile = pid % tiles
    bc = pid // tiles
    channel = bc % C
    batch = bc // C
    pos = tile * BLOCK + tl.arange(0, BLOCK)
    h = pos // 20
    w = pos - h * 20
    mask = pos < S
    acc = tl.zeros((BLOCK,), tl.float32)

    for kh in range(7):
        ih = h + kh - 3
        for kw in range(7):
            iw = w + kw - 3
            valid = mask & (ih >= 0) & (ih < 20) & (iw >= 0) & (iw < 20)
            offsets = (batch * C + channel) * S + ih * 20 + iw
            xv = tl.load(x + offsets, mask=valid, other=0.0)
            wv = tl.load(weight + channel * 49 + kh * 7 + kw)
            acc += xv.to(tl.float32) * wv.to(tl.float32)

    value = acc + tl.load(bias + channel).to(tl.float32)
    value = value.to(tl.float16).to(tl.float32)
    value = value * tl.sigmoid(value)
    offsets = (batch * C + channel) * S + pos
    tl.store(out + offsets, value, mask=mask)


def _fold_conv_bn(layer: YOLOConv):
    bn = layer.bn
    scale = bn.weight.float() * torch.rsqrt(bn.running_var.float() + bn.eps)
    shape = (scale.shape[0],) + (1,) * (layer.conv.weight.ndim - 1)
    weight = (layer.conv.weight.float() * scale.view(shape)).to(
        layer.conv.weight.dtype
    )
    bias = (
        bn.bias.float() - bn.running_mean.float() * scale
    ).to(layer.conv.weight.dtype)
    return weight, bias


class YOLOCIB(nn.Module):
    def __init__(self, c1: int, c2: int, shortcut: bool = True, e: float = 0.5, lk: bool = False):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = nn.Sequential(
            YOLOConv(c1, c1, 3, g=c1),
            YOLOConv(c1, 2 * c_, 1),
            YOLOConv(2 * c_, 2 * c_, 3, g=2 * c_) if not lk else YOLORepVGGDW(2 * c_),
            YOLOConv(2 * c_, c2, 1),
            YOLOConv(c2, c2, 3, g=c2),
        )
        self.add = shortcut and c1 == c2
        self._specialized = c1 == 128 and c2 == 128 and e == 1.0 and lk and self.add
        self._graph = None
        self._graph_input = None
        self._graph_output = None
        self._rep_weight = None
        self._rep_bias = None
        self._folded = None

    def _prepare_weights(self):
        if self._folded is not None:
            return
        first, expand, rep, project, last = self.cv1
        self._folded = (
            _fold_conv_bn(first),
            _fold_conv_bn(expand),
            _fold_conv_bn(project),
            _fold_conv_bn(last),
        )
        conv7 = rep.conv
        conv3 = rep.conv1
        bn7 = conv7.bn
        bn3 = conv3.bn
        scale7 = bn7.weight.float() * torch.rsqrt(bn7.running_var.float() + bn7.eps)
        scale3 = bn3.weight.float() * torch.rsqrt(bn3.running_var.float() + bn3.eps)
        weight7 = conv7.conv.weight.float() * scale7[:, None, None, None]
        weight3 = conv3.conv.weight.float() * scale3[:, None, None, None]
        self._rep_weight = (weight7 + F.pad(weight3, (2, 2, 2, 2))).to(
            conv7.conv.weight.dtype
        )
        self._rep_bias = (
            bn7.bias.float()
            - bn7.running_mean.float() * scale7
            + bn3.bias.float()
            - bn3.running_mean.float() * scale3
        ).to(conv7.conv.weight.dtype)

    def _run_specialized(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        first, expand, rep, project, last = self.cv1
        (first_w, first_b), (expand_w, expand_b), (project_w, project_b), (
            last_w,
            last_b,
        ) = self._folded
        a = torch.empty_like(x)
        _depthwise3[(batch * 128 * 4,)](
            x, x, first_w, first_b, a,
            C=128, S=400, ADD_INPUT=False, BLOCK=128, num_warps=4,
        )

        b = torch.empty((batch, 256, 20, 20), device=x.device, dtype=x.dtype)
        expand_m = 16 if batch == 1 else 32
        project_m = 16 if batch == 1 else 32
        pw_n = 32
        pw_warps = 2 if batch == 1 else 4
        pw_k = 64 if batch == 1 else 128
        _pointwise[(triton.cdiv(400, expand_m), triton.cdiv(256, pw_n), batch)](
            a, expand_w, expand_b, b,
            C=128, N=256, S=400,
            BLOCK_M=expand_m, BLOCK_N=pw_n, BLOCK_K=pw_k,
            num_warps=pw_warps, num_stages=3,
        )

        c = torch.empty_like(b)
        _repvgg_depthwise[(batch * 256 * 4,)](
            b,
            self._rep_weight,
            self._rep_bias,
            c,
            C=256, S=400, BLOCK=128, num_warps=4,
        )

        d = torch.empty_like(x)
        _pointwise[(triton.cdiv(400, project_m), triton.cdiv(128, pw_n), batch)](
            c, project_w, project_b, d,
            C=256, N=128, S=400,
            BLOCK_M=project_m, BLOCK_N=pw_n, BLOCK_K=pw_k,
            num_warps=pw_warps, num_stages=3,
        )

        out = torch.empty_like(x)
        _depthwise3[(batch * 128 * 4,)](
            d, x, last_w, last_b, out,
            C=128, S=400, ADD_INPUT=True, BLOCK=128, num_warps=4,
        )
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            not self._specialized
            or self.training
            or not x.is_cuda
            or x.dtype != torch.float16
            or x.ndim != 4
            or tuple(x.shape[1:]) != (128, 20, 20)
        ):
            y = self.cv1(x)
            return x + y if self.add else y

        if self._graph is None:
            self._prepare_weights()
            self._graph_input = torch.empty_like(x)
            self._graph_input.copy_(x)

            # Compile and populate allocator pools before graph capture.
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):
                    self._run_specialized(self._graph_input)
            stream.synchronize()

            self._graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._graph, stream=stream):
                self._graph_output = self._run_specialized(self._graph_input)
            torch.cuda.current_stream().wait_stream(stream)
            self._graph.replay()
        else:
            self._graph_input.copy_(x)
            self._graph.replay()
        return self._graph_output
