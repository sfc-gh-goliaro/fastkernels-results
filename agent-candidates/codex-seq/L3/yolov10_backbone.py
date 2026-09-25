"""YOLOv10 native backbone."""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

from ..L2.yolov10_c2f import YOLOC2f
from ..L2.yolov10_bottleneck import YOLOBottleneck
from ..L2.yolov10_conv import YOLOConv, _conv_bn_silu
from ..L2.yolov10_psa import YOLOPSA
from ..L2.yolov10_scdown import YOLOSCDown
from ..L2.yolov10_sppf import YOLOSPPF


@triton.jit
def _c2f_expand(
    x,
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

    for k0 in range(0, C, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        xv = tl.load(
            x
            + image[:, None] * (C * S)
            + offs_k[None, :] * S
            + pos[:, None],
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < C),
            other=0.0,
        )
        wv = tl.load(
            weight + offs_n[None, :] * C + offs_k[:, None],
            mask=(offs_n[None, :] < N) & (offs_k[:, None] < C),
            other=0.0,
        )
        acc = tl.dot(xv, wv, acc)

    mask_n = offs_n < N
    mean = tl.load(running_mean + offs_n, mask=mask_n, other=0.0)
    variance = tl.load(running_var + offs_n, mask=mask_n, other=1.0)
    scale = tl.load(bn_weight + offs_n, mask=mask_n, other=0.0)
    bias = tl.load(bn_bias + offs_n, mask=mask_n, other=0.0)
    value = (acc - mean[None, :]) * (
        scale * tl.rsqrt(variance + EPS)
    )[None, :] + bias[None, :]
    squared = value * value
    even = squared * (
        0.2395166094 + squared * (-0.0138038741 + squared * 0.0004331403)
    )
    value = tl.maximum(
        -0.28, tl.minimum(0.5 * value + even, tl.maximum(value, 0.0))
    )
    tl.store(
        out
        + image[:, None] * (N * S)
        + offs_n[None, :] * S
        + pos[:, None],
        value,
        mask=(offs_m[:, None] < M) & mask_n[None, :],
    )


@triton.jit
def _c2f_project(
    first,
    branch2,
    branch3,
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
    BRANCHES: tl.constexpr,
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
    K: tl.constexpr = (2 + BRANCHES) * C

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        from_first = offs_k < 2 * C
        from_second = offs_k < 3 * C
        channel = tl.where(
            from_first,
            offs_k,
            tl.where(from_second, offs_k - 2 * C, offs_k - 3 * C),
        )
        first_ptrs = (
            first
            + image[:, None] * (2 * C * S)
            + channel[None, :] * S
            + pos[:, None]
        )
        second_ptrs = (
            branch2
            + image[:, None] * (C * S)
            + channel[None, :] * S
            + pos[:, None]
        )
        third_ptrs = (
            branch3
            + image[:, None] * (C * S)
            + channel[None, :] * S
            + pos[:, None]
        )
        x_ptrs = tl.where(
            from_first[None, :],
            first_ptrs,
            tl.where(from_second[None, :], second_ptrs, third_ptrs),
        )
        xv = tl.load(
            x_ptrs,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        )
        wv = tl.load(
            weight + offs_n[None, :] * K + offs_k[:, None],
            mask=(offs_n[None, :] < N) & (offs_k[:, None] < K),
            other=0.0,
        )
        acc = tl.dot(xv, wv, acc)

    mask_n = offs_n < N
    mean = tl.load(running_mean + offs_n, mask=mask_n, other=0.0)
    variance = tl.load(running_var + offs_n, mask=mask_n, other=1.0)
    scale = tl.load(bn_weight + offs_n, mask=mask_n, other=0.0)
    bias = tl.load(bn_bias + offs_n, mask=mask_n, other=0.0)
    value = (acc - mean[None, :]) * (
        scale * tl.rsqrt(variance + EPS)
    )[None, :] + bias[None, :]
    squared = value * value
    even = squared * (
        0.2395166094 + squared * (-0.0138038741 + squared * 0.0004331403)
    )
    value = tl.maximum(
        -0.28, tl.minimum(0.5 * value + even, tl.maximum(value, 0.0))
    )
    tl.store(
        out
        + image[:, None] * (N * S)
        + offs_n[None, :] * S
        + pos[:, None],
        value,
        mask=(offs_m[:, None] < M) & mask_n[None, :],
    )


class _C2f(nn.Module):
    def __init__(self, c1: int, c2: int, n: int):
        super().__init__()
        self.c = c2 // 2
        self.cv1 = YOLOConv(c1, 2 * self.c, 1, 1)
        self.cv2 = YOLOConv((2 + n) * self.c, c2, 1, 1)
        self.m = nn.ModuleList(
            YOLOBottleneck(
                self.c,
                self.c,
                shortcut=True,
                k=(3, 3),
                e=1.0,
            )
            for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pieces = list(self.cv1(x).chunk(2, 1))
        pieces.extend(block(pieces[-1]) for block in self.m)
        return self.cv2(torch.cat(pieces, 1))


class YOLOv10Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem1 = YOLOConv(3, 16, 3, 2)
        self.stem2 = YOLOConv(16, 32, 3, 2)
        self.stage2 = _C2f(32, 32, n=1)
        self.down3 = YOLOConv(32, 64, 3, 2)
        self.stage3 = _C2f(64, 64, n=2)
        self.down4 = YOLOSCDown(64, 128, 3, 2)
        self.stage4 = _C2f(128, 128, n=2)
        self.down5 = YOLOSCDown(128, 256, 3, 2)
        self.stage5 = _C2f(256, 256, n=1)
        self.sppf = YOLOSPPF(256, 256, 5)
        self.psa = YOLOPSA(256, 256)
        self._graph_ready = set()
        self._graphs = {}

    @staticmethod
    def _downsample(x: torch.Tensor, layer: YOLOConv) -> torch.Tensor:
        if (
            layer.training
            or not x.is_cuda
            or x.dtype != torch.float16
            or not x.is_contiguous()
            or x.ndim != 4
        ):
            return layer(x)

        batch, channels, height, width = x.shape
        out_channels = layer.conv.weight.shape[0]
        out_height = height // 2
        out_width = width // 2
        out = torch.empty(
            (batch, out_channels, out_height, out_width),
            device=x.device,
            dtype=x.dtype,
        )
        if channels == 3:
            block_m, block_n, block_k, warps = 256, 16, 32, 8
        elif batch == 4:
            block_m, block_n, block_k, warps = 64, 64, 64, 4
        else:
            block_m, block_n, block_k, warps = 32, 64, 64, 4
        _conv_bn_silu[
            (
                triton.cdiv(batch * out_height * out_width, block_m),
                triton.cdiv(out_channels, block_n),
            )
        ](
            x,
            layer.conv.weight,
            layer.bn.weight,
            layer.bn.bias,
            layer.bn.running_mean,
            layer.bn.running_var,
            out,
            M=batch * out_height * out_width,
            N=out_channels,
            C=channels,
            H=height,
            W=width,
            OH=out_height,
            OW=out_width,
            KH=3,
            KW=3,
            SH=2,
            SW=2,
            PH=1,
            PW=1,
            EPS=layer.bn.eps,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=warps,
            num_stages=2,
        )
        return out

    @staticmethod
    def _c2f(x: torch.Tensor, layer: _C2f) -> torch.Tensor:
        if (
            layer.training
            or not x.is_cuda
            or x.dtype != torch.float16
            or not x.is_contiguous()
            or x.ndim != 4
            or x.shape[1] not in (32, 64, 128, 256)
        ):
            return layer(x)

        batch, channels, height, width = x.shape
        spatial = height * width
        first = torch.empty(
            (batch, 2 * layer.c, height, width),
            device=x.device,
            dtype=x.dtype,
        )
        if channels <= 32:
            expand_m, expand_n, expand_k = 64, 32, 32
        elif channels == 64:
            expand_m, expand_n, expand_k = 32, 64, 64
        elif channels == 128:
            expand_m, expand_n, expand_k = 16, 64, 128
        else:
            expand_m, expand_n, expand_k = 16, 64, 128
        cv1 = layer.cv1
        _c2f_expand[
            (
                triton.cdiv(batch * spatial, expand_m),
                triton.cdiv(2 * layer.c, expand_n),
            )
        ](
            x,
            cv1.conv.weight,
            cv1.bn.weight,
            cv1.bn.bias,
            cv1.bn.running_mean,
            cv1.bn.running_var,
            first,
            M=batch * spatial,
            S=spatial,
            C=channels,
            N=2 * layer.c,
            EPS=cv1.bn.eps,
            BLOCK_M=expand_m,
            BLOCK_N=expand_n,
            BLOCK_K=expand_k,
            num_warps=4,
            num_stages=3,
        )
        last = first[:, layer.c:]
        branches = []
        for block in layer.m:
            last = block(last)
            branches.append(last)

        out_channels = layer.cv2.conv.weight.shape[0]
        out = torch.empty(
            (batch, out_channels, height, width),
            device=x.device,
            dtype=x.dtype,
        )
        if layer.c == 16:
            block_m, block_n, block_k = 64, 32, 64
        elif layer.c == 32:
            block_m, block_n, block_k = 32, 64, 64
        elif layer.c == 64:
            block_m, block_n, block_k = 32, 64, 128
        else:
            block_m, block_n, block_k = 16, 64, 128
        cv2 = layer.cv2
        _c2f_project[
            (
                triton.cdiv(batch * spatial, block_m),
                triton.cdiv(out_channels, block_n),
            )
        ](
            first,
            branches[0],
            branches[1] if len(branches) == 2 else branches[0],
            cv2.conv.weight,
            cv2.bn.weight,
            cv2.bn.bias,
            cv2.bn.running_mean,
            cv2.bn.running_var,
            out,
            M=batch * spatial,
            S=spatial,
            C=layer.c,
            N=out_channels,
            BRANCHES=len(branches),
            EPS=cv2.bn.eps,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=4,
            num_stages=3,
        )
        return out

    def _forward_impl(self, x: torch.Tensor):
        x = self._downsample(x, self.stem1)
        x = self.stem2(x)
        p2 = self._c2f(x, self.stage2)
        x = self._downsample(p2, self.down3)
        p3 = self._c2f(x, self.stage3)
        x = self.down4(p3)
        p4 = self._c2f(x, self.stage4)
        x = self.down5(p4)
        p5 = self._c2f(x, self.stage5)
        p5 = self.sppf(p5)
        # The frozen PSA owns a smaller graph cache. Keep capture ownership at
        # this level so its graph setup is not nested inside ours.
        if (
            not self.training
            and p5.is_cuda
            and p5.dtype == torch.float16
            and p5.shape[1:] == (256, 20, 20)
        ):
            p5 = self.psa._forward_fused(p5)
        else:
            p5 = self.psa(p5)
        return {"p3_backbone": p3, "p4_backbone": p4, "p5_backbone": p5}

    def forward(self, x: torch.Tensor):
        captured = (
            not self.training
            and x.is_cuda
            and x.dtype == torch.float16
            and x.is_contiguous()
            and x.ndim == 4
            and x.shape[1:] == (3, 640, 640)
            and x.shape[0] in (1, 4)
        )
        if not captured:
            return self._forward_impl(x)

        key = (tuple(x.shape), x.device)
        cached = self._graphs.get(key)
        if cached is not None:
            graph, graph_input, graph_output = cached
            graph_input.copy_(x)
            graph.replay()
            return graph_output

        if key not in self._graph_ready:
            self._graph_ready.add(key)
            return self._forward_impl(x)

        graph_input = torch.empty_like(x)
        graph_input.copy_(x)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            self._forward_impl(graph_input)
        stream.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            graph_output = self._forward_impl(graph_input)
        torch.cuda.current_stream().wait_stream(stream)
        graph.replay()
        self._graphs[key] = (graph, graph_input, graph_output)
        return graph_output
