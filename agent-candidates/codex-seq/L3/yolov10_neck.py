"""YOLOv10 native neck."""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

from ..L1.interpolate import Interpolate
from ..L2.yolov10_c2f import YOLOC2f, YOLOC2fCIB
from ..L2.yolov10_concat import YOLOConcat
from ..L2.yolov10_conv import YOLOConv
from ..L2.yolov10_scdown import YOLOSCDown


@triton.jit
def _bias_silu_kernel(
    x,
    bias,
    residual,
    out,
    n_elements,
    SPATIAL: tl.constexpr,
    CHANNELS: tl.constexpr,
    RESIDUAL_STRIDE: tl.constexpr,
    ADD_RESIDUAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    channel = (offsets // SPATIAL) % CHANNELS
    value = tl.load(x + offsets, mask=mask).to(tl.float32)
    value += tl.load(bias + channel, mask=mask).to(tl.float32)
    value *= tl.sigmoid(value)
    if ADD_RESIDUAL:
        value = value.to(tl.float16).to(tl.float32)
        image = offsets // (CHANNELS * SPATIAL)
        residual_offsets = (
            image * RESIDUAL_STRIDE
            + channel * SPATIAL
            + offsets % SPATIAL
        )
        value += tl.load(residual + residual_offsets, mask=mask).to(tl.float32)
    tl.store(out + offsets, value, mask=mask)


class _BiasSiLU(nn.Module):
    def __init__(self, bias: torch.Tensor):
        super().__init__()
        self.bias = nn.Parameter(bias)

    def forward(self, x: torch.Tensor):
        out = torch.empty_like(x)
        spatial = x.shape[2] * x.shape[3]
        _bias_silu_kernel[
            (triton.cdiv(x.numel(), 1024),)
        ](
            x,
            self.bias,
            x,
            out,
            x.numel(),
            SPATIAL=spatial,
            CHANNELS=self.bias.numel(),
            RESIDUAL_STRIDE=x.stride(0),
            ADD_RESIDUAL=False,
            BLOCK=1024,
            num_warps=4,
        )
        return out


@triton.jit
def _split_pointwise_silu_kernel(
    x0,
    x1,
    x2,
    weight,
    bias,
    out,
    M: tl.constexpr,
    N: tl.constexpr,
    C0: tl.constexpr,
    C1: tl.constexpr,
    C2: tl.constexpr,
    S: tl.constexpr,
    W: tl.constexpr,
    S0: tl.constexpr,
    W0: tl.constexpr,
    STRIDE0: tl.constexpr,
    STRIDE1: tl.constexpr,
    STRIDE2: tl.constexpr,
    UPSAMPLE: tl.constexpr,
    THREE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    image = offs_m // S
    pos = offs_m % S
    if UPSAMPLE:
        h = pos // W
        source_pos = (h // 2) * W0 + (pos % W) // 2
    else:
        source_pos = pos
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    K: tl.constexpr = C0 + C1 + C2

    for k0 in range(0, K, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        mask_m = offs_m[:, None] < M
        a0 = tl.load(
            x0
            + image[:, None] * STRIDE0
            + k[None, :] * S0
            + source_pos[:, None],
            mask=mask_m & (k[None, :] < C0),
            other=0.0,
        )
        k1 = k - C0
        a1 = tl.load(
            x1
            + image[:, None] * STRIDE1
            + k1[None, :] * S
            + pos[:, None],
            mask=mask_m & (k1[None, :] >= 0) & (k1[None, :] < C1),
            other=0.0,
        )
        if THREE:
            k2 = k1 - C1
            a2 = tl.load(
                x2
                + image[:, None] * STRIDE2
                + k2[None, :] * S
                + pos[:, None],
                mask=mask_m & (k2[None, :] >= 0) & (k2[None, :] < C2),
                other=0.0,
            )
            a = a0 + a1 + a2
        else:
            a = a0 + a1
        b = tl.load(
            weight + offs_n[None, :] * K + k[:, None],
            mask=(offs_n[None, :] < N) & (k[:, None] < K),
            other=0.0,
        )
        acc = tl.dot(a, b, acc)

    value = acc + tl.load(
        bias + offs_n, mask=offs_n < N, other=0.0
    )[None, :]
    value *= tl.sigmoid(value)
    tl.store(
        out + image[:, None] * N * S + offs_n[None, :] * S + pos[:, None],
        value,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


class YOLOv10Neck(nn.Module):
    def __init__(self):
        super().__init__()
        self._upsample = Interpolate()
        self.cat1 = YOLOConcat(1)
        self.c2f_p4 = YOLOC2f(384, 128, n=1, shortcut=False)
        self.cat2 = YOLOConcat(1)
        self.c2f_p3 = YOLOC2f(192, 64, n=1, shortcut=False)
        self.down_p3 = YOLOConv(64, 64, 3, 2)
        self.cat3 = YOLOConcat(1)
        self.c2f_n4 = YOLOC2f(192, 128, n=1, shortcut=False)
        self.down_n4 = YOLOSCDown(128, 128, 3, 2)
        self.cat4 = YOLOConcat(1)
        self.c2fcib_n5 = YOLOC2fCIB(384, 256, n=1, shortcut=True, lk=True)
        self._graph_cache = {}
        self._c2f_fused = False

    @staticmethod
    def _fuse_c2f_block(module: nn.Module):
        for child in list(module.children()):
            YOLOv10Neck._fuse_c2f_block(child)
        name = module.__class__.__name__
        if name in ("YOLOConv", "YOLORepVGGDW"):
            module.fuse()
        if name == "YOLOConv" and module.act.__class__.__name__ == "SiLU":
            bias = module.conv.bias
            module.conv.bias = None
            module.act = _BiasSiLU(bias)
        elif name == "YOLORepVGGDW":
            bias = module.conv.conv.bias
            module.conv.conv.bias = None
            module.act = _BiasSiLU(bias)

    def _prepare_c2f_blocks(self):
        if self._c2f_fused:
            return
        for block in (
            self.c2f_p4,
            self.c2f_p3,
            self.c2f_n4,
            self.c2fcib_n5,
        ):
            self._fuse_c2f_block(block)
        self._c2f_fused = True

    @staticmethod
    def _split_pointwise(layer, x0, x1, x2=None, upsample=False):
        batch = x1.shape[0]
        height, width = x1.shape[2:]
        spatial = height * width
        channels = layer.conv.weight.shape[0]
        c0, c1 = x0.shape[1], x1.shape[1]
        c2 = 0 if x2 is None else x2.shape[1]
        out = torch.empty(
            (batch, channels, height, width), device=x1.device, dtype=x1.dtype
        )
        if spatial == 400:
            block_m, block_n, block_k = 16, 64, 128
        elif spatial == 1600:
            block_m, block_n, block_k = 32, 64, 64
        else:
            block_m, block_n, block_k = 32, 64, 64
        _split_pointwise_silu_kernel[
            (
                triton.cdiv(batch * spatial, block_m),
                triton.cdiv(channels, block_n),
            )
        ](
            x0,
            x1,
            x1 if x2 is None else x2,
            layer.conv.weight,
            layer.act.bias,
            out,
            M=batch * spatial,
            N=channels,
            C0=c0,
            C1=c1,
            C2=c2,
            S=spatial,
            W=width,
            S0=x0.shape[2] * x0.shape[3],
            W0=x0.shape[3],
            STRIDE0=x0.stride(0),
            STRIDE1=x1.stride(0),
            STRIDE2=0 if x2 is None else x2.stride(0),
            UPSAMPLE=upsample,
            THREE=x2 is not None,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=4,
            num_stages=3,
        )
        return out

    @staticmethod
    def _run_cib(cib, x):
        value = x
        layers = list(cib.cv1)
        for layer in layers[:-1]:
            value = layer(value)
        last = layers[-1]
        value = last.conv(value)
        out = torch.empty_like(value)
        spatial = value.shape[2] * value.shape[3]
        _bias_silu_kernel[(triton.cdiv(value.numel(), 1024),)](
            value,
            last.act.bias,
            x,
            out,
            value.numel(),
            SPATIAL=spatial,
            CHANNELS=value.shape[1],
            RESIDUAL_STRIDE=x.stride(0),
            ADD_RESIDUAL=True,
            BLOCK=1024,
            num_warps=4,
        )
        return out

    def _run_c2f(self, block, x0, x1, upsample=False):
        hidden = self._split_pointwise(block.cv1, x0, x1, upsample=upsample)
        first, second = hidden.chunk(2, 1)
        inner_block = block.m[0]
        if inner_block.__class__.__name__ == "YOLOCIB":
            inner = self._run_cib(inner_block, second)
        else:
            inner = inner_block(second)
        return self._split_pointwise(block.cv2, first, second, inner)

    def _forward_captured(self, feats: dict[str, torch.Tensor]):
        p3_backbone = feats["p3_backbone"]
        p4_backbone = feats["p4_backbone"]
        p5_backbone = feats["p5_backbone"]

        p4 = self._run_c2f(
            self.c2f_p4, p5_backbone, p4_backbone, upsample=True
        )
        p3 = self._run_c2f(
            self.c2f_p3, p4, p3_backbone, upsample=True
        )
        down_p3 = self.down_p3(p3)
        n4 = self._run_c2f(self.c2f_n4, down_p3, p4)
        down_n4 = self.down_n4(n4)
        n5 = self._run_c2f(self.c2fcib_n5, down_n4, p5_backbone)
        return [p3, n4, n5]

    def _forward_impl(self, feats: dict[str, torch.Tensor], capture: bool = False):
        if capture:
            return self._forward_captured(feats)
        p3_backbone = feats["p3_backbone"]
        p4_backbone = feats["p4_backbone"]
        p5_backbone = feats["p5_backbone"]

        x = self._upsample(p5_backbone, scale_factor=2.0, mode="nearest")
        x = self.cat1([x, p4_backbone])
        p4 = self.c2f_p4(x)

        x = self._upsample(p4, scale_factor=2.0, mode="nearest")
        x = self.cat2([x, p3_backbone])
        p3 = self.c2f_p3(x)

        x = self.down_p3(p3)
        x = self.cat3([x, p4])
        n4 = self.c2f_n4(x)

        x = self.down_n4(n4)
        x = self.cat4([x, p5_backbone])
        n5 = self.c2fcib_n5(x)
        return [p3, n4, n5]

    def forward(self, feats: dict[str, torch.Tensor]):
        p3 = feats["p3_backbone"]
        p4 = feats["p4_backbone"]
        p5 = feats["p5_backbone"]
        supported = (
            not self.training
            and p3.is_cuda
            and p3.dtype == torch.float16
            and p3.is_contiguous()
            and p4.is_contiguous()
            and p5.is_contiguous()
            and tuple(p3.shape[1:]) == (64, 80, 80)
            and tuple(p4.shape) == (p3.shape[0], 128, 40, 40)
            and tuple(p5.shape) == (p3.shape[0], 256, 20, 20)
            and p3.shape[0] in (1, 4)
        )
        if not supported:
            return self._forward_impl(feats)

        key = (p3.shape[0], p3.device.index)
        cached = self._graph_cache.get(key)
        if cached is None:
            self._prepare_c2f_blocks()
            static_feats = {
                "p3_backbone": torch.empty_like(p3),
                "p4_backbone": torch.empty_like(p4),
                "p5_backbone": torch.empty_like(p5),
            }
            torch._foreach_copy_(
                list(static_feats.values()),
                [p3, p4, p5],
            )

            # Populate Triton caches and lazy per-module state before capture.
            self._forward_impl(static_feats, capture=True)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                outputs = self._forward_impl(static_feats, capture=True)
            cached = (graph, static_feats, outputs)
            self._graph_cache[key] = cached

        graph, static_feats, outputs = cached
        torch._foreach_copy_(
            list(static_feats.values()),
            [p3, p4, p5],
        )
        graph.replay()
        return outputs
