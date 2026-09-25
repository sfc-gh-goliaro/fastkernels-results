"""YOLOv10 CIB (Compact Inverted Block)."""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

from .yolov10_conv import YOLOConv
from .yolov10_repvggdw import YOLORepVGGDW


@triton.jit
def _dw3_kernel(
    x, weight, gamma, beta, mean, var, residual, out,
    n_elements: tl.constexpr,
    ADD_RESIDUAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offs < n_elements
    spatial = offs % 400
    channel_batch = offs // 400
    row = spatial // 20
    col = spatial % 20
    channel = channel_batch % 128
    base = channel_batch * 400

    acc = tl.zeros((BLOCK,), tl.float32)
    for ky in range(3):
        iy = row + ky - 1
        for kx in range(3):
            ix = col + kx - 1
            mask = valid & (iy >= 0) & (iy < 20) & (ix >= 0) & (ix < 20)
            xv = tl.load(x + base + iy * 20 + ix, mask=mask, other=0.0)
            wv = tl.load(weight + channel * 9 + ky * 3 + kx, mask=valid)
            acc += xv * wv

    conv = acc.to(tl.float16)
    norm = (conv - tl.load(mean + channel, mask=valid)) * tl.rsqrt(
        tl.load(var + channel, mask=valid) + 1.0e-3
    )
    norm = norm * tl.load(gamma + channel, mask=valid) + tl.load(
        beta + channel, mask=valid
    )
    activated = norm * tl.sigmoid(norm)
    if ADD_RESIDUAL:
        activated = activated.to(tl.float16) + tl.load(residual + offs, mask=valid)
    tl.store(out + offs, activated, mask=valid)


@triton.jit
def _repdw_kernel(
    x,
    weight7, gamma7, beta7, mean7, var7,
    weight3, gamma3, beta3, mean3, var3,
    out,
    n_elements: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offs < n_elements
    spatial = offs % 400
    channel_batch = offs // 400
    row = spatial // 20
    col = spatial % 20
    channel = channel_batch % 256
    base = channel_batch * 400

    acc7 = tl.zeros((BLOCK,), tl.float32)
    for ky in range(7):
        iy = row + ky - 3
        for kx in range(7):
            ix = col + kx - 3
            mask = valid & (iy >= 0) & (iy < 20) & (ix >= 0) & (ix < 20)
            xv = tl.load(x + base + iy * 20 + ix, mask=mask, other=0.0)
            wv = tl.load(weight7 + channel * 49 + ky * 7 + kx, mask=valid)
            acc7 += xv * wv

    acc3 = tl.zeros((BLOCK,), tl.float32)
    for ky in range(3):
        iy = row + ky - 1
        for kx in range(3):
            ix = col + kx - 1
            mask = valid & (iy >= 0) & (iy < 20) & (ix >= 0) & (ix < 20)
            xv = tl.load(x + base + iy * 20 + ix, mask=mask, other=0.0)
            wv = tl.load(weight3 + channel * 9 + ky * 3 + kx, mask=valid)
            acc3 += xv * wv

    conv7 = acc7.to(tl.float16)
    norm7 = (conv7 - tl.load(mean7 + channel, mask=valid)) * tl.rsqrt(
        tl.load(var7 + channel, mask=valid) + 1.0e-3
    )
    norm7 = norm7 * tl.load(gamma7 + channel, mask=valid) + tl.load(
        beta7 + channel, mask=valid
    )
    conv3 = acc3.to(tl.float16)
    norm3 = (conv3 - tl.load(mean3 + channel, mask=valid)) * tl.rsqrt(
        tl.load(var3 + channel, mask=valid) + 1.0e-3
    )
    norm3 = norm3 * tl.load(gamma3 + channel, mask=valid) + tl.load(
        beta3 + channel, mask=valid
    )
    summed = norm7.to(tl.float16) + norm3.to(tl.float16)
    summed_f32 = summed.to(tl.float32)
    tl.store(out + offs, summed_f32 * tl.sigmoid(summed_f32), mask=valid)


@triton.jit
def _pointwise_kernel(
    x, weight, gamma, beta, mean, var, out,
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    batch = offs_m // 400
    spatial = offs_m % 400
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k = k_start + offs_k
        a_ptrs = x + batch[:, None] * K * 400 + k[None, :] * 400 + spatial[:, None]
        b_ptrs = weight + offs_n[None, :] * K + k[:, None]
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (k[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_n[None, :] < N) & (k[:, None] < K), other=0.0)
        acc = tl.dot(a, b, acc)

    conv = acc.to(tl.float16)
    g = tl.load(gamma + offs_n, mask=offs_n < N)[None, :]
    be = tl.load(beta + offs_n, mask=offs_n < N)[None, :]
    mu = tl.load(mean + offs_n, mask=offs_n < N)[None, :]
    vv = tl.load(var + offs_n, mask=offs_n < N)[None, :]
    norm = (conv - mu) * tl.rsqrt(vv + 1.0e-3) * g + be
    activated = norm * tl.sigmoid(norm)
    out_ptrs = out + batch[:, None] * N * 400 + offs_n[None, :] * 400 + spatial[:, None]
    tl.store(
        out_ptrs,
        activated,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.jit
def _dw3_pointwise_kernel(
    x,
    dw_weight, dw_gamma, dw_beta, dw_mean, dw_var,
    pw_weight, pw_gamma, pw_beta, pw_mean, pw_var,
    out,
    M: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    batch = offs_m // 400
    spatial = offs_m % 400
    row = spatial // 20
    col = spatial % 20
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    for k_start in range(0, 128, BLOCK_K):
        channel = k_start + offs_k
        dw = tl.zeros((BLOCK_M, BLOCK_K), tl.float32)
        for ky in range(3):
            iy = row + ky - 1
            for kx in range(3):
                ix = col + kx - 1
                x_ptrs = (
                    x
                    + batch[:, None] * 128 * 400
                    + channel[None, :] * 400
                    + iy[:, None] * 20
                    + ix[:, None]
                )
                mask = (
                    (offs_m[:, None] < M)
                    & (iy[:, None] >= 0)
                    & (iy[:, None] < 20)
                    & (ix[:, None] >= 0)
                    & (ix[:, None] < 20)
                )
                xv = tl.load(x_ptrs, mask=mask, other=0.0)
                wv = tl.load(
                    dw_weight + channel * 9 + ky * 3 + kx
                )[None, :]
                dw += xv * wv

        conv = dw.to(tl.float16)
        g0 = tl.load(dw_gamma + channel)[None, :]
        b0 = tl.load(dw_beta + channel)[None, :]
        m0 = tl.load(dw_mean + channel)[None, :]
        v0 = tl.load(dw_var + channel)[None, :]
        norm = (conv - m0) * tl.rsqrt(v0 + 1.0e-3) * g0 + b0
        activated = norm * tl.sigmoid(norm)
        w_ptrs = pw_weight + offs_n[None, :] * 128 + channel[:, None]
        pw = tl.load(w_ptrs, mask=offs_n[None, :] < 256, other=0.0)
        acc = tl.dot(activated.to(tl.float16), pw, acc)

    conv = acc.to(tl.float16)
    g1 = tl.load(pw_gamma + offs_n, mask=offs_n < 256)[None, :]
    b1 = tl.load(pw_beta + offs_n, mask=offs_n < 256)[None, :]
    m1 = tl.load(pw_mean + offs_n, mask=offs_n < 256)[None, :]
    v1 = tl.load(pw_var + offs_n, mask=offs_n < 256)[None, :]
    norm = (conv - m1) * tl.rsqrt(v1 + 1.0e-3) * g1 + b1
    activated = norm * tl.sigmoid(norm)
    out_ptrs = (
        out
        + batch[:, None] * 256 * 400
        + offs_n[None, :] * 400
        + spatial[:, None]
    )
    tl.store(
        out_ptrs,
        activated,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < 256),
    )


def _bn_args(layer):
    bn = layer.bn
    return bn.weight, bn.bias, bn.running_mean, bn.running_var


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
        self._workspace = {}

    def _launch_fast(self, x, buffers):
        batch = x.shape[0]
        n128 = batch * 128 * 400
        n256 = batch * 256 * 400
        y1, y2, y3, out = buffers

        l0 = self.cv1[0]
        l1 = self.cv1[1]
        _dw3_pointwise_kernel[(triton.cdiv(batch * 400, 64), 4)](
            x,
            l0.conv.weight, *_bn_args(l0),
            l1.conv.weight, *_bn_args(l1),
            y1,
            M=batch * 400,
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
            num_warps=8, num_stages=3,
        )

        rep = self.cv1[2]
        r7, r3 = rep.conv, rep.conv1
        _repdw_kernel[(triton.cdiv(n256, 256),)](
            y1,
            r7.conv.weight, *_bn_args(r7),
            r3.conv.weight, *_bn_args(r3),
            y2,
            n_elements=n256, BLOCK=256, num_warps=8,
        )

        l3 = self.cv1[3]
        _pointwise_kernel[(triton.cdiv(batch * 400, 32), 2)](
            y2, l3.conv.weight, *_bn_args(l3), y3,
            M=batch * 400, K=256, N=128,
            BLOCK_M=32, BLOCK_N=64, BLOCK_K=32,
            num_warps=4, num_stages=3,
        )

        l4 = self.cv1[4]
        _dw3_kernel[(triton.cdiv(n128, 256),)](
            y3, l4.conv.weight, *_bn_args(l4), x, out,
            n_elements=n128, ADD_RESIDUAL=self.add, BLOCK=256,
            num_warps=8,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Captured YOLOv10n configuration: 128 -> 128 -> 256 -> 256 -> 128.
        if (
            x.is_cuda
            and x.dtype == torch.float16
            and x.shape[1:] == (128, 20, 20)
            and isinstance(self.cv1[2], YOLORepVGGDW)
        ):
            batch = x.shape[0]
            key = (batch, x.device)
            state = self._workspace.get(key)
            if state is None:
                static_x = torch.empty_like(x)
                buffers = (
                    torch.empty((batch, 256, 20, 20), device=x.device, dtype=x.dtype),
                    torch.empty((batch, 256, 20, 20), device=x.device, dtype=x.dtype),
                    torch.empty_like(x),
                    torch.empty_like(x),
                )
                static_x.copy_(x)
                self._launch_fast(static_x, buffers)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    self._launch_fast(static_x, buffers)
                self._workspace[key] = (static_x, buffers, graph)
                return buffers[-1]

            static_x, buffers, graph = state
            static_x.copy_(x)
            graph.replay()
            return buffers[-1]

        y = self.cv1(x)
        return x + y if self.add else y
