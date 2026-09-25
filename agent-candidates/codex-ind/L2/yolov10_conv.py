"""YOLOv10 Conv-BN-Act building block."""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

from ..L1.batch_norm2d import BatchNorm2d
from ..L1.conv2d import Conv2d
from ..L1.silu import SiLU


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 256, "BLOCK_K": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 32, "BLOCK_K": 32}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 32, "BLOCK_K": 64}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 16, "BLOCK_K": 32}, num_warps=2, num_stages=2),
    ],
    key=["HW", "CIN", "COUT", "L"],
)
@triton.jit
def _conv1x1_bn_act_kernel(
    x,
    weight,
    bn_weight,
    bn_bias,
    running_mean,
    running_var,
    out,
    HW: tl.constexpr,
    CIN: tl.constexpr,
    COUT: tl.constexpr,
    L: tl.constexpr,
    EPS: tl.constexpr,
    DO_ACT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    batch = offs_n // HW
    spatial = offs_n - batch * HW
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, CIN, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        w = tl.load(
            weight + offs_m[:, None] * CIN + offs_k[None, :],
            mask=(offs_m[:, None] < COUT) & (offs_k[None, :] < CIN),
            other=0.0,
        )
        inp = tl.load(
            x
            + batch[None, :] * CIN * HW
            + offs_k[:, None] * HW
            + spatial[None, :],
            mask=(offs_k[:, None] < CIN) & (offs_n[None, :] < L),
            other=0.0,
        )
        acc += tl.dot(w, inp)

    gamma = tl.load(bn_weight + offs_m, mask=offs_m < COUT).to(tl.float32)
    beta = tl.load(bn_bias + offs_m, mask=offs_m < COUT).to(tl.float32)
    mean = tl.load(running_mean + offs_m, mask=offs_m < COUT).to(tl.float32)
    var = tl.load(running_var + offs_m, mask=offs_m < COUT).to(tl.float32)
    y = acc * gamma[:, None] * tl.rsqrt(var[:, None] + EPS)
    y += beta[:, None] - mean[:, None] * gamma[:, None] * tl.rsqrt(var[:, None] + EPS)
    if DO_ACT:
        y *= tl.sigmoid(y)
    out_offs = (
        batch[None, :] * COUT * HW
        + offs_m[:, None] * HW
        + spatial[None, :]
    )
    tl.store(
        out + out_offs,
        y,
        mask=(offs_m[:, None] < COUT) & (offs_n[None, :] < L),
    )


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 256, "BLOCK_K": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 16}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 16}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 16}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 16}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 16}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 16, "BLOCK_K": 64}, num_warps=4, num_stages=3),
    ],
    key=["H", "OH", "CIN", "COUT", "L", "STRIDE"],
)
@triton.jit
def _conv3x3_bn_act_kernel(
    x,
    weight,
    bn_weight,
    bn_bias,
    running_mean,
    running_var,
    out,
    H: tl.constexpr,
    W: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    CIN: tl.constexpr,
    COUT: tl.constexpr,
    L: tl.constexpr,
    STRIDE: tl.constexpr,
    PADDING: tl.constexpr,
    EPS: tl.constexpr,
    DO_ACT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    out_hw = OH * OW
    batch = offs_n // out_hw
    out_spatial = offs_n - batch * out_hw
    oy = out_spatial // OW
    ox = out_spatial - oy * OW
    K = CIN * 9
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        channel = offs_k // 9
        kernel_pos = offs_k - channel * 9
        ky = kernel_pos // 3
        kx = kernel_pos - ky * 3
        iy = oy[None, :] * STRIDE + ky[:, None] - PADDING
        ix = ox[None, :] * STRIDE + kx[:, None] - PADDING
        w = tl.load(
            weight + offs_m[:, None] * K + offs_k[None, :],
            mask=(offs_m[:, None] < COUT) & (offs_k[None, :] < K),
            other=0.0,
        )
        inp = tl.load(
            x
            + batch[None, :] * CIN * H * W
            + channel[:, None] * H * W
            + iy * W
            + ix,
            mask=(
                (offs_k[:, None] < K)
                & (offs_n[None, :] < L)
                & (iy >= 0)
                & (iy < H)
                & (ix >= 0)
                & (ix < W)
            ),
            other=0.0,
        )
        acc += tl.dot(w, inp)

    gamma = tl.load(bn_weight + offs_m, mask=offs_m < COUT).to(tl.float32)
    beta = tl.load(bn_bias + offs_m, mask=offs_m < COUT).to(tl.float32)
    mean = tl.load(running_mean + offs_m, mask=offs_m < COUT).to(tl.float32)
    var = tl.load(running_var + offs_m, mask=offs_m < COUT).to(tl.float32)
    inv_std = tl.rsqrt(var + EPS)
    y = acc * (gamma * inv_std)[:, None]
    y += (beta - mean * gamma * inv_std)[:, None]
    if DO_ACT:
        y *= tl.sigmoid(y)
    out_offs = (
        batch[None, :] * COUT * out_hw
        + offs_m[:, None] * out_hw
        + out_spatial[None, :]
    )
    tl.store(
        out + out_offs,
        y,
        mask=(offs_m[:, None] < COUT) & (offs_n[None, :] < L),
    )


def autopad(k: int | tuple[int, int], p=None, d: int = 1):
    if isinstance(k, tuple):
        if d > 1:
            k = tuple(d * (x - 1) + 1 for x in k)
        if p is None:
            return tuple(x // 2 for x in k)
        return p
    if d > 1:
        k = d * (k - 1) + 1
    return k // 2 if p is None else p


def _fuse_conv_bn(conv: Conv2d, bn: BatchNorm2d) -> tuple[torch.Tensor, torch.Tensor]:
    w_conv = conv.weight.clone().view(conv.weight.shape[0], -1)
    w_bn = torch.diag(
        bn.weight.to(dtype=conv.weight.dtype).div(
            torch.sqrt(bn.eps + bn.running_var.to(dtype=conv.weight.dtype))
        )
    )
    fused_weight = torch.mm(w_bn, w_conv).view_as(conv.weight)

    conv_bias = conv.bias
    if conv_bias is None:
        conv_bias = torch.zeros(conv.weight.shape[0], device=conv.weight.device, dtype=conv.weight.dtype)
    b_bn = (
        bn.bias.to(dtype=conv.weight.dtype)
        - bn.weight.to(dtype=conv.weight.dtype)
        .mul(bn.running_mean.to(dtype=conv.weight.dtype))
        .div(torch.sqrt(bn.running_var.to(dtype=conv.weight.dtype) + bn.eps))
    )
    fused_bias = torch.mm(w_bn, conv_bias.reshape(-1, 1)).reshape(-1) + b_bn
    return fused_weight, fused_bias


class YOLOConv(nn.Module):
    default_act = SiLU()

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 1,
        s: int = 1,
        p=None,
        g: int = 1,
        d: int = 1,
        act=True,
    ):
        super().__init__()
        self.conv = Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = BatchNorm2d(c2, eps=1e-3, momentum=0.03)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self._is_fused = False
        self._use_custom = (
            g == 1
            and d == 1
            and isinstance(k, int)
            and k in (1, 3)
            and (act is True or act is False)
        )
        self._do_act = act is True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._is_fused:
            return self.act(self.conv(x))
        if self._use_custom and x.is_cuda and x.dtype == torch.float16 and x.is_contiguous():
            n, _, h, w = x.shape
            kh, kw = self.conv.weight.shape[-2:]
            stride = self.conv.stride[0]
            padding = self.conv.padding[0]
            oh = (h + 2 * padding - kh) // stride + 1
            ow = (w + 2 * padding - kw) // stride + 1
            out = torch.empty((n, self.conv.weight.shape[0], oh, ow), device=x.device, dtype=x.dtype)
            length = n * oh * ow
            c_in = x.shape[1]
            c_out = self.conv.weight.shape[0]
            if kh == 1:
                grid = lambda meta: (
                    triton.cdiv(c_out, meta["BLOCK_M"]),
                    triton.cdiv(length, meta["BLOCK_N"]),
                )
                _conv1x1_bn_act_kernel[grid](
                    x,
                    self.conv.weight,
                    self.bn.weight,
                    self.bn.bias,
                    self.bn.running_mean,
                    self.bn.running_var,
                    out,
                    h * w,
                    c_in,
                    c_out,
                    length,
                    self.bn.eps,
                    self._do_act,
                )
            else:
                grid = lambda meta: (
                    triton.cdiv(c_out, meta["BLOCK_M"]),
                    triton.cdiv(length, meta["BLOCK_N"]),
                )
                _conv3x3_bn_act_kernel[grid](
                    x,
                    self.conv.weight,
                    self.bn.weight,
                    self.bn.bias,
                    self.bn.running_mean,
                    self.bn.running_var,
                    out,
                    h,
                    w,
                    oh,
                    ow,
                    c_in,
                    c_out,
                    length,
                    stride,
                    padding,
                    self.bn.eps,
                    self._do_act,
                )
            return out
        return self.act(self.bn(self.conv(x)))

    @torch.no_grad()
    def fuse(self):
        if self._is_fused:
            return self
        fused_weight, fused_bias = _fuse_conv_bn(self.conv, self.bn)
        self.conv.weight.data.copy_(fused_weight)
        self.conv.bias = nn.Parameter(fused_bias)
        delattr(self, "bn")
        self._is_fused = True
        return self


def fuse_module(module: nn.Module) -> nn.Module:
    from .yolov10_repvggdw import YOLORepVGGDW

    for child in module.children():
        fuse_module(child)
    if isinstance(module, YOLOConv):
        module.fuse()
    elif isinstance(module, YOLORepVGGDW):
        module.fuse()
    return module
