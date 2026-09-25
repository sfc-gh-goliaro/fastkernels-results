"""Oasis 2D patch embedding."""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

from ..L1.conv2d import Conv2d


@triton.jit
def _to_tf32(x):
    # cuDNN rounds to nearest; Triton's native tf32 conversion truncates.
    bits = x.to(tl.int32, bitcast=True)
    bits += 0xFFF + ((bits >> 13) & 1)
    bits &= -8192
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def _patch_embed_nhwc(
    x,
    weight,
    bias,
    out,
    M: tl.constexpr,
    N: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    P: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    image = offs_m // (OH * OW)
    patch = offs_m % (OH * OW)
    oh = patch // OW
    ow = patch % OW
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    K: tl.constexpr = C * P * P
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        channel = offs_k // (P * P)
        pixel = offs_k % (P * P)
        ih = oh[:, None] * P + pixel[None, :] // P
        iw = ow[:, None] * P + pixel[None, :] % P
        a = tl.load(
            x
            + image[:, None] * (C * H * W)
            + channel[None, :] * (H * W)
            + ih * W
            + iw,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            weight + offs_n[None, :] * K + offs_k[:, None],
            mask=(offs_n[None, :] < N) & (offs_k[:, None] < K),
            other=0.0,
        )
        if INPUT_PRECISION == "tf32rn":
            acc = tl.dot(
                _to_tf32(a), _to_tf32(b), acc, input_precision="tf32"
            )
        else:
            acc = tl.dot(a, b, acc, input_precision=INPUT_PRECISION)

    acc += tl.load(bias + offs_n, mask=offs_n < N, other=0.0)[None, :]
    tl.store(
        out + offs_m[:, None] * N + offs_n[None, :],
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


class OasisPatchEmbed(nn.Module):
    def __init__(
        self,
        img_height: int = 256,
        img_width: int = 256,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer=None,
        flatten: bool = True,
    ):
        super().__init__()
        self.img_size = (img_height, img_width)
        self.patch_size = (patch_size, patch_size)
        self.grid_size = (img_height // patch_size, img_width // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.proj = Conv2d(
            in_chans,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=True,
        )
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x: torch.Tensor, random_sample: bool = False) -> torch.Tensor:
        batch, channels, height, width = x.shape
        if not random_sample and (height, width) != self.img_size:
            raise AssertionError(
                f"Input image size ({height}*{width}) doesn't match model {self.img_size}.",
            )
        if (
            x.is_cuda
            and x.dtype == torch.float32
            and channels == 16
            and self.proj.out_channels == 1024
            and self.patch_size == (2, 2)
            and (height, width) == (18, 32)
            and not self.flatten
            and self.norm is None
        ):
            oh, ow = 9, 16
            m, n = batch * oh * ow, 1024
            out = torch.empty((batch, oh, ow, n), device=x.device, dtype=x.dtype)
            _patch_embed_nhwc[(triton.cdiv(m, 32), triton.cdiv(n, 64))](
                x,
                self.proj.weight,
                self.proj.bias,
                out,
                M=m,
                N=n,
                C=channels,
                H=height,
                W=width,
                OH=oh,
                OW=ow,
                P=2,
                INPUT_PRECISION="tf32x3" if batch == 2 else "tf32rn",
                BLOCK_M=32,
                BLOCK_N=64,
                BLOCK_K=64,
                num_warps=4,
                num_stages=2,
            )
            return out
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)
        else:
            x = x.permute(0, 2, 3, 1)
        return self.norm(x) if self.norm is not None else x
