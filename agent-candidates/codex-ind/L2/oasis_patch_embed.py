"""Oasis 2D patch embedding."""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

from ..L1.conv2d import Conv2d


@triton.jit
def _to_tf32(x):
    bits = x.to(tl.int32, bitcast=True)
    rounded = (bits + 0xFFF + ((bits >> 13) & 1)) & -8192
    return rounded.to(tl.float32, bitcast=True)


@triton.jit
def _patch_embed_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    m_size,
    H: tl.constexpr,
    W: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    C: tl.constexpr,
    PATCH: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    DOT_MODE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    image = offs_m // (OH * OW)
    patch_id = offs_m % (OH * OW)
    patch_h = patch_id // OW
    patch_w = patch_id % OW
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        channel = offs_k // (PATCH * PATCH)
        patch_offset = offs_k % (PATCH * PATCH)
        kernel_h = patch_offset // PATCH
        kernel_w = patch_offset % PATCH
        x_offsets = (
            image[:, None] * (C * H * W)
            + channel[None, :] * (H * W)
            + (patch_h[:, None] * PATCH + kernel_h[None, :]) * W
            + patch_w[:, None] * PATCH
            + kernel_w[None, :]
        )
        x = tl.load(
            x_ptr + x_offsets,
            mask=(offs_m[:, None] < m_size) & (offs_k[None, :] < K),
            other=0.0,
        )
        weight_offsets = offs_n[None, :] * K + offs_k[:, None]
        weight = tl.load(
            weight_ptr + weight_offsets,
            mask=(offs_n[None, :] < N) & (offs_k[:, None] < K),
            other=0.0,
        )
        if DOT_MODE == 1:
            acc += tl.dot(x, weight, input_precision="tf32x3")
        else:
            acc += tl.dot(_to_tf32(x), _to_tf32(weight), input_precision="tf32")

    acc += tl.load(bias_ptr + offs_n)[None, :]
    out_offsets = offs_m[:, None] * N + offs_n[None, :]
    tl.store(
        out_ptr + out_offsets,
        acc,
        mask=(offs_m[:, None] < m_size) & (offs_n[None, :] < N),
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
        patch = self.patch_size[0]
        out_height = height // patch
        out_width = width // patch
        embed_dim = self.proj.weight.shape[0]
        use_triton = (
            x.is_cuda
            and x.dtype == torch.float32
            and self.proj.bias is not None
            and self.patch_size[0] == self.patch_size[1]
            and channels == self.proj.weight.shape[1]
        )
        if use_triton:
            m_size = batch * out_height * out_width
            out = torch.empty(
                (batch, out_height, out_width, embed_dim),
                device=x.device,
                dtype=x.dtype,
            )
            k_size = channels * patch * patch
            if k_size == 64:
                block_m = 32 if batch == 2 else 16
                block_n, block_k, num_warps = 64, 64, 4
                dot_mode = 1 if batch == 2 else 2
            else:
                block_m, block_n, block_k, num_warps = 32, 128, 32, 4
                dot_mode = 1
            grid = (
                triton.cdiv(m_size, block_m),
                triton.cdiv(embed_dim, block_n),
            )
            _patch_embed_kernel[grid](
                x,
                self.proj.weight,
                self.proj.bias,
                out,
                m_size,
                H=height,
                W=width,
                OH=out_height,
                OW=out_width,
                C=channels,
                PATCH=patch,
                K=k_size,
                N=embed_dim,
                DOT_MODE=dot_mode,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                BLOCK_K=block_k,
                num_warps=num_warps,
                num_stages=3,
            )
            x = out
            if self.flatten:
                x = x.reshape(batch, out_height * out_width, embed_dim)
        else:
            x = self.proj(x)
            if self.flatten:
                x = x.flatten(2).transpose(1, 2)
            else:
                x = x.permute(0, 2, 3, 1)
        return self.norm(x) if self.norm is not None else x
