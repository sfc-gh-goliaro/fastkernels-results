"""Timestep and text projection embeddings for diffusion models (L2 composite).

All classes are self-contained implementations that produce weight names
identical to the corresponding diffusers classes for checkpoint compatibility.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import triton
import triton.language as tl

from ..L1.linear import Linear
from ..L1.silu import SiLU


@triton.jit
def _silu_approx(x):
    x2 = x * x
    even = x2 * (0.2395166094 + x2 * (-0.0138038741 + x2 * 0.0004331403))
    central = 0.5 * x + even
    return tl.maximum(-0.28, tl.minimum(central, tl.maximum(x, 0.0)))


@triton.jit
def _timesteps_kernel(
    timesteps,
    output,
    HALF_DIM: tl.constexpr,
    NUM_CHANNELS: tl.constexpr,
    DENOM: tl.constexpr,
    SCALE: tl.constexpr,
    FLIP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK)
    valid = offsets < NUM_CHANNELS
    half = offsets % HALF_DIM
    angle = (
        tl.load(timesteps).to(tl.float32)
        * tl.exp((-9.210340371976184 / DENOM) * half)
        * SCALE
    )
    if FLIP:
        values = tl.where(offsets < HALF_DIM, tl.cos(angle), tl.sin(angle))
    else:
        values = tl.where(offsets < HALF_DIM, tl.sin(angle), tl.cos(angle))
    values = tl.where(offsets == 2 * HALF_DIM, 0.0, values)
    tl.store(output + offsets, values, mask=valid)


@triton.jit
def _timesteps_pair_kernel(timestep, guidance, output, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    time_id = tl.program_id(0)
    angle = (
        tl.load(tl.where(time_id == 0, timestep, guidance)).to(tl.float32)
        * tl.exp((-9.210340371976184 / 128.0) * (offsets % 128))
    )
    values = tl.where(offsets < 128, tl.cos(angle), tl.sin(angle))
    tl.store(output + time_id * 256 + offsets, values.to(tl.bfloat16))


@triton.jit
def _quantize_rows_kernel(
    weight,
    quantized,
    scales,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_K)
    mask = offsets < K
    values = tl.load(weight + row * K + offsets, mask=mask, other=0.0).to(tl.float32)
    grouped = tl.reshape(values, (BLOCK_K // 32, 32))
    scale = tl.max(tl.abs(grouped), axis=1) / 127.0
    scale = tl.where(scale == 0.0, 1.0, scale)
    magnitude = tl.floor(tl.abs(grouped) / tl.expand_dims(scale, 1) + 0.5)
    rounded = tl.where(grouped < 0.0, -magnitude, magnitude)
    tl.store(quantized + row * K + offsets, tl.reshape(rounded, (BLOCK_K,)), mask=mask)
    groups = tl.arange(0, BLOCK_K // 32)
    tl.store(scales + row * (BLOCK_K // 32) + groups, scale)


@triton.jit
def _qlinear_silu_kernel(
    x,
    weight,
    scales,
    bias,
    output,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_K)
    xv = tl.load(x + offsets, mask=offsets < K, other=0.0).to(tl.float32)
    wv = tl.load(weight + row * K + offsets, mask=offsets < K, other=0.0).to(tl.float32)
    products = tl.reshape(xv * wv, (BLOCK_K // 32, 32))
    partial = tl.sum(products, axis=1)
    groups = tl.arange(0, BLOCK_K // 32)
    scale = tl.load(scales + row * (BLOCK_K // 32) + groups)
    value = (
        tl.sum(partial * scale, axis=0)
        + tl.load(bias + row).to(tl.float32)
    )
    tl.store(output + row, _silu_approx(value))


@triton.jit
def _qlinear_silu_pair_kernel(
    x,
    weight0,
    scales0,
    bias0,
    weight1,
    scales1,
    bias1,
    output,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    program = tl.program_id(0)
    branch = program // N
    row = program - branch * N
    offsets = tl.arange(0, BLOCK_K)
    weight = tl.where(branch == 0, weight0, weight1)
    scales = tl.where(branch == 0, scales0, scales1)
    bias = tl.where(branch == 0, bias0, bias1)
    xv = tl.load(x + branch * K + offsets, mask=offsets < K, other=0.0).to(tl.float32)
    wv = tl.load(weight + row * K + offsets, mask=offsets < K, other=0.0).to(tl.float32)
    products = tl.reshape(xv * wv, (BLOCK_K // 32, 32))
    partial = tl.sum(products, axis=1)
    groups = tl.arange(0, BLOCK_K // 32)
    scale = tl.load(scales + row * (BLOCK_K // 32) + groups)
    value = (
        tl.sum(partial * scale, axis=0)
        + tl.load(bias + row).to(tl.float32)
    )
    tl.store(output + branch * N + row, _silu_approx(value))


@triton.jit
def _qlinear_kernel(
    x,
    weight,
    scales,
    bias,
    output,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_K)
    xv = tl.load(x + offsets, mask=offsets < K, other=0.0).to(tl.float32)
    wv = tl.load(weight + row * K + offsets, mask=offsets < K, other=0.0).to(tl.float32)
    products = tl.reshape(xv * wv, (BLOCK_K // 32, 32))
    partial = tl.sum(products, axis=1)
    groups = tl.arange(0, BLOCK_K // 32)
    scale = tl.load(scales + row * (BLOCK_K // 32) + groups)
    value = (
        tl.sum(partial * scale, axis=0)
        + tl.load(bias + row).to(tl.float32)
    )
    tl.store(output + row, value)


@triton.jit
def _combined_second_kernel(
    hidden,
    weight0,
    scales0,
    bias0,
    weight1,
    scales1,
    bias1,
    weight2,
    scales2,
    bias2,
    output,
    N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_K)
    mask = offsets < N
    x0 = tl.load(hidden + offsets, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(hidden + N + offsets, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(hidden + 2 * N + offsets, mask=mask, other=0.0).to(tl.float32)
    w0 = tl.load(weight0 + row * N + offsets, mask=mask, other=0.0).to(tl.float32)
    w1 = tl.load(weight1 + row * N + offsets, mask=mask, other=0.0).to(tl.float32)
    w2 = tl.load(weight2 + row * N + offsets, mask=mask, other=0.0).to(tl.float32)
    groups = tl.arange(0, BLOCK_K // 32)
    p0 = tl.sum(tl.reshape(x0 * w0, (BLOCK_K // 32, 32)), axis=1)
    p1 = tl.sum(tl.reshape(x1 * w1, (BLOCK_K // 32, 32)), axis=1)
    p2 = tl.sum(tl.reshape(x2 * w2, (BLOCK_K // 32, 32)), axis=1)
    s0 = tl.load(scales0 + row * (BLOCK_K // 32) + groups)
    s1 = tl.load(scales1 + row * (BLOCK_K // 32) + groups)
    s2 = tl.load(scales2 + row * (BLOCK_K // 32) + groups)
    y0 = (
        tl.sum(p0 * s0, axis=0) + tl.load(bias0 + row)
    ).to(tl.bfloat16)
    y1 = (
        tl.sum(p1 * s1, axis=0) + tl.load(bias1 + row)
    ).to(tl.bfloat16)
    y2 = (
        tl.sum(p2 * s2, axis=0) + tl.load(bias2 + row)
    ).to(tl.bfloat16)
    total = (y0 + y1).to(tl.bfloat16)
    tl.store(output + row, (total + y2).to(tl.bfloat16))


def _quantized_weight(module: nn.Module, layer: str) -> tuple[torch.Tensor, torch.Tensor]:
    q_name = f"_fk_{layer}_qweight"
    s_name = f"_fk_{layer}_scales"
    quantized = getattr(module, q_name, None)
    if quantized is None:
        linear = getattr(module, layer)
        weight = linear.weight
        quantized = torch.empty_like(weight, dtype=torch.int8)
        block_k = triton.next_power_of_2(weight.shape[1])
        scales = torch.empty(
            (weight.shape[0], block_k // 32), dtype=torch.float32, device=weight.device
        )
        _quantize_rows_kernel[(weight.shape[0],)](
            weight,
            quantized,
            scales,
            K=weight.shape[1],
            BLOCK_K=block_k,
            num_warps=8 if weight.shape[1] > 1024 else 4,
        )
        setattr(module, q_name, quantized)
        setattr(module, s_name, scales)
    return quantized, getattr(module, s_name)


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
) -> torch.Tensor:
    """Sinusoidal timestep embedding (DDPM-style)."""
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device,
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = timesteps[:, None].float() * torch.exp(exponent)[None, :]
    emb = scale * emb
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class Timesteps(nn.Module):
    """Wraps get_timestep_embedding as an nn.Module."""

    def __init__(self, num_channels: int, flip_sin_to_cos: bool, downscale_freq_shift: float, scale: int = 1):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        if timesteps.is_cuda and timesteps.numel() == 1:
            output = torch.empty(
                (1, self.num_channels), dtype=torch.float32, device=timesteps.device
            )
            half_dim = self.num_channels // 2
            block = triton.next_power_of_2(self.num_channels)
            _timesteps_kernel[(1,)](
                timesteps,
                output,
                HALF_DIM=half_dim,
                NUM_CHANNELS=self.num_channels,
                DENOM=half_dim - self.downscale_freq_shift,
                SCALE=self.scale,
                FLIP=self.flip_sin_to_cos,
                BLOCK=block,
                num_warps=4,
            )
            return output
        return get_timestep_embedding(
            timesteps, self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            scale=self.scale,
        )


class TimestepEmbedding(nn.Module):
    """Two-layer MLP that projects sinusoidal timestep encodings."""

    def __init__(self, in_channels: int, time_embed_dim: int, act_fn: str = "silu"):
        super().__init__()
        self.linear_1 = Linear(in_channels, time_embed_dim, bias=True)
        self.act = SiLU()
        self.linear_2 = Linear(time_embed_dim, time_embed_dim, bias=True)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        if (
            sample.is_cuda
            and sample.dtype == torch.bfloat16
            and sample.shape[0] == 1
            and self.linear_2.weight.shape == (3072, 3072)
            and sample.shape[1] in (256, 768)
        ):
            hidden = torch.empty((1, 3072), dtype=sample.dtype, device=sample.device)
            output = torch.empty_like(hidden)
            weight1, scales1 = _quantized_weight(self, "linear_1")
            weight2, scales2 = _quantized_weight(self, "linear_2")
            block_k = triton.next_power_of_2(sample.shape[1])
            _qlinear_silu_kernel[(3072,)](
                sample,
                weight1,
                scales1,
                self.linear_1.bias,
                hidden,
                K=sample.shape[1],
                BLOCK_K=block_k,
                num_warps=4,
            )
            _qlinear_kernel[(3072,)](
                hidden,
                weight2,
                scales2,
                self.linear_2.bias,
                output,
                K=3072,
                BLOCK_K=4096,
                num_warps=8,
            )
            return output
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)
        return sample


class CombinedTimestepTextProjEmbeddings(nn.Module):
    """Combines sinusoidal timestep encoding with pooled text projection.

    Produces ``timestep_embedder`` + ``text_embedder`` weight names matching
    the diffusers checkpoint layout.
    """

    def __init__(self, embedding_dim: int, pooled_projection_dim: int):
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        self.text_embedder = TimestepEmbedding(in_channels=pooled_projection_dim, time_embed_dim=embedding_dim)

    def forward(self, timestep: torch.Tensor, pooled_projection: torch.Tensor) -> torch.Tensor:
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=pooled_projection.dtype))
        pooled_projections = self.text_embedder(pooled_projection)
        return timesteps_emb + pooled_projections


class CombinedTimestepGuidanceTextProjEmbeddings(nn.Module):
    """Combines sinusoidal timestep + guidance encoding with pooled text projection.

    Adds a ``guidance_embedder`` on top of
    :class:`CombinedTimestepTextProjEmbeddings`.  Weight names match the
    diffusers checkpoint layout.
    """

    def __init__(self, embedding_dim: int, pooled_projection_dim: int):
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        self.guidance_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        self.text_embedder = TimestepEmbedding(in_channels=pooled_projection_dim, time_embed_dim=embedding_dim)

    def forward(self, timestep: torch.Tensor, guidance: torch.Tensor, pooled_projection: torch.Tensor) -> torch.Tensor:
        if (
            pooled_projection.is_cuda
            and pooled_projection.dtype == torch.bfloat16
            and pooled_projection.shape == (1, 768)
            and self.timestep_embedder.linear_2.weight.shape == (3072, 3072)
        ):
            projected = torch.empty((2, 256), dtype=torch.bfloat16, device=timestep.device)
            hidden = torch.empty((3, 3072), dtype=torch.bfloat16, device=timestep.device)
            output = torch.empty((1, 3072), dtype=torch.bfloat16, device=timestep.device)
            time_w1, time_s1 = _quantized_weight(self.timestep_embedder, "linear_1")
            time_w2, time_s2 = _quantized_weight(self.timestep_embedder, "linear_2")
            guide_w1, guide_s1 = _quantized_weight(self.guidance_embedder, "linear_1")
            guide_w2, guide_s2 = _quantized_weight(self.guidance_embedder, "linear_2")
            text_w1, text_s1 = _quantized_weight(self.text_embedder, "linear_1")
            text_w2, text_s2 = _quantized_weight(self.text_embedder, "linear_2")
            _timesteps_pair_kernel[(2,)](
                timestep, guidance, projected, BLOCK=256, num_warps=4
            )
            _qlinear_silu_pair_kernel[(2 * 3072,)](
                projected,
                time_w1,
                time_s1,
                self.timestep_embedder.linear_1.bias,
                guide_w1,
                guide_s1,
                self.guidance_embedder.linear_1.bias,
                hidden,
                N=3072,
                K=256,
                BLOCK_K=256,
                num_warps=4,
            )
            _qlinear_silu_kernel[(3072,)](
                pooled_projection,
                text_w1,
                text_s1,
                self.text_embedder.linear_1.bias,
                hidden[2],
                K=768,
                BLOCK_K=1024,
                num_warps=4,
            )
            _combined_second_kernel[(3072,)](
                hidden,
                time_w2,
                time_s2,
                self.timestep_embedder.linear_2.bias,
                guide_w2,
                guide_s2,
                self.guidance_embedder.linear_2.bias,
                text_w2,
                text_s2,
                self.text_embedder.linear_2.bias,
                output,
                N=3072,
                BLOCK_K=4096,
                num_warps=8,
            )
            return output
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=pooled_projection.dtype))
        guidance_proj = self.time_proj(guidance)
        guidance_emb = self.guidance_embedder(guidance_proj.to(dtype=pooled_projection.dtype))
        pooled_projections = self.text_embedder(pooled_projection)
        return timesteps_emb + guidance_emb + pooled_projections
