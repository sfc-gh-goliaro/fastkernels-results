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
def _timesteps_kernel(
    timesteps,
    output,
    half_dim: tl.constexpr,
    denominator,
    scale,
    flip: tl.constexpr,
    odd: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK)
    valid = offsets < (2 * half_dim)
    freq_idx = offsets % half_dim
    exponent = -9.210340371976184 * freq_idx.to(tl.float32) / denominator
    angle = tl.load(timesteps).to(tl.float32) * tl.exp(exponent) * scale
    use_cos = (offsets >= half_dim) != flip
    values = tl.where(use_cos, tl.cos(angle), tl.sin(angle))
    tl.store(output + offsets, values, mask=valid)
    if odd:
        tl.store(output + 2 * half_dim, 0.0)


@triton.jit
def _linear_kernel(
    x,
    weight,
    bias,
    output,
    K: tl.constexpr,
    N: tl.constexpr,
    SILU: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((BLOCK_N,), tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k = k_start + offsets_k
        inputs = tl.load(x + k, mask=k < K, other=0.0)
        weights = tl.load(
            weight + offsets_n[:, None] * K + k[None, :],
            mask=(offsets_n[:, None] < N) & (k[None, :] < K),
            other=0.0,
        )
        accumulator += tl.sum(weights * inputs[None, :], axis=1)

    values = accumulator + tl.load(bias + offsets_n, mask=offsets_n < N)
    if SILU:
        values *= tl.sigmoid(values)
    tl.store(output + offsets_n, values, mask=offsets_n < N)


@triton.jit
def _combined_first_kernel(
    timestep,
    guidance,
    pooled,
    time_weight,
    time_bias,
    guidance_weight,
    guidance_bias,
    text_weight,
    text_bias,
    hidden,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row_pid = tl.program_id(0)
    branch = tl.program_id(1)
    offsets_n = row_pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((BLOCK_N,), tl.float32)

    weight = tl.where(
        branch == 0,
        time_weight,
        tl.where(branch == 1, guidance_weight, text_weight),
    )
    bias = tl.where(
        branch == 0,
        time_bias,
        tl.where(branch == 1, guidance_bias, text_bias),
    )
    if branch == 2:
        for k_start in range(0, 768, BLOCK_K):
            k = k_start + offsets_k
            inputs = tl.load(pooled + k, mask=k < 768, other=0.0)
            weights = tl.load(
                weight + offsets_n[:, None] * 768 + k[None, :],
                mask=(offsets_n[:, None] < N) & (k[None, :] < 768),
                other=0.0,
            )
            accumulator += tl.sum(weights * inputs[None, :], axis=1)
    else:
        time_value = tl.where(
            branch == 0, tl.load(timestep), tl.load(guidance)
        ).to(tl.float32)
        k = offsets_k
        freq_idx = k % 128
        angle = time_value * tl.exp(
            -9.210340371976184 * freq_idx.to(tl.float32) / 128.0
        )
        inputs = tl.where(k < 128, tl.cos(angle), tl.sin(angle)).to(
            tl.bfloat16
        )
        weights = tl.load(
            weight + offsets_n[:, None] * 256 + k[None, :],
            mask=(offsets_n[:, None] < N) & (k[None, :] < 256),
            other=0.0,
        )
        accumulator += tl.sum(weights * inputs[None, :], axis=1)

    values = accumulator + tl.load(bias + offsets_n, mask=offsets_n < N)
    values *= tl.sigmoid(values)
    tl.store(
        hidden + branch * N + offsets_n,
        values,
        mask=offsets_n < N,
    )


@triton.jit
def _combined_second_kernel(
    hidden,
    time_weight,
    time_bias,
    guidance_weight,
    guidance_bias,
    text_weight,
    text_bias,
    output,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, BLOCK_K)
    time_acc = tl.zeros((BLOCK_N,), tl.float32)
    guidance_acc = tl.zeros((BLOCK_N,), tl.float32)
    text_acc = tl.zeros((BLOCK_N,), tl.float32)

    for k_start in range(0, N, BLOCK_K):
        k = k_start + offsets_k
        time_x = tl.load(hidden + k, mask=k < N, other=0.0)
        guidance_x = tl.load(hidden + N + k, mask=k < N, other=0.0)
        text_x = tl.load(hidden + 2 * N + k, mask=k < N, other=0.0)
        weight_offsets = offsets_n[:, None] * N + k[None, :]
        mask = (offsets_n[:, None] < N) & (k[None, :] < N)
        time_w = tl.load(time_weight + weight_offsets, mask=mask, other=0.0)
        guidance_w = tl.load(
            guidance_weight + weight_offsets, mask=mask, other=0.0
        )
        text_w = tl.load(text_weight + weight_offsets, mask=mask, other=0.0)
        time_acc += tl.sum(time_w * time_x[None, :], axis=1)
        guidance_acc += tl.sum(guidance_w * guidance_x[None, :], axis=1)
        text_acc += tl.sum(text_w * text_x[None, :], axis=1)

    time_out = (
        time_acc + tl.load(time_bias + offsets_n, mask=offsets_n < N)
    ).to(tl.bfloat16)
    guidance_out = (
        guidance_acc + tl.load(guidance_bias + offsets_n, mask=offsets_n < N)
    ).to(tl.bfloat16)
    text_out = (
        text_acc + tl.load(text_bias + offsets_n, mask=offsets_n < N)
    ).to(tl.bfloat16)
    values = (time_out + guidance_out).to(tl.bfloat16)
    values = (values + text_out).to(tl.bfloat16)
    tl.store(output + offsets_n, values, mask=offsets_n < N)


def _linear_triton(sample: torch.Tensor, layer: Linear, silu: bool) -> torch.Tensor:
    n, k = layer.weight.shape
    output = torch.empty((1, n), dtype=sample.dtype, device=sample.device)
    _linear_kernel[(triton.cdiv(n, 8),)](
        sample,
        layer.weight,
        layer.bias,
        output,
        K=k,
        N=n,
        SILU=silu,
        BLOCK_N=8,
        BLOCK_K=256,
        num_warps=8,
        num_stages=1,
    )
    return output


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
            half_dim = self.num_channels // 2
            output = torch.empty(
                (1, self.num_channels), dtype=torch.float32, device=timesteps.device
            )
            _timesteps_kernel[(1,)](
                timesteps,
                output,
                half_dim=half_dim,
                denominator=half_dim - self.downscale_freq_shift,
                scale=self.scale,
                flip=self.flip_sin_to_cos,
                odd=self.num_channels % 2,
                BLOCK=triton.next_power_of_2(self.num_channels),
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
            and sample.shape == (1, self.linear_1.weight.shape[1])
            and self.linear_1.weight.shape[0] == 3072
            and self.linear_2.weight.shape == (3072, 3072)
        ):
            sample = _linear_triton(sample, self.linear_1, silu=True)
            return _linear_triton(sample, self.linear_2, silu=False)
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
            timestep.is_cuda
            and timestep.dtype == torch.bfloat16
            and timestep.numel() == 1
            and guidance.shape == (1,)
            and pooled_projection.shape == (1, 768)
            and self.timestep_embedder.linear_2.weight.shape == (3072, 3072)
        ):
            n = 3072
            hidden = torch.empty(
                (3, n), dtype=torch.bfloat16, device=timestep.device
            )
            output = torch.empty(
                (1, n), dtype=torch.bfloat16, device=timestep.device
            )
            _combined_first_kernel[(triton.cdiv(n, 16), 3)](
                timestep,
                guidance,
                pooled_projection,
                self.timestep_embedder.linear_1.weight,
                self.timestep_embedder.linear_1.bias,
                self.guidance_embedder.linear_1.weight,
                self.guidance_embedder.linear_1.bias,
                self.text_embedder.linear_1.weight,
                self.text_embedder.linear_1.bias,
                hidden,
                N=n,
                BLOCK_N=16,
                BLOCK_K=256,
                num_warps=8,
                num_stages=1,
            )
            _combined_second_kernel[(triton.cdiv(n, 16),)](
                hidden,
                self.timestep_embedder.linear_2.weight,
                self.timestep_embedder.linear_2.bias,
                self.guidance_embedder.linear_2.weight,
                self.guidance_embedder.linear_2.bias,
                self.text_embedder.linear_2.weight,
                self.text_embedder.linear_2.bias,
                output,
                N=n,
                BLOCK_N=16,
                BLOCK_K=256,
                num_warps=8,
                num_stages=1,
            )
            return output
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=pooled_projection.dtype))
        guidance_proj = self.time_proj(guidance)
        guidance_emb = self.guidance_embedder(guidance_proj.to(dtype=pooled_projection.dtype))
        pooled_projections = self.text_embedder(pooled_projection)
        return timesteps_emb + guidance_emb + pooled_projections
