from __future__ import annotations

import math

import torch
import torch.nn as nn
import triton
import triton.language as tl


def _compute_scaled_inv_freq(
    inv_freq: torch.Tensor,
    scaling_factor: float,
    low_freq_factor: float,
    high_freq_factor: float,
    original_max_position_embeddings: int,
) -> torch.Tensor:
    low_wl = original_max_position_embeddings / low_freq_factor
    high_wl = original_max_position_embeddings / high_freq_factor
    wl = 2 * math.pi / inv_freq
    if low_freq_factor != high_freq_factor:
        smooth = (original_max_position_embeddings / wl - low_freq_factor) / (
            high_freq_factor - low_freq_factor
        )
    else:
        smooth = torch.zeros_like(inv_freq)
    return torch.where(
        wl < high_wl,
        inv_freq,
        torch.where(
            wl > low_wl,
            inv_freq / scaling_factor,
            (1 - smooth) * inv_freq / scaling_factor + smooth * inv_freq,
        ),
    )


@triton.jit
def _rope_kernel(
    positions,
    query,
    key,
    cache,
    query_stride,
    key_stride,
    Q_PAIRS: tl.constexpr,
    K_PAIRS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NEOX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    pair = tl.arange(0, BLOCK)
    half_dim: tl.constexpr = HEAD_DIM // 2

    q_mask = pair < Q_PAIRS
    q_head = pair // half_dim
    rotary = pair % half_dim
    if NEOX:
        x_col = q_head * HEAD_DIM + rotary
        y_col = x_col + half_dim
    else:
        x_col = q_head * HEAD_DIM + 2 * rotary
        y_col = x_col + 1

    pos = tl.load(positions + token)
    cache_row = cache + pos * HEAD_DIM
    cos = tl.load(cache_row + rotary, mask=q_mask)
    sin = tl.load(cache_row + half_dim + rotary, mask=q_mask)

    q_row = query + token * query_stride
    x = tl.load(q_row + x_col, mask=q_mask).to(tl.float32)
    y = tl.load(q_row + y_col, mask=q_mask).to(tl.float32)
    tl.store(q_row + x_col, x * cos - y * sin, mask=q_mask)
    tl.store(q_row + y_col, y * cos + x * sin, mask=q_mask)

    k_mask = pair < K_PAIRS
    k_head = pair // half_dim
    if NEOX:
        kx_col = k_head * HEAD_DIM + rotary
        ky_col = kx_col + half_dim
    else:
        kx_col = k_head * HEAD_DIM + 2 * rotary
        ky_col = kx_col + 1
    k_row = key + token * key_stride
    kx = tl.load(k_row + kx_col, mask=k_mask).to(tl.float32)
    ky = tl.load(k_row + ky_col, mask=k_mask).to(tl.float32)
    tl.store(k_row + kx_col, kx * cos - ky * sin, mask=k_mask)
    tl.store(k_row + ky_col, ky * cos + kx * sin, mask=k_mask)


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int,
        rope_theta: float,
        rope_scaling_factor: float = 1.0,
        rope_low_freq_factor: float = 1.0,
        rope_high_freq_factor: float = 1.0,
        rope_original_max_position_embeddings: int | None = None,
        is_neox_style: bool = True,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.is_neox_style = is_neox_style
        inv_freq = 1.0 / (
            rope_theta
            ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim)
        )
        if (
            rope_scaling_factor != 1.0
            and rope_original_max_position_embeddings is not None
        ):
            inv_freq = _compute_scaled_inv_freq(
                inv_freq,
                rope_scaling_factor,
                rope_low_freq_factor,
                rope_high_freq_factor,
                rope_original_max_position_embeddings,
            )

        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.outer(t, inv_freq)
        cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1).float()
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def forward(self, positions, query, key):
        head_dim = self.head_dim
        q_pairs = query.shape[-1] // 2
        k_pairs = key.shape[-1] // 2
        block = triton.next_power_of_2(q_pairs)
        num_warps = 4 if positions.numel() >= 1024 else 8
        _rope_kernel[(positions.numel(),)](
            positions,
            query,
            key,
            self.cos_sin_cache,
            query.stride(0),
            key.stride(0),
            Q_PAIRS=q_pairs,
            K_PAIRS=k_pairs,
            HEAD_DIM=head_dim,
            NEOX=self.is_neox_style,
            BLOCK=block,
            num_warps=num_warps,
        )
        return query, key
