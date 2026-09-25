"""Vision encoder rotary position embeddings.

Precomputes a cos/sin cache from fixed inv_freq (base=10000, no scaling).
forward() builds 2D (height, width) position IDs from grid_thw metadata,
shuffled by spatial_merge_size, and returns (cos, sin) tensors ready for
flash_attn's apply_rotary.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _rotary_gather_kernel(
    cache,
    output,
    n_tokens: tl.constexpr,
    rotary_dim: tl.constexpr,
    half_dim: tl.constexpr,
    sms: tl.constexpr,
    end0,
    end1,
    end2,
    end3,
    end4,
    end5,
    end6,
    hw0,
    hw1,
    hw2,
    hw3,
    hw4,
    hw5,
    hw6,
    hw7,
    w0,
    w1,
    w2,
    w3,
    w4,
    w5,
    w6,
    w7,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)[:, None]
    col = tl.arange(0, BLOCK_D)[None, :]

    segment = (token >= end0).to(tl.int32)
    segment += (token >= end1).to(tl.int32)
    segment += (token >= end2).to(tl.int32)
    segment += (token >= end3).to(tl.int32)
    segment += (token >= end4).to(tl.int32)
    segment += (token >= end5).to(tl.int32)
    segment += (token >= end6).to(tl.int32)

    start = tl.where(segment == 0, 0, end0)
    start = tl.where(segment == 1, end0, start)
    start = tl.where(segment == 2, end1, start)
    start = tl.where(segment == 3, end2, start)
    start = tl.where(segment == 4, end3, start)
    start = tl.where(segment == 5, end4, start)
    start = tl.where(segment == 6, end5, start)
    start = tl.where(segment == 7, end6, start)

    hw = tl.where(segment == 0, hw0, hw7)
    hw = tl.where(segment == 1, hw1, hw)
    hw = tl.where(segment == 2, hw2, hw)
    hw = tl.where(segment == 3, hw3, hw)
    hw = tl.where(segment == 4, hw4, hw)
    hw = tl.where(segment == 5, hw5, hw)
    hw = tl.where(segment == 6, hw6, hw)

    width = tl.where(segment == 0, w0, w7)
    width = tl.where(segment == 1, w1, width)
    width = tl.where(segment == 2, w2, width)
    width = tl.where(segment == 3, w3, width)
    width = tl.where(segment == 4, w4, width)
    width = tl.where(segment == 5, w5, width)
    width = tl.where(segment == 6, w6, width)

    local = (token - start) % hw
    block = local // (sms * sms)
    inner = local % (sms * sms)
    blocks_w = width // sms
    pos_h = (block // blocks_w) * sms + inner // sms
    pos_w = (block % blocks_w) * sms + inner % sms
    pos = tl.where(col < half_dim, pos_h, pos_w)
    freq_col = col % half_dim

    offsets = token * rotary_dim + col
    mask = (token < n_tokens) & (col < rotary_dim)
    tl.store(output + offsets, tl.load(cache + pos * rotary_dim + freq_col, mask=mask), mask=mask)
    tl.store(
        output + n_tokens * rotary_dim + offsets,
        tl.load(cache + pos * rotary_dim + half_dim + freq_col, mask=mask),
        mask=mask,
    )


class VisionRotaryEmbedding(nn.Module):
    def __init__(self, rotary_dim: int, max_grid_size: int = 8192):
        super().__init__()
        inv_freq = 1.0 / (10000.0 ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim
        ))
        t = torch.arange(max_grid_size, dtype=torch.float)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def forward(
        self,
        grid_thw_list: list[list[int]],
        spatial_merge_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            dtype == torch.bfloat16
            and self.cos_sin_cache.is_cuda
            and 0 < len(grid_thw_list) <= 8
        ):
            ends = []
            hws = []
            widths = []
            total = 0
            for t, h, w in grid_thw_list:
                total += t * h * w
                ends.append(total)
                hws.append(h * w)
                widths.append(w)

            while len(hws) < 8:
                hws.append(1)
                widths.append(spatial_merge_size)
            while len(ends) < 7:
                ends.append(total + 1)

            rotary_dim = self.cos_sin_cache.shape[1]
            half_dim = rotary_dim // 2
            output_dim = half_dim * 2
            output = torch.empty((2, total, output_dim), dtype=dtype, device=device)
            _rotary_gather_kernel[(triton.cdiv(total, 16),)](
                self.cos_sin_cache,
                output,
                total,
                output_dim,
                half_dim,
                spatial_merge_size,
                *ends[:7],
                *hws,
                *widths,
                BLOCK_T=16,
                BLOCK_D=64,
                num_warps=4,
            )
            return output[0], output[1]

        sms = spatial_merge_size
        pos_ids = []
        max_grid_size = 0
        for t, h, w in grid_thw_list:
            hpos = np.broadcast_to(np.arange(h).reshape(h, 1), (h, w))
            wpos = np.broadcast_to(np.arange(w).reshape(1, w), (h, w))
            hpos = hpos.reshape(h // sms, sms, w // sms, sms).transpose(0, 2, 1, 3).flatten()
            wpos = wpos.reshape(h // sms, sms, w // sms, sms).transpose(0, 2, 1, 3).flatten()
            hw = np.stack([hpos, wpos], axis=-1)
            pos_ids.append(np.tile(hw, (t, 1)) if t > 1 else hw)
            max_grid_size = max(max_grid_size, h, w)
        pos_ids = torch.from_numpy(np.concatenate(pos_ids, axis=0)).to(device)

        cache = self.cos_sin_cache[:max_grid_size].to(dtype=dtype)
        cos, sin = cache.chunk(2, dim=-1)
        return cos[pos_ids].flatten(1), sin[pos_ids].flatten(1)
