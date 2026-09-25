"""Bilinear interpolation of learned 2D position embeddings (Qwen3-VL).

Owns a learned embedding weight of (num_grid_per_side^2, hidden_size).
forward() interpolates these onto arbitrary (h, w) grids using bilinear
weights, then reshuffles by spatial_merge_size for the vision encoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

from ..L1.embedding import Embedding


@triton.jit
def _interpolate_kernel(
    weight,
    output,
    output_starts: tl.constexpr,
    num_copies: tl.constexpr,
    h_size: tl.constexpr,
    w_size: tl.constexpr,
    num_grid: tl.constexpr,
    hidden_dim: tl.constexpr,
    merge_size: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    dims = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    valid_row = rows < h_size * w_size
    valid_dim = dims < hidden_dim

    in_merge = rows % (merge_size * merge_size)
    block = rows // (merge_size * merge_size)
    block_w = w_size // merge_size
    h = (block // block_w) * merge_size + in_merge // merge_size
    w = (block % block_w) * merge_size + in_merge % merge_size

    if h_size == num_grid:
        h_pos = h.to(tl.float32)
    else:
        h_pos = h.to(tl.float32) * ((num_grid - 1.0) / (h_size - 1.0))
    if w_size == num_grid:
        w_pos = w.to(tl.float32)
    else:
        w_pos = w.to(tl.float32) * ((num_grid - 1.0) / (w_size - 1.0))

    h0 = h_pos.to(tl.int32)
    w0 = w_pos.to(tl.int32)
    h1 = tl.minimum(h0 + 1, num_grid - 1)
    w1 = tl.minimum(w0 + 1, num_grid - 1)
    dh = h_pos - h0
    dw = w_pos - w0

    w11 = dh * dw
    w10 = dh - w11
    w01 = dw - w11
    w00 = 1.0 - dh - w01

    load_mask = valid_row[:, None] & valid_dim[None, :]
    dim = dims[None, :]
    base00 = (h0 * num_grid + w0)[:, None] * hidden_dim + dim
    base01 = (h0 * num_grid + w1)[:, None] * hidden_dim + dim
    base10 = (h1 * num_grid + w0)[:, None] * hidden_dim + dim
    base11 = (h1 * num_grid + w1)[:, None] * hidden_dim + dim

    v00 = tl.load(weight + base00, mask=load_mask)
    v01 = tl.load(weight + base01, mask=load_mask)
    v10 = tl.load(weight + base10, mask=load_mask)
    v11 = tl.load(weight + base11, mask=load_mask)
    result = (
        v00 * w00[:, None]
        + v01 * w01[:, None]
        + v10 * w10[:, None]
        + v11 * w11[:, None]
    )
    for copy in tl.static_range(num_copies):
        output_row = output_starts[copy] + rows[:, None]
        output_offsets = output_row * hidden_dim + dim
        tl.store(output + output_offsets, result, mask=load_mask)


@triton.jit
def _interpolate_grouped_kernel(
    weight,
    output,
    h_sizes: tl.constexpr,
    w_sizes: tl.constexpr,
    output_starts: tl.constexpr,
    output_groups: tl.constexpr,
    num_groups: tl.constexpr,
    num_copies: tl.constexpr,
    num_grid: tl.constexpr,
    hidden_dim: tl.constexpr,
    merge_size: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    group = tl.program_id(2)
    h_size = h_sizes[0]
    w_size = w_sizes[0]
    for i in tl.static_range(1, num_groups):
        h_size = tl.where(group == i, h_sizes[i], h_size)
        w_size = tl.where(group == i, w_sizes[i], w_size)

    rows = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    dims = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    valid_row = rows < h_size * w_size
    valid_dim = dims < hidden_dim

    in_merge = rows % (merge_size * merge_size)
    block = rows // (merge_size * merge_size)
    block_w = w_size // merge_size
    h = (block // block_w) * merge_size + in_merge // merge_size
    w = (block % block_w) * merge_size + in_merge % merge_size
    h_pos = h.to(tl.float32) * ((num_grid - 1.0) / (h_size - 1.0))
    w_pos = w.to(tl.float32) * ((num_grid - 1.0) / (w_size - 1.0))

    h0 = h_pos.to(tl.int32)
    w0 = w_pos.to(tl.int32)
    h1 = tl.minimum(h0 + 1, num_grid - 1)
    w1 = tl.minimum(w0 + 1, num_grid - 1)
    dh = h_pos - h0
    dw = w_pos - w0
    w11 = dh * dw
    w10 = dh - w11
    w01 = dw - w11
    w00 = 1.0 - dh - w01

    load_mask = valid_row[:, None] & valid_dim[None, :]
    dim = dims[None, :]
    base00 = (h0 * num_grid + w0)[:, None] * hidden_dim + dim
    base01 = (h0 * num_grid + w1)[:, None] * hidden_dim + dim
    base10 = (h1 * num_grid + w0)[:, None] * hidden_dim + dim
    base11 = (h1 * num_grid + w1)[:, None] * hidden_dim + dim
    v00 = tl.load(weight + base00, mask=load_mask)
    v01 = tl.load(weight + base01, mask=load_mask)
    v10 = tl.load(weight + base10, mask=load_mask)
    v11 = tl.load(weight + base11, mask=load_mask)
    result = (
        v00 * w00[:, None]
        + v01 * w01[:, None]
        + v10 * w10[:, None]
        + v11 * w11[:, None]
    )

    for copy in tl.static_range(num_copies):
        store_mask = load_mask & (group == output_groups[copy])
        output_row = output_starts[copy] + rows[:, None]
        output_offsets = output_row * hidden_dim + dim
        tl.store(output + output_offsets, result, mask=store_mask)


class VisionPosEmbedInterpolate(nn.Module):
    def __init__(self, num_position_embeddings: int, hidden_size: int,
                 spatial_merge_size: int):
        super().__init__()
        self._embed = Embedding(num_position_embeddings, hidden_size)
        self.num_grid_per_side = int(num_position_embeddings ** 0.5)
        self.spatial_merge_size = spatial_merge_size
        self.hidden_size = hidden_size

    def forward(
        self,
        grid_thw_list: list[list[int]],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        num_grid = self.num_grid_per_side
        m_size = self.spatial_merge_size
        hidden_dim = self.hidden_size

        total_rows = sum(t * h * w for t, h, w in grid_thw_list)
        output = torch.empty((total_rows, hidden_dim), dtype=dtype, device=device)
        grouped_grids = {}
        output_row_start = 0
        for t, h, w in grid_thw_list:
            segments = grouped_grids.setdefault((h, w), [])
            if segments and segments[-1][0] + segments[-1][1] * h * w == output_row_start:
                segments[-1] = (segments[-1][0], segments[-1][1] + t)
            else:
                segments.append((output_row_start, t))
            output_row_start += t * h * w

        groups = list(grouped_grids.items())
        if len(groups) > 1:
            copy_starts = []
            copy_groups = []
            for group, ((h, w), segments) in enumerate(groups):
                for start, t in segments:
                    for repeat in range(t):
                        copy_starts.append(start + repeat * h * w)
                        copy_groups.append(group)
            grid = (
                triton.cdiv(max(h * w for (h, w), _ in groups), 8),
                triton.cdiv(hidden_dim, 128),
                len(groups),
            )
            _interpolate_grouped_kernel[grid](
                self._embed.emb.weight,
                output,
                tuple(h for (h, _), _ in groups),
                tuple(w for (_, w), _ in groups),
                tuple(copy_starts),
                tuple(copy_groups),
                len(groups),
                len(copy_starts),
                num_grid,
                hidden_dim,
                m_size,
                BLOCK_T=8,
                BLOCK_D=128,
                num_warps=4,
            )
            return output

        for (h, w), segments in groups:
            output_starts = tuple(
                start + repeat * h * w
                for start, t in segments
                for repeat in range(t)
            )
            grid = (
                triton.cdiv(h * w, 8),
                triton.cdiv(hidden_dim, 128),
            )
            _interpolate_kernel[grid](
                self._embed.emb.weight,
                output,
                output_starts,
                len(output_starts),
                h,
                w,
                num_grid,
                hidden_dim,
                m_size,
                BLOCK_T=8,
                BLOCK_D=128,
                num_warps=4,
            )

        return output
