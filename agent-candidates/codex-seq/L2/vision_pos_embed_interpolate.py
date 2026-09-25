"""Fused bilinear interpolation of learned 2D position embeddings."""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

from ..L1.embedding import Embedding


@triton.jit
def _interpolate(
    weight,
    output,
    hidden_dim: tl.constexpr,
    num_grid: tl.constexpr,
    merge_size: tl.constexpr,
    p1: tl.constexpr, p2: tl.constexpr, p3: tl.constexpr,
    p4: tl.constexpr, p5: tl.constexpr, p6: tl.constexpr, p7: tl.constexpr,
    o1: tl.constexpr, o2: tl.constexpr, o3: tl.constexpr,
    o4: tl.constexpr, o5: tl.constexpr, o6: tl.constexpr, o7: tl.constexpr,
    n0: tl.constexpr, n1: tl.constexpr, n2: tl.constexpr, n3: tl.constexpr,
    n4: tl.constexpr, n5: tl.constexpr, n6: tl.constexpr, n7: tl.constexpr,
    t0: tl.constexpr, t1: tl.constexpr, t2: tl.constexpr, t3: tl.constexpr,
    t4: tl.constexpr, t5: tl.constexpr, t6: tl.constexpr, t7: tl.constexpr,
    h0: tl.constexpr, h1: tl.constexpr, h2: tl.constexpr, h3: tl.constexpr,
    h4: tl.constexpr, h5: tl.constexpr, h6: tl.constexpr, h7: tl.constexpr,
    w0: tl.constexpr, w1: tl.constexpr, w2: tl.constexpr, w3: tl.constexpr,
    w4: tl.constexpr, w5: tl.constexpr, w6: tl.constexpr, w7: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_r = tl.program_id(0)
    image = tl.zeros((), tl.int32)
    image += (pid_r >= p1).to(tl.int32)
    image += (pid_r >= p2).to(tl.int32)
    image += (pid_r >= p3).to(tl.int32)
    image += (pid_r >= p4).to(tl.int32)
    image += (pid_r >= p5).to(tl.int32)
    image += (pid_r >= p6).to(tl.int32)
    image += (pid_r >= p7).to(tl.int32)

    p_start = tl.where(image == 0, 0, p1)
    o_start = tl.where(image == 0, 0, o1)
    rows = tl.where(image == 0, n0, n1)
    temporal = tl.where(image == 0, t0, t1)
    height = tl.where(image == 0, h0, h1)
    width = tl.where(image == 0, w0, w1)
    p_start = tl.where(image == 2, p2, p_start)
    p_start = tl.where(image == 3, p3, p_start)
    p_start = tl.where(image == 4, p4, p_start)
    p_start = tl.where(image == 5, p5, p_start)
    p_start = tl.where(image == 6, p6, p_start)
    p_start = tl.where(image == 7, p7, p_start)
    o_start = tl.where(image == 2, o2, o_start)
    o_start = tl.where(image == 3, o3, o_start)
    o_start = tl.where(image == 4, o4, o_start)
    o_start = tl.where(image == 5, o5, o_start)
    o_start = tl.where(image == 6, o6, o_start)
    o_start = tl.where(image == 7, o7, o_start)
    rows = tl.where(image == 2, n2, rows)
    rows = tl.where(image == 3, n3, rows)
    rows = tl.where(image == 4, n4, rows)
    rows = tl.where(image == 5, n5, rows)
    rows = tl.where(image == 6, n6, rows)
    rows = tl.where(image == 7, n7, rows)
    temporal = tl.where(image == 2, t2, temporal)
    temporal = tl.where(image == 3, t3, temporal)
    temporal = tl.where(image == 4, t4, temporal)
    temporal = tl.where(image == 5, t5, temporal)
    temporal = tl.where(image == 6, t6, temporal)
    temporal = tl.where(image == 7, t7, temporal)
    height = tl.where(image == 2, h2, height)
    height = tl.where(image == 3, h3, height)
    height = tl.where(image == 4, h4, height)
    height = tl.where(image == 5, h5, height)
    height = tl.where(image == 6, h6, height)
    height = tl.where(image == 7, h7, height)
    width = tl.where(image == 2, w2, width)
    width = tl.where(image == 3, w3, width)
    width = tl.where(image == 4, w4, width)
    width = tl.where(image == 5, w5, width)
    width = tl.where(image == 6, w6, width)
    width = tl.where(image == 7, w7, width)

    local = (pid_r - p_start) * BLOCK_R + tl.arange(0, BLOCK_R)
    row_mask = local < rows
    spatial = local
    tile_area = merge_size * merge_size
    tile = spatial // tile_area
    within = spatial % tile_area
    y = (tile // (width // merge_size)) * merge_size + within // merge_size
    x = (tile % (width // merge_size)) * merge_size + within % merge_size

    yf = y.to(tl.float32) * (num_grid - 1) / (height - 1)
    xf = x.to(tl.float32) * (num_grid - 1) / (width - 1)
    y0 = yf.to(tl.int32)
    x0 = xf.to(tl.int32)
    y1 = tl.minimum(y0 + 1, num_grid - 1)
    x1 = tl.minimum(x0 + 1, num_grid - 1)
    dy = yf - y0
    dx = xf - x0

    w11 = dy * dx
    w10 = dy - w11
    w01 = dx - w11
    w00 = 1.0 - dy - w01
    w00 = w00.to(tl.bfloat16)
    w01 = w01.to(tl.bfloat16)
    w10 = w10.to(tl.bfloat16)
    w11 = w11.to(tl.bfloat16)

    cols = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = row_mask[:, None] & (cols[None, :] < hidden_dim)
    base00 = (y0 * num_grid + x0)[:, None] * hidden_dim + cols[None, :]
    base01 = (y0 * num_grid + x1)[:, None] * hidden_dim + cols[None, :]
    base10 = (y1 * num_grid + x0)[:, None] * hidden_dim + cols[None, :]
    base11 = (y1 * num_grid + x1)[:, None] * hidden_dim + cols[None, :]
    v00 = tl.load(weight + base00, mask=mask)
    v01 = tl.load(weight + base01, mask=mask)
    v10 = tl.load(weight + base10, mask=mask)
    v11 = tl.load(weight + base11, mask=mask)

    # The reference rounds each BF16 product before reducing the four corners.
    v00 = (v00 * w00[:, None]).to(tl.bfloat16)
    v01 = (v01 * w01[:, None]).to(tl.bfloat16)
    v10 = (v10 * w10[:, None]).to(tl.bfloat16)
    v11 = (v11 * w11[:, None]).to(tl.bfloat16)
    result = v00.to(tl.float32) + v01.to(tl.float32)
    result += v10.to(tl.float32) + v11.to(tl.float32)
    out_offsets = (o_start + local)[:, None] * hidden_dim + cols[None, :]
    tl.store(output + out_offsets, result, mask=mask)
    tl.store(
        output + out_offsets + rows * hidden_dim,
        result,
        mask=mask & (temporal == 2),
    )


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
        if not grid_thw_list:
            return torch.empty((0, self.hidden_size), dtype=dtype, device=device)
        if len(grid_thw_list) > 8:
            raise ValueError("at most eight grids are supported")

        grids = [tuple(map(int, g)) for g in grid_thw_list]
        rows = [h * w for _, h, w in grids]
        output_rows = [t * h * w for t, h, w in grids]
        output = torch.empty(
            (sum(output_rows), self.hidden_size), dtype=dtype, device=device
        )

        padded = grids + [(1, 2, 2)] * (8 - len(grids))
        padded_rows = rows + [0] * (8 - len(rows))
        block_r = 16
        block_d = 128
        program_offsets = [0]
        output_offsets = [0]
        for n, out_n in zip(rows, output_rows):
            program_offsets.append(program_offsets[-1] + triton.cdiv(n, block_r))
            output_offsets.append(output_offsets[-1] + out_n)

        terminal_p = program_offsets[-1]
        terminal_o = output_offsets[-1]
        p = program_offsets[1:-1] + [terminal_p] * (8 - len(rows))
        o = output_offsets[1:-1] + [terminal_o] * (8 - len(rows))
        h = [g[1] for g in padded]
        w = [g[2] for g in padded]
        t = [g[0] for g in padded]

        _interpolate[(terminal_p, triton.cdiv(self.hidden_size, block_d))](
            self._embed.emb.weight,
            output,
            self.hidden_size,
            self.num_grid_per_side,
            self.spatial_merge_size,
            *p,
            *o,
            *padded_rows,
            *t,
            *h,
            *w,
            BLOCK_R=block_r,
            BLOCK_D=block_d,
            num_warps=8 if len(grids) == 1 else 4,
        )
        return output
