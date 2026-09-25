from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl
import triton.language.extra.libdevice as libdevice


@triton.jit
def _flux_pos_embed_kernel(
    ids,
    inv_freq,
    cos_out,
    sin_out,
    n_rows: tl.constexpr,
    out_dim: tl.constexpr,
    axis0_width: tl.constexpr,
    axis1_width: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, 64)
    mask = (rows[:, None] < n_rows) & (cols[None, :] < out_dim)

    axis = tl.where(
        cols < axis0_width,
        0,
        tl.where(cols < axis0_width + axis1_width, 1, 2),
    )
    pos = tl.load(ids + rows[:, None] * 3 + axis[None, :], mask=mask)
    angle = pos.to(tl.float32) * tl.load(
        inv_freq + cols[None, :], mask=cols[None, :] < out_dim
    )

    offsets = rows[:, None] * out_dim + cols[None, :]
    tl.store(cos_out + offsets, libdevice.fast_cosf(angle), mask=mask)
    tl.store(sin_out + offsets, libdevice.fast_sinf(angle), mask=mask)


class FluxPosEmbed(nn.Module):
    def __init__(self, theta: int, axes_dim: list[int] | tuple[int, ...]):
        super().__init__()
        self.theta = theta
        self.axes_dim = list(axes_dim)

        freqs = []
        for dim in self.axes_dim:
            freqs.extend(theta ** (-(2.0 * i) / dim) for i in range(dim // 2))
        self._inv_freq = torch.tensor(freqs, dtype=torch.float32)

    def forward(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not ids.is_cuda:
            pos = ids.float()
            pieces = []
            for axis, dim in enumerate(self.axes_dim):
                freq = torch.arange(0, dim, 2, dtype=torch.float64, device=ids.device)
                pieces.append(pos[:, axis, None] * self.theta ** (-freq / dim))
            angles = torch.cat(pieces, dim=-1)
            return angles.cos(), angles.sin()

        if self._inv_freq.device != ids.device:
            self._inv_freq = self._inv_freq.to(ids.device)

        n_rows = ids.shape[0]
        out_dim = sum(self.axes_dim) // 2
        cos_out = torch.empty((n_rows, out_dim), dtype=torch.float64, device=ids.device)
        sin_out = torch.empty_like(cos_out)
        block_m = 8
        _flux_pos_embed_kernel[(triton.cdiv(n_rows, block_m),)](
            ids,
            self._inv_freq,
            cos_out,
            sin_out,
            n_rows,
            out_dim,
            self.axes_dim[0] // 2,
            self.axes_dim[1] // 2,
            BLOCK_M=block_m,
            num_warps=4,
        )
        return cos_out, sin_out
