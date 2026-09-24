"""Compute M-RoPE 3D position indices for text+vision token sequences.

Builds a (3, seq_len) position tensor where each row encodes temporal,
height, and width positions respectively. Text tokens get identical
positions across all three dimensions; vision tokens get 3D grid indices.

For Qwen3-VL videos, each frame is a separate block of video_token_id
tokens interleaved with timestamp/vision_start/vision_end tokens, so
video_offsets contains one entry per frame (not per video).

For Qwen2-VL videos, all frames are contiguous so video_offsets has one
entry per video, and the full (t, h, w) grid is used.

Optimization notes
------------------
The captured workloads are tiny (349-772 output columns, 1-2 media blocks),
so wall time is entirely Python/NumPy per-call overhead rather than data
movement. Everything here is therefore aimed at cutting the *number* of
NumPy calls and temporaries:

* The output is allocated once and every block is written straight into a
  slice of it, so there is no per-block temporary and no final
  ``concatenate``/``reshape``.
* ``st_idx`` (the running position base) is carried as a Python int derived
  analytically from the previous block's grid extents, replacing the
  ``llm_pos_ids_list[-1].max()`` full-array scans. ``mrope_position_delta``
  falls out of the same counter, so the whole-buffer ``max()`` is gone too.
* ``np.indices((t, h, w))`` (three t*h*w int64 temporaries per block) is
  replaced by three broadcast assignments into a (3, t, h, w) *view* of the
  destination slice.
* Every ``arange`` is a read-only slice of one shared, lazily grown cache,
  so the offset adds (``+ st_idx``) become free as well.
"""

from __future__ import annotations

from operator import itemgetter

import numpy as np
import torch
import torch.nn as nn

_KEY = itemgetter(0)

# Shared read-only ramp. ``_RAMP[a:b]`` is a zero-copy view equal to
# ``np.arange(a, b)``, which removes both the arange allocation and the
# ``+ st_idx`` add from every block fill.
_RAMP = np.arange(4096, dtype=np.int64)


def _ramp(n: int) -> np.ndarray:
    """Return a ramp array with at least *n* entries (grown geometrically)."""
    global _RAMP
    if n > _RAMP.size:
        _RAMP = np.arange(max(n, 2 * _RAMP.size), dtype=np.int64)
    return _RAMP


class MRopeInputPositions(nn.Module):
    """Stateless module that computes M-RoPE 3D positions from token layout."""

    def forward(
        self,
        input_tokens: list[int],
        spatial_merge_size: int,
        image_grid_thw: list[list[int]] | None = None,
        video_grid_thw: list[list[int]] | None = None,
        image_offsets: list[int] | None = None,
        video_offsets: list[int] | None = None,
        video_second_per_grid: list[float] | None = None,
        tokens_per_second: float = 1.0,
    ) -> tuple[torch.Tensor, int]:
        n_tokens = len(input_tokens)
        sms = spatial_merge_size

        # ---- Pass 1: media blocks as (offset, t, merged_h, merged_w, t_factor).
        images = image_grid_thw if image_offsets else None
        videos = video_grid_thw if video_offsets else None

        if images:
            media_items = [
                (off, thw[0], thw[1] // sms, thw[2] // sms, 1.0)
                for off, thw in zip(image_offsets, images)
            ]
        else:
            media_items = []

        if videos:
            total_frames = 0
            for thw in videos:
                total_frames += thw[0]
            if len(video_offsets) == total_frames and total_frames > len(videos):
                # Qwen3-VL: one offset per frame, each frame its own 1-frame grid.
                i = 0
                for thw in videos:
                    mh = thw[1] // sms
                    mw = thw[2] // sms
                    for _ in range(thw[0]):
                        media_items.append((video_offsets[i], 1, mh, mw, 1.0))
                        i += 1
            else:
                # Qwen2-VL: one offset per video, full (t, h, w) grid.
                n_spg = len(video_second_per_grid) if video_second_per_grid else 0
                for i, thw in enumerate(videos):
                    spg = float(video_second_per_grid[i]) if i < n_spg else 1.0
                    media_items.append(
                        (video_offsets[i], thw[0], thw[1] // sms, thw[2] // sms,
                         spg * tokens_per_second)
                    )
        # The baseline sorts unconditionally; captured offsets are already
        # ascending, so scan first and only pay for the sort when they are not.
        n_items = len(media_items)
        if n_items > 1:
            prev = media_items[0][0]
            for k in range(1, n_items):
                cur = media_items[k][0]
                if cur < prev:
                    media_items.sort(key=_KEY)  # stable, like the baseline's
                    break
                prev = cur

        if not media_items:
            positions = np.broadcast_to(np.arange(n_tokens), (3, n_tokens))
            return torch.from_numpy(positions), 0

        # Blocks tile [0, st_end) contiguously, so the total width is known
        # from the last block alone -- no need to sum anything.
        off, gt, gh, gw, _ = media_items[-1]
        st_end = off + gt * gh * gw
        total = st_end if st_end >= n_tokens else n_tokens

        out = np.empty((3, total), dtype=np.int64)
        # Positions never exceed the column count, so ``total + 1`` entries are
        # always enough for every slice taken below.
        ramp = _ramp(total + 1)

        # ---- Pass 2: fill each block in place.
        st = 0
        st_idx = 0
        for offset, grid_t, grid_h, grid_w, t_factor in media_items:
            base = st_idx + offset - st
            if offset > st:
                out[:, st:offset] = ramp[st_idx:base]

            end = offset + grid_t * grid_h * grid_w
            view = out[:, offset:end].reshape(3, grid_t, grid_h, grid_w)
            if t_factor == 1.0:
                if grid_t == 1:
                    view[0] = base
                    t_max = 0
                else:
                    view[0] = ramp[base:base + grid_t].reshape(grid_t, 1, 1)
                    t_max = grid_t - 1
            else:
                # Match the baseline's truncate-toward-zero int cast of the
                # scaled temporal index, applied before the base offset.
                scaled = (np.arange(grid_t) * t_factor).astype(np.int64)
                t_max = int(scaled.max())
                view[0] = (scaled + base).reshape(grid_t, 1, 1)
            view[1] = ramp[base:base + grid_h].reshape(grid_h, 1)
            view[2] = ramp[base:base + grid_w]

            # Next block starts one past this grid block's largest position.
            block_max = grid_h - 1
            if grid_w > grid_h:
                block_max = grid_w - 1
            if t_max > block_max:
                block_max = t_max
            st = end
            st_idx = base + block_max + 1

        if st < n_tokens:
            end_idx = st_idx + n_tokens - st
            out[:, st:n_tokens] = ramp[st_idx:end_idx]
            st_idx = end_idx

        # The final block always holds the global maximum position (st_idx - 1).
        return torch.from_numpy(out), st_idx - n_tokens
