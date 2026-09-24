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
This operator is a pure host-side index build: every input is a Python
list / scalar and the output is a CPU int64 tensor, so there is no device
work to move to the GPU (a launch + D2H copy alone costs more than the
whole op).  The win is in eliminating per-element interpreter and NumPy
dispatch overhead:

* ``_ext`` -- a JIT-compiled C++ extension (``cpp_extension.load_inline``)
  that walks the media items and writes all three rows of the output
  tensor in one pass, with no temporaries.  The whole forward becomes a
  single call.
* ``_forward_numpy`` -- fallback used when the extension cannot be built.
  Allocates the (3, N) result once and fills each segment through
  broadcast stores into reshaped views, instead of the baseline's
  ``np.indices`` + per-segment ``+`` temporaries + final ``concatenate``.

Both paths reproduce the baseline bit-for-bit, including the running
``st_idx = previous_block.max() + 1`` chaining and the truncating
``int64`` cast of the time axis after the ``t_factor`` scale.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# C++ fast path.  One call does argument normalization, the offset sort and
# the full (3, N) fill.  Built at import; any failure falls back to NumPy.
# ---------------------------------------------------------------------------
_CPP_SOURCE = r"""
#include <torch/extension.h>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <tuple>
#include <vector>

namespace {

struct Item {
  int64_t offset;
  int64_t t;
  int64_t h;
  int64_t w;
  double tf;
};

using GridList = std::vector<std::vector<int64_t>>;

// Truncating int64 cast of `idx * tf`, matching numpy's
// `(grid_indices[0] * t_factor).astype(np.int64)`.
inline int64_t scale_t(int64_t idx, double tf) {
  return static_cast<int64_t>(static_cast<double>(idx) * tf);
}

inline void fill_text(int64_t* p0, int64_t* p1, int64_t* p2, int64_t base,
                      int64_t n) {
  for (int64_t i = 0; i < n; ++i) {
    const int64_t v = base + i;
    p0[i] = v;
    p1[i] = v;
    p2[i] = v;
  }
}

}  // namespace

std::tuple<at::Tensor, int64_t> mrope_input_positions(
    int64_t n_tokens,
    int64_t merge,
    std::optional<GridList> image_grid_thw,
    std::optional<GridList> video_grid_thw,
    std::optional<std::vector<int64_t>> image_offsets,
    std::optional<std::vector<int64_t>> video_offsets,
    std::optional<std::vector<double>> video_second_per_grid,
    double tokens_per_second) {
  const auto opts = at::TensorOptions().dtype(at::kLong).device(at::kCPU);

  std::vector<Item> items;

  if (image_grid_thw && image_offsets && !image_grid_thw->empty() &&
      !image_offsets->empty()) {
    const GridList& g = *image_grid_thw;
    const std::vector<int64_t>& off = *image_offsets;
    const size_t n = std::min(g.size(), off.size());
    items.reserve(items.size() + n);
    for (size_t i = 0; i < n; ++i) {
      const std::vector<int64_t>& thw = g[i];
      TORCH_CHECK(thw.size() >= 3, "image_grid_thw entries must be (t, h, w)");
      items.push_back({off[i], thw[0], thw[1] / merge, thw[2] / merge, 1.0});
    }
  }

  if (video_grid_thw && video_offsets && !video_grid_thw->empty() &&
      !video_offsets->empty()) {
    const GridList& g = *video_grid_thw;
    const std::vector<int64_t>& off = *video_offsets;
    int64_t total_frames = 0;
    for (const auto& thw : g) {
      TORCH_CHECK(thw.size() >= 3, "video_grid_thw entries must be (t, h, w)");
      total_frames += thw[0];
    }
    const bool per_frame =
        static_cast<int64_t>(off.size()) == total_frames &&
        total_frames > static_cast<int64_t>(g.size());
    if (per_frame) {
      items.reserve(items.size() + off.size());
      size_t k = 0;
      for (const auto& thw : g) {
        const int64_t mh = thw[1] / merge;
        const int64_t mw = thw[2] / merge;
        for (int64_t f = 0; f < thw[0] && k < off.size(); ++f, ++k) {
          items.push_back({off[k], 1, mh, mw, 1.0});
        }
      }
    } else {
      const size_t n = std::min(g.size(), off.size());
      items.reserve(items.size() + n);
      for (size_t i = 0; i < n; ++i) {
        const std::vector<int64_t>& thw = g[i];
        double spg = 1.0;
        if (video_second_per_grid && i < video_second_per_grid->size()) {
          spg = (*video_second_per_grid)[i];
        }
        items.push_back({off[i], thw[0], thw[1] / merge, thw[2] / merge,
                         spg * tokens_per_second});
      }
    }
  }

  if (items.empty()) {
    at::Tensor out = at::empty({3, n_tokens}, opts);
    int64_t* p0 = out.data_ptr<int64_t>();
    fill_text(p0, p0 + n_tokens, p0 + 2 * n_tokens, 0, n_tokens);
    return {out, int64_t(0)};
  }

  std::stable_sort(items.begin(), items.end(),
                   [](const Item& a, const Item& b) {
                     return a.offset < b.offset;
                   });

  // Pass 1: total length (and the text run of each item, reused in pass 2).
  std::vector<int64_t> text_lens(items.size());
  int64_t total = 0;
  int64_t st = 0;
  for (size_t i = 0; i < items.size(); ++i) {
    const Item& it = items[i];
    int64_t text_len = it.offset - st;
    if (text_len < 0) text_len = 0;
    text_lens[i] = text_len;
    const int64_t n_vis = it.t * it.h * it.w;
    total += text_len + n_vis;
    st = it.offset + n_vis;
  }
  const int64_t tail = (st < n_tokens) ? (n_tokens - st) : 0;
  total += tail;

  at::Tensor out = at::empty({3, total}, opts);
  int64_t* row0 = out.data_ptr<int64_t>();
  int64_t* row1 = row0 + total;
  int64_t* row2 = row1 + total;

  // Pass 2: fill.  `base` is the baseline's running `st_idx`.
  int64_t pos = 0;
  int64_t base = 0;
  for (size_t i = 0; i < items.size(); ++i) {
    const Item& it = items[i];
    const int64_t text_len = text_lens[i];
    fill_text(row0 + pos, row1 + pos, row2 + pos, base, text_len);
    pos += text_len;

    const int64_t gb = base + text_len;  // grid block origin
    const bool scaled = it.tf != 1.0;
    for (int64_t ti = 0; ti < it.t; ++ti) {
      const int64_t tv = gb + (scaled ? scale_t(ti, it.tf) : ti);
      for (int64_t hi = 0; hi < it.h; ++hi) {
        const int64_t hv = gb + hi;
        int64_t* q0 = row0 + pos;
        int64_t* q1 = row1 + pos;
        int64_t* q2 = row2 + pos;
        for (int64_t wi = 0; wi < it.w; ++wi) {
          q0[wi] = tv;
          q1[wi] = hv;
          q2[wi] = gb + wi;
        }
        pos += it.w;
      }
    }

    const int64_t t_max = it.t > 0 ? (scaled ? scale_t(it.t - 1, it.tf)
                                             : it.t - 1)
                                   : 0;
    base = gb + std::max(t_max, std::max(it.h - 1, it.w - 1)) + 1;
  }

  int64_t final_max = base - 1;
  if (tail > 0) {
    fill_text(row0 + pos, row1 + pos, row2 + pos, base, tail);
    final_max = base + tail - 1;
  }

  return {out, final_max + 1 - n_tokens};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mrope_input_positions", &mrope_input_positions,
        "M-RoPE 3D input position build (CPU)");
}
"""


def _build_ext():
    import os

    from torch.utils.cpp_extension import load_inline

    prev = os.environ.get("TORCH_CUDA_ARCH_LIST")
    try:
        # Pure host code: keep nvcc / arch probing out of the build entirely.
        return load_inline(
            name="fk_mrope_input_positions_cpu",
            cpp_sources=_CPP_SOURCE,
            functions=None,
            extra_cflags=["-O3", "-fno-math-errno"],
            with_cuda=False,
            verbose=False,
        )
    finally:
        if prev is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = prev


try:
    _ext = _build_ext()
except Exception:  # pragma: no cover - compiler/toolchain unavailable
    _ext = None


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
        if _ext is not None:
            return _ext.mrope_input_positions(
                len(input_tokens),
                spatial_merge_size,
                image_grid_thw,
                video_grid_thw,
                image_offsets,
                video_offsets,
                None if video_second_per_grid is None
                else [float(x) for x in video_second_per_grid],
                float(tokens_per_second),
            )
        return _forward_numpy(
            input_tokens, spatial_merge_size, image_grid_thw, video_grid_thw,
            image_offsets, video_offsets, video_second_per_grid,
            tokens_per_second,
        )


def _forward_numpy(
    input_tokens,
    spatial_merge_size,
    image_grid_thw,
    video_grid_thw,
    image_offsets,
    video_offsets,
    video_second_per_grid,
    tokens_per_second,
):
    n_tokens = len(input_tokens)

    media_items: list[tuple[int, int, int, int, float]] = []
    if image_grid_thw and image_offsets:
        for i, (t, h, w) in enumerate(image_grid_thw):
            media_items.append((image_offsets[i], t, h // spatial_merge_size,
                                w // spatial_merge_size, 1.0))

    if video_grid_thw and video_offsets:
        total_frames = sum(thw[0] for thw in video_grid_thw)
        per_frame = len(video_offsets) == total_frames and total_frames > len(video_grid_thw)
        if per_frame:
            k = 0
            for t, h, w in video_grid_thw:
                merged_h = h // spatial_merge_size
                merged_w = w // spatial_merge_size
                for _ in range(t):
                    media_items.append((video_offsets[k], 1, merged_h, merged_w, 1.0))
                    k += 1
        else:
            for i, (t, h, w) in enumerate(video_grid_thw):
                second_per_grid = (
                    float(video_second_per_grid[i])
                    if video_second_per_grid and i < len(video_second_per_grid)
                    else 1.0
                )
                media_items.append((video_offsets[i], t, h // spatial_merge_size,
                                    w // spatial_merge_size,
                                    second_per_grid * tokens_per_second))

    if not media_items:
        out = np.empty((3, n_tokens), dtype=np.int64)
        out[:] = np.arange(n_tokens, dtype=np.int64)
        return torch.from_numpy(out), 0

    media_items.sort(key=lambda x: x[0])

    # Pass 1: segment lengths / total width.
    text_lens: list[int] = []
    total = 0
    st = 0
    for offset, grid_t, grid_h, grid_w in (m[:4] for m in media_items):
        text_len = offset - st
        if text_len < 0:
            text_len = 0
        text_lens.append(text_len)
        n_vis = grid_t * grid_h * grid_w
        total += text_len + n_vis
        st = offset + n_vis
    tail = n_tokens - st if st < n_tokens else 0
    total += tail

    out = np.empty((3, total), dtype=np.int64)
    pos = 0
    base = 0
    for (offset, grid_t, grid_h, grid_w, t_factor), text_len in zip(media_items, text_lens):
        if text_len:
            out[:, pos:pos + text_len] = np.arange(base, base + text_len, dtype=np.int64)
            pos += text_len
        gb = base + text_len
        n_vis = grid_t * grid_h * grid_w
        blk = out[:, pos:pos + n_vis]
        # w row: arange(w) tiled over the t*h leading frames/rows.
        blk[2].reshape(grid_t * grid_h, grid_w)[:] = np.arange(
            gb, gb + grid_w, dtype=np.int64)
        # h row: arange(h) repeated w times, tiled over t.
        blk[1].reshape(grid_t, grid_h, grid_w)[:] = np.arange(
            gb, gb + grid_h, dtype=np.int64)[:, None]
        # t row: one constant per frame.
        if grid_t == 1:
            blk[0] = gb
            t_max = 0
        else:
            tvals = np.arange(grid_t, dtype=np.int64)
            if t_factor != 1.0:
                tvals = (tvals * t_factor).astype(np.int64)
            blk[0].reshape(grid_t, grid_h * grid_w)[:] = (tvals + gb)[:, None]
            t_max = int(tvals[-1])
        pos += n_vis
        base = gb + max(t_max, grid_h - 1, grid_w - 1) + 1

    final_max = base - 1
    if tail:
        out[:, pos:pos + tail] = np.arange(base, base + tail, dtype=np.int64)
        final_max = base + tail - 1

    return torch.from_numpy(out), final_max + 1 - n_tokens
