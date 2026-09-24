"""Compute M-RoPE 3D position indices for text+vision token sequences.

Builds a (3, seq_len) position tensor where each row encodes temporal,
height, and width positions respectively. Text tokens get identical
positions across all three dimensions; vision tokens get 3D grid indices.

For Qwen3-VL videos, each frame is a separate block of video_token_id
tokens interleaved with timestamp/vision_start/vision_end tokens, so
video_offsets contains one entry per frame (not per video).

For Qwen2-VL videos, all frames are contiguous so video_offsets has one
entry per video, and the full (t, h, w) grid is used.

Implementation notes
--------------------
The sequence is a run of text tokens, then a grid block, then text, then a
grid block, and so on.  Position indices restart at ``max(previous block) + 1``
at every boundary, which the straightforward implementation expresses by
building one small array per run and per block, reducing each block with
``.max()``, and concatenating everything at the end.

This version emits the same numbers with a single allocation.  Two identities
make that possible:

* A grid block's largest index is ``max(temporal_max, merged_h - 1,
  merged_w - 1) + text_len + start_index``.  Every non-empty spatial grid
  contains coordinate 0 and the rescaled temporal row contains ``0 * factor``,
  so all three terms are non-negative and the block's largest index is at
  least ``text_len + start_index``.
* The text run *preceding* a non-empty grid block tops out at
  ``start_index + text_len - 1``, strictly below that floor.  So a single
  scalar running maximum serves both as "largest index of the previous block"
  (which sets the next run's start index) and as "largest index emitted so
  far" (which sets the returned delta).

The one place the second identity breaks is the *trailing* text run: it has no
following block, so it must raise the running maximum itself.  Likewise an
*empty* grid block contributes no maximum at all, and the text run before it
supplies one.

The total width needs no accumulator either.  Summing ``text_len + block``
over the items telescopes to the final ``st``, so the output width is
``max(st_after_last_item, len(input_tokens))``.

Two lookup tables keep the per-run and per-block cost down.  Both hold nothing
but index arithmetic, so neither depends on the arguments of any call:

* the integers 0, 1, 2, ..., sliced to supply a text run's values directly
  (``table[start:start + length]`` is already offset, so no temporary is
  needed) and to supply the row and column coordinates of a grid block;
* per grid shape, the (3, merged_h * merged_w) coordinate block of a
  single-frame grid at offset zero, so filling one is a single ``np.add`` into
  the output rather than three broadcast stores.  Measured here at 1.2-1.8 us
  against 2.4-3.0 us for the stores, across the recorded grid shapes.

Supported input domain
----------------------
Inside the domain below -- the production-valid one -- this implementation is
exactly equal to the reference on every input: same tensor element for element,
same delta.

Outside it, the families *this workspace tests* are enumerated in the three
groups after the domain. That enumeration is of the tested families, not of every
input a caller could construct: nothing here claims to characterise, say, a
string where an integer belongs, or an object whose ``__index__`` lies.

The domain is:

* ``spatial_merge_size`` dividing each grid so ``merged_h >= 1`` and
  ``merged_w >= 1``;
* offsets that leave room for their own block, i.e. non-overlapping and
  non-decreasing after the sort, so every text run has non-negative length;
* ``image_offsets`` at least as long as ``image_grid_thw``, and (in the
  per-video regime) ``video_offsets`` at least as long as ``video_grid_thw``;
* temporal factors small enough that every *emitted position index* stays
  representable in ``int64``.  The binding constraint is the largest index, not
  the factor: it is ``max(temporal_max, merged_h - 1, merged_w - 1)`` plus the
  running start index, and real factors (seconds per grid times tokens per
  second) leave that many orders of magnitude of headroom.

**(1) Outside the domain; the reference raises, and so does this, with the same
exception type.**

* an offset that overlaps the previous block, leaving a negative text run
  length: ``ValueError``, as the reference's broadcast to a negative width
  raises.  Two media items at one offset are this case whenever the first has a
  non-empty block;
* a degenerate grid (``merged_h == 0`` or ``merged_w == 0``) that the reference
  goes on to reduce -- because another media item follows it, or because a
  trailing text run does: ``ValueError``;
* every run and every block empty, so the reference reduces an empty
  concatenation: ``ValueError``;
* a negative grid dimension: ``ValueError``, as the reference's index
  construction raises.  Checked directly rather than inferred from the product
  of the dimensions, which a negative paired with a zero would hide;
* an offset list shorter than its grid list: ``IndexError``, as the reference's
  indexing raises;
* a temporal factor that is not a scalar: ``ValueError``, since neither
  ``t_factor == 1.0`` here nor ``t_factor != 1.0`` in the reference has a single
  truth value for an array.

**(2) Outside the domain; the reference returns, and this matches it.**

Two sub-cases, which differ in how strong the guarantee is:

* *Deterministic parity.* A **terminal** degenerate grid -- ``merged_h == 0`` or
  ``merged_w == 0`` on the last media item, with nothing after it, so the
  reference never reduces the empty block.  The reference succeeds, taking its
  maximum from the preceding text run, and returns a well-defined value; so does
  this.  ``image_grid_thw=[[1,1,64]]``, ``image_offsets=[4]``, merge 2 and four
  tokens gives width 4 and delta 0 from both.  This sits outside the domain
  because the domain requires positive merged dimensions, but it is not an error
  and the agreement is exact and platform-independent.
* *Platform-defined casting.*  A temporal factor that is not finite.
  ``0 * inf`` is ``nan``, and casting ``nan`` to ``int64`` is platform-defined,
  so the positions are not meaningful values at all.  The two implementations
  agree only because both apply the same float multiply and the same truncating
  cast to the same temporal row -- the agreement is real but it is agreement on
  whatever the platform produces, not on a specified number.

**(3) Outside the domain; the two deliberately disagree.  One family.**

* position indices that overflow ``int64``, which a temporal factor within about
  a factor of two of the largest ``int64`` produces -- roughly 1e18 and up.
  This is **outside** the exact-equivalence domain: the domain above requires
  every emitted position to be representable, and here they are not.  The
  emitted indices wrap, and the reference reduces the *wrapped* values, so its
  running maximum is the largest wrapped index.  This implementation's running
  maximum is a Python integer and does not wrap, so from that point the two
  differ: the tensors still match, the delta differs, and a following media item
  can raise ``OverflowError`` here where the reference returns a wrapped number.

  Matching it would mean reducing every emitted block, which is the cost this
  implementation exists to remove.  Both results are meaningless positions; the
  difference is that one of them is loud.  The delta arithmetic *is* made
  faithful -- ``max + 1 - len(input_tokens)`` wraps here exactly as the
  reference's ``int64`` computes it -- so this category is reached only when the
  indices themselves overflow, not merely when the delta expression does.
"""

from __future__ import annotations

from operator import itemgetter

import numpy as np
import torch
import torch.nn as nn

# Constant index table, read by every call and never written after
# construction.  It holds 0, 1, 2, ... and is sliced to supply a text run's
# values, the row and column coordinates of a grid block, and the temporal
# row.  It is a lookup table of the integers, not a cache of any result:
# nothing about its contents depends on the arguments of any call.
#
# Growth policy: on demand, doubling to keep repeated growth amortized, and
# never shrinking.  The invariant holds over the table's *lifetime*, not per
# call -- a request is served from the existing table whenever that is already
# long enough, so a small request after a large one returns the large table:
#
#   * the published table is read-only, and is never mutated or replaced by a
#     shorter one;
#   * every request is served a table at least as long as it asked for;
#   * the published length is at most
#     max(_INDEX_TABLE_MIN, 2 * largest_request_so_far).
#
# The last clause has to be stated against the largest request ever made, not
# against the current one.  Doubling can overshoot a single request -- asking for
# 1100 entries when the table holds 1024 produces 2048 -- and a later small
# request is then served that 2048-entry table rather than shrinking it.
#
# What bounds the requests themselves is the *caller*: no call asks for more than
# the width of the output it is building.  A text run read straight out of the
# table would need entries up to the run's highest position index, and a large
# temporal factor can push position indices arbitrarily far above the token
# count, so a run whose positions run past the table is written as a table slice
# plus an offset instead -- needing only as many entries as the run is long.
#
# Growth allocates a new array and rebinds the name; the existing array is never
# mutated, and every table is sealed non-writable before it is published.  So a
# concurrent caller holding the old reference keeps a valid table, and every
# caller uses the local reference it was handed.
_INDEX_TABLE_MIN = 1024
_INDEX_TABLE = np.arange(_INDEX_TABLE_MIN, dtype=np.int64)
_INDEX_TABLE.flags.writeable = False

# Coordinate blocks for single-frame grids, keyed by merged grid shape.  Entry
# ``(h, w)`` is the (3, h * w) block a grid of that shape occupies when its
# start index is 0: an all-zero temporal row, then the row and column
# coordinate of each cell.  Adding a scalar start index to it reproduces the
# whole block, so a cached shape costs one ``np.add`` instead of three
# broadcast stores.
#
# Bounded by a total element budget rather than an entry count, since a single
# large grid can cost more than many small ones.  Once the budget is spent, new
# shapes are filled with the broadcast stores instead and nothing is evicted --
# the shapes a real workload repeats are the ones already resident.  Both fill
# routes write identical values.
#
# The budget is approximate under concurrency, not a hard ceiling: the running
# total is a read-modify-write, so two threads first touching different shapes
# can both pass the check and overshoot by up to one template each.  Emitted
# values are unaffected -- a template is immutable once built, and a same-shape
# race only builds the same table twice and discards one.
_GRID_TEMPLATE_BUDGET = 1 << 20  # int64 entries, i.e. 8 MiB
_GRID_TEMPLATES: dict[tuple[int, int], np.ndarray] = {}
_GRID_TEMPLATE_ENTRIES = 0

_OFFSET = itemgetter(0)

_INT64_MIN = -(1 << 63)
_INT64_MAX = (1 << 63) - 1


def _wrap_int64(value: int) -> int:
    """*value* reduced to the range of an int64, as int64 arithmetic would."""
    return ((value + (1 << 63)) & ((1 << 64) - 1)) - (1 << 63)


def _index_table(size: int) -> np.ndarray:
    """Return a read-only table of 0, 1, 2, ... of at least *size* entries.

    The returned table may be considerably longer than *size*: an existing table
    is reused whenever it already suffices, so what bounds the length is the
    largest request the process has made, not this one. See the growth policy
    above.
    """
    global _INDEX_TABLE
    table = _INDEX_TABLE
    if table.shape[0] < size:
        table = np.arange(max(size, 2 * table.shape[0]), dtype=np.int64)
        table.flags.writeable = False
        if _INDEX_TABLE.shape[0] < table.shape[0]:
            _INDEX_TABLE = table
    return table


def _grid_template(grid_h: int, grid_w: int, table: np.ndarray) -> np.ndarray | None:
    """Return the coordinate block for a single-frame grid, or None if adding
    this shape would exceed the template budget."""
    global _GRID_TEMPLATE_ENTRIES
    block = grid_h * grid_w
    if _GRID_TEMPLATE_ENTRIES + 3 * block > _GRID_TEMPLATE_BUDGET:
        return None
    template = np.empty((3, block), dtype=np.int64)
    template[0] = 0
    template[1].reshape(grid_h, grid_w)[...] = table[:grid_h, None]
    template[2].reshape(grid_h, grid_w)[...] = table[:grid_w]
    template.flags.writeable = False
    _GRID_TEMPLATES[(grid_h, grid_w)] = template
    _GRID_TEMPLATE_ENTRIES += 3 * block
    return template


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
        media_items: list[tuple[int, int, int, int, float]] = []
        if image_grid_thw and image_offsets:
            for i, (t, h, w) in enumerate(image_grid_thw):
                media_items.append((image_offsets[i], t,
                                    h // spatial_merge_size,
                                    w // spatial_merge_size, 1.0))

        if video_grid_thw and video_offsets:
            total_frames = sum(thw[0] for thw in video_grid_thw)
            per_frame = (len(video_offsets) == total_frames
                         and total_frames > len(video_grid_thw))
            if per_frame:
                frame_offset_idx = 0
                for t, h, w in video_grid_thw:
                    merged_h = h // spatial_merge_size
                    merged_w = w // spatial_merge_size
                    for _ in range(t):
                        media_items.append(
                            (video_offsets[frame_offset_idx], 1,
                             merged_h, merged_w, 1.0)
                        )
                        frame_offset_idx += 1
            else:
                for i, (t, h, w) in enumerate(video_grid_thw):
                    second_per_grid = (
                        float(video_second_per_grid[i])
                        if video_second_per_grid and i < len(video_second_per_grid)
                        else 1.0
                    )
                    media_items.append(
                        (video_offsets[i], t,
                         h // spatial_merge_size, w // spatial_merge_size,
                         second_per_grid * tokens_per_second)
                    )

        seq_len = len(input_tokens)

        if not media_items:
            table = _index_table(seq_len)
            positions = np.empty((3, seq_len), dtype=np.int64)
            positions[:] = table[:seq_len]
            return torch.from_numpy(positions), 0

        if len(media_items) > 1:
            media_items.sort(key=_OFFSET)

        # Scalar walk over the items: validate the run lengths, find the widest
        # index table any of the fills will need, and reject the inputs whose
        # empty grid block the reference would reduce.  Nothing here touches
        # numpy or scales with the output width.
        st = 0
        table_size = 0
        empty_block = False
        for offset, grid_t, grid_h, grid_w, _ in media_items:
            if empty_block:
                raise ValueError("zero-size array to reduction operation "
                                 "maximum which has no identity")
            text_len = offset - st
            if text_len < 0:
                raise ValueError("all elements of broadcast shape must be "
                                 "non-negative")
            if grid_t < 0 or grid_h < 0 or grid_w < 0:
                # Checked rather than inferred from the product, which two
                # negative dimensions or a negative and a zero would hide.
                raise ValueError("negative dimensions are not allowed")
            if grid_h > table_size:
                table_size = grid_h
            if grid_w > table_size:
                table_size = grid_w
            if grid_t > table_size:
                table_size = grid_t
            block = grid_t * grid_h * grid_w
            empty_block = block == 0
            st = offset + block

        tail_len = seq_len - st
        if tail_len > 0:
            if empty_block:
                raise ValueError("zero-size array to reduction operation "
                                 "maximum which has no identity")
            width = seq_len
        else:
            width = st

        table = _index_table(width if width > table_size else table_size)
        table_len = table.shape[0]
        out = np.empty((3, width), dtype=np.int64)

        # Fill walk: write every text run and every grid block straight into
        # the output, carrying the scalar running maximum.  ``run_max`` is -1
        # until something is emitted, so the first start index is 0.
        st = 0
        run_max = -1
        col = 0
        for offset, grid_t, grid_h, grid_w, t_factor in media_items:
            text_len = offset - st
            start_index = run_max + 1
            if text_len:
                stop = start_index + text_len
                if stop <= table_len:
                    # The table already holds the run's values at this offset,
                    # so the slice is the run -- no temporary, no addition.
                    out[:, col:col + text_len] = table[start_index:stop]
                else:
                    out[:, col:col + text_len] = table[:text_len] + start_index
                col += text_len
            base = text_len + start_index
            block = grid_t * grid_h * grid_w
            if block:
                view = out[:, col:col + block]
                if grid_t == 1 and t_factor == 1.0:
                    # A single unscaled frame has temporal index 0 throughout,
                    # so the whole temporal row is the start index.  A frame
                    # count of one with a factor still takes the general path
                    # below: the factor is applied to index 0 there, which
                    # matters when it is not finite, since 0 * inf is nan and
                    # nan cast to int64 is not 0.
                    template = _GRID_TEMPLATES.get((grid_h, grid_w))
                    if template is None:
                        template = _grid_template(grid_h, grid_w, table)
                    if template is not None:
                        np.add(template, base, out=view)
                    else:
                        view[0] = base
                        view[1].reshape(grid_h, grid_w)[...] = \
                            table[:grid_h, None] + base
                        view[2].reshape(grid_h, grid_w)[...] = \
                            table[:grid_w] + base
                    temporal_max = 0
                else:
                    if t_factor == 1.0:
                        temporal_row = table[:grid_t]
                        temporal_max = grid_t - 1
                    else:
                        # Same float multiply and truncating cast the reference
                        # performs, so the two truncate identically.
                        temporal_row = (table[:grid_t] * t_factor).astype(np.int64)
                        temporal_max = int(temporal_row.max())
                    view[0].reshape(grid_t, grid_h, grid_w)[...] = \
                        (temporal_row + base)[:, None, None]
                    view[1].reshape(grid_t, grid_h, grid_w)[...] = \
                        table[:grid_h, None] + base
                    view[2].reshape(grid_t, grid_h, grid_w)[...] = \
                        table[:grid_w] + base
                col += block
                spatial_max = grid_h - 1 if grid_h > grid_w else grid_w - 1
                run_max = base + (temporal_max if temporal_max > spatial_max
                                  else spatial_max)
            elif text_len:
                # An empty grid block contributes no index, so the text run
                # just written is what raised the running maximum.
                run_max = start_index + text_len - 1
            st = offset + block

        if tail_len > 0:
            start_index = run_max + 1
            stop = start_index + tail_len
            if stop <= table_len:
                out[:, col:col + tail_len] = table[start_index:stop]
            else:
                out[:, col:col + tail_len] = table[:tail_len] + start_index
            run_max = start_index + tail_len - 1

        if run_max < 0:
            # Every run and every block was empty, so the reference reduces an
            # empty array.
            raise ValueError("zero-size array to reduction operation maximum "
                             "which has no identity")

        delta = run_max + 1 - seq_len
        if delta > _INT64_MAX or delta < _INT64_MIN:
            # The running maximum is a Python integer, so it does not wrap
            # where the reference's int64 arithmetic does.  Wrap explicitly,
            # in the same order and at the same width, when the result would
            # otherwise leave the range the reference computes in.
            delta = _wrap_int64(_wrap_int64(run_max + 1) - seq_len)
        return torch.from_numpy(out), delta
