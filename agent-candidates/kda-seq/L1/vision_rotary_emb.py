"""Vision encoder rotary position embeddings.

Precomputes a cos/sin cache from fixed inv_freq (base=10000, no scaling).
forward() builds 2D (height, width) position IDs from grid_thw metadata,
shuffled by spatial_merge_size, and returns (cos, sin) tensors ready for
flash_attn's apply_rotary.

The position-ID shuffle -- ``reshape(h//s, s, w//s, s).transpose(0, 2, 1, 3)`` --
is a fixed permutation with a closed form, so the per-token index array never has
to be materialized on the host and copied over. A single Triton kernel derives
each row's ``(hpos, wpos)`` from a handful of integer ops, gathers the fp32 cache
directly, and lets the store convert to the requested dtype. What crosses the bus
per call is a few words of per-grid metadata instead of an int64 index array the
size of the output.

Since a grid list is almost always short, the metadata usually travels as kernel
scalar arguments and no copy happens at all -- which also keeps a cold global read
off the critical path of every block. Longer lists fall back to a small device
tensor.

Inputs the kernel cannot serve fall through to ``_reference_forward``, a verbatim
copy of the original host implementation, so callers keep the original results
*and* the original exception types on malformed or exotic input.
"""

from __future__ import annotations

import contextlib
import operator

import numpy as np
import torch
import torch.nn as nn
import triton
import triton.language as tl

_FAST_DTYPES = (torch.bfloat16, torch.float16, torch.float32)

# Grid extents and the merge size have to be integers the original's numpy
# arithmetic would also accept. ``operator.index`` is deliberately not used as the
# test: it admits any object with ``__index__``, including ones the original then
# rejects when it compares them or hands them to np.arange.
_INT_TYPES = (int, np.integer)

# Only real sequences are accepted for the grid list and its triples, so the scan
# cannot drain a one-shot iterator before the fallback reads it.
_SEQ_TYPES = (list, tuple)

# nullcontext is stateless and re-entrant, so one instance serves every call that
# needs no device switch -- which is every call in a single-GPU process.
_NO_DEVICE_GUARD = contextlib.nullcontext()

# Grid lists no longer than this carry their metadata in the kernel's argument
# list; longer ones pay for a device tensor. Every captured call has 1, 2 or 8
# grids, and the argument list has to have a fixed arity, so the bound is set just
# past the observed maximum rather than made generous. It has to match the argument
# list of _vision_rope_arg_kernel, which is written out explicitly; the check below
# the kernel keeps the two from drifting.
_MAX_ARG_GRIDS = 8
# Padding for the unused argument slots, flat like the real metadata. t = 0 masks
# every row off and h = w = 1 keep the derived divisors non-zero, so a padded slot is
# inert even though no program is ever launched for one.
_ARG_PAD = (0, 1, 1) * _MAX_ARG_GRIDS

# Rows per program, and warps per row block. The whole output is ~2 MB and the
# kernel measures within noise of an empty launch, so these only need to be big
# enough not to multiply the launch count and small enough to spread ragged
# per-grid row counts over the SMs.
_BLOCK_R = 32
_NUM_WARPS = 4

# Launcher limits the eligibility scan has to respect, because an input can be
# perfectly valid arithmetically and still not be launchable.
#
# A tile is BLOCK_R x DIM elements and Triton refuses more than 2**20 of them, so a
# wide enough cache overflows the tile before it overflows anything else. Rows are
# independent, so BLOCK_R is simply reduced to fit -- measured: every DIM from 64 to
# 8192 compiles at a 8192-element tile, and DIM up to 131072 compiles at BLOCK_R = 1.
# Both operands are powers of two, which is what keeps the reduced BLOCK_R a legal
# tl.arange bound.
_TILE_BUDGET = 8192
_MAX_TILE_ELEMS = 2 ** 20

# The launch's second axis carries the grid index, and CUDA caps grid y at 65535 on
# every compute capability. Measured: 65535 grids launch, 65536 fails with
# "Triton Error [CUDA]: invalid argument".
_MAX_GRID_Y = 65535

# 32-bit index arithmetic is enough for every plausible grid; the scan refuses
# anything that could overflow it rather than silently wrapping.
_INDEX_MAX = 2 ** 31 - 1


@triton.jit
def _rope_tile(cache_ptr, out_ptr, n_rows, row_off, t, h, w, pid_r,
               S: tl.constexpr, HALF: tl.constexpr, DIM: tl.constexpr,
               BLOCK_R: tl.constexpr):
    """Emit one block of rows for a single grid, given that grid's metadata."""
    hw = h * w
    r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    row_mask = r < t * hw

    # Frames of a grid repeat the same 2-D positions, so fold onto one frame.
    i = r % hw
    # Flattening transpose(0, 2, 1, 3) enumerates (bh, bw, ih, iw) with strides
    # (s*w, s*s, s, 1) -- the block-row stride is s*s*(w//s), which equals s*w
    # because s divides w on this path.
    hpos = (i // (S * w)) * S + ((i // S) % S)
    wpos = ((i // (S * S)) % (w // S)) * S + (i % S)

    # Columns [0, HALF) come from hpos, columns [HALF, 2*HALF) from wpos. DIM is
    # rounded up to a power of two; the surplus lanes are masked off, which costs
    # nothing here and keeps the load/store pair boundary-free.
    d = tl.arange(0, DIM)
    src_row = tl.where(d[None, :] < HALF, hpos[:, None], wpos[:, None])
    col = tl.where(d < HALF, d, d - HALF)
    mask = row_mask[:, None] & (d < 2 * HALF)[None, :]

    # Gather fp32 and let the store round once, so the result is bit-identical to
    # converting the whole cache first and gathering afterwards.
    idx = src_row * (2 * HALF) + col[None, :]
    cos_v = tl.load(cache_ptr + idx, mask=mask)
    sin_v = tl.load(cache_ptr + idx + HALF, mask=mask)

    out = (row_off + r)[:, None] * (2 * HALF) + d[None, :]
    tl.store(out_ptr + out, cos_v, mask=mask)
    tl.store(out_ptr + n_rows * (2 * HALF) + out, sin_v, mask=mask)


# Every scalar Triton would otherwise specialize on, listed by name. Triton keys its
# compile cache on whether each integer argument is 1 or divides 16, so without this
# each new combination across the metadata list is a separate compilation -- measured
# at 8 cache entries over 14 grid lists, and ~1.2 ms per miss. Naming the arguments is
# what makes the suppression take effect: ``do_not_specialize`` is silently ignored
# for the elements of a tuple argument.
_ARG_NAMES = ["n_rows"] + [f"{n}{k}" for k in range(_MAX_ARG_GRIDS) for n in "thw"]


@triton.jit(do_not_specialize=_ARG_NAMES)
def _vision_rope_arg_kernel(cache_ptr, out_ptr, n_rows,
                            t0, h0, w0, t1, h1, w1, t2, h2, w2, t3, h3, w3,
                            t4, h4, w4, t5, h5, w5, t6, h6, w6, t7, h7, w7,
                            S: tl.constexpr, HALF: tl.constexpr, DIM: tl.constexpr,
                            BLOCK_R: tl.constexpr, MAX_G: tl.constexpr):
    """Metadata arrives as scalar arguments -- ``(t, h, w)`` per grid, padded to a
    fixed arity. Each program picks out its own grid and accumulates its row offset
    from the grids before it, which costs a few registers and saves both a host-side
    copy and a cold global read per block."""
    ts = (t0, t1, t2, t3, t4, t5, t6, t7)
    hs = (h0, h1, h2, h3, h4, h5, h6, h7)
    ws = (w0, w1, w2, w3, w4, w5, w6, w7)
    g = tl.program_id(1)
    t, h, w, row_off = 0, S, S, 0
    for k in tl.static_range(MAX_G):
        mine = g == k
        t = tl.where(mine, ts[k], t)
        h = tl.where(mine, hs[k], h)
        w = tl.where(mine, ws[k], w)
        row_off += tl.where(k < g, ts[k] * hs[k] * ws[k], 0)
    _rope_tile(cache_ptr, out_ptr, n_rows, row_off, t, h, w, tl.program_id(0),
               S, HALF, DIM, BLOCK_R)


if _vision_rope_arg_kernel.arg_names[2:2 + len(_ARG_NAMES)] != _ARG_NAMES:
    raise RuntimeError("_MAX_ARG_GRIDS does not match _vision_rope_arg_kernel's "
                       "argument list")


@triton.jit(do_not_specialize=["n_rows"])
def _vision_rope_kernel(meta_ptr, cache_ptr, out_ptr, n_rows,
                        S: tl.constexpr, HALF: tl.constexpr, DIM: tl.constexpr,
                        BLOCK_R: tl.constexpr):
    """Metadata arrives as an int32 ``[G, 4]`` tensor of ``(row_off, t, h, w)``.
    Used when the grid list is too long for the argument list."""
    meta = meta_ptr + tl.program_id(1) * 4
    _rope_tile(cache_ptr, out_ptr, n_rows,
               tl.load(meta + 0), tl.load(meta + 1),
               tl.load(meta + 2), tl.load(meta + 3),
               tl.program_id(0), S, HALF, DIM, BLOCK_R)


def _arg_words(triples: tuple[int, ...], n_grids: int) -> tuple[int, ...]:
    """``(t, h, w)`` per grid, padded to the argument kernel's fixed arity."""
    return triples + _ARG_PAD[3 * n_grids:]


def _meta_words(triples: tuple[int, ...], offsets: list[int]) -> list[int]:
    """``(row_off, t, h, w)`` per grid, the layout the metadata tensor carries."""
    return [v for k, off in enumerate(offsets)
            for v in (off, *triples[3 * k:3 * k + 3])]


def _tile_rows(dim: int) -> int:
    """Rows per program, reduced when a full block of them would not fit a tile.

    Both arguments are powers of two, so the result is one too and stays a legal
    ``tl.arange`` bound.
    """
    return max(1, min(_BLOCK_R, _TILE_BUDGET // dim))


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
        # Read the split off the cache rather than recomputing it from
        # rotary_dim: arange(0, rotary_dim, 2) has ceil(rotary_dim/2) entries, so
        # rotary_dim // 2 is wrong for odd rotary_dim.
        #
        # Named _half_width, not half: nn.Module.half() is the standard fp16
        # conversion, and an int attribute of that name shadows it, so callers doing
        # the ordinary thing would get "'int' object is not callable" instead of a
        # cast module.
        self._half_width = cache.shape[-1] // 2
        # The tile shape follows from the cache width alone, so it is settled here
        # rather than recomputed per call: triton.next_power_of_2 and triton.cdiv are
        # constexpr wrappers costing ~0.6 us each, which is real money against a ~36 us
        # host budget. A zero-width cache (rotary_dim = 0) has no tile at all, and the
        # scan refuses it before reading either value.
        self._dim = triton.next_power_of_2(2 * self._half_width) if self._half_width else 0
        self._block_r = _tile_rows(self._dim) if self._dim else 0

    def forward(
        self,
        grid_thw_list: list[list[int]],
        spatial_merge_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        plan = self._scan_grids(grid_thw_list, spatial_merge_size, dtype, device)
        if plan is None:
            return self._reference_forward(grid_thw_list, spatial_merge_size, dtype, device)
        triples, offsets, sms, n_rows, max_rows = plan

        cache = self.cos_sin_cache
        half = self._half_width
        n_grids = len(offsets)
        # cos and sin share one allocation. Both are read-only to the consumer
        # (flash_attn's apply_rotary calls .contiguous() and indexes by stride),
        # so the shared storage carries no hazard and saves an allocation.
        out = torch.empty((2, n_rows, 2 * half), dtype=dtype, device=cache.device)
        block_r = self._block_r
        launch = ((max_rows + block_r - 1) // block_r, n_grids)   # cdiv, spelled out
        # sms is the normalized int, not the caller's object: a numpy integer is a
        # perfectly good merge size to the original but cannot be a Triton
        # constexpr.
        common = dict(S=sms, HALF=half, DIM=self._dim, BLOCK_R=block_r,
                      num_warps=_NUM_WARPS)

        # Triton launches on the *current* device, not on the device its pointer
        # arguments live on, so a module sitting on a non-current GPU would otherwise
        # be handed cross-device pointers. Entering the guard costs ~1.3 us against
        # ~0.2 us to ask which device is current, so it is only paid when they differ
        # -- which on a single-GPU process is never.
        # (_scan_grids has already refused any call whose device does not resolve to
        # the cache's, so the only question left here is whether a switch is needed.)
        guard = (_NO_DEVICE_GUARD
                 if cache.device.index == torch.cuda.current_device()
                 else torch.cuda.device(cache.device))
        with guard:
            if n_grids <= _MAX_ARG_GRIDS:
                _vision_rope_arg_kernel[launch](cache, out, n_rows,
                                                *_arg_words(triples, n_grids),
                                                MAX_G=_MAX_ARG_GRIDS, **common)
            else:
                # Fresh pinned source and fresh device destination: the caching host
                # allocator event-guards reuse of the pinned block, and nothing
                # persistent is written, so concurrent calls on different streams
                # cannot scribble on each other.
                meta = _meta_words(triples, offsets)
                host_meta = torch.tensor(meta, dtype=torch.int32, pin_memory=True)
                dev_meta = torch.empty(len(meta), dtype=torch.int32, device=cache.device)
                dev_meta.copy_(host_meta, non_blocking=True)
                _vision_rope_kernel[launch](dev_meta, cache, out, n_rows, **common)
        return out[0], out[1]

    def _scan_grids(self, grid_thw_list, spatial_merge_size, dtype, device):
        """One pass that both validates the input and collects kernel metadata.

        Returns ``(triples, offsets, sms, n_rows, max_rows)`` for the kernel, or
        ``None`` when anything about the input puts it outside what the kernel
        handles -- in which case the caller replays the original host
        implementation, which is also what reproduces the original exceptions.

        Only real sequences are accepted, never arbitrary iterables: consuming a
        one-shot iterator here would hand a drained object to the fallback.
        """
        cache = self.cos_sin_cache
        if cache.dtype is not torch.float32 or not cache.is_cuda:
            return None
        # A zero-width cache (rotary_dim = 0) is legal for the original, which
        # returns an [N, 0] result, but there is no tile for the kernel to emit.
        if self._half_width < 1:
            return None
        if dtype not in _FAST_DTYPES:
            return None
        if not isinstance(grid_thw_list, _SEQ_TYPES) or not grid_thw_list:
            return None
        # The grid index is the launch's second axis, which CUDA caps.
        if len(grid_thw_list) > _MAX_GRID_Y:
            return None
        # A single row's tile has to fit even after BLOCK_R is reduced to one.
        if self._dim > _MAX_TILE_ELEMS:
            return None

        try:
            dev = torch.device(device)
        except (TypeError, ValueError, RuntimeError):
            return None
        if dev.type != "cuda":
            return None
        if dev.index is None:
            if cache.device.index != torch.cuda.current_device():
                return None
        elif dev.index != cache.device.index:
            return None

        # s >= 1 is checked on its own, ahead of any % or // by s: s == 0 must
        # reach the reference path to raise ZeroDivisionError there, and s < 0
        # would pass a divisibility test (4 % -2 == 0) while the original
        # reshape rejects it.
        if not isinstance(spatial_merge_size, _INT_TYPES):
            return None
        s = operator.index(spatial_merge_size)
        if s < 1:
            return None

        max_pos = cache.shape[0]
        triples: list[int] = []
        offsets: list[int] = []
        n_rows = 0
        max_rows = 0
        for triple in grid_thw_list:
            if not isinstance(triple, _SEQ_TYPES) or len(triple) != 3:
                return None
            t, h, w = triple
            if not (isinstance(t, _INT_TYPES) and isinstance(h, _INT_TYPES)
                    and isinstance(w, _INT_TYPES)):
                return None
            # operator.index is what normalizes a numpy integer to a plain int, which
            # the kernel needs for its constexpr and scalar arguments.
            t, h, w = operator.index(t), operator.index(h), operator.index(w)
            # Positivity, not just a nonzero product: the original tiles only
            # when t > 1, so every t <= 1 yields exactly one frame while the
            # closed form would compute t*h*w rows.
            if t < 1 or h < 1 or w < 1:
                return None
            if h % s or w % s:
                return None
            # Out-of-range positions must raise IndexError from the gather in the
            # reference path, not AssertionError from a guard here.
            if h > max_pos or w > max_pos:
                return None
            rows = t * h * w
            triples += (t, h, w)
            offsets.append(n_rows)
            n_rows += rows
            if rows > max_rows:
                max_rows = rows

        # The flat element count has to fit the kernel's 32-bit index arithmetic. The
        # row count is stated separately to match how the bound is specified, even
        # though _half_width >= 1 makes the element bound the stricter of the two.
        if n_rows > _INDEX_MAX or 2 * n_rows * 2 * self._half_width > _INDEX_MAX:
            return None
        return tuple(triples), offsets, s, n_rows, max_rows

    def _reference_forward(
        self,
        grid_thw_list: list[list[int]],
        spatial_merge_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The original host implementation, kept verbatim so that anything the
        kernel declines behaves exactly as it did before -- same results, same
        exception types."""
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
