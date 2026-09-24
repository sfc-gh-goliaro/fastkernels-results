"""Vision encoder rotary position embeddings.

Precomputes a cos/sin cache from fixed inv_freq (base=10000, no scaling).
forward() builds 2D (height, width) position IDs from grid_thw metadata,
shuffled by spatial_merge_size, and returns (cos, sin) tensors ready for
flash_attn's apply_rotary.

Optimization notes
------------------
The baseline spends its time on host-side bookkeeping, not on math: a per-image
numpy arange/broadcast/transpose/tile pipeline, a concatenate, a host->device
copy of an int64 [N, 2] index tensor, and two int64 advanced-index gathers.
None of that is necessary -- the spatial_merge_size shuffle is a closed form of
the flat token position, so a single Triton launch can derive the positions and
emit both outputs directly:

* ``hpos``/``wpos`` are recovered from the flat position ``p`` inside an image
  frame by digit-decomposing ``p`` in mixed radix ``(h/sms, w/sms, sms, sms)``.
  No index tensor is built, transferred or gathered.
* Per-image metadata (h, w, token offset, t) travels as *scalar kernel
  arguments*, so there is no H2D copy on the critical path at all and no pinned
  staging buffer to recycle.  Up to 8 images ride in one launch.
* One program writes a contiguous run of ``FP`` output elements, which keeps
  every store fully coalesced.  Grid y carries (image, frame), so a token's
  position inside its frame needs no ``% (h*w)``, and the one remaining
  non-constexpr division (``q // bwc``) uses a magic multiply.
* Both outputs live in one ``[2, N, D]`` allocation; ``out[0]`` / ``out[1]`` are
  contiguous ``[N, D]`` views, so a call costs exactly one allocation and one
  kernel launch.
* cos/sin are gathered from the existing fp32 cache and cast in-kernel, which
  is bit-exact with the baseline's cast-then-gather (verified over all 88
  captured grid lists, max_abs_error 0.0).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

# Images carried by a single launch. Captured workloads use <= 8; longer lists
# are processed in consecutive groups of this size.
_MAXIMG = 8

# Element tile per program, and warps per program. Chosen by sweeping
# FP x num_warps over the five benchmarked shapes; this pair puts every one of
# them on the measurement floor (~7.15 us, i.e. the cost of a single launch that
# writes 0.25-3.2 MB of computed bf16 with the L2 flushed).
_FP = 512
_NUM_WARPS = 1

# `q // bwc` is done as `(q * ceil(2**36 / bwc)) >> 36`, which is exact while
# q * bwc < 2**36 and q << 36 < 2**63. bwc <= w <= max_grid_size <= 8192 = 2**13,
# so q < 2**20 is a sufficient (and hugely slack -- captures have q <= 1024)
# condition for both. Anything larger falls back to a real division.
_MAGIC_Q_LIMIT = 1 << 20

_INT_ARGS = ["plane"] + [
    f"{p}{i}" for i in range(_MAXIMG) for p in ("h", "w", "o", "t")
]


@triton.jit(do_not_specialize=_INT_ARGS, do_not_specialize_on_alignment=_INT_ARGS)
def _vision_rope_kernel(
    out_ptr,  # [2, N, D], requested dtype; plane 0 = cos, plane 1 = sin
    cache_ptr,  # [max_grid_size, D] fp32; cols [0, HALF) cos, [HALF, D) sin
    plane,  # N * D: element distance from the cos plane to the sin plane
    h0, w0, o0, t0,
    h1, w1, o1, t1,
    h2, w2, o2, t2,
    h3, w3, o3, t3,
    h4, w4, o4, t4,
    h5, w5, o5, t5,
    h6, w6, o6, t6,
    h7, w7, o7, t7,
    SMS: tl.constexpr,  # spatial_merge_size
    D: tl.constexpr,  # rotary_dim == row width of both the cache and the output
    HALF: tl.constexpr,  # D // 2
    TMAX: tl.constexpr,  # max t in this group; grid y is (image, frame)
    FP: tl.constexpr,  # output elements per program
    MAGIC: tl.constexpr,  # use the magic multiply for `q // bwc`
):
    y = tl.program_id(1)
    img = y // TMAX
    frame = y % TMAX

    # This program's image metadata, selected out of the scalar argument bank.
    # Grid y spans exactly TMAX * (group size), so slots past it are never read.
    h, w, o, t = h0, w0, o0, t0
    m = img == 1
    h, w, o, t = tl.where(m, h1, h), tl.where(m, w1, w), tl.where(m, o1, o), tl.where(m, t1, t)
    m = img == 2
    h, w, o, t = tl.where(m, h2, h), tl.where(m, w2, w), tl.where(m, o2, o), tl.where(m, t2, t)
    m = img == 3
    h, w, o, t = tl.where(m, h3, h), tl.where(m, w3, w), tl.where(m, o3, o), tl.where(m, t3, t)
    m = img == 4
    h, w, o, t = tl.where(m, h4, h), tl.where(m, w4, w), tl.where(m, o4, o), tl.where(m, t4, t)
    m = img == 5
    h, w, o, t = tl.where(m, h5, h), tl.where(m, w5, w), tl.where(m, o5, o), tl.where(m, t5, t)
    m = img == 6
    h, w, o, t = tl.where(m, h6, h), tl.where(m, w6, w), tl.where(m, o6, o), tl.where(m, t6, t)
    m = img == 7
    h, w, o, t = tl.where(m, h7, h), tl.where(m, w7, w), tl.where(m, o7, o), tl.where(m, t7, t)

    if frame >= t:  # this image has fewer frames than the widest one
        return
    hw = h * w
    nelem = hw * D
    start = tl.program_id(0) * FP
    if start >= nelem:  # this image needs fewer tiles than the largest one
        return

    bwc = tl.maximum(w // SMS, 1)
    f = start + tl.arange(0, FP)
    valid = f < nelem

    row = f // D  # token index inside this frame
    col = f - row * D  # column inside the output row
    # Undo the spatial_merge_size block shuffle: the baseline's flattening of
    # (h//sms, sms, w//sms, sms) transposed to (bh, bw, ih, iw) means
    #   row == ((bh * bwc + bw) * SMS + ih) * SMS + iw.
    q = row // (SMS * SMS)
    rem = row - q * (SMS * SMS)
    ih = rem // SMS
    iw = rem - ih * SMS
    if MAGIC:
        bwc64 = bwc.to(tl.int64)
        mg = (68719476736 + bwc64 - 1) // bwc64  # ceil(2**36 / bwc), once per program
        bh = ((q.to(tl.int64) * mg) >> 36).to(tl.int32)
    else:
        bh = q // bwc
    bw = q - bh * bwc

    # Columns [0, HALF) come from cache row hpos, [HALF, D) from cache row wpos;
    # this matches the baseline's cos[pos_ids].flatten(1) ordering.
    lo = col < HALF
    src = cache_ptr + tl.where(lo, bh * SMS + ih, bw * SMS + iw) * D \
        + tl.where(lo, col, col - HALF)
    cos_v = tl.load(src, mask=valid, other=0.0)
    sin_v = tl.load(src + HALF, mask=valid, other=0.0)

    # A frame's output rows are contiguous, so this store run is contiguous too.
    dst = out_ptr + (o + frame * hw) * D + f
    ty = out_ptr.dtype.element_ty
    tl.store(dst, cos_v.to(ty), mask=valid)
    tl.store(dst + plane, sin_v.to(ty), mask=valid)


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
        self.rotary_dim = rotary_dim

    def forward(
        self,
        grid_thw_list: list[list[int]],
        spatial_merge_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        d = self.rotary_dim
        total = 0
        for t, h, w in grid_thw_list:
            total += t * h * w

        out = torch.empty((2, total, d), dtype=dtype, device=device)
        if total == 0:
            return out[0], out[1]

        sms2 = spatial_merge_size * spatial_merge_size
        plane = total * d
        cache = self.cos_sin_cache
        off = 0
        for lo in range(0, len(grid_thw_list), _MAXIMG):
            group = grid_thw_list[lo:lo + _MAXIMG]
            args = []
            tiles = 1
            tmax = 1
            magic = True
            for t, h, w in group:
                args += (h, w, off, t)
                off += t * h * w
                if t > tmax:
                    tmax = t
                n = -(-(h * w * d) // _FP)
                if n > tiles:
                    tiles = n
                if h * w >= _MAGIC_Q_LIMIT * sms2:
                    magic = False
            pad = group[0]
            args += (pad[1], pad[2], 0, 0) * (_MAXIMG - len(group))
            _vision_rope_kernel[(tiles, len(group) * tmax)](
                out, cache, plane, *args,
                SMS=spatial_merge_size, D=d, HALF=d // 2, TMAX=tmax, FP=_FP,
                MAGIC=magic, num_warps=_NUM_WARPS,
            )
        return out[0], out[1]
