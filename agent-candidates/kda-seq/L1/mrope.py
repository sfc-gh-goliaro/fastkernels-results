"""Multi-dimensional Rotary Position Embedding (M-RoPE) for Qwen VL models.

Same contract as the baseline: 3D position tensors ``(3, seq_len)`` carrying
temporal/height/width positions, each indexing a shared ``[max_pos*4, head_dim]``
cos/sin cache, assembled by section into the rotary dim.

Two structural changes against the baseline, both aimed at what is a pure
streaming operation whose floor is one read plus one write of query and key:

* The baseline re-runs ``cos_sin_cache.to(query.dtype)`` on every call,
  converting the whole 537 MB float32 table to bf16 and discarding the result --
  126 us on B200, constant in the token count. The converted table is built once
  here and reused, keyed on the source buffer's identity, its version counter,
  and the requested dtype, so replacing, mutating, or relocating
  ``cos_sin_cache`` cannot serve a stale copy.
* The baseline's 2D path then materializes a ``(3, N, head_dim)`` gather, chunks
  it, and copies both halves contiguous before launching its rotate kernel --
  roughly five launches for one streaming pass. One kernel does it all here: it
  gathers its own cos/sin from the table using the per-lane selected position
  row, so no cos/sin temporary is ever allocated. At N=16384 that kernel moves
  9496 bytes per token in 36.9 us with a cold L2 -- 4.2 TB/s, about half of the
  device's HBM peak. (A repeated-buffer sweep reports 21.7 us / 7.2 TB/s, but that
  is cache-assisted and not the achieved bandwidth.) The four smaller captured
  shapes are bound by host-side launch cost rather than by the kernel, and that
  cost is a CUDA kernel launch itself -- measured at ~5.7 us of CPU here, against
  ~1.9 us for the surrounding Python. A hand-written CUDA extension reaching the
  same kernel was built and measured (see profile/mrope_native_v1/): it saves the
  Triton launcher's Python but not the driver launch, and came out a tie on the
  four small shapes and 2.9 % slower at N=16384, so it is not shipped.

Layout notes. Each cache row is ``[cos(head_dim/2) | sin(head_dim/2)]``, so lane
``j`` reads ``row*head_dim + j`` for cos and ``+ head_dim/2`` for sin. Each token
of query holds ``n_heads * head_dim`` values, and every element is rotated
(``rotary_dim == head_dim``), pairing ``j`` with ``j + head_dim/2`` inside each
head. ``positions`` is read through its real strides: four of the five captured
shapes carry ``stride=(1024, 1)``, and assuming a row stride of ``N`` there would
gather the wrong cos/sin silently. Negative positions are folded against the row
count before an address is formed, matching what ``cache[positions]`` does -- a raw
computation would otherwise read before the allocation -- and a row that is still
out of range after folding is forced to 0 and masked off, so no address is ever
formed outside the table. Set ``TRITON_DEBUG=1`` to have that case abort on device
the way advanced indexing does instead of yielding zeros.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

from . import rotary_emb as _rotary_emb_reg  # noqa: F401 — registers fastkernels_rope ops


@triton.jit
def _fused_mrope_kernel(
    q_ptr, k_ptr, cache_ptr, pos_ptr,
    q_row_stride, k_row_stride,
    pos_dim_stride, pos_tok_stride, cache_row_stride, cache_rows,
    n_qh: tl.constexpr, n_kh: tl.constexpr,
    hd: tl.constexpr, pad_half: tl.constexpr,
    mrope_section_t: tl.constexpr,
    mrope_section_h: tl.constexpr,
    mrope_section_w: tl.constexpr,
    is_interleaved: tl.constexpr,
):
    """One token per program: gather cos/sin once, then rotate head by head.

    Every tile is one-dimensional over the half-dim lanes, so the cos/sin tile
    has the same shape as each query tile and no broadcast is needed. That
    matters more than it looks: with a two-dimensional [n_heads, lanes] query
    tile, Triton removes the resulting layout conversion by rematerializing the
    gather at the wider layout, which re-runs it once per head -- measured at 276
    L2 sectors per token against an ideal of 72, and enough integer address work
    to make the kernel ALU-bound rather than bandwidth-bound.

    The head loop is a Python ``range`` over a ``tl.constexpr`` bound, so it is
    unrolled unconditionally; the resulting batch of 32 independent loads and
    stores is what gives a 32-thread program enough memory-level parallelism.
    """
    pid = tl.program_id(0)
    out_dtype = q_ptr.dtype.element_ty
    half_rd: tl.constexpr = hd // 2

    lane = tl.arange(0, pad_half)
    lane_mask = lane < half_rd

    # Which of the three position rows feeds each lane. Both masks are built
    # from tl.constexpr section sizes, so they fold to compile-time constants:
    # no lookup table, and nothing stands between the position load and the
    # cos/sin gather that depends on it. The expressions mirror the baseline's
    # exactly, including the inclusive `<= 3 * section` bound (harmless there:
    # 3*section is divisible by 3 and can never also satisfy lane % 3 != 0).
    if is_interleaved:
        h_mask = ((lane % 3) == 1) & (lane <= 3 * mrope_section_h)
        w_mask = ((lane % 3) == 2) & (lane <= 3 * mrope_section_w)
    else:
        t_end = mrope_section_t
        h_end = t_end + mrope_section_h
        h_mask = (t_end <= lane) & (lane < h_end)
        w_mask = (h_end <= lane) & (lane < half_rd)

    pos = pos_ptr + pid.to(tl.int64) * pos_tok_stride
    # `cache[positions]` accepts negative indices and counts from the end, so a
    # raw address computation has to fold them the same way or it reads before the
    # allocation. Three scalar folds are cheaper than one over the lane vector.
    p_t = tl.load(pos)
    p_h = tl.load(pos + pos_dim_stride)
    p_w = tl.load(pos + 2 * pos_dim_stride)
    p_t = tl.where(p_t < 0, p_t + cache_rows, p_t)
    p_h = tl.where(p_h < 0, p_h + cache_rows, p_h)
    p_w = tl.where(p_w < 0, p_w + cache_rows, p_w)
    # The baseline's t lanes are the complement of the h and w lanes.
    row = tl.where(h_mask, p_h, tl.where(w_mask, p_w, p_t))

    # Folding is not enough on its own: a position of exactly `cache_rows` stays
    # positive and out of range, and one below `-cache_rows` folds to a value that
    # is still negative. `cache[positions]` rejects both, so no address may be
    # formed from either. The row is forced to 0 for pointer arithmetic and the
    # load is masked off, which makes an out-of-range index safe rather than a
    # read outside the allocation. Under TRITON_DEBUG the assert also aborts on
    # device, matching what advanced indexing does; it compiles away otherwise.
    in_range = (row >= 0) & (row < cache_rows)
    tl.device_assert(in_range, "mrope: position out of range for cos_sin_cache")
    row = tl.where(in_range, row, 0)
    valid = lane_mask & in_range

    # Each cache row is [cos(half_rd) | sin(half_rd)]. `evict_last` asks L2 to keep
    # table lines in preference to the query and key lines streaming past them,
    # which are read once and never revisited. Measured over 13 interleaved passes
    # with every variant pre-warmed: 2.9 % at N=16384 with non-overlapping ranges
    # (36.85-37.89 us against 35.81-35.87), and nothing on the four launch-bound
    # shapes, where the kernel is not what the time goes on.
    cos_off = row * cache_row_stride + lane
    cos = tl.load(cache_ptr + cos_off, mask=valid, other=0.0,
                  eviction_policy="evict_last").to(tl.float32)
    sin = tl.load(cache_ptr + cos_off + half_rd, mask=valid, other=0.0,
                  eviction_policy="evict_last").to(tl.float32)

    # int64 only for the row base; the in-row lane offsets stay 32-bit.
    q_base = q_ptr + pid.to(tl.int64) * q_row_stride + lane
    for h in range(n_qh):
        head = q_base + h * hd
        q1 = tl.load(head, mask=lane_mask, other=0.0).to(tl.float32)
        q2 = tl.load(head + half_rd, mask=lane_mask, other=0.0).to(tl.float32)
        # fp32 multiply-add, rounded once at the store; the baseline rounds twice.
        tl.store(head, (q1 * cos - q2 * sin).to(out_dtype), mask=lane_mask)
        tl.store(head + half_rd, (q2 * cos + q1 * sin).to(out_dtype), mask=lane_mask)

    k_base = k_ptr + pid.to(tl.int64) * k_row_stride + lane
    for h in range(n_kh):
        head = k_base + h * hd
        k1 = tl.load(head, mask=lane_mask, other=0.0).to(tl.float32)
        k2 = tl.load(head + half_rd, mask=lane_mask, other=0.0).to(tl.float32)
        tl.store(head, (k1 * cos - k2 * sin).to(out_dtype), mask=lane_mask)
        tl.store(head + half_rd, (k2 * cos + k1 * sin).to(out_dtype), mask=lane_mask)


def _memo_fresh(memo: tuple, cache: torch.Tensor, dtype: torch.dtype) -> bool:
    """Whether *memo* was derived from exactly this table, for this dtype.

    ``data_ptr`` catches a storage swap through ``.data``, which leaves both the
    object identity and the version counter untouched; shape and stride catch a
    metadata-only rebind onto the same storage; ``_version`` catches an in-place
    write; identity catches a replaced or relocated buffer. Compared field by field
    with short-circuiting rather than as a built tuple, because this runs on every
    call and the cheap fields reject first.
    """
    return (memo[0] is cache
            and memo[1] == cache.data_ptr()
            and memo[2] == cache._version
            and memo[3] is cache.dtype
            and memo[4] == cache.shape
            and memo[5] == cache.stride()
            and memo[6] is dtype)


def _lane_tile(hd: int) -> tuple[int, int]:
    """Padded lane extent and a warp count, chosen statically.

    Each tile is the ``head_dim/2`` lanes of one head, so the warp count follows
    the lane count alone: one warp per 64 lanes, i.e. two elements per thread.
    Measured on B200 at head_dim=128, one warp beat two by 19 % at N=16384 --
    fewer, wider-scheduled programs win over splitting a 64-lane tile.
    Deliberately not ``triton.autotune``: compiling several configs can leave a
    background thread alive into the harness's guarded warmup region.
    """
    pad_half = triton.next_power_of_2(hd // 2)
    return pad_half, max(1, min(8, pad_half // 64))


class MRotaryEmbedding(nn.Module):
    """M-RoPE for Qwen2-VL / Qwen3-VL.

    positions can be either:
      - 1D (seq_len,) for text-only (all 3 dims identical -> standard RoPE)
      - 2D (3, seq_len) for multimodal (T/H/W positions differ)

    mrope_section: list of 3 ints [t, h, w] summing to rotary_dim // 2
    """

    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int,
        rope_theta: float,
        mrope_section: list[int],
        mrope_interleaved: bool = False,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.rotary_dim = head_dim
        self.mrope_section = mrope_section
        self.mrope_interleaved = mrope_interleaved
        assert sum(mrope_section) == head_dim // 2

        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim))
        t = torch.arange(max_position_embeddings * 4, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1).float()
        self.register_buffer("cos_sin_cache", cache, persistent=False)

        # (source, data_ptr, version, src dtype, shape, stride, dtype, converted).
        # A plain attribute rather than a buffer: it is derived state, must stay
        # out of state_dict(), and is rebuilt whenever any field stops matching --
        # which is also what makes a module.to(device) safe.
        self._cast_cache = None
        # Lane tile and warp count depend only on head_dim, so they are settled
        # once here rather than recomputed on every call.
        self._pad_half, self._num_warps = _lane_tile(head_dim)


    def _cast_cache_for(self, dtype: torch.dtype):
        """``cos_sin_cache`` in *dtype* with its row stride and row count.

        Converted at most once per source table.

        Storing it in the compute dtype rather than float32 halves the cos/sin
        bytes streamed and makes the values bit-identical to the baseline's,
        which rotates against a bf16-rounded table too.

        Identity alone is not enough. ``.to(device)`` rebinds the buffer to a new
        object and an in-place write bumps the version counter, but assigning
        ``cache.data = other`` swaps the storage underneath the *same* object
        without necessarily touching either -- so the key also carries the storage
        pointer and the full layout. Built on first use rather than in
        ``__init__``: the compute dtype is not known at construction, and the
        module is moved to the device only after it.
        """
        cache = self.cos_sin_cache
        if cache.dtype == dtype and cache.stride(-1) == 1:
            # Nothing derived is needed; drop the memo so it stops pinning a
            # table this module will never read again.
            self._cast_cache = None
            return cache, cache.stride(0), cache.shape[0]
        memo = self._cast_cache
        if memo is not None and _memo_fresh(memo, cache, dtype):
            return memo[7], memo[8], memo[9]
        converted = cache.to(dtype)
        if converted.stride(-1) != 1:
            # The kernel indexes columns by lane, so the row must be packed.
            # Paid once here rather than assumed.
            converted = converted.contiguous()
        # The row stride and row count travel with the table: they only change when
        # it is rebuilt, so the hot path should not re-read them every call.
        self._cast_cache = (cache, cache.data_ptr(), cache._version, cache.dtype,
                            cache.shape, cache.stride(), dtype, converted,
                            converted.stride(0), converted.shape[0])
        return converted, converted.stride(0), converted.shape[0]

    def _apply_sgl_rope(self, positions_1d, query, key):
        """Apply standard RoPE for 1D positions (decode or text-only)."""
        if torch.compiler.is_compiling():
            # Traced: convert inline so the conversion is part of the graph and
            # Inductor fuses it into the gather, exactly as the baseline does.
            cache = self.cos_sin_cache
            if cache.dtype != query.dtype:
                cache = cache.to(query.dtype)
            from .rotary_emb import RotaryEmbedding
            return RotaryEmbedding.forward_native(
                positions_1d,
                query.view(query.shape[0], -1),
                key.view(key.shape[0], -1),
                self.head_dim, cache,
            )
        torch.ops.fastkernels_rope.rotary_embedding(
            positions_1d,
            query.view(query.shape[0], -1),
            key.view(key.shape[0], -1),
            self.head_dim,
            self._cast_cache_for(query.dtype)[0],
            True,
        )
        return query, key

    def forward_native_2d(self, positions, query, key):
        """Pure PyTorch MRoPE for (3, seq_len) positions -- Inductor-friendly.

        Mirrors the Triton kernel: splits q/k into first/second half, gathers
        cos/sin per T/H/W section, and applies the standard neox-style rotation
        to all head_dim elements.
        """
        cache = self.cos_sin_cache
        if cache.dtype != query.dtype:
            cache = cache.to(query.dtype)

        num_tokens = query.shape[0]
        cos_sin = cache[positions]          # (3, seq_len, head_dim)
        cos, sin = cos_sin.chunk(2, dim=-1) # each (3, seq_len, head_dim/2)

        if self.mrope_interleaved:
            cos = self._apply_interleaved(cos)
            sin = self._apply_interleaved(sin)
        else:
            cos = torch.cat(
                [m[i] for i, m in enumerate(cos.split(self.mrope_section, dim=-1))],
                dim=-1,
            )
            sin = torch.cat(
                [m[i] for i, m in enumerate(sin.split(self.mrope_section, dim=-1))],
                dim=-1,
            )
        # cos, sin: (seq_len, head_dim/2)

        hd = self.head_dim
        half = hd // 2
        q_shape = query.shape
        k_shape = key.shape
        q = query.view(num_tokens, -1, hd)
        k = key.view(num_tokens, -1, hd)

        cos = cos.unsqueeze(1)  # (seq_len, 1, head_dim/2)
        sin = sin.unsqueeze(1)

        q1 = q[..., :half]
        q2 = q[..., half:]
        k1 = k[..., :half]
        k2 = k[..., half:]

        new_q = torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)
        new_k = torch.cat([k1 * cos - k2 * sin, k2 * cos + k1 * sin], dim=-1)

        return new_q.view(q_shape), new_k.view(k_shape)

    def forward(self, positions, query, key):
        """Apply M-RoPE in-place.

        Args:
            positions: (seq_len,) or (3, seq_len) int64 tensor
            query: (seq_len, num_heads, head_dim)
            key: (seq_len, num_kv_heads, head_dim)
        """
        if positions.ndim == 1:
            return self._apply_sgl_rope(positions, query, key)

        if torch.compiler.is_compiling():
            return self.forward_native_2d(positions, query, key)

        # 2D M-RoPE: positions (3, seq_len) with potentially different T/H/W dims (multimodal prefill)
        num_tokens = positions.shape[-1]
        hd = self.head_dim

        # The kernel walks each token's n_heads*head_dim values as one packed
        # block at an arbitrary row stride, and writes in place. Two things must
        # hold: the values within a token are adjacent, and distinct tokens do
        # not share storage -- a row stride below the block width (0 for an
        # expanded tensor) would have every program writing the same bytes.
        q_stride = query.stride()
        k_stride = key.stride()
        if query.ndim == 2:
            n_qh = query.shape[1] // hd
            n_kh = key.shape[1] // hd
            q_packed = q_stride[1] == 1 and q_stride[0] >= n_qh * hd
            k_packed = k_stride[1] == 1 and k_stride[0] >= n_kh * hd
        else:
            n_qh = query.shape[1]
            n_kh = key.shape[1]
            q_packed = (q_stride[2] == 1 and q_stride[1] == hd
                        and query.shape[2] == hd and q_stride[0] >= n_qh * hd)
            k_packed = (k_stride[2] == 1 and k_stride[1] == hd
                        and key.shape[2] == hd and k_stride[0] >= n_kh * hd)

        # Anything else falls back to what the baseline itself does: rotate a
        # contiguous copy and return that, leaving the caller's tensor unmutated.
        if q_packed:
            q_buf, q_row_stride = query, q_stride[0]
        else:
            q_buf = query.reshape(num_tokens, -1).contiguous()
            q_row_stride = q_buf.stride(0)
        if k_packed:
            k_buf, k_row_stride = key, k_stride[0]
        else:
            k_buf = key.reshape(num_tokens, -1).contiguous()
            k_row_stride = k_buf.stride(0)

        cache, cache_row_stride, cache_rows = self._cast_cache_for(query.dtype)
        pos_stride = positions.stride()
        _fused_mrope_kernel[(num_tokens,)](
            q_buf, k_buf, cache, positions,
            q_row_stride, k_row_stride,
            pos_stride[0], pos_stride[1], cache_row_stride, cache_rows,
            n_qh, n_kh, hd, self._pad_half,
            self.mrope_section[0], self.mrope_section[1], self.mrope_section[2],
            self.mrope_interleaved,
            num_warps=self._num_warps,
        )

        if q_buf is not query:
            q_buf = q_buf.view(query.shape)
        if k_buf is not key:
            k_buf = k_buf.view(key.shape)
        return q_buf, k_buf

    def _apply_interleaved(self, x):
        """Reorganize from [TTT...HHH...WWW] to interleaved [THWTHW...]."""
        s = self.mrope_section
        result = x[0].clone()
        result[..., 1:s[1] * 3:3] = x[1, ..., 1:s[1] * 3:3]
        result[..., 2:s[2] * 3:3] = x[2, ..., 2:s[2] * 3:3]
        return result
