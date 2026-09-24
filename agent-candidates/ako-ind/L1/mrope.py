"""Multi-dimensional Rotary Position Embedding (M-RoPE) for Qwen VL models.

Handles 3D position tensors (3, seq_len) representing temporal/height/width
dimensions.  Each dimension's positions index into a shared cos/sin cache, and
the resulting embeddings are assembled by section into the rotary dim.

The 2D (multimodal) path is one fused, allocation-free Triton kernel:

* The cos/sin table is cast to the activation dtype **once** and cached on the
  module.  For Qwen3-VL (max_position_embeddings=262144) the table is
  (1048576, 128) fp32 = 537 MB, so a per-call `cache.to(query.dtype)` moves
  ~805 MB for a call whose real q/k traffic is ~9 MB at seq_len=1000.
* `positions` is handed straight to the kernel (any stride).  There is no
  `cache[positions]` gather buffer, no `chunk`, and no `.contiguous()` copy:
  every lane derives its own T/H/W section from its own frequency index and
  reads only the row that section needs.
* q and k are rotated in place, by the same program, so the three cos/sin rows
  a token needs reach HBM once per token rather than once per tensor.

The single launch is a *programmatic dependent launch* (PDL, sm_90+).  M-RoPE
always consumes a tensor another kernel just produced (the QKV projection in
serving; the input `copy_` under the benchmark harness), and a plain launch
cannot begin dispatching its grid until that producer has fully retired.  With
`launch_pdl=True` the grid is dispatched onto the SMs while the producer drains
and every CTA parks on `gdc_wait()`, so the launch pipeline and the CTA ramp are
paid concurrently with the producer instead of after it.  `gdc_wait()` sits
before the first load of `positions`/`q`/`k` -- all three can be
producer-written -- so ordering is exact.

Launch geometry is `grid = (n_qh // HPP, cdiv(n_tok, BLOCK_T))`: a program owns
BLOCK_T tokens x HPP heads, walked BLOCK_H heads at a time, with head_dim/2
contiguous elements innermost so each thread carries a 128-bit vector.  The
tuple is switched on seq_len -- mid-length sequences need many small CTAs to
fill the SMs, while a 16k-token prefill is purely bandwidth-bound and wants fat
CTAs that read each cos/sin row exactly once.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl
from triton.language.extra.cuda import gdc_wait

from . import rotary_emb as _rotary_emb_reg  # noqa: F401 — registers fastkernels_rope ops

# `griddepcontrol` is sm_90+.  Resolved once, at import, and threaded through as
# a specialization constant so the instruction is never emitted where it is not
# legal and the launch attribute is never set where it does nothing.
try:
    _HAS_PDL = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9
except Exception:  # pragma: no cover - no driver / no device
    _HAS_PDL = False


# ---------------------------------------------------------------------------
# Fused M-RoPE kernel
# ---------------------------------------------------------------------------
@triton.jit
def _mrope_cos_sin(
    pos_ptr, cache_ptr, tt, tm, pos_s0, pos_s1,
    i, is_h, is_w, dmask,
    HD: tl.constexpr, HALF: tl.constexpr, MASK_D: tl.constexpr,
):
    """cos/sin for a (BT, PH) tile, assembled straight from `positions`.

    Reads the three candidate rows *contiguously* (one 128-bit vector per 8
    lanes) and picks per lane.  Letting each lane gather only its own element
    instead compiles to scalar `ld.global.b16`, for exactly the same DRAM
    traffic -- the union of the three rows is touched either way.
    """
    poff = tt.to(tl.int64) * pos_s1
    p_t = tl.load(pos_ptr + poff, mask=tm, other=0)
    p_h = tl.load(pos_ptr + pos_s0 + poff, mask=tm, other=0)
    p_w = tl.load(pos_ptr + 2 * pos_s0 + poff, mask=tm, other=0)
    ci = i[None, :].to(tl.int64)
    o_t = p_t[:, None] * HD + ci
    o_h = p_h[:, None] * HD + ci
    o_w = p_w[:, None] * HD + ci
    if MASK_D:
        dm = dmask[None, :]
        cos = tl.where(is_h[None, :], tl.load(cache_ptr + o_h, mask=dm, other=0.0),
              tl.where(is_w[None, :], tl.load(cache_ptr + o_w, mask=dm, other=0.0),
                                      tl.load(cache_ptr + o_t, mask=dm, other=0.0)))
        sin = tl.where(is_h[None, :], tl.load(cache_ptr + o_h + HALF, mask=dm, other=0.0),
              tl.where(is_w[None, :], tl.load(cache_ptr + o_w + HALF, mask=dm, other=0.0),
                                      tl.load(cache_ptr + o_t + HALF, mask=dm, other=0.0)))
    else:
        cos = tl.where(is_h[None, :], tl.load(cache_ptr + o_h),
              tl.where(is_w[None, :], tl.load(cache_ptr + o_w),
                                      tl.load(cache_ptr + o_t)))
        sin = tl.where(is_h[None, :], tl.load(cache_ptr + o_h + HALF),
              tl.where(is_w[None, :], tl.load(cache_ptr + o_w + HALF),
                                      tl.load(cache_ptr + o_t + HALF)))
    return cos.to(tl.float32)[:, None, :], sin.to(tl.float32)[:, None, :]


@triton.jit
def _mrope_rotate(
    base_ptr, row_stride, cos, sin, tm, i, dmask, h_lo,
    HD: tl.constexpr, HALF: tl.constexpr, NSPAN: tl.constexpr,
    NH: tl.constexpr, BT: tl.constexpr, BH: tl.constexpr,
    MASK_D: tl.constexpr, MASK_H: tl.constexpr,
):
    """Rotate heads [h_lo, h_lo+NSPAN) of a BT-token slab in place, BH at a time."""
    hbase = tl.arange(0, BH)
    toff = tl.arange(0, BT)[:, None, None] * row_stride
    ioff = i[None, None, :]
    for hb in tl.static_range(0, NSPAN, BH):
        hh = h_lo + hb + hbase
        off = toff + hh[None, :, None] * HD + ioff
        m = tm[:, None, None]
        if MASK_H:
            m = m & (hh < NH)[None, :, None]
        if MASK_D:
            m = m & dmask[None, None, :]
        a = tl.load(base_ptr + off, mask=m, other=0.0, eviction_policy='evict_first')
        b = tl.load(base_ptr + off + HALF, mask=m, other=0.0, eviction_policy='evict_first')
        af = a.to(tl.float32)
        bf = b.to(tl.float32)
        tl.store(base_ptr + off, (af * cos - bf * sin).to(a.dtype), mask=m,
                 eviction_policy='evict_first')
        tl.store(base_ptr + off + HALF, (bf * cos + af * sin).to(a.dtype), mask=m,
                 eviction_policy='evict_first')


@triton.jit
def _mrope_fused_kernel(
    q_ptr, k_ptr, pos_ptr, cache_ptr,
    n_tok, pos_s0, pos_s1,
    q_row_stride, k_row_stride,
    HD: tl.constexpr, HALF: tl.constexpr, PH: tl.constexpr,
    N_QH: tl.constexpr, N_KH: tl.constexpr, NSPAN_K: tl.constexpr,
    ST: tl.constexpr, SH: tl.constexpr, SW: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    BT: tl.constexpr, HPP: tl.constexpr, BHQ: tl.constexpr, BHK: tl.constexpr,
    MASK_D: tl.constexpr, MASK_HQ: tl.constexpr, MASK_HK: tl.constexpr,
    PDL: tl.constexpr,
):
    """grid = (n_qh // HPP, cdiv(n_tok, BT)).

    Axis 0 is the head block.  `program_id(0)` is the axis the hardware
    dispatches fastest, so the head blocks of one token block run concurrently
    on neighbouring SMs and share their three cos/sin rows through L2; that is
    what makes HPP < n_qh (more CTAs) affordable.  HPP == n_qh is the other end
    of the trade: fewest CTAs and each cos/sin row read exactly once, which is
    what a long prefill wants.  Head block 0 also owns its tokens of k, so k
    never re-reads a cos/sin row that q already fetched.
    """
    hb = tl.program_id(0)
    t0 = tl.program_id(1) * BT

    # Which mrope section (T / H / W) each frequency index belongs to.  This is
    # the point of the fusion: no gathered-then-masked cos/sin temporary, every
    # lane derives its own section from its own dim index.
    i = tl.arange(0, PH)
    dmask = i < HALF
    if INTERLEAVED:
        is_h = ((i % 3) == 1) & (i <= 3 * SH)
        is_w = ((i % 3) == 2) & (i <= 3 * SW)
    else:
        is_h = (i >= ST) & (i < ST + SH)
        is_w = i >= (ST + SH)

    tt = t0 + tl.arange(0, BT)
    tm = tt < n_tok
    # Grid-dependency barrier.  Everything above is address arithmetic on
    # compile-time values, so it is free to run during the producer's tail;
    # `positions`, `q` and `k` may all have been written by the immediately
    # preceding launch, so the wait goes here, before the first load of any of
    # them.  A no-op unless the producer explicitly triggers completion, and a
    # hard ordering guarantee when it does.
    if PDL:
        gdc_wait()
    cos, sin = _mrope_cos_sin(pos_ptr, cache_ptr, tt, tm, pos_s0, pos_s1,
                              i, is_h, is_w, dmask, HD, HALF, MASK_D)
    _mrope_rotate(q_ptr + t0.to(tl.int64) * q_row_stride, q_row_stride,
                  cos, sin, tm, i, dmask, hb * HPP,
                  HD, HALF, HPP, N_QH, BT, BHQ, MASK_D, MASK_HQ)
    if hb == 0:
        _mrope_rotate(k_ptr + t0.to(tl.int64) * k_row_stride, k_row_stride,
                      cos, sin, tm, i, dmask, 0,
                      HD, HALF, NSPAN_K, N_KH, BT, BHK, MASK_D, MASK_HK)


# ---------------------------------------------------------------------------
# Launch geometry
# ---------------------------------------------------------------------------
# (n_tok <=, BLOCK_T, HPP, BLOCK_H, num_warps, num_stages).
#
# BLOCK_T*BLOCK_H*(head_dim/2) elements over 32*num_warps threads sets the
# per-thread vector width (8 bf16 = one 128-bit access is the target), and
# (n_qh // HPP) * cdiv(n_tok, BLOCK_T) is the CTA count.  Measured on B200
# (148 SMs) over ~150 (BLOCK_T, HPP, BLOCK_H, num_warps, num_stages) points;
# the sweep tables are in ITERATIONS.md.  Mid-length sequences are latency- not
# bandwidth-bound and want small CTAs; the 16k prefill is at ~5.9 TB/s and wants
# HPP = n_qh so the cos/sin rows are read once.
_CONFIGS: tuple[tuple[int, int, int, int, int, int], ...] = (
    #  n_tok <=,  BT, HPP, BHQ, warps, stages
    (      4096,   2,   4,   4,     1,      1),
    (   1 << 30,   8,  16,   2,     2,      1),
)


def _pick(n_tok: int) -> tuple[int, int, int, int, int]:
    for limit, bt, hpp, bh, warps, stages in _CONFIGS:
        if n_tok <= limit:
            return bt, hpp, bh, warps, stages
    return 2, 4, 4, 1, 1


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
        # dtype -> cos/sin table in that dtype.  Built once, on first use, so the
        # 537 MB fp32 -> bf16 conversion never appears in a forward again.
        self._cast_cache: dict = {}

    # -- cached dtype cast ---------------------------------------------------
    def _cache_as(self, dtype: torch.dtype) -> torch.Tensor:
        cache = self.cos_sin_cache
        if cache.dtype == dtype:
            return cache
        got = self._cast_cache.get(dtype)
        if got is None or got.device != cache.device or got.shape != cache.shape:
            got = cache.to(dtype)
            self._cast_cache[dtype] = got
        return got

    def _apply_sgl_rope(self, positions_1d, query, key):
        """Apply standard RoPE for 1D positions (decode or text-only)."""
        cache = self._cache_as(query.dtype)
        if torch.compiler.is_compiling():
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
            cache,
            True,
        )
        return query, key

    def forward_native_2d(self, positions, query, key):
        """Pure PyTorch MRoPE for (3, seq_len) positions -- Inductor-friendly.

        Mirrors the Triton kernel: splits q/k into first/second half, gathers
        cos/sin per T/H/W section, and applies the standard neox-style rotation
        to all head_dim elements.
        """
        cache = self._cache_as(query.dtype)

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

        if torch.compiler.is_compiling() or positions.shape[0] != 3:
            return self.forward_native_2d(positions, query, key)

        hd = self.head_dim
        num_tokens = positions.shape[-1]
        if query.ndim == 2:
            n_qh = query.shape[1] // hd
            n_kh = key.shape[1] // hd
        else:
            n_qh = query.shape[1]
            n_kh = key.shape[1]

        q_flat = query.reshape(num_tokens, -1)
        k_flat = key.reshape(num_tokens, -1)
        if not q_flat.is_contiguous():
            q_flat = q_flat.contiguous()
        if not k_flat.is_contiguous():
            k_flat = k_flat.contiguous()

        if num_tokens:
            half = hd // 2
            ph = triton.next_power_of_2(half)
            bt, hpp, bh, warps, stages = _pick(num_tokens)
            pn_qh = triton.next_power_of_2(n_qh)
            hpp = min(pn_qh, hpp)
            bhq = min(hpp, bh)
            nspan_k = triton.next_power_of_2(n_kh)
            bhk = min(nspan_k, bh)
            grid = (pn_qh // hpp, (num_tokens + bt - 1) // bt)

            _mrope_fused_kernel[grid](
                q_flat, k_flat, positions, self._cache_as(query.dtype),
                num_tokens, positions.stride(0), positions.stride(1),
                n_qh * hd, n_kh * hd,
                hd, half, ph,
                n_qh, n_kh, nspan_k,
                self.mrope_section[0], self.mrope_section[1], self.mrope_section[2],
                self.mrope_interleaved,
                bt, hpp, bhq, bhk,
                ph != half, pn_qh != n_qh, nspan_k != n_kh,
                _HAS_PDL,
                num_warps=warps, num_stages=stages, launch_pdl=_HAS_PDL,
            )

        return q_flat.view_as(query), k_flat.view_as(key)

    def _apply_interleaved(self, x):
        """Reorganize from [TTT...HHH...WWW] to interleaved [THWTHW...]."""
        s = self.mrope_section
        result = x[0].clone()
        result[..., 1:s[1] * 3:3] = x[1, ..., 1:s[1] * 3:3]
        result[..., 2:s[2] * 3:3] = x[2, ..., 2:s[2] * 3:3]
        return result
