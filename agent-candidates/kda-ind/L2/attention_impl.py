"""Attention layer with a Triton fast path for dense unpaged causal prefill.

The layer inherits everything from the baseline and overrides exactly one
dispatch site.  A shape guard decides, per call, whether the varlen prefill can
be served by the local Triton kernel; every regime the guard declines (paged
prefill, decode, mixed batches, tree verify, the Triton-unified route, unusual
layouts, large problems) falls through to the baseline unchanged.

Two measured facts shape the design:

* The benchmark brackets each call between CUDA events *after* enqueueing a
  zero-fill of twice the device L2 (~252 MiB, ~70 us of device time on B200).
  The host therefore runs far ahead of the device, and any submit cost below
  that shadow never reaches the score.  Both this path (~28 us) and the bundled
  FlashAttention-4 launcher (~46 us) sit under it, so the figure of merit is the
  kernel's own *device* time, not launch overhead.
* Every benched case but one is MQA/GQA with a wide ratio (16 query heads per KV
  head).  A grid over query heads would re-read the same K/V tiles once per
  head, which for these sizes costs far more than the arithmetic.  So query
  heads that share a KV head are packed into one program's rows and the K/V
  tiles are read once per group.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from fastkernels.infra.triton_attention_helpers import softmax_step
from fastkernels.tasks.baseline.L2.attention_impl import Attention as _BaselineAttention

# Env switches are resolved once, at import, so the submit path only ever tests
# a module-level bool.  ``FK_ATTN_NO_FAST=1`` forces the guard to decline (used
# to show that declining reproduces the baseline bit for bit); ``FK_ATTN_COUNT=1``
# enables the dispatch tally that proves the fast path really ran.
_FAST_DISABLED = os.environ.get("FK_ATTN_NO_FAST") == "1"
_COUNT = os.environ.get("FK_ATTN_COUNT") == "1"

DISPATCH_COUNTS = {"fast": 0, "tma": 0, "fallback": 0}

# Guard verdicts.  One call decides the route so the submit path walks the
# preconditions once.
_DECLINE = 0
_ROUTE_PACKED = 1
_ROUTE_TMA = 2

# bf16 only.  fp16 would very likely work, but nothing here verifies it -- the
# captured domain is bf16 throughout -- and a guard must never be wider than the
# proven domain.  An fp16 layer falls through to the baseline.
_SUPPORTED_DTYPES = (torch.bfloat16,)
_SUPPORTED_HEAD_DIMS = (64, 128)

# Rows per program, and the KV tile width.  Measured over
# ``profile/probe_config.py``: 32 rows beat 64 and 128 on every size this path
# serves, because the problems are small enough that more, smaller programs fill
# the device better than fewer fat ones, and an fp32 [32, 128] accumulator keeps
# the register file comfortable.
_BLOCK_M = 32
_BLOCK_N = 64

# Dispatch budget, in query-key pairs per query head.  The kernel's device time
# grows with ``max_seqlen_q * min(max_seqlen_k, window)`` while FlashAttention-4
# has a much higher floor but a far better slope (it reaches the tensor cores
# through tcgen05, which Triton does not).  The measured crossover on B200 is
# ~400 tokens for full causal attention and ~800 with a 128-key window -- both
# close to 256*256 pairs -- so the budget is set at that product, one step below
# the crossover rather than at it.
_MAX_TOKEN_PAIRS = 256 * 256

# Absolute cap: sizes past this are compute bound whatever the window, and the
# launch geometry beyond it is not part of the tested domain.
_MAX_SEQLEN_FAST = 4096


def _launch_plan(num_heads: int, heads_per_kv: int, head_dim: int):
    """The complete, immutable launch plan for one layer.

    Returns ``(block_q, grid_heads, out_row_stride, heads_per_prog, head_dim,
    block_n, num_warps, num_stages)`` -- everything a launch needs except the
    grid's query extent, which depends on the call.  Built once per layer and
    cached, so the submit path does no arithmetic beyond one ``cdiv``.

    ``heads_per_prog`` query heads sharing a KV head are packed into the rows of
    a single program, alongside ``block_q`` query positions, so that a program
    holds ``block_q * heads_per_prog == _BLOCK_M`` rows and reads each K/V tile
    once for all of them.  Packing needs a power of two that divides the GQA
    ratio; a ratio without one (an odd ratio, say) simply runs unpacked.

    ``num_stages`` is the only tuned entry that varies: head dim 64 has half the
    shared-memory footprint per KV tile, so it affords one more stage.

    Row count and KV tile come from the offline search in
    ``profile/probe_config.py``; nothing here is decided at run time.
    """
    heads_per_prog = 1
    while (
        heads_per_prog * 2 <= heads_per_kv
        and heads_per_kv % (heads_per_prog * 2) == 0
        and heads_per_prog * 2 <= _BLOCK_M
    ):
        heads_per_prog *= 2
    return (
        _BLOCK_M // heads_per_prog,
        num_heads // heads_per_prog,
        num_heads * head_dim,
        heads_per_prog,
        head_dim,
        _BLOCK_N,
        4,
        2 if head_dim >= 128 else 3,
    )


@triton.jit
def _dense_causal_attn_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    sink_ptr,
    cu_seqlens_q_ptr,
    cu_seqlens_k_ptr,
    scale,
    stride_q_tok,
    stride_k_tok,
    stride_v_tok,
    stride_o_tok,
    HEADS_PER_KV: tl.constexpr,
    HEADS_PER_PROG: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    USE_SINKS: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
):
    """One program per (query-position block, sequence, query-head group).

    Rows are ``(query position, query head)`` pairs -- position major, head
    minor -- so ``HEADS_PER_PROG`` heads that share a KV head see the same K/V
    tiles from registers.

    Keys are addressed in the sequence's own coordinates: query row ``m`` of a
    sequence with ``q_len`` queries over ``k_len`` keys sits at absolute key
    position ``m + key_offset`` where ``key_offset = k_len - q_len``, which puts
    the causal boundary in the right place for prefix-cached sequences as well
    as for the dense ``k_len == q_len`` case.
    """
    tile_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)
    group = tl.program_id(2)

    q_start = tl.load(cu_seqlens_q_ptr + seq_idx)
    q_len = tl.load(cu_seqlens_q_ptr + seq_idx + 1) - q_start
    pos_start = tile_idx * BLOCK_Q
    if pos_start >= q_len:
        return

    k_start = tl.load(cu_seqlens_k_ptr + seq_idx)
    k_len = tl.load(cu_seqlens_k_ptr + seq_idx + 1) - k_start
    key_offset = k_len - q_len

    offs_m = tl.arange(0, BLOCK_Q * HEADS_PER_PROG)
    offs_d = tl.arange(0, HEAD_DIM)
    pos = pos_start + offs_m // HEADS_PER_PROG
    head = group * HEADS_PER_PROG + offs_m % HEADS_PER_PROG
    row_valid = pos < q_len
    # Absolute key position of the newest key each query row may attend to.
    q_abs = pos + key_offset

    q = tl.load(
        q_ptr
        + (q_start + pos)[:, None] * stride_q_tok
        + head[:, None] * HEAD_DIM
        + offs_d[None, :],
        mask=row_valid[:, None],
        other=0.0,
    )

    kv_head = (group * HEADS_PER_PROG) // HEADS_PER_KV
    k_base = k_ptr + k_start * stride_k_tok + kv_head * HEAD_DIM
    v_base = v_ptr + k_start * stride_v_tok + kv_head * HEAD_DIM

    # An attention sink is one extra logit per head that carries no value
    # vector: seeding the online softmax with ``M = sink`` and ``L = 1`` puts
    # exactly ``exp(sink - m)`` into the denominator and nothing into the
    # numerator.  ``sinks = zeros`` is therefore *not* a no-op -- it adds 1 to
    # every row's denominator.  Rows of one program span several heads, so the
    # seed is gathered per row rather than broadcast.
    if USE_SINKS:
        M = tl.load(sink_ptr + head, mask=row_valid, other=float("-inf")).to(tl.float32)
        L = tl.full([BLOCK_Q * HEADS_PER_PROG], 1.0, dtype=tl.float32)
    else:
        M = tl.full([BLOCK_Q * HEADS_PER_PROG], float("-inf"), dtype=tl.float32)
        L = tl.zeros([BLOCK_Q * HEADS_PER_PROG], dtype=tl.float32)
    acc = tl.zeros([BLOCK_Q * HEADS_PER_PROG, HEAD_DIM], dtype=tl.float32)

    # Causal upper bound: no row in this block can see past its own position.
    last_key = tl.minimum(pos_start + BLOCK_Q - 1 + key_offset, k_len - 1)
    loop_hi = last_key // BLOCK_N + 1
    if SLIDING_WINDOW > 0:
        # A window of W keeps keys with ``q_abs - k < W``, so the oldest key any
        # row in this block can see is ``pos_start + key_offset - W + 1``.
        loop_lo = tl.maximum(pos_start + key_offset - SLIDING_WINDOW + 1, 0) // BLOCK_N
    else:
        loop_lo = 0

    for tile in range(loop_lo, loop_hi):
        offs_n = tile * BLOCK_N + tl.arange(0, BLOCK_N)
        col_valid = offs_n < k_len
        k = tl.load(
            k_base + offs_n[:, None] * stride_k_tok + offs_d[None, :],
            mask=col_valid[:, None],
            other=0.0,
        )
        s = tl.dot(q, tl.trans(k)) * scale
        keep = row_valid[:, None] & col_valid[None, :] & (offs_n[None, :] <= q_abs[:, None])
        if SLIDING_WINDOW > 0:
            keep = keep & (q_abs[:, None] - offs_n[None, :] < SLIDING_WINDOW)
        s = tl.where(keep, s, float("-inf"))

        M, L, P, alpha = softmax_step(s, M, L)
        v = tl.load(
            v_base + offs_n[:, None] * stride_v_tok + offs_d[None, :],
            mask=col_valid[:, None],
            other=0.0,
        )
        acc = acc * alpha[:, None] + tl.dot(P.to(v.dtype), v)

    # Rows outside the sequence never collect a finite logit; keep their
    # denominator at 1 so the epilogue stays finite even though it is masked off.
    L = tl.where(L > 0.0, L, 1.0)
    tl.store(
        o_ptr
        + (q_start + pos)[:, None] * stride_o_tok
        + head[:, None] * HEAD_DIM
        + offs_d[None, :],
        (acc / L[:, None]).to(o_ptr.dtype.element_ty),
        mask=row_valid[:, None],
    )


def dense_causal_prefill(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    num_seqs,
    scale,
    heads_per_kv,
    sinks=None,
    sliding_window=0,
    plan=None,
):
    """Causal varlen prefill over dense (unpaged) K/V.

    ``q`` is ``[total_q, num_heads, head_dim]`` and ``k``/``v`` are
    ``[total_k, num_kv_heads, head_dim]``, both possibly row-strided views of a
    wider fused-QKV buffer; the head and head-dim strides must be packed.  The
    output is a freshly allocated contiguous tensor -- never a module-owned
    buffer, which would alias across calls.

    ``plan`` is the layer's cached launch plan from ``_launch_plan``; it is built
    here only when the wrapper is called directly (tests, probes).
    """
    if plan is None:
        plan = _launch_plan(q.shape[1], heads_per_kv, q.shape[2])
    (block_q, grid_heads, out_row_stride, heads_per_prog, head_dim,
     block_n, num_warps, num_stages) = plan
    out = torch.empty((q.shape[0], grid_heads * heads_per_prog, head_dim),
                      dtype=q.dtype, device=q.device)
    grid = (triton.cdiv(max_seqlen_q, block_q), num_seqs, grid_heads)
    _dense_causal_attn_kernel[grid](
        q,
        k,
        v,
        out,
        sinks,
        cu_seqlens_q,
        cu_seqlens_k,
        scale,
        q.stride(0),
        k.stride(0),
        v.stride(0),
        out_row_stride,
        heads_per_kv,
        heads_per_prog,
        block_q,
        head_dim,
        block_n,
        sinks is not None,
        sliding_window,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


# ---------------------------------------------------------------------------
# Mid-size specialization
# ---------------------------------------------------------------------------
# The kernel above serves the small cases, where the device is nearly empty and
# what matters is doing little work.  Between roughly 256 and 1000 tokens the
# problem is large enough that the memory pipeline decides the result, and there
# FlashAttention-4 wins by a mechanism the loop above does not use: bulk TMA
# copies into shared memory, a producer/consumer split across warp groups, and
# tiles wide enough to reach the Blackwell tensor cores.  Triton 3.6 exposes all
# three, so this variant uses them:
#
#   * Q/K/V arrive as host-built tensor descriptors and are loaded with
#     `descriptor.load`, i.e. through TMA, which also handles bounds (out-of-range
#     rows read as zero) so the inner loop carries no masks.
#   * the KV loop is `tl.range(..., warp_specialize=True)`, the surface Triton
#     documents for Blackwell.
#   * 64 rows -- four query positions by the sixteen query heads that share a KV
#     head -- because a `tl.dot` narrower than 64 rows does not lower to
#     `tcgen05` at all (see `profile/mma_lowering.json`).
#   * query blocks are issued in alternating low/high order, so a block near the
#     end of the sequence (which walks every KV tile) is co-scheduled with one
#     near the start (which walks one), instead of all the long blocks landing in
#     the same wave.
_TMA_BLOCK_Q = 4
# 64, not 32: with BLOCK_N=64 the KV tile is wide enough that the two products
# per iteration amortise the softmax dependency chain, which measured 1.13-1.15x
# against FlashAttention-4 in the dispatch window versus 0.8x at BLOCK_N=32.
_TMA_BLOCK_N = 64
_TMA_HEADS = 16
_TMA_HEAD_DIM = 128
_TMA_WARPS = 4
# TMA needs the descriptor's base address 16-byte aligned and every non-innermost
# stride a multiple of 16 bytes; for bf16 that is 8 elements.
_TMA_ALIGN_ELEMS = 8
# Measured window.  On three leases each (`profile/band/window{1,2,3}/`) this
# specialization runs 1.15x FlashAttention-4 at 257 tokens, 1.13x at 400, 1.00x at
# 512 and 0.91x at 656, so it takes over exactly where the packed kernel above
# stops winning and stops short of where it would start losing.
_TMA_MIN_SEQLEN = 257
_TMA_MAX_SEQLEN = 448
_TMA_STAGES = 3


@triton.jit
def _dense_causal_attn_tma_kernel(
    q_desc,
    k_desc,
    v_desc,
    o_desc,
    scale,
    num_q_tiles,
    seq_len,
    HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_N: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
    INTERLEAVE: tl.constexpr,
):
    """Single-segment dense causal prefill over TMA-staged tiles.

    One program per query block; its rows are ``(query position, query head)``
    pairs, position major, so all ``HEADS`` query heads of a KV head share one
    read of each K/V tile.  Restricted to one sequence and to ``k_len == q_len``,
    which is what the guard admits.
    """
    tile_idx = tl.program_id(0)
    if INTERLEAVE:
        # Alternating low/high issue order: pair a short causal block with a long
        # one rather than letting the long tail all start in the last wave.
        half = tile_idx // 2
        tile = tl.where(tile_idx % 2 == 0, half, num_q_tiles - 1 - half)
    else:
        tile = tile_idx
    pos_start = tile * BLOCK_Q

    offs_m = tl.arange(0, BLOCK_Q * HEADS)
    pos = pos_start + offs_m // HEADS
    q_abs = pos  # single segment, k_len == q_len, so key_offset is 0

    q = q_desc.load([pos_start, 0, 0]).reshape(BLOCK_Q * HEADS, HEAD_DIM)

    M = tl.full([BLOCK_Q * HEADS], float("-inf"), dtype=tl.float32)
    L = tl.zeros([BLOCK_Q * HEADS], dtype=tl.float32)
    acc = tl.zeros([BLOCK_Q * HEADS, HEAD_DIM], dtype=tl.float32)

    last_key = tl.minimum(pos_start + BLOCK_Q - 1, seq_len - 1)
    loop_hi = last_key // BLOCK_N + 1

    for kv in tl.range(0, loop_hi, warp_specialize=WARP_SPECIALIZE):
        offs_n = kv * BLOCK_N + tl.arange(0, BLOCK_N)
        k = k_desc.load([kv * BLOCK_N, 0, 0]).reshape(BLOCK_N, HEAD_DIM)
        s = tl.dot(q, tl.trans(k)) * scale
        s = tl.where(offs_n[None, :] <= q_abs[:, None], s, float("-inf"))
        M, L, P, alpha = softmax_step(s, M, L)
        v = v_desc.load([kv * BLOCK_N, 0, 0]).reshape(BLOCK_N, HEAD_DIM)
        acc = acc * alpha[:, None] + tl.dot(P.to(v.dtype), v)

    L = tl.where(L > 0.0, L, 1.0)
    o_desc.store([pos_start, 0, 0],
                 (acc / L[:, None]).reshape(BLOCK_Q, HEADS, HEAD_DIM).to(o_desc.dtype))


def _tma_aligned(t):
    """Can this tensor back a TMA descriptor?

    The innermost dimension must be contiguous and every other stride, plus the
    storage offset, must land on a 16-byte boundary.
    """
    return (
        t.stride(2) == 1
        and t.stride(0) % _TMA_ALIGN_ELEMS == 0
        and t.stride(1) % _TMA_ALIGN_ELEMS == 0
        and t.storage_offset() % _TMA_ALIGN_ELEMS == 0
        and t.data_ptr() % 16 == 0
    )


def dense_causal_prefill_tma(q, k, v, seq_len, scale, *, block_q=_TMA_BLOCK_Q,
                             block_n=_TMA_BLOCK_N, num_warps=_TMA_WARPS,
                             num_stages=_TMA_STAGES, warp_specialize=False,
                             interleave=True):
    """Mid-size dense causal prefill for one sequence, staged through TMA.

    ``q`` is ``[seq_len, 16, 128]`` and ``k``/``v`` are ``[seq_len, 1, 128]``.
    """
    from triton.tools.tensor_descriptor import TensorDescriptor

    heads = q.shape[1]
    head_dim = q.shape[2]
    out = torch.empty((seq_len, heads, head_dim), dtype=q.dtype, device=q.device)
    q_desc = TensorDescriptor(q, [seq_len, heads, head_dim], list(q.stride()),
                              [block_q, heads, head_dim])
    k_desc = TensorDescriptor(k, [seq_len, 1, head_dim], list(k.stride()),
                              [block_n, 1, head_dim])
    v_desc = TensorDescriptor(v, [seq_len, 1, head_dim], list(v.stride()),
                              [block_n, 1, head_dim])
    o_desc = TensorDescriptor(out, [seq_len, heads, head_dim], list(out.stride()),
                              [block_q, heads, head_dim])
    num_q_tiles = triton.cdiv(seq_len, block_q)
    _dense_causal_attn_tma_kernel[(num_q_tiles,)](
        q_desc, k_desc, v_desc, o_desc, scale, num_q_tiles, seq_len,
        heads, head_dim, block_q, block_n, warp_specialize, interleave,
        num_warps=num_warps, num_stages=num_stages)
    return out


class Attention(_BaselineAttention):
    """Baseline attention plus a Triton dense causal prefill fast path."""

    # Per-instance cache of everything the guard can decide from ``__init__``
    # arguments alone.  ``None`` means "not computed yet", ``False`` means this
    # layer configuration can never take the fast path.
    _dense_fast_cfg = None

    def _dense_fast_config(self):
        """Static half of the guard: head dim, GQA ratio, window, sinks.

        Computed once per layer so the submit path reads a tuple instead of
        re-deriving the same constants on every call.
        """
        cfg = False
        if (
            self.attention_chunk_size is None
            and not self._triton_only
            and self.head_size in _SUPPORTED_HEAD_DIMS
            and self.num_kv_heads > 0
            and self.num_heads % self.num_kv_heads == 0
        ):
            sinks = self._fa3_sinks
            # The bundled FlashAttention build wants exactly one sink per query
            # head in the model dtype, so anything else has to reach the baseline
            # (and raise there) rather than be quietly served here.  Dtype and
            # device travel with ``Module.to``, so they are re-checked per call.
            if sinks is None or (
                sinks.dim() == 1
                and sinks.shape[0] == self.num_heads
                and sinks.stride(0) == 1
            ):
                heads_per_kv = self.num_heads // self.num_kv_heads
                window = self.sliding_window if self.sliding_window is not None else 0
                # The mid-size specialization is deliberately narrow: it is written
                # for the one shape class that occurs here with more than 256
                # tokens, and it has no sink or window handling at all.
                tma_ok = (
                    sinks is None
                    and window == 0
                    and self.num_kv_heads == 1
                    and self.num_heads == _TMA_HEADS
                    and self.head_size == _TMA_HEAD_DIM
                )
                cfg = (
                    heads_per_kv,
                    sinks,
                    window,
                    _launch_plan(self.num_heads, heads_per_kv, self.head_size),
                    tma_ok,
                )
        self._dense_fast_cfg = cfg
        return cfg

    def _forward_pure(self, q, k, v, k_cache, v_cache, ctx):
        cfg = self._dense_fast_cfg
        if cfg is None:
            cfg = self._dense_fast_config()
        if cfg and not _FAST_DISABLED:
            route = self._dense_fast_route(q, k, v, k_cache, v_cache, ctx, cfg)
            if route == _ROUTE_PACKED:
                if _COUNT:
                    DISPATCH_COUNTS["fast"] += 1
                heads_per_kv, sinks, window, plan = cfg[:4]
                cu_q = ctx.cu_seqlens_q
                return dense_causal_prefill(
                    q,
                    k,
                    v,
                    cu_q,
                    ctx.cu_seqlens_k,
                    ctx.max_seqlen_q,
                    cu_q.shape[0] - 1,
                    self.scale,
                    heads_per_kv,
                    sinks,
                    window,
                    plan,
                )
            if route == _ROUTE_TMA:
                if _COUNT:
                    DISPATCH_COUNTS["tma"] += 1
                return dense_causal_prefill_tma(q, k, v, ctx.max_seqlen_q, self.scale)
        if _COUNT:
            DISPATCH_COUNTS["fallback"] += 1
        return super()._forward_pure(q, k, v, k_cache, v_cache, ctx)

    def _dense_fast_route(self, q, k, v, k_cache, v_cache, ctx,
                          cfg=(1, None, 0, None, False)):
        """Per-call half of the guard: which kernel, if any, may serve this call.

        Returns `_ROUTE_PACKED`, `_ROUTE_TMA`, or `_DECLINE`.  Deliberately
        narrower than either kernel's proven domain -- everything it rejects is
        served by the baseline, so a rejection is a performance non-event rather
        than a correctness one.
        """
        if not ctx.is_prefill or ctx.is_mixed or getattr(ctx, "is_tree_verify", False):
            return _DECLINE
        max_q = ctx.max_seqlen_q
        max_k = ctx.max_seqlen_k
        if max_q > _MAX_SEQLEN_FAST or max_k > _MAX_SEQLEN_FAST:
            return _DECLINE
        # Pick the candidate route from the cheap scalars before touching the
        # tensors, so an oversized call declines without paying for the rest.  A
        # window bounds how many keys each query row actually reads, so it is the
        # span, not the sequence length, that the packed kernel's budget buys.
        window = cfg[2]
        span = min(max_k, window) if window else max_k
        if max_q * span <= _MAX_TOKEN_PAIRS:
            want = _ROUTE_PACKED
        elif cfg[4] and _TMA_MIN_SEQLEN <= max_q <= _TMA_MAX_SEQLEN and max_k == max_q:
            want = _ROUTE_TMA
        else:
            return _DECLINE
        # Only dense, unpaged prefill.  A block table under either the plain or
        # the sliding-window key means the KV lives in a paged cache; a non-empty
        # cache pair means a store the fast path must not step around.
        if ctx.block_tables is not None or ctx.sliding_block_tables is not None:
            return _DECLINE
        if k_cache.numel() or v_cache.numel():
            return _DECLINE
        # Inference only: the kernel has no backward.
        if torch.is_grad_enabled() or q.requires_grad or k.requires_grad or v.requires_grad:
            return _DECLINE
        cu_q = ctx.cu_seqlens_q
        cu_k = ctx.cu_seqlens_k
        if cu_q is None or cu_k is None or cu_q.shape != cu_k.shape:
            return _DECLINE
        if cu_q.dtype != torch.int32 or cu_k.dtype != torch.int32:
            return _DECLINE
        # The kernel indexes these directly, so a strided or higher-rank view
        # would be misread.
        if cu_q.dim() != 1 or cu_k.dim() != 1:
            return _DECLINE
        if cu_q.stride(0) != 1 or cu_k.stride(0) != 1:
            return _DECLINE
        dtype = q.dtype
        if dtype not in _SUPPORTED_DTYPES or k.dtype != dtype or v.dtype != dtype:
            return _DECLINE
        device = q.device
        if device.type != "cuda" or k.device != device or v.device != device:
            return _DECLINE
        if cu_q.device != device or cu_k.device != device:
            return _DECLINE
        sinks = cfg[1]
        if sinks is not None and (sinks.dtype != dtype or sinks.device != device):
            return _DECLINE
        # Row-strided views of a fused QKV buffer are expected; anything with a
        # gap inside a token's heads is not.
        head_dim = q.shape[2]
        if q.stride(2) != 1 or k.stride(2) != 1 or v.stride(2) != 1:
            return _DECLINE
        if q.stride(1) != head_dim or k.stride(1) != head_dim or v.stride(1) != head_dim:
            return _DECLINE
        if k.shape != v.shape or k.stride() != v.stride():
            return _DECLINE
        if want == _ROUTE_PACKED:
            return _ROUTE_PACKED
        # The specialization additionally needs one dense segment whose keys and
        # queries line up, and descriptor-compatible alignment.
        if (
            cu_q.shape[0] == 2
            and q.shape[0] == max_q
            and k.shape[0] == max_q
            and _tma_aligned(q)
            and _tma_aligned(k)
            and _tma_aligned(v)
        ):
            return _ROUTE_TMA
        return _DECLINE
