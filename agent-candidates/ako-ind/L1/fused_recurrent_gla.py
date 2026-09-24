"""Fused recurrent GLA — hand-written single-pass Triton kernels.

The captured workload is decode-dominated: ``T == 1`` with a persistent
``float32`` recurrent state of shape ``[B, H, K, V]`` that is read in and
written back on every call.  With ``K=256, V=512, H=5`` that state is 2.5 MB per
batch element while ``q/k/v/gk`` together are ~10 KB per element, so the op is
purely a state-streaming (DRAM bandwidth) problem.

Two kernels, both written here:

``_fwd_decode`` (``T == 1``)
    One program per ``(batch*head, v-tile)``.  It walks ``K`` in tiles and for
    each tile does *one* fused pass: load the state tile, scale it by
    ``exp(gk)``, add the rank-1 update ``k (x) v``, store the new state tile and
    accumulate its contribution to ``o``.  The state is read once and written
    once with no intermediate materialization, and ``o`` is complete inside the
    program -- no cross-program reduction and no second launch, so the whole op
    is a single kernel.

``_fwd_seq`` (``T > 1`` / varlen)
    Register-resident ``[BK, BV]`` state tile with a plain sequential loop over
    time -- the same recurrence in the same order of operations as the reference,
    so numerics match.  No chunked or parallel-scan reformulation: the captured
    ``gk`` is raw ``randn``, so a cumulative-decay factorisation would divide by
    ``exp`` of a large partial sum and blow up.  Here ``BK`` is small (8 rows) so
    that the per-step ``o`` reduction stays inside each thread's registers, which
    means ``K`` is split across programs and ``_reduce_o`` sums the partials.

``_reduce_o``
    Sums the ``NK`` partial ``o`` buffers and casts to the output dtype in one
    kernel, so the ``T > 1`` path is two launches rather than three.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# T == 1 (decode): stream the state once, one pass, one launch.
# ---------------------------------------------------------------------------
@triton.jit
def _fwd_decode(
    q, k, v, gk, o, h0, ht,
    scale,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NV: tl.constexpr,
    NK: tl.constexpr,
    USE_H0: tl.constexpr,
    USE_GK: tl.constexpr,
    STORE_HT: tl.constexpr,
):
    """Unmasked: the host only dispatches here when BK | K and BV | V."""
    pid = tl.program_id(0)
    i_nh = pid // NV
    i_v = pid % NV
    i_n = i_nh // H
    i_h = i_nh % H

    offs_v = i_v * BV + tl.arange(0, BV)
    nh64 = i_nh.to(tl.int64)
    xoff = (i_n.to(tl.int64) * H + i_h) * K
    p_q = q + xoff
    p_k = k + xoff
    if USE_H0:
        p_h = h0 + nh64 * (K * V) + offs_v[None, :]
    if STORE_HT:
        p_t = ht + nh64 * (K * V) + offs_v[None, :]

    b_v = tl.load(v + (i_n.to(tl.int64) * H + i_h) * V + offs_v).to(tl.float32)
    # Deferred o-reduction: accumulate a [BK, BV] partial and reduce once at the
    # end, so the K loop stays a barrier-free load/fma/store stream.
    t_acc = tl.zeros([BK, BV], dtype=tl.float32)

    for i_k in range(NK):
        offs_k = i_k * BK + tl.arange(0, BK)
        off_h = offs_k[:, None] * V

        b_q = tl.load(p_q + offs_k).to(tl.float32) * scale
        b_k = tl.load(p_k + offs_k).to(tl.float32)

        if USE_H0:
            b_h = tl.load(p_h + off_h).to(tl.float32)
            if USE_GK:
                b_g = tl.load(gk + xoff + offs_k).to(tl.float32)
                b_h = b_h * tl.exp(b_g)[:, None]
        else:
            b_h = tl.zeros([BK, BV], dtype=tl.float32)
        b_h += b_k[:, None] * b_v[None, :]

        if STORE_HT:
            tl.store(p_t + off_h, b_h)
        t_acc += b_h * b_q[:, None]

    acc = tl.sum(t_acc, 0)
    tl.store(o + (i_n.to(tl.int64) * H + i_h) * V + offs_v, acc.to(o.dtype.element_ty))


# ---------------------------------------------------------------------------
# General T (register-resident state, sequential over time).
# ---------------------------------------------------------------------------
@triton.jit
def _fwd_seq(
    q, k, v, gk, o, h0, ht, cu_seqlens,
    scale, T, ALL,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NV: tl.constexpr,
    NK: tl.constexpr,
    USE_H0: tl.constexpr,
    USE_GK: tl.constexpr,
    STORE_HT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    pid = tl.program_id(0)
    i_nh = pid // (NV * NK)
    rem = pid % (NV * NK)
    i_k = rem // NV
    i_v = rem % NV
    i_n = i_nh // H
    i_h = i_nh % H

    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        t_len = eos - bos
    else:
        bos = i_n.to(tl.int64) * T
        t_len = T

    offs_k = i_k * BK + tl.arange(0, BK)
    offs_v = i_v * BV + tl.arange(0, BV)
    m_k = offs_k < K
    m_v = offs_v < V
    m_h = m_k[:, None] & m_v[None, :]
    off_h = offs_k[:, None] * V + offs_v[None, :]

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_H0:
        b_h += tl.load(h0 + i_nh.to(tl.int64) * (K * V) + off_h,
                       mask=m_h, other=0.0).to(tl.float32)

    xoff = (bos * H + i_h) * K
    p_q = q + xoff
    p_k = k + xoff
    p_v = v + (bos * H + i_h) * V
    p_o = o + (i_k.to(tl.int64) * ALL + bos) * (H * V) + i_h * V
    if USE_GK:
        p_g = gk + xoff

    # The recurrence is serial in t, and q/k/v/gk for step t+1 do not depend on
    # the state, so the loads are hoisted one step ahead of the compute: without
    # this the ~300-cycle L2 latency sits on the critical path of every step
    # (Triton's own pipeliner does not touch this loop -- num_stages is inert).
    # On the last step the advance is 0, so the redundant prefetch stays in
    # bounds and its result is simply unused.
    b_q = tl.load(p_q + offs_k, mask=m_k, other=0.0).to(tl.float32) * scale
    b_k = tl.load(p_k + offs_k, mask=m_k, other=0.0).to(tl.float32)
    b_v = tl.load(p_v + offs_v, mask=m_v, other=0.0).to(tl.float32)
    if USE_GK:
        b_g = tl.load(p_g + offs_k, mask=m_k, other=0.0).to(tl.float32)

    for t in range(t_len):
        adv = tl.where(t + 1 < t_len, 1, 0)
        p_q += adv * (H * K)
        p_k += adv * (H * K)
        p_v += adv * (H * V)
        n_q = tl.load(p_q + offs_k, mask=m_k, other=0.0).to(tl.float32) * scale
        n_k = tl.load(p_k + offs_k, mask=m_k, other=0.0).to(tl.float32)
        n_v = tl.load(p_v + offs_v, mask=m_v, other=0.0).to(tl.float32)
        if USE_GK:
            p_g += adv * (H * K)
            n_g = tl.load(p_g + offs_k, mask=m_k, other=0.0).to(tl.float32)
            b_h = b_h * tl.exp(b_g)[:, None]
        b_h += b_k[:, None] * b_v[None, :]
        b_o = tl.sum(b_h * b_q[:, None], 0)
        tl.store(p_o + offs_v, b_o.to(o.dtype.element_ty), mask=m_v)
        p_o += H * V
        b_q = n_q
        b_k = n_k
        b_v = n_v
        if USE_GK:
            b_g = n_g

    if STORE_HT:
        tl.store(ht + i_nh.to(tl.int64) * (K * V) + off_h, b_h, mask=m_h)


# ---------------------------------------------------------------------------
# Cross-tile o reduction (only when the general-T kernel has to split K).
# ---------------------------------------------------------------------------
@triton.jit
def _reduce_o(src, dst, n_elem, NK: tl.constexpr, BLK: tl.constexpr):
    offs = tl.program_id(0) * BLK + tl.arange(0, BLK)
    m = offs < n_elem
    acc = tl.zeros([BLK], dtype=tl.float32)
    for i in range(NK):
        acc += tl.load(src + i.to(tl.int64) * n_elem + offs, mask=m, other=0.0)
    tl.store(dst + offs, acc.to(dst.dtype.element_ty), mask=m)


# ---------------------------------------------------------------------------
# Host-side configuration and launch.
# ---------------------------------------------------------------------------
try:
    _raw_stream = torch._C._cuda_getCurrentRawStream
except AttributeError:  # pragma: no cover
    _raw_stream = None


_DECODE_CFG: dict = {}
_SEQ_CFG: dict = {}

# Tuned on B200 (see ITERATIONS.md).  Decode: a [64, 64] fp32 state tile with 4
# warps streams the state at 6.5 TB/s, above a bare 671 MB device-to-device copy.
# General-T: an [8, 128] tile on a *single* warp -- one warp means the per-step
# o-reduction needs no barriers at all, and 8 rows x 128 lanes keeps that
# reduction inside each thread's registers.
_DEC_BK, _DEC_BV, _DEC_NW, _DEC_NS = 64, 64, 4, 2
_SEQ_BK, _SEQ_BV, _SEQ_NW, _SEQ_NS = 8, 128, 1, 2
_RED = (1024, 512)         # o-reduce: (starting block, minimum grid)


def _pow2_div(n: int, want: int) -> int:
    """Largest power-of-two divisor of ``n`` that is <= ``want``."""
    d = 1
    while d * 2 <= want and n % (d * 2) == 0:
        d *= 2
    return d


def _decode_config(K: int, V: int):
    cfg = _DECODE_CFG.get((K, V))
    if cfg is None:
        bk, bv, nw, ns = _DEC_BK, _DEC_BV, _DEC_NW, _DEC_NS
        bk, bv = _pow2_div(K, bk), _pow2_div(V, bv)
        # The decode kernel is unmasked, so it needs BK | K and BV | V.  If the
        # best power-of-two divisors are degenerate (odd K or V), the masked
        # general-T kernel handles T == 1 correctly too, so route there instead.
        cfg = (bk, bv, V // bv, K // bk, nw, ns,
               K % bk == 0 and V % bv == 0 and bk >= 8 and bv >= 8)
        _DECODE_CFG[(K, V)] = cfg
    return cfg


def _seq_config(K: int, V: int):
    cfg = _SEQ_CFG.get((K, V))
    if cfg is None:
        bk, bv, nw, ns = _SEQ_BK, _SEQ_BV, _SEQ_NW, _SEQ_NS
        bk, bv = min(bk, K), min(bv, V)
        cfg = (bk, bv, (V + bv - 1) // bv, (K + bk - 1) // bk, nw, ns)
        _SEQ_CFG[(K, V)] = cfg
    return cfg


# ---------------------------------------------------------------------------
# Fast launch.
#
# Triton's Python dispatcher costs ~8 us per call (argument binding, runtime
# specialization, cache-key hashing).  Every shape here except B=256 is only
# 6-60 us of GPU work, so that dispatch is the dominant cost.  After the first
# launch of a given (kernel, specialization) combination we keep the
# CompiledKernel and call its C launcher directly, exactly as
# ``JITFunction.run`` does.
#
# The cache key has to pin everything Triton specializes on, or we would reuse a
# kernel compiled for different assumptions:
#   * every ``tl.constexpr`` value;
#   * runtime ints (Triton turns ``1`` into a constexpr and marks multiples of
#     16 as divisible) -- so those are keyed on their exact value;
#   * every pointer's dtype (it becomes the pointee type in the signature);
#   * the device (a CUfunction handle belongs to one device's module);
#   * pointer alignment -- Triton marks 16B-aligned pointers as divisible, so the
#     fast path is only taken when *all* pointers are 16B aligned (they always
#     are: both the torch caching allocator and the benchmark's shifting pool
#     hand out 256B-aligned storage).  Anything else falls back to the dispatcher.
# ---------------------------------------------------------------------------
_LAUNCH: dict = {}


def _launch(jit_fn, key, grid, args, dev, nw, ns):
    ent = _LAUNCH.get(key)
    if ent is not None:
        try:
            run, fn, meta = ent
            run(grid, 1, 1, _raw_stream(dev), fn, meta, None, None, None, *args)
            return
        except Exception:  # pragma: no cover - drop the entry and use the dispatcher
            _LAUNCH.pop(key, None)   # all three kernels are idempotent, so a
            # partially-issued retry cannot corrupt anything
    compiled = jit_fn[(grid,)](*args, num_warps=nw, num_stages=ns)
    if key is not None and _raw_stream is not None and compiled is not None:
        try:
            _LAUNCH[key] = (compiled.run, compiled.function, compiled.packed_metadata)
        except Exception:  # pragma: no cover - keep the slow path working
            pass


class FusedRecurrentGLA(nn.Module):
    """Fused recurrent GLA — single-pass state-streaming Triton kernels."""

    def forward(
        self,
        q: torch.Tensor,  # [B, T, H, K]
        k: torch.Tensor,  # [B, T, H, K]
        v: torch.Tensor,  # [B, T, H, V]
        gk: torch.Tensor | None = None,  # [B, T, H, K]
        scale: float | None = None,
        initial_state: torch.Tensor | None = None,  # [B, H, K, V]
        output_final_state: bool = False,
        cu_seqlens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T, H, K = q.shape
        V = v.shape[-1]
        if scale is None:
            scale = K ** -0.5
        use_h0 = initial_state is not None
        use_gk = gk is not None

        if not q.is_contiguous():
            q = q.contiguous()
        if not k.is_contiguous():
            k = k.contiguous()
        if not v.is_contiguous():
            v = v.contiguous()
        if use_gk and not gk.is_contiguous():
            gk = gk.contiguous()
        if use_h0 and not initial_state.is_contiguous():
            initial_state = initial_state.contiguous()

        N = B if cu_seqlens is None else cu_seqlens.numel() - 1
        ht = q.new_empty(N, H, K, V, dtype=torch.float32) if output_final_state else None
        dev = q.get_device()

        if T == 1 and cu_seqlens is None:
            bk, bv, nv, nk, nw, ns, ok = _decode_config(K, V)
            if ok:
                o = q.new_empty(B, 1, H, V)
                algn = (q.data_ptr() | k.data_ptr() | v.data_ptr() | o.data_ptr()
                        | (gk.data_ptr() if use_gk else 0)
                        | (initial_state.data_ptr() if use_h0 else 0)
                        | (ht.data_ptr() if output_final_state else 0)) & 15
                _launch(_fwd_decode,
                        ((dev, H, K, V, bk, bv, nw, ns, q.dtype, k.dtype, v.dtype,
                          o.dtype, None if gk is None else gk.dtype,
                          None if initial_state is None else initial_state.dtype,
                          None if ht is None else ht.dtype)
                         if algn == 0 else None),
                        N * H * nv,
                        (q, k, v, gk, o, initial_state, ht, scale,
                         H, K, V, bk, bv, nv, nk, use_h0, use_gk, output_final_state),
                        dev, nw, ns)
                return o, ht

        bk, bv, nv, nk, nw, ns = _seq_config(K, V)
        varlen = cu_seqlens is not None
        o = (q.new_empty(B, T, H, V) if nk == 1
             else q.new_empty(nk, B, T, H, V, dtype=torch.float32))
        algn = (q.data_ptr() | k.data_ptr() | v.data_ptr() | o.data_ptr()
                | (gk.data_ptr() if use_gk else 0)
                | (initial_state.data_ptr() if use_h0 else 0)
                | (ht.data_ptr() if output_final_state else 0)
                | (cu_seqlens.data_ptr() if varlen else 0)) & 15
        _launch(_fwd_seq,
                ((dev, H, K, V, bk, bv, nv, nk, T, B * T, nw, ns, varlen,
                  q.dtype, k.dtype, v.dtype, o.dtype,
                  None if gk is None else gk.dtype,
                  None if initial_state is None else initial_state.dtype,
                  None if ht is None else ht.dtype,
                  None if cu_seqlens is None else cu_seqlens.dtype)
                 if algn == 0 else None),
                N * H * nv * nk,
                (q, k, v, gk, o, initial_state, ht, cu_seqlens, scale, T, B * T,
                 H, K, V, bk, bv, nv, nk, use_h0, use_gk, output_final_state, varlen),
                dev, nw, ns)
        if nk > 1:
            po, o = o, q.new_empty(B, T, H, V)
            n_elem = B * T * H * V
            # o is small (a few hundred KB), so the reduction is launch- and
            # occupancy-bound, not bandwidth-bound: size the block so it fills
            # the machine rather than using a fixed 1024.
            blk, min_grid = _RED
            while blk > 64 and (n_elem + blk - 1) // blk < min_grid:
                blk //= 2
            nwr = max(1, min(8, blk // 128))
            _launch(_reduce_o,
                    ((dev, "red", nk, n_elem, blk, nwr, po.dtype, o.dtype)
                     if (po.data_ptr() | o.data_ptr()) & 15 == 0 else None),
                    (n_elem + blk - 1) // blk,
                    (po, o, n_elem, nk, blk), dev, nwr, 2)
        return o, ht
