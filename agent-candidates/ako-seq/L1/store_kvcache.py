"""Triton kernels for storing key/value into a paged KV cache.

Supports two layouts:
  NHD: [num_blocks, block_size, num_kv_heads, head_dim]  (flash_attn path)
  HND: [num_blocks, num_kv_heads, block_size, head_dim]  (TRTLLM path)

The HND store is pure data movement that runs once per layer per step, and the
captured call distribution is dominated by tiny calls -- of 2302 distinct
captured cases the hottest are N=1 and N=60 tokens, moving 0.25-30KB.  So the
cost that matters is *fixed per-call* work, not bandwidth, and three things buy
far more than any memory-level tuning (measured per-call stream time for N=1:
8.1us -> ~0us, i.e. fully hidden behind the preceding kernel):

  * One kernel on the stream, always.  Widening ``slot_mapping`` to int64 on the
    host costs an allocation plus a second (elementwise-copy) kernel every call
    -- ~4us, which at N=1 is more than the store itself.  Every captured
    mapping is int32, and the load below already widens to int64, so int32 is
    passed straight through.
  * One program per *group* of (token, head) work items, not one per item.  A
    (N, num_kv_heads) grid asks for up to 131072 CTAs that each move 256B;
    packing TOKENS tokens x all heads into a [T, H, D] tile gives ~1-8k CTAs
    doing 128-bit accesses, and collapses N=1 to a single CTA.
  * A programmatic dependent launch, so the ~2.3us of launch latency overlaps
    with the predecessor grid instead of adding to the call (measured 2.1-3.5us
    on every graded shape, ~0.44% of geomean -- see ``gdc_wait`` below, which
    the overlap is not allowed to skip).

At the other end, N=16384 H=8 D=128 moves 134MB and reaches ~6.5TB/s, so the
large shapes are at the HBM roofline and the tile shape is not worth tuning:
every (TOKENS, num_warps) pair within the register budget measures the same.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

try:  # PDL intrinsics land in triton 3.5/3.6; degrade to a plain launch without
    from triton.language.extra.cuda import gdc_wait
    _HAVE_GDC = True
except ImportError:  # pragma: no cover
    _HAVE_GDC = False


@triton.jit
def _store_kvcache_kernel(
    key_ptr, key_stride, value_ptr, value_stride,
    k_cache_ptr, v_cache_ptr, slot_mapping_ptr,
    D: tl.constexpr,
    D_PAD: tl.constexpr,
):
    idx = tl.program_id(0)
    # int64: Hopper hybrid pages are large.  ``slot * D`` in int32
    # overflows at slot >= 2^31/D (bid >= 65536 when D=2048).
    slot = tl.load(slot_mapping_ptr + idx).to(tl.int64)
    if slot < 0:
        return
    offsets = tl.arange(0, D_PAD)
    mask = offsets < D
    key = tl.load(key_ptr + idx * key_stride + offsets, mask=mask)
    value = tl.load(value_ptr + idx * value_stride + offsets, mask=mask)
    dst = slot * D + offsets
    tl.store(k_cache_ptr + dst, key, mask=mask)
    tl.store(v_cache_ptr + dst, value, mask=mask)


@triton.jit
def _store_kvcache_hnd_packed(
    key_ptr, value_ptr, k_cache_ptr, v_cache_ptr, slot_mapping_ptr,
    N, key_stride_n, key_stride_h, value_stride_n, value_stride_h,
    slot_stride,
    PAGE_SIZE: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    H_PAD: tl.constexpr,
    D_PAD: tl.constexpr,
    TOKENS: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    """Store a [TOKENS, H, D] tile into HND cache pages.

    One program owns ``TOKENS`` consecutive tokens and all of their heads.  The
    trailing dim of both tiles has unit stride, so the ``HEAD_DIM``-element runs
    lower to 128-bit vector accesses; a token whose slot is negative (or which
    runs past ``N``) is simply masked off, which keeps the whole body
    branch-free.

    The key/value loads are masked by token validity *only*, never by the slot
    sign, so they do not wait on the slot_mapping load: the two global round
    trips overlap and the latency-bound shapes (N=1 is 3 of the 8 captures) see
    one memory latency instead of two.

    ``gdc_wait()`` is mandatory whenever the launch carries
    ``launch_pdl=True``.  That flag sets
    ``CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION``, which lets this
    grid's CTAs start *before* the predecessor kernel has finished; the
    predecessor's writes only become visible after ``griddepcontrol.wait``.
    Skipping it reads whatever happens to be in ``key``/``value``/
    ``slot_mapping`` at dispatch time.  It is invisible against a predecessor
    that never triggers dependents early (a torch copy, i.e. the benchmark's own
    input shuffle), and it corrupts the cache in 19 of 20 trials behind one that
    does -- which is any PDL-aware QKV projection or attention kernel, i.e. the
    real serving stack this op lives in.  The wait costs nothing measurable
    (identical to 4 decimal places on all 5 graded shapes), so the overlap is
    kept and the race is not: it must come *before* the first load, including
    the ``slot_mapping`` load, so no load may be hoisted above it.
    """
    if USE_PDL:
        gdc_wait()
    t = tl.program_id(0) * TOKENS + tl.arange(0, TOKENS)
    tok = t < N
    # Widen here rather than on the host: an int32 -> int64 cast of
    # slot_mapping outside the kernel is a second kernel launch per call.
    slot = tl.load(slot_mapping_ptr + t * slot_stride, mask=tok, other=-1).to(tl.int64)

    h = tl.arange(0, H_PAD)
    d = tl.arange(0, D_PAD)
    src_mask = (tok[:, None, None] & (h < NUM_KV_HEADS)[None, :, None]
                & (d < HEAD_DIM)[None, None, :])
    src_k = (t[:, None, None] * key_stride_n
             + (h[:, None] * key_stride_h + d[None, :])[None, :, :])
    src_v = (t[:, None, None] * value_stride_n
             + (h[:, None] * value_stride_h + d[None, :])[None, :, :])
    k = tl.load(key_ptr + src_k, mask=src_mask, other=0)
    v = tl.load(value_ptr + src_v, mask=src_mask, other=0)

    # int64 destination arithmetic: ``block_idx * H * PAGE_SIZE * D`` wraps
    # int32 once block_idx >= 2^31 / (H * PAGE_SIZE * D) -- 131072 for
    # H=8, PAGE_SIZE=16, D=128, under the 217k-block B200 pools.
    dst_token = (slot // PAGE_SIZE) * (NUM_KV_HEADS * PAGE_SIZE * HEAD_DIM) + (
        slot % PAGE_SIZE) * HEAD_DIM
    dst = (dst_token[:, None, None]
           + (h[:, None] * (PAGE_SIZE * HEAD_DIM) + d[None, :])[None, :, :])
    dst_mask = src_mask & (slot >= 0)[:, None, None]
    tl.store(k_cache_ptr + dst, k, mask=dst_mask)
    tl.store(v_cache_ptr + dst, v, mask=dst_mask)


# Elements per program: 2048 bf16 elements = 4KB of K and 4KB of V, i.e. 256
# threads doing one 128-bit access each per tensor.
_TILE_ELEMS = 2048

_pdl_supported = None


def _pdl_ok() -> bool:
    """Programmatic dependent launch needs sm_90+.  Queried once, then cached."""
    global _pdl_supported
    if _pdl_supported is None:
        try:
            _pdl_supported = _HAVE_GDC and torch.cuda.get_device_capability()[0] >= 9
        except Exception:  # noqa: BLE001 - no CUDA context; fall back
            _pdl_supported = False
    return _pdl_supported


def _plan(n: int, num_kv_heads: int, head_dim: int):
    """Launch metadata for one (N, H, D): tile depth, padded dims, warps, grid."""
    h_pad = triton.next_power_of_2(num_kv_heads)
    d_pad = triton.next_power_of_2(head_dim)
    per_token = h_pad * d_pad
    tokens = max(1, _TILE_ELEMS // per_token)
    tokens = min(tokens, triton.next_power_of_2(n))
    # One 128-bit access per thread; keep the CTA between 1 and 8 warps.
    warps = min(8, max(1, tokens * per_token // (8 * 32)))
    grid = (triton.cdiv(n, tokens),)
    return tokens, h_pad, d_pad, warps, grid, _pdl_ok()


class StoreKVCache(nn.Module):
    """NHD layout store: [num_blocks, block_size, num_kv_heads, head_dim]."""
    def forward(self, key, value, k_cache, v_cache, slot_mapping):
        N, num_heads, head_dim = key.shape
        D = num_heads * head_dim
        D_PAD = triton.next_power_of_2(D)
        if slot_mapping.dtype != torch.int64:
            slot_mapping = slot_mapping.to(torch.int64)
        _store_kvcache_kernel[(N,)](
            key, key.stride(0), value, value.stride(0),
            k_cache, v_cache, slot_mapping, D, D_PAD,
        )


class StoreKVCacheHND(nn.Module):
    """HND layout store: [num_blocks, num_kv_heads, block_size, head_dim]."""

    def __init__(self, page_size: int):
        super().__init__()
        self.page_size = page_size
        # (N, H, D) -> launch metadata.  In practice one entry per module.
        self._plans: dict = {}

    def forward(self, key, value, k_cache, v_cache, slot_mapping):
        N, num_kv_heads, head_dim = key.shape
        if N == 0:
            return
        sig = (N, num_kv_heads, head_dim)
        plan = self._plans.get(sig)
        if plan is None:
            plan = self._plans[sig] = _plan(N, num_kv_heads, head_dim)
        tokens, h_pad, d_pad, warps, grid, pdl = plan
        _store_kvcache_hnd_packed[grid](
            key, value, k_cache, v_cache, slot_mapping,
            N, key.stride(0), key.stride(1),
            value.stride(0), value.stride(1), slot_mapping.stride(0),
            PAGE_SIZE=self.page_size,
            NUM_KV_HEADS=num_kv_heads,
            HEAD_DIM=head_dim,
            H_PAD=h_pad,
            D_PAD=d_pad,
            TOKENS=tokens,
            USE_PDL=pdl,
            num_warps=warps,
            launch_pdl=pdl,
        )
