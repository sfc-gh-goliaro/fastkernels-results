"""vLLM-aligned Attention layer with paged KV cache.

Mirrors vLLM's ``Attention`` class (from
``vllm/model_executor/layers/attention/attention.py``):

    forward(query, key, value) -> torch.Tensor

Inputs and outputs are **flat** ``[N, num_heads * head_dim]`` tensors.
KV cache metadata is obtained from the global ``Context`` (via
``get_context()``), matching vLLM's ``get_forward_context()`` pattern.

Backend selection (flash_attn vs TRTLLM-gen) is handled at init time
via ``AttnBackendConfig``.  The engine discovers this module for KV cache
assignment through duck-typing (``hasattr(module, "k_cache")``).

TODO(tech-debt): CUDA graph capture is incompatible with chunked local
attention because the metadata remapping (cu_seqlens, block_tables) varies
per batch.  vLLM disables CUDA graphs when chunked local attention is
active.  If/when we add CUDA graph support, we need to handle this case.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ....infra.context import get_context, get_attn_backend_config
from ..L1.store_kvcache import StoreKVCache, StoreKVCacheHND

try:  # Triton is what the fused prefill path below is written in.
    import triton
    import triton.language as tl
    _TRITON_OK = True
except Exception:  # pragma: no cover - no Triton -> library path only
    _TRITON_OK = False

try:  # FlashAttention-4's CuTe entry point, for reaching its compiled kernel.
    from vllm.vllm_flash_attn.cute.interface import _flash_attn_fwd as _CUTE_FWD
except Exception:  # pragma: no cover - no FA4 -> wrapper path only
    _CUTE_FWD = None

_TRITON_MIN_LAUNCH_GRID_SIZE_2D = 128
_TRITON_NUM_PAR_SOFTMAX_SEGMENTS = 16

# Largest head size the trtllm-gen paged kernels advertise.  Above this,
# SM100 layers drop to TRITON_ATTN (FA4 TMEM-rejects 512).  Hopper uses FA4
# for those heads instead; see ``_triton_only`` below.
_TRTLLM_MAX_HEAD_SIZE = 256

from ..L1.triton_unified_attention import (
    unified_attention as _triton_unified_attention,
)
from ....infra.kv_quant_mode import KVQuantMode as _VllmKVQuantMode
import inspect as _inspect

_TRITON_UNIFIED_ACCEPTS_KV_QUANT = (
    "kv_quant_mode" in _inspect.signature(_triton_unified_attention).parameters
)


# ---------------------------------------------------------------------------
# Fused single-sequence causal prefill.  Every call this layer actually sees is
# one dense causal sequence over fresh q/k/v with no paged cache in play, and
# for short sequences a purpose-built kernel beats the library dispatch by
# 2-4x on device time.  ``_fused_limit`` is where that stops being true and
# ``_fused_prefill`` is the guard that decides.
# ---------------------------------------------------------------------------
_LOG2E = 1.4426950408889634
# Masked-out score sentinel.  Finite (not -inf) so a tile that is fully masked
# for some row keeps ``exp2`` and the running max NaN-free; the first real score
# in a later tile rescales that row by ``exp2(-1e30 - m)`` == 0 exactly, which
# discards the sentinel mass.  Every row ``i < N`` sees key ``i``, so no row
# ends the loop with only sentinels.
_NEG = -1.0e30

if _TRITON_OK:
    _TL_LOG2E = tl.constexpr(_LOG2E)
    _TL_NEG = tl.constexpr(_NEG)

    @triton.jit(do_not_specialize=["N"])
    def _fused_attn_kernel(Q, K, V, Out, Sink, qk_scale, N,
                           SQ0: tl.constexpr, SQ1: tl.constexpr,
                           SK0: tl.constexpr, SK1: tl.constexpr,
                           SV0: tl.constexpr, SV1: tl.constexpr,
                           SO: tl.constexpr, G: tl.constexpr, D: tl.constexpr,
                           BM: tl.constexpr, BN: tl.constexpr,
                           HAS_SINK: tl.constexpr, WINDOW: tl.constexpr):
        """One CTA per (query-row tile, query head); online-softmax over KV.

        ``Q`` is ``[N, H * D]``, ``K``/``V`` are ``[N, Hkv * D]``, and all three
        are column slices of a fused QKV buffer in the captured workloads -- so
        the row stride is *not* ``H * D`` and both strides are carried.  They are
        ``constexpr`` rather than runtime arguments because Triton's divisibility
        specialization is what lets it vectorize the strided row loads -- handing
        the same values in as plain ints measured 2.7x slower.
        GQA/MQA is a pure index map, query head ``h`` reading kv head ``h // G``.
        ``Out`` is freshly allocated and contiguous.
        """
        start_m = tl.program_id(0)
        h = tl.program_id(1)
        offs_m = start_m * BM + tl.arange(0, BM)
        offs_d = tl.arange(0, D)
        m_mask = offs_m < N
        q = tl.load(Q + offs_m[:, None] * SQ0
                    + ((h * D + offs_d) * SQ1)[None, :],
                    mask=m_mask[:, None], other=0.0)

        acc = tl.zeros([BM, D], dtype=tl.float32)
        m_i = tl.full([BM], _TL_NEG, tl.float32)
        l_i = tl.zeros([BM], tl.float32)

        hi = tl.minimum((start_m + 1) * BM, N)
        if WINDOW > 0:
            lo = ((tl.maximum(start_m * BM - WINDOW + 1, 0)) // BN) * BN
        else:
            lo = 0
        kvd = (h // G) * D + offs_d
        kb = K + (kvd * SK1)[None, :]
        vb = V + (kvd * SV1)[None, :]
        for start_n in range(lo, hi, BN):
            offs_n = start_n + tl.arange(0, BN)
            n_mask = offs_n < N
            k = tl.load(kb + offs_n[:, None] * SK0, mask=n_mask[:, None],
                        other=0.0)
            qk = tl.dot(q, tl.trans(k)) * qk_scale
            keep = (offs_n[None, :] <= offs_m[:, None]) & n_mask[None, :]
            if WINDOW > 0:
                keep = keep & (offs_n[None, :] > offs_m[:, None] - WINDOW)
            qk = tl.where(keep, qk, _TL_NEG)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            p = tl.exp2(qk - m_ij[:, None])
            alpha = tl.exp2(m_i - m_ij)
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None]
            vt = tl.load(vb + offs_n[:, None] * SV0, mask=n_mask[:, None],
                         other=0.0)
            acc = tl.dot(p.to(vt.dtype), vt, acc)
            m_i = m_ij

        if HAS_SINK:
            # gpt-oss attention sink: one extra logit per query head that only
            # ever lands in the softmax denominator.
            l_i = l_i + tl.exp2(tl.load(Sink + h).to(tl.float32) * _TL_LOG2E
                                - m_i)
        acc = acc / l_i[:, None]
        tl.store(Out + offs_m[:, None] * SO + (h * D + offs_d)[None, :],
                 acc.to(Out.dtype.element_ty), mask=m_mask[:, None])

    @triton.jit
    def _fused_attn_n1_kernel(Q, K, V, Out, Sink, scale,
                              SQ1: tl.constexpr, SK1: tl.constexpr,
                              SV1: tl.constexpr, G: tl.constexpr,
                              D: tl.constexpr, HAS_SINK: tl.constexpr):
        """N == 1: causal attention over a single key is analytically ``v``.

        With a sink the softmax is over two logits, so the value is scaled by
        ``sigmoid(q.k * scale - sink)``.  One CTA per query head, no MMA, and
        no row stride at all -- there is only one row.
        """
        h = tl.program_id(0)
        d = tl.arange(0, D)
        kvd = (h // G) * D + d
        o = tl.load(V + kvd * SV1)
        if HAS_SINK:
            q = tl.load(Q + (h * D + d) * SQ1).to(tl.float32)
            k = tl.load(K + kvd * SK1).to(tl.float32)
            s = tl.sum(q * k) * scale
            w = 1.0 / (1.0 + tl.exp2((tl.load(Sink + h).to(tl.float32) - s)
                                     * _TL_LOG2E))
            o = (o.to(tl.float32) * w).to(o.dtype)
        tl.store(Out + h * D + d, o)


_FUSED_DTYPES = frozenset((torch.bfloat16, torch.float16))
_FUSED_HEAD_SIZES = frozenset((16, 32, 64, 128, 256))

# Where FlashAttention's own kernel starts winning, and the tile shape to use
# below that.  Both come from a profiled device-time sweep (fused kernel vs the
# FA4 kernel in the same call, `dev/sweep6.py`):
#
#   H=16 Hkv=1 D=128   N=  60  2.7 vs  7.2 us   N= 512  11.3 vs 11.6 us
#                      N= 128  3.8 vs  7.3 us   N= 656  16.5 vs 14.5 us
#                      N= 256  6.5 vs  8.9 us   N=1000  22.4 vs 17.3 us
#   H=32 Hkv=4 D=64    N=  60  2.8 vs  6.9 us   N= 256   5.9 vs  9.1 us
#   (sinks, window)    N= 128  3.9 vs  7.1 us   N= 656   9.1 vs  9.5 us
#
# Everything in this benchmark is *device* bound -- the harness puts a 253 MB
# L2 flush (measured 70 us) on the stream ahead of every timed call, so the
# launch queue is always deep and the measured window is the device time of the
# kernels between the two events.  Which kernel is faster is therefore the whole
# routing question, and past a few hundred tokens FA4's tcgen05 pipeline is
# simply a better GEMM than anything `tl.dot` produces here (at 16 K tokens it
# is 1560 us against 4034 us, so the long cases stay on the library outright).
def _fused_limit(head_size: int) -> int:
    return 512 if head_size >= 128 else 768


def _fused_tile_shape(head_size: int, N: int):
    """(BM, BN, num_warps, num_stages) for one query head of this geometry.

    The fp32 accumulator is ``BM x head_size``, so BM trades register/tmem
    pressure and causal-mask waste against the number of KV steps -- and at
    these token counts the grid is small enough that a narrow BM, i.e. more
    CTAs, is what keeps the SMs busy.
    """
    if head_size >= 128:
        if N <= 128:
            return 16, 64, 4, 2
        if N <= 256:
            return 32, 64, 4, 3
        return 64, 128, 8, 3
    if N <= 64:
        return 16, 64, 4, 2
    if N <= 256:
        return 32, 64, 4, 3
    return 64, 64, 4, 2


# Cap on distinct (tile, strides, alignment) layouts one layer will specialize
# for.  Each one is a Triton compile plus a cached launcher; a workload that
# somehow presents more than this falls back to the library path rather than
# recompiling without bound.
_FUSED_MAX_VARIANTS = 8


def _n1_warps(head_size: int) -> int:
    return max(1, min(4, head_size // 32))


# Cache-key stand-in for the N == 1 kernel, which has no tile at all.
_N1_TILE = (0, 0, 0, 0)


try:
    from torch._C import _cuda_getCurrentRawStream as _cuda_raw_stream
except ImportError:  # pragma: no cover - older torch
    def _cuda_raw_stream(idx):
        return torch.cuda.current_stream(idx).cuda_stream

_cuda_current_device = torch.cuda.current_device


def _raw_stream():
    """Current CUDA stream as a raw handle, without building a Stream object.

    Same two calls Triton's own launcher makes, so a capture-mode stream is
    picked up exactly as it would be on the normal launch path.
    """
    return _cuda_raw_stream(_cuda_current_device())


class _FastLaunch:
    """Cached direct launch of one already-compiled Triton kernel.

    ``JITFunction.run`` re-derives the specialization key, the backend options
    and the launch metadata on every call -- ~9 us, against a kernel whose
    device time is 1-3 us.  Compiling once through the normal path and then
    calling the ``CompiledKernel``'s launcher directly leaves only the argument
    marshalling, ~3 us.  Under this benchmark's timing loop that is not the
    binding constraint (see ``_fused_limit``), but it is free, and it keeps the
    layer's host cost an order of magnitude under the library path's for any
    caller that *is* launch-bound.

    A launcher-signature mismatch (a different Triton) raises rather than
    computing a wrong answer, and the caller treats that as "no fast path".

    Correctness rests on the specialization key being constant across calls:
    ``N`` is ``do_not_specialize`` in the kernel above, every other non-pointer
    argument is a constexpr baked into ``tail``, and pointer *alignment* -- the
    one remaining specialization axis -- is part of the caller's cache key.
    Triton's launch hooks are bypassed, so a Triton-level profiler sees nothing
    from the cached path; the first (compiling) call still goes through it.
    """

    __slots__ = ("launch", "func", "meta", "tail")

    def __init__(self, compiled, tail):
        self.launch = compiled.run        # triggers _init_handles()
        self.func = compiled.function
        self.meta = compiled.packed_metadata
        # The launcher takes *every* declared parameter, constexprs included;
        # those are fixed per layer, so they are bound here.
        self.tail = tail

    def __call__(self, g0, g1, args):
        self.launch(g0, g1, 1, _raw_stream(), self.func, self.meta, None,
                    None, None, *args, *self.tail)


# ---------------------------------------------------------------------------
# Direct launch of FlashAttention-4's already-compiled kernel.
#
# Above the fused crossover this layer calls FlashAttention, and on Blackwell
# that means FA4 -- a CuTe-DSL kernel whose *launcher* is pure Python.  Reaching
# the compiled kernel goes through five frames that all re-derive per-call what
# is constant for the layer:
#
#   nn.Module.__call__ -> TRTLLMPrefill.forward -> torch._dynamo.disable
#     -> vllm flash_attn_varlen_func -> cute _flash_attn_fwd
#       -> JITCache[compile_key](*call_args)
#
# Measured host cost per call at each level on the 656-token case (`dev/
# hostlayers.py`), against an FA4 kernel whose device time is 14.4 us:
#
#   forward_impl (baseline)   45.9 us      vllm flash_attn_varlen_func  33.3 us
#   TRTLLMPrefill.forward     37.5 us      cute _flash_attn_fwd         27.0 us
#   fa_utils (dynamo)         35.2 us      JITCache[key] direct          4.7 us
#
# Most of the 27 us inside ``_flash_attn_fwd`` is building a ~50-element compile
# key and validating tensors; every element of that key is fixed once the token
# count and dtype are, so the compiled kernel and its argument list can be
# captured once and reused.  ``dev/hostprobe.py`` measured that on this path
# host time converts to measured latency at ~0.8 us per us (FA4's launcher sits
# close to the harness's 70 us per-iteration device budget, so anything on top
# of it spills into the window), which is what makes this worth doing.
#
# The capture is done by *running the real library call* with FA4's JIT cache
# temporarily replaced by a recorder, so whatever the wrapper chain decides --
# tile shape, num_splits, sinks, window -- is what gets captured; nothing is
# reimplemented or assumed.  It is then proved bit-for-bit against that same
# call's output before being trusted, and any deviation retires the fast path.
# ---------------------------------------------------------------------------

# Cap on distinct (token count, dtype) launches one layer will capture.
_FA4_MAX_VARIANTS = 4


class _CuteKeySpy:
    """Stand-in for FA4's JIT cache that records one launch.

    Installed on ``_flash_attn_fwd.compile_cache`` for the duration of a single
    priming call.  Both lookup paths are recorded because a warm cache answers
    ``__contains__`` and a cold one goes through ``__setitem__`` first, and the
    launch itself is wrapped to capture the positional argument list.
    """

    __slots__ = ("inner", "key", "args", "calls")

    def __init__(self, inner):
        self.inner = inner
        self.key = None
        self.args = None
        self.calls = 0

    def __getitem__(self, key):
        self.key = key
        fn = self.inner[key]
        spy = self

        def record(*args):
            spy.calls += 1
            spy.args = args
            return fn(*args)

        return record

    def __setitem__(self, key, value):
        self.inner[key] = value

    def __contains__(self, key):
        self.key = key
        return key in self.inner


def _cute_slots(args, q, k, v, out, cu_q, cu_k):
    """Positions in ``args`` holding each operand, located by (address, numel).

    The recorded argument list is FA4's own and its layout differs between
    architectures and library versions, so the operands are found rather than
    assumed.  On this path ``cu_seqlens_q`` and ``cu_seqlens_k`` are usually the
    *same* tensor, which is why a two-hit match is resolved positionally there
    instead of being rejected.  Returns ``None`` if anything is ambiguous, which
    the caller treats as "no fast path".
    """
    def find(t):
        return [j for j, a in enumerate(args)
                if isinstance(a, torch.Tensor)
                and a.data_ptr() == t.data_ptr() and a.numel() == t.numel()]

    slots = []
    for t in (q, k, v, out):
        hits = find(t)
        if len(hits) != 1:
            return None
        slots.append(hits[0])
    if cu_q is cu_k:
        hits = find(cu_q)
        if len(hits) != 2:
            return None
        slots.extend(hits)
    else:
        hits_q, hits_k = find(cu_q), find(cu_k)
        if len(hits_q) != 1 or len(hits_k) != 1:
            return None
        slots.append(hits_q[0])
        slots.append(hits_k[0])
    return tuple(slots)


class _CuteDirect:
    """One captured FA4 launch: the compiled kernel plus its argument list.

    Every argument except the four operands and the two ``cu_seqlens`` is
    constant for the layer (softmax scale, window, sink vector, and a tail of
    ``None``s for the paged / block-sparse / fp8 features this path does not
    use), so the list is kept as a template and only the varying slots are
    filled.  The captured tensors are dropped from the template so a 16 K-token
    case does not pin its priming call's buffers for the life of the layer.
    """

    __slots__ = ("fn", "tmpl", "slots")

    def __init__(self, fn, args, slots):
        tmpl = list(args)
        for i in slots:
            tmpl[i] = None
        self.fn = fn
        self.tmpl = tuple(tmpl)
        self.slots = slots

    def __call__(self, q, k, v, out, cu_q, cu_k):
        args = list(self.tmpl)
        iq, ik, iv, io, icq, ick = self.slots
        args[iq] = q
        args[ik] = k
        args[iv] = v
        args[io] = out
        args[icq] = cu_q
        args[ick] = cu_k
        self.fn(*args)


def _chunked_prefill_remap(
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    block_tables: torch.Tensor | None,
    attention_chunk_size: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, int, int, torch.Tensor | None]:
    """Remap prefill metadata into chunked local-attention virtual batches.

    Follows vLLM's ``make_local_attention_virtual_batches`` algorithm: each
    original sequence is split into ``attention_chunk_size``-wide chunks that
    the kernel sees as independent sequences.

    Returns (cu_seqlens_q', cu_seqlens_k', max_seqlen_q', max_seqlen_k',
             block_tables').
    """
    device = cu_seqlens_q.device
    cu_q_np = cu_seqlens_q.cpu().numpy()
    cu_k_np = cu_seqlens_k.cpu().numpy()

    q_seqlens = cu_q_np[1:] - cu_q_np[:-1]
    k_seqlens = cu_k_np[1:] - cu_k_np[:-1]
    batch_size = len(q_seqlens)

    q_tokens_in_first_block = np.minimum(
        attention_chunk_size - ((k_seqlens - q_seqlens) % attention_chunk_size),
        q_seqlens,
    ).astype(np.int32)
    tokens_in_last_block = (
        attention_chunk_size + (k_seqlens % -attention_chunk_size)
    ).astype(np.int32)

    local_blocks = (
        1 + np.ceil(
            np.maximum(q_seqlens - q_tokens_in_first_block, 0) / attention_chunk_size
        ).astype(np.int32)
    )

    cu_num_blocks = np.cumsum(local_blocks)
    virtual_batches = int(cu_num_blocks[-1])

    block_offsets = np.repeat(cu_num_blocks - local_blocks, local_blocks)
    arange = np.arange(virtual_batches, dtype=np.int32) - block_offsets
    rarange = np.repeat(local_blocks, local_blocks) - arange - 1

    seqlens_q_local = np.repeat(
        q_seqlens - q_tokens_in_first_block, local_blocks,
    ).astype(np.int32)
    seqlens_q_local[arange == 0] = q_tokens_in_first_block
    seqlens_q_local[arange > 0] = np.minimum(
        seqlens_q_local - attention_chunk_size * (arange - 1),
        attention_chunk_size,
    )[arange > 0]

    cu_seqlens_q_local = np.empty(virtual_batches + 1, dtype=np.int32)
    np.cumsum(seqlens_q_local, out=cu_seqlens_q_local[1:])
    cu_seqlens_q_local[0] = 0

    seqlens_k_local = np.full(virtual_batches, attention_chunk_size, dtype=np.int32)
    seqlens_k_local[cu_num_blocks - 1] = tokens_in_last_block

    cu_seqlens_k_local = np.empty(virtual_batches + 1, dtype=np.int32)
    np.cumsum(seqlens_k_local, out=cu_seqlens_k_local[1:])
    cu_seqlens_k_local[0] = 0

    max_seqlen_q = int(seqlens_q_local.max()) if virtual_batches > 0 else 0
    max_seqlen_k = int(seqlens_k_local.max()) if virtual_batches > 0 else 0

    cu_q_out = torch.from_numpy(cu_seqlens_q_local).to(device=device)
    cu_k_out = torch.from_numpy(cu_seqlens_k_local).to(device=device)

    block_tables_out = None
    if block_tables is not None and block_size > 0:
        assert attention_chunk_size % block_size == 0
        pages_per_chunk = attention_chunk_size // block_size

        k_seqstarts_absolute = np.repeat(k_seqlens, local_blocks) - (
            rarange * attention_chunk_size
            + np.repeat(tokens_in_last_block, local_blocks)
        )
        block_starts = k_seqstarts_absolute // block_size

        block_indices = (
            block_starts[:, None]
            + np.arange(pages_per_chunk, dtype=np.int32)
        )
        block_indices = block_indices.reshape(-1).clip(
            max=block_tables.shape[1] - 1,
        )
        batch_indices = np.repeat(
            np.arange(batch_size, dtype=np.int32),
            local_blocks * pages_per_chunk,
        )

        bi_torch = torch.from_numpy(batch_indices)
        bk_torch = torch.from_numpy(block_indices)
        block_tables_out = block_tables[bi_torch, bk_torch].view(
            virtual_batches, -1,
        )

    return cu_q_out, cu_k_out, max_seqlen_q, max_seqlen_k, block_tables_out


def _chunked_decode_remap(
    cache_seqlens: torch.Tensor,
    block_tables: torch.Tensor | None,
    attention_chunk_size: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor | None, int]:
    """Remap decode metadata so the kernel only attends within the last chunk.

    Returns (cache_seqlens', block_tables', max_context_len').
    """
    local_seqlens = torch.clamp(cache_seqlens, max=attention_chunk_size)
    max_context_len = int(local_seqlens.max().item()) if local_seqlens.numel() > 0 else 0

    if block_tables is not None and block_size > 0:
        assert attention_chunk_size % block_size == 0
        pages_per_chunk = attention_chunk_size // block_size
        chunk_start_page = (cache_seqlens - local_seqlens) // block_size
        offsets = torch.arange(pages_per_chunk, device=block_tables.device)
        page_indices = chunk_start_page.unsqueeze(1) + offsets
        page_indices = page_indices.clamp(max=block_tables.shape[1] - 1)
        block_tables = torch.gather(block_tables, 1, page_indices)

    return local_seqlens, block_tables, max_context_len


class Attention(nn.Module):

    def __init__(self, num_heads: int, head_size: int, scale: float,
                 num_kv_heads: int | None = None,
                 sliding_window: int | None = None,
                 sinks: torch.nn.Parameter | None = None,
                 attention_chunk_size: int | None = None,
                 prefer_triton: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.sliding_window = sliding_window
        self.sinks = sinks
        self.attention_chunk_size = attention_chunk_size

        # TODO(tech-debt): For chunked local attention layers the KV cache
        # could be limited to ``attention_chunk_size`` tokens per layer instead
        # of ``max_seq_len``, following vLLM's ``ChunkedLocalAttentionSpec``.
        # This is not needed for correctness but would reduce memory usage.
        self.k_cache = self.v_cache = torch.tensor([])

        attn_cfg = get_attn_backend_config()
        self._block_size = attn_cfg.block_size

        # Per-layer backend selection, mirroring vLLM's per-KV-cache-group
        # choice rather than one global backend.  Reproduced by running
        # ``CudaPlatform.get_valid_backends`` for each config; on SM100:
        #
        #   head_size 128/256, DECODER      -> FLASHINFER  (trtllm-gen)
        #   ENCODER_ONLY / ENCODER_DECODER  -> FLASH_ATTN  ("attention type
        #       not supported" excludes FlashInfer) -- see whisper_attention
        #   PrefixLM bidirectional (Gemma4 sliding, 256) -> TRITON_ATTN
        #       (FlashInfer rejects mm_prefix; FA3 fails "mm_prefix requires
        #       FA4").  Gemma4 global layers (512) stay Triton on SM100
        #       because FA4 TMEM-rejects head_size>128; on Hopper they run
        #       FLASH_ATTN FA4 (FA3 caps at 256, FA4 is valid on SM90).
        #
        # ``prefer_triton`` opts a layer into the mm_prefix / FA3-reject case.
        # ``head_size > 256`` is only forced to Triton on the trtllm (SM100)
        # path; Hopper uses FA4 for those heads.
        #
        # The Triton unified kernel indexes the cache as
        # ``[num_blocks, block_size, num_kv_heads, head_size]`` (NHD), as does
        # the SDPA fallback's ``_cache_seq``, so a layer routed away from
        # trtllm-gen must also be *allocated* NHD -- hence the layout is a
        # per-layer property the engine reads back when sizing the cache.
        # Note this does not depend on whether the Triton kernel imported: an
        # HND cache would silently transpose the head and block dims for
        # either consumer.
        self._triton_only = prefer_triton or (
            head_size > _TRTLLM_MAX_HEAD_SIZE and attn_cfg.use_trtllm
        )
        self._use_trtllm = attn_cfg.use_trtllm and not self._triton_only
        self.kv_layout = "HND" if self._use_trtllm else "NHD"

        # Native FA3/TRTLLM path: sinks -> s_aux, sliding window -> window_size.
        # Held as a plain attribute (not a submodule parameter): the sinks
        # Parameter is already owned by the enclosing attention block, and
        # ``process_weights_after_loading`` may swap in an FP32 copy for the
        # trtllm-gen kernels, which nn.Module would reject on a parameter slot.
        object.__setattr__(self, "_fa3_sinks", sinks)
        self._fa3_window_size = (
            (sliding_window - 1, 0) if sliding_window is not None else (-1, -1)
        )

        self._use_custom_op = False
        self._layer_name = ""
        self.register_buffer(
            "_triton_kv_scale",
            torch.tensor(1.0, dtype=torch.float32),
            persistent=False,
        )
        self._decode_cu_seqlens_q: torch.Tensor | None = None
        self._triton_seq_threshold_3d = max(
            1, _TRITON_MIN_LAUNCH_GRID_SIZE_2D // self.num_kv_heads,
        )
        self._triton_softmax_segm_output: torch.Tensor | None = None
        self._triton_softmax_segm_max: torch.Tensor | None = None
        self._triton_softmax_segm_expsum: torch.Tensor | None = None

        if self._use_trtllm:
            self.store_kvcache = StoreKVCacheHND(page_size=attn_cfg.block_size)
            from ..L1.flashinfer_prefill import TRTLLMPrefill
            from ..L1.flashinfer_decode import TRTLLMDecode
            self.prefill_op = TRTLLMPrefill(
                self.num_heads, self.num_kv_heads, head_size,
            )
            self.decode_op = TRTLLMDecode(
                self.num_heads, self.num_kv_heads, head_size,
            )
        else:
            self.store_kvcache = StoreKVCache()
            from ..L1.flash_attn_prefill import FlashAttnPrefill
            from ..L1.flash_attn_decode import FlashAttnDecode
            self.prefill_op = FlashAttnPrefill(
                self.num_heads, self.num_kv_heads, head_size,
            )
            self.decode_op = FlashAttnDecode(
                self.num_heads, self.num_kv_heads, head_size,
                page_size=self._block_size,
            )
            self.decode_op._window_size = self._fa3_window_size

        from .tree_attn_prefill import TreeAttnPrefill
        self.tree_attn_op = TreeAttnPrefill(
            self.num_heads, self.num_kv_heads, head_size,
        )

        # Native-path kwargs are fixed at init, so build the dict once instead
        # of on every call (see ``_forward_pure`` / ``_forward_mixed``).
        self._fa_extra = self._build_fa_extra()
        self._setup_fused()

    def _build_fa_extra(self) -> dict:
        fa_extra = {}
        if self._fa3_sinks is not None:
            fa_extra["s_aux"] = self._fa3_sinks
        if self._fa3_window_size != (-1, -1):
            fa_extra["window_size"] = self._fa3_window_size
        return fa_extra

    def _setup_fused(self) -> None:
        """Decide the fused-prefill specialization once, here.

        A dense single-sequence causal prefill is the only thing this layer is
        ever asked for in practice, and for short sequences the library kernel
        is 2-4x the device time of a purpose-built one (see ``_fused_limit``).
        So ``forward_impl`` is reduced to a guard over cheap attribute reads
        plus one launch, and every derived quantity it needs -- widths, group
        size, log2-folded scale, sink validity, the crossover -- is computed
        here.  Only the tile shape is left to the call, because it depends on
        the token count.  As a side effect the host path drops from ~45 us to
        ~10 us per call, which does not show up in this benchmark (the harness
        keeps the launch queue deep) but matters to any launch-bound caller.
        """
        h, hkv, d = self.num_heads, self.num_kv_heads, self.head_size
        self._q_width = h * d
        self._kv_width = hkv * d
        self._group_size = h // hkv if hkv else 1
        self._qk_scale = self.scale * _LOG2E
        self._fused_window = (0 if self.sliding_window is None
                              else int(self.sliding_window))
        self._fused_cache = {}
        sink = self._fa3_sinks
        sink_ok = sink is None or (sink.is_cuda and sink.numel() == h
                                   and sink.is_contiguous())
        self._fused_sink = sink if sink_ok else None
        # Is the dense single-sequence shortcut in ``forward_impl`` valid at
        # all?  Chunked local attention needs the virtual-batch remap, and a
        # ``_triton_only`` layer's unpaged path is the torch fallback (which
        # drops sinks and the window), so both keep the general dispatch.
        self._dense_ok = (self.attention_chunk_size is None
                          and not self._triton_only)
        self._fused_ok = (
            _TRITON_OK
            and sink_ok
            and d in _FUSED_HEAD_SIZES
            and hkv > 0 and h % hkv == 0
        )
        self._fused_max_n = _fused_limit(d)
        # Direct-launch state for the library path (see ``_CuteDirect``).
        self._fa4_cache = {}
        self._fa4_ok = _CUTE_FWD is not None

    def _library_prefill(self, query, key, value, N, ctx):
        """Above the fused crossover: the same FlashAttention call
        ``_forward_pure`` makes, with none of the dispatch it does not need.

        Every branch that call sits behind has already been decided by the
        guard in ``forward_impl`` -- one sequence, no block table, empty cache,
        no chunked local attention -- and ``_fa_extra`` (sinks / window) was
        built at init.  What is left is three views and the call itself.  Worth
        doing rather than falling through: on the 656- and 1000-token cases the
        library path is close enough to the harness's host/device balance point
        that ~10 us of extra host time costs 7-9 us of measured latency.
        """
        h, d = self.num_heads, self.head_size
        hkv = self.num_kv_heads
        o = self.prefill_op(
            query.view(N, h, d), key.view(N, hkv, d), value.view(N, hkv, d),
            cu_seqlens_q=ctx.cu_seqlens_q, cu_seqlens_k=ctx.cu_seqlens_k,
            max_seqlen_q=N, max_seqlen_k=N,
            softmax_scale=self.scale, causal=True, **self._fa_extra,
        )
        return o.reshape(N, self._q_width)

    def _direct_prefill(self, query, key, value, N, ctx):
        """``_library_prefill`` with FA4's Python launcher removed.

        Same kernel, same arguments, ~7 us of host work instead of ~46 us -- see
        the table above ``_CuteKeySpy``.  The first call for a given token count
        and dtype goes through the library and captures the launch; every later
        one fills six slots of the recorded argument list and calls the compiled
        kernel.  Anything unexpected falls back to ``_library_prefill``, which is
        the untouched library call.
        """
        cu_q = ctx.cu_seqlens_q
        cu_k = ctx.cu_seqlens_k
        spec = (N, query.dtype, cu_q.numel())
        direct = self._fa4_cache.get(spec)
        if direct is None:
            return self._fa4_capture(query, key, value, N, ctx, spec)
        h, d = self.num_heads, self.head_size
        hkv = self.num_kv_heads
        out = query.new_empty((N, h, d))
        try:
            direct(query.view(N, h, d), key.view(N, hkv, d),
                   value.view(N, hkv, d), out, cu_q, cu_k)
        except Exception:
            # The captured launch reaches past FA4's own validation, so a
            # tensor it rejects raises here rather than computing a wrong
            # answer.  Retire the fast path and let the wrapper handle it.
            self._fa4_ok = False
            self._fa4_cache.clear()
            return self._library_prefill(query, key, value, N, ctx)
        return out.view(N, self._q_width)

    def _fa4_capture(self, query, key, value, N, ctx, spec):
        """Prime one launch: run the library call with FA4's JIT cache spied on.

        Nothing about the kernel is chosen here -- the wrapper picks the tile
        shape, split count, sink and window arguments exactly as it does today,
        and what gets kept is whatever it launched.  Correctness rests on two
        things.  First, every element of FA4's ~50-element compile key is fixed
        once the token count and dtype are (the tile shape, ``q_stage``,
        ``pack_gqa``, sink/window presence and split count all derive from the
        layer's geometry and ``max_seqlen_q``), which is what ``spec`` pins;
        strides are *not* in that key -- they are dynamic in the compiled kernel
        -- so one capture serves both the contiguous tensors ``_clone_tree``
        produces and the captured strides ``_ShiftingPool`` preserves.  Second,
        the captured launch is checked bit-for-bit against the wrapper's own
        output on this very call before it is ever reused.
        """
        h, d = self.num_heads, self.head_size
        hkv = self.num_kv_heads
        # 16-byte pointer and stride alignment is baked into the compiled
        # kernel (``to_cute_tensor(..., assumed_align=16)``), and grad-tracking
        # inputs would make FA4 allocate an LSE and change its compile key.
        step = 16 // query.element_size()
        if (len(self._fa4_cache) >= _FA4_MAX_VARIANTS
                or query.stride(-1) != 1 or key.stride(-1) != 1
                or value.stride(-1) != 1
                or ((query.data_ptr() | key.data_ptr() | value.data_ptr())
                    & 15)
                or ((query.stride(0) | key.stride(0) | value.stride(0) | d)
                    & (step - 1))
                or query.requires_grad or key.requires_grad
                or value.requires_grad):
            self._fa4_ok = False
            return self._library_prefill(query, key, value, N, ctx)

        inner = _CUTE_FWD.compile_cache
        spy = _CuteKeySpy(inner)
        _CUTE_FWD.compile_cache = spy
        try:
            ref = self._library_prefill(query, key, value, N, ctx)
        finally:
            _CUTE_FWD.compile_cache = inner

        # Not one FA4 launch (an FA2/FA3 build, a paged branch, or a wrapper
        # that enqueues more than the one kernel): stay on the library path.
        if spy.calls != 1 or spy.args is None or spy.key is None:
            self._fa4_ok = False
            return ref
        try:
            args = list(spy.args)
            slots = _cute_slots(args, query.view(N, h, d),
                                key.view(N, hkv, d), value.view(N, hkv, d),
                                ref, ctx.cu_seqlens_q, ctx.cu_seqlens_k)
            if slots is None:
                self._fa4_ok = False
                return ref
            direct = _CuteDirect(inner[spy.key], args, slots)
            probe = query.new_empty((N, h, d))
            direct(query.view(N, h, d), key.view(N, hkv, d),
                   value.view(N, hkv, d), probe, ctx.cu_seqlens_q,
                   ctx.cu_seqlens_k)
            # Same kernel, same launch configuration, same inputs, and
            # ``num_splits == 1`` so no atomics: equality is exact or the
            # capture is not what the wrapper ran.
            ok = torch.equal(probe.view(N, self._q_width), ref)
        except Exception:
            ok = False
        if not ok:
            self._fa4_ok = False
            return ref
        self._fa4_cache[spec] = direct
        return ref

    def _fused_prefill(self, query, key, value, N):
        """Fused causal attention over one dense sequence, or None to fall back.

        Returns ``None`` (rather than raising) for any layout the kernel is not
        the right answer for -- mismatched dtypes, an unexpected shape, or more
        distinct layouts than the launcher cache will specialize for -- so the
        caller just falls through to the library call.
        """
        dt = query.dtype
        if (dt is not key.dtype or dt is not value.dtype
                or dt not in _FUSED_DTYPES):
            return None
        # The kernel addresses q/k/v by (row, head, dim) arithmetic rather than
        # through a view, so the shapes it assumes are checked here instead of
        # being enforced by the ``.view()`` calls on the general path.  This also
        # pins the rank, which the stride lookups below rely on.
        kvw = self._kv_width
        if (query.shape != (N, self._q_width) or key.shape != (N, kvw)
                or value.shape != (N, kvw)):
            return None
        tile = _N1_TILE if N == 1 else _fused_tile_shape(self.head_size, N)
        # The compiled kernel is specialized on the exact strides and on 16-byte
        # pointer alignment, and the cached launcher skips re-deriving Triton's
        # key, so those *are* the cache key.
        spec = (tile, query.stride(), key.stride(), value.stride(),
                not ((query.data_ptr() | key.data_ptr() | value.data_ptr())
                     & 15))
        launch = self._fused_cache.get(spec)
        if launch is None:
            return self._fused_compile(query, key, value, N, spec)
        out = query.new_empty((N, self._q_width))
        h = self.num_heads
        try:
            if N == 1:
                launch(h, 1,
                       (query, key, value, out, self._fused_sink, self.scale))
            else:
                bm = tile[0]
                launch((N + bm - 1) // bm, h,
                       (query, key, value, out, self._fused_sink,
                        self._qk_scale, N))
        except Exception:
            # ``_FastLaunch`` reaches into Triton's launcher directly, so a
            # Triton whose launcher signature differs from the one this was
            # written against would raise here rather than compute a wrong
            # answer.  Retire the fast path and let the library handle it; the
            # output buffer is discarded either way.
            self._fused_ok = False
            self._fused_cache.clear()
            return None
        return out

    def _fused_compile(self, query, key, value, N, spec):
        """First call for one layout: compile through Triton's normal path and
        keep the resulting ``CompiledKernel``'s launcher for every later call."""
        if len(self._fused_cache) >= _FUSED_MAX_VARIANTS:
            return None                      # too many layouts: library path
        tile, sq, sk, sv, aligned = spec
        bm, bn, warps, stages = tile
        if not aligned:
            # A base pointer that is not 16-byte aligned would need its own
            # specialization; rare enough to not be worth a cache slot.
            return None
        # The kernel does its element offsets in int32 (Triton's index type for
        # these operands).  Checked once, here, instead of per call: the token
        # cap keeps this true for any sane layout, but a pathological row stride
        # would silently wrap.
        span = max(N * abs(s[0]) + w * abs(s[1])
                   for s, w in ((sq, self._q_width), (sk, self._kv_width),
                                (sv, self._kv_width)))
        if span >= 2 ** 31:
            return None
        out = query.new_empty((N, self._q_width))
        h = self.num_heads
        has_sink = self._fused_sink is not None
        try:
            if N == 1:
                tail = (sq[1], sk[1], sv[1], self._group_size, self.head_size,
                        has_sink)
                compiled = _fused_attn_n1_kernel[(h, 1, 1)](
                    query, key, value, out, self._fused_sink, self.scale,
                    SQ1=tail[0], SK1=tail[1], SV1=tail[2], G=tail[3],
                    D=tail[4], HAS_SINK=tail[5],
                    num_warps=_n1_warps(self.head_size))
            else:
                tail = (sq[0], sq[1], sk[0], sk[1], sv[0], sv[1],
                        self._q_width, self._group_size, self.head_size,
                        bm, bn, has_sink, self._fused_window)
                compiled = _fused_attn_kernel[((N + bm - 1) // bm, h, 1)](
                    query, key, value, out, self._fused_sink, self._qk_scale,
                    N, SQ0=tail[0], SQ1=tail[1], SK0=tail[2], SK1=tail[3],
                    SV0=tail[4], SV1=tail[5], SO=tail[6], G=tail[7],
                    D=tail[8], BM=tail[9], BN=tail[10], HAS_SINK=tail[11],
                    WINDOW=tail[12], num_warps=warps, num_stages=stages)
        except Exception:
            # A tile that will not fit this device (registers, shared memory,
            # tensor memory) must not turn a working layer into a hard failure:
            # retire the fused path and let the library handle everything.
            self._fused_ok = False
            return None
        self._fused_cache[spec] = _FastLaunch(compiled, tail)
        return out

    def set_trtllm_workspace(self, workspace: torch.Tensor):
        if self._use_trtllm:
            self.decode_op._workspace = workspace
            self.prefill_op._workspace = workspace

    def process_weights_after_loading(self) -> None:
        """Prime the FP32 attention-sink copy the trtllm-gen kernels need.

        The two kernels this layer can dispatch to disagree on the sink dtype:
        ``trtllm_batch_{decode,context}_with_kv_cache`` reject anything but
        float32 (``attention_sinks must be a float tensor``) while the
        FlashAttention build vLLM bundles rejects anything but the model dtype
        (``learnable_sink must be bfloat16``) -- and a trtllm layer still falls
        back to FlashAttention for unpaged prefill.  So the layer keeps the
        checkpoint parameter and each trtllm op holds its own converted copy,
        materialized here (as vLLM does in
        ``FlashInferImpl.process_weights_after_loading``) rather than inside a
        forward or a graph capture.
        """
        self._fa_extra = self._build_fa_extra()
        # A captured FA4 launch holds the sink vector it was primed with, so it
        # must not outlive a change to the native-path kwargs.
        self._fa4_cache = {}
        if self.sinks is None or not self._use_trtllm:
            return
        self.prefill_op.prime_sinks(self.sinks)
        self.decode_op.prime_sinks(self.sinks)

    def forward_impl(self, query: torch.Tensor, key: torch.Tensor,
                     value: torch.Tensor) -> torch.Tensor:
        """Core attention logic, callable from both eager and custom-op paths."""
        ctx = get_context()
        N = query.shape[0]

        # One dense causal sequence with no KV cache in play.  This is what the
        # layer actually runs, and it needs none of the dispatch below: no
        # store_kvcache, no chunked remap (which would sync on ``.cpu()``), no
        # paged/decode/mixed/tree-verify branch, no ``_group_*`` indirection.
        # ``max_seqlen_q == N`` is what makes it *one* sequence: any
        # multi-segment cu_seqlens would leave the longest segment below the
        # token count.  Everything here is an attribute read or a C call, no
        # device reads, so a layer that does not qualify pays ~1 us to find out.
        if (self._dense_ok and ctx.is_prefill and not ctx.is_mixed
                and not ctx.is_tree_verify
                and ctx.block_tables is None
                and ctx.sliding_block_tables is None
                and ctx.max_seqlen_q == N and ctx.max_seqlen_k == N
                and self.k_cache.numel() == 0):
            if self._fused_ok and N <= self._fused_max_n:
                out = self._fused_prefill(query, key, value, N)
                if out is not None:
                    return out
            if self._fa4_ok:
                return self._direct_prefill(query, key, value, N, ctx)
            return self._library_prefill(query, key, value, N, ctx)

        q = query.view(N, self.num_heads, self.head_size)
        k = key.view(N, self.num_kv_heads, self.head_size)
        v = value.view(N, self.num_kv_heads, self.head_size)

        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            self.store_kvcache(k, v, k_cache, v_cache, self._group_slot_mapping(ctx))

        if getattr(ctx, "is_tree_verify", False):
            o = self.tree_attn_op(
                q, k_cache, v_cache,
                block_table_prefix=ctx.tree_block_table_prefix,
                cache_seqlens_prefix=ctx.tree_cache_seqlens_prefix,
                cu_seqlens_q_prefix=ctx.tree_cu_seqlens_q_prefix,
                max_seqlen_q_prefix=ctx.tree_max_seqlen_q_prefix,
                max_seqlen_k_prefix=ctx.tree_max_seqlen_k_prefix,
                page_table_expand=ctx.tree_page_table_expand,
                cache_seqlens_expand=ctx.tree_cache_seqlens_expand,
                cu_seqlens_q_expand=ctx.tree_cu_seqlens_q_expand,
                max_seqlen_k_expand=ctx.tree_num_verify_tokens,
                block_size=self._block_size,
                softmax_scale=self.scale,
            )
        elif ctx.is_mixed:
            if self._triton_only:
                can_use_triton = (
                    self._can_use_triton_unified(k_cache, self._group_prefill_block_tables(ctx))
                    and (ctx.num_decode_tokens == 0 or self._group_decode_block_tables(ctx) is not None)
                )
                if can_use_triton:
                    o = self._forward_mixed_triton(q, k_cache, v_cache, ctx)
                else:
                    o = self._forward_mixed_torch(q, k_cache, v_cache, ctx)
                return o.reshape(N, self.num_heads * self.head_size)
            o = self._forward_mixed(q, k_cache, v_cache, ctx)
        else:
            if self._triton_only:
                if self._can_use_triton_unified(k_cache, self._group_block_tables(ctx)):
                    o = self._forward_pure_triton(q, k_cache, v_cache, ctx)
                else:
                    o = self._forward_pure_torch(q, k, v, k_cache, v_cache, ctx)
                return o.reshape(N, self.num_heads * self.head_size)
            o = self._forward_pure(q, k, v, k_cache, v_cache, ctx)

        return o.reshape(N, self.num_heads * self.head_size)

    def _sliding_group_tensor(self, tensor):
        if tensor is None:
            return None
        gid = getattr(self, "_sliding_group_id", None)
        if gid is None:
            return tensor
        return tensor[gid].contiguous()

    def _group_slot_mapping(self, ctx):
        if self.sliding_window and ctx.sliding_slot_mapping is not None:
            return self._sliding_group_tensor(ctx.sliding_slot_mapping)
        return ctx.slot_mapping

    def _group_block_tables(self, ctx):
        if self.sliding_window and ctx.sliding_block_tables is not None:
            return self._sliding_group_tensor(ctx.sliding_block_tables)
        return ctx.block_tables

    def _group_prefill_block_tables(self, ctx):
        if self.sliding_window and ctx.sliding_prefill_block_tables is not None:
            return self._sliding_group_tensor(ctx.sliding_prefill_block_tables)
        return ctx.prefill_block_tables

    def _group_decode_block_tables(self, ctx):
        if self.sliding_window and ctx.sliding_decode_block_tables is not None:
            return self._sliding_group_tensor(ctx.sliding_decode_block_tables)
        return ctx.decode_block_tables

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor) -> torch.Tensor:
        if self._use_custom_op:
            return torch.ops.fastkernels.unified_attention(
                query, key, value, self._layer_name,
            )
        return self.forward_impl(query, key, value)

    def _forward_pure(self, q, k, v, k_cache, v_cache, ctx):
        fa_extra = self._fa_extra

        if ctx.is_prefill:
            cu_q = ctx.cu_seqlens_q
            cu_k = ctx.cu_seqlens_k
            msq = ctx.max_seqlen_q
            msk = ctx.max_seqlen_k
            bt = self._group_block_tables(ctx)

            if self.attention_chunk_size is not None:
                cu_q, cu_k, msq, msk, bt = _chunked_prefill_remap(
                    cu_q, cu_k, bt, self.attention_chunk_size, self._block_size,
                )

            if bt is not None:
                return self.prefill_op(
                    q, k_cache, v_cache,
                    cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
                    max_seqlen_q=msq, max_seqlen_k=msk,
                    softmax_scale=self.scale, causal=True,
                    block_table=bt, **fa_extra,
                )
            return self.prefill_op(
                q, k, v,
                cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
                max_seqlen_q=msq, max_seqlen_k=msk,
                softmax_scale=self.scale, causal=True,
                **fa_extra,
            )

        cache_seqlens = ctx.context_lens
        bt = self._group_block_tables(ctx)
        max_ctx = ctx.max_context_len

        if self.attention_chunk_size is not None:
            cache_seqlens, bt, max_ctx = _chunked_decode_remap(
                cache_seqlens, bt, self.attention_chunk_size, self._block_size,
            )

        return self.decode_op(
            q, k_cache, v_cache,
            cache_seqlens=cache_seqlens, block_table=bt,
            softmax_scale=self.scale, causal=True,
            max_seq_len=max_ctx, **fa_extra,
        )

    def _can_use_triton_unified(
        self,
        k_cache: torch.Tensor,
        block_tables: torch.Tensor | None,
    ) -> bool:
        return (
            # The kernel reads the cache as [num_blocks, block_size,
            # num_kv_heads, head_size]; an HND-allocated layer would have its
            # head and block dims transposed.
            self.kv_layout == "NHD"
            and self.attention_chunk_size is None
            and k_cache.numel() > 0
            and block_tables is not None
        )

    def _get_decode_cu_seqlens_q(
        self,
        num_tokens: int,
        device: torch.device,
    ) -> torch.Tensor:
        needed = num_tokens + 1
        cached = self._decode_cu_seqlens_q
        if cached is None or cached.device != device or cached.numel() < needed:
            cached = torch.arange(needed, dtype=torch.int32, device=device)
            self._decode_cu_seqlens_q = cached
        return cached[:needed]

    def _triton_kv_descale(
        self,
        num_seqs: int,
        num_kv_heads: int,
    ) -> torch.Tensor:
        return self._triton_kv_scale.expand(num_seqs, num_kv_heads)

    def _get_triton_3d_buffers(
        self,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output = self._triton_softmax_segm_output
        if output is None or output.device != device:
            threshold = self._triton_seq_threshold_3d
            segments = _TRITON_NUM_PAR_SOFTMAX_SEGMENTS
            head_dim_padded = 1 << (self.head_size - 1).bit_length()
            self._triton_softmax_segm_output = torch.empty(
                (threshold, self.num_heads, segments, head_dim_padded),
                dtype=torch.float32,
                device=device,
            )
            self._triton_softmax_segm_max = torch.empty(
                (threshold, self.num_heads, segments),
                dtype=torch.float32,
                device=device,
            )
            self._triton_softmax_segm_expsum = torch.empty(
                (threshold, self.num_heads, segments),
                dtype=torch.float32,
                device=device,
            )
        return (
            self._triton_softmax_segm_output,
            self._triton_softmax_segm_max,
            self._triton_softmax_segm_expsum,
        )

    def _forward_paged_triton(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        seqused_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        block_tables: torch.Tensor,
    ) -> torch.Tensor:
        out = torch.empty_like(q)
        num_seqs = int(seqused_k.shape[0])
        kv_descale = self._triton_kv_descale(num_seqs, k_cache.shape[2])
        triton_extra = {}
        if max_seqlen_q == 1 and num_seqs <= self._triton_seq_threshold_3d:
            segm_output, segm_max, segm_expsum = self._get_triton_3d_buffers(
                q.device,
            )
            triton_extra = {
                "seq_threshold_3D": self._triton_seq_threshold_3d,
                "num_par_softmax_segments": _TRITON_NUM_PAR_SOFTMAX_SEGMENTS,
                "softmax_segm_output": segm_output,
                "softmax_segm_max": segm_max,
                "softmax_segm_expsum": segm_expsum,
            }
        if _TRITON_UNIFIED_ACCEPTS_KV_QUANT:
            triton_extra["kv_quant_mode"] = _VllmKVQuantMode.NONE
        _triton_unified_attention(
            q=q,
            k=k_cache,
            v=v_cache,
            out=out,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            seqused_k=seqused_k,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.scale,
            causal=True,
            window_size=self._fa3_window_size,
            block_table=block_tables,
            softcap=0.0,
            q_descale=None,
            k_descale=kv_descale,
            v_descale=kv_descale,
            sinks=self._fa3_sinks,
            **triton_extra,
        )
        return out

    def _forward_pure_triton(self, q, k_cache, v_cache, ctx):
        if ctx.is_prefill:
            seqused_k = ctx.cu_seqlens_k[1:] - ctx.cu_seqlens_k[:-1]
            return self._forward_paged_triton(
                q,
                k_cache,
                v_cache,
                ctx.cu_seqlens_q,
                seqused_k,
                ctx.max_seqlen_q,
                ctx.max_seqlen_k,
                self._group_block_tables(ctx),
            )

        cu_q = self._get_decode_cu_seqlens_q(q.shape[0], q.device)
        return self._forward_paged_triton(
            q,
            k_cache,
            v_cache,
            cu_q,
            ctx.context_lens,
            1,
            ctx.max_context_len,
            self._group_block_tables(ctx),
        )

    def _repeat_kv_for_heads(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_kv_heads == self.num_heads:
            return x
        repeat = self.num_heads // self.num_kv_heads
        return x.repeat_interleave(repeat, dim=1)

    def _cache_seq(self, cache: torch.Tensor, block_table: torch.Tensor,
                   length: int) -> torch.Tensor:
        pages = (length + self._block_size - 1) // self._block_size
        block_ids = block_table[:pages].to(torch.long)
        return cache[block_ids].reshape(
            -1, self.num_kv_heads, self.head_size,
        )[:length]

    def _sdpa_one(self, q_seq: torch.Tensor, k_seq: torch.Tensor,
                  v_seq: torch.Tensor, key_offset: int = 0) -> torch.Tensor:
        q_len = q_seq.size(0)
        k_len = k_seq.size(0)
        k_seq = self._repeat_kv_for_heads(k_seq)
        v_seq = self._repeat_kv_for_heads(v_seq)
        q4 = q_seq.transpose(0, 1).unsqueeze(0)
        k4 = k_seq.transpose(0, 1).unsqueeze(0)
        v4 = v_seq.transpose(0, 1).unsqueeze(0)
        q_pos = key_offset + torch.arange(q_len, device=q_seq.device)
        k_pos = torch.arange(k_len, device=q_seq.device)
        mask = k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)
        out = F.scaled_dot_product_attention(
            q4, k4, v4, attn_mask=mask,
            dropout_p=0.0, scale=self.scale,
        )
        return out.squeeze(0).transpose(0, 1)

    def _prefill_torch_from_tensors(self, q, k, v, cu_q, cu_k) -> torch.Tensor:
        out = torch.empty_like(q)
        num_seqs = cu_q.numel() - 1
        for i in range(num_seqs):
            qs = int(cu_q[i].item())
            qe = int(cu_q[i + 1].item())
            ks = int(cu_k[i].item())
            ke = int(cu_k[i + 1].item())
            key_offset = (ke - ks) - (qe - qs)
            out[qs:qe] = self._sdpa_one(
                q[qs:qe], k[ks:ke], v[ks:ke], key_offset=key_offset,
            )
        return out

    def _prefill_torch_from_cache(self, q, k_cache, v_cache, cu_q, cu_k,
                                  block_tables) -> torch.Tensor:
        out = torch.empty_like(q)
        num_seqs = cu_q.numel() - 1
        for i in range(num_seqs):
            qs = int(cu_q[i].item())
            qe = int(cu_q[i + 1].item())
            k_len = int((cu_k[i + 1] - cu_k[i]).item())
            q_len = qe - qs
            k_seq = self._cache_seq(k_cache, block_tables[i], k_len)
            v_seq = self._cache_seq(v_cache, block_tables[i], k_len)
            out[qs:qe] = self._sdpa_one(
                q[qs:qe], k_seq, v_seq, key_offset=k_len - q_len,
            )
        return out

    def _decode_torch(self, q, k_cache, v_cache, cache_seqlens,
                      block_tables) -> torch.Tensor:
        out = torch.empty_like(q)
        for i in range(q.size(0)):
            k_len = int(cache_seqlens[i].item())
            k_seq = self._cache_seq(k_cache, block_tables[i], k_len)
            v_seq = self._cache_seq(v_cache, block_tables[i], k_len)
            out[i:i + 1] = self._sdpa_one(
                q[i:i + 1], k_seq, v_seq, key_offset=k_len - 1,
            )
        return out

    def _forward_pure_torch(self, q, k, v, k_cache, v_cache, ctx):
        bt = self._group_block_tables(ctx)
        if ctx.is_prefill:
            if bt is not None and k_cache.numel():
                return self._prefill_torch_from_cache(
                    q, k_cache, v_cache,
                    ctx.cu_seqlens_q, ctx.cu_seqlens_k,
                    bt,
                )
            return self._prefill_torch_from_tensors(
                q, k, v, ctx.cu_seqlens_q, ctx.cu_seqlens_k,
            )
        return self._decode_torch(
            q, k_cache, v_cache, ctx.context_lens, bt,
        )

    def _forward_mixed(self, q, k_cache, v_cache, ctx):
        fa_extra = self._fa_extra

        np_ = ctx.num_prefill_tokens
        nd = ctx.num_decode_tokens
        out = torch.empty_like(q)

        if np_ > 0:
            cu_q = ctx.prefill_cu_seqlens_q
            cu_k = ctx.prefill_cu_seqlens_k
            msq = ctx.prefill_max_seqlen_q
            msk = ctx.prefill_max_seqlen_k
            bt = self._group_prefill_block_tables(ctx)

            if self.attention_chunk_size is not None:
                cu_q, cu_k, msq, msk, bt = _chunked_prefill_remap(
                    cu_q, cu_k, bt, self.attention_chunk_size, self._block_size,
                )

            pq = q[:np_].contiguous() if self._use_trtllm else q[:np_]
            out[:np_] = self.prefill_op(
                pq, k_cache, v_cache,
                cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
                max_seqlen_q=msq, max_seqlen_k=msk,
                softmax_scale=self.scale, causal=True,
                block_table=bt, **fa_extra,
            )

        if nd > 0:
            cache_seqlens = ctx.decode_context_lens
            bt = self._group_decode_block_tables(ctx)
            max_ctx = ctx.decode_max_context_len

            if self.attention_chunk_size is not None:
                cache_seqlens, bt, max_ctx = _chunked_decode_remap(
                    cache_seqlens, bt,
                    self.attention_chunk_size, self._block_size,
                )

            out[np_:] = self.decode_op(
                q[np_:], k_cache, v_cache,
                cache_seqlens=cache_seqlens, block_table=bt,
                softmax_scale=self.scale, causal=True,
                max_seq_len=max_ctx, **fa_extra,
            )
        return out

    def _forward_mixed_triton(self, q, k_cache, v_cache, ctx):
        np_ = ctx.num_prefill_tokens
        nd = ctx.num_decode_tokens
        out = torch.empty_like(q)

        if np_ > 0:
            prefill_seqused_k = (
                ctx.prefill_cu_seqlens_k[1:] - ctx.prefill_cu_seqlens_k[:-1]
            )
            out[:np_] = self._forward_paged_triton(
                q[:np_],
                k_cache,
                v_cache,
                ctx.prefill_cu_seqlens_q,
                prefill_seqused_k,
                ctx.prefill_max_seqlen_q,
                ctx.prefill_max_seqlen_k,
                self._group_prefill_block_tables(ctx),
            )

        if nd > 0:
            cu_q = self._get_decode_cu_seqlens_q(nd, q.device)
            out[np_:] = self._forward_paged_triton(
                q[np_:],
                k_cache,
                v_cache,
                cu_q,
                ctx.decode_context_lens,
                1,
                ctx.decode_max_context_len,
                self._group_decode_block_tables(ctx),
            )
        return out

    def _forward_mixed_torch(self, q, k_cache, v_cache, ctx):
        np_ = ctx.num_prefill_tokens
        nd = ctx.num_decode_tokens
        out = torch.empty_like(q)
        if np_ > 0:
            out[:np_] = self._prefill_torch_from_cache(
                q[:np_],
                k_cache,
                v_cache,
                ctx.prefill_cu_seqlens_q,
                ctx.prefill_cu_seqlens_k,
                self._group_prefill_block_tables(ctx),
            )
        if nd > 0:
            out[np_:] = self._decode_torch(
                q[np_:],
                k_cache,
                v_cache,
                ctx.decode_context_lens,
                self._group_decode_block_tables(ctx),
            )
        return out
