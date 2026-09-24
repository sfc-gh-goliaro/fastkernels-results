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

fastkernels candidate: the *dense* (cache-free) prefill branch gets its own
Triton kernel -- see the ``_dense_prefill_kernel`` banner below.  Every other
path (paged prefill/decode, mixed batches, tree verify, the Triton-unified and
SDPA fallbacks) is the baseline's, unchanged.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ....infra.context import get_context, get_attn_backend_config
from ..L1.store_kvcache import StoreKVCache, StoreKVCacheHND

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


# ###########################################################################
# Dense (unpaged) varlen prefill -- hand-written Triton flash attention
#
# With no block table this layer runs its dense branch, handing Q/K/V straight
# to the FlashAttention build; on SM100 that is FA4, whose CuTeDSL launcher is
# pure Python.  Its cost is flat in the problem size -- ~13 us of dispatch for
# anything up to a few hundred tokens -- while the captured workload for this
# layer is dominated by *short* dense prefills (1, 60, 656, 1000 tokens).  At
# those lengths the kernel itself is a rounding error and the launcher is the
# whole latency.
#
# ``_dense_prefill_kernel`` replaces that with a single Triton launch: fp32
# online-softmax accumulators, ``exp2`` with ``log2(e)`` folded into the score
# scale, K read already transposed so the QK GEMM needs no layout conversion,
# and one CTA per (row block, head, sequence).  It covers everything the dense
# branch can be handed -- varlen ``cu_seqlens``, GQA/MQA, bottom-right causal
# alignment, a sliding window, and attention sinks.
#
# Longer prefills stay on FA4: its warp-specialized SM100 kernel sustains
# ~1 PFLOP/s on the 16k-token shape, which a ``tl.dot`` pipeline does not
# reach, and there the launcher cost is noise.  ``_DENSE_FAST_MAX_WORK`` is the
# measured crossover (~1.2e8 query-key-element-channels: N<=192 for 16 heads of
# 128, or N<=512 once a 128-wide sliding window bounds the key extent).
# ###########################################################################

import triton
import triton.language as tl

_LOG2E = 1.4426950408889634
_LOG2E_TL = tl.constexpr(1.4426950408889634)

# Head sizes the kernel tiles exactly (``tl.arange`` wants a power of two).
_DENSE_FAST_HEAD_SIZES = frozenset((16, 32, 64, 128, 256))

# q_tokens * keys_per_query * num_heads * head_size below which one Triton
# launch beats FA4's dispatch overhead.
_DENSE_FAST_MAX_WORK = 120_000_000


@triton.jit
def _dense_prefill_kernel(
    Q, K, V, O, SINKS, CU_Q, CU_K,
    sqt, sqh, skt, skh, svt, svh, sot,
    qk_scale, window,
    GQA: tl.constexpr, D: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    HAS_SINK: tl.constexpr, HAS_WINDOW: tl.constexpr,
    SEQ1: tl.constexpr,
):
    """Causal varlen attention, one CTA per (row block, head, sequence).

    ``qk_scale`` carries ``softmax_scale * log2(e)`` so the softmax can use
    ``exp2`` directly.  Q/K/V are read through explicit row/head strides, so a
    slice of a fused QKV buffer is consumed in place -- no staging copy.

    ``SEQ1`` specializes the every-sequence-is-one-token step (the second
    hottest captured shape): a softmax over a single key is exactly 1, so the
    output is that key's value row and both GEMMs -- with them the whole
    tensor-memory round trip -- drop out.  A sink *would* change the result
    (it takes a share of the weight), so the caller keeps sink layers on the
    general path.
    """
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)

    q_beg = tl.load(CU_Q + pid_b)
    q_len = tl.load(CU_Q + pid_b + 1) - q_beg
    m_start = pid_m * BLOCK_M
    if m_start >= q_len:
        return
    k_beg = tl.load(CU_K + pid_b)
    k_len = tl.load(CU_K + pid_b + 1) - k_beg

    if SEQ1:
        offs_d1 = tl.arange(0, D)
        o_ptr = O + q_beg * sot + pid_h * D + offs_d1
        if k_len > 0:
            tl.store(o_ptr, tl.load(V + k_beg * svt + (pid_h // GQA) * svh
                                    + offs_d1))
        else:
            tl.store(o_ptr, tl.zeros([D], dtype=O.dtype.element_ty))
        return
    # Bottom-right causal alignment (FlashAttention's convention): query i of
    # the sequence attends keys 0..i+delta.
    delta = k_len - q_len

    offs_d = tl.arange(0, D)
    offs_n = tl.arange(0, BLOCK_N)
    offs_m = m_start + tl.arange(0, BLOCK_M)
    rmask = offs_m < q_len
    q_pos = offs_m + delta
    q = tl.load(Q + q_beg * sqt + pid_h * sqh
                + offs_m[:, None] * sqt + offs_d[None, :],
                mask=rmask[:, None], other=0.0)
    kv_h = pid_h // GQA
    k_base = K + k_beg * skt + kv_h * skh
    v_base = V + k_beg * svt + kv_h * svh

    # -1e30, not -inf: under a sliding window a whole tile can be masked out,
    # and -inf would make the rescale exp2(-inf - -inf) = NaN.
    m_i = tl.full([BLOCK_M], -1.0e30, dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    lo = 0
    if HAS_WINDOW:
        lo = tl.maximum(m_start + delta - window, 0) // BLOCK_N * BLOCK_N
    hi = tl.minimum(m_start + BLOCK_M + delta, k_len)
    for start_n in tl.range(lo, hi, BLOCK_N):
        n = start_n + offs_n
        nm = n < k_len
        kt = tl.load(k_base + offs_d[:, None] + n[None, :] * skt,
                     mask=nm[None, :], other=0.0)
        qk = tl.dot(q, kt) * qk_scale
        keep = (n[None, :] <= q_pos[:, None]) & nm[None, :]
        if HAS_WINDOW:
            keep = keep & (n[None, :] >= (q_pos - window)[:, None])
        qk = tl.where(keep, qk, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v = tl.load(v_base + n[:, None] * svt + offs_d[None, :],
                    mask=nm[:, None], other=0.0)
        acc = tl.dot(p.to(v.dtype), v, acc)
        m_i = m_new

    if HAS_SINK:
        # An attention sink is an extra logit with no value: it only enters the
        # softmax denominator.  Rebase the running max first so exp2 cannot
        # overflow when the sink dominates the row.
        sk = tl.load(SINKS + pid_h).to(tl.float32) * _LOG2E_TL
        m_new = tl.maximum(m_i, sk)
        alpha = tl.exp2(m_i - m_new)
        l_i = l_i * alpha + tl.exp2(sk - m_new)
        acc = acc * alpha[:, None]

    # l_i == 0 only for a row whose window excluded every key; emit zeros.
    acc = acc / tl.where(l_i == 0.0, 1.0, l_i)[:, None]
    tl.store(O + q_beg * sot + pid_h * D
             + offs_m[:, None] * sot + offs_d[None, :],
             acc.to(O.dtype.element_ty), mask=rmask[:, None])


def _dense_prefill_tile(max_seqlen_q: int):
    """(BLOCK_M, BLOCK_N, num_warps, num_stages) for a dense prefill.

    Short sequences want the smallest tile that still feeds ``tl.dot``: fewer
    bytes of shared memory and fewer masked lanes per CTA.
    """
    if max_seqlen_q <= 64:
        return 16, 64, 4, 2
    if max_seqlen_q <= 256:
        return 32, 64, 4, 2
    return 64, 64, 4, 2


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

        # Dense-prefill fast path (see the kernel banner above).  Decided once
        # here so ``forward`` only reads flags; ``_triton_only`` layers keep
        # their own unified-attention path and are excluded.
        self._dense_fast = (
            head_size in _DENSE_FAST_HEAD_SIZES
            and not self._triton_only
            and self.attention_chunk_size is None
            and self.num_heads % self.num_kv_heads == 0
        )
        self._dense_fast_gqa = self.num_heads // self.num_kv_heads
        self._dense_fast_scale = scale * _LOG2E
        self._dense_fast_window = (
            sliding_window - 1 if sliding_window is not None else -1
        )
        # Per-query-token budget: the work threshold divided by the channels
        # every query row touches.
        self._dense_fast_area = max(
            1, _DENSE_FAST_MAX_WORK // (self.num_heads * head_size),
        )

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
        if self.sinks is None or not self._use_trtllm:
            return
        self.prefill_op.prime_sinks(self.sinks)
        self.decode_op.prime_sinks(self.sinks)

    def forward_impl(self, query: torch.Tensor, key: torch.Tensor,
                     value: torch.Tensor) -> torch.Tensor:
        """Core attention logic, callable from both eager and custom-op paths."""
        ctx = get_context()
        N = query.shape[0]
        if self._dense_prefill_ok(ctx, N, query.dtype):
            return self._dense_prefill(query, key, value, ctx)

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

    def _dense_prefill_ok(self, ctx, num_tokens: int, dtype) -> bool:
        """Is this step a dense prefill small enough for the Triton kernel?

        A populated KV cache is excluded rather than handled: the baseline path
        also has to *write* the new tokens into it, and that is not what this
        kernel is for.
        """
        if not self._dense_fast or not ctx.is_prefill or ctx.is_mixed:
            return False
        if dtype is not torch.bfloat16 and dtype is not torch.float16:
            return False
        if self.k_cache.numel() or self._group_block_tables(ctx) is not None:
            return False
        if getattr(ctx, "is_tree_verify", False):
            return False
        keys = ctx.max_seqlen_k
        window = self._dense_fast_window
        if 0 <= window < keys - 1:
            keys = window + 1
        return num_tokens * keys <= self._dense_fast_area

    def _dense_prefill(self, query, key, value, ctx):
        """Varlen causal dense prefill through the Triton kernel."""
        T = query.shape[0]
        H = self.num_heads
        D = self.head_size
        q = query.view(T, H, D)
        k = key.view(T, self.num_kv_heads, D)
        v = value.view(T, self.num_kv_heads, D)
        out = torch.empty((T, H * D), dtype=query.dtype, device=query.device)
        if T == 0:
            return out
        max_seqlen_q = ctx.max_seqlen_q
        max_seqlen_k = ctx.max_seqlen_k
        cu_q = ctx.cu_seqlens_q
        # A window at least as wide as the longest key run masks nothing, so
        # compile the specialization without it.
        window = self._dense_fast_window
        if window >= max_seqlen_k - 1:
            window = -1
        seq1 = (max_seqlen_q == 1 and max_seqlen_k == 1
                and self._fa3_sinks is None)
        if seq1:
            BLOCK_M, BLOCK_N, warps, stages = 16, 16, 1, 1
        else:
            BLOCK_M, BLOCK_N, warps, stages = _dense_prefill_tile(max_seqlen_q)
        grid = (-(-max_seqlen_q // BLOCK_M), H, cu_q.numel() - 1)
        _dense_prefill_kernel[grid](
            q, k, v, out, self._fa3_sinks, cu_q, ctx.cu_seqlens_k,
            q.stride(0), q.stride(1), k.stride(0), k.stride(1),
            v.stride(0), v.stride(1), H * D,
            self._dense_fast_scale, window,
            GQA=self._dense_fast_gqa, D=D,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            HAS_SINK=self._fa3_sinks is not None,
            HAS_WINDOW=window >= 0, SEQ1=seq1,
            num_warps=warps, num_stages=stages,
        )
        return out

    def _forward_pure(self, q, k, v, k_cache, v_cache, ctx):
        fa_extra = {}
        if self._fa3_sinks is not None:
            fa_extra["s_aux"] = self._fa3_sinks
        if self._fa3_window_size != (-1, -1):
            fa_extra["window_size"] = self._fa3_window_size

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
        fa_extra = {}
        if self._fa3_sinks is not None:
            fa_extra["s_aux"] = self._fa3_sinks
        if self._fa3_window_size != (-1, -1):
            fa_extra["window_size"] = self._fa3_window_size

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
