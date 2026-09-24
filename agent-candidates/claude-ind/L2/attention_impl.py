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
# Dense (unpaged) varlen causal attention.
#
# The hot shape for this layer is a freshly-projected prefill whose KV cache
# is still empty: ``_forward_pure`` hands q/k/v straight to
# ``flash_attn_varlen_func`` with no page table.  Two costs dominate there.
# FA4's CuTeDSL launcher spends ~50 us of *host* time per call, which swamps
# every shape below a couple of thousand tokens; and its tile choice for
# those shapes leaves the machine idle.  The kernel below (Triton, one CTA
# per (q-tile, head, sequence)) replaces that path.  Every other dispatch --
# paged prefill/decode, mixed batches, tree verify, chunked local attention,
# the Triton-unified and SDPA fallbacks -- is untouched.
#
# The kernel wins below ``_FK_MAX_Q_ELEMS`` query elements and loses above it:
# FA4 is a warp-specialized tcgen05 pipeline that overlaps its softmax with
# the MMAs, which Triton cannot express here (``warp_specialize=True`` fails
# to compile this loop, and ``num_ctas=2`` trips an assert in PlanCTA), so
# once there is enough work to hide FA4's startup the baseline dispatch is
# faster and long prefills keep using it.
#
# ``_fk_launcher`` caches the compiled kernel and calls its launcher directly:
# ``JITFunction.run`` re-derives the specialization key from all twelve
# runtime arguments on every call (~15 us), which is the same trap FA4 falls
# into.  Bypassing it costs ~6 us instead.
# ###########################################################################

import triton
import triton.language as tl

_FK_NEG = tl.constexpr(-1.0e30)
_FK_LOG2E = tl.constexpr(1.4426950408889634)


@triton.jit
def _fk_inner(acc, l_i, m_i, q, K, V, koff, voff, stride_kn, stride_vn,
              offs_m, offs_n, offs_d, qk_scale, lo, hi, seqlen_k, diff,
              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
              HEAD_DIM: tl.constexpr, MASKED: tl.constexpr,
              WINDOW_LEFT: tl.constexpr):
    for start_n in tl.range(lo, hi, BLOCK_N):
        n = start_n + offs_n
        if MASKED:
            nm = n < seqlen_k
            kt = tl.load(K + koff + n[None, :] * stride_kn + offs_d[:, None],
                         mask=nm[None, :], other=0.0)
            qk = tl.dot(q, kt) * qk_scale
            keep = (offs_m[:, None] + diff) >= n[None, :]
            if WINDOW_LEFT >= 0:
                keep = keep & ((offs_m[:, None] + diff - WINDOW_LEFT) <= n[None, :])
            qk = tl.where(keep & nm[None, :], qk, _FK_NEG)
            m_new = tl.maximum(m_i, tl.max(qk, 1))
            p = tl.math.exp2(qk - m_new[:, None])
        else:
            kt = tl.load(K + koff + n[None, :] * stride_kn + offs_d[:, None])
            qk = tl.dot(q, kt)
            m_new = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            p = tl.math.exp2(qk * qk_scale - m_new[:, None])
        alpha = tl.math.exp2(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        if MASKED:
            vt = tl.load(V + voff + n[:, None] * stride_vn + offs_d[None, :],
                         mask=(n < seqlen_k)[:, None], other=0.0)
        else:
            vt = tl.load(V + voff + n[:, None] * stride_vn + offs_d[None, :])
        acc = tl.dot(p.to(vt.dtype), vt, acc)
        m_i = m_new
    return acc, l_i, m_i


@triton.jit
def _fk_fwd(Q, K, V, Out, Sinks, CuQ, CuK, sm_scale,
            stride_qm, stride_kn, stride_vn, stride_om,
            GQA: tl.constexpr, HEAD_DIM: tl.constexpr,
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
            WINDOW_LEFT: tl.constexpr, HAS_SINK: tl.constexpr):
    start_m = tl.program_id(0)
    h = tl.program_id(1)
    b = tl.program_id(2)

    q_start = tl.load(CuQ + b)
    seqlen_q = tl.load(CuQ + b + 1) - q_start
    k_start = tl.load(CuK + b)
    seqlen_k = tl.load(CuK + b + 1) - k_start

    m_lo = start_m * BLOCK_M
    ok = (m_lo < seqlen_q).to(tl.int32)

    kh = h // GQA
    offs_m = m_lo + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    m_mask = offs_m < seqlen_q
    q = tl.load(Q + (q_start + offs_m)[:, None] * stride_qm + h * HEAD_DIM
                + offs_d[None, :], mask=m_mask[:, None], other=0.0)

    koff = k_start * stride_kn + kh * HEAD_DIM
    voff = k_start * stride_vn + kh * HEAD_DIM

    diff = seqlen_k - seqlen_q
    qk_scale = sm_scale * _FK_LOG2E

    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    m_i = tl.full([BLOCK_M], _FK_NEG, dtype=tl.float32)

    hi = tl.minimum(seqlen_k, m_lo + BLOCK_M + diff) * ok
    if WINDOW_LEFT >= 0:
        lo = tl.maximum(0, m_lo + diff - WINDOW_LEFT) // BLOCK_N * BLOCK_N
        acc, l_i, m_i = _fk_inner(acc, l_i, m_i, q, K, V, koff, voff,
                                  stride_kn, stride_vn, offs_m, offs_n, offs_d,
                                  qk_scale, lo, hi, seqlen_k, diff,
                                  BLOCK_M, BLOCK_N, HEAD_DIM, True, WINDOW_LEFT)
    else:
        n1 = (tl.minimum(m_lo + diff + 1, seqlen_k) // BLOCK_N * BLOCK_N) * ok
        acc, l_i, m_i = _fk_inner(acc, l_i, m_i, q, K, V, koff, voff,
                                  stride_kn, stride_vn, offs_m, offs_n, offs_d,
                                  qk_scale, 0, n1, seqlen_k, diff,
                                  BLOCK_M, BLOCK_N, HEAD_DIM, False, -1)
        acc, l_i, m_i = _fk_inner(acc, l_i, m_i, q, K, V, koff, voff,
                                  stride_kn, stride_vn, offs_m, offs_n, offs_d,
                                  qk_scale, n1, hi, seqlen_k, diff,
                                  BLOCK_M, BLOCK_N, HEAD_DIM, True, -1)

    if HAS_SINK:
        s = tl.load(Sinks + h).to(tl.float32) * _FK_LOG2E
        m_new = tl.maximum(m_i, s)
        alpha = tl.math.exp2(m_i - m_new)
        l_i = l_i * alpha + tl.math.exp2(s - m_new)
        acc = acc * alpha[:, None]

    acc = acc / tl.where(l_i == 0.0, 1.0, l_i)[:, None]
    tl.store(Out + (q_start + offs_m)[:, None] * stride_om + h * HEAD_DIM
             + offs_d[None, :], acc.to(Out.dtype.element_ty),
             mask=m_mask[:, None])


_FK_HEAD_DIMS = frozenset((16, 32, 64, 128, 256))
_FK_LAUNCHERS: dict = {}

# Query elements (tokens * heads * head_dim) below which the Triton kernel
# below beats the FA4 dense path, measured on B200 across the three head
# geometries this layer is built with (16x128 MQA, 32x64 GQA-8, 32x128 GQA-4).
# The crossover tracks the query element count rather than the token count:
# both backends are latency-bound while the grid cannot fill the GPU, and FA4
# pulls ahead once there is enough work to hide its warp-specialized tcgen05
# pipeline's startup cost -- which lands near 6.5e5 elements for every
# geometry tested (320 tokens at 16 heads x 128, 160 at 32 x 128).
_FK_MAX_Q_ELEMS = 655360


# The q tile plus ``num_stages`` buffered k/v tiles have to fit in a CTA's
# 227 KB of shared memory.  Budgeting below the hardware limit leaves room for
# the staging buffer Triton adds for the second dot's operand.
_FK_SMEM_BUDGET = 190000


def _fk_config(max_q: int, head_dim: int):
    """(BLOCK_M, BLOCK_N, num_warps, num_stages) for a query length.

    Tuned per bucket on B200 against the benchmark's own timing loop, across
    all three head geometries this layer is built with.  Short prefills are
    parallelism-starved -- the grid is ``ceil(max_q / BLOCK_M) * heads`` CTAs,
    so a 60-token batch with BLOCK_M=64 leaves most of the GPU idle and wants
    the narrowest q tile.  Longer ones want a wide tile so the per-block
    softmax and the output rescale amortize over more MMA work.  ``BLOCK_N``
    then shrinks until the tiles fit in shared memory, which only bites for
    head_size 256.
    """
    if max_q <= 32:
        bm, bn, warps, stages = 16, 32, 4, 1
    elif max_q <= 320:
        bm, bn, warps, stages = 16, 64, 4, 2
    elif max_q <= 640:
        bm, bn, warps, stages = 64, 64, 4, 2
    else:
        bm, bn, warps, stages = 128, 128, 8, 2
    while (bm + 2 * stages * bn) * head_dim * 2 > _FK_SMEM_BUDGET and bn > 16:
        bn //= 2
    return bm, bn, warps, stages


def _fk_make_launcher(args, warps, stages):
    """Compile ``_fk_fwd`` for *args* and return ``launch(g0, g1, g2, args)``.

    The fast path hands the compiled kernel's own launcher the grid, the
    stream and the arguments, skipping ``JITFunction.run``'s per-call
    specialization work.  If that ABI is not what this Triton build expects
    the closure falls back to the ordinary indexed launch; if the kernel will
    not build at all (a block shape this head size cannot afford), ``None``
    sends the caller back to the baseline dispatch.
    """
    kw = {"GQA": args[12], "HEAD_DIM": args[13], "BLOCK_M": args[14],
          "BLOCK_N": args[15], "WINDOW_LEFT": args[16], "HAS_SINK": args[17]}
    try:
        from triton.runtime import driver
        kernel = _fk_fwd.warmup(*args[:12], grid=(1, 1, 1), num_warps=warps,
                                num_stages=stages, **kw)
        kernel._init_handles()
        run = kernel.run
        func = kernel.function
        packed = kernel.packed_metadata
        active = driver.active
        device = active.get_current_device()
        stream_of = active.get_current_stream

        def launch(g0, g1, g2, a):
            run(g0, g1, g2, stream_of(device), func, packed,
                None, None, None, *a)

        launch(1, 1, 1, args)   # prove the ABI before trusting it
        return launch
    except Exception:
        pass
    try:
        def launch(g0, g1, g2, a):
            _fk_fwd[(g0, g1, g2)](
                *a[:12], num_warps=warps, num_stages=stages,
                GQA=a[12], HEAD_DIM=a[13], BLOCK_M=a[14], BLOCK_N=a[15],
                WINDOW_LEFT=a[16], HAS_SINK=a[17])
        launch(1, 1, 1, args)
        return launch
    except Exception:
        return None


def _fk_dense_attn(q, k, v, out, cu_q, cu_k, max_q, num_seqs, num_heads,
                   gqa, head_dim, scale, sinks, window_left):
    """Varlen causal attention over unpaged q/k/v shaped ``[N, heads*dim]``.

    Returns ``out``, or ``None`` if the kernel could not be built for this
    geometry -- the caller then runs the baseline dispatch.
    """
    if max_q <= 0:
        return None
    bm, bn, warps, stages = _fk_config(max_q, head_dim)
    has_sink = sinks is not None
    args = (q, k, v, out, sinks, cu_q, cu_k, scale,
            q.stride(0), k.stride(0), v.stride(0), out.stride(0),
            gqa, head_dim, bm, bn, window_left, has_sink)
    # Triton bakes argument specializations into the code it emits: 16-byte
    # aligned pointers, and strides it can prove are 1 or a multiple of 16
    # (which is what lets the q/k/v loads vectorize).  A cached launcher is
    # only valid for arguments that specialize the same way, so the strides,
    # the scale and the pointer alignment are all part of the key.
    key = (gqa, head_dim, bm, bn, window_left, has_sink, q.dtype, q.device,
           args[8], args[9], args[10], args[11], scale,
           (q.data_ptr() | k.data_ptr() | v.data_ptr()) & 15)
    try:
        launch = _FK_LAUNCHERS[key]
    except KeyError:
        if len(_FK_LAUNCHERS) > 256:     # pathological stride churn; start over
            _FK_LAUNCHERS.clear()
        launch = _FK_LAUNCHERS[key] = _fk_make_launcher(args, warps, stages)
    if launch is None:
        return None
    launch(-(-max_q // bm), num_heads, num_seqs, args)
    return out


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

        # Dense-prefill fast path (see _fk_dense_attn).  ``_triton_only``
        # layers (head_size > 256, or mm_prefix) keep the baseline dispatch so
        # their reference numerics are untouched.
        self._fk_window_left = (
            sliding_window - 1 if sliding_window is not None else -1
        )
        self._fk_gqa = self.num_heads // self.num_kv_heads
        self._fk_no_sliding_groups = not sliding_window
        # A sliding window caps the key range each query scans, so the work
        # per token stops growing with the sequence and the crossover moves
        # out; measured at ~2x on the 128-wide window this layer is built with.
        self._fk_max_q = _FK_MAX_Q_ELEMS // max(1, num_heads * head_size)
        if sliding_window:
            self._fk_max_q *= 2
        self._fk_ok = (
            not self._triton_only
            and attention_chunk_size is None
            and head_size in _FK_HEAD_DIMS
            and self.num_heads % self.num_kv_heads == 0
        )

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

        # Unpaged causal prefill: nothing to write to the cache, no page table
        # to walk.  Kept as one flat condition so the common case costs a
        # handful of attribute loads.
        if (self._fk_ok and ctx.is_prefill and not ctx.is_mixed
                and ctx.max_seqlen_q <= self._fk_max_q
                and not getattr(ctx, "is_tree_verify", False)
                and not self.k_cache.numel()
                and ctx.block_tables is None
                and (self._fk_no_sliding_groups
                     or ctx.sliding_block_tables is None)
                and query.stride(-1) == 1 and key.stride(-1) == 1
                and value.stride(-1) == 1):
            cu_q = ctx.cu_seqlens_q
            n_heads = self.num_heads
            head_size = self.head_size
            fast = _fk_dense_attn(
                query, key, value,
                query.new_empty((query.shape[0], n_heads * head_size)),
                cu_q, ctx.cu_seqlens_k, ctx.max_seqlen_q, cu_q.shape[0] - 1,
                n_heads, self._fk_gqa, head_size, self.scale,
                self._fa3_sinks, self._fk_window_left,
            )
            if fast is not None:
                return fast

        N = query.shape[0]

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
