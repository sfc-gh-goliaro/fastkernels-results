"""TRTLLM-gen paged attention decode, with a hand-written small-batch fast path.

Accepts the same interface as FlashAttnDecode so that LlamaAttention can
dispatch to either backend without branch logic.

Two paths, selected host-side on shapes and flags only (never on a device
value, so ``forward`` stays sync-free and CUDA-graph capturable):

* **Long-context small-batch flash-decode** (below) -- a split-K,
  online-softmax paged decode written here.  At batch 1 with one KV head and a
  ~124k-token context, trtllm-gen reaches ~1.8 TB/s of a ~6 TB/s achievable HBM
  read on B200: with a single request there is no batch dimension to
  parallelise over, so its KV-split heuristic both under-fills the machine and
  over-pays the cross-CTA reduction.  16 query heads against 1 KV head is a
  gather-and-reduce, not a GEMM, so the decode is just a stream of 4 KB pages
  with an fp32 online softmax on top.
* **trtllm-gen** for everything else, byte-for-byte the call the parent
  shipped, so nothing that falls through can regress.  Two groups fall through
  for different reasons:
    - *Large batch* (e.g. [981, 32, 128] over 8 KV heads): already at or above
      the pure-HBM roofline -- 7.4 TB/s effective, which beats a streaming read
      because a wide page table gives real L2 hits.  There is no multiple there.
    - *Short context* (below ``MIN_CTX``): a split-K decode needs two kernel
      launches, ~2.1 us each on this part, and at a few hundred tokens of KV
      that overhead is larger than the whole attention.  The measured crossover
      is ~24k tokens.
"""

import torch
import torch.nn as nn
from flashinfer.decode import get_trtllm_gen_fmha_module
from flashinfer.utils import device_support_pdl, get_device_sm_count

try:
    import triton
    import triton.language as tl
    _HAVE_TRITON = True
except Exception:                                            # pragma: no cover
    _HAVE_TRITON = False


def prime_trtllm_sinks(module: nn.Module, sinks: torch.Tensor | None) -> None:
    """Materialize the FP32 attention-sink copy the trtllm-gen kernels need.

    ``trtllm_batch_decode_with_kv_cache`` /
    ``trtllm_batch_context_with_kv_cache`` hard-check
    ``attention_sinks.dtype == float32``, while the FlashAttention build vLLM
    bundles asserts the opposite for the same weights
    (``learnable_sink must be bfloat16``).  So the conversion cannot live on the
    layer -- only the op knows which kernel it is about to call.  vLLM does the
    same conversion once per layer in
    ``FlashInferImpl.process_weights_after_loading``; call this from the owning
    attention layer's post-load hook so the copy never lands inside a forward
    or a CUDA-graph capture.
    """
    if sinks is None:
        module._sinks_fp32 = None
    elif sinks.dtype == torch.float32:
        module._sinks_fp32 = sinks
    else:
        module._sinks_fp32 = sinks.detach().to(torch.float32)
    module._sinks_src = sinks


def trtllm_sinks(module: nn.Module, s_aux: torch.Tensor | None):
    """Return the FP32 view of ``s_aux``, priming the cache if needed."""
    if s_aux is None or s_aux.dtype == torch.float32:
        return s_aux
    if module._sinks_fp32 is None or module._sinks_src is not s_aux:
        prime_trtllm_sinks(module, s_aux)
    return module._sinks_fp32


if _HAVE_TRITON:

    @triton.jit
    def _kv_tile(q, m_i, l_i, acc, kbase, vbase, btp, t0, end, scale,
                 sk_p, sk_t, sv_p, sv_t, offs_d,
                 BLOCK_N: tl.constexpr, PAGE: tl.constexpr,
                 MASKED: tl.constexpr, I64: tl.constexpr):
        """One BLOCK_N-token slab: paged gather + online-softmax update."""
        offs_n = t0 + tl.arange(0, BLOCK_N)
        # One page id per lane. The PAGE-fold redundancy is L1-resident and
        # costs less than reshaping an [NPG, PAGE] tile into [BLOCK_N], which
        # forces a shared-memory relayout of the address vector (measured 2x
        # slower on the mainloop).
        if MASKED:
            nmask = offs_n < end
            pg = tl.load(btp + offs_n // PAGE, mask=nmask, other=0)
        else:
            pg = tl.load(btp + offs_n // PAGE)
        if I64:
            pg = pg.to(tl.int64)
            tok = (offs_n % PAGE).to(tl.int64)
        else:
            tok = offs_n % PAGE
        koff = (pg * sk_p + tok * sk_t)[:, None] + offs_d[None, :]
        voff = (pg * sv_p + tok * sv_t)[:, None] + offs_d[None, :]
        if MASKED:
            k = tl.load(kbase + koff, mask=nmask[:, None], other=0.0)
        else:
            k = tl.load(kbase + koff)
        s = tl.dot(q, tl.trans(k)) * scale
        if MASKED:
            s = tl.where(nmask[None, :], s, -1e30)
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        if MASKED:
            v = tl.load(vbase + voff, mask=nmask[:, None], other=0.0)
        else:
            v = tl.load(vbase + voff)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        return m_new, l_i, acc

    @triton.jit
    def _flash_decode_split(
        Q, K, V, BT, SL, OUT, PO, PM, PL, scale,
        sq_b, sq_h, sk_p, sk_h, sk_t, sv_p, sv_h, sv_t, sbt_b,
        so_b, so_h, spo_b, spo_h, spm_b, spm_h,
        tokens_per_split, nsplit,
        NKV: tl.constexpr, H_PER_KV: tl.constexpr, BLOCK_H: tl.constexpr,
        D: tl.constexpr, BLOCK_N: tl.constexpr, PAGE: tl.constexpr,
        WRITE_FINAL: tl.constexpr, I64: tl.constexpr,
    ):
        """One CTA = one (request, kv head, KV slice).  Online fp32 softmax.

        With ``WRITE_FINAL`` (only when the whole context is one split) the CTA
        normalises and writes the answer itself, so no combine launch is needed.
        Otherwise it leaves an unnormalised ``(m, l, O)`` partial behind.
        """
        pid_s = tl.program_id(0)
        pid_bh = tl.program_id(1)
        b = pid_bh // NKV
        kvh = pid_bh % NKV

        seqlen = tl.load(SL + b).to(tl.int32)
        start = pid_s * tokens_per_split
        # The last split runs to the true seq_len rather than to its nominal
        # slice end. ``tokens_per_split`` is derived from the caller's
        # ``max_seq_len``, which is only ever a *bound*; if a caller understates
        # it the tail lands on this CTA instead of being silently dropped.
        end = tl.minimum(start + tokens_per_split, seqlen)
        if pid_s == nsplit - 1:
            end = seqlen

        offs_h = tl.arange(0, BLOCK_H)
        offs_d = tl.arange(0, D)
        hmask = offs_h < H_PER_KV
        qh = kvh * H_PER_KV + offs_h
        q = tl.load(Q + b * sq_b + qh[:, None] * sq_h + offs_d[None, :],
                    mask=hmask[:, None], other=0.0)

        m_i = tl.full([BLOCK_H], -1e30, tl.float32)
        l_i = tl.zeros([BLOCK_H], tl.float32)
        acc = tl.zeros([BLOCK_H, D], tl.float32)

        kbase = K + kvh * sk_h
        vbase = V + kvh * sv_h
        btp = BT + b * sbt_b

        # Full slabs unpredicated, one masked tail slab: every tile but the last
        # of the last split is exactly BLOCK_N tokens, and predicating them costs
        # a mask register per load plus the -1e30 select on the whole score tile.
        full_end = start + ((tl.maximum(end - start, 0) // BLOCK_N) * BLOCK_N)
        for t0 in range(start, full_end, BLOCK_N):
            m_i, l_i, acc = _kv_tile(q, m_i, l_i, acc, kbase, vbase, btp, t0,
                                     end, scale, sk_p, sk_t, sv_p, sv_t, offs_d,
                                     BLOCK_N, PAGE, False, I64)
        if full_end < end:
            m_i, l_i, acc = _kv_tile(q, m_i, l_i, acc, kbase, vbase, btp,
                                     full_end, end, scale, sk_p, sk_t, sv_p,
                                     sv_t, offs_d, BLOCK_N, PAGE, True, I64)

        if WRITE_FINAL:
            o = acc / tl.where(l_i > 0, l_i, 1.0)[:, None]
            tl.store(OUT + b * so_b + qh[:, None] * so_h + offs_d[None, :],
                     o.to(OUT.dtype.element_ty), mask=hmask[:, None])
        else:
            # Partials are [B, H, NSPLIT, D]: the combine then walks one head's
            # splits as a single contiguous run instead of striding by H*D.
            empty = end <= start
            tl.store(PM + b * spm_b + qh * spm_h + pid_s,
                     tl.where(empty, -1e30, m_i), mask=hmask)
            tl.store(PL + b * spm_b + qh * spm_h + pid_s,
                     tl.where(empty, 0.0, l_i), mask=hmask)
            tl.store(PO + b * spo_b + qh[:, None] * spo_h + pid_s * D
                     + offs_d[None, :], acc.to(PO.dtype.element_ty),
                     mask=hmask[:, None])

    @triton.jit
    def _flash_decode_combine(
        PO, PM, PL, OUT, nsplit,
        so_b, so_h, spo_b, spo_h, spm_b, spm_h,
        D: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_S: tl.constexpr,
        MAX_S: tl.constexpr,
    ):
        """Merge the per-split (m, l, O) partials for one (request, head, D slice).

        Slicing the head dim across CTAs, on top of (b, h), is what makes this
        cheap.  The split kernel needs a couple of hundred CTAs to reach peak
        bandwidth, so it leaves that many partials per head; a (b, h)-only grid
        is then 16 CTAs each chasing ~280 dependent 512 B loads, which measured
        3x slower than this shape.
        """
        b = tl.program_id(0)
        h = tl.program_id(1)
        offs_d = tl.program_id(2) * BLOCK_D + tl.arange(0, BLOCK_D)
        pm_p = PM + b * spm_b + h * spm_h
        pl_p = PL + b * spm_b + h * spm_h
        po_p = PO + b * spo_b + h * spo_h

        # m and l are one float per split: read them all at once and get the
        # global max / denominator without touching the wide O partials.
        offs_a = tl.arange(0, MAX_S)
        amask = offs_a < nsplit
        m_v = tl.load(pm_p + offs_a, mask=amask, other=-1e30)
        l_v = tl.load(pl_p + offs_a, mask=amask, other=0.0)
        m_g = tl.max(m_v)
        den = tl.sum(tl.exp(m_v - m_g) * l_v)

        acc = tl.zeros([BLOCK_D], tl.float32)
        for s0 in range(0, nsplit, BLOCK_S):
            offs_s = s0 + tl.arange(0, BLOCK_S)
            smask = offs_s < nsplit
            w = tl.exp(tl.load(pm_p + offs_s, mask=smask, other=-1e30) - m_g)
            po = tl.load(po_p + offs_s[:, None] * D + offs_d[None, :],
                         mask=smask[:, None], other=0.0)
            acc += tl.sum(w[:, None] * po, 0)
        o = acc / tl.where(den > 0, den, 1.0)
        tl.store(OUT + b * so_b + h * so_h + offs_d, o.to(OUT.dtype.element_ty))


class TRTLLMDecode(nn.Module):
    # Fast-path tuning, hardcoded rather than autotuned so the score cannot move
    # with a re-selected config. The split kernel has no batch dimension to
    # parallelise over at batch 1, so the KV split is its only source of
    # occupancy -- and it is register/SMEM limited to 2 CTAs per SM, which is
    # what TARGET_CTAS targets.
    FAST_MAX_BATCH = 8
    TARGET_CTAS = 296         # 2 CTAs/SM x 148 SMs
    MAX_SPLITS = 296          # nsplit is capped by TARGET_CTAS anyway
    BLOCK_N = 64              # tokens per mainloop slab (4 pages at page_size 16)
    NUM_WARPS = 2
    NUM_STAGES = 6
    MIN_CTX = 32768           # fall through to trtllm-gen below this bound
    C_BLOCK_D = 32            # combine: head-dim slice per CTA
    C_BLOCK_S = 64            # combine: splits per tile
    C_WARPS = 1
    C_STAGES = 2

    def __init__(self, num_qo_heads: int, num_kv_heads: int, head_dim: int,
                 workspace: torch.Tensor | None = None):
        super().__init__()
        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.sm_scale = head_dim ** -0.5
        if workspace is None:
            workspace = torch.zeros(
                512 * 1024 * 1024, dtype=torch.uint8, device="cuda"
            )
        self._workspace = workspace
        self._sinks_fp32: torch.Tensor | None = None
        self._sinks_src: torch.Tensor | None = None
        # Per-call invariants, resolved once. ``backend='auto'`` in the public
        # API re-reads the compute capability on every call only to land on
        # trtllm-gen for sm_100/sm_103; this module is Blackwell-only, so pin it.
        self._run = None
        self._ws_bytes = workspace.numel() * workspace.element_size()
        self._sm_count = 0
        self._pdl = False
        self._dev = -2
        if workspace.is_cuda:
            self._resolve(workspace.device)
        # Split-K scratch, allocated once so ``forward`` never mallocs. Only the
        # head geometry the fast path accepts gets a buffer.
        self._h_per_kv = num_qo_heads // max(1, num_kv_heads)
        self._po = self._pm = self._pl = None
        if (_HAVE_TRITON and workspace.is_cuda and self._h_per_kv >= 16
                and self._h_per_kv & (self._h_per_kv - 1) == 0):
            dev = workspace.device
            n = self.FAST_MAX_BATCH * self.MAX_SPLITS
            self._po = torch.empty(n * num_qo_heads * head_dim,
                                   dtype=torch.float32, device=dev)
            self._pm = torch.empty(n * num_qo_heads, dtype=torch.float32, device=dev)
            self._pl = torch.empty(n * num_qo_heads, dtype=torch.float32, device=dev)

    def _resolve(self, device: torch.device) -> None:
        """Bind the trtllm-gen op and the device-derived launch constants."""
        self._run = get_trtllm_gen_fmha_module().trtllm_paged_attention_decode
        self._sm_count = get_device_sm_count(device)
        self._pdl = device_support_pdl(device)
        self._dev = device.index if device.index is not None else 0

    def prime_sinks(self, sinks: torch.Tensor | None) -> None:
        prime_trtllm_sinks(self, sinks)

    # -- hand-written small-batch path ------------------------------------
    def _fast_ok(self, q, k_cache, v_cache, block_table, cache_seqlens,
                 s_aux, window_size) -> bool:
        """Host-side predicate: shapes, dtypes and flags only -- no device reads."""
        if self._po is None or s_aux is not None:
            return False
        if window_size is not None and window_size[0] >= 0:
            return False
        if cache_seqlens is None or block_table is None:
            return False
        b, h, d = q.shape
        if (b > self.FAST_MAX_BATCH or h != self.num_qo_heads
                or d != self.head_dim):
            return False
        if q.dtype != torch.bfloat16 or k_cache.dtype != torch.bfloat16:
            return False
        if v_cache.dtype != torch.bfloat16 or k_cache.dim() != 4:
            return False
        if k_cache.shape[1] != self.num_kv_heads or k_cache.shape[3] != d:
            return False
        if v_cache.shape != k_cache.shape:
            return False
        page = k_cache.shape[2]
        if page & (page - 1) or page < 16:
            return False
        # Innermost dim must be unit-stride for the vector loads.
        return (q.stride(2) == 1 and k_cache.stride(3) == 1
                and v_cache.stride(3) == 1 and block_table.stride(1) == 1
                and cache_seqlens.stride(0) == 1)

    def _fast_decode(self, q, k_cache, v_cache, cache_seqlens, block_table,
                     scale, bound):
        b, h, d = q.shape
        nkv = self.num_kv_heads
        page = k_cache.shape[2]
        bn = max(self.BLOCK_N, page)
        n_tiles = max(1, -(-int(bound) // bn))
        n_prog = b * nkv
        nsplit = min(n_tiles, max(1, self.TARGET_CTAS // n_prog), self.MAX_SPLITS)
        tiles_per_split = -(-n_tiles // nsplit)
        nsplit = min(nsplit, -(-n_tiles // tiles_per_split))
        tokens_per_split = -(-n_tiles // nsplit) * bn

        out = torch.empty_like(q)
        po = pm = pl = out
        spo_b = spo_h = spm_b = spm_h = 0
        if nsplit > 1:
            po = self._po.view(self.FAST_MAX_BATCH, h, self.MAX_SPLITS, d)
            pm = self._pm.view(self.FAST_MAX_BATCH, h, self.MAX_SPLITS)
            pl = self._pl.view(self.FAST_MAX_BATCH, h, self.MAX_SPLITS)
            spo_b, spo_h = po.stride(0), po.stride(1)
            spm_b, spm_h = pm.stride(0), pm.stride(1)

        _flash_decode_split[(nsplit, n_prog)](
            q, k_cache, v_cache, block_table, cache_seqlens, out, po, pm, pl,
            scale,
            q.stride(0), q.stride(1),
            k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
            v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
            block_table.stride(0),
            out.stride(0), out.stride(1),
            spo_b, spo_h, spm_b, spm_h,
            tokens_per_split, nsplit,
            NKV=nkv, H_PER_KV=self._h_per_kv, BLOCK_H=self._h_per_kv,
            D=d, BLOCK_N=bn, PAGE=page, WRITE_FINAL=(nsplit == 1),
            I64=(k_cache.numel() > 2 ** 31 or v_cache.numel() > 2 ** 31),
            num_warps=self.NUM_WARPS, num_stages=self.NUM_STAGES,
        )
        if nsplit > 1:
            bd = min(self.C_BLOCK_D, d)
            _flash_decode_combine[(b, h, d // bd)](
                po, pm, pl, out, nsplit,
                out.stride(0), out.stride(1), spo_b, spo_h, spm_b, spm_h,
                D=d, BLOCK_D=bd, BLOCK_S=self.C_BLOCK_S,
                MAX_S=triton.next_power_of_2(nsplit),
                num_warps=self.C_WARPS, num_stages=self.C_STAGES,
            )
        return out

    def forward(self, q, k_cache, v_cache, cache_seqlens=None,
                block_table=None, softmax_scale=None, causal=True,
                max_seq_len=None, s_aux=None, window_size=None, **kwargs):
        # trtllm-gen requires a contiguous query: with a batched (multi-request)
        # decode, the query view is non-contiguous and the TMA load reads later
        # rows at the wrong stride -> only row 0 is correct, the rest are garbage.
        # vLLM's FlashInfer backend and our own TRTLLMPrefill both do this; the
        # decode path was missing it.
        #
        # block_tables / seq_lens MUST be contiguous too: the trtllm-gen kernel
        # reads the page table assuming a dense [batch, max_pages] row-major
        # layout. The engine's eager/CUDA-graph decode buffers hand us a column
        # slice (``_eager_block_tables[:n, :bt_cols]``) whose row stride is the
        # full ``max_num_blocks``, not ``bt_cols`` -> every row > 0 would read
        # its page ids from the wrong offset (garbage pages), so only row 0
        # stayed correct and all other sequences in the batch were corrupted.
        # vLLM likewise asserts is_strictly_contiguous(block_tables/seq_lens).
        #
        # The engine's steady-state buffers are already dense, so gate each copy
        # on ``is_contiguous()`` -- an unconditional ``.contiguous()`` still pays
        # a dispatcher round trip per call to hand back the same tensor.
        if not q.is_contiguous():
            q = q.contiguous()
        if not block_table.is_contiguous():
            block_table = block_table.contiguous()
        if cache_seqlens is not None and not cache_seqlens.is_contiguous():
            cache_seqlens = cache_seqlens.contiguous()
        if max_seq_len is None:
            # ``max_seq_len`` only feeds trtllm-gen's launch heuristics (CTAs per
            # KV sequence and tileSizeQ); the mainloop bounds itself with the
            # per-request ``seq_lens``, so any upper bound is numerically exact.
            # Deriving it from the page table is that bound -- and it avoids the
            # ``cache_seqlens.max().item()`` device-to-host sync the generic path
            # does, which drains the whole pipeline once per decode step and
            # bars CUDA-graph capture outright.
            max_seq_len = block_table.size(-1) * k_cache.size(-2)
        if self._dev != q.get_device():
            self._resolve(q.device)
        scale = self.sm_scale if softmax_scale is None else softmax_scale
        if (max_seq_len >= self.MIN_CTX
                and self._fast_ok(q, k_cache, v_cache, block_table,
                                  cache_seqlens, s_aux, window_size)):
            return self._fast_decode(q, k_cache, v_cache, cache_seqlens,
                                     block_table, scale, max_seq_len)
        # Attention sinks and the sliding window must be forwarded explicitly.
        # The caller names them ``s_aux`` / ``window_size`` (the FlashAttention
        # spelling); trtllm-gen calls them ``sinks`` / ``window_left``. Letting
        # them fall into **kwargs silently dropped both, which is a *numerical*
        # bug, not a crash: gpt-oss-120b (sinks + alternating sliding window)
        # scored 0.8 of 385 matching tokens against vLLM. vLLM passes both here
        # (flashinfer.py: window_left=self.window_left, sinks=self.sinks).
        out = torch.empty_like(q)
        self._run(
            out,
            None,                                       # out_scale_factor
            q,
            k_cache,                                    # HND: [pages, H, page, D]
            v_cache,
            self._workspace,
            block_table,
            cache_seqlens,
            1,                                          # max_q_len (q_len_per_req)
            max_seq_len,
            scale,
            1.0,                                        # bmm2_scale
            -1.0,                                       # o_sf_scale (unused)
            -1,                                         # o_sf_vec_size (unused)
            0,                                          # o_sf_start_index
            q.size(0),                                  # batch_size
            window_size[0] if window_size is not None and window_size[0] >= 0 else -1,
            0,                                          # sparse_mla_top_k
            self._sm_count,
            self._pdl,
            self._ws_bytes,
            trtllm_sinks(self, s_aux),
            None,                                       # cum_seq_lens_q
            None,                                       # key_block_scales
            None,                                       # value_block_scales
            None,                                       # skip_softmax threshold
            True,                                       # uses_shared_paged_kv_idx
            None,                                       # lse
            0,                                          # lse_stride_tokens
            0,                                          # lse_stride_heads
        )
        return out
