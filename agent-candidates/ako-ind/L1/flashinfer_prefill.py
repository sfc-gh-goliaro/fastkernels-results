"""TRTLLM-gen paged attention prefill kernel (via FlashInfer, Blackwell only).

Accepts the same cu_seqlens-based interface as FlashAttnPrefill so that
LlamaAttention can dispatch to either backend without branch logic.

Backend routing
---------------
trtllm-gen is the fastest paged-prefill kernel here for the smallest calls, but
its SM100 kernel carries a ~11 us fixed cost -- measured as the intercept of a
two-point fit over the 487- and 804-token cases (0.97 GFLOP -> 13.2 us,
2.65 GFLOP -> 16.9 us, i.e. ~450 TFLOP/s marginal plus 11 us) -- and above that
it leaves real time on the table for two different reasons:

* **Long contexts** lose because with a page size of 16 and a KV tile of 128,
  *every* Blackwell paged-KV kernel falls off TMA onto a cp.async gather (FA4
  spells this ``paged_kv_non_tma = page_size not in [None, tile_n]``).  On a
  32 K-token context it is cheaper to materialize the pages this call actually
  touches (~34 MB) once and hand FA4 a *ragged* problem it runs on the TMA path
  at ~880 TFLOP/s versus trtllm-gen's ~765.

* **Short contexts with a lot of total work** lose to the causal mask.  At
  FA4's SM100-mandated ``tile_m = 128``, a 269-token sequence covers 36 315
  useful key/query pairs with 98 304 MMA slots.  Shrinking ``tile_n`` from 128
  to 64 halves the waste in the triangular tiles, and on the 61x269 case that
  is worth 1.17x on the attention kernel.  ``tile_m = 64`` would be better
  still but is structurally closed: it dies in ``make_tmem_copy`` at both
  head_dim 64 and 128.

So the routing has three arms, keyed on shape metadata only -- head dim, page
size, table width, dtypes, ``max_seqlen_k`` -- never on a value read out of a
tensor, so it costs no device sync and picks the same arm on every iteration:

    work = q_tokens * max_seqlen_k * num_qo_heads
    work <  3e7                        -> trtllm-gen, unchanged
    work >= 3e7, short kv, no sinks    -> FA4 paged, tile_mn = (128, 64)
    work >= 3e7, otherwise             -> page gather + FA4 varlen ragged

Below ~3e7 the gather plus FA4's launch cost more than the kernel saves (0.90x
measured on the 487- and 804-token cases, whose kernels are only 13-17 us to
begin with), and FA4 paged is a wash there too (12.7 vs 13.2 us on the
487-token case) -- not enough to justify a fourth arm.

Per-call overhead
-----------------
The kv sequence lengths trtllm-gen wants are ``cu_seqlens_k`` differences, and
computing them with torch costs a separate 1.4-1.9 us elementwise kernel --
10% of the *whole* forward on the short cases, where the attention kernel is
only 13 us.  Two things remove almost all of it: the ragged arm never needs
them at all (so the computation moved inside the arm that does), and for a
single sequence ``cu_seqlens_k[1:]`` already *is* the length vector, a free
view, because ``cu_seqlens[0]`` is 0 by definition.  Multi-sequence calls fall
back to a one-block Triton kernel, marginally cheaper than torch's.
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl
from flashinfer.prefill import trtllm_batch_context_with_kv_cache

from ....infra.fa_utils import FA_VERSION, flash_attn_varlen_func
from .flashinfer_decode import prime_trtllm_sinks, trtllm_sinks

try:  # FA4's CuTe entry point, for the knobs the vLLM wrapper does not expose
    from vllm.vllm_flash_attn.cute.interface import _flash_attn_fwd
except Exception:  # pragma: no cover - falls back to the wrapper's defaults
    _flash_attn_fwd = None

# q_tokens * max_seqlen_k * num_qo_heads above which FA4 beats trtllm-gen's
# paged kernel.  Measured crossover sits between 1.0e7 (804-token case, 0.92x)
# and 1.4e8 (61x269 case, 1.17x); 3e7 keeps >3x margin either side.
_FA4_WORK_MIN = 3e7

# max_seqlen_k below which FA4's narrow KV tile wins.  tile_n = 64 is 1.04x on a
# 269-token context and 0.65x on a 32 K one, so the cutoff only has to separate
# those two classes.
_NARROW_TILE_MAX_KV = 512


@triton.jit
def _gather_pages(KSRC, VSRC, KDST, VDST, BT, CU, NTD, NPAGE, NPG, SBT,
                  PS: tl.constexpr, D: tl.constexpr, HKPSD: tl.constexpr,
                  BLK: tl.constexpr):
    """Copy the pages named by ``BT`` out of an HND paged cache into ragged
    varlen KV packed at exactly ``cu_seqlens_k`` offsets.

    Program ``p`` owns page slot ``p % NPG`` of sequence ``p // NPG``, read from
    row ``p // NPG`` of the page table at its true row stride ``SBT`` -- taking
    the stride rather than a ``block_table[:, :NPG]`` slice avoids a 1.4-2.7 us
    ``.contiguous()`` copy on any call whose table is wider than the pages it
    uses (which is every call here).  Source page holds ``[num_kv_heads,
    page_size, head_dim]`` contiguously; the destination is head-major
    ``[num_kv_heads, capacity, head_dim]`` (a fixed head stride ``NTD`` lets the
    whole address be shape arithmetic).  One program per (slot, kv head) moves
    one ``page_size * head_dim`` run, contiguous on both sides, so the copy
    streams at bandwidth.

    The tail page of a sequence is truncated to the tokens that exist, and slots
    past a sequence's end drop out entirely -- that is what keeps the segments
    tightly packed so ``cu_seqlens_k`` can be handed to FA4 as-is.  Page ids are
    clamped, not masked: a page table may carry padding entries in those dead
    slots, and clamping is free where a bounds check is not.
    """
    p = tl.program_id(0)
    h = tl.program_id(1).to(tl.int64)
    seq = p // NPG
    slot = p - seq * NPG
    start = tl.load(CU + seq).to(tl.int64) + slot * PS
    n = tl.minimum(tl.load(CU + seq + 1).to(tl.int64) - start, PS)
    if n > 0:
        page = tl.load(BT + seq * SBT + slot).to(tl.int64)
        page = tl.minimum(tl.maximum(page, 0), NPAGE - 1)
        i = tl.arange(0, BLK)
        m = i < n * D
        src = page * HKPSD + h * (PS * D) + i
        dst = h * NTD + start * D + i
        tl.store(KDST + dst, tl.load(KSRC + src, mask=m, other=0), mask=m)
        tl.store(VDST + dst, tl.load(VSRC + src, mask=m, other=0), mask=m)


@triton.jit
def _seq_lens_kernel(CU, OUT, N, BLK: tl.constexpr):
    """``OUT[i] = CU[i + 1] - CU[i]`` for a handful of sequences, in one block."""
    i = tl.arange(0, BLK)
    m = i < N
    tl.store(OUT + i, tl.load(CU + i + 1, mask=m, other=0)
             - tl.load(CU + i, mask=m, other=0), mask=m)


class TRTLLMPrefill(nn.Module):
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
        # Scratch for the ragged path, keyed by buffer shape + dtype. Reusing it
        # is safe because the gather fully rewrites every byte FA4 will read
        # before FA4 runs; it just keeps the allocator off the critical path.
        self._kv_scratch: dict = {}
        self._len_scratch: dict = {}

    def prime_sinks(self, sinks: torch.Tensor | None) -> None:
        prime_trtllm_sinks(self, sinks)

    # -- per-call bookkeeping ------------------------------------------------
    def _seq_lens(self, cu_seqlens_k):
        """kv length per sequence, as cheaply as the shape allows."""
        n = cu_seqlens_k.shape[0] - 1
        if n == 1:
            # cu_seqlens[0] is 0 by definition, so the suffix *is* the lengths:
            # a contiguous view, no kernel at all.
            return cu_seqlens_k[1:]
        key = (n, cu_seqlens_k.dtype, cu_seqlens_k.device)
        buf = self._len_scratch.get(key)
        if buf is None:
            buf = torch.empty(n, dtype=cu_seqlens_k.dtype,
                              device=cu_seqlens_k.device)
            self._len_scratch[key] = buf
        _seq_lens_kernel[(1,)](cu_seqlens_k, buf, n,
                               BLK=triton.next_power_of_2(n), num_warps=1)
        return buf

    # -- FA4 paths ----------------------------------------------------------
    def _fa4_ok(self, q, k, v, max_seqlen_k, block_table, s_aux,
                window_size) -> bool:
        """Shape-only guard: is this call in the class where FA4 beats trtllm?"""
        if FA_VERSION != 4 or window_size is not None:
            return False
        if k.dim() != 4 or v.dim() != 4 or k.shape != v.shape:
            return False
        if not (k.is_contiguous() and v.is_contiguous()):
            return False
        # HND layout, FA4-supported head dim.
        head_dim = k.shape[-1]
        if k.shape[1] != self.num_kv_heads or head_dim != self.head_dim:
            return False
        if head_dim > 128 or head_dim % 8 != 0:
            return False
        if q.dtype != k.dtype or k.dtype not in (torch.bfloat16, torch.float16):
            return False
        # FA4 wants the sink vector in bfloat16, one entry per qo head.
        if s_aux is not None and (s_aux.dtype != torch.bfloat16
                                  or s_aux.shape != (self.num_qo_heads,)):
            return False
        page_size = k.shape[-2]
        n_pages = (max_seqlen_k + page_size - 1) // page_size
        if n_pages > block_table.shape[-1]:
            return False
        work = q.shape[0] * max_seqlen_k * self.num_qo_heads
        return work >= _FA4_WORK_MIN

    def _narrow_tile_ok(self, max_seqlen_k, block_table, s_aux) -> bool:
        """Short-context arm: FA4 paged with a 64-wide KV tile, no gather."""
        return (_flash_attn_fwd is not None and s_aux is None
                and self.head_dim == 128
                and max_seqlen_k <= _NARROW_TILE_MAX_KV
                and block_table.dtype == torch.int32)

    def _ragged_kv(self, k, v, block_table, cu_seqlens_k, n_pages):
        """Materialize the touched pages as ragged varlen KV for FA4's TMA path.

        Returns ``(k_r, v_r)`` packed at ``cu_seqlens_k`` offsets, as transposed
        views of head-major buffers -- that keeps ``stride(-1) == 1``, all FA4
        asks for, while letting the gather write contiguous runs.  The buffers
        are sized to the page-aligned worst case, so their tail is slack that
        FA4 never reads.
        """
        num_pages, num_kv, page_size, head_dim = k.shape
        n_slots = block_table.shape[0] * n_pages
        cap = n_slots * page_size
        key = (num_kv, cap, head_dim, k.dtype)
        buf = self._kv_scratch.get(key)
        if buf is None:
            buf = (torch.empty((num_kv, cap, head_dim), dtype=k.dtype,
                               device=k.device),
                   torch.empty((num_kv, cap, head_dim), dtype=v.dtype,
                               device=v.device))
            self._kv_scratch[key] = buf
        kd, vd = buf
        _gather_pages[(n_slots, num_kv)](
            k, v, kd, vd, block_table, cu_seqlens_k, cap * head_dim, num_pages,
            n_pages, block_table.stride(0),
            PS=page_size, D=head_dim, HKPSD=num_kv * page_size * head_dim,
            BLK=triton.next_power_of_2(page_size * head_dim), num_warps=4,
        )
        return kd.transpose(0, 1), vd.transpose(0, 1)

    def forward(self, q, k, v, cu_seqlens_q, cu_seqlens_k,
                max_seqlen_q, max_seqlen_k, softmax_scale=None,
                causal=True, block_table=None, s_aux=None,
                window_size=None, **kwargs):
        if block_table is not None:
            q = q.contiguous()
            # trtllm-gen reads the page table as a dense row-major tensor; a
            # non-contiguous block_table (e.g. a column slice of a wider buffer)
            # makes every row > 0 read wrong page ids. Match vLLM, which asserts
            # is_strictly_contiguous here. See TRTLLMDecode for the full story.
            block_table = block_table.contiguous()
            scale = softmax_scale if softmax_scale is not None else self.sm_scale
            if self._fa4_ok(q, k, v, max_seqlen_k, block_table, s_aux,
                            window_size):
                if self._narrow_tile_ok(max_seqlen_k, block_table, s_aux):
                    # Short kv: keep the cache paged (the gather would cost more
                    # than the non-TMA page walk saves at this length) and pay
                    # only for a narrower KV tile.  k/v are permuted to FA4's
                    # NHD page layout; that is a view, and head_dim keeps
                    # stride 1, which is FA4's only layout requirement.
                    return _flash_attn_fwd(
                        q, k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3),
                        cu_seqlens_q=cu_seqlens_q,
                        seqused_k=self._seq_lens(cu_seqlens_k),
                        max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
                        page_table=block_table, softmax_scale=scale,
                        causal=True, num_splits=1, tile_mn=(128, 64),
                    )[0]
                page_size = k.shape[-2]
                n_pages = (max_seqlen_k + page_size - 1) // page_size
                k_r, v_r = self._ragged_kv(k, v, block_table, cu_seqlens_k,
                                           n_pages)
                # The gather packs to exactly the caller's cu_seqlens_k, so it
                # goes straight back in and no seqused_k is needed. Causal is
                # hard-coded to match the trtllm-gen branch below, which always
                # runs the causal kernel.
                return flash_attn_varlen_func(
                    q, k_r, v_r,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_k=max_seqlen_k,
                    softmax_scale=scale,
                    causal=True,
                    fa_version=FA_VERSION,
                    # See the dense branch: FA4's split-KV kernel fails to
                    # compile in this vLLM build, and a 32 K-token prefill is
                    # compute-bound anyway.
                    num_splits=1,
                    **({} if s_aux is None else {"s_aux": s_aux}),
                )
            seq_lens = self._seq_lens(cu_seqlens_k)
            return trtllm_batch_context_with_kv_cache(
                query=q,
                kv_cache=(k, v),
                workspace_buffer=self._workspace,
                block_tables=block_table,
                seq_lens=seq_lens,
                max_q_len=max_seqlen_q,
                max_kv_len=max_seqlen_k,
                bmm1_scale=scale,
                bmm2_scale=1.0,
                batch_size=seq_lens.shape[0],
                cum_seq_lens_q=cu_seqlens_q,
                cum_seq_lens_kv=cu_seqlens_k,
                # See TRTLLMDecode: sinks and the sliding window arrive under
                # their FlashAttention names and must be translated, not
                # swallowed by **kwargs -- dropping them corrupts numerics
                # silently.
                window_left=(
                    window_size[0] if window_size is not None
                    and window_size[0] >= 0 else -1
                ),
                sinks=trtllm_sinks(self, s_aux),
                kv_layout="HND",
            )
        # Dense (unpaged) fallback: same FlashAttention build/version vLLM
        # would use for this device.  Sinks/window must be carried across here
        # too, under FlashAttention's own parameter names.
        fa_extra = {}
        if s_aux is not None:
            fa_extra["s_aux"] = s_aux
        if window_size is not None:
            fa_extra["window_size"] = window_size
        return flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale if softmax_scale is not None else self.sm_scale,
            causal=causal,
            fa_version=FA_VERSION,
            # Compute-bound dense prefill gains nothing from KV-splitting, but the
            # FA4 (SM100 CuTe) auto heuristic still picks the split-KV kernel for
            # mid-size seqlens -- which fails to compile in this vLLM build
            # (TYPE_UNSTABLE_JOIN on ``n_block_first``).  Pin the unsplit kernel.
            num_splits=1,
            **fa_extra,
        )
