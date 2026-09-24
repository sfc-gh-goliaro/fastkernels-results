"""TRTLLM-gen paged attention prefill kernel (via FlashInfer, Blackwell only).

Accepts the same cu_seqlens-based interface as FlashAttnPrefill so that
LlamaAttention can dispatch to either backend without branch logic.

``trtllm_batch_context_with_kv_cache`` reads the page table directly and is the
default for any paged call, and it is the numerical anchor every other path here
is checked against.

There is a second way to serve a paged call: de-page the KV the batch actually
references into a packed varlen buffer and run *dense* attention over it. Dense
FlashAttention is measurably faster than its own paged form on this device -- the
page-table indirection costs more than the extra KV round-trip, once the context
is long enough to amortise it. A purpose-built gather moves the referenced KV at
several TB/s (tens of microseconds), so on a long-context chunked-prefill shape
the de-paged path wins by over 10%, while on a short shape the two extra kernel
launches lose to trtllm-gen outright. A host-only gate decides, in two stages:
whether the path can represent the input at all, then whether it pays.

Two properties of the de-paged path worth stating up front:

* **It requires every request to have at least one KV token** -- ``cu_seqlens_k``
  strictly increasing -- and *rejects* a call that does not, with a device-side
  assertion rather than a wrong answer. That is a precondition of the calling
  contract, not an extra restriction: vLLM's scheduler refuses to schedule a
  request with no new tokens (``v1/core/sched/scheduler.py``, ``assert
  num_new_tokens > 0``, with the ``num_new_tokens == 0`` branches skipping the
  request), the per-request KV length is then
  ``num_computed_tokens + num_scheduled_tokens``
  (``v1/worker/gpu_model_runner.py``, ``self.seq_lens[:num_reqs] = ...``), and that
  vector is what reaches this operator (``v1/attention/backends/flashinfer.py``,
  ``prefill_seq_lens = seq_lens[prefill_start:]``). So every scheduled request has
  ``seq_lens >= num_scheduled_tokens >= 1``. Rejecting is right rather than merely
  convenient because the paged kernel does not define this case either: it never
  writes an empty request's output rows, so the caller reads whatever the output
  allocation happened to contain, and there is no value to agree with.
* The profitability test estimates work as ``T_q * max_seqlen_k * H_q``, which
  overstates it for a batch whose requests differ wildly in length -- the true
  per-request pair count is a device value. Every configuration the gate admits is
  measured end to end against the paged path over the whole captured population
  (``tools/validate_depaged_population.py``), so the estimate's looseness is
  bounded by evidence rather than by argument.
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl
from flashinfer.prefill import trtllm_batch_context_with_kv_cache

from ....infra.fa_utils import FA_VERSION, flash_attn_varlen_func
from .flashinfer_decode import prime_trtllm_sinks, trtllm_sinks

# One program handles BLOCK_T logical KV tokens of one (request, kv head). At 16
# it maps to exactly one physical page, so each program does one contiguous
# page-sized read per tensor; larger blocks span several pages and trade more
# page-table loads for fewer programs. 32 measured fastest on the long-context
# shape this path is gated to, and the gather is well under 1% of that shape's
# total anyway.
_GATHER_BLOCK_T = 32

# Page size the gather is written for. Every captured configuration uses 16, and
# a different one changes the read granularity enough that it should be measured
# rather than assumed, so the gate rejects it instead.
_PAGE_SIZE = 16

# Head dims with measured dense-vs-paged numerics. 256 is a real captured
# configuration but has no evidence here, so it stays on the paged path.
_DEPAGED_HEAD_DIMS = (64, 128)

# Absolute cap on the packed k+v buffer. The packed length must be bounded
# host-side by ``batch * max_seqlen_k`` (the true length lives on the device),
# and that bound is loose for a ragged batch, so cap it in bytes rather than
# trusting the shape. Insurance for arbitrary inputs: the largest captured
# configuration needs well under 1 GiB.
_PACK_BUDGET_BYTES = 1536 * 1024 * 1024

# The context must be this much longer than the query block before de-paging is
# considered: the signature of chunked prefill / prefix cache, where dense
# attention's advantage is large enough to pay for the gather.
_CONTEXT_GROWTH = 2.0

# Minimum attention work (query tokens x context x query heads) for the two extra
# kernel launches to amortise. Below it, trtllm-gen's fixed ~20us cost is
# unbeatable by anything that gathers first.
_WORK_FLOOR = 1e9

# Minimum per-request query block. Running every captured configuration the gate
# admits (tools/validate_depaged_population.py) split cleanly on this and nothing
# else: with max_seqlen_q >= 2048 all 55 admitted cases land between 1.088x and
# 1.174x, while the six cases below it scatter from 0.881x to 1.271x -- one of them
# an outright loss -- and no host-known quantity distinguishes the winners from the
# loser there. The observed max_seqlen_q values jump straight from 496 to 4096, so
# this floor sits with ~4x margin on both sides rather than being fitted to the
# case that failed. Giving up four small-block wins to remove one loss is the
# trade this gate is supposed to make.
_MIN_QUERY_BLOCK = 2048


@triton.jit
def _gather_paged_kv_kernel(
    k_ptr, v_ptr, out_k_ptr, out_v_ptr, block_table_ptr, cu_seqlens_k_ptr,
    k_stride_page, k_stride_head, k_stride_slot,
    v_stride_page, v_stride_head, v_stride_slot,
    bt_stride_row,
    out_stride_token, out_stride_head,
    PAGE_SIZE: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_T: tl.constexpr,
):
    """Pack the KV a batch of requests actually references into varlen rows.

    Row ``cu_seqlens_k[b] + t`` of the output holds
    ``kv[block_table[b, t // PAGE_SIZE], h, t % PAGE_SIZE, :]``, i.e. exactly the
    layout dense varlen attention expects, with no page indirection left.

    The per-request bounds are read from ``cu_seqlens_k`` *inside* the kernel, so
    the launch needs only host-known sizes and the caller never has to copy a
    device length back to the host to size or mask this.
    """
    pid_t = tl.program_id(0)
    b = tl.program_id(1)
    h = tl.program_id(2)

    start = tl.load(cu_seqlens_k_ptr + b).to(tl.int64)
    end = tl.load(cu_seqlens_k_ptr + b + 1).to(tl.int64)

    t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    mask = t < (end - start)

    # Duplicate physical page ids are normal (a page table may map two logical
    # pages onto one physical page); this is a pure read, so they need no care
    # beyond loading the id per token rather than per block.
    #
    # int64 throughout the address arithmetic: a page id times a page stride
    # overflows int32 for caches this operator legitimately accepts. At
    # H_kv=8, D=128 the page stride is 16384, so page id 131072 is already past
    # 2^31, and nothing in the gate bounds the *source* cache -- the packed-byte
    # budget only bounds the destination.
    page = tl.load(block_table_ptr + b.to(tl.int64) * bt_stride_row
                   + t // PAGE_SIZE, mask=mask, other=0).to(tl.int64)
    slot = (t % PAGE_SIZE).to(tl.int64)

    d = tl.arange(0, HEAD_DIM).to(tl.int64)
    src = (page[:, None] * k_stride_page + h.to(tl.int64) * k_stride_head
           + slot[:, None] * k_stride_slot + d[None, :])
    src_v = (page[:, None] * v_stride_page + h.to(tl.int64) * v_stride_head
             + slot[:, None] * v_stride_slot + d[None, :])
    dst = ((start + t)[:, None] * out_stride_token
           + h.to(tl.int64) * out_stride_head + d[None, :])

    tl.store(out_k_ptr + dst, tl.load(k_ptr + src, mask=mask[:, None]),
             mask=mask[:, None])
    tl.store(out_v_ptr + dst, tl.load(v_ptr + src_v, mask=mask[:, None]),
             mask=mask[:, None])


@triton.jit
def _zero_packed_tail_kernel(
    out_k_ptr, out_v_ptr, cu_seqlens_k_ptr, batch, n_rows,
    out_stride_token, out_stride_head,
    HEAD_DIM: tl.constexpr, BLOCK_T: tl.constexpr,
):
    """Zero the packed buffer's rows past ``cu_seqlens_k[batch]``.

    The gather writes only rows a request owns, so a buffer kept across calls
    still holds the previous call's KV in the rows this call does not reach.
    Dense attention reads whole KV tiles and so touches some of them, and while a
    finite leftover is masked out exactly, a NaN or Inf one is not -- which would
    let a call with perfectly finite inputs inherit a NaN from an earlier, longer
    call. Clearing the tail makes that impossible instead of making it the
    caller's problem.

    The boundary is a device value, so the grid is sized from the host-known row
    count and the start offset is applied here; programs that fall past the end
    mask off entirely.
    """
    pid = tl.program_id(0)
    h = tl.program_id(1)
    packed = tl.load(cu_seqlens_k_ptr + batch).to(tl.int64)
    row = packed + pid * BLOCK_T + tl.arange(0, BLOCK_T)
    mask = row < n_rows
    d = tl.arange(0, HEAD_DIM).to(tl.int64)
    off = (row[:, None] * out_stride_token
           + h.to(tl.int64) * out_stride_head + d[None, :])
    zero = tl.zeros((BLOCK_T, HEAD_DIM), dtype=out_k_ptr.dtype.element_ty)
    tl.store(out_k_ptr + off, zero, mask=mask[:, None])
    tl.store(out_v_ptr + off, zero, mask=mask[:, None])


# The tail sweep touches at most a handful of real rows, so it is launch-bound;
# a wide block keeps the grid small.
_ZERO_TAIL_BLOCK_T = 512


def gather_paged_kv(k, v, block_table, cu_seqlens_k, max_seqlen_k, out,
                    block_t: int = _GATHER_BLOCK_T):
    """De-page ``k``/``v`` into ``out`` (``[2, T_ub, H_kv, D]``), returning the
    packed ``(k_pack, v_pack)`` views dense varlen attention can consume.

    ``out`` may be longer than the packed context. Those trailing rows are
    cleared rather than left alone: the attention kernel bounds its KV *scores*
    by ``cu_seqlens_k`` but still reads the tile that straddles the end, so their
    contents have to be finite.

    Assumes, as the operator's contract, that ``cu_seqlens_k`` is non-decreasing
    with ``cu_seqlens_k[-1] <= out.shape[1]``, that every page id in
    ``block_table`` is a valid index into ``k``, and that
    ``block_table.shape[1] * PAGE_SIZE`` covers every request's context. None of
    those can be checked host-side without reading device memory, and the paged
    kernel makes the same assumptions.
    """
    _, h_kv, page_size, head_dim = k.shape
    batch = cu_seqlens_k.numel() - 1
    k_pack, v_pack = out[0], out[1]
    n_rows = k_pack.shape[0]
    _gather_paged_kv_kernel[(triton.cdiv(max_seqlen_k, block_t), batch, h_kv)](
        k, v, k_pack, v_pack, block_table, cu_seqlens_k,
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        block_table.stride(0),
        k_pack.stride(0), k_pack.stride(1),
        PAGE_SIZE=page_size, HEAD_DIM=head_dim, BLOCK_T=block_t,
    )
    _zero_packed_tail_kernel[
        (triton.cdiv(n_rows, _ZERO_TAIL_BLOCK_T), h_kv)](
        k_pack, v_pack, cu_seqlens_k, batch, n_rows,
        k_pack.stride(0), k_pack.stride(1),
        HEAD_DIM=head_dim, BLOCK_T=_ZERO_TAIL_BLOCK_T,
    )
    return k_pack, v_pack


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
        # (device, stream, h_kv, head_dim, dtype) -> packed scratch buffer.
        self._kv_pack: dict[tuple, torch.Tensor] = {}

    def prime_sinks(self, sinks: torch.Tensor | None) -> None:
        prime_trtllm_sinks(self, sinks)

    def forward(self, q, k, v, cu_seqlens_q, cu_seqlens_k,
                max_seqlen_q, max_seqlen_k, softmax_scale=None,
                causal=True, block_table=None, s_aux=None,
                window_size=None, **kwargs):
        if block_table is None:
            return self._dense_fallback(
                q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                softmax_scale, causal, s_aux, window_size,
            )
        # Both gate stages read shapes, dtypes, strides and devices only. Nothing
        # here touches a device value, because a device->host copy would
        # serialise against the work already queued on the stream and cost far
        # more than the path it is choosing between could save.
        if (self._depaged_eligible(q, k, v, cu_seqlens_q, cu_seqlens_k,
                                   softmax_scale, causal, block_table, s_aux,
                                   window_size)
                and self._depaged_worthwhile(q, k, cu_seqlens_k,
                                             max_seqlen_q, max_seqlen_k)):
            return self._depaged_dense(
                q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                softmax_scale, block_table, s_aux,
            )
        return self._paged_context(
            q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
            softmax_scale, block_table, s_aux, window_size,
        )

    def _depaged_eligible(self, q, k, v, cu_seqlens_q, cu_seqlens_k,
                          softmax_scale, causal, block_table, s_aux,
                          window_size) -> bool:
        """Can the de-paged dense path represent this call at all?

        Purely a correctness question -- never a performance one. Anything this
        rejects is served by the paged path, which is the anchor, so the honest
        default for an untested configuration is to reject it.
        """
        if FA_VERSION != 4:
            return False
        # A scale of exactly zero makes dense attention return NaN here, while
        # trtllm-gen handles the degenerate uniform-attention case correctly.
        # Every non-zero scale down to 1e-12 agrees, so the exclusion is this
        # narrow. A tensor scale is rejected too: comparing it would read a
        # device value.
        if softmax_scale is not None:
            if not isinstance(softmax_scale, (int, float)):
                return False
            if softmax_scale == 0:
                return False
        # A sliding window makes the reachable KV a sliver of the context;
        # gathering the whole thing would be waste even where it is correct.
        if window_size is not None:
            return False
        # The paged branch is unconditionally causal, so a non-causal call must
        # go there to keep the two paths' masking identical.
        if causal is not True:
            return False
        if q.dim() != 3 or k.dim() != 4 or v.shape != k.shape:
            return False
        if k.shape[-2] != _PAGE_SIZE or k.shape[-1] not in _DEPAGED_HEAD_DIMS:
            return False
        # D innermost is what both the gather's vectorised loads and dense
        # attention need; anything else would require repairing a
        # multi-gigabyte cache, which is never worth it.
        if k.stride(-1) != 1 or v.stride(-1) != 1 or q.stride(-1) != 1:
            return False
        # bf16 only. fp16 is representable by both kernels but every comparison
        # behind this path was measured in bf16, and the plan's own risk table
        # leaves fp16 drift open -- so it takes the anchor until measured.
        if k.dtype is not torch.bfloat16:
            return False
        if q.dtype != k.dtype or v.dtype != k.dtype:
            return False
        if cu_seqlens_q.dtype != torch.int32 or cu_seqlens_k.dtype != torch.int32:
            return False
        if not cu_seqlens_q.is_contiguous() or not cu_seqlens_k.is_contiguous():
            return False
        if cu_seqlens_k.numel() < 2 or cu_seqlens_q.numel() != cu_seqlens_k.numel():
            return False
        if block_table.dim() != 2 or block_table.dtype != torch.int32:
            return False
        # The gather reads the page table by its real strides, so a column slice
        # of a wider buffer is fine; a non-unit innermost stride is not.
        if block_table.stride(-1) != 1:
            return False
        if block_table.shape[0] != cu_seqlens_k.numel() - 1:
            return False
        # Sinks reach dense attention under FlashAttention's own dtype rule
        # (bf16); converting here would only hide a caller-side mistake. The test
        # is on the *shape*, not the element count: FA4 hard-asserts a 1-D
        # (num_qo_heads,) sink, so a (1, num_qo_heads) tensor has the right numel
        # and still raises inside the kernel, where the paged path would have
        # accepted it.
        if s_aux is not None and (
                s_aux.dtype != torch.bfloat16
                or s_aux.dim() != 1
                or s_aux.shape[0] != self.num_qo_heads
                or s_aux.stride(0) != 1):
            return False
        dev = k.device
        if any(t.device != dev for t in (q, v, cu_seqlens_q, cu_seqlens_k,
                                         block_table)):
            return False
        return not (s_aux is not None and s_aux.device != dev)

    def _depaged_worthwhile(self, q, k, cu_seqlens_k,
                            max_seqlen_q, max_seqlen_k) -> bool:
        """Does de-paging pay for itself here? Biased towards the paged path.

        Deliberately a narrow specialisation with wide margins, not a general
        cost model: the exact per-request KV lengths live on the device, so no
        host-side rule can be general without the synchronisation this path
        exists to avoid. Every term is host-known.
        """
        batch = cu_seqlens_k.numel() - 1
        h_kv, head_dim = k.shape[1], k.shape[3]
        # Upper bound on packed rows: each request contributes at most
        # max_seqlen_k. Loose for a ragged batch, which is what the byte cap is
        # for.
        pack_bytes = 2 * batch * max_seqlen_k * h_kv * head_dim * k.element_size()
        if pack_bytes > _PACK_BUDGET_BYTES:
            return False
        if max_seqlen_k < _CONTEXT_GROWTH * max_seqlen_q:
            return False
        # A long query block per request, not merely a lot of query tokens in
        # total: the dense kernel's advantage is only reliable in that regime.
        if max_seqlen_q < _MIN_QUERY_BLOCK:
            return False
        return q.shape[0] * max_seqlen_k * q.shape[1] >= _WORK_FLOOR

    def _packed_kv_buffer(self, t_ub, h_kv, head_dim, dtype, device):
        """Packed k/v scratch of at least ``t_ub`` rows, private to this stream.

        Keyed by ``(device, current stream)``, not just by the instance. Work
        issued on one stream is ordered against itself, so reusing a buffer across
        calls on the same stream is safe; two calls on *different* streams have no
        such ordering, and a single per-instance buffer would let one call's gather
        overwrite rows the other's attention is still reading. Rewriting the buffer
        on every call is no defence when the rewrites overlap.

        Zero-initialised, never ``torch.empty``: dense attention reads whole KV
        tiles, so the tile straddling the end of the packed context also reads
        rows past it. A finite value there is masked out exactly -- measured, and
        true even for values as large as 1e4 -- but a NaN or Inf is already in
        the QK product before the mask applies and propagates into the output,
        and uninitialised memory can hold either. The gather clears the trailing
        rows on every call as well, so a buffer grown by one call cannot hand a
        stale non-finite row to a later, shorter one.
        """
        # Reading the current stream is a host-side query on the caller's own
        # stream object; it queues nothing and synchronises nothing.
        key = (device.type, device.index, torch.cuda.current_stream(device).stream_id,
               h_kv, head_dim, dtype)
        buf = self._kv_pack.get(key)
        if buf is None or buf.shape[1] < t_ub:
            buf = torch.zeros(2, t_ub, h_kv, head_dim,
                              dtype=dtype, device=device)
            self._kv_pack[key] = buf
        return buf[:, :t_ub]

    def _depaged_dense(self, q, k, v, cu_seqlens_q, cu_seqlens_k,
                       max_seqlen_q, max_seqlen_k, softmax_scale,
                       block_table, s_aux):
        batch = cu_seqlens_k.numel() - 1
        h_kv, head_dim = k.shape[1], k.shape[3]
        # Every request must have at least one KV token. This is not a
        # conservatism: with an empty request the paged kernel leaves that
        # request's output rows *unwritten*, so the caller reads whatever was in
        # the freshly allocated output tensor -- measured by writing 3.5 into the
        # allocator and reading 3.5 back out of those rows. There is no value for
        # this path to agree with, and dense attention would instead return a
        # confidently wrong number, so the input is rejected rather than served.
        # Callers do not generate it: a prefill request with no context cannot
        # occur upstream, and the benchmark's own input builder clamps every KV
        # length to at least 1.
        #
        # The lengths live on the device, so the check is queued as device work
        # and never read back here; a violation faults the stream instead of
        # silently returning the wrong answer.
        torch._assert_async(
            torch.all(cu_seqlens_k[1:] > cu_seqlens_k[:-1]),
            "TRTLLMPrefill: the de-paged path requires every request to have at "
            "least one KV token (cu_seqlens_k must be strictly increasing)",
        )
        out = self._packed_kv_buffer(batch * max_seqlen_k, h_kv, head_dim,
                                     k.dtype, k.device)
        # Every row dense attention can read is written from this call's k, v and
        # block_table: the requests' own rows by the gather, the trailing rows by
        # the tail sweep. No part of a previous call's KV survives into this one.
        k_pack, v_pack = gather_paged_kv(k, v, block_table, cu_seqlens_k,
                                         max_seqlen_k, out, _GATHER_BLOCK_T)
        fa_extra = {}
        if s_aux is not None:
            fa_extra["s_aux"] = s_aux
        return flash_attn_varlen_func(
            q, k_pack, v_pack,
            # cu_seqlens_k is already the packed buffer's offset vector -- the
            # gather writes request b at row cu_seqlens_k[b] precisely so no
            # second metadata tensor is needed.
            cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale if softmax_scale is not None else self.sm_scale,
            # Matches the paged branch, which is causal regardless of the flag,
            # and FA4's bottom-right alignment agrees with trtllm-gen's.
            causal=True,
            fa_version=FA_VERSION,
            # Same reason as the dense fallback: the auto heuristic picks a
            # split-KV kernel that fails to compile in this build.
            num_splits=1,
            **fa_extra,
        )

    def _paged_context(self, q, k, v, cu_seqlens_q, cu_seqlens_k,
                       max_seqlen_q, max_seqlen_k, softmax_scale,
                       block_table, s_aux, window_size):
        q = q.contiguous()
        seq_lens = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
        batch_size = seq_lens.shape[0]
        # trtllm-gen reads the page table as a dense row-major tensor; a
        # non-contiguous block_table (e.g. a column slice of a wider buffer)
        # makes every row > 0 read wrong page ids. Match vLLM, which asserts
        # is_strictly_contiguous here. See TRTLLMDecode for the full story.
        block_table = block_table.contiguous()
        seq_lens = seq_lens.contiguous()
        return trtllm_batch_context_with_kv_cache(
            query=q,
            kv_cache=(k, v),
            workspace_buffer=self._workspace,
            block_tables=block_table,
            seq_lens=seq_lens,
            max_q_len=max_seqlen_q,
            max_kv_len=max_seqlen_k,
            bmm1_scale=softmax_scale if softmax_scale is not None else self.sm_scale,
            bmm2_scale=1.0,
            batch_size=batch_size,
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

    def _dense_fallback(self, q, k, v, cu_seqlens_q, cu_seqlens_k,
                        max_seqlen_q, max_seqlen_k, softmax_scale,
                        causal, s_aux, window_size):
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
