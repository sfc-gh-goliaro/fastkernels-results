"""Paged attention decode (TRTLLM-gen compatible interface), hand-written.

Drop-in replacement for the FlashInfer ``trtllm_batch_decode_with_kv_cache``
wrapper: a paged bf16 K/V cache in ``HND`` layout
(``[num_pages, num_kv_heads, page_size, head_dim]``), one query token per
request, GQA, optional attention sinks (``s_aux``) and an optional sliding
window.

Implementation: flash-decoding (split-K over the KV sequence) in two Triton
kernels.

* ``_split_kernel`` -- one CTA per ``(request, kv_head, kv_split)``.  It holds
  that request's GQA group of queries (padded to ``BH`` rows so the MMA has a
  legal ``M``), streams ``SPLIT`` KV tokens through an online-softmax loop and
  writes the *normalized* partial context plus its log-sum-exp (base 2).
* ``_reduce_kernel`` -- one CTA per ``(request, qo_head, head_dim chunk)``;
  a single-pass log-sum-exp merge of that request's partials, folding the
  attention sink into the denominator.

A context short enough to need only one split skips the merge entirely: the
split kernel writes the final output, and the GQA group is spread over several
CTAs (grid dim 2) to make up the lost parallelism.

The KV cache dominates: hundreds of GB of traffic for a large batch with long
contexts, against ~10 MB of everything else.  So the split kernel is written as
a pure streaming read -- ``BN`` tokens per step, ``head_dim``-contiguous loads,
32-bit address math, no masking in the full-block loop -- and the split count is
planned to fill the GPU while keeping the partials (``MID``) a negligible
fraction of that traffic.
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl

_LOG2E = 1.4426950408889634

# Launch-planning knobs.  ``_CTA_TARGET`` is the CTA count we aim for: two waves
# of a 148-SM Blackwell part.  A batch of 1000 requests already exceeds it, so
# there the split count is driven by the per-CTA work cap in ``_CFG`` (which
# keeps the sequence-length imbalance of a real batch off the critical path); a
# single-request decode instead grows the split count to reach the target.
_CTA_TARGET = 296
_REDUCE_TARGET = 512
_REDUCE_TILE = 2048
# Per-head_dim inner-loop shape:
#   (KV tokens per step, warps, pipeline stages, register cap, per-CTA token cap)
# The register cap is what makes this kernel stream at the DRAM roofline: left to
# itself ptxas spends ~194 registers per thread on the [BN, head_dim] tile, only
# two CTAs fit per SM, and there is not enough memory parallelism in flight to
# cover HBM latency (measured 5.2 -> 5.9 TB/s on a B200 with the cap).
_CFG = {
    64: (256, 4, 2, 144, 8192),
    128: (128, 4, 2, 160, 16384),
}
_CFG_DEFAULT = (64, 4, 2, 128, 8192)
# A short context is latency-bound, not bandwidth-bound: a narrower tile starts
# faster and splits into more (shorter) CTAs.
_SHORT_LEN = 1024
_CFG_SHORT = (64, 4, 2, None, 8192)
# Cap on the redundant KV traffic the head-split single-kernel path may incur.
_REDUNDANT_BYTES = 8 << 20


def _ceil(a, b):
    return -(-a // b)


@triton.jit
def _split_kernel(
    Q, K, V, BT, SL, MID, LSE,
    sq_b, sq_h,
    sk_b, sk_h, sk_t,
    sbt_b,
    sm_b, sm_h, sm_s,
    sl_b, sl_h,
    qk_scale, n_kv_heads, window_left,
    GQA: tl.constexpr, BH: tl.constexpr, D: tl.constexpr,
    BN: tl.constexpr, SPLIT: tl.constexpr, PAGE: tl.constexpr,
    USE_WIN: tl.constexpr, I32: tl.constexpr, FINAL: tl.constexpr,
    HPC: tl.constexpr,
):
    pid = tl.program_id(0)
    sp = tl.program_id(1)
    b = pid // n_kv_heads
    hk = pid % n_kv_heads
    hr = tl.arange(0, BH)
    hs = tl.program_id(2)
    hmask = (hr < HPC) & (hs * HPC + hr < GQA)
    # ``HPC < GQA`` only in the unsplit (FINAL) path: with a single KV split and
    # a small context there are not enough (request, kv_head) pairs to fill the
    # GPU, so the GQA group is spread over several CTAs instead.  Re-reading the
    # (tiny) KV range per CTA is cheaper than a second kernel launch.
    oh = hk * GQA + hs * HPC + hr

    seqlen = tl.load(SL + b)
    lo = sp * SPLIT
    hi = tl.minimum(lo + SPLIT, seqlen)
    if USE_WIN:
        lo = tl.maximum(lo, seqlen - 1 - window_left)
    d = tl.arange(0, D)
    if lo >= hi:
        if FINAL:
            tl.store(MID + b * sm_b + oh[:, None] * sm_h + d[None, :],
                     tl.zeros((BH, D), tl.float32).to(MID.dtype.element_ty),
                     mask=hmask[:, None])
        return

    q = tl.load(Q + b * sq_b + oh[:, None] * sq_h + d[None, :],
                mask=hmask[:, None], other=0.0)

    m_i = tl.full((BH,), float("-inf"), tl.float32)
    l_i = tl.zeros((BH,), tl.float32)
    acc = tl.zeros((BH, D), tl.float32)

    kbase = K + hk * sk_h
    vbase = V + hk * sk_h
    btbase = BT + b * sbt_b
    # Full BN-token blocks first: predicated loads and the -inf select cost
    # several percent of the streaming rate, so keep them out of the hot loop.
    nfull = ((hi - lo) // BN) * BN
    for s in range(lo, lo + nfull, BN):
        t = s + tl.arange(0, BN)
        pg = tl.load(btbase + t // PAGE)
        tok = t % PAGE
        if I32:
            off = pg[:, None] * sk_b + tok[:, None] * sk_t + d[None, :]
        else:
            off = (pg[:, None].to(tl.int64) * sk_b
                   + tok[:, None] * sk_t + d[None, :])
        k = tl.load(kbase + off)
        qk = tl.dot(q, tl.trans(k)) * qk_scale
        v = tl.load(vbase + off)
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(V.dtype.element_ty), v, acc)
        m_i = m_new
    if lo + nfull < hi:
        t = lo + nfull + tl.arange(0, BN)
        tm = t < hi
        pg = tl.load(btbase + t // PAGE, mask=tm, other=0)
        tok = t % PAGE
        if I32:
            off = pg[:, None] * sk_b + tok[:, None] * sk_t + d[None, :]
        else:
            off = (pg[:, None].to(tl.int64) * sk_b
                   + tok[:, None] * sk_t + d[None, :])
        k = tl.load(kbase + off, mask=tm[:, None], other=0.0)
        qk = tl.dot(q, tl.trans(k)) * qk_scale
        qk = tl.where(tm[None, :], qk, float("-inf"))
        v = tl.load(vbase + off, mask=tm[:, None], other=0.0)
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(V.dtype.element_ty), v, acc)
        m_i = m_new

    acc = acc / l_i[:, None]
    tl.store(MID + b * sm_b + oh[:, None] * sm_h + sp * sm_s + d[None, :],
             acc.to(MID.dtype.element_ty), mask=hmask[:, None])
    if not FINAL:
        tl.store(LSE + b * sl_b + oh * sl_h + sp, m_i + tl.math.log2(l_i),
                 mask=hmask)


@triton.jit
def _reduce_kernel(
    MID, LSE, OUT, SL, SINK,
    sm_b, sm_h, sm_s,
    sl_b, sl_h,
    so_b, so_h,
    window_left, log2e,
    BD: tl.constexpr, BS: tl.constexpr, SPLIT: tl.constexpr, NS: tl.constexpr,
    HAS_SINK: tl.constexpr, USE_WIN: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    dc = tl.program_id(2)
    seqlen = tl.load(SL + b)
    d = dc * BD + tl.arange(0, BD)
    obase = OUT + b * so_b + h * so_h
    # Clamp to the allocated split count: a caller that under-reports
    # ``max_seq_len`` must not send this kernel off the end of MID/LSE.
    ns = tl.minimum(tl.cdiv(seqlen, SPLIT), NS)
    s0 = 0
    if USE_WIN:
        s0 = tl.maximum(0, seqlen - 1 - window_left) // SPLIT
    if ns <= s0:
        tl.store(obase + d, tl.zeros((BD,), tl.float32).to(OUT.dtype.element_ty))
        return

    lbase = LSE + b * sl_b + h * sl_h
    mbase = MID + b * sm_b + h * sm_h
    m_i = float("-inf")
    den = 0.0
    acc = tl.zeros((BD,), tl.float32)
    # Single pass: rescale the running (acc, den) whenever a later split raises
    # the max, exactly as the online softmax in the split kernel does.
    for s in range(s0, ns, BS):
        sp = s + tl.arange(0, BS)
        vm = sp < ns
        lv = tl.load(lbase + sp, mask=vm, other=float("-inf"))
        m_new = tl.maximum(m_i, tl.max(lv))
        alpha = tl.math.exp2(m_i - m_new)
        w = tl.math.exp2(lv - m_new)
        mid = tl.load(mbase + sp[:, None] * sm_s + d[None, :], mask=vm[:, None],
                      other=0.0)
        acc = acc * alpha + tl.sum(w[:, None] * mid.to(tl.float32), 0)
        den = den * alpha + tl.sum(w)
        m_i = m_new
    if HAS_SINK:
        sink = tl.load(SINK + h).to(tl.float32) * log2e
        m_new = tl.maximum(m_i, sink)
        alpha = tl.math.exp2(m_i - m_new)
        acc = acc * alpha
        den = den * alpha + tl.math.exp2(sink - m_new)
    tl.store(obase + d, (acc / den).to(OUT.dtype.element_ty))


def prime_trtllm_sinks(module: nn.Module, sinks: torch.Tensor | None) -> None:
    """API-compatible cache of the attention-sink vector.

    The kernel reads ``s_aux`` in its native dtype, so unlike the trtllm-gen
    path there is no FP32 copy to materialize; the hook stays so an owning
    attention layer can still call it from its post-load hook.
    """
    module._sinks_fp32 = sinks
    module._sinks_src = sinks


def trtllm_sinks(module: nn.Module, s_aux: torch.Tensor | None):
    return s_aux


class TRTLLMDecode(nn.Module):
    def __init__(self, num_qo_heads: int, num_kv_heads: int, head_dim: int,
                 workspace: torch.Tensor | None = None):
        super().__init__()
        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.sm_scale = head_dim ** -0.5
        self._workspace = workspace
        self._sinks_fp32: torch.Tensor | None = None
        self._sinks_src: torch.Tensor | None = None
        self._plan_key = None
        self._plan = None

    def prime_sinks(self, sinks: torch.Tensor | None) -> None:
        prime_trtllm_sinks(self, sinks)

    # -- launch planning ---------------------------------------------------
    @staticmethod
    def _plan_for(batch, n_qo, n_kv, head_dim, page, max_len, kv_strides,
                  num_pages):
        gqa = n_qo // n_kv
        bh = max(16, triton.next_power_of_2(gqa))
        if max_len <= _SHORT_LEN:
            bn, warps, stages, maxnreg, split_max = _CFG_SHORT
        else:
            bn, warps, stages, maxnreg, split_max = _CFG.get(head_dim, _CFG_DEFAULT)
        nbh = batch * n_kv
        nblk = _ceil(max_len, bn)
        if nblk <= 2:
            # One or two steps of work in total: splitting it would cost more in
            # merge-kernel launch than it saves.
            ns, split = 1, nblk * bn
        else:
            ns = min(max(1, _ceil(_CTA_TARGET, nbh)), nblk)
            split = min(_ceil(_ceil(max_len, ns), bn) * bn, split_max)
            ns = max(1, _ceil(max_len, split))
        # With one split the merge kernel is pure overhead, so spread the GQA
        # group over ``nsub`` CTAs to recover parallelism instead.
        nsub, hpc = 1, gqa
        if ns == 1:
            kv_bytes = max(1, nbh * max_len * head_dim * 4)
            while (nsub < gqa and nbh * nsub * 2 <= _CTA_TARGET
                   and kv_bytes * nsub * 2 <= _REDUNDANT_BYTES):
                nsub *= 2
            hpc = _ceil(gqa, nsub)
        # Merge kernel: (request, qo_head) alone does not fill the GPU for a
        # single-request decode, so tile the head dim as well.
        bd = head_dim
        while bd > 16 and batch * n_qo * (head_dim // bd) < _REDUCE_TARGET:
            bd //= 2
        bs = max(8, min(triton.next_power_of_2(ns), max(16, _REDUCE_TILE // bd)))
        # Page-table offsets fit in int32 unless the cache is enormous; 32-bit
        # address math in the hot loop is measurably faster than 64-bit.
        s_b, s_h, s_t = kv_strides
        i32 = ((num_pages - 1) * s_b + (n_kv - 1) * s_h + (page - 1) * s_t
               + head_dim) < (1 << 31)
        return dict(gqa=gqa, bh=bh, split=split, ns=ns, bn=bn, i32=i32,
                    nsub=nsub, hpc=hpc, warps=warps, stages=stages,
                    launch=({} if maxnreg is None else {"maxnreg": maxnreg}),
                    nbh=nbh, bd=bd, bs=bs,
                    nd=head_dim // bd, mid_shape=(batch, n_qo, ns, head_dim))

    def forward(self, q, k_cache, v_cache, cache_seqlens=None,
                block_table=None, softmax_scale=None, causal=True,
                max_seq_len=None, s_aux=None, window_size=None, **kwargs):
        if block_table.stride(-1) != 1:
            block_table = block_table.contiguous()
        if cache_seqlens.stride(-1) != 1:
            cache_seqlens = cache_seqlens.contiguous()
        batch, n_qo, head_dim = q.shape
        n_kv = k_cache.shape[1]
        page = k_cache.shape[2]
        bt_cap = block_table.shape[1] * page
        if max_seq_len is None:
            max_len = bt_cap
        else:
            max_len = max(1, min(int(max_seq_len), bt_cap))

        key = (batch, n_qo, n_kv, head_dim, page, max_len)
        p = self._plan
        if key != self._plan_key:
            p = self._plan_for(batch, n_qo, n_kv, head_dim, page, max_len,
                               k_cache.stride()[:3], k_cache.shape[0])
            p["mid"] = torch.empty(p["mid_shape"], dtype=torch.float32,
                                   device=q.device)
            p["lse"] = torch.empty(p["mid_shape"][:3], dtype=torch.float32,
                                   device=q.device)
            self._plan = p
            self._plan_key = key
        mid, lse, ns = p["mid"], p["lse"], p["ns"]
        out = torch.empty((batch, n_qo, head_dim), dtype=q.dtype, device=q.device)

        window_left = -1
        if window_size is not None and window_size[0] >= 0:
            window_left = int(window_size[0])
        scale = self.sm_scale if softmax_scale is None else softmax_scale
        # One split and no sink/window: the split kernel already produces the
        # normalized context, so write straight to the output and skip the merge.
        final = ns == 1 and s_aux is None and window_left < 0
        dst = out if final else mid

        _split_kernel[(p["nbh"], ns, p["nsub"] if final else 1)](
            q, k_cache, v_cache, block_table, cache_seqlens, dst, lse,
            q.stride(0), q.stride(1),
            k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
            block_table.stride(0),
            dst.stride(0), dst.stride(1), 0 if final else mid.stride(2),
            lse.stride(0), lse.stride(1),
            scale * _LOG2E, n_kv, window_left,
            GQA=p["gqa"], BH=p["bh"], D=head_dim, BN=p["bn"], SPLIT=p["split"],
            PAGE=page, USE_WIN=(window_left >= 0), I32=p["i32"],
            FINAL=final, HPC=(p["hpc"] if final else p["gqa"]),
            num_warps=p["warps"], num_stages=p["stages"], **p["launch"],
        )
        if final:
            return out
        _reduce_kernel[(batch, n_qo, p["nd"])](
            mid, lse, out, cache_seqlens, s_aux,
            mid.stride(0), mid.stride(1), mid.stride(2),
            lse.stride(0), lse.stride(1),
            out.stride(0), out.stride(1),
            window_left, _LOG2E,
            BD=p["bd"], BS=p["bs"], SPLIT=p["split"], NS=ns,
            HAS_SINK=(s_aux is not None), USE_WIN=(window_left >= 0),
            num_warps=4, num_stages=2,
        )
        return out
