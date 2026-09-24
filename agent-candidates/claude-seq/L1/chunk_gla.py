"""Chunk GLA — fused Triton chunked prefill kernel.

Same math as ``fla.ops.gla.chunk_gla`` (chunked linear attention with a
log-space vector forget gate), restructured for this workload:

* the chunk-local cumsum, both intra-chunk ``A`` passes and the gated
  ``q``/``k`` materialization collapse into one ``prep`` kernel, so the
  ``[BT, BT]`` intra-chunk score block comes out of a single
  ``[BT, K] x [K, BT]`` matmul instead of a ``BC``-step serial loop;
* the inter-chunk recurrence and the output pass either run as two kernels
  through a materialized per-chunk state (best when that state tensor is
  amortized over a long sequence) or as one kernel holding the whole
  ``[K, BV]`` state in registers (best for the short packed sequences, which
  are launch bound);
* no host synchronization -- the varlen chunk table is rebuilt inside each
  kernel from ``cu_seqlens`` by a small in-register scan, so nothing is read
  back to the CPU and no grid size depends on device data;
* launches go straight to the cached ``CompiledKernel`` and the scratch
  tensors share two allocations, keeping the per-call host cost near the bare
  driver launch cost.

Tensor layout matches FLA's convention (``[B, T, H, K]``).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

RCP_LN2 = triton.language.constexpr(1.4426950408889634)
BT = 64

# The captured calls span 5-CTA launches (a 64-token prefill) and 15k-CTA
# launches (a 200k-token batch), which want opposite trade-offs: tiny grids are
# launch/latency bound and want few kernels and many programs, big grids want
# small tiles and a state tensor that is read once. Each tuple below is
# (small-grid, large-grid). Overridable by the tuning scripts in dev/.
SMALL_GRID = 2048          # program-count threshold between the two regimes
FUSE = (True, False)       # single-kernel recurrence+output per regime
KSPLIT = (4, 1)            # prep programs per (chunk, head); >1 emits partial A
BK_PREP = (64, 32)
NW_PREP = (2, 4)
NS_PREP = (3, 3)
BV_FUSE = (32, 64)         # fused-path state tile; only the small entry is used
NW_FUSE = (8, 8)
NS_FUSE = (2, 2)
BK_STATE = (128, 128)      # split-path tiles; only the large entry is used
BV_STATE = (128, 128)
NW_STATE = (4, 4)
NS_STATE = (2, 2)
BV_OUT = (128, 128)
NW_OUT = (8, 8)
NS_OUT = (2, 2)
BV_ZERO = 128              # tail-zeroing tile inside the prep kernel
A_BF16 = (True, True)      # bf16 (vs tf32) mma for the intra-chunk A block


@triton.jit
def _seq_of_chunk(cu, i_c, N, BT: tl.constexpr, NP2: tl.constexpr):
    """(bos, Tn, i_t, found) for global chunk ``i_c`` of a packed varlen batch.

    Rebuilding the chunk table per program costs one short scan and removes the
    device->host copy a precomputed table would need.
    """
    o_n = tl.arange(0, NP2)
    s0 = tl.load(cu + o_n, mask=o_n < N, other=0).to(tl.int32)
    s1 = tl.load(cu + o_n + 1, mask=o_n < N, other=0).to(tl.int32)
    cnt = tl.where(o_n < N, (s1 - s0 + (BT - 1)) // BT, 0)
    base = tl.cumsum(cnt, 0) - cnt
    sel = (o_n < N) & (base <= i_c) & (i_c < base + cnt)
    return (tl.sum(tl.where(sel, s0, 0)), tl.sum(tl.where(sel, s1 - s0, 0)),
            i_c - tl.sum(tl.where(sel, base, 0)), tl.sum(sel.to(tl.int32)))


@triton.jit
def _seq_base(cu, i_n, N, BT: tl.constexpr, NP2: tl.constexpr):
    """(bos, Tn, first chunk index) for sequence ``i_n``."""
    o_n = tl.arange(0, NP2)
    s0 = tl.load(cu + o_n, mask=o_n < N, other=0).to(tl.int32)
    s1 = tl.load(cu + o_n + 1, mask=o_n < N, other=0).to(tl.int32)
    cnt = tl.where(o_n < N, (s1 - s0 + (BT - 1)) // BT, 0)
    return (tl.sum(tl.where(o_n == i_n, s0, 0)),
            tl.sum(tl.where(o_n == i_n, s1 - s0, 0)),
            tl.sum(tl.where(o_n < i_n, cnt, 0)))


@triton.jit
def _load_A(aux, AOFF, ASTR, tok, o_i, m_t, BT: tl.constexpr, KS: tl.constexpr):
    """The intra-chunk score block, summing the per-K-slice partials in fp32."""
    p = aux + AOFF + tok[:, None] * BT + o_i[None, :]
    b_A = tl.load(p, mask=m_t[:, None], other=0.0)
    for j in range(1, KS):
        b_A += tl.load(p + j * ASTR, mask=m_t[:, None], other=0.0)
    return b_A


@triton.jit(do_not_specialize=['scale', 'T', 'NT', 'N', 'KOFF', 'AOFF', 'ASTR'])
def _prep_kernel(
    q, k, g, cu, buf, aux, o,
    scale, T, NT, N, KOFF, AOFF, ASTR,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    KPP: tl.constexpr,
    BVZ: tl.constexpr,
    ABF: tl.constexpr,
    NP2: tl.constexpr,
    NCZ: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """Per (chunk, head, K slice): local cumsum of ``g``, gated ``q``/``k``, ``A``.

    Each program owns ``KPP`` of the ``K`` columns. When ``KPP < K`` the grid
    gains a third dimension and every program emits a partial ``A`` block for
    the consumer to sum -- a little extra traffic for the program count the
    short-sequence launches need.

    Programs at or past ``NCZ`` are tail-clearing blocks: rows at or past
    ``cu_seqlens[-1]`` are never produced by the chunk path but the reference
    leaves them zero, and they are disjoint from every real write.
    """
    i_c = tl.program_id(0)
    i_h = tl.program_id(1)
    i_ks = tl.program_id(2)
    o_i = tl.arange(0, BT)

    if IS_VARLEN:
        if i_c >= NCZ:
            if i_ks != 0:
                return
            o_t = (i_c - NCZ) * BT + o_i
            m_t = (o_t >= tl.load(cu + N).to(tl.int32)) & (o_t < T)
            tok = o_t.to(tl.int64) * H + i_h
            for j in range(tl.cdiv(V, BVZ)):
                o_v = j * BVZ + tl.arange(0, BVZ)
                tl.store(o + tok[:, None] * V + o_v[None, :],
                         tl.zeros([BT, BVZ], dtype=o.dtype.element_ty),
                         mask=m_t[:, None] & (o_v < V)[None, :])
            return
        bos, Tn, i_t, found = _seq_of_chunk(cu, i_c, N, BT, NP2)
        if found == 0:
            return
    else:
        i_t = i_c % NT
        bos = (i_c // NT) * T
        Tn = T

    o_t = i_t * BT + o_i
    m_t = o_t < Tn
    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    row = (bos.to(tl.int64) * H + i_h) * K + o_t.to(tl.int64)[:, None] * (H * K)
    for i_k in range(tl.cdiv(KPP, BK)):
        o_k = i_ks * KPP + i_k * BK + tl.arange(0, BK)
        m = m_t[:, None] & (o_k < K)[None, :]
        p = row + o_k[None, :]

        b_g = tl.load(g + p, mask=m, other=0.0).to(tl.float32)
        b_gc = tl.cumsum(b_g, 0) * RCP_LN2
        # Rows past the chunk tail load 0, so row BT-1 still carries the last
        # valid cumsum and row BT//2 is a safe mid-chunk exponent reference.
        g_last = tl.sum(tl.where(o_i[:, None] == BT - 1, b_gc, 0.0), 0)
        g_mid = tl.sum(tl.where(o_i[:, None] == BT // 2, b_gc, 0.0), 0)

        b_q = tl.load(q + p, mask=m, other=0.0)
        f_qg = b_q * tl.math.exp2(b_gc) * scale
        tl.store(buf + p, f_qg.to(buf.dtype.element_ty), mask=m)

        b_k = tl.load(k + p, mask=m, other=0.0)
        f_kd = b_k * tl.math.exp2(g_last[None, :] - b_gc)
        tl.store(buf + KOFF + p, f_kd.to(buf.dtype.element_ty), mask=m)

        tl.store(aux + (i_c * H + i_h) * K + o_k, tl.math.exp2(g_last), mask=o_k < K)

        # A = (q 2^{gc-gmid}) . (k 2^{gmid-gc})^T; referencing the exponents to
        # the chunk midpoint keeps both factors within 2^+-40 of unity.
        b_l = f_qg * tl.math.exp2(-g_mid)[None, :]
        b_r = f_kd * tl.math.exp2(g_mid - g_last)[None, :]
        if ABF:
            b_A += tl.dot(b_l.to(buf.dtype.element_ty), tl.trans(b_r).to(buf.dtype.element_ty))
        else:
            b_A += tl.dot(b_l, tl.trans(b_r))

    m_A = (o_i[:, None] >= o_i[None, :]) & (o_i < Tn - i_t * BT)[None, :]
    p_A = (AOFF + i_ks * ASTR + (bos.to(tl.int64) * H + i_h) * BT
           + o_t.to(tl.int64)[:, None] * (H * BT) + o_i[None, :])
    tl.store(aux + p_A, tl.where(m_A, b_A, 0.0), mask=m_t[:, None])


@triton.jit(do_not_specialize=['T', 'NT', 'N', 'KOFF'])
def _state_kernel(
    buf, v, aux, h0, h, ht, cu,
    T, NT, N, KOFF,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NP2: tl.constexpr,
    USE_H0: tl.constexpr,
    STORE_HT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """Sequential inter-chunk recurrence; writes the chunk-entry states."""
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, Tn, base = _seq_base(cu, i_n, N, BT, NP2)
    else:
        base, bos, Tn = i_n * NT, i_n * T, T

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    m_k = o_k < K
    m_v = o_v < V
    m_h = m_k[:, None] & m_v[None, :]

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_H0:
        b_h += tl.load(h0 + i_nh.to(tl.int64) * (K * V) + o_k[:, None] * V + o_v[None, :],
                       mask=m_h, other=0.0).to(tl.float32)

    for i_t in range(tl.cdiv(Tn, BT)):
        p_h = h + ((base + i_t).to(tl.int64) * H + i_h) * (K * V) + o_k[:, None] * V + o_v[None, :]
        tl.store(p_h, b_h.to(h.dtype.element_ty), mask=m_h)

        o_t = i_t * BT + tl.arange(0, BT)
        m_t = o_t < Tn
        tok = (bos.to(tl.int64) * H + i_h) + o_t.to(tl.int64) * H
        b_kd = tl.load(buf + KOFF + tok[None, :] * K + o_k[:, None],
                       mask=m_k[:, None] & m_t[None, :], other=0.0)
        b_v = tl.load(v + tok[:, None] * V + o_v[None, :],
                      mask=m_t[:, None] & m_v[None, :], other=0.0)
        b_gl = tl.load(aux + ((base + i_t) * H + i_h) * K + o_k, mask=m_k, other=0.0)
        b_h = b_h * b_gl[:, None] + tl.dot(b_kd, b_v)

    if STORE_HT:
        tl.store(ht + i_nh.to(tl.int64) * (K * V) + o_k[:, None] * V + o_v[None, :],
                 b_h, mask=m_h)


@triton.jit(do_not_specialize=['T', 'NT', 'N', 'AOFF', 'ASTR'])
def _out_kernel(
    buf, v, aux, h, o, cu,
    T, NT, N, AOFF, ASTR,
    H: tl.constexpr,
    K: tl.constexpr,
    KP2: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    KS: tl.constexpr,
    NP2: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """o = (q 2^{gc}) H_chunk + tril(A) v."""
    i_c = tl.program_id(0)
    i_h = tl.program_id(1)
    o_i = tl.arange(0, BT)
    o_k = tl.arange(0, KP2)
    m_k = o_k < K

    if IS_VARLEN:
        bos, Tn, i_t, found = _seq_of_chunk(cu, i_c, N, BT, NP2)
        if found == 0:
            return
    else:
        i_t = i_c % NT
        bos = (i_c // NT) * T
        Tn = T

    o_t = i_t * BT + o_i
    m_t = o_t < Tn
    row = (bos.to(tl.int64) * H + i_h) * K + o_t.to(tl.int64)[:, None] * (H * K)
    tok = (bos.to(tl.int64) * H + i_h) + o_t.to(tl.int64) * H
    b_qg = tl.load(buf + row + o_k[None, :], mask=m_t[:, None] & m_k[None, :], other=0.0)
    b_A = _load_A(aux, AOFF, ASTR, tok, o_i, m_t, BT, KS).to(b_qg.dtype)
    hb = (i_c.to(tl.int64) * H + i_h) * (K * V)

    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = o_v < V
        m_tv = m_t[:, None] & m_v[None, :]
        b_h = tl.load(h + hb + o_k[:, None] * V + o_v[None, :],
                      mask=m_k[:, None] & m_v[None, :], other=0.0)
        b_v = tl.load(v + tok[:, None] * V + o_v[None, :], mask=m_tv, other=0.0)
        b_o = tl.dot(b_qg, b_h) + tl.dot(b_A, b_v)
        tl.store(o + tok[:, None] * V + o_v[None, :], b_o.to(o.dtype.element_ty), mask=m_tv)


@triton.jit(do_not_specialize=['T', 'NT', 'N', 'KOFF', 'AOFF', 'ASTR'])
def _fused_kernel(
    buf, v, aux, h0, o, ht, cu,
    T, NT, N, KOFF, AOFF, ASTR,
    H: tl.constexpr,
    K: tl.constexpr,
    KP2: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    KS: tl.constexpr,
    NP2: tl.constexpr,
    USE_H0: tl.constexpr,
    STORE_HT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """Recurrence and output in one pass, with the state live in registers."""
    i_v = tl.program_id(0)
    i_nh = tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, Tn, base = _seq_base(cu, i_n, N, BT, NP2)
    else:
        base, bos, Tn = i_n * NT, i_n * T, T

    o_i = tl.arange(0, BT)
    o_k = tl.arange(0, KP2)
    m_k = o_k < K
    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V
    m_h = m_k[:, None] & m_v[None, :]

    b_h = tl.zeros([KP2, BV], dtype=tl.float32)
    if USE_H0:
        b_h += tl.load(h0 + i_nh.to(tl.int64) * (K * V) + o_k[:, None] * V + o_v[None, :],
                       mask=m_h, other=0.0).to(tl.float32)

    for i_t in range(tl.cdiv(Tn, BT)):
        o_t = i_t * BT + o_i
        m_t = o_t < Tn
        tok = (bos.to(tl.int64) * H + i_h) + o_t.to(tl.int64) * H
        b_qg = tl.load(buf + tok[:, None] * K + o_k[None, :],
                       mask=m_t[:, None] & m_k[None, :], other=0.0)
        b_A = _load_A(aux, AOFF, ASTR, tok, o_i, m_t, BT, KS).to(b_qg.dtype)
        b_v = tl.load(v + tok[:, None] * V + o_v[None, :],
                      mask=m_t[:, None] & m_v[None, :], other=0.0)
        b_o = tl.dot(b_qg, b_h.to(b_qg.dtype)) + tl.dot(b_A, b_v)
        tl.store(o + tok[:, None] * V + o_v[None, :], b_o.to(o.dtype.element_ty),
                 mask=m_t[:, None] & m_v[None, :])

        b_kd = tl.load(buf + KOFF + tok[None, :] * K + o_k[:, None],
                       mask=m_k[:, None] & m_t[None, :], other=0.0)
        b_gl = tl.load(aux + ((base + i_t) * H + i_h) * K + o_k, mask=m_k, other=0.0)
        b_h = b_h * b_gl[:, None] + tl.dot(b_kd, b_v)

    if STORE_HT:
        tl.store(ht + i_nh.to(tl.int64) * (K * V) + o_k[:, None] * V + o_v[None, :],
                 b_h, mask=m_h)


_KERNEL_CACHE: dict = {}
_PLAN_CACHE: dict = {}
_raw_stream = torch._C._cuda_getCurrentRawStream


def _compiled(jit, key, args, num_warps, num_stages):
    """The cached ``CompiledKernel``; launching it skips the JIT dispatch.

    The key must cover everything the compiled code depends on -- the constexpr
    tuple, the launch shape and the operand dtypes.
    """
    key = key + (num_warps, num_stages, tuple(a.dtype for a in args if hasattr(a, 'dtype')))
    ck = _KERNEL_CACHE.get(key)
    if ck is None:
        ck = jit.warmup(*args, grid=(1, 1, 1), num_warps=num_warps, num_stages=num_stages)
        ck._init_handles()
        _KERNEL_CACHE[key] = ck
    return ck


def _prepare(x):
    """fla's ``input_guard`` equivalent, plus the 16B alignment the JIT assumes."""
    if x is None:
        return None
    if not x.is_contiguous():
        return x.contiguous()
    return x.clone() if x.data_ptr() % 16 else x


def _plan(B, T, H, K, V, N, varlen, use_h0, ofs, dev, dt, dtv):
    """Grid sizes, buffer offsets and the compiled kernels for one shape."""
    if varlen:
        NT = 0
        NC = (T + BT - 1) // BT
        NTB = NC + N
        NP2 = 16
        while NP2 < N:
            NP2 *= 2
    else:
        NT = (T + BT - 1) // BT
        NTB = NT * B
        NC = 0
        NP2 = 16
    kp2 = 16
    while kp2 < K:
        kp2 *= 2
    # Varlen segments are one chunk each in this workload, so N*H is the
    # realistic program count there; NTB*H is exact for the dense case.
    i = 0 if (N if varlen else NTB) * H < SMALL_GRID else 1
    nbk = (K + BK_PREP[i] - 1) // BK_PREP[i]
    tpp = -(-nbk // min(KSPLIT[i], nbk))     # K tiles per prep program
    ks = -(-nbk // tpp)
    kpp = tpp * BK_PREP[i]
    # buf holds the gated q and k; aux holds 2^g_last and the fp32 A partials.
    koff = (B * T * H * K + 7) // 8 * 8
    aoff = (NTB * H * K + 3) // 4 * 4
    astr = (B * T * H * BT + 3) // 4 * 4

    fq = torch.empty(8, dtype=torch.float32, device=dev)
    bq = torch.empty(8, dtype=dt, device=dev)
    vq = torch.empty(8, dtype=dtv, device=dev)
    iq = torch.empty(8, dtype=torch.int64, device=dev)
    cq = iq if varlen else fq

    prep_c = (H, K, V, BT, BK_PREP[i], kpp, BV_ZERO, A_BF16[i], NP2, NTB, varlen)
    ck_p = _compiled(_prep_kernel, ('p',) + prep_c,
                     (bq, bq, bq, cq, bq, fq, vq, 0.1, T, NT, N, koff, aoff, astr) + prep_c,
                     NW_PREP[i], NS_PREP[i])
    sizes = (2 * koff, aoff + ks * astr)
    if FUSE[i]:
        bvf = BV_FUSE[i]
        fuse_c = (H, K, kp2, V, BT, bvf, ks, NP2, use_h0, ofs, varlen)
        ck_f = _compiled(_fused_kernel, ('f',) + fuse_c,
                         (bq, vq, fq, fq, vq, fq, cq, T, NT, N, koff, aoff, astr) + fuse_c,
                         NW_FUSE[i], NS_FUSE[i])
        grids = (-(-V // bvf), N * H, ks)
        return (NTB, NT, koff, aoff, astr, sizes, NTB + NC, True, grids,
                prep_c, fuse_c, dev.index, ck_p, ck_f, None)

    bks, bvs = BK_STATE[i], BV_STATE[i]
    state_c = (H, K, V, BT, bks, bvs, NP2, use_h0, ofs, varlen)
    ck_s = _compiled(_state_kernel, ('s',) + state_c,
                     (bq, vq, fq, fq, bq, fq, cq, T, NT, N, koff) + state_c,
                     NW_STATE[i], NS_STATE[i])
    out_c = (H, K, kp2, V, BT, BV_OUT[i], ks, NP2, varlen)
    ck_o = _compiled(_out_kernel, ('o',) + out_c,
                     (bq, vq, fq, bq, vq, cq, T, NT, N, aoff, astr) + out_c,
                     NW_OUT[i], NS_OUT[i])
    grids = (-(-K // bks), -(-V // bvs), ks)
    return (NTB, NT, koff, aoff, astr, sizes, NTB + NC, False, grids,
            prep_c, state_c, dev.index, ck_p, ck_s, (out_c, ck_o))


def _gla_fwd(q, k, v, g, scale, h0, ofs, cu):
    B, T, H, K = q.shape
    V = v.shape[-1]
    varlen = cu is not None
    N = cu.numel() - 1 if varlen else B
    key = (B, T, H, K, V, N, varlen, h0 is not None, ofs, q.dtype, v.dtype, q.device)
    plan = _PLAN_CACHE.get(key)
    if plan is None:
        plan = _plan(B, T, H, K, V, N, varlen, h0 is not None, ofs, q.device, q.dtype, v.dtype)
        _PLAN_CACHE[key] = plan
    (NTB, NT, koff, aoff, astr, sizes, NOZ, fuse, grids, prep_c, sec_c,
     devix, ck_p, ck_2, tail) = plan
    if scale is None:
        scale = K ** -0.5

    buf = q.new_empty(sizes[0])
    aux = q.new_empty(sizes[1], dtype=torch.float32)
    ht = q.new_empty(N, H, K, V, dtype=torch.float32) if ofs else aux
    o = torch.empty_like(v)
    cu_ = cu if varlen else aux
    st = _raw_stream(devix)

    ck_p.run(NOZ, H, grids[2], st, ck_p.function, ck_p.packed_metadata, None, None, None,
             q, k, g, cu_, buf, aux, o, scale, T, NT, N, koff, aoff, astr, *prep_c)
    if fuse:
        ck_2.run(grids[0], grids[1], 1, st, ck_2.function, ck_2.packed_metadata,
                 None, None, None, buf, v, aux, h0 if h0 is not None else aux, o, ht, cu_,
                 T, NT, N, koff, aoff, astr, *sec_c)
    else:
        h = q.new_empty(NTB * H * K * V)
        ck_2.run(grids[0], grids[1], N * H, st, ck_2.function, ck_2.packed_metadata,
                 None, None, None, buf, v, aux, h0 if h0 is not None else aux, h, ht, cu_,
                 T, NT, N, koff, *sec_c)
        out_c, ck_o = tail
        ck_o.run(NTB, H, 1, st, ck_o.function, ck_o.packed_metadata, None, None, None,
                 buf, v, aux, h, o, cu_, T, NT, N, aoff, astr, *out_c)
    return o, (ht if ofs else None)


class ChunkGLA(nn.Module):
    """Fused Triton chunk GLA kernel."""

    def forward(
        self,
        q: torch.Tensor,  # [B, T, H, K]
        k: torch.Tensor,  # [B, T, H, K]
        v: torch.Tensor,  # [B, T, H, V]
        g: torch.Tensor,  # [B, T, H, K]  log-space forget gate
        scale: float | None = None,
        initial_state: torch.Tensor | None = None,  # [B, H, K, V]
        output_final_state: bool = False,
        cu_seqlens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return _gla_fwd(
            _prepare(q), _prepare(k), _prepare(v), _prepare(g),
            scale, _prepare(initial_state), output_final_state, _prepare(cu_seqlens),
        )
