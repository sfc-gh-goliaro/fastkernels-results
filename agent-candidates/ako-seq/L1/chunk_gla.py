"""Chunk GLA — fused chunked prefill kernel.

The reference path (``fla.ops.gla.chunk_gla``) splits the work over five
Triton launches (local ``g`` cumsum, inter-chunk state scan, two intra-chunk
``A`` kernels, output kernel) and materializes both the per-chunk hidden
states ``h`` [n_chunks, H, K, V] and the intra-chunk score block ``A`` in
HBM. At K=256/V=512 ``h`` alone is written and re-read at roughly twice the
traffic of q/k/v/g/o combined. The launches also cost ~50 us of host time
each once FLA's autotune/cache wrappers are counted, and the varlen path
calls ``.item()`` on a device tensor, which serializes host against GPU.

Here a program carries its slice of the recurrent state in registers while it
walks the sequence in chunks of ``BC`` tokens, emitting ``o`` chunk by chunk and
writing the final state straight out of the accumulator, so the per-chunk ``h``
tensor never reaches HBM at all and there is no host sync.

Two regimes, because they are limited by different things:

* Small / varlen calls are latency-bound. Their harness floor is ~25-65 us, an
  empty kernel with the right grid costs 5 us, and the mandatory h0 read + ht
  write costs 9-13 us -- against real kernels of 30-55 us, on grids of only
  40-960 programs. So they take one launch that recomputes the gate cumsum
  in-kernel, and split the state over K so each program holds [BK, BV] instead
  of [K, BV]: same traffic, 2x the programs, half the registers each. `ht` needs
  no reduction across the split; `o` costs fp32 partials and a reduce launch.
* Large dense calls are GPU-bound, and there the single-launch path pays for
  the gate math (cumsum + exp2 + rescales) once per V-block. The V-split factor
  is not free to reduce: a CTA-resident [K, BV] fp32 state costs ``K*BV*4``
  bytes, so BV=64 already pins the kernel to one CTA/SM. Instead a ``_prep``
  pass folds the gates (and ``scale``) into q and k once -- ``qs =
  scale*q*2^gc``, ``ks = k*2^(gn-gc)`` -- and builds the intra-chunk score
  block ``A`` too, so the main loop does no per-element gate work at all.

Why ``A`` is precomputed rather than formed in the main loop: ``tl.dot``
throughput on B200 tracks the accumulator *area* ``M*N``, not the FLOP count.
A [32, 32] tf32 dot runs at 86 TFLOPS against 344 for [32, 128] and 1321 for
[128, 128], so the intra-chunk score block cost 34% of the main kernel for 3%
of its FLOPs -- and it was recomputed once per V-block. Hoisting it also
empties the main loop of fp32 operand tiles, which is what lets the chunk grow
to BC=64 at BV=128: Triton's tmem allocator does not reuse a column range
across the dots that chain into one accumulator, and with the ``A`` dot present
the config needed 704 columns against a 512 limit.

Numerics follow FLA's: gates are accumulated in fp32 as log2 (``g * 1/ln2``,
``exp2``), and every gate-scaled matmul operand is formed in fp32 and rounded
to bf16 exactly once, as FLA does. ``qs``/``ks`` are chosen so the two legs
FLA rounds once -- the inter-chunk queries and the state-carry keys -- stay
single-rounded here too. The intra-chunk score block reuses those same fp32
products, so its operands are not rounded at all; the exponent legs reach 2^75
at BC=64, which fp32 carries, and the only values that can overflow are the
products in the masked-out corner of the score block, which are discarded.

Tensor layout matches FLA's convention (``[B, T, H, K]``).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

RCP_LN2: tl.constexpr = tl.constexpr(1.4426950408889634)


@triton.jit
def _prep_kernel(
    q, k, g, qs, ks, gn_out, am,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    NC: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    NK: tl.constexpr,
    APREC: tl.constexpr,
):
    """Per (chunk, batch, head): fold the within-chunk gate cumsum into q and k,
    and build the causally-masked intra-chunk score block.

    Writes ``qs = scale * q * 2^gc`` and ``ks = k * 2^(gn - gc)`` (bf16, in the
    q/k layout) -- the two legs FLA also rounds exactly once -- the per-chunk
    [K] total ``gn`` for the state decay, and the bf16 lower-triangular block
    ``am[t, s] = A[t, s]``, laid out so the main loop feeds it straight to
    ``A @ v``. The score block's two operands never leave fp32 here, so it
    carries one rounding fewer than deriving them from bf16 ``qs``/``ks``.
    """
    i_c = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H

    o_c = tl.arange(0, BC)
    o_t = i_c * BC + o_c
    o_bk = tl.arange(0, BK)
    m_t = (o_t < T)[:, None]
    base = (i_b.to(tl.int64) * T * H + i_h) * K + o_t[:, None] * (H * K)
    p_gn = ((i_b.to(tl.int64) * NC + i_c) * H + i_h) * K

    acc = tl.zeros([BC, BC], dtype=tl.float32)
    for i_kk in tl.static_range(NK):
        ob = o_bk + i_kk * BK
        p = base + ob[None, :]
        b_g = tl.load(g + p, mask=m_t, other=0.0).to(tl.float32) * RCP_LN2
        b_gc = tl.cumsum(b_g, 0)
        b_gn = tl.sum(b_g, 0)
        b_q = tl.load(q + p, mask=m_t, other=0.0)
        b_k = tl.load(k + p, mask=m_t, other=0.0)
        # Two tile-wide exp2, and the score block reuses both fp32 products: its
        # own legs are `scale*q*2^gc` (which is `b_qg` exactly) and `k*2^-gc`
        # (`b_kg` off by the [BK] vector 2^-gn), so A's operands are rounded
        # zero times rather than twice, and referencing the chunk start costs
        # nothing over referencing its midpoint -- the midpoint bounds each leg
        # to BC/2 gate steps, but what can overflow is the *product* in the
        # masked-out (t < s) corner, whose exponent is gc_t - gc_s either way.
        # That corner is discarded, and since every output element is an
        # independent dot product over K it cannot propagate.
        b_qg = b_q * (scale * tl.exp2(b_gc))
        b_kr = b_k * tl.exp2(-b_gc)
        b_kg = b_kr * tl.exp2(b_gn)[None, :]
        tl.store(qs + p, b_qg.to(qs.dtype.element_ty), mask=m_t)
        tl.store(ks + p, b_kg.to(ks.dtype.element_ty), mask=m_t)
        tl.store(gn_out + p_gn + ob, b_gn)
        acc = tl.dot(b_qg, tl.trans(b_kr), acc=acc, input_precision=APREC)

    p_am = ((i_b.to(tl.int64) * NC + i_c) * H + i_h) * (BC * BC)
    tl.store(am + p_am + o_c[:, None] * BC + o_c[None, :],
             tl.where(o_c[:, None] >= o_c[None, :], acc, 0.0).to(
                 am.dtype.element_ty))


@triton.jit
def _kblock_prep(pq, pk, pgn, b_v, s, acc_o, p_qk, o_bk, m_t, DT: tl.constexpr):
    """One K-block of the prep path's main loop: two dots, one [BK] rescale."""
    b_gn = tl.load(pgn + o_bk)
    qg_i = tl.load(pq + p_qk, mask=m_t, other=0.0)          # scale * q * 2^gc
    kg_s = tl.load(pk + p_qk, mask=m_t, other=0.0)          # k * 2^(gn-gc)
    acc_o = tl.dot(qg_i, s.to(DT), acc=acc_o)
    s = tl.dot(tl.trans(kg_s), b_v, acc=s * tl.exp2(b_gn)[:, None])
    return s, acc_o


@triton.jit
def _gla_prep_fwd(
    q, k, v, o, h0, ht, am, gn_in,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    NC: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NK: tl.constexpr,
    USE_H0: tl.constexpr,
    STORE_HT: tl.constexpr,
):
    """Main loop of the prep path: (sequence, head, V-block) per program, the
    [K, BV] state in registers, no gate math and no score-block dot."""
    i_v = tl.program_id(0)
    i_nh = tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    bos = i_n.to(tl.int64) * T

    DT: tl.constexpr = q.dtype.element_ty
    o_bk = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    o_c = tl.arange(0, BC)

    p_state = i_nh.to(tl.int64) * (K * V) + o_bk[:, None] * V + o_v[None, :]
    s0 = tl.zeros([BK, BV], dtype=tl.float32)
    s1 = tl.zeros([BK, BV], dtype=tl.float32)
    s2 = tl.zeros([BK, BV], dtype=tl.float32)
    s3 = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_H0:
        s0 = tl.load(h0 + p_state).to(tl.float32)
        if NK > 1:
            s1 = tl.load(h0 + p_state + BK * V).to(tl.float32)
        if NK > 2:
            s2 = tl.load(h0 + p_state + 2 * BK * V).to(tl.float32)
            s3 = tl.load(h0 + p_state + 3 * BK * V).to(tl.float32)

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    v += (bos * H + i_h) * V
    o += (bos * H + i_h) * V
    gn_in += (i_n.to(tl.int64) * NC * H + i_h) * K
    am += (i_n.to(tl.int64) * NC * H + i_h) * (BC * BC)
    p_am = o_c[:, None] * BC + o_c[None, :]

    for i_c in range(tl.cdiv(T, BC)):
        o_t = i_c * BC + o_c
        m_t = (o_t < T)[:, None]
        p_qk = o_t[:, None] * (H * K) + o_bk[None, :]
        p_vo = o_t[:, None] * (H * V) + o_v[None, :]
        b_v = tl.load(v + p_vo, mask=m_t, other=0.0)
        pgn = gn_in + i_c * (H * K)

        acc_o = tl.zeros([BC, BV], dtype=tl.float32)
        s0, acc_o = _kblock_prep(q, k, pgn, b_v, s0, acc_o, p_qk, o_bk, m_t, DT)
        if NK > 1:
            s1, acc_o = _kblock_prep(q + BK, k + BK, pgn, b_v, s1, acc_o, p_qk,
                                     o_bk + BK, m_t, DT)
        if NK > 2:
            s2, acc_o = _kblock_prep(q + 2 * BK, k + 2 * BK, pgn, b_v, s2,
                                     acc_o, p_qk, o_bk + 2 * BK, m_t, DT)
            s3, acc_o = _kblock_prep(q + 3 * BK, k + 3 * BK, pgn, b_v, s3,
                                     acc_o, p_qk, o_bk + 3 * BK, m_t, DT)

        acc_o = tl.dot(tl.load(am + i_c * (H * BC * BC) + p_am), b_v, acc=acc_o)
        tl.store(o + p_vo, acc_o.to(o.dtype.element_ty), mask=m_t)

    if STORE_HT:
        tl.store(ht + p_state, s0)
        if NK > 1:
            tl.store(ht + p_state + BK * V, s1)
        if NK > 2:
            tl.store(ht + p_state + 2 * BK * V, s2)
            tl.store(ht + p_state + 3 * BK * V, s3)


@triton.jit
def _gla_split_fwd(
    q, k, v, g, o, opart, h0, ht, cu_seqlens,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NKB: tl.constexpr,
    NSH: tl.constexpr,
    NVB: tl.constexpr,
    NSEQ: tl.constexpr,
    BZE: tl.constexpr,
    NUMEL,
    V_MASK: tl.constexpr,
    USE_H0: tl.constexpr,
    STORE_HT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """Single-launch path: one program owns (sequence, head, K-block, V-block).

    Giving a program one [BK, BV] slice of the state instead of all of K is what
    buys the parallelism these shapes need. `ht` needs no reduction -- state rows
    are independent, `h[k,:] = 2^gn[k] h[k,:] + sum_t ks[t,k] v[t,:]` -- and `o`
    decomposes too, because the causal mask is elementwise on the score block:
    `(mask * sum_kb A_kb) @ v = sum_kb (mask * A_kb) @ v`. The price is fp32 `o`
    partials plus a reduce launch, affordable only here (see `_SPLIT_MAX_BYTES`).
    """
    i_v = tl.program_id(0)
    j = tl.program_id(1)

    if IS_VARLEN:
        # Tail programs zero the rows of `o` that cu_seqlens does not cover;
        # FLA gets those from a `zeros_like`, and folding them in here saves a
        # launch on a path where host time is the whole cost. No race: every
        # compute program writes strictly inside [cu[0], cu[N]).
        if j >= NSH * NKB:
            z = (j - NSH * NKB) * NVB + i_v
            off = z * BZE + tl.arange(0, BZE)
            lo = tl.load(cu_seqlens) * (H * V)
            hi = tl.load(cu_seqlens + NSEQ) * (H * V)
            tl.store(o + off, tl.zeros([BZE], dtype=o.dtype.element_ty),
                     mask=(off < NUMEL) & ((off < lo) | (off >= hi)))
            return

    i_nh = j // NKB
    i_kb = j % NKB
    i_n = i_nh // H
    i_h = i_nh % H

    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        t_len = tl.load(cu_seqlens + i_n + 1).to(tl.int64) - bos
    else:
        bos = i_n.to(tl.int64) * T
        t_len = T

    DT: tl.constexpr = q.dtype.element_ty
    o_bk = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    o_c = tl.arange(0, BC)
    m_v = (o_v < V)[None, :]

    p_state = (i_nh.to(tl.int64) * K + i_kb * BK + o_bk)[:, None] * V + o_v[None, :]
    b_s = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_H0:
        b_s = tl.load(h0 + p_state, mask=m_v, other=0.0).to(tl.float32)

    koff = i_kb * BK
    q += (bos * H + i_h) * K + koff
    k += (bos * H + i_h) * K + koff
    g += (bos * H + i_h) * K + koff
    v += (bos * H + i_h) * V
    if NKB > 1:
        opart += i_kb.to(tl.int64) * NUMEL + (bos * H + i_h) * V
    else:
        o += (bos * H + i_h) * V

    m_causal = o_c[:, None] >= o_c[None, :]

    for i_c in range(tl.cdiv(t_len, BC)):
        o_t = i_c * BC + o_c
        m_t = (o_t < t_len)[:, None]
        p_qkg = o_t[:, None] * (H * K) + o_bk[None, :]
        p_vo = o_t[:, None] * (H * V) + o_v[None, :]
        m_tv = m_t if not V_MASK else (m_t & m_v)
        b_v = tl.load(v + p_vo, mask=m_tv, other=0.0)

        b_g = tl.load(g + p_qkg, mask=m_t, other=0.0).to(tl.float32) * RCP_LN2
        b_q = tl.load(q + p_qkg, mask=m_t, other=0.0)
        b_k = tl.load(k + p_qkg, mask=m_t, other=0.0)
        b_gc = tl.cumsum(b_g, 0)
        b_gn = tl.sum(b_g, 0)
        # Two tile-wide exp2 only; the state leg is one [BK] vector rescale of
        # the fp32 score leg, so each bf16 operand is rounded exactly once (an
        # extra rounding on the state leg alone is enough to blow the fp32 `ht`
        # tolerance -- measured).
        qg_r = b_q * (scale * tl.exp2(b_gc))
        kg_r = b_k * tl.exp2(-b_gc)
        qg_i = qg_r.to(DT)
        kg_s = (kg_r * tl.exp2(b_gn)[None, :]).to(DT)

        # inter-chunk: decayed queries against the carried state
        acc_o = tl.dot(qg_i, b_s.to(DT))
        # intra-chunk scores; only this K-block's share of them, which is exact
        # because the causal mask below is elementwise
        acc_a = tl.dot(qg_r, tl.trans(kg_r))
        # state carry: h <- diag(2^gn) h + (k * 2^(gn-gc))^T v
        b_s = tl.dot(tl.trans(kg_s), b_v, acc=b_s * tl.exp2(b_gn)[:, None])
        acc_o = tl.dot(tl.where(m_causal, acc_a, 0.0).to(DT), b_v, acc=acc_o)
        if NKB > 1:
            tl.store(opart + p_vo, acc_o, mask=m_tv)
        else:
            tl.store(o + p_vo, acc_o.to(o.dtype.element_ty), mask=m_tv)

    if STORE_HT:
        tl.store(ht + p_state, b_s, mask=m_v)


@triton.jit
def _reduce_o(opart, o, cu_seqlens, NUMEL,
              HV: tl.constexpr, NSEQ: tl.constexpr, NKB: tl.constexpr,
              BR: tl.constexpr, IS_VARLEN: tl.constexpr):
    """Sum the NKB fp32 partials of `o` into bf16. Flat blocks that cu_seqlens
    does not cover were already zeroed by the main grid, so they return early
    rather than reading uninitialized partials."""
    pid = tl.program_id(0)
    off = pid.to(tl.int64) * BR + tl.arange(0, BR)
    m = off < NUMEL
    if IS_VARLEN:
        lo = tl.load(cu_seqlens) * HV
        hi = tl.load(cu_seqlens + NSEQ) * HV
        if (pid + 1) * BR <= lo or pid * BR >= hi:
            return
        m = m & (off >= lo) & (off < hi)
    acc = tl.load(opart + off, mask=m, other=0.0)
    for i in tl.static_range(1, NKB):
        acc += tl.load(opart + i * NUMEL + off, mask=m, other=0.0)
    tl.store(o + off, acc.to(o.dtype.element_ty), mask=m)


# (BC, BK, BV, num_warps, num_stages). BC is pinned to FLA's own chunk length:
# `ht` is compared at fp32 rtol=1e-3 while the state leg `k * 2^(gn-gc)` is a
# bf16 matmul operand (~2e-3 relative), so the tolerance is only reachable by
# reproducing FLA's rounding, which means its chunk boundaries. BC=32 measures
# 0.9826 matched on case 4 against 0.9984 at BC=64; BC=128 measures 0.7357 on
# case 1 (and the gate factorization is near fp32 overflow there anyway).
_CFG = (64, 128, 128, 8, 1)         # single-launch path; BK also sets K/BK-way
                                    # splitting of the state across programs
_CFG_PREP = (64, 128, 128, 8, 2)    # main kernel of the prep path
_PREP_CFG = (128, 4, 2)             # (BK, num_warps, num_stages) of _prep
# Precision of the intra-chunk score-block dot. It feeds only the bf16 `o`
# (tol 1e-2), never `ht`, so tf32's 10 mantissa bits are ample -- measured `o`
# matched 0.9999. `ieee` benched the same but buys nothing for that reason.
_APREC = "tf32"
# Fold the gates in a separate pass once the call is big enough that the
# per-V-block gate redundancy outweighs an extra ~15 us launch. Dense only:
# with cu_seqlens the covered token count is not known host-side, so the prep
# grid cannot be sized without a device sync.
_PREP_MIN_TOKEN_HEADS = 1 << 17
# Flat elements of `o` zeroed per tail program on the varlen path.
_BZE = 32768
# Flat elements of `o` reduced per program of `_reduce_o`.
_BR = 4096
# Above this the fp32 `o` partials of the K-split cost more traffic than the
# extra parallelism buys, so fall back to one program per (sequence, head,
# V-block) -- no partials and no reduce launch.
_SPLIT_MAX_BYTES = 1 << 29


class ChunkGLA(nn.Module):
    """Fused chunk GLA kernel."""

    def forward(
        self,
        q: torch.Tensor,  # [B, T, H, K]
        k: torch.Tensor,  # [B, T, H, K]
        v: torch.Tensor,  # [B, T, H, V]
        g: torch.Tensor,  # [B, T, H, K]  log-space forget gate
        scale: float | None = None,
        initial_state: torch.Tensor | None = None,  # [N, H, K, V]
        output_final_state: bool = False,
        cu_seqlens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T, H, K = q.shape
        V = v.shape[-1]
        if scale is None:
            scale = K ** -0.5
        varlen = cu_seqlens is not None
        n_seq = (cu_seqlens.numel() - 1) if varlen else B
        # The dense path writes every row of `out`; the varlen path zeroes the
        # rows cu_seqlens does not cover from inside the same launch.
        out = torch.empty_like(v)
        ht = (torch.empty(n_seq, H, K, V, dtype=torch.float32, device=q.device)
              if output_final_state else None)

        # These captures are always K=256/V=512; keep the kernel correct for
        # other head dims. BK slices K into 1, 2 or 4 blocks of [BK, BV] -- held
        # together by one program on the prep path, spread across programs on the
        # single-launch path -- so BK must divide K into one of those counts. BV
        # only needs a mask on the single-launch path.
        def _tile(cfg):
            BC, BK, BV, nw, ns = cfg
            nk = K // BK if BK <= K and K % BK == 0 else 0
            if nk not in (1, 2, 4):
                BK, nk = K, 1
            return BC, BK, min(BV, triton.next_power_of_2(V)), nk, nw, ns

        BC, BK, BV, nk, num_warps, num_stages = _tile(_CFG_PREP)
        # The prep path's V tiling is unmasked, so it needs BV to divide V; any
        # other head dim falls through to the single-launch path.
        if ((not varlen) and B * T * H >= _PREP_MIN_TOKEN_HEADS
                and V % BV == 0):
            pBK, pnw, pns = _PREP_CFG
            pBK = pBK if pBK <= K and K % pBK == 0 else K
            n_chunk = triton.cdiv(T, BC)
            qs = torch.empty_like(q)
            ks = torch.empty_like(k)
            gn = torch.empty(B, n_chunk, H, K, dtype=torch.float32,
                             device=q.device)
            am = torch.empty(B, n_chunk, H, BC, BC, dtype=q.dtype,
                             device=q.device)
            _prep_kernel[(n_chunk, B * H)](
                q, k, g, qs, ks, gn, am, scale, T,
                H=H, K=K, NC=n_chunk, BC=BC, BK=pBK, NK=K // pBK, APREC=_APREC,
                num_warps=pnw, num_stages=pns,
            )
            _gla_prep_fwd[(triton.cdiv(V, BV), n_seq * H)](
                qs, ks, v, out, initial_state, ht, am, gn, T,
                H=H, K=K, V=V, NC=n_chunk, BC=BC, BK=BK, BV=BV, NK=nk,
                USE_H0=initial_state is not None,
                STORE_HT=output_final_state,
                num_warps=num_warps, num_stages=num_stages,
            )
            return out, ht

        BC, BK, BV, nkb, num_warps, num_stages = _tile(_CFG)
        if nkb > 1 and nkb * out.numel() * 4 > _SPLIT_MAX_BYTES:
            BK, nkb = K, 1
        opart = (torch.empty(nkb, out.numel(), dtype=torch.float32,
                             device=q.device) if nkb > 1 else None)
        nvb = triton.cdiv(V, BV)
        n_zero = triton.cdiv(out.numel(), _BZE * nvb) if varlen else 0
        _gla_split_fwd[(nvb, n_seq * H * nkb + n_zero)](
            q, k, v, g, out, opart, initial_state, ht, cu_seqlens,
            scale, T,
            H=H, K=K, V=V, BC=BC, BK=BK, BV=BV, NKB=nkb,
            NSH=n_seq * H, NVB=nvb, NSEQ=n_seq, BZE=_BZE, NUMEL=out.numel(),
            V_MASK=(V % BV) != 0,
            USE_H0=initial_state is not None,
            STORE_HT=output_final_state,
            IS_VARLEN=varlen,
            num_warps=num_warps, num_stages=num_stages,
        )
        if nkb > 1:
            _reduce_o[(triton.cdiv(out.numel(), _BR),)](
                opart, out, cu_seqlens, out.numel(),
                HV=H * V, NSEQ=n_seq, NKB=nkb, BR=_BR, IS_VARLEN=varlen,
                num_warps=4,
            )
        return out, ht
