"""Fused split-KV flash-decoding MLA kernel (Triton) over a paged latent cache.

A paged latent cache entry is ``kv_lora_rank + qk_rope_head_dim`` wide (576 =
512 + 64) and doubles as **K** (all 576 lanes) and **V** (the first 512 lanes).
The kernel exploits that: every KV tile is pulled from HBM exactly once into
SRAM and consumed twice -- as the ``B`` operand of ``QK^T`` and as the ``B``
operand of ``P*V``.  No separate V tensor is materialized or re-read.

Each program owns 16 query rows (one head group of one ``(request, q position)``
pair -- ``q_len`` and any head count above 16 are folded into the grid, never
into the mma M) against a KV strip: ~30 flop/byte against a machine balance an
order of magnitude higher, so the kernel is HBM bound and the whole game is
keeping the loads saturated:

* a ``BLOCK_N`` tile is *exactly one page* (``BLOCK_N`` divides the page size and
  every tile start is a multiple of ``BLOCK_N``), so the page comes out of the
  block table with **one scalar load** and the row base with one scalar multiply
  -- not a ``BLOCK_N``-lane gather plus ``BLOCK_N`` 64-bit address computations.
  Worth 5% on the long-context shape;
* the KV loop is a **single** instantiation -- the ragged last tile reads a
  clamped (still valid) page and is neutralised by one select on the logit
  tile, instead of a second predicated copy of the body.  The two-path version
  needed 255 regs/thread with 118 spills; this one needs 184 with none, which
  is the difference between 3.5 and 5.2 TB/s;
* the softmax row sums come out of an mma against a ones tile rather than a
  ``tl.sum``, which on a ``[16, BN]`` mma-layout tile is a cross-warp reduction
  through shared memory;
* the softmax reference is **fixed for a whole pass**, so no cross-warp
  ``tl.max(qk, 1)`` and no accumulator rescale sit on the critical path of every
  tile.  That is only exact while no exponent saturates, which is not assumed
  but *proved*: the exact row max is carried along elementwise (free -- each
  thread maxes its own lanes) and checked once per pass, with a retry that
  cannot need more than two passes.  Worth ~9% on the long-context shape.

Long context is at the bandwidth floor: KV is split across ``ns`` CTAs per
request that each emit a partial ``(acc, m, l)``, merged by a log-sum-exp
rescale pass.  Small/mid shapes are latency bound, so when one split per
request already fills the GPU the split kernel writes the final output itself
and no second kernel is launched.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

# --------------------------------------------------------------------------
# Tunables (the private A/B harness patches these; defaults are the benched
# winners).
# --------------------------------------------------------------------------
BLOCK_N = 64          # KV tokens per inner iteration
NUM_WARPS = 4
NUM_STAGES = 2        # cp.async depth of the KV loop
LOOP_RANGE = 1        # 1: tl.range(num_stages=..)  0: plain range + launch stages
UNROLL = 1            # tl.range loop_unroll_factor
TOK_PER_SPLIT = 1536  # bound per-CTA KV work so long requests stay balanced
MIN_CTAS = 296        # ~2 per SM: floor on total CTAs when the batch is small
MIN_TOK_SPLIT = 256   # never split below this; a 2nd kernel costs ~2us
SMALL_BN = 64         # BLOCK_N for latency-bound shapes (few tokens per request)
SMALL_WARPS = 8       # more warps help the one-tile-per-CTA latency-bound path
SMALL_MSL = 256
MAX_NSPLIT = 32
COMBINE_WARPS = 4
CLAMP = 64.0          # exponent window half-width; p stays in [2^-CL, 2^CL]
NEG = tl.constexpr(float('-inf'))
PM_NEG = tl.constexpr(-1.0e38)


@triton.jit
def _ld(ptr, off, DC: tl.constexpr, rmask, MASKED: tl.constexpr,
        PART: tl.constexpr, R):
    """Load one [rows, DC] lane-chunk starting at lane `off`."""
    d = off + tl.arange(0, DC)
    if MASKED:
        if PART:
            return tl.load(ptr + d[None, :], mask=rmask[:, None] & (d < R)[None, :],
                           other=0.0)
        return tl.load(ptr + d[None, :], mask=rmask[:, None], other=0.0)
    if PART:
        return tl.load(ptr + d[None, :], mask=(d < R)[None, :], other=0.0)
    return tl.load(ptr + d[None, :])


@triton.jit
def _kv_step(a0, a1, a2, a3, m_i, l_i, smx, q0, q1, q2, q3, q_pe,
             KV, BT, sbt_row, skv_p, skv_t, start, hi, last,
             qk_scale2, ones, R,
             M: tl.constexpr, PE: tl.constexpr, PAGE: tl.constexpr,
             BN: tl.constexpr, DC: tl.constexpr,
             PART: tl.constexpr, CL: tl.constexpr, ONEPG: tl.constexpr):
    """One KV tile: loaded once, consumed as K (all lanes) then as V (latent).

    Out-of-range tokens in the ragged last tile read a *clamped* (still valid)
    page instead of being predicated off, and are neutralised by one select on
    the [M, BN] logit tile.  That keeps the loop to a single instantiation --
    a separate masked tail path doubles the code and pushes the register
    allocator over the edge.

    The softmax reference ``m_i`` is *fixed* for the whole pass, so neither a
    cross-warp ``tl.max(qk, 1)`` nor an accumulator rescale is on the critical
    path of a tile -- worth ~75 us of 890 on the long-context shape.  The exact
    row max is still obtained, but as an *elementwise* running max ``smx``: each
    thread maxes only its own lanes, so there is no reduction and no barrier.
    One reduce of ``smx`` per *pass* then tells the caller whether the fixed
    reference was in range.  See the retry loop in
    :func:`_mla_decode_split`.
    """
    n = start + tl.arange(0, BN)
    if ONEPG:
        # `start` is a multiple of BN and PAGE is a multiple of BN, so the whole
        # tile lies inside ONE page: the block table needs a single scalar read
        # and the row base a single scalar multiply, instead of a BN-lane gather
        # plus BN 64-bit address computations.  The ragged last tile simply
        # reads the rest of its (existing, valid) page and is neutralised by the
        # same select on the logits, so no clamp is needed either.
        pg = tl.load(BT + sbt_row + start // PAGE)
        kvp = KV + (pg.to(tl.int64) * skv_p
                    + (start % PAGE + tl.arange(0, BN)).to(tl.int64) * skv_t)[:, None]
    else:
        nc = tl.minimum(n, last)
        pg = tl.load(BT + sbt_row + nc // PAGE)
        row = pg.to(tl.int64) * skv_p + (nc % PAGE).to(tl.int64) * skv_t
        kvp = KV + tl.multiple_of(row, 64)[:, None]

    k0 = _ld(kvp, 0, DC, None, False, PART, R)
    k1 = _ld(kvp, DC, DC, None, False, PART, R)
    k2 = _ld(kvp, 2 * DC, DC, None, False, PART, R)
    k3 = _ld(kvp, 3 * DC, DC, None, False, PART, R)
    kpe = _ld(kvp, R, PE, None, False, False, R)

    qk = tl.dot(q_pe, tl.trans(kpe))
    qk = tl.dot(q0, tl.trans(k0), acc=qk)
    qk = tl.dot(q1, tl.trans(k1), acc=qk)
    qk = tl.dot(q2, tl.trans(k2), acc=qk)
    qk = tl.dot(q3, tl.trans(k3), acc=qk)
    qk = tl.where(n[None, :] < hi, qk * qk_scale2, NEG)

    smx = tl.maximum(smx, qk)
    # Clamped at CL, so p, l_i and acc cannot overflow fp32 for any reference;
    # when the caller's check passes, no element was clamped and this is exactly
    # exp2(qk - m_i) -- an exact softmax numerator against a fixed per-row
    # reference, which is why no accumulator rescale is ever needed.
    p = tl.exp2(tl.minimum(qk - m_i[:, None], CL))
    pb = p.to(kpe.dtype)
    # Row sums via an mma against a ones tile: a tl.sum along the mma-layout N
    # axis is a cross-warp reduction through shared memory.
    l_i = l_i + tl.sum(tl.dot(pb, ones), 1) * 0.125
    a0 = tl.dot(pb, k0, acc=a0)
    a1 = tl.dot(pb, k1, acc=a1)
    a2 = tl.dot(pb, k2, acc=a2)
    a3 = tl.dot(pb, k3, acc=a3)
    return a0, a1, a2, a3, m_i, l_i, smx


@triton.jit
def _mla_decode_split(
    Q, KV, BT, SL, O, PO, PM,
    qk_scale2, out_scale,
    sq_m,
    skv_p, skv_t,
    sbt_b,
    so_m,
    spo_b, spo_s, spo_m,
    spm_b, spm_s, spm_m,
    H, R, QL, RG,
    M: tl.constexpr, DC: tl.constexpr, PE: tl.constexpr,
    PAGE: tl.constexpr, BN: tl.constexpr, SPLIT_LEN: tl.constexpr,
    SINGLE: tl.constexpr, MASK_M: tl.constexpr, STAGES: tl.constexpr,
    PART: tl.constexpr, CL: tl.constexpr, ONEPG: tl.constexpr,
    UNR: tl.constexpr,
):
    # One program owns M query rows of one (request, q position) pair.  q_len and
    # any head count above M are folded into program_id(0) -- they all share the
    # same KV strip, and keeping M pinned at the native mma 16 avoids Triton's
    # M>=64 tcgen05 path, whose tensor-memory budget a 512-wide fp32
    # accumulator cannot fit.
    g = tl.program_id(0)
    s = tl.program_id(1)
    bq = g // RG
    hg = (g % RG) * M
    b = bq // QL

    L = tl.load(SL + b)
    lo = s * SPLIT_LEN
    hi = tl.minimum(lo + SPLIT_LEN, L)

    m_off = tl.arange(0, M)
    qm = (hg + m_off) < H
    d_c = tl.arange(0, DC)

    if lo >= hi:
        # No KV in this split.  Publish a neutral (m, l) so the combine pass
        # skips it and never reads the uninitialized partial accumulator.
        if SINGLE:
            op = O + (bq * H + hg + m_off)[:, None] * so_m
            z = tl.zeros([M, DC], dtype=O.dtype.element_ty)
            for c in range(4):
                d = c * DC + d_c
                tl.store(op + d[None, :], z, mask=qm[:, None] & (d < R)[None, :])
        else:
            pm = PM + g * spm_b + s * spm_s + m_off * spm_m
            tl.store(pm, PM_NEG, mask=qm)
            tl.store(pm + 1, 0.0, mask=qm)
        return

    qp = Q + (bq * H + hg + m_off)[:, None] * sq_m
    q0 = _ld(qp, 0, DC, qm, MASK_M, PART, R)
    q1 = _ld(qp, DC, DC, qm, MASK_M, PART, R)
    q2 = _ld(qp, 2 * DC, DC, qm, MASK_M, PART, R)
    q3 = _ld(qp, 3 * DC, DC, qm, MASK_M, PART, R)
    q_pe = _ld(qp, R, PE, qm, MASK_M, False, R)

    ones = tl.full([BN, 8], 1.0, dtype=Q.dtype.element_ty)
    sbt_row = b * sbt_b
    last = hi - 1
    end = lo + ((hi - lo + BN - 1) // BN) * BN

    zero = tl.zeros([M, DC], dtype=tl.float32)
    a0 = zero
    a1 = zero
    a2 = zero
    a3 = zero
    l_i = tl.zeros([M], dtype=tl.float32)
    m_i = tl.zeros([M], dtype=tl.float32)
    smx = tl.full([M, BN], NEG, dtype=tl.float32)

    # An online running max puts a cross-warp reduce *and* a barrier on the
    # critical path of every tile: ~75 us of 890 here.  A reference held fixed
    # for a whole pass costs neither, but it is only exact while no exponent
    # saturates the CL ceiling -- so don't assume that, prove it:
    #
    #   * `smx` carries the exact max **elementwise**, which is free: every
    #     thread maxes only its own lanes, so there is no reduction and no
    #     barrier inside the loop.  One reduce afterwards turns it into the
    #     exact per-row max `em`, once per pass instead of once per tile;
    #   * the pass is exact iff `|em - m_i| <= CL` for every row.  `em - m_i` is
    #     the largest exponent actually reached, so `<= CL` means `tl.minimum`
    #     never bound anything and `p` is exactly `exp2(qk - m_i)`, while
    #     `>= -CL` keeps the largest `p` at or above `2^-CL` -- neither end of
    #     the fp32 range is touched;
    #   * otherwise re-run with the reference set to `em` itself.  Every
    #     exponent is then `<= 0` with the largest exactly `2^0`, so the second
    #     pass can neither saturate nor underflow, and `smx` is monotone so it
    #     re-derives the same `em` and terminates.  **At most two passes, ever,
    #     and the pass that is accepted is exact.**  (A NaN logit compares false
    #     and is accepted, exactly as the online-max version would.)
    #
    # The retry wraps the *same* loop, so the body is still instantiated once;
    # r1's dead end was a *duplicated* body -- 255 regs, 118 spills, -8%.
    redo = 1
    while redo == 1:
        a0 = zero
        a1 = zero
        a2 = zero
        a3 = zero
        l_i = tl.zeros([M], dtype=tl.float32)
        for start in tl.range(lo, end, BN,
                              num_stages=(STAGES if STAGES > 0 else None),
                              loop_unroll_factor=UNR):
            a0, a1, a2, a3, m_i, l_i, smx = _kv_step(
                a0, a1, a2, a3, m_i, l_i, smx, q0, q1, q2, q3, q_pe,
                KV, BT, sbt_row, skv_p, skv_t, start, hi, last,
                qk_scale2, ones, R, M, PE, PAGE, BN, DC, PART, CL, ONEPG)
        em = tl.max(smx, 1)
        redo = 0
        if tl.max(tl.abs(em - m_i)) > CL:
            m_i = em
            redo = 1

    if SINGLE:
        sc = (out_scale / l_i)[:, None]
        op = O + (bq * H + hg + m_off)[:, None] * so_m
        for c in range(4):
            d = c * DC + d_c
            v = ((a0 if c == 0 else a1 if c == 1 else a2 if c == 2 else a3)
                 * sc).to(O.dtype.element_ty)
            if MASK_M or PART:
                tl.store(op + d[None, :], v, mask=qm[:, None] & (d < R)[None, :])
            else:
                tl.store(op + d[None, :], v)
    else:
        pop = PO + g * spo_b + s * spo_s + m_off[:, None] * spo_m
        pm = PM + g * spm_b + s * spm_s + m_off * spm_m
        for c in range(4):
            d = c * DC + d_c
            v = a0 if c == 0 else a1 if c == 1 else a2 if c == 2 else a3
            if MASK_M or PART:
                tl.store(pop + d[None, :], v, mask=qm[:, None] & (d < R)[None, :])
            else:
                tl.store(pop + d[None, :], v)
        if MASK_M:
            tl.store(pm, m_i, mask=qm)
            tl.store(pm + 1, l_i, mask=qm)
        else:
            tl.store(pm, m_i)
            tl.store(pm + 1, l_i)


@triton.jit
def _mla_combine(
    PO, PM, O, out_scale,
    spo_b, spo_s, spo_m,
    spm_b, spm_s, spm_m,
    so_m,
    H, RG,
    M: tl.constexpr, R: tl.constexpr, NS: tl.constexpr, DC: tl.constexpr,
):
    g = tl.program_id(0)
    m = tl.program_id(1)
    hg = (g % RG) * M + m
    if hg >= H:
        return
    s = tl.arange(0, NS)

    pm = PM + g * spm_b + s * spm_s + m * spm_m
    mv = tl.load(pm)
    lv = tl.load(pm + 1)
    gmax = tl.max(mv)
    w = tl.exp2(mv - gmax)
    inv = out_scale / tl.maximum(tl.sum(lv * w), 1e-30)
    live = mv > -5.0e37

    base = PO + g * spo_b + s[:, None] * spo_s + m * spo_m
    ob = O + ((g // RG) * H + hg) * so_m
    for c in tl.range(0, R, DC):
        d = c + tl.arange(0, DC)
        po = tl.load(base + d[None, :], mask=live[:, None], other=0.0)
        v = tl.sum(w[:, None] * po, 0) * inv
        tl.store(ob + d, v.to(O.dtype.element_ty))


class FlashInferMLADecode(nn.Module):
    """Split-KV flash-decoding MLA over a paged latent KV cache."""

    # Sized for the largest captured decode (997 requests x 4 splits x 16 rows
    # x 512 fp32 partials); grown on demand for anything bigger.
    _WORKSPACE_BYTES = 160 * 1024 * 1024

    def __init__(
        self,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        kv_lora_rank: int,
        workspace: torch.Tensor | None = None,
    ):
        super().__init__()
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.kv_lora_rank = kv_lora_rank
        self._workspace = workspace

    @property
    def available(self) -> bool:
        return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9

    def ensure_workspaces(self, device: torch.device) -> None:
        """Materialize scratch before graph capture: a buffer first allocated
        inside a capture region belongs to that graph's private pool, and later
        graphs replaying against it fault."""
        self._scratch(self._WORKSPACE_BYTES, torch.device(device))

    def _scratch(self, nbytes: int, device: torch.device) -> torch.Tensor:
        ws = self._workspace
        if ws is None or ws.device != device or ws.numel() < nbytes:
            ws = torch.empty(max(nbytes, self._WORKSPACE_BYTES),
                             dtype=torch.uint8, device=device)
            self._workspace = ws
        return ws

    def forward(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        softmax_scale: float,
        max_seq_len: int,
        bmm2_scale: float = 1.0,
    ):
        if kv_cache.dim() == 4:
            kv_cache = kv_cache.squeeze(1)
        B, QL, H, D = q.shape
        R = self.kv_lora_rank
        PE = D - R
        PAGE = kv_cache.shape[-2]
        M = 16
        rg = -(-H // M)
        ngrp = B * QL * rg
        dc = triton.next_power_of_2(-(-R // 4))

        # The kernel indexes q / out with a flat (q_len, head) row index, so the
        # two inner dims must form one dense run.  Contiguous inputs (what the
        # decode path produces) take neither branch.
        if (q.stride(-1) != 1 or q.stride(2) != D
                or (QL > 1 and q.stride(1) != H * D)):
            q = q.contiguous()
        if block_table.stride(-1) != 1:
            block_table = block_table.contiguous()

        out = torch.empty((B, QL, H, R), dtype=q.dtype, device=q.device)

        msl = max(int(max_seq_len), 1)
        small = msl <= SMALL_MSL
        bn = SMALL_BN if small else BLOCK_N
        nwarps = SMALL_WARPS if small else NUM_WARPS
        # Split count: bound per-CTA work (load balance across wildly different
        # cache_seqlens) and, for a small batch, put at least MIN_CTAS CTAs on
        # the machine.  Never split finer than one KV tile -- an extra split is
        # a wasted CTA plus a wider reduction.
        want = max(-(-msl // TOK_PER_SPLIT), -(-MIN_CTAS // B))
        cap = min(MAX_NSPLIT, -(-msl // bn), -(-msl // MIN_TOK_SPLIT))
        ns = 1 << (max(1, min(want, cap)).bit_length() - 1)
        split_len = -(-(-(-msl // ns)) // bn) * bn
        need = max(1, -(-msl // split_len))
        ns = 1 << (need - 1).bit_length() if need > 1 else 1

        qk_scale2 = float(softmax_scale) * 1.4426950408889634
        sq, skv, so = q.stride(), kv_cache.stride(), out.stride()
        common = dict(M=M, DC=dc, PE=PE, PAGE=PAGE, BN=bn, SPLIT_LEN=split_len,
                      MASK_M=(H % M != 0), PART=(4 * dc != R), CL=float(CLAMP),
                      ONEPG=(PAGE % bn == 0),
                      STAGES=(NUM_STAGES if LOOP_RANGE else 0), UNR=UNROLL,
                      num_warps=nwarps,
                      num_stages=(1 if LOOP_RANGE else NUM_STAGES))

        if ns == 1:
            _mla_decode_split[(ngrp, 1)](
                q, kv_cache, block_table, cache_seqlens, out, out, out,
                qk_scale2, float(bmm2_scale),
                sq[2], skv[0], skv[1], block_table.stride(0),
                so[2], 0, 0, 0, 0, 0, 0, H, R, QL, rg,
                SINGLE=True, **common,
            )
            return out, None

        po_elems = ngrp * ns * M * R
        pm_elems = ngrp * ns * M * 2
        flat = self._scratch((po_elems + pm_elems) * 4, q.device).view(torch.float32)
        po = flat[:po_elems].view(ngrp, ns, M, R)
        pm = flat[po_elems:po_elems + pm_elems].view(ngrp, ns, M, 2)

        _mla_decode_split[(ngrp, ns)](
            q, kv_cache, block_table, cache_seqlens, out, po, pm,
            qk_scale2, float(bmm2_scale),
            sq[2], skv[0], skv[1], block_table.stride(0),
            so[2],
            po.stride(0), po.stride(1), po.stride(2),
            pm.stride(0), pm.stride(1), pm.stride(2), H, R, QL, rg,
            SINGLE=False, **common,
        )

        cdc = min(R, max(32, 2048 // ns))
        _mla_combine[(ngrp, M)](
            po, pm, out, float(bmm2_scale),
            po.stride(0), po.stride(1), po.stride(2),
            pm.stride(0), pm.stride(1), pm.stride(2),
            so[2], H, rg,
            M=M, R=R, NS=ns, DC=cdc,
            num_warps=COMBINE_WARPS, num_stages=1,
        )
        return out, None
