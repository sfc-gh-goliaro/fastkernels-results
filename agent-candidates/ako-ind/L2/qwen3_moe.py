"""Qwen3 Mixture-of-Experts block with two fused FP8 expert-compute paths.

FP8 W8A8 block-scaled expert weights (128x128 weight blocks, per-token-group
activation scales, FP32 accumulation).  Which path runs is decided per shape by
the mean number of routed rows per expert, because the two have different cost
floors and cross over at about one 128-row alignment block per expert:

**A. DeepGEMM m-grouped contiguous** (mean rows/expert >= 128, e.g. M=16384).
DeepGEMM's grouped FP8 GEMM runs at ~2500/2200 TFLOPS here -- roughly 2.6x the
hand-written Triton GEMMs below -- but it needs its A operand permuted into one
``[M_sum, K]`` buffer whose per-expert spans start on 128-row boundaries.  The
reference builds that layout out of separate torch ops (argsort + searchsorted +
scatter + a full zero-fill of the padded buffer) and ends up spending more time
moving data than computing.  Here the layout is produced by fused kernels that
write straight into the formats DeepGEMM wants:

  1. metadata            per-expert counts -> 128-aligned offsets -> a
                         destination row per (token, slot), plus DeepGEMM's
                         per-row expert-id vector.  Four tiny kernels, no
                         argsort, no searchsorted, no host sync.
  2. ``_scatter_quant8`` one pass: gather, per-128-group FP8 quantization, and
                         scatter into the m-grouped buffer, emitting the scale
                         directly in DeepGEMM's packed-UE8M0 mn-major layout
  3. DeepGEMM GEMM1
  4. ``_silu_quant_dg``  SiLU-mul + FP8 requantization, emitted in the same
                         packed layout for GEMM2, skipping padding rows
  5. DeepGEMM GEMM2
  6. ``_reduce_dg``      fused unpermute + top-k weighted reduction (replaces
                         the reference's [M, top_k, K] gather plus ``moe_sum``)

**B. Fused Triton chain** (everything below that, down to M=1).  Nothing is
permuted at all: GEMM1 gathers its A rows straight out of the quantized
activations, and the padding granularity is BM=64 instead of 128.

  ``_quant_act`` -> metadata -> ``_gemm1s`` + ``_silu_quant`` (or ``_gemm1``
  with the SiLU-mul + requant fused into its epilogue, and the activation
  quantization folded in as well at tiny M) -> ``_gemm2`` -> ``_reduce``

Neither path uses ``moe_align`` block padding, an ``M_sum x K`` zero-filled
activation buffer, ``argsort``/``searchsorted``, a host sync, or a separate
``silu_and_mul`` / activation-requant / ``moe_sum`` pass.

The two paths deliberately differ in *where* the routed weight is applied,
following whichever the reference itself uses at that shape: path A stores the
unweighted bf16 per-expert output and applies ``topk_weight`` in fp32 during the
reduce (matching the reference's DeepGEMM path, which makes path A bit-exact
against it), while path B folds the weight into GEMM2's epilogue before the bf16
store once the reference has fallen back to its own Triton kernel.  bf16
rounding of GEMM2's output is the dominant error term at these magnitudes, so
getting its placement wrong shows up as a hard correctness failure, not as
noise.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import triton
import triton.language as tl

from ....infra.tp import _tp_rank, _tp_size
from ..L1.allreduce import AllReduce
from ..L1.topk_softmax import TopKSoftmax
from ..L2.fused_experts import FusedExperts
from ..L2.parallel_linear import ReplicatedLinear

_FP8_BLOCK = 128
_FP8_MAX = 448.0
_QUANT_EPS = 1e-10

_FMAX = tl.constexpr(448.0)
_RFMAX = tl.constexpr(1.0 / 448.0)
_EPS = tl.constexpr(1e-10)


# ---------------------------------------------------------------------------
# Scratch buffers.  One set shared by every layer (layers run sequentially),
# grow-only so steady state does no allocation at all.
# ---------------------------------------------------------------------------
class _Scratch:
    def __init__(self):
        self.bufs: dict[str, torch.Tensor] = {}

    def get(self, key, numel, dtype, device, zero=False):
        b = self.bufs.get(key)
        if b is None or b.numel() < numel or b.dtype != dtype or b.device != device:
            b = (torch.zeros if zero else torch.empty)(
                numel, dtype=dtype, device=device)
            self.bufs[key] = b
        return b[:numel]


_SCRATCH = _Scratch()


# ---------------------------------------------------------------------------
# 1) per-token-group FP8 activation quantization
# ---------------------------------------------------------------------------
@triton.jit
def _quant_act(x_ptr, q_ptr, s_ptr, M, K: tl.constexpr, NG: tl.constexpr,
               BM: tl.constexpr, G: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_g = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    vm = rm < M
    cols = pid_g * G + tl.arange(0, G)
    p = rm[:, None] * K + cols[None, :]
    x = tl.load(x_ptr + p, mask=vm[:, None], other=0.0).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x), axis=1), _EPS)
    sc = tl.math.exp2(tl.math.ceil(tl.math.log2(amax * _RFMAX)))
    q = tl.clamp(x / sc[:, None], -_FMAX, _FMAX)
    tl.store(q_ptr + p, q.to(q_ptr.dtype.element_ty), mask=vm[:, None])
    tl.store(s_ptr + rm * NG + pid_g, sc, mask=vm)


# ---------------------------------------------------------------------------
# 2) routing metadata: counts -> offsets -> expert-grouped slot list
# ---------------------------------------------------------------------------
@triton.jit
def _meta_count(ids_ptr, cnt_ptr, NT, E: tl.constexpr, EP: tl.constexpr,
                BLK: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BLK + tl.arange(0, BLK)
    m = off < NT
    ids = tl.load(ids_ptr + off, mask=m, other=E)
    tl.atomic_add(cnt_ptr + ids, 1, mask=m)


@triton.jit
def _meta_scan(cnt_ptr, cnt2_ptr, off_ptr, cur_ptr, blkoff_ptr, E: tl.constexpr,
               EP: tl.constexpr, BM: tl.constexpr):
    idx = tl.arange(0, EP)
    m = idx < E
    c = tl.load(cnt_ptr + idx, mask=m, other=0)
    # Republish the histogram and reset the atomic accumulator in the same pass,
    # so the next call needs no separate zero-fill launch.
    tl.store(cnt2_ptr + idx, c, mask=m)
    tl.store(cnt_ptr + idx, tl.zeros((EP,), dtype=tl.int32), mask=m)
    cs = tl.cumsum(c, axis=0)
    o = cs - c
    tl.store(off_ptr + idx, o, mask=m)
    tl.store(off_ptr + E, tl.sum(c, axis=0))
    tl.store(cur_ptr + idx, o, mask=m)
    nb = (c + (BM - 1)) // BM
    bs = tl.cumsum(nb, axis=0)
    tl.store(blkoff_ptr + idx, bs - nb, mask=m)
    tl.store(blkoff_ptr + E, tl.sum(nb, axis=0))


@triton.jit
def _meta_fill(ids_ptr, cur_ptr, slot_ptr, NT, E: tl.constexpr,
               BLK: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BLK + tl.arange(0, BLK)
    m = off < NT
    ids = tl.load(ids_ptr + off, mask=m, other=E)
    pos = tl.atomic_add(cur_ptr + ids, 1, mask=m)
    tl.store(slot_ptr + pos, off, mask=m)


@triton.jit
def _meta_small(ids_ptr, cnt_ptr, off_ptr, slot_ptr, blkoff_ptr, NT,
                E: tl.constexpr, EP: tl.constexpr, NTP: tl.constexpr,
                BM: tl.constexpr):
    """Single-CTA, atomic-free metadata for tiny token counts."""
    i = tl.arange(0, NTP)
    vi = i < NT
    ids = tl.load(ids_ptr + i, mask=vi, other=E)
    ei = tl.arange(0, EP)
    em = ei < E
    # counts
    hit = (ids[:, None] == ei[None, :]) & vi[:, None]
    c = tl.sum(hit.to(tl.int32), axis=0)
    cs = tl.cumsum(c, axis=0)
    o = cs - c
    tl.store(cnt_ptr + ei, c, mask=em)
    tl.store(off_ptr + ei, o, mask=em)
    tl.store(off_ptr + E, tl.sum(c, axis=0))
    nb = (c + (BM - 1)) // BM
    bs = tl.cumsum(nb, axis=0)
    tl.store(blkoff_ptr + ei, bs - nb, mask=em)
    tl.store(blkoff_ptr + E, tl.sum(nb, axis=0))
    # rank of each slot inside its expert (stable: ascending flat index)
    same = (ids[:, None] == ids[None, :]) & (i[None, :] < i[:, None])
    rank = tl.sum((same & vi[None, :]).to(tl.int32), axis=1)
    base = tl.sum(tl.where(hit, o[None, :], 0), axis=1)
    tl.store(slot_ptr + base + rank, i, mask=vi)


# ---------------------------------------------------------------------------
# 3) GEMM1: gathered A, gate+up side by side, SiLU-mul + FP8 requant epilogue
# ---------------------------------------------------------------------------
@triton.jit
def _gemm1(a_ptr, as_ptr, w_ptr, ws_ptr, hq_ptr, hs_ptr,
           slot_ptr, cnt_ptr, off_ptr, blkoff_ptr,
           K: tl.constexpr, N: tl.constexpr, E: tl.constexpr,
           EP: tl.constexpr, TOPK: tl.constexpr,
           NG: tl.constexpr, NGH: tl.constexpr, WSK: tl.constexpr,
           WSN: tl.constexpr, NTT,
           BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
           ASV: tl.constexpr = 0, QI: tl.constexpr = False):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    idx = tl.arange(0, EP)
    bo = tl.load(blkoff_ptr + idx, mask=idx < E, other=0x7FFFFFFF)
    tot = tl.load(blkoff_ptr + E)
    if pid_m >= tot:
        return
    sel = bo <= pid_m
    e = tl.sum(sel.to(tl.int32), axis=0) - 1
    bo_e = tl.max(tl.where(sel, bo, -1), axis=0)
    m0 = (pid_m - bo_e) * BM
    cnt_e = tl.load(cnt_ptr + e)
    off_e = tl.load(off_ptr + e)

    rm = m0 + tl.arange(0, BM)
    vm = rm < cnt_e
    flat = tl.load(slot_ptr + off_e + rm, mask=vm, other=0)
    tok = flat // TOPK

    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    NKB: tl.constexpr = K // BK
    kv = tl.arange(0, NKB)

    a_ptrs = a_ptr + tok[:, None] * K + offs_k[None, :]
    wb = w_ptr + e.to(tl.int64) * (2 * N * K)
    bg_ptrs = wb + offs_n[None, :] * K + offs_k[:, None]
    bu_ptrs = bg_ptrs + N * K

    # Every scale this block will ever need, loaded ONCE into registers.  A
    # global load *inside* the k-loop is not covered by Triton's cp.async
    # multi-stage pipeline, so its ~500-cycle latency lands directly on the
    # accumulator update and stalls the next MMA -- measured ~2x slower.
    wsb = ws_ptr + e.to(tl.int64) * (WSN * WSK)
    as_row = as_ptr + tok * NG
    if QI:
        pass
    elif ASV == 0:
        asc_all = tl.load(as_ptr + tok[:, None] * NG + kv[None, :],
                          mask=vm[:, None], other=0.0)
    bg_all = tl.load(wsb + ((pid_n * BN) // 128) * WSK + kv)
    bu_all = tl.load(wsb + ((N + pid_n * BN) // 128) * WSK + kv)

    accg = tl.zeros((BM, BN), dtype=tl.float32)
    accu = tl.zeros((BM, BN), dtype=tl.float32)
    for kb in range(0, NKB):
        bg = tl.load(bg_ptrs)
        bu = tl.load(bu_ptrs)
        pick = kv == kb
        if QI:
            # Tiny-M path: quantize the gathered activation group in-register so
            # the separate quant kernel (a whole launch) disappears.  BK == the
            # 128-element quantization group, so this is bit-identical to it.
            xb = tl.load(a_ptrs, mask=vm[:, None], other=0.0).to(tl.float32)
            xam = tl.maximum(tl.max(tl.abs(xb), axis=1), _EPS)
            sa = tl.math.exp2(tl.math.ceil(tl.math.log2(xam * _RFMAX)))
            a = tl.clamp(xb / sa[:, None], -_FMAX,
                         _FMAX).to(hq_ptr.dtype.element_ty)
        else:
            a = tl.load(a_ptrs, mask=vm[:, None], other=0.0)
            if ASV == 0:
                sa = tl.sum(tl.where(pick[None, :], asc_all, 0.0), axis=1)
            else:
                sa = tl.load(as_row + kb, mask=vm, other=0.0)
        gsl = tl.sum(tl.where(pick, bg_all, 0.0), axis=0)
        usl = tl.sum(tl.where(pick, bu_all, 0.0), axis=0)
        accg += tl.dot(a, bg) * (sa * gsl)[:, None]
        accu += tl.dot(a, bu) * (sa * usl)[:, None]
        a_ptrs += BK
        bg_ptrs += BK
        bu_ptrs += BK

    g = accg.to(tl.bfloat16).to(tl.float32)
    ub = accu.to(tl.bfloat16)
    sig = (g / (1.0 + tl.exp(-g))).to(tl.bfloat16)
    y = (sig * ub).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(y), axis=1), _EPS)
    sc = tl.math.exp2(tl.math.ceil(tl.math.log2(amax * _RFMAX)))
    yq = tl.clamp(y / sc[:, None], -_FMAX, _FMAX)

    prow = off_e + rm
    tl.store(hq_ptr + prow[:, None] * N + offs_n[None, :],
             yq.to(hq_ptr.dtype.element_ty), mask=vm[:, None])
    # ``hs`` is stored transposed ([N/128, NT]) so GEMM2's per-k scale load is a
    # single coalesced 4*BM-byte access instead of a strided one.
    tl.store(hs_ptr + pid_n * NTT + prow, sc, mask=vm)


# --- alternative GEMM1: single accumulator over the full 2N, SiLU-mul + fp8
# requantization split into its own pass (A/B against the fused epilogue) ---
@triton.jit
def _gemm1s(a_ptr, as_ptr, w_ptr, ws_ptr, o_ptr,
            slot_ptr, cnt_ptr, off_ptr, blkoff_ptr,
            K: tl.constexpr, N2: tl.constexpr, E: tl.constexpr,
            EP: tl.constexpr, TOPK: tl.constexpr,
            NG: tl.constexpr, WSK: tl.constexpr, WSN: tl.constexpr,
            BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    idx = tl.arange(0, EP)
    bo = tl.load(blkoff_ptr + idx, mask=idx < E, other=0x7FFFFFFF)
    tot = tl.load(blkoff_ptr + E)
    if pid_m >= tot:
        return
    sel = bo <= pid_m
    e = tl.sum(sel.to(tl.int32), axis=0) - 1
    bo_e = tl.max(tl.where(sel, bo, -1), axis=0)
    m0 = (pid_m - bo_e) * BM
    cnt_e = tl.load(cnt_ptr + e)
    off_e = tl.load(off_ptr + e)
    rm = m0 + tl.arange(0, BM)
    vm = rm < cnt_e
    flat = tl.load(slot_ptr + off_e + rm, mask=vm, other=0)
    tok = flat // TOPK
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    NKB: tl.constexpr = K // BK
    kv = tl.arange(0, NKB)
    a_ptrs = a_ptr + tok[:, None] * K + offs_k[None, :]
    wb = w_ptr + e.to(tl.int64) * (N2 * K)
    b_ptrs = wb + offs_n[None, :] * K + offs_k[:, None]
    asc_all = tl.load(as_ptr + tok[:, None] * NG + kv[None, :],
                      mask=vm[:, None], other=0.0)
    bs_all = tl.load(ws_ptr + e.to(tl.int64) * (WSN * WSK)
                     + ((pid_n * BN) // 128) * WSK + kv)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for kb in range(0, NKB):
        a = tl.load(a_ptrs, mask=vm[:, None], other=0.0)
        b = tl.load(b_ptrs)
        pick = kv == kb
        sa = tl.sum(tl.where(pick[None, :], asc_all, 0.0), axis=1)
        sb = tl.sum(tl.where(pick, bs_all, 0.0), axis=0)
        acc += tl.dot(a, b) * (sa * sb)[:, None]
        a_ptrs += BK
        b_ptrs += BK
    prow = off_e + rm
    tl.store(o_ptr + prow[:, None] * N2 + offs_n[None, :],
             acc.to(o_ptr.dtype.element_ty), mask=vm[:, None])


@triton.jit
def _silu_quant(y_ptr, hq_ptr, hs_ptr, NT, NTT, N: tl.constexpr,
                BM: tl.constexpr, G: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    vm = rm < NT
    cols = pid_n * G + tl.arange(0, G)
    gp = y_ptr + rm[:, None] * (2 * N) + cols[None, :]
    g = tl.load(gp, mask=vm[:, None], other=0.0).to(tl.float32)
    ub = tl.load(gp + N, mask=vm[:, None], other=0.0)
    sig = (g / (1.0 + tl.exp(-g))).to(tl.bfloat16)
    y = (sig * ub).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(y), axis=1), _EPS)
    sc = tl.math.exp2(tl.math.ceil(tl.math.log2(amax * _RFMAX)))
    yq = tl.clamp(y / sc[:, None], -_FMAX, _FMAX)
    tl.store(hq_ptr + rm[:, None] * N + cols[None, :],
             yq.to(hq_ptr.dtype.element_ty), mask=vm[:, None])
    tl.store(hs_ptr + pid_n * NTT + rm, sc, mask=vm)


# ---------------------------------------------------------------------------
# 4) GEMM2: contiguous A, scatter rows back to token-major order
# ---------------------------------------------------------------------------
@triton.jit
def _gemm2(hq_ptr, hs_ptr, w_ptr, ws_ptr, o_ptr, tw_ptr,
           slot_ptr, cnt_ptr, off_ptr, blkoff_ptr,
           K: tl.constexpr, N: tl.constexpr, E: tl.constexpr,
           EP: tl.constexpr, NGH: tl.constexpr, WSK: tl.constexpr,
           WSN: tl.constexpr, NTT,
           BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
           MULW: tl.constexpr, ASV: tl.constexpr = 0):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    idx = tl.arange(0, EP)
    bo = tl.load(blkoff_ptr + idx, mask=idx < E, other=0x7FFFFFFF)
    tot = tl.load(blkoff_ptr + E)
    if pid_m >= tot:
        return
    sel = bo <= pid_m
    e = tl.sum(sel.to(tl.int32), axis=0) - 1
    bo_e = tl.max(tl.where(sel, bo, -1), axis=0)
    m0 = (pid_m - bo_e) * BM
    cnt_e = tl.load(cnt_ptr + e)
    off_e = tl.load(off_ptr + e)

    rm = m0 + tl.arange(0, BM)
    vm = rm < cnt_e
    prow = off_e + rm

    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    NKB: tl.constexpr = N // BK
    NKP: tl.constexpr = 16 if NKB <= 16 else 32
    kv = tl.arange(0, NKP)
    km = kv < NKB

    a_ptrs = hq_ptr + prow[:, None] * N + offs_k[None, :]
    wb = w_ptr + e.to(tl.int64) * (K * N)
    b_ptrs = wb + offs_n[None, :] * N + offs_k[:, None]

    # Hoist all per-k scales out of the loop (see _gemm1).
    if ASV == 0:
        asc_all = tl.load(hs_ptr + kv[None, :] * NTT + prow[:, None],
                          mask=vm[:, None] & km[None, :], other=0.0)
    wsb = ws_ptr + e.to(tl.int64) * (WSN * WSK) + ((pid_n * BN) // 128) * WSK
    bsc_all = tl.load(wsb + kv, mask=km, other=0.0)

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for kb in range(0, NKB):
        a = tl.load(a_ptrs, mask=vm[:, None], other=0.0)
        b = tl.load(b_ptrs)
        pick = kv == kb
        if ASV == 0:
            sa = tl.sum(tl.where(pick[None, :], asc_all, 0.0), axis=1)
        else:
            sa = tl.load(hs_ptr + kb * NTT + prow, mask=vm, other=0.0)
        sb = tl.sum(tl.where(pick, bsc_all, 0.0), axis=0)
        acc += tl.dot(a, b) * (sa * sb)[:, None]
        a_ptrs += BK
        b_ptrs += BK

    flat = tl.load(slot_ptr + off_e + rm, mask=vm, other=0)
    if MULW:
        # Mirror the reference Triton MoE path, which folds the routed weight
        # into the GEMM epilogue *before* the bf16 store.
        acc = acc * tl.load(tw_ptr + flat, mask=vm, other=0.0)[:, None]
    tl.store(o_ptr + flat[:, None] * K + offs_n[None, :],
             acc.to(o_ptr.dtype.element_ty), mask=vm[:, None])


# ---------------------------------------------------------------------------
# 5) top-k weighted cross-expert reduction
# ---------------------------------------------------------------------------
@triton.jit
def _reduce(i3_ptr, tw_ptr, o_ptr, M, K: tl.constexpr, TOPK: tl.constexpr,
            BK: tl.constexpr, USEW: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    s = tl.arange(0, TOPK)
    cols = pid_n * BK + tl.arange(0, BK)
    base = pid_m * TOPK
    t = tl.load(i3_ptr + (base + s)[:, None] * K + cols[None, :]).to(tl.float32)
    if USEW:
        w = tl.load(tw_ptr + base + s).to(tl.float32)
        t = t * w[:, None]
    r = tl.sum(t, axis=0)
    tl.store(o_ptr + pid_m * K + cols, r.to(o_ptr.dtype.element_ty))


# ---------------------------------------------------------------------------
# 6) DeepGEMM m-grouped-contiguous path
#
# DeepGEMM's grouped FP8 GEMM runs at ~2500/2200 TFLOPS on these shapes, ~2.6x
# the hand-written Triton GEMMs above, but it needs its A operand permuted into
# one ``[M_sum, K]`` buffer whose per-expert spans start on 128-row boundaries.
# The reference builds that layout in separate torch ops (argsort +
# searchsorted + scatter + a full zero-fill) and spends more time moving data
# than computing.  Here the whole layout is produced by two fused kernels that
# write straight into DeepGEMM's required formats, including the packed-UE8M0
# scale-factor layout (mn-major ``int32``, four 128-element k-groups packed per
# word, byte == the fp32 biased-exponent field of the power-of-two scale --
# verified bit-identical to ``get_mn_major_tma_aligned_packed_ue8m0_tensor``).
# ---------------------------------------------------------------------------
@triton.jit
def _meta_scan_dg(cnt_ptr, cnt2_ptr, aoff_ptr, cur_ptr, E: tl.constexpr,
                  EP: tl.constexpr, ALIGN: tl.constexpr):
    """Histogram -> ALIGN-aligned per-expert row offsets, in one CTA."""
    idx = tl.arange(0, EP)
    m = idx < E
    c = tl.load(cnt_ptr + idx, mask=m, other=0)
    tl.store(cnt2_ptr + idx, c, mask=m)
    tl.store(cnt_ptr + idx, tl.zeros((EP,), dtype=tl.int32), mask=m)
    ac = ((c + (ALIGN - 1)) // ALIGN) * ALIGN
    cs = tl.cumsum(ac, axis=0)
    o = cs - ac
    tl.store(aoff_ptr + idx, o, mask=m)
    tl.store(aoff_ptr + E, tl.sum(ac, axis=0))
    tl.store(cur_ptr + idx, o, mask=m)


@triton.jit
def _meta_fill_dg(ids_ptr, cur_ptr, dest_ptr, NT, E: tl.constexpr,
                  BLK: tl.constexpr):
    """dest[flat] = the row of the m-grouped buffer this (token, slot) lands on."""
    pid = tl.program_id(0)
    off = pid * BLK + tl.arange(0, BLK)
    m = off < NT
    ids = tl.load(ids_ptr + off, mask=m, other=E)
    pos = tl.atomic_add(cur_ptr + ids, 1, mask=m)
    tl.store(dest_ptr + off, pos, mask=m)


@triton.jit
def _meta_idx(aoff_ptr, cnt_ptr, mi_ptr, E: tl.constexpr, EP: tl.constexpr,
              ALIGN: tl.constexpr):
    """Per-row expert id for DeepGEMM (-1 = skip).  Expert spans are
    ALIGN-aligned, so one program owns exactly one expert's ALIGN-row block and
    resolves it with a register reduction instead of a searchsorted."""
    pid = tl.program_id(0)
    r0 = pid * ALIGN
    rows = r0 + tl.arange(0, ALIGN)
    idx = tl.arange(0, EP)
    ao = tl.load(aoff_ptr + idx, mask=idx < E, other=0x7FFFFFFF)
    tot = tl.load(aoff_ptr + E)
    if r0 >= tot:
        tl.store(mi_ptr + rows, tl.full((ALIGN,), -1, tl.int32))
        return
    sel = ao <= r0
    e = tl.sum(sel.to(tl.int32), axis=0) - 1
    ao_e = tl.max(tl.where(sel, ao, -1), axis=0)
    c = tl.load(cnt_ptr + e)
    tl.store(mi_ptr + rows, tl.where(rows - ao_e < c, e, -1))


@triton.jit
def _scatter_quant(x_ptr, dest_ptr, aq_ptr, sf_ptr, NT, AMS,
                   K: tl.constexpr, TOPK: tl.constexpr, NG: tl.constexpr,
                   KP: tl.constexpr):
    """Fused gather + per-128-group FP8 quantization + scatter into the
    m-grouped contiguous buffer, writing the scale straight into DeepGEMM's
    packed-UE8M0 mn-major layout.  One program per (token, slot): the eight
    slots of a token are consecutive program ids, so the bf16 source row is
    read once out of L2 for all eight copies."""
    pid = tl.program_id(0)
    if pid >= NT:
        return
    tok = pid // TOPK
    d = tl.load(dest_ptr + pid).to(tl.int64)
    g = tl.arange(0, NG)[:, None] * 128 + tl.arange(0, 128)[None, :]
    x = tl.load(x_ptr + tok.to(tl.int64) * K + g).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x), axis=1), _EPS)
    sc = tl.math.exp2(tl.math.ceil(tl.math.log2(amax * _RFMAX)))
    q = tl.clamp(x / sc[:, None], -_FMAX, _FMAX)
    tl.store(aq_ptr + d * K + g, q.to(aq_ptr.dtype.element_ty))
    b = (sc.to(tl.int32, bitcast=True) >> 23) & 0xFF
    packed = tl.sum(tl.reshape(b, (KP, 4)) << (tl.arange(0, 4) * 8)[None, :],
                    axis=1)
    tl.store(sf_ptr + tl.arange(0, KP).to(tl.int64) * AMS + d, packed)


@triton.jit
def _scatter_quant8(x_ptr, dest_ptr, aq_ptr, sf_ptr, M, AMS,
                    K: tl.constexpr, TOPK: tl.constexpr, NG: tl.constexpr,
                    KP: tl.constexpr):
    """Same as :func:`_scatter_quant`, but one program per *token*: the bf16 row
    is read and quantized once and the FP8 result is broadcast to all TOPK
    destination rows.  Cuts the read side of the scatter from TOPK*M*K to M*K
    (1.07 GB -> 134 MB at M=16384) at the cost of eight stores per program."""
    tok = tl.program_id(0)
    g = tl.arange(0, NG)[:, None] * 128 + tl.arange(0, 128)[None, :]
    x = tl.load(x_ptr + tok.to(tl.int64) * K + g).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x), axis=1), _EPS)
    sc = tl.math.exp2(tl.math.ceil(tl.math.log2(amax * _RFMAX)))
    q = tl.clamp(x / sc[:, None], -_FMAX, _FMAX).to(aq_ptr.dtype.element_ty)
    b = (sc.to(tl.int32, bitcast=True) >> 23) & 0xFF
    packed = tl.sum(tl.reshape(b, (KP, 4)) << (tl.arange(0, 4) * 8)[None, :],
                    axis=1)
    kp = tl.arange(0, KP).to(tl.int64) * AMS
    for s in tl.static_range(TOPK):
        d = tl.load(dest_ptr + tok * TOPK + s).to(tl.int64)
        tl.store(aq_ptr + d * K + g, q)
        tl.store(sf_ptr + kp + d, packed)


@triton.jit
def _silu_quant_dg(y_ptr, mi_ptr, hq_ptr, sf_ptr, AMS, N: tl.constexpr,
                   BM: tl.constexpr, ALIGN: tl.constexpr):
    """SiLU-mul + FP8 requantization of GEMM1's output, emitted directly in
    DeepGEMM's layout for GEMM2.  Each program owns 512 columns (four 128-element
    quantization groups) so it can pack a whole scale word, and skips blocks
    whose rows are alignment padding."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    r0 = pid_m * BM
    if tl.load(mi_ptr + r0) < 0:
        return
    rows = (r0 + tl.arange(0, BM)).to(tl.int64)
    # Rows past an expert's real count are alignment padding that GEMM1 never
    # wrote; mask them so the requantized tail is a deterministic zero instead
    # of whatever silu() makes of uninitialized memory.
    vm = tl.load(mi_ptr + rows) >= 0
    cols = pid_n * 512 + tl.arange(0, 4)[:, None] * 128 + tl.arange(0, 128)[None, :]
    p = y_ptr + rows[:, None, None] * (2 * N) + cols[None, :, :]
    g = tl.load(p, mask=vm[:, None, None], other=0.0).to(tl.float32)
    ub = tl.load(p + N, mask=vm[:, None, None], other=0.0)
    sig = (g / (1.0 + tl.exp(-g))).to(tl.bfloat16)
    y = (sig * ub).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(y), axis=2), _EPS)
    sc = tl.math.exp2(tl.math.ceil(tl.math.log2(amax * _RFMAX)))
    yq = tl.clamp(y / sc[:, :, None], -_FMAX, _FMAX)
    tl.store(hq_ptr + rows[:, None, None] * N + cols[None, :, :],
             yq.to(hq_ptr.dtype.element_ty))
    b = (sc.to(tl.int32, bitcast=True) >> 23) & 0xFF
    packed = tl.sum(b << (tl.arange(0, 4) * 8)[None, :], axis=1)
    tl.store(sf_ptr + pid_n * AMS + rows, packed)


@triton.jit
def _reduce_dg(m2_ptr, dest_ptr, tw_ptr, o_ptr, K: tl.constexpr,
               TOPK: tl.constexpr, BK: tl.constexpr):
    """Gather the top-k per-expert rows back to token order and reduce them with
    the routed weights in fp32 -- replaces the reference's separate unpermute
    (a full [M, top_k, K] materialization) plus moe_sum."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    s = tl.arange(0, TOPK)
    base = pid_m * TOPK
    d = tl.load(dest_ptr + base + s).to(tl.int64)
    cols = pid_n * BK + tl.arange(0, BK)
    t = tl.load(m2_ptr + d[:, None] * K + cols[None, :]).to(tl.float32)
    w = tl.load(tw_ptr + base + s).to(tl.float32)
    r = tl.sum(t * w[:, None], axis=0)
    tl.store(o_ptr + pid_m.to(tl.int64) * K + cols, r.to(o_ptr.dtype.element_ty))


# ---------------------------------------------------------------------------
def _deepgemm_layout_ok(hidden_states, w13, w2) -> bool:
    try:
        from ..L1.moe_grouped_gemm import _valid_deep_gemm
    except Exception:
        return hidden_states.size(0) >= 128
    return bool(_valid_deep_gemm(hidden_states, w13, w2)
                and not torch.cuda.is_current_stream_capturing())


# --- DeepGEMM handles, resolved once -----------------------------------------
try:
    import deep_gemm as _dg

    _DG_ALIGN = int(_dg.get_mk_alignment_for_contiguous_layout())
    _DG_GEMM = _dg.m_grouped_fp8_gemm_nt_contiguous
    _DG_XFORM = _dg.transform_sf_into_required_layout
except Exception:  # pragma: no cover - DeepGEMM absent
    _dg = None
    _DG_ALIGN = 128
    _DG_GEMM = None
    _DG_XFORM = None


def _dg_wscale(cache, tag, ws, mn, k, e):
    """DeepGEMM's B-side scale layout, transformed once and cached.  Expert
    weights never change after loading, so paying this per call (as the
    reference does) is pure overhead."""
    key = (ws.data_ptr(), ws._version, mn, k)
    hit = cache.get(tag)
    if hit is not None and hit[0] == key:
        return hit[1]
    t = _DG_XFORM(ws, mn=mn, k=k, recipe=(1, _FP8_BLOCK, _FP8_BLOCK),
                  num_groups=e, is_sfa=False)
    cache[tag] = (key, t)
    return t


# Tile / launch knobs, all in one place so the dev sweeps can poke them without
# editing kernels.  Values are what the sweeps in ITERATIONS.md landed on.
_TUNE = {
    # BM=16 (rather than 64) while there is less than this many tokens/expert
    "bm_small_nt": 4.0,
    # BM -> (num_warps, num_stages) for GEMM1 / GEMM2
    "g1": {16: (4, 3), 32: (4, 4), 64: (4, 4), 128: (4, 4)},
    "g2": {16: (4, 4), 32: (4, 3), 64: (4, 4), 128: (8, 3)},
    "bn2": 128,          # GEMM2 output-tile width
    "bkr": 1024,         # _reduce column tile
    "rw": 2,             # _reduce num_warps (fewer is strictly better here)
    "asv1": 1,           # GEMM1 a-scale: 1 = per-k load, 0 = register tile
    "asv2": 0,           # GEMM2 a-scale: 0 = register tile wins here
    "g1mode": 1,         # 1 = split GEMM1 + separate SiLU-mul/requant pass
    "g1mode_min": 1024,  # ...but only from this many (token, expert) pairs up
    "qi_nt": 64,         # fold activation quant into GEMM1 below this NT
    "force_bm": 0,       # sweep override only
    "dg": 1,             # enable the DeepGEMM path at all
    # DeepGEMM's cost floor is E*ALIGN padded rows regardless of M, so it only
    # wins once the mean rows-per-expert reaches roughly one alignment block.
    # There its padding is at most 2x and its ~2.5x MMA advantage dominates;
    # below about half that, the Triton chain measures faster (M=643: 332 vs
    # 372 us, M=314: 517 vs 542 us) because it pads to BM=64 rather than
    # ALIGN=128 and never materializes a permuted copy at all.
    "dg_min_per_e": _FP8_BLOCK,
    "sq_bm": 16,         # _silu_quant_dg rows per program
    "bkrd": 1024,        # _reduce_dg column tile
    "rwd": 2,            # _reduce_dg num_warps
    "sqw": 8,            # _scatter_quant num_warps
    "sq8": 1,            # 1 = one program per token (single read), 0 = per slot
    "sqs": 4,            # _silu_quant_dg num_warps
    # sweep overrides (0 = use the tables above)
    "g1w": 0, "g1s": 0, "g2w": 0, "g2s": 0, "bn1": 128, "bn1f": 128,
}


def _ws(tag, bm):
    """(num_warps, num_stages) for GEMM tag at block size bm, sweep-overridable."""
    w, st = _TUNE[tag][bm]
    return _TUNE[tag + "w"] or w, _TUNE[tag + "s"] or st


def _pick_bm(nt: int, e: int) -> int:
    """Rows per expert-block.  BM=64 is the sweet spot for every shape with real
    per-expert occupancy (measured against 16/32/128 at M=314..16384); only the
    near-empty case, where each expert holds ~1 token, prefers the smallest
    tile."""
    if _TUNE["force_bm"]:
        return _TUNE["force_bm"]
    return 16 if nt <= _TUNE["bm_small_nt"] * e else 64


class Qwen3MoE(nn.Module):
    """Qwen3 Mixture-of-Experts with a fused Triton grouped-GEMM chain.

    Weights (FP8 mode):
      gate:     [num_experts, hidden_size] (bfloat16, replicated)
      w13:      [E, 2*moe_intermediate_per_tp, hidden_size] (float8_e4m3fn)
      w13_scale:[E, scale_rows_13, scale_cols_13] (float32)
      w2:       [E, hidden_size, moe_intermediate_per_tp] (float8_e4m3fn)
      w2_scale: [E, scale_rows_2, scale_cols_2] (float32)

    Weights (BF16 mode):
      gate:  [num_experts, hidden_size]
      w13:   [E, 2*moe_intermediate_per_tp, hidden_size]
      w2:    [E, hidden_size, moe_intermediate_per_tp]
    """

    def __init__(self, config, quant_config: dict | None = None):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.hidden_size = config.hidden_size
        tp = _tp_size()
        self.tp_size = tp
        self.intermediate_per_tp = config.moe_intermediate_size // tp
        self.renormalize = getattr(config, "norm_topk_prob", True)
        self.use_fp8 = quant_config is not None

        self.gate = ReplicatedLinear(
            config.hidden_size, config.num_experts, bias=False,
        )

        w13_rows = 2 * self.intermediate_per_tp
        w2_cols = self.intermediate_per_tp

        if self.use_fp8:
            block_size = quant_config.get("weight_block_size", [128, 128])
            self.block_shape = block_size
            block_n, block_k = block_size[0], block_size[1]

            self.w13 = nn.Parameter(torch.empty(
                config.num_experts, w13_rows, config.hidden_size,
                dtype=torch.float8_e4m3fn,
            ), requires_grad=False)
            self.w13_scale = nn.Parameter(torch.ones(
                config.num_experts,
                math.ceil(w13_rows / block_n),
                math.ceil(config.hidden_size / block_k),
                dtype=torch.float32,
            ), requires_grad=False)

            self.w2 = nn.Parameter(torch.empty(
                config.num_experts, config.hidden_size, w2_cols,
                dtype=torch.float8_e4m3fn,
            ), requires_grad=False)
            self.w2_scale = nn.Parameter(torch.ones(
                config.num_experts,
                math.ceil(config.hidden_size / block_n),
                math.ceil(w2_cols / block_k),
                dtype=torch.float32,
            ), requires_grad=False)

            self.w13.weight_loader = self._w13_weight_loader_fp8
            self.w13_scale.weight_loader = self._w13_scale_loader
            self.w2.weight_loader = self._w2_weight_loader_fp8
            self.w2_scale.weight_loader = self._w2_scale_loader
        else:
            self.block_shape = None
            self.w13 = nn.Parameter(torch.empty(
                config.num_experts, w13_rows, config.hidden_size,
            ))
            self.w13.weight_loader = self._w13_weight_loader

            self.w2 = nn.Parameter(torch.empty(
                config.num_experts, config.hidden_size, w2_cols,
            ))
            self.w2.weight_loader = self._w2_weight_loader

            self.w13_scale = None
            self.w2_scale = None

        self.topk_softmax = TopKSoftmax()
        self.fused_experts = FusedExperts()
        self.allreduce = AllReduce()

        # Custom-op dispatch for torch.compile (set by engine after model init)
        self._use_custom_op = False
        self._layer_name = ""
        # DeepGEMM B-side scale layouts, transformed on first use and cached.
        self._dgws: dict = {}

    # --- BF16 weight loaders ---

    def _w13_weight_loader(self, param, loaded_weight, expert_id: int, is_w1: bool):
        tp, rank = _tp_size(), _tp_rank()
        N = self.intermediate_per_tp
        shard = loaded_weight.narrow(0, rank * N, N)
        offset = 0 if is_w1 else N
        param.data[expert_id, offset:offset + N, :].copy_(shard)

    def _w2_weight_loader(self, param, loaded_weight, expert_id: int):
        tp, rank = _tp_size(), _tp_rank()
        N = self.intermediate_per_tp
        param.data[expert_id].copy_(loaded_weight.narrow(1, rank * N, N))

    # --- FP8 weight loaders ---

    def _w13_weight_loader_fp8(self, param, loaded_weight, expert_id: int, is_w1: bool):
        tp, rank = _tp_size(), _tp_rank()
        N = self.intermediate_per_tp
        shard = loaded_weight.narrow(0, rank * N, N)
        offset = 0 if is_w1 else N
        param.data[expert_id, offset:offset + N, :].copy_(shard)

    def _w13_scale_loader(self, param, loaded_weight, expert_id: int, is_w1: bool):
        tp, rank = _tp_size(), _tp_rank()
        block_n = self.block_shape[0]
        N = self.intermediate_per_tp
        scale_rows_per_shard = math.ceil(N / block_n)
        full_scale_rows = loaded_weight.shape[0]
        rows_per_tp = full_scale_rows // tp
        src = loaded_weight.narrow(0, rank * rows_per_tp, rows_per_tp)
        offset = 0 if is_w1 else scale_rows_per_shard
        param.data[expert_id, offset:offset + rows_per_tp, :].copy_(src)

    def _w2_weight_loader_fp8(self, param, loaded_weight, expert_id: int):
        tp, rank = _tp_size(), _tp_rank()
        N = self.intermediate_per_tp
        param.data[expert_id].copy_(loaded_weight.narrow(1, rank * N, N))

    def _w2_scale_loader(self, param, loaded_weight, expert_id: int):
        tp, rank = _tp_size(), _tp_rank()
        block_k = self.block_shape[1]
        N = self.intermediate_per_tp
        scale_cols_per_shard = math.ceil(N / block_k)
        full_scale_cols = loaded_weight.shape[1]
        cols_per_tp = full_scale_cols // tp
        src = loaded_weight.narrow(1, rank * cols_per_tp, cols_per_tp)
        param.data[expert_id].copy_(src)

    # --- fused expert compute -------------------------------------------

    def _fused_ok(self, hidden_states) -> bool:
        """The fused chain assumes 128-aligned hidden / intermediate sizes and
        the standard 128x128 weight-scale blocking.  Anything else falls back to
        the reference grouped-GEMM path so correctness never depends on it."""
        bs = self.block_shape
        return (hidden_states.is_contiguous()
                and self.hidden_size % _FP8_BLOCK == 0
                and self.intermediate_per_tp % _FP8_BLOCK == 0
                and bs is not None and bs[0] == _FP8_BLOCK
                and bs[1] == _FP8_BLOCK
                and self.w13_scale is not None and self.w2_scale is not None
                and self.w13_scale.dtype == torch.float32
                and self.w2_scale.dtype == torch.float32)

    def _dg_ok(self, hidden_states) -> bool:
        """DeepGEMM's m-grouped contiguous layout is usable here.  Adds two
        constraints on top of the reference's own ``_valid_deep_gemm``: the
        packed-UE8M0 scale word covers four 128-element k-groups, so both
        reduction dims must be multiples of 4*128."""
        return (_DG_GEMM is not None
                and self.hidden_size % (4 * _FP8_BLOCK) == 0
                and self.intermediate_per_tp % (4 * _FP8_BLOCK) == 0
                and _DG_ALIGN == _FP8_BLOCK
                and _deepgemm_layout_ok(hidden_states, self.w13, self.w2))

    def _fused_moe_dg(self, hidden_states, topk_weights, topk_ids,
                      _SCRATCH=_SCRATCH):
        """DeepGEMM m-grouped-contiguous expert compute.

        The A operand is built by one fused gather+quantize+scatter kernel that
        writes straight into the padded m-grouped buffer and into DeepGEMM's
        packed-UE8M0 scale layout; the intermediate is emitted in the same
        layout by the SiLU-mul epilogue pass; the unpermute and the top-k sum
        are one fused weighted gather.  Nothing here allocates, argsorts,
        searchsorteds, zero-fills a padded buffer, or syncs with the host.
        """
        M, K = hidden_states.shape
        E, TOPK = self.num_experts, self.top_k
        N, N2 = self.intermediate_per_tp, 2 * self.intermediate_per_tp
        NT = M * TOPK
        dev = hidden_states.device
        NG, KP = K // _FP8_BLOCK, K // (4 * _FP8_BLOCK)
        NGH, KPH = N // _FP8_BLOCK, N // (4 * _FP8_BLOCK)
        EP = triton.next_power_of_2(E)
        A = _DG_ALIGN
        # Worst-case padded row count (every expert loses up to A-1 rows to
        # alignment).  Fixed rather than data-dependent, so no host sync is
        # needed; the tail blocks carry expert id -1 and DeepGEMM skips them.
        MS = NT + E * (A - 1)
        MS += (-MS) % A

        # -- routing metadata: counts -> aligned offsets -> destination rows --
        cnt = _SCRATCH.get("cnt", E + 1, torch.int32, dev, zero=True)
        cnt2 = _SCRATCH.get("cnt2", E + 1, torch.int32, dev)
        aoff = _SCRATCH.get("aoff", E + 1, torch.int32, dev)
        cur = _SCRATCH.get("cur", E + 1, torch.int32, dev)
        dest = _SCRATCH.get("dest", NT, torch.int32, dev)
        m_idx = _SCRATCH.get("midx", MS, torch.int32, dev)
        ids = topk_ids.view(-1)
        blk = 1024
        _meta_count[(triton.cdiv(NT, blk),)](
            ids, cnt, NT, E=E, EP=EP, BLK=blk, num_warps=4)
        _meta_scan_dg[(1,)](cnt, cnt2, aoff, cur, E=E, EP=EP, ALIGN=A,
                            num_warps=4)
        _meta_fill_dg[(triton.cdiv(NT, blk),)](
            ids, cur, dest, NT, E=E, BLK=blk, num_warps=4)
        _meta_idx[(MS // A,)](aoff, cnt2, m_idx, E=E, EP=EP, ALIGN=A,
                              num_warps=4)

        # -- fused gather + quantize + scatter into the m-grouped buffer ------
        a_q = _SCRATCH.get("dgaq", MS * K, torch.float8_e4m3fn, dev).view(MS, K)
        a_sf = (_SCRATCH.get("dgas", KP * MS, torch.int32, dev)
                .view(KP, MS).transpose(0, 1))
        if _TUNE["sq8"]:
            _scatter_quant8[(M,)](
                hidden_states, dest, a_q, a_sf, M, MS,
                K=K, TOPK=TOPK, NG=NG, KP=KP, num_warps=_TUNE["sqw"])
        else:
            _scatter_quant[(NT,)](
                hidden_states, dest, a_q, a_sf, NT, MS,
                K=K, TOPK=TOPK, NG=NG, KP=KP, num_warps=_TUNE["sqw"])

        # -- GEMM1 -----------------------------------------------------------
        ws13 = _dg_wscale(self._dgws, "w13", self.w13_scale, N2, K, E)
        mm = _SCRATCH.get("dgmm", MS * max(N2, K), torch.bfloat16, dev)
        mm1 = mm[:MS * N2].view(MS, N2)
        _DG_GEMM((a_q, a_sf), (self.w13, ws13), mm1, m_idx)

        # -- SiLU-mul + requantization, emitted in DeepGEMM's layout ---------
        hq = _SCRATCH.get("dghq", MS * N, torch.float8_e4m3fn, dev).view(MS, N)
        h_sf = (_SCRATCH.get("dghs", KPH * MS, torch.int32, dev)
                .view(KPH, MS).transpose(0, 1))
        sbm = _TUNE["sq_bm"]
        _silu_quant_dg[(MS // sbm, KPH)](
            mm1, m_idx, hq, h_sf, MS, N=N, BM=sbm, ALIGN=A,
            num_warps=_TUNE["sqs"])

        # -- GEMM2 (reuses GEMM1's output buffer) ----------------------------
        ws2 = _dg_wscale(self._dgws, "w2", self.w2_scale, K, N, E)
        mm2 = mm[:MS * K].view(MS, K)
        _DG_GEMM((hq, h_sf), (self.w2, ws2), mm2, m_idx)

        # -- fused unpermute + top-k weighted reduction ----------------------
        out = torch.empty(M, K, dtype=hidden_states.dtype, device=dev)
        bkr = _TUNE["bkrd"]
        while K % bkr:
            bkr //= 2
        _reduce_dg[(M, K // bkr)](
            mm2, dest, topk_weights, out, K=K, TOPK=TOPK, BK=bkr,
            num_warps=_TUNE["rwd"])
        return out

    def _fused_moe(self, hidden_states, topk_weights, topk_ids,
                   _SCRATCH=_SCRATCH):
        M, K = hidden_states.shape
        E = self.num_experts
        TOPK = self.top_k
        N = self.intermediate_per_tp
        NT = M * TOPK
        dev = hidden_states.device
        NG = K // _FP8_BLOCK
        NGH = N // _FP8_BLOCK
        EP = triton.next_power_of_2(E)
        # The reference implementation switches between two numerically distinct
        # reduction orders: its DeepGEMM path stores the *unweighted* bf16
        # per-expert output and applies the routed weight in fp32 during the
        # unpermute-reduce, while its Triton fallback folds the weight into the
        # GEMM epilogue before the bf16 store.  Follow whichever one applies so
        # the bf16 rounding (the dominant error term at these magnitudes) lines
        # up on every scenario.
        mulw = not _deepgemm_layout_ok(hidden_states, self.w13, self.w2)

        # -- activation quant --------------------------------------------
        a_q = _SCRATCH.get("aq", M * K, torch.float8_e4m3fn, dev)
        a_s = _SCRATCH.get("as", M * NG, torch.float32, dev)
        # Two GEMM1 shapes: the fused one (gate/up side by side, SiLU-mul +
        # requantization in the epilogue) and a split one (single accumulator
        # over the whole 2N, separate activation pass).  The split form has a
        # cheaper accumulator and more programs, which wins once there is real
        # work; the fused form saves a launch and a round trip, which wins when
        # the call is launch-bound.  For tiny token counts GEMM1 also folds the
        # activation quantization in, removing another launch entirely.
        use_g1s = _TUNE["g1mode"] and NT >= _TUNE["g1mode_min"]
        qi = (not use_g1s) and NT <= _TUNE["qi_nt"]
        if not qi:
            qbm = 1 if M <= 64 else 16
            _quant_act[(triton.cdiv(M, qbm), NG)](
                hidden_states, a_q, a_s, M, K, NG, BM=qbm, G=_FP8_BLOCK,
                num_warps=4,
            )

        # -- routing metadata --------------------------------------------
        BM = _pick_bm(NT, E)
        cnt = _SCRATCH.get("cnt", E + 1, torch.int32, dev, zero=True)
        cnt2 = _SCRATCH.get("cnt2", E + 1, torch.int32, dev)
        off = _SCRATCH.get("off", E + 1, torch.int32, dev)
        cur = _SCRATCH.get("cur", E + 1, torch.int32, dev)
        blkoff = _SCRATCH.get("blkoff", E + 1, torch.int32, dev)
        slot = _SCRATCH.get("slot", NT, torch.int32, dev)
        ids = topk_ids.view(-1)

        if NT <= 128:
            _meta_small[(1,)](
                ids, cnt2, off, slot, blkoff, NT, E=E, EP=EP,
                NTP=triton.next_power_of_2(NT), BM=BM, num_warps=4,
            )
        else:
            blk = 1024
            _meta_count[(triton.cdiv(NT, blk),)](
                ids, cnt, NT, E=E, EP=EP, BLK=blk, num_warps=4,
            )
            _meta_scan[(1,)](cnt, cnt2, off, cur, blkoff, E=E, EP=EP, BM=BM,
                             num_warps=4)
            _meta_fill[(triton.cdiv(NT, blk),)](
                ids, cur, slot, NT, E=E, BLK=blk, num_warps=4,
            )

        nb_max = min(NT, triton.cdiv(NT, BM) + E)

        # -- GEMM1 (gather + silu-mul + requant) -------------------------
        hq = _SCRATCH.get("hq", NT * N, torch.float8_e4m3fn, dev)
        hs = _SCRATCH.get("hs", NT * NGH, torch.float32, dev)
        w13s = self.w13_scale
        if use_g1s:
            mm1 = _SCRATCH.get("mm1", NT * 2 * N, torch.bfloat16, dev)
            _gemm1s[(nb_max, 2 * N // _TUNE["bn1"])](
                a_q, a_s, self.w13, w13s, mm1,
                slot, cnt2, off, blkoff,
                K=K, N2=2 * N, E=E, EP=EP, TOPK=TOPK, NG=NG,
                WSK=w13s.shape[2], WSN=w13s.shape[1],
                BM=BM, BN=_TUNE["bn1"], BK=128,
                num_warps=_ws("g1", BM)[0], num_stages=_ws("g1", BM)[1],
            )
            sbm = 16 if NT > 256 else 1
            _silu_quant[(triton.cdiv(NT, sbm), NGH)](
                mm1, hq, hs, NT, NT, N=N, BM=sbm, G=_FP8_BLOCK, num_warps=4,
            )
        else:
            bn1f = _TUNE["bn1f"]
            _gemm1[(nb_max, N // bn1f)](
                hidden_states if qi else a_q, a_s, self.w13, w13s, hq, hs,
                slot, cnt2, off, blkoff,
                K=K, N=N, E=E, EP=EP, TOPK=TOPK, NG=NG, NGH=NGH,
                WSK=w13s.shape[2], WSN=w13s.shape[1], NTT=NT,
                BM=BM, BN=bn1f, BK=128, ASV=_TUNE["asv1"], QI=qi,
                num_warps=_ws("g1", BM)[0], num_stages=_ws("g1", BM)[1],
            )

        # -- GEMM2 + top-k reduction -------------------------------------
        w2s = self.w2_scale
        i3 = _SCRATCH.get("i3", NT * K, torch.bfloat16, dev)
        bn2 = _TUNE["bn2"]
        _gemm2[(nb_max, K // bn2)](
            hq, hs, self.w2, w2s, i3, topk_weights,
            slot, cnt2, off, blkoff,
            K=K, N=N, E=E, EP=EP, NGH=NGH,
            WSK=w2s.shape[2], WSN=w2s.shape[1], NTT=NT,
            BM=BM, BN=bn2, BK=128, MULW=mulw, ASV=_TUNE["asv2"],
            num_warps=_ws("g2", BM)[0], num_stages=_ws("g2", BM)[1],
        )

        # -- top-k reduction ---------------------------------------------
        out = torch.empty(M, K, dtype=hidden_states.dtype, device=dev)
        bkr = _TUNE["bkr"]
        while K % bkr:
            bkr //= 2
        _reduce[(M, K // bkr)](
            i3, topk_weights, out, M, K=K, TOPK=TOPK, BK=bkr,
            USEW=not mulw, num_warps=_TUNE["rw"],
        )
        return out

    def _core(self, hidden_states, scratch=_SCRATCH):
        """Router + expert compute for an already-flattened ``[M, hidden]``
        input.  ``scratch`` is injectable so a caller can hand the chain a
        private buffer pool instead of the shared grow-only one."""
        router_logits = self.gate(hidden_states)
        topk_weights, topk_ids = self.topk_softmax(
            router_logits, self.top_k, renormalize=self.renormalize,
        )

        if self.use_fp8 and self._fused_ok(hidden_states):
            if (_TUNE["dg"]
                    and hidden_states.size(0) * self.top_k
                    >= self.num_experts * _TUNE["dg_min_per_e"]
                    and self._dg_ok(hidden_states)):
                return self._fused_moe_dg(hidden_states, topk_weights, topk_ids,
                                          scratch)
            return self._fused_moe(hidden_states, topk_weights, topk_ids, scratch)
        if self.use_fp8:
            return self.fused_experts(
                hidden_states, self.w13, self.w2,
                topk_weights, topk_ids, self.num_experts,
                w13_scale=self.w13_scale, w2_scale=self.w2_scale,
                use_fp8_w8a8=True, block_shape=self.block_shape,
            )
        return self.fused_experts(
            hidden_states, self.w13, self.w2,
            topk_weights, topk_ids, self.num_experts,
            w13_scale=None, w2_scale=None,
            use_fp8_w8a8=False, block_shape=None,
        )

    def forward_impl(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Core MoE logic, callable from both eager and custom-op paths."""
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_size)

        out = self._core(hidden_states)

        if self.tp_size > 1 and not self._use_custom_op:
            out = self.allreduce(out)

        return out.view(orig_shape)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._use_custom_op:
            # The all-reduce stays *outside* the opaque op: inside it Inductor
            # cannot see the collective, so ``AllReduceFusedAddRMSNormPass`` has
            # nothing to match at the MoE end of the layer -- half of every
            # layer's collectives. vLLM keeps its MoE reduction in traced Python
            # for the same reason (``moe_runner._maybe_reduce_final_output``).
            out = torch.ops.fastkernels.moe_forward(hidden_states, self._layer_name)
            if self.tp_size > 1:
                out = self.allreduce(out)
            return out
        return self.forward_impl(hidden_states)
