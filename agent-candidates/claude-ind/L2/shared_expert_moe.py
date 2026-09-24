"""Shared-expert MoE as one graph-replayed Triton pipeline.

The baseline hands the routed half to trtllm-gen's monolithic BF16 MoE kernel
(via FlashInfer) and runs the router, the shared expert and the gated epilogue
around it as five more launches. On B200 that kernel is a good one -- at 26
tokens it streams the ~1.2 GB of selected expert weights at ~5 TB/s, which is
~90% of what this GPU delivers -- but the layer as a whole is dominated by
something else at decode sizes: its Python launcher (AutoTuner lookup, runner
construction, tuning-config rebuild) costs ~0.7-1.0 ms of *host* time per call,
an order of magnitude more than the ~60-500 us of GPU work.

So this candidate rewrites the whole layer body -- router GEMM, top-k, routed
experts, shared expert, gated epilogue -- as seven Triton kernels with no host
synchronisation and no data-dependent shapes, then records them into one CUDA
graph per token count. A layer then costs four CUDA calls on the host (copy in,
replay, copy out) instead of a Python round trip per sub-op, and the GPU side is
within a few percent of trtllm-gen's (both are HBM-bound on the same bytes),
ahead of it at a single token where there is no way to fill the machine and
fewer padded tiles is what matters.

Pipeline (capture-safe: static grids, device-side bounds):

    logits -> route -> gemm1 -> gemm2 -> epilogue      (routed critical path)
          shared_gu -> shared_down .............       (side stream)

1. ``_router_logits_kernel``  ``x @ gate.weight^T``. Only 2 MB of weights, so at
   small batches K is split to get enough CTAs; the FP32 partials go to separate
   slices, which needs no atomics and no memset. Also clears the histogram that
   the next kernel fills.
2. ``_route_kernel``  per token: top-k by repeated arg-max over BF16-rounded
   logits, softmax over the selected ones (identical to softmax over all experts
   plus the top-k renormalisation), the expert histogram and slot assignment in
   one atomic, this token's row of the routed accumulator zeroed, and -- the row
   is already resident -- the shared-expert gate's ``[hidden] -> [1]`` dot.
3. ``_gemm1_kernel``  grouped ``x[tokens_of_e] @ w13[e]^T`` with SwiGLU fused
   into the epilogue: one pass over w13, both halves in the same program so the
   gathered activation tile is read once.
4. ``_gemm2_kernel``  grouped ``h[slots_of_e] @ w2[e]^T``, routing-weighted and
   FP32-atomically accumulated into a [T, H] buffer -- the atomics *are* the
   top-k reduction, so no per-slot output staging and no reduce kernel.
5. ``_shared_gu_kernel`` / ``_shared_down_kernel``  the shared expert. It depends
   only on x, so it rides a side stream and its 6 MB of weights stream while the
   routed experts' do; only the final add is left on the critical path.
6. ``_epilogue_kernel``  ``routed + shared * sigmoid(gate)`` -> BF16 output.

Three things carry most of the performance:

* **Expert buckets instead of a sort.** ``_route_kernel`` writes each (token, k)
  pair straight into ``expert * CAP + rank`` with one atomic, so there is no
  permutation kernel; the grouped GEMMs recover each expert's slot range and
  tile range by redoing the histogram prefix sums in-register (five 512-wide
  reductions, see :func:`_tile_map`), which beats launching a scan and
  serialising the pipeline against it.
* **BF16-rounded logits.** The reference router is ``F.linear`` in BF16, and ~13%
  of tokens have two experts whose logits collapse to the same BF16 value at the
  k/k+1 boundary, where the winner is decided by index order. Rounding before
  the top-k (and letting ties go to the lower index, as ``torch.topk`` and
  trtllm-gen both do) is what keeps the selected expert sets equal; a more
  precise router would disagree with the reference on those tokens.
* **Enough CTAs, not bigger tiles.** Every grouped-GEMM tile shape between
  BN=32 and BN=256 lands within a few percent of the same ~4.9 TB/s, so the
  tiles are chosen to spread each expert's 6 MB over enough CTAs to saturate
  HBM rather than for MMA efficiency -- an expert sees only a handful of tokens
  until the batch is large, so BM stays at the MMA minimum of 16.

Anything the fast path does not cover -- grouped top-k, correction bias, sigmoid
routing, a non-unit routed scaling factor, a config it has no tiling for, or a
batch past ``_FAST_MAX_TOKENS`` where trtllm-gen's padding no longer costs
anything -- falls back to the baseline body, including trtllm-gen's weight
layout.
"""
from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

import triton
import triton.language as tl

from ....infra.tp import _tp_rank, _tp_size
from ..L1.allreduce import AllReduce
from ..L1.gate_linear import GateLinear
from ..L1.grouped_topk import GroupedTopK
from ..L1.moe_shared_gate_add import moe_shared_gate_add
from ..L1.silu_and_mul import SiluAndMul
from .trtllm_bf16_moe import (
    TrtLlmBf16MoE,
    prepare_trtllm_bf16_moe_weights,
    trtllm_bf16_moe_supported,
)
from .fused_experts import FusedExperts
from .parallel_linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)


def trtllm_routing_method_type(
    routing: str,
    renormalize: bool,
    has_e_score_bias: bool,
    num_expert_group: int | None,
) -> int | None:
    """Map a routing config to a trtllm-gen ``RoutingMethodType``, else None.

    Mirrors vLLM's ``get_routing_method_type``
    (``vllm/model_executor/layers/fused_moe/config.py``). ``None`` stands for
    vLLM's ``Unspecified``, which ``TrtLlmBf16ExpertsMonolithic`` does not
    accept -- callers fall back to the Triton path in that case.
    """
    if has_e_score_bias:
        if routing != "sigmoid" or not renormalize:
            return None
        if (num_expert_group or 0) > 0:
            return 2  # DeepSeekV3
        return None
    if routing == "sigmoid":
        return 6 if renormalize else 8  # SigmoidRenorm / Sigmoid
    if routing == "softmax":
        return 4 if renormalize else 0  # RenormalizeNaive / Default
    return None


# ---------------------------------------------------------------------------
# Triton pipeline
# ---------------------------------------------------------------------------
@triton.jit
def _tile_map(count_ptr, t, E: tl.constexpr, BM: tl.constexpr):
    """Decode flat tile id *t* -> (expert, slot_base, m_tile, count, n_tiles).

    Each expert owns ``ceil(count_e / BM)`` consecutive tile ids.
    """
    ea = tl.arange(0, E)
    c = tl.load(count_ptr + ea)
    tiles = (c + (BM - 1)) // BM
    e = tl.sum(tl.where(tl.cumsum(tiles, axis=0) <= t, 1, 0), axis=0)
    before = ea < e
    slot_base = tl.sum(tl.where(before, c, 0), axis=0)
    m_tile = t - tl.sum(tl.where(before, tiles, 0), axis=0)
    cnt = tl.sum(tl.where(ea == e, c, 0), axis=0)
    return e, slot_base, m_tile, cnt, tl.sum(tiles, axis=0)


@triton.jit
def _router_logits_kernel(
    x_ptr, w_ptr, out_ptr, count_ptr, n_tokens,
    H: tl.constexpr, E: tl.constexpr, KS: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    """logits_part[ks, T, E] = x[:, k_slice] @ w[:, k_slice]^T (FP32 partials).

    Also clears the expert histogram that ``_route_kernel`` fills atomically --
    it is only read after this kernel completes, so the memset rides along
    instead of being its own launch.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    ks = tl.program_id(2)
    if (pid_m == 0) & (ks == 0):
        # Exactly E // BN programs run along axis 1, so BN entries each covers
        # the histogram once and only once.
        ec = pid_n * BN + tl.arange(0, BN)
        tl.store(count_ptr + ec, tl.zeros((BN,), tl.int32))
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    m_mask = offs_m < n_tokens
    offs_k = ks * (H // KS) + tl.arange(0, BK)
    x_ptrs = x_ptr + offs_m[:, None].to(tl.int64) * H + offs_k[None, :]
    w_ptrs = w_ptr + offs_n[:, None].to(tl.int64) * H + offs_k[None, :]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for _ in range(0, H // KS, BK):
        a = tl.load(x_ptrs, mask=m_mask[:, None], other=0.0)
        b = tl.load(w_ptrs)
        acc = tl.dot(a, tl.trans(b), acc)
        x_ptrs += BK
        w_ptrs += BK
    tl.store(out_ptr + (ks * n_tokens + offs_m[:, None]).to(tl.int64) * E + offs_n[None, :],
             acc, mask=m_mask[:, None])


@triton.jit
def _route_kernel(
    logits_ptr, x_ptr, gate_w_ptr, racc_ptr,
    slot_tok_ptr, slot_w_ptr, count_ptr, gate_ptr, n_tokens,
    H: tl.constexpr, E: tl.constexpr, TOPK: tl.constexpr, TP: tl.constexpr,
    KS: tl.constexpr, CAP: tl.constexpr, BH: tl.constexpr,
    HAS_GATE: tl.constexpr, LOG_DT: tl.constexpr,
):
    """One CTA per token: top-k, softmax weights, expert bucket, gate dot.

    The logits are rounded to BF16 before the top-k because ``F.linear`` -- what
    the reference router runs -- does: ~13% of tokens have two experts whose
    logits collapse to the same BF16 value at the k/k+1 boundary, where the
    winner is then decided by index order. Keeping FP32 precision here would
    pick a different expert set than the reference on those tokens.
    """
    m = tl.program_id(0)
    mi = m.to(tl.int64)
    ea = tl.arange(0, E)
    part = tl.zeros((E,), tl.float32)
    for ks in tl.static_range(KS):
        part += tl.load(logits_ptr + (ks * n_tokens + mi) * E + ea)
    cur = part.to(LOG_DT).to(tl.float32)
    jj = tl.arange(0, TP)
    vals = tl.full((TP,), float("-inf"), tl.float32)
    ids = tl.zeros((TP,), tl.int32)
    for j in range(TOPK):
        v, i = tl.max(cur, axis=0, return_indices=True)
        i = i.to(tl.int32)
        vals = tl.where(jj == j, v, vals)
        ids = tl.where(jj == j, i, ids)
        cur = tl.where(ea == i, float("-inf"), cur)
    kmask = jj < TOPK
    p = tl.where(kmask, tl.exp(vals - tl.max(vals, axis=0)), 0.0)
    w = p / tl.sum(p, axis=0)
    rank = tl.atomic_add(count_ptr + ids, tl.full((TP,), 1, tl.int32), mask=kmask)
    dst = ids.to(tl.int64) * CAP + rank
    tl.store(slot_tok_ptr + dst, tl.full((TP,), m, tl.int32), mask=kmask)
    tl.store(slot_w_ptr + dst, w, mask=kmask)

    zero = tl.zeros((BH,), tl.float32)
    for k in range(0, H, BH):
        tl.store(racc_ptr + mi * H + k + tl.arange(0, BH), zero)

    if HAS_GATE:
        acc = tl.zeros((BH,), tl.float32)
        for k in range(0, H, BH):
            kk = k + tl.arange(0, BH)
            acc += (tl.load(x_ptr + mi * H + kk).to(tl.float32)
                    * tl.load(gate_w_ptr + kk).to(tl.float32))
        tl.store(gate_ptr + mi, tl.sum(acc, axis=0))


@triton.jit
def _gemm1_kernel(
    x_ptr, w13_ptr, h_ptr, slot_tok_ptr, count_ptr,
    H: tl.constexpr, I: tl.constexpr, E: tl.constexpr, CAP: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    """Grouped ``x[tokens_of_e] @ w13[e]^T`` with SwiGLU fused into the epilogue.

    Both halves of w13 are consumed by the same program, so the gathered
    activation tile is read once for the gate and the up projection and the
    BF16 intermediate is written once.
    """
    t = tl.program_id(0)
    e, slot_base, m_tile, cnt, total = _tile_map(count_ptr, t, E, BM)
    if t < total:
        rows = m_tile * BM + tl.arange(0, BM)
        valid = rows < cnt
        ei = e.to(tl.int64)
        tok = tl.load(slot_tok_ptr + ei * CAP + rows, mask=valid, other=0).to(tl.int64)
        offs_n = tl.program_id(1) * BN + tl.arange(0, BN)
        offs_k = tl.arange(0, BK)
        w1p = w13_ptr + ei * (2 * I * H) + offs_n[:, None].to(tl.int64) * H + offs_k[None, :]
        w3p = w1p + I * H
        xp = x_ptr + tok[:, None] * H + offs_k[None, :]
        a1 = tl.zeros((BM, BN), tl.float32)
        a3 = tl.zeros((BM, BN), tl.float32)
        for _ in range(0, H, BK):
            a = tl.load(xp, mask=valid[:, None], other=0.0)
            a1 = tl.dot(a, tl.trans(tl.load(w1p)), a1)
            a3 = tl.dot(a, tl.trans(tl.load(w3p)), a3)
            xp += BK
            w1p += BK
            w3p += BK
        slots = (slot_base + rows).to(tl.int64)
        tl.store(h_ptr + slots[:, None] * I + offs_n[None, :],
                 ((a1 * tl.sigmoid(a1)) * a3).to(h_ptr.dtype.element_ty),
                 mask=valid[:, None])


@triton.jit
def _gemm2_kernel(
    h_ptr, w2_ptr, out_ptr, slot_tok_ptr, slot_w_ptr, count_ptr,
    H: tl.constexpr, I: tl.constexpr, E: tl.constexpr, CAP: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    """Grouped ``h[slots_of_e] @ w2[e]^T``, routing-weighted.

    The top-k reduction *is* the FP32 atomic accumulate into ``out``: one pass,
    no per-slot output staging and no separate reduce kernel.
    """
    t = tl.program_id(0)
    e, slot_base, m_tile, cnt, total = _tile_map(count_ptr, t, E, BM)
    if t < total:
        rows = m_tile * BM + tl.arange(0, BM)
        valid = rows < cnt
        ei = e.to(tl.int64)
        tok = tl.load(slot_tok_ptr + ei * CAP + rows, mask=valid, other=0).to(tl.int64)
        sw = tl.load(slot_w_ptr + ei * CAP + rows, mask=valid, other=0.0)
        slots = (slot_base + rows).to(tl.int64)
        offs_n = tl.program_id(1) * BN + tl.arange(0, BN)
        offs_k = tl.arange(0, BK)
        w2p = w2_ptr + ei * (H * I) + offs_n[:, None].to(tl.int64) * I + offs_k[None, :]
        hp = h_ptr + slots[:, None] * I + offs_k[None, :]
        acc = tl.zeros((BM, BN), tl.float32)
        for _ in range(0, I, BK):
            a = tl.load(hp, mask=valid[:, None], other=0.0)
            acc = tl.dot(a, tl.trans(tl.load(w2p)), acc)
            hp += BK
            w2p += BK
        tl.atomic_add(out_ptr + tok[:, None] * H + offs_n[None, :],
                      acc * sw[:, None], mask=valid[:, None])


@triton.jit
def _shared_gu_kernel(
    x_ptr, w_ptr, hs_ptr, n_tokens,
    H: tl.constexpr, I: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    """Shared expert's merged gate/up GEMM with SwiGLU fused in."""
    offs_m = tl.program_id(0) * BM + tl.arange(0, BM)
    offs_n = tl.program_id(1) * BN + tl.arange(0, BN)
    m_mask = offs_m < n_tokens
    offs_k = tl.arange(0, BK)
    xp = x_ptr + offs_m[:, None].to(tl.int64) * H + offs_k[None, :]
    w1p = w_ptr + offs_n[:, None].to(tl.int64) * H + offs_k[None, :]
    w3p = w1p + I * H
    a1 = tl.zeros((BM, BN), tl.float32)
    a3 = tl.zeros((BM, BN), tl.float32)
    for _ in range(0, H, BK):
        a = tl.load(xp, mask=m_mask[:, None], other=0.0)
        a1 = tl.dot(a, tl.trans(tl.load(w1p)), a1)
        a3 = tl.dot(a, tl.trans(tl.load(w3p)), a3)
        xp += BK
        w1p += BK
        w3p += BK
    tl.store(hs_ptr + offs_m[:, None].to(tl.int64) * I + offs_n[None, :],
             ((a1 * tl.sigmoid(a1)) * a3).to(hs_ptr.dtype.element_ty),
             mask=m_mask[:, None])


@triton.jit
def _shared_down_kernel(
    hs_ptr, w_ptr, out_ptr, n_tokens,
    H: tl.constexpr, I: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    """Shared expert's down projection. Runs on the side stream, so its weights
    stream while the routed experts' do."""
    offs_m = tl.program_id(0) * BM + tl.arange(0, BM)
    offs_n = tl.program_id(1) * BN + tl.arange(0, BN)
    m_mask = offs_m < n_tokens
    offs_k = tl.arange(0, BK)
    hp = hs_ptr + offs_m[:, None].to(tl.int64) * I + offs_k[None, :]
    wp = w_ptr + offs_n[:, None].to(tl.int64) * I + offs_k[None, :]
    acc = tl.zeros((BM, BN), tl.float32)
    for _ in range(0, I, BK):
        a = tl.load(hp, mask=m_mask[:, None], other=0.0)
        acc = tl.dot(a, tl.trans(tl.load(wp)), acc)
        hp += BK
        wp += BK
    tl.store(out_ptr + offs_m[:, None].to(tl.int64) * H + offs_n[None, :],
             acc.to(out_ptr.dtype.element_ty), mask=m_mask[:, None])


@triton.jit
def _epilogue_kernel(
    shared_ptr, racc_ptr, gate_ptr, out_ptr, n_tokens,
    H: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
    HAS_GATE: tl.constexpr, HAS_SHARED: tl.constexpr,
):
    """``routed + shared * sigmoid(gate)`` -- all that is left on the critical
    path once the shared expert's GEMMs have been moved off it."""
    offs_m = tl.program_id(0) * BM + tl.arange(0, BM)
    offs_n = tl.program_id(1) * BN + tl.arange(0, BN)
    m_mask = offs_m < n_tokens
    idx = offs_m[:, None].to(tl.int64) * H + offs_n[None, :]
    acc = tl.load(racc_ptr + idx, mask=m_mask[:, None], other=0.0)
    if HAS_SHARED:
        sh = tl.load(shared_ptr + idx, mask=m_mask[:, None], other=0.0).to(tl.float32)
        if HAS_GATE:
            sh = sh * tl.sigmoid(tl.load(gate_ptr + offs_m, mask=m_mask,
                                         other=0.0))[:, None]
        acc += sh
    tl.store(out_ptr + idx, acc.to(out_ptr.dtype.element_ty), mask=m_mask[:, None])


def _plan(n_tokens: int) -> dict:
    """Tile shapes and the router's K-split, measured per token-count regime.

    The grouped GEMMs run at ~92% of this GPU's achievable read bandwidth in
    every regime, so their tiles are picked for everything else: BM stays at the
    MMA minimum (an expert sees a handful of tokens until the batch is large),
    and BN is set so that the selected experts' weights are streamed by enough
    CTAs to fill the machine -- at a couple of tokens that means narrow tiles
    (ten experts must keep 148 SMs busy), at hundreds it means wide ones. The
    router GEMM has only 2 MB of weights to read, so its K is split instead.
    """
    if n_tokens <= 8:
        return dict(KSR=4, BMR=16, BNR=32, BKR=128,
                    BM1=16, BN1=32, BK1=128, W1=2, S1=4,
                    BM2=16, BN2=64, BK2=128, W2=2, S2=3,
                    BMS=16, BNS=16, BKS=64, BMD=16, BND=32, BKD=128,
                    BME=16, BNE=256)
    return dict(KSR=8, BMR=16, BNR=128, BKR=256,
                BM1=16, BN1=64, BK1=128, W1=4, S1=4,
                BM2=16, BN2=128, BK2=128, W2=2, S2=2,
                BMS=256, BNS=32, BKS=64, BMD=32, BND=32, BKD=128,
                BME=32, BNE=256)


def _fit(v: int, dim: int, lo: int = 16) -> int:
    """Largest power of two <= *v* that divides *dim* (at least *lo*).

    Every block size indexes a ``tl.arange``, so it must be a power of two, and
    the kernels tile their N/K dimensions without masking, so it must divide the
    dimension. For the shapes this layer actually runs (all powers of two) this
    returns *v* unchanged; it only matters for odd configurations.
    """
    v = max(lo, min(v, dim))
    while v > lo and dim % v:
        v //= 2
    return v


def _sane(p: dict, H: int, I: int, Is: int, E: int) -> dict:
    """Clamp a plan to sizes the kernels can actually tile.

    Without this a BK larger than its reduction would skip the K-loop entirely
    and silently produce zeros, and a BN that does not divide its dimension
    would drop columns.
    """
    p = dict(p)
    Is = max(64, Is)
    p["KSR"] = _fit(p["KSR"], H, lo=1)
    p["BKR"] = _fit(p["BKR"], H // p["KSR"], lo=16)
    p["BK1"] = _fit(p["BK1"], H)
    p["BK2"] = _fit(p["BK2"], I)
    p["BKS"] = _fit(p["BKS"], H)
    p["BKD"] = _fit(p["BKD"], Is)
    p["BNR"] = _fit(p["BNR"], E)
    p["BN1"] = _fit(p["BN1"], I)
    p["BN2"] = _fit(p["BN2"], H)
    p["BNS"] = _fit(p["BNS"], Is)
    p["BND"] = _fit(p["BND"], H)
    p["BNE"] = _fit(p["BNE"], H)
    p["BH"] = _fit(512, H, lo=16)
    return p


class _FastPath:
    """Static buffers + one CUDA graph for a single token count."""

    __slots__ = ("owner", "n_tokens", "p", "buf", "cap", "graph", "x_static",
                 "out_static", "tiles1", "tiles2", "gate_w", "gu_w", "down_w",
                 "side", "eager", "fork_ev", "join_ev")

    def __init__(self, owner: "SharedExpertMoE", n_tokens: int):
        self.owner = owner
        self.n_tokens = n_tokens
        p = self.p = _sane(_plan(n_tokens),
                           owner.hidden_size, owner.intermediate_per_tp,
                           owner.shared_intermediate, owner.num_experts)
        dev = owner.fast_w13.device
        act = owner.fast_w13.dtype
        H, E, K = owner.hidden_size, owner.num_experts, owner.top_k
        I, Is = owner.intermediate_per_tp, owner.shared_intermediate
        slots = n_tokens * K
        # Bucket capacity: an expert can in principle take every token.
        cap = self.cap = n_tokens

        def f(*shape, dt=torch.float32):
            return torch.empty(shape, device=dev, dtype=dt)

        self.x_static = f(n_tokens, H, dt=act)
        self.out_static = f(n_tokens, H, dt=act)
        self.buf = dict(
            logits=f(p["KSR"], n_tokens, E),
            count=torch.zeros(E, device=dev, dtype=torch.int32),
            slot_tok=f(E * cap, dt=torch.int32),
            slot_w=f(E * cap),
            h=f(slots, I, dt=act),
            racc=f(n_tokens, H),
            hs=f(n_tokens, Is, dt=act) if Is else f(1, 1, dt=act),
            gate=f(n_tokens),
            shared=f(n_tokens, H, dt=act) if Is else f(1, 1, dt=act),
        )
        # Grid bound for the grouped GEMMs: sum_e ceil(c_e / BM) never exceeds
        # min(E, slots) + slots // BM. Static per token count (capture-safe); the
        # kernels recompute the exact count and exit early.
        self.tiles1 = min(E, slots) + slots // p["BM1"] + 1
        self.tiles2 = min(E, slots) + slots // p["BM2"] + 1
        # Weight handles are resolved once: the recorded graph bakes in their
        # addresses, so weights may be updated in place afterwards but not
        # replaced (the bench loads them before ``process_weights_after_loading``).
        self.gate_w = (owner.shared_expert_gate.weight.detach().reshape(-1)
                       if owner.shared_expert_gate is not None else self.buf["gate"])
        if owner.has_shared_expert:
            sh = getattr(owner, owner.shared_expert_attr_name)
            self.gu_w = sh.gate_up_proj.weight.detach()
            self.down_w = sh.down_proj.weight.detach()
        else:
            self.gu_w = self.down_w = self.buf["hs"]
        self.graph = None
        self.eager = False
        # Fork/join for the side stream. The stream and both events are created
        # up front: events allocated during capture do not record into the graph,
        # which silently drops the side-stream work from the replay.
        self.side = torch.cuda.Stream() if owner.has_shared_expert else None
        self.fork_ev = torch.cuda.Event()
        self.join_ev = torch.cuda.Event()

    # -- launches -----------------------------------------------------------
    def _launch_shared(self) -> None:
        """The shared expert's two GEMMs: independent of routing, so they ride
        the side stream while the routed experts' weights stream."""
        o, p, b = self.owner, self.p, self.buf
        Is = o.shared_intermediate
        _shared_gu_kernel[(triton.cdiv(self.n_tokens, p["BMS"]),
                           triton.cdiv(Is, p["BNS"]))](
            self.x_static, self.gu_w, b["hs"], self.n_tokens,
            H=o.hidden_size, I=Is,
            BM=p["BMS"], BN=p["BNS"], BK=p["BKS"], num_warps=4, num_stages=3,
        )
        _shared_down_kernel[(triton.cdiv(self.n_tokens, p["BMD"]),
                             o.hidden_size // p["BND"])](
            b["hs"], self.down_w, b["shared"], self.n_tokens,
            H=o.hidden_size, I=Is,
            BM=p["BMD"], BN=p["BND"], BK=p["BKD"], num_warps=4, num_stages=3,
        )

    def _launch_routed(self) -> None:
        o, p, b = self.owner, self.p, self.buf
        T = self.n_tokens
        H, E, K = o.hidden_size, o.num_experts, o.top_k
        I = o.intermediate_per_tp
        _router_logits_kernel[(triton.cdiv(T, p["BMR"]), E // p["BNR"], p["KSR"])](
            self.x_static, o.gate.weight, b["logits"], b["count"], T,
            H=H, E=E, KS=p["KSR"], BM=p["BMR"], BN=p["BNR"], BK=p["BKR"],
            num_warps=4, num_stages=3,
        )
        _route_kernel[(T,)](
            b["logits"], self.x_static, self.gate_w, b["racc"],
            b["slot_tok"], b["slot_w"], b["count"], b["gate"], T,
            H=H, E=E, TOPK=K, TP=o._topk_pad, KS=p["KSR"], CAP=self.cap,
            BH=p["BH"], HAS_GATE=o.shared_expert_gate is not None,
            LOG_DT=(tl.float16 if self.x_static.dtype == torch.float16
                    else tl.bfloat16),
            num_warps=4,
        )
        _gemm1_kernel[(self.tiles1, I // p["BN1"])](
            self.x_static, o.fast_w13, b["h"], b["slot_tok"], b["count"],
            H=H, I=I, E=E, CAP=self.cap,
            BM=p["BM1"], BN=p["BN1"], BK=p["BK1"],
            num_warps=p["W1"], num_stages=p["S1"],
        )
        _gemm2_kernel[(self.tiles2, H // p["BN2"])](
            b["h"], o.fast_w2, b["racc"], b["slot_tok"], b["slot_w"], b["count"],
            H=H, I=I, E=E, CAP=self.cap,
            BM=p["BM2"], BN=p["BN2"], BK=p["BK2"],
            num_warps=p["W2"], num_stages=p["S2"],
        )

    def _launch_epilogue(self, out: torch.Tensor) -> None:
        o, p, b = self.owner, self.p, self.buf
        _epilogue_kernel[(triton.cdiv(self.n_tokens, p["BME"]),
                          o.hidden_size // p["BNE"])](
            b["shared"], b["racc"], b["gate"], out, self.n_tokens,
            H=o.hidden_size, BM=p["BME"], BN=p["BNE"],
            HAS_GATE=o.shared_expert_gate is not None,
            HAS_SHARED=o.has_shared_expert, num_warps=4, num_stages=2,
        )

    def run(self, out: torch.Tensor, overlap: bool = True) -> None:
        """The whole layer. The shared expert only depends on x, so it runs on a
        side stream alongside the routed chain and is joined for the epilogue."""
        if overlap and self.side is not None:
            self.fork_ev.record()
            with torch.cuda.stream(self.side):
                self.fork_ev.wait()
                self._launch_shared()
                self.join_ev.record()
            self._launch_routed()
            self.join_ev.wait()
        else:
            if self.side is not None:
                self._launch_shared()
            self._launch_routed()
        self._launch_epilogue(out)

    # -- entry --------------------------------------------------------------
    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.graph is None and not self.eager:
            self._capture(x)
        self.x_static.copy_(x)
        if self.graph is None:
            out = torch.empty_like(self.out_static)
            self.run(out)
            return out
        self.graph.replay()
        return self.out_static.clone()

    def _capture(self, x: torch.Tensor) -> None:
        """Warm up (JIT + first-touch) eagerly, then record one graph.

        Everything the pipeline touches is pre-allocated, so capture records
        kernels only -- replay is then a single host call for the whole layer.
        """
        self.x_static.copy_(x)
        try:
            for _ in range(2):
                self.run(self.out_static)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self.run(self.out_static)
            self.graph = g
        except Exception:  # noqa: BLE001 - capture is an optimisation, not a must
            self.eager = True
            torch.cuda.synchronize()


class _TPSwiGLUMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int,
                 reduce_results: bool = True):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size, intermediate_size],
        )
        # ``reduce_results=False`` lets the caller add this partial to the
        # routed-expert partial and all-reduce the sum once, instead of
        # all-reducing both separately. vLLM does the same.
        self.down_proj = RowParallelLinear(
            intermediate_size, hidden_size, reduce_results=reduce_results,
        )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x = x.reshape(-1, orig_shape[-1])
        out = self.down_proj(self.act_fn(self.gate_up_proj(x)))
        return out.view(*orig_shape[:-1], out.shape[-1])


class SharedExpertMoE(nn.Module):
    # Above this many tokens every expert is selected many times over, so
    # trtllm-gen's per-expert padding stops mattering and its monolithic kernel
    # is the faster of the two (measured crossover ~768 tokens on B200); the
    # fused path keeps the decode and small-prefill range.
    _FAST_MAX_TOKENS = 768
    # One graph + buffer set per distinct token count. A serving run sees few,
    # but cap the cache so a pathological caller cannot grow it without bound.
    _MAX_PATHS = 32

    def __init__(
        self,
        *,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        moe_intermediate_size: int,
        routing: Literal["sigmoid", "softmax"] = "softmax",
        correction_bias: bool = False,
        renormalize: bool = True,
        routed_scaling_factor: float = 1.0,
        use_grouped_topk: bool = False,
        num_expert_group: int = 1,
        topk_group: int = 1,
        force_grouped_topk_sorted: bool = False,
        keep_router_weights_fp32: bool = False,
        shared_expert_intermediate_size: int = 0,
        shared_expert_attr_name: str = "shared_expert",
        shared_expert_gate: bool = False,
        reduce_results: bool = True,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.routing = routing
        self.correction_bias = correction_bias
        self.renormalize = renormalize
        self.routed_scaling_factor = routed_scaling_factor
        self.use_grouped_topk = use_grouped_topk
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.keep_router_weights_fp32 = keep_router_weights_fp32

        tp = _tp_size()
        self.tp_size = tp
        self.intermediate_per_tp = moe_intermediate_size // tp

        self.gate = ReplicatedLinear(hidden_size, num_experts, bias=False)
        if correction_bias:
            self.gate.e_score_correction_bias = nn.Parameter(torch.zeros(num_experts))
            self.gate.e_score_correction_bias.weight_loader = (
                lambda p, w: p.data.copy_(w)
            )
        if use_grouped_topk:
            self.gate_linear = GateLinear()
            self.grouped_topk = GroupedTopK(
                scoring_func=routing,
                renormalize=renormalize,
                routed_scaling_factor=1.0,
                force_sorted=force_grouped_topk_sorted,
            )
        else:
            self.gate_linear = None
            self.grouped_topk = None

        n = self.intermediate_per_tp
        self.w13 = nn.Parameter(torch.empty(num_experts, 2 * n, hidden_size))
        self.w13.weight_loader = self._w13_weight_loader
        self.w2 = nn.Parameter(torch.empty(num_experts, hidden_size, n))
        self.w2.weight_loader = self._w2_weight_loader

        self.fused_experts = FusedExperts()
        self.allreduce = AllReduce()
        # ``reduce_results=False`` hands the un-reduced partial sum back to the
        # caller, which folds the collective into its next norm.
        self.reduce_results = reduce_results
        self._use_custom_op = False
        self._layer_name = ""

        self._trtllm_routing = trtllm_routing_method_type(
            routing, renormalize, correction_bias, num_expert_group,
        )
        self.has_shared_expert = shared_expert_intermediate_size > 0
        self.shared_expert_attr_name = shared_expert_attr_name
        self.shared_intermediate = (
            shared_expert_intermediate_size // tp if self.has_shared_expert else 0
        )

        # What the fused pipeline covers -- i.e. what this layer is captured
        # running. Everything else keeps the baseline body verbatim.
        self._fast = (
            routing == "softmax"
            and renormalize
            and not correction_bias
            and not use_grouped_topk
            and routed_scaling_factor == 1.0
            and top_k <= 16
            and num_experts >= 64
            and (num_experts & (num_experts - 1)) == 0
            and hidden_size % 128 == 0
            and self.intermediate_per_tp % 128 == 0
            and (not self.has_shared_expert or self.shared_intermediate % 64 == 0)
            and torch.cuda.is_available()
        )
        self._topk_pad = 1 << max(0, (top_k - 1).bit_length())
        self._paths: dict[int, _FastPath] = {}
        self.fast_w13 = None
        self.fast_w2 = None

        self.use_trtllm = (
            trtllm_bf16_moe_supported() and self._trtllm_routing is not None
        )
        self.trtllm_moe = (
            TrtLlmBf16MoE(
                num_experts=num_experts,
                top_k=top_k,
                intermediate_size_per_partition=self.intermediate_per_tp,
                routing_method_type=self._trtllm_routing,
                num_expert_group=num_expert_group if use_grouped_topk else None,
                topk_group=topk_group if use_grouped_topk else None,
                routed_scaling_factor=(
                    routed_scaling_factor if routed_scaling_factor != 1.0 else None
                ),
            )
            if self.use_trtllm
            else None
        )
        self._trtllm_weights_ready = False

        if self.has_shared_expert:
            setattr(
                self,
                shared_expert_attr_name,
                _TPSwiGLUMLP(
                    hidden_size, shared_expert_intermediate_size,
                    # Defer the shared expert's reduce so it can be folded into
                    # the routed output's -- one all-reduce per layer instead of
                    # two.
                    reduce_results=(_tp_size() == 1),
                ),
            )
        if shared_expert_gate:
            self.shared_expert_gate = ReplicatedLinear(hidden_size, 1, bias=False)
        else:
            self.shared_expert_gate = None

    def _w13_weight_loader(
        self,
        param,
        loaded_weight,
        expert_id: int,
        is_w1: bool | None = None,
        is_gate: bool | None = None,
    ):
        if is_w1 is None and is_gate is None:
            raise TypeError("must pass is_w1 or is_gate to w13 loader")
        is_first = bool(is_w1 if is_w1 is not None else is_gate)
        rank = _tp_rank()
        n = self.intermediate_per_tp
        shard = loaded_weight.narrow(0, rank * n, n)
        offset = 0 if is_first else n
        param.data[expert_id, offset:offset + n, :].copy_(shard)

    def _w2_weight_loader(self, param, loaded_weight, expert_id: int):
        rank = _tp_rank()
        n = self.intermediate_per_tp
        param.data[expert_id].copy_(loaded_weight.narrow(1, rank * n, n))

    def process_weights_after_loading(self) -> None:
        """Keep the checkpoint layout for the fused path, shuffled for trtllm-gen.

        The fused GEMMs want exactly what the loaders already produce (both
        expert tensors K-contiguous), while trtllm-gen -- still used above
        ``_FAST_MAX_TOKENS`` -- needs its 4D BlockMajorK shuffle, which
        overwrites the parameters. So the fused path keeps its own handles on the
        original tensors and the parameters go on to be shuffled as in the
        baseline.
        """
        if self._fast and self.fast_w13 is None:
            self.fast_w13 = self.w13.data
            self.fast_w2 = self.w2.data
        if not self.use_trtllm or self._trtllm_weights_ready:
            return
        w13, w2 = prepare_trtllm_bf16_moe_weights(self.w13.data, self.w2.data)
        self.w13 = nn.Parameter(w13, requires_grad=False)
        self.w2 = nn.Parameter(w2, requires_grad=False)
        self._trtllm_weights_ready = True

    def _route(self, router_logits: torch.Tensor):
        if self.grouped_topk is not None:
            e_score_correction_bias = (
                self.gate.e_score_correction_bias if self.correction_bias else None
            )
            return self.grouped_topk(
                router_logits,
                e_score_correction_bias,
                num_expert_group=self.num_expert_group,
                topk_group=self.topk_group,
                topk=self.top_k,
            )
        if self.routing == "sigmoid":
            scores = torch.sigmoid(router_logits.float())
            if self.correction_bias:
                scores_for_choice = scores + self.gate.e_score_correction_bias
                _, topk_ids = scores_for_choice.topk(self.top_k, dim=-1)
                topk_weights = scores.gather(-1, topk_ids)
            else:
                topk_weights, topk_ids = scores.topk(self.top_k, dim=-1)
        else:
            scores = torch.softmax(router_logits.float(), dim=-1)
            topk_weights, topk_ids = scores.topk(self.top_k, dim=-1)
        if self.renormalize:
            topk_weights = topk_weights / (
                topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            )
        return topk_weights.to(torch.float32), topk_ids.to(torch.int32)

    # Token count below which the shared-expert gate projection is folded into
    # the epilogue kernel instead of run as its own gemv.
    _FUSE_GATE_MAX_TOKENS = 256

    def _shared_expert_output(
        self, hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Return ``(shared_output, raw_gate)``, both ``None`` without a shared expert.

        The gate's sigmoid and the scaling are deliberately *not* applied here:
        the caller folds them into the routed-output add with one kernel (see
        :func:`moe_shared_gate_add`). ``raw_gate`` is ``None`` when there is
        nothing for the caller to apply -- either the shared expert has no gate,
        or the gate projection itself is being folded into that same kernel.
        """
        if not self.has_shared_expert:
            return None, None
        shared_mlp = getattr(self, self.shared_expert_attr_name)
        out = shared_mlp(hidden_states)
        if self.shared_expert_gate is None:
            return out, None
        if hidden_states.shape[0] <= self._FUSE_GATE_MAX_TOKENS:
            return out, None
        return out, self.shared_expert_gate(hidden_states)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, self.hidden_size)
        if self._use_custom_op:
            # All-reduce stays outside the opaque op so the decoder's fused
            # AR+norm can still match it. vLLM / KimiMoE do the same.
            output = torch.ops.fastkernels.moe_forward(
                hidden_states, self._layer_name,
            )
            if self.tp_size > 1 and self.reduce_results:
                output = self.allreduce(output)
            return output.view(orig_shape)
        return self.forward_impl(hidden_states).view(orig_shape)

    def forward_impl(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._fast_ok(hidden_states):
            output = self._forward_fast(hidden_states)
            if self.tp_size > 1 and self.reduce_results:
                output = self.allreduce(output)
            return output
        return self._forward_baseline(hidden_states)

    def _fast_ok(self, hidden_states: torch.Tensor) -> bool:
        """Whether the fused pipeline handles this call.

        Above ``_FAST_MAX_TOKENS`` every expert is selected many times over, so
        trtllm-gen's padding no longer costs anything and its monolithic kernel
        is the faster of the two -- measured crossover is ~768 tokens on B200.
        """
        n = hidden_states.shape[0]
        return (
            self._fast
            and 0 < n <= self._FAST_MAX_TOKENS
            and hidden_states.is_cuda
            and hidden_states.dtype in (torch.bfloat16, torch.float16)
            and hidden_states.dtype == self.w13.dtype
        )

    def _forward_fast(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()
        n_tokens = hidden_states.shape[0]
        path = self._paths.get(n_tokens)
        if path is None:
            if self.fast_w13 is None:
                self.fast_w13, self.fast_w2 = self.w13.data, self.w2.data
            if len(self._paths) >= self._MAX_PATHS:
                self._paths.pop(next(iter(self._paths)))
            path = _FastPath(self, n_tokens)
            self._paths[n_tokens] = path
        return path(hidden_states)

    def _forward_baseline(self, hidden_states: torch.Tensor) -> torch.Tensor:
        shared_output, shared_gate = self._shared_expert_output(hidden_states)
        if self.gate_linear is not None:
            router_logits = self.gate_linear(
                hidden_states,
                self.gate.weight,
                out_dtype=torch.float32,
            )
        else:
            router_logits = self.gate(hidden_states)

        if self.use_trtllm:
            # Routing, both GEMMs and the weighted reduction happen inside the
            # kernel, including ``routed_scaling_factor``.
            routed_output = self.trtllm_moe(
                hidden_states,
                self.w13,
                self.w2,
                router_logits,
                routing_bias=(
                    self.gate.e_score_correction_bias
                    if self.correction_bias
                    else None
                ),
            )
        else:
            topk_weights, topk_ids = self._route(router_logits)
            if not self.keep_router_weights_fp32:
                topk_weights = topk_weights.to(hidden_states.dtype)

            routed_output = self.fused_experts(
                hidden_states, self.w13, self.w2,
                topk_weights, topk_ids, self.num_experts,
            )
            if self.routed_scaling_factor != 1.0:
                routed_output = routed_output * self.routed_scaling_factor

        # Add the shared expert's *unreduced* partial first, then all-reduce the
        # sum once. Both terms are per-rank partial sums over the same output
        # space, so summing before the reduce is exact, and it halves the
        # all-reduce count for layers that have a shared expert.
        if shared_output is None:
            output = routed_output
        elif self.shared_expert_gate is None:
            output = routed_output + shared_output
        elif shared_gate is None:
            # Small batch: the epilogue kernel projects the gate itself.
            output = moe_shared_gate_add(
                routed_output, shared_output,
                hidden_states=hidden_states,
                gate_weight=self.shared_expert_gate.weight,
            )
        else:
            output = moe_shared_gate_add(
                routed_output, shared_output, shared_gate,
            )
        if self.tp_size > 1 and self.reduce_results and not self._use_custom_op:
            output = self.allreduce(output)
        return output
