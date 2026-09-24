from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import triton
import triton.language as tl

from ....infra.cuda_ext import lazy_op
from ....infra.tp import _tp_rank, _tp_size
from ..L1.allreduce import AllReduce
from ..L1.gate_linear import GateLinear
from ..L1.grouped_topk import GroupedTopK
from ..L1.moe_align import MoeAlign, _C as _ALIGN_C
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

# ###########################################################################
# Fused shared-expert MoE (routing + both grouped GEMMs + shared expert +
# gated epilogue) written from scratch for this layer's shape.
#
# The reference path hands the routed experts to trtllm-gen and runs the shared
# expert, its gate projection and the epilogue as separate torch/Triton ops.
# Two things make that expensive here:
#
# * At decode sizes the layer is pure launch latency: a [1, 2048] forward is
#   ~41 us of GPU work, and the reference spends anywhere from 80 us to 0.97 ms
#   of wall time on it depending on how loaded the host is -- the host walking
#   the flashinfer autotuner plus ~20 torch ops. A python-level op costs 7-25 us
#   here, so the only way down is to issue fewer of them.
# * The shared expert is structurally just one more expert: its intermediate
#   size equals the routed experts' and its gate is a per-token scalar. Folding
#   it in as expert ``E`` with routing weight ``sigmoid(gate)`` deletes the two
#   dense GEMMs, the activation, the gate gemv and the epilogue add, and lets
#   the top-k reduction absorb the gating for free.
#
# So this path keeps one expert-weight tensor of ``E + 1`` experts and one
# router weight of ``E + 1`` rows, and runs: router GEMM -> top-k -> alignment
# -> grouped GEMM1 (SwiGLU in its epilogue) -> grouped GEMM2 -> top-k reduction,
# six launches with no intermediate the reference does not also materialize. A
# couple of tokens take a separate CUDA path (``shared_expert_moe_fk.cu``) that
# issues the whole layer from one pybind call.
#
# The grouped GEMMs run at the measured streaming bandwidth of this device
# (GEMM1 moves 839 MiB in 141 us at 26 tokens, 5.96 TB/s, against 5.62 TB/s for
# a plain contiguous read of the same bytes), so at decode sizes the remaining
# difference against the reference is launch count, not kernel quality.
# ###########################################################################

_C = lazy_op("shared_expert_moe_fk", "shared_expert_moe_fk.cu")

# Token count up to which the whole layer is issued from one pybind call (see
# ``shared_expert_moe_fk.cu``). Past a few tokens the grouped-GEMM path is GPU
# bound and the host has time to issue its launches while the GPU works, and its
# MMA tiles beat a gemv; at one token the launches *are* the latency, and this
# path takes the host off the critical path entirely.
_SMALL_MAX_TOKENS = 4

_MIN_BLOCKS = 296  # 2 waves on a 148-SM B200; target for split-K sizing

# Grouped-GEMM tiles per alignment block size, measured on B200 for this layer's
# N/K (GEMM1 2I=1024 x H=2048, GEMM2 H=2048 x I=512). Small BM is bandwidth
# bound -- the tile only has to keep enough weight loads in flight -- while
# BM>=128 is MMA bound and wants the widest BN that still fits TMEM.
# Key 0 is the unsorted (one-pair-per-block) regime, which wants narrow tiles to
# get enough CTAs out of a handful of pairs.
_G1_TILES = {0: (64, 64, 4, 4), 16: (256, 64, 4, 4), 64: (128, 64, 8, 3),
             128: (256, 64, 8, 4)}
_G2_TILES = {0: (128, 64, 4, 4), 16: (128, 64, 4, 3), 64: (128, 64, 4, 3),
             128: (256, 64, 8, 4)}


def _ROUTER_TILE(M: int):
    """(BM, BN, num_warps, num_stages) for the router projection.

    Tiny M is a bandwidth problem over 2 MiB of router weight (split-K carries
    the parallelism); large M is a real GEMM and wants the wide tile.
    """
    if M <= 64:
        return 16, 64, 4, 3
    if M <= 2048:
        return 64, 128, 8, 3
    return 128, 256, 8, 4


def _pick_tile(table: dict, bm: int, n: int, k: int):
    bn, bk, nw, ns = table[bm]
    while n % bn:
        bn //= 2
    while k % bk:
        bk //= 2
    return bn, bk, nw, ns


def _next_pow2(n: int) -> int:
    return 1 << max(0, (int(n) - 1).bit_length())


@triton.jit
def _router_gemm_kernel(
    X, W, OUT,
    M, sxm, som_k, som_m,
    K: tl.constexpr, EP: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, SK: tl.constexpr,
):
    """``OUT[sk, m, n] = sum_{k in chunk sk} X[m, k] * W[n, k]`` in fp32.

    ``W`` is the router weight with the shared-expert gate appended as row ``E``
    and zero rows up to ``EP``, so one GEMM produces the routing logits and the
    shared expert's gate. Split along K because at decode sizes the m/n tiling
    alone leaves most of the machine idle while 2 MiB of router weight streams.
    """
    pid = tl.program_id(0)
    sk = tl.program_id(1)
    nn = EP // BN
    pm = pid // nn
    pn = pid % nn
    om = pm * BM + tl.arange(0, BM)
    on = pn * BN + tl.arange(0, BN)
    ok = tl.arange(0, BK)
    kc = K // SK
    k0 = sk * kc
    mmask = om < M
    a_ptr = X + om[:, None] * sxm + (k0 + ok)[None, :]
    b_ptr = W + on[None, :] * K + (k0 + ok)[:, None]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for _ in range(kc // BK):
        a = tl.load(a_ptr, mask=mmask[:, None], other=0.0)
        b = tl.load(b_ptr)
        acc = tl.dot(a, b, acc)
        a_ptr += BK
        b_ptr += BK
    tl.store(OUT + sk * som_k + om[:, None] * som_m + on[None, :], acc,
             mask=mmask[:, None])


@triton.jit
def _topk_kernel(
    LOG, TW, TE,
    M, som_k, som_m, stw,
    E: tl.constexpr, TK: tl.constexpr, S: tl.constexpr,
    BE: tl.constexpr, SK: tl.constexpr, HAS_SHARED: tl.constexpr,
    ROUND_LOGITS: tl.constexpr,
):
    """softmax -> top-k -> renormalize, plus the shared expert's gate slot.

    ``softmax(l)_e / sum_{e in topk} softmax(l)_e`` is exactly a softmax over the
    selected logits, so the renormalized weights need no full-vector sum. The
    k-th largest value is found with TK max-reductions (masking by *index*, so
    duplicate logits cannot knock out two experts at once), then the selected
    experts are compacted with a cumsum -- top-k order is irrelevant downstream
    because the weights are only ever summed.
    """
    m = tl.program_id(0)
    if m >= M:
        return
    e = tl.arange(0, BE)
    base = LOG + m * som_m
    v = tl.load(base + e, mask=e < E, other=float("-inf"))
    for sk in tl.static_range(1, SK):
        v += tl.load(base + sk * som_k + e, mask=e < E, other=0.0)
    # The reference's router projection is an ``F.linear`` in the weight dtype, so
    # its logits reach the routing rounded to 16 bits. Near-ties in the top-k tail
    # are much coarser than fp32 would make them, and which expert wins changes
    # the output by a whole expert's contribution -- so round here too instead of
    # being gratuitously more precise than the thing we must match.
    if ROUND_LOGITS == 1:
        v = v.to(tl.bfloat16).to(tl.float32)
    elif ROUND_LOGITS == 2:
        v = v.to(tl.float16).to(tl.float32)
    vmax = tl.max(v, 0)
    cur = v
    thr = vmax
    for _ in range(TK):
        thr, am = tl.max(cur, 0, return_indices=True)
        cur = tl.where(e == am, float("-inf"), cur)
    # Everything strictly above the k-th value is in; the slots that remain go
    # to the lowest-indexed experts *tied at* the k-th value. Truncating the
    # whole ``v >= thr`` set by index instead would drop a higher-scoring expert
    # in favour of a tied-at-threshold one with a smaller index.
    gt = v > thr
    eq = v == thr
    room = TK - tl.sum(gt.to(tl.int32), 0)
    sel = gt | (eq & (tl.cumsum(eq.to(tl.int32), 0) - 1 < room))
    pos = tl.cumsum(sel.to(tl.int32), 0) - 1
    ex = tl.where(sel, tl.exp(v - vmax), 0.0)
    w = ex / tl.sum(ex, 0)
    tl.store(TW + m * stw + pos, w, mask=sel)
    tl.store(TE + m * stw + pos, e.to(tl.int32), mask=sel)
    if HAS_SHARED:
        g = tl.load(base + E)
        for sk in tl.static_range(1, SK):
            g += tl.load(base + sk * som_k + E)
        tl.store(TW + m * stw + TK, tl.sigmoid(g))
        tl.store(TE + m * stw + TK, E)


@triton.jit
def _moe_gemm1_kernel(
    X, SID, EID, NPP, W, C,
    numel, sxm, swe, scm,
    H: tl.constexpr, N2: tl.constexpr, S: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    SORTED: tl.constexpr,
):
    """Grouped ``[rows, H] x [H, 2I]`` with SwiGLU folded into the epilogue.

    ``W``'s output rows are stored gate/up *interleaved* (``2i`` gate, ``2i+1``
    up), so one ``BN``-wide accumulator holds both halves of ``BN/2`` activations
    and ``tl.split`` pairs them up -- no second accumulator (which spills TMEM and
    costs ~6x) and no separate pass over the 400+ MiB of raw GEMM1 output.

    Rows are gathered through ``SID`` (flat ``token * S + slot`` ids, padding
    entries equal ``numel``); padded rows read zero and store zero so GEMM2 needs
    no validity mask.
    """
    pid = tl.program_id(0)
    npn = N2 // BN
    pm = pid // npn
    pn = pid % npn
    rows = pm * BM + tl.arange(0, BM)
    if SORTED:
        if pm * BM >= tl.load(NPP):
            return
        pair = tl.load(SID + rows)
    else:
        # ``numel`` small enough that grouping buys nothing: one block per
        # (token, slot) pair, so no alignment kernel and no expert histogram.
        pair = tl.where(tl.arange(0, BM) == 0, pm, numel)
    e = tl.load(EID + pm)
    tok = (pair // S).to(tl.int32)
    ok = tl.arange(0, BK)
    on = pn * BN + tl.arange(0, BN)
    a_ptr = X + tok[:, None] * sxm + ok[None, :]
    b_ptr = W + e * swe + on[None, :] * H + ok[:, None]
    amask = (pair < numel)[:, None]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for _ in range(H // BK):
        a = tl.load(a_ptr, mask=amask, other=0.0)
        b = tl.load(b_ptr)
        acc = tl.dot(a, b, acc)
        a_ptr += BK
        b_ptr += BK
    g, u = tl.split(tl.reshape(acc, (BM, BN // 2, 2)))
    h = (g * tl.sigmoid(g)) * u
    oi = (pn * BN // 2) + tl.arange(0, BN // 2)
    tl.store(C + rows[:, None] * scm + oi[None, :], h.to(C.dtype.element_ty))


@triton.jit
def _moe_gemm2_kernel(
    G1, SID, EID, NPP, W, TW, C,
    numel, sg1, swe, scm,
    I: tl.constexpr, H: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    SORTED: tl.constexpr,
):
    """Grouped ``[rows, I] x [I, H]``, routing weight applied in the epilogue.

    Results are scattered to ``C[pair]`` rather than left in sorted order so the
    reduction reads a token's ``S`` slots contiguously.
    """
    pid = tl.program_id(0)
    npn = H // BN
    pm = pid // npn
    pn = pid % npn
    rows = pm * BM + tl.arange(0, BM)
    if SORTED:
        if pm * BM >= tl.load(NPP):
            return
        pair = tl.load(SID + rows)
    else:
        pair = tl.where(tl.arange(0, BM) == 0, pm, numel)
    e = tl.load(EID + pm)
    ok = tl.arange(0, BK)
    on = pn * BN + tl.arange(0, BN)
    a_ptr = G1 + rows[:, None] * sg1 + ok[None, :]
    b_ptr = W + e * swe + on[None, :] * I + ok[:, None]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for _ in range(I // BK):
        a = tl.load(a_ptr)
        b = tl.load(b_ptr)
        acc = tl.dot(a, b, acc)
        a_ptr += BK
        b_ptr += BK
    w = tl.load(TW + pair, mask=pair < numel, other=0.0)
    tl.store(C + pair[:, None] * scm + on[None, :],
             (acc * w[:, None]).to(C.dtype.element_ty),
             mask=(pair < numel)[:, None])


@triton.jit
def _reduce_kernel(OS, OUT, sos, sout, S: tl.constexpr, BN: tl.constexpr):
    """Sum a token's ``S`` slot outputs (routing weights already applied)."""
    m = tl.program_id(0)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    p = OS + m * S * sos + n
    acc = tl.load(p).to(tl.float32)
    for s in tl.static_range(1, S):
        acc += tl.load(p + s * sos).to(tl.float32)
    tl.store(OUT + m * sout + n, acc.to(OUT.dtype.element_ty))



class _FusedMoE:
    """Everything this layer needs, specialized once per module.

    Holds the ``E + 1`` expert weight tensors (routed experts plus the shared
    expert as expert ``E``), the ``EP x H`` router weight (routing logits plus
    the shared-expert gate as row ``E``), and a per-token-count plan of tile
    shapes, grids and scratch buffers so a forward issues no allocation-shaped
    python work beyond the launches themselves.
    """

    def __init__(self, mod: "SharedExpertMoE"):
        self.H = int(mod.hidden_size)
        self.E = int(mod.num_experts)
        self.TK = int(mod.top_k)
        self.I = int(mod.intermediate_per_tp)
        self.N2 = 2 * self.I
        self.has_shared = bool(mod.has_shared_expert)
        self.S = self.TK + (1 if self.has_shared else 0)
        self.EG = self.E + (1 if self.has_shared else 0)
        # Router output width, padded so one BN tile size divides it.
        self.EP = -(-(self.E + 1) // 256) * 256
        self.align = MoeAlign()
        # 1 = round the logits to bf16 before top-k, 2 = to fp16, 0 = not at all
        # (the grouped-top-k path's gate projection already returns fp32).
        self.round_logits = 0 if mod.gate_linear is not None else (
            1 if mod.w13.dtype == torch.bfloat16 else 2)
        # ``moe_topk`` keeps one expert score per register lane and packs it with
        # the expert index into a u32, so it needs E <= 32*32 and top_k <= 32.
        self.cuda_ok = (self.E <= 512 and self.TK <= 32 and self.round_logits != 0)
        # The single-call decode path additionally needs the shared expert (it is
        # the 11th slot) and the divisibility its gemv tiling assumes.
        self.small_ok = (self.cuda_ok and self.has_shared
                         and self.H % 128 == 0 and self.I % 32 == 0)
        self._small_fwd = None
        self._topk_fwd = None
        self._reduce_fwd = None
        self._align_fwd = None
        self.w13 = None
        self.w2 = None
        self.rw = None
        self._plans: dict[int, dict] = {}

    # -- eligibility ------------------------------------------------------
    @staticmethod
    def supported(mod: "SharedExpertMoE") -> bool:
        if mod.routing != "softmax" or not mod.renormalize:
            return False
        if mod.correction_bias or mod.use_grouped_topk:
            return False
        if mod.routed_scaling_factor != 1.0:
            return False
        if mod.tp_size != 1:
            return False
        if not torch.cuda.is_available():
            return False
        E, H, I, TK = mod.num_experts, mod.hidden_size, mod.intermediate_per_tp, mod.top_k
        if H % 64 or I % 64 or TK < 1 or TK > E:
            return False
        if mod.has_shared_expert:
            shared = getattr(mod, mod.shared_expert_attr_name, None)
            if shared is None:
                return False
            if tuple(shared.gate_up_proj.weight.shape) != (2 * I, H):
                return False
            if tuple(shared.down_proj.weight.shape) != (H, I):
                return False
            if mod.shared_expert_gate is None:
                return False
        # 32-bit element offsets everywhere (keeps the GEMM addressing cheap).
        if (E + 1) * 2 * I * H >= 2 ** 31:
            return False
        return True

    # -- weights ----------------------------------------------------------
    def prepare(self, mod: "SharedExpertMoE") -> None:
        """Fold the shared expert in as expert ``E`` and the gate as router row ``E``.

        The originals are replaced rather than kept beside the packed copies:
        w13 alone is 2 GiB at this layer's size.
        """
        if self.w13 is not None:
            return
        w13, w2 = mod.w13.data, mod.w2.data
        dev, dt = w13.device, w13.dtype
        E, I, H, N2 = self.E, self.I, self.H, self.N2
        if self.has_shared:
            shared = getattr(mod, mod.shared_expert_attr_name)
            gu = shared.gate_up_proj.weight.data
            dn = shared.down_proj.weight.data
            p13 = torch.empty(E + 1, N2, H, dtype=dt, device=dev)
            p13[:E, 0::2].copy_(w13[:, :I])
            p13[:E, 1::2].copy_(w13[:, I:])
            p13[E, 0::2].copy_(gu[:I])
            p13[E, 1::2].copy_(gu[I:])
            p2 = torch.empty(E + 1, H, I, dtype=dt, device=dev)
            p2[:E].copy_(w2)
            p2[E].copy_(dn)
            mod.w13 = nn.Parameter(p13, requires_grad=False)
            mod.w2 = nn.Parameter(p2, requires_grad=False)
        else:
            p13 = torch.empty(E, N2, H, dtype=dt, device=dev)
            p13[:, 0::2].copy_(w13[:, :I])
            p13[:, 1::2].copy_(w13[:, I:])
            mod.w13 = nn.Parameter(p13, requires_grad=False)
            p2 = w2.contiguous()
        rw = torch.zeros(self.EP, H, dtype=dt, device=dev)
        rw[:E].copy_(mod.gate.weight.data)
        if self.has_shared:
            rw[E].copy_(mod.shared_expert_gate.weight.data.reshape(H))
        self.w13, self.w2, self.rw = p13, p2, rw

    # -- per-M plan -------------------------------------------------------
    def _build_plan(self, M: int, dev, dt) -> dict:
        H, I, N2, EP, S, EG, TK, E = (
            self.H, self.I, self.N2, self.EP, self.S, self.EG, self.TK, self.E)

        # Router GEMM: split K until the grid covers the machine. 2 MiB of
        # router weight over three n-tiles is otherwise a handful of CTAs.
        bm_r, bn_r, nw_r, ns_r = _ROUTER_TILE(M)
        bk_r = 64
        nm = -(-M // bm_r)
        nn = EP // bn_r
        sk = 1
        while sk < 8 and nm * nn * sk < _MIN_BLOCKS and H % (2 * sk * bk_r) == 0:
            sk *= 2
        logits = torch.empty(sk, M, EP, dtype=torch.float32, device=dev)

        # Alignment block: match the average rows per expert so padded rows
        # (which cost real MMA work) stay in proportion to real ones. Below
        # ``EG / 4`` pairs almost every pair has its own expert, so grouping only
        # buys a few percent of weight traffic and is not worth the alignment
        # launch (which is pure fixed cost at decode sizes).
        numel = M * S
        avg = numel / EG
        bm = 128 if avg >= 48 else (64 if avg >= 24 else 16)
        sorted_rows = numel * 4 > EG
        if sorted_rows:
            max_padded = (numel * bm if numel < EG
                          else numel + EG * (bm - 1))
            max_blocks = -(-max_padded // bm)
        else:
            max_blocks = numel
            max_padded = numel * bm

        tw = torch.empty(M, S, dtype=torch.float32, device=dev)
        te = torch.empty(M, S, dtype=torch.int32, device=dev)
        # Resolve the frozen alignment op's buffers once and call its extension
        # directly: its python wrapper re-does a plan lookup and a reshape per
        # call, and a python-level call costs as much here as the kernel.
        align = (self.align._build_plan((te.shape, bm, EG, dev), te, bm, EG)
                 if sorted_rows else None)
        g1 = torch.empty(max_padded, I, dtype=dt, device=dev)
        oslots = torch.empty(numel, H, dtype=dt, device=dev)

        key = bm if sorted_rows else 0
        bn1, bk1, nw1, ns1 = _pick_tile(_G1_TILES, key, N2, H)
        bn2, bk2, nw2, ns2 = _pick_tile(_G2_TILES, key, H, I)
        red_bn = min(1024, H)
        return dict(
            sk=sk, bm_r=bm_r, bn_r=bn_r, bk_r=bk_r, nw_r=nw_r, ns_r=ns_r,
            grid_r=(nm * nn, sk),
            bm=bm, numel=numel, sorted_rows=sorted_rows,
            grid1=(max_blocks * (N2 // bn1),),
            bn1=bn1, bk1=bk1, nw1=nw1, ns1=ns1,
            grid2=(max_blocks * (H // bn2),),
            bn2=bn2, bk2=bk2, nw2=nw2, ns2=ns2,
            grid_red=(M, H // red_bn), red_bn=red_bn,
            logits=logits, tw=tw, te=te, g1=g1, oslots=oslots,
            te_flat=te.view(-1), align=align,
            be=_next_pow2(E),
        )

    def _plan(self, M: int, dev, dt) -> dict:
        p = self._plans.get(M)
        if p is None:
            p = self._build_plan(M, dev, dt)
            self._plans[M] = p
        return p

    # -- forward ----------------------------------------------------------
    def run(self, x: torch.Tensor) -> torch.Tensor:
        M = x.shape[0]
        if M == 0:
            return torch.empty_like(x)
        if x.stride(1) != 1:
            x = x.contiguous()
        if self.small_ok and M <= _SMALL_MAX_TOKENS:
            fwd = self._small_fwd
            if fwd is None:
                fwd = self._small_fwd = _C.shared_expert_moe_small
            return fwd(x, self.rw, self.w13, self.w2, self.TK, self.E,
                       self.has_shared)
        H, I, N2, EP, S, EG, TK, E = (
            self.H, self.I, self.N2, self.EP, self.S, self.EG, self.TK, self.E)
        p = self._plan(M, x.device, x.dtype)
        logits, tw, te, g1, oslots = (
            p["logits"], p["tw"], p["te"], p["g1"], p["oslots"])

        _router_gemm_kernel[p["grid_r"]](
            x, self.rw, logits,
            M, x.stride(0), M * EP, EP,
            K=H, EP=EP, BM=p["bm_r"], BN=p["bn_r"], BK=p["bk_r"], SK=p["sk"],
            num_warps=p["nw_r"], num_stages=p["ns_r"],
        )
        if self.cuda_ok:
            fn = self._topk_fwd
            if fn is None:
                fn = self._topk_fwd = _C.moe_topk
            fn(logits, tw, te, self.rw, E, TK, self.has_shared)
        else:
            _topk_kernel[(M,)](
                logits, tw, te,
                M, M * EP, EP, S,
                E=E, TK=TK, S=S, BE=p["be"], SK=p["sk"],
                HAS_SHARED=self.has_shared, ROUND_LOGITS=self.round_logits,
                num_warps=1,
            )
        if p["sorted_rows"]:
            sorted_ids, expert_ids, npp, n_exp, bs = p["align"]
            fn = self._align_fwd
            if fn is None:
                fn = self._align_fwd = _ALIGN_C.moe_align_fused
            fn(p["te_flat"], n_exp, bs, sorted_ids, expert_ids, npp)
        else:
            sorted_ids = expert_ids = p["te_flat"]
            npp = te
        _moe_gemm1_kernel[p["grid1"]](
            x, sorted_ids, expert_ids, npp, self.w13, g1,
            p["numel"], x.stride(0), N2 * H, I,
            H=H, N2=N2, S=S, BM=p["bm"], BN=p["bn1"], BK=p["bk1"],
            SORTED=p["sorted_rows"], num_warps=p["nw1"], num_stages=p["ns1"],
        )
        _moe_gemm2_kernel[p["grid2"]](
            g1, sorted_ids, expert_ids, npp, self.w2, tw, oslots,
            p["numel"], I, H * I, H,
            I=I, H=H, BM=p["bm"], BN=p["bn2"], BK=p["bk2"],
            SORTED=p["sorted_rows"], num_warps=p["nw2"], num_stages=p["ns2"],
        )
        if self.cuda_ok:
            fn = self._reduce_fwd
            if fn is None:
                fn = self._reduce_fwd = _C.moe_reduce
            return fn(oslots, M, S)
        out = torch.empty(M, H, dtype=x.dtype, device=x.device)
        _reduce_kernel[p["grid_red"]](
            oslots, out, H, H, S=S, BN=p["red_bn"], num_warps=4,
        )
        return out


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

        # trtllm-gen BF16 MoE: what vLLM 0.26 runs for this MoE on Blackwell
        # (``FLASHINFER_TRTLLM`` unquantized backend ->
        # ``TrtLlmBf16ExpertsMonolithic``). It fuses routing, both GEMMs and the
        # weighted reduction into one kernel, replacing gate + top-k +
        # ``_fused_moe_kernel``.
        self._trtllm_routing = trtllm_routing_method_type(
            routing, renormalize, correction_bias, num_expert_group,
        )
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

        self.has_shared_expert = shared_expert_intermediate_size > 0
        self.shared_expert_attr_name = shared_expert_attr_name
        if self.has_shared_expert:
            setattr(
                self,
                shared_expert_attr_name,
                _TPSwiGLUMLP(
                    hidden_size, shared_expert_intermediate_size,
                    # Defer the shared expert's reduce so it can be folded into
                    # the routed output's -- one all-reduce per layer instead of
                    # two. Decode profile: cross_device_reduce_1stage was 18.8%
                    # of decode time at ~3 all-reduces per layer per step where
                    # 2 suffice.
                    reduce_results=(_tp_size() == 1),
                ),
            )
        if shared_expert_gate:
            self.shared_expert_gate = ReplicatedLinear(hidden_size, 1, bias=False)
        else:
            self.shared_expert_gate = None

        # The hand-written fused path replaces routing + both grouped GEMMs +
        # the shared expert + the gated epilogue. When it applies, the trtllm-gen
        # weight shuffle must not run: this path wants the plain [E, 2I, H] /
        # [E, H, I] layout.
        # ``supported`` is a config check only: params are still fp32 here (the
        # harness casts after construction), so the dtype gate lives in
        # ``_get_fused``, which also builds the packed weights on first use.
        self._fused_ok = _FusedMoE.supported(self)
        self._fused = None
        if self._fused_ok:
            self.use_trtllm = False
            self.trtllm_moe = None

    def _get_fused(self):
        fused = self._fused
        if fused is None and self._fused_ok:
            if self.w13.dtype in (torch.bfloat16, torch.float16):
                fused = self._fused = _FusedMoE(self)
                fused.prepare(self)
            else:
                self._fused_ok = False
        return fused

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
        """Shuffle expert weights into trtllm-gen's 4D BlockMajorK layout.

        Mirrors vLLM's ``convert_to_unquantized_kernel_format`` for the
        ``FLASHINFER_TRTLLM`` backend, which runs once after loading. The
        original ``[E, 2*I, H]`` / ``[E, H, I]`` tensors are replaced, so the
        Triton path is unavailable afterwards -- guarded by ``use_trtllm``.
        """
        if self._fused_ok:
            self._get_fused()
            return
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
    # the epilogue kernel instead of run as its own gemv. The gemv is a
    # ``[hidden] -> [1]`` dot whose 5.33 us is all launch latency, so folding
    # wins while that dominates; past this the projection is a real GEMM and the
    # kernel's per-tile recomputation of the dot would start to cost more than
    # the launch it saves.
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
        if (self._fused_ok and not self._use_custom_op
                and hidden_states.dim() == 2):
            return self.forward_impl(hidden_states)
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
        fused = self._get_fused()
        if fused is not None:
            return fused.run(hidden_states)
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
        # all-reduce count for layers that have a shared expert. When the shared
        # expert is gated, its sigmoid, the scaling and the add are a single
        # kernel -- three separate elementwise launches per layer is what
        # Inductor fuses away for vLLM, and at batch 1 that is pure overhead.
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
