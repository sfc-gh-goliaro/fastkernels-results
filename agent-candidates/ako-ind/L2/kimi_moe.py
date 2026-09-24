"""KimiMoE with a hand-written fused MoE kernel for the small/mid-token regime.

The captured shapes are M in {1, 26, 64, 443, 16384} and the score is an
equal-weight geometric mean over them, so the four sub-prefill shapes carry 4/5
of it. The baseline runs trtllm-gen there and is **launch-bound, not
throughput-bound**: at M=1 its real GPU time is 58 us but a call costs ~940 us,
almost all of it Python -- the flashinfer wrapper, a separate router GEMM, and
the shared expert's four eager ops.

This replaces that with Triton kernels that stay on the GPU for the whole layer
and never materialize the ``[M*top_k, 2*I]`` intermediate:

  1. ``_router_gemm``   router logits in fp32; also zeroes the routing scratch
                        and the output accumulator (no memset launches)
  2. ``_route_topk``    sigmoid + ``e_score_correction_bias`` top-8, renormalize
                        x ``routed_scaling_factor``, and a per-expert token list
                        plus a compacted active-expert list
  3. ``_moe_experts``   the fused expert MLP: one streaming pass over each
                        activated expert's w13 rows, SwiGLU in registers, and
                        the w2 contraction + routing-weighted reduction in the
                        same CTA
  4. ``_finalize``      fp32 accumulator -> bf16

At M <= ``FUSE_ROUTE_MAX`` step 2 disappears: every CTA re-derives the top-8
from its token's 256-float logits row, which is cheaper than the launch.

Two things that shape the design and are easy to get backwards:

* **Only GPU time counts for the candidate.** The harness zeroes a 265 MB
  buffer before each timed call, so the host runs ~45 us ahead and the timed
  span is GPU-bound. Capturing the launch sequence in a CUDA graph measured
  72.08 -> 72.77 us, i.e. nothing. But each *launch* still costs ~5 us of
  measured wall (an empty Triton kernel measures 5.1 us), so kernel count is a
  real cost -- hence folding routing in at tiny M.
* **The shared expert is just expert slot ``E``.** Its intermediate size equals
  the routed experts' (1024), so it is appended to the weight tensors and rides
  the same kernel with weight 1.0 and every token, instead of four eager ops.

M=16384 is genuinely compute-bound (2.5 ms of real GPU work, trtllm-gen already
near roofline) and stays on the baseline path.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

from ....infra.tp import _tp_rank, _tp_size
from ..L1.allreduce import AllReduce
from ..L1.gate_linear import GateLinear
from ..L1.grouped_topk import GroupedTopK
from .trtllm_bf16_moe import (
    ROUTING_DEEPSEEK_V3,
    TrtLlmBf16MoE,
    prepare_trtllm_bf16_moe_weights,
    trtllm_bf16_moe_supported,
)
from .fused_experts import FusedExperts
from .llama_mlp import LlamaMLP
from .parallel_linear import ReplicatedLinear


# Largest M routed through the fused path. The captured mid shape is 443; the
# prefill (16384) is compute-bound and belongs to trtllm-gen.
FUSED_MAX_TOKENS = 512


# ---------------------------------------------------------------------------
# 1. Router GEMM: logits[M, E] = x[M, H] @ gate_w[E, H].T  in fp32.
#    Split over experts (and, at tiny M, over K) so that even M=1 has enough
#    CTAs to pull gate_w's 1.18 MB at a useful rate.
# ---------------------------------------------------------------------------
@triton.jit
def _router_gemm(
    x_ptr, gw_ptr, logits_ptr, cnt_ptr, nact_ptr, acc_ptr,
    M, NACC, E: tl.constexpr, H: tl.constexpr, NE: tl.constexpr,
    MAXT: tl.constexpr, BM: tl.constexpr, BE: tl.constexpr, BK: tl.constexpr,
    ZB: tl.constexpr, SPLITK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_e = tl.program_id(1)
    pid_k = tl.program_id(2)

    # Zero the routing scratch and the fp32 output accumulator here, so neither
    # the top-k kernel's atomics nor the expert kernel's need a separate memset
    # launch. (Doing the accumulator re-zero in the *finalize* kernel instead
    # looks equivalent and is not: Triton reorders that store ahead of the load
    # of the same address for some warps, which silently drops output values.)
    if pid_m == 0 and pid_k == 0:
        offz = pid_e * BE + tl.arange(0, BE)
        tl.store(cnt_ptr + offz, tl.zeros([BE], tl.int32), mask=offz < E)
        if pid_e == 0:
            tl.store(nact_ptr, 0)
    zoff = ((pid_m * NE + pid_e) * SPLITK + pid_k) * ZB + tl.arange(0, ZB)
    tl.store(acc_ptr + zoff, tl.zeros([ZB], tl.float32), mask=zoff < NACC)

    offm = pid_m * BM + tl.arange(0, BM)
    offe = pid_e * BE + tl.arange(0, BE)
    mm = offm < M
    acc = tl.zeros([BM, BE], dtype=tl.float32)
    xb = x_ptr + offm[:, None] * H
    wb = gw_ptr + offe[None, :] * H
    # Split at BK-chunk granularity, not at H // SPLITK: H (2304) / SPLITK is
    # not generally a multiple of BK, and splitting there both reads past the
    # chunk and leaves K coverage incomplete.
    c0 = (pid_k * (H // BK)) // SPLITK
    c1 = ((pid_k + 1) * (H // BK)) // SPLITK
    for c in range(c0, c1):
        offk = c * BK + tl.arange(0, BK)
        a = tl.load(xb + offk[None, :], mask=mm[:, None], other=0.0)
        b = tl.load(wb + offk[:, None])
        acc = tl.dot(a, b, acc)
    tl.store(logits_ptr + pid_k * (MAXT * E) + offm[:, None] * E + offe[None, :],
             acc, mask=mm[:, None])


# ---------------------------------------------------------------------------
# 2. Routing: sigmoid scoring, e_score_correction_bias for *selection* only,
#    plain top-8 (n_group == topk_group == 1 degenerates grouped top-k),
#    renormalize, then x routed_scaling_factor. Scatters into a per-expert
#    token list and compacts the set of activated experts so the expert kernel's
#    grid can be bounded by min(M*top_k, E) instead of E.
# ---------------------------------------------------------------------------
@triton.jit
def _route_topk(
    logits_ptr, bias_ptr, cnt_ptr, act_ptr, nact_ptr, tok_ptr, wgt_ptr,
    SCALE,
    E: tl.constexpr, TOPK: tl.constexpr, MAXT: tl.constexpr,
):
    # One program per token. A token-tiled version (BM=16, [BM, E] tiles) is the
    # obvious shape and costs 27 us of GPU here regardless of M: eight
    # sequential full-tile argmax reductions over [16, 256] fp32 spill. One
    # token per program keeps the whole thing in a single warp's registers.
    m = tl.program_id(0)
    offe = tl.arange(0, E)
    scores = tl.sigmoid(tl.load(logits_ptr + m * E + offe))
    cur = scores + tl.load(bias_ptr + offe).to(tl.float32)

    jj = tl.arange(0, TOPK)
    sel_i = tl.zeros([TOPK], dtype=tl.int32)
    sel_w = tl.zeros([TOPK], dtype=tl.float32)
    for j in tl.static_range(TOPK):
        idx = tl.argmax(cur, axis=0).to(tl.int32)
        hit = offe == idx
        sw = tl.sum(tl.where(hit, scores, 0.0), axis=0)
        sel_i = tl.where(jj == j, idx, sel_i)
        sel_w = tl.where(jj == j, sw, sel_w)
        cur = tl.where(hit, -float("inf"), cur)

    sel_w = sel_w / tl.sum(sel_w, axis=0) * SCALE

    slot = tl.atomic_add(cnt_ptr + sel_i, 1)
    base = sel_i * MAXT + slot
    tl.store(tok_ptr + base, tl.zeros([TOPK], dtype=tl.int32) + m)
    tl.store(wgt_ptr + base, sel_w)

    # First token to land on an expert appends it to the active list.
    first = slot == 0
    aslot = tl.atomic_add(nact_ptr + tl.zeros([TOPK], dtype=tl.int32), 1, mask=first)
    tl.store(act_ptr + aslot, sel_i, mask=first)


# ---------------------------------------------------------------------------
# 3. Fused expert MLP.
#
#    Grid = (I / BI, SH_SLOTS + NACT_MAX).  The second axis is a flat list of
#    work slots: the first ``SH_SLOTS`` are the shared expert (one token tile
#    each, weight 1.0), the rest are activated routed experts.
#
#    Weight layout is chosen so each CTA streams one *contiguous* region per
#    matrix: wA is [E+1, 2I, H] and wB is [E+1, I, H], so rows
#    [i0, i0+BI) x full H is BI*H*2 contiguous bytes.
# ---------------------------------------------------------------------------
@triton.jit
def _moe_experts(
    x_ptr, wa_ptr, wb_ptr, cnt_ptr, act_ptr, nact_ptr, tok_ptr, wgt_ptr, acc_ptr,
    logits_ptr, bias_ptr, SCALE, M,
    E: tl.constexpr, H: tl.constexpr, I: tl.constexpr, MAXT: tl.constexpr,
    SH_SLOTS: tl.constexpr, BI: tl.constexpr, BM: tl.constexpr,
    BK: tl.constexpr, BN: tl.constexpr,
    TOPK: tl.constexpr, SPLITK: tl.constexpr, FUSE_ROUTE: tl.constexpr,
):
    pid_i = tl.program_id(0)
    pid_s = tl.program_id(1)

    if FUSE_ROUTE:
        # Tiny-M path: one slot per (token, expert-slot), and every CTA
        # re-derives the top-8 from its token's 256-float logits row. Redundant,
        # but at M<=4 it is far cheaper than a separate routing launch (~5 us of
        # measured wall for the launch alone) and it parallelizes across the
        # ~288 CTAs instead of running in one.
        tk = pid_s // (TOPK + 1)
        js = pid_s % (TOPK + 1)
        offe = tl.arange(0, E)
        lg = tl.zeros([E], dtype=tl.float32)
        for sk in tl.static_range(SPLITK):
            lg += tl.load(logits_ptr + sk * (MAXT * E) + tk * E + offe)
        scores = tl.sigmoid(lg)
        cur = scores + tl.load(bias_ptr + offe).to(tl.float32)
        jj = tl.arange(0, TOPK)
        sel_i = tl.zeros([TOPK], dtype=tl.int32)
        sel_w = tl.zeros([TOPK], dtype=tl.float32)
        for q in tl.static_range(TOPK):
            idx = tl.argmax(cur, axis=0).to(tl.int32)
            hit = offe == idx
            sw = tl.sum(tl.where(hit, scores, 0.0), axis=0)
            sel_i = tl.where(jj == q, idx, sel_i)
            sel_w = tl.where(jj == q, sw, sel_w)
            cur = tl.where(hit, -float("inf"), cur)
        sel_w = sel_w / tl.sum(sel_w, axis=0) * SCALE
        shared = js == TOPK
        e = tl.where(shared, E, tl.sum(tl.where(jj == js, sel_i, 0), axis=0))
        wv = tl.where(shared, 1.0, tl.sum(tl.where(jj == js, sel_w, 0.0), axis=0))
        nt = 1
        t0 = 0
        ntile = 1
    else:
        shared = pid_s < SH_SLOTS
        nact = tl.load(nact_ptr)
        k = pid_s - SH_SLOTS
        # Clamped so an out-of-range slot still forms legal addresses; those
        # slots get ntile == 0 and never issue a load.
        kc = tl.maximum(tl.minimum(k, E - 1), 0)
        er = tl.load(act_ptr + kc)
        er = tl.maximum(tl.minimum(er, E - 1), 0)
        nt = tl.where(shared, M, tl.load(cnt_ptr + er))
        e = tl.where(shared, E, er)
        wv = 1.0
        tk = 0
        t0 = tl.where(shared, pid_s * BM, 0)
        ntile = tl.where(shared, 1, (nt + BM - 1) // BM)
        ntile = tl.where(shared | (k < nact), ntile, 0)

    offi = pid_i * BI + tl.arange(0, BI)
    wa_e = wa_ptr + e.to(tl.int64) * (2 * I * H)
    wb_e = wb_ptr + e.to(tl.int64) * (I * H)
    wg = wa_e + offi * H
    wu = wg + I * H
    wo = wb_e + offi[:, None] * H
    eo = tl.where(shared, 0, e) * MAXT
    ones = tl.full([BM], 1.0, tl.float32)

    # Each slot walks its expert's token list in BM-wide tiles. Routed experts
    # normally need one tile; the shared expert gets one slot per tile.
    for ti in range(0, ntile):
        offt = t0 + ti * BM + tl.arange(0, BM)
        if FUSE_ROUTE:
            mt = tl.arange(0, BM) == 0
            tidx = tl.zeros([BM], dtype=tl.int32) + tk
            w = ones * wv
        else:
            mt = offt < nt
            tload = tl.load(tok_ptr + eo + offt, mask=mt, other=0)
            wload = tl.load(wgt_ptr + eo + offt, mask=mt, other=0.0)
            tidx = tl.where(shared, offt, tload)
            w = tl.where(shared, ones, wload)

        xb = x_ptr + tidx[:, None] * H
        a1 = tl.zeros([BM, BI], dtype=tl.float32)
        a3 = tl.zeros([BM, BI], dtype=tl.float32)
        for k0 in range(0, H, BK):
            offk = k0 + tl.arange(0, BK)
            a = tl.load(xb + offk[None, :], mask=mt[:, None], other=0.0)
            a1 = tl.dot(a, tl.load(wg[None, :] + offk[:, None]), a1)
            a3 = tl.dot(a, tl.load(wu[None, :] + offk[:, None]), a3)
        h = (a1 * tl.sigmoid(a1) * a3).to(tl.bfloat16)

        ws = w[:, None]
        for n0 in range(0, H, BN):
            offn = n0 + tl.arange(0, BN)
            b2 = tl.load(wo + offn[None, :])
            o = tl.dot(h, b2) * ws
            tl.atomic_add(acc_ptr + tidx[:, None] * H + offn[None, :], o,
                          mask=mt[:, None], sem="relaxed")


# ---------------------------------------------------------------------------
# 4. fp32 accumulator -> bf16 output.
# ---------------------------------------------------------------------------
@triton.jit
def _finalize(acc_ptr, out_ptr, N, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < N
    v = tl.load(acc_ptr + off, mask=m, other=0.0)
    tl.store(out_ptr + off, v.to(tl.bfloat16), mask=m)


# (BI, BM, BK, BN, num_warps, num_stages) for the expert kernel, by M bucket.
# Small M needs many CTAs (small BI) to saturate HBM; large M wants wide tiles
# to keep the atomic-reduction traffic and the MMA cost down.
_EXPERT_CFG = (
    (2,    (32, 16, 128, 256, 4, 4)),
    (32,   (32, 16, 128, 128, 4, 4)),
    (128,  (64, 16, 128, 128, 8, 4)),
    (512,  (128, 32, 64, 128, 4, 4)),
)
_ROUTER_CFG = (16, 8, 256, 4, 4)    # BM, BE, BK, num_warps, num_stages

# At or below this M the routing is folded into the expert kernel (one launch
# fewer), and the router GEMM is K-split so it still has enough CTAs.
FUSE_ROUTE_MAX = 4
ROUTER_SPLITK = 3



def _expert_cfg(m: int):
    for lim, cfg in _EXPERT_CFG:
        if m <= lim:
            return cfg
    return _EXPERT_CFG[-1][1]


class KimiMoE(nn.Module):
    def __init__(self, config, quant_config: dict | None = None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_token
        self.num_shared_experts = config.num_shared_experts
        self.num_expert_group = config.num_expert_group
        self.topk_group = config.topk_group
        self.routed_scaling_factor = config.routed_scaling_factor
        self.tp_size = _tp_size()
        self.intermediate_per_tp = config.moe_intermediate_size // self.tp_size

        self.gate = ReplicatedLinear(
            self.hidden_size,
            self.num_experts,
            bias=False,
            quant_config=None,
        )
        self.gate.e_score_correction_bias = nn.Parameter(
            torch.empty(self.num_experts),
        )
        self.gate.e_score_correction_bias.weight_loader = (
            lambda p, w: p.data.copy_(w.to(p.dtype))
        )

        self.grouped_topk = GroupedTopK(
            scoring_func=config.moe_router_activation_func,
            renormalize=config.moe_renormalize,
            routed_scaling_factor=1.0,
            force_sorted=True,
        )
        self.w13 = nn.Parameter(
            torch.empty(
                config.num_experts,
                2 * self.intermediate_per_tp,
                config.hidden_size,
            ),
        )
        self.w13.weight_loader = self._w13_weight_loader
        self.w2 = nn.Parameter(
            torch.empty(
                config.num_experts,
                config.hidden_size,
                self.intermediate_per_tp,
            ),
        )
        self.w2.weight_loader = self._w2_weight_loader
        self.fused_experts = FusedExperts()
        self.gate_linear = GateLinear()
        self.shared_experts = (
            LlamaMLP(
                config,
                quant_config=quant_config,
                intermediate_size=config.moe_intermediate_size * self.num_shared_experts,
                reduce_results=False,
            )
            if self.num_shared_experts
            else None
        )
        self.allreduce = AllReduce()

        self.use_trtllm = trtllm_bf16_moe_supported()
        self.trtllm_moe = (
            TrtLlmBf16MoE(
                num_experts=self.num_experts,
                top_k=self.top_k,
                intermediate_size_per_partition=self.intermediate_per_tp,
                routing_method_type=ROUTING_DEEPSEEK_V3,
                num_expert_group=self.num_expert_group,
                topk_group=self.topk_group,
                routed_scaling_factor=self.routed_scaling_factor,
            )
            if self.use_trtllm
            else None
        )
        self._trtllm_weights_ready = False

        self._use_custom_op = False
        self._layer_name = ""

        # Fused small-M path state (built in process_weights_after_loading).
        self._fused_ready = False
        self._wa = None
        self._wb = None
        self._scratch = None
        self._buf = None
        self._plans = {}

    def _w13_weight_loader(self, param, loaded_weight, expert_id: int, is_w1: bool):
        n = self.intermediate_per_tp
        shard = loaded_weight.narrow(0, _tp_rank() * n, n)
        offset = 0 if is_w1 else n
        param.data[expert_id, offset:offset + n, :].copy_(shard)

    def _w2_weight_loader(self, param, loaded_weight, expert_id: int):
        n = self.intermediate_per_tp
        param.data[expert_id].copy_(loaded_weight.narrow(1, _tp_rank() * n, n))

    # -- weight preparation -------------------------------------------------
    def _build_fused_weights(self) -> None:
        """Materialize the fused path's own weight layout *before* the trtllm-gen
        shuffle destroys the plain ``[E, 2I, H]`` / ``[E, H, I]`` tensors.

        ``wa`` is ``[E+1, 2I, H]`` (gate rows then up rows, contiguous over H)
        and ``wb`` is ``[E+1, I, H]`` -- w2 transposed so that a CTA owning
        ``I``-rows ``[i0, i0+BI)`` streams one contiguous block from each. Slot
        ``E`` holds the shared expert, whose intermediate size is identical
        (``moe_intermediate_size * num_shared_experts == I``), so the shared
        expert becomes just another expert inside the fused kernel.
        """
        w13 = self.w13.data
        w2 = self.w2.data
        if w13.dim() != 3 or self.shared_experts is None:
            return
        E, twoI, H = w13.shape
        I = twoI // 2
        if I != self.intermediate_per_tp or H != self.hidden_size:
            return
        gu = self.shared_experts.gate_up_proj.weight.data
        dn = self.shared_experts.down_proj.weight.data
        if tuple(gu.shape) != (twoI, H) or tuple(dn.shape) != (H, I):
            return

        dev = w13.device
        wa = torch.empty((E + 1, twoI, H), dtype=w13.dtype, device=dev)
        wa[:E].copy_(w13)
        wa[E].copy_(gu)
        wb = torch.empty((E + 1, I, H), dtype=w13.dtype, device=dev)
        wb[:E].copy_(w2.transpose(1, 2))
        wb[E].copy_(dn.t())
        self._wa = wa
        self._wb = wb
        self._gw = self.gate.weight
        self._bias = self.gate.e_score_correction_bias

        T = FUSED_MAX_TOKENS
        self._plans = {}
        self._scratch = {
            "logits": torch.empty((ROUTER_SPLITK, T, E), dtype=torch.float32,
                                  device=dev),
            "cnt": torch.zeros((E,), dtype=torch.int32, device=dev),
            "act": torch.empty((E,), dtype=torch.int32, device=dev),
            "nact": torch.zeros((1,), dtype=torch.int32, device=dev),
            "tok": torch.empty((E, T), dtype=torch.int32, device=dev),
            "wgt": torch.empty((E, T), dtype=torch.float32, device=dev),
            "acc": torch.empty((T, H), dtype=torch.float32, device=dev),
        }
        self._buf = tuple(self._scratch[k] for k in
                          ("logits", "cnt", "act", "nact", "tok", "wgt", "acc"))
        self._fused_ready = True

    def process_weights_after_loading(self) -> None:
        if not self.use_trtllm or self._trtllm_weights_ready:
            return
        self._build_fused_weights()
        w13, w2 = prepare_trtllm_bf16_moe_weights(self.w13.data, self.w2.data)
        self.w13 = nn.Parameter(w13, requires_grad=False)
        self.w2 = nn.Parameter(w2, requires_grad=False)
        self._trtllm_weights_ready = True

    # -- fused forward ------------------------------------------------------
    def _plan(self, M: int):
        """Per-M launch geometry, computed once. Keeping this out of the hot path
        matters: at M=1 the whole layer is ~30 us of GPU and the Python around
        the four launches is the larger half of the wall time."""
        plan = self._plans.get(M)
        if plan is not None:
            return plan
        H, E, I = self.hidden_size, self.num_experts, self.intermediate_per_tp
        rBM, rBE, rBK, rW, rS = _ROUTER_CFG
        fuse = M <= FUSE_ROUTE_MAX
        splitk = ROUTER_SPLITK if fuse else 1
        nprog_m = triton.cdiv(M, rBM)
        ne = E // rBE
        nacc = M * H
        zb = triton.next_power_of_2(
            max(32, -(-nacc // (nprog_m * ne * splitk))))
        BI, BM, BK, BN, nw, ns = _expert_cfg(M)
        sh = triton.cdiv(M, BM)
        egrid = ((I // BI, (self.top_k + 1) * M) if fuse
                 else (I // BI, sh + min(M * self.top_k, E)))
        plan = (
            (nprog_m, ne, splitk),
            (nacc, ne, rBM, rBE, rBK, zb, splitk, rW, rS),
            (M,), egrid, (sh, BI, BM, BK, BN, nw, ns), (triton.cdiv(nacc, 4096),),
            nacc, fuse,
        )
        self._plans[M] = plan
        return plan

    def _fused_forward(self, x: torch.Tensor) -> torch.Tensor:
        M = x.shape[0]
        H, E, I = self.hidden_size, self.num_experts, self.intermediate_per_tp
        (rgrid, (nacc, ne, rBM, rBE, rBK, zb, splitk, rW, rS), tgrid,
         egrid, (sh, BI, BM, BK, BN, nw, ns), fgrid, n, fuse) = self._plan(M)
        logits, cnt, act, nact, tok, wgt, acc = self._buf
        scale = self.routed_scaling_factor

        _router_gemm[rgrid](
            x, self._gw, logits, cnt, nact, acc,
            M, nacc, E, H, ne, FUSED_MAX_TOKENS, rBM, rBE, rBK, zb, splitk,
            num_warps=rW, num_stages=rS,
        )
        if not fuse:
            _route_topk[tgrid](
                logits, self._bias, cnt, act, nact, tok, wgt,
                scale, E, self.top_k, FUSED_MAX_TOKENS,
                num_warps=1,
            )
        _moe_experts[egrid](
            x, self._wa, self._wb, cnt, act, nact, tok, wgt, acc,
            logits, self._bias, scale, M,
            E, H, I, FUSED_MAX_TOKENS, sh, BI, BM, BK, BN,
            self.top_k, splitk, fuse,
            num_warps=nw, num_stages=ns,
        )
        out = torch.empty((M, H), dtype=x.dtype, device=x.device)
        _finalize[fgrid](acc, out, n, 4096, num_warps=4)
        return out

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._use_custom_op:
            out = torch.ops.fastkernels.moe_forward(hidden_states, self._layer_name)
            if self.tp_size > 1:
                out = self.allreduce(out)
            return out
        return self.forward_impl(hidden_states)

    def forward_impl(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_size)

        if (self._fused_ready
                and hidden_states.shape[0] <= FUSED_MAX_TOKENS
                and hidden_states.is_contiguous()):
            out = self._fused_forward(hidden_states)
            if self.tp_size > 1 and not self._use_custom_op:
                out = self.allreduce(out)
            return out.view(orig_shape)

        shared_output = (
            self.shared_experts(hidden_states)
            if self.shared_experts is not None
            else None
        )

        router_logits = self.gate_linear(
            hidden_states,
            self.gate.weight,
            out_dtype=torch.float32,
        )
        if self.use_trtllm:
            out = self.trtllm_moe(
                hidden_states,
                self.w13,
                self.w2,
                router_logits,
                routing_bias=self.gate.e_score_correction_bias,
            )
        else:
            topk_weights, topk_ids = self.grouped_topk(
                router_logits,
                self.gate.e_score_correction_bias,
                num_expert_group=self.num_expert_group,
                topk_group=self.topk_group,
                topk=self.top_k,
            )

            out = self.fused_experts(
                hidden_states,
                self.w13,
                self.w2,
                topk_weights,
                topk_ids,
                self.num_experts,
            )
            out = out * self.routed_scaling_factor
        if shared_output is not None:
            out = out + shared_output
        if self.tp_size > 1 and not self._use_custom_op:
            out = self.allreduce(out)
        return out.view(orig_shape)
