from __future__ import annotations

import os
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


# ---------------------------------------------------------------------------
# Small-batch (decode) fused path -- hand-written Triton.
# ---------------------------------------------------------------------------
# At the captured decode shapes (M in 1..69) the reference path is not
# GPU-bound at all: its GPU work is 43us at M=1 and 353us at M=60, while a
# forward takes ~940us of *host* time (~670us of it inside flashinfer's
# trtllm_bf16_moe python wrapper). Meanwhile top_k=10 of 512 experts touches
# only 10 experts at M=1 and ~360 at M=60, so the routed GEMMs are a batched
# GEMV whose floor is (touched expert weights)/HBM-bandwidth.
#
# So this path does two things: it issues the whole layer in a handful of
# Triton launches, and it reads each *touched* expert's weights exactly once.
# The shared expert is folded in as expert id ``E`` -- its gate_up/down have
# exactly the same shapes as a routed expert's w13/w2, and its sigmoid gate is
# just another per-token routing weight -- so the gated add is free.

_FAST_NSPLIT = 32
_FAST_BI = 32
_FAST_BH = 64


@triton.jit
def _gate_logits_kernel(
    x_ptr, gw_ptr, logits_ptr, cnt_ptr, y32_ptr, act_ptr, nact_ptr,
    M,
    H: tl.constexpr, E: tl.constexpr, LS: tl.constexpr,
    BM: tl.constexpr, BE: tl.constexpr, BK: tl.constexpr, NSPLIT: tl.constexpr,
):
    """router logits = x @ gate.T (rounded to bf16, as F.linear would), plus
    the two zero-fills the later kernels need (per-expert counters, fp32
    accumulator) so they do not each cost their own launch."""
    mb = tl.program_id(0)
    s = tl.program_id(1)
    rows = mb * BM + tl.arange(0, BM)
    rmask = rows < M

    # zero fp32 output accumulator slice [BM, H/NSPLIT]
    HS: tl.constexpr = H // NSPLIT
    if s < NSPLIT:
        cols = s * HS + tl.arange(0, HS)
        tl.store(y32_ptr + rows[:, None] * H + cols[None, :],
                 tl.zeros([BM, HS], tl.float32), mask=rmask[:, None])
    # zero the per-expert token counters (E of them) + set the shared slot
    if mb == 0 and s < NSPLIT:
        ce = s * BE + tl.arange(0, BE)
        tl.store(cnt_ptr + ce, tl.zeros([BE], tl.int32))
        if s == 0:
            tl.store(cnt_ptr + E + tl.arange(0, 1),
                     tl.full([1], M, tl.int32))
            # The shared expert is always active and takes active slot 0.
            tl.store(act_ptr + tl.arange(0, 1), tl.full([1], E, tl.int32))
            tl.store(nact_ptr + tl.arange(0, 1), tl.full([1], 1, tl.int32))

    ee = s * BE + tl.arange(0, BE)
    acc = tl.zeros([BM, BE], tl.float32)
    for k0 in range(0, H, BK):
        kk = k0 + tl.arange(0, BK)
        xt = tl.load(x_ptr + rows[:, None] * H + kk[None, :],
                     mask=rmask[:, None], other=0.0)
        wt = tl.load(gw_ptr + ee[:, None] * H + kk[None, :])
        acc = tl.dot(xt, tl.trans(wt), acc)
    tl.store(logits_ptr + rows[:, None] * LS + ee[None, :],
             acc.to(tl.bfloat16), mask=rmask[:, None])


@triton.jit
def _topk_scatter_kernel(
    logits_ptr, cnt_ptr, tok_ptr, wt_ptr, act_ptr, nact_ptr,
    MAXPE,
    E: tl.constexpr, LS: tl.constexpr, TOPK: tl.constexpr, TKP: tl.constexpr,
):
    """softmax-renormalized top-k, then scatter each (expert, token, weight)
    into that expert's bucket. One CTA per token; ranks come from an atomic
    counter so no sort/prefix-sum pass is needed.

    Selection is an iterative argmax that takes the *lowest* expert index
    among equal logits. That matters: the logits are bf16, so at these
    magnitudes ~12% of tokens have an exact tie between the 10th and 11th
    largest, and picking the other one changes the output well outside
    tolerance."""
    t = tl.program_id(0)
    ee = tl.arange(0, E)
    v = tl.load(logits_ptr + t * LS + ee).to(tl.float32)
    mx = tl.max(v, axis=0)

    kk = tl.arange(0, TKP)
    sel_id = tl.zeros([TKP], tl.int32)
    sel_p = tl.zeros([TKP], tl.float32)
    for k in range(TOPK):
        m2 = tl.max(v, axis=0)
        idx = tl.min(tl.where(v == m2, ee, E), axis=0)
        sel_id = tl.where(kk == k, idx, sel_id)
        sel_p = tl.where(kk == k, tl.exp(m2 - mx), sel_p)
        v = tl.where(ee == idx, float("-inf"), v)
    valid = kk < TOPK
    w = sel_p / tl.sum(tl.where(valid, sel_p, 0.0), axis=0)

    pos = tl.atomic_add(cnt_ptr + sel_id, 1, mask=valid)
    dst = sel_id * MAXPE + pos
    tl.store(tok_ptr + dst, t, mask=valid)
    tl.store(wt_ptr + dst, w, mask=valid)

    # Whoever wins rank 0 for an expert appends it to the dense active list, so
    # the expert kernels launch CTAs only for experts that actually have work
    # (a full [E+1, tiles] grid is 8208 CTAs and costs 6us of pure launch).
    first = valid & (pos == 0)
    nfirst = tl.sum(first.to(tl.int32), axis=0)
    if nfirst > 0:
        base = tl.atomic_add(nact_ptr, nfirst)
        rank = tl.cumsum(first.to(tl.int32), axis=0) - first.to(tl.int32)
        tl.store(act_ptr + base + rank, sel_id, mask=first)

    # The shared expert's gate projection rides along as column E of the same
    # gate GEMM (its weight row is appended to the router's), so here it is one
    # scalar load, and its sigmoid becomes expert E's routing weight.
    g = tl.load(logits_ptr + t * LS + E).to(tl.float32)
    tl.store(tok_ptr + E * MAXPE + t, t)
    tl.store(wt_ptr + E * MAXPE + t, 1.0 / (1.0 + tl.exp(-g)))


@triton.jit
def _moe_gemm1_kernel(
    x_ptr, w13_ptr, a_ptr, cnt_ptr, tok_ptr, act_ptr, nact_ptr,
    MAXPE,
    H: tl.constexpr, I: tl.constexpr,
    BM: tl.constexpr, BI: tl.constexpr, BK: tl.constexpr,
    NTILE: tl.constexpr,
):
    """a[slot, i] = silu(x @ w1[e].T) * (x @ w3[e].T) for every token routed to
    expert ``e``. One CTA per (expert, intermediate tile): untouched experts
    exit before reading a single weight byte, and a touched expert's w13 slice
    is streamed exactly once."""
    pid = tl.program_id(0)
    ai = pid // NTILE
    if ai < tl.load(nact_ptr):
        e = tl.load(act_ptr + ai)
        cnt = tl.load(cnt_ptr + e)
        ii = (pid % NTILE) * BI + tl.arange(0, BI)
        wb = w13_ptr + e.to(tl.int64) * (2 * I * H)
        for b0 in range(0, cnt, BM):
            slot = b0 + tl.arange(0, BM)
            smask = slot < cnt
            tok = tl.load(tok_ptr + e * MAXPE + slot, mask=smask, other=0)
            acc1 = tl.zeros([BM, BI], tl.float32)
            acc3 = tl.zeros([BM, BI], tl.float32)
            for k0 in range(0, H, BK):
                kk = k0 + tl.arange(0, BK)
                xt = tl.load(x_ptr + tok[:, None] * H + kk[None, :],
                             mask=smask[:, None], other=0.0)
                w1 = tl.load(wb + ii[:, None] * H + kk[None, :])
                w3 = tl.load(wb + (ii + I)[:, None] * H + kk[None, :])
                acc1 = tl.dot(xt, tl.trans(w1), acc1)
                acc3 = tl.dot(xt, tl.trans(w3), acc3)
            a = (acc1 * tl.sigmoid(acc1)) * acc3
            tl.store(a_ptr + (e * MAXPE + slot)[:, None] * I + ii[None, :],
                     a.to(tl.bfloat16), mask=smask[:, None])


@triton.jit
def _moe_gemm2_kernel(
    a_ptr, w2_ptr, y32_ptr, cnt_ptr, tok_ptr, wt_ptr, act_ptr, nact_ptr,
    MAXPE,
    H: tl.constexpr, I: tl.constexpr,
    BM: tl.constexpr, BH: tl.constexpr, BK: tl.constexpr,
    NTILE: tl.constexpr,
):
    """y[token] += weight * (a @ w2[e].T), scatter-accumulated in fp32. Same
    one-CTA-per-(expert, output tile) shape, so w2[e] is also read once."""
    pid = tl.program_id(0)
    ai = pid // NTILE
    if ai < tl.load(nact_ptr):
        e = tl.load(act_ptr + ai)
        cnt = tl.load(cnt_ptr + e)
        hh = (pid % NTILE) * BH + tl.arange(0, BH)
        wb = w2_ptr + e.to(tl.int64) * (H * I)
        for b0 in range(0, cnt, BM):
            slot = b0 + tl.arange(0, BM)
            smask = slot < cnt
            tok = tl.load(tok_ptr + e * MAXPE + slot, mask=smask, other=0)
            rw = tl.load(wt_ptr + e * MAXPE + slot, mask=smask, other=0.0)
            acc = tl.zeros([BM, BH], tl.float32)
            for k0 in range(0, I, BK):
                kk = k0 + tl.arange(0, BK)
                at = tl.load(a_ptr + (e * MAXPE + slot)[:, None] * I + kk[None, :],
                             mask=smask[:, None], other=0.0)
                w2t = tl.load(wb + hh[:, None] * I + kk[None, :])
                acc = tl.dot(at, tl.trans(w2t), acc)
            # ``sem="relaxed"``: the accumulator is not read until the next
            # kernel, so the default acq_rel fence on every one of these is
            # pure cost (it was worth 2x on a large-M variant of this scatter).
            tl.atomic_add(y32_ptr + tok[:, None] * H + hh[None, :],
                          acc * rw[:, None], mask=smask[:, None], sem="relaxed")


@triton.jit
def _cast_kernel(y32_ptr, out_ptr, n, BLK: tl.constexpr):
    off = tl.program_id(0) * BLK + tl.arange(0, BLK)
    m = off < n
    tl.store(out_ptr + off, tl.load(y32_ptr + off, mask=m).to(out_ptr.dtype.element_ty),
             mask=m)


class _MoEScratch:
    """Per-token-count scratch, shared by every layer (they run one at a time)."""

    __slots__ = ("logits", "cnt", "tok", "wt", "abuf", "y32", "act", "nact",
                 "maxpe", "nact_ub", "lstride")

    def __init__(self, M, H, E, I, topk, device):
        # An expert can, in the worst case, be picked by every token, so the
        # per-expert bucket has to be M wide.
        self.maxpe = M
        n = (E + 1) * self.maxpe
        self.lstride = E + E // _FAST_NSPLIT
        self.logits = torch.empty(M, self.lstride, dtype=torch.bfloat16,
                                  device=device)
        self.cnt = torch.empty(E + 1, dtype=torch.int32, device=device)
        self.tok = torch.empty(n, dtype=torch.int32, device=device)
        self.wt = torch.empty(n, dtype=torch.float32, device=device)
        self.abuf = torch.empty(n, I, dtype=torch.bfloat16, device=device)
        self.y32 = torch.empty(M, H, dtype=torch.float32, device=device)
        self.act = torch.empty(E + 1, dtype=torch.int32, device=device)
        self.nact = torch.empty(1, dtype=torch.int32, device=device)
        # Host-side upper bound on the number of active experts: M*top_k picks
        # can cover at most that many distinct experts, plus the shared one.
        self.nact_ub = min(E, M * topk) + 1


_SCRATCH: dict = {}


def _get_scratch(M, H, E, I, topk, device):
    key = (M, H, E, I, topk, device)
    s = _SCRATCH.get(key)
    if s is None:
        s = _MoEScratch(M, H, E, I, topk, device)
        _SCRATCH[key] = s
    return s


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

        self._fast_w13 = None
        self._fast_w2 = None
        self._fast_gate_w = None
        self._graphs = {}
        self._graph_ok = True

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

        n2 = self.intermediate_per_tp
        self._fast_ok = (
            routing == "softmax" and renormalize and not correction_bias
            and not use_grouped_topk and routed_scaling_factor == 1.0
            and self.has_shared_expert and self.shared_expert_gate is not None
            and tp == 1
            and shared_expert_intermediate_size == moe_intermediate_size
            and num_experts % _FAST_NSPLIT == 0
            and hidden_size % (_FAST_NSPLIT * 16) == 0
            and n2 % _FAST_BI == 0 and hidden_size % _FAST_BH == 0
        )

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
        # The small-M path needs the *unshuffled* [E, 2I, H] / [E, H, I]
        # weights, which the trtllm conversion below replaces. Build its fused
        # [E+1, ...] copy (expert E == the shared expert) first, then let the
        # original 3D tensors go.
        self._build_fast_weights()
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
        if (hidden_states.dim() == 2 and not self._use_custom_op
                and self._fast_ok and hidden_states.is_cuda
                and 0 < hidden_states.shape[0] <= self._FAST_MAX_TOKENS):
            # Every captured call site already hands us ``[tokens, hidden]``, so
            # the reshape/view pair around ``forward_impl`` is two no-op aten
            # dispatches (~3 us of *host* time) on a path whose entire GPU cost
            # at batch 1 is 27 us. ``_fast_ok`` implies tp == 1, so there is no
            # all-reduce to apply either.
            out = self._forward_fast(hidden_states)
            if out is not None:
                return out
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

    # Token count up to which the fused Triton path runs. Above it the layer is
    # a real (compute-bound) grouped GEMM and the reference kernel's tensor-core
    # schedule wins, so it keeps that shape.
    _FAST_MAX_TOKENS = 1024

    def _build_fast_weights(self) -> None:
        """Fuse the routed experts and the shared expert into one [E+1, ...] pair.

        The shared expert's ``gate_up``/``down`` have exactly a routed expert's
        w13/w2 shapes, so giving it expert id ``E`` lets one kernel do both and
        turns ``+ shared * sigmoid(gate)`` into just another routing weight.
        """
        if not self._fast_ok or self._fast_w13 is not None:
            return
        w13, w2 = self.w13.data, self.w2.data
        if w13.dim() != 3 or w2.dim() != 3:
            return
        E, n, H = self.num_experts, self.intermediate_per_tp, self.hidden_size
        shared = getattr(self, self.shared_expert_attr_name)
        gu = shared.gate_up_proj.weight.data
        dn = shared.down_proj.weight.data
        sg = self.shared_expert_gate.weight.data
        if (tuple(w13.shape) != (E, 2 * n, H) or tuple(w2.shape) != (E, H, n)
                or tuple(gu.shape) != (2 * n, H) or tuple(dn.shape) != (H, n)
                or sg.numel() != H or w13.dtype != torch.bfloat16):
            return
        fw13 = torch.empty(E + 1, 2 * n, H, dtype=w13.dtype, device=w13.device)
        fw13[:E].copy_(w13)
        fw13[E].copy_(gu)
        fw2 = torch.empty(E + 1, H, n, dtype=w2.dtype, device=w2.device)
        fw2[:E].copy_(w2)
        fw2[E].copy_(dn)
        # Router rows + the shared-expert gate row in one matrix, padded out to a
        # whole column block so the gate GEMM stays a clean tiling.
        be = E // _FAST_NSPLIT
        gw = torch.zeros(E + be, H, dtype=w13.dtype, device=w13.device)
        gw[:E].copy_(self.gate.weight.data)
        gw[E].copy_(sg.reshape(-1))
        self._fast_gate_w = gw
        self._fast_w13 = fw13
        self._fast_w2 = fw2

    # Replaying a captured graph costs ~2us of host time against ~15us per
    # Triton launch, and at M=1 the layer is host-issue-bound (5 launches
    # ~= 80us of python against ~40us of GPU work), so the graph is what makes
    # the decode shapes GPU-bound. Shapes are stable per call site, which is
    # the precondition; ``FASTKERNELS_MOE_NO_GRAPH=1`` forces the eager path
    # (needed to attribute per-kernel time in a profile).
    _MAX_GRAPHS = 32

    def _forward_fast(self, x: torch.Tensor) -> torch.Tensor | None:
        if self._fast_w13 is None:
            self._build_fast_weights()
            if self._fast_w13 is None:
                return None
        M = x.shape[0]
        if self._graph_ok:
            entry = self._graphs.get(M)
            if entry is None:
                entry = self._capture(x, M)
            if entry is not None:
                graph, xbuf, out = entry
                xbuf.copy_(x)
                graph.replay()
                return out
        return self._run_fast(x, M)

    def _capture(self, x: torch.Tensor, M: int):
        if (os.environ.get("FASTKERNELS_MOE_NO_GRAPH") == "1"
                or len(self._graphs) >= self._MAX_GRAPHS
                or torch.cuda.is_current_stream_capturing()):
            self._graph_ok = False
            return None
        try:
            xbuf = torch.empty_like(x)
            xbuf.copy_(x)
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    self._run_fast(xbuf, M)
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = self._run_fast(xbuf, M)
        except Exception:
            self._graph_ok = False
            self._graphs.clear()
            return None
        entry = (graph, xbuf, out)
        self._graphs[M] = entry
        return entry

    def _run_fast(self, x: torch.Tensor, M: int) -> torch.Tensor:
        H, E, I = self.hidden_size, self.num_experts, self.intermediate_per_tp
        topk = self.top_k
        sc = _get_scratch(M, H, E, I, topk, x.device)
        # Wider token blocks past the decode range: the gate weight is re-read
        # once per token block, so at M in the hundreds BM=16 turns a 2 MB
        # matrix into tens of MB of L2 traffic.
        bmg, gbk, gns = (16, 512, 4) if M <= 64 else (64, 256, 3)
        _gate_logits_kernel[(triton.cdiv(M, bmg), _FAST_NSPLIT + 1)](
            x, self._fast_gate_w, sc.logits, sc.cnt, sc.y32, sc.act, sc.nact, M,
            H=H, E=E, LS=sc.lstride, BM=bmg, BE=E // _FAST_NSPLIT, BK=gbk,
            NSPLIT=_FAST_NSPLIT, num_warps=2, num_stages=gns,
        )
        _topk_scatter_kernel[(M,)](
            sc.logits, sc.cnt, sc.tok, sc.wt, sc.act, sc.nact, sc.maxpe,
            E=E, LS=sc.lstride, TOPK=topk, TKP=triton.next_power_of_2(topk),
            num_warps=1,
        )
        nt1 = I // _FAST_BI
        _moe_gemm1_kernel[(sc.nact_ub * nt1,)](
            x, self._fast_w13, sc.abuf, sc.cnt, sc.tok, sc.act, sc.nact, sc.maxpe,
            H=H, I=I, BM=16, BI=_FAST_BI, BK=128, NTILE=nt1,
            num_warps=2, num_stages=5,
        )
        nt2 = H // _FAST_BH
        _moe_gemm2_kernel[(sc.nact_ub * nt2,)](
            sc.abuf, self._fast_w2, sc.y32, sc.cnt, sc.tok, sc.wt, sc.act, sc.nact,
            sc.maxpe, H=H, I=I, BM=16, BH=_FAST_BH, BK=128, NTILE=nt2,
            # More warps per CTA only pays once several tokens share an expert.
            num_warps=4 if M >= 128 else 2, num_stages=3,
        )
        out = torch.empty(M, H, dtype=x.dtype, device=x.device)
        _cast_kernel[(triton.cdiv(M * H, 2048),)](sc.y32, out, M * H, BLK=2048,
                                                  num_warps=4)
        return out

    def forward_impl(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if (self._fast_ok and hidden_states.shape[0] <= self._FAST_MAX_TOKENS
                and hidden_states.is_cuda and hidden_states.shape[0] > 0):
            out = self._forward_fast(hidden_states)
            if out is not None:
                if self.tp_size > 1 and self.reduce_results:
                    out = self.allreduce(out)
                return out
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
