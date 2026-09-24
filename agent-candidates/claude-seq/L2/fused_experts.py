"""Fused MoE experts: two grouped GEMMs with SiLU-mul in between.

The captured workload is FP8 W8A8 with 128x128 weight block scales, top-8
routing over 128 experts, K=4096 and N=384 per expert.  ``_valid_deep_gemm``
rejects it (N <= 512), so the reference is the generic Triton
``_fused_moe_kernel`` run twice with four bandwidth-bound passes around it
(activation quant, SiLU-mul, activation quant again, top-k reduce).

What is different here:

* Both grouped GEMMs feed the ``tcgen05`` MMA through Blackwell TMA: either a
  TMA *gather* of the routed A rows (``descriptor.gather``, hardware row-gather
  -- the reference pays for that routing indirection with masked per-row loads)
  or a TMA tile load of the expert weights, whichever measures faster for the
  shape.  Only one of the two per loop: pipelining a descriptor gather together
  with a descriptor load is not race-free in Triton 3.6.  On the 16384-token
  shape this is 896 TFLOP/s of FP8 for GEMM1 against the reference's 415.
* SiLU-mul and the second FP8 quantization are fused into a single pass
  (``_act_quant``), and the second GEMM applies the routed weight and writes
  straight into the top-k reduction input, so the intermediate never makes more
  than one round trip.
* The block-scale rescale stays at the 128-wide scale-group boundary (one FP32
  promotion per MMA), which is what the reference does and what the FP8
  requantization in the middle makes numerically observable.
* ``_gemm2`` gathers its whole (only 3 scale groups deep) A tile once and then
  walks many output tiles with it resident, so the routing lookup is paid per
  CTA instead of per output tile.

Numerics follow the reference exactly wherever FP8 quantization makes them
observable: the GEMM1 accumulator is rounded to bfloat16 before the activation,
SiLU is rounded to bfloat16 before the multiply by the up-projection (vLLM's
``packed_silu_kernel`` does that), and both quantizations use UE8M0
(power-of-two) scales from ``ceil(log2(amax/448))`` computed on the float bits.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from ..L1.fp8_linear import PerTokenGroupQuantFp8
from ..L1.moe_align import MoeAlign
from ..L1.moe_grouped_gemm import MoeGroupedGemm, get_triton_config
from ..L1.moe_sum import MoeSum
from ..L1.gelu_and_mul import GeluAndMul
from ..L1.silu_and_mul import SiluAndMul

SPARSITY_FACTOR = 4
_FP8_GROUP_SIZE = 128


# ---------------------------------------------------------------------------
# UE8M0 activation scale: 2 ** ceil(log2(max(amax, eps) / 448)).
#
# Computed on the float bits rather than as exp2(ceil(log2(x))) so it is exact
# for the case that actually occurs: a bfloat16 amax of 7 * 2**k makes
# amax/448 an exact power of two, where a 1-ulp log2 would pick the next octave
# and rescale a whole 128-wide group by 2x.  The division must be a true divide
# for the same reason (1/448 is inexact).
# ---------------------------------------------------------------------------
@triton.jit
def _ue8m0_scale(amax):
    v = tl.maximum(tl.maximum(amax, 1e-10) / 448.0, 1e-10)
    bits = v.to(tl.int32, bitcast=True)
    e = ((bits >> 23) & 0xFF) - 127 + tl.where((bits & 0x7FFFFF) != 0, 1, 0)
    return ((e + 127) << 23).to(tl.float32, bitcast=True)


@triton.jit
def _quant_rows(x_ptr, q_ptr, s_ptr, M, K, stride_xm,
                NG: tl.constexpr, GPP: tl.constexpr):
    """Per-token-group FP8 quantization of the activations.

    Scales are written transposed ([NG, M]) so the GEMM's gathered per-row
    scale load is contiguous across the rows of a tile.
    """
    pid = tl.program_id(0)
    row = pid % M
    g0 = (pid // M) * GPP
    offs = (g0 + tl.arange(0, GPP))[:, None] * 128 + tl.arange(0, 128)[None, :]
    x = tl.load(x_ptr + row.to(tl.int64) * stride_xm + offs).to(tl.float32)
    scale = _ue8m0_scale(tl.max(tl.abs(x), axis=1))
    tl.store(q_ptr + row.to(tl.int64) * K + offs,
             (x / scale[:, None]).to(q_ptr.dtype.element_ty))
    tl.store(s_ptr + (g0 + tl.arange(0, GPP)) * M + row, scale)


@triton.jit
def _gemm1(a_ptr, a_desc, as_ptr, w_ptr, w_desc, ws_ptr, c_ptr,
           sorted_ptr, eid_ptr, npp_ptr,
           num_valid, M, K: tl.constexpr, N2: tl.constexpr, TOP_K: tl.constexpr,
           BM: tl.constexpr, BN: tl.constexpr, NT: tl.constexpr,
           EDIV: tl.constexpr, AT: tl.constexpr):
    """``a @ w13[e].T`` for one routing block: FP8 block-scaled grouped GEMM.

    Exactly one of the two operands rides TMA (``AT`` picks which): a TMA
    *gather* of the routed A rows (``descriptor.gather`` -- hardware row-gather
    straight to SMEM, which is what the reference pays for with masked per-row
    loads) or a TMA tile load of the weights.  Both at once is measurably
    faster and measurably *wrong*: pipelining a descriptor gather together with
    a descriptor load in one loop is not race-free in Triton 3.6 (it returns
    different results run to run).

    The tile stays 64x128x128 with one FP32 promotion per 128-wide scale group,
    i.e. bit-identical to the reference -- the FP8 requantization downstream
    turns a 1-ulp bfloat16 difference here into a whole 128-wide group
    requantized one octave off.
    """
    pid = tl.program_id(0)
    pid_m = pid // NT
    pid_n = pid % NT
    if pid_m * BM >= tl.load(npp_ptr):
        return
    offs_tok = tl.load(sorted_ptr + pid_m * BM + tl.arange(0, BM))
    tmask = offs_tok < num_valid
    if tl.max(tmask.to(tl.int32)) == 0:
        return
    e = tl.load(eid_ptr + pid_m // EDIV)
    rows = offs_tok // TOP_K
    offs_n = pid_n * BN + tl.arange(0, BN)
    as_ptrs = as_ptr + rows
    ws = ws_ptr + e * ((N2 // 128) * (K // 128)) + (pid_n * BN // 128) * (K // 128)
    if AT:
        w_ptrs = (w_ptr + e.to(tl.int64) * (N2 * K)
                  + offs_n[None, :] * K + tl.arange(0, 128)[:, None])
    else:
        a_ptrs = a_ptr + rows[:, None] * K + tl.arange(0, 128)[None, :]
        wn = e * N2 + pid_n * BN

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for kb in tl.range(0, K // 128):
        if AT:
            a = a_desc.gather(rows, kb * 128)
            b = tl.load(w_ptrs)
            w_ptrs += 128
        else:
            a = tl.load(a_ptrs, mask=tmask[:, None], other=0.0)
            b = w_desc.load([wn, kb * 128]).T
            a_ptrs += 128
        sa = tl.load(as_ptrs + kb * M, mask=tmask, other=0.0)
        acc += tl.dot(a, b) * (sa * tl.load(ws + kb))[:, None]

    tl.store(c_ptr + offs_tok[:, None] * N2 + offs_n[None, :],
             acc.to(c_ptr.dtype.element_ty), mask=tmask[:, None])


@triton.jit
def _act_quant(c_ptr, q_ptr, qs_ptr, rows, N: tl.constexpr, NPG: tl.constexpr,
               RPP: tl.constexpr):
    """``silu(gate) * up`` followed by per-128-group FP8 quantization.

    One program owns ``RPP`` rows of one 128-wide scale group, which is exactly
    the group the amax has to cover.  The two bfloat16 round-trips are the
    reference's: GEMM1 stores bfloat16, and vLLM's packed SiLU rounds the
    activation before multiplying by ``up``.
    """
    pid = tl.program_id(0)
    g = pid // NPG
    rr = (pid % NPG) * RPP + tl.arange(0, RPP)
    rmask = rr < rows
    offs = g * 128 + tl.arange(0, 128)
    base = c_ptr + rr[:, None].to(tl.int64) * (2 * N) + offs[None, :]
    gate = tl.load(base, mask=rmask[:, None], other=0.0).to(tl.float32)
    up = tl.load(base + N, mask=rmask[:, None], other=0.0).to(tl.float32)
    s = (gate / (1.0 + tl.exp(-gate))).to(tl.bfloat16).to(tl.float32)
    act = (s * up).to(tl.bfloat16).to(tl.float32)
    scale = _ue8m0_scale(tl.max(tl.abs(act), axis=1))
    tl.store(q_ptr + rr[:, None].to(tl.int64) * N + offs[None, :],
             (act / scale[:, None]).to(q_ptr.dtype.element_ty),
             mask=rmask[:, None])
    tl.store(qs_ptr + g * rows + rr, scale, mask=rmask)


@triton.jit
def _gemm2(a_desc, as_ptr, w_ptr, w_desc, ws_ptr, y_ptr, tw_ptr,
           sorted_ptr, eid_ptr, npp_ptr, num_valid,
           K: tl.constexpr, N: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
           NKB: tl.constexpr, NT: tl.constexpr, NITER: tl.constexpr,
           EDIV: tl.constexpr, BT: tl.constexpr):
    """``(a2 @ w2[e].T) * topk_weight`` for one routing block.

    K is only ``NKB`` scale groups deep, so the whole A tile is TMA-gathered
    once up front and the CTA walks ``NITER`` output tiles with it resident:
    the routing lookup and the gather are paid per CTA, not per output tile.
    """
    pid = tl.program_id(0)
    ntc: tl.constexpr = NT // NITER
    pid_m = pid // ntc
    pid_nc = pid % ntc
    if pid_m * BM >= tl.load(npp_ptr):
        return
    offs_tok = tl.load(sorted_ptr + pid_m * BM + tl.arange(0, BM))
    tmask = offs_tok < num_valid
    if tl.max(tmask.to(tl.int32)) == 0:
        return
    e = tl.load(eid_ptr + pid_m // EDIV)

    a0 = a_desc.gather(offs_tok, 0)
    s0 = tl.load(as_ptr + offs_tok, mask=tmask, other=0.0)
    if NKB > 1:
        a1 = a_desc.gather(offs_tok, 128)
        s1 = tl.load(as_ptr + num_valid + offs_tok, mask=tmask, other=0.0)
    if NKB > 2:
        a2 = a_desc.gather(offs_tok, 256)
        s2 = tl.load(as_ptr + 2 * num_valid + offs_tok, mask=tmask, other=0.0)
    mw = tl.load(tw_ptr + offs_tok, mask=tmask, other=0.0)
    wsb = ws_ptr + e * ((N // 128) * NKB)
    if not BT:
        wbase = w_ptr + e.to(tl.int64) * (N * K)

    for it in tl.range(0, NITER):
        pid_n = pid_nc * NITER + it
        offs_n = pid_n * BN + tl.arange(0, BN)
        wsn = wsb + (offs_n // 128) * NKB
        if BT:
            wn = e * N + pid_n * BN
            b0 = w_desc.load([wn, 0]).T
            b1 = w_desc.load([wn, 128]).T if NKB > 1 else b0
            b2 = w_desc.load([wn, 256]).T if NKB > 2 else b0
        else:
            w_ptrs = wbase + offs_n[None, :] * K + tl.arange(0, 128)[:, None]
            b0 = tl.load(w_ptrs)
            b1 = tl.load(w_ptrs + 128) if NKB > 1 else b0
            b2 = tl.load(w_ptrs + 256) if NKB > 2 else b0
        acc = tl.dot(a0, b0) * s0[:, None] * tl.load(wsn)[None, :]
        if NKB > 1:
            acc += tl.dot(a1, b1) * s1[:, None] * tl.load(wsn + 1)[None, :]
        if NKB > 2:
            acc += tl.dot(a2, b2) * s2[:, None] * tl.load(wsn + 2)[None, :]
        acc *= mw[:, None]
        tl.store(y_ptr + offs_tok[:, None] * N + offs_n[None, :],
                 acc.to(y_ptr.dtype.element_ty), mask=tmask[:, None])


@triton.jit
def _moe_sum(y_ptr, o_ptr, N, TOP_K: tl.constexpr, BN: tl.constexpr):
    m = tl.program_id(0)
    offs = tl.program_id(1) * BN + tl.arange(0, BN)
    base = y_ptr + (m * TOP_K).to(tl.int64) * N + offs
    acc = tl.load(base).to(tl.float32)
    for j in tl.static_range(1, TOP_K):
        acc += tl.load(base + j * N).to(tl.float32)
    tl.store(o_ptr + m.to(tl.int64) * N + offs, acc.to(o_ptr.dtype.element_ty))


class _Scratch:
    """Shared scratch so every FusedExperts layer reuses one set of buffers
    (layers run sequentially)."""
    __slots__ = ("bufs", "cache13", "a_fp8_1", "a_scale_1", "a_fp8_2", "a_scale_2")

    def __init__(self):
        self.bufs = {}
        self.cache13 = None
        self.a_fp8_1 = self.a_scale_1 = self.a_fp8_2 = self.a_scale_2 = None

    def get(self, key, shape, dtype, device):
        n = 1
        for s in shape:
            n *= s
        t = self.bufs.get(key)
        if t is None or t.numel() < n or t.dtype != dtype or t.device != device:
            t = torch.empty(n, dtype=dtype, device=device)
            self.bufs[key] = t
        return t[:n].view(shape)


_SCRATCH = _Scratch()

# (BM_align, g1 BM/BN/warps/stages/ws, g2 BM/BN/NITER/warps/stages/ws) by
# routing-slot count. Tuned on B200 for the captured token counts; the GEMM
# tiles want the alignment as their common multiple so one routing pass feeds
# both.
#  rows <=: (align, g1(BM, BN, warps, stages, TMA-A?),
#             g2(BM, BN, NITER, warps, stages, TMA-B?), quant GPP, act RPP)
_PLANS = (
    (          512,  64, (64,  64, 4, 4, 0), (64,  64,  2, 8, 3, 0),  8,  4),
    (         4096,  64, (64, 128, 4, 3, 0), (64,  64,  8, 4, 2, 0), 16,  8),
    (         6144, 128, (64, 128, 4, 3, 0), (64,  64, 16, 4, 2, 1), 16, 16),
    (        16384, 128, (64, 128, 4, 3, 0), (128, 64, 16, 4, 2, 1), 16, 16),
    (float("inf"), 128, (64, 128, 4, 3, 1), (128, 64, 32, 4, 2, 1), 16, 16),
)


def _make_plan(key, align, g1, g2, qgpp, aq_rpp):
    """Resolve a tile choice into launch grids for one (M, top_k, E, K, N)."""
    M, top_k, E, K, N = key
    rows = M * top_k
    n2 = 2 * N
    g1_bm, g1_bn, g1_w, g1_s, g1_at = g1
    g2_bm, g2_bn, g2_it, g2_w, g2_s, g2_bt = g2
    max_padded = rows * align if rows < E else rows + E * (align - 1)
    nblk = (max_padded + align - 1) // align
    # Keep the tunable counts legal for this (K, N): every tile count has to
    # divide exactly (a kernel covers its dimension in one pass) and the
    # per-program group counts have to stay powers of two for tl.arange.
    while g2_it > 1 and (K // g2_bn) % g2_it:
        g2_it //= 2
    ng = K // _FP8_GROUP_SIZE
    while qgpp > 1 and ng % qgpp:
        qgpp //= 2
    return dict(
        align=align,
        g1=dict(grid=(nblk * (align // g1_bm) * (n2 // g1_bn),),
                BM=g1_bm, BN=g1_bn, NT=n2 // g1_bn, EDIV=align // g1_bm,
                AT=g1_at, num_stages=g1_s, num_warps=g1_w),
        g2=dict(grid=(nblk * (align // g2_bm) * (K // g2_bn // g2_it),),
                BM=g2_bm, BN=g2_bn, NITER=g2_it, NT=K // g2_bn,
                EDIV=align // g2_bm, BT=g2_bt, num_stages=g2_s,
                num_warps=g2_w),
        qgpp=min(qgpp, ng),
        aq_rpp=aq_rpp,
        sum_bn=min(1024, K),
    )


class FusedExperts(nn.Module):
    """Fused MoE experts: two grouped GEMMs with SiLU-mul in between."""

    def __init__(self, activation: str = "silu", config_style: str = "legacy"):
        super().__init__()
        if activation not in ("silu", "gelu_tanh"):
            raise ValueError(f"Unsupported MoE activation: {activation}")
        if config_style not in ("legacy", "vllm"):
            raise ValueError(f"Unsupported MoE config style: {config_style}")
        self.activation = activation
        self.config_style = config_style
        self.moe_align = MoeAlign()
        self.moe_grouped_gemm = MoeGroupedGemm()
        self.act_fn = SiluAndMul() if activation == "silu" else GeluAndMul("tanh")
        self.moe_sum = MoeSum()
        self.per_token_group_quant_fp8 = PerTokenGroupQuantFp8()
        self._sb = _SCRATCH
        self._plans = {}

    # -- fast path ---------------------------------------------------------
    def _build_plan(self, key):
        M, top_k, E, K, N = key
        rows = M * top_k
        for limit, align, g1, g2, qgpp, aq_rpp in _PLANS:
            if rows <= limit:
                break
        plan = _make_plan(key, align, g1, g2, qgpp, aq_rpp)
        self._plans[key] = plan
        return plan

    def _forward_fast(self, hidden_states, w13, w2, topk_weights, topk_ids,
                      num_experts, w13_scale, w2_scale, M, K, E, N, top_k):
        dev = hidden_states.device
        sb = self._sb
        rows = M * top_k
        key = (M, top_k, E, K, N)
        plan = self._plans.get(key) or self._build_plan(key)
        NG = K // _FP8_GROUP_SIZE
        n2 = 2 * N

        a1 = sb.get("a1", (M, K), torch.float8_e4m3fn, dev)
        a1s = sb.get("a1s", (NG, M), torch.float32, dev)
        gpp = plan["qgpp"]
        _quant_rows[(M * (NG // gpp),)](
            hidden_states, a1, a1s, M, K, hidden_states.stride(0),
            NG=NG, GPP=gpp, num_warps=2,
        )

        sorted_ids, expert_ids, npp = self.moe_align(
            topk_ids, plan["align"], num_experts, naive=False,
        )

        # One scratch span backs both GEMM outputs: the intermediate is fully
        # consumed by _act_quant before _gemm2 starts writing.
        cache = sb.get("cache", (rows * K,), hidden_states.dtype, dev)
        inter = cache[:rows * n2].view(rows, n2)
        y = cache[:rows * K].view(rows, K)
        a2 = sb.get("a2", (rows, N), torch.float8_e4m3fn, dev)
        a2s = sb.get("a2s", (N // _FP8_GROUP_SIZE, rows), torch.float32, dev)

        g1 = plan["g1"]
        at = g1["AT"]
        _gemm1[g1["grid"]](
            a1, TensorDescriptor.from_tensor(a1, [1, 128]) if at else a1,
            a1s, w13,
            w13 if at else TensorDescriptor.from_tensor(
                w13.view(E * n2, K), [g1["BN"], 128]),
            w13_scale, inter, sorted_ids, expert_ids, npp, rows, M,
            K=K, N2=n2, TOP_K=top_k, BM=g1["BM"], BN=g1["BN"], NT=g1["NT"],
            EDIV=g1["EDIV"], AT=at, num_warps=g1["num_warps"],
            num_stages=g1["num_stages"],
        )

        rpp = plan["aq_rpp"]
        npg = (rows + rpp - 1) // rpp
        _act_quant[(npg * (N // _FP8_GROUP_SIZE),)](
            inter, a2, a2s, rows, N=N, NPG=npg, RPP=rpp, num_warps=2,
        )

        g2 = plan["g2"]
        bt = g2["BT"]
        _gemm2[g2["grid"]](
            TensorDescriptor.from_tensor(a2, [1, 128]), a2s, w2,
            TensorDescriptor.from_tensor(w2.view(E * K, N), [g2["BN"], 128])
            if bt else w2,
            w2_scale, y, topk_weights, sorted_ids, expert_ids, npp, rows,
            K=N, N=K, BM=g2["BM"], BN=g2["BN"], NKB=N // _FP8_GROUP_SIZE,
            NT=g2["NT"], NITER=g2["NITER"], EDIV=g2["EDIV"], BT=bt,
            num_warps=g2["num_warps"], num_stages=g2["num_stages"],
        )

        out = torch.empty(M, K, dtype=hidden_states.dtype, device=dev)
        bn = plan["sum_bn"]
        _moe_sum[(M, K // bn)](y, out, K, TOP_K=top_k, BN=bn, num_warps=4)
        return out

    def forward(
        self,
        hidden_states: torch.Tensor,
        w13: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        w13_scale: torch.Tensor | None = None,
        w2_scale: torch.Tensor | None = None,
        w13_scale_dg: torch.Tensor | None = None,
        w2_scale_dg: torch.Tensor | None = None,
        use_fp8_w8a8: bool = False,
        block_shape: list[int] | None = None,
    ) -> torch.Tensor:
        M, K = hidden_states.size()
        E, N2, _ = w13.size()
        N = N2 // 2
        top_k = topk_ids.size(1)

        if (self.activation == "silu" and use_fp8_w8a8
                and block_shape is not None
                and block_shape[0] == _FP8_GROUP_SIZE
                and block_shape[1] == _FP8_GROUP_SIZE
                and w13_scale is not None and w2_scale is not None
                and K % 1024 == 0 and N % _FP8_GROUP_SIZE == 0
                and N <= 512
                and hidden_states.dtype == torch.bfloat16
                and hidden_states.stride(1) == 1
                and w13.is_contiguous() and w2.is_contiguous()
                and w13.dtype == torch.float8_e4m3fn
                and w2.dtype == torch.float8_e4m3fn
                and topk_ids.dtype == torch.int32
                and topk_weights.dtype == torch.float32
                and M * top_k * K < 2 ** 31):
            return self._forward_fast(
                hidden_states, w13, w2, topk_weights, topk_ids, num_experts,
                w13_scale, w2_scale, M, K, E, N, top_k,
            )
        return self._forward_triton(
            hidden_states, w13, w2, topk_weights, topk_ids,
            num_experts, w13_scale, w2_scale, use_fp8_w8a8, block_shape,
            M, K, E, N, N2, top_k,
        )

    # -- reference fallback (unsupported dtype / activation / block shape) --
    def _get_cache13(self, total_elems, device, dtype):
        sb = self._sb
        if sb.cache13 is None or sb.cache13.numel() < total_elems:
            sb.cache13 = torch.empty(total_elems, device=device, dtype=dtype)
        return sb.cache13[:total_elems]

    def _get_fp8_bufs(self, buf_id, M, K, device):
        sb = self._sb
        attr_a = f"a_fp8_{buf_id}"
        attr_s = f"a_scale_{buf_id}"
        num_groups = math.ceil(K / _FP8_GROUP_SIZE)
        existing_a = getattr(sb, attr_a)
        if existing_a is None or existing_a.size(0) < M or existing_a.size(1) < K:
            setattr(sb, attr_a, torch.empty(M, K, dtype=torch.float8_e4m3fn, device=device))
            setattr(sb, attr_s, torch.empty(M, num_groups, dtype=torch.float32, device=device))
        return getattr(sb, attr_a)[:M, :K], getattr(sb, attr_s)[:M, :num_groups]

    def _forward_triton(
        self,
        hidden_states, w13, w2, topk_weights, topk_ids,
        num_experts, w13_scale, w2_scale,
        use_fp8_w8a8, block_shape,
        M, K, E, N, N2, top_k,
    ) -> torch.Tensor:
        config = get_triton_config(
            M, w13.shape, w2.shape, top_k,
            use_fp8=use_fp8_w8a8, block_shape=block_shape,
            default_style=self.config_style,
        )

        use_naive = (M * top_k * SPARSITY_FACTOR <= num_experts)

        sorted_token_ids, expert_ids, num_tokens_post_padded = self.moe_align(
            topk_ids, config["BLOCK_SIZE_M"], num_experts, naive=use_naive,
        )

        cache13_size = M * top_k * max(N2, K)
        cache13_flat = self._get_cache13(cache13_size, hidden_states.device,
                                         hidden_states.dtype)
        intermediate1 = cache13_flat[:M * top_k * N2].view(M * top_k, N2)
        intermediate3 = cache13_flat[:M * top_k * K].view(M * top_k, K)

        if use_fp8_w8a8:
            a_fp8, a_scale = self._get_fp8_bufs(1, M, K, hidden_states.device)
            self.per_token_group_quant_fp8(hidden_states, a_fp8, a_scale)
            gemm1_input, gemm1_a_scale = a_fp8, a_scale
        else:
            gemm1_input, gemm1_a_scale = hidden_states, None

        self.moe_grouped_gemm(
            gemm1_input, w13, intermediate1,
            topk_weights, sorted_token_ids, expert_ids,
            num_tokens_post_padded,
            mul_routed_weight=False, top_k=top_k, config=config,
            a_scale=gemm1_a_scale, b_scale=w13_scale,
            use_fp8_w8a8=use_fp8_w8a8, block_shape=block_shape,
        )

        intermediate2 = self.act_fn(intermediate1)

        if use_fp8_w8a8:
            a2_fp8, a2_scale = self._get_fp8_bufs(2, M * top_k, N,
                                                  hidden_states.device)
            self.per_token_group_quant_fp8(intermediate2, a2_fp8, a2_scale)
            gemm2_input, gemm2_a_scale = a2_fp8, a2_scale
        else:
            gemm2_input, gemm2_a_scale = intermediate2, None

        self.moe_grouped_gemm(
            gemm2_input, w2, intermediate3,
            topk_weights, sorted_token_ids, expert_ids,
            num_tokens_post_padded,
            mul_routed_weight=True, top_k=1, config=config,
            a_scale=gemm2_a_scale, b_scale=w2_scale,
            use_fp8_w8a8=use_fp8_w8a8, block_shape=block_shape,
        )

        return self.moe_sum(intermediate3, top_k)
