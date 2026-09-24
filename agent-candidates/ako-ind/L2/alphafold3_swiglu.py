"""SwiGLU activation and AdaLN for AlphaFold3 (L2 composites).

SwiGLU: SiLU(linear_a(x)) * linear_b(x)
AdaLN: Adaptive Layer Normalization

Reference: openfold3/core/model/primitives/activations.py SwiGLU
           openfold3/core/model/primitives/normalization.py AdaLN

Every captured case is tiny and skinny-M (16..1536 rows against c_in/c_out of
at most 768/1536), so neither class is compute-bound.  What the benchmark
measures is the *number of kernels* in the stream -- each one costs ~2us there,
independent of how little work it does -- plus the (cold) bytes each one moves.
The baseline spends 4 launches per SwiGLU (two GEMMs, silu, mul) and ~12 per
AdaLN (two fp32-promoted LayerNorms at three launches each, two GEMMs, sigmoid,
add, mul).  This drops that to 1 kernel for AdaLN and 1-2 for SwiGLU.

AdaLN is a single Triton kernel.  layer_norm_s, the gate GEMM, the shift GEMM,
layer_norm_a, the sigmoid, the add and the multiply all happen in one program:
the two Linears that read the same s_norm become two ``tl.dot``s off one shared
normalized tile, so each weight is read exactly once and no intermediate ever
round-trips through HBM.

SwiGLU picks between two shapes of the same idea, per (M, c_in, c_out):

* Small weights -- one fused Triton kernel, same trick: two ``tl.dot``s off one
  shared x tile plus the silu/mul epilogue.
* Large weights -- one cuBLAS GEMM against a stacked [2*c_out, c_in] weight,
  then a Triton epilogue that consumes both halves.  Here the *weight read*
  dominates and a skinny-M Triton tiling can only field 24-96 CTAs, which
  streams 4.7 MB in ~8us where cuBLAS (split-K, many more CTAs) needs ~4.1us;
  the extra launch is cheaper than that gap.  See ITERATIONS.md.

Parameters are exactly the baseline's (``linear_a.weight``,
``linear_b.weight``, ``layer_norm_s.weight``, ``linear_g.weight``,
``linear_g.bias``, ``linear_s.weight``).  The fused kernels take the two
parallel weights as two pointers, and the one derived tensor that does exist
(SwiGLU's stacked weight) is a lazily built plain attribute, not a Parameter --
the benchmark rebuilds the module and calls ``load_state_dict(..., strict=False)``
with the baseline's keys, so a new parameter name would silently keep its random
init.

Numerics follow the baseline rather than merely clearing tolerance: both
LayerNorm reductions run in fp32 (the baseline promotes with
``F.layer_norm(x.float(), ...)``), ``layer_norm_a`` has neither scale nor offset
while ``layer_norm_s`` is weight-only, and every value is rounded back to bf16
at exactly the points where the baseline materializes a bf16 tensor.  SwiGLU
comes out bit-exact against the reference on the captured shapes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

from ..L1.sigmoid import Sigmoid
from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear
from ..L1.silu import SiLU

_EPS = tl.constexpr(1e-5)


# ---------------------------------------------------------------------------
# SwiGLU: one kernel = two GEMMs over a shared x tile + silu/mul epilogue.
# ---------------------------------------------------------------------------
@triton.jit
def _swiglu_kernel(X, WA, WB, O, M,
                   K: tl.constexpr, N: tl.constexpr,
                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    rowmask = (rm < M)[:, None]

    xp = X + (rm[:, None] * K + rk[None, :])
    ap = WA + (rn[:, None] * K + rk[None, :])
    bp = WB + (rn[:, None] * K + rk[None, :])

    acc_a = tl.zeros([BM, BN], dtype=tl.float32)
    acc_b = tl.zeros([BM, BN], dtype=tl.float32)
    for _ in tl.static_range(K // BK):
        xt = tl.load(xp, mask=rowmask, other=0.0)
        acc_a = tl.dot(xt, tl.trans(tl.load(ap)), acc_a)
        acc_b = tl.dot(xt, tl.trans(tl.load(bp)), acc_b)
        xp += BK
        ap += BK
        bp += BK

    # The baseline rounds both GEMM results (and the silu output) to bf16;
    # round at the same points so the fused path tracks it closely.
    va = acc_a.to(tl.bfloat16).to(tl.float32)
    vb = acc_b.to(tl.bfloat16).to(tl.float32)
    out = (va * tl.sigmoid(va)).to(tl.bfloat16).to(tl.float32) * vb
    tl.store(O + (rm[:, None] * N + rn[None, :]), out.to(tl.bfloat16),
             mask=rowmask)


# ---------------------------------------------------------------------------
# SwiGLU epilogue for the stacked-GEMM path: reads the [M, 2*c_out] output of a
# single cuBLAS GEMM against the stacked [2*c_out, c_in] weight and applies
# silu(first half) * second half.  Used when the two weights are big enough
# that the *weight read* dominates: cuBLAS streams 4.7 MB in ~4.1us where the
# hand-written Triton GEMM needs ~8us at the 24-96 CTAs a skinny-M tiling can
# field, so one cuBLAS GEMM + this epilogue (2 kernels) beats one fused Triton
# kernel.  Below that size the fused kernel wins and this is unused.
# ---------------------------------------------------------------------------
@triton.jit
def _swiglu_epilogue(Y, O, M, N: tl.constexpr,
                     BM: tl.constexpr, BN: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rowmask = (rm < M)[:, None]
    yp = Y + (rm[:, None] * (2 * N) + rn[None, :])
    va = tl.load(yp, mask=rowmask, other=0.0).to(tl.float32)
    vb = tl.load(yp + N, mask=rowmask, other=0.0).to(tl.float32)
    out = (va * tl.sigmoid(va)).to(tl.bfloat16).to(tl.float32) * vb
    tl.store(O + (rm[:, None] * N + rn[None, :]), out.to(tl.bfloat16),
             mask=rowmask)


# ---------------------------------------------------------------------------
# AdaLN: one kernel = layer_norm(s) -> gate GEMM + shift GEMM -> layer_norm(a)
# -> sigmoid gate -> add -> mul.
#
# ONE_K / ONE_A specialize the (dominant) case where a whole row of s / a is
# one tile: the row is then loaded once and mean, variance and the GEMM operand
# all come off the same registers instead of three passes over it.
# ---------------------------------------------------------------------------
@triton.jit
def _adaln_kernel(A, S, LNW, WG, BG, WS, O, M,
                  CA: tl.constexpr, CS: tl.constexpr,
                  BM: tl.constexpr, BN: tl.constexpr,
                  BK: tl.constexpr, BA: tl.constexpr,
                  ONE_K: tl.constexpr, ONE_A: tl.constexpr,
                  FULL_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    rowmask = (rm < M)[:, None]

    sbase = S + (rm[:, None] * CS + rk[None, :])
    gp = WG + (rn[:, None] * CS + rk[None, :])
    sp = WS + (rn[:, None] * CS + rk[None, :])
    acc_g = tl.zeros([BM, BN], dtype=tl.float32)
    acc_s = tl.zeros([BM, BN], dtype=tl.float32)

    if ONE_K:
        sv = tl.load(sbase, mask=rowmask, other=0.0).to(tl.float32)
        d = sv - (tl.sum(sv, 1) / CS)[:, None]
        srstd = 1.0 / tl.sqrt(tl.sum(d * d, 1) / CS + _EPS)
        sn = (d * srstd[:, None] * tl.load(LNW + rk).to(tl.float32)[None, :]).to(tl.bfloat16)
        acc_g = tl.dot(sn, tl.trans(tl.load(gp)), acc_g)
        acc_s = tl.dot(sn, tl.trans(tl.load(sp)), acc_s)
    else:
        # fp32 reduction, two passes so the variance never leans on
        # E[x^2] - mean^2 cancellation.
        ssum = tl.zeros([BM], dtype=tl.float32)
        for i in tl.static_range(CS // BK):
            ssum += tl.sum(tl.load(sbase + i * BK, mask=rowmask, other=0.0).to(tl.float32), 1)
        smean = ssum / CS
        svar = tl.zeros([BM], dtype=tl.float32)
        for i in tl.static_range(CS // BK):
            d = tl.load(sbase + i * BK, mask=rowmask, other=0.0).to(tl.float32) - smean[:, None]
            svar += tl.sum(d * d, 1)
        srstd = 1.0 / tl.sqrt(svar / CS + _EPS)
        for i in tl.static_range(CS // BK):
            d = tl.load(sbase + i * BK, mask=rowmask, other=0.0).to(tl.float32) - smean[:, None]
            w = tl.load(LNW + i * BK + rk).to(tl.float32)
            sn = (d * srstd[:, None] * w[None, :]).to(tl.bfloat16)
            acc_g = tl.dot(sn, tl.trans(tl.load(gp + i * BK)), acc_g)
            acc_s = tl.dot(sn, tl.trans(tl.load(sp + i * BK)), acc_s)

    # --- layer_norm_a: no scale, no offset, fp32 reduction. ------------------
    if ONE_A:
        av = tl.load(A + (rm[:, None] * CA + tl.arange(0, BA)[None, :]),
                     mask=rowmask, other=0.0).to(tl.float32)
        amean = tl.sum(av, 1) / CA
        ad = av - amean[:, None]
        arstd = 1.0 / tl.sqrt(tl.sum(ad * ad, 1) / CA + _EPS)
        if FULL_N:
            an = (ad * arstd[:, None]).to(tl.bfloat16).to(tl.float32)
        else:
            av2 = tl.load(A + (rm[:, None] * CA + rn[None, :]),
                          mask=rowmask, other=0.0).to(tl.float32)
            an = ((av2 - amean[:, None]) * arstd[:, None]).to(tl.bfloat16).to(tl.float32)
    else:
        ra = tl.arange(0, BA)
        abase = A + (rm[:, None] * CA + ra[None, :])
        asum = tl.zeros([BM], dtype=tl.float32)
        for i in tl.static_range(CA // BA):
            asum += tl.sum(tl.load(abase + i * BA, mask=rowmask, other=0.0).to(tl.float32), 1)
        amean = asum / CA
        avar = tl.zeros([BM], dtype=tl.float32)
        for i in tl.static_range(CA // BA):
            d = tl.load(abase + i * BA, mask=rowmask, other=0.0).to(tl.float32) - amean[:, None]
            avar += tl.sum(d * d, 1)
        arstd = 1.0 / tl.sqrt(avar / CA + _EPS)
        av = tl.load(A + (rm[:, None] * CA + rn[None, :]), mask=rowmask, other=0.0).to(tl.float32)
        an = ((av - amean[:, None]) * arstd[:, None]).to(tl.bfloat16).to(tl.float32)

    # --- epilogue: sigmoid(gate) * (a_norm + shift) -------------------------
    # sigmoid's input *and* output are bf16 tensors in the baseline
    # (bf16 linear output -> torch.sigmoid -> bf16); round at both.
    g = tl.sigmoid((acc_g + tl.load(BG + rn).to(tl.float32)).to(tl.bfloat16).to(tl.float32))
    g = g.to(tl.bfloat16).to(tl.float32)
    shift = acc_s.to(tl.bfloat16).to(tl.float32)
    out = g * (an + shift).to(tl.bfloat16).to(tl.float32)
    tl.store(O + (rm[:, None] * CA + rn[None, :]), out.to(tl.bfloat16), mask=rowmask)


# ---------------------------------------------------------------------------
# Launch-configuration tables.  Entries are the best of a sweep over
# BM/BN/BK/BA/num_warps/num_stages at the captured shapes, timed through the
# benchmark's own timing loop -- CUDA-graph replay and ``do_bench`` both lie
# here (the former is L2-warm, the latter CPU-bound; see ITERATIONS.md).
# Anything unseen falls back to the shape of those winners.
# ---------------------------------------------------------------------------
def _divisor(n: int, cap: int) -> int:
    b = cap
    while b >= 16:
        if n % b == 0:
            return b
        b //= 2
    return 0


# Fused-Triton-GEMM path, (M, c_in, c_out) -> (BM, BN, BK, num_warps,
# num_stages).  BM=16 everywhere: these are weight-read bound, so many small
# CTAs beat few fat ones even at M=368.
_SWIGLU_CFG = {
    (368, 128, 256): (16, 16, 128, 1, 1),
    (256, 128, 256): (16, 16, 128, 1, 1),
    (128, 64, 256): (16, 32, 64, 4, 1),
}

# Stacked-cuBLAS-GEMM path, (M, c_in, c_out) -> (BM, BN, num_warps, num_stages)
# for the epilogue.  Keyed the same way; membership here selects the path.
_SWIGLU_EPI_CFG = {
    (16, 384, 1536): (16, 64, 4, 1),
    (16, 768, 1536): (16, 128, 8, 2),
}

# Above this many weight elements per Linear, one cuBLAS GEMM over the stacked
# weight plus the epilogue above beats the fused Triton GEMM (measured on the
# captured shapes: 589824 and 1179648 elements go stacked, 32768 and 16384 stay
# fused).
_STACK_MIN_WEIGHT = 1 << 17

# (M, c_a, c_s) -> (BM, BN, BK, BA, num_warps, num_stages).  Every config in a
# 128-point sweep of the 128/128 shapes lands within 0.05us of 17.38us, so these
# pick the structurally cleanest point of that plateau: BN = BA = c_a, i.e. one
# n-block per row-block, which lets one load of the a row serve both
# layer_norm_a's statistics and the epilogue.
_ADALN_CFG = {
    (16, 768, 384): (16, 32, 128, 128, 8, 1),
    (1536, 128, 128): (16, 128, 128, 128, 8, 2),
    (384, 128, 128): (16, 128, 128, 128, 8, 2),
    (368, 128, 128): (16, 128, 128, 128, 8, 2),
}


class SwiGLU(nn.Module):
    """SwiGLU activation: SiLU(Wa x) * Wb x.

    Args:
        c_in: Number of input channels
        c_out: Number of output channels
    """

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.silu = SiLU()
        self.linear_a = Linear(c_in, c_out, bias=False)
        self.linear_b = Linear(c_in, c_out, bias=False)
        self.c_in = c_in
        self.c_out = c_out
        # shape -> launch plan; inference-only, so a plain dict is enough.
        self._plans: dict = {}
        # Lazily built [2*c_out, c_in] stack of linear_a/linear_b, plus the
        # identity of the weights it was built from.  NOT a Parameter and NOT a
        # buffer: the benchmark rebuilds this module and calls
        # load_state_dict(..., strict=False) with the *baseline's* keys, so a new
        # parameter name would silently keep its random init.  Built on the first
        # forward, which runs after weight loading, and re-derived whenever
        # either source weight is replaced, moved or written in place.
        self._stack = None
        self._stack_key = None

    def _ref(self, x):
        return self.silu(self.linear_a(x)) * self.linear_b(x)

    @staticmethod
    def _wkey(wa, wb):
        return (id(wa), wa.data_ptr(), wa._version,
                id(wb), wb.data_ptr(), wb._version)

    def _stacked(self, wa, wb):
        key = self._wkey(wa, wb)
        if self._stack is None or self._stack_key != key:
            self._stack = torch.cat((wa.detach(), wb.detach()), 0).contiguous()
            self._stack_key = key
        return self._stack

    def _plan(self, shape):
        c_in, c_out = self.c_in, self.c_out
        wa, wb = self.linear_a.weight, self.linear_b.weight
        if (len(shape) < 1 or shape[-1] != c_in
                or tuple(wa.shape) != (c_out, c_in)
                or tuple(wb.shape) != (c_out, c_in)
                or wa.dtype is not torch.bfloat16 or wb.dtype is not torch.bfloat16
                or not wa.is_contiguous() or not wb.is_contiguous()):
            return None
        m = 1
        for d in shape[:-1]:
            m *= d
        if m == 0:
            return None
        oshape = tuple(shape[:-1]) + (c_out,)
        key = (m, c_in, c_out)
        stacked = (key in _SWIGLU_EPI_CFG
                   or (key not in _SWIGLU_CFG and c_in * c_out >= _STACK_MIN_WEIGHT))
        if stacked:
            cfg = _SWIGLU_EPI_CFG.get(key)
            if cfg is None:
                bn = _divisor(c_out, 64)
                if not bn:
                    return None
                cfg = (16, bn, 4, 1)
            bm, bn, nw, ns = cfg
            grid = (triton.cdiv(m, bm), c_out // bn)
            return (True, grid, m, c_out, bm, bn, nw, ns, oshape)
        cfg = _SWIGLU_CFG.get(key)
        if cfg is None:
            bk = _divisor(c_in, 128)
            bn = _divisor(c_out, 32)
            if not bk or not bn:
                return None
            cfg = (16, bn, bk, 4, 1)
        bm, bn, bk, nw, ns = cfg
        grid = (triton.cdiv(m, bm), c_out // bn)
        return (False, grid, m, c_in, c_out, bm, bn, bk, nw, ns, oshape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (x.dtype is not torch.bfloat16 or not x.is_cuda
                or not x.is_contiguous()):
            return self._ref(x)
        plan = self._plans.get(x.shape)
        if plan is None:
            plan = self._plan(x.shape)
            self._plans[x.shape] = plan
            if plan is None:
                return self._ref(x)
        if plan[0]:
            _, grid, m, n, bm, bn, nw, ns, oshape = plan
            y = torch.mm(x.reshape(m, -1),
                         self._stacked(self.linear_a.weight,
                                       self.linear_b.weight).t())
            out = torch.empty(oshape, dtype=torch.bfloat16, device=x.device)
            _swiglu_epilogue[grid](y, out, m, N=n, BM=bm, BN=bn,
                                   num_warps=nw, num_stages=ns)
            return out
        _, grid, m, k, n, bm, bn, bk, nw, ns, oshape = plan
        out = torch.empty(oshape, dtype=torch.bfloat16, device=x.device)
        _swiglu_kernel[grid](x, self.linear_a.weight, self.linear_b.weight, out, m,
                             K=k, N=n, BM=bm, BN=bn, BK=bk,
                             num_warps=nw, num_stages=ns)
        return out


class AdaLN(nn.Module):
    """Adaptive Layer Normalization matching the reference AdaLN.

    Submodule structure matches checkpoint keys:
    - layer_norm_s: LayerNorm(c_s), weight-only
    - linear_g: Linear(c_s, c_a, bias=True) — gating
    - linear_s: Linear(c_s, c_a, bias=False) — additive conditioning

    Reference: openfold3/core/model/primitives/normalization.py AdaLN

    Args:
        c_a: Activation channel dimension
        c_s: Conditioning channel dimension
    """

    def __init__(self, c_a: int, c_s: int):
        super().__init__()
        self.c_a = c_a
        self.c_s = c_s

        self.layer_norm_a = LayerNorm(c_a, create_scale=False, create_offset=False)
        self.layer_norm_s = LayerNorm(c_s, create_offset=False)
        self.sigmoid = Sigmoid()
        self.linear_g = Linear(c_s, c_a, bias=True)
        self.linear_s = Linear(c_s, c_a, bias=False)
        self._plans: dict = {}

    def _ref(self, a, s):
        s_norm = self.layer_norm_s(s)
        g = self.sigmoid(self.linear_g(s_norm))
        a_norm = self.layer_norm_a(a)
        return g * (a_norm + self.linear_s(s_norm))

    def _plan(self, ashape, sshape):
        c_a, c_s = self.c_a, self.c_s
        wg, ws = self.linear_g.weight, self.linear_s.weight
        lnw, bg = self.layer_norm_s.weight, self.linear_g.bias
        if (len(ashape) < 1 or len(sshape) < 1
                or ashape[-1] != c_a or sshape[-1] != c_s
                or lnw is None or bg is None
                or tuple(wg.shape) != (c_a, c_s) or tuple(ws.shape) != (c_a, c_s)
                or wg.dtype is not torch.bfloat16 or ws.dtype is not torch.bfloat16
                or lnw.dtype is not torch.bfloat16 or bg.dtype is not torch.bfloat16
                or not (wg.is_contiguous() and ws.is_contiguous()
                        and lnw.is_contiguous() and bg.is_contiguous())):
            return None
        m = 1
        for d in ashape[:-1]:
            m *= d
        ms = 1
        for d in sshape[:-1]:
            ms *= d
        if m != ms or m == 0:
            return None
        # ``a`` may carry a leading unit dim that ``s`` lacks; that broadcast is
        # unit-only, so the row counts above already agree and the result is
        # just the wider of the two shapes.
        try:
            oshape = torch.broadcast_shapes(ashape, sshape[:-1] + (c_a,))
        except RuntimeError:
            return None
        on = 1
        for d in oshape:
            on *= d
        if on != m * c_a:
            return None
        cfg = _ADALN_CFG.get((m, c_a, c_s))
        if cfg is None:
            bk = _divisor(c_s, 128)
            ba = _divisor(c_a, 256)
            # prefer one n-block per row-block so ONE_A/FULL_N kick in
            bn = ba if ba == c_a else _divisor(c_a, 64)
            if not bk or not ba or not bn:
                return None
            cfg = (16, bn, bk, ba, 4, 3)
        bm, bn, bk, ba, nw, ns = cfg
        grid = (triton.cdiv(m, bm), c_a // bn)
        return (grid, m, c_a, c_s, bm, bn, bk, ba, nw, ns, tuple(oshape))

    def forward(self, a: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        if (a.dtype is not torch.bfloat16 or s.dtype is not torch.bfloat16
                or not a.is_cuda or not s.is_cuda
                or not a.is_contiguous() or not s.is_contiguous()):
            return self._ref(a, s)
        key = (a.shape, s.shape)
        plan = self._plans.get(key)
        if plan is None:
            plan = self._plan(a.shape, s.shape)
            self._plans[key] = plan
            if plan is None:
                return self._ref(a, s)
        grid, m, c_a, c_s, bm, bn, bk, ba, nw, ns, oshape = plan
        out = torch.empty(oshape, dtype=torch.bfloat16, device=a.device)
        _adaln_kernel[grid](a, s, self.layer_norm_s.weight,
                            self.linear_g.weight, self.linear_g.bias,
                            self.linear_s.weight, out, m,
                            CA=c_a, CS=c_s, BM=bm, BN=bn, BK=bk, BA=ba,
                            ONE_K=(c_s == bk), ONE_A=(c_a == ba),
                            FULL_N=(c_a == bn and c_a == ba),
                            num_warps=nw, num_stages=ns)
        return out
