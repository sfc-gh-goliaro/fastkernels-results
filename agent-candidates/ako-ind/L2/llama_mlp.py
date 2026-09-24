"""Llama SwiGLU MLP block: gate_up_proj -> SiluAndMul -> down_proj.

Also used by DeepSeek V3's "shared expert": pass ``reduce_results=False``
and override ``intermediate_size`` with ``moe_intermediate_size *
n_shared_experts``.  See ``L2/deepseek_moe.py``.

Two changes over the reference three-launch pipeline, both aimed at the decode
regime where the block is pure HBM streaming (~352 MB of bf16 weights for
h=4096/i=14336 against a few hundred KB of activations):

1. ``M == 1`` -- a fused pair of hand-written Triton GEMV kernels replaces the
   whole pipeline.  cuBLAS is far off the read roofline at a single row, and
   without tensor cores in the way the tiles can shrink to 4-8 weight rows with
   256-1024-element K chunks, which is what actually saturates HBM here: long
   contiguous runs per weight row plus enough blocks to keep ~1000 loads in
   flight.  SiLU*mul is folded into the gate_up epilogue so the [1, 2i]
   intermediate never reaches memory, and accumulation is fp32.

2. ``M > 1`` -- keep cuBLAS for both GEMMs (it is genuinely good at these
   shapes; a Triton replacement measured slower at every M > 1 tried) but
   replace ``SiluAndMul`` with a flat-grid Triton kernel.  The vendored CUDA
   activation launches exactly ``num_tokens`` blocks, so at M = 60 it runs 60
   blocks on 148 SMs and costs ~9 us for ~5 MB of traffic.  A flat grid over
   M*i elements is bit-identical (same rounding order, verified exact) and
   drops that to ~7 us -- and, being a Triton launch, it can carry
   ``launch_pdl``, which is where most of the mid-M win actually comes from.

Every weight load in the GEMV path is tagged ``evict_first``: the weights are
streamed once per call and never reused, so keeping them out of L2 leaves the
cache to the activations that *are* re-read by every block.  Accumulation is
fp32 throughout.

fp8 weights, tensor parallelism, bias and non-half dtypes all fall through to
the reference path unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

from .parallel_linear import MergedColumnParallelLinear, RowParallelLinear
from ..L1.silu_and_mul import SiluAndMul

# Programmatic Dependent Launch: lets the down kernel be resident and do its
# address setup while the gate_up kernel drains, instead of paying a full
# launch gap between two kernels that each run for only ~20 us.  Worth ~2 us of
# a ~36 us call.  Optional -- older Triton builds lack the intrinsics.
try:
    from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait
    _HAS_PDL = True
except ImportError:  # pragma: no cover
    _HAS_PDL = False


def _no_pdl(exc: TypeError) -> bool:
    """A Triton without the ``launch_pdl`` launch kwarg -- disable and retry."""
    global _HAS_PDL
    if _HAS_PDL and 'launch_pdl' in str(exc):
        _HAS_PDL = False
        return True
    return False


@triton.jit
def _silu_mul(g_f32, u_f32, odt: tl.constexpr):
    """SiLU*mul with the *reference's* rounding, not a more accurate one.

    The reference pipeline stores the gate_up GEMM result to bf16, so
    ``silu_and_mul`` sees bf16 inputs and rounds its own silu result back to
    bf16 before the multiply.  Staying in fp32 throughout would be more
    accurate but drifts ~0.4% from the reference -- enough to push a few
    percent of elements outside the bf16 rtol=1e-2 band once down_proj sums
    10^4 of them.  So round exactly where the reference rounds.
    """
    g = g_f32.to(odt).to(tl.float32)
    u = u_f32.to(odt).to(tl.float32)
    s = (g / (1.0 + tl.exp(-g))).to(odt).to(tl.float32)
    return (s * u).to(odt)


# ---------------------------------------------------------------------------
# flat-grid SiLU*mul over y[M, 2i] -> act[M, i]
# ---------------------------------------------------------------------------
@triton.jit
def _silu_mul_kernel(Y, A, MI, I, N2, BLOCK: tl.constexpr, PDL: tl.constexpr):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = o < MI
    r = o // I
    base = Y + r * N2 + (o - r * I)
    if PDL:
        gdc_wait()
    g = tl.load(base, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(base + I, mask=mask, other=0.0).to(tl.float32)
    tl.store(A + o, _silu_mul(g, u, A.dtype.element_ty), mask=mask)


# ---------------------------------------------------------------------------
# M == 1: gate_up + SiLU*mul, then down (optionally split along K)
# ---------------------------------------------------------------------------
@triton.jit
def _gu_gemv(X, W, A, H, I, BN: tl.constexpr, BK: tl.constexpr,
             PDL: tl.constexpr):
    pid = tl.program_id(0)
    offs_n = pid * BN + tl.arange(0, BN)
    accg = tl.zeros([BN], dtype=tl.float32)
    accu = tl.zeros([BN], dtype=tl.float32)
    wg = W + offs_n[:, None] * H
    wu = W + (offs_n + I)[:, None] * H
    for k in range(0, H, BK):
        ok = k + tl.arange(0, BK)
        xv = tl.load(X + ok, eviction_policy='evict_last').to(tl.float32)
        bg = tl.load(wg + ok[None, :], eviction_policy='evict_first').to(tl.float32)
        bu = tl.load(wu + ok[None, :], eviction_policy='evict_first').to(tl.float32)
        accg += tl.sum(bg * xv[None, :], axis=1)
        accu += tl.sum(bu * xv[None, :], axis=1)
    tl.store(A + offs_n, _silu_mul(accg, accu, A.dtype.element_ty))
    if PDL:
        gdc_launch_dependents()


@triton.jit
def _down_gemv(A, W, O, P, I, BJ: tl.constexpr, BK: tl.constexpr,
               SPLIT: tl.constexpr, KPER: tl.constexpr, PDL: tl.constexpr):
    """The [BJ, BK] fp32 accumulator is deliberately *not* reduced per K step.

    ``tl.sum(..., axis=1)`` inside the loop costs a cross-lane (and, once BK
    exceeds a warp's worth of elements, cross-warp via SMEM) reduction on every
    iteration.  Keeping the partial products in a register tile and reducing
    once at the end runs the loop at the pure-read rate -- measured 31.7 -> 27.6
    us on the (4096, 14336) down weight, i.e. exactly the streaming-read floor.
    """
    pj = tl.program_id(0)
    ps = tl.program_id(1)
    offs_j = pj * BJ + tl.arange(0, BJ)
    acc = tl.zeros([BJ, BK], dtype=tl.float32)
    wp = W + offs_j[:, None] * I
    if PDL:
        # As late as possible: the weight-pointer arithmetic above overlaps the
        # producer's tail; this must precede the first load of ``A``.
        gdc_wait()
    for k in range(ps * KPER, (ps + 1) * KPER, BK):
        ok = k + tl.arange(0, BK)
        av = tl.load(A + ok, eviction_policy='evict_last').to(tl.float32)
        acc += tl.load(wp + ok[None, :],
                       eviction_policy='evict_first').to(tl.float32) * av[None, :]
    r = tl.sum(acc, axis=1)
    if SPLIT == 1:
        tl.store(O + offs_j, r.to(O.dtype.element_ty))
    else:
        tl.store(P + ps * tl.num_programs(0) * BJ + offs_j, r)


@triton.jit
def _reduce(P, O, N, SPLIT: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    acc = tl.load(P + offs, mask=mask, other=0.0)
    for s in tl.static_range(1, SPLIT):
        acc += tl.load(P + s * N + offs, mask=mask, other=0.0)
    tl.store(O + offs, acc.to(O.dtype.element_ty), mask=mask)


# ---------------------------------------------------------------------------
# launch-plan selection (computed once per (h, i), then cached)
# ---------------------------------------------------------------------------
def _first_div(n, cands):
    for c in cands:
        if n % c == 0:
            return c
    return None


# Winning tiles from a full BN/BK/warps/stages/split sweep on B200 (sm_100)
# under the harness' L2-flushed timing loop.  Shapes outside the table use the
# generic divisor rules below, which pick the same shape of config.
_TUNED_GEMV = {
    # (h, i): (gu BN, gu BK, gu warps, gu stages,
    #          down BJ, down BK, down SPLIT, down warps, down stages)
    (4096, 14336): (8, 1024, 4, 4, 2, 1024, 1, 2, 4),
    (2304, 9216): (4, 256, 2, 3, 4, 1024, 1, 8, 3),
}


def _gemv_plan(h, i):
    """Return an (M == 1) launch plan, or None if the shape does not fit."""
    tuned = _TUNED_GEMV.get((h, i))
    if tuned is not None:
        gbn, gbk, gnw, gns, dbj, dbk, split, dnw, dns = tuned
    else:
        gbn = _first_div(i, (8, 4, 16, 32))
        gbk = _first_div(h, (512, 256, 128, 64))
        dbj = _first_div(h, (2, 4, 8, 16))
        dbk = _first_div(i, (1024, 512, 2048, 256, 128, 64))
        gnw, gns, dnw, dns = 4, 4, 4, 3
        if None in (gbn, gbk, dbj, dbk):
            return None
        while dbj * dbk > 16384:      # keep the register tile in budget
            dbk //= 2
        split = 1
        # Split K until the down grid can fill the machine several times over.
        while (h // dbj) * split < 512 and i % (split * 2 * dbk) == 0:
            split *= 2
    # The kernels do not mask the N/K axes, so divisibility is required.
    if i % gbn or h % gbk or h % dbj or i % (split * dbk):
        return None
    return (gbn, gbk, gnw, gns, dbj, dbk, split, i // split, dnw, dns)


_GEMV_PLANS: dict[tuple[int, int], tuple | None] = {}


def _mlp_gemv(x, w_gu, w_down, h, i):
    key = (h, i)
    if key in _GEMV_PLANS:
        plan = _GEMV_PLANS[key]
    else:
        plan = _gemv_plan(h, i)
        _GEMV_PLANS[key] = plan
    if plan is None:
        return None
    gbn, gbk, gnw, gns, dbj, dbk, split, kper, dnw, dns = plan

    act = torch.empty((1, i), dtype=x.dtype, device=x.device)
    out = torch.empty((1, h), dtype=x.dtype, device=x.device)
    njob = h // dbj
    part = (torch.empty(split * h, dtype=torch.float32, device=x.device)
            if split > 1 else act)
    pdl = _HAS_PDL
    try:
        _gu_gemv[(i // gbn,)](x, w_gu, act, h, i, gbn, gbk, pdl,
                              num_warps=gnw, num_stages=gns, launch_pdl=pdl)
        _down_gemv[(njob, split)](act, w_down, out, part, i, dbj, dbk, split,
                                  kper, pdl, num_warps=dnw, num_stages=dns,
                                  launch_pdl=pdl)
    except TypeError as exc:
        if not _no_pdl(exc):
            raise
        _gu_gemv[(i // gbn,)](x, w_gu, act, h, i, gbn, gbk, False,
                              num_warps=gnw, num_stages=gns)
        _down_gemv[(njob, split)](act, w_down, out, part, i, dbj, dbk, split,
                                  kper, False, num_warps=dnw, num_stages=dns)
    if split > 1:
        _reduce[(triton.cdiv(h, 1024),)](part, out, h, split, 1024, num_warps=4)
    return out


def _silu_cfg(n):
    """Pick (BLOCK, warps) so the grid stays several waves deep.

    The point of this kernel is block count -- at n ~ 10^6 a 512-element block
    gives ~2000 blocks where the vendored one-block-per-token kernel gives 60.
    """
    if n <= 1 << 21:
        return 512, 4
    if n <= 1 << 24:
        return 1024, 4
    return 4096, 8


def _silu_mul_triton(y, i):
    m2 = y.shape[0]
    act = torch.empty((m2, i), dtype=y.dtype, device=y.device)
    n = m2 * i
    block, nw = _silu_cfg(n)
    grid = (triton.cdiv(n, block),)
    try:
        _silu_mul_kernel[grid](y, act, n, i, y.stride(0), block, _HAS_PDL,
                               num_warps=nw, launch_pdl=_HAS_PDL)
    except TypeError as exc:
        if not _no_pdl(exc):
            raise
        _silu_mul_kernel[grid](y, act, n, i, y.stride(0), block, False,
                               num_warps=nw)
    return act


class LlamaMLP(nn.Module):
    def __init__(self, config, quant_config: dict | None = None,
                 hidden_size: int | None = None,
                 intermediate_size: int | None = None,
                 reduce_results: bool = True):
        super().__init__()
        h = hidden_size if hidden_size is not None else config.hidden_size
        i = intermediate_size if intermediate_size is not None else config.intermediate_size
        self.gate_up_proj = MergedColumnParallelLinear(
            h, [i] * 2,
            quant_config=quant_config,
        )
        self.down_proj = RowParallelLinear(
            i, h,
            quant_config=quant_config,
            reduce_results=reduce_results,
        )
        self.act_fn = SiluAndMul()
        self._h = h
        # The fast paths need plain (non-fp8) weights, no bias and no all-reduce.
        self._fast_ok = (
            quant_config is None
            and self.down_proj.tp_size == 1
            and self.gate_up_proj.bias is None
            and self.down_proj.bias is None
        )

    def forward(self, x):
        if self._fast_ok and x.is_cuda and x.stride(-1) == 1:
            w_gu = self.gate_up_proj.weight
            w_down = self.down_proj.weight
            if (x.shape[-1] == self._h and x.dtype in (torch.bfloat16, torch.float16)
                    and x.dtype == w_gu.dtype == w_down.dtype
                    and w_gu.is_contiguous() and w_down.is_contiguous()):
                i = w_down.shape[1]
                m = x.numel() // self._h
                if m == 1:
                    out = _mlp_gemv(x.reshape(1, self._h), w_gu, w_down, self._h, i)
                    if out is not None:
                        return out.reshape(*x.shape[:-1], self._h)
                else:
                    y = self.gate_up_proj(x)
                    if y.is_contiguous():
                        act = _silu_mul_triton(y.reshape(m, 2 * i), i)
                        return self.down_proj(act.reshape(*x.shape[:-1], i))
                    return self.down_proj(self.act_fn(y))
        x = self.gate_up_proj(x)
        x = self.act_fn(x)
        return self.down_proj(x)
