"""Linear (matrix multiply) kernels.

Matmul: y = x @ w.T (+ bias)   -- F.linear semantics
BMM:    c = a @ b              -- torch.matmul semantics
Linear: parametric wrapper holding weight/bias.

`_mm` below is a hand-written strided tiled MMA GEMM (batched, arbitrary
operand strides, optional fused bias, optional split-K reduction through
`_red`).  Tile shape / warp count / pipeline depth / CTA order / split factor
are compile-time constants chosen per captured (L, M, K, N) in `_MM_CFG`.

`_MM_CFG` is also the dispatch gate: a shape reaches `_mm` only when it has a
tuned entry, i.e. only when this kernel was measured *faster than the reference
on this GPU* and verified to reproduce the reference's numerics.  On B200 that
is true for the tf32 batched shapes, where cuBLAS falls back to a legacy sm_80
CUTLASS kernel; for bf16 it dispatches Blackwell-native `nvjet_sm100_*`
kernels with 2-CTA clusters and tile shapes (48x64, 40x64, 64x8, ...) that
Triton cannot express, and those are left to the reference.  See ITERATIONS.md
for the full per-shape measurement table.

Deferring is not only a speed choice: for fp32 the reference silently switches
between exact-fp32 and TF32 kernels depending on shape, and a candidate has to
match whichever it picked (the benchmark's fp32 tolerance is atol 1e-5 /
rtol 1e-3 on 99% of elements), so guessing is a correctness risk.
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------
@triton.jit
def _mm(A, B, BIAS, C, M, N, K,
        sab, sam, sak, sbb, sbk, sbn, scb, scm,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
        GM: tl.constexpr, GN: tl.constexpr, NFAST: tl.constexpr,
        SK: tl.constexpr, KC: tl.constexpr, PSTRIDE: tl.constexpr,
        HAS_BIAS: tl.constexpr, EVEN_M: tl.constexpr, EVEN_N: tl.constexpr,
        EVEN_K: tl.constexpr, TF32R: tl.constexpr, PREC: tl.constexpr):
    """c[l, M, N] = a[l, M, K] @ b[l, K, N] (+ bias[N]) for arbitrary strides.

    grid = (tiles, L[, SK]).  ``NFAST`` picks which of the two output axes the
    fastest-varying program id walks, which decides whether concurrent CTAs
    share their A tile or their B tile in L2.  With ``SK > 1`` the reduction is
    cut into SK chunks of KC and each chunk stores an fp32 partial slab at
    ``C + sk * PSTRIDE`` for `_red` to sum.
    """
    pid = tl.program_id(0)
    lid = tl.program_id(1)
    if NFAST:
        pm = pid // GN
        pn = pid % GN
    else:
        pm = pid % GM
        pn = pid // GM
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    mm = rm < M
    mn = rn < N
    if SK == 1:
        k0 = 0
        kend = K
    else:
        sk = tl.program_id(2)
        k0 = sk * KC
        kend = min(k0 + KC, K)
    ap = A + lid * sab + rm[:, None] * sam + (k0 + rk)[None, :] * sak
    bp = B + lid * sbb + (k0 + rk)[:, None] * sbk + rn[None, :] * sbn
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for kk in range(k0, kend, BK):
        if EVEN_K:
            a = tl.load(ap) if EVEN_M else tl.load(ap, mask=mm[:, None], other=0.0)
            b = tl.load(bp) if EVEN_N else tl.load(bp, mask=mn[None, :], other=0.0)
        else:
            mk = (kk + rk) < kend
            if EVEN_M:
                a = tl.load(ap, mask=mk[None, :], other=0.0)
            else:
                a = tl.load(ap, mask=mm[:, None] & mk[None, :], other=0.0)
            if EVEN_N:
                b = tl.load(bp, mask=mk[:, None], other=0.0)
            else:
                b = tl.load(bp, mask=mk[:, None] & mn[None, :], other=0.0)
        if TF32R:
            # Round both operands to tf32 with round-to-nearest-even *before*
            # the dot.  The tf32 MMA path truncates the fp32 mantissa, which
            # doubles the quantization error and biases it; rounding first
            # reproduces what the reference tf32 kernel computes (matched ratio
            # 0.999 vs 0.57 for plain input_precision="tf32").
            ai = a.to(tl.int32, bitcast=True)
            a = ((ai + 0x0FFF + ((ai >> 13) & 1)) & -8192).to(tl.float32, bitcast=True)
            bi = b.to(tl.int32, bitcast=True)
            b = ((bi + 0x0FFF + ((bi >> 13) & 1)) & -8192).to(tl.float32, bitcast=True)
        acc = tl.dot(a, b, acc, input_precision=PREC)
        ap += BK * sak
        bp += BK * sbk
    if SK == 1:
        if HAS_BIAS:
            acc += tl.load(BIAS + rn, mask=mn, other=0.0).to(tl.float32)[None, :]
        cp = C + lid * scb + rm[:, None] * scm + rn[None, :]
        if EVEN_M and EVEN_N:
            tl.store(cp, acc.to(C.dtype.element_ty))
        else:
            tl.store(cp, acc.to(C.dtype.element_ty), mask=mm[:, None] & mn[None, :])
    else:
        cp = C + sk * PSTRIDE + lid * scb + rm[:, None] * scm + rn[None, :]
        if EVEN_M and EVEN_N:
            tl.store(cp, acc)
        else:
            tl.store(cp, acc, mask=mm[:, None] & mn[None, :])


@triton.jit
def _red(P, BIAS, Y, TOT, N: tl.constexpr, SK: tl.constexpr,
         PSTRIDE: tl.constexpr, BLOCK: tl.constexpr, HAS_BIAS: tl.constexpr):
    """Sum `_mm`'s SK fp32 partial slabs, add bias, cast to the output dtype."""
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < TOT
    acc = tl.load(P + i, mask=m, other=0.0)
    for s in tl.static_range(1, SK):
        acc += tl.load(P + s * PSTRIDE + i, mask=m, other=0.0)
    if HAS_BIAS:
        acc += tl.load(BIAS + (i % N), mask=m, other=0.0).to(tl.float32)
    tl.store(Y + i, acc.to(Y.dtype.element_ty), mask=m)


# ---------------------------------------------------------------------------
# Per-shape configs: (L, M, K, N) -> (BM, BN, BK, warps, stages, n_fastest,
#                                     split_k[, num_ctas])
#
# Only shapes listed here run on `_mm`; anything else uses the reference op.
# Entries were picked by an exhaustive grid sweep measured with the benchmark's
# own timing loop (see ITERATIONS.md for the sweep sizes and the shapes that
# lost, so they are not re-explored).
# ---------------------------------------------------------------------------
_MM_CFG = {
    # tf32 batched matmul: the reference is a legacy sm_80 CUTLASS TF32 kernel.
    (12, 77, 64, 77): (16, 64, 64, 8, 4, 1, 1),
    (12, 77, 77, 64): (16, 32, 32, 4, 3, 1, 1),
}


# ---------------------------------------------------------------------------
# Launch plans (built once per shape, then cached)
# ---------------------------------------------------------------------------
_PLAN = {}
_RED_BLOCK = 2048


def _mm_plan(L, M, N, K, dtype, has_bias):
    cfg = _MM_CFG.get((L, M, K, N))
    if cfg is None:
        return None
    bm, bn, bk, warps, stages, nfast, sk = cfg[:7]
    ctas = cfg[7] if len(cfg) > 7 else 1
    bk = min(bk, max(16, triton.next_power_of_2(K)))
    gm = triton.cdiv(M, bm)
    gn = triton.cdiv(N, bn)
    f32 = dtype is torch.float32
    kc = triton.cdiv(triton.cdiv(K, sk), bk) * bk if sk > 1 else K
    if sk > 1:
        sk = triton.cdiv(K, kc)
    meta = dict(BM=bm, BN=bn, BK=bk, GM=gm, GN=gn, NFAST=nfast,
                SK=sk, KC=kc,
                PSTRIDE=(L * gm * bm * gn * bn if sk > 1 else 0),
                HAS_BIAS=(has_bias and sk == 1),
                EVEN_M=(M % bm == 0), EVEN_N=(N % bn == 0),
                EVEN_K=(K % bk == 0 and (sk == 1 or kc % bk == 0)),
                TF32R=f32, PREC="tf32" if f32 else "ieee",
                num_warps=warps, num_stages=stages, num_ctas=ctas)
    grid = (gm * gn, L) if sk == 1 else (gm * gn, L, sk)
    return (grid, meta, sk, gm * bm, gn * bn)


def _run_mm(a, b, bias, c, L, M, N, K, sa, sb, sc, plan):
    grid, meta, sk, pm, pn = plan
    if sk == 1:
        _mm[grid](a, b, bias, c, M, N, K, sa[0], sa[1], sa[2],
                  sb[0], sb[1], sb[2], sc[0], sc[1], **meta)
        return c
    tot = L * pm * pn
    part = torch.empty(sk * tot, dtype=torch.float32, device=a.device)
    _mm[grid](a, b, None, part, M, N, K, sa[0], sa[1], sa[2],
              sb[0], sb[1], sb[2], pn * pm, pn, **meta)
    nout = c.numel()
    _red[(triton.cdiv(nout, _RED_BLOCK),)](
        part, bias, c, nout, N=N, SK=sk, PSTRIDE=tot, BLOCK=_RED_BLOCK,
        HAS_BIAS=bias is not None, num_warps=4)
    return c


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------
def _linear(x, w, bias):
    """F.linear: y[..., N] = x[..., K] @ w[N, K].T (+ bias[N])."""
    if (w.ndim != 2 or w.dtype is not x.dtype or not x.is_contiguous()
            or w.stride(1) != 1 or x.shape[-1] != w.shape[1]
            or (bias is not None and (bias.dtype is not x.dtype
                                      or not bias.is_contiguous()))):
        return torch.nn.functional.linear(x, w, bias)
    N, K = w.shape
    if K == 0 or N == 0 or x.numel() == 0:
        return torch.nn.functional.linear(x, w, bias)
    M = x.numel() // K
    key = (M, N, K, x.dtype, bias is not None)
    if key in _PLAN:
        plan = _PLAN[key]
    else:
        plan = _mm_plan(1, M, N, K, x.dtype, bias is not None)
        _PLAN[key] = plan
    if plan is None:
        return torch.nn.functional.linear(x, w, bias)
    y = torch.empty(x.shape[:-1] + (N,), dtype=x.dtype, device=x.device)
    _run_mm(x, w, bias, y, 1, M, N, K, (0, K, 1), (0, 1, w.stride(0)), (0, N), plan)
    return y


def _batch_of(t):
    """(batch, batch_stride) collapsed from the leading dims, or None if the
    batch dims cannot be described by a single stride."""
    batch, stride = 1, 0
    for i in range(t.ndim - 2):
        if t.shape[i] == 1:
            continue
        if batch != 1:
            return None
        batch, stride = t.shape[i], t.stride(i)
    return batch, stride


def _matmul(a, b):
    """torch.matmul for the batched-3D/4D case; anything else defers."""
    if (b.dtype is not a.dtype or a.ndim < 3 or a.ndim != b.ndim
            or a.shape[:-2] != b.shape[:-2] or a.stride(-1) != 1
            or b.shape[-2] != a.shape[-1]):
        return torch.matmul(a, b)
    ba = _batch_of(a)
    bb = _batch_of(b)
    if ba is None or bb is None or ba[0] != bb[0]:
        return torch.matmul(a, b)
    L = ba[0]
    M, K = a.shape[-2], a.shape[-1]
    N = b.shape[-1]
    if M == 0 or N == 0 or K == 0:
        return torch.matmul(a, b)
    key = ("bmm", L, M, K, N, a.dtype)
    if key in _PLAN:
        plan = _PLAN[key]
    else:
        plan = _mm_plan(L, M, N, K, a.dtype, False)
        _PLAN[key] = plan
    if plan is None:
        return torch.matmul(a, b)
    c = torch.empty(a.shape[:-1] + (N,), dtype=a.dtype, device=a.device)
    _run_mm(a, b, None, c, L, M, N, K,
            (ba[1], a.stride(-2), a.stride(-1)),
            (bb[1], b.stride(-2), b.stride(-1)),
            (_batch_of(c)[1], c.stride(-2)), plan)
    return c


# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------
class Matmul(nn.Module):
    """Pure functional linear: takes input, weight, and optional bias as forward args."""

    def forward(self, input, weight, bias=None):
        return _linear(input, weight, bias)


class BMM(nn.Module):
    """Batch matrix multiply: torch.matmul(a, b)."""

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return _matmul(a, b)


class Linear(nn.Module):
    """Parametric linear: stores weight and bias internally."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.matmul = Matmul()

    def forward(self, input):
        return self.matmul(input, self.weight, self.bias)
