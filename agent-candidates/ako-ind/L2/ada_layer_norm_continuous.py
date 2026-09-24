"""Adaptive continuous layer norm for diffusion transformers (L2 composite).

Used as the final output norm in FLUX (``norm_out``).  Projects the
conditioning embedding through SiLU + Linear into per-channel scale and
shift, then applies LayerNorm with those modulations.

Fused two-launch Triton pipeline
--------------------------------
The captured shapes are batch-1 (``x[1, 4096|1024, 3072]``,
``conditioning_embedding[1, 3072]``), so ``scale``/``shift`` are per-channel
vectors shared by every token.  That collapses the whole module into:

1. ``_modulation_kernel`` -- a small prologue that fuses SiLU + the
   ``C -> 2*N`` projection (with bias) and folds the LayerNorm affine into
   per-channel coefficients ``a[c] = w[c] * (1 + scale[c])`` and
   ``b[c] = bias[c] * (1 + scale[c]) + shift[c]``.  Nothing here touches the
   big tensor, so no broadcast multiply/add pass over ``x`` survives.
2. ``_ada_layer_norm_chunked_kernel`` / ``_ada_layer_norm_kernel`` --
   row-per-program LayerNorm over the last dim.  ``x`` is read from HBM once,
   mean/rstd are computed in fp32 (the epilogue re-reads the row out of L2, so
   it costs no HBM traffic), and ``y = (x-mean)*rstd*a[c] + b[c]`` goes out in
   the input dtype.  The chunked form keeps every lane live and wins from
   ~2048 rows up; below that the wide single-tile form has better occupancy.

That is one read plus one write of ``x`` (plus the unavoidable read of the
projection weight), versus the baseline's separate silu / addmm / layer_norm /
mul / add passes.  PDL (``launch_pdl``) lets the norm kernel stream ``x`` and
finish its reduction while the prologue is still draining.

The captured FLUX config has ``elementwise_affine=False``, so ``w``/``bias``
above drop out and the coefficients are just ``1 + scale`` and ``shift``.  They
are then stored in the activation dtype rather than fp32: the reference forms
``1 + scale`` in bf16 too, so bf16 storage is *lossless here* and halves the
coefficient traffic into every norm program (measured 13.7 -> 10.4 us on the
4096-row norm kernel).  With affine parameters the fold needs a real fp32
product and the coefficients stay fp32.

Both kernels are bandwidth-bound, so what is left is L2 residency rather than
arithmetic: see the ``_PR_EV_W`` / ``_LN_EV_X`` block below.

All reductions and the modulation math run in fp32, so the only numeric
difference from the reference is that it rounds the normalized value to bf16
before applying the modulation and we do not -- worth ~1 bf16 ulp, far inside
the bf16 tolerance (measured max_abs 3.1e-2 with 100% of elements matched).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# PDL intrinsics (Triton 3.6+).  No-ops on older Triton.
# ---------------------------------------------------------------------------
try:  # pragma: no cover - depends on the installed Triton
    from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait

    _HAS_PDL = True
except ImportError:  # pragma: no cover
    _HAS_PDL = False

    @triton.jit
    def gdc_launch_dependents():
        pass

    @triton.jit
    def gdc_wait():
        pass


# ---------------------------------------------------------------------------
# Kernel 1: SiLU + projection, folded into per-channel LayerNorm coefficients.
# ---------------------------------------------------------------------------
@triton.jit
def _proj_acc(COND, W, offs_n, nmask, C: tl.constexpr, BLOCK_N: tl.constexpr,
              BLOCK_K: tl.constexpr, DT: tl.constexpr, EVEN_N: tl.constexpr,
              EVEN_K: tl.constexpr, EV_W: tl.constexpr):
    """``dot(silu(cond), W[offs_n, :])`` accumulated in fp32.

    ``C`` is a constexpr so the K loop is a fully unrolled ``static_range``:
    every 128-bit load of the weight tile is issued up front, which is what
    gets this read-only stream near HBM peak (a rolled loop with a small tile
    leaves far too little in flight -- and a persistent grid-stride form is
    worse still, since Triton does not pipeline across the outer iterations).
    """
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    wp = W + offs_n[:, None].to(tl.int64) * C
    for k0 in tl.static_range(0, C, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        if EVEN_K:
            c = tl.load(COND + offs_k).to(tl.float32)
        else:
            c = tl.load(COND + offs_k, mask=offs_k < C, other=0.0).to(tl.float32)
        # SiLU, rounded to the linear's dtype exactly like the reference
        # (``self.silu(cond).to(x.dtype)``) before the matmul.  Every program
        # activates the *whole* conditioning vector, so this is 768x redundant;
        # it costs 0.36 us isolated (2 MUFU/element: ex2 plus the rcp inside
        # div.full).  A 1-MUFU form -- ``x/2*(1+tanh.approx(x/2))`` -- does
        # recover that 0.36 us in isolation, but end-to-end it measures
        # identical once the weight stream is no longer pinned in L2, so the
        # exact form is kept.  Hoisting the SiLU into its own launch is much
        # worse (+3.7 us): the extra launch costs more than the MUFU work.
        s = (c / (1.0 + tl.exp2(-1.4426950408889634 * c))).to(DT).to(tl.float32)
        if EVEN_N and EVEN_K:
            w = tl.load(wp + offs_k[None, :],
                        eviction_policy=EV_W).to(tl.float32)
        else:
            m = nmask[:, None]
            if not EVEN_K:
                m = m & (offs_k < C)[None, :]
            w = tl.load(wp + offs_k[None, :], mask=m, other=0.0,
                        eviction_policy=EV_W).to(tl.float32)
        acc += tl.sum(w * s[None, :], axis=1)
    return acc


@triton.jit
def _modulation_kernel(
    COND, W, BIAS, LNW, LNB, AB,
    N, stride_cond, stride_ab,
    C: tl.constexpr, HAS_BIAS: tl.constexpr, HAS_LNW: tl.constexpr,
    HAS_LNB: tl.constexpr, COUPLED: tl.constexpr, DT: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, EVEN_N: tl.constexpr,
    EVEN_K: tl.constexpr, USE_PDL: tl.constexpr, EV_W: tl.constexpr,
):
    """Write ``AB[b] = [a(0..N), b(0..N)]`` for one conditioning row.

    ``COUPLED`` (the LayerNorm has a bias) needs ``scale`` and ``shift`` of the
    same channel in one program.  Otherwise the two halves are independent and
    the grid runs flat over the 2*N projection outputs, which doubles the CTA
    count for a given ``BLOCK_N``.
    """
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)
    cond = COND + pid_b * stride_cond
    out = AB + pid_b * stride_ab
    offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    abdt = AB.dtype.element_ty

    if COUPLED:
        nmask = offs < N
        scale = _proj_acc(cond, W, offs, nmask, C, BLOCK_N, BLOCK_K, DT,
                          EVEN_N, EVEN_K, EV_W)
        shift = _proj_acc(cond, W, offs + N, nmask, C, BLOCK_N, BLOCK_K, DT,
                          EVEN_N, EVEN_K, EV_W)
        if HAS_BIAS:
            scale += tl.load(BIAS + offs, mask=nmask, other=0.0).to(tl.float32)
            shift += tl.load(BIAS + offs + N, mask=nmask,
                             other=0.0).to(tl.float32)
        # Round through the linear's output dtype and form ``1 + scale`` in
        # that dtype, matching the reference's intermediate precision.
        t = (1.0 + scale.to(DT).to(tl.float32)).to(DT).to(tl.float32)
        shift = shift.to(DT).to(tl.float32)
        a = t
        if HAS_LNW:
            a = t * tl.load(LNW + offs, mask=nmask, other=0.0).to(tl.float32)
        b = shift + tl.load(LNB + offs, mask=nmask, other=0.0).to(tl.float32) * t
        tl.store(out + offs, a.to(abdt), mask=nmask)
        tl.store(out + offs + N, b.to(abdt), mask=nmask)
    else:
        nmask = offs < 2 * N
        v = _proj_acc(cond, W, offs, nmask, C, BLOCK_N, BLOCK_K, DT,
                      EVEN_N, EVEN_K, EV_W)
        if HAS_BIAS:
            v += tl.load(BIAS + offs, mask=nmask, other=0.0).to(tl.float32)
        v = v.to(DT).to(tl.float32)
        is_scale = offs < N
        # first half -> a = (1 + scale) [* lnw];  second half -> b = shift
        v = tl.where(is_scale, (1.0 + v).to(DT).to(tl.float32), v)
        if HAS_LNW:
            w = tl.load(LNW + tl.where(is_scale, offs, 0), mask=is_scale,
                        other=1.0).to(tl.float32)
            v = v * tl.where(is_scale, w, 1.0)
        tl.store(out + offs, v.to(abdt), mask=nmask)

    if USE_PDL:
        gdc_launch_dependents()


# ---------------------------------------------------------------------------
# Kernel 2: row-per-program LayerNorm with the folded coefficients.
# ---------------------------------------------------------------------------
@triton.jit
def _ada_layer_norm_kernel(
    X, Y, AB, T, stride_ab, eps,
    N: tl.constexpr, ROWS: tl.constexpr, BLOCK: tl.constexpr,
    EVEN_N: tl.constexpr, EVEN_T: tl.constexpr, USE_PDL: tl.constexpr,
    EV_X: tl.constexpr, EV_AB: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_b = tl.program_id(1)
    rows = pid * ROWS + tl.arange(0, ROWS)
    offs = tl.arange(0, BLOCK)
    base = (pid_b.to(tl.int64) * T + rows[:, None]) * N + offs[None, :]

    if EVEN_N and EVEN_T:
        x = tl.load(X + base, eviction_policy=EV_X).to(tl.float32)
        mean = tl.sum(x, axis=1) * (1.0 / N)
        xc = x - mean[:, None]
    else:
        m = tl.full([1, 1], True, tl.int1)
        if not EVEN_N:
            m = m & (offs[None, :] < N)
        if not EVEN_T:
            m = m & (rows[:, None] < T)
        x = tl.load(X + base, mask=m, other=0.0,
                    eviction_policy=EV_X).to(tl.float32)
        mean = tl.sum(x, axis=1) * (1.0 / N)
        xc = tl.where(m, x - mean[:, None], 0.0)
    var = tl.sum(xc * xc, axis=1) * (1.0 / N)
    rstd = 1.0 / tl.sqrt(var + eps)

    # Everything above is independent of the prologue, so wait as late as
    # possible: the x stream overlaps the prologue's tail.
    if USE_PDL:
        gdc_wait()
    abp = AB + pid_b * stride_ab
    if EVEN_N:
        a = tl.load(abp + offs, eviction_policy=EV_AB).to(tl.float32)
        b = tl.load(abp + N + offs, eviction_policy=EV_AB).to(tl.float32)
    else:
        nm = offs < N
        a = tl.load(abp + offs, mask=nm, other=0.0,
                    eviction_policy=EV_AB).to(tl.float32)
        b = tl.load(abp + N + offs, mask=nm, other=0.0,
                    eviction_policy=EV_AB).to(tl.float32)
    y = xc * (rstd[:, None] * a[None, :]) + b[None, :]

    if EVEN_N and EVEN_T:
        tl.store(Y + base, y.to(Y.dtype.element_ty))
    else:
        tl.store(Y + base, y.to(Y.dtype.element_ty), mask=m)


@triton.jit
def _ada_layer_norm_chunked_kernel(
    X, Y, AB, T, stride_ab, eps,
    N: tl.constexpr, ROWS: tl.constexpr, CHUNK: tl.constexpr,
    USE_PDL: tl.constexpr, EV_X: tl.constexpr, EV_AB: tl.constexpr,
):
    """Same math, but the row is walked in ``CHUNK``-wide pieces.

    ``N`` is a multiple of ``CHUNK`` so every lane is live (the single-tile
    kernel has to pad 3072 up to a 4096-wide power-of-2 tile and idles a
    quarter of its lanes).  The reduction pass leaves the row in L1, so the
    re-read in the epilogue costs no HBM traffic -- one HBM read, one write,
    with full lane utilisation.
    """
    pid = tl.program_id(0)
    pid_b = tl.program_id(1)
    rows = pid * ROWS + tl.arange(0, ROWS)
    row0 = (pid_b.to(tl.int64) * T) + rows
    rp = X + row0[:, None] * N
    # Shifted one-pass moments: accumulate around x[row, 0] so that
    # ``E[d^2] - E[d]^2`` cancels against the row's own spread rather than
    # against its offset -- as stable as a two-pass reduction, and the extra
    # subtract is free on a bandwidth-bound kernel.
    k = tl.load(X + row0 * N).to(tl.float32)
    s = tl.zeros([ROWS], tl.float32)
    ss = tl.zeros([ROWS], tl.float32)
    for j in tl.static_range(0, N, CHUNK):
        o = j + tl.arange(0, CHUNK)
        d = tl.load(rp + o[None, :],
                    eviction_policy=EV_X).to(tl.float32) - k[:, None]
        s += tl.sum(d, axis=1)
        ss += tl.sum(d * d, axis=1)
    dmean = s * (1.0 / N)
    mean = k + dmean
    var = ss * (1.0 / N) - dmean * dmean
    rstd = 1.0 / tl.sqrt(tl.maximum(var, 0.0) + eps)

    if USE_PDL:
        gdc_wait()
    abp = AB + pid_b * stride_ab
    yp = Y + ((pid_b.to(tl.int64) * T) + rows[:, None]) * N
    for j in tl.static_range(0, N, CHUNK):
        o = j + tl.arange(0, CHUNK)
        v = tl.load(rp + o[None, :]).to(tl.float32)
        a = tl.load(abp + o, eviction_policy=EV_AB).to(tl.float32)
        b = tl.load(abp + N + o, eviction_policy=EV_AB).to(tl.float32)
        y = (v - mean[:, None]) * (rstd[:, None] * a[None, :]) + b[None, :]
        tl.store(yp + o[None, :], y.to(Y.dtype.element_ty))


# ---------------------------------------------------------------------------
# Host side.
# ---------------------------------------------------------------------------
# Launch geometry.  Tuned end-to-end on B200 (isolated per-kernel tuning
# misleads here: with PDL the two kernels overlap, so the prologue's CTA count
# trades against the norm kernel's occupancy).
_PR_N = 8                # projection outputs per prologue program
_PR_K = 1024             # K chunk of the (statically unrolled) prologue loop
_PR_WARPS = 4
_LN_CHUNK = 512          # chunked norm: elements per pass step
_LN_CHUNK_ROWS = 1
_LN_CHUNK_WARPS = 1
_LN_CHUNK_MIN_ROWS = 2048  # below this the wide single-tile program wins
_LN_TILE_ROWS = 1
_LN_TILE_WARPS = 4
# L2 residency policies.  These are not interchangeable and each was measured
# end-to-end; the right hint depends on whether the stream is re-read.
#
# The projection weight is read exactly once per call, so it must NOT be pinned:
# every timed call starts with L2 full of dirty lines (the harness flushes 253 MB
# before recording the start event), and `evict_last` on a read-once stream makes
# the allocator evict those *dirty* lines instead of our own clean ones, paying a
# writeback for all 36 MiB.  Switching this one hint to `evict_first` is worth
# 2.0 us at 4096 rows (35.84 vs 37.90, every rep of every run).  At 1024 rows it
# is worth 0.1-2.0 us: the old form is bimodal between two ~2 us rungs across
# reps while this one sits on the low rung every time.  2.49x -> 2.66x geomean.
# `evict_first` specifically, not merely dropping the hint: the default policy is
# 2.0 us slower than `evict_first` at 4096 rows.
#
# `x` is the opposite case: the chunked norm kernel walks the row twice (moments,
# then the epilogue) and the second pass reads it back out of L2, so pinning it
# is load-bearing -- dropping to the default policy costs 2.0 us at 4096 rows and
# `evict_first` costs 4.1 us.  The coefficient buffer is 12 KB read by every
# program, so it is pinned for the same reason.
_PR_EV_W = "evict_first"   # 36 MiB projection weight: read once, do not pin
_LN_EV_X = "evict_last"    # x: the chunked epilogue re-reads the row from L2
_LN_EV_AB = "evict_last"   # 12 KB of coefficients, read by every program

_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16, torch.float32)
_TL_DTYPE = {torch.bfloat16: tl.bfloat16, torch.float16: tl.float16,
             torch.float32: tl.float32}


class _ProjParams(nn.Module):
    """Weight/bias container for the conditioning projection.

    Parameter names match the reference ``Linear`` so the benchmark's
    ``load_state_dict`` shares weights.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None


class _NormParams(nn.Module):
    """Optional LayerNorm affine parameters (names match the reference)."""

    def __init__(self, normalized_shape: int, elementwise_affine: bool,
                 create_scale: bool = True, create_offset: bool = True):
        super().__init__()
        if elementwise_affine and create_scale:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
        else:
            self.register_parameter("weight", None)
        if elementwise_affine and create_offset:
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter("bias", None)


class AdaLayerNormContinuous(nn.Module):
    r"""
    Adaptive normalization layer with a norm layer (layer_norm or rms_norm).

    Args:
        embedding_dim (`int`): Embedding dimension to use during projection.
        conditioning_embedding_dim (`int`): Dimension of the input condition.
        elementwise_affine (`bool`, defaults to `True`):
            Boolean flag to denote if affine transformation should be applied.
        eps (`float`, defaults to 1e-5): Epsilon factor.
        bias (`bool`, defaults to `True`): Whether to use bias in the linear layer.
        norm_type (`str`, defaults to `"layer_norm"`):
            Normalization layer to use. Values supported: "layer_norm", "rms_norm".
    """

    def __init__(
        self,
        embedding_dim: int,
        conditioning_embedding_dim: int,
        elementwise_affine=True,
        eps=1e-5,
        bias=True,
        norm_type="layer_norm",
        promote_fp32: bool = True,
    ):
        super().__init__()
        if norm_type != "layer_norm":
            raise ValueError(f"unknown norm_type {norm_type}")
        self.embedding_dim = embedding_dim
        self.conditioning_embedding_dim = conditioning_embedding_dim
        self.eps = eps
        self.promote_fp32 = promote_fp32
        self.linear = _ProjParams(conditioning_embedding_dim, embedding_dim * 2,
                                  bias=bias)
        self.norm = _NormParams(embedding_dim, elementwise_affine)
        self._ab = None          # persistent coefficient scratch
        self._plans: dict = {}   # launch plan cache, keyed by shape/dtype

    # -- fallback for layouts the fused path does not cover ------------------
    def _reference(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        emb = F.linear(F.silu(cond).to(x.dtype), self.linear.weight,
                       self.linear.bias)
        scale, shift = torch.chunk(emb, 2, dim=1)
        w, b = self.norm.weight, self.norm.bias
        shape = (self.embedding_dim,)
        if self.promote_fp32:
            xn = F.layer_norm(
                x.float(), shape,
                None if w is None else w.float(),
                None if b is None else b.float(), self.eps).to(x.dtype)
        else:
            xn = F.layer_norm(x, shape, w, b, self.eps)
        return xn * (1 + scale)[:, None, :] + shift[:, None, :]

    # -- fused path ---------------------------------------------------------
    def _plan(self, x: torch.Tensor, cond: torch.Tensor) -> dict:
        n = self.embedding_dim
        coupled = self.norm.bias is not None
        if cond.shape[0] == 1:
            batches, rows = 1, x.numel() // n
        else:
            batches, rows = cond.shape[0], x.numel() // (n * cond.shape[0])
        block = triton.next_power_of_2(n)
        # Chunked (full-lane) norm needs enough rows to fill the machine with
        # its 1-warp programs; below that the wide single-tile program wins.
        chunk = _LN_CHUNK if (rows >= _LN_CHUNK_MIN_ROWS
                              and n % _LN_CHUNK == 0) else 0
        ln_rows = _LN_CHUNK_ROWS if chunk else _LN_TILE_ROWS
        while ln_rows > 1 and rows % ln_rows:
            ln_rows //= 2
        pr_n, pr_k = _PR_N, min(_PR_K, triton.next_power_of_2(
            self.conditioning_embedding_dim))
        return dict(
            rows=rows, batches=batches, block=block, stride_ab=2 * n,
            chunk=chunk, ln_grid=(triton.cdiv(rows, ln_rows), batches),
            ln_rows=ln_rows,
            ln_warps=_LN_CHUNK_WARPS if chunk else _LN_TILE_WARPS,
            even_n=(block == n), even_t=(rows % ln_rows == 0),
            pr_grid=(triton.cdiv(n if coupled else 2 * n, pr_n), batches),
            pr_n=pr_n, pr_k=pr_k, pr_warps=_PR_WARPS, coupled=coupled,
            pr_even_n=((n if coupled else 2 * n) % pr_n == 0),
            pr_even_k=(self.conditioning_embedding_dim % pr_k == 0),
            ab_dtype=(torch.float32 if (self.norm.weight is not None
                                        or coupled) else x.dtype),
            dt=_TL_DTYPE[x.dtype],
        )

    def _fused(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        n = self.embedding_dim
        key = (x.shape, cond.shape, x.dtype)
        plan = self._plans.get(key)
        if plan is None:
            plan = self._plan(x, cond)
            self._plans[key] = plan

        ab = self._ab
        nb = plan["batches"]
        if (ab is None or ab.shape[0] != nb or ab.dtype != plan["ab_dtype"]
                or ab.device != x.device):
            ab = torch.empty((nb, 2 * n), dtype=plan["ab_dtype"], device=x.device)
            self._ab = ab

        pdl = _HAS_PDL
        _modulation_kernel[plan["pr_grid"]](
            cond, self.linear.weight, self.linear.bias,
            self.norm.weight, self.norm.bias, ab,
            n, cond.stride(0), plan["stride_ab"],
            C=self.conditioning_embedding_dim,
            HAS_BIAS=self.linear.bias is not None,
            HAS_LNW=self.norm.weight is not None,
            HAS_LNB=plan["coupled"], COUPLED=plan["coupled"], DT=plan["dt"],
            BLOCK_N=plan["pr_n"], BLOCK_K=plan["pr_k"],
            EVEN_N=plan["pr_even_n"], EVEN_K=plan["pr_even_k"], USE_PDL=pdl,
            EV_W=_PR_EV_W,
            num_warps=plan["pr_warps"], num_stages=1, launch_pdl=pdl,
        )
        y = torch.empty_like(x)
        if plan["chunk"]:
            _ada_layer_norm_chunked_kernel[plan["ln_grid"]](
                x, y, ab, plan["rows"], plan["stride_ab"], self.eps,
                N=n, ROWS=plan["ln_rows"], CHUNK=plan["chunk"], USE_PDL=pdl,
                EV_X=_LN_EV_X, EV_AB=_LN_EV_AB,
                num_warps=plan["ln_warps"], num_stages=1, launch_pdl=pdl,
            )
        else:
            _ada_layer_norm_kernel[plan["ln_grid"]](
                x, y, ab, plan["rows"], plan["stride_ab"], self.eps,
                N=n, ROWS=plan["ln_rows"], BLOCK=plan["block"],
                EVEN_N=plan["even_n"], EVEN_T=plan["even_t"], USE_PDL=pdl,
                EV_X=_LN_EV_X, EV_AB=_LN_EV_AB,
                num_warps=plan["ln_warps"], num_stages=1, launch_pdl=pdl,
            )
        return y

    def forward(self, x: torch.Tensor, conditioning_embedding: torch.Tensor
                ) -> torch.Tensor:
        if (x.dim() == 3 and x.shape[-1] == self.embedding_dim
                and x.dtype in _SUPPORTED_DTYPES and x.is_cuda
                and (x.dtype != torch.float32
                     or not torch.backends.cuda.matmul.allow_tf32)
                and self.embedding_dim <= 8192 and x.is_contiguous()
                and conditioning_embedding.dim() == 2
                and conditioning_embedding.shape[-1] == self.conditioning_embedding_dim
                and conditioning_embedding.shape[0] in (1, x.shape[0])
                and conditioning_embedding.stride(-1) == 1):
            return self._fused(x, conditioning_embedding)
        return self._reference(x, conditioning_embedding)
