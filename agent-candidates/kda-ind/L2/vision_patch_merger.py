"""Patch merger for Qwen vision encoders.

Merges spatial_merge_size^2 patches into one via norm + 2-layer MLP.

Unified across Qwen2-VL and Qwen3-VL:
  - use_postshuffle_norm: Qwen3 DeepStack mergers norm after spatial reshape.

The two GEMMs are left on their incumbent libraries: profiling puts the nvJet
2-SM cooperative tcgen05 kernels at 95.1% and 96.5% of peak sustained
tensor-pipe active, so there is nothing to win there. The norm is a different
story -- PyTorch's vectorized_layer_norm_kernel gives one 128-thread block per
1152-wide row, so every reduction crosses shared memory and __syncthreads. It
is the only kernel in the profile with real barrier pressure (barrier = 1.69
warps/issue) and it moves 298 MB in 205.9 us at N=64680, 1.45 TB/s against a
measured 5.11 TB/s device-copy figure. That is the headroom this file spends.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

from ..L1.gelu import GELU
from ..L1.layer_norm import LayerNorm
from .parallel_linear import ColumnParallelLinear, RowParallelLinear

# torch._addmm_activation is a private ATen op, so its presence is checked
# rather than assumed. It absorbs fc1's bias and GELU into the GEMM epilogue,
# which deletes the standalone GELU kernel outright: measured 429.0 us against
# 516.2 us for addmm + F.gelu at M=16170.
_HAS_ADDMM_ACTIVATION = hasattr(torch, "_addmm_activation")

# The CUTLASS epilogue implements *tanh* GELU, not the exact-erf GELU this
# module's F.gelu(approximate="none") reference computes. An fp32-accumulator
# oracle settles it: the epilogue's output agrees with a tanh reference on
# 0.99310 of elements bit-for-bit in bf16 and with an erf reference on only
# 0.87067 (mean |delta| 2.3e-6 against 1.2e-4). The substitution costs about
# 5e-4 of the matched ratio in bf16, whose 1e-2 tolerance absorbs it, but it
# exceeds fp16's 1e-3 bound -- measured 0.98931 against the fp16 reference,
# under the 0.99 the harness requires. fp16 therefore does not take the fast
# path at all; it runs the reference expression end to end.
_FAST_DTYPE = torch.bfloat16

# Normalized widths the fast path will launch. The row decomposition below is
# general, but a launched specialization has to come with measured register and
# local-memory evidence, and that evidence is per configuration -- so only the
# widths in this tuple ship, and every other construction runs the reference
# expression. 1152 is the pre-shuffle benched width (context_dim) and 4608 the
# post-shuffle one (context_dim * spatial_merge_size**2). Widening this tuple
# means extending profile/vision-patch-merger-ln-round1-evidence/ to cover the
# new configurations.
_PROFILED_NORM_DIMS = (1152, 4608)

# ptxas contracts the normalization arithmetic into FMAs, and its rearrangement
# leaves the row mean carrying a ~1e-8 relative error instead of being exact.
# On ordinary data that is invisible. On a row whose true variance is at or
# below (mean * 1e-8)^2 it is fatal, because rsqrt(var + eps) then amplifies the
# residual into an O(1) offset: a bf16 row constant at 9984 came out 7.4e-2 away
# from F.layer_norm, which returns the bias exactly. The defect is in ptxas, not
# in the Triton/LLVM front end -- an inline-asm optimisation barrier on the sum,
# on the mean, and on the residual all leave it in place, and only --fmad=false
# removes it. Cost of turning contraction off, measured wall-clock in a single
# process on identical source so the A/B is clean: 56.3 -> 60.4 us at N=64680
# and 24.6 -> 27.6 us at N=20680, under 1% of end-to-end latency and the same
# order of premium the design already pays to keep the two-pass
# mean-then-variance form. It also lowers register pressure.
_FP_FUSION = False

# Elements per thread the row-resident kernel is tuned for. An offline sweep of
# S in {1,2,4,8} x num_warps in {1,2,4,8,16} put the optimum at ~64-72 fp32
# values held live per thread; below that there is not enough per-thread work to
# hide the load latency, above it the two chunks spill to local memory. Both
# shipped configurations land at exactly 72 and NCU measures zero local-memory
# traffic for each (profile/vision-patch-merger-ln-round1-evidence/).
_ELEMS_PER_THREAD = 72
_WARP = 32
# The two helpers below stay general so that they are total functions -- they
# must never hand back a configuration that cannot launch -- while
# _PROFILED_NORM_DIMS decides what actually ships. num_warps is capped at 8,
# bounding the width they will decompose; rows per program are capped at 64
# because reaching the elements-per-thread target at a very narrow width
# otherwise wants a tall [S, 1] low chunk, which was the one configuration in
# the wider sweep that spilled. Neither cap binds at 1152 or 4608.
_MAX_WARPS = 8
_MAX_FAST_DIM = _ELEMS_PER_THREAD * _WARP * _MAX_WARPS
_MAX_ROWS_PER_PROG = 64


@triton.jit
def _layer_norm_row_seg(X, Y, G, B, eps, n_rows,
                        HI: tl.constexpr, LO: tl.constexpr, S: tl.constexpr):
    """LayerNorm over rows of width HI + LO, both powers of two.

    A single next_power_of_2(1152) = 2048 tile would mask off 44% of its lanes
    and tops out at 2488 GB/s. Splitting the row into two dense power-of-two
    chunks keeps every lane busy; with one warp per row-group the reduction is
    intra-thread accumulation plus warp shuffles, so no shared memory and no
    __syncthreads.

    profile/vision-patch-merger-ln-round1-evidence/ profiles this source at both
    launchable configurations alongside the reference F.layer_norm, in one GPU
    lease so their numbers are mutually comparable. The decisive counters are
    exact instruction counts rather than sampled ratios: at HI=1024, LO=128,
    S=2, num_warps=1 the kernel executes **zero** shared-memory load and store
    instructions, where the reference executes hundreds of thousands of them, so
    the reduction demonstrably stays inside the warp. At HI=4096, LO=512, S=1,
    num_warps=2 the second warp reintroduces a cross-warp reduction and 62040
    shared loads and stores appear, which the source-level report localizes to
    Triton's reduction helper -- so the one-warp argument does not carry to that
    path, and only correctness and absence of spill are claimed for it. Occupancy,
    register counts and sampled stall ratios are in that report; they are not
    repeated here, so that re-profiling never obliges an edit to this file.

    Two-pass mean-then-variance rather than E[x^2] - E[x]^2. The one-pass form
    is 2.5% faster and passes the benched inputs identically, but loses
    precision catastrophically once |mean| >> std -- data this bench never
    generates and a real vision encoder can.
    """
    # int64 row index: rows * (HI + LO) overflows int32 above ~466k rows, and
    # the kernel is memory-saturated, so the wider arithmetic is free.
    r = tl.program_id(0).to(tl.int64) * S + tl.arange(0, S)
    m = (r < n_rows)[:, None]
    base = r[:, None] * (HI + LO)
    ca = tl.arange(0, HI)[None, :]
    cb = tl.arange(0, LO)[None, :]
    xa = tl.load(X + base + ca, mask=m, other=0.0).to(tl.float32)
    xb = tl.load(X + base + HI + cb, mask=m, other=0.0).to(tl.float32)
    # Correctly-rounded division, not the reciprocal multiply the compiler would
    # otherwise pick. `sum * (1/N)` folds into the `x - mu` subtract as an FMA,
    # so the product never rounds to fp32 and mu carries a ~2^-27 relative
    # error. That is invisible on normal data and catastrophic on a row whose
    # true variance is at or below (mean * 2^-27)^2: on a bf16 row constant at
    # 9984 the residual came out at 7.4e-5 instead of 0, and rsqrt(var + eps)
    # then amplified it to a 7.4e-2 constant offset on every output element --
    # against a reference that returns the bias exactly. div.rn forces mu to
    # materialize, which makes `x - mu` exact for a constant row.
    n = tl.full((1, 1), HI + LO, tl.float32)
    mu = tl.fdiv((tl.sum(xa, 1) + tl.sum(xb, 1))[:, None], n, ieee_rounding=True)
    # Masked rows contribute zero to the variance, so a short final row-group
    # cannot poison its neighbours' statistics.
    da = tl.where(m, xa - mu, 0.0)
    db = tl.where(m, xb - mu, 0.0)
    var = tl.fdiv((tl.sum(da * da, 1) + tl.sum(db * db, 1))[:, None], n,
                  ieee_rounding=True)
    rs = tl.rsqrt(var + eps)
    ga = tl.load(G + ca).to(tl.float32)
    gb = tl.load(G + HI + cb).to(tl.float32)
    ba = tl.load(B + ca).to(tl.float32)
    bb = tl.load(B + HI + cb).to(tl.float32)
    tl.store(Y + base + ca, (da * rs * ga + ba).to(Y.dtype.element_ty), mask=m)
    tl.store(Y + base + HI + cb, (db * rs * gb + bb).to(Y.dtype.element_ty), mask=m)


def _row_segments(dim: int) -> tuple[int, int] | None:
    """Split *dim* into two power-of-two chunks, or None if it does not split.

    A power-of-two width splits evenly; anything else peels off its high bit and
    the remainder must itself be a power of two. The LO > 0 term matters: a bare
    (LO & (LO - 1)) == 0 test admits LO = 0, which happens at dim = 1, and
    tl.arange(0, 0) is not a legal Triton range.
    """
    if dim < 2 or dim > _MAX_FAST_DIM:
        return None
    hi = 1 << (dim.bit_length() - 1)
    lo = dim - hi
    if lo == 0:
        hi = lo = dim >> 1
    if lo <= 0 or (lo & (lo - 1)) != 0:
        return None
    return hi, lo


def _launch_shape(dim: int) -> tuple[int, int]:
    """Rows per program and warps per program for a row of width *dim*.

    Holds S * dim / (32 * num_warps) at or just under the tuned elements per
    thread: dim = 1152 -> (2, 1), dim = 4608 -> (1, 2), both at exactly 72.
    """
    num_warps = 1
    while dim > _ELEMS_PER_THREAD * _WARP * num_warps:
        num_warps *= 2
    rows = 1
    budget = _ELEMS_PER_THREAD * _WARP * num_warps
    while rows * 2 * dim <= budget and rows * 2 <= _MAX_ROWS_PER_PROG:
        rows *= 2
    return rows, num_warps


def _affine_ok(p: torch.Tensor, xn: torch.Tensor, dim: int) -> bool:
    """Whether an affine parameter is laid out the way the kernel indexes it.

    The kernel addresses the scale and offset as dense arrays (`G + ca`), so a
    parameter that is the right shape but strided -- a slice of a wider tensor,
    say -- would silently read the wrong elements, where F.layer_norm handles it.
    Shape, dtype and device are checked for the same reason: where the reference
    would raise, the fast path must not instead compute something.
    """
    return (p.shape == (dim,) and p.is_contiguous()
            and p.dtype == xn.dtype and p.device == xn.device)


def _fast_layer_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
                     eps: float, hi: int, lo: int, rows_per_prog: int,
                     num_warps: int) -> torch.Tensor:
    """LayerNorm over the trailing hi + lo elements of contiguous *x*."""
    n_rows = x.numel() // (hi + lo)
    y = torch.empty_like(x)
    _layer_norm_row_seg[(triton.cdiv(n_rows, rows_per_prog),)](
        x, y, weight, bias, eps, n_rows,
        HI=hi, LO=lo, S=rows_per_prog, num_warps=num_warps,
        enable_fp_fusion=_FP_FUSION,
    )
    return y


class VisionPatchMerger(nn.Module):
    """Patch merger: norm -> flatten -> MLP(fc1, GELU, fc2).

    Qwen3 DeepStack mergers set use_postshuffle_norm=True to norm after reshape.
    """

    def __init__(self, d_model: int, context_dim: int,
                 spatial_merge_size: int = 2,
                 use_postshuffle_norm: bool = False,
                 eps: float = 1e-6):
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size ** 2)
        self.use_postshuffle_norm = use_postshuffle_norm
        norm_dim = self.hidden_size if use_postshuffle_norm else context_dim
        # See VisionBlock: vLLM's vision path uses plain nn.LayerNorm on
        # bf16, and our fp32 promotion costs two full-tensor copies here.
        self.norm = LayerNorm(norm_dim, eps=eps, promote_fp32=False)
        self.fc1 = ColumnParallelLinear(self.hidden_size, self.hidden_size, bias=True)
        self.act = GELU()
        self.fc2 = RowParallelLinear(self.hidden_size, d_model, bias=True)

        # Row decomposition and launch geometry depend only on the normalized
        # width, so they are settled here. Nothing tensor-valued is cached:
        # _prepare_module rebinds p.data when it casts or moves parameters, so a
        # stored weight view would go stale -- silently old weights, or a device
        # mismatch. .t() is metadata-only and is recomputed per call.
        self._norm_dim = norm_dim
        seg = _row_segments(norm_dim) if norm_dim in _PROFILED_NORM_DIMS else None
        if seg is None:
            self._hi = self._lo = self._rows_per_prog = self._num_warps = 0
        else:
            self._hi, self._lo = seg
            self._rows_per_prog, self._num_warps = _launch_shape(norm_dim)
        # One construction-time bool for everything that cannot change without
        # replacing a submodule; _HAS_ADDMM_ACTIVATION is a module-level import
        # probe, and the width decomposition is fixed by norm_dim. hidden_size is
        # in here because spatial_merge_size=0 makes it 0, and the per-call
        # `x.numel() % hidden_size` would then raise ZeroDivisionError out of the
        # predicate itself -- where the reference expression instead raises from
        # view(-1, 0). A guard that raises is not a guard.
        self._fast_eligible = (self._hi > 0 and self.hidden_size > 0
                               and _HAS_ADDMM_ACTIVATION)

    # ---- fast-path predicate -------------------------------------------------

    def _norm_input_ok(self, xn: torch.Tensor) -> bool:
        """Whether the Triton norm may replace self.norm on *xn*.

        *xn* is the tensor the norm actually sees, i.e. already reshaped in the
        post-shuffle configuration, so xn.shape[-1] == norm_dim is the same
        check in both configurations.
        """
        norm = self.norm
        return (
            xn.shape[-1] == self._norm_dim
            # A swapped-in norm may disagree about the width or want the fp32
            # promotion, in which case F.layer_norm is the only correct answer.
            and norm.promote_fp32 is False
            and norm.normalized_shape == (self._norm_dim,)
            and norm.weight is not None
            and norm.bias is not None
            and _affine_ok(norm.weight, xn, self._norm_dim)
            and _affine_ok(norm.bias, xn, self._norm_dim)
        )

    def _fast_path_ok(self, x: torch.Tensor) -> bool:
        """Whether the guarded fast path covers this call.

        Every term that fails routes the call to the full baseline expression
        rather than to a partially optimized path, so a miss can only cost
        latency, never fidelity.
        """
        fc1, fc2 = self.fc1, self.fc2
        return (
            # Everything the optimized path needs is decided here, so a miss
            # returns the complete reference expression and never a hybrid of
            # the two.
            self._fast_eligible
            and x.is_cuda
            and x.numel() > 0
            # fp16 is excluded outright: the fused epilogue is unsafe there and
            # a norm-only fast path would be exactly the partial path this
            # predicate exists to prevent.
            and x.dtype is _FAST_DTYPE
            # Non-contiguous input goes to the baseline rather than through
            # .contiguous(): in the post-shuffle configuration the baseline's
            # x.view(-1, hidden_size) raises where .contiguous().view(...)
            # would succeed, and the candidate must not accept input the
            # baseline rejects.
            and x.is_contiguous()
            and x.numel() % self.hidden_size == 0
            # No backward kernel is written, so training routes to the baseline
            # expression and keeps its autograd graph.
            and not torch.is_grad_enabled()
            and not fc1.use_fp8
            and fc1.bias is not None
            and self.act.approximate == "none"
            # fc2 keeps its own forward, so these three are conservative rather
            # than load-bearing: a sharded or quantized fc2 would still compose
            # correctly with the fused fc1. They stay because the fused epilogue
            # was only ever measured against a single-rank bf16 fc2.
            and not fc2.use_fp8
            and fc2.tp_size == 1
            and fc2.tp_rank == 0
        )

    # ---- forward -------------------------------------------------------------

    def _baseline_forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_postshuffle_norm:
            x = self.norm(x.view(-1, self.hidden_size))
        else:
            x = self.norm(x).view(-1, self.hidden_size)
        return self.fc2(self.act(self.fc1(x)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._fast_path_ok(x):
            return self._baseline_forward(x)

        if self.use_postshuffle_norm:
            xn = x.view(-1, self.hidden_size)
        else:
            xn = x
        if not self._norm_input_ok(xn):
            return self._baseline_forward(x)

        z = _fast_layer_norm(xn, self.norm.weight, self.norm.bias, self.norm.eps,
                             self._hi, self._lo, self._rows_per_prog,
                             self._num_warps)
        # The kernel writes a contiguous output, so the patch merge is
        # metadata-only -- no shuffle kernel.
        z = z.view(-1, self.hidden_size)

        # Unconditional: _addmm_activation's availability and the bf16
        # restriction are both settled by the predicate above.
        h = torch._addmm_activation(self.fc1.bias, z, self.fc1.weight.t(),
                                    use_gelu=True)
        # fc2 keeps its own forward. Bypassing it with F.linear was measured
        # same-process at M=440, 5170 and 16170 with interleaved, order-swapped
        # repetitions: ratios 1.000, 0.951, 1.019 against a per-repetition
        # spread of 286-342 us, i.e. no effect distinguishable from clock drift.
        # Its host-side work sits behind queued GPU work, so there was nothing
        # to win, and calling the submodule keeps its fp8 and tensor-parallel
        # semantics rather than restating them here.
        return self.fc2(h)
