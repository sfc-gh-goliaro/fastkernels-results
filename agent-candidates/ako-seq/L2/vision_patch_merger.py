"""Patch merger for Qwen vision encoders.

Merges spatial_merge_size^2 patches into one via norm + 2-layer MLP.

Unified across Qwen2-VL and Qwen3-VL:
  - use_postshuffle_norm: Qwen3 DeepStack mergers norm after spatial reshape.

The two bf16 GEMMs (hidden x hidden and hidden x d_model, 4608-square and
4608x4096 as captured) are 75-88% of the device time and are left to cuBLAS's
``nvjet_sm100_tst_*_2cta_*`` kernels, but *the layout they are handed is ours*
-- see ``_prepare``'s K-major weight copies below, which is where the small
shapes' last win came from. Owning the multiply itself was measured and lost at
both ends of the shape range: a fully swept Triton GEMM (the frozen
``L1/linear.py:_mm``, ~680 configs over tile / BK / warps / stages / CTA order /
cluster) runs 1.34x slower than cuBLAS at M=440 and 1.26x slower at M=16170.
See ITERATIONS.md for the tables. What is left to win is the glue: at the
64680-row shape the normalization pass is 52 us and the GELU pass 40 us of a
~900 us call, and at the 1760-row shape the two GEMMs are 35 us of 47 us.

* **Both GEMMs get a materialized K-major weight, not the ``.t()`` view.**
  ``F.linear`` hands cuBLAS ``w`` with stride ``(1, K)``; ``_prepare`` instead
  builds ``w.t().contiguous()`` (stride ``(N, 1)``) once and calls
  ``torch.addmm`` against that. Same tile, same 2-CTA cluster -- the heuristic
  just selects ``..._bz_bias_NNT`` over ``..._bz_bias_TNT``, and the B-tile
  loads become contiguous along N. It is worth almost nothing where the weight
  read is amortized and a lot where it is not, which is exactly the shape the
  captures' small case has (device us, cold L2, 3 interleaved reps):

      M=440    fc1 19.85 -> 18.16   fc2 18.68 -> 17.32   pair -7.9%
      M=5170   fc1 125.34 -> 124.29 fc2 105.38 -> 104.87 pair -0.7%
      M=16170  fc1 406.89 -> 405.22 fc2 355.78 -> 356.45 pair -0.1%

  End to end at M=440 that is 51.2 -> 47.1 us, and the output stays *bit*
  identical to the reference's. The cost is one extra weight-sized allocation
  per module (80 MB here, built once outside the hot path) and a widened
  weight-mutation window -- see ``_prep_state``.
* **The normalization pass reads only what exists.** gamma/beta are broadcast
  loads re-read by every one of the (up to 64680) row programs, and at N=1152
  that is not free: 56.3 us with both, 52.2 us with beta alone, 50.2 us with
  neither, against 58.4 us for the L1 kernel. Which loads the kernel contains is
  a compile-time constant resolved from the parameters' actual *values*, so an
  all-ones gamma or an all-zeros beta costs nothing *and* leaves the stored
  result bit-identical to the reference's. At 5.93 TB/s the remaining pass is at
  copy speed -- the minimum for a materialized normalized operand.
* **fc1's GELU rides the GEMM epilogue, but only where that is actually
  cheaper.** cuBLASLt's nvjet family has no GELU epilogue (it has bias, and
  ReLU+bias, but not GELU), so asking for one drops fc1 onto a slower
  ``cutlass3x_sm100_tensorop_*`` kernel; the penalty is close to what the
  separate pass costs, and which side wins depends on whether the intermediate
  fits in L2. See ``_EPI_MIN_M``.
* **Three launches above the threshold, four below, and no Python arithmetic in
  the hot path.** Everything shape-invariant is resolved in ``__init__``, the
  parameter-dependent state is cached behind an identity+version+address guard,
  and the normalization kernel is invoked through the compiled kernel's own C
  launcher rather than Triton's per-call binder (the descent
  ``L1/layer_norm.py`` documents: 12 us of Python per launch down to 3). Host
  issue cost per call: 57 us for the reference, 42 us here on the 3-launch path.
  Both sit under the benchmark's per-iteration GPU time, so this is headroom
  rather than a measured win.

Folding the affine into fc1 -- ``W1'[:, k] = W1[:, k] * g[k % norm_dim]``,
``b1' = b1 + W1 @ tile(beta)`` -- is exact *algebra* but not exact
*arithmetic*, and measured a bad trade; see ITERATIONS.md. It moves beta from
before the bf16 rounding of fc1's input to after it, and the two roundings are
independent, so reference and candidate disagree by ~0.5% of the (1e-2, 1e-2)
tolerance budget: 99.54% of output elements matched against 99.95% without the
fold, and 96.8% at spatial_merge_size=3 where K is 6912. It buys 2 us of a
~920 us call. Not worth spending correctness margin on.

Numerics: the one intended departure from the reference is the epilogue on the
M >= _EPI_MIN_M path, which evaluates GELU on the fp32 accumulator (the
reference rounds fc1's output to bf16 first) in cuBLAS's tanh form rather than
exact erf. That is well inside one bf16 ULP of the intermediate; 99.95% of final
elements land inside the benchmark's (1e-2, 1e-2) bound at checkpoint weight
scale, and the deviation grows with |fc1 out| / atol, so it is the epilogue that
gates the fused path to fp16/bf16.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra.cuda import gdc_wait
from torch._C import _cuda_getCurrentRawStream as _raw_stream
from torch._C import _cuda_getDevice as _cur_device

from ..L1.gelu import GELU
from ..L1.layer_norm import LayerNorm
from .parallel_linear import ColumnParallelLinear, RowParallelLinear

# Widest normalized row the register tile covers; above this the fused path
# defers to the reference (no captured merger comes close -- 1152 or 4608).
_MAX_N = 16384

# The fused path is fp16/bf16 only. The epilogue's GELU differs from the exact
# erf reference by ~2e-4 absolute, which is nothing against a (1e-2, 1e-2)
# tolerance but is 20x fp32's atol of 1e-5, and fc2 then amplifies it by
# ~sqrt(K)*|W2|. No captured merger is fp32. Same split ``L1/gelu.py`` makes
# when it routes fp32 to the accurate intrinsic.
#
# Separately, and not fixed here: fp32 does not reproduce the *baseline* through
# ``_reference`` either (98.4% of elements matched against the required 99%),
# because the frozen L1 LayerNorm's fused fp32 reduction differs from
# ``F.layer_norm``'s by more than fp32's (1e-5, 1e-3) bound survives through two
# 4608-wide GEMMs. That is inherited from the L1 winner -- an identical
# measurement comes out of a merger built straight from the baseline source --
# and every captured merger is bf16.
_EPI_DTYPES = (torch.bfloat16, torch.float16)

_HAS_ADDMM_ACT = hasattr(torch, "_addmm_activation")

# Smallest M for which fc1's GELU is cheaper inside cuBLAS's epilogue than as a
# separate pass.
#
# Asking for the epilogue is not a free win: cuBLASLt's nvjet_sm100 family has a
# fused bias and a fused *ReLU*+bias (``..._bz_relubias_TNT``, measured at
# exactly the plain-bias kernel's speed) but no GELU, so requesting GELU makes
# the heuristic fall back to ``cutlass3x_sm100_tensorop_*`` -- which costs +2.1
# us at M=440, +16.5 us at M=5170 and +30.7 us at M=16170 on fc1 alone. That is
# roughly what a separate GELU pass over the same intermediate costs, so the two
# are within a few percent of each other everywhere, and which one wins is
# decided by L2 residency: below ~M=2500 the M x 4608 intermediate and both
# weight matrices all fit in the 126 MB L2, the separate pass is nearly free, and
# only the slower GEMM shows. Measured with the benchmark itself, one flag apart
# (candidate ms): M=440 49.1 separate / 52.3 fused, M=5170 300.0 / 297.9,
# M=5940 336.9 / 326.7, M=6292 375.8 / 363.5, M=16170 951.2 / 917.0. The
# threshold sits in the wide insensitive band between the two regimes.
_EPI_MIN_M = 4096


@triton.jit
def _merge_norm(X, Z, G, BT, N: tl.constexpr, eps: tl.constexpr,
                B0: tl.constexpr, B1: tl.constexpr, TWO: tl.constexpr,
                MASK1: tl.constexpr, HAS_G: tl.constexpr, HAS_B: tl.constexpr):
    """z = (x - mean) * rstd * gamma + beta, one program per normalized row.

    Structurally ``L1/layer_norm.py``'s fused row kernel -- the row is covered
    by one or two power-of-two tiles summing to exactly N (1152 = 1024 + 128,
    4608 = 4096 + 512) rather than a masked ``next_pow2`` tile that would idle
    43% of its lanes, and is reduced in fp32 by the shifted one-pass formula:
    subtract the row's own first element, then accumulate ``sum(d)`` and
    ``sum(d*d)`` in the same pass so the two reduction trees pipeline instead of
    serializing. Shifting by a real data point keeps ``sq/N - off^2`` cancelling
    only the *shifted* mean, so precision survives ``|mean| >> std``;
    ``maximum(., 0)`` covers a row whose rounded variance lands a hair below
    zero, where eps of exactly 0 would give NaN instead of the reference's inf.

    Two deliberate differences from the L1 kernel, both measured at N=1152:

    * ``HAS_G`` / ``HAS_B`` are resolved from the parameters' *values*, not just
      their presence, so an all-ones gamma or an all-zeros beta compiles the
      broadcast load out entirely. 58.4 us (L1) -> 56.3 us both -> 52.2 us beta
      only -> 50.2 us neither, on the 64680-row shape.
    * The *store* keeps the default eviction policy where the L1 kernel uses
      ``evict_first``. Z is not dead on arrival here: it is fc1's A operand and
      the GEMM re-reads it tile-at-a-time, so demoting it costs 2 us (52.2 us
      -> 50.2 us, i.e. 5.93 TB/s, essentially copy speed). X keeps
      ``evict_first`` -- that one really is never revisited.
    """
    row = tl.program_id(0)
    base = row.to(tl.int64) * N
    c0 = tl.arange(0, B0)
    # This grid always runs downstream of whatever produced X, and a dependent
    # launch costs a fixed ~2 us of otherwise-idle GPU time (see L1/gelu.py,
    # which uses PDL for the same reason). ``launch_pdl`` lets the grid be staged
    # while the producer drains; gdc_wait() before the first load keeps the
    # dependency honest.
    gdc_wait()
    shift = tl.load(X + base).to(tl.float32)
    d0 = tl.load(X + base + c0,
                 eviction_policy="evict_first").to(tl.float32) - shift
    acc = tl.sum(d0, axis=0)
    sq = tl.sum(d0 * d0, axis=0)
    if TWO:
        c1 = B0 + tl.arange(0, B1)
        if MASK1:
            m1 = c1 < N
            # Padding lanes must contribute 0 to both sums, so zero them after
            # the shift rather than loading ``other=0.0``.
            d1 = tl.where(m1, tl.load(X + base + c1, mask=m1,
                                      eviction_policy="evict_first")
                          .to(tl.float32) - shift, 0.0)
        else:
            d1 = tl.load(X + base + c1,
                         eviction_policy="evict_first").to(tl.float32) - shift
        acc += tl.sum(d1, axis=0)
        sq += tl.sum(d1 * d1, axis=0)
    inv_n: tl.constexpr = 1.0 / N
    off = acc * inv_n                       # mean, relative to the shift
    rstd = 1.0 / tl.sqrt(tl.maximum(sq * inv_n - off * off, 0.0) + eps)

    y0 = (d0 - off) * rstd
    if HAS_G:
        y0 = y0 * tl.load(G + c0).to(tl.float32)
    if HAS_B:
        y0 = y0 + tl.load(BT + c0).to(tl.float32)
    tl.store(Z + base + c0, y0.to(Z.dtype.element_ty))
    if TWO:
        y1 = (d1 - off) * rstd
        if MASK1:
            if HAS_G:
                y1 = y1 * tl.load(G + c1, mask=m1).to(tl.float32)
            if HAS_B:
                y1 = y1 + tl.load(BT + c1, mask=m1).to(tl.float32)
            tl.store(Z + base + c1, y1.to(Z.dtype.element_ty), mask=m1)
        else:
            if HAS_G:
                y1 = y1 * tl.load(G + c1).to(tl.float32)
            if HAS_B:
                y1 = y1 + tl.load(BT + c1).to(tl.float32)
            tl.store(Z + base + c1, y1.to(Z.dtype.element_ty))


def _tile_split(n: int):
    """Cover exactly *n* columns with one or two power-of-two tiles.

    ``B0`` is the highest set bit of *n*; the remainder gets a second tile,
    masked only when the remainder is not itself a power of two (never the case
    for any captured width -- 1152 and 4608 both split cleanly).
    """
    b0 = 1 << (n.bit_length() - 1)
    rem = n - b0
    if rem == 0:
        return b0, 1, False, False
    b1 = triton.next_power_of_2(rem)
    return b0, b1, True, b1 != rem


def _num_warps_for(n: int) -> int:
    """Warps per row-program -- the crossovers measured in ``L1/layer_norm.py``.

    One warp holding the whole row gives the most loads in flight per thread and
    the cheapest reduction (a single intra-warp shuffle tree, no shared memory,
    no barrier), and wins outright up to ~2048 columns; past that the row stops
    fitting in one warp's registers. Re-measured for this kernel at N=1152 over
    {1, 2, 4} warps and {1, 2, 4, 8} rows per program: 1 warp, 1 row is the best
    cell of the grid at every captured row count, and packing rows into a
    program to amortize the gamma/beta broadcast loses badly (56.3 us -> 76.8 us
    at 4 rows on the 64680-row shape) -- the 2-D tile costs more in per-lane
    access width than the broadcast ever cost.
    """
    if n <= 2048:
        return 1
    if n <= 3072:
        return 2
    if n <= 8192:
        return 4
    return 8


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

        # ---- shape-invariant launch state -----------------------------------
        # No parameter is *read* here: the harness (and any real weight loader)
        # fills them after construction, so everything value-dependent is
        # deferred to _prepare.
        self._nd = norm_dim
        # With the norm *after* the spatial reshape the reduction runs over the
        # flat buffer in hidden_size-wide rows, so x's logical last dim is
        # irrelevant as long as it is contiguous and its numel divides
        # hidden_size -- which is what the Qwen3 DeepStack mergers actually pass
        # (use_postshuffle_norm=True with x:[R, 1, context_dim]). With the norm
        # *before* the reshape the last dim has to be exactly context_dim, or
        # the reference would be normalizing over different groups.
        self._last_free = use_postshuffle_norm
        if 0 < norm_dim <= _MAX_N:
            self._tile = (norm_dim, eps) + _tile_split(norm_dim)
            self._num_warps = _num_warps_for(norm_dim)
        else:
            self._tile = ()           # fused path disabled
            self._num_warps = 0

        # ---- parameter-dependent state, built on the first forward ----------
        self._prep_key: tuple | None = None
        self._ready = False
        self._cargs: tuple = ()       # _merge_norm's trailing constexprs
        self._w1n: torch.Tensor | None = None   # fc1.weight.T, materialized
        self._w2n: torch.Tensor | None = None   # fc2.weight.T, materialized
        self._b1: torch.Tensor | None = None
        self._b2: torch.Tensor | None = None
        self._g: torch.Tensor | None = None
        self._bt: torch.Tensor | None = None

        # ---- direct-launch cache for _merge_norm ----------------------------
        self._ldtype: torch.dtype | None = None
        self._ldev = -1
        self._lrun = None
        self._lpre: tuple = ()
        self._lpost: tuple = ()

        # Weights arriving through ``load_state_dict`` land in the *existing*
        # Parameter, so belt-and-braces: drop the cache from the load hook as
        # well as from the guard in _prepare.
        hook = getattr(self, "register_load_state_dict_post_hook", None)
        if hook is not None:
            hook(_invalidate_prep)

    # ------------------------------------------------------------------
    # parameter-dependent setup
    # ------------------------------------------------------------------
    def _prep_state(self):
        """Fingerprint of every parameter ``_prepare`` inspects.

        Three components per tensor, because each catches a different way new
        weights arrive: ``id`` catches a *replaced* Parameter, ``_version``
        catches an in-place ``copy_`` (which is what ``load_state_dict`` does,
        verified to bump the counter), and ``data_ptr`` catches ``p.data = ...``
        rebinding the storage underneath a Parameter that keeps its identity --
        which is exactly what the harness's dtype cast does. Ten attribute
        loads, ~0.5 us; the ``load_state_dict`` post-hook covers the same ground
        from the other side. The one hole left is a bare ``p.data.copy_()``
        *after* a forward has already run -- ``L1/layer_norm.py`` documents the
        same window for its own affine cache, and weight loading completes
        before the first forward in every path we have.

        That window is **wider than it was before the K-major weight copies**,
        and deliberately so. ``.data.copy_()`` bumps neither ``p._version`` nor
        ``p.data._version`` (verified), so it is undetectable without reading the
        weight -- which would mean a device sync on every call. When the GEMMs
        ran off ``w.detach()``, an alias sharing the Parameter's storage, such a
        write was picked up anyway; a materialized transpose is a snapshot, so it
        is not. This is the same trade every engine that pre-packs weights makes
        (and ``ColumnParallelLinear``/``RowParallelLinear`` come from one).
        Everything that actually loads weights is covered: ``load_state_dict``
        assigns through the Parameter (bumps ``_version``) *and* fires the
        post-hook, ``p.data = ...`` and the harness's dtype cast move
        ``data_ptr``, and a replaced Parameter changes ``id``. Verified: a
        candidate that has run a forward and then reloads weights produces output
        bit-identical to a freshly built one.
        """
        out = []
        for t in (self.norm.weight, self.norm.bias, self.fc1.weight,
                  self.fc1.bias, self.fc2.weight, self.fc2.bias):
            if t is None:
                out.append(None)
            else:
                out.append((id(t), t._version, t.data_ptr()))
        return tuple(out)

    def _prepare(self, dtype: torch.dtype) -> bool:
        """Resolve the launch constants that depend on parameter values.

        Everything here is loop-invariant across calls: which affine loads the
        normalization kernel needs, the transposed fc1 weight cuBLAS wants, and
        detached handles for the two GEMMs. Returns False when the fused path
        does not apply, in which case ``forward`` runs the reference.

        ``detach()`` rather than the Parameter: the compiled kernel is
        specialized on gamma/beta's *dtype*, and ``p.data = p.data.to(...)``
        re-points a Parameter without changing its identity. A detached alias
        shares storage -- so ordinary in-place weight updates are still picked
        up -- but pins the dtype the kernel was compiled for.
        """
        key = self._prep_state()
        if key == self._prep_key:
            return self._ready
        self._prep_key = key
        self._ready = False
        self._ldtype = None               # any change re-specializes the launch
        if not _HAS_ADDMM_ACT or not self._tile:
            return False
        fc1, fc2 = self.fc1, self.fc2
        w1, b1, w2, b2 = fc1.weight, fc1.bias, fc2.weight, fc2.bias
        g, bt = self.norm.weight, self.norm.bias
        if (fc1.use_fp8 or fc2.use_fp8 or fc2.tp_size > 1
                or b1 is None or b2 is None
                or w1.ndim != 2 or w2.ndim != 2 or not w1.is_cuda
                or w1.dtype is not dtype or b1.dtype is not dtype
                or w2.dtype is not dtype or b2.dtype is not dtype
                or w1.shape[1] != self.hidden_size
                or w2.shape[1] != self.hidden_size
                or not w1.is_contiguous() or not w2.is_contiguous()
                or not b1.is_contiguous() or not b2.is_contiguous()):
            return False
        for t in (g, bt):
            if t is not None and (t.numel() != self._nd or t.dtype is not dtype
                                  or not t.is_contiguous()):
                return False
        # Value-dependent, and the whole point of the specialization: an
        # all-ones gamma / all-zeros beta is the identity, so compile the
        # broadcast load out instead of reading it once per row program. Exact,
        # not approximate -- ``y * 1.0`` and ``y + 0.0`` are bit-preserving in
        # fp32, so Z stays bit-identical to the reference's.
        has_g = g is not None and not bool(torch.all(g == 1))
        has_b = bt is not None and bool(torch.any(bt != 0))
        self._cargs = self._tile + (has_g, has_b)
        # Both GEMMs get a *materialized* K-major copy of their weight rather
        # than the ``.t()`` view ``F.linear`` hands cuBLAS; see the module
        # docstring for the measurements. Nothing keeps a plain alias of the
        # weight: the fallback path goes through ``self.fc1``/``self.fc2``, so
        # these two copies are the only weights the fused path reads.
        self._w1n = w1.detach().t().contiguous()
        self._w2n = w2.detach().t().contiguous()
        self._b1 = b1.detach()
        self._b2 = b2.detach()
        # Unused pointers still have to be *passed*, and Triton specializes on
        # pointer alignment, so a placeholder has to be as aligned as the real
        # thing. ``None`` is not an option (the kernel signature is fixed), and
        # a freshly allocated parameter is 256-byte aligned either way.
        self._g = g.detach() if g is not None else None
        self._bt = bt.detach() if bt is not None else None
        self._ready = True
        return True

    # ------------------------------------------------------------------
    # normalization launch
    # ------------------------------------------------------------------
    def _norm_setup(self, x, z, g, bt, rows: int) -> None:
        """First fused call for this dtype: compile, then memoize the compiled
        kernel's own C launcher.

        ``kernel[grid](...)`` re-binds and re-specializes every argument, hashes
        them and rebuilds the launch metadata on each call -- ~12 us of Python,
        which at the 1760-row shape is a quarter of the whole operator. Every
        argument here is constexpr except the four pointers, so the only
        per-call quantity Triton's binder would derive is pointer *alignment*;
        hence the ``& 15`` guard in ``forward`` and the refusal to memoize off a
        misaligned first call. See ``L1/layer_norm.py:_launch_setup`` for the
        same descent and its measurements (12 us -> 4 us -> 3.1 us).
        """
        kern = _merge_norm[(rows,)](x, z, g, bt, *self._cargs,
                                    num_warps=self._num_warps, launch_pdl=True)
        if (x.data_ptr() & 15 or z.data_ptr() & 15
                or g.data_ptr() & 15 or bt.data_ptr() & 15):
            return
        # ``CudaLauncher`` is Triton-internal, so every piece is fetched
        # defensively: if a future Triton reshapes it, or the kernel turns out
        # to need scratch (ours never does -- no in-kernel allocation, no
        # profiling), we simply never memoize and every call keeps going through
        # the supported ``kernel[grid](...)`` path. Slower, still right.
        launcher = None if kern is None else kern.run
        raw_launch = getattr(launcher, "launch", None)
        if (raw_launch is not None
                and x.get_device() == _cur_device()
                and getattr(launcher, "global_scratch_size", None) == 0
                and getattr(launcher, "profile_scratch_size", None) == 0):
            self._lrun = raw_launch
            self._lpre = (
                kern.function,
                launcher.launch_cooperative_grid, launcher.launch_pdl,
                None, None,                      # global / profile scratch
                kern.packed_metadata,
                None, None, None,                # launch metadata, 2 hooks
            )
            self._lpost = (g.data_ptr(), bt.data_ptr()) + self._cargs
            self._ldev = x.get_device()
            self._ldtype = x.dtype

    # ------------------------------------------------------------------
    # the unfused reference path -- anything the fused path does not cover
    # ------------------------------------------------------------------
    def _reference(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_postshuffle_norm:
            x = self.norm(x.view(-1, self.hidden_size))
        else:
            x = self.norm(x).view(-1, self.hidden_size)
        return self.fc2(self.act(self.fc1(x)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.hidden_size
        total = x.numel()
        if (self._num_warps
                and x.dtype in _EPI_DTYPES
                and x.is_cuda
                and x.is_contiguous()
                and (self._last_free or x.shape[-1] == self._nd)
                and total % h == 0
                and total
                and not torch.is_grad_enabled()
                and self._prepare(x.dtype)):
            # The normalized buffer is the same flat buffer in both modes: with
            # the norm before the spatial reshape it is R rows of context_dim
            # laid end to end, which *is* M rows of hidden_size. So the reshape
            # is free and only the reduction width differs.
            nd = self._nd
            z = torch.empty(total // h, h, dtype=x.dtype, device=x.device)
            g = self._g if self._g is not None else z
            bt = self._bt if self._bt is not None else z
            rows = total // nd
            xp, zp = x.data_ptr(), z.data_ptr()
            if (x.dtype is self._ldtype
                    and x.get_device() == self._ldev
                    and _cur_device() == self._ldev
                    and not (xp & 15) and not (zp & 15)):
                self._lrun(rows, 1, 1, _raw_stream(self._ldev),
                           *self._lpre, xp, zp, *self._lpost)
            else:
                self._norm_setup(x, z, g, bt, rows)
            if z.shape[0] >= _EPI_MIN_M:
                # fc1 + bias + GELU in one cuBLAS epilogue, then fc2 + bias.
                return torch.addmm(
                    self._b2,
                    torch._addmm_activation(self._b1, z, self._w1n,
                                            use_gelu=True),
                    self._w2n)
            # Separate GELU pass, keeping cuBLAS on its faster non-epilogue
            # kernel for fc1. ``self.act`` is the frozen L1 winner, tile-order
            # heuristic and all.
            return torch.addmm(
                self._b2, self.act(torch.addmm(self._b1, z, self._w1n)),
                self._w2n)
        return self._reference(x)


def _invalidate_prep(module, incompatible_keys):
    """``load_state_dict`` post-hook: new weights, so re-derive the launch
    constants (which affine loads exist, the transposed fc1 view) on the next
    forward."""
    module._prep_key = None
    module._ready = False
