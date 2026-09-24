"""Standard LayerNorm wrapping F.layer_norm with optional affine parameters.

Supports create_scale and create_offset flags matching the reference
openfold3/core/model/primitives/normalization.py LayerNorm.

The fast path is one fused row-wise Triton kernel, which removes three
different costs depending on which captured shape you look at:

* **The fp32 cast sandwich** (``promote_fp32=True``, the narrow shapes). The
  reference path launches THREE kernels -- ``x.float()`` materializing a full
  fp32 copy, ``F.layer_norm`` over it, then ``.to(orig_dtype)`` -- and moves
  ~5x the minimum bytes purely to obtain an fp32 reduction. The kernel loads
  the low-precision row once, promotes in registers, reduces in fp32, applies
  the affine in fp32 and stores in the original dtype: one launch, one
  traversal, fp32 promotion for free.
* **ATen's own layer_norm** (``promote_fp32=False``, the 4608-wide shapes,
  where there is no sandwich to remove). ``native_layer_norm`` re-reads the row
  for the normalization pass; measured on B200 it runs at 2.5x the time of a
  plain read+write of the same bytes, where this kernel -- single load, row
  resident in registers -- runs at 1.11x, i.e. 5.9 TB/s of the 6.5 TB/s a
  ``torch.add`` achieves on the same tensor.
* **Per-call host cost.** At the captured sizes (a single 256-element row at
  the small end) Python dispatch dominates the kernel, so the launch config is
  resolved in ``__init__``, the fp32 affine is cached, and after the first call
  the compiled kernel is invoked through its own C launcher rather than
  Triton's per-call binder/specializer: 12 us of Python per call down to 3 us.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from torch._C import _cuda_getCurrentRawStream as _raw_stream
from torch._C import _cuda_getDevice as _cur_device

# Widest row we keep entirely in registers. Above this the register tile stops
# fitting and the row goes back to F.layer_norm.
_MAX_N = 16384

# dtypes the fused kernel handles (load -> fp32 in registers -> store back).
_FAST_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


@triton.jit
def _layer_norm_fwd(
    X, Y, W, B,
    N: tl.constexpr, eps: tl.constexpr,
    B0: tl.constexpr, B1: tl.constexpr, TWO: tl.constexpr, MASK1: tl.constexpr,
    HAS_W: tl.constexpr, HAS_B: tl.constexpr,
):
    """One program per normalized row; the whole row lives in registers.

    The row is covered by one or two **power-of-two tiles that sum to exactly
    N** (4608 = 4096 + 512, 384 = 256 + 128, ...) rather than a single masked
    ``next_pow2(N)`` tile. A masked 8192-lane tile over a 4608-wide row idles
    43% of its lanes and needs 4x the warps to hold the same rows in flight;
    measured on B200, the exact split is 55.6 us against 83.0 us at
    16170x4608.

    The row is loaded once and reduced in fp32 by the **shifted one-pass**
    formula: subtract the row's own first element ``c``, then accumulate
    ``sum(x-c)`` and ``sum((x-c)^2)`` in the same pass. Those two reduction
    trees are independent, so they pipeline instead of serializing the way a
    literal two-pass ``mean``-then-``sum((x-mean)^2)`` does -- worth 24% of the
    device time on a single narrow row (1.13 us -> 0.86 us at N=256) and 5% at
    16170x4608, where it also drops the register count from 255 to 227.

    Shifting by a real data point is what makes this safe: ``x-c`` is on the
    scale of the row's spread, so the ``sq/N - off^2`` subtraction cancels only
    the shifted mean, not the (potentially huge) raw mean -- unlike the naive
    ``E[x^2] - E[x]^2``, which loses all precision when ``|mean| >> std``. The
    ``maximum(., 0)`` covers the one remaining case: a row whose rounded
    variance lands a hair below zero, where an eps of exactly 0 would otherwise
    produce NaN instead of the reference's inf.

    ``evict_first`` on both the input row and the output row: neither is ever
    revisited, and demoting them leaves L2 to the weight/bias broadcast that
    every one of the (up to 16170) programs re-reads. Do not also set
    ``cache_modifier=".cg"`` on these loads -- ptxas fails on that combination
    with this toolchain, and it measured no faster on its own.
    """
    row = tl.program_id(0)
    base = row.to(tl.int64) * N
    c0 = tl.arange(0, B0)
    shift = tl.load(X + base).to(tl.float32)
    d0 = tl.load(X + base + c0,
                 eviction_policy="evict_first").to(tl.float32) - shift
    acc = tl.sum(d0, axis=0)
    sq = tl.sum(d0 * d0, axis=0)
    if TWO:
        c1 = B0 + tl.arange(0, B1)
        if MASK1:
            m1 = c1 < N
            # The padding lanes must contribute 0 to both sums, so zero them
            # after the shift rather than loading ``other=0.0``.
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
    var = sq * inv_n - off * off
    rstd = 1.0 / tl.sqrt(tl.maximum(var, 0.0) + eps)

    y0 = (d0 - off) * rstd
    if HAS_W:
        y0 = y0 * tl.load(W + c0).to(tl.float32)
    if HAS_B:
        y0 = y0 + tl.load(B + c0).to(tl.float32)
    tl.store(Y + base + c0, y0.to(Y.dtype.element_ty),
             eviction_policy="evict_first")
    if TWO:
        y1 = (d1 - off) * rstd
        if MASK1:
            if HAS_W:
                y1 = y1 * tl.load(W + c1, mask=m1).to(tl.float32)
            if HAS_B:
                y1 = y1 + tl.load(B + c1, mask=m1).to(tl.float32)
            tl.store(Y + base + c1, y1.to(Y.dtype.element_ty), mask=m1,
                     eviction_policy="evict_first")
        else:
            if HAS_W:
                y1 = y1 * tl.load(W + c1).to(tl.float32)
            if HAS_B:
                y1 = y1 + tl.load(B + c1).to(tl.float32)
            tl.store(Y + base + c1, y1.to(Y.dtype.element_ty),
                     eviction_policy="evict_first")


def _tile_split(n: int):
    """Cover exactly *n* columns with one or two power-of-two tiles.

    ``B0`` is the highest set bit of *n*; the remainder gets a second tile,
    masked only when the remainder is not itself a power of two (never the case
    for any captured width).
    """
    b0 = 1 << (n.bit_length() - 1)
    rem = n - b0
    if rem == 0:
        return b0, 1, False, False
    b1 = triton.next_power_of_2(rem)
    return b0, b1, True, b1 != rem


def _num_warps_for(n: int) -> int:
    """Warps per row-program.

    Far fewer warps than the textbook choice. One warp holding the whole row
    gives the most loads in flight per thread and the cheapest reduction (a
    single intra-warp shuffle tree -- no shared memory, no barrier), and it wins
    outright up to ~2048 columns: measured on B200 at 40 MB per width,
    N=1024 is 27.1 us at 1 warp against 59.2 us at 8. Past that the row stops
    fitting in one warp's registers and the widths have to be spread out --
    N=8192 spills at 1 warp (73 us) and wants 4 (30 us); N=16384 wants 8.
    Thresholds measured, not derived; the crossovers are sharp.
    """
    if n <= 2048:
        return 1
    if n <= 3072:
        return 2
    if n <= 8192:
        return 4
    return 8


class LayerNorm(nn.Module):
    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        create_scale: bool = True,
        create_offset: bool = True,
        promote_fp32: bool = True,
    ):
        super().__init__()
        self.normalized_shape = (normalized_shape,)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.promote_fp32 = promote_fp32

        if elementwise_affine and create_scale:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
        else:
            self.register_parameter("weight", None)

        if elementwise_affine and create_offset:
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter("bias", None)

        # fp32 views of weight/bias, filled on the first forward (see there).
        # A separate flag rather than a None check on _w32, so a module with no
        # affine params does not retry the cast every call.
        self._cast_done = False
        self._src_w: torch.Tensor | None = None
        self._src_b: torch.Tensor | None = None
        self._w32: torch.Tensor | None = None
        self._b32: torch.Tensor | None = None

        # Everything the fused launch needs that does not depend on the input,
        # resolved here so the hot forward does no arithmetic. ``_n = 0``
        # disables the fast path (row too wide for the register tile).
        n = int(normalized_shape)
        if 0 < n <= _MAX_N:
            self._n = n
            b0, b1, two, mask1 = _tile_split(n)
            self._num_warps = _num_warps_for(n)
            # Trailing constexpr arguments of _layer_norm_fwd, in signature
            # order; HAS_W / HAS_B are appended once the affine is resolved.
            self._tile_args = (n, eps, b0, b1, two, mask1)
        else:
            self._n = 0
            self._tile_args = ()

        # Direct-launch cache, keyed on the input dtype (the only thing left
        # that can change the compiled kernel). See _launch_setup.
        self._ldtype: torch.dtype | None = None
        self._lrun = None
        self._lpre: tuple = ()
        self._lpost: tuple = ()
        self._ldev = -1
        self._cargs: tuple = ()
        # The affine snapshots the kernel reads, plus their addresses. Triton's
        # launcher accepts a raw int for a pointer argument, which skips a
        # ``data_ptr()`` round trip through Python per tensor per call; the
        # tensors are kept referenced here so the storage stays alive and so an
        # in-place weight update is still picked up (same storage, same
        # address).
        self._kw: torch.Tensor | None = None
        self._kb: torch.Tensor | None = None
        self._kwp: int | None = None
        self._kbp: int | None = None
        # The parameter dict itself, so the per-call "has weight/bias been
        # replaced?" guard is two dict lookups instead of two trips through
        # nn.Module.__getattr__ (which is a Python-level function and costs
        # more than the guard it feeds).
        self._pdict = self._parameters

    # ------------------------------------------------------------------
    # weight / bias preparation
    # ------------------------------------------------------------------
    def _fp32_affine(self):
        """fp32 copies of weight/bias, cached across calls.

        Cast weight/bias to fp32 ONCE, not per call. These are parameters, so
        the cast is loop-invariant, but re-running it cost ~50 kernel launches
        per decode step across the 21 indexer compute layers -- the same defect
        as the indexer rope re-casting its cos/sin cache every call. Weight
        loading completes before the first forward, so a lazy cache is safe.
        Re-derive if the parameter object was replaced or moved. ``_w32`` is a
        plain attribute, not a buffer, so ``module.to(device)`` would not move
        it -- 72 modules across the tree use this op, and a stale cache there
        would be a device mismatch (or worse, silently old weights). The guard
        is two identity compares, ~100 ns against the ~2 us kernel launch it
        saves.
        """
        if (not self._cast_done
                or self._src_w is not self.weight
                or self._src_b is not self.bias):
            w, b = self.weight, self.bias
            self._src_w, self._src_b = w, b
            self._w32 = (w.float()
                         if w is not None and w.dtype != torch.float32 else w)
            self._b32 = (b.float()
                         if b is not None and b.dtype != torch.float32 else b)
            self._cast_done = True
            self._ldtype = None  # invalidate the launch cache
        return self._w32, self._b32

    def _kernel_affine(self):
        """weight/bias in the form the fused kernel wants.

        With both the reduction and the affine already in fp32 inside the
        kernel, ``promote_fp32`` no longer changes the arithmetic:
        ``w.float()`` of a bf16 parameter is exact, so either setting produces
        the same numbers. We still route through the fp32 cache when
        ``promote_fp32`` is set (so an fp32 master weight stays fp32) and hand
        the parameter through otherwise -- matching what ``F.layer_norm``
        would have been given in each case.

        ``detach()`` rather than the parameter itself: the compiled kernel is
        specialized on the weight's *dtype*, and ``p.data = p.data.to(...)``
        re-points a Parameter without changing its identity, which the identity
        guard cannot see. A detached alias shares storage -- so ordinary
        in-place weight updates are still picked up -- but pins the dtype the
        kernel was compiled for, turning a silent-garbage failure into the same
        stale-value window ``_w32`` already documents.
        """
        if self.promote_fp32:
            w, b = self._fp32_affine()
        else:
            w, b = self.weight, self.bias
        return (None if w is None else w.detach(),
                None if b is None else b.detach())

    # ------------------------------------------------------------------
    # the unfused reference path -- anything the kernel does not cover
    # ------------------------------------------------------------------
    def _reference(self, x: torch.Tensor) -> torch.Tensor:
        if not self.promote_fp32:
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps,
            )
        # Promote to fp32 for the reduction to match vLLM's
        # ``vllm/model_executor/layers/layernorm.py:LayerNorm`` which keeps
        # ``weight`` / ``bias`` in fp32 and runs the reduction in fp32.
        # Matters for the DeepSeek-V3.2 indexer ``k_norm`` — running the
        # reduction in bf16 biases the variance enough to shift the
        # FP8-quantized indexer K cache, which in turn changes the top-2048
        # selection in every sparse layer.
        orig_dtype = x.dtype
        weight, bias = self._fp32_affine()
        return F.layer_norm(
            x.float(), self.normalized_shape, weight, bias, self.eps,
        ).to(orig_dtype)

    # ------------------------------------------------------------------
    # fused path
    # ------------------------------------------------------------------
    def _launch_setup(self, x: torch.Tensor) -> torch.Tensor:
        """First fused call for this input dtype: compile, then memoize the
        compiled kernel's own launcher.

        Triton's ``kernel[grid](...)`` re-binds and re-specializes every
        argument, hashes them and rebuilds the launch metadata on each call --
        ~12 us of Python here, more than the whole rest of the operator at the
        captured sizes. Everything the binder derives is invariant for this
        module (every argument is constexpr except four pointers), except for
        Triton's *pointer-alignment* specialization -- hence the ``& 15`` guard
        in ``forward`` before the cached launcher is used, and the refusal to
        memoize off a misaligned first call.

        So we descend two levels and keep the pieces:

        * ``CompiledKernel.run`` is the ``CudaLauncher``, whose ``__call__``
          defines a closure and makes two scratch-allocation calls per launch;
          with both scratch sizes 0 that is pure overhead, so we hold its
          ``.launch`` (the generated C entry point) and pass the arguments
          ``__call__`` would have inserted.
        * The invariant prefix (function handle, cooperative/PDL flags, the
          two scratch slots, packed metadata, the three hook slots) and the
          invariant suffix (weight/bias addresses plus every constexpr) are
          pre-built as tuples, so a launch is two star-unpacks rather than a
          dozen attribute loads.

        Measured at N=256: 12.0 us for ``kernel[grid](...)``, 4.0 us through
        ``CompiledKernel.run``, 3.1 us here.
        """
        src_w, src_b = self.weight, self.bias
        if not self.promote_fp32 and (
                (src_w is not None and src_w.dtype != x.dtype)
                or (src_b is not None and src_b.dtype != x.dtype)):
            # F.layer_norm itself rejects a low-precision input with a
            # differently-typed affine when there is no fp32 promotion; go
            # through the reference so the caller sees the same error rather
            # than a silently-more-permissive fused result.
            return self._reference(x)
        w, b = self._kernel_affine()
        cargs = self._tile_args + (w is not None, b is not None)
        y = torch.empty_like(x)
        kern = _layer_norm_fwd[(x.numel() // self._n,)](
            x, y, w, b, *cargs, num_warps=self._num_warps,
        )
        aligned = not (x.data_ptr() & 15) and not (y.data_ptr() & 15)
        if w is not None:
            aligned = aligned and not (w.data_ptr() & 15)
        if b is not None:
            aligned = aligned and not (b.data_ptr() & 15)
        # ``CudaLauncher`` is Triton-internal, so every piece we reach for is
        # fetched defensively: if a future Triton reshapes it, or the kernel
        # turns out to need scratch (ours never does -- no in-kernel allocation
        # and no profiling), we simply never memoize and every call keeps going
        # through the supported ``kernel[grid](...)`` path. Slower, still right.
        launcher = None if kern is None else kern.run
        raw_launch = getattr(launcher, "launch", None)
        if (aligned and raw_launch is not None
                and x.get_device() == _cur_device()
                and getattr(launcher, "global_scratch_size", None) == 0
                and getattr(launcher, "profile_scratch_size", None) == 0):
            self._kw, self._kb = w, b
            self._kwp = None if w is None else w.data_ptr()
            self._kbp = None if b is None else b.data_ptr()
            self._src_w, self._src_b = src_w, src_b
            self._cargs = cargs
            self._lrun = raw_launch
            self._lpre = (
                kern.function,
                launcher.launch_cooperative_grid, launcher.launch_pdl,
                None, None,                      # global / profile scratch
                kern.packed_metadata,
                None, None, None,                # launch metadata, 2 hooks
            )
            self._lpost = (self._kwp, self._kbp) + cargs
            self._ldev = x.get_device()
            self._ldtype = x.dtype
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = self._n
        dev = self._ldev
        p = self._pdict
        if (x.dtype is self._ldtype
                and p["weight"] is self._src_w
                and p["bias"] is self._src_b
                and x.ndim
                and x.is_contiguous()
                and x.shape[-1] == n
                and x.get_device() == dev
                and _cur_device() == dev
                and not torch.is_grad_enabled()):
            xp = x.data_ptr()
            if not (xp & 15):  # the alignment the compiled kernel assumes
                y = torch.empty_like(x)
                self._lrun(x.numel() // n, 1, 1, _raw_stream(dev),
                           *self._lpre, xp, y.data_ptr(), *self._lpost)
                return y
        if (n
                and x.dtype in _FAST_DTYPES
                and x.is_cuda
                and x.ndim
                and x.is_contiguous()
                and x.shape[-1] == n
                and not torch.is_grad_enabled()):
            return self._launch_setup(x)
        return self._reference(x)
