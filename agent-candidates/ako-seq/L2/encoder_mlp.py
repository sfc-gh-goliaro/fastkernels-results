"""Feed-forward blocks for encoder models.

Both modules are a cuBLAS GEMM followed by a cheap tail, and on B200 the tail is
where the money is.  The fp16 GEMMs land on Blackwell-native ``nvjet_sm100_*``
kernels with a fused bias epilogue -- one kernel, 2-CTA clusters, tile shapes
Triton cannot express -- so the matmul is left to cuBLAS (L1 ``Linear``'s
``_MM_CFG`` gate defers every 2-D fp16 shape for exactly this reason, and the
fast paths below reproduce that decision directly).  What is removed here is the
*passes around it*, and the per-call cost of issuing them.

Three costs, all measured under the benchmark's own timing loop (see
ITERATIONS.md for the tables):

* **Kernel count.**  A trivial 1-element kernel costs 2.06 us of event-to-event
  time here (16 of them cost 33 us).  At M=64 both modules move well under 1 MB,
  so launches -- not bandwidth -- are the whole cost.
* **Bandwidth**, which only bites at M=2048.
* **Per-call Python.**  The harness's 253 MB L2-flush kernel gives the GPU 68 us
  of work to chew on before the measured window, which hides *some* host time --
  but not as much as a bare-op probe suggests: injecting 15 us of extra CPU into
  ``EncoderOutput.forward`` costs 3.9 us of measured time at M=64 and 1.0 us at
  M=512.  So the launch path is worth shortening, and both modules issue their
  Triton kernel through a memoized C launcher rather than
  ``kernel[grid](...)``.

``EncoderOutput`` is restructured:

    baseline:  gemm(+bias) -> add -> layer_norm            (3 kernels, 4 at M=64)
    this file: gemm        -> fused add+bias+layer_norm     (2 kernels, 3 at M=64)

with four separate wins folded into that one epilogue kernel:

1. **One pass instead of three.**  The baseline materializes ``dense(h) +
   input_tensor`` and then re-reads it.  The fused kernel loads the GEMM output
   and the residual, adds them, normalizes in registers and stores once, *over
   the GEMM's own output buffer*: 20 MB of traffic down to 12 MB at M=2048, one
   launch removed at every M, and one allocation removed from the host path.
2. **The dense bias moves out of the GEMM.**  At M=64 the second GEMM is
   split-K (K=4096 against 64 rows), so cuBLAS runs a separate
   ``splitKreduce_kernel``, and asking it to also apply the bias costs a
   measured 1.95 us (13.4 -> 15.4 us for the GEMM pair alone).  The epilogue is
   already reading a full 1024-wide row, so it applies the bias for free.
3. **The weight is kept [K, N] row-major.**  ``F.linear``'s ``w.t()`` gives
   cuBLAS the "TNT" nvjet variant; a ``[K, N]`` contiguous weight selects "NNT",
   which measures ~1.95 us faster for K=4096 -> N=1024.  Verified against
   layout- *and* buffer-swapped controls (four independent 8 MB weight buffers,
   both orders), and again end to end.  The same probe found no difference for
   K=1024 -> N=4096, so the transpose is gated on ``K > N``: it costs a cached
   8 MB copy and only pays where the weight read dominates a small-M GEMM.
4. **PDL.**  ``launch_pdl`` + ``gdc_wait()`` stages the epilogue's grid while the
   GEMM drains; worth 1.9-2.0 us at M=512 and M=2048.

``EncoderIntermediate`` stays two kernels, because the one-kernel version loses:
folding GELU into the GEMM via ``torch._addmm_activation(..., use_gelu=True)``
drops off nvjet onto ``cutlass3x_sm100_tensorop_*_gelu``, 1.24 us slower at M=64
and 3.43 us slower at M=2048 -- more than the launch and the 16 MB round trip it
saves.  nvjet has a *ReLU* epilogue at zero cost (``_relubias_TNT``) but no GELU,
which is why cuBLAS falls back.  So the GELU pass stays, in place over the GEMM's
output, with its launch shape retuned for these three tensors: L1's ``GELU``
picks ``BLOCK=2048, 4 warps`` above its 0.5 Mi-element threshold and at M=512
that costs 2.0 us against ``BLOCK=2048, 8 warps``.  The GELU math is L1's -- one
``tanh.approx.f32`` -- and ``GELU`` remains the fallback for everything the
retuned launch does not cover.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._C import _cuda_getCurrentRawStream as _raw_stream
from torch._C import _cuda_getDevice as _cur_device
from triton.language.extra.cuda import gdc_wait

from ..L1.gelu import GELU
from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear

# Widest row the fused epilogue keeps entirely in registers; above this it falls
# back to the unfused L1 path (the same bound L1 layer_norm uses).
_MAX_N = 16384
_LOWP = (torch.float16, torch.bfloat16)
_FAST_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


# ---------------------------------------------------------------------------
# Memoized Triton launch
# ---------------------------------------------------------------------------
class _CachedLaunch:
    """The compiled kernel's own C entry point, plus its invariant arguments.

    Triton's ``kernel[grid](...)`` re-binds and re-specializes every argument,
    hashes them and rebuilds the launch metadata on each call -- ~12 us of
    Python, and at M=64 that is *not* free (see the module docstring).  So we
    descend two levels and keep the pieces, as L1 layer_norm does:

    * ``CompiledKernel.run`` is the ``CudaLauncher``, whose ``__call__`` defines
      a closure and makes two scratch-allocation calls per launch; with both
      scratch sizes 0 that is pure overhead, so we hold its ``.launch`` (the
      generated C entry point) and pass the arguments ``__call__`` would have
      inserted itself.
    * The invariant prefix (function handle, cooperative/PDL flags, the two
      scratch slots, packed metadata, three hook slots) is pre-built as a tuple,
      so a launch is one star-unpack instead of a dozen attribute loads.

    Everything is fetched defensively: if a future Triton reshapes
    ``CudaLauncher``, or a kernel turns out to need scratch (ours never do -- no
    in-kernel allocation, no profiling), ``ready`` stays False and every call
    keeps going through the supported ``kernel[grid](...)`` path.  Slower, still
    right.
    """

    __slots__ = ("ready", "run", "pre")

    def __init__(self):
        self.ready = False
        self.run = None
        self.pre = ()

    def bind(self, kern) -> bool:
        launcher = None if kern is None else getattr(kern, "run", None)
        raw = getattr(launcher, "launch", None)
        if (raw is None
                or getattr(launcher, "global_scratch_size", None) != 0
                or getattr(launcher, "profile_scratch_size", None) != 0):
            return False
        self.run = raw
        self.pre = (
            kern.function,
            launcher.launch_cooperative_grid, launcher.launch_pdl,
            None, None,                      # global / profile scratch
            kern.packed_metadata,
            None, None, None,                # launch metadata, 2 hooks
        )
        self.ready = True
        return True


def _tile_split(n: int):
    """Cover exactly *n* columns with one or two power-of-two tiles.

    From L1 layer_norm: a masked ``next_pow2(n)`` tile idles up to half its
    lanes, while ``4608 = 4096 + 512`` covers the row exactly.  ``B0`` is the
    highest set bit; the remainder gets a second tile, masked only when it is not
    itself a power of two.  ``n = 1024`` (every captured shape here) takes the
    single-tile branch.
    """
    b0 = 1 << (n.bit_length() - 1)
    rem = n - b0
    if rem == 0:
        return b0, 1, False, False
    b1 = triton.next_power_of_2(rem)
    return b0, b1, True, b1 != rem


def _num_warps_for(n: int) -> int:
    """Warps per row-program.

    Four times L1 layer_norm's count, and measured rather than inherited: L1
    gives one warp per row up to n=2048, but this kernel loads *two* rows plus
    three broadcast vectors, so a single warp cannot keep enough of the row in
    flight.  At n=1024 (every captured shape) 4 warps is the best choice at all
    three M -- 1 and 2 warps cost a full 2 us at M=64 and M=512, 8 costs 1 us at
    M=2048, 16 costs 5 us.

    Above 2048 the counts are extrapolated on register pressure alone (no
    captured shape reaches them), keeping ~256 fp32 lanes of row per warp.
    """
    if n <= 2048:
        return 4
    if n <= 8192:
        return 8
    return 16


# ---------------------------------------------------------------------------
# Fused residual-add + bias + LayerNorm epilogue (EncoderOutput)
# ---------------------------------------------------------------------------
@triton.jit
def _add_ln_fwd(A, R, DB, Y, W, B,
                N: tl.constexpr, eps: tl.constexpr,
                B0: tl.constexpr, B1: tl.constexpr, TWO: tl.constexpr,
                MASK1: tl.constexpr, HAS_DB: tl.constexpr,
                HAS_W: tl.constexpr, HAS_B: tl.constexpr):
    """One program per row: ``Y = layer_norm(A + DB + R) * W + B``.

    ``A`` is the GEMM output, ``R`` the residual, ``DB`` the GEMM's own (skipped)
    bias.  All three are summed in fp32 *in registers*, so the row is traversed
    once -- the baseline's separate ``add`` kernel wrote the sum to HBM only for
    ``layer_norm`` to read it straight back.  ``Y`` may alias ``A`` (it does):
    every tile is loaded before any tile is stored, and rows do not overlap.

    The reduction is L1 layer_norm's **shifted one pass**: subtract the row's own
    first element and accumulate ``sum(d)`` and ``sum(d*d)`` together.  The two
    trees are independent so they pipeline, instead of serializing the way a
    literal mean-then-variance pass does, and shifting by a real data point means
    ``sq/N - off*off`` cancels only the *shifted* mean -- so precision survives a
    row whose ``|mean| >> std``, which the naive ``E[x^2] - E[x]^2`` does not.
    ``maximum(., 0)`` covers a rounded variance landing a hair below zero, where
    an eps of exactly 0 would give NaN instead of the reference's inf.

    ``gdc_wait()`` pairs with ``launch_pdl=True``: this kernel always runs
    downstream of the GEMM (or of cuBLAS's split-K reduction), so its grid is
    staged while the producer drains and only the first load waits.

    ``evict_first`` on the row loads and the store: neither is revisited, and
    demoting them leaves L2 to ``DB``/``W``/``B``, which every one of the (up to
    2048) programs re-reads.
    """
    row = tl.program_id(0)
    base = row.to(tl.int64) * N
    c0 = tl.arange(0, B0)
    # The GEMM may still be draining; wait before reading its output.
    gdc_wait()
    # Shift by the row's own first element.  These scalar loads touch the same
    # cache lines the vector loads below want, so they cost only the instruction.
    shift = tl.load(A + base).to(tl.float32) + tl.load(R + base).to(tl.float32)
    if HAS_DB:
        shift += tl.load(DB).to(tl.float32)

    s0 = (tl.load(A + base + c0, eviction_policy="evict_first").to(tl.float32)
          + tl.load(R + base + c0, eviction_policy="evict_first").to(tl.float32))
    if HAS_DB:
        s0 += tl.load(DB + c0).to(tl.float32)
    d0 = s0 - shift
    acc = tl.sum(d0, axis=0)
    sq = tl.sum(d0 * d0, axis=0)
    if TWO:
        c1 = B0 + tl.arange(0, B1)
        if MASK1:
            m1 = c1 < N
            s1 = (tl.load(A + base + c1, mask=m1,
                          eviction_policy="evict_first").to(tl.float32)
                  + tl.load(R + base + c1, mask=m1,
                            eviction_policy="evict_first").to(tl.float32))
            if HAS_DB:
                s1 += tl.load(DB + c1, mask=m1).to(tl.float32)
            # Padding lanes must contribute 0 to both sums, so zero them after
            # the shift rather than loading ``other=0.0``.
            d1 = tl.where(m1, s1 - shift, 0.0)
        else:
            s1 = (tl.load(A + base + c1,
                          eviction_policy="evict_first").to(tl.float32)
                  + tl.load(R + base + c1,
                            eviction_policy="evict_first").to(tl.float32))
            if HAS_DB:
                s1 += tl.load(DB + c1).to(tl.float32)
            d1 = s1 - shift
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


# ---------------------------------------------------------------------------
# GELU epilogue (EncoderIntermediate)
# ---------------------------------------------------------------------------
# Exact-mode GELU on one SFU op, from L1 gelu: x*Phi(x) is refactored as
# 0.5x(1+tanh(y)) with y = x*(A1 + A3 x^2 + A5 x^4) least-squares fitted to half
# the logit of Phi(x), weighted by d(gelu)/dy so the error is flat in the output.
# Worst-case |error| vs x*Phi(x) over x in [-8, 8] is 3.0e-5 -- below one fp16
# output ULP, against a (1e-2, 1e-2) tolerance.  Capping x^2 keeps the tanh
# argument positive past the quintic's turnover at |x| ~ 11, so gelu(x) -> x
# still holds in the tail.
_A1 = tl.constexpr(0.79745782)
_A3 = tl.constexpr(0.037051035)
_A5 = tl.constexpr(-0.000358865)
_UCAP = tl.constexpr(64.0)

# Launch shape for the [M, 4096] fp16 activation, swept against the benchmark's
# own timing loop at M=64/512/2048.  BLOCK=2048 with 8 warps is optimal at all
# three (13.34 / 15.36 / 27.65 us end to end).  BLOCK dominates at M=2048 (4096
# and 2048 tie, 1024 costs 2.0, 512 costs 6.1, 256 costs 14.3 us); at M=512 it is
# the warp count that separates 15.4 from 16.4, and L1's BLOCK=2048/4-warp choice
# lands on the wrong side of that one.
_G_BLOCK = 2048
_G_WARPS = 8


@triton.jit
def _gelu_kernel(X, Y, n, EXACT_TILES: tl.constexpr, BLOCK: tl.constexpr):
    """``Y = gelu(X)`` over a flat buffer, fp16/bf16.  ``Y`` may alias ``X``.

    ``launch_pdl`` + ``gdc_wait()``: this always runs on the GEMM's output, so
    the grid is staged while the GEMM drains and only the first load waits.

    ``EXACT_TILES`` drops the bounds mask when ``BLOCK`` divides ``n`` -- true for
    every captured shape, since ``n = M * 4096``.
    """
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    gdc_wait()
    if EXACT_TILES:
        x = tl.load(X + off).to(tl.float32)
    else:
        m = off < n
        x = tl.load(X + off, mask=m).to(tl.float32)
    u = tl.minimum(x * x, _UCAP)
    p = (_A5 * u + _A3) * u + _A1
    h = 0.5 * x
    t = tl.inline_asm_elementwise("tanh.approx.f32 $0, $1;", "=f,f", [x * p],
                                  dtype=tl.float32, is_pure=True, pack=1)
    y = (h * t + h).to(X.dtype.element_ty)
    if EXACT_TILES:
        tl.store(Y + off, y)
    else:
        tl.store(Y + off, y, mask=m)


# ---------------------------------------------------------------------------
# Written split-K GEMM + fused reduce/bias/residual/LayerNorm (EncoderOutput)
# ---------------------------------------------------------------------------
# Only for the shapes in ``_SK_CFG``, i.e. only where this pair was *measured*
# faster than cuBLAS and verified against the reference -- the same discipline
# L1 ``Linear``'s ``_MM_CFG`` gate uses.  It wins in exactly one place, and the
# reason is structural rather than a better matmul:
#
#   K=4096 against M=64 rows leaves cuBLAS no choice but to partition K, so
#   ``torch.mm`` is *two* kernels (``nvjet_..._splitK_`` plus
#   ``cublasLt::splitKreduce_kernel``) and the LayerNorm epilogue is a third.
#   Owning the split-K means owning the reduction, and a reduction that already
#   holds complete output rows can do the row statistics for free -- so the
#   whole module becomes two kernels.  Measured 13.33 vs 15.36 us against the
#   cuBLAS path in the ranking harness (five interleaved reps, both orders),
#   i.e. exactly the ~2.05 us one launch costs here.
#
# At M=512/2048 cuBLAS does *not* split K, its GEMM is one kernel, and this
# kernel loses: the fusion can only ever save one launch (2.05 us) while the
# written mainloop gives up 2-6 us to nvjet on those shapes.  Both fall through
# to the cuBLAS path below.  See ITERATIONS.md for the per-shape table.
@triton.jit
def _sk_mm(A, B, P, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
           BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
           KC: tl.constexpr, NS: tl.constexpr):
    """``P[sk] = A[:, k0:k0+KC] @ B[k0:k0+KC, :]`` -- fp32 partials, grid (tiles, SK).

    fp32 partials rather than fp16: they are only ``SK * M * N * 4`` bytes (2 MB
    at the gated shape) and they stay in L2 for the reducer, so the cheaper store
    would buy nothing and would put a rounding step in front of the LayerNorm
    that the reference does not have.

    ``gdc_wait()`` pairs with ``launch_pdl=True``: this is the *first* kernel of
    the module, so its producer is the benchmark's own input copy rather than
    another kernel of ours -- staging the grid against that copy is worth the
    same ~2 us it is worth anywhere else, but the wait is what makes it correct.
    """
    pid = tl.program_id(0)
    sk = tl.program_id(1)
    nn: tl.constexpr = N // BN
    rm = (pid // nn) * BM + tl.arange(0, BM)
    rn = (pid % nn) * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    k0 = sk * KC
    ap = A + rm[:, None] * K + (k0 + rk)[None, :]
    bp = B + (k0 + rk)[:, None] * N + rn[None, :]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    gdc_wait()
    for _ in tl.range(0, KC // BK, num_stages=NS):
        acc = tl.dot(tl.load(ap), tl.load(bp), acc)
        ap += BK
        bp += BK * N
    tl.store(P + sk * (M * N) + rm[:, None] * N + rn[None, :], acc)


@triton.jit
def _sk_red_ln(P, R, DB, Y, W, B, M: tl.constexpr, N: tl.constexpr,
               SK: tl.constexpr, eps: tl.constexpr, SKC: tl.constexpr,
               HAS_DB: tl.constexpr, HAS_W: tl.constexpr, HAS_B: tl.constexpr):
    """One program per row: sum the SK partials, add bias + residual, LayerNorm.

    This replaces *both* ``cublasLt::splitKreduce_kernel`` and the separate
    epilogue.  Because the row arrives complete and in registers, the statistics
    are a plain two-pass mean-then-variance -- strictly more accurate than the
    shifted one-pass form ``_add_ln_fwd`` needs, and free here, since neither
    pass touches memory.

    ``SKC`` partials are summed per 2-D tile (``[SKC, N]`` fp32) rather than one
    at a time: Triton streams the tile with cp.async and accumulates per thread,
    which a scalar loop over ``SK`` does not get.  ``SKC=4`` at N=1024 is a 16 KB
    tile, inside the register budget.
    """
    row = tl.program_id(0)
    c = tl.arange(0, N)
    base = row.to(tl.int64) * N
    gdc_wait()
    s = tl.arange(0, SKC)
    pp = P + base + c[None, :] + s[:, None] * (M * N)
    a = tl.sum(tl.load(pp), axis=0)
    for k in tl.static_range(SKC, SK, SKC):
        a += tl.sum(tl.load(pp + k * (M * N)), axis=0)
    a += tl.load(R + base + c).to(tl.float32)
    if HAS_DB:
        a += tl.load(DB + c).to(tl.float32)
    inv: tl.constexpr = 1.0 / N
    mu = tl.sum(a, axis=0) * inv
    d = a - mu
    # sum(d*d) cannot be negative in fp32, so no clamp is needed; eps == 0 with a
    # constant row still gives the reference's inf rather than a NaN.
    rstd = 1.0 / tl.sqrt(tl.sum(d * d, axis=0) * inv + eps)
    y = d * rstd
    if HAS_W:
        y = y * tl.load(W + c).to(tl.float32)
    if HAS_B:
        y = y + tl.load(B + c).to(tl.float32)
    tl.store(Y + base + c, y.to(Y.dtype.element_ty))


# (M, K, N) -> (BM, BN, BK, SK, gemm warps, gemm stages, reduce warps, SKC).
# Populated only from measured, verified wins.
_SK_CFG = {
    (64, 4096, 1024): (64, 64, 128, 8, 4, 4, 8, 4),
}


class EncoderIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.intermediate_act_fn = GELU()
        # The retuned epilogue reproduces GELU()'s *exact* mode only; a config
        # asking for the tanh approximation goes to L1's module, which has both.
        self._exact = self.intermediate_act_fn.approximate == "none"
        self._launch = _CachedLaunch()
        self._ldtype: torch.dtype | None = None
        self._ldev = -1

    def _gelu_(self, g: torch.Tensor) -> torch.Tensor:
        """In-place exact GELU over ``g`` (the GEMM's own fresh output buffer).

        In place rather than into an ``empty_like``: it measured identical on the
        GPU and removes an allocation from the host path, which at M=64 is not
        free.
        """
        n = g.numel()
        gp = g.data_ptr()
        exact = n % _G_BLOCK == 0
        # Triton specializes an ``int`` argument on "divisible by 16" and
        # "equals 1", and a pointer on 16-byte alignment, so the memoized
        # launcher may only be reused for arguments on the same side of all
        # three -- hence the guards rather than a bare dtype check.
        if (self._launch.ready
                and g.dtype is self._ldtype
                and g.get_device() == self._ldev
                and _cur_device() == self._ldev
                and exact
                and not (gp & 15)
                and not (n & 15)):
            self._launch.run(triton.cdiv(n, _G_BLOCK), 1, 1,
                             _raw_stream(self._ldev), *self._launch.pre,
                             gp, gp, n, exact, _G_BLOCK)
            return g
        kern = _gelu_kernel[(triton.cdiv(n, _G_BLOCK),)](
            g, g, n, exact, _G_BLOCK, num_warps=_G_WARPS, launch_pdl=True)
        if (exact and not (gp & 15) and not (n & 15)
                and g.get_device() == _cur_device()
                and self._launch.bind(kern)):
            self._ldtype = g.dtype
            self._ldev = g.get_device()
        return g

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        g = self.dense(hidden_states)
        if (self._exact
                and g.dtype in _LOWP
                and g.is_cuda
                and g.is_contiguous()
                and g.numel()
                and not torch.is_grad_enabled()):
            return self._gelu_(g)
        return self.intermediate_act_fn(g)


class EncoderOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = Linear(config.intermediate_size, config.hidden_size, bias=True)
        # promote_fp32=False: vLLM's bert.py / roberta.py use a plain
        # nn.LayerNorm here (see encoder_embeddings for the full rationale).
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps,
                                   promote_fp32=False)

        # Everything the fused launch needs that does not depend on the input.
        # ``_n = 0`` disables the fused path (row too wide for a register tile).
        n = int(config.hidden_size)
        self._k = int(config.intermediate_size)
        if 0 < n <= _MAX_N:
            self._n = n
            b0, b1, two, mask1 = _tile_split(n)
            self._cargs = (n, float(config.layer_norm_eps), b0, b1, two, mask1)
            self._warps = _num_warps_for(n)
        else:
            self._n = 0
            self._cargs = ()
            self._warps = 4

        # The parameter dicts themselves: a Parameter lives in ``_parameters``,
        # not in ``__dict__``, so ``self.dense.weight`` goes through
        # ``nn.Module.__getattr__`` -- a Python-level call, and there are six of
        # them per forward.  Two cached dicts turn that into dict lookups.
        self._dp = self.dense._parameters
        self._lp = self.LayerNorm._parameters

        self._launch = _CachedLaunch()
        self._ldtype: torch.dtype | None = None
        self._ldev = -1
        self._ltail: tuple = ()      # (W ptr, B ptr) + constexprs
        self._ldbp = 0               # dense-bias pointer, 0 when there is none
        # ``dense.weight`` in the layout handed to cuBLAS, plus the guards that
        # invalidate it.
        #
        # The Parameter identities below are held inside tuples, not assigned
        # directly: ``nn.Module.__setattr__`` intercepts any Parameter value and
        # calls ``register_parameter``, so ``self._bw = w`` silently published
        # ``dense.weight`` a second time under the name ``_bw``.  That put an 8 MB
        # alias in ``state_dict()`` and -- worse for the launch path this file
        # spends effort shortening -- moved every later read of ``self._bw`` onto
        # ``nn.Module.__getattr__``, a Python-level call, twice per forward.  A
        # tuple is not a Parameter, so it stays in ``__dict__``, and identity
        # comparison still works through the element.
        self._bwref: tuple = ()      # (dense.weight,)
        self._bver = -1
        self._bwp = -1
        self._bmat: torch.Tensor | None = None
        self._btail: tuple = ()      # (dense.bias, LayerNorm.weight, LayerNorm.bias)

        # Written split-K path: the subset of _SK_CFG that matches this module's
        # (K, N), keyed by the row count, so forward() is one dict lookup.
        self._skm = {k[0]: v for k, v in _SK_CFG.items()
                     if k[1] == self._k and k[2] == n}
        self._skmm = _CachedLaunch()
        self._skrd = _CachedLaunch()
        self._skp: torch.Tensor | None = None   # fp32 partials, reused per call
        self._skargs: tuple = ()
        self._sktail: tuple = ()
        self._skdt: torch.dtype | None = None
        self._skdev = -1

    # ------------------------------------------------------------------
    # weight preparation
    # ------------------------------------------------------------------
    def _gemm_mat(self, w: torch.Tensor) -> torch.Tensor:
        """The right-hand operand for ``x @ w.T``, cached across calls.

        For ``K > N`` this is ``w.t().contiguous()`` -- a full 8 MB copy, so it
        must not be redone per call -- and otherwise just the ``w.t()`` view.
        Weight loading completes before the first forward, so building it lazily
        is safe.

        Four guards, not one.  Identity catches a replaced Parameter (a fresh
        ``load_state_dict``) and ``_version`` catches an in-place update, but
        ``p.data = p.data.to(...)`` re-points a Parameter *without* touching
        either -- which is how ``_prepare_module`` itself casts a module -- so
        the data pointer and dtype are compared too.  Without those the cache
        silently serves a weight of the wrong dtype or from freed storage.
        """
        wp = w.data_ptr()
        ref = self._bwref
        if (not ref or ref[0] is not w or self._bver != w._version
                or self._bwp != wp or self._bmat.dtype is not w.dtype):
            d = w.detach()
            self._bmat = d.t().contiguous() if w.shape[1] > w.shape[0] else d.t()
            self._bwref = (w,)
            self._bver = w._version
            self._bwp = wp
            self._launch.ready = False   # rebind: the bound pointers may move
            self._skmm.ready = False
        return self._bmat

    # ------------------------------------------------------------------
    # written split-K path (only shapes in _SK_CFG reach it)
    # ------------------------------------------------------------------
    def _sk_forward(self, x, r, w, db, lw, lb, cfg):
        """Two kernels for what cuBLAS needs three: split-K GEMM, then a reduce
        that also does bias + residual + LayerNorm.  Returns None to decline.

        The partials buffer is allocated once and reused: it is pure scratch, the
        reducer consumes it in the same launch pair that wrote it, and at M=64 an
        extra 2 MB allocation per call is host time we measurably cannot afford.
        """
        BM, BN, BK, SK, NW, NS, RW, SKC = cfg
        m, n, k = x.shape[0], self._n, self._k
        if not self._cargs or m % BM or n % BN or (k // SK) % BK:
            return None   # _sk_mm stores unmasked; only exact tilings may run
        if (db is None or lw is None or lb is None
                or db.dtype is not x.dtype or lw.dtype is not x.dtype
                or lb.dtype is not x.dtype or not db.is_contiguous()):
            return None
        b = self._gemm_mat(w)                     # [K, N] contiguous
        if self._skp is None or self._skp.shape != (SK, m, n) \
                or self._skp.device != x.device:
            self._skp = torch.empty(SK, m, n, device=x.device, dtype=torch.float32)
            self._skmm.ready = False
        p = self._skp
        y = torch.empty(m, n, device=x.device, dtype=x.dtype)
        g0 = (m // BM) * (n // BN)
        eps = self._cargs[1]
        if (self._skmm.ready and self._skrd.ready
                and x.dtype is self._skdt and x.get_device() == self._skdev
                and _cur_device() == self._skdev):
            xp, yp = x.data_ptr(), y.data_ptr()
            if not ((xp | yp | r.data_ptr()) & 15):
                st = _raw_stream(self._skdev)
                self._skmm.run(g0, SK, 1, st, *self._skmm.pre,
                               xp, b.data_ptr(), p.data_ptr(), *self._skargs)
                self._skrd.run(m, 1, 1, st, *self._skrd.pre,
                               p.data_ptr(), r.data_ptr(), db.data_ptr(), yp,
                               lw.data_ptr(), lb.data_ptr(), *self._sktail)
                return y
        kmm = _sk_mm[(g0, SK)](x, b, p, m, n, k, BM, BN, BK, k // SK, NS,
                               num_warps=NW, launch_pdl=True)
        krd = _sk_red_ln[(m,)](p, r, db, y, lw, lb, m, n, SK, eps, SKC,
                               True, True, True, num_warps=RW, launch_pdl=True)
        ptrs = (x.data_ptr(), b.data_ptr(), p.data_ptr(), y.data_ptr(),
                r.data_ptr(), db.data_ptr(), lw.data_ptr(), lb.data_ptr())
        dev = x.get_device()
        if (not any(q & 15 for q in ptrs) and dev == _cur_device()
                and self._skmm.bind(kmm) and self._skrd.bind(krd)):
            self._skargs = (m, n, k, BM, BN, BK, k // SK, NS)
            self._sktail = (m, n, SK, eps, SKC, True, True, True)
            self._skdt = x.dtype
            self._skdev = dev
        return y

    # ------------------------------------------------------------------
    # fused path
    # ------------------------------------------------------------------
    def _fused(self, x: torch.Tensor, r: torch.Tensor, w, db, lw, lb):
        """First fused call for this configuration: launch normally, then
        memoize.  Returns the result, or None if the fused path cannot run."""
        if ((lw is not None and lw.dtype is not x.dtype)
                or (lb is not None and lb.dtype is not x.dtype)
                or (db is not None and (db.dtype is not x.dtype
                                        or not db.is_contiguous()))):
            # F.layer_norm rejects a low-precision input with a differently
            # typed affine when promote_fp32 is off, and a mistyped dense bias
            # would silently widen what we accept; stay on the reference.
            return None
        g = torch.mm(x, self._gemm_mat(w))
        cargs = self._cargs + (db is not None, lw is not None, lb is not None)
        kern = _add_ln_fwd[(x.shape[0],)](
            g, r, db, g, lw, lb, *cargs,
            num_warps=self._warps, launch_pdl=True)
        ptrs = [g.data_ptr(), r.data_ptr()]
        if db is not None:
            ptrs.append(db.data_ptr())
        if lw is not None:
            ptrs.append(lw.data_ptr())
        if lb is not None:
            ptrs.append(lb.data_ptr())
        dev = x.get_device()
        if (not any(p & 15 for p in ptrs) and dev == _cur_device()
                and self._launch.bind(kern)):
            self._ldbp = 0 if db is None else db.data_ptr()
            self._ltail = ((0 if lw is None else lw.data_ptr(),
                            0 if lb is None else lb.data_ptr()) + cargs)
            self._ldtype = x.dtype
            self._ldev = dev
            self._btail = (db, lw, lb)
        return g

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        dp = self._dp
        w = dp["weight"]
        # Written split-K GEMM + fused reduce/LN, for the measured shapes only.
        if self._skm:
            cfg = self._skm.get(hidden_states.shape[0] if hidden_states.ndim == 2
                                else -1)
            if (cfg is not None
                    and hidden_states.dtype is torch.float16
                    and hidden_states.is_cuda
                    and hidden_states.is_contiguous()
                    and hidden_states.shape[1] == self._k
                    and w.ndim == 2 and w.shape[0] == self._n
                    and w.shape[1] == self._k
                    and w.dtype is hidden_states.dtype
                    and input_tensor.dtype is hidden_states.dtype
                    and input_tensor.is_contiguous()
                    and input_tensor.shape[0] == hidden_states.shape[0]
                    and input_tensor.shape[1] == self._n
                    and not torch.is_grad_enabled()):
                y = self._sk_forward(hidden_states, input_tensor, w, dp["bias"],
                                     self._lp["weight"], self._lp["bias"], cfg)
                if y is not None:
                    return y
        # Memoized launch. Every guard is an identity compare, a small int
        # compare or a bit test; together they cost far less than the ~12 us of
        # binding they replace. The parameter guards are what make the cached
        # pointers and the cached [K, N] weight safe -- see _gemm_mat for why
        # identity and _version alone are not enough. Anything that fails falls
        # through to _fused, which rebinds, or to the L1 reference.
        if (self._launch.ready
                and hidden_states.dtype is self._ldtype
                and self._bwref
                and w is self._bwref[0]
                and self._bver == w._version
                and w.data_ptr() == self._bwp
                and w.dtype is self._ldtype
                and w.shape[0] == self._n
                and w.shape[1] == self._k
                and (dp["bias"], self._lp["weight"], self._lp["bias"])
                == self._btail
                and hidden_states.ndim == 2
                and hidden_states.shape[1] == self._k
                and input_tensor.dtype is self._ldtype
                and input_tensor.shape[0] == hidden_states.shape[0]
                and input_tensor.shape[1] == self._n
                and hidden_states.is_contiguous()
                and input_tensor.is_contiguous()
                and hidden_states.get_device() == self._ldev
                and _cur_device() == self._ldev
                and not torch.is_grad_enabled()):
            g = torch.mm(hidden_states, self._bmat)
            gp = g.data_ptr()
            rp = input_tensor.data_ptr()
            if not ((gp | rp) & 15):   # the alignment the compiled kernel assumes
                self._launch.run(hidden_states.shape[0], 1, 1,
                                 _raw_stream(self._ldev), *self._launch.pre,
                                 gp, rp, self._ldbp, gp, *self._ltail)
                return g
            dbp, lwp, lbp = self._btail
            _add_ln_fwd[(hidden_states.shape[0],)](
                g, input_tensor, dbp, g, lwp, lbp,
                *self._ltail[2:], num_warps=self._warps, launch_pdl=True)
            return g

        if (self._n
                and hidden_states.ndim == 2
                and hidden_states.dtype in _FAST_DTYPES
                and hidden_states.is_cuda
                and hidden_states.is_contiguous()
                and w.ndim == 2
                and w.shape[0] == self._n
                and w.shape[1] == hidden_states.shape[1] == self._k
                and w.dtype is hidden_states.dtype
                and input_tensor.dtype is hidden_states.dtype
                and input_tensor.is_contiguous()
                and input_tensor.shape[0] == hidden_states.shape[0]
                and input_tensor.shape[1] == self._n
                and hidden_states.numel()
                and not torch.is_grad_enabled()):
            y = self._fused(hidden_states, input_tensor, w, dp["bias"],
                            self._lp["weight"], self._lp["bias"])
            if y is not None:
                return y
        return self.LayerNorm(self.dense(hidden_states) + input_tensor)
