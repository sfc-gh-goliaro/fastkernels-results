"""Outer product mean for AlphaFold3 (L2).

Implements AF3 Algorithm 9. Computes an outer product of MSA
representations and averages over the MSA dimension to produce
a pair representation update.

Reference: openfold3/core/model/layers/outer_product_mean.py OuterProductMean

The whole forward is one Triton kernel, and the outer product is never built
---------------------------------------------------------------------------
The eager formulation is ~12 separate operators (layer_norm, two projections,
two mask multiplies, two transposes, the outer einsum, a reshape, the output
projection, the norm einsum, an add and a divide).  At the captured shape --
``m: bf16[1, 8, 16, 64]``, ``c_hidden=32``, ``c_z=128`` -- that is ~190 us of
host dispatch and a global-memory round trip per intermediate, including a
512 KB ``[N_res, N_res, c_hidden**2]`` outer product that exists only to be
contracted away.  So this file has no ops: ``_opm_fwd`` is the whole algorithm
in one launch.

What the kernel computes is a *reassociation* of Algorithm 9.  Writing
``W = linear_out.weight`` as ``W[z]`` reshaped to ``[c_hidden, c_hidden]``,

    out[i, j, z] = (bias[z] + sum_s a[s, i, :]^T W[z] b[s, j, :]) / norm[i, j]

and both matrix products can be pushed all the way into the *weights*:

    Y[z] = W1^T W[z]                            [c_m, c_hidden]
    t    = lnm @ Y[z]                            (so ``a`` never exists)
    b    = lnm @ W2^T
    out[i, j, z] = sum over the joined (s, h) axis of t[i, (s,h)] b[j, (s,h)]

``lnm`` is the layer-normed MSA tile with the mask already applied, its rows
ordered ``(residue, seq)`` so that both ``[N_seq*N_res, h] -> [N_res, N_seq*h]``
reshapes are linear-index identities and no transpose is needed to join ``s``
onto the contraction axis.  Consequences, all measured (see ITERATIONS.md):

* **the MSA-axis reduction happens inside a ``tl.dot``.**  The previous kernel
  formed the outer-product slice with a 4D broadcast-reduce on the vector units;
  that was its single largest stage.  Here ``s`` is folded into the pair dot's
  ``K`` (``N_seq * c_hidden = 256``), which needs no padding of the length-8 MSA
  axis -- the thing that defeated every direct tensor-core attempt before.  The
  whole contraction, both dots plus the store, is 0.45 us of a 5.4 us kernel;
  it used to be ~3.0 us.
* **``Y[z]`` depends only on weights,** so it issues while the MSA tile is still
  in flight and costs nothing.  Folding ``W[z]`` into the weight rather than into
  the activations also takes one dot off the post-layer-norm dependency chain,
  which is what the kernel is actually bound by (0.3 us).
* **the pair dot's M and N are the two residue axes,** so ``tl.dot`` forces
  ``PB >= 16`` on both -- the whole 16-residue axis at the captured shape.  One
  CTA therefore owns one ``c_z`` channel and the entire pair block, and the
  layer-norm tile is shared by ``t`` and ``b`` instead of loaded twice.  The
  grid is ``(pair blocks, c_z, batch)``.

Every call-invariant load -- both projections, the ``W[z]`` slice, the bias, the
layer-norm affine -- is issued before the MSA tile is touched: the benchmark
flushes L2 before each timed call, so all of them are cold misses and want to
overlap rather than serialize behind the layer-norm.  Hoisting the ``W[z]`` load
alone is worth 0.6 us, and issuing the MSA tile's load *before* the weights-only
``Y[z]`` dot -- rather than after, which is what reads naturally -- is worth
another 0.6, because Triton will not hoist a load across an intervening dot.

The host side is the other half of the eager cost, and after the first call the
compiled kernel is invoked through its own C launcher rather than Triton's
per-call binder (the technique the L1 ``LayerNorm`` documents at length): every
argument except the nine pointers is ``constexpr``, so a launch is one call with
two pre-built tuples.  That is ~11 us of Python per forward, which the harness
hides entirely behind its own cache-flush memset -- host time stopped mattering
once it dropped under ~40 us, so it is not tuned further.

Numerics follow the eager path stage for stage -- fp32 layer-norm reduction and
affine, low-precision operands with fp32 accumulation for every matmul, bias
before the divide -- while dropping the intermediate rounding the eager path only
incurs because it has to write each stage to memory.  The reassociation does
change the accumulation order, so the folded weights are rounded once to the
input dtype and the tolerance is re-checked against the real baseline in
``tools/verify.py``.  Anything the kernel does not cover (fp32/fp64 input, mixed
parameter precision, a broadcast mask, non-contiguous or CPU tensors, autograd,
or a shape whose register tiles will not fit) falls through to ``_reference``
below, which is the eager code verbatim.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import triton
import triton.compiler.errors
import triton.language as tl
import triton.runtime.errors
from torch._C import _cuda_getCurrentRawStream as _raw_stream
from torch._C import _cuda_getDevice as _cur_device

from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear


# Dtypes the fused kernel is allowed to claim.  fp32 is deliberately excluded:
# the reference would route it through cuBLAS kernels that switch between exact
# fp32 and TF32 by shape, and the fp32 tolerance (atol 1e-5) is tight enough
# that matching whichever one it picked is a correctness risk, not a win.
_FAST_DTYPES = (torch.bfloat16, torch.float16)

# Register-block ceiling: elements in one layer-norm tile.  A shape that cannot
# fit under it takes the reference path rather than spilling.  At the captured
# shape the tile is 8 K elements and the kernel uses 146 registers with no
# spills; ``tools/inspect_kern.py`` reports both.
_MAX_TILE = 1 << 15
_MAX_ACC = 1 << 14

# Preferred (pair block, warps, pipeline stages).  ``_plan`` grows the pair
# block to cover the whole residue axis when the tile budget allows, which is
# what lets one layer-norm tile serve both ``t`` and ``b``.
#
# 4 warps, not 8: the front end alone is 1.2 us faster at 8 warps, but the three
# dots are 1.4 us slower there, because the pair dot is only 16x16 and spreading
# it over more warps buys nothing while costing a cross-warp reduction.  Once the
# contraction stopped being a vector-unit broadcast this became the crossover,
# so it was re-swept rather than inherited.  ``num_stages`` is inert (no loop).
_CFG_DEFAULT = "16,4,2"
try:
    _PB, _WARPS, _STAGES = (
        int(v) for v in os.environ.get("OPM_CFG", _CFG_DEFAULT).split(","))
except ValueError:  # a malformed override must not break the module
    _PB, _WARPS, _STAGES = (int(v) for v in _CFG_DEFAULT.split(","))


@triton.jit
def _load(M, MASK, mbase, kbase, r0,
          S: tl.constexpr, R: tl.constexpr, C: tl.constexpr,
          SB: tl.constexpr, CB: tl.constexpr, PB: tl.constexpr,
          HAS_MASK: tl.constexpr, RMASK: tl.constexpr, CMASK: tl.constexpr):
    """Issue the MSA-tile and mask loads for ``PB`` residues from ``r0``.

    Deliberately split from the layer-norm below.  Every load in this kernel is a
    cold miss -- the benchmark flushes L2 before each timed call -- so all of them
    have to be in flight before any arithmetic waits on any one of them.  Doing
    the weights-only ``Y[z]`` dot before this costs 0.6 us: that dot blocks on the
    weight loads, and the tile load is only issued once it retires.

    Rows are ``(residue, seq)`` -- the reverse of the input's layout -- so that
    ``[PB * SB, h] -> [PB, SB * h]`` is a pure reshape further down and the ``s``
    axis lands inside the pair dot's ``K`` with no register shuffle.  Padding rows
    (``s >= N_seq`` or past the last residue) load as zero, which makes their
    layer-norm the affine bias rather than a NaN, and the mask multiply then
    zeroes them -- so they contribute nothing to the contraction and nothing to
    ``norm``.
    """
    cid = tl.arange(0, CB)
    cok = cid < C
    rid = tl.arange(0, PB * SB)
    r_i = r0 + rid // SB
    s_i = rid % SB
    rok = (s_i < S) & (r_i < R)
    xoff = mbase + (s_i * R + r_i)[:, None] * C + cid[None, :]
    if RMASK:
        x = tl.load(M + xoff, other=0.0,
                    mask=(rok[:, None] & cok[None, :]) if CMASK else rok[:, None])
    elif CMASK:
        x = tl.load(M + xoff, mask=cok[None, :], other=0.0)
    else:
        x = tl.load(M + xoff)

    if HAS_MASK:
        koff = kbase + s_i * R + r_i
        mk = (tl.load(MASK + koff, mask=rok, other=0.0) if RMASK
              else tl.load(MASK + koff)).to(tl.float32)
    elif RMASK:
        mk = tl.where(rok, 1.0, 0.0)
    else:
        mk = tl.full([PB * SB], 1.0, tl.float32)
    return x, mk


@triton.jit
def _norm(x, mk, lnw, lnb,
          C: tl.constexpr, CB: tl.constexpr, SB: tl.constexpr, PB: tl.constexpr,
          LN_EPS: tl.constexpr, HAS_LNW: tl.constexpr, HAS_LNB: tl.constexpr,
          CMASK: tl.constexpr, LP: tl.constexpr):
    """``layer_norm(x) * mask``: fp32 reduction and affine, low-precision result.

    The mask goes on the tile once here, so both projections inherit it and their
    ``[PB * SB, c_hidden]`` results go straight from the dot to low precision.
    """
    xf = x.to(tl.float32)
    mu = tl.sum(xf, 1) * (1.0 / C)
    xc = xf - mu[:, None]
    if CMASK:
        xc = tl.where((tl.arange(0, CB) < C)[None, :], xc, 0.0)
    var = tl.sum(xc * xc, 1) * (1.0 / C)
    ln = xc * tl.rsqrt(var + LN_EPS)[:, None]
    if HAS_LNW:
        ln = ln * lnw[None, :]
    if HAS_LNB:
        ln = ln + lnb[None, :]
    return (ln * mk[:, None]).to(LP), tl.reshape(mk, [PB, SB])


@triton.jit
def _opm_fwd(
    M, MASK, OUT, LNW, LNB, W1, W2, WOUT, BOUT,
    S: tl.constexpr, R: tl.constexpr, C: tl.constexpr,
    H: tl.constexpr, Z: tl.constexpr,
    SB: tl.constexpr, CB: tl.constexpr, HB: tl.constexpr,
    PB: tl.constexpr, ND: tl.constexpr, SHARED: tl.constexpr,
    LN_EPS: tl.constexpr, EPS: tl.constexpr,
    HAS_MASK: tl.constexpr, HAS_LNW: tl.constexpr, HAS_LNB: tl.constexpr,
    RMASK: tl.constexpr, CMASK: tl.constexpr, HMASK: tl.constexpr,
    PMASK: tl.constexpr, NB1: tl.constexpr,
):
    """``out[bt, i0:i0+PB, d0:d0+PB, z]`` for one CTA.

    ``RMASK`` / ``CMASK`` / ``HMASK`` / ``PMASK`` say whether the padded block
    dims actually overhang the real extent (MSA rows and residues, channels,
    ``c_hidden``, and the residue-pair block); where they do not, the guards fold
    away and the loads and the store are unmasked.  ``SHARED`` is set when the
    pair block covers the whole residue axis, so one tile serves both sides.
    """
    pid = tl.program_id(0)
    z = tl.program_id(1)
    bt = tl.program_id(2)
    lp = W1.dtype.element_ty
    od = OUT.dtype.element_ty
    if SHARED:
        # One pair block covers the whole residue axis, so both block offsets are
        # zero at compile time and every tile address folds to a constant.
        i0 = 0
        d0 = 0
    else:
        i0 = (pid // ND) * PB
        d0 = (pid % ND) * PB
    if NB1:
        mbase = 0
        kbase = 0
    else:
        mbase = bt * (S * R * C)
        kbase = bt * (S * R)

    cid = tl.arange(0, CB)
    hid = tl.arange(0, HB)
    cok = cid < C
    hok = hid < H

    # ---- every cold miss first ------------------------------------------
    # linear_1 / linear_2 hold [c_hidden, c_m] and are only used transposed.
    # Loading row-major and transposing keeps the load coalesced and needs no
    # host-side copy, so an in-place weight update is picked up with no cache to
    # invalidate -- and the transpose is free: it compiles to a shared-memory
    # descriptor change (``ttg.memdesc_trans``), not a data movement.  Loading
    # pre-transposed instead measured identical, which is how that was confirmed.
    woff = hid[:, None] * C + cid[None, :]
    if HMASK and CMASK:
        wmk = hok[:, None] & cok[None, :]
    elif HMASK:
        wmk = tl.broadcast_to(hok[:, None], [HB, CB])
    elif CMASK:
        wmk = tl.broadcast_to(cok[None, :], [HB, CB])
    if HMASK or CMASK:
        w1 = tl.load(W1 + woff, mask=wmk, other=0.0)
        w2 = tl.load(W2 + woff, mask=wmk, other=0.0)
    else:
        w1 = tl.load(W1 + woff)
        w2 = tl.load(W2 + woff)

    # linear_out weight rows are the flattened (h1, h2) pair: [c_z, H * H].
    zoff = z * (H * H) + hid[:, None] * H + hid[None, :]
    if HMASK:
        wz = tl.load(WOUT + zoff, mask=hok[:, None] & hok[None, :], other=0.0)
    else:
        wz = tl.load(WOUT + zoff)
    bo = tl.load(BOUT + z).to(tl.float32)
    if HAS_LNW:
        lnw = (tl.load(LNW + cid, mask=cok, other=0.0) if CMASK
               else tl.load(LNW + cid)).to(tl.float32)
    else:
        lnw = tl.zeros([CB], tl.float32)
    if HAS_LNB:
        lnb = (tl.load(LNB + cid, mask=cok, other=0.0) if CMASK
               else tl.load(LNB + cid)).to(tl.float32)
    else:
        lnb = tl.zeros([CB], tl.float32)

    xi, mki = _load(M, MASK, mbase, kbase, i0, S, R, C, SB, CB, PB,
                    HAS_MASK, RMASK, CMASK)
    if SHARED:
        xd = xi
        mkd = mki
    else:
        xd, mkd = _load(M, MASK, mbase, kbase, d0, S, R, C, SB, CB, PB,
                        HAS_MASK, RMASK, CMASK)

    # ---- then arithmetic ------------------------------------------------
    # Y[z] = W1^T W[z].  Weights only, so this dot fills the shadow of the tile
    # loads above; it is what lets `a = ln @ W1^T` be skipped entirely.
    yz = tl.dot(tl.trans(w1), wz).to(lp)
    w2t = tl.trans(w2)

    lhs_i, mk_i = _norm(xi, mki, lnw, lnb, C, CB, SB, PB, LN_EPS,
                        HAS_LNW, HAS_LNB, CMASK, lp)
    if SHARED:
        lhs_d = lhs_i
        mk_d = mk_i
    else:
        lhs_d, mk_d = _norm(xd, mkd, lnw, lnb, C, CB, SB, PB, LN_EPS,
                            HAS_LNW, HAS_LNB, CMASK, lp)

    t2 = tl.reshape(tl.dot(lhs_i, yz).to(lp), [PB, SB * HB])
    b2 = tl.reshape(tl.dot(lhs_d, w2t).to(lp), [PB, SB * HB])
    # norm[ii, dd] = sum_s mask[s, i0+ii] * mask[s, d0+dd]
    nrm = tl.sum(mk_i[:, None, :] * mk_d[None, :, :], 2)
    res = (tl.dot(t2, tl.trans(b2)) + bo) / (nrm + EPS)

    ri = i0 + tl.arange(0, PB)
    rd = d0 + tl.arange(0, PB)
    ooff = ri[:, None] * (R * Z) + rd[None, :] * Z + z
    if not NB1:
        ooff += bt * (R * R * Z)
    if PMASK:
        tl.store(OUT + ooff, res.to(od),
                 mask=(ri[:, None] < R) & (rd[None, :] < R))
    else:
        tl.store(OUT + ooff, res.to(od))


class OuterProductMean(nn.Module):
    """AF3 Algorithm 9: Outer product mean.

    Args:
        c_m: MSA embedding channel dimension
        c_z: Pair embedding channel dimension
        c_hidden: Hidden channel dimension
        eps: Epsilon for numerical stability
    """

    def __init__(self, c_m: int, c_z: int, c_hidden: int, eps: float = 1e-3):
        super().__init__()
        self.c_m = c_m
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.eps = eps

        self.layer_norm = LayerNorm(c_m)
        self.linear_1 = Linear(c_m, c_hidden, bias=False)
        self.linear_2 = Linear(c_m, c_hidden, bias=False)
        self.linear_out = Linear(c_hidden ** 2, c_z, bias=True)

        self._fusable = c_hidden > 0 and c_z > 0 and c_m > 0

        # Direct-launch cache, keyed on everything that can change the compiled
        # kernel or the grid.  Populated on the first fused call for a key.
        self._key: tuple | None = None
        self._nofast: set = set()
        self._lrun = None
        self._lpre: tuple = ()
        self._lpost: tuple = ()
        self._grid: tuple = ()
        self._oshape: tuple = ()
        self._ldev = -1
        # Parameters the cached launch reads by address.  Held so the storage
        # stays alive, and identity-guarded per call: an in-place weight update
        # keeps the address (and is picked up), replacing a Parameter object does
        # not (the same window the L1 modules document).
        self._pw: tuple = ()
        self._pln: tuple = (0, 0)
        self._pd = (self.layer_norm._parameters, self.linear_1._parameters,
                    self.linear_2._parameters, self.linear_out._parameters)

    # ------------------------------------------------------------------
    # the unfused reference path -- anything the kernel does not cover
    # ------------------------------------------------------------------
    def _reference(
        self,
        m: torch.Tensor,
        mask: torch.Tensor | None,
        chunk_size: int | None,
        inplace_safe: bool,
    ) -> torch.Tensor:
        if mask is None:
            mask = m.new_ones(m.shape[:-1])

        ln = self.layer_norm(m)

        mask = mask.unsqueeze(-1)
        a = self.linear_1(ln) * mask
        b = self.linear_2(ln) * mask

        del ln

        # [*, N_res, N_seq, C]
        a = a.transpose(-2, -3)
        b = b.transpose(-2, -3)

        # [*, N_res, N_res, C, C]
        outer = torch.einsum("...bac,...dae->...bdce", a, b)

        # [*, N_res, N_res, C * C]
        outer = outer.reshape(outer.shape[:-2] + (-1,))

        # [*, N_res, N_res, C_z]
        outer = self.linear_out(outer)

        # Normalization: count valid sequence pairs per residue pair
        norm = torch.einsum("...abc,...adc->...bdc", mask, mask)
        norm = norm + self.eps

        outer = outer / norm

        return outer

    # ------------------------------------------------------------------
    # fused path
    # ------------------------------------------------------------------
    def _plan(self, m: torch.Tensor, mask: torch.Tensor | None):
        """Block shape / grid / constexpr tail for this input, or None.

        Every ``tl.dot`` fixes part of the shape: ``Y[z] = W1^T W[z]`` needs
        ``CB, HB >= 16``, the projections need ``PB * SB >= 16``, and the pair
        dot -- whose M and N are the two residue axes and whose K is the joined
        ``(seq, c_hidden)`` axis -- needs ``PB >= 16`` and ``SB * HB >= 16``.
        Beyond that the only choice is how much of the residue axis one CTA
        takes: covering all of it lets a single layer-norm tile serve both
        projections, which is worth more than anything else here, so ``PB`` is
        grown to the whole axis whenever the tile budget allows and dropped back
        to the 16 the pair dot requires when it does not.
        """
        S, R, C = m.shape[-3], m.shape[-2], m.shape[-1]
        if C != self.c_m or S < 1 or R < 1:
            return None
        H, Z = self.c_hidden, self.c_z
        SB = triton.next_power_of_2(S)
        CB = triton.next_power_of_2(C)
        HB = triton.next_power_of_2(H)
        RB = triton.next_power_of_2(R)
        if CB < 16 or HB < 16 or SB * HB < 16 or CB * HB > _MAX_TILE:
            return None
        # Mixed parameter/input precision would change which matmul the
        # reference picks (and for fp32 params, whether it is TF32).
        for p in self.parameters():
            if p.dtype != m.dtype:
                return None

        # Cover the whole residue axis in one pair block if the tile budget
        # allows -- that is what makes SHARED possible -- otherwise fall back to
        # the 16 the pair dot's M and N require and tile the axis.
        PB = min(max(16, RB), max(16, _PB))
        if PB * SB * CB > _MAX_TILE or PB * SB * HB > _MAX_TILE:
            PB = 16
        nI = (R + PB - 1) // PB
        shared = 1 if nI == 1 else 0
        live = 1 if shared else 2          # layer-norm tiles alive at once
        if (live * PB * SB * CB > _MAX_TILE or live * PB * SB * HB > _MAX_TILE
                or PB * PB > _MAX_ACC):
            return None

        lnw = self._pd[0]["weight"]
        lnb = self._pd[0]["bias"]
        cargs = (
            S, R, C, H, Z, SB, CB, HB, PB, nI, shared,
            float(self.layer_norm.eps), float(self.eps),
            mask is not None, lnw is not None, lnb is not None,
            SB != S or nI * PB != R, CB != C, HB != H, nI * PB != R,
            m.numel() == S * R * C,
        )
        grid = (nI * nI, Z, m.numel() // (S * R * C))
        return cargs, grid

    def _launch_setup(self, m, mask, key):
        plan = self._plan(m, mask)
        if plan is None:
            self._key = None
            return None
        cargs, grid = plan
        pd0, pd1, pd2, pd3 = self._pd
        lnw, lnb = pd0["weight"], pd0["bias"]
        w1, w2 = pd1["weight"], pd2["weight"]
        wout, bout = pd3["weight"], pd3["bias"]
        if bout is None or w1 is None or w2 is None or wout is None:
            self._key = None
            return None
        oshape = m.shape[:-3] + (m.shape[-2], m.shape[-2], self.c_z)
        out = torch.empty(oshape, dtype=m.dtype, device=m.device)

        dummy = m if mask is None else mask
        params = (lnw if lnw is not None else m, lnb if lnb is not None else m,
                  w1, w2, wout, bout)
        try:
            kern = _opm_fwd[grid](
                m, dummy, out, *params, *cargs,
                num_warps=_WARPS, num_stages=_STAGES,
            )
        except (triton.runtime.errors.OutOfResources,
                triton.compiler.errors.CompilationError):
            # A shape whose register/shared budget the plan's ceilings did not
            # catch, or that the installed Triton will not compile.  Remember
            # this key so we do not pay the failed compile again, and take the
            # reference -- other shapes are unaffected.
            self._key = None
            self._nofast.add(key)
            return None

        # Memoize the compiled kernel's own C launcher: Triton's
        # ``kernel[grid](...)`` re-binds, re-specializes and re-hashes every
        # argument on each call, which is more Python than this whole operator is
        # GPU work.  Every argument but the pointers is constexpr, so the only
        # per-call specialization left is pointer alignment -- hence the 16-byte
        # checks here and in ``forward``.  Every internal attribute is fetched
        # defensively; if any is missing we never memoize and keep going through
        # the supported path.
        ptrs = (m, out, dummy) + params
        if all(not (t.data_ptr() & 15) for t in ptrs):
            launcher = None if kern is None else kern.run
            raw = getattr(launcher, "launch", None)
            if (raw is not None
                    and m.get_device() == _cur_device()
                    and getattr(launcher, "global_scratch_size", None) == 0
                    and getattr(launcher, "profile_scratch_size", None) == 0):
                self._pw = params
                self._pln = (lnw, lnb)
                self._lrun = raw
                self._lpre = (
                    kern.function,
                    launcher.launch_cooperative_grid, launcher.launch_pdl,
                    None, None,                   # global / profile scratch
                    kern.packed_metadata,
                    None, None, None,             # launch metadata, 2 hooks
                )
                self._lpost = tuple(p.data_ptr() for p in params) + cargs
                self._grid = grid
                self._oshape = oshape
                self._ldev = m.get_device()
                self._key = key
            else:
                self._key = None
        else:
            self._key = None
        return out

    def forward(
        self,
        m: torch.Tensor,
        mask: torch.Tensor | None = None,
        chunk_size: int | None = None,
        inplace_safe: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            m:    [*, N_seq, N_res, C_m] MSA embedding
            mask: [*, N_seq, N_res] MSA mask

        Returns:
            [*, N_res, N_res, C_z] pair embedding update
        """
        if (self._fusable
                and m.ndim >= 3
                and m.dtype in _FAST_DTYPES
                and m.is_cuda
                and m.is_contiguous()
                and not torch.is_grad_enabled()
                and (mask is None
                     or (mask.shape == m.shape[:-1] and mask.is_contiguous()
                         and mask.dtype == m.dtype))):
            key = (m.shape, m.dtype, mask is None, m.get_device())
            if key == self._key:
                pd0, pd1, pd2, pd3 = self._pd
                pw, pln = self._pw, self._pln
                if (pd1["weight"] is pw[2] and pd2["weight"] is pw[3]
                        and pd3["weight"] is pw[4] and pd3["bias"] is pw[5]
                        and pd0["weight"] is pln[0] and pd0["bias"] is pln[1]
                        and _cur_device() == self._ldev):
                    mp = m.data_ptr()
                    kp = mp if mask is None else mask.data_ptr()
                    if not ((mp | kp) & 15):
                        out = torch.empty(self._oshape, dtype=m.dtype,
                                          device=m.device)
                        op = out.data_ptr()
                        if not (op & 15):
                            g = self._grid
                            self._lrun(g[0], g[1], g[2],
                                       _raw_stream(self._ldev), *self._lpre,
                                       mp, kp, op, *self._lpost)
                            return out
            elif key not in self._nofast:
                out = self._launch_setup(m, mask, key)
                if out is not None:
                    return out
        return self._reference(m, mask, chunk_size, inplace_safe)
