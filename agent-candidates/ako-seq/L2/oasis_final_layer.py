"""Oasis final DiT projection layer, collapsed into two fused Triton kernels.

The layer is ``x -> LayerNorm -> *(1+scale) + shift -> Linear(1024, 64)`` with
``shift, scale = Linear(1024, 2048)(SiLU(c))``.  The baseline spends 67-81 us of
*device* time on it (the harness's ``l2.zero_()`` runs the GPU tens of
microseconds behind the host, so the scored window is pure device time and host
cost is free), and moves under 6 MB doing it -- so this is a launch-count and
memory-layout problem, not a FLOP problem.  Two kernels, and it cannot be one:
every output row needs ``shift``/``scale``, a 1024-deep reduction of ``c``
against a multi-megabyte weight, so stage 2 has a real grid-wide dependency on
stage 1.

**The layout.**  ``x`` is captured NON-contiguous: ``[1, T, 9, 16, 1024]`` with
strides ``(., 147456, 16, 1, 144)``, i.e. a ``[T, 1024, 144]`` contiguous buffer
viewed channels-last.  So ``p = h*16 + w`` collapses to a stride-1 index and the
144 *rows* of a frame are contiguous at a fixed feature ``j``, while the
LayerNorm axis walks stride 144.  Row-blocking is therefore what gives coalesced
loads, and ``F.layer_norm``'s need to materialize a contiguous fp32 copy of this
view is a large part of why the baseline is slow.

**Halving the dominant stream.**  Writing ``g = 1+scale``, ``b = shift``,
``V = linear.weight``:

    out[p,n] = rstd_p * (A[p,n] - mu_p * Gv[n]) + Bv[n] + bias[n]
    A[p,n]   = sum_j x[j,p] * g_j * V[n,j]
    Gv[n]    = sum_j g_j * V[n,j]        Bv[n] = sum_j b_j * V[n,j]

``b`` (1024 numbers per frame) is used ONLY through ``Bv`` (64 numbers per
frame), and ``Bv[t,n] = sum_k silu(c)[t,k] * (V @ Wshift)[n,k] + (V @ bshift)[n]``.
``V @ Wshift`` is a [64, 1024] matrix derivable from the weights, so it replaces
the entire 2 MB ``Wshift`` read; ``Gv`` goes the same way through ``V @ Wscale``.
``g`` is still needed elementwise, so ``Wscale`` stays.  Everything stage 1 must
produce is then ONE uniform GEMV against a single packed matrix

    A = [ Wscale ; V @ Wshift ; V @ Wscale ]        (1024 + 64 + 64) x 1024

with the ``+1`` of ``(1+scale)`` and ``colsum(V)`` folded into its fp32 bias.
2.25 MB instead of 4 MB, one kernel, no branches -- and it also removes two
per-iteration cross-tile reductions from stage 2.

**Why that identity is also what makes stage 2 single-pass.**  ``A`` does not
involve ``mu`` or ``rstd``, so the projection can be accumulated in the SAME
traversal of ``x`` that reduces the row statistics; ``mu``/``rstd`` are applied
in the epilogue.  The textbook fusion needs two passes (reduce, then re-read to
normalize).  The normalized/modulated activation is never written to HBM.

Both kernels are shaped by what the harness actually measures, which is *device
time with a cold, dirty L2*: it fills a 253 MB buffer before every timed call,
so the GPU runs tens of microseconds behind the host (host cost is free) and
every window is quantised to 2.048 us levels with **one level per kernel
launch** -- an empty kernel measures 5.12 us, two of them 7.23, eight 19.55,
against 0.61 us each in a warm CUDA-graph replay.  A warm graph replay ranks
configurations *differently*, so nothing here was tuned against one.

* **Stage 1 has no MMA.**  The obvious form is a ``tl.dot`` of a [16, K/SK]
  ``silu(c)`` operand against the packed slab, but ``T <= 6`` while the MMA's
  minimum is 16 rows: it pads 10 dead rows, and -- because ``BN`` then cannot go
  below the MMA's minimum N=16 either -- it forces parallelism to come from
  split-K rather than from the n axis.  That is expensive *for stage 2*, which
  sums the SK partial planes with one vector load per plane per k-iteration and
  loses about a level per doubling of SK (at SK=32 it collapsed to ~100 us).
  Unrolling the T rows as independent reduce-along-k's removes both
  constraints: 9.25 us against 13.31 for the MMA form at SK=2, with ``BN = 4``
  giving 288 CTAs from ``N/BN`` alone and only a 2-way split.  Interleaving the
  partial planes so stage 2 could read them as one ``[BK, SK]`` tile was tried
  and is worse (stage 2 13.3 -> 17.4 us).
* **Stage 2's traversal is CTA count only; coalescing is worth zero.**  Row
  runs are ``BM*2`` bytes, so it looks like a large BM must coalesce better --
  but at a FIXED grid the run length does not matter at all.  Measured on the
  traversal alone (T=4, cold, every variant reading the identical bytes): a
  32 B-run row block at grid 36 is 9.18 us and a fully CONTIGUOUS read at grid
  36 is 9.17; contiguous only improves at grid >= 72 (7.17) where the row block
  cannot follow.  And perfect 288 B p-runs are never better at any grid --
  11.02 us at grid 32, 9.07 at grid 128 -- i.e. never below the 9.15 that every
  row blocking already reaches.  So the strided traversal has a hard floor of
  ~9.15 us and 7.14 is unreachable for this access pattern; there is no
  transaction waste to remove, and BM should be chosen purely for grid size.
  That is why ``BM`` here is 8, below the MMA's 16-row minimum (``tl.dot`` pads
  M internally, which costs nothing measurable): it doubles the grid to
  ``T*18``.  BM=8/BK=512 measures 11.26 us against 13.25 for BM=16/BK=256 --
  one full quantisation level, and proof that ~2 us of stage 2 was never the
  traversal.  Splitting the j axis instead (grid T*S, partial planes, a third
  launch) measured 19.5 against 13.3.
* **PDL is worth exactly one level.**  ``gdc_launch_dependents()`` at the end of
  stage 1 and ``gdc_wait()`` before stage 2's k-loop, with ``launch_pdl`` on
  **stage 2 only**: 21.45 -> 19.44 us.  Stage 1 deliberately does not get the
  attribute -- with it, stage 1 would also be allowed to start before the
  harness's own flush kernel finished, which moves work outside the measured
  window instead of making it faster.

Numerics: both reductions and the affine run in fp32; only the two MMA operands
are fp16, matching what the reference feeds cuBLAS.  Weight packing, the fp32
biases, launch grids, the ``mod`` scratch buffer and both direct C launchers are
resolved on the first call (``__init__`` runs before the harness re-points
``p.data`` to fp16 and ``load_state_dict``s over it), so steady-state
``forward`` is two ``launch()`` calls and one ``torch.empty``.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._C import _cuda_getCurrentRawStream as _raw_stream
from torch._C import _cuda_getDevice as _cur_device

try:  # Triton 3.6+; the intrinsics are no-ops without launch_pdl anyway
    from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait
    _HAS_PDL = True
except Exception:  # noqa: BLE001 - older Triton: PDL simply stays off
    _HAS_PDL = False

from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear
from ..L1.silu import SiLU

# Padded M for the adaLN GEMV: T <= 6 in every capture and the MMA's minimum is
# 16 rows, so one tile covers every frame count.  Also the row stride of the
# `mod` scratch plane, which is why it is a plain constant.
_MT = 16
_MTC = tl.constexpr(_MT)


@triton.jit
def _adaln_gemv(C, WP, BIAS, OUT,
                T: tl.constexpr, K: tl.constexpr, N: tl.constexpr,
                BN: tl.constexpr, SK: tl.constexpr, PDL: tl.constexpr):
    """OUT[sk, t, n] = sum_{k in chunk sk} silu(c[t,k]) * A[n,k] (+ BIAS[n]).

    grid = (N/BN, SK).  ``WP`` is A packed ``[n_block, sk, K/SK, BN]`` so each
    CTA's whole slab is ONE contiguous run, loaded as a single tile with no
    k-loop, and reused for all T rows.

    No MMA.  The obvious form is ``tl.dot`` of a [16, K/SK] ``silu(c)`` operand
    against the [K/SK, BN] slab, but T <= 6 while the MMA's minimum is 16 rows,
    so that pads 10 dead rows AND -- because ``BN`` then cannot go below the
    MMA's minimum N=16 -- forces parallelism to come from split-K instead of
    from the n axis.  That matters because stage 2 pays for every extra split
    (one vector load per plane per k-iteration).  Unrolling the T rows as
    independent reduce-along-k's of a broadcast multiply removes both
    constraints: measured at SK=2 it is 9.25 us against 13.31 for the MMA form,
    and it lets BN drop to 4 so ``N/BN`` alone supplies 288 CTAs.
    """
    pid = tl.program_id(0)
    sk = tl.program_id(1)
    kc: tl.constexpr = K // SK
    ri = tl.arange(0, BN)
    kk = tl.arange(0, kc)
    w = tl.load(WP + (pid * SK + sk) * (kc * BN) + kk[:, None] * BN + ri[None, :])
    wf = w.to(tl.float32)
    n = pid * BN + ri
    b = tl.load(BIAS + n)
    cp = C + sk * kc
    for t in range(T):
        cv = tl.load(cp + t * K + kk).to(tl.float32)
        cv = cv * tl.sigmoid(cv)
        a = tl.sum(wf * cv[:, None], 0)
        if sk == 0:
            a += b
        tl.store(OUT + sk * (_MTC * N) + t * N + n, a)
    if PDL:
        # Producer signal: lets stage 2's CTAs be scheduled while this kernel's
        # tail drains, which hides stage 2's launch latency. Only stage 2 is
        # launched with the PDL attribute, so this kernel itself gains no early
        # start against whatever the caller queued before it.
        gdc_launch_dependents()


@triton.jit
def _final_proj(X, VT, LB, MOD, OUT,
                T: tl.constexpr, P: tl.constexpr, K: tl.constexpr,
                N: tl.constexpr, NM: tl.constexpr, ST: tl.constexpr,
                SJ: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr,
                NB: tl.constexpr, EPS: tl.constexpr, EXACT: tl.constexpr,
                SK: tl.constexpr, HR: tl.constexpr, PDL: tl.constexpr):
    """LayerNorm + (1+scale) modulation + final projection, one pass over x.

    ``acc`` accumulates ``A[p,n]`` from the raw input tile while ``ha``/``hb``
    accumulate the shifted tile and its square; the row statistics are reduced
    out of them once, after the loop.  The shift is the row's own first element,
    so ``s2/K - off^2`` cancels only the shifted mean rather than a potentially
    huge raw one.
    """
    pid = tl.program_id(0)
    t = pid // NB
    p = (pid % NB) * BM + tl.arange(0, BM)
    rn = tl.arange(0, N)
    xb = X + t * ST + p
    mb = MOD + t * NM
    if EXACT:
        shift = tl.load(xb).to(tl.float32)
    else:
        mp = p < P
        shift = tl.load(xb, mask=mp, other=0.0).to(tl.float32)
    if PDL:
        # As late as possible but strictly before the first read of MOD: the
        # row base addresses and the shift constant (both from x) are resolved
        # in the overlap window.
        gdc_wait()
    acc = tl.zeros((BM, N), tl.float32)
    # HR rows of hoisted mean/variance accumulator.  HR == BK keeps the whole
    # tile (no reduction inside the loop at all); HR == 0 reduces every
    # iteration; in between the tile is pre-folded HR-ways first, trading a
    # cheap same-thread fold for a smaller register footprint.
    if HR:
        ha = tl.zeros((HR, BM), tl.float32)
        hb = tl.zeros((HR, BM), tl.float32)
    else:
        ha = tl.zeros((BM,), tl.float32)
        hb = tl.zeros((BM,), tl.float32)
    for k0 in tl.range(0, K, BK):
        kk = k0 + tl.arange(0, BK)
        if EXACT:
            xt = tl.load(xb[None, :] + kk[:, None] * SJ)
        else:
            xt = tl.load(xb[None, :] + kk[:, None] * SJ, mask=mp[None, :],
                         other=0.0)
        d = xt.to(tl.float32) - shift[None, :]
        if HR == BK:
            ha += d
            hb += d * d
        elif HR:
            ha += tl.sum(tl.reshape(d, (BK // HR, HR, BM)), 0)
            hb += tl.sum(tl.reshape(d * d, (BK // HR, HR, BM)), 0)
        else:
            ha += tl.sum(d, 0)
            hb += tl.sum(d * d, 0)
        v = tl.load(VT + kk[:, None] * N + rn[None, :])
        g = tl.load(mb + kk)
        for s in tl.static_range(1, SK):
            g += tl.load(mb + s * (_MTC * NM) + kk)
        acc = tl.dot(tl.trans(xt), v * g.to(v.dtype)[:, None], acc)
    inv: tl.constexpr = 1.0 / K
    if HR:
        off = tl.sum(ha, 0) * inv
        var = tl.sum(hb, 0) * inv - off * off
    else:
        off = ha * inv
        var = hb * inv - off * off
    rstd = 1.0 / tl.sqrt(tl.maximum(var, 0.0) + EPS)
    bv = tl.load(mb + K + rn)
    gv = tl.load(mb + K + N + rn)
    for s in tl.static_range(1, SK):
        o2 = s * (_MTC * NM)
        bv += tl.load(mb + o2 + K + rn)
        gv += tl.load(mb + o2 + K + N + rn)
    o = ((acc - (shift + off)[:, None] * gv[None, :]) * rstd[:, None]
         + bv[None, :] + tl.load(LB + rn)[None, :])
    op = OUT + (t * P + p[:, None]) * N + rn[None, :]
    if EXACT:
        tl.store(op, o.to(OUT.dtype.element_ty))
    else:
        tl.store(op, o.to(OUT.dtype.element_ty), mask=mp[:, None])


# --------------------------------------------------------------------------
# Launch configs, measured per frame count (see ITERATIONS.md for the sweeps).
#   stage 1: (BN, SK, num_warps)
#   stage 2: (BM, BK, num_warps, num_stages, HR)
# Measured as a PAIR (the two stages are coupled through SK and share the window
# overhead), cold L2, PDL on.  The pair is exactly additive -- stage 1 alone
# 11.23 + stage 2 alone 11.26 - 5.12 for the one shared window = 17.37, which is
# what the pair measures -- so each half can be judged on its own bytes.
# Because a single draw of ONE config moved a whole 2.048 us level between runs
# (r1's own config: 19.38 us at T=4 in one process, 17.54 in the next), the two
# arms were measured INTERLEAVED in one process, 5 rounds each, medians:
#   T:              2      3      4      5      6
#   BM=16 BK=256   17.17  17.44  17.47  19.49  19.49   (r1)
#   BM= 8 BK=512   15.38  15.39  17.44  17.41  19.46   (here)
# One full level at T=2,3,5 and neutral at T=4,6; never worse.  BM=16/BK=128/
# w=2/HR=32 is joint-best everywhere too, so the win is the grid (T*18 instead
# of T*9), not the specific tile.  SK stays 1: with BN=8 the n axis alone gives
# 144 CTAs, so stage 1 needs no split and stage 2 pays nothing for one.  Stage 1
# alone is a level faster at (BN=8, SK=4, w=1)/grid 576 (9.25 us), but in the
# pair every SK>1 config landed in the same bin or worse at every T.
# --------------------------------------------------------------------------
_CFG1 = (8, 1, 4)
_CFG2 = (8, 512, 4, 3, 0)
# Programmatic dependent launch. `launch_pdl` is set on stage 2 ONLY, so stage
# 2's CTAs can be scheduled while stage 1's tail drains (worth one full 2.05 us
# quantisation level, measured) while stage 1 itself gains no early start
# relative to whatever the caller queued before it.
_PDL = os.environ.get("OFL_PDL", "1") != "0"

# Per-frame-count overrides, empty because the sweep found none worth having:
# the single config above is the best or joint-best at every captured T.  Kept
# as the hook a future retune would use.
_CFG1_T: dict[int, tuple] = {}
_CFG2_T: dict[int, tuple] = {}


def _env_cfg(name, default):
    raw = os.environ.get(name)
    return default if not raw else tuple(int(v) for v in raw.split(","))


class _Plan:
    """Everything a launch needs for one frame count, resolved once."""

    __slots__ = ("grid1x", "grid1y", "grid2", "run1", "run2", "pre1", "pre2",
                 "post1", "mid2", "post2", "mod", "out_shape", "dev")


class OasisFinalLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        # Submodule structure kept verbatim: the harness shares weights via
        # load_state_dict(baseline.state_dict()), so parameter names have to
        # match, and these are the fallback for any layout the fused path does
        # not cover.
        self.norm_final = LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.linear = Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.ModuleList(
            [
                SiLU(),
                Linear(hidden_size, 2 * hidden_size, bias=True),
            ]
        )

        self.hidden_size = int(hidden_size)
        self.out_features = int(patch_size * patch_size * out_channels)
        self.eps = 1e-6
        self._cfg1 = _env_cfg("OFL_CFG1", _CFG1)
        self._cfg2 = _env_cfg("OFL_CFG2", _CFG2)
        self._cfg1_t = {} if os.environ.get("OFL_CFG1") else _CFG1_T
        self._cfg2_t = {} if os.environ.get("OFL_CFG2") else _CFG2_T

        # Lazily prepared: __init__ runs before the harness casts p.data to
        # fp16 and copies the shared weights in, so nothing derived from the
        # parameters can be built here.
        self._packed = None       # (A packed, fp32 bias, V^T, linear bias fp32)
        self._plans: dict = {}
        self._sig = None
        self._ver = None
        self._devobj = None
        self._geom = None         # (H, W, P) the packing/plans were built for
        self._aw = self._ab = self._lw = self._lb = None

    # ------------------------------------------------------------------
    # host-side preparation
    # ------------------------------------------------------------------
    def _versions(self):
        """Cheap staleness guard for the packed weights.

        ``load_state_dict`` and the harness's ``p.data = p.data.to(fp16)`` both
        keep the Parameter object while changing its contents or dtype, so
        identity alone cannot see them.  ``_version`` bumps on every in-place
        write; four reads cost ~0.2 us against the 2.8 us launch they guard.
        """
        return (self._aw._version, self._ab._version, self._lw._version,
                self._lb._version)

    def _weight_sig(self):
        return (self._aw.data_ptr(), self._ab.data_ptr(), self._lw.data_ptr(),
                self._lb.data_ptr(), self._aw.dtype, self._lw.dtype)

    def _prepare(self, x):
        """Build the packed adaLN matrix and its fp32 bias; reset the plans.

        Returns False when the fused path cannot serve this module.
        """
        alin = self.adaLN_modulation[1]
        lin = self.linear
        aw, ab = alin.weight, alin.bias
        lw, lb = lin.weight, lin.bias
        K = self.hidden_size
        N = self.out_features
        bn, sk = self._cfg1[0], self._cfg1[1]
        if (ab is None or lb is None
                or N & (N - 1) or N < 16   # tl.arange / MMA need a pow2 N >= 16
                or aw.dtype is not x.dtype or lw.dtype is not x.dtype
                or tuple(aw.shape) != (2 * K, K) or tuple(lw.shape) != (N, K)
                or not aw.is_contiguous() or not lw.is_contiguous()
                or not ab.is_contiguous() or not lb.is_contiguous()
                or (K + 2 * N) % bn or K % sk
                or (K // sk) & (K // sk - 1)   # tl.arange needs a pow2 K/SK
                or K % self._cfg2[1]
                or (self._cfg2[4] and self._cfg2[1] % self._cfg2[4])):
            return False
        self._aw, self._ab, self._lw, self._lb = aw, ab, lw, lb
        f = torch.float32
        wsh = aw[:K].detach().to(f)          # shift rows, [j, k]
        wsc = aw[K:].detach().to(f)          # scale rows
        v = lw.detach().to(f)                # [N, K]
        absh = ab[:K].detach().to(f)
        absc = ab[K:].detach().to(f)
        # A = [Wscale ; V@Wshift ; V@Wscale], packed [n_block, sk, K/SK, BN] so
        # every CTA reads one contiguous slab.
        a32 = torch.cat([wsc, v @ wsh, v @ wsc])
        # V @ Wshift is a product of two weight matrices, so its entries are
        # ~K times larger than either; with pathological weights that can leave
        # fp16 range, and the MMA operand has to be fp16. Checked once, here.
        if not (torch.isfinite(a32).all().item()
                and a32.abs().max().item() < 6e4):
            self._packed = None
            return False
        a = a32.to(aw.dtype)
        ntot = a.shape[0]
        kc = K // sk
        a = a.view(ntot // bn, bn, sk, kc).permute(0, 2, 3, 1).contiguous()
        bias = torch.cat([absc + 1.0, v @ absh, v @ absc + v.sum(1)]).contiguous()
        self._packed = (a, bias, lw.detach().t().contiguous(),
                        lb.detach().to(f).contiguous())
        self._sig = self._weight_sig()
        self._ver = self._versions()
        self._devobj = x.device
        self._geom = (int(x.shape[2]), int(x.shape[3]),
                      int(x.shape[2]) * int(x.shape[3]))
        self._plans = {}
        return True

    def _build_plan(self, x, c, T):
        """Compile both kernels for this frame count and memoize their raw C
        launchers plus the invariant argument tuples.

        ``kernel[grid](...)`` re-binds, re-specializes and hashes every argument
        on each call -- 7.2 us of host time here against 2.8 us for the compiled
        kernel's own C entry point -- so we descend to the latter and pre-build
        the argument prefix around it, as the L1 LayerNorm does.  Returns None
        if any piece is not where we expect, in which case the fully supported
        (slower) path is used.
        """
        wp, bias, vt, lbf = self._packed
        K = self.hidden_size
        N = self.out_features
        ntot = K + 2 * N
        dev = x.get_device()
        bn, sk, w1 = self._cfg1
        bm, bk2, w2, s2, hr = self._cfg2_t.get(T, self._cfg2)
        _, _, pp = self._geom
        nb = -(-pp // bm)
        exact = (pp % bm == 0)
        st, sj = x.stride(1), x.stride(4)
        # One scratch plane per split, each _MT rows regardless of T, so the
        # buffer is shared by every frame count.
        mod = torch.empty(sk * _MT * ntot, dtype=torch.float32, device=x.device)
        out = torch.empty((1, T) + self._geom[:2] + (N,), dtype=x.dtype,
                          device=x.device)

        g1 = (ntot // bn, sk)
        pdl = _HAS_PDL and _PDL
        k1 = _adaln_gemv[g1](c, wp, bias, mod, T=T, K=K, N=ntot, BN=bn,
                             SK=sk, PDL=pdl, num_warps=w1)
        g2 = (T * nb,)
        k2 = _final_proj[g2](x, vt, lbf, mod, out, T=T, P=pp, K=K, N=N,
                             NM=ntot, ST=st, SJ=sj, BM=bm, BK=bk2, NB=nb,
                             EPS=self.eps, EXACT=exact, SK=sk, HR=hr,
                             PDL=pdl, num_warps=w2, num_stages=s2,
                             launch_pdl=pdl)
        for kern in (k1, k2):
            launcher = None if kern is None else kern.run
            if (launcher is None or getattr(launcher, "launch", None) is None
                    or getattr(launcher, "global_scratch_size", None) != 0
                    or getattr(launcher, "profile_scratch_size", None) != 0):
                return None
        if dev != _cur_device():
            return None
        # The compiled kernels are specialized on 16-byte pointer alignment.
        for t in (c, wp, bias, mod, x, vt, lbf, out):
            if t.data_ptr() & 15:
                return None

        def pieces(kern):
            launcher = kern.run
            return (launcher.launch,
                    (kern.function, launcher.launch_cooperative_grid,
                     launcher.launch_pdl, None, None, kern.packed_metadata,
                     None, None, None))

        p = _Plan()
        p.run1, p.pre1 = pieces(k1)
        p.run2, p.pre2 = pieces(k2)
        p.grid1x, p.grid1y = g1
        p.grid2 = g2[0]
        p.mod = mod
        p.dev = dev
        mp = mod.data_ptr()
        p.post1 = (wp.data_ptr(), bias.data_ptr(), mp, T, K, ntot, bn, sk, pdl)
        p.mid2 = (vt.data_ptr(), lbf.data_ptr(), mp)
        p.post2 = (T, pp, K, N, ntot, st, sj, bm, bk2, nb, self.eps, exact,
                   sk, hr, pdl)
        p.out_shape = out.shape
        return p

    # ------------------------------------------------------------------
    # fallback: the composed reference
    # ------------------------------------------------------------------
    def _reference(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        modulation = c
        for layer in self.adaLN_modulation:
            modulation = layer(modulation)
        shift, scale = modulation.chunk(2, dim=-1)
        while shift.dim() < x.dim():
            shift = shift.unsqueeze(-2)
            scale = scale.unsqueeze(-2)
        h = self.norm_final(x) * (1 + scale) + shift
        return self.linear(h)

    def _eligible(self, x, c):
        """The fused path assumes the captured channels-last view of x: a
        contiguous [T, K, P] buffer seen as [1, T, H, W, K]."""
        if not (x.is_cuda and x.dtype is torch.float16 and c.dtype is x.dtype
                and c.is_contiguous() and x.ndim == 5 and c.ndim == 3
                and x.shape[0] == 1 and c.shape[0] == 1
                and x.shape[4] == self.hidden_size
                and c.shape[2] == self.hidden_size
                and c.shape[1] == x.shape[1]
                and 0 < x.shape[1] <= _MT
                and not torch.is_grad_enabled()):
            return False
        h, w = int(x.shape[2]), int(x.shape[3])
        return (x.stride(3) == 1 and x.stride(2) == w and x.stride(4) == h * w
                and x.stride(1) == self.hidden_size * h * w)

    def _slow_path(self, x, c):
        if not self._eligible(x, c):
            return self._reference(x, c)
        if (self._packed is None or self._sig != self._weight_sig()
                or self._ver != self._versions()
                or self._geom[0] != x.shape[2] or self._geom[1] != x.shape[3]):
            if not self._prepare(x):
                return self._reference(x, c)
        T = x.shape[1]
        if T in self._plans:
            p = self._plans[T]
        else:
            p = self._build_plan(x, c, T)
            self._plans[T] = p
        if p is None or p.dev != x.get_device():
            return self._reference(x, c)
        return self._launch(p, x, c)

    def _launch(self, p, x, c):
        out = torch.empty(p.out_shape, dtype=torch.float16, device=self._devobj)
        st = _raw_stream(p.dev)
        p.run1(p.grid1x, p.grid1y, 1, st, *p.pre1, c.data_ptr(), *p.post1)
        p.run2(p.grid2, 1, 1, st, *p.pre2, x.data_ptr(), *p.mid2,
               out.data_ptr(), *p.post2)
        return out

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        p = self._plans.get(x.shape[1])
        if (p is not None and self._versions() == self._ver
                and x.get_device() == p.dev):
            return self._launch(p, x, c)
        return self._slow_path(x, c)
