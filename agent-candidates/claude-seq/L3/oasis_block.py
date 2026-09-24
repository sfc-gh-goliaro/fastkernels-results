"""Oasis spatio-temporal DiT block -- adaLN, both norms and every residual fused.

The captured workload is ``x: fp16[1, T, 9, 16, 1024]`` with ``T = 2..6`` and
``c: fp16[1, T, 1024]``: ``M = T * 144 <= 864`` rows of width 1024, 43 GFLOP at
``T = 6``.  That is ~50 us of tensor-core work, and the reference composition
takes 780 us: the operator is almost entirely *overhead*.  Measured here on the
best frozen-winner composition of the same ops at
``T = 6``: 56 kernel launches, 417 us of GPU time, 814 us of host time, scored
780 us.  Two costs drive it, both measured on this B200 inside the scorer's own
timing loop (L2 flush, start event, forward, end event):

* ~4.5 us of GPU command time per launch (an empty Triton kernel in a chain of
  16 measures 3.7 us; a 1.7 MB elementwise kernel measures 4.5 us), and
* ~7.7 us of host time per launch, of which the L2 flush queued ahead of the
  start event hides only the first ~60 us.

So the score is roughly ``max(gpu_time, host_time - 60us)`` and *launch count* is
the thing to minimise.  231 us of the baseline's 417 us of GPU time was pure
elementwise glue: ``_modulate``'s two ``repeat`` copies (12 KB each, 5.7 us
apiece -- all latency), its ``1 + scale``, its ``mul`` and its ``add``,
``_gate``'s ``repeat`` and ``mul``, and the residual ``add``, 36 launches in all.

This file keeps the module tree (the scorer shares weights through
``load_state_dict``) and rewrites the *composition* as five Triton kernels that
absorb all of that glue, leaving 18 launches:

``_adaln``
    ``silu(c) @ [Ws; Wt].T + [bs; bt]`` -- **both** modulations in one launch,
    with ``silu`` in the GEMM prologue and the ``1 + scale`` of the two scale
    chunks in the epilogue, writing a ``[12, R, 1024]`` buffer whose planes are
    the two ``chunk(6)`` splits.  ``R = B*T <= 6``, so this is a pure 25 MB
    weight stream: tiled ``[BLOCK_N, BLOCK_K]`` over the weight with the
    contraction as the contiguous axis and ``tl.dot(w, a.T)`` -- the natural
    ``[BLOCK_K, BLOCK_N]`` orientation of a GEMM leaves the weight read strided
    and measured 27 us against 11 us for this one.
``_norm_mod``
    LayerNorm over the last axis followed by ``x * (1 + scale) + shift``, one
    pass, reading ``x`` through arbitrary ``(row, pixel, channel)`` strides.  The
    captured ``x`` is **not** contiguous -- ``[.., .., 16, 1, 144]``, a
    ``[b, t, d, h, w]`` buffer viewed as ``[b, t, h, w, d]`` -- and tiling over
    ``(pixel, channel)`` instead of ``(row, channel)`` keeps those reads one
    32 B sector per 16 pixels instead of one per element.
``_add_norm_mod``
    ``x + gate * y`` (the residual) *and* the next LayerNorm + modulation in one
    launch: the three kernels between two GEMMs become one.  The tail residual
    reuses it with ``NORM=False``.
``_sattn``
    Spatial axial attention -- rotary from a ``[144, 32]`` cos/sin table, flash
    attention over the 144-pixel sequence, and both layout changes -- reading the
    qkv projection in place and writing the output projection's input in place.
    ``log2(e)`` is folded into the score scale so the inner loop uses ``exp2``.
``_gelu_tanh``
    The MLP activation, fp32 math as ATen evaluates it.

The nine plain GEMMs stay on torch's fp16 path (``mm``/``addmm`` with ``out=``, so
the bias rides cuBLASLt's epilogue and nothing is allocated per call): cuBLAS'
``nvjet_sm100_*_2cta_*`` kernels run these shapes ~1.8x faster than the best
``tl.dot`` formulation measured on them (12.7 us vs 23.7 us on 864x3072x1024),
which is also why the frozen L2 temporal winner delegates its own projections --
so the fusion is what this file writes and the plain GEMMs are handed over.
Temporal attention stays on that frozen L2 winner, which already collapses both
rotary chains, all four permute-copies and the attention into one kernel.

Per-call host work is a hot path too, so everything that does not depend on the
input pointers -- grids, block sizes, strides, the rotary tables, the stacked
adaLN weights, the transposed weight views, the 12 modulation planes, every
scratch buffer, and the specialised Triton binaries bound to their raw launchers
(:class:`_Launcher`) -- is resolved once per shape into a :class:`_Plan`.  What is
left per call is four pointer patches, one allocation and 18 launches.

Those 18 are then cut to 4 on the host.  Only three of them touch memory the
caller owns: ``_adaln`` reads ``c``, ``_norm_mod`` reads ``x`` (and republishes it
contiguously), and the tail residual writes the output.  Everything between them
addresses nothing but plan-owned buffers, so it is captured once per shape into a
CUDA graph and replayed as a single launch -- which is also worth ~1.5 us of GPU
command time on each of the 15 kernels inside it, because a graph node launches
cheaper than a driver call.  That one change took the geomean from 190 us to
125 us; it is the largest single win in this file, and it is only available
because the fusion above had already made the middle pointer-independent.

Numerics follow the reference op for op rather than approximating it.  The
reference LayerNorm promotes to fp32, so the reduction is fp32 and the result is
rounded to fp16 before the modulation; ``1 + scale``, ``x * (1+scale)``,
``+ shift``, ``gate * y`` and the residual add are each rounded to fp16 exactly
where torch rounds them; ``silu`` and ``gelu`` evaluate in fp32 and round once,
as ATen does; and the rotary runs in fp16 from a table built by ``torch.cos`` /
``torch.sin`` on the fp16 frequencies, because at ``max_freq = 256`` the axial
frequencies reach 402, where one fp16 ulp is 0.25 and ``cos`` of it is
effectively chaotic.  Deviation from the reference is 5.9e-3 at worst against a
``1e-2 + 1e-2|y|`` bound, with 100% of elements inside it -- the same order as
the frozen-winner composition's own fp16 noise.

Measured end to end against the reference composition, ``T = 2..6``:
104 / 116 / 120 / 128 / 132 us against 1.76 / 1.72 / 1.71 / 1.69 / 1.76 ms, a
14.5x geomean.  Of the ~132 us at ``T = 6``, ~76 us is the nine cuBLAS GEMMs,
~12 us the adaLN weight stream, and the rest the five kernels here -- so what is
left is mostly work that was measured to be faster delegated than rewritten.

Absolute numbers on this machine are only good to ~50%: ``bench`` fails to lock
the B200's clocks (its ``nvidia-smi -lgc`` needs a sudo that is not there), so
they float between 1155 and 1965 MHz while the GPU pool is shared, and a
*GPU*-bound candidate tracks that while a *host*-bound baseline does not -- the
same code scored 8.0-10.9x and 13.3-17.1x in runs twenty minutes apart.  Every
comparison quoted here is therefore an A/B taken back to back in one process.

What was tried and rejected, all measured on these shapes:

* Fusing the output projection with the gated residual and the next LayerNorm
  into one kernel.  ``N`` is the full row, so the tile has to be
  ``[BLOCK_M, 1024]``: that leaves 54 CTAs at ``M = 864``, each re-reading the
  whole 2 MB weight, and the best of 108 configs measured 60.6 us against 20 us
  for ``addmm`` plus the separate residual kernel.
* Overlapping the adaLN weight stream with the compute-bound spatial
  projections, by forking ten of its twelve planes onto a side stream inside the
  graph.  Capture succeeds and the result is correct, but a graph with a
  cross-stream fork replays far slower here -- the geomean went 125 us -> 196 us,
  swamping the ~8 us the overlap was worth.
* A Triton ``fc1`` with the activation in its epilogue, to drop the GELU pass:
  1.5-2x off cuBLAS on ``864x4096x1024``, which is more than the 6 us the pass
  costs.  The same measurement is why every plain GEMM here is delegated.

Anything the fast path does not cover -- a non-fp16/bf16 or non-CUDA input, a
layout whose ``(h, w)`` strides are not a plain row-major pair, an unaligned
input pointer, ``B*T > 16``, an unexpected head dim, an affine norm, a rotary
table that does not span the head, a weight that moved, or a call under
``enable_grad`` -- falls through to :meth:`_eager`, the reference forward kept
verbatim.
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

import triton
import triton.language as tl

from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear
from ..L1.oasis_rotary import OasisRotaryEmbedding
from ..L1.silu import SiLU
from ..L2.oasis_mlp import OasisMLP
from ..L2.oasis_spatial_axial_attention import OasisSpatialAxialAttention
from ..L2.oasis_temporal_axial_attention import OasisTemporalAxialAttention

_LOG2E = 1.4426950408889634
_DEBUG = bool(os.environ.get("FK_OASIS_BLOCK_DEBUG"))
_FAST_DTYPES = (torch.float16, torch.bfloat16)


# ###########################################################################
# Reference helpers (used by the eager fallback)
# ###########################################################################
def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    fixed_dims = [1] * len(shift.shape[1:])
    shift = shift.repeat(x.shape[0] // shift.shape[0], *fixed_dims)
    scale = scale.repeat(x.shape[0] // shift.shape[0], *fixed_dims)
    while shift.dim() < x.dim():
        shift = shift.unsqueeze(-2)
        scale = scale.unsqueeze(-2)
    return x * (1 + scale) + shift


def _gate(x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    fixed_dims = [1] * len(g.shape[1:])
    g = g.repeat(x.shape[0] // g.shape[0], *fixed_dims)
    while g.dim() < x.dim():
        g = g.unsqueeze(-2)
    return g * x


# ###########################################################################
# Kernel 1 -- silu + both adaLN projections + the ``1 + scale`` epilogue
# ###########################################################################
@triton.jit
def _adaln(C, W, B, OUT, R,
           DIM: tl.constexpr, RP: tl.constexpr, NTILE: tl.constexpr,
           BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, NPROG: tl.constexpr):
    """``OUT[j, r, d] = silu(C)[r] . W[j*DIM + d] + B[j*DIM + d]`` (+1 if scale).

    ``W`` is the two ``[6*DIM, DIM]`` adaLN weights stacked, so plane ``j`` is
    chunk ``j % 6`` of the spatial (``j < 6``) or temporal modulation.  Chunks 1
    and 4 are ``scale_msa`` / ``scale_mlp``; the reference immediately forms
    ``1 + scale`` in fp16, so that is folded in here and each of the four
    normalisation launches carries one fewer op.

    ``R`` rows against ``12*DIM`` columns is a GEMV, so the loop is written
    weight-major: the tile is ``[BLOCK_N, BLOCK_K]`` with the contraction
    contiguous, and ``tl.dot(w, a.T)`` accumulates ``[BLOCK_N, RP]``.  Padding
    ``R`` up to ``RP = 16`` for the MMA costs nothing -- the kernel is a pure
    weight stream -- while the conventional ``[BLOCK_K, BLOCK_N]`` weight tile
    leaves the read strided by ``DIM`` and measured 2.5x slower.

    This is the one kernel that reads ``c``, so it is also the one launch that
    cannot move inside the graph.  Splitting it -- the two planes the first
    LayerNorm needs issued here and the other ten forked onto a side stream
    *inside* the graph, where 21 MB of weight streaming would hide behind the
    compute-bound spatial projections -- was tried and abandoned: the capture
    succeeds and the arithmetic is right, but a graph with a cross-stream fork
    replays far slower here (the scored geomean went 125 us -> 196 us), which
    swamps the ~8 us the overlap was worth.
    """
    rr = tl.arange(0, RP)
    rk = tl.arange(0, BLOCK_K)
    mask_r = rr < R

    for tile in range(tl.program_id(0), NTILE, NPROG):
        rn = tile * BLOCK_N + tl.arange(0, BLOCK_N)
        acc = tl.zeros((BLOCK_N, RP), dtype=tl.float32)
        c_ptrs = C + rr[:, None] * DIM + rk[None, :]
        w_ptrs = W + rn[:, None] * DIM + rk[None, :]
        for _ in range(0, DIM // BLOCK_K):
            cv = tl.load(c_ptrs, mask=mask_r[:, None], other=0.0)
            # F.silu on a half tensor computes in fp32 and rounds once
            cf = cv.to(tl.float32)
            a = (cf * tl.sigmoid(cf)).to(cv.dtype)
            acc = tl.dot(tl.load(w_ptrs), tl.trans(a), acc)
            c_ptrs += BLOCK_K
            w_ptrs += BLOCK_K

        acc += tl.load(B + rn).to(tl.float32)[:, None]
        val = acc.to(OUT.dtype.element_ty)
        j = (tile * BLOCK_N) // DIM
        six = j % 6
        if (six == 1) or (six == 4):
            val = (1.0 + val.to(tl.float32)).to(OUT.dtype.element_ty)
        tl.store(OUT + j * R * DIM + rr[None, :] * DIM + (rn % DIM)[:, None],
                 val, mask=mask_r[None, :])


# ###########################################################################
# Kernel 2 -- LayerNorm + modulation, over strided or contiguous x
# ###########################################################################
@triton.jit
def _norm_mod(X, SH, SC, Y, XC, M, HW, EPS, sxr, sxp, sxd,
              N: tl.constexpr, BLOCK_P: tl.constexpr):
    """``Y[m] = layernorm(X[m]) * SC[m // HW] + SH[m // HW]``, and ``XC = X``.

    ``X`` is addressed as ``r * sxr + p * sxp + d * sxd`` with ``r = m // HW``
    the ``(batch, time)`` row and ``p = m % HW`` the pixel, which covers both the
    contiguous 5-D layout and the captured channel-major one.  ``SC`` already
    holds ``1 + scale``.

    The row is in registers anyway, so it is also written out contiguously: this
    is the only kernel that reads the caller's ``x``, which is what lets
    everything between here and the last residual be captured in a CUDA graph,
    and it turns the first residual's read of ``x`` into a coalesced one.
    """
    rm = tl.program_id(0) * BLOCK_P + tl.arange(0, BLOCK_P)
    mask = rm < M
    r = rm // HW
    p = rm % HW
    d = tl.arange(0, N)

    xraw = tl.load(X + r[:, None] * sxr + p[:, None] * sxp + d[None, :] * sxd,
                   mask=mask[:, None], other=0.0)
    tl.store(XC + rm[:, None] * N + d[None, :], xraw, mask=mask[:, None])
    xv = xraw.to(tl.float32)
    mu = tl.sum(xv, 1) / N
    xc = xv - mu[:, None]
    var = tl.sum(xc * xc, 1) / N
    xh = (xc * (1.0 / tl.sqrt(var + EPS))[:, None]).to(Y.dtype.element_ty)

    moff = r[:, None] * N + d[None, :]
    sc = tl.load(SC + moff, mask=mask[:, None], other=0.0)
    sh = tl.load(SH + moff, mask=mask[:, None], other=0.0)
    yv = (xh.to(tl.float32) * sc.to(tl.float32)).to(Y.dtype.element_ty)
    yv = (yv.to(tl.float32) + sh.to(tl.float32)).to(Y.dtype.element_ty)
    tl.store(Y + rm[:, None] * N + d[None, :], yv, mask=mask[:, None])


# ###########################################################################
# Kernel 3 -- gated residual, then the next LayerNorm + modulation
# ###########################################################################
@triton.jit
def _add_norm_mod(X, YIN, G, SH, SC, XO, YO, M, HW, EPS, sxr, sxp, sxd,
                  N: tl.constexpr, BLOCK_P: tl.constexpr, NORM: tl.constexpr):
    """``XO = X + G[m // HW] * YIN`` and, with ``NORM``, ``YO = normmod(XO)``."""
    rm = tl.program_id(0) * BLOCK_P + tl.arange(0, BLOCK_P)
    mask = rm < M
    r = rm // HW
    p = rm % HW
    d = tl.arange(0, N)

    x = tl.load(X + r[:, None] * sxr + p[:, None] * sxp + d[None, :] * sxd,
                mask=mask[:, None], other=0.0)
    y = tl.load(YIN + rm[:, None] * N + d[None, :], mask=mask[:, None], other=0.0)
    g = tl.load(G + r[:, None] * N + d[None, :], mask=mask[:, None], other=0.0)
    # gate, then residual, each rounded where torch rounds it
    gated = (g.to(tl.float32) * y.to(tl.float32)).to(x.dtype)
    xo = (x.to(tl.float32) + gated.to(tl.float32)).to(x.dtype)
    tl.store(XO + rm[:, None] * N + d[None, :], xo, mask=mask[:, None])

    if NORM:
        xv = xo.to(tl.float32)
        mu = tl.sum(xv, 1) / N
        xc = xv - mu[:, None]
        var = tl.sum(xc * xc, 1) / N
        xh = (xc * (1.0 / tl.sqrt(var + EPS))[:, None]).to(x.dtype)
        moff = r[:, None] * N + d[None, :]
        sc = tl.load(SC + moff, mask=mask[:, None], other=0.0)
        sh = tl.load(SH + moff, mask=mask[:, None], other=0.0)
        yv = (xh.to(tl.float32) * sc.to(tl.float32)).to(x.dtype)
        yv = (yv.to(tl.float32) + sh.to(tl.float32)).to(x.dtype)
        tl.store(YO + rm[:, None] * N + d[None, :], yv, mask=mask[:, None])


# ###########################################################################
# Kernel 4 -- spatial axial attention: rotary + flash attention + layouts
# ###########################################################################
@triton.jit
def _sattn(QKV, COS, SIN, O, QK_SCALE,
           S: tl.constexpr, D: tl.constexpr, HD: tl.constexpr,
           BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """grid = ``(cdiv(S, BLOCK_M), B*T, heads)``.

    ``QKV`` is ``[B*T*S, 3*HD]``; row ``bt * S + p``, column
    ``part * HD + head * D + d``.  ``O`` is ``[B*T*S, HD]``, already the layout
    the output projection consumes, so neither ``permute`` copy happens.
    ``COS``/``SIN`` are ``[S, D/2]``: the reference's ``repeat_interleave(2)``
    makes the two halves of each adjacent pair share an angle, so only the
    distinct ones are stored and the pair is rotated in-register through
    ``tl.split`` / ``tl.join``.
    """
    pid_m = tl.program_id(0)
    bt = tl.program_id(1)
    h = tl.program_id(2)
    d2: tl.constexpr = D // 2
    base = bt * S * (3 * HD) + h * D

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rd = tl.arange(0, D)
    rh = tl.arange(0, d2)
    mask_m = rm < S

    q = tl.load(QKV + base + rm[:, None] * (3 * HD) + rd[None, :],
                mask=mask_m[:, None], other=0.0)
    ct = tl.load(COS + rm[:, None] * d2 + rh[None, :], mask=mask_m[:, None], other=0.0)
    st = tl.load(SIN + rm[:, None] * d2 + rh[None, :], mask=mask_m[:, None], other=0.0)
    q0, q1 = tl.split(tl.reshape(q, (BLOCK_M, d2, 2)))
    q = tl.reshape(tl.join(q0 * ct - q1 * st, q1 * ct + q0 * st), (BLOCK_M, D))

    acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)
    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for n0 in range(0, tl.cdiv(S, BLOCK_N)):
        rn = n0 * BLOCK_N + tl.arange(0, BLOCK_N)
        nm = rn < S
        k = tl.load(QKV + base + HD + rn[:, None] * (3 * HD) + rd[None, :],
                    mask=nm[:, None], other=0.0)
        ck = tl.load(COS + rn[:, None] * d2 + rh[None, :], mask=nm[:, None], other=0.0)
        sk = tl.load(SIN + rn[:, None] * d2 + rh[None, :], mask=nm[:, None], other=0.0)
        k0, k1 = tl.split(tl.reshape(k, (BLOCK_N, d2, 2)))
        k = tl.reshape(tl.join(k0 * ck - k1 * sk, k1 * ck + k0 * sk), (BLOCK_N, D))

        qk = tl.dot(q, tl.trans(k)) * QK_SCALE
        if S % BLOCK_N != 0:
            qk = tl.where(nm[None, :], qk, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v = tl.load(QKV + base + 2 * HD + rn[:, None] * (3 * HD) + rd[None, :],
                    mask=nm[:, None], other=0.0)
        acc = tl.dot(p.to(v.dtype), v, acc)
        m_i = m_new

    acc = acc / l_i[:, None]
    tl.store(O + bt * S * HD + h * D + rm[:, None] * HD + rd[None, :],
             acc.to(O.dtype.element_ty), mask=mask_m[:, None])


# ###########################################################################
# Kernel 5 -- tanh-approximate GELU
# ###########################################################################
@triton.jit
def _gelu_tanh(X, Y, NEL, BLOCK: tl.constexpr):
    """ATen's tanh-approximate GELU, rewritten around one ``ex2.approx``.

    ``0.5 x (1 + tanh(z)) == x * sigmoid(2z)``, so with
    ``z = sqrt(2/pi) (x + 0.044715 x^3)`` the whole activation is
    ``x / (1 + exp2(-2 z log2(e)))`` -- a single hardware exponential instead of
    libdevice's software ``tanhf``, which measured 15.4 us against 12.0 us here
    (a plain ``copy_`` of the same 14 MB is 11.3 us, so this is 0.7 us over the
    memory floor).  Accuracy is unchanged where it matters: the deviation from a
    float64 GELU is 1.00e-3 against ``tanhf``'s 9.7e-4, i.e. both are one fp16
    ulp at these magnitudes, and 86% of elements come out bit-identical to
    ``F.gelu(approximate="tanh")``.
    """
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = o < NEL
    x = tl.load(X + o, mask=m, other=0.0).to(tl.float32)
    u = -2.3016759014368986 * (x + 0.044715 * x * x * x)
    tl.store(Y + o, (x / (1.0 + tl.math.exp2(u))).to(Y.dtype.element_ty), mask=m)


# ###########################################################################
# Cheap repeated launches
# ###########################################################################
class _Launcher:
    """One Triton kernel, specialised once and pre-bound to its argument row.

    ``kernel[grid](*args)`` re-binds every argument, recomputes the
    specialisation key and re-looks-up the compile cache on each call -- ~4 us of
    host time whose result is constant once the shape is fixed.  It is resolved
    once here and the raw C launcher is called afterwards with exactly the row
    ``JITFunction.run`` would have built; :meth:`set` patches the few slots whose
    tensor changes per call (all torch allocations are 512 B aligned, so the
    baked pointer-alignment specialisation still holds).  Any deviation in that
    private ABI is caught on the first call and demotes the launcher to the
    public path for good.
    """

    __slots__ = ("_jit", "_grid", "_args", "_row", "_run", "_fast", "_kw", "_base")

    def __init__(self, jit_fn, grid, args, num_warps=4, num_stages=3):
        self._jit = jit_fn
        self._grid = tuple(grid)
        self._args = list(args)
        self._kw = {"num_warps": num_warps, "num_stages": num_stages}
        self._fast = False
        compiled = jit_fn[self._grid](*args, **self._kw)
        try:
            from triton import knobs
            g = self._grid + (1, 1)
            self._run = compiled.run
            self._base = 9
            self._row = [g[0], g[1], g[2], 0, compiled.function,
                         compiled.packed_metadata, None,
                         knobs.runtime.launch_enter_hook,
                         knobs.runtime.launch_exit_hook, *args]
            self._fast = True
        except Exception:
            pass

    def set(self, i, v):
        self._args[i] = v
        if self._fast:
            self._row[self._base + i] = v

    def __call__(self, stream):
        if self._fast:
            self._row[3] = stream
            try:
                self._run(*self._row)
                return
            except Exception:
                self._fast = False
        self._jit[self._grid](*self._args, **self._kw)


# ###########################################################################
# Launch configuration
# ###########################################################################
# (BLOCK_N, BLOCK_K, num_warps, num_stages, persistent) for _adaln.
_ADALN_CFG = (128, 128, 8, 4, False)
# (BLOCK_P, num_warps, num_stages) for the two normalisation kernels.
_NORM_CFG = (8, 8, 2)
# (BLOCK_M, BLOCK_N, num_warps, num_stages) for _sattn.
_SATTN_CFG = (16, 32, 1, 3)
# (BLOCK, num_warps) for _gelu_tanh.
_GELU_CFG = (2048, 4)
# rows of ``c`` the adaLN MMA is padded to
_RP = 16


class _Plan:
    """Everything about one input shape that does not depend on the pointers.

    ``_norm_mod`` is the only kernel that reads the caller's ``x`` and ``_adaln``
    the only one that reads ``c``; the last residual is the only one that writes
    the caller's output.  Everything in between touches nothing but plan-owned
    buffers, so it is captured once into a CUDA graph and replayed as a single
    launch -- 15 of the 18 launches, and with them ~110 us of the ~190 us of host
    dispatch this block would otherwise spend per call (a ``cuLaunchKernelEx``
    alone measures ~5 us of host time on this machine).  Capture is best-effort:
    if it fails for any reason the same closure runs eagerly.

    The scratch buffers are per-shape and reused, so two concurrent ``forward``
    calls for the same shape on the same module (two threads, or two streams)
    would share them, and a captured graph makes that explicit rather than worse.
    That is the usual workspace trade-off for an inference kernel; the eager path
    has no such constraint if it is ever needed.
    """

    __slots__ = ("dtype", "device", "flat", "out_shape", "cflat", "run",
                 "l_adaln", "l_norm", "l_anm", "l_sattn", "l_gelu", "keys",
                 "hold", "graph")

    def __init__(self, blk, x, c):
        B, T, Hh, W, dim = x.shape
        dev, dt = x.device, x.dtype
        self.dtype, self.device = dt, dev
        HW = Hh * W
        R = B * T
        M = R * HW
        self.out_shape = (B, T, Hh, W, dim)
        self.flat = (M, dim)
        self.cflat = (R, dim)
        eps = blk.s_norm1.eps
        heads = blk.s_attn.heads
        D = dim // heads
        nsm = torch.cuda.get_device_properties(dev).multi_processor_count

        # strides of the (possibly channel-major) input and of a plain [M, dim]
        sx = (x.stride(1), x.stride(3), x.stride(4))
        cx = (HW * dim, dim, 1)

        # adaLN weights, the two projections stacked so one launch covers both
        sa, ta = blk.s_adaLN_modulation[1], blk.t_adaLN_modulation[1]
        wcat = torch.cat((sa.weight, ta.weight), 0).contiguous()
        bcat = torch.cat((sa.bias, ta.bias), 0).contiguous()

        # spatial rotary tables, built by the reference so the fp16 rounding of
        # the max_freq=256 frequencies is reproduced rather than approximated
        freqs = blk.s_attn.rotary_emb.get_axial_freqs(Hh, W).reshape(HW, D)
        cos = freqs.cos()[:, ::2].contiguous().to(dt)
        sin = freqs.sin()[:, ::2].contiguous().to(dt)

        # scratch, allocated once per shape
        mod = torch.empty((12, R, dim), device=dev, dtype=dt)
        mp = [mod[i] for i in range(12)]
        y = torch.empty((M, dim), device=dev, dtype=dt)
        qkv = torch.empty((M, 3 * dim), device=dev, dtype=dt)
        ctx = torch.empty((M, dim), device=dev, dtype=dt)
        proj = torch.empty((M, dim), device=dev, dtype=dt)
        xb = [torch.empty((M, dim), device=dev, dtype=dt) for _ in range(4)]
        nh = blk.s_mlp.fc1.weight.shape[0]
        h0 = torch.empty((M, nh), device=dev, dtype=dt)
        h1 = torch.empty((M, nh), device=dev, dtype=dt)

        # transposed weight views, so the hot path never calls .t()
        wqkv_s = blk.s_attn.to_qkv.weight.t()
        wo_s, bo_s = blk.s_attn.to_out.weight.t(), blk.s_attn.to_out.bias
        s1, sb1 = blk.s_mlp.fc1.weight.t(), blk.s_mlp.fc1.bias
        s2, sb2 = blk.s_mlp.fc2.weight.t(), blk.s_mlp.fc2.bias
        t1, tb1 = blk.t_mlp.fc1.weight.t(), blk.t_mlp.fc1.bias
        t2, tb2 = blk.t_mlp.fc2.weight.t(), blk.t_mlp.fc2.bias

        bn, bk, aw, ast, apers = _ADALN_CFG
        ntile = (12 * dim) // bn
        nprog = min(nsm, ntile) if apers else ntile
        self.l_adaln = _Launcher(
            _adaln, (nprog,), (c.view(R, dim), wcat, bcat, mod, R,
                               dim, _RP, ntile, bn, bk, nprog), aw, ast)

        bp, nw, ns = _NORM_CFG
        gnorm = (triton.cdiv(M, bp),)
        self.l_norm = _Launcher(
            _norm_mod, gnorm,
            (x, mp[0], mp[1], y, xb[0], M, HW, eps, sx[0], sx[1], sx[2], dim, bp),
            nw, ns)

        anm = []
        for i, norm in enumerate((True, True, True, False)):
            j = 2 + 3 * i
            anm.append(_Launcher(
                _add_norm_mod, gnorm,
                (xb[i], proj, mp[j], mp[j + 1] if norm else mp[j],
                 mp[j + 2] if norm else mp[j], xb[i + 1] if norm else y, y,
                 M, HW, eps, cx[0], cx[1], cx[2], dim, bp, norm), nw, ns))
        self.l_anm = anm

        smb, snb, sw, sst = _SATTN_CFG
        self.l_sattn = _Launcher(
            _sattn, (triton.cdiv(HW, smb), R, heads),
            (qkv, cos, sin, ctx, D ** -0.5 * _LOG2E, HW, D, dim, smb, snb), sw, sst)

        gb, gw = _GELU_CFG
        self.l_gelu = _Launcher(
            _gelu_tanh, (triton.cdiv(h0.numel(), gb),), (h0, h1, h0.numel(), gb), gw)

        mm, addmm = torch.mm, torch.addmm
        t_attn = blk.t_attn
        five = self.out_shape
        l_sattn, l_gelu = self.l_sattn, self.l_gelu
        a0, a1, a2, a3 = anm
        tout = []

        def body(stream):
            """The pointer-independent middle: spatial attention, both MLPs,
            temporal attention and the three inner residuals."""
            mm(y, wqkv_s, out=qkv)
            l_sattn(stream)
            addmm(bo_s, ctx, wo_s, out=proj)
            a0(stream)

            addmm(sb1, y, s1, out=h0)
            l_gelu(stream)
            addmm(sb2, h1, s2, out=proj)
            a1(stream)

            o = t_attn(y.view(five)).view(M, dim)
            tout[:] = (o,)      # the graph baked its pointer; keep it alive
            a2.set(1, o)
            a2(stream)

            addmm(tb1, y, t1, out=h0)
            l_gelu(stream)
            addmm(tb2, h1, t2, out=proj)

        self.graph = None
        self.hold = (wcat, bcat, cos, sin, mod, mp, y, qkv, ctx, proj, xb, h0, h1,
                     wqkv_s, wo_s, s1, s2, t1, t2, body, tout)
        self.keys = blk._param_keys()

        graph = _capture(body, dev)
        self.graph = graph
        replay = graph.replay if graph is not None else None
        l_adaln, l_norm = self.l_adaln, self.l_norm

        def run(xin, cin, out, stream):
            l_adaln.set(0, cin)
            l_adaln(stream)
            l_norm.set(0, xin)
            l_norm(stream)
            if replay is None:
                body(stream)
            else:
                replay()
            a3.set(5, out)
            a3(stream)

        self.run = run


def _capture(body, dev):
    """Warm up *body* on a side stream, then capture it into a CUDA graph.

    The warm-up is what lets the capture be clean: it drives every lazy
    initialisation (cuBLASLt handles and workspaces, the frozen temporal
    attention's own per-shape plan and its Triton binary) before the capture
    stream opens, so the only allocations inside the capture are the intermediate
    the temporal projection returns -- which comes from the graph's private pool
    and is held for the graph's lifetime.  Best-effort: on any failure the caller
    keeps running *body* eagerly, which costs ~65 us per call but is otherwise the
    same code -- set ``FK_OASIS_BLOCK_DEBUG=1`` to see why a capture was refused
    rather than having to infer it from the latency.
    """
    try:
        side = torch.cuda.Stream(device=dev)
        side.wait_stream(torch.cuda.current_stream(dev))
        with torch.cuda.stream(side):
            for _ in range(3):
                body(torch._C._cuda_getCurrentRawStream(dev.index))
        torch.cuda.current_stream(dev).wait_stream(side)
        torch.cuda.synchronize(dev)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            body(torch._C._cuda_getCurrentRawStream(dev.index))
        return graph
    except Exception:
        if _DEBUG:
            import traceback
            traceback.print_exc(file=sys.stderr)
        try:
            torch.cuda.synchronize(dev)
        except Exception:
            pass
        return None


class SpatioTemporalDiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        *,
        mlp_ratio: float = 4.0,
        is_causal: bool = True,
        spatial_rotary_emb: OasisRotaryEmbedding,
        temporal_rotary_emb: OasisRotaryEmbedding,
    ):
        super().__init__()
        self.s_norm1 = LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.s_attn = OasisSpatialAxialAttention(
            hidden_size,
            heads=num_heads,
            dim_head=hidden_size // num_heads,
            rotary_emb=spatial_rotary_emb,
        )
        self.s_norm2 = LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.s_mlp = OasisMLP(
            hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            approximate_tanh=True,
        )
        self.s_adaLN_modulation = nn.Sequential(
            SiLU(),
            Linear(hidden_size, 6 * hidden_size, bias=True),
        )

        self.t_norm1 = LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.t_attn = OasisTemporalAxialAttention(
            hidden_size,
            heads=num_heads,
            dim_head=hidden_size // num_heads,
            rotary_emb=temporal_rotary_emb,
            is_causal=is_causal,
        )
        self.t_norm2 = LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.t_mlp = OasisMLP(
            hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            approximate_tanh=True,
        )
        self.t_adaLN_modulation = nn.Sequential(
            SiLU(),
            Linear(hidden_size, 6 * hidden_size, bias=True),
        )
        self._plans: dict = {}

    # -- plan validity -------------------------------------------------------
    def _params(self):
        return (self.s_adaLN_modulation[1].weight, self.s_adaLN_modulation[1].bias,
                self.t_adaLN_modulation[1].weight, self.t_adaLN_modulation[1].bias,
                self.s_attn.to_qkv.weight, self.s_attn.to_out.weight,
                self.s_attn.to_out.bias, self.s_attn.rotary_emb.freqs,
                self.s_mlp.fc1.weight, self.s_mlp.fc1.bias,
                self.s_mlp.fc2.weight, self.s_mlp.fc2.bias,
                self.t_mlp.fc1.weight, self.t_mlp.fc1.bias,
                self.t_mlp.fc2.weight, self.t_mlp.fc2.bias)

    def _param_keys(self):
        """Pointer + version of every parameter a plan bakes in, so an in-place
        weight update or a ``load_state_dict(assign=True)`` invalidates it.

        Two of the bakings are *copies* -- the stacked adaLN weight and the rotary
        cos/sin tables -- which a version bump is the only way to notice, and the
        rest are raw pointers inside launch rows and transposed views, which a
        reassigned ``p.data`` would leave dangling.  Checking all sixteen costs
        ~13 us of host time per call; that is affordable only because the graph
        left the forward GPU-bound by 2.7x, and it is the reason this guard is
        stricter than the usual two-pointer check."""
        return tuple((p.data_ptr(), p._version) for p in self._params())

    def _apply(self, *args, **kwargs):
        self._plans.clear()
        return super()._apply(*args, **kwargs)

    def _load_from_state_dict(self, *args, **kwargs):
        self._plans.clear()
        return super()._load_from_state_dict(*args, **kwargs)

    # -- plan construction ---------------------------------------------------
    def _build_plan(self, x: torch.Tensor, c: torch.Tensor):
        if x.ndim != 5 or c.ndim != 3 or not x.is_cuda:
            return None
        if x.dtype not in _FAST_DTYPES or c.dtype is not x.dtype:
            return None
        if torch.is_grad_enabled():
            return None
        B, T, Hh, W, dim = x.shape
        if c.shape != (B, T, dim) or not c.is_contiguous():
            return None
        R = B * T
        if R == 0 or R > _RP or Hh * W == 0:
            return None
        # the kernels address x as r * sxr + p * sxp + d * sxd
        if x.stride(2) != W * x.stride(3):
            return None
        if B != 1 and x.stride(0) != T * x.stride(1):
            return None

        heads = self.s_attn.heads
        if heads <= 0 or dim % heads:
            return None
        D = dim // heads
        if D % 2 or D > 256 or D < 16:
            return None
        for p in self._params():
            if p is None or p.dtype is not x.dtype or not p.is_contiguous():
                return None
        for ln in (self.s_norm1, self.s_norm2, self.t_norm1, self.t_norm2):
            if ln.weight is not None or ln.bias is not None:
                return None
            if ln.normalized_shape != (dim,) or ln.eps != self.s_norm1.eps:
                return None
        if self.s_adaLN_modulation[1].weight.shape != (6 * dim, dim):
            return None
        if self.t_adaLN_modulation[1].weight.shape != (6 * dim, dim):
            return None
        if self.s_attn.to_qkv.weight.shape != (3 * dim, dim):
            return None
        if self.s_attn.to_qkv.bias is not None:
            return None
        if self.s_attn.to_out.weight.shape != (dim, dim):
            return None
        if self.s_mlp.fc1.weight.shape[1] != dim or self.s_mlp.fc2.weight.shape[0] != dim:
            return None
        if self.s_mlp.fc1.weight.shape[0] != self.s_mlp.fc2.weight.shape[1]:
            return None
        if self.t_mlp.fc1.weight.shape != self.s_mlp.fc1.weight.shape:
            return None
        if self.t_mlp.fc2.weight.shape != self.s_mlp.fc2.weight.shape:
            return None
        if dim % _ADALN_CFG[0] or dim % _ADALN_CFG[1]:
            return None
        freqs = self.s_attn.rotary_emb.get_axial_freqs(Hh, W)
        if freqs.shape[-1] != D or freqs.numel() != Hh * W * D:
            return None
        try:
            return _Plan(self, x, c)
        except Exception:
            return None

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        key = (x.shape, x.stride(), c.shape, x.dtype)
        plan = self._plans.get(key)
        if plan is None:
            if key in self._plans:
                return self._eager(x, c)
            plan = self._build_plan(x, c)
            self._plans[key] = plan
            if plan is None:
                return self._eager(x, c)
        elif (torch.is_grad_enabled() or not c.is_contiguous()
                or ((x.data_ptr() | c.data_ptr()) & 15)
                or plan.keys != self._param_keys()):
            # a moved weight, an unaligned view or a graph-recording call: the
            # baked launch rows no longer describe this call
            if plan.keys != self._param_keys():
                self._plans.clear()
            return self._eager(x, c)
        out = torch.empty(plan.flat, device=plan.device, dtype=plan.dtype)
        plan.run(x, c.view(plan.cflat), out,
                 torch._C._cuda_getCurrentRawStream(x.device.index))
        return out.view(plan.out_shape)

    # -- reference path ------------------------------------------------------
    def _eager(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        s_shift_msa, s_scale_msa, s_gate_msa, s_shift_mlp, s_scale_mlp, s_gate_mlp = (
            self.s_adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + _gate(self.s_attn(_modulate(self.s_norm1(x), s_shift_msa, s_scale_msa)), s_gate_msa)
        x = x + _gate(self.s_mlp(_modulate(self.s_norm2(x), s_shift_mlp, s_scale_mlp)), s_gate_mlp)

        t_shift_msa, t_scale_msa, t_gate_msa, t_shift_mlp, t_scale_mlp, t_gate_mlp = (
            self.t_adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + _gate(self.t_attn(_modulate(self.t_norm1(x), t_shift_msa, t_scale_msa)), t_gate_msa)
        x = x + _gate(self.t_mlp(_modulate(self.t_norm2(x), t_shift_mlp, t_scale_mlp)), t_gate_mlp)
        return x
