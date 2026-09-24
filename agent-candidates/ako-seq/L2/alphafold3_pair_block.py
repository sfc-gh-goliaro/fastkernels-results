"""PairBlock for AlphaFold3 -- the whole block as ten fused Triton kernels.

Shared pair-representation update block used by PairFormer, MSA module,
and template embedder. Sequence: TriMulOut -> TriMulIn -> TriAttStart ->
TriAttEnd -> SwiGLUTransition.

Reference: openfold3/core/model/latent/base_blocks.py PairBlock

What this operator actually costs
--------------------------------
The captured shape is ``z: bf16[1, 16, 16, 128]`` with ``pair_mask[1, 16, 16]``:
256 pair rows, 64 KiB of activations, ~146 M MACs for the whole block -- a
fraction of a microsecond of B200 tensor-core work. The eager composition issues
**109 kernels** and spends ~1.5 ms of host time getting them onto the GPU. The
arithmetic is irrelevant; what is left, once the launches are gone, is 1.1 MiB of
*weights* and a chain of dependent launches.

Three measurements set the target (all under the harness's own conditions; see
ITERATIONS.md):

* The harness enqueues a 253 MiB L2 flush before its start event. That fill costs
  72 us of device time, so the GPU runs ~70 us behind the host and the measured
  window is **pure device time** -- host work is free up to that budget.
* A trivial kernel's device time is ~1.55 us at any grid from 8 to 256 programs.
  Back-to-back that is ~3.3 us per launch, or **~1.0 us with programmatic
  dependent launch**. So the floor for an N-kernel block is ~N us, not ~3N.
* First touch of a kernel's own weights, with L2 cold, costs ~+3 us. Paying that
  eleven times is worth more than the arithmetic.

Why splitting beats fusing here
-------------------------------
The obvious move is to collapse the block into as few kernels as possible. A
first version did exactly that -- five kernels, one per cross-row dependency,
every row-local stage riding in an epilogue -- and scored 13.5x, because those
kernels ran at **7-16 us each**. A program that owns a complete pair row must
load a complete weight matrix (up to 160 KiB for a concatenated projection), and
with only 8-16 programs there is nothing resident to hide that latency behind.
Measured on the tri-mul input projection (``[256,128] @ [128,640]``):

| shape of the grid                                   | device time |
|-----------------------------------------------------|-------------|
| 8 programs, each loading all 160 KiB of weights     | 6.5 us      |
| same, weight loop software-pipelined (num_stages=3) | 4.3 us      |
| 80 programs, each loading one 32 KiB column slice   | **2.15 us** |
| cuBLAS on the same GEMM                             | 2.4 us      |

Splitting the *output columns* across programs is what reaches the floor: it cuts
per-program weight bytes by the split factor, and every program's loads are
independent so they are all in flight at once. So the design is column-split
kernels plus PDL, not maximal fusion -- 33.7x against the 5-kernel version's
13.5x.

Where the seams are
-------------------
A layer norm reduces over the channel axis, so a program can only normalize rows
it holds *completely*; a column-split producer therefore cannot be fused with the
layer norm that consumes it. That makes each of the block's seven layer norms a
natural kernel boundary. Two of them are crossed anyway, by having every column
task redundantly recompute the whole residual row it needs (``_epi_tm_proj``,
``_epi_ta_proj``): that costs one extra ``[CH, C]`` weight slice per program and
saves a round trip through memory plus a launch. Crossing the remaining seams the
same way would need the full 256 KiB of a SwiGLU weight or an all-rows einsum
operand per program, which is back to the 5-kernel failure mode.

That redundancy is the block's single most expensive line of code: ablating the
recomputed ``dot(layer_norm_out(x), linear_z)`` out of each epilogue is worth
1.15 us in *both* fused kernels. It is still the right trade. Splitting them back
apart into a whole-row epilogue kernel plus a plain projection -- 10 kernels ->
12, redundancy 6x/9x -> 1x -- measures **2.1 us worse** (42.02 against 39.94 on
the flush proxy, and +1.5 us in the sum of per-kernel warm costs): with one block
per SM the wall clock is per-program latency, and the un-fused epilogue's 16
programs plus two extra dependent launches cost more than the recompute.

The ten kernels:

  1. ``_tm_proj``     tri-mul-out: layer_norm_in + a/b projections + gates + mask,
                      and the L2 warm-up of the whole weight arena (see below)
  2. ``_tm_einsum``   the ``ij,jk`` contraction as a *batched* 16x16x16 dot with
                      the channel axis as the batch
  3. ``_epi_tm_proj`` tri-mul-out epilogue + tri-mul-in's input projection
  4. ``_tm_einsum``   tri-mul-in's contraction
  5. ``_epi_ta_proj`` tri-mul-in epilogue + starting-node q/k/v/g/bias projection
  6. ``_ta_attn``     starting-node attention + gate + linear_o + residual
  7. ``_ta_proj``     ending-node q/k/v/g/bias projection
  8. ``_ta_attn``     ending-node attention + epilogue
  9. ``_sg_hidden``   transition: layer_norm + SiLU(a) * b
 10. ``_sg_out``      transition: linear_out + mask + the block's final residual

(``_ingest`` is still here but is reachable only on the CUDA-graph fallback path,
where the caller's inputs have to be staged into fixed buffers for capture.)

The two triangle multiplications differ only in which index each operand is
contracted over, and the two attentions only in whether the pair matrix is read
transposed. Both differences are passed as strides, so the reference's three
permute-copies and two ``transpose(-2, -3)`` copies disappear into index
arithmetic, and there are seven compiled kernels rather than ten.

Warming L2 without spending a launch
------------------------------------
The harness flushes L2 with a 253 MiB fill immediately before its start event, so
every weight is cold and each kernel would pay full HBM latency on its first
touch. Reading the whole 1.1 MiB arena in a kernel of its own does fix that, but
the launch costs as much as it saves: measured on the flush proxy, a standalone
load-based ``_ingest`` and no warm-up at all are the *same* 41.95 us.

``prefetch.global.L2`` breaks the tie. It has no result, so nothing in the issuing
program depends on it, and it can therefore ride inside a kernel that is doing
other work: ``_tm_proj``'s programs each fire ``PFPP`` of them at the arena before
touching anything of their own, and the lines land while the rest of the pipeline
is still queueing. Worth **2.0 us** (39.94 against 41.95 with the prefetch
removed) and one launch fewer.

Reading across two grids
------------------------
Every kernel issues all of its loads before its layer-norm reduction -- the
reduction is a dependency barrier the scheduler will not hoist loads across --
which is also what gives PDL a real overlap window. The stronger version of that
is to notice which loads need the wait *at all*.

A chain of PDL launches only defers grid k+1 relative to grid k: grid k+1 cannot
trigger grid k+2 until every one of its blocks has run
``gdc_launch_dependents()``, which is after its own ``gdc_wait()`` on grid k.
So anything grid k+2 reads that was written by grid k is already final when k+2
starts, and its load belongs *above* ``gdc_wait()``. Three of the four seams in
this block qualify (``ABG``'s gate columns and ``Z`` in both epilogues, the
ending-node attention's residual, ``_sg_out``'s residual), as does every load of
the caller's own ``z``/``pair_mask``. Hoisting them -- plus moving the starting
node's residual load off the tail of ``_ta_attn``, where it was a fully exposed
round trip after the softmax -- is worth ~1.5 us.

Numerics
--------
fp32 for every reduction (both layer-norm passes, the einsum accumulator, the
softmax, every tensor-core accumulator); bf16 at the points where the reference
also rounds -- layer-norm outputs feeding a projection, the einsum result, the
gated activations, the attention probabilities -- so the fused path tracks the
reference's own rounding instead of drifting from it. Bench tolerance is
atol=rtol=1e-2 over 99% of elements; measured worst case is 1.0000 matched, with
a max absolute deviation of 1-2 bf16 ulps.

Where the 39.9 us goes
----------------------
Measured by timing N forwards in one flush window (``dev/prox.py --nfwd``) and by
prefix-truncating the launch chain (``dev/prefix.py``):

| bucket                                             |    us |
|----------------------------------------------------|------:|
| harness: 2 input-pool copies + the event pair      |  9.25 |
| PDL launch floor, 10 kernels x ~0.9               |  ~9   |
| work inside the kernels, warm                      | ~17   |
| cold-miss residue after the arena prefetch         |  ~4   |

The launch floor is measured directly: back-to-back trivial kernels cost
**0.87 us** each with PDL against 2.59 eager and 1.51 as CUDA-graph nodes. The
in-kernel work is *not* concentrated anywhere -- ten kernels at 0.9-3.1 us of
warm work each -- and ncu puts 42-53% of every kernel's stalls in
``long_scoreboard`` (global-load dependency) at 12.9% occupancy with one block
per SM, so what is left is dependent-load latency with nothing resident to hide
it behind, not arithmetic and not bandwidth.

Host mechanics
--------------
Kernels are invoked through the compiled kernel's own C launcher with a
fully pre-built argument list (~3.4 us of Python each against ~9 us for
``kernel[grid](...)``), and every intermediate buffer is allocated once and
reused, so a forward does exactly one allocation -- the output. Host cost is not
on the critical path at this size, but it is nearly free to remove and it is the
one thing that would matter if a scoring draw were ever host-contended.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._C import _cuda_getCurrentRawStream as _raw_stream
from torch._C import _cuda_getDevice as _cur_device

try:  # Triton 3.6+; the project floor is 3.5, so this is probed, not assumed
    from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait
    _HAS_PDL = True
except Exception:  # pragma: no cover - older Triton
    _HAS_PDL = False

from .alphafold3_triangle_multiplication import (
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)
from .alphafold3_triangle_attention import TriangleAttention
from .alphafold3_swiglu_transition import SwiGLUTransition

# Input dtypes the fused path handles. fp32 is deliberately excluded: the
# reference's F.linear runs an exact-fp32 cuBLAS kernel there while tl.dot would
# drop to tf32, and the fp32 tolerance (atol 1e-5) does not cover that gap.
_FUSED_DTYPES = (torch.bfloat16, torch.float16)

# Tile shapes. All swept on B200 under the harness's own conditions; see
# ITERATIONS.md for the tables. The pattern: the column block wants to be 32-64
# (smaller stops feeding the MMA, larger concentrates weight bytes into fewer
# programs), and the row block wants the grid at >= ~32 programs.
_BR = 16        # pair rows per program
_CB = 64        # output columns per program (triangle multiplication, transition)
_CQ = 64        # output columns per program (q/k/v/g/bias projection)
_CA = 32        # output columns per program (attention out-projection)
_EB = 8         # einsum channels per program (the batch dim of its 3-D dot)
_SB, _ST = 8, 32    # transition out-projection: rows, columns per program
_TK = 256       # transition out-projection: reduction chunk
_IBL = 2048     # _ingest elements per program

# 8 warps by default, 4 for the three kernels that measured faster there (both
# attentions and the transition out-projection -- see ``_WARPS4`` at the launch
# site). Counter to the usual "small tile -> few warps" instinct
# (and to what the frozen L1 layer_norm winner found for a bare norm), but
# measured: 36.9 us on the flush proxy at 8 warps against 41.1 at 4 and 47.3 at
# 2. These kernels are latency-bound on 16-84 KiB of independent loads, not
# register-bound, and with 11-528 programs occupancy is irrelevant -- so warps
# buy in-flight requests and cost nothing.
_WARPS = 8
_WARPS4 = 4   # both ``_ta_attn`` launches and ``_sg_out``

# Programmatic dependent launch. The block is eleven strictly dependent kernels
# whose every weight / layer-norm-affine load is independent of the previous
# kernel's output, so each ``gdc_wait()`` sits *after* that prologue and the
# consumer's static loads issue while the producer drains. Measured per-launch
# floor with trivial kernels: ~1.0 us with PDL against ~3.3 us without.
_PDL = _HAS_PDL

# CUDA-graph capture is the fallback for a Triton without the PDL intrinsics,
# not an addition to it: measured together, PDL alone wins (45.3 us against
# 47.4 us on the flush proxy, 53.2 us against 56.4 us on the bench). Capture
# needs constant per-node arguments, which is why it also turns on input staging.
_GRAPH = not _PDL

# ---------------------------------------------------------------------------
# Prologue
# ---------------------------------------------------------------------------
@triton.jit
def _pf(P, off):
    """Fire-and-forget ``prefetch.global.L2`` of the 128 B lines at ``P + off``.

    Unlike a ``.cg`` load there is no result to keep alive, so nothing in the
    kernel depends on it: the requests are issued and the program retires while
    they are still in flight. That is what lets the block's L2 warm-up ride
    inside the first real kernel instead of costing a launch of its own --
    measured, a load-based warm-up of the whole arena is 4.2 us of chain time and
    the same warm-up as prefetches is 2.2 us standalone and ~0 us distributed.
    """
    tl.inline_asm_elementwise("prefetch.global.L2 [$1];", "=r,l",
                              [P.to(tl.int64) + off.to(tl.int64)],
                              dtype=tl.int32, is_pure=False, pack=1)


@triton.jit
def _ingest(Z, SZ, MK, SM, W, SINK,
            NZ: tl.constexpr, NM: tl.constexpr, NP: tl.constexpr, BL: tl.constexpr,
            PDL: tl.constexpr):
    """Stage the inputs into fixed buffers and pull the weight arena into L2.

    **Only reached on the CUDA-graph fallback path** (a Triton without the PDL
    intrinsics). ``z`` and ``pair_mask`` are the only pointers the caller owns and
    they move every call; copying them into plan-owned buffers makes every *other*
    kernel's argument list constant, which is what lets the middle of the pipeline
    be captured once. Since the kernel exists there anyway it also warms the
    weight arena, which the flush leaves cold.

    On the PDL path this launch is gone: the warm-up moved into ``_tm_proj``'s
    ``prefetch.global.L2`` prologue, which does the same job without a launch (see
    the module docstring). Reading the arena here rather than prefetching it is
    deliberate -- graph nodes are not overlapped, so there is nothing to hide a
    fire-and-forget prefetch behind.

    The arena sum is stored so the loads are not dead code; ``SINK`` is one fp32
    per program and is never read.
    """
    pid = tl.program_id(0)
    o = pid * BL + tl.arange(0, BL)
    if PDL:
        gdc_wait()
    mz = o < NZ
    tl.store(SZ + o, tl.load(Z + o, mask=mz, other=0), mask=mz)
    mm = o < NM
    tl.store(SM + o, tl.load(MK + o, mask=mm, other=0), mask=mm)
    tl.store(SINK + pid,
             tl.sum(tl.load(W + o, mask=o < NP, other=0).to(tl.float32), axis=0))
    if PDL:
        gdc_launch_dependents()


# ---------------------------------------------------------------------------
# Shared device helpers
# ---------------------------------------------------------------------------
@triton.jit
def _ln(x, w, b, C: tl.constexpr, EPS: tl.constexpr):
    """Row-wise layer norm of an ``[M, C]`` fp32 tile, affine in fp32.

    Two passes over a tile already resident in registers, so the second costs
    nothing and there is no reason to use the numerically weaker
    ``E[x^2] - E[x]^2``. ``w``/``b`` are *already-loaded* fp32 affine tiles: every
    kernel here issues all of its loads before the reduction, because the
    reduction is a dependency barrier the scheduler will not hoist loads across,
    and one extra exposed memory round trip is ~0.5-2 us at this size.
    """
    mu = tl.sum(x, axis=1) * (1.0 / C)
    d = x - mu[:, None]
    var = tl.sum(d * d, axis=1) * (1.0 / C)
    r = 1.0 / tl.sqrt(var + EPS)
    return d * r[:, None] * w[None, :] + b[None, :]


@triton.jit
def _sig(x):
    return 1.0 / (1.0 + tl.exp(-x))


@triton.jit
def _wt(WT, col, K: tl.constexpr, NW: tl.constexpr, LD: tl.constexpr):
    """``[K, NW]`` slice of a pre-transposed weight starting at column ``col``."""
    return tl.load(WT + tl.arange(0, K)[:, None] * LD + (col + tl.arange(0, NW))[None, :])


# ---------------------------------------------------------------------------
# Triangle multiplication
# ---------------------------------------------------------------------------
@triton.jit
def _tm_task(t, CH: tl.constexpr, CB: tl.constexpr, NCA: tl.constexpr):
    """Decode a triangle-multiplication input-projection column task.

    Tasks ``[0, 2*NCA)`` produce a column block of ``a`` or ``b`` -- each needs the
    *same* column range of two different projections (the value and its sigmoid
    gate). The remaining tasks produce the output gate, which needs only one; it
    aims both weight offsets at the same slice and discards the second dot rather
    than branching, so the kernel stays divergence-free and every load can be
    issued before the layer-norm reduction.

    Returns (is_ab, value-weight column, gate-weight column, output column).
    """
    ab = t < 2 * NCA
    cbi = tl.where(ab, t % NCA, t - 2 * NCA) * CB
    cp = tl.where(ab, (t // NCA) * 2 * CH + cbi, 4 * CH + cbi)
    return (ab, cp, cp + tl.where(ab, CH, 0),
            tl.where(ab, (t // NCA) * CH + cbi, 2 * CH + cbi))


@triton.jit
def _tm_store(xl, rows, m, wp, wg, ab, col, ABG,
              C: tl.constexpr, CH: tl.constexpr, CB: tl.constexpr, DT: tl.constexpr):
    """One column block of ``a``, ``b`` or the output gate, from a normalized tile.

    All three land in one buffer ``ABG = [a | b | gate]``, so the store has no
    branch on its destination either.
    """
    p = tl.dot(xl, wp)
    tl.store(ABG + rows[:, None] * (2 * CH + C) + (col + tl.arange(0, CB))[None, :],
             tl.where(ab, m * _sig(tl.dot(xl, wg)) * p, _sig(p)).to(DT))


@triton.jit
def _ta_store(xl, lr, w, cb, QKVG, LD: tl.constexpr, CB: tl.constexpr, DT: tl.constexpr):
    """One column block of the triangle-attention q/k/v/g/bias projection."""
    tl.store(QKVG + lr[:, None] * LD + (cb + tl.arange(0, CB))[None, :],
             tl.dot(xl, w).to(DT))


@triton.jit
def _epi_pre(ABG, Z, rows, C: tl.constexpr, CH: tl.constexpr):
    """The ``_epi_tile`` inputs that do **not** come from the predecessor grid.

    ``ABG``'s gate columns were written *two* grids back (the ``_tm_einsum`` in
    between writes only ``X``) and ``Z`` is a tensor written two grids back or by
    the caller, so both are already final when this grid starts -- see
    "Reading across two grids" in the module docstring. Loading them above
    ``gdc_wait()`` puts one more memory round trip inside the PDL overlap window.
    """
    c = tl.arange(0, C)
    g = tl.load(ABG + rows[:, None] * (2 * CH + C) + (2 * CH + c)[None, :]).to(tl.float32)
    zv = tl.load(Z + rows[:, None] * C + c[None, :]).to(tl.float32)
    return g, zv


@triton.jit
def _epi_tile(X, ZOUT, g, zv, wz, lw, lb, rows,
              C: tl.constexpr, CH: tl.constexpr, EPS: tl.constexpr, DT: tl.constexpr):
    """``layer_norm_out`` -> ``linear_z`` -> output gate -> residual, whole row.

    Returns the fp32 ``[BR, C]`` tile as well as storing it, so a fused caller can
    feed the next stage's layer norm from registers instead of a round trip
    through memory. ``wz``/``lw``/``lb`` and ``g``/``zv`` are all preloaded by the
    caller, so the only load left here is ``X`` -- the one value that genuinely
    comes from the predecessor grid.
    """
    x = tl.load(X + rows[:, None] * CH + tl.arange(0, CH)[None, :]).to(tl.float32)
    z2 = zv + tl.dot(_ln(x, lw, lb, CH, EPS).to(DT), wz) * g
    tl.store(ZOUT + rows[:, None] * C + tl.arange(0, C)[None, :], z2.to(DT))
    return z2


@triton.jit
def _tm_proj(Z, M, LNW, LNB, WT, ABG, ARENA, NLINE,
             C: tl.constexpr, CH: tl.constexpr, BR: tl.constexpr, CB: tl.constexpr,
             NCA: tl.constexpr, EPS: tl.constexpr, DT: tl.constexpr,
             PDL: tl.constexpr, PFPP: tl.constexpr):
    """``layer_norm_in`` -> a, b and the output gate, one column slice per program.

    Only used for the first triangle multiplication; the second one's projection
    is fused into the first one's epilogue (see ``_epi_tm_proj``).

    With ``PFPP > 0`` this kernel also carries the whole block's L2 warm-up: each
    program fires ``PFPP`` ``prefetch.global.L2`` at the weight arena before it
    touches anything of its own. Nothing waits on the result, so the requests
    land while the later kernels are still queueing -- which is what let the
    standalone ``_ingest`` launch go away on the PDL path.

    ``gdc_wait()`` is *not* optional here even though nothing in this kernel
    reads a predecessor kernel's output: the harness's own input-shifting copy of
    ``z``/``pair_mask`` is the preceding stream work, and a PDL grid may start
    before it finishes. (Measured, dropping the wait is 2 us *worse* anyway --
    the kernel then races the warm-up for bandwidth.)
    """
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    c = tl.arange(0, C)
    if PFPP > 0:
        pid = tl.program_id(0) * tl.num_programs(1) + tl.program_id(1)
        _pf(ARENA, tl.minimum(pid * PFPP + tl.arange(0, PFPP), NLINE - 1) * 128)
    ab, cp, cg, col = _tm_task(tl.program_id(1), CH, CB, NCA)
    wp = _wt(WT, cp, C, CB, 4 * CH + C)
    wg = _wt(WT, cg, C, CB, 4 * CH + C)
    lw = tl.load(LNW + c)
    lb = tl.load(LNB + c)
    if PDL:
        gdc_wait()
    m = tl.load(M + rows)[:, None].to(tl.float32)
    x = tl.load(Z + rows[:, None] * C + c[None, :]).to(tl.float32)
    _tm_store(_ln(x, lw, lb, C, EPS).to(DT), rows, m, wp, wg, ab, col, ABG,
              C, CH, CB, DT)
    if PDL:
        gdc_launch_dependents()


@triton.jit
def _epi_tm_proj(X, ABG, Z, ZOUT, LNOW, LNOB, WZT, M, LNW, LNB, WT, ABG2,
                 C: tl.constexpr, CH: tl.constexpr, CH2: tl.constexpr,
                 BR: tl.constexpr, CB: tl.constexpr, NCA: tl.constexpr,
                 EPS: tl.constexpr, DT: tl.constexpr, PDL: tl.constexpr):
    """One triangle multiplication's epilogue + the next one's input projection.

    The layer norm between them reduces over the channel axis, so a program can
    only feed it rows it holds *completely* -- which normally makes it a kernel
    boundary, since the epilogue is column-split. Crossing it instead by having
    every column task recompute the whole ``[BR, C]`` residual row costs one extra
    ``[CH, C]`` weight slice per program and saves the round trip through memory
    plus a launch: measured in isolation, 5.89 us as two kernels against 4.28 us
    fused.
    """
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    c = tl.arange(0, C)
    ab, cp, cg, col = _tm_task(tl.program_id(1), CH2, CB, NCA)
    wp = _wt(WT, cp, C, CB, 4 * CH2 + C)
    wg = _wt(WT, cg, C, CB, 4 * CH2 + C)
    wz = _wt(WZT, 0, CH, C, C)
    lo = tl.load(LNOW + tl.arange(0, CH))
    lb0 = tl.load(LNOB + tl.arange(0, CH))
    lw = tl.load(LNW + c)
    lb = tl.load(LNB + c)
    m = tl.load(M + rows)[:, None].to(tl.float32)
    g, zv = _epi_pre(ABG, Z, rows, C, CH)
    if PDL:
        gdc_wait()
    z2 = _epi_tile(X, ZOUT, g, zv, wz, lo, lb0, rows, C, CH, EPS, DT)
    _tm_store(_ln(z2, lw, lb, C, EPS).to(DT), rows, m, wp, wg, ab, col, ABG2,
              C, CH2, CB, DT)
    if PDL:
        gdc_launch_dependents()


@triton.jit
def _epi_ta_proj(X, ABG, Z, ZOUT, LNOW, LNOB, WZT, LNW, LNB, WT, QKVG,
                 N: tl.constexpr, C: tl.constexpr, CH: tl.constexpr, LD: tl.constexpr,
                 BR: tl.constexpr, CB: tl.constexpr,
                 EPS: tl.constexpr, DT: tl.constexpr, PDL: tl.constexpr):
    """The second triangle multiplication's epilogue + the starting-node projection.

    Same trick as ``_epi_tm_proj``; the consumer here is the attention input
    projection, whose column tasks tile a different concatenated weight. The
    starting node reads the pair matrix untransposed, so there is no index
    remapping between the two halves.
    """
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    c = tl.arange(0, C)
    cb = tl.program_id(1) * CB
    w = _wt(WT, cb, C, CB, LD)
    wz = _wt(WZT, 0, CH, C, C)
    lo = tl.load(LNOW + tl.arange(0, CH))
    lb0 = tl.load(LNOB + tl.arange(0, CH))
    lw = tl.load(LNW + c)
    lb = tl.load(LNB + c)
    g, zv = _epi_pre(ABG, Z, rows, C, CH)
    if PDL:
        gdc_wait()
    z2 = _epi_tile(X, ZOUT, g, zv, wz, lo, lb0, rows, C, CH, EPS, DT)
    _ta_store(_ln(z2, lw, lb, C, EPS).to(DT), rows, w, cb, QKVG, LD, CB, DT)
    if PDL:
        gdc_launch_dependents()


@triton.jit
def _tm_einsum(ABG, X, N: tl.constexpr, C: tl.constexpr, CH: tl.constexpr,
               EB: tl.constexpr, SAI: tl.constexpr, SAJ: tl.constexpr,
               SBK: tl.constexpr, SBJ: tl.constexpr, DT: tl.constexpr,
               PDL: tl.constexpr):
    """``x[i,k,c] = sum_j a[..,c] * b[..,c]`` as a batched dot over the channels.

    The contraction is not a matmul -- ``c`` is a free index shared by both
    operands -- but it *is* ``EB`` independent 16x16x16 matmuls, which is exactly
    a 3-D ``tl.dot``. Splitting over channels makes every program's footprint two
    ``[EB, N, N]`` tiles (8 KiB at EB=8) and needs no weights at all, so this is
    the cheapest kernel in the block. The alternative -- one program per outer
    index, accumulating ``N`` rank-1 broadcast updates -- has to read all of
    ``b`` (64 KiB) per program and measured 4.3 us against 2.1 us here.

    Outgoing vs incoming is entirely in the strides: outgoing contracts
    ``a[i,j]`` with ``b[k,j]``, incoming ``a[j,i]`` with ``b[j,k]``.
    """
    cb = tl.program_id(0) * EB
    bb = tl.program_id(1) * (N * N)
    c = cb + tl.arange(0, EB)
    i = tl.arange(0, N)
    j = tl.arange(0, N)
    k = tl.arange(0, N)
    W: tl.constexpr = 2 * CH + C
    if PDL:
        gdc_wait()
    a3 = tl.load(ABG + (bb + i[None, :, None] * SAI + j[None, None, :] * SAJ) * W
                 + c[:, None, None])
    b3 = tl.load(ABG + (bb + k[None, None, :] * SBK + j[None, :, None] * SBJ) * W
                 + CH + c[:, None, None])
    o = tl.dot(a3, b3)
    tl.store(X + (bb + i[None, :, None] * N + k[None, None, :]) * CH + c[:, None, None],
             o.to(DT))
    if PDL:
        gdc_launch_dependents()


# ---------------------------------------------------------------------------
# Triangle attention
# ---------------------------------------------------------------------------
@triton.jit
def _ta_proj(Z, LNW, LNB, WT, QKVG,
             N: tl.constexpr, C: tl.constexpr, LD: tl.constexpr,
             BR: tl.constexpr, CB: tl.constexpr,
             SZI: tl.constexpr, SZJ: tl.constexpr,
             EPS: tl.constexpr, DT: tl.constexpr, PDL: tl.constexpr):
    """``layer_norm`` -> q / k / v / g and the triangle bias, one slice per program.

    q, k, v, g and the bias share one output buffer and one concatenated weight of
    width ``LD``, zero-padded so ``LD`` is a whole number of column blocks. Every
    program is then the same shape -- one ``[C, CB]`` weight slice and one store
    -- with no branch for the (only ``H`` wide) bias task.

    ``SZI``/``SZJ`` are the pair matrix's row strides. Swapping them turns the
    ending node's ``x = x.transpose(-2, -3)`` into the index map
    ``logical (i, j) -> actual (j, i)``, at no cost.
    """
    lr = tl.program_id(0) * BR + tl.arange(0, BR)
    cb = tl.program_id(1) * CB
    c = tl.arange(0, C)
    # ``lr`` is a flat logical row over the whole batch; the transpose applies
    # within each N x N pair matrix, so split the batch index off first.
    nn2: tl.constexpr = N * N
    rows = (lr // nn2) * nn2 + ((lr % nn2) // N) * SZI + (lr % N) * SZJ
    w = _wt(WT, cb, C, CB, LD)
    lw = tl.load(LNW + c)
    lb = tl.load(LNB + c)
    if PDL:
        gdc_wait()
    x = tl.load(Z + rows[:, None] * C + c[None, :]).to(tl.float32)
    _ta_store(_ln(x, lw, lb, C, EPS).to(DT), lr, w, cb, QKVG, LD, CB, DT)
    if PDL:
        gdc_launch_dependents()


@triton.jit
def _ta_attn(QKVG, M, WOT, Z, ZOUT,
             N: tl.constexpr, C: tl.constexpr, LD: tl.constexpr,
             H: tl.constexpr, HB: tl.constexpr, D: tl.constexpr, CA: tl.constexpr,
             SMI: tl.constexpr, SMJ: tl.constexpr,
             SZI: tl.constexpr, SZJ: tl.constexpr,
             SCALE: tl.constexpr, INF: tl.constexpr, DT: tl.constexpr,
             PDL: tl.constexpr, ZE: tl.constexpr):
    """Attention over one pair axis, all heads at once, + gate + linear_o + residual.

    One program per (outer index, output column block). At ``N = 16`` the score
    matrix for one head is a single MMA tile, so there is no flash-style KV loop
    -- and the ``H`` heads are ``H`` independent 16x16 problems, which is exactly
    a 3-D ``tl.dot``. Doing them batched rather than in an unrolled head loop is
    what makes this kernel cheap: q, k, v, g and the bias become four wide
    independent loads issued together instead of four serialized per-head
    load/softmax/load chains. ``HB`` is ``H`` padded to the batch size ``tl.dot``
    accepts; the padding lanes carry zeros and are dropped by the reshape.

    ``linear_o`` contracts the head axis, so the gated per-head outputs are
    permuted back to ``[N, H*D]`` and consumed by a single dot; splitting the
    output columns across programs re-runs the (tiny) attention per block and in
    exchange cuts each program's ``linear_o`` slice to ``[H*D, CA]``.

    The mask bias and the triangle bias are consumed inline and never
    materialized. Starting vs ending node is index math only.
    """
    i = tl.program_id(0)
    cb = tl.program_id(1) * CA
    bb = tl.program_id(2) * (N * N)
    h = tl.arange(0, HB)
    j = tl.arange(0, N)
    kk = tl.arange(0, N)
    d = tl.arange(0, D)
    col = cb + tl.arange(0, CA)
    hd = (h[:, None, None] * D + d[None, None, :])
    qrow = (bb + i * N + j[None, :, None]) * LD
    krow = (bb + i * N + kk[None, :, None]) * LD
    valid = (h < H)[:, None, None]
    hpad = tl.arange(0, HB * D)
    wo = tl.load(WOT + hpad[:, None] * C + col[None, :],
                 mask=(hpad < H * D)[:, None], other=0.0)
    # The mask is the caller's own tensor. ``ZE`` marks the ending node, whose
    # residual is the starting node's output -- written two grids back, so final
    # already. The starting node's residual is its predecessor's output and has
    # to come after the wait, but it is still issued *before* the dots rather
    # than after the softmax, so it overlaps them instead of extending the chain.
    zo = (bb + i * SZI + j * SZJ)[:, None] * C + col[None, :]
    mb = (tl.load(M + bb + i * SMI + kk * SMJ).to(tl.float32) - 1.0) * INF
    if ZE:
        zres = tl.load(Z + zo).to(tl.float32)
    if PDL:
        gdc_wait()
    if not ZE:
        zres = tl.load(Z + zo).to(tl.float32)
    q = tl.load(QKVG + qrow + hd, mask=valid, other=0.0)
    kt = tl.load(QKVG + krow + (C + hd), mask=valid, other=0.0)
    v = tl.load(QKVG + krow + (2 * C + hd), mask=valid, other=0.0)
    g = tl.load(QKVG + qrow + (3 * C + hd), mask=valid, other=0.0)
    tb = tl.load(QKVG + (bb + j[None, :, None] * N + kk[None, None, :]) * LD
                 + 4 * C + h[:, None, None], mask=valid, other=0.0)

    s = tl.dot(q, tl.permute(kt, (0, 2, 1))) * SCALE + mb[None, None, :]
    s += tb.to(tl.float32)
    p = tl.exp(s - tl.max(s, axis=2)[:, :, None])
    p = (p / tl.sum(p, axis=2)[:, :, None]).to(DT)
    o = tl.dot(p, v) * _sig(g.to(tl.float32))
    # [HB, N, D] -> [N, HB*D]; the HB-H padding rows are zero, and linear_o's
    # weight slice is only H*D tall, so they contract away.
    oc = tl.reshape(tl.permute(o.to(DT), (1, 0, 2)), (N, HB * D))
    acc = tl.dot(oc, wo)

    tl.store(ZOUT + zo, (zres + acc).to(DT))
    if PDL:
        gdc_launch_dependents()


# ---------------------------------------------------------------------------
# SwiGLU transition
# ---------------------------------------------------------------------------
@triton.jit
def _sg_hidden(Z, LNW, LNB, WT, HB,
               C: tl.constexpr, CT: tl.constexpr, BR: tl.constexpr, CB: tl.constexpr,
               EPS: tl.constexpr, DT: tl.constexpr, PDL: tl.constexpr):
    """``layer_norm`` -> ``SiLU(linear_a(x)) * linear_b(x)``, one column slice each."""
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    cb = tl.program_id(1) * CB
    c = tl.arange(0, C)
    wa = _wt(WT, cb, C, CB, 2 * CT)
    wb = _wt(WT, CT + cb, C, CB, 2 * CT)
    lw = tl.load(LNW + c)
    lb = tl.load(LNB + c)
    if PDL:
        gdc_wait()
    x = tl.load(Z + rows[:, None] * C + c[None, :]).to(tl.float32)
    xl = _ln(x, lw, lb, C, EPS).to(DT)
    a = tl.dot(xl, wa).to(DT).to(tl.float32)
    b = tl.dot(xl, wb).to(DT).to(tl.float32)
    tl.store(HB + rows[:, None] * CT + (cb + tl.arange(0, CB))[None, :],
             ((a * _sig(a)).to(DT).to(tl.float32) * b).to(DT))
    if PDL:
        gdc_launch_dependents()


@triton.jit
def _sg_out(HB, M, WOT, Z, OUT,
            C: tl.constexpr, CT: tl.constexpr, BR: tl.constexpr, CB: tl.constexpr,
            TB: tl.constexpr, EPS: tl.constexpr, DT: tl.constexpr, PDL: tl.constexpr):
    """``linear_out`` -> mask -> the block's final residual.

    ``linear_out`` contracts the whole ``CT``-wide hidden state, so the reduction
    is walked in ``TB``-wide chunks; the loop is a plain ``range`` so Triton
    software-pipelines the weight slices rather than trying to hold all of
    ``[CT, CB]`` at once.
    """
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    col = tl.program_id(1) * CB + tl.arange(0, CB)
    # mask: the caller's tensor; Z: written two grids back by the ending-node
    # attention (the ``_sg_hidden`` in between writes only ``HB``).
    m = tl.load(M + rows)[:, None].to(tl.float32)
    o = rows[:, None] * C + col[None, :]
    zv = tl.load(Z + o).to(tl.float32)
    if PDL:
        gdc_wait()
    acc = tl.zeros([BR, CB], tl.float32)
    for t in range(0, CT, TB):
        hh = tl.load(HB + rows[:, None] * CT + (t + tl.arange(0, TB))[None, :])
        acc += tl.dot(hh, tl.load(WOT + (t + tl.arange(0, TB))[:, None] * C + col[None, :]))
    tl.store(OUT + o, (zv + acc * m).to(DT))
    if PDL:
        gdc_launch_dependents()


# ---------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------
def _pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


class _Unsupported(Exception):
    """This (shape, dtype) combination is not covered by the fused path."""


class _Call:
    """One kernel, launched through the compiled kernel's own C entry point.

    ``kernel[grid](...)`` re-binds, re-specializes and re-hashes every argument
    on each call (~9 us of Python); with every argument here either a raw pointer
    or a constexpr there is nothing for it to derive that is not already known.
    ``argv`` is the complete argument list, so a launch is one Python call with no
    tuple building, and the handful of entries that change per forward are patched
    in place by index.
    """

    __slots__ = ("run", "argv")

    def __init__(self, kern, grid, args, device):
        launcher = kern.run
        self.run = launcher.launch
        g = tuple(grid) + (1,) * (3 - len(grid))
        self.argv = [g[0], g[1], g[2], _raw_stream(device), kern.function,
                     launcher.launch_cooperative_grid, launcher.launch_pdl,
                     None, None, kern.packed_metadata, None, None, None, *args]

    @staticmethod
    def usable(kern) -> bool:
        launcher = getattr(kern, "run", None)
        return (launcher is not None
                and getattr(launcher, "launch", None) is not None
                and getattr(launcher, "global_scratch_size", None) == 0
                and getattr(launcher, "profile_scratch_size", None) == 0)


_ARGV0 = 13  # index of the first kernel argument inside _Call.argv


class _Plan:
    """Compiled kernels + workspace + pre-built launch arguments for one shape.

    Built on the first forward rather than in ``__init__`` so externally loaded
    weights are what gets folded into the concatenated / pre-transposed buffers,
    and invalidated by ``_apply`` / ``_load_from_state_dict`` on the owning module
    so a later ``.to()`` or checkpoint load rebuilds it.
    """

    def __init__(self, mod, z: torch.Tensor, mask: torch.Tensor):
        dt, dev = z.dtype, z.device
        B = 1
        for s in z.shape[:-3]:
            B *= s
        N, C = z.shape[-2], z.shape[-1]
        R = B * N * N
        self.n, self.c, self.dtype, self.device = N, C, dt, dev
        self.numel = z.numel()

        tmo, tmi = mod.tri_mul_out, mod.tri_mul_in
        tas, tae = mod.tri_att_start, mod.tri_att_end
        pt = mod.pair_transition
        CH = tmo.c_hidden
        H, D = tas.no_heads, tas.c_hidden
        HD = H * D
        HB = triton.next_power_of_2(H)
        CT = pt.n * pt.c_in

        epss = {tmo.layer_norm_in.eps, tmo.layer_norm_out.eps,
                tmi.layer_norm_in.eps, tmi.layer_norm_out.eps,
                tas.layer_norm.eps, tae.layer_norm.eps, pt.layer_norm.eps}
        BR = min(_BR, R)
        CB = min(_CB, C, CH)
        EB = min(_EB, CH)
        SB, ST = min(_SB, R), min(_ST, C)
        CA = min(_CA, C)
        CQ = min(_CQ, 4 * HD)
        TBK = min(_TK, CT)
        # ``N >= 16``: the score matrix is the contraction operand of the PV
        # product and tl.dot needs K >= 16. ``HD == C``: ``_ta_proj`` writes
        # q/k/v/g at 4*HD row stride and ``_ta_attn`` reads them back at HD-wide
        # head-group offsets, so the two have to agree.
        if not (len(epss) == 1 and _pow2(N) and 16 <= N <= 64
                and _pow2(C) and _pow2(CH) and _pow2(D) and _pow2(HD) and HD == C
                and _pow2(CB) and _pow2(EB) and _pow2(ST) and _pow2(TBK)
                and _pow2(CA) and C % CA == 0 and CB >= H and CQ >= H
                and _pow2(CQ) and (4 * HD) % CQ == 0
                and R % BR == 0 and C % CB == 0 and CH % CB == 0 and CH % EB == 0
                and R % SB == 0 and C % ST == 0 and CT % TBK == 0
                and (4 * HD) % CB == 0 and CT % CB == 0
                and tmi.c_hidden == CH and tae.no_heads == H
                and tae.c_hidden == D and pt.c_in == C):
            raise _Unsupported
        eps = epss.pop()

        # Projection weights live in one contiguous arena so ``_prefetch`` can
        # walk them as a single flat range (and so they share L2 sets).
        pend = []

        def wt(*ws):
            """Pre-transposed ``[K, sum(N_i)]`` view of concatenated weight rows."""
            t = torch.cat(ws, 0).t().contiguous()
            pend.append(t)
            return t

        def f32(t):
            return t.detach().float().contiguous()

        # The triangle-bias block is padded out to a full column block so every
        # ``_ta_proj`` program is the same shape (and tl.dot's N >= 16 is met).
        zpad = torch.zeros(CQ - H, C, dtype=dt, device=dev)
        W = {}
        for tag, m in (("o", tmo), ("i", tmi)):
            W["ln" + tag] = (f32(m.layer_norm_in.weight), f32(m.layer_norm_in.bias))
            W["w1" + tag] = wt(m.linear_a_p.weight, m.linear_a_g.weight,
                               m.linear_b_p.weight, m.linear_b_g.weight,
                               m.linear_g.weight)
            W["lo" + tag] = (f32(m.layer_norm_out.weight), f32(m.layer_norm_out.bias))
            W["wz" + tag] = wt(m.linear_z.weight)
        for tag, m in (("s", tas), ("e", tae)):
            W["ln" + tag] = (f32(m.layer_norm.weight), f32(m.layer_norm.bias))
            W["w2" + tag] = wt(m.mha.linear_q.weight, m.mha.linear_k.weight,
                               m.mha.linear_v.weight, m.mha.linear_g.weight,
                               m.linear_z.weight, zpad)
            W["wo" + tag] = wt(m.mha.linear_o.weight)
        W["lnt"] = (f32(pt.layer_norm.weight), f32(pt.layer_norm.bias))
        W["w3t"] = wt(pt.swiglu.linear_a.weight, pt.swiglu.linear_b.weight)
        W["wot"] = wt(pt.linear_out.weight)
        arena = torch.empty(sum(t.numel() for t in pend), dtype=dt, device=dev)
        off = 0
        remap = {}
        for t in pend:
            v = arena[off:off + t.numel()].view(t.shape)
            v.copy_(t)
            remap[id(t)] = v
            off += t.numel()
        W = {k: (remap.get(id(v), v) if torch.is_tensor(v) else v) for k, v in W.items()}
        self.arena = arena
        self.weights = W  # keeps the storages alive; the kernels hold raw pointers

        def buf(*shape):
            return torch.empty(shape, dtype=dt, device=dev)

        S = {n: buf(R, 2 * CH + C) for n in ("abgo", "abgi")}
        S.update({n: buf(R, CH) for n in ("xo", "xi")})
        S.update({n: buf(R, C) for n in ("z2", "z3", "z4", "z5", "out")})
        LDA = 4 * HD + CQ          # q | k | v | g | triangle bias | zero pad
        S.update({n: buf(R, LDA) for n in ("qs", "qe")})
        S["hb"] = buf(R, CT)
        S["sz"] = buf(R, C)
        S["sm"] = buf(R)
        self.scratch = S

        DT = tl.bfloat16 if dt is torch.bfloat16 else tl.float16
        scale = float(D) ** -0.5
        inf = tas.inf
        NCA, NCG = CH // CB, C // CB
        # Stride pairs, in units of pair rows. ``_tm_*``: outgoing contracts
        # a[i,j] with b[k,j], incoming a[j,i] with b[j,k]. ``_ta_*``: the ending
        # node reads and writes the pair matrix (and the mask) transposed.
        OUTG, INCO = (N, 1, N, 1), (1, N, 1, N)
        FWD, TRN = (N, 1), (1, N)

        K, A, G, NW = [], [], [], []

        def add(kern, grid, args, warps=_WARPS):
            K.append(kern)
            G.append(grid)
            A.append(args)
            NW.append(warps)

        # Without graph capture there is no reason to stage the caller's inputs:
        # the kernels can read them in place and ``_ingest`` only has to warm L2.
        stage = _GRAPH
        szb, smb = (S["sz"], S["sm"]) if stage else (z, mask)
        # ``_ingest`` survives only on the CUDA-graph fallback path, where the
        # inputs *have* to be staged into fixed buffers for capture. On the PDL
        # path its other job -- warming the weight arena -- moved into
        # ``_tm_proj``'s prefetch prologue, which does it without a launch.
        PBL = _IBL
        if stage:
            npf = (max(arena.numel(), z.numel()) + PBL - 1) // PBL
            S["sink"] = torch.empty(npf, dtype=torch.float32, device=dev)
            add(_ingest, (npf,),
                [z, S["sz"], mask, S["sm"], arena, S["sink"],
                 z.numel(), mask.numel(), arena.numel(), PBL, _PDL])

        # Only the first projection stands alone; the other three input stages
        # are fused into the epilogue of whatever produced their rows.
        # One ``prefetch.global.L2`` per 128 B arena line, spread evenly over this
        # kernel's programs; zero when ``_ingest`` is already warming the arena.
        ntm = (R // BR) * (2 * NCA + NCG)
        nline = (arena.numel() * arena.element_size() + 127) // 128
        pfpp = 0 if stage else triton.next_power_of_2((nline + ntm - 1) // ntm)
        add(_tm_proj, (R // BR, 2 * NCA + NCG),
            [szb, smb, W["lno"][0], W["lno"][1], W["w1o"], S["abgo"], arena, nline,
             C, CH, BR, CB, NCA, eps, DT, _PDL, pfpp])
        add(_tm_einsum, (CH // EB, B),
            [S["abgo"], S["xo"], N, C, CH, EB, *OUTG, DT, _PDL])
        add(_epi_tm_proj, (R // BR, 2 * NCA + NCG),
            [S["xo"], S["abgo"], szb, S["z2"],
             W["loo"][0], W["loo"][1], W["wzo"],
             smb, W["lni"][0], W["lni"][1], W["w1i"], S["abgi"],
             C, CH, CH, BR, CB, NCA, eps, DT, _PDL])
        add(_tm_einsum, (CH // EB, B),
            [S["abgi"], S["xi"], N, C, CH, EB, *INCO, DT, _PDL])
        add(_epi_ta_proj, (R // BR, LDA // CQ),
            [S["xi"], S["abgi"], S["z2"], S["z3"],
             W["loi"][0], W["loi"][1], W["wzi"],
             W["lns"][0], W["lns"][1], W["w2s"], S["qs"],
             N, C, CH, LDA, BR, CQ, eps, DT, _PDL])

        add(_ta_attn, (N, C // CA, B),
            [S["qs"], smb, W["wos"], S["z3"], S["z4"],
             N, C, LDA, H, HB, D, CA, *FWD, *FWD, scale, inf, DT, _PDL, 0],
            _WARPS4)
        add(_ta_proj, (R // BR, LDA // CQ),
            [S["z4"], W["lne"][0], W["lne"][1], W["w2e"],
             S["qe"], N, C, LDA, BR, CQ, *TRN, eps, DT, _PDL])
        add(_ta_attn, (N, C // CA, B),
            [S["qe"], smb, W["woe"], S["z4"], S["z5"],
             N, C, LDA, H, HB, D, CA, *TRN, *TRN, scale, inf, DT, _PDL, 1],
            _WARPS4)

        add(_sg_hidden, (R // BR, CT // CB),
            [S["z5"], W["lnt"][0], W["lnt"][1], W["w3t"], S["hb"],
             C, CT, BR, CB, eps, DT, _PDL])
        add(_sg_out, (R // SB, C // ST),
            [S["hb"], smb, W["wot"], S["z5"], S["out"],
             C, CT, SB, ST, TBK, eps, DT, _PDL],
            _WARPS4)

        # Compile with the real tensors -- Triton infers pointer types from the
        # objects it is handed -- then bake the resulting addresses into the
        # launch arguments.
        compiled = [k[g](*a, num_warps=w, launch_pdl=_PDL)
                    for k, g, a, w in zip(K, G, A, NW)]
        if not all(_Call.usable(k) for k in compiled):
            raise _Unsupported
        # Held for the process's lifetime: ``_Call`` keeps only the raw CUfunction
        # handle and packed metadata, so the CompiledKernel has to stay alive.
        self.compiled = compiled
        raw = [[x.data_ptr() if torch.is_tensor(x) else x for x in a] for a in A]
        dix = dev.index if dev.index is not None else _cur_device()
        self.calls = [_Call(k, g, a, dix) for k, g, a in zip(compiled, G, raw)]
        self.dix = dix
        self.stream = self.calls[0].argv[3]

        # Every argument slot holding a tensor the plan does *not* own has to be
        # re-pointed on each call. Found by identity rather than written down by
        # hand: the caller's ``z``/``mask`` appear in four kernels between them,
        # and a missed slot is a read of freed memory -- silent, and only when
        # the caller happens to pass a fresh tensor.
        def slots(t):
            return tuple((ci, _ARGV0 + ai)
                         for ci, a in enumerate(A) for ai, x in enumerate(a)
                         if x is t)

        self.z_slots = slots(z)
        self.mask_slots = slots(mask)
        self.out_slots = slots(S["out"])
        assert self.z_slots and self.mask_slots and self.out_slots
        self.graph = None
        if _GRAPH:
            # Graph capture needs every node's argument list to be constant, which
            # is exactly why ``_ingest`` stages z/mask: only the first kernel may
            # touch the caller's inputs and only the last its output.
            assert all(ci == 0 for ci, _ in self.z_slots + self.mask_slots)
            assert all(ci == len(self.calls) - 1 for ci, _ in self.out_slots)
            self._capture()

    def _capture(self):
        """Capture the fixed-pointer middle of the pipeline into a CUDA graph.

        Only reached when the PDL intrinsics are unavailable. The cost being
        attacked is *device-side* dispatch, not host Python: measured back to
        back, an eagerly launched kernel costs ~3.3 us of GPU-side serialization
        against ~1.4 us as a graph node (and ~1.0 us with PDL, which is why PDL
        is preferred outright). Falls back to eager launches if capture fails.
        """
        inner = self.calls[1:-1]
        try:
            g = torch.cuda.CUDAGraph()
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                st = _raw_stream(self.dix)
                for c in inner:
                    c.argv[3] = st
                for _ in range(2):
                    for c in inner:
                        c.run(*c.argv)
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            with torch.cuda.graph(g):
                st = _raw_stream(self.dix)
                for c in inner:
                    c.argv[3] = st
                    c.run(*c.argv)
            torch.cuda.synchronize()
        except Exception:
            self.graph = None
        else:
            self.graph = g
        # The captured launchers must never be invoked directly again -- their
        # stream slot now names the (dead) capture stream.
        self.inner = inner

    def matches(self, z, mask) -> bool:
        # The alignment test is not optional: Triton specialized the compiled
        # kernels on the 16-byte alignment of the pointers they were built with,
        # so a misaligned buffer would silently miscompute.
        return (z.dtype is self.dtype and z.device == self.device
                and z.shape[-1] == self.c and z.shape[-2] == self.n
                and z.shape[-3] == self.n and z.numel() == self.numel
                and z.is_contiguous() and mask is not None
                and mask.is_contiguous() and mask.dtype is self.dtype
                and not (z.data_ptr() & 15) and not (mask.data_ptr() & 15))

    def run(self, z, mask, out):
        calls = self.calls
        zp, mp = z.data_ptr(), mask.data_ptr()
        for ci, ai in self.z_slots:
            calls[ci].argv[ai] = zp
        for ci, ai in self.mask_slots:
            calls[ci].argv[ai] = mp
        op = out.data_ptr()
        for ci, ai in self.out_slots:
            calls[ci].argv[ai] = op
        stream = _raw_stream(self.dix)
        g = self.graph
        if stream != self.stream:
            self.stream = stream
            for c in (calls if g is None else (calls[0], calls[-1])):
                c.argv[3] = stream
        if g is None:
            for c in calls:
                c.run(*c.argv)
        else:
            first, last = calls[0], calls[-1]
            first.run(*first.argv)
            g.replay()
            last.run(*last.argv)
        return out


class PairBlock(nn.Module):
    """Shared pair stack block for AF3 PairFormer / MSA module / template.

    Args:
        c_z: Pair embedding channel dimension
        c_hidden_mul: Hidden dim for triangle multiplication
        c_hidden_pair_att: Per-head hidden dim for triangle attention
        no_heads_pair: Number of heads in triangle attention
        transition_n: Scale of pair transition hidden dimension
        pair_dropout: Dropout rate (unused in inference baseline)
        inf: Large constant for masking
    """

    def __init__(
        self,
        c_z: int,
        c_hidden_mul: int,
        c_hidden_pair_att: int,
        no_heads_pair: int,
        transition_n: int,
        pair_dropout: float = 0.0,
        fuse_projection_weights: bool = False,
        inf: float = 1e9,
    ):
        super().__init__()

        self.tri_mul_out = TriangleMultiplicationOutgoing(c_z, c_hidden_mul)
        self.tri_mul_in = TriangleMultiplicationIncoming(c_z, c_hidden_mul)

        self.tri_att_start = TriangleAttention(
            c_z, c_hidden_pair_att, no_heads_pair, starting=True, inf=inf,
        )
        self.tri_att_end = TriangleAttention(
            c_z, c_hidden_pair_att, no_heads_pair, starting=False, inf=inf,
        )

        self.pair_transition = SwiGLUTransition(c_in=c_z, n=transition_n)

        self._plan: _Plan | None = None
        self._no_fuse = False
        self._rejected: set = set()

    # The plan folds the parameters into concatenated / pre-transposed buffers,
    # so any wholesale change to them has to invalidate it. Both hooks fire
    # before the first forward in normal use (``.to(device)``, then a checkpoint
    # load), which is exactly why the plan is built lazily.
    def _apply(self, *args, **kwargs):
        self._plan = None
        return super()._apply(*args, **kwargs)

    def _load_from_state_dict(self, *args, **kwargs):
        self._plan = None
        return super()._load_from_state_dict(*args, **kwargs)

    def _reference(self, z, pair_mask, pair_trans_mask):
        z = z + self.tri_mul_out(z, mask=pair_mask)
        z = z + self.tri_mul_in(z, mask=pair_mask)
        z = z + self.tri_att_start(z, mask=pair_mask)
        z = z + self.tri_att_end(z, mask=pair_mask)
        z = z + self.pair_transition(z, mask=pair_trans_mask)
        return z

    def forward(
        self,
        z: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size: int | None = None,
        use_deepspeed_evo_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
        _mask_trans: bool = True,
        _attn_chunk_size: int | None = None,
    ) -> torch.Tensor:
        """
        Args:
            z:         [*, N, N, C_z] pair embedding
            pair_mask: [*, N, N] pair mask

        Returns:
            [*, N, N, C_z] updated pair embedding
        """
        plan = self._plan
        if plan is not None and _mask_trans and plan.matches(z, pair_mask):
            return plan.run(z, pair_mask, torch.empty_like(z))

        key = (z.dtype, z.device, tuple(z.shape))
        if (not self._no_fuse and key not in self._rejected
                and _mask_trans and z.is_cuda
                and z.dtype in _FUSED_DTYPES and z.ndim >= 3
                and z.shape[-2] == z.shape[-3] and z.is_contiguous()
                and pair_mask is not None and pair_mask.dtype is z.dtype
                and pair_mask.is_contiguous()
                and pair_mask.shape == z.shape[:-1]
                and not torch.is_grad_enabled()):
            try:
                plan = _Plan(self, z, pair_mask)
            except _Unsupported:
                # A shape the kernels do not cover. Remembered per shape rather
                # than latched for the module, so a module that sees one odd
                # shape does not lose the fused path for every later one.
                self._rejected.add(key)
            except Exception:  # a Triton/driver surprise must not lose the op
                self._no_fuse = True
            else:
                self._plan = plan
                return plan.run(z, pair_mask, torch.empty_like(z))

        return self._reference(z, pair_mask, pair_mask if _mask_trans else None)
